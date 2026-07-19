"""
LinkedIn job scraper using requests + BeautifulSoup.
Scrapes the public LinkedIn job search pages.
"""
import logging
import random
import re
import time
from typing import List, Optional
from urllib.parse import quote_plus

import requests
from bs4 import BeautifulSoup
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from .base import BaseScraper, JobPosting

logger = logging.getLogger(__name__)

from profile_policy import LINKEDIN_LOCAL_GEO
from salary_rules import MAX_BASE_SALARY

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/121.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
    "Cache-Control": "max-age=0",
}

# LinkedIn experience-level facet: 4=mid-senior, 5=director, 6=executive.
# Kept pre-encoded because search URLs are built by string templates with
# quote_plus applied to the keywords only — there is no urlencode layer.
SENIORITY_FACET = "&f_E=4%2C5%2C6"


def _parse_salary(text: str) -> tuple[Optional[float], Optional[float]]:
    """
    Parse salary range from text.
    Handles: "$100,000", "$100k", "$100K-$150K", "100000 - 150000", etc.
    Returns (salary_min, salary_max) as floats or (None, None).
    """
    if not text:
        return None, None

    # Strip 401(k) / 401k references before parsing to avoid confusing them
    # with salary figures (e.g. "401(k) matching" → "$401K" false positive).
    cleaned = re.sub(r"\b401\s*\([Kk]\)", "", text)
    cleaned = re.sub(r"\b401[Kk]\b", "", cleaned)

    # Pattern: $xxx,xxx or $xxxK or $xxx.xx with optional K
    pattern = r"\$(\d{1,3}(?:,\d{3})*(?:\.\d+)?)\s*[Kk]?"
    matches = re.findall(pattern, cleaned)
    values = []
    for m in matches:
        val = float(m.replace(",", ""))
        # If value looks like it's in K format (< 1000), multiply by 1000
        if val < 1000:
            val *= 1000
        values.append(val)

    # Also handle plain "150K" or "150k" without $ sign
    k_pattern = r"(?<!\$)\b(\d{2,3})[Kk]\b"
    k_matches = re.findall(k_pattern, cleaned)
    for m in k_matches:
        val = float(m) * 1000
        values.append(val)

    if not values:
        return None, None

    values = sorted(set(values))
    filtered = []
    for v in values:
        if v > MAX_BASE_SALARY:
            logger.warning("_parse_salary: discarding $%.0f — exceeds MAX_BASE_SALARY cap", v)
        elif v >= 20000:
            filtered.append(v)
    values = filtered

    if not values:
        return None, None
    if len(values) == 1:
        return values[0], values[0]

    sal_min, sal_max = values[0], values[-1]

    # Discard high end if range is impossibly wide (> 5× the low end)
    if sal_min > 0 and sal_max / sal_min > 5:
        logger.warning(
            "_parse_salary: suspicious range $%.0f–$%.0f (ratio %.1f), dropping high end",
            sal_min, sal_max, sal_max / sal_min,
        )
        return sal_min, sal_min

    return sal_min, sal_max


class LinkedInScraper(BaseScraper):
    """Scrapes LinkedIn public job search pages."""

    BASE_SEARCH_URL = (
        "https://www.linkedin.com/jobs/search/"
        "?keywords={query}&f_WT=2&f_TPR=r86400&position=1&pageNum=0"
    )
    COMPANY_SEARCH_URL = (
        "https://www.linkedin.com/jobs/search/"
        "?keywords={query}&f_TPR=r604800&f_C={company_id}&position=1&pageNum=0"
    )
    LOCAL_SEARCH_URL = (
        "https://www.linkedin.com/jobs/search/"
        "?keywords={query}&location={location}&distance=25"
        "&f_TPR=r86400&position=1&pageNum=0"
    )
    # Guest fragment endpoint serving result pages past the first HTML page.
    # start is an absolute card offset; arbitrary (non-multiple-of-25) values
    # are accepted (verified live 2026-07-13).
    FRAGMENT_SEARCH_URL = (
        "https://www.linkedin.com/jobs-guest/jobs/api/seeMoreJobPostings/search"
        "?keywords={query}&f_WT=2&f_TPR=r86400&start={start}"
    )
    LOCAL_LOCATION = LINKEDIN_LOCAL_GEO
    BASE_JOB_URL = "https://www.linkedin.com/jobs/view/{job_id}/"

    # 429 rate-limit handling
    MAX_429_RETRIES = 2          # retries per request after a 429
    BASE_429_WAIT = 5            # seconds; escalates 5, 10, ... per retry
    MAX_429_WAIT = 60
    CIRCUIT_429_THRESHOLD = 6    # consecutive 429s trip the breaker for the run

    def __init__(self, config: dict, known_urls: Optional[set] = None):
        super().__init__(config)
        self.session = requests.Session()
        self.session.headers.update(HEADERS)
        self._consecutive_429 = 0
        self._rate_limited = False  # circuit breaker: pause per-card fetches
        # Monotonic timestamp of the last detail-request start; pacing sleeps
        # count time the request itself already spent toward the politeness
        # gap. Instance-level so it spans keyword/company/local passes (the
        # whole chain runs serialized on one thread).
        self._last_detail_start: Optional[float] = None
        # Canonical LinkedIn URLs already in the DB — cards matching these
        # skip the expensive per-card description fetch entirely.
        self.known_urls: set = known_urls or set()
        # URLs emitted this run, shared across keyword/company/local paths.
        self.seen_urls: set = set()
        self._known_skipped = 0  # per-query skip counter (reset in scrape)
        # Keyword searches walk up to max_pages result pages (~25 cards each).
        # Floor at 1: a 0/negative misconfig must degrade to a single page's
        # worth of results, never silently scrape zero jobs (cap = max_pages*25).
        self.max_pages = max(1, int(config.get("linkedin_max_pages", 3)))
        # Queries that get the seniority facet (trims entry-level noise on
        # adjacent-function searches); config lists queries, never params.
        self.facet_queries = set(
            config.get("linkedin_seniority_facet_queries", []) or []
        )

    def _facet_suffix(self, query: str) -> str:
        return SENIORITY_FACET if query in self.facet_queries else ""

    def _build_search_url(self, query: str) -> str:
        return (
            self.BASE_SEARCH_URL.format(query=quote_plus(query))
            + self._facet_suffix(query)
        )

    def _build_fragment_url(self, query: str, start: int) -> str:
        return (
            self.FRAGMENT_SEARCH_URL.format(query=quote_plus(query), start=start)
            + self._facet_suffix(query)
        )

    @staticmethod
    def _retry_after_seconds(resp) -> Optional[int]:
        """Parse a Retry-After header expressed in seconds; ignore date form."""
        val = resp.headers.get("Retry-After")
        if not val:
            return None
        try:
            return max(0, int(val))
        except (TypeError, ValueError):
            return None

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((requests.ConnectionError, requests.Timeout)),
        reraise=True,
    )
    def _fetch_page(self, url: str) -> Optional[str]:
        """Fetch a page, returning HTML text or None on error.

        On 429, retries with escalating backoff (honoring Retry-After) up to
        MAX_429_RETRIES. After CIRCUIT_429_THRESHOLD consecutive 429s the circuit
        breaker trips (self._rate_limited), which pauses per-card description
        fetches so a hard rate-limit doesn't burn the whole run on 30s sleeps.
        """
        for attempt in range(self.MAX_429_RETRIES + 1):
            try:
                resp = self.session.get(url, timeout=15)
            except (requests.ConnectionError, requests.Timeout):
                raise  # Let tenacity handle retries
            except Exception as e:
                self.logger.warning("Unexpected error fetching %s: %s", url, e)
                return None

            if resp.status_code == 429:
                self._consecutive_429 += 1
                if self._consecutive_429 >= self.CIRCUIT_429_THRESHOLD and not self._rate_limited:
                    self._rate_limited = True
                    self.logger.warning(
                        "LinkedIn rate-limited %d× consecutively — pausing per-card "
                        "description fetches for the rest of this run.",
                        self._consecutive_429,
                    )
                if attempt < self.MAX_429_RETRIES:
                    wait = self._retry_after_seconds(resp) or min(
                        self.BASE_429_WAIT * (2 ** attempt), self.MAX_429_WAIT
                    )
                    self.logger.warning(
                        "LinkedIn 429 (attempt %d/%d). Waiting %ds then retrying...",
                        attempt + 1, self.MAX_429_RETRIES, wait,
                    )
                    time.sleep(wait)
                    continue
                self.logger.warning("LinkedIn 429 after %d retries; giving up on %s",
                                    self.MAX_429_RETRIES, url)
                return None

            # Non-429 response: LinkedIn is responding, reset rate-limit state.
            self._consecutive_429 = 0
            self._rate_limited = False
            if resp.status_code == 403:
                self.logger.warning("LinkedIn returned 403 (likely blocked). URL: %s", url)
                return None
            try:
                resp.raise_for_status()
            except requests.HTTPError as e:
                self.logger.warning("HTTP error fetching %s: %s", url, e)
                return None
            return resp.text
        return None

    def _fetch_job_description(self, job_url: str) -> str:
        """Fetch full job description from LinkedIn job detail page."""
        if self._rate_limited:
            # Circuit breaker tripped — keep the card's metadata, skip the fetch.
            return ""
        # Politeness pacing: 1-3s jittered gap between detail-request STARTS.
        # Time the previous request already spent (fetch, 429 backoff, parsing)
        # counts toward the gap — only the remainder is slept, so a request
        # slower than the target adds no dead time on top.
        target = random.uniform(1, 3)
        if self._last_detail_start is not None:
            target -= time.monotonic() - self._last_detail_start
        time.sleep(max(0.0, target))
        self._last_detail_start = time.monotonic()
        html = self._fetch_page(job_url)
        if not html:
            return ""
        try:
            soup = BeautifulSoup(html, "lxml")
            # Try multiple selectors for job description
            for selector in [
                ".show-more-less-html__markup",
                "#job-details",
                ".description__text",
                ".jobs-description__content",
            ]:
                el = soup.select_one(selector)
                if el:
                    return el.get_text(separator=" ", strip=True)
        except Exception as e:
            self.logger.debug("Error parsing job description from %s: %s", job_url, e)
        return ""

    def _parse_html(
        self, html: str, query: str, limit: Optional[int] = None
    ) -> tuple[List[JobPosting], int]:
        """Parse LinkedIn search result HTML.

        Returns (new_jobs, cards_seen): cards_seen counts every card parsed
        (capped at limit), new_jobs contains only jobs whose canonical URL is
        not already known (DB) or seen (this run).
        """
        jobs: List[JobPosting] = []
        try:
            soup = BeautifulSoup(html, "lxml")
        except Exception as e:
            self.logger.error("Failed to parse LinkedIn HTML: %s", e)
            return jobs, 0

        cards = (
            soup.select("li.jobs-search__results-list > div.base-card") or
            soup.select("div.base-card") or
            soup.select("li > div[data-entity-urn]")
        )
        if not cards:
            cards = (
                soup.select(".job-search-card") or
                soup.select("a.base-card__full-link")
            )

        self.logger.info("LinkedIn found %d cards for query: %s", len(cards), query)

        cards = cards[: limit if limit is not None else self.max_results]
        for card in cards:
            try:
                job = self._parse_card(card, query)
                if job:
                    jobs.append(job)
            except Exception as e:
                self.logger.debug("Error parsing LinkedIn card: %s", e)
                continue

        return jobs, len(cards)

    def scrape(self, query: str) -> List[JobPosting]:
        """Scrape LinkedIn for jobs matching the query, walking result pages.

        Page 0 is the public HTML search page; subsequent pages come from the
        guest fragment endpoint with start=<cards seen so far>. The walk stops
        at the card cap (linkedin_max_pages * 25), on an empty page, or after
        two consecutive pages with zero new URLs — one all-known page is weak
        signal because the configured queries overlap heavily.
        """
        self._known_skipped = 0
        cap = self.max_pages * 25

        url = self._build_search_url(query)
        self.logger.info("LinkedIn scraping: %s", url)
        html = self._fetch_page(url)
        if not html:
            self.logger.warning("LinkedIn returned no content for query: %s", query)
            return []

        jobs, cards_seen = self._parse_html(html, query, limit=cap)
        pages = 1
        zero_new_streak = 0 if jobs else (1 if cards_seen else 0)

        while 0 < cards_seen < cap and zero_new_streak < 2:
            time.sleep(random.uniform(1, 2))
            frag_url = self._build_fragment_url(query, start=cards_seen)
            self.logger.info("LinkedIn scraping (page %d): %s", pages + 1, frag_url)
            frag_html = self._fetch_page(frag_url)
            if not frag_html:
                break
            frag_jobs, frag_cards = self._parse_html(
                frag_html, query, limit=cap - cards_seen
            )
            pages += 1
            if frag_cards == 0:
                break
            cards_seen += frag_cards
            if frag_jobs:
                jobs.extend(frag_jobs)
                zero_new_streak = 0
            else:
                zero_new_streak += 1

        self.logger.info(
            'LinkedIn query done: "%s" pages=%d cards=%d known_skipped=%d new=%d',
            query, pages, cards_seen, self._known_skipped, len(jobs),
        )
        return jobs

    def scrape_companies(self, queries: List[str]) -> List[JobPosting]:
        """Company-targeted searches for linkedin_company_searches, re-using
        all configured queries with f_C={company_id} appended."""
        return self._run_company_searches(
            self.config.get("linkedin_company_searches", {}) or {}, queries
        )

    def scrape_local_companies(self) -> List[JobPosting]:
        """Company-targeted searches for local-area fallback companies.

        Uses ONLY local_search_queries (1-2 queries) — never the full
        search_queries set — to stay inside the LinkedIn 429 budget.
        """
        return self._run_company_searches(
            self.config.get("linkedin_local_company_searches", {}) or {},
            self.config.get("local_search_queries", []) or [],
        )

    def _run_company_searches(
        self, company_searches: dict, queries: List[str]
    ) -> List[JobPosting]:
        if not company_searches or not queries:
            return []

        all_jobs: List[JobPosting] = []

        for company_name, company_id in company_searches.items():
            self.logger.info(
                "LinkedIn company-targeted search: %s (ID: %s)", company_name, company_id
            )
            for i, query in enumerate(queries):
                # Drop "remote" from company-targeted queries — it suppresses results
                # when combined with f_C. Remote preference is handled by scoring.
                company_query = re.sub(r'\bremote\b', '', query, flags=re.IGNORECASE).strip()
                url = self.COMPANY_SEARCH_URL.format(
                    query=quote_plus(company_query), company_id=company_id
                )
                self.logger.info("LinkedIn company scraping: %s", url)

                html = self._fetch_page(url)
                if not html:
                    self.logger.warning(
                        "LinkedIn returned no content for %s / query: %s", company_name, query
                    )
                else:
                    page_jobs, _ = self._parse_html(html, query)
                    all_jobs.extend(page_jobs)

                if i < len(queries) - 1:
                    time.sleep(self.rate_limit)

            time.sleep(self.rate_limit)

        self.logger.info(
            "LinkedIn company searches returned %d unique jobs.", len(all_jobs)
        )
        return all_jobs

    def scrape_local(self) -> List[JobPosting]:
        """Local-area-scoped keyword searches (location param, no f_WT=2
        remote filter) for local_search_queries."""
        queries = self.config.get("local_search_queries", []) or []
        if not queries:
            return []

        all_jobs: List[JobPosting] = []
        for i, query in enumerate(queries):
            url = self.LOCAL_SEARCH_URL.format(
                query=quote_plus(query), location=quote_plus(self.LOCAL_LOCATION)
            )
            self.logger.info("LinkedIn local scraping: %s", url)
            html = self._fetch_page(url)
            if not html:
                self.logger.warning(
                    "LinkedIn returned no content for local query: %s", query
                )
            else:
                page_jobs, _ = self._parse_html(html, query)
                all_jobs.extend(page_jobs)
            if i < len(queries) - 1:
                time.sleep(self.rate_limit)

        self.logger.info("LinkedIn local search returned %d unique jobs.", len(all_jobs))
        return all_jobs

    @staticmethod
    def _normalize_job_url(href: str) -> str:
        """
        Normalize a LinkedIn job URL to a canonical form.
        Strips query params, fragments, and reconstructs from the numeric job ID
        so that the same job always produces the same URL string regardless of
        which search path (keyword vs company-targeted) fetched it.
        e.g. https://www.linkedin.com/jobs/view/1234567/?refId=abc&trackingId=xyz
             → https://www.linkedin.com/jobs/view/1234567/
        """
        # Strip query params and fragments first
        href = href.split("?")[0].split("#")[0].strip()
        # LinkedIn guest URLs are slug-form with the numeric ID at the END, e.g.
        # /jobs/view/senior-product-manager-at-google-4433250484 — capture the
        # trailing digits (optional slug prefix) so the plain numeric form and
        # any slug for the same ID collapse to one canonical URL.
        m = re.search(r"/jobs/view/(?:[^/?#]*-)?(\d+)", href)
        if m:
            return f"https://www.linkedin.com/jobs/view/{m.group(1)}/"
        return href

    def _parse_card(self, card: BeautifulSoup, query: str) -> Optional[JobPosting]:
        """Parse a single LinkedIn job card into a JobPosting."""
        # Extract URL
        link_el = card.select_one("a.base-card__full-link") or card.select_one("a[href*='/jobs/view/']")
        if not link_el:
            return None
        job_url = self._normalize_job_url(link_el.get("href", ""))
        if not job_url:
            return None
        # Dedup before the expensive detail fetch: skip anything already in
        # the DB (known_urls) or already emitted this run (seen_urls).
        if job_url in self.known_urls or job_url in self.seen_urls:
            self._known_skipped += 1
            return None

        # Extract title
        title_el = card.select_one(".base-search-card__title") or card.select_one("h3")
        title = title_el.get_text(strip=True) if title_el else ""
        if not title:
            return None

        # Extract company
        company_el = card.select_one(".base-search-card__subtitle") or \
                     card.select_one(".job-search-card__company-name") or \
                     card.select_one("h4")
        company = company_el.get_text(strip=True) if company_el else ""

        # Extract location
        location_el = card.select_one(".job-search-card__location") or \
                      card.select_one(".base-search-card__metadata")
        location = location_el.get_text(strip=True) if location_el else ""

        # Extract date
        date_el = card.select_one(".job-search-card__listdate") or \
                  card.select_one("time")
        date_posted = None
        if date_el:
            date_posted = date_el.get("datetime") or date_el.get_text(strip=True)

        self.seen_urls.add(job_url)

        # Fetch description
        description = ""
        try:
            description = self._fetch_job_description(job_url)
        except Exception as e:
            self.logger.debug("Failed to fetch description for %s: %s", job_url, e)

        # Parse salary
        salary_min, salary_max = _parse_salary(description)

        return JobPosting(
            title=title,
            company=company,
            location=self._normalize_location(location),
            url=job_url,
            description=description,
            salary_min=salary_min,
            salary_max=salary_max,
            date_posted=date_posted,
            source="linkedin",
        )
