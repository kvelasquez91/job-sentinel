"""
Greenhouse and Lever job board scrapers.
Both use public JSON APIs - no HTML parsing needed.
"""
import html as html_module
import logging
import re
import time
from typing import List, Optional

import requests
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from local_area import is_local_commuter_area
from .base import BaseScraper, JobPosting

logger = logging.getLogger(__name__)

from profile_policy import AI_TITLE_KEYWORDS, PRODUCT_TITLE_KEYWORDS
from salary_rules import MAX_BASE_SALARY, extract_salary_regex

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/121.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/html,*/*",
}

# Board slugs come from config.yaml (`greenhouse_slugs` / `lever_slugs` /
# `ashby_slugs`). These module defaults apply only when a key is absent.
GREENHOUSE_SLUGS: dict = {}
LEVER_SLUGS: dict = {}
ASHBY_SLUGS: dict = {}

# Title keywords that indicate PM or product leadership roles (profile_policy.PRODUCT_TITLE_KEYWORDS).
# Keywords that indicate AI/ML relevance IN A TITLE, stricter than description check (profile_policy.AI_TITLE_KEYWORDS).


def _is_relevant_title(title: str) -> bool:
    """Check if a job title matches product management roles."""
    t = title.lower()
    return any(kw in t for kw in PRODUCT_TITLE_KEYWORDS)


def _mentions_ai(text: str) -> bool:
    """Check if text mentions AI/ML keywords (for title matching)."""
    t = text.lower()
    return any(kw in t for kw in AI_TITLE_KEYWORDS)


def _title_passes_filter(title: str) -> bool:
    """Job must have a product/PM title. AI-in-title alone is not enough."""
    return _is_relevant_title(title)


# Common filler words dropped when matching a title against target phrases.
_TITLE_STOPWORDS = {"of", "the", "and", "for", "a", "an", "to", "in", "on", "with", "at"}


def _tokenize_title(title: str) -> set:
    """Significant lowercase word tokens of a title (punctuation & stopwords removed)."""
    words = re.findall(r"[a-z0-9]+", (title or "").lower())
    return {w for w in words if w not in _TITLE_STOPWORDS}


# `is_local_commuter_area(location, local_locations)` is imported from `local_area`
# (the shared matcher, also used by engine/scorer.py and engine/llm_scorer.py) and
# re-exported here so smartrecruiters.py / workday.py keep importing it from
# `.greenhouse`. It matches a listed city ONLY when directly followed by the
# configured state context, so e.g. "Springfield, NC" / "Portland, OR" are NOT local.
def local_title_passes(title: str, location: str, local_locations,
                       local_title_token_sets) -> bool:
    """True if the job is in the local commuter area AND its title token-supersets
    one of the broadened local target titles. Lets local digital roles through the
    strict PM filter without loosening it for remote-company jobs."""
    if not title or not location or not local_title_token_sets:
        return False
    if not is_local_commuter_area(location, local_locations):
        return False
    title_tokens = _tokenize_title(title)
    return any(toks <= title_tokens for toks in local_title_token_sets)


def _parse_salary(text: str) -> tuple[Optional[float], Optional[float]]:
    """Parse salary from text. Returns (min, max) or (None, None)."""
    if not text:
        return None, None

    pattern = r"\$(\d{1,3}(?:,\d{3})*(?:\.\d+)?)\s*[Kk]?"
    matches = re.findall(pattern, text)
    values = []
    for m in matches:
        val = float(m.replace(",", ""))
        if val < 1000:
            val *= 1000
        values.append(val)

    k_pattern = r"(?<!\$)\b(\d{2,3})[Kk]\b"
    k_matches = re.findall(k_pattern, text)
    for m in k_matches:
        val = float(m) * 1000
        values.append(val)

    if not values:
        return None, None

    filtered = []
    for v in sorted(set(values)):
        if v > MAX_BASE_SALARY:
            logger.warning("_parse_salary: discarding $%.0f — exceeds MAX_BASE_SALARY cap", v)
        elif v >= 20000:
            filtered.append(v)
    values = filtered

    if not values:
        return None, None
    if len(values) == 1:
        return values[0], values[0]
    return values[0], values[-1]


# Sentences on a posting page must mention one of these to have their dollar
# figures treated as salary — company-hosted pages (e.g. stripe.com) are full
# of marketing dollar amounts that must never leak into salary fields.
_SALARY_CONTEXT_KEYWORDS = (
    "salary", "compensation", "pay range", "base pay", "on target earnings",
    "hiring range", "pay scale", "wage",
)


def extract_salary_from_page_text(text: str) -> tuple[Optional[float], Optional[float]]:
    """Extract a salary range from full posting-page text.

    Some boards (Stripe) omit pay data from the Greenhouse API entirely and
    render it only on the company-hosted posting page. Page text is noisy, so
    dollar figures only count when their sentence contains a salary-context
    keyword.
    """
    if not text:
        return None, None
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        lowered = sentence.lower()
        if any(kw in lowered for kw in _SALARY_CONTEXT_KEYWORDS):
            lo, hi = extract_salary_regex(sentence)
            if lo is not None:
                return lo, hi
    return None, None


def _clean_html(text: str) -> str:
    """Strip HTML tags and decode HTML entities from text."""
    if not text:
        return ""
    clean = re.sub(r"<[^>]+>", " ", text)
    clean = html_module.unescape(clean)
    clean = re.sub(r"\s+", " ", clean).strip()
    return clean


class GreenhouseScraper(BaseScraper):
    """
    Scrapes Greenhouse and Lever public job board APIs.
    These are reliable JSON APIs that don't require authentication.
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self.session = requests.Session()
        self.session.headers.update(HEADERS)
        # Profile target titles as token sets. A job title matches a target when
        # it contains all of the target's significant words (order-independent),
        # so "Senior Product Manager AI" matches "Senior Product Manager, AI
        # Platform" — full-phrase substring matching missed these and silently
        # dropped Greenhouse/Lever/Ashby yield to zero.
        target_titles = config.get("profile", {}).get("target_titles", [])
        self._profile_title_token_sets: List[set] = [
            toks for toks in (_tokenize_title(t) for t in target_titles) if toks
        ]

        # Local-area broadening (Task 11): a job located in the local commuter
        # area may ALSO pass on a broadened local_target_titles match, without
        # loosening the filter for remote-company jobs elsewhere.
        self._local_locations = config.get("local_locations") or []
        self._local_title_token_sets: List[set] = [
            toks for toks in (_tokenize_title(t)
                              for t in (config.get("local_target_titles") or [])) if toks
        ]

    def _passes_title_filter(self, title: str) -> bool:
        """Decide whether a job title is relevant for the loaded profile.

        A title passes if it matches the built-in PM/product keyword baseline
        OR contains all significant words of one of the profile's
        ``target_titles``. Profile targets can only *broaden* the baseline; they
        can never narrow it below the built-in filter (which is what previously
        let a set of over-specific target phrases zero out every board).
        """
        if _title_passes_filter(title):
            return True
        if self._profile_title_token_sets:
            title_tokens = _tokenize_title(title)
            return any(toks <= title_tokens for toks in self._profile_title_token_sets)
        return False

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        retry=retry_if_exception_type((requests.ConnectionError, requests.Timeout)),
        reraise=True,
    )
    def _fetch_json(self, url: str) -> Optional[dict | list]:
        """Fetch JSON from URL, returning parsed data or None on error."""
        try:
            resp = self.session.get(url, timeout=15)
            if resp.status_code == 404:
                self.logger.debug("404 for URL: %s (company may not use this board)", url)
                return None
            if resp.status_code == 429:
                self.logger.warning("Rate limited (429) for URL: %s", url)
                time.sleep(10)
                return None
            resp.raise_for_status()
            return resp.json()
        except requests.HTTPError as e:
            self.logger.debug("HTTP error fetching %s: %s", url, e)
            return None
        except (requests.ConnectionError, requests.Timeout):
            raise
        except Exception as e:
            self.logger.debug("Error fetching JSON from %s: %s", url, e)
            return None

    def scrape(self, query: str) -> List[JobPosting]:
        """
        For Greenhouse/Lever, the query is not used as a search term.
        Instead, we scrape all relevant companies' boards and filter by title/keywords.
        This method is called once per 'query' in scrape_all, but we return results
        only on the first call to avoid duplicate scraping.
        """
        # Use query as a signal - only scrape on first call
        if query != list(self.config.get("search_queries", [""]))[0]:
            return []
        return self._scrape_all_boards()

    def _scrape_all_boards(self) -> List[JobPosting]:
        """Scrape all configured Greenhouse, Lever, and Ashby boards.

        Profile-specific slugs from the config are merged with the module-level
        defaults so that each profile can add its own target companies without
        losing the shared baseline list.
        """
        all_jobs: List[JobPosting] = []
        rate = self.config.get("scraping", {}).get("rate_limit_seconds", 1)

        # If the profile supplies explicit slugs, use ONLY those (even if empty).
        # If the key is absent from config (None), fall back to module-level defaults.
        config_gh = self.config.get("greenhouse_slugs")
        greenhouse_slugs = GREENHOUSE_SLUGS if config_gh is None else config_gh

        config_lv = self.config.get("lever_slugs")
        lever_slugs = LEVER_SLUGS if config_lv is None else config_lv

        config_ab = self.config.get("ashby_slugs")
        ashby_slugs = ASHBY_SLUGS if config_ab is None else config_ab

        # Scrape Greenhouse boards
        for company_name, slug in greenhouse_slugs.items():
            try:
                jobs = self._scrape_greenhouse(company_name, slug)
                all_jobs.extend(jobs)
                self.logger.info("Greenhouse %s: %d relevant jobs", company_name, len(jobs))
                time.sleep(rate)
            except Exception as e:
                self.logger.error("Error scraping Greenhouse for %s: %s", company_name, e)

        # Scrape Lever boards
        for company_name, slug in lever_slugs.items():
            try:
                jobs = self._scrape_lever(company_name, slug)
                all_jobs.extend(jobs)
                self.logger.info("Lever %s: %d relevant jobs", company_name, len(jobs))
                time.sleep(rate)
            except Exception as e:
                self.logger.error("Error scraping Lever for %s: %s", company_name, e)

        # Scrape Ashby boards
        for company_name, slug in ashby_slugs.items():
            try:
                jobs = self._scrape_ashby(company_name, slug)
                all_jobs.extend(jobs)
                self.logger.info("Ashby %s: %d relevant jobs", company_name, len(jobs))
                time.sleep(rate)
            except Exception as e:
                self.logger.error("Error scraping Ashby for %s: %s", company_name, e)

        self.logger.info("Greenhouse/Lever/Ashby total jobs before dedup: %d", len(all_jobs))
        return all_jobs

    def _scrape_greenhouse(self, company_name: str, slug: str) -> List[JobPosting]:
        """Scrape a company's Greenhouse job board via the public API."""
        url = f"https://api.greenhouse.io/v1/boards/{slug}/jobs"
        data = self._fetch_json(url)
        if not data:
            return []

        jobs_data = data.get("jobs", []) if isinstance(data, dict) else data
        relevant_jobs: List[JobPosting] = []

        for job_data in jobs_data:
            try:
                title = job_data.get("title", "")
                if not title:
                    continue

                description_raw = _clean_html(
                    job_data.get("content", "") or
                    job_data.get("description", "")
                )

                # Get location (computed before the title filter so a local
                # digital role can be evaluated against the broadened net).
                # Coerced to str: an API returning a non-string/null "name"
                # must not reach local_title_passes/is_local_commuter_area as anything
                # but a string (that code path is now reachable even for
                # titles that fail the strict filter, so a malformed shape
                # here must not raise).
                location = ""
                location_data = job_data.get("location", {})
                if isinstance(location_data, dict):
                    location = str(location_data.get("name") or "")
                elif isinstance(location_data, str):
                    location = location_data

                # Also check metadata for additional locations
                metadata = job_data.get("metadata", [])
                if isinstance(metadata, list):
                    for meta in metadata:
                        if isinstance(meta, dict) and meta.get("name", "").lower() in ("location", "remote"):
                            loc_val = meta.get("value", "")
                            if loc_val:
                                location = str(loc_val)
                                break

                # Filter: must be a product/PM role or mention AI, OR (for
                # local-commuter-area jobs only) match the broadened local_target_titles.
                if not (self._passes_title_filter(title)
                        or local_title_passes(title, location, self._local_locations,
                                               self._local_title_token_sets)):
                    continue

                job_url = job_data.get("absolute_url", "")
                if not job_url:
                    job_id = job_data.get("id", "")
                    if job_id:
                        job_url = f"https://boards.greenhouse.io/{slug}/jobs/{job_id}"

                if not job_url:
                    continue

                date_posted = job_data.get("updated_at") or job_data.get("published_at")

                # Fetch full description if not included
                if not description_raw:
                    detail = self._fetch_greenhouse_detail(slug, job_data.get("id"))
                    description_raw = detail or ""

                salary_min, salary_max = _parse_salary(description_raw)

                # Some boards (Stripe) never expose pay data via the API — the
                # range exists only on the company-hosted posting page.
                if salary_min is None:
                    salary_min, salary_max = extract_salary_from_page_text(
                        self._fetch_posting_page_text(job_url))

                relevant_jobs.append(JobPosting(
                    title=title,
                    company=company_name.replace("-", " ").title(),
                    location=self._normalize_location(location),
                    url=job_url,
                    description=description_raw[:5000],  # Cap description length
                    salary_min=salary_min,
                    salary_max=salary_max,
                    date_posted=date_posted,
                    source="greenhouse",
                ))
            except Exception as e:
                self.logger.debug("Error parsing Greenhouse job for %s: %s", company_name, e)
                continue

        return relevant_jobs

    def _fetch_posting_page_text(self, url: str) -> str:
        """Fetch a job's public posting page and return its visible text.

        Used as a salary fallback: some boards (Stripe) publish the pay range
        only on the company-hosted page, never through the Greenhouse API.
        Returns "" on any error — the caller treats that as "no salary found".
        """
        if not url:
            return ""
        try:
            resp = self.session.get(url, timeout=15)
            if resp.status_code != 200:
                self.logger.debug("Posting page %s returned %s", url, resp.status_code)
                return ""
            return _clean_html(resp.text)
        except Exception as e:
            self.logger.debug("Error fetching posting page %s: %s", url, e)
            return ""

    def _fetch_greenhouse_detail(self, slug: str, job_id: Optional[int]) -> str:
        """Fetch full job details from Greenhouse API."""
        if not job_id:
            return ""
        url = f"https://api.greenhouse.io/v1/boards/{slug}/jobs/{job_id}"
        data = self._fetch_json(url)
        if not data:
            return ""
        content = data.get("content", "") or data.get("description", "")
        return _clean_html(content)

    def _scrape_ashby(self, company_name: str, slug: str) -> List[JobPosting]:
        """Scrape a company's Ashby job board via the public posting API."""
        url = f"https://api.ashbyhq.com/posting-api/job-board/{slug}"
        data = self._fetch_json(url)
        if not data:
            return []

        postings = data.get("jobs", data.get("jobPostings", [])) if isinstance(data, dict) else []
        relevant_jobs: List[JobPosting] = []

        for posting in postings:
            try:
                title = posting.get("title", "")
                if not title:
                    continue

                description_raw = _clean_html(
                    posting.get("descriptionHtml", "") or
                    posting.get("description", "")
                )

                # Coerced to str: local_title_passes/is_local_commuter_area is now
                # reachable even for titles that fail the strict filter, so a
                # malformed (non-string) location shape must not raise.
                location = str(posting.get("locationName") or posting.get("location") or "")

                if not (self._passes_title_filter(title)
                        or local_title_passes(title, location, self._local_locations,
                                               self._local_title_token_sets)):
                    continue

                job_url = posting.get("jobUrl", "") or posting.get("applyUrl", "")
                if not job_url:
                    continue

                date_posted = posting.get("publishedAt") or posting.get("updatedAt")

                salary_min, salary_max = _parse_salary(description_raw)

                # Ashby sometimes has compensation in a dedicated field
                comp = posting.get("compensation", {})
                if isinstance(comp, dict) and not salary_min:
                    salary_min = comp.get("minValue") or comp.get("min")
                    salary_max = comp.get("maxValue") or comp.get("max")

                relevant_jobs.append(JobPosting(
                    title=title,
                    company=company_name,
                    location=self._normalize_location(location),
                    url=job_url,
                    description=description_raw[:5000],
                    salary_min=float(salary_min) if salary_min else None,
                    salary_max=float(salary_max) if salary_max else None,
                    date_posted=str(date_posted) if date_posted else None,
                    source="ashby",
                ))
            except Exception as e:
                self.logger.debug("Error parsing Ashby job for %s: %s", company_name, e)
                continue

        return relevant_jobs

    def _scrape_lever(self, company_name: str, slug: str) -> List[JobPosting]:
        """Scrape a company's Lever job board."""
        url = f"https://api.lever.co/v0/postings/{slug}?mode=json"
        data = self._fetch_json(url)
        if not data:
            return []

        postings = data if isinstance(data, list) else data.get("postings", [])
        relevant_jobs: List[JobPosting] = []

        for posting in postings:
            try:
                title = posting.get("text", "")
                if not title:
                    continue

                description_raw = _clean_html(
                    posting.get("descriptionPlain", "") or
                    posting.get("description", "")
                )

                # Get categories/location. Coerced to str: local_title_passes/
                # is_local_commuter_area is now reachable even for titles that fail the
                # strict filter, so a malformed (non-string) location shape
                # must not raise.
                categories = posting.get("categories", {})
                location = ""
                if isinstance(categories, dict):
                    location = str(
                        categories.get("location") or categories.get("commitment") or ""
                    )

                # Filter: must be a product/PM role or mention AI, OR (for
                # local-commuter-area jobs only) match the broadened local_target_titles.
                if not (self._passes_title_filter(title)
                        or local_title_passes(title, location, self._local_locations,
                                               self._local_title_token_sets)):
                    continue

                job_url = posting.get("hostedUrl", "") or posting.get("applyUrl", "")
                if not job_url:
                    continue

                # Lever uses createdAt as Unix timestamp in milliseconds
                created_at = posting.get("createdAt")
                date_posted = None
                if created_at:
                    try:
                        from datetime import datetime
                        ts = int(created_at) / 1000  # ms to seconds
                        date_posted = datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d")
                    except Exception:
                        date_posted = str(created_at)

                salary_min, salary_max = _parse_salary(description_raw)

                relevant_jobs.append(JobPosting(
                    title=title,
                    company=company_name.title(),
                    location=self._normalize_location(location),
                    url=job_url,
                    description=description_raw[:5000],
                    salary_min=salary_min,
                    salary_max=salary_max,
                    date_posted=date_posted,
                    source="lever",
                ))
            except Exception as e:
                self.logger.debug("Error parsing Lever job for %s: %s", company_name, e)
                continue

        return relevant_jobs
