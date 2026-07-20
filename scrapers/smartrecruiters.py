"""
SmartRecruiters job scraper.

SmartRecruiters is a widely-used ATS. Companies publish jobs through a public
REST API that requires no authentication.

Currently configured for Palo Alto Networks and Arista Networks.

API endpoint:
  GET https://api.smartrecruiters.com/v1/companies/{company_id}/postings
  Params: limit, offset (we page the whole board and filter titles locally;
          the server-side `q` search matched almost nothing for our queries)
  Returns: {"content": [...], "totalFound": N, "offset": N, "limit": N}

Job object key fields:
  id             — posting ID
  name           — job title
  location       — {city, country, remote}
  releasedDate   — ISO 8601 date string
  ref            — canonical job URL

Research conducted 2026-03-31.
"""
import logging
import time
from typing import List, Optional

import requests
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from .base import BaseScraper, JobPosting, clean_description
from .greenhouse import _title_passes_filter, _tokenize_title, local_title_passes
from salary_rules import extract_salary_regex

logger = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/121.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json",
    "Accept-Language": "en-US,en;q=0.9",
}

BASE_URL = "https://api.smartrecruiters.com/v1/companies"
PAGE_SIZE = 100  # SmartRecruiters supports up to 100 per page

# Companies come exclusively from config.yaml's `smartrecruiters_companies` —
# there are no built-in defaults (main.py only builds this scraper when that
# key exists).
DEFAULT_SMARTRECRUITERS_COMPANIES: list = []


class SmartRecruitersScraper(BaseScraper):
    """
    Scrapes SmartRecruiters job boards using the public
    /v1/companies/{company_id}/postings JSON API. No authentication required.
    Paginates via the `offset` parameter.
    """

    source_name = "smartrecruiters"

    def __init__(self, config: dict):
        super().__init__(config)
        companies_from_config = config.get("smartrecruiters_companies")
        if companies_from_config:
            self.companies = companies_from_config
        else:
            self.companies = DEFAULT_SMARTRECRUITERS_COMPANIES

        self.session = requests.Session()
        self.session.headers.update(HEADERS)

        # Per-run cache of detail-URL → jobAd. The /postings LIST endpoint omits
        # jobAd, so we fetch each posting's detail record once for its sections.
        self._detail_cache: dict = {}

        # Profile target titles as token sets (see greenhouse.py for the shared
        # matching approach). A title matches a target when it contains all of
        # the target's significant words, order-independent.
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

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _passes_title_filter(self, title: str) -> bool:
        """Decide whether a job title is relevant for the loaded profile.

        A title passes if it matches the shared PM/product keyword baseline
        (``greenhouse._title_passes_filter``) OR contains all significant words
        of one of the profile's ``target_titles``. Profile targets can only
        broaden the baseline; they can never narrow it below the built-in filter.
        """
        if not title:
            return False
        if _title_passes_filter(title):
            return True
        if self._profile_title_token_sets:
            title_tokens = _tokenize_title(title)
            return any(toks <= title_tokens for toks in self._profile_title_token_sets)
        return False

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((requests.ConnectionError, requests.Timeout)),
        reraise=True,
    )
    def _get_postings(self, company_id: str, params: dict) -> Optional[dict]:
        """GET the SmartRecruiters postings API, return parsed JSON or None."""
        url = f"{BASE_URL}/{company_id}/postings"
        try:
            resp = self.session.get(url, params=params, timeout=20)
            if resp.status_code == 429:
                self.logger.warning(
                    "SmartRecruiters rate limited (429) for %s. Waiting 30s...",
                    company_id,
                )
                time.sleep(30)
                return None
            if resp.status_code in (403, 404, 503):
                self.logger.warning(
                    "SmartRecruiters returned %d for company %s",
                    resp.status_code, company_id,
                )
                return None
            resp.raise_for_status()
            return resp.json()
        except requests.HTTPError as e:
            self.logger.warning(
                "HTTP error from SmartRecruiters for %s: %s", company_id, e
            )
            return None
        except (requests.ConnectionError, requests.Timeout):
            raise
        except Exception as e:
            self.logger.warning(
                "Unexpected error from SmartRecruiters for %s: %s", company_id, e
            )
            return None

    def _fetch_job_ad(self, ref_url: str) -> dict:
        """Fetch a posting's detail record and return its jobAd.

        The /postings list omits jobAd; only the per-posting detail endpoint
        (posting["ref"] is that API URL) carries the description sections.
        Best-effort — returns {} on any error so the job is still stored
        (scored on title/company/location). Only successful fetches are
        cached: a transient failure must not pin {} for the rest of the run.
        """
        if not ref_url or not ref_url.startswith("http"):
            return {}
        if ref_url in self._detail_cache:
            return self._detail_cache[ref_url]
        job_ad: dict = {}
        fetched = False
        try:
            resp = self.session.get(ref_url, timeout=20)
            if resp.ok:
                job_ad = (resp.json() or {}).get("jobAd") or {}
                fetched = True
            else:
                self._log_detail_fetch_failure(
                    "SmartRecruiters: detail fetch returned %d for %s",
                    resp.status_code, ref_url,
                )
        except Exception as e:
            self._log_detail_fetch_failure(
                "SmartRecruiters: detail fetch failed for %s: %s", ref_url, e
            )
        if fetched:
            self._count_detail_fetch_success()
            self._detail_cache[ref_url] = job_ad
        return job_ad

    def _parse_posting(self, posting: dict, company_name: str,
                       company_id: str) -> Optional[JobPosting]:
        """Parse a single SmartRecruiters posting into a JobPosting.

        company_id is the config board slug (e.g. "BoschGroup") — it forms
        the public posting-page URL, which is NOT anything the API returns:
        the row's `ref` field is the API's JSON self-link."""
        title = (posting.get("name") or "").strip()
        if not title:
            return None

        # Build location string (computed before the title filter so a local
        # digital role can be evaluated against the broadened net). city/
        # country are coerced to str: local_title_passes/is_local_commuter_area is
        # now reachable even for titles that fail the strict filter, so a
        # malformed (non-string) location shape must not raise.
        loc = posting.get("location") or {}
        city = str(loc.get("city") or "")
        country = str(loc.get("country") or "")
        remote = loc.get("remote", False)
        if remote:
            location = "Remote"
        else:
            parts = [p for p in (city, country) if p]
            location = ", ".join(parts)

        # Filter: must be a product/PM role or mention AI, OR (for local
        # commuter-area jobs only) match the broadened local_target_titles.
        if not (self._passes_title_filter(title)
                or local_title_passes(title, location, self._local_locations,
                                       self._local_title_token_sets)):
            return None

        # Job URL — the PUBLIC posting page, built from board slug + posting
        # id. `ref` is the API's JSON self-link (api.smartrecruiters.com),
        # not a page a human can read: storing it (pre-2026-07-20) gave the
        # dashboard raw-JSON job links and kept the tailor's Tier-1
        # smartrecruiters extractor (which parses jobs.smartrecruiters.com
        # URLs) from ever firing. No posting id -> drop the row rather than
        # regress to an API link.
        posting_id = str(posting.get("id") or "").strip()
        if not posting_id:
            return None
        job_url = f"https://jobs.smartrecruiters.com/{company_id}/{posting_id}"

        date_posted = posting.get("releasedDate") or posting.get("createdOn")
        if date_posted and "T" in str(date_posted):
            date_posted = str(date_posted).split("T")[0]

        # Description lives inside jobAd.sections — but the /postings LIST
        # endpoint omits jobAd entirely; only the per-posting detail record
        # carries it. Fetch the detail record (posting["ref"] is that API URL)
        # when the list row has no jobAd — unless the URL is already stored
        # with a description (steady-state runs skip the GET). An inline
        # jobAd, if present, is used directly with no extra fetch.
        job_ad = posting.get("jobAd")
        if job_ad is None and job_url not in self.known_description_urls:
            job_ad = self._fetch_job_ad(posting.get("ref") or "")
        sections = (job_ad or {}).get("sections") or {}
        description = ""
        for section_key in ("jobDescription", "qualifications", "additionalInformation"):
            section = sections.get(section_key) or {}
            text = section.get("text") or ""
            if text:
                description = f"{description} {text}".strip()

        # Normalize once here (idempotent — JobPosting re-cleans and caps):
        # salary extraction needs clean text, not the sections' raw HTML.
        description = clean_description(description)
        sal_min, sal_max = extract_salary_regex(description)

        return JobPosting(
            title=title,
            company=company_name,
            location=self._normalize_location(location),
            url=job_url,
            description=description,
            salary_min=sal_min,
            salary_max=sal_max,
            date_posted=date_posted,
            source="smartrecruiters",
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def _scrape_company(self, company_config: dict) -> List[JobPosting]:
        """Fetch all postings for one company and filter titles locally.

        The API's server-side ``q`` search matched almost nothing for our narrow
        queries (a board with hundreds of postings returned 0), so we page through
        the whole board and apply the profile title filter in-process instead.
        """
        company_id = company_config["company_id"]
        company_name = company_config["company"]
        jobs: List[JobPosting] = []
        offset = 0

        while len(jobs) < self.max_results:
            params = {"limit": PAGE_SIZE, "offset": offset}
            data = self._get_postings(company_id, params)
            if not data:
                break

            content = data.get("content") or []
            total = data.get("totalFound", 0)
            if not content:
                break

            for posting in content:
                if len(jobs) >= self.max_results:
                    break
                try:
                    job = self._parse_posting(posting, company_name, company_id)
                    if job:
                        jobs.append(job)
                except Exception as e:
                    self.logger.debug(
                        "Error parsing SmartRecruiters posting for %s: %s",
                        company_name, e,
                    )

            offset += len(content)
            if offset >= total:
                break
            time.sleep(self.rate_limit)

        self.logger.info(
            "SmartRecruiters %s: %d relevant jobs", company_name, len(jobs)
        )
        return jobs

    def scrape(self, query: str) -> List[JobPosting]:
        """Scrape every configured company once.

        The board API is not keyword-searched (titles are filtered locally), so we
        only run on the first query to avoid rescraping each company once per
        query — matching the GreenhouseScraper pattern. ``scrape_all`` (inherited
        from BaseScraper) dedups the combined results by URL.
        """
        first_query = (self.config.get("search_queries") or [""])[0]
        if query != first_query:
            return []

        all_jobs: List[JobPosting] = []
        seen_urls: set = set()
        for i, company_config in enumerate(self.companies):
            try:
                self.logger.info(
                    "SmartRecruiters: scraping %s (%d/%d)",
                    company_config["company"], i + 1, len(self.companies),
                )
                for job in self._scrape_company(company_config):
                    if job.url and job.url not in seen_urls:
                        seen_urls.add(job.url)
                        all_jobs.append(job)
            except Exception as e:
                self.logger.error(
                    "SmartRecruiters: error scraping %s: %s",
                    company_config.get("company", "unknown"), e, exc_info=True,
                )
            if i < len(self.companies) - 1:
                time.sleep(self.rate_limit)

        return all_jobs
