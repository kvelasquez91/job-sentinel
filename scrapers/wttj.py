"""
Welcome to the Jungle (WTTJ) job scraper.

WTTJ's job search is Algolia-backed. The frontend's client search key is origin-
restricted (not auth/UA-restricted), so a plain HTTP client works as long as the
`Origin` header is set. The credentials below are the public client values WTTJ
injects at runtime via https://www.welcometothejungle.com/api/env; we hardcode the
current ones and re-fetch /api/env (a Googlebot UA clears WTTJ's WAF *for that
endpoint*) to pick up a rotated key if Algolia ever returns 403.

The Algolia index is ~90% French, so we filter to `language:en`.

The Algolia hit carries only `profile` (the requirements list) — not the full role
text — and the scorer leans on JD text, so for each job that passes the title pre-filter
we fetch its full description from WTTJ's public REST API:
    GET https://api.welcometothejungle.com/api/v1/organizations/{org_slug}/jobs/{slug}
and read the `job.description` (+ `profile`/`key_missions` when present). The `www.*` HTML
job page is WAF-challenged (HTTP 202 JS-challenge under any volume), but the `api.*` JSON
endpoint serves the SAME job un-challenged, so it is the reliable description source. If
the API call fails we fall back to the Algolia `profile` (requirements only); the job is
still returned and scored. Section HTML is normalized to plain text downstream in
JobPosting (see scrapers.base.clean_description).

WTTJ salaries are frequently EUR/GBP; the scorer's comp rubric assumes USD, so we only
pass salary through when the currency is USD and the period is yearly.

Validation: 2026-07-10 (Algolia query → 200 + hits; api.welcometothejungle.com job → 200
+ full `description` across 5 orgs; www.* job page → 202 WAF challenge).
"""
import json
import logging
import re
import time
from typing import List, Optional

import requests
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from profile_policy import (
    WTTJ_DEFAULT_QUERIES as _DEFAULT_WTTJ_QUERIES,
    WTTJ_TITLE_KEYWORDS as _TITLE_KEYWORDS,
)
from .base import BaseScraper, JobPosting

logger = logging.getLogger(__name__)

WTTJ_BASE = "https://www.welcometothejungle.com"
# Public REST API (api.* subdomain) — serves job details un-WAF'd, unlike www.*.
WTTJ_API_BASE = "https://api.welcometothejungle.com/api/v1"

# Public client Algolia credentials (origin-restricted). Refreshed from /api/env on 403.
ALGOLIA_APP_ID = "CSEKHVMS53"
ALGOLIA_API_KEY = "4bd8f6215d0cc52b26430765769e65a0"
ALGOLIA_JOBS_INDEX = "wk_cms_jobs_production"

_ENV_URL = f"{WTTJ_BASE}/api/env"
# Googlebot UA clears WTTJ's WAF for /api/env...
_GOOGLEBOT_UA = "Mozilla/5.0 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)"
# ...but the HTML job pages are served to a normal browser UA and 403 Googlebot.
_BROWSER_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)

HITS_PER_PAGE = 50

# Broad title pre-filter (the LLM scorer does the precise relevance ranking). Anchored on
# "product manager" + product/AI leadership; intentionally excludes "product marketing",
# design, compliance, and engineering titles, which would otherwise sneak in.
# (profile_policy.WTTJ_TITLE_KEYWORDS)

# WTTJ-specific search terms (role-only). WTTJ's Algolia matches "remote"/"AI ML" as
# literal text, so the shared `search_queries` (tuned for LinkedIn-style keyword search)
# return almost nothing here. Override via `wttj_queries` in config.yaml.
# (profile_policy.WTTJ_DEFAULT_QUERIES)


def _to_float(value) -> Optional[float]:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


class WTTJScraper(BaseScraper):
    """
    Scrapes Welcome to the Jungle via its public Algolia jobs index. No auth required;
    the search key is origin-restricted, so we send the `Origin` header. English jobs
    only. Paginates via Algolia `page`/`nbPages`. Full descriptions come from each job
    page's JobPosting JSON-LD.
    """

    source_name = "wttj"

    def __init__(self, config: dict):
        super().__init__(config)
        self.app_id = ALGOLIA_APP_ID
        self.api_key = ALGOLIA_API_KEY
        self.index = ALGOLIA_JOBS_INDEX

        self.session = requests.Session()

        # Per-run cache of API-URL → description. The same job can surface across
        # several of WTTJ's queries; caching means we hit the detail API once.
        self._desc_cache: dict = {}

        # WTTJ uses its OWN role-only query set (see _DEFAULT_WTTJ_QUERIES) instead of the
        # shared search_queries, and a bounded result cap so it doesn't flood the run.
        self.queries = config.get("wttj_queries") or _DEFAULT_WTTJ_QUERIES
        self.max_results = config.get("wttj_max_results", 20)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _algolia_url(self) -> str:
        return f"https://{self.app_id}-dsn.algolia.net/1/indexes/{self.index}/query"

    def _algolia_headers(self) -> dict:
        return {
            "Content-Type": "application/json",
            "Origin": WTTJ_BASE,
            "Referer": f"{WTTJ_BASE}/",
            "x-algolia-application-id": self.app_id,
            "x-algolia-api-key": self.api_key,
        }

    def _passes_title_filter(self, title: str) -> bool:
        """
        Broad PM / product-leadership / AI-product pre-filter. WTTJ titles rarely contain
        the profile's exact multi-word target phrases (e.g. "senior product manager ai")
        as a substring, so we match on role keywords here and let the LLM scorer do the
        precise relevance ranking. Excludes product *marketing* / design / compliance roles.
        """
        if not title:
            return False
        return any(kw in title.lower() for kw in _TITLE_KEYWORDS)

    def _refresh_keys(self) -> bool:
        """Re-fetch /api/env (Googlebot UA bypasses the WAF) to pick up rotated Algolia keys."""
        try:
            resp = self.session.get(_ENV_URL, headers={"User-Agent": _GOOGLEBOT_UA}, timeout=20)
            if resp.status_code != 200:
                return False
            m = re.search(r"\{.*\}", resp.text, re.DOTALL)
            if not m:
                return False
            env = json.loads(m.group(0))
            self.app_id = env.get("ALGOLIA_APPLICATION_ID", self.app_id)
            self.api_key = env.get("ALGOLIA_API_KEY_CLIENT", self.api_key)
            self.logger.info("WTTJ: refreshed Algolia credentials from /api/env")
            return True
        except Exception as exc:
            self.logger.warning("WTTJ: failed to refresh keys from /api/env: %s", exc)
            return False

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((requests.ConnectionError, requests.Timeout)),
        reraise=True,
    )
    def _algolia_query(self, query: str, page: int) -> Optional[dict]:
        """POST one page of results to the Algolia jobs index. Returns parsed JSON or None."""
        payload = {
            "query": query,
            "hitsPerPage": HITS_PER_PAGE,
            "page": page,
            "filters": "language:en",
            "analytics": False,
        }
        try:
            resp = self.session.post(self._algolia_url(), headers=self._algolia_headers(),
                                     json=payload, timeout=20)
            if resp.status_code == 403 and self._refresh_keys():
                resp = self.session.post(self._algolia_url(), headers=self._algolia_headers(),
                                         json=payload, timeout=20)
            if resp.status_code == 429:
                self.logger.warning("WTTJ Algolia rate limited (429). Waiting 30s...")
                time.sleep(30)
                return None
            if resp.status_code != 200:
                self.logger.warning("WTTJ Algolia returned %d for '%s' page %d",
                                    resp.status_code, query, page)
                return None
            return resp.json()
        except (requests.ConnectionError, requests.Timeout):
            raise
        except Exception as exc:
            self.logger.warning("WTTJ Algolia error for '%s' page %d: %s", query, page, exc)
            return None

    def _fetch_description(self, org_slug: str, slug: str) -> str:
        """Fetch the full job description from WTTJ's public REST API.

        The `www.*` HTML job page is WAF-challenged (HTTP 202), but the `api.*`
        JSON endpoint serves the same job un-challenged. We read `job.description`
        and append `job.key_missions` / `job.profile` when present (some postings
        put responsibilities or requirements in those instead). Returns raw
        section HTML joined by blank lines — normalized to plain text downstream
        in JobPosting. Best-effort — "" on any error, so the caller can fall
        back to the Algolia `profile`. Only successful (200) fetches are
        cached: a transient WAF/5xx must not pin "" for the rest of the run.
        """
        if not (org_slug and slug):
            return ""
        api_url = f"{WTTJ_API_BASE}/organizations/{org_slug}/jobs/{slug}"
        if api_url in self._desc_cache:
            return self._desc_cache[api_url]
        text = ""
        fetched = False
        try:
            resp = self.session.get(api_url, headers={
                "User-Agent": _BROWSER_UA,
                "Accept": "application/json",
                "Accept-Language": "en-US,en;q=0.9",
            }, timeout=20)
            if resp.status_code == 200:
                job = (resp.json() or {}).get("job") or {}
                parts = [job.get(k) for k in ("description", "key_missions", "profile")]
                text = "\n\n".join(p for p in parts if p)
                fetched = True
            else:
                self._log_detail_fetch_failure(
                    "WTTJ: detail fetch returned %d for %s",
                    resp.status_code, api_url,
                )
        except Exception as exc:
            self._log_detail_fetch_failure(
                "WTTJ: detail fetch failed for %s: %s", api_url, exc
            )
        if fetched:
            self._count_detail_fetch_success()
            self._desc_cache[api_url] = text
        return text

    def _parse_hit(self, hit: dict) -> Optional[JobPosting]:
        """Map a single Algolia hit to a JobPosting (fetching the full description)."""
        title = (hit.get("name") or "").strip()
        if not title or not self._passes_title_filter(title):
            return None

        org = hit.get("organization") or {}
        company = (org.get("name") or "").strip()
        org_slug = org.get("slug") or ""
        slug = hit.get("slug") or ""
        if not (org_slug and slug):
            return None
        url = f"{WTTJ_BASE}/en/companies/{org_slug}/jobs/{slug}"

        # Location — KEEP the country (the scorer caps non-US locations), even when remote.
        offices = hit.get("offices") or []
        office = offices[0] if offices else {}
        loc_parts = [p for p in (office.get("city"), office.get("country")) if p]
        base_loc = ", ".join(loc_parts)
        if hit.get("remote") in ("fulltime", "partial", "punctual"):
            location = f"Remote ({base_loc})" if base_loc else "Remote"
        else:
            location = base_loc or "Not specified"

        # Salary — only trust USD/yearly (the scorer's comp rubric assumes USD).
        sal_min = sal_max = None
        if (hit.get("salary_currency") or "").upper() == "USD" and \
                (hit.get("salary_period") or "").lower() in ("yearly", "year"):
            sal_min = _to_float(hit.get("salary_minimum"))
            sal_max = _to_float(hit.get("salary_maximum"))

        date_posted = hit.get("published_at")
        if date_posted and "T" in str(date_posted):
            date_posted = str(date_posted).split("T")[0]

        # Full JD from the detail API; fall back to the Algolia `profile`
        # (requirements only) if the API is unavailable. Skipped entirely for
        # URLs already stored with a description (steady-state runs).
        if url in self.known_description_urls:
            description = ""
        else:
            description = self._fetch_description(org_slug, slug) or (hit.get("profile") or "")

        return JobPosting(
            title=title,
            company=company,
            location=location,
            url=url,
            description=description or "",
            salary_min=sal_min,
            salary_max=sal_max,
            date_posted=date_posted,
            source="wttj",
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def scrape(self, query: str) -> List[JobPosting]:
        """Query the WTTJ Algolia jobs index for `query`, paginating up to max_results."""
        jobs: List[JobPosting] = []
        seen_urls: set = set()
        page = 0

        while len(jobs) < self.max_results:
            data = self._algolia_query(query, page)
            if not data:
                break
            hits = data.get("hits") or []
            if not hits:
                break

            for hit in hits:
                if len(jobs) >= self.max_results:
                    break
                try:
                    job = self._parse_hit(hit)
                except Exception as exc:
                    self.logger.debug("WTTJ: error parsing hit: %s", exc)
                    continue
                if job and job.url not in seen_urls:
                    seen_urls.add(job.url)
                    jobs.append(job)
                    time.sleep(min(self.rate_limit, 1.0))  # polite delay between job-page fetches

            nb_pages = data.get("nbPages", 1)
            page += 1
            if page >= nb_pages:
                break
            time.sleep(self.rate_limit)

        self.logger.info("WTTJ: %d matching jobs for query '%s'", len(jobs), query)
        return jobs

    def scrape_all(self, queries: List[str]) -> List[JobPosting]:
        """
        Run WTTJ's OWN query set (self.queries) rather than the caller's shared
        search_queries — WTTJ's Algolia needs role-only terms. Dedup-by-URL and
        rate-limiting come from the base implementation.
        """
        return super().scrape_all(self.queries)
