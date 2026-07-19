"""
Eightfold AI job scraper.

Eightfold AI is an AI-powered talent intelligence platform. Companies using
Eightfold host their career portals at {company}.eightfold.ai or under a
custom domain. The public-facing career portal exposes a JSON API at
/api/apply/v2/jobs that requires no authentication.

Tenant config fields:
  subdomain   — used when the portal lives at {subdomain}.eightfold.ai
  base_url    — optional; overrides subdomain-based URL for custom domains
  domain      — the company's root domain, passed as the `domain` API param

Currently configured for Fluor (fluor.eightfold.ai) by default.
The owner's config.yaml adds Netflix.

Netflix notes (confirmed 2026-04-01):
  The user-facing frontend is at jobs.netflix.com (Next.js), but the
  Eightfold API backend lives at explore.jobs.netflix.net. Use:
    base_url: https://explore.jobs.netflix.net
  Job detail URLs use /careers/job/{id} on the same host.
  API responses include a canonicalPositionUrl field with the correct URL.

API endpoint:
  GET {base_url}/api/apply/v2/jobs
  Params: domain, hl, query, start, num
  Returns: {"positions": [...], "count": <total>}
"""
import datetime
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

from .base import BaseScraper, JobPosting, ats_search_queries, clean_description
from .greenhouse import _title_passes_filter, _tokenize_title
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

# Tenants come exclusively from config.yaml's `eightfold_tenants` — there are
# no built-in defaults (main.py only builds this scraper when that key exists).
DEFAULT_EIGHTFOLD_TENANTS: list = []

PAGE_SIZE = 10

# Epoch values at or above this are milliseconds (as seconds they'd be year
# ~5138); below it they are seconds. Guards against mixed units across tenants.
_EPOCH_MS_THRESHOLD = 10 ** 11


def _epoch_to_iso_date(ts) -> Optional[str]:
    """Convert an epoch timestamp to a ``YYYY-MM-DD`` string (UTC).

    Eightfold's ``t_update``/``t_create`` are epoch SECONDS; a millisecond value
    is detected by magnitude and rescaled. Returns None for missing/invalid input.
    """
    try:
        ts = int(ts)
    except (TypeError, ValueError):
        return None
    if ts <= 0:
        return None
    if ts >= _EPOCH_MS_THRESHOLD:
        ts = ts / 1000
    try:
        return datetime.datetime.fromtimestamp(
            ts, tz=datetime.timezone.utc
        ).strftime("%Y-%m-%d")
    except (OverflowError, OSError, ValueError):
        return None


class EightfoldScraper(BaseScraper):
    """
    Scrapes Eightfold AI career portals using the public /api/apply/v2/jobs
    JSON endpoint. No authentication is required. Paginate via the `start`
    parameter (0-indexed offset).
    """

    source_name = "eightfold"

    def __init__(self, config: dict):
        super().__init__(config)
        tenants_from_config = config.get("eightfold_tenants")
        if tenants_from_config:
            self.tenants = tenants_from_config
        else:
            self.tenants = DEFAULT_EIGHTFOLD_TENANTS

        self.session = requests.Session()
        self.session.headers.update(HEADERS)

        # Per-run cache of detail-URL → description. The search API returns
        # job_description empty, so we fetch each position's detail record once.
        self._detail_cache: dict = {}

        # Main-pass title gate (mirrors workday.py): baseline PM/product
        # keywords (shared with greenhouse/lever/ashby/smartrecruiters/
        # successfactors/workday) OR profile.target_titles.
        target_titles = config.get("profile", {}).get("target_titles", [])
        self._target_title_token_sets: List[set] = [
            toks for toks in (_tokenize_title(t) for t in target_titles) if toks
        ]

    def _passes_main_title(self, title: str) -> bool:
        """Baseline-or-target-titles title gate for the MAIN scrape pass —
        same semantics as greenhouse/lever/ashby/smartrecruiters/
        successfactors/workday's main pass."""
        if _title_passes_filter(title):
            return True
        toks = _tokenize_title(title)
        return any(s <= toks for s in self._target_title_token_sets)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _api_url(self, tenant: dict) -> str:
        """Build the Eightfold jobs API URL.

        Supports both the standard subdomain pattern and custom base URLs
        (e.g. Netflix at jobs.netflix.com).
        """
        base_url = tenant.get("base_url")
        if base_url:
            return f"{base_url.rstrip('/')}/api/apply/v2/jobs"
        return f"https://{tenant['subdomain']}.eightfold.ai/api/apply/v2/jobs"

    def _job_url(self, tenant: dict, pid: int) -> str:
        """Build the human-readable job detail URL."""
        base_url = tenant.get("base_url")
        if base_url:
            return f"{base_url.rstrip('/')}/jobs/{pid}"
        return (
            f"https://{tenant['subdomain']}.eightfold.ai/careers/job"
            f"?pid={pid}&domain={tenant['domain']}"
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((requests.ConnectionError, requests.Timeout)),
        reraise=True,
    )
    def _get_jobs(self, url: str, params: dict) -> Optional[dict]:
        """GET the Eightfold jobs API, return parsed JSON or None on error."""
        try:
            resp = self.session.get(url, params=params, timeout=20)
            if resp.status_code == 429:
                self.logger.warning(
                    "Eightfold rate limited (429) at %s. Waiting 30s...", url
                )
                time.sleep(30)
                return None
            if resp.status_code in (403, 404, 503):
                self.logger.warning(
                    "Eightfold returned %d for %s", resp.status_code, url
                )
                return None
            resp.raise_for_status()
            return resp.json()
        except requests.HTTPError as e:
            self.logger.warning("HTTP error from Eightfold %s: %s", url, e)
            return None
        except (requests.ConnectionError, requests.Timeout):
            raise
        except Exception as e:
            self.logger.warning("Unexpected error from Eightfold %s: %s", url, e)
            return None

    def _fetch_description(self, tenant: dict, pid) -> str:
        """Fetch a single position's detail record for its job_description HTML.

        The search API returns job_description empty; the detail endpoint
        (/api/apply/v2/jobs/{pid}?domain=...) carries the full posting.
        Best-effort — returns "" on ANY error: _get_jobs re-raises
        ConnectionError/Timeout after its tenacity retries, and letting that
        escape here aborts the whole tenant scrape (losing every job already
        parsed) or the backfill run. Only successful fetches are cached, so a
        transient failure doesn't pin "" for the rest of the run.
        """
        detail_url = f"{self._api_url(tenant)}/{pid}"
        if detail_url in self._detail_cache:
            return self._detail_cache[detail_url]
        desc = ""
        err = None
        try:
            data = self._get_jobs(detail_url, {"domain": tenant.get("domain", "")})
        except Exception as e:
            err = e
            data = None
        if data is None:
            # _get_jobs already logged the HTTP status; this line classifies
            # it as a lost description and feeds the run summary.
            self._log_detail_fetch_failure(
                "Eightfold %s: detail fetch failed for %s%s",
                tenant.get("company", "unknown"), detail_url,
                f": {err}" if err else "",
            )
        else:
            self._count_detail_fetch_success()
            desc = data.get("job_description") or ""
            self._detail_cache[detail_url] = desc
        return desc

    def _parse_position(self, pos: dict, tenant: dict) -> Optional[JobPosting]:
        """Parse a single Eightfold position entry into a JobPosting."""
        pid = pos.get("id")
        title = (pos.get("name") or "").strip()
        if not title or not pid:
            return None

        # Build location from available fields
        city = pos.get("city") or ""
        state = pos.get("state") or ""
        country = pos.get("country") or ""
        parts = [p for p in (city, state, country) if p]
        location = ", ".join(parts) if parts else (pos.get("location") or "")

        job_url = pos.get("canonicalPositionUrl") or self._job_url(tenant, pid)

        # Eightfold t_update/t_create are epoch SECONDS (not milliseconds).
        date_posted = _epoch_to_iso_date(pos.get("t_update") or pos.get("t_create"))

        # The v2 SEARCH API returns job_description as an EMPTY string; the full
        # posting HTML lives only on the per-position detail endpoint. Fetch it
        # when the list row's job_description is blank — unless the URL is
        # already stored with a description (steady-state runs skip the GET).
        raw_desc = pos.get("job_description") or ""
        if not raw_desc.strip() and job_url not in self.known_description_urls:
            raw_desc = self._fetch_description(tenant, pid)
        # Normalize once here (idempotent — JobPosting re-cleans and caps):
        # salary extraction needs clean text, and the old per-scraper strip
        # flattened the list/paragraph structure the dashboard renders.
        description = clean_description(raw_desc)
        sal_min, sal_max = extract_salary_regex(description)

        return JobPosting(
            title=title,
            company=tenant["company"],
            location=self._normalize_location(location),
            url=job_url,
            description=description,
            salary_min=sal_min,
            salary_max=sal_max,
            date_posted=date_posted,
            source="eightfold",
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def _scrape_tenant(self, tenant: dict, query: str) -> List[JobPosting]:
        """
        Scrape all pages from a single Eightfold tenant for a given query.
        Paginates until all results are fetched or max_results is reached.
        """
        api_url = self._api_url(tenant)
        company = tenant["company"]
        jobs: List[JobPosting] = []
        offset = 0
        rejected_by_title = 0

        self.logger.info("Eightfold: scraping %s for '%s'", company, query)

        while len(jobs) < self.max_results:
            params = {
                "domain": tenant["domain"],
                "hl": "en",
                "query": query,
                "start": offset,
                "num": PAGE_SIZE,
            }
            data = self._get_jobs(api_url, params)
            if not data:
                break

            positions = data.get("positions", []) or []
            total = data.get("count", 0)

            if not positions:
                break

            for pos in positions:
                if len(jobs) >= self.max_results:
                    break
                # Main-pass title gate (baseline OR target_titles) — checked
                # BEFORE _parse_position so an off-gate posting never costs a
                # per-position detail fetch.
                title = str(pos.get("name") or "").strip()
                if not self._passes_main_title(title):
                    rejected_by_title += 1
                    continue
                job = self._parse_position(pos, tenant)
                if job:
                    jobs.append(job)

            offset += len(positions)
            self.logger.debug(
                "Eightfold %s: fetched %d/%d (offset %d)",
                company, len(jobs), total, offset,
            )

            if offset >= total:
                break

            time.sleep(self.rate_limit)

        if rejected_by_title:
            self.logger.debug(
                "Eightfold %s: title gate rejected %d posting(s) for query '%s'",
                company, rejected_by_title, query,
            )
        self.logger.info(
            "Eightfold %s: %d jobs for query '%s'", company, len(jobs), query
        )
        return jobs

    def scrape(self, query: str) -> List[JobPosting]:
        """Scrape all configured Eightfold tenants for the given query."""
        all_jobs: List[JobPosting] = []
        seen_urls: set = set()

        for i, tenant in enumerate(self.tenants):
            try:
                jobs = self._scrape_tenant(tenant, query)
                for job in jobs:
                    if job.url not in seen_urls:
                        seen_urls.add(job.url)
                        all_jobs.append(job)
            except Exception as e:
                self.logger.error(
                    "Eightfold: error scraping tenant %s: %s",
                    tenant.get("company", "unknown"), e, exc_info=True,
                )

            if i < len(self.tenants) - 1:
                time.sleep(self.rate_limit)

        return all_jobs

    def scrape_all(self, queries: List[str]) -> List[JobPosting]:
        """
        Scrape all tenants for all queries. Deduplicates by URL across
        tenants and queries. Rate-limits between each (tenant × query) call.

        Eightfold's search is server-side (the `query` API param), so the
        LinkedIn-phrased queries the orchestrator passes are swapped for
        ats_search_queries() — see scrapers/base.py.
        """
        queries = ats_search_queries(self.config, fallback=queries) or queries
        all_jobs: List[JobPosting] = []
        seen_urls: set = set()

        for qi, query in enumerate(queries):
            for ti, tenant in enumerate(self.tenants):
                try:
                    self.logger.info(
                        "Eightfold [query %d/%d, tenant %d/%d]: %s @ %s",
                        qi + 1, len(queries), ti + 1, len(self.tenants),
                        query, tenant["company"],
                    )
                    jobs = self._scrape_tenant(tenant, query)
                    new_count = 0
                    for job in jobs:
                        if job.url not in seen_urls:
                            seen_urls.add(job.url)
                            all_jobs.append(job)
                            new_count += 1
                    self.logger.info(
                        "Eightfold %s / '%s': %d new jobs",
                        tenant["company"], query, new_count,
                    )
                except Exception as e:
                    self.logger.error(
                        "Eightfold: error on tenant %s query '%s': %s",
                        tenant.get("company", "unknown"), query, e, exc_info=True,
                    )

                if not (qi == len(queries) - 1 and ti == len(self.tenants) - 1):
                    time.sleep(self.rate_limit)

        self.logger.info("Eightfold total unique jobs: %d", len(all_jobs))
        self._log_detail_fetch_summary()
        return all_jobs
