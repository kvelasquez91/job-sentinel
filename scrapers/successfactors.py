"""
SAP SuccessFactors "Career Site Builder" scraper.

Many large manufacturers (especially in smaller commuter-area markets) host
careers on SuccessFactors CSB sites: server-rendered HTML, no auth, no JS required.

Endpoints (relative to a tenant's base_url, e.g. https://jobs.michelinman.com):
  Search:  /search/?q=&sortColumn=referencedate&sortDirection=desc&startrow={N}
           25 rows per page, <tr class="data-row"> rows.
  Detail:  href from <a class="jobTitle-link">; description lives in
           <span itemprop="description">.

Config:
  successfactors_tenants:
    - company: Michelin
      base_url: https://jobs.michelinman.com

Title filtering happens locally (profile target_titles + local_target_titles);
detail pages are fetched only for title-passing jobs to keep request volume low.
"""
import logging
import time
from typing import List, Optional
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from .base import BaseScraper, JobPosting
from .greenhouse import _title_passes_filter, _tokenize_title
from salary_rules import extract_salary_regex

logger = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/121.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml",
    "Accept-Language": "en-US,en;q=0.9",
}

PAGE_SIZE = 25  # CSB search pages return 25 rows
# Hard cap on search pages per tenant. Some CSB tenants (e.g. ZF Group) clamp an
# out-of-range startrow back to a non-empty page instead of returning an empty
# one, so "no rows" alone can't terminate pagination. 200 pages * 25 = 5000 jobs
# is well beyond any real board we filter.
MAX_PAGES = 200
SEARCH_PATH = "/search/?q=&sortColumn=referencedate&sortDirection=desc&startrow={start}"


class SuccessFactorsScraper(BaseScraper):
    """Scrapes SuccessFactors Career Site Builder tenants (HTML, no auth)."""

    def __init__(self, config: dict):
        super().__init__(config)
        self.tenants = config.get("successfactors_tenants") or []
        self.session = requests.Session()
        self.session.headers.update(HEADERS)

        target_titles = (
            config.get("profile", {}).get("target_titles", [])
            + config.get("local_target_titles", [])
        )
        self._profile_title_token_sets: List[set] = [
            toks for toks in (_tokenize_title(t) for t in target_titles) if toks
        ]

    def _passes_title_filter(self, title: str) -> bool:
        """Baseline PM/product filter OR any profile/local target title match."""
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
    def _get_html(self, url: str) -> Optional[str]:
        try:
            resp = self.session.get(url, timeout=20)
            if resp.status_code in (403, 404, 429, 503):
                self.logger.warning(
                    "SuccessFactors returned %d for %s", resp.status_code, url
                )
                return None
            resp.raise_for_status()
            return resp.text
        except requests.HTTPError as e:
            self.logger.warning("HTTP error from SuccessFactors %s: %s", url, e)
            return None
        except (requests.ConnectionError, requests.Timeout):
            raise
        except Exception as e:
            self.logger.warning("Unexpected error from SuccessFactors %s: %s", url, e)
            return None

    def _fetch_description(self, job_url: str) -> str:
        html = self._get_html(job_url)
        if not html:
            return ""
        soup = BeautifulSoup(html, "lxml")
        desc = soup.find(attrs={"itemprop": "description"})
        if desc:
            return desc.get_text(" ", strip=True)[:5000]
        job_div = soup.find("div", class_="job")
        if job_div:
            return job_div.get_text(" ", strip=True)[:5000]
        return ""

    def _scrape_tenant(self, tenant: dict) -> List[JobPosting]:
        base_url = tenant["base_url"].rstrip("/")
        company = tenant["company"]
        jobs: List[JobPosting] = []
        seen_row_urls: set = set()
        start = 0
        page = 0

        while len(jobs) < self.max_results and page < MAX_PAGES:
            page += 1
            url = base_url + SEARCH_PATH.format(start=start)
            html = self._get_html(url)
            if not html:
                break
            soup = BeautifulSoup(html, "lxml")
            rows = soup.select("tr.data-row")
            if not rows:
                break

            # Detect out-of-range startrow that a tenant clamps back to an
            # already-seen page (never returns an empty page): if this page adds
            # no new job URLs, we've stopped making progress — stop paging.
            page_urls = {
                (a.get("href") or "").strip()
                for a in soup.select("tr.data-row a.jobTitle-link")
            }
            page_urls.discard("")
            if page_urls and page_urls <= seen_row_urls:
                break
            seen_row_urls |= page_urls

            for row in rows:
                if len(jobs) >= self.max_results:
                    break
                try:
                    link = row.select_one("a.jobTitle-link")
                    if not link:
                        continue
                    title = link.get_text(strip=True)
                    if not self._passes_title_filter(title):
                        continue
                    job_url = urljoin(base_url + "/", (link.get("href") or "").strip())
                    if not job_url:
                        continue
                    loc_el = row.select_one("span.jobLocation")
                    location = loc_el.get_text(strip=True) if loc_el else ""
                    date_el = row.select_one("span.jobDate")
                    date_posted = date_el.get_text(strip=True) if date_el else None

                    time.sleep(self.rate_limit)
                    description = self._fetch_description(job_url)
                    sal_min, sal_max = extract_salary_regex(description)

                    jobs.append(JobPosting(
                        title=title,
                        company=company,
                        location=self._normalize_location(location),
                        url=job_url,
                        description=description,
                        salary_min=sal_min,
                        salary_max=sal_max,
                        date_posted=date_posted,
                        source="successfactors",
                    ))
                except Exception as e:
                    self.logger.debug(
                        "Error parsing SuccessFactors row for %s: %s", company, e
                    )

            start += PAGE_SIZE
            time.sleep(self.rate_limit)

        self.logger.info("SuccessFactors %s: %d relevant jobs", company, len(jobs))
        return jobs

    def scrape(self, query: str) -> List[JobPosting]:
        """Scrape every configured tenant once (boards are title-filtered
        locally, so only the first query triggers work — the SmartRecruiters
        pattern; scrape_all dedups by URL)."""
        first_query = (self.config.get("search_queries") or [""])[0]
        if query != first_query:
            return []

        all_jobs: List[JobPosting] = []
        seen_urls: set = set()
        for i, tenant in enumerate(self.tenants):
            try:
                self.logger.info(
                    "SuccessFactors: scraping %s (%d/%d)",
                    tenant.get("company", "?"), i + 1, len(self.tenants),
                )
                for job in self._scrape_tenant(tenant):
                    if job.url and job.url not in seen_urls:
                        seen_urls.add(job.url)
                        all_jobs.append(job)
            except Exception as e:
                self.logger.error(
                    "SuccessFactors: error scraping %s: %s",
                    tenant.get("company", "unknown"), e, exc_info=True,
                )
            if i < len(self.tenants) - 1:
                time.sleep(self.rate_limit)
        return all_jobs
