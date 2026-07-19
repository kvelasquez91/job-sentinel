"""
Workday job scraper using the internal Workday JSON API.

Each company on Workday has a tenant URL and site path. The API accepts POST
requests with a JSON payload and returns paginated job listings as JSON.

Only companies confirmed to use Workday ATS are included. Companies on other
ATS platforms (Eightfold, SAP SuccessFactors, custom in-house systems, etc.)
are excluded — they require separate scrapers.

Cloudflare Bot Management bypass: tenants that return 422 on normal requests
are retried using Playwright (headless Chromium) to extract real CF cookies,
which are then injected into the requests.Session for all subsequent API calls.
Playwright is an optional dependency; non-CF tenants work without it.
"""
import datetime
import json
import logging
import re
import time
from typing import List, Optional

import requests
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from .base import DESCRIPTION_MAX_LEN, BaseScraper, JobPosting, clean_description
from .greenhouse import _title_passes_filter, _tokenize_title, is_local_commuter_area


def _parse_relative_date(text, today: Optional[datetime.date] = None) -> Optional[str]:
    """Parse Workday's relative ``postedOn`` text into a ``YYYY-MM-DD`` string.

    Handles "Posted Today", "Posted Yesterday", "Posted N Days Ago",
    "Posted N+ Days Ago", and passes through values already in ISO date form.
    Returns None for empty or unrecognized text (so expiry falls back to
    ``created_at``). Stored verbatim, these strings sorted greater than any ISO
    cutoff, so Workday jobs never expired.
    """
    if not text:
        return None
    s = str(text).strip()
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", s):
        return s
    today = today or datetime.date.today()
    low = s.lower()
    if "today" in low:
        return today.isoformat()
    if "yesterday" in low:
        return (today - datetime.timedelta(days=1)).isoformat()
    match = re.search(r"(\d+)", low)
    if match and "day" in low:
        return (today - datetime.timedelta(days=int(match.group(1)))).isoformat()
    return None

try:
    from playwright.sync_api import sync_playwright
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False

logger = logging.getLogger(__name__)


class CloudflareBlockedError(Exception):
    """Raised when Workday returns 422 (Cloudflare Bot Management blocked)."""


class WorkdayAuthError(Exception):
    """Raised when the jobs API POST is rejected at auth level (401/403),
    usually meaning the cached session cookies went stale mid-run."""

    def __init__(self, status: int, url: str):
        super().__init__(f"Workday auth rejection ({status}) at {url}")
        self.status = status

from salary_rules import MAX_BASE_SALARY

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/121.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json",
    "Accept-Language": "en-US,en;q=0.9",
    "Content-Type": "application/json",
}

# Headers used when fetching the HTML landing page to seed session cookies.
LANDING_PAGE_HEADERS = {
    "Accept": (
        "text/html,application/xhtml+xml,application/xml;"
        "q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Upgrade-Insecure-Requests": "1",
}

# Tenants come exclusively from config.yaml's `workday_tenants` — there are no
# built-in defaults (main.py only builds this scraper when that key exists).
DEFAULT_WORKDAY_TENANTS: list = []

PAGE_SIZE = 20


class WorkdayScraper(BaseScraper):
    """
    Scrapes Workday career sites for job postings using the internal JSON API.

    Each tenant is scraped independently. Results from all tenants are combined
    and deduplicated by URL in the parent scrape_all method.
    """

    source_name = "workday"

    def __init__(self, config: dict):
        super().__init__(config)
        tenants_from_config = config.get("workday_tenants")
        if tenants_from_config:
            self.tenants = tenants_from_config
        else:
            self.tenants = DEFAULT_WORKDAY_TENANTS

        self.session = requests.Session()
        self.session.headers.update(HEADERS)

        # Per-run cache of detail-URL → description. A job can match several
        # search queries and be re-parsed each time; caching means we fetch its
        # description page at most once per scrape run.
        self._detail_cache: dict = {}

        # Per-run cache of tenant_url → extra headers from a SUCCESSFUL
        # landing-page establishment (2xx/3xx). Cookies persist on the shared
        # session between queries, so the landing GET (and its 1s "look
        # human" pause) only needs to run once per tenant, not once per
        # (tenant × query) pair. _invalidate_session_headers drops the entry
        # on any POST failure, failed auth retry, or CF block, so the next
        # use re-establishes — the same recovery surface as the old per-pair
        # establishment.
        self._session_headers_cache: dict = {}

        # Monotonic timestamp of the last hit per Workday CLUSTER host, for
        # politeness pacing between (tenant × query) pairs (see _pace_cluster).
        self._cluster_last_hit: dict = {}

        # Local-area broadening (Task 11): Workday's search is server-side
        # (searchText posted to the API), so the remote AI-PM queries never
        # match local digital roles on direct-API tenants. Tenants tagged
        # local: true get one extra empty-searchText pass, kept only when
        # BOTH in the local commuter area AND matching a broadened title filter.
        self._local_locations = config.get("local_locations") or []
        self._local_title_token_sets: List[set] = [
            toks for toks in (_tokenize_title(t)
                              for t in (config.get("local_target_titles") or [])) if toks
        ]
        target_titles = config.get("profile", {}).get("target_titles", [])
        self._target_title_token_sets: List[set] = [
            toks for toks in (_tokenize_title(t) for t in target_titles) if toks
        ]

    def _passes_broad_title(self, title: str) -> bool:
        """Broadened title filter used ONLY for the local-tenant pass: baseline
        PM/product keywords OR profile.target_titles OR local_target_titles."""
        if _title_passes_filter(title):
            return True
        toks = _tokenize_title(title)
        return any(s <= toks for s in self._target_title_token_sets + self._local_title_token_sets)

    def _passes_main_title(self, title: str) -> bool:
        """Title gate for the MAIN scrape pass (every tenant, every configured
        search query): baseline PM/product keywords OR profile.target_titles —
        the same baseline-OR-target_titles semantics as greenhouse/lever/
        ashby/smartrecruiters/successfactors. Deliberately narrower than
        _passes_broad_title (no local_target_titles): that broadening exists
        only for the local-tenant pass, which supplies its own posting_filter
        and must not be re-narrowed by this gate (see _scrape_tenant_paced /
        _scrape_tenant_via_playwright)."""
        if _title_passes_filter(title):
            return True
        toks = _tokenize_title(title)
        return any(s <= toks for s in self._target_title_token_sets)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _api_url(self, tenant: dict) -> str:
        """Build the Workday jobs API endpoint URL for a tenant."""
        tenant_url = tenant["tenant_url"]
        company_slug = tenant.get("company_slug", tenant["company"].lower())
        site_path = tenant["site_path"]
        return f"https://{tenant_url}/wday/cxs/{company_slug}/{site_path}/jobs"

    def _detail_url(self, tenant: dict, external_path: str) -> str:
        """Build the Workday job detail API URL."""
        tenant_url = tenant["tenant_url"]
        company_slug = tenant.get("company_slug", tenant["company"].lower())
        site_path = tenant["site_path"]
        return f"https://{tenant_url}/wday/cxs/{company_slug}/{site_path}{external_path}"

    def _job_page_url(self, tenant: dict, external_path: str) -> str:
        """Build the human-readable job URL."""
        tenant_url = tenant["tenant_url"]
        # external_path typically looks like /job/Some-City/Job-Title_JR-12345
        return f"https://{tenant_url}/en-US{external_path}"

    def _establish_session_for_tenant(self, tenant: dict) -> tuple:
        """
        GET the careers landing page to seed session cookies (Cloudflare clearance,
        XSRF-TOKEN, etc.) before the API POST.  Returns (extra_headers, ok):
        extra headers—Referer, Origin, and any CSRF token—to attach to the
        subsequent POST call, and whether the landing GET actually completed
        (callers must not cache a failed, CSRF-less establishment).
        """
        tenant_url = tenant["tenant_url"]
        company_slug = tenant.get("company_slug", tenant["company"].lower())
        site_path = tenant["site_path"]
        landing_url = f"https://{tenant_url}/{company_slug}/{site_path}"

        extra_headers: dict = {
            "Referer": landing_url,
            "Origin": f"https://{tenant_url}",
        }

        ok = False
        try:
            resp = self.session.get(
                landing_url,
                headers=LANDING_PAGE_HEADERS,
                timeout=20,
            )
            self.logger.debug(
                "Landing-page GET %s → %d  cookies=%s",
                landing_url, resp.status_code,
                list(self.session.cookies.keys()),
            )
            # Workday sets XSRF-TOKEN (or similar) that must be echoed in the
            # X-XSRF-TOKEN request header for the API POST.
            csrf = (
                self.session.cookies.get("XSRF-TOKEN")
                or self.session.cookies.get("csrf-token")
                or self.session.cookies.get("X-CSRF-TOKEN")
                or self.session.cookies.get("wd-browser-id")
            )
            if csrf:
                extra_headers["X-XSRF-TOKEN"] = csrf
                self.logger.debug("CSRF token captured for %s", tenant["company"])
            # A 4xx/5xx landing page (CF challenge, WAF block) sets no usable
            # cookies — it must count as failed so it is never cached.
            ok = resp.ok
        except Exception as e:
            self.logger.debug(
                "Session establishment failed for %s: %s", tenant["company"], e
            )

        # Brief pause so the session looks human before we POST.
        time.sleep(1)
        return extra_headers, ok

    def _get_session_headers(self, tenant: dict, force_refresh: bool = False) -> dict:
        """Per-tenant session establishment behind a per-run cache.

        Only SUCCESSFUL establishments are cached — a swallowed landing-GET
        failure returns degraded (CSRF-less) headers, and pinning those for
        every subsequent query would turn one network blip into a whole-run
        outage for that tenant, where the old per-pair behavior retried."""
        key = tenant["tenant_url"]
        if not force_refresh:
            cached = self._session_headers_cache.get(key)
            if cached is not None:
                return cached
        headers, ok = self._establish_session_for_tenant(tenant)
        if ok:
            self._session_headers_cache[key] = headers
        else:
            self._session_headers_cache.pop(key, None)
        return headers

    def _invalidate_session_headers(self, tenant: dict) -> None:
        """Single home for cache invalidation: the next _get_session_headers
        for this tenant re-runs the landing GET, restoring the old per-pair
        establishment's recovery behavior for whatever failure preceded."""
        self._session_headers_cache.pop(tenant["tenant_url"], None)

    @staticmethod
    def _cluster_host(tenant_url: str) -> str:
        """alpha.wd5.myworkdayjobs.com → wd5.myworkdayjobs.com. Several
        tenants share a wdN cluster and anti-bot can operate at cluster
        level, so politeness pacing is keyed on the cluster host rather than
        the full tenant subdomain."""
        parts = tenant_url.split(".", 1)
        return parts[1] if len(parts) == 2 else tenant_url

    def _pace_cluster(self, tenant: dict) -> None:
        """Wait only as long as needed to keep >= rate_limit seconds between
        consecutive hits to the same Workday cluster. Consecutive
        (tenant × query) pairs almost always target different clusters, so
        this replaces the old unconditional inter-pair sleep with ~zero
        waiting while preserving same-cluster spacing."""
        last = self._cluster_last_hit.get(self._cluster_host(tenant["tenant_url"]))
        if last is not None:
            wait = self.rate_limit - (time.monotonic() - last)
            if wait > 0:
                time.sleep(wait)

    def _mark_cluster_hit(self, tenant: dict) -> None:
        self._cluster_last_hit[self._cluster_host(tenant["tenant_url"])] = (
            time.monotonic()
        )

    def _scrape_tenant_via_playwright(
        self, tenant: dict, query: str, posting_filter=None
    ) -> List["JobPosting"]:
        """
        Scrape a Cloudflare Bot Management-protected Workday tenant using
        Playwright's browser context for all HTTP requests.

        posting_filter, when given, is applied to each raw posting dict before
        parsing (and before the per-posting detail fetch).

        Playwright navigates to the careers landing page first (passing the CF
        JS challenge), then uses context.request.post() for all API calls so
        that CF cookies AND the browser TLS fingerprint stay consistent.
        """
        if not PLAYWRIGHT_AVAILABLE:
            self.logger.warning(
                "Workday %s: Cloudflare 422 but playwright not installed — skipping. "
                "Install: pip install playwright && python -m playwright install chromium",
                tenant["company"],
            )
            return []

        api_url = self._api_url(tenant)
        company = tenant["company"]
        tenant_url = tenant["tenant_url"]
        company_slug = tenant.get("company_slug", tenant["company"].lower())
        site_path = tenant["site_path"]
        landing_url = f"https://{tenant_url}/{company_slug}/{site_path}"

        self.logger.info(
            "CF bypass (Playwright): scraping %s for '%s'", company, query
        )

        jobs: List[JobPosting] = []
        rejected_by_title = 0

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(
                user_agent=HEADERS["User-Agent"],
                extra_http_headers={"Accept-Language": "en-US,en;q=0.9"},
            )
            page = context.new_page()
            try:
                page.goto(landing_url, wait_until="networkidle", timeout=45000)
            except Exception as e:
                self.logger.debug(
                    "Playwright goto timed out for %s: %s — continuing with collected cookies",
                    company, e,
                )

            # Detect Workday maintenance page — the tenant is down server-side,
            # not a bot management issue.  Bail out early to avoid silent zeros.
            if "community.workday.com/maintenance" in page.url:
                self.logger.warning(
                    "CF bypass (Playwright) %s: tenant is on Workday maintenance page (%s)"
                    " — skipping (Workday infrastructure issue, not a bot-management problem)",
                    company, page.url,
                )
                browser.close()
                return []

            offset = 0
            while len(jobs) < self.max_results:
                payload = {
                    "appliedFacets": {},
                    "limit": PAGE_SIZE,
                    "offset": offset,
                    "searchText": query,
                }

                cookies_dict = {c["name"]: c["value"] for c in context.cookies()}
                csrf = (
                    cookies_dict.get("XSRF-TOKEN")
                    or cookies_dict.get("csrf-token")
                    or cookies_dict.get("X-CSRF-TOKEN")
                    or cookies_dict.get("wd-browser-id")
                )
                req_headers = {
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                    "Accept-Language": "en-US,en;q=0.9",
                    "Referer": landing_url,
                    "Origin": f"https://{tenant_url}",
                }
                if csrf:
                    req_headers["X-XSRF-TOKEN"] = csrf

                try:
                    # Use XMLHttpRequest via page.evaluate() so the API call
                    # runs in the browser's JS context — same CF cookies AND
                    # TLS fingerprint as the navigation.  XHR is preferred over
                    # fetch() to avoid RUM/analytics wrappers that some Workday
                    # tenants inject and that can interfere with fetch calls.
                    result = page.evaluate(
                        """async ({url, payload, headers}) => {
                            return new Promise((resolve) => {
                                const xhr = new XMLHttpRequest();
                                xhr.open("POST", url, true);
                                for (const [k, v] of Object.entries(headers)) {
                                    xhr.setRequestHeader(k, v);
                                }
                                xhr.withCredentials = true;
                                xhr.onreadystatechange = function() {
                                    if (xhr.readyState !== 4) return;
                                    if (xhr.status >= 200 && xhr.status < 300) {
                                        try {
                                            resolve({status: xhr.status, data: JSON.parse(xhr.responseText)});
                                        } catch(e) {
                                            resolve({status: xhr.status, data: null});
                                        }
                                    } else {
                                        resolve({status: xhr.status, data: null});
                                    }
                                };
                                xhr.onerror = function() { resolve({status: 0, data: null}); };
                                xhr.send(JSON.stringify(payload));
                            });
                        }""",
                        {"url": api_url, "payload": payload, "headers": req_headers},
                    )
                    if result["data"] is None:
                        self.logger.warning(
                            "CF bypass (Playwright) %s: XHR returned %d",
                            company, result["status"],
                        )
                        break
                    data = result["data"]
                except Exception as e:
                    self.logger.warning(
                        "CF bypass (Playwright) %s: XHR failed: %s", company, e
                    )
                    break

                postings = data.get("jobPostings", []) or []
                total = data.get("total", 0)

                if not postings:
                    break

                for posting in postings:
                    if len(jobs) >= self.max_results:
                        break
                    if posting_filter and not posting_filter(posting):
                        continue
                    # Main-pass title gate — see _scrape_tenant_paced for why
                    # this only applies when posting_filter is None (the local
                    # pass supplies its own, broader filter).
                    if posting_filter is None and not self._passes_main_title(
                        str(posting.get("title") or "").strip()
                    ):
                        rejected_by_title += 1
                        continue
                    # Parse without the requests-session detail fetch — CF is
                    # blocking that session — then fetch the description
                    # through the browser context that passed the challenge.
                    job = self._parse_posting(posting, tenant, fetch_description=False)
                    if job:
                        if job.url not in self.known_description_urls:
                            raw = self._fetch_description_via_page(
                                page, tenant, posting.get("externalPath") or ""
                            )
                            if raw:
                                # __post_init__ already ran, so clean here.
                                job.description = clean_description(raw)[:DESCRIPTION_MAX_LEN]
                        jobs.append(job)

                offset += len(postings)
                self.logger.debug(
                    "CF bypass (Playwright) %s: fetched %d/%d jobs (offset %d)",
                    company, len(jobs), total, offset,
                )

                if offset >= total:
                    break

                time.sleep(self.rate_limit)

            browser.close()

        if rejected_by_title:
            self.logger.debug(
                "CF bypass (Playwright) %s: title gate rejected %d posting(s) for query '%s'",
                company, rejected_by_title, query,
            )
        self.logger.info(
            "CF bypass (Playwright) %s: %d jobs for query '%s'", company, len(jobs), query
        )
        return jobs

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((requests.ConnectionError, requests.Timeout)),
        reraise=True,
    )
    def _post_jobs(self, url: str, payload: dict, extra_headers: Optional[dict] = None) -> Optional[dict]:
        """POST to Workday jobs API, return parsed JSON or None on error.

        Raises WorkdayAuthError on 401/403 (stale/rejected session — the
        caller may re-establish and retry) and CloudflareBlockedError on 422;
        other failures log and return None."""
        try:
            resp = self.session.post(url, json=payload, headers=extra_headers or {}, timeout=20)
            if resp.status_code == 422:
                raise CloudflareBlockedError(f"Cloudflare Bot Management blocked (422) at {url}")
            if resp.status_code == 429:
                self.logger.warning("Workday rate limited (429) at %s. Waiting 30s...", url)
                time.sleep(30)
                return None
            if resp.status_code in (401, 403):
                self.logger.warning("Workday returned %d for %s", resp.status_code, url)
                raise WorkdayAuthError(resp.status_code, url)
            if resp.status_code in (404, 503):
                self.logger.warning("Workday returned %d for %s", resp.status_code, url)
                return None
            resp.raise_for_status()
            return resp.json()
        except (CloudflareBlockedError, WorkdayAuthError):
            raise
        except requests.HTTPError as e:
            self.logger.warning("HTTP error POSTing to %s: %s", url, e)
            return None
        except (requests.ConnectionError, requests.Timeout):
            raise
        except Exception as e:
            self.logger.warning("Unexpected error POSTing to %s: %s", url, e)
            return None

    def _fetch_description(self, tenant: dict, external_path: str) -> str:
        """Fetch the full job description from the Workday detail endpoint.

        The jobPostings LIST payload carries no description, so each posting's
        detail record must be fetched separately. Returns the RAW description
        HTML — normalization happens centrally in JobPosting/clean_description
        (stripping here flattened the list/paragraph structure the dashboard
        renders). Best-effort: "" on any error (the job is still stored,
        scored on title/etc.). Only successful fetches are cached: a transient
        429/timeout must not pin "" for the rest of the run — save_jobs can
        repair a blank stored description on a later run, but only if a later
        fetch actually succeeds.
        """
        url = self._detail_url(tenant, external_path)
        if url in self._detail_cache:
            return self._detail_cache[url]
        description = ""
        fetched = False
        try:
            resp = self.session.get(url, timeout=20)
            if resp.ok:
                data = resp.json()
                # Description is nested under jobPostingInfo.jobDescription
                job_info = data.get("jobPostingInfo", {})
                description = job_info.get("jobDescription", "") or ""
                fetched = True
            else:
                self._log_detail_fetch_failure(
                    "Workday %s: detail fetch returned %d for %s",
                    tenant["company"], resp.status_code, url,
                )
        except Exception as e:
            self._log_detail_fetch_failure(
                "Workday %s: detail fetch failed for %s: %s",
                tenant["company"], url, e,
            )
        if fetched:
            self._count_detail_fetch_success()
            self._detail_cache[url] = description
        return description

    def _fetch_description_via_page(self, page, tenant: dict, external_path: str) -> str:
        """Fetch a posting's detail record through the Playwright page.

        On Cloudflare-blocked tenants the plain requests session is exactly
        what CF is rejecting, so detail fetches must ride the same browser
        context (cookies + TLS fingerprint) that passed the JS challenge —
        the same XHR-in-page trick _scrape_tenant_via_playwright uses for the
        list POST. Shares _detail_cache with the requests path; caches only
        successful fetches. Best-effort: "" on any error.
        """
        if not external_path:
            return ""
        url = self._detail_url(tenant, external_path)
        if url in self._detail_cache:
            return self._detail_cache[url]
        description = ""
        fetched = False
        try:
            result = page.evaluate(
                """async (url) => new Promise((resolve) => {
                    const xhr = new XMLHttpRequest();
                    xhr.open("GET", url, true);
                    xhr.setRequestHeader("Accept", "application/json");
                    xhr.withCredentials = true;
                    xhr.onreadystatechange = function() {
                        if (xhr.readyState !== 4) return;
                        if (xhr.status >= 200 && xhr.status < 300) {
                            try {
                                resolve({ok: true, data: JSON.parse(xhr.responseText)});
                            } catch (e) {
                                resolve({ok: false, data: null});
                            }
                        } else {
                            resolve({ok: false, data: null});
                        }
                    };
                    xhr.onerror = function() { resolve({ok: false, data: null}); };
                    xhr.send();
                })""",
                url,
            )
            if result and result.get("ok"):
                job_info = (result.get("data") or {}).get("jobPostingInfo") or {}
                description = job_info.get("jobDescription") or ""
                fetched = True
            else:
                self._log_detail_fetch_failure(
                    "Workday %s: CF bypass (Playwright) detail fetch failed for %s",
                    tenant["company"], url,
                )
        except Exception as e:
            self._log_detail_fetch_failure(
                "Workday %s: CF bypass (Playwright) detail fetch failed for %s: %s",
                tenant["company"], url, e,
            )
        if fetched:
            self._count_detail_fetch_success()
            self._detail_cache[url] = description
        return description

    def _parse_posting(
        self, posting: dict, tenant: dict, fetch_description: bool = True
    ) -> Optional[JobPosting]:
        """Parse a single Workday jobPostings entry into a JobPosting.

        fetch_description=False skips the per-posting detail GET — used by the
        Playwright CF-bypass path, which fetches through the browser context
        instead (the plain requests session is exactly what CF is blocking).
        """
        title = posting.get("title", "").strip()
        if not title:
            return None

        external_path = posting.get("externalPath", "")
        if not external_path:
            return None

        job_url = self._job_page_url(tenant, external_path)
        location = str(posting.get("locationsText") or "").strip()
        date_posted = _parse_relative_date(posting.get("postedOn", ""))

        # bulletFields may contain salary info
        bullet_fields = posting.get("bulletFields", []) or []
        salary_text = " ".join(str(b) for b in bullet_fields if b)

        salary_min, salary_max = self._parse_salary(salary_text)

        # The list payload carries no description — fetch it from the detail
        # endpoint (cached per run). Best-effort: "" if the fetch fails.
        # Skipped for URLs already stored with a description (steady-state
        # runs would otherwise re-download every detail page for nothing).
        description = ""
        if fetch_description and job_url not in self.known_description_urls:
            description = self._fetch_description(tenant, external_path)

        return JobPosting(
            title=title,
            company=tenant["company"],
            location=self._normalize_location(location),
            url=job_url,
            description=description,
            salary_min=salary_min,
            salary_max=salary_max,
            date_posted=date_posted,
            source="workday",
        )

    @staticmethod
    def _parse_salary(text: str):
        """Parse salary range from text. Returns (min, max) floats or (None, None)."""
        import re
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
        for m in re.findall(k_pattern, text):
            values.append(float(m) * 1000)

        if not values:
            return None, None

        filtered = []
        for v in sorted(set(values)):
            if v > MAX_BASE_SALARY:
                logger.warning("_parse_salary: discarding $%.0f — exceeds MAX_BASE_SALARY cap", v)
            elif v >= 20_000:
                filtered.append(v)
        if not filtered:
            return None, None
        return (filtered[0], filtered[0]) if len(filtered) == 1 else (filtered[0], filtered[-1])

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def _scrape_tenant(
        self, tenant: dict, query: str, posting_filter=None
    ) -> List[JobPosting]:
        """
        Scrape all pages from a single Workday tenant for a given query.
        Paginates until all results are fetched or max_results is reached.

        posting_filter, when given, is applied to each raw posting dict BEFORE
        parsing — and therefore before the per-posting detail fetch, so
        callers that discard most postings (the local-tenant pass) don't pay
        a detail GET for every discard.

        Cluster politeness lives here (pace on entry, mark on exit) so every
        caller gets it without repeating the pair at each call site.
        """
        self._pace_cluster(tenant)
        try:
            return self._scrape_tenant_paced(tenant, query, posting_filter)
        finally:
            # Mark even on failure — the attempt itself hit the cluster.
            self._mark_cluster_hit(tenant)

    def _scrape_tenant_paced(
        self, tenant: dict, query: str, posting_filter=None
    ) -> List[JobPosting]:
        api_url = self._api_url(tenant)
        company = tenant["company"]
        jobs: List[JobPosting] = []
        offset = 0
        rejected_by_title = 0

        self.logger.info("Workday: scraping %s for '%s'", company, query)

        # Seed session cookies / CSRF token via a browser-like GET first
        # (cached per tenant — cookies persist on the shared session).
        extra_headers = self._get_session_headers(tenant)
        auth_retry_done = False

        try:
            while len(jobs) < self.max_results:
                payload = {
                    "appliedFacets": {},
                    "limit": PAGE_SIZE,
                    "offset": offset,
                    "searchText": query,
                }
                try:
                    data = self._post_jobs(api_url, payload, extra_headers=extra_headers)
                except WorkdayAuthError as auth_err:
                    if auth_retry_done:
                        # The freshly established session was rejected too —
                        # drop it so the tenant's next query starts clean, and
                        # give up this query like the old flow did.
                        self._invalidate_session_headers(tenant)
                        break
                    # Stale session (cookies can expire mid-run): one paced
                    # fresh establishment + retry, so a tenant that would have
                    # succeeded under the old per-pair establishment still
                    # yields its jobs.
                    auth_retry_done = True
                    self.logger.info(
                        "Workday %s: %s — re-establishing session and retrying once",
                        company, auth_err,
                    )
                    time.sleep(self.rate_limit)
                    extra_headers = self._get_session_headers(tenant, force_refresh=True)
                    try:
                        data = self._post_jobs(api_url, payload, extra_headers=extra_headers)
                    except WorkdayAuthError:
                        self._invalidate_session_headers(tenant)
                        break

                if not data:
                    if data is None:
                        # Request-level failure (404/503/429/HTTP error), not
                        # an empty page: don't trust the cached session for
                        # the tenant's next query — re-establishing on failure
                        # matches the old per-pair recovery surface.
                        self._invalidate_session_headers(tenant)
                    break

                postings = data.get("jobPostings", []) or []
                total = data.get("total", 0)

                if not postings:
                    break

                for posting in postings:
                    if len(jobs) >= self.max_results:
                        break
                    if posting_filter and not posting_filter(posting):
                        continue
                    # Main-pass title gate (baseline OR target_titles) — checked
                    # BEFORE _parse_posting so an off-gate posting never costs a
                    # per-posting detail fetch. Only the main pass (posting_filter
                    # is None) is gated here: the local-tenant pass supplies its
                    # own posting_filter (the broader _passes_broad_title) and
                    # must not be narrowed further by this stricter gate.
                    if posting_filter is None and not self._passes_main_title(
                        str(posting.get("title") or "").strip()
                    ):
                        rejected_by_title += 1
                        continue
                    job = self._parse_posting(posting, tenant)
                    if job:
                        jobs.append(job)

                offset += len(postings)
                self.logger.debug(
                    "Workday %s: fetched %d/%d jobs (offset %d)",
                    company, len(jobs), total, offset,
                )

                # Stop if we've fetched all available results
                if offset >= total:
                    break

                time.sleep(self.rate_limit)
        except CloudflareBlockedError as cf_err:
            # Don't let a CF-blocked session linger in the cache — the
            # Playwright path uses its own browser context, and any later
            # plain-requests use should start fresh.
            self._invalidate_session_headers(tenant)
            self.logger.warning(
                "Workday %s: %s — switching to Playwright CF bypass", company, cf_err
            )
            return self._scrape_tenant_via_playwright(
                tenant, query, posting_filter=posting_filter
            )

        if rejected_by_title:
            self.logger.debug(
                "Workday %s: title gate rejected %d posting(s) for query '%s'",
                company, rejected_by_title, query,
            )
        self.logger.info("Workday %s: %d jobs for query '%s'", company, len(jobs), query)
        return jobs

    def scrape(self, query: str) -> List[JobPosting]:
        """
        Scrape all configured Workday tenants for the given query.
        Paces requests per cluster host (see _pace_cluster).
        """
        all_jobs: List[JobPosting] = []
        seen_urls: set = set()

        for tenant in self.tenants:
            try:
                jobs = self._scrape_tenant(tenant, query)
                for job in jobs:
                    if job.url not in seen_urls:
                        seen_urls.add(job.url)
                        all_jobs.append(job)
            except Exception as e:
                self.logger.error(
                    "Workday: error scraping tenant %s: %s",
                    tenant.get("company", "unknown"), e, exc_info=True,
                )

        return all_jobs

    def scrape_all(self, queries: List[str]) -> List[JobPosting]:
        """
        Scrape all tenants for all queries. Deduplicates by URL across
        tenants and queries. Paces requests per cluster host (see _pace_cluster).
        """
        all_jobs: List[JobPosting] = []
        seen_urls: set = set()

        for qi, query in enumerate(queries):
            for ti, tenant in enumerate(self.tenants):
                try:
                    self.logger.info(
                        "Workday [query %d/%d, tenant %d/%d]: %s @ %s",
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
                        "Workday %s / '%s': %d new jobs", tenant["company"], query, new_count
                    )
                except Exception as e:
                    self.logger.error(
                        "Workday: error on tenant %s query '%s': %s",
                        tenant.get("company", "unknown"), query, e, exc_info=True,
                    )

        self._scrape_local_tenants(all_jobs, seen_urls)

        self.logger.info("Workday total unique jobs: %d", len(all_jobs))
        self._log_detail_fetch_summary()
        return all_jobs

    def _scrape_local_tenants(self, all_jobs: List[JobPosting], seen_urls: set) -> None:
        """Extra filter-locally pass for tenants tagged ``local: true``.

        Workday's search is server-side, so the remote AI-PM queries above
        never surface local digital roles on direct-API tenants (e.g. a
        "Director, Digital Growth Marketing & CRM" role in a commuter-area
        city). This runs ONE additional pass per local tenant with an empty
        searchText — reusing ``_scrape_tenant``'s existing paging +
        CF/Playwright fallback, capped by ``self.max_results`` — and keeps
        only jobs that are BOTH in the local commuter area AND pass the
        broadened title filter. Mutates
        ``all_jobs``/``seen_urls`` in place, deduped by URL against jobs
        already collected by the normal query loop.
        """
        local_tenants = [t for t in self.tenants if t.get("local")]

        def _local_prefilter(posting: dict) -> bool:
            # Mirror the post-parse filter on the raw list fields so the
            # per-posting detail fetch only runs for postings that can
            # actually be kept — the empty-searchText pass returns the whole
            # board and discards most of it.
            title = str(posting.get("title") or "").strip()
            location = self._normalize_location(
                str(posting.get("locationsText") or "").strip()
            )
            return bool(title) and is_local_commuter_area(location, self._local_locations) \
                and self._passes_broad_title(title)

        for i, tenant in enumerate(local_tenants):
            try:
                self.logger.info(
                    "Workday local pass [%d/%d]: %s", i + 1, len(local_tenants), tenant["company"]
                )
                local_hits = [
                    j for j in self._scrape_tenant(
                        tenant, "", posting_filter=_local_prefilter
                    )
                    if is_local_commuter_area(j.location or "", self._local_locations)
                    and self._passes_broad_title(j.title or "")
                ]
                new_count = 0
                for job in local_hits:
                    if job.url not in seen_urls:
                        seen_urls.add(job.url)
                        all_jobs.append(job)
                        new_count += 1
                self.logger.info(
                    "Workday local pass %s: %d new jobs", tenant["company"], new_count
                )
            except Exception as e:
                self.logger.error(
                    "Workday: error on local pass for tenant %s: %s",
                    tenant.get("company", "unknown"), e, exc_info=True,
                )
