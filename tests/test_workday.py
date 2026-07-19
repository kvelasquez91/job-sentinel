"""Tests for Workday relative-date parsing.

Regression: Workday's postedOn is relative text ("Posted 4 Days Ago") stored
verbatim in date_posted. String-comparing that against an ISO cutoff always
sorted greater, so Workday jobs never expired (1,145 stuck 'new').
"""
import datetime

from scrapers.workday import _parse_relative_date

TODAY = datetime.date(2026, 7, 2)


def test_posted_today():
    assert _parse_relative_date("Posted Today", today=TODAY) == "2026-07-02"


def test_posted_yesterday():
    assert _parse_relative_date("Posted Yesterday", today=TODAY) == "2026-07-01"


def test_posted_n_days_ago():
    assert _parse_relative_date("Posted 4 Days Ago", today=TODAY) == "2026-06-28"


def test_posted_30_plus_days_ago():
    assert _parse_relative_date("Posted 30+ Days Ago", today=TODAY) == "2026-06-02"


def test_iso_date_passes_through():
    assert _parse_relative_date("2026-06-15", today=TODAY) == "2026-06-15"


def test_unrecognized_returns_none():
    assert _parse_relative_date("", today=TODAY) is None
    assert _parse_relative_date(None, today=TODAY) is None
    assert _parse_relative_date("Whenever", today=TODAY) is None


# ---------------------------------------------------------------------------
# Task 11: local-tenant filter-locally pass (local commuter area).
#
# Workday's search is server-side (POST searchText=query), so the remote
# AI-PM queries never match local digital roles on direct-API tenants (e.g.
# a retail chain's "Director, Digital Growth Marketing & CRM" in Riverton,
# IL). Tenants tagged local: true get one extra pass with an empty
# searchText (reusing _scrape_tenant's existing paging + CF/Playwright
# fallback), keeping only jobs that are BOTH in the local commuter area AND
# pass a broadened title filter (baseline PM OR profile.target_titles OR
# local_target_titles). Remote-company jobs (and non-local tenants) must
# keep the strict filter.
# ---------------------------------------------------------------------------
from scrapers.base import JobPosting
from scrapers.workday import WorkdayScraper
from policy_fixtures import patch_state_pattern, patch_greenhouse_gates

LOCAL_LOCATIONS = [
    "Springfield", "Riverton", "Fairview", "Lakewood", "Cedar Falls",
]
LOCAL_TARGET_TITLES = [
    "Digital Product Manager",
    "Director Digital Transformation",
    "Director Digital",
    "Head of Innovation",
    "Director Innovation",
    "IT Product Manager",
    "Product Owner",
    "Director Data Analytics",
    "Head of Digital",
    "AI Product Manager",
]

WORKDAY_CONFIG = {
    "profile": {
        "target_titles": [
            "Senior Product Manager AI",
            "Director Product AI",
            "Head of AI",
            "VP Product",
            "Product Lead GenAI",
        ]
    },
    "search_queries": ["Senior Product Manager AI remote"],
    "local_locations": LOCAL_LOCATIONS,
    "local_target_titles": LOCAL_TARGET_TITLES,
    "workday_tenants": [
        {
            "company": "Denny's",
            "tenant_url": "dennys.wd1.myworkdayjobs.com",
            "company_slug": "dennys",
            "site_path": "Dennys",
            "local": True,
        },
        {
            "company": "NVIDIA",
            "tenant_url": "nvidia.wd5.myworkdayjobs.com",
            "company_slug": "nvidia",
            "site_path": "NVIDIAExternalCareerSite",
        },
    ],
    "scraping": {"max_results_per_source": 50, "rate_limit_seconds": 0},
}


def _job(title, location, url):
    return JobPosting(
        title=title,
        company="Denny's",
        location=location,
        url=url,
        description="",
        source="workday",
    )


def test_local_pass_captures_local_digital_role(monkeypatch):
    """A local:true tenant gets an extra empty-searchText pass. Local-area
    jobs with a broad-title match (baseline PM OR target_titles OR
    local_target_titles) are kept; a decoy with the SAME title but a
    non-local location is dropped; a local job with an irrelevant title is
    dropped (no broad-title match)."""
    patch_state_pattern(monkeypatch)
    sc = WorkdayScraper(WORKDAY_CONFIG)

    local_digital_role = _job(
        "Director, Digital Growth Marketing & CRM",
        "Riverton, IL - 203 E. Main Street (29319)",
        "https://dennys.wd1.myworkdayjobs.com/en-US/Dennys/job/local-digital",
    )
    remote_decoy_same_title = _job(
        "Director, Digital Growth Marketing & CRM",
        "New York, NY",
        "https://dennys.wd1.myworkdayjobs.com/en-US/Dennys/job/remote-decoy",
    )
    local_irrelevant_title = _job(
        "Line Cook",
        "Springfield, IL",
        "https://dennys.wd1.myworkdayjobs.com/en-US/Dennys/job/line-cook",
    )

    empty_search_results = [local_digital_role, remote_decoy_same_title, local_irrelevant_title]

    def fake_scrape_tenant(tenant, query, posting_filter=None):
        # Normal query pass (both tenants) — no results; the test only cares
        # about the local pass triggered for the local:true tenant.
        if query != "":
            return []
        assert tenant["company"] == "Denny's", "only the local:true tenant should get an empty-searchText pass"
        return empty_search_results

    sc._scrape_tenant = fake_scrape_tenant

    jobs = sc.scrape_all(["Senior Product Manager AI remote"])
    urls = {j.url for j in jobs}

    assert local_digital_role.url in urls
    assert remote_decoy_same_title.url not in urls, "same local title but non-local location must be dropped"
    assert local_irrelevant_title.url not in urls, "local-area job with no broad-title match must be dropped"


def test_non_local_tenant_has_no_local_pass():
    """A tenant without local: true (e.g. NVIDIA) must NOT get the extra
    empty-searchText pass — _scrape_tenant(tenant, "") is never called for it."""
    sc = WorkdayScraper(WORKDAY_CONFIG)

    empty_search_calls = []

    def fake_scrape_tenant(tenant, query, posting_filter=None):
        if query == "":
            empty_search_calls.append(tenant["company"])
            return []
        # Normal query pass — return nothing so the test focuses on the
        # local-pass call pattern, not on parsing.
        return []

    sc._scrape_tenant = fake_scrape_tenant

    sc.scrape_all(["Senior Product Manager AI remote"])

    assert "Denny's" in empty_search_calls, "local:true tenant should get the local pass"
    assert "NVIDIA" not in empty_search_calls, "non-local tenant must not get the local pass"


# ---------------------------------------------------------------------------
# Dogfood finding: Eightfold applies NO title gate at all, and Workday's
# _passes_broad_title gate applies ONLY to the optional local-tenant pass —
# the MAIN tenant scrape (the normal query loop, every tenant) saved
# everything the tenant search API returned, contradicting the documented
# "every ATS scraper is gated" claim. _passes_main_title applies the same
# baseline-OR-target_titles semantics as greenhouse/lever/ashby/
# smartrecruiters/successfactors to Workday's MAIN pass — BEFORE the
# per-posting detail fetch, so an off-gate posting never costs an API call.
# ---------------------------------------------------------------------------
import requests as _requests_main_gate  # noqa: E402  (see _FakeResp below, defined later in this module)


def _main_gate_single_tenant_config(rate_limit=0):
    return {
        "scraping": {"rate_limit_seconds": rate_limit, "max_results_per_source": 50},
        "workday_tenants": [
            {"company": "AlphaCo", "tenant_url": "alpha.wd5.myworkdayjobs.com",
             "site_path": "External", "company_slug": "alphaco"},
        ],
    }


def _main_gate_posting(title, jr="JR1"):
    return {
        "title": title,
        "externalPath": f"/job/Remote/{jr}",
        "locationsText": "Remote",
        "postedOn": "Posted Today",
        "bulletFields": [],
    }


class _MainGateFakeResp:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self.ok = 200 <= status_code < 300
        self._payload = payload if payload is not None else {}

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise _requests_main_gate.HTTPError(f"HTTP {self.status_code}")


def _main_gate_is_landing(url):
    return "/wday/cxs/" not in url


def test_main_pass_rejects_off_gate_title(monkeypatch):
    """The MAIN scrape pass (normal query loop) must reject a title matching
    neither the baseline gate nor profile.target_titles — and reject it
    BEFORE the per-posting detail fetch (no wasted API call)."""
    patch_greenhouse_gates(monkeypatch)
    sc = WorkdayScraper(_main_gate_single_tenant_config())
    monkeypatch.setattr("scrapers.workday.time.sleep", lambda s: None)

    posting = _main_gate_posting("Staff Accountant")
    get_calls = []

    def fake_get(url, **kwargs):
        get_calls.append(url)
        return _MainGateFakeResp(200)

    monkeypatch.setattr(sc.session, "get", fake_get)
    monkeypatch.setattr(
        sc.session, "post",
        lambda url, **kw: _MainGateFakeResp(200, {"jobPostings": [posting], "total": 1}),
    )

    jobs = sc.scrape_all(["query one"])

    assert jobs == []
    assert len(get_calls) == 1, (
        f"only the landing-page GET should fire — the detail fetch must be "
        f"skipped for a title the gate rejects, got {get_calls}"
    )


def test_main_pass_accepts_baseline_title(monkeypatch):
    """A title matching the baseline PM/product gate passes the MAIN pass."""
    patch_greenhouse_gates(monkeypatch)
    sc = WorkdayScraper(_main_gate_single_tenant_config())
    monkeypatch.setattr("scrapers.workday.time.sleep", lambda s: None)

    posting = _main_gate_posting("Senior Product Manager, AI")

    def fake_get(url, **kwargs):
        if _main_gate_is_landing(url):
            return _MainGateFakeResp(200)
        return _MainGateFakeResp(200, {"jobPostingInfo": {"jobDescription": "JD"}})

    monkeypatch.setattr(sc.session, "get", fake_get)
    monkeypatch.setattr(
        sc.session, "post",
        lambda url, **kw: _MainGateFakeResp(200, {"jobPostings": [posting], "total": 1}),
    )

    jobs = sc.scrape_all(["query one"])

    assert len(jobs) == 1
    assert jobs[0].title == "Senior Product Manager, AI"


def test_main_pass_accepts_target_titles_with_empty_baseline(monkeypatch):
    """With an EMPTY baseline gate, a title that token-matches
    profile.target_titles must still pass the MAIN pass."""
    import scrapers.greenhouse as gh
    monkeypatch.setattr(gh, "PRODUCT_TITLE_KEYWORDS", [])
    monkeypatch.setattr(gh, "AI_TITLE_KEYWORDS", [])

    cfg = _main_gate_single_tenant_config()
    cfg["profile"] = {"target_titles": ["Director Digital Transformation"]}
    sc = WorkdayScraper(cfg)
    monkeypatch.setattr("scrapers.workday.time.sleep", lambda s: None)

    posting = _main_gate_posting("Director of Digital Transformation, EMEA")

    def fake_get(url, **kwargs):
        if _main_gate_is_landing(url):
            return _MainGateFakeResp(200)
        return _MainGateFakeResp(200, {"jobPostingInfo": {"jobDescription": "JD"}})

    monkeypatch.setattr(sc.session, "get", fake_get)
    monkeypatch.setattr(
        sc.session, "post",
        lambda url, **kw: _MainGateFakeResp(200, {"jobPostings": [posting], "total": 1}),
    )

    jobs = sc.scrape_all(["query one"])

    assert len(jobs) == 1


def test_main_pass_yields_nothing_when_gate_and_target_titles_empty(monkeypatch):
    """Both the baseline gate AND profile.target_titles empty (the neutral-
    tree condition) means the MAIN pass yields NOTHING — dormant until
    configured, same semantics as greenhouse."""
    import scrapers.greenhouse as gh
    monkeypatch.setattr(gh, "PRODUCT_TITLE_KEYWORDS", [])
    monkeypatch.setattr(gh, "AI_TITLE_KEYWORDS", [])

    sc = WorkdayScraper(_main_gate_single_tenant_config())  # no profile.target_titles
    monkeypatch.setattr("scrapers.workday.time.sleep", lambda s: None)

    posting = _main_gate_posting("AI Product Manager")

    monkeypatch.setattr(sc.session, "get", lambda url, **kw: _MainGateFakeResp(200))
    monkeypatch.setattr(
        sc.session, "post",
        lambda url, **kw: _MainGateFakeResp(200, {"jobPostings": [posting], "total": 1}),
    )

    jobs = sc.scrape_all(["query one"])

    assert jobs == []


# ---------------------------------------------------------------------------
# Regression: locationsText present-but-None.
#
# Workday's API can return {"locationsText": null} (key present, value None).
# dict.get("locationsText", "") only substitutes the default when the key is
# ABSENT — a present None value passes straight through, so None.strip() raised
# AttributeError and crashed _parse_posting for every posting on that tenant.
# ---------------------------------------------------------------------------
def test_parse_posting_null_location_does_not_crash():
    sc = WorkdayScraper(WORKDAY_CONFIG)
    sc._fetch_description = lambda tenant, external_path: ""  # no network in unit test
    tenant = {
        "company": "Denny's",
        "tenant_url": "dennys.wd1.myworkdayjobs.com",
        "company_slug": "dennys",
        "site_path": "Dennys",
    }
    posting = {
        "title": "Director, Digital Growth Marketing & CRM",
        "externalPath": "/job/Riverton/Director_JR-123",
        "locationsText": None,  # key present, value None
        "postedOn": "Posted 2 Days Ago",
    }

    job = sc._parse_posting(posting, tenant)

    assert job is not None
    assert job.location == ""


# ---------------------------------------------------------------------------
# Regression: descriptions were never captured.
#
# The jobPostings LIST payload carries no description; each posting's detail
# record must be fetched separately. _fetch_description() existed but was never
# called — _parse_posting hardcoded description="" — so every Workday job was
# stored blank ("No description available" in the dashboard). _parse_posting
# must now call _fetch_description and use its result.
# ---------------------------------------------------------------------------
def test_parse_posting_populates_description_from_detail():
    sc = WorkdayScraper(WORKDAY_CONFIG)
    tenant = {
        "company": "NVIDIA",
        "tenant_url": "nvidia.wd5.myworkdayjobs.com",
        "company_slug": "nvidia",
        "site_path": "NVIDIAExternalCareerSite",
    }
    calls = []

    def fake_fetch(t, external_path):
        calls.append(external_path)
        return "Build AI products end to end."

    sc._fetch_description = fake_fetch
    posting = {
        "title": "Senior Product Manager, AI",
        "externalPath": "/job/Santa-Clara/PM_JR-1",
        "locationsText": "Santa Clara, CA",
        "postedOn": "Posted 2 Days Ago",
    }

    job = sc._parse_posting(posting, tenant)

    assert job is not None
    assert job.description == "Build AI products end to end."
    assert calls == ["/job/Santa-Clara/PM_JR-1"], "must fetch the detail record for this posting"


def test_fetch_description_returns_raw_html_and_caches(monkeypatch):
    """The fetcher returns RAW HTML (cleaning is central, in JobPosting —
    per-scraper stripping flattened the list/paragraph structure) and a job
    matching several queries must hit the detail endpoint at most once."""
    sc = WorkdayScraper(WORKDAY_CONFIG)
    tenant = {
        "company": "NVIDIA",
        "tenant_url": "nvidia.wd5.myworkdayjobs.com",
        "company_slug": "nvidia",
        "site_path": "NVIDIAExternalCareerSite",
    }

    class FakeResp:
        ok = True

        @staticmethod
        def json():
            return {"jobPostingInfo": {"jobDescription": "<p>Full <b>JD</b> text</p>"}}

    get_calls = []

    def fake_get(url, timeout=20):
        get_calls.append(url)
        return FakeResp()

    monkeypatch.setattr(sc.session, "get", fake_get)

    d1 = sc._fetch_description(tenant, "/job/CA/PM_JR-2")
    d2 = sc._fetch_description(tenant, "/job/CA/PM_JR-2")

    assert d1 == "<p>Full <b>JD</b> text</p>"
    assert d2 == d1
    assert len(get_calls) == 1, "second call must be served from cache"


def test_fetch_description_failure_not_cached(monkeypatch):
    """A transient 429/timeout must NOT be cached as '' for the run: main.py
    never repairs description on a duplicate URL, so a negative-cached failure
    stores the job permanently blank."""
    sc = WorkdayScraper(WORKDAY_CONFIG)
    tenant = {
        "company": "NVIDIA",
        "tenant_url": "nvidia.wd5.myworkdayjobs.com",
        "company_slug": "nvidia",
        "site_path": "NVIDIAExternalCareerSite",
    }

    class _Bad:
        ok = False
        status_code = 429

    class _Good:
        ok = True

        @staticmethod
        def json():
            return {"jobPostingInfo": {"jobDescription": "JD"}}

    calls = []

    def fake_get(url, timeout=20):
        calls.append(url)
        return _Bad() if len(calls) == 1 else _Good()

    monkeypatch.setattr(sc.session, "get", fake_get)
    assert sc._fetch_description(tenant, "/job/CA/PM_JR-3") == ""
    assert sc._fetch_description(tenant, "/job/CA/PM_JR-3") == "JD"
    assert len(calls) == 2, "failure must not be negative-cached"


def test_parse_posting_fetch_description_false_skips_network():
    """The Playwright CF-bypass path parses with fetch_description=False —
    firing the plain-requests fetch there is a doomed GET against the very
    session Cloudflare just blocked."""
    sc = WorkdayScraper(WORKDAY_CONFIG)
    sc._fetch_description = lambda tenant, external_path: (_ for _ in ()).throw(
        AssertionError("must not fetch when fetch_description=False")
    )
    tenant = {
        "company": "NVIDIA",
        "tenant_url": "nvidia.wd5.myworkdayjobs.com",
        "company_slug": "nvidia",
        "site_path": "NVIDIAExternalCareerSite",
    }
    posting = {
        "title": "Senior Product Manager, AI",
        "externalPath": "/job/Santa-Clara/PM_JR-4",
        "locationsText": "Santa Clara, CA",
        "postedOn": "Posted 2 Days Ago",
    }

    job = sc._parse_posting(posting, tenant, fetch_description=False)

    assert job is not None
    assert job.description == ""


def test_parse_posting_skips_fetch_for_urls_already_described():
    """Steady-state daily runs: a URL already stored WITH a description must
    not cost a detail GET again (main.py seeds known_description_urls)."""
    sc = WorkdayScraper(WORKDAY_CONFIG)
    tenant = {
        "company": "NVIDIA",
        "tenant_url": "nvidia.wd5.myworkdayjobs.com",
        "company_slug": "nvidia",
        "site_path": "NVIDIAExternalCareerSite",
    }
    posting = {
        "title": "Senior Product Manager, AI",
        "externalPath": "/job/Santa-Clara/PM_JR-5",
        "locationsText": "Santa Clara, CA",
        "postedOn": "Posted 2 Days Ago",
    }
    sc.known_description_urls = {sc._job_page_url(tenant, "/job/Santa-Clara/PM_JR-5")}
    sc._fetch_description = lambda tenant, external_path: (_ for _ in ()).throw(
        AssertionError("must not fetch a URL already stored with a description")
    )

    job = sc._parse_posting(posting, tenant)

    assert job is not None
    assert job.description == ""


# ---------------------------------------------------------------------------
# Perf: per-tenant session caching + per-cluster rate pacing.
#
# Run 137 measured Workday at 25.4 min, of which ~14 min was (a) re-running
# the landing-page GET + fixed 1s "look human" sleep for every (tenant x
# query) pair when cookies already persist on the shared session, and (b) an
# unconditional 2s sleep between consecutive pairs that almost always target
# DIFFERENT tenant hosts. The fixes: establish the session once per tenant
# per run (invalidating on auth failure, with one fresh-session retry), and
# key the inter-pair politeness wait on the shared Workday cluster host
# (everything after the first dot of tenant_url) so same-cluster pacing is
# preserved while cross-cluster waits drop to zero.
# ---------------------------------------------------------------------------
import requests as _requests


class _FakeResp:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self.ok = 200 <= status_code < 300
        self._payload = payload if payload is not None else {}

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise _requests.HTTPError(f"HTTP {self.status_code}")


def _two_tenant_config(rate_limit=7, second_tenant_url="beta.wd1.myworkdayjobs.com"):
    return {
        "scraping": {"rate_limit_seconds": rate_limit, "max_results_per_source": 50},
        "workday_tenants": [
            {"company": "AlphaCo", "tenant_url": "alpha.wd5.myworkdayjobs.com",
             "site_path": "External", "company_slug": "alphaco"},
            {"company": "BetaCo", "tenant_url": second_tenant_url,
             "site_path": "Careers", "company_slug": "betaco"},
        ],
    }


def _is_landing(url):
    return "/wday/cxs/" not in url


def test_landing_session_established_once_per_tenant(monkeypatch):
    """The landing-page GET (cookie/CSRF seed) must run once per TENANT per
    run, not once per (tenant x query) pair — cookies persist on the shared
    requests.Session between queries."""
    sc = WorkdayScraper(_two_tenant_config())
    monkeypatch.setattr("scrapers.workday.time.sleep", lambda s: None)

    landing_gets = []

    def fake_get(url, **kwargs):
        if _is_landing(url):
            landing_gets.append(url)
        return _FakeResp(200)

    monkeypatch.setattr(sc.session, "get", fake_get)
    monkeypatch.setattr(
        sc.session, "post",
        lambda url, **kw: _FakeResp(200, {"jobPostings": [], "total": 0}),
    )

    sc.scrape_all(["query one", "query two"])

    assert len(landing_gets) == 2, (
        f"expected 1 landing GET per tenant across all queries, got {len(landing_gets)}: {landing_gets}"
    )


def test_failed_landing_get_is_not_cached(monkeypatch):
    """A failed establishment must not pin degraded (CSRF-less) headers for
    the rest of the run: the next query for that tenant retries the landing
    GET, matching today's per-pair retry behavior."""
    cfg = _two_tenant_config()
    cfg["workday_tenants"] = cfg["workday_tenants"][:1]
    sc = WorkdayScraper(cfg)
    monkeypatch.setattr("scrapers.workday.time.sleep", lambda s: None)

    landing_attempts = []

    def fake_get(url, **kwargs):
        if _is_landing(url):
            landing_attempts.append(url)
            if len(landing_attempts) == 1:
                raise _requests.ConnectionError("network blip")
        return _FakeResp(200)

    monkeypatch.setattr(sc.session, "get", fake_get)
    monkeypatch.setattr(
        sc.session, "post",
        lambda url, **kw: _FakeResp(200, {"jobPostings": [], "total": 0}),
    )

    sc.scrape_all(["query one", "query two"])

    assert len(landing_attempts) == 2, (
        "failed landing GET must be retried on the tenant's next query, not cached"
    )


def test_stale_cached_session_reestablished_on_403(monkeypatch):
    """A 403 on the API POST means the cached session went stale: the scraper
    must re-establish (fresh landing GET) and retry that query once, so the
    tenant still yields its jobs instead of silently returning none."""
    patch_greenhouse_gates(monkeypatch)  # posting's title must clear the main-pass gate
    cfg = _two_tenant_config()
    cfg["workday_tenants"] = cfg["workday_tenants"][:1]
    sc = WorkdayScraper(cfg)
    monkeypatch.setattr("scrapers.workday.time.sleep", lambda s: None)

    posting = {
        "title": "AI Product Manager",
        "externalPath": "/job/Remote/AI-PM_JR1",
        "locationsText": "Remote",
        "postedOn": "Posted Today",
        "bulletFields": [],
    }
    landing_gets = []
    posts = []

    def fake_get(url, **kwargs):
        if _is_landing(url):
            landing_gets.append(url)
            return _FakeResp(200)
        return _FakeResp(200, {"jobPostingInfo": {"jobDescription": "desc"}})

    def fake_post(url, **kwargs):
        posts.append(url)
        if len(posts) == 2:  # first POST of query 2 — stale session
            return _FakeResp(403)
        return _FakeResp(200, {"jobPostings": [posting], "total": 1})

    monkeypatch.setattr(sc.session, "get", fake_get)
    monkeypatch.setattr(sc.session, "post", fake_post)

    jobs = sc.scrape_all(["query one", "query two"])

    assert len(posts) == 3, "stale-session 403 must trigger exactly one fresh-session retry"
    assert len(landing_gets) == 2, "the retry must re-establish the session first"
    assert len(jobs) == 1  # same URL both queries — dedup keeps one


def _deterministic_clock(monkeypatch):
    """Patch scrapers.workday.time so pacing math is wall-clock-independent:
    sleep() records its argument and advances the fake clock; monotonic()
    reads it. Returns the recorded-sleeps list."""
    sleeps = []
    clock = {"t": 1000.0}

    def fake_sleep(s):
        sleeps.append(s)
        clock["t"] += s

    monkeypatch.setattr("scrapers.workday.time.sleep", fake_sleep)
    monkeypatch.setattr("scrapers.workday.time.monotonic", lambda: clock["t"])
    return sleeps


def test_no_fixed_sleep_between_different_cluster_tenants(monkeypatch):
    """Consecutive (tenant x query) pairs on DIFFERENT Workday clusters need
    no politeness wait between them — the unconditional inter-pair sleep was
    ~8 min/run of dead waiting. Same-cluster revisits inside the rate-limit
    window must still wait (next test)."""
    sc = WorkdayScraper(_two_tenant_config(rate_limit=7))
    sleeps = _deterministic_clock(monkeypatch)

    monkeypatch.setattr(sc.session, "get", lambda url, **kw: _FakeResp(200))
    monkeypatch.setattr(
        sc.session, "post",
        lambda url, **kw: _FakeResp(200, {"jobPostings": [], "total": 0}),
    )

    sc.scrape_all(["query one"])

    rate_waits = [s for s in sleeps if s >= 6]
    assert rate_waits == [], (
        f"alpha.wd5 -> beta.wd1 are different clusters; no inter-pair wait expected, got {rate_waits}"
    )


def test_same_cluster_revisit_still_paced(monkeypatch):
    """Two tenants on the SAME wdN cluster hit back-to-back must keep the
    rate-limit spacing (anti-bot can operate per cluster, not per subdomain)."""
    sc = WorkdayScraper(
        _two_tenant_config(rate_limit=7, second_tenant_url="gamma.wd5.myworkdayjobs.com")
    )
    sleeps = _deterministic_clock(monkeypatch)

    monkeypatch.setattr(sc.session, "get", lambda url, **kw: _FakeResp(200))
    monkeypatch.setattr(
        sc.session, "post",
        lambda url, **kw: _FakeResp(200, {"jobPostings": [], "total": 0}),
    )

    sc.scrape_all(["query one"])

    assert any(s > 6 for s in sleeps), (
        f"same-cluster back-to-back pairs must still be paced ~rate_limit apart, got {sleeps}"
    )


def test_error_landing_response_is_not_cached(monkeypatch):
    """A landing page answering 503/403 (an HTTP response, not an exception)
    sets no usable cookies — it must not be cached as an establishment, or
    one CF challenge would pin degraded headers for the whole run."""
    cfg = _two_tenant_config()
    cfg["workday_tenants"] = cfg["workday_tenants"][:1]
    sc = WorkdayScraper(cfg)
    monkeypatch.setattr("scrapers.workday.time.sleep", lambda s: None)

    landing_gets = []

    def fake_get(url, **kwargs):
        if _is_landing(url):
            landing_gets.append(url)
            return _FakeResp(503)
        return _FakeResp(200)

    monkeypatch.setattr(sc.session, "get", fake_get)
    monkeypatch.setattr(
        sc.session, "post",
        lambda url, **kw: _FakeResp(200, {"jobPostings": [], "total": 0}),
    )

    sc.scrape_all(["query one", "query two"])

    assert len(landing_gets) == 2, (
        "an error landing response must be retried on the tenant's next query, not cached"
    )


def test_post_failure_invalidates_cached_session(monkeypatch):
    """A non-auth POST failure (404/503) must drop the cached session so the
    tenant's NEXT query re-establishes — the old per-pair establishment
    recovered from every failure mode this way, not just 401/403."""
    cfg = _two_tenant_config()
    cfg["workday_tenants"] = cfg["workday_tenants"][:1]
    sc = WorkdayScraper(cfg)
    monkeypatch.setattr("scrapers.workday.time.sleep", lambda s: None)

    landing_gets = []
    posts = []

    def fake_get(url, **kwargs):
        if _is_landing(url):
            landing_gets.append(url)
        return _FakeResp(200)

    def fake_post(url, **kwargs):
        posts.append(url)
        if len(posts) == 1:
            return _FakeResp(503)
        return _FakeResp(200, {"jobPostings": [], "total": 0})

    monkeypatch.setattr(sc.session, "get", fake_get)
    monkeypatch.setattr(sc.session, "post", fake_post)

    sc.scrape_all(["query one", "query two"])

    assert len(landing_gets) == 2, (
        "a failed POST must invalidate the cached session for the next query"
    )


def test_malformed_tenant_entry_does_not_kill_other_tenants(monkeypatch):
    """A config tenant entry missing tenant_url must only lose that tenant
    (logged), not abort the whole Workday scrape — pacing/establishment
    lookups run inside the per-tenant containment."""
    patch_greenhouse_gates(monkeypatch)  # posting's title must clear the main-pass gate
    cfg = _two_tenant_config()
    cfg["workday_tenants"] = [
        {"company": "BrokenCo", "site_path": "External"},  # no tenant_url
        cfg["workday_tenants"][1],
    ]
    sc = WorkdayScraper(cfg)
    monkeypatch.setattr("scrapers.workday.time.sleep", lambda s: None)

    posting = {
        "title": "AI Product Manager",
        "externalPath": "/job/Remote/AI-PM_JR9",
        "locationsText": "Remote",
        "postedOn": "Posted Today",
        "bulletFields": [],
    }

    def fake_get(url, **kwargs):
        if _is_landing(url):
            return _FakeResp(200)
        return _FakeResp(200, {"jobPostingInfo": {"jobDescription": "d"}})

    monkeypatch.setattr(sc.session, "get", fake_get)
    monkeypatch.setattr(
        sc.session, "post",
        lambda url, **kw: _FakeResp(200, {"jobPostings": [posting], "total": 1}),
    )

    jobs = sc.scrape_all(["query one"])

    assert len(jobs) == 1, "the healthy tenant must still contribute its jobs"
    assert jobs[0].company == "BetaCo"
