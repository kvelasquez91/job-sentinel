"""Detail-fetch failures must be visible at production log level (INFO).

Regression: the list-API scrapers (workday/eightfold/smartrecruiters/wttj)
fetch each job's detail endpoint for the description, but logged every fetch
failure at DEBUG only. Production runs log at INFO, so a systemic failure
(403 blocking, endpoint change) would leave zero log trace until descriptions
went blank corpus-wide. Required behavior, shared via BaseScraper:

  * each failed detail fetch logs at WARNING (with company / URL / status),
    capped at DETAIL_FAILURE_WARN_LIMIT per run — the rest log at DEBUG;
  * scrape_all ends with one per-scraper summary line: INFO when zero
    failures, WARNING when some failed, ERROR when >50% failed (so it lands
    in logs/errors.log);
  * scrapers that attempted no detail fetches stay silent (no 0/0 noise).
"""
import logging
from types import SimpleNamespace

import pytest

from scrapers.base import BaseScraper
from scrapers.eightfold import EightfoldScraper
from scrapers.smartrecruiters import SmartRecruitersScraper
from scrapers.workday import WorkdayScraper
from scrapers.wttj import WTTJScraper


class _DummyScraper(BaseScraper):
    source_name = "dummy"

    def scrape(self, query):
        return []


def _records(caplog, level, needle):
    return [r for r in caplog.records
            if r.levelno == level and needle in r.getMessage()]


# ---------------------------------------------------------------------------
# BaseScraper helpers
# ---------------------------------------------------------------------------

def test_first_n_failures_warn_rest_debug(caplog):
    sc = _DummyScraper({})
    caplog.set_level(logging.DEBUG)
    for i in range(sc.DETAIL_FAILURE_WARN_LIMIT + 2):
        sc._log_detail_fetch_failure("detail fetch failed for job %d", i)
    warned = _records(caplog, logging.WARNING, "detail fetch failed for job")
    debugged = _records(caplog, logging.DEBUG, "detail fetch failed for job")
    assert len(warned) == sc.DETAIL_FAILURE_WARN_LIMIT
    assert len(debugged) == 2
    assert sc._detail_fetch_failures == sc.DETAIL_FAILURE_WARN_LIMIT + 2


def test_summary_info_when_no_failures(caplog):
    sc = _DummyScraper({})
    caplog.set_level(logging.DEBUG)
    for _ in range(3):
        sc._count_detail_fetch_success()
    sc._log_detail_fetch_summary()
    assert _records(caplog, logging.INFO, "dummy: 0/3 detail fetches failed")


def test_summary_warning_when_some_failures(caplog):
    sc = _DummyScraper({})
    caplog.set_level(logging.DEBUG)
    sc._log_detail_fetch_failure("boom")
    for _ in range(9):
        sc._count_detail_fetch_success()
    sc._log_detail_fetch_summary()
    assert _records(caplog, logging.WARNING, "dummy: 1/10 detail fetches failed")


def test_summary_error_when_majority_fail(caplog):
    sc = _DummyScraper({})
    caplog.set_level(logging.DEBUG)
    for _ in range(3):
        sc._log_detail_fetch_failure("boom")
    sc._count_detail_fetch_success()
    sc._log_detail_fetch_summary()
    assert _records(caplog, logging.ERROR, "dummy: 3/4 detail fetches failed")


def test_summary_silent_when_no_attempts(caplog):
    sc = _DummyScraper({})
    caplog.set_level(logging.DEBUG)
    sc._log_detail_fetch_summary()
    assert not [r for r in caplog.records if "detail fetches" in r.getMessage()]


def test_base_scrape_all_emits_summary(caplog):
    sc = _DummyScraper({"scraping": {"rate_limit_seconds": 0}})
    caplog.set_level(logging.DEBUG)

    def scrape(query):
        sc._log_detail_fetch_failure("boom for %s", query)
        sc._count_detail_fetch_success()
        sc._count_detail_fetch_success()
        return []

    sc.scrape = scrape
    sc.scrape_all(["q1"])
    assert _records(caplog, logging.WARNING, "dummy: 1/3 detail fetches failed")


# ---------------------------------------------------------------------------
# Workday wiring
# ---------------------------------------------------------------------------

WD_CONFIG = {
    "profile": {"target_titles": ["Product Manager"]},
    "workday_tenants": [
        {"company": "NVIDIA", "tenant_url": "nvidia.wd5.myworkdayjobs.com",
         "company_slug": "nvidia", "site_path": "NVIDIAExternalCareerSite"},
    ],
    "scraping": {"max_results_per_source": 10, "rate_limit_seconds": 0},
}
WD_TENANT = WD_CONFIG["workday_tenants"][0]


class _Resp:
    def __init__(self, status_code, payload=None):
        self.status_code = status_code
        self.ok = 200 <= status_code < 300
        self._payload = payload or {}

    def json(self):
        return self._payload


def test_workday_detail_failure_warns_with_company_and_status(caplog, monkeypatch):
    sc = WorkdayScraper(WD_CONFIG)
    caplog.set_level(logging.DEBUG)
    monkeypatch.setattr(sc.session, "get", lambda url, timeout=20: _Resp(403))
    assert sc._fetch_description(WD_TENANT, "/job/CA/PM_JR-1") == ""
    warned = _records(caplog, logging.WARNING, "NVIDIA")
    assert warned and "403" in warned[0].getMessage()
    assert sc._detail_fetch_failures == 1


def test_workday_detail_exception_counts_failure(caplog, monkeypatch):
    sc = WorkdayScraper(WD_CONFIG)
    caplog.set_level(logging.DEBUG)

    def boom(url, timeout=20):
        raise OSError("connection reset")

    monkeypatch.setattr(sc.session, "get", boom)
    assert sc._fetch_description(WD_TENANT, "/job/CA/PM_JR-2") == ""
    assert sc._detail_fetch_failures == 1
    assert _records(caplog, logging.WARNING, "NVIDIA")


def test_workday_detail_success_counts_attempt_not_failure(monkeypatch):
    sc = WorkdayScraper(WD_CONFIG)
    monkeypatch.setattr(
        sc.session, "get",
        lambda url, timeout=20: _Resp(200, {"jobPostingInfo": {"jobDescription": "JD"}}),
    )
    assert sc._fetch_description(WD_TENANT, "/job/CA/PM_JR-3") == "JD"
    assert sc._detail_fetch_attempts == 1
    assert sc._detail_fetch_failures == 0


def test_workday_playwright_detail_failure_counts(caplog):
    sc = WorkdayScraper(WD_CONFIG)
    caplog.set_level(logging.DEBUG)
    page = SimpleNamespace(evaluate=lambda script, url: {"ok": False, "data": None})
    assert sc._fetch_description_via_page(page, WD_TENANT, "/job/CA/PM_JR-4") == ""
    assert sc._detail_fetch_failures == 1
    assert _records(caplog, logging.WARNING, "NVIDIA")


def test_workday_scrape_all_emits_summary(caplog, monkeypatch):
    sc = WorkdayScraper(WD_CONFIG)
    caplog.set_level(logging.DEBUG)

    def fake_scrape_tenant(tenant, query, posting_filter=None):
        sc._log_detail_fetch_failure("boom")
        sc._count_detail_fetch_success()
        sc._count_detail_fetch_success()
        return []

    monkeypatch.setattr(sc, "_scrape_tenant", fake_scrape_tenant)
    sc.scrape_all(["q1"])
    assert _records(caplog, logging.WARNING, "workday: 1/3 detail fetches failed")


# ---------------------------------------------------------------------------
# Eightfold wiring
# ---------------------------------------------------------------------------

EF_CONFIG = {
    "eightfold_tenants": [{"company": "Fluor", "subdomain": "fluor", "domain": "fluor.com"}],
    "scraping": {"rate_limit_seconds": 0},
}
EF_TENANT = EF_CONFIG["eightfold_tenants"][0]


def test_eightfold_detail_failure_warns(caplog, monkeypatch):
    sc = EightfoldScraper(EF_CONFIG)
    caplog.set_level(logging.DEBUG)
    monkeypatch.setattr(sc, "_get_jobs", lambda url, params: None)
    assert sc._fetch_description(EF_TENANT, 1) == ""
    assert sc._detail_fetch_failures == 1
    assert _records(caplog, logging.WARNING, "Fluor")


def test_eightfold_detail_success_counts(monkeypatch):
    sc = EightfoldScraper(EF_CONFIG)
    monkeypatch.setattr(
        sc, "_get_jobs", lambda url, params: {"job_description": "<p>JD</p>"}
    )
    assert sc._fetch_description(EF_TENANT, 2) == "<p>JD</p>"
    assert sc._detail_fetch_attempts == 1
    assert sc._detail_fetch_failures == 0


def test_eightfold_scrape_all_emits_summary(caplog, monkeypatch):
    sc = EightfoldScraper(EF_CONFIG)
    caplog.set_level(logging.DEBUG)

    def fake_scrape_tenant(tenant, query):
        sc._log_detail_fetch_failure("boom")
        sc._count_detail_fetch_success()
        sc._count_detail_fetch_success()
        return []

    monkeypatch.setattr(sc, "_scrape_tenant", fake_scrape_tenant)
    sc.scrape_all(["q1"])
    assert _records(caplog, logging.WARNING, "eightfold: 1/3 detail fetches failed")


# ---------------------------------------------------------------------------
# SmartRecruiters wiring
# ---------------------------------------------------------------------------

SR_CONFIG = {
    "profile": {"target_titles": ["Product Manager"]},
    "smartrecruiters_companies": [{"company": "Test Co", "company_id": "TestCo"}],
    "scraping": {"rate_limit_seconds": 0},
}


def test_smartrecruiters_detail_failure_warns_with_status(caplog, monkeypatch):
    sc = SmartRecruitersScraper(SR_CONFIG)
    caplog.set_level(logging.DEBUG)
    monkeypatch.setattr(sc.session, "get", lambda url, timeout=20: _Resp(500))
    ref = "https://api.smartrecruiters.com/v1/companies/TestCo/postings/1"
    assert sc._fetch_job_ad(ref) == {}
    assert sc._detail_fetch_failures == 1
    warned = _records(caplog, logging.WARNING, ref)
    assert warned and "500" in warned[0].getMessage()


def test_smartrecruiters_scrape_all_emits_summary(caplog, monkeypatch):
    sc = SmartRecruitersScraper(SR_CONFIG)
    caplog.set_level(logging.DEBUG)

    def fake_scrape(query):
        sc._log_detail_fetch_failure("boom")
        sc._count_detail_fetch_success()
        sc._count_detail_fetch_success()
        return []

    monkeypatch.setattr(sc, "scrape", fake_scrape)
    sc.scrape_all(["q1"])
    assert _records(caplog, logging.WARNING,
                    "smartrecruiters: 1/3 detail fetches failed")


# ---------------------------------------------------------------------------
# WTTJ wiring
# ---------------------------------------------------------------------------

WTTJ_CONFIG = {
    "profile": {"target_titles": ["Product Manager"]},
    "scraping": {"max_results_per_source": 10, "rate_limit_seconds": 0},
}


def test_wttj_detail_failure_warns_with_status(caplog, monkeypatch):
    sc = WTTJScraper(WTTJ_CONFIG)
    caplog.set_level(logging.DEBUG)
    monkeypatch.setattr(
        sc.session, "get", lambda url, headers=None, timeout=20: _Resp(202)
    )
    assert sc._fetch_description("org", "slug") == ""
    assert sc._detail_fetch_failures == 1
    warned = _records(caplog, logging.WARNING, "org")
    assert warned and "202" in warned[0].getMessage()


def test_wttj_detail_success_counts(monkeypatch):
    sc = WTTJScraper(WTTJ_CONFIG)
    monkeypatch.setattr(
        sc.session, "get",
        lambda url, headers=None, timeout=20: _Resp(200, {"job": {"description": "JD"}}),
    )
    assert sc._fetch_description("org", "slug") == "JD"
    assert sc._detail_fetch_attempts == 1
    assert sc._detail_fetch_failures == 0
