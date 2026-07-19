"""Tests for Eightfold timestamp parsing.

Regression: Eightfold's t_update/t_create are epoch SECONDS, but the scraper
divided by 1000 (treating them as milliseconds), so every job's date_posted
became 1970-01-21 — which then auto-expired the job the next day.
"""
from scrapers.eightfold import EightfoldScraper, _epoch_to_iso_date
from policy_fixtures import patch_greenhouse_gates

EF_CONFIG = {
    "eightfold_tenants": [{"company": "Fluor", "subdomain": "fluor", "domain": "fluor.com"}],
    "scraping": {"rate_limit_seconds": 0},
}
EF_TENANT = {"company": "Fluor", "subdomain": "fluor", "domain": "fluor.com"}


def test_epoch_seconds_not_treated_as_milliseconds():
    # Real Eightfold t_update (epoch seconds, ~2026) must not collapse to 1970.
    d = _epoch_to_iso_date(1779148800)
    assert d is not None and d.startswith("2026"), d


def test_millisecond_timestamps_are_also_handled():
    # Same instant expressed in milliseconds must yield the same date.
    assert _epoch_to_iso_date(1779148800000) == _epoch_to_iso_date(1779148800)


def test_invalid_timestamps_return_none():
    assert _epoch_to_iso_date(None) is None
    assert _epoch_to_iso_date("") is None
    assert _epoch_to_iso_date(0) is None


# ---------------------------------------------------------------------------
# Regression: descriptions were never captured.
#
# The v2 SEARCH API returns job_description as an EMPTY string; the full posting
# HTML lives only on the per-position detail endpoint. The scraper read the
# (empty) list value, so every Eightfold job was stored blank. _parse_position
# must fall back to a detail fetch when the list job_description is empty.
# ---------------------------------------------------------------------------
def test_parse_position_fetches_description_when_list_empty():
    sc = EightfoldScraper(EF_CONFIG)
    calls = []

    def fake_fetch(tenant, pid):
        calls.append(pid)
        return "<p>Own the AI product roadmap.</p>"

    sc._fetch_description = fake_fetch
    pos = {
        "id": 123,
        "name": "AI Product Manager",
        "job_description": "",  # search API returns this empty
        "t_update": 1779148800,
        "city": "Remote",
    }

    job = sc._parse_position(pos, EF_TENANT)

    assert job is not None
    assert job.description == "Own the AI product roadmap."  # tags stripped
    assert calls == [123], "must fetch the detail record when the list value is empty"


def test_parse_position_uses_list_description_when_present():
    """If the list row already carries job_description, use it — no detail fetch."""
    sc = EightfoldScraper(EF_CONFIG)
    sc._fetch_description = lambda tenant, pid: (_ for _ in ()).throw(
        AssertionError("must not fetch when list description is present")
    )
    pos = {
        "id": 123,
        "name": "AI Product Manager",
        "job_description": "<p>Inline JD.</p>",
        "t_update": 1779148800,
        "city": "Remote",
    }

    job = sc._parse_position(pos, EF_TENANT)

    assert job is not None
    assert job.description == "Inline JD."


# ---------------------------------------------------------------------------
# The detail fetch must be best-effort: _get_jobs re-raises ConnectionError/
# Timeout after its tenacity retries, and an escaping exception aborts the
# whole tenant scrape (losing every job already parsed) or the backfill run.
# Failures must also not be negative-cached — the same job seen again later in
# the run should retry, because main.py never repairs a blank stored row.
# ---------------------------------------------------------------------------
def test_fetch_description_survives_errors_and_does_not_cache_failure(monkeypatch):
    import requests

    sc = EightfoldScraper(EF_CONFIG)
    calls = []

    def flaky(url, params):
        calls.append(url)
        if len(calls) == 1:
            raise requests.ConnectionError("network down")
        return {"job_description": "<p>JD</p>"}

    monkeypatch.setattr(sc, "_get_jobs", flaky)
    assert sc._fetch_description(EF_TENANT, 1) == ""  # error -> best-effort ""
    assert sc._fetch_description(EF_TENANT, 1) == "<p>JD</p>"
    assert len(calls) == 2, "failure must not be negative-cached"


def test_description_keeps_list_structure():
    """Eightfold must not pre-flatten HTML: clean_description preserves the
    bullet/paragraph structure the dashboard renders."""
    sc = EightfoldScraper(EF_CONFIG)
    sc._fetch_description = (
        lambda tenant, pid: "<ul><li>Ship roadmap</li><li>Talk to users</li></ul>"
    )
    pos = {"id": 9, "name": "AI Product Manager", "job_description": "",
           "t_update": 1779148800, "city": "Remote"}

    job = sc._parse_position(pos, EF_TENANT)

    assert job is not None
    assert "• Ship roadmap" in job.description
    assert "• Talk to users" in job.description


# ---------------------------------------------------------------------------
# Dogfood finding: Eightfold applied NO title gate at all — the main tenant
# scrape saved every job the search API returned, contradicting the
# documented "every ATS scraper is gated" claim. _passes_main_title applies
# the same baseline-OR-target_titles semantics as greenhouse/lever/ashby/
# smartrecruiters/successfactors/workday, in _scrape_tenant's loop — BEFORE
# the per-position detail fetch, so an off-gate posting never costs an API
# call. (_parse_position itself stays gate-free, per the tests above.)
# ---------------------------------------------------------------------------
def test_main_pass_rejects_off_gate_title(monkeypatch):
    """The MAIN scrape pass must reject a title matching neither the
    baseline gate nor profile.target_titles — and reject it BEFORE the
    per-position detail fetch (no wasted API call)."""
    patch_greenhouse_gates(monkeypatch)
    sc = EightfoldScraper(EF_CONFIG)

    pos = {"id": 1, "name": "Staff Accountant", "job_description": "",
           "t_update": 1779148800, "city": "Remote"}
    list_url = sc._api_url(EF_TENANT)
    calls = []

    def fake_get_jobs(url, params):
        calls.append(url)
        if url == list_url:
            return {"positions": [pos], "count": 1}
        return {"job_description": "should not be fetched"}

    monkeypatch.setattr(sc, "_get_jobs", fake_get_jobs)

    jobs = sc.scrape_all(["query one"])

    assert jobs == []
    assert calls == [list_url], (
        f"the gate must reject before the detail fetch, got calls={calls}"
    )


def test_main_pass_accepts_baseline_title(monkeypatch):
    """A title matching the baseline PM/product gate passes the MAIN pass."""
    patch_greenhouse_gates(monkeypatch)
    sc = EightfoldScraper(EF_CONFIG)

    pos = {"id": 2, "name": "Senior Product Manager, AI", "job_description": "",
           "t_update": 1779148800, "city": "Remote"}
    list_url = sc._api_url(EF_TENANT)
    detail_url = f"{list_url}/2"

    def fake_get_jobs(url, params):
        if url == list_url:
            return {"positions": [pos], "count": 1}
        if url == detail_url:
            return {"job_description": "<p>JD</p>"}
        return None

    monkeypatch.setattr(sc, "_get_jobs", fake_get_jobs)

    jobs = sc.scrape_all(["query one"])

    assert len(jobs) == 1
    assert jobs[0].title == "Senior Product Manager, AI"


def test_main_pass_accepts_target_titles_with_empty_baseline(monkeypatch):
    """With an EMPTY baseline gate, a title that token-matches
    profile.target_titles must still pass the MAIN pass."""
    import scrapers.greenhouse as gh
    monkeypatch.setattr(gh, "PRODUCT_TITLE_KEYWORDS", [])
    monkeypatch.setattr(gh, "AI_TITLE_KEYWORDS", [])

    cfg = dict(EF_CONFIG)
    cfg["profile"] = {"target_titles": ["Director Digital Transformation"]}
    sc = EightfoldScraper(cfg)

    pos = {"id": 3, "name": "Director of Digital Transformation, EMEA",
           "job_description": "", "t_update": 1779148800, "city": "Remote"}
    list_url = sc._api_url(EF_TENANT)
    detail_url = f"{list_url}/3"

    def fake_get_jobs(url, params):
        if url == list_url:
            return {"positions": [pos], "count": 1}
        if url == detail_url:
            return {"job_description": "<p>JD</p>"}
        return None

    monkeypatch.setattr(sc, "_get_jobs", fake_get_jobs)

    jobs = sc.scrape_all(["query one"])

    assert len(jobs) == 1


def test_main_pass_yields_nothing_when_gate_and_target_titles_empty(monkeypatch):
    """Both the baseline gate AND profile.target_titles empty (the neutral-
    tree condition) means the MAIN pass yields NOTHING — dormant until
    configured, same semantics as greenhouse."""
    import scrapers.greenhouse as gh
    monkeypatch.setattr(gh, "PRODUCT_TITLE_KEYWORDS", [])
    monkeypatch.setattr(gh, "AI_TITLE_KEYWORDS", [])

    sc = EightfoldScraper(EF_CONFIG)  # no profile.target_titles

    pos = {"id": 4, "name": "AI Product Manager", "job_description": "",
           "t_update": 1779148800, "city": "Remote"}
    list_url = sc._api_url(EF_TENANT)

    def fake_get_jobs(url, params):
        if url == list_url:
            return {"positions": [pos], "count": 1}
        return {"job_description": "should not be fetched"}

    monkeypatch.setattr(sc, "_get_jobs", fake_get_jobs)

    jobs = sc.scrape_all(["query one"])

    assert jobs == []
