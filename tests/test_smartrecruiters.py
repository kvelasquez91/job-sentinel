"""Tests for SmartRecruitersScraper.

Covers the two bugs that made it return 0 jobs for every company:
  1. It passed the full search phrase to the API's server-side ``q`` param, so
     a company with hundreds of postings returned 0. The scraper must fetch a
     company's postings and filter by title *locally*.
  2. Its target_titles filter used brittle full-phrase substring matching (same
     bug fixed in greenhouse.py), rejecting real titles like "Product Manager, X".
"""
from scrapers.smartrecruiters import SmartRecruitersScraper
from policy_fixtures import patch_state_pattern, patch_greenhouse_gates

PROFILE_CONFIG = {
    "profile": {
        "target_titles": [
            "Senior Product Manager AI",
            "Director Product AI",
            "Head of AI",
            "VP Product",
            "Product Lead GenAI",
        ]
    },
    "smartrecruiters_companies": [{"company": "Test Co", "company_id": "TestCo"}],
    "search_queries": ["Senior Product Manager AI remote"],
}


def _scraper():
    return SmartRecruitersScraper(PROFILE_CONFIG)


def _posting(title):
    return {
        "name": title,
        "location": {"remote": True},
        "ref": f"https://jobs.smartrecruiters.com/{abs(hash(title))}",
        "id": "x",
        "releasedDate": "2026-07-01T00:00:00Z",
        "jobAd": {"sections": {}},
    }


def test_flat_product_manager_title_passes_despite_narrow_targets(monkeypatch):
    patch_greenhouse_gates(monkeypatch)
    assert _scraper()._passes_title_filter("Product Manager, API Growth")


def test_target_title_matches_despite_punctuation():
    assert _scraper()._passes_title_filter("Senior Product Manager, AI Platform")


def test_irrelevant_title_rejected():
    assert not _scraper()._passes_title_filter("Principal Hardware Engineer, ASIC")


def test_scrape_company_fetches_all_and_filters_locally(monkeypatch):
    """Fetch a company's postings and filter by title locally — never rely on a
    narrow server-side ``q`` search (which returned 0 for real boards)."""
    patch_greenhouse_gates(monkeypatch)
    sc = _scraper()
    calls = []

    def fake_get_postings(company_id, params):
        calls.append(params)
        if params.get("offset", 0) > 0:
            return {"content": [], "totalFound": 2}
        return {
            "content": [
                _posting("Product Manager, Compute Platform"),
                _posting("Staff Hardware Engineer, ASIC"),
            ],
            "totalFound": 2,
        }

    sc._get_postings = fake_get_postings
    jobs = sc._scrape_company({"company": "Test Co", "company_id": "TestCo"})

    assert [j.title for j in jobs] == ["Product Manager, Compute Platform"]
    # No narrow server-side query: `q` must be absent/empty so filtering is local.
    assert all(not p.get("q") for p in calls), f"unexpected server-side q: {calls}"


# ---------------------------------------------------------------------------
# Task 11: location-gated local-title broadening (local commuter area).
# See tests/test_greenhouse.py for the full rationale — the strict AI-PM
# filter must accept local_target_titles ONLY when the job's location is in
# the local commuter area, leaving remote-company jobs on the strict filter
# unchanged.
# ---------------------------------------------------------------------------

LOCAL_CONFIG = {
    "profile": {
        "target_titles": [
            "Senior Product Manager AI",
            "Director Product AI",
            "Head of AI",
            "VP Product",
            "Product Lead GenAI",
        ]
    },
    "smartrecruiters_companies": [{"company": "Test Co", "company_id": "TestCo"}],
    "search_queries": ["Senior Product Manager AI remote"],
    "local_locations": [
        "Springfield", "Riverton", "Fairview", "Lakewood", "Cedar Falls",
    ],
    "local_target_titles": [
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
    ],
}


def _local_posting(title, city):
    # SmartRecruiters postings often carry the state in the city field itself
    # (e.g. "Springfield, IL"); country left blank so the built location
    # string is exactly the city text, matching real-world local postings.
    return {
        "name": title,
        "location": {"city": city, "country": "", "remote": False},
        "ref": f"https://jobs.smartrecruiters.com/{abs(hash((title, city)))}",
        "id": "x",
        "releasedDate": "2026-07-01T00:00:00Z",
        "jobAd": {"sections": {}},
    }


def test_local_title_passes_in_local_area_but_not_elsewhere(monkeypatch):
    """A local_target_title ("Director Digital Transformation") located in
    Springfield, IL must pass; the identical title in Austin, TX must be
    dropped (strict filter unchanged for non-local jobs). A normal remote
    AI-PM title must still pass regardless of location."""
    patch_state_pattern(monkeypatch)
    sc = SmartRecruitersScraper(LOCAL_CONFIG)

    local_job = _local_posting("Director Digital Transformation", "Springfield, IL")
    remote_decoy = _local_posting("Director Digital Transformation", "Austin, TX")
    remote_pm_job = _posting("Senior Product Manager, AI Platform")

    parsed_local = sc._parse_posting(local_job, "Test Co")
    parsed_decoy = sc._parse_posting(remote_decoy, "Test Co")
    parsed_remote_pm = sc._parse_posting(remote_pm_job, "Test Co")

    assert parsed_local is not None
    assert parsed_local.location == "Springfield, IL"
    assert parsed_decoy is None, "non-local job with a local-only title must still be dropped"
    assert parsed_remote_pm is not None


def test_malformed_location_does_not_crash_for_filter_failing_title():
    """Regression: moving location-string construction before the title
    filter (so local_title_passes can see it) means `_parse_posting` now
    builds `", ".join(parts)` from `city`/`country` for EVERY posting,
    including ones whose title fails the strict filter — a code path that
    never ran before this change. A non-string city/country (a malformed
    API response) must not raise `TypeError` from `", ".join`; the posting
    should just be dropped, like any other filter-failing posting, not
    crash the caller."""
    sc = SmartRecruitersScraper(LOCAL_CONFIG)

    malformed_city_posting = {
        "name": "Warehouse Associate",  # fails the strict filter
        "location": {"city": {"unexpected": "shape"}, "country": "US", "remote": False},
        "ref": "https://jobs.smartrecruiters.com/malformed-city",
        "id": "x",
        "releasedDate": "2026-07-01T00:00:00Z",
        "jobAd": {"sections": {}},
    }

    result = sc._parse_posting(malformed_city_posting, "Test Co")  # must not raise

    assert result is None


# ---------------------------------------------------------------------------
# Regression: descriptions were never captured.
#
# The description lives in jobAd.sections, but the /postings LIST endpoint omits
# jobAd entirely — it only appears on the per-posting detail record (the row's
# `ref` API URL). The scraper read jobAd off the list row, so every posting was
# stored blank. _parse_posting must fetch the detail record when the list row
# has no jobAd.
# ---------------------------------------------------------------------------
def test_parse_posting_fetches_description_when_list_omits_jobad(monkeypatch):
    patch_greenhouse_gates(monkeypatch)
    sc = _scraper()
    fetched = []

    def fake_fetch_job_ad(ref):
        fetched.append(ref)
        return {
            "sections": {
                "jobDescription": {"text": "Own the AI product roadmap."},
                "qualifications": {"text": "5+ years PM."},
            }
        }

    sc._fetch_job_ad = fake_fetch_job_ad

    # Real list rows carry NO jobAd key (unlike the detail record).
    posting = {
        "name": "Product Manager, AI Platform",
        "location": {"remote": True},
        "ref": "https://api.smartrecruiters.com/v1/companies/TestCo/postings/42",
        "id": "42",
        "releasedDate": "2026-07-01T00:00:00Z",
    }

    job = sc._parse_posting(posting, "Test Co")

    assert job is not None
    assert "Own the AI product roadmap." in job.description
    assert "5+ years PM." in job.description
    assert fetched == ["https://api.smartrecruiters.com/v1/companies/TestCo/postings/42"]


def test_parse_posting_uses_inline_jobad_without_fetch(monkeypatch):
    """If a jobAd is already present on the row (e.g. detail record), use it
    directly — no extra detail fetch."""
    patch_greenhouse_gates(monkeypatch)
    sc = _scraper()
    sc._fetch_job_ad = lambda ref: (_ for _ in ()).throw(
        AssertionError("must not fetch when jobAd is inline")
    )

    posting = {
        "name": "Product Manager, AI Platform",
        "location": {"remote": True},
        "ref": "https://api.smartrecruiters.com/v1/companies/TestCo/postings/42",
        "id": "42",
        "releasedDate": "2026-07-01T00:00:00Z",
        "jobAd": {"sections": {"jobDescription": {"text": "Inline description."}}},
    }

    job = sc._parse_posting(posting, "Test Co")

    assert job is not None
    assert job.description == "Inline description."


def test_fetch_job_ad_failure_not_cached(monkeypatch):
    """A transient detail-fetch failure must NOT pin {} for the whole run —
    the same posting re-seen later should retry."""
    sc = _scraper()
    calls = []

    class _Bad:
        ok = False
        status_code = 500

    class _Good:
        ok = True

        @staticmethod
        def json():
            return {"jobAd": {"sections": {"jobDescription": {"text": "X"}}}}

    def fake_get(url, timeout=20):
        calls.append(url)
        return _Bad() if len(calls) == 1 else _Good()

    monkeypatch.setattr(sc.session, "get", fake_get)
    ref = "https://api.smartrecruiters.com/v1/companies/T/postings/1"
    assert sc._fetch_job_ad(ref) == {}
    assert sc._fetch_job_ad(ref) == {"sections": {"jobDescription": {"text": "X"}}}
    assert len(calls) == 2, "failure must not be negative-cached"
