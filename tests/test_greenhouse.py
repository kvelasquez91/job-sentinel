"""Tests for GreenhouseScraper's title filtering.

Regression coverage for the profile ``target_titles`` filter that silently
zeroed out Greenhouse/Lever/Ashby yield: the narrow config phrases (e.g.
"Senior Product Manager AI") were matched as verbatim substrings, so real
titles like "Product Manager, API Growth" never passed.
"""
import logging

from scrapers.greenhouse import GreenhouseScraper, extract_salary_from_page_text
from policy_fixtures import patch_state_pattern, patch_greenhouse_gates

# A plausible owner's narrow target-title phrases
# that broke substring matching. Kept inline so the test is self-contained.
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
    "search_queries": ["Senior Product Manager AI remote"],
}

# Real titles from Anthropic's public Greenhouse board that are genuine PM roles.
REAL_PM_TITLES = [
    "Product Manager, API Growth",
    "Product Manager, Compute Platform",
    "Research Product Manager, Model Behaviors",
    "Product Manager, Developer Productivity",
]

# Real titles that must NOT pass — no product/PM/AI-leadership signal.
IRRELEVANT_TITLES = [
    "Software Engineer, Backend",
    "Head of Logistics",
    "Head of FX & Risk",
    "Data Engineering Manager, Product",  # engineering manager, not a PM role
]


def _scraper():
    return GreenhouseScraper(PROFILE_CONFIG)


def test_flat_product_manager_titles_pass_despite_narrow_profile_targets(monkeypatch):
    """A profile whose target_titles are narrow must not reject the built-in
    PM/product baseline — flat "Product Manager, X" titles have to pass."""
    patch_greenhouse_gates(monkeypatch)
    sc = _scraper()
    for title in REAL_PM_TITLES:
        assert sc._passes_title_filter(title), f"{title!r} should pass the filter"


def test_target_title_matches_despite_punctuation_and_extra_words():
    """"Senior Product Manager AI" must match "Senior Product Manager, AI
    Platform" — token-based, not verbatim-substring."""
    sc = _scraper()
    assert sc._passes_title_filter("Senior Product Manager, AI Platform")
    assert sc._passes_title_filter("Head of AI, Applied Research")


def test_irrelevant_titles_are_rejected():
    sc = _scraper()
    for title in IRRELEVANT_TITLES:
        assert not sc._passes_title_filter(title), f"{title!r} should be rejected"


def test_profile_targets_do_not_require_verbatim_phrase(monkeypatch):
    """Guard against the substring regression: a real PM title that contains
    none of the target phrases verbatim must still pass."""
    patch_greenhouse_gates(monkeypatch)
    sc = _scraper()
    title = "Product Manager, Claude Code Model Performance"
    assert not any(t.lower() in title.lower() for t in PROFILE_CONFIG["profile"]["target_titles"])
    assert sc._passes_title_filter(title)


# ---------------------------------------------------------------------------
# Task 11: location-gated local-title broadening (local commuter area).
#
# The strict AI-PM filter drops legitimate local digital roles (e.g. a
# retail-chain's "Director, Digital Growth Marketing & CRM") because they
# don't match the remote AI-PM keyword baseline or profile.target_titles. For
# jobs located in the local commuter area, local_target_titles should ALSO be
# accepted — but this must NOT loosen the filter for remote-company jobs
# (same title, non-local location must still be dropped).
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


def _greenhouse_job(title, location_name):
    return {
        "title": title,
        "location": {"name": location_name},
        "absolute_url": f"https://boards.greenhouse.io/testco/jobs/{abs(hash((title, location_name)))}",
        "id": abs(hash((title, location_name))),
        "content": "Some description with no salary info.",
        "updated_at": "2026-07-01T00:00:00Z",
    }


def test_local_title_passes_in_local_area_but_not_elsewhere(monkeypatch):
    """A local_target_title ("Director Digital Transformation") located in
    Springfield, IL must pass; the identical title in Austin, TX must be
    dropped (strict filter unchanged for non-local jobs). A normal remote
    AI-PM title must still pass regardless of location."""
    patch_state_pattern(monkeypatch)
    sc = GreenhouseScraper(LOCAL_CONFIG)

    local_job = _greenhouse_job("Director Digital Transformation", "Springfield, IL")
    remote_decoy = _greenhouse_job("Director Digital Transformation", "Austin, TX")
    remote_pm_job = _greenhouse_job("Senior Product Manager, AI Platform", "Remote - US")

    sc._fetch_json = lambda url: {"jobs": [local_job, remote_decoy, remote_pm_job]}
    # Keep the test hermetic: salary-less descriptions trigger the posting-page
    # salary fallback, which must not hit the network here.
    sc._fetch_posting_page_text = lambda url: ""

    jobs = sc._scrape_greenhouse("Test Co", "testco")
    titles_and_locations = {(j.title, j.location) for j in jobs}

    assert ("Director Digital Transformation", "Springfield, IL") in titles_and_locations
    assert not any(
        j.title == "Director Digital Transformation" and j.location == "Austin, TX"
        for j in jobs
    ), "non-local job with a local-only title must still be dropped"
    assert any(j.title == "Senior Product Manager, AI Platform" for j in jobs)


def test_malformed_location_does_not_crash_for_filter_failing_title(caplog):
    """Regression: moving the location computation before the title-filter
    check (so local_title_passes can see it) means location is now built for
    EVERY job, including ones whose title fails the strict filter — a code
    path that never ran before this change. A location shape that isn't a
    plain string (e.g. location.name: null, or a nested object) must not
    raise inside local_title_passes/is_local_commuter_area; the job should be
    silently dropped via the normal filter `continue`, not via the per-job
    except-and-log-at-debug fallback (a bare `jobs == []` assertion alone
    can't tell these apart, since both produce an empty result — asserting
    on the debug log is what actually proves no exception was swallowed)."""
    sc = GreenhouseScraper(LOCAL_CONFIG)
    null_name_job = {
        "title": "Warehouse Associate",  # fails the strict filter
        "location": {"name": None},  # API returned {"name": null}
        "absolute_url": "https://boards.greenhouse.io/testco/jobs/1",
        "id": 1,
        "content": "",
        "updated_at": "2026-07-01T00:00:00Z",
    }
    malformed_name_job = {
        "title": "Warehouse Associate",  # fails the strict filter
        "location": {"name": {"unexpected": "shape"}},  # non-string name
        "absolute_url": "https://boards.greenhouse.io/testco/jobs/2",
        "id": 2,
        "content": "",
        "updated_at": "2026-07-01T00:00:00Z",
    }

    sc._fetch_json = lambda url: {"jobs": [null_name_job, malformed_name_job]}

    with caplog.at_level(logging.DEBUG):
        jobs = sc._scrape_greenhouse("Test Co", "testco")  # must not raise

    assert jobs == []
    assert not any("Error parsing Greenhouse job" in r.message for r in caplog.records), (
        "a malformed location must be filtered out cleanly, not crash and get "
        "silently caught by the per-job except handler"
    )


def test_ashby_malformed_location_does_not_crash_for_filter_failing_title(caplog):
    """Same regression as above, for the Ashby parse path: a non-string
    `location`/`locationName` on a filter-failing posting must not raise."""
    sc = GreenhouseScraper(LOCAL_CONFIG)
    malformed_posting = {
        "title": "Warehouse Associate",  # fails the strict filter
        "location": {"unexpected": "shape"},  # non-string, falls back from locationName
        "jobUrl": "https://jobs.ashbyhq.com/testco/warehouse",
        "id": "x",
        "publishedAt": "2026-07-01T00:00:00Z",
    }
    sc._fetch_json = lambda url: {"jobs": [malformed_posting]}

    with caplog.at_level(logging.DEBUG):
        jobs = sc._scrape_ashby("Test Co", "testco")  # must not raise

    assert jobs == []
    assert not any("Error parsing Ashby job" in r.message for r in caplog.records), (
        "a malformed location must be filtered out cleanly, not crash and get "
        "silently caught by the per-job except handler"
    )


def test_lever_malformed_location_does_not_crash_for_filter_failing_title(caplog):
    """Same regression as above, for the Lever parse path: a non-string
    `categories.location`/`categories.commitment` on a filter-failing
    posting must not raise."""
    sc = GreenhouseScraper(LOCAL_CONFIG)
    malformed_posting = {
        "text": "Warehouse Associate",  # fails the strict filter
        "categories": {"location": {"unexpected": "shape"}},  # non-string location
        "hostedUrl": "https://jobs.lever.co/testco/warehouse",
    }
    sc._fetch_json = lambda url: [malformed_posting]

    with caplog.at_level(logging.DEBUG):
        jobs = sc._scrape_lever("Test Co", "testco")  # must not raise

    assert jobs == []
    assert not any("Error parsing Lever job" in r.message for r in caplog.records), (
        "a malformed location must be filtered out cleanly, not crash and get "
        "silently caught by the per-job except handler"
    )


# ---------------------------------------------------------------------------
# Posting-page salary fallback.
#
# Regression: Stripe's Greenhouse API carries NO pay data anywhere (`content`
# has no dollar figures, `pay_input_ranges`/`metadata` are null) — the range
# is rendered only on the company-hosted posting page (stripe.com "Pay and
# benefits" section). When the description yields no salary, the scraper must
# fetch the posting page and extract the range from salary-context sentences
# only, so unrelated dollar figures on the page never leak into salary fields.
# ---------------------------------------------------------------------------

# Mirrors the actual stripe.com "Pay and benefits" copy for gh_jid=8064526,
# surrounded by marketing dollar-noise that must NOT be parsed as salary.
STRIPE_PAGE_TEXT = (
    "Stripe processes $640,000 in payments for merchants every second and has "
    "helped businesses raise over $500,000 in funding rounds. "
    "Pay and benefits. The annual US base salary range for this role is "
    "$162,400 - $243,600. For sales roles, the range provided is the role's "
    'On Target Earnings ("OTE") range. '
    "Additional benefits for this role may include: equity, company bonus or "
    "sales commissions/bonuses; 401(k) plan; medical, dental, and vision benefits."
)


def test_extract_salary_from_page_text_finds_range_in_salary_sentence():
    lo, hi = extract_salary_from_page_text(STRIPE_PAGE_TEXT)
    assert lo == 162400
    assert hi == 243600


def test_extract_salary_from_page_text_ignores_non_compensation_dollars():
    """Dollar figures outside a salary/pay/compensation sentence must not be
    treated as salary — a page with only marketing numbers yields nothing."""
    noise_only = (
        "Stripe processes $640,000 in payments for merchants every second and "
        "has helped businesses raise over $500,000 in funding rounds."
    )
    assert extract_salary_from_page_text(noise_only) == (None, None)
    assert extract_salary_from_page_text("") == (None, None)


def test_greenhouse_salary_fallback_uses_posting_page(monkeypatch):
    """When the API description has no salary, the scraper must fetch the
    company-hosted posting page and extract the range from it."""
    patch_greenhouse_gates(monkeypatch)
    sc = _scraper()
    job = _greenhouse_job("AI Product Manager, Professional Services",
                          "New York/ San Francisco")
    sc._fetch_json = lambda url: {"jobs": [job]}
    fetched_urls = []

    def fake_page_fetch(url):
        fetched_urls.append(url)
        return STRIPE_PAGE_TEXT

    sc._fetch_posting_page_text = fake_page_fetch

    jobs = sc._scrape_greenhouse("Stripe", "stripe")
    assert len(jobs) == 1
    assert jobs[0].salary_min == 162400
    assert jobs[0].salary_max == 243600
    assert fetched_urls == [job["absolute_url"]]


def test_greenhouse_salary_fallback_skipped_when_description_has_salary():
    """No extra page fetch when the API description already yields a range."""
    sc = _scraper()
    job = _greenhouse_job("Senior Product Manager, AI Platform", "Remote - US")
    job["content"] = "Great role. The base salary range is $150,000 - $200,000."
    sc._fetch_json = lambda url: {"jobs": [job]}

    def must_not_fetch(url):
        raise AssertionError("posting page must not be fetched when salary is in the description")

    sc._fetch_posting_page_text = must_not_fetch

    jobs = sc._scrape_greenhouse("Stripe", "stripe")
    assert len(jobs) == 1
    assert jobs[0].salary_min == 150000
    assert jobs[0].salary_max == 200000


def test_greenhouse_salary_fallback_tolerates_page_fetch_failure():
    """A failed/empty page fetch leaves salary unset — never crashes the job."""
    sc = _scraper()
    job = _greenhouse_job("Senior Product Manager, AI Platform", "Remote - US")
    sc._fetch_json = lambda url: {"jobs": [job]}
    sc._fetch_posting_page_text = lambda url: ""

    jobs = sc._scrape_greenhouse("Stripe", "stripe")
    assert len(jobs) == 1
    assert jobs[0].salary_min is None
    assert jobs[0].salary_max is None
