from unittest import mock

from scrapers.wttj import WTTJScraper
from policy_fixtures import patch_wttj_gates

CONFIG = {
    "profile": {"target_titles": ["Product Manager", "Director Product", "Head of AI"]},
    "scraping": {"max_results_per_source": 50, "rate_limit_seconds": 0},
}


def _hit(name="Senior Product Manager", org="Acme", oslug="acme", slug="senior-pm",
         currency="USD", period="yearly", smin=200000, smax=260000, remote="fulltime",
         city="New York", country="United States"):
    return {
        "name": name,
        "organization": {"name": org, "slug": oslug},
        "slug": slug,
        "salary_currency": currency, "salary_period": period,
        "salary_minimum": smin, "salary_maximum": smax,
        "remote": remote,
        "offices": [{"city": city, "country": country}],
        "published_at": "2026-06-05T02:00:00.000+02:00",
        "profile": "",
    }


def test_parse_hit_maps_fields(monkeypatch):
    patch_wttj_gates(monkeypatch)
    s = WTTJScraper(CONFIG)
    with mock.patch.object(s, "_fetch_description", return_value="We build AI/ML LLM products. $200k-$260k."):
        job = s._parse_hit(_hit())
    assert job is not None
    assert job.title == "Senior Product Manager"
    assert job.company == "Acme"
    assert job.url == "https://www.welcometothejungle.com/en/companies/acme/jobs/senior-pm"
    assert job.source == "wttj"
    assert job.salary_min == 200000 and job.salary_max == 260000
    assert "AI/ML" in job.description
    assert job.date_posted == "2026-06-05"
    assert "United States" in job.location


def test_parse_hit_title_filter_rejects_unrelated():
    s = WTTJScraper(CONFIG)
    with mock.patch.object(s, "_fetch_description", return_value="x"):
        assert s._parse_hit(_hit(name="Warehouse Associate")) is None


def test_parse_hit_non_usd_salary_dropped(monkeypatch):
    patch_wttj_gates(monkeypatch)
    s = WTTJScraper(CONFIG)
    with mock.patch.object(s, "_fetch_description", return_value="desc"):
        job = s._parse_hit(_hit(currency="EUR"))
    assert job is not None
    assert job.salary_min is None and job.salary_max is None


def test_parse_hit_requires_slugs_for_url():
    s = WTTJScraper(CONFIG)
    with mock.patch.object(s, "_fetch_description", return_value="desc"):
        assert s._parse_hit(_hit(oslug="", slug="")) is None


def test_scrape_paginates_and_caps_at_max_results(monkeypatch):
    patch_wttj_gates(monkeypatch)
    s = WTTJScraper(CONFIG)
    s.max_results = 3
    pages = [
        {"hits": [_hit(slug=f"j{i}") for i in range(2)], "nbPages": 5},
        {"hits": [_hit(slug=f"k{i}") for i in range(2)], "nbPages": 5},
    ]

    def fake_query(query, page):
        return pages[page] if page < len(pages) else {"hits": [], "nbPages": 5}

    monkeypatch.setattr(s, "_algolia_query", fake_query)
    monkeypatch.setattr(s, "_fetch_description", lambda org_slug, slug: "desc")
    jobs = s.scrape("product manager")
    assert len(jobs) == 3  # capped at max_results, did not run away across all pages
    assert all(j.source == "wttj" for j in jobs)
    assert all(j.url for j in jobs)


# --- Description fetch via WTTJ's public REST API (not the WAF-blocked www.* page) ---
#
# The Algolia hit carries only `profile` (requirements); the full role text is
# fetched from api.welcometothejungle.com. Regression: the scraper used to read
# the JSON-LD off the www.* HTML job page, which is WAF-challenged (HTTP 202),
# so it usually fell back to the requirements-only `profile` — descriptions were
# "lacking" (see the Believe "Automation & Process Lead" report).


class _FakeResp:
    def __init__(self, status_code, payload):
        self.status_code = status_code
        self._payload = payload

    def json(self):
        return self._payload


def test_fetch_description_reads_full_jd_from_api(monkeypatch):
    s = WTTJScraper(CONFIG)
    captured = {}

    def fake_get(url, headers=None, timeout=None):
        captured["url"] = url
        return _FakeResp(200, {"job": {
            "description": "<p>The role: own automation end-to-end.</p>",
            "profile": "<ul><li>6+ years PM experience.</li></ul>",
            "key_missions": "",
        }})

    monkeypatch.setattr(s.session, "get", fake_get)
    out = s._fetch_description("believe-digital", "senior-pm_paris")

    assert ("api.welcometothejungle.com/api/v1/organizations/believe-digital"
            "/jobs/senior-pm_paris") in captured["url"]
    # Full role text AND the requirements are both included.
    assert "The role: own automation end-to-end." in out
    assert "6+ years PM experience." in out


def test_fetch_description_caches_per_run(monkeypatch):
    s = WTTJScraper(CONFIG)
    calls = []

    def fake_get(url, headers=None, timeout=None):
        calls.append(url)
        return _FakeResp(200, {"job": {"description": "Role text."}})

    monkeypatch.setattr(s.session, "get", fake_get)
    a = s._fetch_description("org", "slug")
    b = s._fetch_description("org", "slug")
    assert a == b == "Role text."
    assert len(calls) == 1, "second call must be served from cache"


def test_fetch_description_empty_on_waf_or_error(monkeypatch):
    s = WTTJScraper(CONFIG)
    monkeypatch.setattr(s.session, "get", lambda *a, **k: _FakeResp(202, {}))
    assert s._fetch_description("org", "slug") == ""  # WAF-style non-200 → empty


def test_fetch_description_requires_slugs_and_skips_network(monkeypatch):
    s = WTTJScraper(CONFIG)

    def boom(*a, **k):
        raise AssertionError("must not hit the network without slugs")

    monkeypatch.setattr(s.session, "get", boom)
    assert s._fetch_description("", "slug") == ""
    assert s._fetch_description("org", "") == ""


def test_parse_hit_falls_back_to_profile_when_api_empty(monkeypatch):
    """If the detail API yields nothing, the job still gets the Algolia
    `profile` (requirements) rather than an empty description."""
    patch_wttj_gates(monkeypatch)
    s = WTTJScraper(CONFIG)
    monkeypatch.setattr(s, "_fetch_description", lambda org_slug, slug: "")
    hit = _hit()
    hit["profile"] = "<ul><li>Requirements only.</li></ul>"

    job = s._parse_hit(hit)

    assert job is not None
    assert "Requirements only." in job.description  # cleaned by JobPosting


# --- WTTJ-specific query set ---

def test_uses_configured_wttj_queries():
    cfg = dict(CONFIG, wttj_queries=["foo", "bar"])
    assert WTTJScraper(cfg).queries == ["foo", "bar"]


def test_falls_back_to_default_queries(monkeypatch):
    patch_wttj_gates(monkeypatch)
    s = WTTJScraper(CONFIG)  # CONFIG has no wttj_queries
    assert "Product Manager" in s.queries and len(s.queries) >= 4


def test_scrape_all_uses_own_queries(monkeypatch):
    s = WTTJScraper(CONFIG)
    s.queries = ["q1", "q2"]
    seen = []
    monkeypatch.setattr(s, "scrape", lambda q: seen.append(q) or [])
    s.scrape_all(["caller-queries-ignored-by-wttj"])
    assert seen == ["q1", "q2"]


def test_fetch_description_error_not_cached(monkeypatch):
    """A transient WAF/5xx response must NOT be cached: the same job seen via a
    later query in the run should retry (main.py never repairs a blank row)."""
    s = WTTJScraper(CONFIG)
    calls = []

    def fake_get(url, headers=None, timeout=None):
        calls.append(url)
        if len(calls) == 1:
            return _FakeResp(202, {})
        return _FakeResp(200, {"job": {"description": "Role."}})

    monkeypatch.setattr(s.session, "get", fake_get)
    assert s._fetch_description("org", "slug") == ""
    assert s._fetch_description("org", "slug") == "Role."
    assert len(calls) == 2, "failure must not be negative-cached"
