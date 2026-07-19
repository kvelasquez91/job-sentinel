"""Tests for SuccessFactorsScraper (Career Site Builder HTML)."""
from scrapers.successfactors import SuccessFactorsScraper
from policy_fixtures import patch_greenhouse_gates

CONFIG = {
    "profile": {"target_titles": ["Senior Product Manager AI", "VP Product"]},
    "local_target_titles": ["Director Digital Transformation", "IT Product Manager"],
    "successfactors_tenants": [
        {"company": "Michelin", "base_url": "https://jobs.michelinman.com"}
    ],
    "search_queries": ["Senior Product Manager AI remote"],
}

SEARCH_HTML = """
<html><body><table><tbody>
<tr class="data-row">
  <td><a class="jobTitle-link" href="/job/Springfield-Digital-Product-Manager/123/">
    Digital Product Manager</a></td>
  <td><span class="jobLocation">Springfield, IL, US</span></td>
  <td><span class="jobDate">Jul 1, 2026</span></td>
</tr>
<tr class="data-row">
  <td><a class="jobTitle-link" href="/job/Springfield-Tire-Builder/456/">
    Tire Builder</a></td>
  <td><span class="jobLocation">Springfield, IL, US</span></td>
  <td><span class="jobDate">Jul 1, 2026</span></td>
</tr>
</tbody></table></body></html>
"""

DETAIL_HTML = """
<html><body><span itemprop="description">
Lead digital product strategy for our AI-driven manufacturing platform.
</span></body></html>
"""


def _scraper():
    return SuccessFactorsScraper(CONFIG)


def test_local_target_titles_broaden_filter():
    sc = _scraper()
    assert sc._passes_title_filter("Director, Digital Transformation")
    assert sc._passes_title_filter("IT Product Manager - ERP")
    assert not sc._passes_title_filter("Tire Builder")


def test_parse_search_page_filters_titles_and_builds_urls(monkeypatch):
    patch_greenhouse_gates(monkeypatch)
    sc = _scraper()
    pages = {"0": SEARCH_HTML, "25": "<html><body></body></html>"}
    sc._get_html = lambda url: (
        DETAIL_HTML if "/job/" in url else pages.get(
            url.split("startrow=")[-1], "<html></html>")
    )
    jobs = sc._scrape_tenant({"company": "Michelin",
                              "base_url": "https://jobs.michelinman.com"})
    assert [j.title for j in jobs] == ["Digital Product Manager"]
    assert jobs[0].url == (
        "https://jobs.michelinman.com/job/Springfield-Digital-Product-Manager/123/")
    assert jobs[0].location == "Springfield, IL, US"
    assert "digital product strategy" in jobs[0].description
    assert jobs[0].source == "successfactors"
    assert jobs[0].company == "Michelin"


def test_scrape_runs_only_on_first_query():
    sc = _scraper()
    sc._scrape_tenant = lambda t: []
    assert sc.scrape("some other query") == []


# A search page whose rows never pass the title filter and which the tenant
# returns for EVERY startrow (ZF Group clamps out-of-range startrow back to a
# non-empty page instead of returning an empty one). The old loop only exited
# on an empty page or on filling max_results, so this spun forever.
WRAP_HTML = """
<html><body><table><tbody>
<tr class="data-row">
  <td><a class="jobTitle-link" href="/job/Stuttgart-Tire-Builder/1/">
    Tire Builder</a></td>
  <td><span class="jobLocation">Stuttgart, DE</span></td>
</tr>
<tr class="data-row">
  <td><a class="jobTitle-link" href="/job/Stuttgart-Verfahrensmechaniker/2/">
    Verfahrensmechaniker</a></td>
  <td><span class="jobLocation">Stuttgart, DE</span></td>
</tr>
</tbody></table></body></html>
"""


def test_scrape_tenant_terminates_when_pages_never_empty():
    """Regression: a tenant that never returns an empty page (out-of-range
    startrow wraps to already-seen rows) must not loop forever."""
    config = dict(CONFIG)
    config["scraping"] = {"rate_limit_seconds": 0}
    sc = SuccessFactorsScraper(config)

    calls = {"n": 0}

    def fake_get(url):
        if "/job/" in url:
            return ""  # no title passes the filter, so this is never reached
        calls["n"] += 1
        assert calls["n"] < 200, "infinite pagination loop (page never empties)"
        return WRAP_HTML

    sc._get_html = fake_get
    jobs = sc._scrape_tenant({"company": "ZF Group",
                              "base_url": "https://jobs.zf.com"})
    assert jobs == []
    assert calls["n"] < 200
