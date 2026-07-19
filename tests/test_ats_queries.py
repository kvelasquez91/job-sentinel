"""ATS query source (Workday/Eightfold): their search is server-side, so
feeding them the LinkedIn-phrased `search_queries` (recruiter phrasing plus a
" remote" suffix) returns ~0 from a tenant's own search box — a local-first
owner's verified tenants silently yield nothing. ats_search_queries() picks
the query set those scrapers search with instead:

    explicit `ats_search_queries` config key
    > profile.target_titles + local_target_titles (deduped, order kept)
    > `search_queries` with the word "remote" stripped
"""
from scrapers.base import ats_search_queries


def test_explicit_key_wins():
    cfg = {
        "ats_search_queries": ["Process Engineer"],
        "profile": {"target_titles": ["Ignored Title"]},
        "search_queries": ["Ignored remote"],
    }
    assert ats_search_queries(cfg) == ["Process Engineer"]


def test_titles_fallback_merges_and_dedupes_preserving_order():
    cfg = {
        "profile": {"target_titles": ["Senior Process Engineer",
                                      "Process Safety Engineer"]},
        "local_target_titles": ["Process Engineer", "process safety engineer"],
        "search_queries": ["Senior Process Engineer remote"],
    }
    assert ats_search_queries(cfg) == [
        "Senior Process Engineer", "Process Safety Engineer", "Process Engineer"]


def test_search_queries_fallback_strips_remote_word():
    cfg = {"search_queries": ["Senior Widget Engineer remote",
                              "Remote Director of Widgets"]}
    assert ats_search_queries(cfg) == [
        "Senior Widget Engineer", "Director of Widgets"]


def test_stripping_collapses_whitespace_drops_empties_and_dupes():
    # "Remote" alone strips to nothing; the two Widget variants collapse to
    # one query; interior "remote" must not leave a double space behind.
    cfg = {"search_queries": ["Remote", "Widget Engineer remote",
                              "Widget remote Engineer"]}
    assert ats_search_queries(cfg) == ["Widget Engineer"]


def test_all_sources_empty_returns_empty():
    assert ats_search_queries({}) == []


# --- wiring: the two server-side-search scrapers must search with these -----


def test_workday_scrape_all_searches_titles_not_linkedin_queries(monkeypatch):
    from scrapers.workday import WorkdayScraper
    cfg = {
        "profile": {"target_titles": ["Senior Process Engineer"]},
        "local_target_titles": ["Process Engineer"],
        "search_queries": ["Senior Process Engineer remote"],
        "workday_tenants": [{
            "company": "Example",
            "tenant_url": "example.wd5.myworkdayjobs.com",
            "company_slug": "example",
            "site_path": "Example",
        }],
        "scraping": {"max_results_per_source": 50, "rate_limit_seconds": 0},
    }
    sc = WorkdayScraper(cfg)
    seen = []
    monkeypatch.setattr(
        sc, "_scrape_tenant",
        lambda tenant, query, posting_filter=None: (seen.append(query), [])[1])
    sc.scrape_all(cfg["search_queries"])
    assert seen == ["Senior Process Engineer", "Process Engineer"]


def test_eightfold_scrape_all_searches_titles_not_linkedin_queries(monkeypatch):
    from scrapers.eightfold import EightfoldScraper
    cfg = {
        "profile": {"target_titles": ["Senior Process Engineer"]},
        "search_queries": ["Senior Process Engineer remote"],
        "eightfold_tenants": [{
            "company": "Example",
            "subdomain": "example",
            "domain": "example.com",
        }],
        "scraping": {"max_results_per_source": 50, "rate_limit_seconds": 0},
    }
    sc = EightfoldScraper(cfg)
    seen = []
    monkeypatch.setattr(
        sc, "_scrape_tenant",
        lambda tenant, query: (seen.append(query), [])[1])
    sc.scrape_all(cfg["search_queries"])
    assert seen == ["Senior Process Engineer"]


def test_workday_scrape_all_falls_back_to_passed_queries_when_config_bare(monkeypatch):
    # No ats key, no titles, no search_queries in config: keep the caller's
    # queries rather than searching nothing.
    from scrapers.workday import WorkdayScraper
    cfg = {
        "workday_tenants": [{
            "company": "Example",
            "tenant_url": "example.wd5.myworkdayjobs.com",
            "company_slug": "example",
            "site_path": "Example",
        }],
        "scraping": {"max_results_per_source": 50, "rate_limit_seconds": 0},
    }
    sc = WorkdayScraper(cfg)
    seen = []
    monkeypatch.setattr(
        sc, "_scrape_tenant",
        lambda tenant, query, posting_filter=None: (seen.append(query), [])[1])
    sc.scrape_all(["Senior Product Manager AI remote"])
    assert seen == ["Senior Product Manager AI"]
