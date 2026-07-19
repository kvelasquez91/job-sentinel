"""HN Who's Hiring scraper: Algolia API parsing, filtering, salary extraction."""
import logging

import scrapers.hn_whoishiring as hn_mod
from scrapers.hn_whoishiring import HNWhoIsHiringScraper
from policy_fixtures import patch_hn_gates

CFG = {
    "scraping": {"max_results_per_source": 50, "rate_limit_seconds": 0},
    "search_queries": ["Senior Product Manager AI remote"],
    "hn_whoishiring_enabled": True,
}

STORY_ID = "41000000"

STORY_HITS = {"hits": [
    {"objectID": STORY_ID, "title": "Ask HN: Who is hiring? (July 2026)"},
]}

COMMENTS_PAGE = {
    "nbPages": 1,
    "hits": [
        {   # relevant: PM + remote + salary
            "objectID": "41000001", "parent_id": 41000000,
            "created_at": "2026-07-02T12:00:00Z",
            "comment_text": ("<p>AcmeAI | Senior Product Manager, LLM Platform | "
                             "REMOTE (US) | $210K-$260K + equity</p>"
                             "<p>We build agentic workflows for the enterprise.</p>"),
        },
        {   # not a PM role
            "objectID": "41000002", "parent_id": 41000000,
            "created_at": "2026-07-02T12:01:00Z",
            "comment_text": "<p>DataCo | Senior Backend Engineer | REMOTE | Go, Postgres</p>",
        },
        {   # PM but onsite-only, no remote mention
            "objectID": "41000003", "parent_id": 41000000,
            "created_at": "2026-07-02T12:02:00Z",
            "comment_text": "<p>OfficeCorp | Product Manager | NYC office only, 5 days</p>",
        },
        {   # a reply, not a top-level posting
            "objectID": "41000004", "parent_id": 41000001,
            "created_at": "2026-07-02T12:03:00Z",
            "comment_text": "<p>Is this Product Manager role remote-friendly?</p>",
        },
    ],
}


def _scraper(monkeypatch):
    s = HNWhoIsHiringScraper(CFG)

    def fake_get_json(url, params=None):
        if "search_by_date" in url and (params or {}).get("tags", "").startswith("story"):
            return STORY_HITS
        return COMMENTS_PAGE

    monkeypatch.setattr(s, "_get_json", fake_get_json)
    return s


def test_parses_relevant_top_level_pm_comment(monkeypatch):
    patch_hn_gates(monkeypatch)
    jobs = _scraper(monkeypatch).scrape("Senior Product Manager AI remote")
    assert len(jobs) == 1
    j = jobs[0]
    assert j.company == "AcmeAI"
    assert "Product Manager" in j.title
    assert j.location == "Remote"
    assert j.url == "https://news.ycombinator.com/item?id=41000001"
    assert (j.salary_min, j.salary_max) == (210_000.0, 260_000.0)
    assert j.source == "hn"
    assert j.date_posted == "2026-07-02"


def test_non_first_query_is_a_noop(monkeypatch):
    jobs = _scraper(monkeypatch).scrape("Director Product AI ML remote")
    assert jobs == []


def test_iterates_multiple_comment_pages(monkeypatch):
    """The comment query paginates newest-first; multiple pages should all be
    consumed (and merged into results) when nbPages says more remain."""
    patch_hn_gates(monkeypatch)
    monkeypatch.setattr(hn_mod.time, "sleep", lambda *_: None)
    s = HNWhoIsHiringScraper(CFG)

    page0 = {
        "nbPages": 2,
        "hits": [
            {   # relevant top-level posting, newest page
                "objectID": "41000101", "parent_id": 41000000,
                "created_at": "2026-07-05T09:00:00Z",
                "comment_text": "<p>PageZeroCo | Senior Product Manager | REMOTE | $180K-$200K</p>",
            },
        ],
    }
    page1 = {
        "nbPages": 2,
        "hits": [
            {   # relevant top-level posting, older page
                "objectID": "41000102", "parent_id": 41000000,
                "created_at": "2026-07-01T09:00:00Z",
                "comment_text": "<p>PageOneCo | Product Manager | REMOTE | $150K-$170K</p>",
            },
        ],
    }

    def fake_get_json(url, params=None):
        if "search_by_date" in url and (params or {}).get("tags", "").startswith("story"):
            return STORY_HITS
        page = (params or {}).get("page", 0)
        return page0 if page == 0 else page1

    monkeypatch.setattr(s, "_get_json", fake_get_json)
    jobs = s.scrape("Senior Product Manager AI remote")

    assert len(jobs) == 2
    assert {j.url for j in jobs} == {
        "https://news.ycombinator.com/item?id=41000101",
        "https://news.ycombinator.com/item?id=41000102",
    }


def test_warns_and_stops_at_page_cap(monkeypatch, caplog):
    """When MAX_COMMENT_PAGES is hit while nbPages indicates more pages
    remain, the scraper must stop (not hang) and log a WARNING rather than
    silently dropping older postings."""
    patch_hn_gates(monkeypatch)
    monkeypatch.setattr(hn_mod, "MAX_COMMENT_PAGES", 1)
    monkeypatch.setattr(hn_mod.time, "sleep", lambda *_: None)
    s = HNWhoIsHiringScraper(CFG)

    # Always the same finite page, claiming far more pages exist than the
    # (monkeypatched) cap allows -- guards against an infinite loop.
    capped_page = {
        "nbPages": 5,
        "hits": [
            {   # relevant top-level posting
                "objectID": "41000201", "parent_id": 41000000,
                "created_at": "2026-07-05T09:00:00Z",
                "comment_text": "<p>CappedCo | Product Manager | REMOTE | $160K-$180K</p>",
            },
        ],
    }

    def fake_get_json(url, params=None):
        if "search_by_date" in url and (params or {}).get("tags", "").startswith("story"):
            return STORY_HITS
        return capped_page

    monkeypatch.setattr(s, "_get_json", fake_get_json)

    with caplog.at_level(logging.WARNING):
        jobs = s.scrape("Senior Product Manager AI remote")

    assert len(jobs) == 1
    assert "MAX_COMMENT_PAGES" in caplog.text
