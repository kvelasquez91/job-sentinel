"""Tests for scraper-run yield summarization (silent-failure visibility).

A scraper that returns 0 jobs must be distinguishable from a healthy run — the
daily automation previously logged "0 jobs" at INFO identically to success, so
boards stayed broken for months unnoticed.
"""
from main import format_sources, summarize_scraper_run


def test_healthy_run_has_no_empty_sources():
    summary = summarize_scraper_run({"LinkedIn": 139, "Workday": 147, "WTTJ": 59})
    assert summary["total"] == 345
    assert summary["empty"] == []
    assert summary["all_empty"] is False


def test_single_dead_source_is_flagged():
    summary = summarize_scraper_run({"LinkedIn": 139, "SmartRecruiters": 0, "Workday": 147})
    assert summary["empty"] == ["SmartRecruiters"]
    assert summary["all_empty"] is False
    assert summary["total"] == 286


def test_all_sources_empty_is_flagged_as_total_failure():
    summary = summarize_scraper_run({"LinkedIn": 0, "Greenhouse/Lever": 0, "WTTJ": 0})
    assert summary["all_empty"] is True
    assert summary["empty"] == ["LinkedIn", "Greenhouse/Lever", "WTTJ"]
    assert summary["total"] == 0


def test_no_scrapers_is_not_a_total_failure():
    """An empty count map (no scrapers ran) is vacuous, not an all-empty run."""
    summary = summarize_scraper_run({})
    assert summary["all_empty"] is False
    assert summary["total"] == 0
    assert summary["empty"] == []


def test_format_sources_lists_only_contributing_scrapers():
    counts = {"LinkedIn": 139, "SmartRecruiters": 0, "WTTJ": 59}
    assert format_sources(counts) == "linkedin,wttj"


def test_format_sources_empty_when_nothing_contributed():
    assert format_sources({"LinkedIn": 0}) == ""
    assert format_sources({}) == ""


def test_format_sources_preserves_insertion_order():
    counts = {"WTTJ": 3, "LinkedIn": 10, "Workday": 0, "Eightfold": 2}
    assert format_sources(counts) == "wttj,linkedin,eightfold"


def test_run_scrapers_registers_successfactors_adapter(monkeypatch):
    """The successfactors_tenants config key must register the SuccessFactors
    scraper under its display name — the wiring guarantee for local-area tenants.
    (iCIMS/Oracle adapters were dropped; SuccessFactors is the only new adapter.)"""
    import main as main_mod
    from scrapers.linkedin import LinkedInScraper
    from scrapers.greenhouse import GreenhouseScraper
    from scrapers.successfactors import SuccessFactorsScraper

    for cls in (LinkedInScraper, GreenhouseScraper, SuccessFactorsScraper):
        monkeypatch.setattr(cls, "scrape_all", lambda self, queries: [])

    config = {
        "search_queries": ["x"],
        "scraping": {"rate_limit_seconds": 0},
        "successfactors_tenants": [{"company": "A", "base_url": "https://a.example"}],
    }
    _jobs, counts = main_mod.run_scrapers(config, dry_run=True)
    assert "SuccessFactors" in counts


import sqlite3

from main import _linkedin_company_queries, load_known_linkedin_urls


def test_company_queries_use_explicit_list_when_configured():
    config = {
        "linkedin_company_search_queries": ["Senior Product Manager AI remote"],
        "search_queries": ["a", "b", "c"],
    }
    assert _linkedin_company_queries(config, ["a", "b", "c"]) == [
        "Senior Product Manager AI remote"
    ]


def test_company_queries_fall_back_to_keyword_queries():
    assert _linkedin_company_queries({}, ["a", "b"]) == ["a", "b"]
    assert _linkedin_company_queries(
        {"linkedin_company_search_queries": []}, ["a"]
    ) == ["a"]


def test_load_known_linkedin_urls_filters_by_source():
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE jobs (url TEXT, source TEXT)")
    conn.executemany(
        "INSERT INTO jobs VALUES (?, ?)",
        [
            ("https://www.linkedin.com/jobs/view/1/", "linkedin"),
            ("https://www.linkedin.com/jobs/view/2/", "linkedin"),
            ("https://boards.greenhouse.io/x/jobs/3", "greenhouse"),
        ],
    )
    urls = load_known_linkedin_urls(conn)
    assert urls == {
        "https://www.linkedin.com/jobs/view/1/",
        "https://www.linkedin.com/jobs/view/2/",
    }


def test_load_known_linkedin_urls_survives_missing_table():
    conn = sqlite3.connect(":memory:")
    assert load_known_linkedin_urls(conn) == set()


# ---------------------------------------------------------------------------
# Concurrent scraper execution (perf). The sequential loop made total scrape
# time the SUM of all sources (~56 min measured); each scraper targets a
# different host family, so they run in parallel threads with per-host pacing
# preserved inside each scraper. The merge into all_jobs/seen_urls/counts must
# stay in the fixed scrapers-list order regardless of completion order, so
# URL-dedup winners and per-source counts are identical to the sequential run.
# ---------------------------------------------------------------------------
import threading
import time as _time

from scrapers.base import JobPosting


def _posting(title, company, url):
    return JobPosting(
        title=title, company=company, location="Remote",
        url=url, description="d", source="test",
    )


def test_scrapers_run_concurrently(monkeypatch):
    """Both scrapers must be in flight at the same time: each waits on a
    2-party barrier inside scrape_all, which times out (broken barrier) if
    the orchestrator runs them one after the other."""
    import main as main_mod
    from scrapers.linkedin import LinkedInScraper
    from scrapers.greenhouse import GreenhouseScraper

    barrier = threading.Barrier(2, timeout=5)

    def li_scrape_all(self, queries):
        barrier.wait()
        return [_posting("A", "CoA", "https://x/a")]

    def gh_scrape_all(self, queries):
        barrier.wait()
        return [_posting("B", "CoB", "https://x/b")]

    monkeypatch.setattr(LinkedInScraper, "scrape_all", li_scrape_all)
    monkeypatch.setattr(GreenhouseScraper, "scrape_all", gh_scrape_all)

    config = {"search_queries": ["q"], "scraping": {"rate_limit_seconds": 0}}
    jobs, counts = main_mod.run_scrapers(config)

    assert {j.url for j in jobs} == {"https://x/a", "https://x/b"}
    assert counts == {"LinkedIn": 1, "Greenhouse/Lever": 1}


def test_parallel_merge_keeps_fixed_scraper_order(monkeypatch):
    """A URL returned by both LinkedIn and Greenhouse must be credited to
    LinkedIn (fixed merge order) even when Greenhouse finishes first."""
    import main as main_mod
    from scrapers.linkedin import LinkedInScraper
    from scrapers.greenhouse import GreenhouseScraper

    shared = "https://x/shared"

    def li_scrape_all(self, queries):
        _time.sleep(0.3)  # force LinkedIn to finish AFTER Greenhouse
        return [_posting("S1", "CoS1", shared), _posting("L", "CoL", "https://x/l")]

    def gh_scrape_all(self, queries):
        return [_posting("S2", "CoS2", shared), _posting("G", "CoG", "https://x/g")]

    monkeypatch.setattr(LinkedInScraper, "scrape_all", li_scrape_all)
    monkeypatch.setattr(GreenhouseScraper, "scrape_all", gh_scrape_all)

    config = {"search_queries": ["q"], "scraping": {"rate_limit_seconds": 0}}
    jobs, counts = main_mod.run_scrapers(config)

    assert counts == {"LinkedIn": 2, "Greenhouse/Lever": 1}
    winner = next(j for j in jobs if j.url == shared)
    assert winner.title == "S1", "shared URL must belong to LinkedIn (merge order), not completion order"


def test_one_scraper_failure_does_not_block_others(monkeypatch):
    """A scraper raising inside its thread logs the error, counts 0, and the
    other scrapers' jobs still come through."""
    import main as main_mod
    from scrapers.linkedin import LinkedInScraper
    from scrapers.greenhouse import GreenhouseScraper

    def li_scrape_all(self, queries):
        raise RuntimeError("boom")

    def gh_scrape_all(self, queries):
        return [_posting("G", "CoG", "https://x/g")]

    monkeypatch.setattr(LinkedInScraper, "scrape_all", li_scrape_all)
    monkeypatch.setattr(GreenhouseScraper, "scrape_all", gh_scrape_all)

    config = {"search_queries": ["q"], "scraping": {"rate_limit_seconds": 0}}
    jobs, counts = main_mod.run_scrapers(config)

    assert {j.url for j in jobs} == {"https://x/g"}
    assert counts["LinkedIn"] == 0
    assert counts["Greenhouse/Lever"] == 1


def test_linkedin_company_pass_survives_keyword_failure_and_merges_last(monkeypatch):
    """The company-targeted pass still runs when the keyword pass raised, stays
    on the LinkedIn thread (after the keyword pass), and merges AFTER all base
    scrapers — so a URL Greenhouse also found is credited to Greenhouse."""
    import main as main_mod
    from scrapers.linkedin import LinkedInScraper
    from scrapers.greenhouse import GreenhouseScraper

    shared = "https://x/shared"

    def li_scrape_all(self, queries):
        raise RuntimeError("keyword pass down")

    def li_scrape_companies(self, queries):
        return [_posting("C-dup", "CoCdup", shared), _posting("C", "CoC", "https://x/c")]

    def gh_scrape_all(self, queries):
        _time.sleep(0.3)  # company pass finishes first; merge order must still win
        return [_posting("S", "CoS", shared)]

    monkeypatch.setattr(LinkedInScraper, "scrape_all", li_scrape_all)
    monkeypatch.setattr(LinkedInScraper, "scrape_companies", li_scrape_companies)
    monkeypatch.setattr(GreenhouseScraper, "scrape_all", gh_scrape_all)

    config = {
        "search_queries": ["q"],
        "scraping": {"rate_limit_seconds": 0},
        "linkedin_company_searches": True,
    }
    jobs, counts = main_mod.run_scrapers(config)

    winner = next(j for j in jobs if j.url == shared)
    assert winner.title == "S", "company-pass jobs must merge after base scrapers"
    assert counts["LinkedIn"] == 1  # 0 from failed keyword pass + 1 unique company job
    assert counts["Greenhouse/Lever"] == 1
    assert any(j.url == "https://x/c" for j in jobs)


def test_malformed_scraper_result_only_zeroes_that_source(monkeypatch):
    """A scraper returning a malformed list (e.g. a None element) must lose
    only its own source — the old sequential loop's try wrapped scrape AND
    merge, so the merge exception never aborted the whole run."""
    import main as main_mod
    from scrapers.linkedin import LinkedInScraper
    from scrapers.greenhouse import GreenhouseScraper

    def li_scrape_all(self, queries):
        return [_posting("A", "CoA", "https://x/a")]

    def gh_scrape_all(self, queries):
        return [None]  # parse bug: a non-JobPosting slipped into the list

    monkeypatch.setattr(LinkedInScraper, "scrape_all", li_scrape_all)
    monkeypatch.setattr(GreenhouseScraper, "scrape_all", gh_scrape_all)

    config = {"search_queries": ["q"], "scraping": {"rate_limit_seconds": 0}}
    jobs, counts = main_mod.run_scrapers(config)

    assert {j.url for j in jobs} == {"https://x/a"}
    assert counts["Greenhouse/Lever"] == 0
    assert counts["LinkedIn"] == 1


def test_local_layer_discarded_when_companies_pass_fails(monkeypatch):
    """If scrape_local succeeds but scrape_local_companies raises, the whole
    local layer is discarded and 'LinkedIn Local' stays unset — a partial
    merge would feed the source-regression baseline a healthy-looking row
    while the fallback half is broken."""
    import main as main_mod
    from scrapers.linkedin import LinkedInScraper
    from scrapers.greenhouse import GreenhouseScraper

    monkeypatch.setattr(
        LinkedInScraper, "scrape_all", lambda self, queries: []
    )
    monkeypatch.setattr(GreenhouseScraper, "scrape_all", lambda self, queries: [])
    monkeypatch.setattr(
        LinkedInScraper, "scrape_local",
        lambda self: [_posting("L", "CoL", "https://x/local")],
    )

    def boom(self):
        raise RuntimeError("local companies pass down")

    monkeypatch.setattr(LinkedInScraper, "scrape_local_companies", boom)

    config = {
        "search_queries": ["q"],
        "scraping": {"rate_limit_seconds": 0},
        "local_search_queries": ["local q"],
    }
    jobs, counts = main_mod.run_scrapers(config)

    assert "LinkedIn Local" not in counts
    assert not any(j.url == "https://x/local" for j in jobs)


# ---------------------------------------------------------------------------
# Batch streaming (scoring-behind-scrape): run_scrapers can hand each
# scraper's jobs to an on_batch callback ON THE MAIN THREAD as that scraper
# finishes — while slower scrapers are still running — so saving and LLM
# scoring overlap the scrape. Ranks encode today's fixed merge order for the
# reconciling saver (LinkedIn=0, others by list position, LinkedIn
# company/local last).
# ---------------------------------------------------------------------------


def test_on_batch_streams_fast_scraper_before_slow_one_finishes(monkeypatch):
    import main as main_mod
    from scrapers.linkedin import LinkedInScraper
    from scrapers.greenhouse import GreenhouseScraper

    events = []
    gh_batch_seen = threading.Event()

    def li_scrape_all(self, queries):
        assert gh_batch_seen.wait(timeout=5), (
            "Greenhouse's batch must reach on_batch while LinkedIn still runs")
        return [_posting("L", "CoL", "https://x/l")]

    def gh_scrape_all(self, queries):
        return [_posting("G", "CoG", "https://x/g")]

    def on_batch(name, batch, rank):
        events.append((name, [j.url for j in batch], rank,
                       threading.current_thread() is threading.main_thread()))
        if name == "Greenhouse/Lever":
            gh_batch_seen.set()

    monkeypatch.setattr(LinkedInScraper, "scrape_all", li_scrape_all)
    monkeypatch.setattr(GreenhouseScraper, "scrape_all", gh_scrape_all)

    config = {"search_queries": ["q"], "scraping": {"rate_limit_seconds": 0}}
    jobs, counts = main_mod.run_scrapers(config, on_batch=on_batch)

    assert ("Greenhouse/Lever", ["https://x/g"], 1, True) in events
    assert ("LinkedIn", ["https://x/l"], 0, True) in events
    assert counts == {"LinkedIn": 1, "Greenhouse/Lever": 1}
    assert {j.url for j in jobs} == {"https://x/l", "https://x/g"}


def test_on_batch_streams_linkedin_subpasses_separately(monkeypatch):
    """The keyword pass must stream BEFORE the company pass completes (the
    chain runs serially for minutes), with the company pass ranked last —
    matching its position in the old fixed merge order."""
    import main as main_mod
    from scrapers.linkedin import LinkedInScraper
    from scrapers.greenhouse import GreenhouseScraper

    events = []
    kw_streamed = threading.Event()

    def li_scrape_all(self, queries):
        return [_posting("K", "CoK", "https://x/kw")]

    def li_scrape_companies(self, queries):
        assert kw_streamed.wait(timeout=5), (
            "keyword batch must stream before the company pass finishes")
        return [_posting("C", "CoC", "https://x/co")]

    def on_batch(name, batch, rank):
        events.append((name, rank))
        if any(j.url == "https://x/kw" for j in batch):
            kw_streamed.set()

    monkeypatch.setattr(LinkedInScraper, "scrape_all", li_scrape_all)
    monkeypatch.setattr(LinkedInScraper, "scrape_companies", li_scrape_companies)
    monkeypatch.setattr(
        GreenhouseScraper, "scrape_all",
        lambda self, q: [_posting("G", "CoG", "https://x/g")])

    config = {
        "search_queries": ["q"],
        "scraping": {"rate_limit_seconds": 0},
        "linkedin_company_searches": True,
    }
    jobs, counts = main_mod.run_scrapers(config, on_batch=on_batch)

    ranks = dict(events)
    assert ranks["LinkedIn"] == 0
    assert ranks["LinkedIn company"] > ranks["Greenhouse/Lever"] == 1
    assert counts["LinkedIn"] == 2
    assert counts["Greenhouse/Lever"] == 1


def test_on_batch_exception_deferred_until_scrape_completes(monkeypatch):
    """A save-side failure must not kill in-flight scrapers: the callback
    error is logged, remaining batches still stream, and the exception
    surfaces after the scrape — where a save_jobs crash landed today."""
    import main as main_mod
    import pytest as _pytest
    from scrapers.linkedin import LinkedInScraper
    from scrapers.greenhouse import GreenhouseScraper

    seen = []

    def li_scrape_all(self, queries):
        _time.sleep(0.2)
        return [_posting("L", "CoL", "https://x/l")]

    def gh_scrape_all(self, queries):
        return [_posting("G", "CoG", "https://x/g")]

    def on_batch(name, batch, rank):
        seen.append(name)
        if name == "Greenhouse/Lever":
            raise RuntimeError("save blew up")

    monkeypatch.setattr(LinkedInScraper, "scrape_all", li_scrape_all)
    monkeypatch.setattr(GreenhouseScraper, "scrape_all", gh_scrape_all)

    config = {"search_queries": ["q"], "scraping": {"rate_limit_seconds": 0}}
    with _pytest.raises(RuntimeError, match="save blew up"):
        main_mod.run_scrapers(config, on_batch=on_batch)
    assert "LinkedIn" in seen, "later batches must still stream after a callback error"
