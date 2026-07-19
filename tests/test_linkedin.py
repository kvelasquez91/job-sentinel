"""Tests for LinkedIn URL canonicalization and 429 rate-limit handling.

Regressions:
  - _normalize_job_url only matched digits immediately after /jobs/view/, so
    real slug-form URLs (numeric ID at the end) were stored raw and the same
    job appeared under multiple URLs (66 duplicate rows).
  - A 429 slept 30s then discarded the request (no retry, no cap), burning
    ~17 of a 35-minute run on rate-limit sleeps that bought nothing.
  - The per-card politeness sleep ran on top of each detail request's own
    ~1s duration, stretching the intended 1-3s inter-request gap to ~1.8-3.8s
    (~3-4.5 min of dead time per run across keyword/company/local passes).
"""
import pytest
import requests

import scrapers.linkedin as linkedin
from scrapers.linkedin import LinkedInScraper


def _sc():
    return LinkedInScraper({})


class FakeResp:
    def __init__(self, status, headers=None, text="OK"):
        self.status_code = status
        self.headers = headers or {}
        self.text = text

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code}")


class FakeSession:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = 0

    def get(self, url, **kwargs):
        self.calls += 1
        return self._responses.pop(0) if self._responses else FakeResp(200)


# --- URL canonicalization ---

def test_normalize_slug_form_url_extracts_trailing_id():
    url = ("https://www.linkedin.com/jobs/view/"
           "senior-product-manager-geo-consumer-at-google-4433250484?refId=abc")
    assert LinkedInScraper._normalize_job_url(url) == \
        "https://www.linkedin.com/jobs/view/4433250484/"


def test_normalize_numeric_form_url():
    url = "https://www.linkedin.com/jobs/view/1234567/?trk=xyz"
    assert LinkedInScraper._normalize_job_url(url) == \
        "https://www.linkedin.com/jobs/view/1234567/"


def test_same_id_different_slugs_collapse_to_one_url():
    a = LinkedInScraper._normalize_job_url(
        "https://www.linkedin.com/jobs/view/senior-pm-geo-consumer-at-google-4433250484")
    b = LinkedInScraper._normalize_job_url(
        "https://www.linkedin.com/jobs/view/senior-pm-geo-consumer-maps-at-google-4433250484")
    assert a == b == "https://www.linkedin.com/jobs/view/4433250484/"


def test_normalize_no_id_returns_stripped_href():
    assert LinkedInScraper._normalize_job_url(
        "https://www.linkedin.com/jobs/view/?x=1") == \
        "https://www.linkedin.com/jobs/view/"


# --- 429 handling ---

def test_fetch_page_retries_429_then_returns_content(monkeypatch):
    monkeypatch.setattr(linkedin.time, "sleep", lambda *a, **k: None)
    sc = _sc()
    sc.session = FakeSession([FakeResp(429), FakeResp(200, text="<html>ok</html>")])
    assert sc._fetch_page("https://www.linkedin.com/jobs/search/?x") == "<html>ok</html>"
    assert sc.session.calls == 2  # retried after the 429


def test_repeated_429s_trip_circuit_breaker_and_skip_descriptions(monkeypatch):
    monkeypatch.setattr(linkedin.time, "sleep", lambda *a, **k: None)
    sc = _sc()
    sc.session = FakeSession([FakeResp(429)] * 50)
    for _ in range(4):
        sc._fetch_page("https://www.linkedin.com/jobs/search/?x")
    assert sc._rate_limited is True
    # once tripped, per-card description fetches are skipped without a request
    calls_before = sc.session.calls
    assert sc._fetch_job_description("https://www.linkedin.com/jobs/view/1/") == ""
    assert sc.session.calls == calls_before


# ---------------------------------------------------------------------------
# Local-area searches
# ---------------------------------------------------------------------------

LOCAL_CONFIG = {
    "search_queries": ["Senior Product Manager AI remote"],
    "local_search_queries": ["Digital Product Manager", "Director Digital Transformation"],
    "linkedin_local_company_searches": {"Milliken": 12345},
    "scraping": {"rate_limit_seconds": 0},
}

# LinkedInScraper.LOCAL_LOCATION is a class attribute bound at import time to
# profile_policy.LINKEDIN_LOCAL_GEO (ambient — "" in a neutral tree). Patch
# it directly so the URL-building test is config-agnostic.
_FIXTURE_LOCAL_GEO = "Springfield, Illinois, United States"


def test_scrape_local_uses_configured_location_and_no_remote_filter(monkeypatch):
    from scrapers.linkedin import LinkedInScraper
    monkeypatch.setattr(LinkedInScraper, "LOCAL_LOCATION", _FIXTURE_LOCAL_GEO)
    sc = LinkedInScraper(LOCAL_CONFIG)
    urls = []
    sc._fetch_page = lambda url: urls.append(url) or None
    sc.scrape_local()
    assert len(urls) == 2
    for url in urls:
        assert "location=Springfield%2C+Illinois%2C+United+States" in url
        assert "f_WT=2" not in url   # no remote-only work-type filter


def test_scrape_local_companies_uses_local_queries_only():
    from scrapers.linkedin import LinkedInScraper
    sc = LinkedInScraper(LOCAL_CONFIG)
    urls = []
    sc._fetch_page = lambda url: urls.append(url) or None
    sc.scrape_local_companies()
    # 1 company x 2 local queries — NOT the full search_queries set
    assert len(urls) == 2
    assert all("f_C=12345" in u for u in urls)


def test_local_methods_empty_without_config():
    from scrapers.linkedin import LinkedInScraper
    sc = LinkedInScraper({"search_queries": ["x"], "scraping": {"rate_limit_seconds": 0}})
    assert sc.scrape_local() == []
    assert sc.scrape_local_companies() == []


# --- dedup-before-fetch ---

def _card_html(job_id, title="Senior Product Manager, AI"):
    return (
        f'<div class="base-card">'
        f'<a class="base-card__full-link" '
        f'href="https://www.linkedin.com/jobs/view/{job_id}/?refId=x&trk=y"></a>'
        f'<h3 class="base-search-card__title">{title}</h3>'
        f'<h4 class="base-search-card__subtitle">Acme Corp</h4>'
        f'<span class="job-search-card__location">United States (Remote)</span>'
        f'</div>'
    )


def _page_html(*job_ids):
    return "<div>" + "".join(_card_html(j) for j in job_ids) + "</div>"


def test_known_url_card_skips_detail_fetch(monkeypatch):
    known = {"https://www.linkedin.com/jobs/view/111/"}
    sc = LinkedInScraper({}, known_urls=known)
    fetched = []
    monkeypatch.setattr(
        sc, "_fetch_job_description", lambda url: fetched.append(url) or "desc"
    )
    jobs, cards_seen = sc._parse_html(_page_html(111, 222), "q")
    assert cards_seen == 2
    assert [j.url for j in jobs] == ["https://www.linkedin.com/jobs/view/222/"]
    assert fetched == ["https://www.linkedin.com/jobs/view/222/"]
    assert sc._known_skipped == 1


def test_seen_url_not_reemitted_across_calls(monkeypatch):
    sc = LinkedInScraper({})
    monkeypatch.setattr(sc, "_fetch_job_description", lambda url: "")
    jobs1, _ = sc._parse_html(_page_html(111), "q")
    jobs2, _ = sc._parse_html(_page_html(111, 333), "q")
    assert [j.url for j in jobs1] == ["https://www.linkedin.com/jobs/view/111/"]
    assert [j.url for j in jobs2] == ["https://www.linkedin.com/jobs/view/333/"]


def test_parse_html_respects_limit(monkeypatch):
    sc = LinkedInScraper({})
    monkeypatch.setattr(sc, "_fetch_job_description", lambda url: "")
    jobs, cards_seen = sc._parse_html(_page_html(1, 2, 3, 4, 5), "q", limit=3)
    assert cards_seen == 3
    assert len(jobs) == 3


def test_company_search_skips_urls_seen_by_keyword_path(monkeypatch):
    sc = LinkedInScraper(
        {"linkedin_company_searches": {"Acme": 42}, "search_queries": []}
    )
    monkeypatch.setattr(sc, "_fetch_job_description", lambda url: "")
    monkeypatch.setattr(sc, "_fetch_page", lambda url: _page_html(111, 444))
    monkeypatch.setattr(linkedin.time, "sleep", lambda s: None)
    sc.seen_urls.add("https://www.linkedin.com/jobs/view/111/")
    jobs = sc._run_company_searches({"Acme": 42}, ["PM"])
    assert [j.url for j in jobs] == ["https://www.linkedin.com/jobs/view/444/"]


# --- pagination with early-stop ---

def _paging_scraper(monkeypatch, pages, config=None, known_urls=None):
    """Scraper whose _fetch_page returns queued page HTML and records URLs."""
    sc = LinkedInScraper(config or {}, known_urls=known_urls)
    monkeypatch.setattr(sc, "_fetch_job_description", lambda url: "")
    monkeypatch.setattr(linkedin.time, "sleep", lambda s: None)
    calls = []

    def fake_fetch(url):
        calls.append(url)
        return pages[len(calls) - 1] if len(calls) <= len(pages) else ""

    monkeypatch.setattr(sc, "_fetch_page", fake_fetch)
    return sc, calls


def test_pagination_walks_and_stops_at_card_cap(monkeypatch):
    page0 = _page_html(*range(100, 125))   # 25 new cards
    page1 = _page_html(*range(200, 225))   # 25 new cards
    sc, calls = _paging_scraper(monkeypatch, [page0, page1],
                                config={"linkedin_max_pages": 2})
    jobs = sc.scrape("q")
    assert len(jobs) == 50               # cap = 2 * 25
    assert len(calls) == 2
    assert "start=25" in calls[1]
    assert "seeMoreJobPostings" in calls[1]


def test_max_pages_floors_at_one_for_zero_or_negative():
    assert LinkedInScraper({"linkedin_max_pages": 0}).max_pages == 1
    assert LinkedInScraper({"linkedin_max_pages": -5}).max_pages == 1
    assert LinkedInScraper({"linkedin_max_pages": 3}).max_pages == 3
    assert LinkedInScraper({}).max_pages == 3  # default unchanged


def test_fragment_start_counts_cards_not_new_jobs(monkeypatch):
    known = {f"https://www.linkedin.com/jobs/view/{i}/" for i in (100, 101)}
    page0 = _page_html(100, 101, 102)    # 3 cards, 1 new
    page1 = _page_html()                 # empty -> halt
    sc, calls = _paging_scraper(monkeypatch, [page0, page1], known_urls=known)
    jobs = sc.scrape("q")
    assert len(jobs) == 1
    assert "start=3" in calls[1]         # cards seen, not new-job count


def test_early_stop_after_two_consecutive_allknown_pages(monkeypatch):
    known = {f"https://www.linkedin.com/jobs/view/{i}/" for i in (201, 202, 301, 302)}
    page0 = _page_html(100, 101)         # 2 new
    page1 = _page_html(201, 202)         # all known -> streak 1
    page2 = _page_html(301, 302)         # all known -> streak 2, stop
    page3 = _page_html(400)              # must never be requested
    sc, calls = _paging_scraper(monkeypatch, [page0, page1, page2, page3],
                                known_urls=known)
    jobs = sc.scrape("q")
    assert len(jobs) == 2
    assert len(calls) == 3


def test_single_allknown_page_does_not_stop_walk(monkeypatch):
    known = {"https://www.linkedin.com/jobs/view/201/"}
    page0 = _page_html(100)              # 1 new
    page1 = _page_html(201)              # all known -> streak 1
    page2 = _page_html(300)              # new again -> streak resets
    page3 = _page_html()                 # empty -> halt
    sc, calls = _paging_scraper(monkeypatch, [page0, page1, page2, page3],
                                known_urls=known)
    jobs = sc.scrape("q")
    assert [j.url for j in jobs] == [
        "https://www.linkedin.com/jobs/view/100/",
        "https://www.linkedin.com/jobs/view/300/",
    ]
    assert len(calls) == 4


def test_empty_page0_returns_no_jobs_single_fetch(monkeypatch):
    sc, calls = _paging_scraper(monkeypatch, [_page_html()])
    assert sc.scrape("q") == []
    assert len(calls) == 1               # no fragment fetch after empty page 0


def test_summary_log_line_emitted(monkeypatch, caplog):
    import logging as _logging
    page0 = _page_html(100)
    sc, calls = _paging_scraper(monkeypatch, [page0, _page_html()])
    with caplog.at_level(_logging.INFO, logger="LinkedInScraper"):
        sc.scrape("my query")
    assert any(
        'LinkedIn query done: "my query" pages=2 cards=1 known_skipped=0 new=1'
        in r.message for r in caplog.records
    )


# --- seniority facet ---

def test_facet_appended_only_for_listed_queries():
    sc = LinkedInScraper(
        {"linkedin_seniority_facet_queries": ["AI Program Manager remote"]}
    )
    faceted = sc._build_search_url("AI Program Manager remote")
    plain = sc._build_search_url("Senior Product Manager AI remote")
    assert faceted.endswith("&f_E=4%2C5%2C6")
    assert "f_E=" not in plain


def test_facet_applies_to_fragment_urls_too():
    sc = LinkedInScraper(
        {"linkedin_seniority_facet_queries": ["Product Owner AI remote"]}
    )
    frag = sc._build_fragment_url("Product Owner AI remote", start=25)
    assert "seeMoreJobPostings" in frag
    assert "start=25" in frag
    assert frag.endswith("&f_E=4%2C5%2C6")


def test_no_facet_config_key_means_no_facets():
    sc = LinkedInScraper({})
    assert "f_E=" not in sc._build_search_url("AI Program Manager remote")


# --- detail-fetch pacing (wall-clock anchored) ---

def _detail_clock(monkeypatch):
    """Patch scrapers.linkedin's time so pacing math is wall-clock-independent:
    sleep() records its argument and advances the fake clock; monotonic()
    reads it. Returns (sleeps, clock); fakes bump clock["t"] to simulate a
    request taking real time."""
    sleeps = []
    clock = {"t": 1000.0}

    def fake_sleep(s):
        sleeps.append(s)
        clock["t"] += s

    monkeypatch.setattr("scrapers.linkedin.time.sleep", fake_sleep)
    monkeypatch.setattr("scrapers.linkedin.time.monotonic", lambda: clock["t"])
    return sleeps, clock


def _pin_jitter(monkeypatch, value):
    """Pin random.uniform to a fixed target, recording each requested range."""
    ranges = []

    def fake_uniform(a, b):
        ranges.append((a, b))
        return value

    monkeypatch.setattr("scrapers.linkedin.random.uniform", fake_uniform)
    return ranges


def test_detail_sleep_subtracts_elapsed_fetch_time(monkeypatch):
    """The 1-3s politeness jitter is a target gap between detail-request
    STARTS: time the previous request already spent counts toward it, so the
    pre-fetch sleep is only the remainder, never the full jitter again."""
    sc = _sc()
    sleeps, clock = _detail_clock(monkeypatch)
    ranges = _pin_jitter(monkeypatch, 2.5)

    def fake_fetch(url):
        clock["t"] += 1.0  # the detail request itself takes 1s
        return "<html></html>"

    monkeypatch.setattr(sc, "_fetch_page", fake_fetch)

    sc._fetch_job_description("https://www.linkedin.com/jobs/view/1/")
    sc._fetch_job_description("https://www.linkedin.com/jobs/view/2/")

    assert ranges == [(1, 3), (1, 3)]  # jitter range itself is untouched
    # First fetch has no anchor yet -> full target; second pays only the
    # remainder after the 1s the previous request already consumed.
    assert sleeps == [pytest.approx(2.5), pytest.approx(1.5)]


def test_slow_detail_fetch_yields_zero_sleep(monkeypatch):
    """A request slower than the target gap (e.g. a 429-backoff retry) already
    IS the politeness pause — no additional sleep gets stacked on top."""
    sc = _sc()
    sleeps, clock = _detail_clock(monkeypatch)
    _pin_jitter(monkeypatch, 2.0)

    def fake_fetch(url):
        clock["t"] += 5.0  # slow request, longer than the 2.0s target
        return "<html></html>"

    monkeypatch.setattr(sc, "_fetch_page", fake_fetch)

    sc._fetch_job_description("https://www.linkedin.com/jobs/view/1/")
    sc._fetch_job_description("https://www.linkedin.com/jobs/view/2/")

    assert sleeps[1] == pytest.approx(0.0)
    assert all(s <= 2.0 for s in sleeps)  # no sleep ever exceeds the target


def test_anchor_advances_even_when_no_sleep_was_needed(monkeypatch):
    """The anchor must track the LAST request's start even when that request
    needed no pre-sleep: a slow fetch that freezes the anchor in the past
    would make every later elapsed-time reading grow monotonically and
    silently zero out pacing for the rest of the run."""
    sc = _sc()
    sleeps, clock = _detail_clock(monkeypatch)
    _pin_jitter(monkeypatch, 2.0)

    durations = iter([5.0, 0.1, 0.1])  # slow request, then two fast ones

    def fake_fetch(url):
        clock["t"] += next(durations)
        return "<html></html>"

    monkeypatch.setattr(sc, "_fetch_page", fake_fetch)

    sc._fetch_job_description("https://www.linkedin.com/jobs/view/1/")
    sc._fetch_job_description("https://www.linkedin.com/jobs/view/2/")
    sc._fetch_job_description("https://www.linkedin.com/jobs/view/3/")

    # 1st: full target (no anchor). 2nd: elapsed 5.0 >= 2.0 -> zero sleep.
    # 3rd: anchored to the 2nd request's START, elapsed 0.1 -> sleep 1.9 —
    # NOT another zero measured from the stale pre-slow-fetch anchor.
    assert sleeps == [
        pytest.approx(2.0), pytest.approx(0.0), pytest.approx(1.9),
    ]


def test_card_loop_paces_at_target_not_target_plus_fetch(monkeypatch):
    """Through the real _parse_card path, consecutive detail requests start
    ~target seconds apart; only the first card pays the full sleep."""
    sc = _sc()
    sleeps, clock = _detail_clock(monkeypatch)
    _pin_jitter(monkeypatch, 2.0)

    def fake_fetch(url):
        clock["t"] += 1.0
        return ""  # no description HTML — pacing must still apply

    monkeypatch.setattr(sc, "_fetch_page", fake_fetch)

    sc._parse_html(_page_html(1, 2, 3), "q")

    assert sleeps == [
        pytest.approx(2.0), pytest.approx(1.0), pytest.approx(1.0),
    ]
