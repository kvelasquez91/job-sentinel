"""The backfill writers must sanitize before writing.

The scraper path is cleaned centrally by JobPosting.__post_init__, but the
backfill script does direct SQL UPDATEs — every writer must run the fetched
text through clean_description or it reintroduces the literal-HTML /
entity-encoded-text bug the sanitizer exists to fix (observed live: raw
'<p><strong>…' stored by the SmartRecruiters backfiller).
"""
import sqlite3

import scripts.backfill_descriptions as bf


def _conn():
    conn = sqlite3.connect(":memory:")
    conn.execute(
        "CREATE TABLE jobs (id INTEGER PRIMARY KEY, url TEXT, source TEXT, "
        "description TEXT)"
    )
    return conn


def test_smartrecruiters_backfill_sanitizes_html(monkeypatch):
    monkeypatch.setattr(bf, "POLITE_DELAY", 0)

    class FakeSR:
        def __init__(self, config):
            pass

        def _fetch_job_ad(self, url):
            return {"sections": {
                "jobDescription": {"text": "<p>Own the roadmap</p>"},
                "qualifications": {"text": "<ul><li>5+ years</li></ul>"},
            }}

    monkeypatch.setattr(bf, "SmartRecruitersScraper", FakeSR)
    conn = _conn()
    conn.execute(
        "INSERT INTO jobs (url, source, description) VALUES "
        "('https://api.smartrecruiters.com/v1/companies/T/postings/1', "
        "'smartrecruiters', '')"
    )

    bf.backfill_smartrecruiters(conn, {})

    desc = conn.execute("SELECT description FROM jobs").fetchone()[0]
    assert "<" not in desc and ">" not in desc, desc
    assert "Own the roadmap" in desc
    assert "5+ years" in desc


def test_workday_backfill_decodes_entities_and_strips(monkeypatch):
    monkeypatch.setattr(bf, "POLITE_DELAY", 0)

    class FakeWD:
        def __init__(self, config):
            pass

        def _establish_session_for_tenant(self, tenant):
            return {}

        def _fetch_description(self, tenant, external_path):
            # Raw HTML with entities, as the (fixed) fetcher now returns.
            return "<p>R&amp;D lead</p>&lt;li&gt;Ship it&lt;/li&gt;"

    monkeypatch.setattr(bf, "WorkdayScraper", FakeWD)
    conn = _conn()
    conn.execute(
        "INSERT INTO jobs (url, source, description) VALUES "
        "('https://acme.wd1.myworkdayjobs.com/en-US/careers/job/City/Role_R1', "
        "'workday', '')"
    )
    config = {"workday_tenants": [{
        "company": "Acme",
        "tenant_url": "acme.wd1.myworkdayjobs.com",
        "company_slug": "acme",
        "site_path": "careers",
    }]}

    bf.backfill_workday(conn, config)

    desc = conn.execute("SELECT description FROM jobs").fetchone()[0]
    assert "&amp;" not in desc and "<" not in desc, desc
    assert "R&D lead" in desc
    assert "Ship it" in desc


def test_eightfold_backfill_uses_central_cleaner(monkeypatch):
    monkeypatch.setattr(bf, "POLITE_DELAY", 0)

    class FakeEF:
        def __init__(self, config):
            pass

        def _fetch_description(self, tenant, pid):
            return "<ul><li>Ship features</li></ul>"

    monkeypatch.setattr(bf, "EightfoldScraper", FakeEF)
    conn = _conn()
    conn.execute(
        "INSERT INTO jobs (url, source, description) VALUES "
        "('https://acme.eightfold.ai/careers/job?pid=123&domain=acme.com', "
        "'eightfold', '')"
    )
    config = {"eightfold_tenants": [{
        "company": "Acme", "subdomain": "acme", "domain": "acme.com",
    }]}

    bf.backfill_eightfold(conn, config)

    desc = conn.execute("SELECT description FROM jobs").fetchone()[0]
    assert "<" not in desc and ">" not in desc, desc
    assert "• Ship features" in desc  # structure preserved, not flattened


def test_default_sources_exclude_wttj():
    """The wttj backfiller refreshes EVERY row (not just empty ones) — it must
    be opt-in, not part of the no-arg default."""
    assert set(bf.DEFAULT_SOURCES) == {"workday", "eightfold", "smartrecruiters"}
    assert "wttj" in bf.BACKFILLERS  # still available explicitly
