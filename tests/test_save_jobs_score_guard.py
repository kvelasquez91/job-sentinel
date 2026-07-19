"""save_jobs must never overwrite an LLM-blended score with a keyword-only one.

jobs.score becomes a 0.4*kw + 0.6*llm blend once llm_score is set. The
duplicate-URL update path used to compare a fresh keyword-only score against
that blend and overwrite it whenever kw > blend — re-inflating keyword-heavy
junk the LLM had buried (2026-07-12 audit: 7 live rows corrupted).
"""
import main as main_mod
from engine.scorer import JobScorer
from scrapers.base import JobPosting


def _setup(tmp_path):
    conn = main_mod.init_database(str(tmp_path / "jobs.db"))
    scorer = JobScorer({"scoring": {"alert_threshold": 60}})
    return conn, scorer


def _incoming(url):
    # Keyword-rich remote posting: keyword score lands well above both
    # stored scores used below (75 under current rules).
    return JobPosting(
        title="Senior Product Manager, AI Platform", company="Acme",
        location="Remote", url=url,
        description="AI ML LLM GenAI product manager enterprise automation")


def test_llm_blended_row_is_not_clobbered(tmp_path):
    conn, scorer = _setup(tmp_path)
    # Stored row: the LLM judged it poorly (llm 10 -> blend 22).
    # normalized_title stays NULL so the incoming job reaches the
    # URL-duplicate branch (see module docstring).
    conn.execute(
        "INSERT INTO jobs (title, company, url, location, description, score, "
        "llm_score, profile) VALUES "
        "('Old Title', 'Acme', 'u1', 'Remote', 'x', 22, 10.0, 'testuser')")
    conn.commit()
    main_mod.save_jobs(conn, [_incoming("u1")], 1, scorer)
    row = conn.execute(
        "SELECT score, llm_score FROM jobs WHERE url='u1'").fetchone()
    assert row["score"] == 22          # blend untouched
    assert row["llm_score"] == 10.0    # LLM component untouched


def test_keyword_only_row_still_updates(tmp_path):
    conn, scorer = _setup(tmp_path)
    conn.execute(
        "INSERT INTO jobs (title, company, url, location, description, score, "
        "llm_score, profile) VALUES "
        "('Old Title', 'Acme', 'u1', 'Remote', 'x', 5, NULL, 'testuser')")
    conn.commit()
    main_mod.save_jobs(conn, [_incoming("u1")], 1, scorer)
    row = conn.execute("SELECT score FROM jobs WHERE url='u1'").fetchone()
    assert row["score"] > 5   # keyword-vs-keyword improvement still allowed


def test_salary_only_update_preserves_llm_explanation(tmp_path):
    conn, scorer = _setup(tmp_path)
    # LLM-scored row with no salary yet. normalized_title stays NULL so the
    # incoming job (different title) reaches the URL-duplicate branch rather
    # than the title-dedup path (see module docstring).
    conn.execute(
        "INSERT INTO jobs (title, company, url, location, description, score, "
        "llm_score, match_explanation, salary_min, salary_max, profile) VALUES "
        "('Old Title', 'Acme', 'u1', 'Remote', 'x', 48, 80.0, "
        "'LLM: strong AI PM fit', NULL, NULL, 'testuser')")
    conn.commit()
    from scrapers.base import JobPosting
    incoming = JobPosting(
        title="Senior Product Manager, AI Platform", company="Acme",
        location="Remote", url="u1",
        description="AI ML LLM GenAI product manager enterprise automation",
        salary_min=200000, salary_max=260000)
    main_mod.save_jobs(conn, [incoming], 1, scorer)
    row = conn.execute(
        "SELECT score, match_explanation, salary_min, salary_max "
        "FROM jobs WHERE url='u1'").fetchone()
    assert row["match_explanation"] == "LLM: strong AI PM fit"  # preserved, not NULL
    assert row["salary_min"] == 200000 and row["salary_max"] == 260000
    assert row["score"] == 48  # LLM blend untouched
