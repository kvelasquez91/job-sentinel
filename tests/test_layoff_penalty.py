"""Layoff-flagged companies get a small keyword-score penalty at save time."""
import main as main_mod
from engine.scorer import JobScorer
from scrapers.base import JobPosting


def _setup(tmp_path, penalty):
    conn = main_mod.init_database(str(tmp_path / "jobs.db"))
    conn.execute(
        "INSERT INTO company_insights (company_name, company_name_normalized, "
        "has_recent_layoffs, fetched_at) VALUES ('Shrinking Co', 'shrinking co', 1, '2026-07-01')")
    conn.commit()
    scorer = JobScorer({"scoring": {"alert_threshold": 60, "layoff_penalty": penalty}})
    return conn, scorer


def _job(company):
    return JobPosting(
        title="Senior Product Manager AI", company=company, location="Remote",
        url=f"https://x/{company}", description="AI LLM GenAI product leadership")


def test_layoff_company_scores_lower(tmp_path):
    conn, scorer = _setup(tmp_path, penalty=5)
    main_mod.save_jobs(conn, [_job("Shrinking Co"), _job("Steady Co")], 1, scorer)
    rows = dict(conn.execute("SELECT company, score FROM jobs").fetchall())
    assert rows["Shrinking Co"] == rows["Steady Co"] - 5
    expl = conn.execute(
        "SELECT match_explanation FROM jobs WHERE company = 'Shrinking Co'").fetchone()[0]
    assert "layoff" in expl.lower()


def test_penalty_zero_disables(tmp_path):
    conn, scorer = _setup(tmp_path, penalty=0)
    main_mod.save_jobs(conn, [_job("Shrinking Co"), _job("Steady Co")], 1, scorer)
    rows = dict(conn.execute("SELECT company, score FROM jobs").fetchall())
    assert rows["Shrinking Co"] == rows["Steady Co"]


def test_apply_layoff_penalty_flagged_company():
    from engine.scorer import apply_layoff_penalty
    assert apply_layoff_penalty(50, "Shrinking Co.", {"shrinking co"}, 5) == 45


def test_apply_layoff_penalty_not_flagged_or_disabled():
    from engine.scorer import apply_layoff_penalty
    assert apply_layoff_penalty(50, "Steady Co", {"shrinking co"}, 5) == 50
    assert apply_layoff_penalty(50, "Shrinking Co", {"shrinking co"}, 0) == 50
    assert apply_layoff_penalty(0, "Shrinking Co", {"shrinking co"}, 5) == 0
    assert apply_layoff_penalty(3, "Shrinking Co", {"shrinking co"}, 5) == 0


def test_load_layoff_companies_reads_flags(tmp_path):
    from engine.scorer import load_layoff_companies
    conn, _ = _setup(tmp_path, penalty=5)  # _setup inserts 'shrinking co' flagged
    assert load_layoff_companies(conn) == {"shrinking co"}


def test_load_layoff_companies_missing_table(tmp_path):
    import sqlite3
    from engine.scorer import load_layoff_companies
    conn = sqlite3.connect(str(tmp_path / "bare.db"))  # no tables at all
    assert load_layoff_companies(conn) == set()


def test_reblend_preserves_penalty(tmp_path):
    """THE 2026-07-12 audit bug: a reblend immediately after save must be a
    no-op for a penalized row — before this fix, reblend stripped the -5."""
    from engine.llm_scorer import LLMScorer
    conn, scorer = _setup(tmp_path, penalty=5)
    main_mod.save_jobs(conn, [_job("Shrinking Co")], 1, scorer)
    before = conn.execute("SELECT score FROM jobs").fetchone()[0]
    changed = LLMScorer(db_path=str(tmp_path / "jobs.db")).reblend_all(
        profile="testuser",
        config={"scoring": {"alert_threshold": 60, "layoff_penalty": 5}})
    after = conn.execute("SELECT score FROM jobs").fetchone()[0]
    assert changed == 0
    assert after == before
