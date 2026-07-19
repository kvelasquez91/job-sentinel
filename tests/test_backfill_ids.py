"""apply_llm_scores_to_db(ids=[...]) restricts scoring to the given job ids.

Used by the targeted high-comp backfill so it scores ONLY the qualifying
cleared stubs instead of draining the whole NULL-score pool (which, in the
live DB, also contains unrelated backlog rows). The per-row scoring call is
stubbed — no CLI in tests."""
import sqlite3

import engine.llm_scorer as llm
from engine.llm_scorer import LLMScorer


def _make_db(path):
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE jobs (id INTEGER PRIMARY KEY, title TEXT, company TEXT, "
        "location TEXT, description TEXT DEFAULT '', salary_min REAL, "
        "salary_max REAL, score INTEGER DEFAULT 0, llm_score REAL, "
        "profile TEXT DEFAULT 'testuser', status TEXT DEFAULT 'new', "
        "run_id INTEGER, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)")
    for i in (1, 2, 3):
        conn.execute(
            "INSERT INTO jobs (id, title, company, location, llm_score) "
            "VALUES (?, 'Director of Product', 'Co', 'San Francisco, CA', NULL)",
            (i,))
    conn.commit()
    conn.close()


def test_ids_restricts_scoring_to_those_rows(tmp_path, monkeypatch):
    db = str(tmp_path / "jobs.db")
    _make_db(db)
    s = LLMScorer(db_path=db)
    s._available = True
    monkeypatch.setattr(llm, "_load_feedback_examples", lambda *a, **k: ([], []))

    seen = []

    def _fake_score_one_job(row):
        seen.append(row["id"])
        return None  # nothing to write; we only care which rows were fed in

    monkeypatch.setattr(s, "_score_one_job", _fake_score_one_job)

    s.apply_llm_scores_to_db(ids=[1, 3], profile="testuser")

    assert sorted(seen) == [1, 3]  # row 2 (also NULL) was NOT scored
