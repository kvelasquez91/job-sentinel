"""rejudge_filter(ids=[...]) — the content-scoped surgical re-judge.

Only the given ids are re-judged; everything else keeps its verdicts.
The judge call itself is stubbed — no CLI in tests."""
import json
import sqlite3

from engine.llm_scorer import LLMScorer

_BLOB = json.dumps({
    "must_haves": [{"term": "AI", "aliases": [], "verdict": "absent"}],
    "knockouts": [{"requirement": "Onsite in San Francisco, CA",
                   "verdict": "failed"}],
    "title_variants": [],
    "title_alignment": "close",
    "inventory_sha256": "stale-sha",
})

_JUDGED = {
    "must_haves": [{"term": "AI", "verdict": "evidenced", "evidence": "x"}],
    "knockouts": [{"requirement": "Onsite in San Francisco, CA",
                   "verdict": "met", "reason": "high-comp exception"}],
    "title_claim": "close",
}


def _make_db(path):
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE jobs (id INTEGER PRIMARY KEY, title TEXT, "
        "description TEXT DEFAULT '', salary_min REAL, salary_max REAL, "
        "status TEXT DEFAULT 'new', profile TEXT DEFAULT 'testuser', "
        "filter_score REAL, filter_score_master REAL, filter_source TEXT, "
        "filter_knockout INTEGER, filter_json TEXT, "
        "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)")
    for i in (1, 2):
        conn.execute(
            "INSERT INTO jobs (id, title, salary_min, salary_max, "
            "filter_source, filter_knockout, filter_json) "
            "VALUES (?, 'Head of Product', 305000, 385000, 'master', 1, ?)",
            (i, _BLOB))
    conn.commit()
    conn.close()


def test_ids_scopes_the_rejudge(tmp_path, monkeypatch):
    db = str(tmp_path / "jobs.db")
    _make_db(db)
    s = LLMScorer(db_path=db)
    s._available = True
    s.judge_basis_text = "CANDIDATE INVENTORY"
    s.judge_basis_sha = "fresh-sha"
    s.judge_basis = "inventory"
    monkeypatch.setattr(s, "_judge_filter", lambda *a, **k: dict(_JUDGED))

    assert s.rejudge_filter(ids=[1], workers=1, profile="testuser") == 1

    conn = sqlite3.connect(db)
    ko1 = conn.execute("SELECT filter_knockout FROM jobs WHERE id=1").fetchone()[0]
    ko2 = conn.execute("SELECT filter_knockout FROM jobs WHERE id=2").fetchone()[0]
    conn.close()
    assert ko1 == 0  # location KO lifted by the stubbed 'met' verdict
    assert ko2 == 1  # untouched — not in ids
