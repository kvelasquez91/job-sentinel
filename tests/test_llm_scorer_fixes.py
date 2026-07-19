"""Tests for the #7 LLM-scoring fixes:

- _parse_llm_response returns NULL (not a fabricated 50) when dimensions are absent.
- _apply_caps scans the same description window the LLM prompt saw (not a narrower one).
- rescore_all(force=True) recovers the keyword baseline before clearing llm_score
  (so a forced rescore does not compound the blend into itself).
- A bounded backfill retries genuinely-unscored rows (oldest first, skipping
  expired/terminal statuses) so transient failures self-heal.
"""
import os
import tempfile

import engine.llm_scorer as scorer_mod
from engine.llm_scorer import LLMScorer
import main
from policy_fixtures import patch_pm_prefilter

_FULL_DIMS = (
    '{"role_match":28,"seniority_match":16,"remote_location":20,"ai_domain_fit":18,'
    '"comp_match":9,"explanation":"Strong PM/AI fit.","salary_min":230000,"salary_max":260000}'
)


def _fake_run(*_a, **_k):
    return {"text": _FULL_DIMS, "usage": {}, "cost_usd": 0.0, "is_error": False, "session_id": None}


def _tmpdb():
    path = os.path.join(tempfile.mkdtemp(), "t.db")
    return path, main.init_database(path)


def _insert(conn, url, status="new", score=50, llm_score=None, created_at="2026-06-01 00:00:00"):
    conn.execute(
        "INSERT INTO jobs (title,company,url,description,score,llm_score,profile,status,created_at) "
        "VALUES (?,?,?,?,?,?,?,?,?)",
        ("Senior Product Manager AI", "Acme", url,
         "We build AI/ML/LLM products for the enterprise.", score, llm_score, "testuser", status, created_at),
    )
    conn.commit()


# --- #3: fabricated-50 -> NULL on missing dims ---

def test_missing_dims_and_fit_score_returns_none():
    s = LLMScorer(db_path=":memory:")
    assert s._parse_llm_response('{"explanation":"hi"}') == (None, None, None, None, None, None, None, None)


def test_missing_dims_with_fit_score_still_returns_none():
    s = LLMScorer(db_path=":memory:")
    assert s._parse_llm_response('{"fit_score":77,"explanation":"hi"}') == (None, None, None, None, None, None, None, None)


def test_full_dims_still_score_normally():  # regression guard
    s = LLMScorer(db_path=":memory:")
    score, _expl, _mn, _mx, _emn, _emx, dims, _fr = s._parse_llm_response(
        '{"role_match":25,"seniority_match":15,"remote_location":18,"ai_domain_fit":10,'
        '"comp_match":5,"explanation":"ok"}')
    assert score == 73 and dims is not None


# --- #4: AI-cap window aligned to the prompt window ---

def test_ai_cap_not_applied_when_signal_is_past_2000_chars(monkeypatch):
    patch_pm_prefilter(monkeypatch)
    s = LLMScorer(db_path=":memory:")
    dims = {"role_match": 20, "seniority_match": 11, "remote_location": 20, "ai_domain_fit": 15, "comp_match": 9}
    desc = " " * 2900 + "we build ML pipelines and LLM systems"  # AI signal at char ~2900
    out = s._apply_caps(dims, "Senior Product Manager", "Remote (US)", desc, None)
    assert out["ai_domain_fit"] == 15  # LLM saw it (<=6000); cap must not override


def test_ai_cap_still_fires_without_any_ai_signal(monkeypatch):  # regression guard
    patch_pm_prefilter(monkeypatch)
    s = LLMScorer(db_path=":memory:")
    dims = {"role_match": 20, "seniority_match": 11, "remote_location": 20, "ai_domain_fit": 15, "comp_match": 9}
    out = s._apply_caps(dims, "Senior Product Manager", "Remote (US)", "generic product role, nothing special", None)
    assert out["ai_domain_fit"] == 8


# --- #2: force-rescore recovers the keyword baseline ---

def test_force_rescore_recovers_keyword_baseline(monkeypatch):
    monkeypatch.setattr(scorer_mod, "is_cli_available", lambda: False)  # skip real scoring
    path, conn = _tmpdb()
    # score=68 is a blended value round(0.4*kw + 0.6*60); the true kw was 80.
    _insert(conn, "u1", score=68, llm_score=60)
    LLMScorer(db_path=path).rescore_all(force=True, profile="testuser")
    row = conn.execute("SELECT score, llm_score FROM jobs WHERE url='u1'").fetchone()
    assert row[0] == 80 and row[1] is None


# --- #1: bounded backfill of genuinely-unscored rows ---

def test_backfill_scores_oldest_eligible_and_skips_terminal(monkeypatch):
    monkeypatch.setattr(scorer_mod, "is_cli_available", lambda: True)
    monkeypatch.setattr(scorer_mod, "run_claude", _fake_run)
    path, conn = _tmpdb()
    _insert(conn, "new1", status="new", created_at="2026-01-01 00:00:00")
    _insert(conn, "new2", status="new", created_at="2026-01-02 00:00:00")
    _insert(conn, "new3", status="new", created_at="2026-01-03 00:00:00")
    _insert(conn, "exp", status="expired", created_at="2026-01-01 00:00:00")
    _insert(conn, "applied", status="applied", created_at="2026-01-01 00:00:00")

    scored = LLMScorer(db_path=path).apply_llm_scores_to_db(profile="testuser", backfill_limit=2)
    assert scored == 2

    def _llm(url):
        return conn.execute("SELECT llm_score FROM jobs WHERE url=?", (url,)).fetchone()[0]
    assert _llm("new1") is not None and _llm("new2") is not None  # oldest two
    assert _llm("new3") is None                                    # over the limit
    assert _llm("exp") is None and _llm("applied") is None         # terminal, excluded


# --- daily per-run cap: score the highest keyword-score jobs first, bounded ---

def test_daily_cap_scores_highest_keyword_score_first(monkeypatch):
    monkeypatch.setattr(scorer_mod, "is_cli_available", lambda: True)
    monkeypatch.setattr(scorer_mod, "run_claude", _fake_run)
    path, conn = _tmpdb()
    for url, kw in [("a", 90), ("b", 80), ("c", 70), ("d", 60)]:
        conn.execute(
            "INSERT INTO jobs (title,company,url,description,score,llm_score,profile,status,run_id) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            ("Senior Product Manager AI", "Acme", url,
             "We build AI/ML/LLM products.", kw, None, "testuser", "new", 5),
        )
    conn.commit()

    scored = LLMScorer(db_path=path).apply_llm_scores_to_db(run_id=5, profile="testuser", limit=2)
    assert scored == 2
    got = {r[0] for r in conn.execute("SELECT url FROM jobs WHERE llm_score IS NOT NULL").fetchall()}
    assert got == {"a", "b"}  # only the two highest keyword scores were LLM-scored


# --- comp estimates: parsed, clamped, persisted ---

def test_parse_extracts_comp_estimates():
    s = LLMScorer(db_path=":memory:")
    _sc, _e, _mn, _mx, emin, emax, _d, _fr = s._parse_llm_response(
        '{"role_match":25,"seniority_match":15,"remote_location":18,"ai_domain_fit":10,'
        '"comp_match":5,"explanation":"ok","salary_min":null,"salary_max":null,'
        '"est_total_comp_min":200000,"est_total_comp_max":260000}')
    assert (emin, emax) == (200000.0, 260000.0)


def test_parse_clamps_absurd_estimates_to_none():
    s = LLMScorer(db_path=":memory:")
    _sc, _e, _mn, _mx, emin, emax, _d, _fr = s._parse_llm_response(
        '{"role_match":25,"seniority_match":15,"remote_location":18,"ai_domain_fit":10,'
        '"comp_match":5,"explanation":"ok","est_total_comp_min":5000,"est_total_comp_max":9000000}')
    assert (emin, emax) == (None, None)


def test_estimates_persisted_to_db(monkeypatch):
    monkeypatch.setattr(scorer_mod, "is_cli_available", lambda: True)
    monkeypatch.setattr(scorer_mod, "run_claude", lambda *a, **k: {
        "text": ('{"role_match":28,"seniority_match":16,"remote_location":20,'
                 '"ai_domain_fit":18,"comp_match":9,"explanation":"fit",'
                 '"salary_min":null,"salary_max":null,'
                 '"est_total_comp_min":300000,"est_total_comp_max":380000}'),
        "usage": {}, "cost_usd": 0.0, "is_error": False, "session_id": None})
    path, conn = _tmpdb()
    conn.execute(
        "INSERT INTO jobs (title,company,url,description,score,llm_score,profile,status,run_id) "
        "VALUES ('Director of AI Product','Acme','u-est','We build AI/ML LLM products.',80,NULL,'testuser','new',9)")
    conn.commit()
    LLMScorer(db_path=path).apply_llm_scores_to_db(run_id=9, profile="testuser")
    row = conn.execute("SELECT salary_est_min, salary_est_max FROM jobs WHERE url='u-est'").fetchone()
    assert (row[0], row[1]) == (300000.0, 380000.0)


# --- feedback loop: more/less examples reach the scoring prompt ---

def test_format_feedback_block_empty_when_no_feedback():
    assert scorer_mod.format_feedback_block([], []) == ""


def test_format_feedback_block_lists_examples():
    block = scorer_mod.format_feedback_block(
        ["Head of AI @ Acme"], ["Product Manager, Tires @ RubberCo"])
    assert "CANDIDATE FEEDBACK" in block
    assert "Head of AI @ Acme" in block
    assert "RubberCo" in block


def test_load_feedback_examples_reads_db():
    path, conn = _tmpdb()
    _insert(conn, "fb1")
    _insert(conn, "fb2")
    conn.execute("UPDATE jobs SET feedback = 'more' WHERE url = 'fb1'")
    conn.execute("UPDATE jobs SET feedback = 'less' WHERE url = 'fb2'")
    conn.commit()
    more, less = scorer_mod._load_feedback_examples(path, "testuser")
    assert more == ["Senior Product Manager AI @ Acme"]
    assert less == ["Senior Product Manager AI @ Acme"]


def test_feedback_block_lands_in_prompt(monkeypatch):
    captured = {}

    def spy_run(prompt, **kwargs):
        captured["prompt"] = prompt
        captured["system"] = kwargs.get("system_prompt") or ""
        return _fake_run()

    monkeypatch.setattr(scorer_mod, "run_claude", spy_run)
    s = LLMScorer(db_path=":memory:")
    s._feedback_block = scorer_mod.format_feedback_block(["Head of AI @ Acme"], [])
    s._call_claude("PM", "Co", "AI role", location="Remote")
    # feedback_block is static per-run context — it's formatted into the
    # system template (the cache-eligible prefix), not the per-job user prompt.
    assert "CANDIDATE FEEDBACK" in captured["system"]
    assert "Head of AI @ Acme" in captured["system"]
