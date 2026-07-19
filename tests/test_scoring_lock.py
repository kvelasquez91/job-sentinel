"""Cost-safety guards on the LLM scoring pass.

1. Cross-process lock: a second scoring pass (e.g. a manual backfill script
   overlapping the launchd daily run) must skip with a WARNING instead of
   double-scoring the same NULL rows — double subscription spend.
2. Incremental flushes: results are written every _FLUSH_EVERY completions so
   a mid-pass crash forfeits at most one batch of CLI spend, not the whole
   pass.

Never invokes the real claude CLI — run_claude is mocked, matching
tests/test_llm_scorer.py.
"""
import fcntl
import json
import logging
import os
import sqlite3
from unittest import mock

import pytest

import engine.llm_scorer as scorer_mod
from engine.llm_scorer import LLMScorer

_ENVELOPE_TEXT = (
    '{"role_match":28,"seniority_match":16,"remote_location":20,"ai_domain_fit":18,'
    '"comp_match":9,"fit_score":91,"explanation":"Strong PM/AI fit.",'
    '"salary_min":230000,"salary_max":260000,'
    '"est_total_comp_min":280000,"est_total_comp_max":360000}'
)


def _fake_run(*_args, **_kwargs):
    return {"text": _ENVELOPE_TEXT, "usage": {}, "cost_usd": 0.0,
            "is_error": False, "session_id": None}


_GOOD_DESC = "We build AI/ML LLM products for enterprise GenAI roadmaps. " * 10


def _seed_unscored_jobs(db, n):
    import main as main_mod
    conn = main_mod.init_database(str(db))
    for jid in range(1, n + 1):
        conn.execute(
            "INSERT INTO jobs (id, title, company, location, url, description, "
            "score, status, profile) VALUES (?, 'Director of AI Product', 'Acme', "
            "'Remote (US)', ?, ?, 50, 'new', 'testuser')",
            (jid, f"https://x/{jid}", _GOOD_DESC))
    conn.commit()
    conn.close()


def _scorer(db, monkeypatch):
    s = LLMScorer(db_path=str(db), model="claude-sonnet-5")
    s.judge_basis_text, s.judge_basis_sha, s.judge_basis = (
        "INVENTORY TEXT", "sha1" * 16, "inventory")
    monkeypatch.setattr(scorer_mod, "is_cli_available", lambda: True)
    return s


def _hold_lock(db):
    """Acquire the scoring lock the way a concurrent process would."""
    lock_path = os.path.join(os.path.dirname(os.path.abspath(str(db))),
                             ".scoring.lock")
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o644)
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    return fd


def _release_lock(fd):
    fcntl.flock(fd, fcntl.LOCK_UN)
    os.close(fd)


def _scored_count(db):
    conn = sqlite3.connect(str(db))
    n = conn.execute(
        "SELECT COUNT(*) FROM jobs WHERE llm_score IS NOT NULL").fetchone()[0]
    conn.close()
    return n


# ---------------------------------------------------------------------------
# Cross-process lock
# ---------------------------------------------------------------------------

def test_apply_skips_when_lock_held(tmp_path, monkeypatch, caplog):
    db = tmp_path / "jobs.db"
    _seed_unscored_jobs(db, 1)
    s = _scorer(db, monkeypatch)
    fd = _hold_lock(db)
    try:
        with mock.patch.object(scorer_mod, "run_claude",
                               side_effect=_fake_run) as m:
            with caplog.at_level(logging.WARNING):
                count = s.apply_llm_scores_to_db(profile="testuser", workers=1)
    finally:
        _release_lock(fd)
    assert count == 0
    assert m.call_count == 0                      # zero CLI spend
    assert _scored_count(db) == 0                 # nothing written
    assert any("another scoring pass" in r.message.lower()
               for r in caplog.records)


def test_backfill_filter_skips_when_lock_held(tmp_path, monkeypatch, caplog):
    import main as main_mod
    db = tmp_path / "jobs.db"
    conn = main_mod.init_database(str(db))
    conn.execute(
        "INSERT INTO jobs (id, title, company, location, url, description, "
        "score, llm_score, status, profile) VALUES (1, 'Director of AI Product', "
        "'Acme', 'Remote (US)', 'https://x/1', ?, 50, 80, 'new', 'testuser')",
        (_GOOD_DESC,))
    conn.commit()
    conn.close()
    s = _scorer(db, monkeypatch)
    fd = _hold_lock(db)
    try:
        with mock.patch.object(scorer_mod, "run_claude",
                               side_effect=_fake_run) as m:
            with caplog.at_level(logging.WARNING):
                count = s.backfill_filter(profile="testuser", workers=1)
    finally:
        _release_lock(fd)
    assert count == 0
    assert m.call_count == 0
    assert any("another scoring pass" in r.message.lower()
               for r in caplog.records)


def test_rejudge_filter_skips_when_lock_held(tmp_path, monkeypatch, caplog):
    import main as main_mod
    db = tmp_path / "jobs.db"
    conn = main_mod.init_database(str(db))
    # A stale master-basis row that WOULD be re-judged if the lock weren't held.
    stale_blob = json.dumps({
        "version": 2,
        "must_haves": [{"term": "Generative AI", "aliases": [],
                        "verdict": "absent", "evidence": ""}],
        "title_variants": ["Director of AI Product"],
        "title_alignment": "close", "title_claim": "none",
        "knockouts": [], "uncapped_score": 10,
        "inventory_sha256": "oldsha", "basis": "inventory",
        "judged_at": "2026-07-09T08:00:00",
    })
    conn.execute(
        "INSERT INTO jobs (id, title, company, url, description, score, "
        "status, profile, llm_score, filter_json, filter_source) "
        "VALUES (1, 'PM', 'Co', 'https://x/1', ?, 50, 'new', 'testuser', 70, ?, "
        "'master')", (_GOOD_DESC, stale_blob))
    conn.commit()
    conn.close()
    s = _scorer(db, monkeypatch)
    fd = _hold_lock(db)
    try:
        with mock.patch.object(scorer_mod, "run_claude",
                               side_effect=_fake_run) as m:
            with caplog.at_level(logging.WARNING):
                count = s.rejudge_filter(profile="testuser", workers=1)
    finally:
        _release_lock(fd)
    assert count == 0
    assert m.call_count == 0
    assert any("another scoring pass" in r.message.lower()
               for r in caplog.records)


def test_lock_released_after_pass(tmp_path, monkeypatch):
    db = tmp_path / "jobs.db"
    _seed_unscored_jobs(db, 1)
    s = _scorer(db, monkeypatch)
    with mock.patch.object(scorer_mod, "run_claude", side_effect=_fake_run):
        count = s.apply_llm_scores_to_db(profile="testuser", workers=1)
    assert count == 1
    # The lock must be free again — a later pass (or process) can take it.
    fd = _hold_lock(db)
    _release_lock(fd)


# ---------------------------------------------------------------------------
# Incremental flushes
# ---------------------------------------------------------------------------

def test_apply_flushes_incrementally(tmp_path, monkeypatch):
    n = scorer_mod._FLUSH_EVERY + 5
    db = tmp_path / "jobs.db"
    _seed_unscored_jobs(db, n)
    s = _scorer(db, monkeypatch)
    with mock.patch.object(s, "_write_results",
                           wraps=s._write_results) as spy:
        with mock.patch.object(scorer_mod, "run_claude", side_effect=_fake_run):
            count = s.apply_llm_scores_to_db(profile="testuser", workers=1)
    assert count == n
    assert _scored_count(db) == n
    batch_sizes = [len(c.args[0]) for c in spy.call_args_list]
    assert len(batch_sizes) >= 2, "results must be flushed mid-pass, not once at the end"
    assert all(b <= scorer_mod._FLUSH_EVERY for b in batch_sizes)
    assert sum(batch_sizes) == n


def test_mid_pass_crash_preserves_flushed_results(tmp_path, monkeypatch):
    """A crash after the first flush must not forfeit already-scored rows."""
    flush = scorer_mod._FLUSH_EVERY
    n = flush + 5
    db = tmp_path / "jobs.db"
    _seed_unscored_jobs(db, n)
    s = _scorer(db, monkeypatch)

    calls = {"n": 0}

    def _dies_after_flush(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] > flush:
            # BaseException, like a real OOM kill / Ctrl-C: escapes the
            # per-future `except Exception` and aborts the whole pass.
            raise KeyboardInterrupt
        return _fake_run()

    with mock.patch.object(scorer_mod, "run_claude",
                           side_effect=_dies_after_flush):
        with pytest.raises(KeyboardInterrupt):
            s.apply_llm_scores_to_db(profile="testuser", workers=1)
    assert _scored_count(db) == flush
