"""Tests for LLM-free score re-blending (LLMScorer.reblend_all).

reblend_all re-runs the keyword scorer against each stored posting (so
scorer-rule changes like the location cap take effect) and re-blends it with
the row's EXISTING llm_score using the current weights — making ZERO LLM
calls. Use after changing keyword rules or blend weights without spending
subscription budget.
"""
import os
import tempfile

import engine.llm_scorer as scorer_mod
from engine.llm_scorer import LLMScorer, _KW_WEIGHT, _LLM_WEIGHT
from engine.scorer import JobScorer
from scrapers.base import JobPosting
from policy_fixtures import patch_scorer_keywords
import main

_CFG = {"scoring": {"alert_threshold": 60}}


def _tmpdb():
    path = os.path.join(tempfile.mkdtemp(), "t.db")
    return path, main.init_database(path)


def _insert(conn, url, title, location, description, score, llm_score=None,
            profile="testuser"):
    conn.execute(
        "INSERT INTO jobs (title,company,url,location,description,score,"
        "llm_score,profile,status) VALUES (?,?,?,?,?,?,?,?,?)",
        (title, "Acme", url, location, description, score, llm_score, profile,
         "new"))
    conn.commit()


def _expected_kw(title, location, description):
    """Keyword score computed independently, so tests pin the blend + cap
    propagation, not the keyword-scoring internals (covered in test_scorer)."""
    job = JobPosting(title=title, company="Acme", location=location, url="x",
                     description=description)
    return JobScorer(_CFG).score(job)


def test_reblend_llm_scored_row_uses_new_weights():
    path, conn = _tmpdb()
    desc = "We build AI/ML/LLM products for the enterprise."
    _insert(conn, "u1", "Senior Product Manager AI", "Remote", desc,
            score=999, llm_score=73)  # stale stored score
    LLMScorer(db_path=path).reblend_all(profile="testuser", config=_CFG)
    kw = _expected_kw("Senior Product Manager AI", "Remote", desc)
    expected = min(100, round(_KW_WEIGHT * kw + _LLM_WEIGHT * 73))
    row = conn.execute("SELECT score FROM jobs WHERE url='u1'").fetchone()
    assert row[0] == expected


def test_reblend_keyword_only_row_uses_pure_keyword_score():
    path, conn = _tmpdb()
    desc = "We build AI/ML/LLM products."
    _insert(conn, "u1", "Senior Product Manager AI", "Remote", desc,
            score=0, llm_score=None)
    LLMScorer(db_path=path).reblend_all(profile="testuser", config=_CFG)
    kw = _expected_kw("Senior Product Manager AI", "Remote", desc)
    row = conn.execute("SELECT score FROM jobs WHERE url='u1'").fetchone()
    assert row[0] == kw


def test_reblend_applies_location_cap_to_named_city(monkeypatch):
    """A strong role in a named non-remote city: the keyword component is
    capped at 40 (not zeroed), so the re-blend surfaces it instead of burying
    it — this is the big-company-SVP-in-a-named-city case from the audit."""
    patch_scorer_keywords(monkeypatch)
    path, conn = _tmpdb()
    _insert(conn, "u1", "Senior Product Manager", "New York, NY",
            "AI and ML product work", score=35, llm_score=88)
    LLMScorer(db_path=path).reblend_all(profile="testuser", config=_CFG)
    kw = _expected_kw("Senior Product Manager", "New York, NY",
                      "AI and ML product work")
    assert kw == 40  # cap propagates through the keyword scorer
    expected = min(100, round(_KW_WEIGHT * 40 + _LLM_WEIGHT * 88))
    row = conn.execute("SELECT score FROM jobs WHERE url='u1'").fetchone()
    assert row[0] == expected and row[0] >= 60  # now clears the alert line


def test_reblend_makes_no_llm_calls(monkeypatch):
    """Hard guarantee: reblend must never reach the CLI."""
    def _boom(*_a, **_k):
        raise AssertionError("reblend_all must not call the LLM")
    monkeypatch.setattr(scorer_mod, "run_claude", _boom)
    monkeypatch.setattr(scorer_mod, "is_cli_available", _boom)
    path, conn = _tmpdb()
    _insert(conn, "u1", "Senior Product Manager AI", "Remote",
            "We build AI/ML/LLM products.", score=10, llm_score=73)
    updated = LLMScorer(db_path=path).reblend_all(profile="testuser", config=_CFG)
    assert updated == 1


def test_reblend_only_touches_named_profile():
    path, conn = _tmpdb()
    _insert(conn, "u1", "Senior Product Manager AI", "Remote", "AI/ML/LLM",
            score=999, llm_score=73, profile="other")
    updated = LLMScorer(db_path=path).reblend_all(profile="testuser", config=_CFG)
    assert updated == 0
    row = conn.execute("SELECT score FROM jobs WHERE url='u1'").fetchone()
    assert row[0] == 999  # other-profile row untouched


def test_reblend_applies_layoff_penalty_to_blend():
    """Penalty hits the keyword component BEFORE blending, mirroring how
    save-time stores a penalized keyword score that the LLM pass blends."""
    path, conn = _tmpdb()
    conn.execute(
        "INSERT INTO company_insights (company_name, company_name_normalized, "
        "has_recent_layoffs, fetched_at) VALUES ('Acme', 'acme', 1, '2026-07-01')")
    conn.commit()
    desc = "We build AI/ML/LLM products for the enterprise."
    _insert(conn, "u1", "Senior Product Manager AI", "Remote", desc,
            score=999, llm_score=73)
    cfg = {"scoring": {"alert_threshold": 60, "layoff_penalty": 5}}
    LLMScorer(db_path=path).reblend_all(profile="testuser", config=cfg)
    kw = _expected_kw("Senior Product Manager AI", "Remote", desc) - 5
    expected = min(100, round(_KW_WEIGHT * kw + _LLM_WEIGHT * 73))
    row = conn.execute("SELECT score FROM jobs WHERE url='u1'").fetchone()
    assert row[0] == expected


def test_reblend_dry_run_reports_without_writing():
    path, conn = _tmpdb()
    _insert(conn, "u1", "Senior Product Manager AI", "Remote",
            "We build AI/ML/LLM products.", score=999, llm_score=73)
    would = LLMScorer(db_path=path).reblend_all(
        profile="testuser", config=_CFG, dry_run=True)
    assert would == 1
    row = conn.execute("SELECT score FROM jobs WHERE url='u1'").fetchone()
    assert row[0] == 999  # nothing written
    # Real pass applies it; a second dry-run then reports clean.
    assert LLMScorer(db_path=path).reblend_all(profile="testuser", config=_CFG) == 1
    assert LLMScorer(db_path=path).reblend_all(
        profile="testuser", config=_CFG, dry_run=True) == 0
