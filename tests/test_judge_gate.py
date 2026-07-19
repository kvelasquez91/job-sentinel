"""Stage-2 judge gate: jobs whose blended score can't surface skip the judge.

Alerts fire at 60 and auto-tailor needs score AND filter >= 60 (inclusive),
so a job blending below judge_min_score (default 40) never surfaces — its
judge call is pure waste.
Gated rows write the existing '{}' sentinel (renders "—") and the backfill
target SQL applies the same floor so they are not re-billed later.
"""
import json
import sqlite3
from unittest import mock

import pytest

import engine.llm_scorer as scorer_mod
from engine.llm_scorer import LLMScorer

# Stage-1 envelope: dims sum to 91 and survive _apply_caps unchanged for a
# 'Director of AI Product' @ 'Remote (US)' row with AI keywords in the
# description and salary_max 260000.
_SCORING_TEXT = (
    '{"role_match":28,"seniority_match":16,"remote_location":20,'
    '"ai_domain_fit":18,"comp_match":9,"explanation":"Strong fit.",'
    '"salary_min":230000,"salary_max":260000,'
    '"est_total_comp_min":280000,"est_total_comp_max":360000,'
    '"filter":{"must_have_keywords":[{"term":"Generative AI","aliases":["GenAI"]}],'
    '"title_variants":[],"title_alignment":"close",'
    '"knockouts":[{"requirement":"8+ years product management","verdict":"met"}]}}'
)
_JUDGE_TEXT = json.dumps({
    "must_haves": [{"term": "Generative AI", "verdict": "explicit",
                    "evidence": "Directed Example Corp GenAI integration"}],
    "knockouts": [{"requirement": "8+ years product management",
                   "verdict": "met", "reason": "8.7 years",
                   "required_years": 8, "candidate_years": 8.7}],
    "title_claim": "close",
})


def _fake_two_stage(prompt, **kwargs):
    text = (_JUDGE_TEXT if "screening judge" in (kwargs.get("system_prompt") or "")
            else _SCORING_TEXT)
    return {"text": text, "usage": {}, "cost_usd": 0.0,
            "is_error": False, "session_id": None}


def _scorer(db_path=":memory:"):
    s = LLMScorer(db_path=db_path, model="claude-sonnet-5")
    s.judge_basis_text, s.judge_basis_sha, s.judge_basis = (
        "INVENTORY TEXT", "sha1" * 16, "inventory")
    return s


def _row(kw_score):
    return {"id": 7, "title": "Director of AI Product", "company": "Acme",
            "location": "Remote (US)", "salary_min": None, "salary_max": None,
            "description": "We build AI/ML LLM products.", "score": kw_score}


@pytest.fixture(autouse=True)
def _pin_gate(monkeypatch):
    """Pin the gate to 50 so a config.yaml override can't skew these tests."""
    monkeypatch.setattr(scorer_mod, "_JUDGE_MIN_SCORE", 50)


# A weak-fit stage-1 envelope: dims sum to 40 (well under every _apply_caps
# ceiling for this row), so the blend stays below the gate no matter the
# weights. Carries a real filter block so the skipped judge is provably the
# gate's doing, not missing filter data.
_LOW_FIT_TEXT = (
    '{"role_match":15,"seniority_match":10,"remote_location":8,'
    '"ai_domain_fit":5,"comp_match":2,"explanation":"Weak fit.",'
    '"salary_min":null,"salary_max":null,'
    '"filter":{"must_have_keywords":[{"term":"Generative AI","aliases":["GenAI"]}],'
    '"title_variants":[],"title_alignment":"none","knockouts":[]}}'
)


def _fake_low_fit(prompt, **kwargs):
    return {"text": _LOW_FIT_TEXT, "usage": {}, "cost_usd": 0.0,
            "is_error": False, "session_id": None}


def test_low_blended_score_skips_judge_and_writes_sentinel():
    # kw=10, llm=40 -> blended round(0.4*10 + 0.6*40) = 28 < 50 -> gated.
    with mock.patch.object(scorer_mod, "run_claude",
                           side_effect=_fake_low_fit) as m:
        result = _scorer()._score_one_job(_row(kw_score=10))
    assert m.call_count == 1                       # scoring call only, no judge
    assert result[1] == 28                         # blended final score
    assert result[-1] == (None, None, "none", None, "{}")   # sentinel


def test_high_blended_score_still_judges():
    # kw=50, llm=91 -> blended 66 >= 50 -> judge runs as before.
    with mock.patch.object(scorer_mod, "run_claude",
                           side_effect=_fake_two_stage) as m:
        result = _scorer()._score_one_job(_row(kw_score=50))
    assert m.call_count == 2                       # scoring + judge
    filter_fields = result[-1]
    assert filter_fields[2] == "master"            # real judged fields written


def test_gate_zero_disables_gating(monkeypatch):
    monkeypatch.setattr(scorer_mod, "_JUDGE_MIN_SCORE", 0)
    with mock.patch.object(scorer_mod, "run_claude",
                           side_effect=_fake_two_stage) as m:
        _scorer()._score_one_job(_row(kw_score=10))
    assert m.call_count == 2                       # judge runs even at blended 42


def test_backfill_filter_skips_rows_below_gate(tmp_path, monkeypatch):
    import main as main_mod
    db = tmp_path / "jobs.db"
    conn = main_mod.init_database(str(db))
    good_desc = "We build AI/ML LLM products for enterprise GenAI roadmaps. " * 10
    rows = [
        # (id, score, filter_json) — both live, both real descriptions.
        (1, 30, "{}"),    # below gate -> excluded from backfill target set
        (2, 80, None),    # above gate -> targeted (scoring + judge = 2 calls)
    ]
    for jid, score, fjson in rows:
        conn.execute(
            "INSERT INTO jobs (id, title, company, location, url, description, "
            "score, llm_score, filter_json, status, profile) "
            "VALUES (?, 'Director of AI Product', 'Acme', 'Remote (US)', ?, ?, "
            "?, 80, ?, 'new', 'testuser')",
            (jid, f"https://x/{jid}", good_desc, score, fjson))
    conn.commit()
    conn.close()

    s = _scorer(db_path=str(db))
    monkeypatch.setattr(scorer_mod, "is_cli_available", lambda: True)
    with mock.patch.object(scorer_mod, "run_claude",
                           side_effect=_fake_two_stage) as m:
        count = s.backfill_filter(profile="testuser", workers=1)

    assert m.call_count == 2      # only job 2 was attempted (scoring + judge)
    assert count == 1
    conn = sqlite3.connect(str(db))
    got = dict(conn.execute(
        "SELECT id, filter_score FROM jobs WHERE filter_score IS NOT NULL"))
    conn.close()
    assert set(got) == {2}
