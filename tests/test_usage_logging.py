"""Per-pass LLM usage accounting: accumulate run_claude usage, log one summary.

run_claude() returns usage (incl. cache_read/cache_creation token counts) and
a notional cost_usd; the scorer used to discard both. These tests pin the
accumulator arithmetic and the one-line-per-pass summary log.
"""
import json
import logging
from unittest import mock

import engine.llm_scorer as scorer_mod
from engine.llm_scorer import LLMScorer


def _result(inp=1000, out=200, read=0, write=0, cost=0.01):
    return {"text": "{}", "usage": {
        "input_tokens": inp, "output_tokens": out,
        "cache_read_input_tokens": read, "cache_creation_input_tokens": write,
    }, "cost_usd": cost, "is_error": False, "session_id": None}


def test_track_usage_accumulates_and_resets():
    s = LLMScorer(db_path=":memory:")
    s._track_usage(_result(inp=1000, out=200, read=500, write=100, cost=0.01))
    s._track_usage(_result(inp=2000, out=300, read=0, write=0, cost=0.02))
    assert s._usage["calls"] == 2
    assert s._usage["input_tokens"] == 3000
    assert s._usage["output_tokens"] == 500
    assert s._usage["cache_read_input_tokens"] == 500
    assert s._usage["cache_creation_input_tokens"] == 100
    assert round(s._usage["cost_usd"], 4) == 0.03
    s._reset_usage()
    assert s._usage["calls"] == 0
    assert s._usage["input_tokens"] == 0


def test_track_usage_tolerates_missing_usage_fields():
    s = LLMScorer(db_path=":memory:")
    s._track_usage({"text": "{}", "usage": {}, "cost_usd": 0.0})
    s._track_usage({"text": "{}"})          # no usage / cost_usd keys at all
    s._track_usage({"text": "{}", "usage": None, "cost_usd": None})
    assert s._usage["calls"] == 3
    assert s._usage["input_tokens"] == 0
    assert s._usage["cost_usd"] == 0.0


def test_log_usage_summary_emits_totals_and_is_silent_when_empty(caplog):
    s = LLMScorer(db_path=":memory:")
    with caplog.at_level(logging.INFO):
        s._log_usage_summary("empty pass")        # zero calls -> no log line
    assert not [m for m in caplog.messages if "LLM usage" in m]

    s._track_usage(_result(inp=4000, out=500, read=1500, write=200, cost=0.05))
    s._track_usage(_result(inp=4000, out=500, read=1500, write=200, cost=0.05))
    with caplog.at_level(logging.INFO):
        s._log_usage_summary("scoring pass")
    lines = [m for m in caplog.messages if "LLM usage [scoring pass]" in m]
    assert lines, "expected one usage summary line"
    assert "2 calls" in lines[0]
    assert "in=8000" in lines[0]
    assert "cache_read=3000" in lines[0]
    assert "cache_write=400" in lines[0]
    assert "out=1000" in lines[0]
    assert "$0.10" in lines[0]


# --- integration: a scoring pass tracks both stage-1 and stage-2 calls -------

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


def _fake_two_stage_with_usage(prompt, **kwargs):
    text = (_JUDGE_TEXT if "screening judge" in (kwargs.get("system_prompt") or "")
            else _SCORING_TEXT)
    return {"text": text,
            "usage": {"input_tokens": 4000, "output_tokens": 500,
                      "cache_read_input_tokens": 1500,
                      "cache_creation_input_tokens": 200},
            "cost_usd": 0.05, "is_error": False, "session_id": None}


def test_apply_pass_logs_usage_summary(tmp_path, monkeypatch, caplog):
    import main as main_mod
    db = tmp_path / "jobs.db"
    conn = main_mod.init_database(str(db))
    conn.execute(
        "INSERT INTO jobs (id, title, company, location, url, description, score, "
        "status, profile) VALUES (7, 'Director of AI Product', 'Acme', "
        "'Remote (US)', 'https://x/7', ?, 50, 'new', 'testuser')",
        ("We build AI/ML LLM products for enterprise GenAI roadmaps. " * 10,))
    conn.commit()
    conn.close()

    s = LLMScorer(db_path=str(db), model="claude-sonnet-5")
    s.judge_basis_text, s.judge_basis_sha, s.judge_basis = (
        "INVENTORY TEXT", "sha1" * 16, "inventory")
    monkeypatch.setattr(scorer_mod, "is_cli_available", lambda: True)
    with mock.patch.object(scorer_mod, "run_claude",
                           side_effect=_fake_two_stage_with_usage):
        with caplog.at_level(logging.INFO):
            s.apply_llm_scores_to_db(profile="testuser", workers=1)

    lines = [m for m in caplog.messages if "LLM usage [scoring pass]" in m]
    assert lines, "expected a usage summary at the end of the pass"
    assert "2 calls" in lines[0]      # stage-1 scoring + stage-2 judge
    assert "in=8000" in lines[0]
    assert "cache_read=3000" in lines[0]
