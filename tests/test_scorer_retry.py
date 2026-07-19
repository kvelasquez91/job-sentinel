"""Retry policy: transient CLI errors retry (3 attempts); timeouts never retry.

ClaudeCLITimeout exists so callers don't re-run a deterministic-latency prompt
at the same ceiling (see claude_cli.py's docstring). resume_tailor already
excludes timeouts from retry; these tests pin the scorer to the same policy.
"""
from unittest import mock

import pytest
import tenacity.nap

import engine.llm_scorer as scorer_mod
from engine.llm_scorer import LLMScorer


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    """Make tenacity's exponential backoff instant so retry tests run fast.

    tenacity.nap.sleep calls time.sleep dynamically, so patching the sleep
    attribute on the time module tenacity.nap imported neuters the waits.
    """
    monkeypatch.setattr(tenacity.nap.time, "sleep", lambda _s: None)


def _scorer():
    s = LLMScorer(db_path=":memory:", model="claude-sonnet-5")
    s.judge_basis_text, s.judge_basis_sha, s.judge_basis = (
        "INVENTORY TEXT", "sha1" * 16, "inventory")
    return s


_FILTER_RAW = {
    "must_have_keywords": [{"term": "RAG", "aliases": []}],
    "knockouts": [],
    "title_variants": [],
    "title_alignment": "none",
}


def test_scoring_timeout_is_not_retried():
    with mock.patch.object(
        scorer_mod, "run_claude",
        side_effect=scorer_mod.ClaudeCLITimeout("claude CLI killed: timed out after 60s"),
    ) as m:
        result = _scorer()._call_claude("Director of AI Product", "Acme", "desc")
    assert m.call_count == 1          # a timeout must never be retried
    assert result == (None,) * 8      # graceful failure — job stays unscored


def test_scoring_transient_error_retries_three_times():
    with mock.patch.object(
        scorer_mod, "run_claude",
        side_effect=scorer_mod.ClaudeCLIError("transient boom"),
    ) as m:
        result = _scorer()._call_claude("Director of AI Product", "Acme", "desc")
    assert m.call_count == 3          # initial attempt + 2 retries
    assert result == (None,) * 8


def test_judge_timeout_is_not_retried():
    with mock.patch.object(
        scorer_mod, "run_claude",
        side_effect=scorer_mod.ClaudeCLITimeout("claude CLI killed: timed out after 60s"),
    ) as m:
        judged = _scorer()._judge_filter("Director of AI Product", _FILTER_RAW)
    assert m.call_count == 1
    assert judged is None             # caller then leaves filter fields NULL


def test_judge_transient_error_retries_three_times():
    with mock.patch.object(
        scorer_mod, "run_claude",
        side_effect=scorer_mod.ClaudeCLIError("transient boom"),
    ) as m:
        judged = _scorer()._judge_filter("Director of AI Product", _FILTER_RAW)
    assert m.call_count == 3
    assert judged is None
