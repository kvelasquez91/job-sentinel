"""Tests for CLI flag resolution in main.py.

Covers apply_scrape_only, mode-flag mutual exclusion (validate_mode_flags),
orphaned-flag detection, the --rescore-force confirmation gate, and the
rescore_all dry-run path. None of these tests may ever invoke the claude CLI —
LLM entry points are monkeypatched to raise if reached.
"""
import os
import sys
import tempfile
from argparse import Namespace

import pytest

import main
from main import apply_scrape_only, confirm_rescore_force, validate_mode_flags


def test_scrape_only_forces_both_skip_flags():
    args = Namespace(scrape_only=True, skip_llm=False, skip_company_intel=False)
    apply_scrape_only(args)
    assert args.skip_llm is True
    assert args.skip_company_intel is True


def test_flags_untouched_without_scrape_only():
    args = Namespace(scrape_only=False, skip_llm=False, skip_company_intel=False)
    apply_scrape_only(args)
    assert args.skip_llm is False
    assert args.skip_company_intel is False


def test_explicit_skip_flags_survive_scrape_only():
    args = Namespace(scrape_only=True, skip_llm=True, skip_company_intel=True)
    apply_scrape_only(args)
    assert args.skip_llm is True
    assert args.skip_company_intel is True


# ---------------------------------------------------------------------------
# validate_mode_flags — mutual exclusion + orphaned flags
# ---------------------------------------------------------------------------

def _args(**overrides):
    """Namespace mirroring main.py's parser defaults."""
    base = dict(
        dry_run=False, scrape_only=False, dashboard=False, profile="testuser",
        config=None, db="data/jobs.db", skip_company_intel=False,
        enrich_companies=False, skip_llm=False, skip_auto_tailor=False,
        rescore_all=False, rescore_force=False, rescore_sample=False,
        reblend=False, rescore_run=None, backfill_filter=False,
        rejudge_filter=False, rejudge_filter_all=False,
        filter_since_hours=None, log_level="INFO", verbose=False,
        dismiss_job=None,
    )
    base.update(overrides)
    return Namespace(**base)


def test_no_mode_flags_passes():
    validate_mode_flags(_args())


def test_single_mode_flag_passes():
    validate_mode_flags(_args(rescore_force=True))


def test_two_mode_flags_error_names_both(capsys):
    with pytest.raises(SystemExit) as exc:
        validate_mode_flags(_args(reblend=True, rescore_force=True))
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "--reblend" in err and "--rescore-force" in err


def test_rescore_run_counts_as_mode():
    with pytest.raises(SystemExit) as exc:
        validate_mode_flags(_args(rescore_run=7, dashboard=True))
    assert exc.value.code == 2


def test_dismiss_job_counts_as_mode():
    with pytest.raises(SystemExit) as exc:
        validate_mode_flags(_args(dismiss_job="42", rescore_all=True))
    assert exc.value.code == 2


def test_filter_since_hours_orphaned_errors(capsys):
    with pytest.raises(SystemExit) as exc:
        validate_mode_flags(_args(filter_since_hours=48))
    assert exc.value.code == 2
    assert "--filter-since-hours" in capsys.readouterr().err


def test_filter_since_hours_ok_with_rejudge_filter():
    validate_mode_flags(_args(rejudge_filter=True, filter_since_hours=48))


def test_filter_since_hours_ok_with_backfill_filter():
    validate_mode_flags(_args(backfill_filter=True, filter_since_hours=48))


def test_dry_run_errors_with_rescore_all(capsys):
    """--rescore-all doesn't support --dry-run — silently billing while the
    user believes it's a dry run is exactly the bug this guards against."""
    with pytest.raises(SystemExit) as exc:
        validate_mode_flags(_args(rescore_all=True, dry_run=True))
    assert exc.value.code == 2
    assert "--dry-run" in capsys.readouterr().err


def test_dry_run_ok_with_reblend():
    validate_mode_flags(_args(reblend=True, dry_run=True))


def test_dry_run_ok_with_rescore_force():
    validate_mode_flags(_args(rescore_force=True, dry_run=True))


def test_dry_run_ok_alone():
    validate_mode_flags(_args(dry_run=True))


# ---------------------------------------------------------------------------
# confirm_rescore_force — interactive gate on the destructive full re-score
# ---------------------------------------------------------------------------

class _TTYStdin:
    def isatty(self):
        return True


class _PipeStdin:
    def isatty(self):
        return False


def test_confirm_accepts_yes(monkeypatch):
    monkeypatch.setattr(sys, "stdin", _TTYStdin())
    monkeypatch.setattr("builtins.input", lambda prompt: "yes")
    assert confirm_rescore_force(120, "testuser") is True


def test_confirm_rejects_anything_else(monkeypatch):
    monkeypatch.setattr(sys, "stdin", _TTYStdin())
    monkeypatch.setattr("builtins.input", lambda prompt: "y")
    assert confirm_rescore_force(120, "testuser") is False


def test_confirm_handles_eof(monkeypatch):
    monkeypatch.setattr(sys, "stdin", _TTYStdin())

    def _raise(prompt):
        raise EOFError

    monkeypatch.setattr("builtins.input", _raise)
    assert confirm_rescore_force(120, "testuser") is False


def test_confirm_refuses_noninteractive_stdin(monkeypatch):
    """Piped/cron invocations must not be able to auto-confirm."""
    monkeypatch.setattr(sys, "stdin", _PipeStdin())
    monkeypatch.setattr(
        "builtins.input",
        lambda prompt: (_ for _ in ()).throw(AssertionError("input() must not be called")),
    )
    assert confirm_rescore_force(120, "testuser") is False


def test_confirm_prompt_shows_count(monkeypatch):
    seen = {}
    monkeypatch.setattr(sys, "stdin", _TTYStdin())

    def _capture(prompt):
        seen["prompt"] = prompt
        return "no"

    monkeypatch.setattr("builtins.input", _capture)
    confirm_rescore_force(345, "testuser")
    assert "345" in seen["prompt"]


# ---------------------------------------------------------------------------
# LLMScorer.rescore_all dry-run — reports, writes nothing, calls nothing
# ---------------------------------------------------------------------------

def _tmpdb():
    path = os.path.join(tempfile.mkdtemp(), "t.db")
    return path, main.init_database(path)


def _insert(conn, url, score=50, llm_score=None, profile="testuser"):
    conn.execute(
        "INSERT INTO jobs (title,company,url,location,description,score,"
        "llm_score,profile,status) VALUES (?,?,?,?,?,?,?,?,?)",
        ("PM", "Acme", url, "Remote", "desc", score, llm_score, profile, "new"))
    conn.commit()


def _forbid_llm(monkeypatch):
    """Any path that would reach the claude CLI fails the test immediately."""
    from engine.llm_scorer import LLMScorer

    def _boom(self, *a, **kw):
        raise AssertionError("apply_llm_scores_to_db must not run in a dry run")

    monkeypatch.setattr(LLMScorer, "apply_llm_scores_to_db", _boom)


def test_rescore_force_dry_run_counts_all_profile_rows(monkeypatch):
    from engine.llm_scorer import LLMScorer
    _forbid_llm(monkeypatch)
    path, conn = _tmpdb()
    _insert(conn, "u1", score=70, llm_score=80)
    _insert(conn, "u2", score=40, llm_score=None)
    _insert(conn, "u3", score=40, llm_score=55, profile="other")
    count = LLMScorer(db_path=path).rescore_all(
        force=True, profile="testuser", dry_run=True)
    assert count == 2


def test_rescore_force_dry_run_writes_nothing(monkeypatch):
    from engine.llm_scorer import LLMScorer
    _forbid_llm(monkeypatch)
    path, conn = _tmpdb()
    _insert(conn, "u1", score=70, llm_score=80)
    LLMScorer(db_path=path).rescore_all(force=True, profile="testuser", dry_run=True)
    row = conn.execute("SELECT score, llm_score FROM jobs WHERE url='u1'").fetchone()
    assert (row[0], row[1]) == (70, 80)


def test_rescore_nonforce_dry_run_counts_null_rows(monkeypatch):
    from engine.llm_scorer import LLMScorer
    _forbid_llm(monkeypatch)
    path, conn = _tmpdb()
    _insert(conn, "u1", llm_score=80)
    _insert(conn, "u2", llm_score=None)
    count = LLMScorer(db_path=path).rescore_all(
        force=False, profile="testuser", dry_run=True)
    assert count == 1


# ---------------------------------------------------------------------------
# main() dispatch integration
# ---------------------------------------------------------------------------

def test_main_conflicting_modes_exit_2(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["main.py", "--reblend", "--dashboard"])
    with pytest.raises(SystemExit) as exc:
        main.main()
    assert exc.value.code == 2


def test_main_orphaned_filter_since_hours_exit_2(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["main.py", "--filter-since-hours", "48"])
    with pytest.raises(SystemExit) as exc:
        main.main()
    assert exc.value.code == 2


def test_main_rescore_force_dry_run_writes_nothing(monkeypatch):
    _forbid_llm(monkeypatch)
    path, conn = _tmpdb()
    _insert(conn, "u1", score=70, llm_score=80)
    monkeypatch.setattr(
        sys, "argv", ["main.py", "--rescore-force", "--dry-run", "--db", path])
    main.main()
    row = conn.execute("SELECT score, llm_score FROM jobs WHERE url='u1'").fetchone()
    assert (row[0], row[1]) == (70, 80)


def test_main_rescore_force_aborts_when_not_confirmed(monkeypatch):
    from engine.llm_scorer import LLMScorer
    _forbid_llm(monkeypatch)
    monkeypatch.setattr(LLMScorer, "is_available", lambda self: True)
    monkeypatch.setattr(main, "confirm_rescore_force", lambda count, profile: False)
    path, conn = _tmpdb()
    _insert(conn, "u1", score=70, llm_score=80)
    monkeypatch.setattr(sys, "argv", ["main.py", "--rescore-force", "--db", path])
    with pytest.raises(SystemExit) as exc:
        main.main()
    assert exc.value.code == 1  # refused rescore must not look like success
    row = conn.execute("SELECT score, llm_score FROM jobs WHERE url='u1'").fetchone()
    assert (row[0], row[1]) == (70, 80)


# ---------------------------------------------------------------------------
# Python version guard
# ---------------------------------------------------------------------------

def test_python_version_guard(monkeypatch):
    import main as main_mod
    monkeypatch.setattr(main_mod.sys, "version_info", (3, 11, 9))
    with pytest.raises(SystemExit):
        main_mod._check_python_version()
    monkeypatch.setattr(main_mod.sys, "version_info", (3, 12, 0))
    main_mod._check_python_version()  # must not raise


# ---------------------------------------------------------------------------
# PROFILE_KEY threading — the --profile default must track config, never a
# hardcoded literal (see profile_policy.PROFILE_KEY).
# ---------------------------------------------------------------------------

def test_profile_default_comes_from_config(monkeypatch):
    """--profile default must track profile_policy.PROFILE_KEY, never a literal."""
    import profile_policy
    assert profile_policy.PROFILE_KEY  # non-empty
    import main as main_mod
    # argparse default is captured at parser build; assert the module wiring
    assert main_mod.PROFILE_KEY == profile_policy.PROFILE_KEY
