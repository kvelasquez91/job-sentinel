"""_tailor_worker error categorization for the manual (dashboard) tailor path.

The engine now RAISES ClaudeCLIError on CLI failure (2026-07-15 final-review
fix) instead of swallowing it into empty analysis objects — these tests pin
the worker arm that categorizes it, so the manual path keeps surfacing a
failed task instead of a mislabeled success.
"""
import importlib

from claude_cli import ClaudeCLIError, ClaudeCLITimeout

app_mod = importlib.import_module("dashboard.app")


def _seed_task(task_id: str) -> None:
    with app_mod._tailor_lock:
        app_mod._tailor_tasks[task_id] = {
            "task_id": task_id,
            "job_id": 1,
            "status": "pending",
            "progress": "Starting...",
            "steps": [],
            "result": None,
            "error": None,
            "error_category": None,
            "created_at": "2026-07-15T00:00:00",
            "completed_at": None,
        }


def _run_worker_with(monkeypatch, exc, task_id):
    def dead_pipeline(job, db_path, progress=None):
        raise exc
    monkeypatch.setattr(app_mod, "run_tailor_pipeline", dead_pipeline)
    _seed_task(task_id)
    app_mod._tailor_worker(task_id, 1, "https://x/1", "Acme", "PM")
    with app_mod._tailor_lock:
        return app_mod._tailor_tasks.pop(task_id)


def test_claude_cli_error_fails_task_as_auth_error(tmp_path, monkeypatch):
    task = _run_worker_with(
        monkeypatch, ClaudeCLIError("claude CLI produced no output (exit=1)"),
        "t-cli-err")
    assert task["status"] == "failed"
    assert task["error_category"] == "auth_error"
    assert "claude" in task["error"].lower()


def test_claude_cli_timeout_subclass_hits_same_arm(tmp_path, monkeypatch):
    """ClaudeCLITimeout (the fail-fast subclass the engine lets through
    un-retried) must land in the same ClaudeCLIError arm."""
    task = _run_worker_with(
        monkeypatch, ClaudeCLITimeout("claude CLI killed: timed out after 300s"),
        "t-cli-timeout")
    assert task["status"] == "failed"
    assert task["error_category"] == "auth_error"
