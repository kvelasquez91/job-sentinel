"""Session-wide fixtures.

Tests that run main.main() call setup_logging(), which attaches rotating
file handlers to the root logger — and those handlers persist for the rest
of the pytest session, leaking every later test's log records into the
production logs/ directory. Redirect the whole session to a temp dir.
"""
import logging
import os

import pytest


@pytest.fixture(scope="session", autouse=True)
def isolate_log_dir(tmp_path_factory):
    """Point setup_logging's file handlers at a temp dir, never real logs/."""
    log_dir = tmp_path_factory.mktemp("job_sentinel_logs")
    previous = os.environ.get("JOB_SENTINEL_LOG_DIR")
    os.environ["JOB_SENTINEL_LOG_DIR"] = str(log_dir)
    yield
    if previous is None:
        os.environ.pop("JOB_SENTINEL_LOG_DIR", None)
    else:
        os.environ["JOB_SENTINEL_LOG_DIR"] = previous
    root = logging.getLogger()
    for handler in list(root.handlers):
        if isinstance(handler, logging.FileHandler):
            root.removeHandler(handler)
            handler.close()


@pytest.fixture(autouse=True)
def _resume_summary_fixture(monkeypatch):
    """When no personalized resume_summary is configured (fresh/shared checkout),
    tests run against a fixture summary instead of erroring."""
    import engine.llm_scorer as _scorer_mod
    if _scorer_mod.RESUME_SUMMARY is None:
        monkeypatch.setattr(_scorer_mod, "RESUME_SUMMARY", "TEST RESUME SUMMARY (fixture)")
