"""Tests never write to the real logs/ directory.

Any test that runs main.main() ends up calling setup_logging(), which
attaches rotating file handlers to the root logger. Those handlers persist
for the rest of the pytest session, so every later test's log records leak
into the production log. setup_logging must honor JOB_SENTINEL_LOG_DIR so
conftest.py can redirect the whole session to a temp dir.
"""
import logging
import os
from pathlib import Path

import main


def _close_root_handlers():
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()


def test_setup_logging_honors_log_dir_env(tmp_path, monkeypatch):
    redirected = tmp_path / "redirected"
    monkeypatch.setenv("JOB_SENTINEL_LOG_DIR", str(redirected))
    try:
        main.setup_logging()
        logging.getLogger("job_sentinel").error("isolation probe")
        for handler in logging.getLogger().handlers:
            handler.flush()
        assert (redirected / "job_sentinel.log").exists()
        assert "isolation probe" in (redirected / "job_sentinel.log").read_text()
        assert "isolation probe" in (redirected / "errors.log").read_text()
    finally:
        _close_root_handlers()


def test_setup_logging_defaults_to_logs_dir(tmp_path, monkeypatch):
    monkeypatch.delenv("JOB_SENTINEL_LOG_DIR", raising=False)
    monkeypatch.chdir(tmp_path)
    try:
        main.setup_logging()
        logging.getLogger("job_sentinel").info("default dir probe")
        for handler in logging.getLogger().handlers:
            handler.flush()
        assert (tmp_path / "logs" / "job_sentinel.log").exists()
        assert (tmp_path / "logs" / "errors.log").exists()
    finally:
        _close_root_handlers()


def test_session_log_dir_redirected_away_from_repo():
    log_dir = os.environ.get("JOB_SENTINEL_LOG_DIR")
    assert log_dir, "conftest.py must set JOB_SENTINEL_LOG_DIR for the session"
    repo_logs = Path(__file__).resolve().parent.parent / "logs"
    assert Path(log_dir).resolve() != repo_logs.resolve()
