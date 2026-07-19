"""
Error monitoring for Job Sentinel.

Features:
  - Records every run result (success/failure) to the run_history SQLite table
  - Writes failures to errors.log (in addition to the main log)

Email alerting was removed; failures are surfaced via run_history (read by the
dashboard) and errors.log.
"""
import logging
import sqlite3
import traceback as tb_module
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Run history helpers
# ---------------------------------------------------------------------------

def record_run_result(
    db_path: str,
    status: str,          # "success" | "failed"
    error_message: Optional[str] = None,
    run_id: Optional[int] = None,
) -> None:
    """Append a run result row to the run_history table."""
    try:
        conn = sqlite3.connect(db_path)
        conn.execute(
            """INSERT INTO run_history (run_id, status, error_message, ran_at)
               VALUES (?, ?, ?, ?)""",
            (run_id, status, error_message, datetime.now().isoformat()),
        )
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error("Failed to record run history: %s", e)


# ---------------------------------------------------------------------------
# Error logging
# ---------------------------------------------------------------------------

def log_error_to_file(error: Exception, traceback_str: str, errors_log: str) -> None:
    """Append a structured error entry to errors.log."""
    try:
        with open(errors_log, "a", encoding="utf-8") as f:
            f.write("\n" + "=" * 70 + "\n")
            f.write(f"TIMESTAMP : {datetime.now().isoformat()}\n")
            f.write(f"ERROR TYPE: {type(error).__name__}\n")
            f.write(f"MESSAGE   : {error}\n")
            f.write("TRACEBACK :\n")
            f.write(traceback_str)
            f.write("\n")
    except Exception as e:
        logger.error("Could not write to errors.log: %s", e)


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

def handle_run_failure(
    error: Exception,
    db_path: str,
    errors_log: str,
    run_id: Optional[int] = None,
) -> None:
    """
    All-in-one handler for a failed run:
      1. Log traceback to errors.log
      2. Record failure in the run_history table
    """
    traceback_str = tb_module.format_exc()
    log_error_to_file(error, traceback_str, errors_log)
    record_run_result(db_path, "failed", str(error)[:1000], run_id)
