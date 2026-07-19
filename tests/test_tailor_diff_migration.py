"""Tailor-diff schema migration: five columns, JSON validity CHECKs."""
import sqlite3

import pytest

import main as main_mod

NEW_HISTORY_COLS = {"final_text", "final_text_source",
                    "warnings_json", "edit_verdicts_json"}


def _cols(conn, table):
    return {r["name"] for r in conn.execute(f"PRAGMA table_info({table})")}


def test_migration_adds_columns_and_is_idempotent(tmp_path):
    db = str(tmp_path / "jobs.db")
    conn = main_mod.init_database(db)
    conn.close()
    conn = main_mod.init_database(db)  # second run must be a no-op
    assert NEW_HISTORY_COLS <= _cols(conn, "tailor_history")
    assert "tailor_issue_json" in _cols(conn, "jobs")
    conn.close()


def test_json_check_rejects_malformed_and_accepts_null_and_valid(tmp_path):
    conn = main_mod.init_database(str(tmp_path / "jobs.db"))
    with conn:  # NULL and valid JSON pass
        conn.execute("INSERT INTO tailor_history (job_id, warnings_json, "
                     "edit_verdicts_json) VALUES (1, NULL, '[]')")
        conn.execute("INSERT INTO tailor_history (job_id, warnings_json) "
                     "VALUES (1, '{\"layout\": []}')")
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("INSERT INTO tailor_history (job_id, warnings_json) "
                     "VALUES (1, 'not json{')")
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("INSERT INTO tailor_history (job_id, edit_verdicts_json) "
                     "VALUES (1, 'nope')")
    conn.close()


def test_jobs_tailor_issue_json_check(tmp_path):
    conn = main_mod.init_database(str(tmp_path / "jobs.db"))
    with conn:
        conn.execute("INSERT INTO jobs (title, company, url, profile) "
                     "VALUES ('t', 'c', 'https://x/1', 'testuser')")
        conn.execute("UPDATE jobs SET tailor_issue_json = '{\"reverted\": 1}'")
        conn.execute("UPDATE jobs SET tailor_issue_json = NULL")
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("UPDATE jobs SET tailor_issue_json = '{broken'")
    conn.close()
