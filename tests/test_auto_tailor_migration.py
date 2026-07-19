"""Auto-tailor migration: columns exist; grandfather backfill runs exactly once.

The backfill is locked to the column-creation branch (2026-07-15 spec): a
re-run of init_database (every dashboard restart + every daily run) must
NEVER exempt rows inserted after the column shipped.
"""
import sqlite3

import main as main_mod


def _cols(conn):
    return {r[1] for r in conn.execute("PRAGMA table_info(jobs)")}


def _insert(conn, url, score=80, filter_score=75.0):
    conn.execute(
        "INSERT INTO jobs (title, company, url, score, filter_score, status, profile) "
        "VALUES ('PM', 'Co', ?, ?, ?, 'new', 'testuser')",
        (url, score, filter_score),
    )
    conn.commit()


def test_fresh_db_has_auto_tailor_columns(tmp_path):
    conn = main_mod.init_database(str(tmp_path / "jobs.db"))
    assert {"auto_tailor_exempt", "auto_tailor_attempts"} <= _cols(conn)
    conn.close()


def test_backfill_grandfathers_once_and_only_once(tmp_path):
    db = str(tmp_path / "jobs.db")
    conn = main_mod.init_database(db)

    # Simulate a pre-feature DB: drop the new columns (SQLite >= 3.35 supports
    # DROP COLUMN; venv ships 3.50), then seed pre-feature rows.
    conn.execute("ALTER TABLE jobs DROP COLUMN auto_tailor_exempt")
    conn.execute("ALTER TABLE jobs DROP COLUMN auto_tailor_attempts")
    conn.commit()
    _insert(conn, "https://x/old-qualifier")                      # 80/75 -> exempt
    _insert(conn, "https://x/old-low-filter", score=90, filter_score=55.0)
    _insert(conn, "https://x/old-unjudged", score=90, filter_score=None)
    conn.close()

    # First re-init re-adds the columns -> backfill exempts old qualifiers only.
    conn = main_mod.init_database(db)
    rows = {r["url"]: r["auto_tailor_exempt"] for r in
            conn.execute("SELECT url, auto_tailor_exempt FROM jobs")}
    assert rows["https://x/old-qualifier"] == 1
    assert rows["https://x/old-low-filter"] is None
    assert rows["https://x/old-unjudged"] is None
    _insert(conn, "https://x/fresh-qualifier")   # arrives AFTER the column exists
    conn.close()

    # Second re-init: column already present -> backfill branch unreachable.
    conn = main_mod.init_database(db)
    rows = {r["url"]: r["auto_tailor_exempt"] for r in
            conn.execute("SELECT url, auto_tailor_exempt FROM jobs")}
    assert rows["https://x/fresh-qualifier"] is None   # NOT retroactively exempted
    assert rows["https://x/old-qualifier"] == 1
    conn.close()
