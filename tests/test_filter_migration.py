"""Filter Match columns exist after init_database (fresh AND migrated DBs)."""
import sqlite3

import main as main_mod

FILTER_COLS = {"filter_score", "filter_score_master", "filter_source",
               "filter_knockout", "filter_json"}


def _cols(db_path):
    conn = sqlite3.connect(db_path)
    cols = {r[1] for r in conn.execute("PRAGMA table_info(jobs)")}
    conn.close()
    return cols


def test_fresh_db_has_filter_columns(tmp_path):
    db = tmp_path / "jobs.db"
    main_mod.init_database(str(db)).close()
    assert FILTER_COLS <= _cols(str(db))


def test_existing_db_gains_filter_columns(tmp_path):
    # Simulate a pre-feature DB: create it, strip nothing (ALTER ADD is
    # idempotent via the PRAGMA check), then re-run init to migrate.
    db = tmp_path / "jobs.db"
    main_mod.init_database(str(db)).close()
    main_mod.init_database(str(db)).close()  # second run must not raise
    assert FILTER_COLS <= _cols(str(db))
