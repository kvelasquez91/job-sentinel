"""Source-yield regression detection over runs.source_counts history."""
import json

import main as main_mod


def _db_with_history(tmp_path, histories):
    conn = main_mod.init_database(str(tmp_path / "jobs.db"))
    for h in histories:
        conn.execute(
            "INSERT INTO runs (started_at, status, profile, source_counts) "
            "VALUES (datetime('now'), 'completed', 'testuser', ?)", (json.dumps(h),))
    conn.commit()
    return conn


def test_flags_collapsed_source(tmp_path):
    conn = _db_with_history(tmp_path, [{"LinkedIn": 40, "Eightfold": 12}] * 7)
    warnings = main_mod.detect_source_regressions(
        conn, {"LinkedIn": 42, "Eightfold": 1}, current_run_id=999)
    assert len(warnings) == 1
    assert "Eightfold" in warnings[0]


def test_quiet_when_yields_are_normal(tmp_path):
    conn = _db_with_history(tmp_path, [{"LinkedIn": 40}] * 7)
    assert main_mod.detect_source_regressions(conn, {"LinkedIn": 35}, current_run_id=999) == []


def test_low_volume_sources_are_ignored(tmp_path):
    # trailing avg < 5 — too noisy to alert on
    conn = _db_with_history(tmp_path, [{"Lever": 2}] * 7)
    assert main_mod.detect_source_regressions(conn, {"Lever": 0}, current_run_id=999) == []
