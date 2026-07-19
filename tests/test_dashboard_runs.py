"""/api/runs must honor ?limit= (default 10) so the dashboard can label
session dividers for every run on screen."""
import asyncio
import importlib

import main as main_mod

app_mod = importlib.import_module("dashboard.app")


def _seed_runs(tmp_path, n):
    db = tmp_path / "jobs.db"
    conn = main_mod.init_database(str(db))
    # Seconds increment so ISO strings sort in insert order (n <= 60).
    conn.executemany(
        "INSERT INTO runs (started_at, status, profile) VALUES (?, 'completed', 'testuser')",
        [(f"2026-06-01T00:00:{i:02d}",) for i in range(n)],
    )
    conn.commit()
    conn.close()
    return db


def test_runs_default_limit_is_10(tmp_path, monkeypatch):
    monkeypatch.setattr(app_mod, "DB_PATH", str(_seed_runs(tmp_path, 15)))
    runs = asyncio.run(app_mod.get_runs())
    assert len(runs) == 10


def test_runs_limit_param_returns_more(tmp_path, monkeypatch):
    monkeypatch.setattr(app_mod, "DB_PATH", str(_seed_runs(tmp_path, 15)))
    runs = asyncio.run(app_mod.get_runs(limit=50))
    assert len(runs) == 15


def test_runs_rows_have_label_fields_newest_first(tmp_path, monkeypatch):
    monkeypatch.setattr(app_mod, "DB_PATH", str(_seed_runs(tmp_path, 3)))
    runs = asyncio.run(app_mod.get_runs(limit=50))
    assert runs[0]["id"] is not None
    assert runs[0]["started_at"] == "2026-06-01T00:00:02"   # newest first
    assert runs[-1]["started_at"] == "2026-06-01T00:00:00"


def test_runs_limit_clamped_to_at_least_1(tmp_path, monkeypatch):
    monkeypatch.setattr(app_mod, "DB_PATH", str(_seed_runs(tmp_path, 5)))
    runs = asyncio.run(app_mod.get_runs(limit=0))
    assert len(runs) == 1
