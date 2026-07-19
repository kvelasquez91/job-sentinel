"""dashboard settings API: auto-tailor toggle GET/PUT."""
import asyncio
import importlib
import sqlite3

import main as main_mod
import settings_store

app_mod = importlib.import_module("dashboard.app")


def _db(tmp_path):
    db = tmp_path / "jobs.db"
    main_mod.init_database(str(db)).close()
    return db


def test_get_falls_back_to_config_default(tmp_path, monkeypatch):
    monkeypatch.setattr(app_mod, "DB_PATH", str(_db(tmp_path)))
    monkeypatch.setattr(app_mod, "_auto_tailor_config_default",
                        lambda: {"enabled": True})
    assert asyncio.run(app_mod.get_auto_tailor_setting()) == {"enabled": True}
    monkeypatch.setattr(app_mod, "_auto_tailor_config_default", lambda: {})
    assert asyncio.run(app_mod.get_auto_tailor_setting()) == {"enabled": False}


def test_put_roundtrip_beats_config(tmp_path, monkeypatch):
    monkeypatch.setattr(app_mod, "DB_PATH", str(_db(tmp_path)))
    monkeypatch.setattr(app_mod, "_auto_tailor_config_default",
                        lambda: {"enabled": True})
    out = asyncio.run(app_mod.put_auto_tailor_setting(
        app_mod.AutoTailorSetting(enabled=False)))
    assert out == {"enabled": False}
    assert asyncio.run(app_mod.get_auto_tailor_setting()) == {"enabled": False}
    out = asyncio.run(app_mod.put_auto_tailor_setting(
        app_mod.AutoTailorSetting(enabled=True)))
    assert asyncio.run(app_mod.get_auto_tailor_setting()) == {"enabled": True}


def test_put_writes_the_value_the_daily_run_reads(tmp_path, monkeypatch):
    """Cross-process contract: main.py's pass reads the same key PUT writes."""
    db = _db(tmp_path)
    monkeypatch.setattr(app_mod, "DB_PATH", str(db))
    asyncio.run(app_mod.put_auto_tailor_setting(
        app_mod.AutoTailorSetting(enabled=False)))
    conn = sqlite3.connect(str(db))
    assert settings_store.auto_tailor_enabled(conn, {"enabled": True}) is False
    conn.close()
