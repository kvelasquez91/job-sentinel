"""settings_store: get/set round-trip + auto-tailor toggle semantics."""
import main as main_mod
import settings_store


def _conn(tmp_path):
    return main_mod.init_database(str(tmp_path / "jobs.db"))


def test_get_missing_key_returns_default(tmp_path):
    conn = _conn(tmp_path)
    assert settings_store.get_setting(conn, "nope") is None
    assert settings_store.get_setting(conn, "nope", default="x") == "x"


def test_set_then_get_and_overwrite(tmp_path):
    conn = _conn(tmp_path)
    settings_store.set_setting(conn, "k", "1")
    assert settings_store.get_setting(conn, "k") == "1"
    settings_store.set_setting(conn, "k", "0")  # UPSERT, not UNIQUE violation
    assert settings_store.get_setting(conn, "k") == "0"


def test_auto_tailor_enabled_prefers_db_over_config(tmp_path):
    conn = _conn(tmp_path)
    # No DB key: config decides; absent 'enabled' defaults False.
    assert settings_store.auto_tailor_enabled(conn, {"enabled": True}) is True
    assert settings_store.auto_tailor_enabled(conn, {}) is False
    # DB key set: it wins over config in BOTH directions.
    settings_store.set_setting(conn, settings_store.AUTO_TAILOR_KEY, "0")
    assert settings_store.auto_tailor_enabled(conn, {"enabled": True}) is False
    settings_store.set_setting(conn, settings_store.AUTO_TAILOR_KEY, "1")
    assert settings_store.auto_tailor_enabled(conn, {"enabled": False}) is True


def test_settings_survive_reinit(tmp_path):
    db = tmp_path / "jobs.db"
    conn = main_mod.init_database(str(db))
    settings_store.set_setting(conn, "k", "1")
    conn.close()
    conn = main_mod.init_database(str(db))  # IF NOT EXISTS must not clobber
    assert settings_store.get_setting(conn, "k") == "1"
    conn.close()
