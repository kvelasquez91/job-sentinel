"""The /api/profiles endpoint must hide profiles that have only expired jobs."""
import asyncio
import importlib
import sqlite3

# NOTE: `import dashboard.app as app_mod` would bind the FastAPI *instance*, not
# the module — dashboard/__init__.py does `from .app import app`, which shadows
# the submodule attribute on the package. importlib gets the real module so
# monkeypatch.setattr(app_mod, "DB_PATH", ...) works.
app_mod = importlib.import_module("dashboard.app")


def _make_db(tmp_path):
    db = tmp_path / "jobs.db"
    conn = sqlite3.connect(db)
    conn.execute(
        "CREATE TABLE jobs (id INTEGER PRIMARY KEY, title TEXT, company TEXT, "
        "url TEXT, status TEXT, profile TEXT)"
    )
    conn.executemany(
        "INSERT INTO jobs (title, company, url, status, profile) VALUES (?, ?, ?, ?, ?)",
        [
            ("Job A", "Co", "https://x/1", "new", "testuser"),
            ("Job B", "Co", "https://x/2", "expired", "testuser"),
            ("Job C", "Co", "https://x/3", "expired", "alex"),
        ],
    )
    conn.commit()
    conn.close()
    return db


def test_profiles_endpoint_hides_profiles_with_only_expired_jobs(tmp_path, monkeypatch):
    monkeypatch.setattr(app_mod, "DB_PATH", str(_make_db(tmp_path)))
    profiles = asyncio.run(app_mod.get_profiles())
    assert profiles == ["testuser"]
