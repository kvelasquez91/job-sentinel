"""prune_old_expired: old expired rows die; human-touched rows survive."""
import main as main_mod


def _db(tmp_path):
    conn = main_mod.init_database(str(tmp_path / "jobs.db"))
    rows = [
        # (url, status, created_at, feedback, notes)
        ("old-plain",   "expired", "2025-11-01 00:00:00", None,   None),   # deleted
        ("old-feedback","expired", "2025-11-01 00:00:00", "more", None),   # kept
        ("old-notes",   "expired", "2025-11-01 00:00:00", None,   "hi"),   # kept
        ("fresh",       "expired", "2026-07-01 00:00:00", None,   None),   # kept (recent)
        ("live",        "new",     "2025-11-01 00:00:00", None,   None),   # kept (not expired)
    ]
    conn.executemany(
        "INSERT INTO jobs (title, company, url, status, created_at, feedback, notes, profile) "
        "VALUES ('PM', 'Co', ?, ?, ?, ?, ?, 'testuser')", rows)
    conn.commit()
    return conn


def test_prune_deletes_only_old_untouched_expired(tmp_path):
    conn = _db(tmp_path)
    deleted = main_mod.prune_old_expired(conn, "testuser", retention_days=180)
    assert deleted == 1
    urls = {r[0] for r in conn.execute("SELECT url FROM jobs").fetchall()}
    assert urls == {"old-feedback", "old-notes", "fresh", "live"}
