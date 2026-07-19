"""
Tiny key-value settings store in jobs.db, shared by the dashboard process
and main.py's daily run. Values are TEXT; booleans use '1'/'0'.

The dashboard writes here (PUT /api/settings/auto-tailor); the daily run
reads here before its post-run auto-tailor pass. config.yaml supplies the
default only until the first write — no write-on-read, so the settings row
exists only once the toggle has actually been flipped.
"""
import sqlite3
from datetime import datetime
from typing import Optional

AUTO_TAILOR_KEY = "auto_tailor_enabled"


def get_setting(conn: sqlite3.Connection, key: str,
                default: Optional[str] = None) -> Optional[str]:
    """Return the stored value for key, or default when the key is absent."""
    row = conn.execute(
        "SELECT value FROM settings WHERE key = ?", (key,)
    ).fetchone()
    return row[0] if row is not None else default


def set_setting(conn: sqlite3.Connection, key: str, value: str) -> None:
    """UPSERT a setting and commit."""
    conn.execute(
        """INSERT INTO settings (key, value, updated_at) VALUES (?, ?, ?)
           ON CONFLICT(key) DO UPDATE SET value = excluded.value,
                                          updated_at = excluded.updated_at""",
        (key, value, datetime.now().isoformat()),
    )
    conn.commit()


def auto_tailor_enabled(conn: sqlite3.Connection, at_cfg: dict) -> bool:
    """Is the auto-tailor pass on? The dashboard toggle (DB) wins; config.yaml
    auto_tailor.enabled is only the default until the toggle is first flipped."""
    val = get_setting(conn, AUTO_TAILOR_KEY)
    if val is None:
        return bool(at_cfg.get("enabled", False))
    return val == "1"
