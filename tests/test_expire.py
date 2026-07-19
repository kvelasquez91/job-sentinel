"""Tests for expire_stale_new_jobs date handling.

The expiry query string-compared date_posted against an ISO cutoff. A non-ISO
value (e.g. Workday's "Posted 30+ Days Ago") sorted greater than the cutoff, so
such jobs never expired. Expiry must validate date_posted and fall back to
created_at when it isn't a real date.
"""
import datetime

import main


def _conn():
    return main.init_database(":memory:")


def _insert(conn, url, date_posted, created_at, status="new", profile="testuser"):
    conn.execute(
        "INSERT INTO jobs (title, company, url, date_posted, created_at, status, profile) "
        "VALUES (?,?,?,?,?,?,?)",
        ("T", "C", url, date_posted, created_at, status, profile),
    )
    conn.commit()


def _status(conn, url):
    return conn.execute("SELECT status FROM jobs WHERE url=?", (url,)).fetchone()[0]


def test_non_iso_date_posted_falls_back_to_created_at():
    conn = _conn()
    now = datetime.datetime.now()
    old = (now - datetime.timedelta(days=30)).isoformat(" ")
    recent = now.isoformat(" ")
    # Non-ISO date_posted with OLD created_at must expire (regression: never did).
    _insert(conn, "stale", "Posted 30+ Days Ago", old)
    # Non-ISO date_posted with RECENT created_at must stay new.
    _insert(conn, "fresh", "Posted 30+ Days Ago", recent)
    main.expire_stale_new_jobs(conn, profile="testuser")
    assert _status(conn, "stale") == "expired"
    assert _status(conn, "fresh") == "new"


def test_iso_date_posted_still_governs_expiry():
    conn = _conn()
    now = datetime.datetime.now().isoformat(" ")
    _insert(conn, "old", "2020-01-01", now)
    _insert(conn, "today", datetime.date.today().isoformat(), now)
    main.expire_stale_new_jobs(conn, profile="testuser")
    assert _status(conn, "old") == "expired"
    assert _status(conn, "today") == "new"


def test_null_date_posted_uses_created_at():
    conn = _conn()
    now = datetime.datetime.now()
    _insert(conn, "oldnull", None, (now - datetime.timedelta(days=30)).isoformat(" "))
    _insert(conn, "newnull", None, now.isoformat(" "))
    main.expire_stale_new_jobs(conn, profile="testuser")
    assert _status(conn, "oldnull") == "expired"
    assert _status(conn, "newnull") == "new"
