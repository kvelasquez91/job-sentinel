"""The dashboard's default sort is 'added' — newest scrape run first.

The owner chose this default (2026-07-13 "Dashboard default sort by newest
jobs" session); the original uncommitted edit was lost to concurrent-
session branch surgery in the shared working dir, so this test pins the
static-file default the way the prompt-surface tests pin prompt text.
`updateSortArrows()` runs on DOMContentLoaded, so the state default alone
drives the rendered header/pill state.

Also pins that the FETCH order matches the DISPLAY order: the client
shows jobs grouped by run (newest run first) but /api/jobs used to
default to score DESC, so once the corpus exceeded the fetch limit the
dropped rows were the lowest-scored ones scattered across every run —
including the newest — and run-divider counts silently undercounted.
"""
import asyncio
import importlib
import os
import re

import main as main_mod

_app_mod = importlib.import_module("dashboard.app")

_INDEX = os.path.join(os.path.dirname(__file__), "..",
                      "dashboard", "static", "index.html")


def test_default_sort_is_added_newest_first():
    with open(_INDEX, encoding="utf-8") as f:
        html = f.read()
    m = re.search(r"^\s*sortBy:\s*'(\w+)',", html, re.MULTILINE)
    assert m, "state.sortBy default not found in dashboard/static/index.html"
    assert m.group(1) == "added", (
        f"dashboard default sort must be 'added' (newest first), got '{m.group(1)}'")
    d = re.search(r"^\s*sortDir:\s*'(\w+)',", html, re.MULTILINE)
    assert d and d.group(1) == "desc", "default sort direction must be 'desc'"


def test_added_column_displays_added_date_not_posting_date():
    """In the Added (session) view the date cell must render created_at —
    the sort key — not date_posted, or the column looks unsorted (a job
    posted weeks ago but scraped today sits in Today's run group)."""
    with open(_INDEX, encoding="utf-8") as f:
        html = f.read()
    assert re.search(
        r"sessionView \? formatAddedDate\(job\.created_at\) : formatDate\(job\.date_posted\)",
        html), "date cell must be mode-aware: created_at in Added view, date_posted otherwise"
    assert "function formatAddedDate(" in html


# --- Fetch order must match display order (truncation regression) ---

def _seed_runs(tmp_path, runs):
    """runs: list of (run_id, count, score) — count jobs per scrape run."""
    db = tmp_path / "jobs.db"
    conn = main_mod.init_database(str(db))
    for run_id, count, score in runs:
        conn.executemany(
            "INSERT INTO jobs (title, company, url, description, score, "
            "status, profile, run_id) VALUES (?, ?, ?, 'd', ?, 'new', 'testuser', ?)",
            [(f"Job r{run_id}-{i}", "Co", f"https://x/{run_id}/{i}",
              score + (i % 5), run_id) for i in range(count)],
        )
    conn.commit()
    conn.close()
    return db


def _get_jobs(**kw):
    args = dict(min_score=None, status=None, sort_by="added",
                limit=2000, profile="testuser", show_dismissed=False)
    args.update(kw)
    return asyncio.run(_app_mod.get_jobs(**args))


def test_api_added_sort_keeps_newest_runs_complete_under_limit(tmp_path, monkeypatch):
    """sort_by=added must order run-major (newest run first) so LIMIT
    truncation drops the OLDEST runs whole, never rows from recent runs.
    Newest run gets deliberately LOW scores: under the old score-DESC
    default it would be truncated away entirely."""
    monkeypatch.setattr(_app_mod, "DB_PATH", str(_seed_runs(
        tmp_path, [(1, 50, 90), (2, 50, 50), (3, 30, 10)])))

    jobs = _get_jobs(limit=60)
    assert len(jobs) == 60
    run_ids = [j["run_id"] for j in jobs]
    # Newest run (3) survives truncation completely despite low scores.
    assert run_ids.count(3) == 30
    # Run-major order, newest first; the oldest run is what gets dropped.
    assert run_ids == sorted(run_ids, reverse=True)
    assert 1 not in run_ids


def test_api_added_sort_orders_by_score_within_run_and_nulls_last(tmp_path, monkeypatch):
    db = _seed_runs(tmp_path, [(1, 3, 80), (2, 3, 20)])
    conn = main_mod.init_database(str(db))
    conn.execute(
        "INSERT INTO jobs (title, company, url, description, score, status, "
        "profile, run_id) VALUES ('Legacy', 'Co', 'https://x/legacy', 'd', "
        "99, 'new', 'testuser', NULL)")
    conn.commit()
    conn.close()
    monkeypatch.setattr(_app_mod, "DB_PATH", str(db))

    jobs = _get_jobs()
    # Legacy NULL-run_id rows group at the oldest end, matching the client
    # (which maps null run_id to -1).
    assert jobs[-1]["run_id"] is None
    for run_id in (1, 2):
        scores = [j["score"] for j in jobs if j["run_id"] == run_id]
        assert scores == sorted(scores, reverse=True)


def test_frontend_fetches_in_display_order_with_truncation_flag():
    """The /api/jobs fetch must request sort_by=added — the same order the
    default view displays — and flag when the payload hit the fetch limit
    so truncation is visible instead of silent."""
    with open(_INDEX, encoding="utf-8") as f:
        html = f.read()
    m = re.search(r"apiFetch\(`/api/jobs\?([^`]*)`\)", html)
    assert m, "jobs list fetch not found in dashboard/static/index.html"
    assert "sort_by=added" in m.group(1), (
        "fetch order must match the default display order (sort_by=added), "
        "otherwise LIMIT truncation silently drops rows from recent runs")
    assert "JOBS_FETCH_LIMIT" in m.group(1), (
        "fetch limit must use the shared JOBS_FETCH_LIMIT constant so the "
        "truncation check can't drift from the requested limit")
    assert re.search(r"const JOBS_FETCH_LIMIT\s*=", html)
    assert "fetchTruncated" in html, (
        "payload hitting the fetch limit must set a truncation flag the UI surfaces")
