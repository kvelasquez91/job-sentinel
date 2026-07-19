"""
FastAPI dashboard for Job Sentinel.
Serves the web UI and provides REST API endpoints for job data.
"""
import json
import logging
import os
import re
import sqlite3
import threading
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from claude_cli import ClaudeCLIError
import settings_store
import yaml
import statistics

from engine.scorer import seniority_bucket
from engine.llm_scorer import prefilter_job, clean_for_llm, extract_salary_regex
from local_area import LOCAL_COMMUTER_CITIES, build_local_area_regex
from profile_policy import (
    PROFILE_KEY,
    DASHBOARD_PAGE_TITLE,
    DASHBOARD_COMP_TIERS,
    DASHBOARD_PROFILES,
    LOCAL_CITIES,
)

_LOCAL_RE = build_local_area_regex(LOCAL_COMMUTER_CITIES)

# Config-driven local matcher shared with the client via /api/ui-config.
# The pattern string is handed to index.html so its Local filter uses the
# same strict-gap matcher as the server (no client/server local-match drift).
_UI_LOCAL_RE = build_local_area_regex(LOCAL_CITIES)
_UI_LOCAL_PATTERN = _UI_LOCAL_RE.pattern if _UI_LOCAL_RE else None


def _is_local_loc(location) -> bool:
    return bool(_LOCAL_RE and _LOCAL_RE.search((location or "").lower()))

import main
import tailor_diff

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Optional resume_tailor integration
# ---------------------------------------------------------------------------
try:
    from resume_tailor.jd_extractor import JDExtractionError
    from resume_tailor.pipeline import run_tailor_pipeline
    from resume_tailor.config import DOCX_OUTPUT_DIR
    from resume_tailor.google_api import GoogleAPIClient
    from resume_tailor.tailor_engine import (
        _SKILL_SUBCATEGORY_LABELS,
        _MASTER_TITLE_LINE,
    )
    from googleapiclient.errors import HttpError as _GoogleHttpError
    import google.auth.exceptions as _google_auth_exc
    _TAILOR_AVAILABLE = True
except ImportError as _tailor_import_err:
    logging.getLogger(__name__).warning(
        "resume_tailor module not importable — tailor endpoints will return 503: %s",
        _tailor_import_err,
    )
    _TAILOR_AVAILABLE = False
    DOCX_OUTPUT_DIR = os.path.join("data", "tailored_resumes")  # fallback for download
    GoogleAPIClient = None
    _SKILL_SUBCATEGORY_LABELS = {}
    _MASTER_TITLE_LINE = ""

    class JDExtractionError(Exception):  # type: ignore[no-redef]
        pass

    class _GoogleHttpError(Exception):  # type: ignore[no-redef]
        pass

    class _google_auth_exc:  # type: ignore[no-redef]
        class TransportError(Exception):
            pass

        class RefreshError(Exception):
            pass


# ---------------------------------------------------------------------------
# Async task store (in-memory; single-user tool)
# ---------------------------------------------------------------------------
_tailor_tasks: dict = {}
_tailor_lock = threading.Lock()
_tailor_watchdogs: dict = {}  # task_id -> threading.Timer

_PIPELINE_TIMEOUT_SECS = 600  # 10 minutes — Opus-via-CLI runs are slower than the API


def _cancel_watchdog(task_id: str) -> None:
    """Cancel and remove the timeout watchdog for a task."""
    watchdog = _tailor_watchdogs.pop(task_id, None)
    if watchdog:
        watchdog.cancel()


def _timeout_watchdog(task_id: str) -> None:
    """Mark a still-running task as timed out. Called by threading.Timer."""
    _tailor_watchdogs.pop(task_id, None)
    with _tailor_lock:
        task = _tailor_tasks.get(task_id)
        if task and task["status"] == "running":
            logging.getLogger(__name__).warning(
                "Tailor task %s exceeded %ds timeout", task_id, _PIPELINE_TIMEOUT_SECS
            )
            task.update({
                "status": "failed",
                "error_category": "timeout_error",
                "progress": "Timed out after 10 minutes",
                "error": (
                    "Tailoring pipeline timed out after 10 minutes. "
                    "Try again — if it keeps timing out, the job posting may be unusually slow to process."
                ),
                "completed_at": datetime.now().isoformat(),
            })

logger = logging.getLogger(__name__)

DB_PATH = os.environ.get("JOB_SENTINEL_DB", "data/jobs.db")
STATIC_DIR = Path(__file__).parent / "static"


def get_db_connection() -> sqlite3.Connection:
    """Create and return a SQLite connection with row_factory."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: initialize DB on startup."""
    # Run main.py's full init (CREATE TABLE IF NOT EXISTS + _migrate_columns) so
    # every column/table main.py's daily run has ever migrated or created
    # (salary_est_*, tracking fields, tailored_resume_url/tailored_at,
    # tailor_history, etc.) exists here too — the dashboard process never
    # otherwise runs main.py's migrations, so without this, columns/tables
    # added there 404/500 in the API until the next scheduled run touches the
    # DB. init_database is now the single source of truth for this schema.
    main.init_database(DB_PATH).close()

    db_dir = Path(DB_PATH).parent
    db_dir.mkdir(parents=True, exist_ok=True)
    yield


app = FastAPI(
    title="Job Sentinel Dashboard",
    description="Job Sentinel — job monitoring dashboard",
    version="1.0.0",
    lifespan=lifespan,
)


# --- Pydantic Models ---

class JobStatusUpdate(BaseModel):
    status: str  # "applied" | "saved" | "not_interested" | "new"


VALID_STAGES = {"applied", "phone_screen", "interview", "offer", "rejected", "ghosted"}


class TrackingUpdate(BaseModel):
    stage: Optional[str] = None
    notes: Optional[str] = None
    next_action_date: Optional[str] = None
    offer_json: Optional[str] = None


class FeedbackUpdate(BaseModel):
    direction: Optional[str] = None  # "more" | "less" | None (clear)


class TailorRequest(BaseModel):
    job_id: int


class AutoTailorSetting(BaseModel):
    enabled: bool


class JobResponse(BaseModel):
    id: int
    title: str
    company: str
    location: Optional[str] = None
    url: str
    description: Optional[str] = None
    salary_min: Optional[float] = None
    salary_max: Optional[float] = None
    date_posted: Optional[str] = None
    source: Optional[str] = None
    score: int
    status: str
    profile: Optional[str] = None
    match_explanation: Optional[str] = None
    llm_score: Optional[float] = None
    llm_explanation: Optional[str] = None
    created_at: Optional[str] = None


# --- Helper ---

def row_to_dict(row: sqlite3.Row) -> dict:
    """Convert sqlite3.Row to plain dict."""
    return dict(row)


# _filter_gate memo: job id -> (fingerprint-of-inputs, gate). The frontend
# polls /api/jobs every 5 minutes and row content almost never changes, so
# without this every poll re-ran extract_salary_regex/prefilter_job on every
# unscored row forever (and re-logged every salary discard). The fingerprint
# covers exactly the fields the gate reads, so any row edit recomputes.
_FILTER_GATE_CACHE: dict = {}
_FILTER_GATE_CACHE_MAX = 50_000   # ~5x corpus; clear-all guard, not an LRU


def _filter_gate(d: dict) -> Optional[dict]:
    """Why the scorer's prefilter hard-skips a still-unscored row (or None).

    Returns {"kind": "onsite"|"off-target", "reason": <str>} so the dashboard
    can show a distinct chip instead of a bare "—" for jobs that were
    intentionally skipped (never sent to the LLM) rather than merely
    not-yet-scored. Uses the SAME prefilter_job the scorer uses, so the two
    never drift. Only cap <= 15 is a hard skip; higher caps still get an LLM
    filter score and so are not "gated". Scored rows return None.
    """
    if d.get("filter_score") is not None:
        return None
    job_id = d.get("id")
    fingerprint = hash((d.get("title"), d.get("location"), d.get("description"),
                        d.get("salary_min"), d.get("salary_max")))
    if job_id is not None:
        cached = _FILTER_GATE_CACHE.get(job_id)
        if cached is not None and cached[0] == fingerprint:
            return cached[1]
    desc = clean_for_llm(d.get("description") or "")
    # quiet: this path re-reads rows already reported at scrape/score time —
    # a salary discard here is never new information.
    regex_min, regex_max = extract_salary_regex(desc, quiet=True)
    cap, reason = prefilter_job(
        d.get("title") or "", d.get("location") or "", desc,
        salary_min=d.get("salary_min") or regex_min,
        salary_max=d.get("salary_max") or regex_max)
    if cap is None or cap > 15:
        gate = None
    else:
        kind = "onsite" if "location" in (reason or "").lower() else "off-target"
        gate = {"kind": kind, "reason": reason}
    if job_id is not None:
        if len(_FILTER_GATE_CACHE) >= _FILTER_GATE_CACHE_MAX:
            _FILTER_GATE_CACHE.clear()
        _FILTER_GATE_CACHE[job_id] = (fingerprint, gate)
    return gate


# --- Routes ---

@app.get("/", response_class=HTMLResponse)
async def serve_index():
    """Serve the main dashboard HTML."""
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="index.html not found")
    return HTMLResponse(content=index_path.read_text(encoding="utf-8"))


@app.get("/api/jobs", response_class=JSONResponse)
async def get_jobs(
    min_score: Optional[int] = Query(None, description="Minimum score filter"),
    status: Optional[str] = Query(None, description="Status filter"),
    sort_by: Optional[str] = Query("score", description="Sort field: score|date|company|created_at|added"),
    limit: Optional[int] = Query(2000, description="Max results"),
    profile: str = Query(PROFILE_KEY, description="Profile filter"),
    show_dismissed: bool = Query(False, description="Include dismissed jobs"),
):
    """Return all jobs with optional filtering and sorting."""
    valid_statuses = {"new", "saved", "applied", "not_interested", "expired"}
    valid_sorts = {"score", "date", "company", "created_at", "added"}

    if status and status not in valid_statuses:
        raise HTTPException(status_code=400, detail=f"Invalid status. Must be one of: {valid_statuses}")

    sort_col = {
        "score": "score DESC",
        "date": "date_posted DESC NULLS LAST",
        "company": "company ASC",
        "created_at": "created_at DESC",
        # Mirrors the dashboard's default "Added" display order (run-major,
        # newest run first, NULL run_id at the oldest end like the client's
        # `run_id ?? -1`) so LIMIT truncation drops the oldest runs whole
        # instead of low scorers scattered across every run.
        "added": "run_id DESC NULLS LAST, score DESC",
    }.get(sort_by or "score", "score DESC")

    query = "SELECT * FROM jobs WHERE profile = ?"
    params = [profile]

    if min_score is not None:
        query += " AND score >= ?"
        params.append(min_score)

    if status:
        query += " AND status = ?"
        params.append(status)
    else:
        # By default hide expired jobs — stale listings no longer actionable
        query += " AND status != 'expired'"

    if not show_dismissed:
        query += " AND (dismissed IS NULL OR dismissed = 0)"

    query += f" ORDER BY {sort_col}"

    if limit:
        query += " LIMIT ?"
        params.append(limit)

    try:
        conn = get_db_connection()
        with conn:
            rows = conn.execute(query, params).fetchall()
        conn.close()
        results = []
        for r in rows:
            d = row_to_dict(r)
            # Compute the prefilter gate BEFORE popping description (it needs it).
            d["filter_gate"] = _filter_gate(d)
            # List payload stays slim — the detail endpoint serves the description.
            d.pop("description", None)
            # filter_json is fat detail — served by the detail endpoint, like
            # the description (toggleDetail lazy-fetches and merges the row).
            d.pop("filter_json", None)
            results.append(d)
        return results
    except sqlite3.OperationalError as e:
        logger.warning("DB error in get_jobs: %s", e)
        return []


@app.get("/api/jobs/{job_id}", response_class=JSONResponse)
async def get_job(job_id: int):
    """Return a single job by ID."""
    try:
        conn = get_db_connection()
        with conn:
            row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        conn.close()
    except sqlite3.OperationalError as e:
        raise HTTPException(status_code=500, detail=f"Database error: {e}")

    if not row:
        raise HTTPException(status_code=404, detail="Job not found")
    return row_to_dict(row)


@app.get("/api/jobs/{job_id}/comps", response_class=JSONResponse)
async def get_job_comps(job_id: int, profile: str = Query(PROFILE_KEY)):
    """Market comparables from our own DB: same seniority bucket + locality,
    salary-posted jobs from the last 180 days."""
    try:
        conn = get_db_connection()
        with conn:
            job = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
            rows = (
                conn.execute(
                    "SELECT id, title, location, salary_max FROM jobs "
                    "WHERE profile = ? AND salary_max IS NOT NULL "
                    "AND created_at >= date('now', '-180 days')",
                    (profile,),
                ).fetchall()
                if job else []
            )
        conn.close()
    except sqlite3.OperationalError as e:
        raise HTTPException(status_code=500, detail=f"Database error: {e}")
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    bucket = seniority_bucket(job["title"])
    local = _is_local_loc(job["location"])
    vals = sorted(
        r["salary_max"] for r in rows
        if r["id"] != job_id
        and seniority_bucket(r["title"]) == bucket
        and _is_local_loc(r["location"]) == local
    )
    out = {"count": len(vals), "bucket": bucket, "local": local,
           "median": None, "p25": None, "p75": None}
    if len(vals) >= 4:
        q = statistics.quantiles(vals, n=4)
        out.update({"median": statistics.median(vals), "p25": q[0], "p75": q[2]})
    return out


@app.patch("/api/jobs/{job_id}/status", response_class=JSONResponse)
async def update_job_status(job_id: int, body: JobStatusUpdate):
    """Update the status of a job (applied/saved/not_interested/new)."""
    valid_statuses = {"new", "saved", "applied", "not_interested", "expired"}
    if body.status not in valid_statuses:
        raise HTTPException(status_code=400, detail=f"Invalid status: {body.status}")

    try:
        conn = get_db_connection()
        with conn:
            result = conn.execute(
                "UPDATE jobs SET status = ? WHERE id = ?", (body.status, job_id)
            )
            if result.rowcount and body.status == "applied":
                conn.execute(
                    "UPDATE jobs SET applied_at = COALESCE(applied_at, datetime('now')) WHERE id = ?",
                    (job_id,),
                )
            row = (
                conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
                if result.rowcount else None
            )
        conn.close()
    except sqlite3.OperationalError as e:
        raise HTTPException(status_code=500, detail=f"Database error: {e}")

    if row is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return row_to_dict(row)


@app.patch("/api/jobs/{job_id}/dismiss", response_class=JSONResponse)
async def dismiss_job(job_id: int):
    """Mark a job as dismissed (hidden from main dashboard view)."""
    try:
        conn = get_db_connection()
        with conn:
            result = conn.execute(
                "UPDATE jobs SET dismissed = 1 WHERE id = ?", (job_id,)
            )
            row = (
                conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
                if result.rowcount else None
            )
        conn.close()
    except sqlite3.OperationalError as e:
        raise HTTPException(status_code=500, detail=f"Database error: {e}")
    if row is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return row_to_dict(row)


@app.patch("/api/jobs/{job_id}/undismiss", response_class=JSONResponse)
async def undismiss_job(job_id: int):
    """Restore a previously dismissed job."""
    try:
        conn = get_db_connection()
        with conn:
            result = conn.execute(
                "UPDATE jobs SET dismissed = 0 WHERE id = ?", (job_id,)
            )
            row = (
                conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
                if result.rowcount else None
            )
        conn.close()
    except sqlite3.OperationalError as e:
        raise HTTPException(status_code=500, detail=f"Database error: {e}")
    if row is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return row_to_dict(row)


@app.patch("/api/jobs/{job_id}/tracking", response_class=JSONResponse)
async def update_job_tracking(job_id: int, body: TrackingUpdate):
    """Update application-tracking fields. Empty string clears a field to NULL."""
    updates = {}
    for field in ("stage", "notes", "next_action_date", "offer_json"):
        val = getattr(body, field)
        if val is not None:
            updates[field] = val if val != "" else None

    if not updates:
        raise HTTPException(status_code=400, detail="No tracking fields provided")
    stage_val = updates.get("stage")
    if stage_val is not None and stage_val not in VALID_STAGES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid stage. Must be one of: {sorted(VALID_STAGES)}")
    offer_val = updates.get("offer_json")
    if offer_val is not None:
        try:
            json.loads(offer_val)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="offer_json must be valid JSON")

    set_clause = ", ".join(f"{k} = ?" for k in updates)
    params = list(updates.values()) + [job_id]
    try:
        conn = get_db_connection()
        with conn:
            result = conn.execute(f"UPDATE jobs SET {set_clause} WHERE id = ?", params)
            row = (
                conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
                if result.rowcount else None
            )
        conn.close()
    except sqlite3.OperationalError as e:
        raise HTTPException(status_code=500, detail=f"Database error: {e}")
    if row is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return row_to_dict(row)


@app.patch("/api/jobs/{job_id}/feedback", response_class=JSONResponse)
async def update_job_feedback(job_id: int, body: FeedbackUpdate):
    """Store 'more'/'less' calibration feedback for the LLM scorer (NULL clears)."""
    if body.direction is not None and body.direction not in ("more", "less"):
        raise HTTPException(status_code=400, detail="direction must be 'more', 'less', or null")
    try:
        conn = get_db_connection()
        with conn:
            result = conn.execute(
                "UPDATE jobs SET feedback = ? WHERE id = ?", (body.direction, job_id))
            row = (
                conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
                if result.rowcount else None
            )
        conn.close()
    except sqlite3.OperationalError as e:
        raise HTTPException(status_code=500, detail=f"Database error: {e}")
    if row is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return row_to_dict(row)


@app.get("/api/stats", response_class=JSONResponse)
async def get_stats(
    profile: str = Query(PROFILE_KEY, description="Profile filter"),
):
    """Return aggregate statistics about the job database."""
    try:
        conn = get_db_connection()
        with conn:
            total = conn.execute("SELECT COUNT(*) FROM jobs WHERE profile = ?", (profile,)).fetchone()[0]
            avg_score_row = conn.execute("SELECT AVG(score) FROM jobs WHERE profile = ?", (profile,)).fetchone()
            avg_score = round(avg_score_row[0] or 0, 1)
            top_score = conn.execute("SELECT MAX(score) FROM jobs WHERE profile = ?", (profile,)).fetchone()[0] or 0

            # Jobs added today
            today_str = datetime.now().strftime("%Y-%m-%d")
            jobs_today = conn.execute(
                "SELECT COUNT(*) FROM jobs WHERE profile = ? AND date(created_at) = ?", (profile, today_str)
            ).fetchone()[0]

            # Top companies by job count
            top_companies_rows = conn.execute(
                """
                SELECT company, COUNT(*) as count
                FROM jobs
                WHERE profile = ?
                GROUP BY company
                ORDER BY count DESC
                LIMIT 10
                """,
                (profile,)
            ).fetchall()
            top_companies = [{"company": r[0], "count": r[1]} for r in top_companies_rows]

            # Score distribution
            buckets = {
                "80-100": conn.execute("SELECT COUNT(*) FROM jobs WHERE profile = ? AND score >= 80", (profile,)).fetchone()[0],
                "60-79": conn.execute("SELECT COUNT(*) FROM jobs WHERE profile = ? AND score >= 60 AND score < 80", (profile,)).fetchone()[0],
                "40-59": conn.execute("SELECT COUNT(*) FROM jobs WHERE profile = ? AND score >= 40 AND score < 60", (profile,)).fetchone()[0],
                "0-39": conn.execute("SELECT COUNT(*) FROM jobs WHERE profile = ? AND score < 40", (profile,)).fetchone()[0],
            }

            # Last run info
            last_run_row = conn.execute(
                "SELECT completed_at, total_scraped, total_new, total_above_threshold FROM runs "
                "WHERE status = 'completed' AND profile = ? ORDER BY completed_at DESC LIMIT 1",
                (profile,)
            ).fetchone()

        conn.close()

        last_run = None
        if last_run_row:
            last_run = {
                "completed_at": last_run_row[0],
                "total_scraped": last_run_row[1],
                "total_new": last_run_row[2],
                "total_above_threshold": last_run_row[3],
            }

        return {
            "total_jobs": total,
            "avg_score": avg_score,
            "top_score": top_score,
            "jobs_today": jobs_today,
            "top_companies": top_companies,
            "score_distribution": buckets,
            "last_run": last_run,
        }
    except sqlite3.OperationalError as e:
        logger.warning("DB error in get_stats: %s", e)
        return {
            "total_jobs": 0,
            "avg_score": 0,
            "top_score": 0,
            "jobs_today": 0,
            "top_companies": [],
            "score_distribution": {"80-100": 0, "60-79": 0, "40-59": 0, "0-39": 0},
            "last_run": None,
        }


def _comp_bucket_labels(tiers):
    """Histogram bucket labels derived from the same comp tiers the chips use."""
    t = [int(x) for x in tiers]
    return [f"<{t[0]//1000}k", f"{t[0]//1000}-{t[1]//1000}k",
            f"{t[1]//1000}-{t[2]//1000}k", f"{t[2]//1000}k+"]


@app.get("/api/analytics", response_class=JSONResponse)
async def get_analytics(profile: str = Query(PROFILE_KEY)):
    """Market analytics: weekly inflow, live comp distribution, 14-day source yield."""
    try:
        conn = get_db_connection()
        with conn:
            weekly_rows = conn.execute(
                "SELECT strftime('%Y-%W', created_at) AS wk, COUNT(*) AS n, "
                "SUM(CASE WHEN score >= 80 THEN 1 ELSE 0 END) AS high "
                "FROM jobs WHERE profile = ? AND created_at >= date('now', '-56 days') "
                "GROUP BY wk ORDER BY wk",
                (profile,),
            ).fetchall()
            comp_rows = conn.execute(
                "SELECT COALESCE(salary_max, salary_est_max) AS comp FROM jobs "
                "WHERE profile = ? AND status = 'new' "
                "AND COALESCE(salary_max, salary_est_max) IS NOT NULL",
                (profile,),
            ).fetchall()
            source_rows = conn.execute(
                "SELECT source, COUNT(*) AS n, "
                "SUM(CASE WHEN score >= 80 THEN 1 ELSE 0 END) AS high "
                "FROM jobs WHERE profile = ? AND created_at >= date('now', '-14 days') "
                "GROUP BY source ORDER BY n DESC",
                (profile,),
            ).fetchall()
        conn.close()
    except sqlite3.OperationalError as e:
        logger.warning("DB error in get_analytics: %s", e)
        return {"weekly": [], "comp_buckets": {}, "sources_14d": []}

    labels = _comp_bucket_labels(DASHBOARD_COMP_TIERS)
    t = [int(x) for x in DASHBOARD_COMP_TIERS]
    buckets = {k: 0 for k in labels}
    for r in comp_rows:
        c = r["comp"]
        if c < t[0]:
            buckets[labels[0]] += 1
        elif c < t[1]:
            buckets[labels[1]] += 1
        elif c < t[2]:
            buckets[labels[2]] += 1
        else:
            buckets[labels[3]] += 1

    return {
        "weekly": [{"week": r["wk"], "new_jobs": r["n"], "high_matches": r["high"] or 0}
                   for r in weekly_rows],
        "comp_buckets": buckets,
        "sources_14d": [{"source": r["source"] or "?", "jobs": r["n"], "high": r["high"] or 0}
                        for r in source_rows],
    }


@app.get("/api/profiles", response_class=JSONResponse)
async def get_profiles():
    """Return distinct profiles that still have at least one non-expired job."""
    try:
        conn = get_db_connection()
        rows = conn.execute(
            "SELECT DISTINCT profile FROM jobs WHERE status != 'expired' ORDER BY profile"
        ).fetchall()
        conn.close()
        return [r["profile"] if isinstance(r, dict) else r[0] for r in rows]
    except sqlite3.OperationalError as e:
        logger.warning("DB error in get_profiles: %s", e)
        return [PROFILE_KEY]


@app.get("/api/ui-config", response_class=JSONResponse)
async def get_ui_config():
    """Config-driven UI values — index.html hardcodes no owner policy."""
    return {
        "page_title": DASHBOARD_PAGE_TITLE,
        "default_profile": PROFILE_KEY,
        "profiles": DASHBOARD_PROFILES,
        "comp_tiers": DASHBOARD_COMP_TIERS,
        "local_pattern": _UI_LOCAL_PATTERN,
    }


@app.get("/api/companies", response_class=JSONResponse)
async def get_companies():
    """Return all cached company insights keyed by normalized company name."""
    try:
        conn = get_db_connection()
        with conn:
            rows = conn.execute("SELECT * FROM company_insights").fetchall()
        conn.close()
        return {r["company_name_normalized"]: row_to_dict(r) for r in rows}
    except sqlite3.OperationalError as e:
        logger.warning("DB error in get_companies: %s", e)
        return {}


@app.get("/api/companies/{company_name}", response_class=JSONResponse)
async def get_company(company_name: str):
    """Return insights for a specific company (by normalized name or exact name)."""
    import re
    normalized = re.sub(r"[^\w\s]", "", company_name.lower()).strip()
    try:
        conn = get_db_connection()
        with conn:
            row = conn.execute(
                "SELECT * FROM company_insights WHERE company_name_normalized = ?",
                (normalized,),
            ).fetchone()
        conn.close()
    except sqlite3.OperationalError as e:
        raise HTTPException(status_code=500, detail=f"Database error: {e}")

    if not row:
        raise HTTPException(status_code=404, detail="Company not found")
    return row_to_dict(row)


@app.get("/api/runs", response_class=JSONResponse)
async def get_runs(limit: int = 10):
    """Return recent scraper runs, newest first. limit clamped to 1..5000;
    the dashboard requests a high limit to label session dividers."""
    limit = max(1, min(limit, 5000))
    try:
        conn = get_db_connection()
        with conn:
            rows = conn.execute(
                "SELECT * FROM runs ORDER BY started_at DESC LIMIT ?", (limit,)
            ).fetchall()
        conn.close()
        return [row_to_dict(r) for r in rows]
    except sqlite3.OperationalError as e:
        logger.warning("DB error in get_runs: %s", e)
        return []


# ---------------------------------------------------------------------------
# Settings — auto-tailor toggle (read by main.py's post-run pass)
# ---------------------------------------------------------------------------

def _auto_tailor_config_default() -> dict:
    """config.yaml's auto_tailor section, {} when unreadable. Read directly
    rather than via main.load_config, which sys.exits on a missing file —
    a bad config must degrade the toggle default, not kill the request."""
    try:
        with open("config.yaml", "r", encoding="utf-8") as f:
            return (yaml.safe_load(f) or {}).get("auto_tailor") or {}
    except (OSError, yaml.YAMLError):
        return {}


@app.get("/api/settings/auto-tailor", response_class=JSONResponse)
async def get_auto_tailor_setting():
    """Current toggle state: DB value if ever set, else config.yaml default."""
    conn = get_db_connection()
    try:
        return {"enabled": settings_store.auto_tailor_enabled(
            conn, _auto_tailor_config_default())}
    finally:
        conn.close()


@app.put("/api/settings/auto-tailor", response_class=JSONResponse)
async def put_auto_tailor_setting(body: AutoTailorSetting):
    """Flip the toggle; the next daily run reads this before its pass.
    Deliberately no _TAILOR_AVAILABLE 503: the toggle governs the daily
    run's pass, not this process's tailor endpoints."""
    conn = get_db_connection()
    try:
        settings_store.set_setting(
            conn, settings_store.AUTO_TAILOR_KEY, "1" if body.enabled else "0")
        return {"enabled": body.enabled}
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Resume tailoring — background worker
# ---------------------------------------------------------------------------

def _tailor_worker(
    task_id: str,
    job_id: int,
    job_url: str,
    company: str,
    title: str,
) -> None:
    """Run the tailoring pipeline in a background thread and map its outcome
    onto the shared task dict. The heavy lifting lives in
    resume_tailor.pipeline.run_tailor_pipeline; this wrapper only does
    task-status bookkeeping and error categorization."""

    def progress(msg: str) -> None:
        with _tailor_lock:
            task = _tailor_tasks.get(task_id)
            if task:
                task["progress"] = msg
                task["steps"].append(msg)

    def _fail(category: str, progress_msg: str, error_msg: str) -> None:
        _cancel_watchdog(task_id)
        with _tailor_lock:
            _tailor_tasks[task_id].update({
                "status": "failed",
                "error_category": category,
                "progress": progress_msg,
                "error": error_msg,
                "completed_at": datetime.now().isoformat(),
            })

    try:
        with _tailor_lock:
            _tailor_tasks[task_id]["status"] = "running"

        result = run_tailor_pipeline(
            {"id": job_id, "url": job_url, "company": company, "title": title},
            db_path=DB_PATH,
            progress=progress,
        )

        _cancel_watchdog(task_id)
        with _tailor_lock:
            _tailor_tasks[task_id].update({
                "status": "completed",
                "progress": "Resume tailored successfully!",
                "completed_at": datetime.now().isoformat(),
                "result": {k: result[k] for k in (
                    "google_doc_url", "docx_filename", "ats_score",
                    "keywords_matched", "keywords_total", "issues", "warnings",
                    "company", "title", "total_input_tokens",
                    "total_output_tokens", "est_cost", "issue_facts")},
            })
        logger.info(
            "Tailor task %s completed: ATS %d, %d keywords",
            task_id, result["ats_score"], result["keywords_matched"],
        )

    except JDExtractionError as exc:
        logger.warning("Task %s: JD extraction failed for %s: %s", task_id, job_url, exc)
        _fail(
            "jd_extraction_error",
            "Failed: could not extract job description",
            "The job posting page couldn't be read automatically. "
            "Try a different job URL or a supported job board (Greenhouse, Lever, Ashby, etc.). "
            f"Details: {exc}",
        )

    except ClaudeCLIError:
        logger.error("Task %s: Claude CLI error", task_id)
        _fail(
            "auth_error",
            "Failed: Claude CLI error",
            "the `claude` CLI failed — ensure Claude Code is installed and logged in to your subscription.",
        )

    except (
        _GoogleHttpError,
        FileNotFoundError,
        _google_auth_exc.TransportError,
        _google_auth_exc.RefreshError,
    ) as exc:
        if isinstance(exc, FileNotFoundError):
            msg = (
                "Google credentials file not found — "
                f"run scripts/setup_google_auth.py to authenticate. ({exc})"
            )
        elif isinstance(exc, (_google_auth_exc.TransportError, _google_auth_exc.RefreshError)):
            msg = (
                "Google OAuth token expired or invalid — "
                "run scripts/setup_google_auth.py again to re-authorize."
            )
        else:
            http_status = getattr(getattr(exc, "resp", None), "status", None)
            if http_status in (401, 403):
                msg = "Google API auth failed — run scripts/setup_google_auth.py again."
            else:
                msg = f"Google API error: {exc}"
        partial = getattr(exc, "partial_doc_id", None)
        if partial:
            msg += f" (A partial Google Doc was created before the failure — ID: {partial})"
        logger.error("Task %s: Google API error: %s", task_id, exc)
        _fail("google_api_error", "Failed: Google API error", msg)

    except Exception as exc:
        msg = str(exc)
        partial = getattr(exc, "partial_doc_id", None)
        if partial:
            msg += f" (A partial Google Doc was created before the failure — ID: {partial})"
        logger.error("Tailor task %s failed: %s", task_id, exc, exc_info=True)
        _fail("unknown_error", f"Failed: {str(exc)[:100]}", msg)


# ---------------------------------------------------------------------------
# Resume tailoring — API endpoints
# ---------------------------------------------------------------------------

@app.post("/api/tailor", response_class=JSONResponse)
async def start_tailor(body: TailorRequest):
    """
    Kick off an async resume tailoring task for the given job.
    Returns a task_id immediately; poll /api/tailor/status/{task_id} for progress.
    """
    if not _TAILOR_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Resume tailoring module not available. Check server logs.",
        )

    try:
        conn = get_db_connection()
        with conn:
            row = conn.execute("SELECT * FROM jobs WHERE id = ?", (body.job_id,)).fetchone()
        conn.close()
    except sqlite3.OperationalError as exc:
        raise HTTPException(status_code=500, detail=f"Database error: {exc}")

    if not row:
        raise HTTPException(status_code=404, detail="Job not found")

    job = row_to_dict(row)
    task_id = str(uuid.uuid4())

    with _tailor_lock:
        _tailor_tasks[task_id] = {
            "task_id": task_id,
            "job_id": body.job_id,
            "status": "pending",
            "progress": "Starting...",
            "steps": [],
            "result": None,
            "error": None,
            "error_category": None,
            "created_at": datetime.now().isoformat(),
            "completed_at": None,
        }

    thread = threading.Thread(
        target=_tailor_worker,
        args=(task_id, job["id"], job["url"], job.get("company") or "", job.get("title") or ""),
        daemon=True,
        name=f"tailor-{task_id[:8]}",
    )
    thread.start()

    watchdog = threading.Timer(_PIPELINE_TIMEOUT_SECS, _timeout_watchdog, args=(task_id,))
    watchdog.daemon = True
    watchdog.start()
    _tailor_watchdogs[task_id] = watchdog

    logger.info("Started tailor task %s for job %s (%s)", task_id, job["id"], job.get("title"))

    return {"task_id": task_id, "status": "pending"}


@app.get("/api/tailor/status/{task_id}", response_class=JSONResponse)
async def tailor_status(task_id: str):
    """Return the current status and progress of a tailoring task."""
    with _tailor_lock:
        task = _tailor_tasks.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return task


@app.get("/api/tailor/download/{task_id}")
async def tailor_download(task_id: str):
    """Download the exported .docx file for a completed tailoring task."""
    with _tailor_lock:
        task = _tailor_tasks.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    if task["status"] != "completed" or not task.get("result"):
        raise HTTPException(status_code=400, detail="Task is not completed yet")

    docx_filename = task["result"]["docx_filename"]
    docx_path = Path(DOCX_OUTPUT_DIR) / docx_filename
    if not docx_path.exists():
        raise HTTPException(status_code=404, detail="Export file not found on disk")

    return FileResponse(
        path=str(docx_path),
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        filename=docx_filename,
    )


@app.post("/api/tailor/retry/{task_id}", response_class=JSONResponse)
async def tailor_retry(task_id: str):
    """Retry a failed tailoring task (reuses the same task_id)."""
    if not _TAILOR_AVAILABLE:
        raise HTTPException(status_code=503, detail="Resume tailoring module not available")

    with _tailor_lock:
        task = _tailor_tasks.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    if task["status"] not in ("failed",):
        raise HTTPException(status_code=400, detail="Only failed tasks can be retried")

    job_id = task["job_id"]
    try:
        conn = get_db_connection()
        with conn:
            row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        conn.close()
    except sqlite3.OperationalError as exc:
        raise HTTPException(status_code=500, detail=f"Database error: {exc}")

    if not row:
        raise HTTPException(status_code=404, detail="Job not found")

    job = row_to_dict(row)

    with _tailor_lock:
        _tailor_tasks[task_id].update(
            {
                "status": "pending",
                "progress": "Retrying...",
                "steps": [],
                "result": None,
                "error": None,
                "error_category": None,
                "completed_at": None,
            }
        )

    thread = threading.Thread(
        target=_tailor_worker,
        args=(task_id, job["id"], job["url"], job.get("company") or "", job.get("title") or ""),
        daemon=True,
        name=f"tailor-retry-{task_id[:8]}",
    )
    thread.start()

    # Cancel any stale watchdog from the previous run and start a fresh one
    _cancel_watchdog(task_id)
    watchdog = threading.Timer(_PIPELINE_TIMEOUT_SECS, _timeout_watchdog, args=(task_id,))
    watchdog.daemon = True
    watchdog.start()
    _tailor_watchdogs[task_id] = watchdog

    logger.info("Retrying tailor task %s for job %s", task_id, job["id"])

    return {"task_id": task_id, "status": "pending"}


# ---------------------------------------------------------------------------
# Tailor history endpoints
# ---------------------------------------------------------------------------

@app.get("/api/tailor/history", response_class=JSONResponse)
async def get_tailor_history(job_id: Optional[int] = Query(None, description="Filter by job ID")):
    """Return all tailor history records, most recent first. Optionally filtered by job_id."""
    try:
        conn = get_db_connection()
        with conn:
            if job_id is not None:
                rows = conn.execute(
                    "SELECT * FROM tailor_history WHERE job_id = ? ORDER BY created_at DESC",
                    (job_id,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM tailor_history ORDER BY created_at DESC"
                ).fetchall()
        conn.close()
    except sqlite3.OperationalError as e:
        logger.warning("DB error in get_tailor_history: %s", e)
        return []

    results = []
    for row in rows:
        d = row_to_dict(row)
        # Parse JSON columns for list response (keep edits_json as string for bandwidth)
        d["ats_issues"] = json.loads(d["ats_issues"] or "[]")
        d["keywords_list"] = json.loads(d["keywords_list"] or "[]")
        # edits_json left as string — only parse in the detail endpoint
        results.append(d)
    return results


@app.get("/api/tailor/history/{history_id}", response_class=JSONResponse)
async def get_tailor_history_detail(history_id: int):
    """Return a single tailor history record with full parsed detail including edits."""
    try:
        conn = get_db_connection()
        with conn:
            row = conn.execute(
                "SELECT * FROM tailor_history WHERE id = ?", (history_id,)
            ).fetchone()
        conn.close()
    except sqlite3.OperationalError as e:
        raise HTTPException(status_code=500, detail=f"Database error: {e}")

    if not row:
        raise HTTPException(status_code=404, detail="History record not found")

    d = row_to_dict(row)
    d["ats_issues"] = json.loads(d["ats_issues"] or "[]")
    d["keywords_list"] = json.loads(d["keywords_list"] or "[]")
    d["edits"] = json.loads(d["edits_json"] or "{}")
    return d


@app.get("/api/tailor/history/download/{history_id}")
async def tailor_history_download(history_id: int):
    """Download the .docx file for a history record."""
    try:
        conn = get_db_connection()
        with conn:
            row = conn.execute(
                "SELECT docx_path, job_title, company FROM tailor_history WHERE id = ?",
                (history_id,),
            ).fetchone()
        conn.close()
    except sqlite3.OperationalError as e:
        raise HTTPException(status_code=500, detail=f"Database error: {e}")

    if not row:
        raise HTTPException(status_code=404, detail="History record not found")

    docx_path = row["docx_path"]
    if not docx_path or not Path(docx_path).exists():
        raise HTTPException(status_code=404, detail="Export file not found on disk")

    filename = Path(docx_path).name
    return FileResponse(
        path=str(docx_path),
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        filename=filename,
    )


def _tailor_diff_payload(row: sqlite3.Row) -> dict:
    """Diff payload for a history row that HAS final_text. All stored-JSON
    parsing is defensive: malformed columns yield diff_unavailable, not a 500."""
    try:
        edits_dict = json.loads(row["edits_json"] or "{}")
    except (TypeError, ValueError):
        return {"diff_unavailable": "stored edits_json is not valid JSON"}
    _, master_text = tailor_diff.unwrap_edits(edits_dict)
    if not master_text:
        return {"diff_unavailable": "no master text stored with this tailor"}

    def _load(col, default):
        try:
            return json.loads(row[col]) if row[col] else default
        except (TypeError, ValueError):
            return default

    return {
        "blocks": tailor_diff.compute_diff_blocks(master_text, row["final_text"]),
        "verdicts": _load("edit_verdicts_json", []),
        "warnings": _load("warnings_json", {}),
        "final_text_source": row["final_text_source"] or "pipeline",
    }


@app.get("/api/tailor/diff/{history_id}", response_class=JSONResponse)
async def get_tailor_diff(history_id: int):
    """Ground-truth diff for one tailor. Pure read — never writes, and has
    no resume_tailor dependency (rows with stored text diff even when the
    tailor imports fail)."""
    try:
        conn = get_db_connection()
        with conn:
            row = conn.execute(
                "SELECT * FROM tailor_history WHERE id = ?",
                (history_id,)).fetchone()
        conn.close()
    except sqlite3.OperationalError as e:
        raise HTTPException(status_code=500, detail=f"Database error: {e}")
    if not row:
        raise HTTPException(status_code=404, detail="History record not found")
    if row["final_text"] is None:
        return {"final_text_missing": True}
    return _tailor_diff_payload(row)


_DOC_ID_RE = re.compile(r"/d/([A-Za-z0-9_-]+)")


@app.post("/api/tailor/diff/{history_id}/sync", response_class=JSONResponse)
async def sync_tailor_diff(history_id: int):
    """Explicit backfill for backlog rows: fetch the tailored Google Doc
    once, compute verdicts, cache first-writer-wins. Idempotent — a row
    that already has final_text returns its payload without fetching."""
    if not _TAILOR_AVAILABLE:
        raise HTTPException(
            status_code=503, detail="resume_tailor module not available")
    try:
        conn = get_db_connection()
        with conn:
            row = conn.execute(
                "SELECT * FROM tailor_history WHERE id = ?",
                (history_id,)).fetchone()
        conn.close()
    except sqlite3.OperationalError as e:
        raise HTTPException(status_code=500, detail=f"Database error: {e}")
    if not row:
        raise HTTPException(status_code=404, detail="History record not found")
    if row["final_text"] is not None:
        return _tailor_diff_payload(row)

    m = _DOC_ID_RE.search(row["google_doc_url"] or "")
    if not m:
        raise HTTPException(
            status_code=404, detail="No Google Doc URL stored for this record")
    try:
        edits_dict = json.loads(row["edits_json"] or "{}")
    except (TypeError, ValueError):
        return {"diff_unavailable": "stored edits_json is not valid JSON"}
    _, master_text = tailor_diff.unwrap_edits(edits_dict)
    if not master_text:
        return {"diff_unavailable": "no master text stored with this tailor"}

    try:
        client = GoogleAPIClient()
        client.authenticate()
        text = client.extract_plain_text(client.read_document(m.group(1)))
    except Exception as exc:  # auth down, doc deleted/403 — write nothing
        raise HTTPException(
            status_code=502, detail=f"Google Doc fetch failed: {exc}")

    verdicts = tailor_diff.compute_edit_verdicts(
        edits_dict, text, _SKILL_SUBCATEGORY_LABELS, _MASTER_TITLE_LINE)
    try:
        conn = get_db_connection()
        with conn:
            # First writer wins: a concurrent duplicate becomes a no-op and
            # both callers return whatever the row now holds.
            conn.execute(
                "UPDATE tailor_history SET final_text = ?, "
                "final_text_source = 'live_fetch', edit_verdicts_json = ? "
                "WHERE id = ? AND final_text IS NULL",
                (text, json.dumps(verdicts), history_id))
            row = conn.execute(
                "SELECT * FROM tailor_history WHERE id = ?",
                (history_id,)).fetchone()
        conn.close()
    except sqlite3.OperationalError as e:
        raise HTTPException(status_code=500, detail=f"Database error: {e}")
    return _tailor_diff_payload(row)


# Mount static files last to avoid route conflicts
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
