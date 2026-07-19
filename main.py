#!/usr/bin/env python3
"""
Job Sentinel - Automated job sourcing engine for Senior PM / AI roles
"""
import argparse
import json
import logging
import logging.handlers
import queue
import re
import sqlite3
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import yaml

from engine.scorer import apply_layoff_penalty, load_layoff_companies
from filter_match import EFFECTIVE_SCORE_SQL
from profile_policy import PROFILE_KEY

import settings_store
from claude_cli import ClaudeCLIError, MAX_CONCURRENCY as CLAUDE_CLI_MAX_CONCURRENCY

MIN_PYTHON = (3, 12)


def _check_python_version() -> None:
    """Fail fast on unsupported interpreters (tested only on 3.12+)."""
    if sys.version_info < MIN_PYTHON:
        sys.exit(
            f"Job Sentinel requires Python {'.'.join(map(str, MIN_PYTHON))}+ "
            f"(this is {sys.version.split()[0]}). Recreate the venv with a newer Python."
        )

# ---------------------------------------------------------------------------
# Logging setup (called early, before anything else)
# ---------------------------------------------------------------------------

def setup_logging(log_level: str = "INFO", verbose: bool = False) -> None:
    """
    Configure root logger with:
      - Console (stdout) handler
      - RotatingFileHandler for main log  (5 MB × 5 backups)
      - RotatingFileHandler for error log (2 MB × 10 backups)
    """
    # JOB_SENTINEL_LOG_DIR redirects file logs (tests set it via conftest.py
    # so pytest runs never write into the real logs/ directory)
    logs_dir = Path(os.environ.get("JOB_SENTINEL_LOG_DIR", "logs"))
    logs_dir.mkdir(parents=True, exist_ok=True)

    level = logging.DEBUG if verbose else getattr(logging, log_level.upper(), logging.INFO)

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    root = logging.getLogger()
    root.setLevel(level)
    # Clear any handlers added before this call (e.g. basicConfig defaults)
    root.handlers.clear()

    # --- Console ---
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level)
    console.setFormatter(fmt)
    root.addHandler(console)

    # --- Main rotating log ---
    main_handler = logging.handlers.RotatingFileHandler(
        logs_dir / "job_sentinel.log",
        maxBytes=5 * 1024 * 1024,   # 5 MB
        backupCount=5,
        encoding="utf-8",
    )
    main_handler.setLevel(level)
    main_handler.setFormatter(fmt)
    root.addHandler(main_handler)

    # --- Error-only rotating log ---
    error_handler = logging.handlers.RotatingFileHandler(
        logs_dir / "errors.log",
        maxBytes=2 * 1024 * 1024,   # 2 MB
        backupCount=10,
        encoding="utf-8",
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(fmt)
    root.addHandler(error_handler)


logger = logging.getLogger("job_sentinel")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(path: str = "config.yaml") -> dict:
    """Load and return config from YAML file."""
    config_path = Path(path)
    if not config_path.exists():
        logger.error("Config file not found: %s", path)
        sys.exit(1)
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    logger.info("Config loaded from %s", path)
    return config


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------

def init_database(db_path: str) -> sqlite3.Connection:
    """
    Initialize SQLite database with required tables.
    Runs ADD COLUMN migrations for new columns on existing tables.
    Returns an open connection.
    """
    db_dir = Path(db_path).parent
    db_dir.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    # Three writers can now overlap (streaming saves on this conn, the
    # scoring consumer's flushes, enrichment upserts) plus the dashboard
    # process; the 5s default busy timeout is a cliff a WAL checkpoint can
    # push a writer over — and a lost batch save fails the whole run.
    conn.execute("PRAGMA busy_timeout=30000")

    conn.executescript("""
        CREATE TABLE IF NOT EXISTS jobs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            company TEXT NOT NULL,
            location TEXT,
            url TEXT UNIQUE NOT NULL,
            description TEXT,
            salary_min REAL,
            salary_max REAL,
            date_posted TEXT,
            source TEXT,
            score INTEGER DEFAULT 0,
            status TEXT DEFAULT 'new',
            match_explanation TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            run_id INTEGER,
            normalized_title TEXT,
            normalized_company TEXT,
            profile TEXT NOT NULL DEFAULT 'default'
        );

        CREATE TABLE IF NOT EXISTS runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            started_at TIMESTAMP,
            completed_at TIMESTAMP,
            total_scraped INTEGER DEFAULT 0,
            total_new INTEGER DEFAULT 0,
            total_above_threshold INTEGER DEFAULT 0,
            sources TEXT,
            status TEXT DEFAULT 'running',
            profile TEXT DEFAULT 'default'
        );

        CREATE TABLE IF NOT EXISTS company_insights (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            company_name TEXT NOT NULL,
            company_name_normalized TEXT UNIQUE NOT NULL,
            is_public INTEGER DEFAULT 0,
            stock_ticker TEXT,
            stock_trend TEXT DEFAULT 'unknown',
            stock_change_30d REAL,
            headcount_estimate TEXT,
            headcount_trend TEXT DEFAULT 'unknown',
            has_recent_layoffs INTEGER DEFAULT 0,
            layoff_details TEXT DEFAULT '[]',
            recent_news_sentiment TEXT DEFAULT 'neutral',
            recent_headlines TEXT DEFAULT '[]',
            health_score INTEGER DEFAULT 50,
            health_summary TEXT,
            health_flags TEXT DEFAULT '[]',
            data_sources TEXT DEFAULT '[]',
            fetched_at TEXT NOT NULL,
            cache_ttl_hours INTEGER DEFAULT 48
        );

        CREATE TABLE IF NOT EXISTS run_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id INTEGER,
            status TEXT NOT NULL,
            error_message TEXT,
            ran_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS tailor_history (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            job_id              INTEGER,
            created_at          TIMESTAMP,
            google_doc_url      TEXT,
            docx_path           TEXT,
            ats_score           INTEGER,
            ats_passed          BOOLEAN,
            ats_issues          TEXT,
            keywords_matched    INTEGER,
            keywords_total      INTEGER,
            keywords_list       TEXT,
            edits_json          TEXT,
            est_cost            REAL,
            total_input_tokens  INTEGER,
            total_output_tokens INTEGER,
            job_title           TEXT,
            company             TEXT
        );

        CREATE TABLE IF NOT EXISTS settings (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_jobs_score ON jobs(score DESC);
        CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
        CREATE INDEX IF NOT EXISTS idx_jobs_created ON jobs(created_at DESC);
        CREATE INDEX IF NOT EXISTS idx_company_insights_norm ON company_insights(company_name_normalized);
        CREATE INDEX IF NOT EXISTS idx_run_history_ran_at ON run_history(ran_at DESC);
    """)
    conn.commit()

    # --- Column migrations (safe on existing DBs) ---
    # Must run before any index that references columns added by migration
    _migrate_columns(conn)

    # --- Indexes that depend on migrated columns ---
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_jobs_norm_dedup ON jobs(normalized_title, normalized_company)"
    )
    conn.commit()

    # --- Score audit (observability) ---
    # DB-level trigger so EVERY writer of jobs.score is captured: save_jobs,
    # reblend, the LLM pass, and manual sqlite edits alike. Added after a
    # 2026-07-12 audit found an unattributable overnight score change.
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS score_audit (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            job_id INTEGER NOT NULL,
            old_score INTEGER,
            new_score INTEGER,
            changed_at TEXT DEFAULT (datetime('now'))
        );
        CREATE INDEX IF NOT EXISTS idx_score_audit_job ON score_audit(job_id);
        CREATE TRIGGER IF NOT EXISTS trg_jobs_score_audit
        AFTER UPDATE OF score ON jobs
        WHEN COALESCE(OLD.score, -999) != COALESCE(NEW.score, -999)
        BEGIN
            INSERT INTO score_audit (job_id, old_score, new_score)
            VALUES (OLD.id, OLD.score, NEW.score);
        END;
    """)
    conn.commit()

    logger.info("Database initialized at %s", db_path)
    return conn


def expire_stale_new_jobs(conn: sqlite3.Connection, profile: str, max_age_days: int = 14) -> int:
    """
    Set status='expired' for any job that is still 'new' but whose date_posted
    is older than max_age_days. Scoped to the given profile.
    Falls back to created_at when date_posted is NULL.
    Returns the number of rows updated.
    """
    # date(date_posted) normalizes/validates the value: it returns the ISO date
    # for a real date and NULL for anything unparseable (e.g. Workday's
    # "Posted 30+ Days Ago"), so bad values fall back to created_at instead of
    # sorting greater than the cutoff and never expiring.
    result = conn.execute(
        """UPDATE jobs SET status = 'expired'
           WHERE status = 'new'
             AND profile = ?
             AND (
               (date(date_posted) IS NOT NULL AND date(date_posted) < date('now', ?))
               OR (date(date_posted) IS NULL AND date(created_at) < date('now', ?))
             )""",
        (profile, f"-{max_age_days} days", f"-{max_age_days} days"),
    )
    conn.commit()
    expired = result.rowcount
    if expired:
        logger.info(
            "Expired %d stale 'new' jobs (date_posted or created_at > %d days ago).",
            expired, max_age_days,
        )
    return expired


def prune_old_expired(conn: sqlite3.Connection, profile: str, retention_days: int = 180) -> int:
    """Hard-delete expired rows past retention that carry no human data.

    Rows with feedback/stage/notes are kept forever — they feed the LLM
    feedback loop and the application history. 180 days matches the comps
    benchmark window so deleting never shrinks the benchmark corpus.
    """
    result = conn.execute(
        """DELETE FROM jobs
           WHERE profile = ? AND status = 'expired'
             AND date(created_at) < date('now', ?)
             AND feedback IS NULL AND stage IS NULL AND notes IS NULL""",
        (profile, f"-{retention_days} days"),
    )
    conn.commit()
    if result.rowcount:
        logger.info("Pruned %d expired jobs older than %d days.", result.rowcount, retention_days)
    return result.rowcount


def _cleanup_db_on_init(conn: sqlite3.Connection, profile: str = PROFILE_KEY, retention_days: int = 180) -> None:
    """Run dedup and stale-job cleanup on startup."""
    try:
        removed = dedup_existing_jobs(conn, profile=profile)
        if removed:
            logger.info("Startup dedup cleaned %d duplicate rows from DB", removed)
    except Exception as e:
        logger.warning("Startup dedup failed (non-fatal): %s", e)

    try:
        expire_stale_new_jobs(conn, profile=profile)
    except Exception as e:
        logger.warning("Stale job expiry failed (non-fatal): %s", e)

    try:
        prune_old_expired(conn, profile=profile, retention_days=retention_days)
    except Exception as e:
        logger.warning("Expired-row pruning failed (non-fatal): %s", e)


def _migrate_columns(conn: sqlite3.Connection) -> None:
    """
    Add new columns to existing tables if they don't exist yet.
    Uses PRAGMA table_info to check existence before ALTER TABLE so real
    DB errors are never silently swallowed.
    Also backfills normalized_title / normalized_company for existing rows.
    """
    migrations = [
        # (table, column, definition)
        ("jobs", "llm_score", "REAL"),
        ("jobs", "llm_explanation", "TEXT"),
        ("jobs", "normalized_title", "TEXT"),
        ("jobs", "normalized_company", "TEXT"),
        ("company_insights", "glassdoor_overall", "REAL"),
        ("company_insights", "glassdoor_culture", "REAL"),
        ("company_insights", "glassdoor_wlb", "REAL"),
        ("company_insights", "glassdoor_management", "REAL"),
        ("jobs", "profile", "TEXT NOT NULL DEFAULT 'default'"),
        ("runs", "profile", "TEXT DEFAULT 'default'"),
        ("jobs", "dismissed", "INTEGER DEFAULT 0"),
        ("jobs", "salary_est_min", "REAL"),
        ("jobs", "salary_est_max", "REAL"),
        ("jobs", "applied_at", "TEXT"),
        ("jobs", "stage", "TEXT"),
        ("jobs", "notes", "TEXT"),
        ("jobs", "next_action_date", "TEXT"),
        ("jobs", "offer_json", "TEXT"),
        ("jobs", "tailored_resume_url", "TEXT"),
        ("jobs", "tailored_at", "TEXT"),
        ("jobs", "feedback", "TEXT"),
        ("runs", "source_counts", "TEXT"),
        ("jobs", "filter_score", "REAL"),
        ("jobs", "filter_score_master", "REAL"),
        ("jobs", "filter_source", "TEXT"),
        ("jobs", "filter_knockout", "INTEGER"),
        ("jobs", "filter_json", "TEXT"),
        # Tailor diff: ground-truth final text + verdicts + badge facts.
        # CHECK guards the one real malformed-JSON vector (manual sqlite3
        # edits); json.dumps itself is atomic. NULL always passes.
        ("tailor_history", "final_text", "TEXT"),
        ("tailor_history", "final_text_source", "TEXT"),
        ("tailor_history", "warnings_json",
         "TEXT CHECK (warnings_json IS NULL OR json_valid(warnings_json))"),
        ("tailor_history", "edit_verdicts_json",
         "TEXT CHECK (edit_verdicts_json IS NULL "
         "OR json_valid(edit_verdicts_json))"),
        ("jobs", "tailor_issue_json",
         "TEXT CHECK (tailor_issue_json IS NULL "
         "OR json_valid(tailor_issue_json))"),
    ]

    # Build a per-table column cache from PRAGMA so we only query once per table.
    table_columns: dict = {}
    for table, col, defn in migrations:
        if table not in table_columns:
            rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
            table_columns[table] = {row["name"] for row in rows}

        if col not in table_columns[table]:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {defn}")
            conn.commit()
            table_columns[table].add(col)
            logger.info("Migration: added column %s.%s (%s)", table, col, defn)
        else:
            logger.debug("Migration: column %s.%s already exists, skipping", table, col)

    # Backfill normalized_title / normalized_company for rows that pre-date the migration.
    needs_backfill = conn.execute(
        "SELECT COUNT(*) FROM jobs WHERE normalized_title IS NULL OR normalized_company IS NULL"
    ).fetchone()[0]

    if needs_backfill:
        rows = conn.execute(
            "SELECT id, title, company FROM jobs "
            "WHERE normalized_title IS NULL OR normalized_company IS NULL"
        ).fetchall()
        conn.executemany(
            "UPDATE jobs SET normalized_title = ?, normalized_company = ? WHERE id = ?",
            [(_norm_dedup(r["title"]), _norm_dedup(r["company"]), r["id"]) for r in rows],
        )
        conn.commit()
        logger.info(
            "Migration: backfilled normalized_title/normalized_company for %d existing rows",
            needs_backfill,
        )

    # Profile indexes (safe to run on existing DBs via IF NOT EXISTS)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_jobs_profile ON jobs(profile)")
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_jobs_profile_score ON jobs(profile, score DESC)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_jobs_profile_llm_score ON jobs(profile, llm_score)"
    )
    conn.commit()

    # --- Auto-tailor columns + one-time grandfather backfill ----------------
    # The backfill MUST stay inside the column-creation branch: it runs once,
    # in the same commit that adds the column, so a re-run of init_database
    # (every dashboard restart + every daily run) can never re-exempt freshly
    # ingested jobs. See the 2026-07-15 auto-tailor gate design (private repo notes).
    jobs_cols = {row["name"] for row in conn.execute("PRAGMA table_info(jobs)").fetchall()}
    if "auto_tailor_attempts" not in jobs_cols:
        conn.execute("ALTER TABLE jobs ADD COLUMN auto_tailor_attempts INTEGER DEFAULT 0")
        conn.commit()
        logger.info("Migration: added column jobs.auto_tailor_attempts")
    if "auto_tailor_exempt" not in jobs_cols:
        conn.execute("ALTER TABLE jobs ADD COLUMN auto_tailor_exempt INTEGER")
        # Grandfather snapshot: everything already qualifying at ship time
        # stays manual (the owner has seen those rows in the dashboard for weeks).
        # Deliberately broader than the runtime gate (no status/dismissed/
        # profile filter) and hard-coded at the shipped 60/60 floors — this is
        # a snapshot of the rule at ship time, not a config read. Recovery:
        # re-run this UPDATE by hand via sqlite3 if it ever needs widening.
        cur = conn.execute(
            f"""UPDATE jobs SET auto_tailor_exempt = 1
                WHERE tailored_at IS NULL
                  AND {EFFECTIVE_SCORE_SQL} >= 60
                  AND filter_score >= 60"""
        )
        conn.commit()  # ALTER + backfill land together (crash-safe)
        logger.info(
            "Migration: added jobs.auto_tailor_exempt; grandfathered %d rows "
            "(pre-feature qualifiers stay manual)", cur.rowcount,
        )


# ---------------------------------------------------------------------------
# Deduplication helpers
# ---------------------------------------------------------------------------

def _norm_dedup(text: str) -> str:
    """Normalize title/company for dedup: lowercase, strip punctuation, collapse whitespace."""
    text = re.sub(r"[^\w\s]", "", (text or "").lower())
    return re.sub(r"\s+", " ", text).strip()


def _job_completeness(job) -> float:
    """Score how complete a job's data is (higher = keep this one)."""
    score = 0.0
    if job.salary_min:
        score += 2
    if job.salary_max:
        score += 2
    if job.description:
        score += min(len(job.description) / 500, 3)
    return score


def dedup_jobs_in_memory(jobs: list) -> list:
    """
    Deduplicate a list of JobPosting objects by (normalized title, normalized company).
    Both title AND company must match for a job to be considered a duplicate.
    When duplicates exist, keeps the one with the most complete data.
    """
    seen: dict = {}
    dropped = 0
    for job in jobs:
        key = (_norm_dedup(job.title), _norm_dedup(job.company))
        if key not in seen:
            seen[key] = job
        else:
            existing = seen[key]
            new_score = _job_completeness(job)
            old_score = _job_completeness(existing)
            if new_score > old_score:
                logger.debug(
                    "Dedup: dropping [%s | %s | %s] (score=%.1f) in favor of [%s | %s] (score=%.1f)",
                    existing.title, existing.company, existing.source, old_score,
                    job.source, job.url, new_score,
                )
                seen[key] = job
            else:
                logger.debug(
                    "Dedup: dropping [%s | %s | %s] (score=%.1f), keeping [%s | %s] (score=%.1f)",
                    job.title, job.company, job.source, new_score,
                    existing.source, existing.url, old_score,
                )
            dropped += 1
    if dropped:
        logger.info("In-memory dedup: removed %d cross-source duplicates", dropped)
    return list(seen.values())


def dedup_existing_jobs(conn: sqlite3.Connection, profile: str = PROFILE_KEY) -> int:
    """
    Remove duplicate rows from the jobs table, keeping the most complete record
    per (normalized title, normalized company) within a profile.
    Returns number of rows deleted.
    """
    rows = conn.execute(
        "SELECT id, title, company, salary_min, salary_max, description, "
        "normalized_title, normalized_company FROM jobs WHERE profile = ?",
        (profile,),
    ).fetchall()

    groups: dict = {}
    for row in rows:
        # Prefer stored normalized columns; fall back to computing on the fly
        # for rows that pre-date the migration (shouldn't happen after backfill).
        norm_title = row["normalized_title"] or _norm_dedup(row["title"])
        norm_company = row["normalized_company"] or _norm_dedup(row["company"])
        key = (norm_title, norm_company)
        groups.setdefault(key, []).append(dict(row))

    to_delete = []
    for jobs_group in groups.values():
        if len(jobs_group) <= 1:
            continue
        jobs_group.sort(
            key=lambda j: (
                bool(j.get("salary_min")) * 2
                + bool(j.get("salary_max")) * 2
                + min(len(j.get("description") or "") / 500, 3)
            ),
            reverse=True,
        )
        for j in jobs_group[1:]:
            to_delete.append(j["id"])

    if to_delete:
        conn.executemany("DELETE FROM jobs WHERE id = ?", [(i,) for i in to_delete])
        conn.commit()
        logger.info("DB dedup: removed %d duplicate job rows", len(to_delete))

    return len(to_delete)


# ---------------------------------------------------------------------------
# Streaming save (scoring-behind-scrape overlap)
#
# Batches are saved as each scraper finishes so the LLM scoring consumer can
# run DURING the scrape. Same-run collisions must resolve to EXACTLY the
# state the batch flow (dedup_jobs_in_memory + one save_jobs call) produces:
#   - within a batch        -> dedup_jobs_in_memory verbatim;
#   - same URL twice        -> the copy from the earlier fixed-merge-order
#                              source (rank) wins, as run_scrapers' seen_urls
#                              merge did;
#   - same title+company    -> completeness-wins with fixed-order tiebreak
#                              (the old global scan kept the incumbent on a
#                              tie, and its incumbent order WAS merge order),
#                              and the winner's WHOLE identity is what gets
#                              stored — a later winning copy swaps the row in
#                              place and re-NULLs any scoring computed on the
#                              loser;
#   - copy touching a PRIOR-run row (same key or same url) -> DEFERRED: no
#                              write until finalize_streaming_save applies
#                              the final winner once through save_jobs. In
#                              the old flow only the global winner ever
#                              reached save_jobs, so mid-scrape fills from
#                              non-final copies must not exist; deferring
#                              costs no scoring overlap since those rows were
#                              scored in prior runs.
# ---------------------------------------------------------------------------

class StreamingSaveState:
    """Cross-batch bookkeeping for reconcile_save_batch."""

    def __init__(self):
        # (norm_title, norm_company) -> {"job", "row_id" (None = not inserted
        # this run: deferred prior-row contact, or a save that errored — both
        # re-applied through save_jobs at finalize), "rank"}
        self.key_winners: dict = {}
        self.url_owner: dict = {}   # url -> winner key
        self.dropped = 0            # title+company drops (dedup log parity)
        self.stats = {"total_jobs": 0, "total_new": 0, "total_above_threshold": 0}


_SWAP_CONTENT_SQL = """
    UPDATE jobs SET
        title = ?, company = ?, location = ?, url = ?, description = ?,
        salary_min = ?, salary_max = ?, date_posted = ?, source = ?,
        score = ?, match_explanation = ?,
        normalized_title = ?, normalized_company = ?,
        llm_score = NULL, llm_explanation = NULL,
        salary_est_min = NULL, salary_est_max = NULL,
        filter_score = NULL, filter_score_master = NULL, filter_source = NULL,
        filter_knockout = NULL, filter_json = NULL
    WHERE id = ?
"""


def _score_job_with_layoff(job, scorer, layoff_companies: set,
                           layoff_penalty: int) -> None:
    """Keyword-score a job and apply the layoff caution — THE single home
    for this step, used by save_jobs and the streaming swap so a scored job
    can never differ by which path stored it."""
    score, explanation = scorer.score_and_explain(job)
    penalized = apply_layoff_penalty(score, job.company, layoff_companies,
                                     layoff_penalty)
    if penalized != score:
        explanation = f"{explanation} [Caution: recent layoffs reported at {job.company}.]"
        score = penalized
    job.score = score
    job.match_explanation = explanation


def _swap_row_to_copy(conn, row_id: int, job, scorer,
                      layoff_companies: set, layoff_penalty: int) -> None:
    """Replace a this-run row's identity/content with a winning copy and
    re-NULL scoring outputs — anything the consumer already computed was
    based on the loser's inputs. Never called for prior-run rows."""
    _score_job_with_layoff(job, scorer, layoff_companies, layoff_penalty)
    conn.execute(_SWAP_CONTENT_SQL, (
        job.title, job.company, job.location, job.url, job.description,
        job.salary_min, job.salary_max, job.date_posted, job.source,
        job.score, job.match_explanation,
        _norm_dedup(job.title), _norm_dedup(job.company), row_id,
    ))
    conn.commit()


def reconcile_save_batch(conn, jobs: list, source_rank: int,
                         state: StreamingSaveState, run_id: int, scorer,
                         profile_name: str = PROFILE_KEY) -> None:
    """Save one scraper's batch, reconciling against this run's earlier
    batches so the final table matches the old batch flow byte for byte.
    finalize_streaming_save MUST run after the last batch."""
    layoff_penalty = int(scorer.config.get("scoring", {}).get("layoff_penalty", 0) or 0)
    layoff_companies = load_layoff_companies(conn) if layoff_penalty else set()

    # Within-batch duplicates resolve exactly as the old flow resolved them:
    # this is the same function it used, scoped to the batch.
    jobs = dedup_jobs_in_memory(jobs)

    survivors: list = []
    survivor_keys: list = []
    for job in jobs:
        if not job.url:
            continue
        key = (_norm_dedup(job.title), _norm_dedup(job.company))

        # URL contest first — run_scrapers' seen_urls check preceded the
        # in-memory dedup, so same-URL pairs never reached completeness.
        owner_key = state.url_owner.get(job.url)
        if owner_key is not None:
            owner = state.key_winners.get(owner_key)
            if owner is None or source_rank >= owner["rank"]:
                continue  # today's fixed order already had a better claim
            if owner_key != key and key in state.key_winners:
                logger.debug("Streaming save: degenerate url+key conflict for "
                             "%s — keeping incumbents", job.url)
                continue
            if owner["row_id"] is not None:
                _swap_row_to_copy(conn, owner["row_id"], job, scorer,
                                  layoff_companies, layoff_penalty)
            state.key_winners.pop(owner_key, None)
            state.key_winners[key] = {"job": job, "row_id": owner["row_id"],
                                      "rank": source_rank}
            state.url_owner[job.url] = key
            continue

        winner = state.key_winners.get(key)
        if winner is not None:
            new_c, old_c = _job_completeness(job), _job_completeness(winner["job"])
            # The old global scan kept the incumbent on ties — and its
            # incumbency order was the FIXED merge order, not arrival order.
            beats = new_c > old_c or (new_c == old_c and source_rank < winner["rank"])
            if beats:
                if winner["row_id"] is not None:
                    try:
                        _swap_row_to_copy(conn, winner["row_id"], job, scorer,
                                          layoff_companies, layoff_penalty)
                    except sqlite3.IntegrityError:
                        # The winning copy's URL exists as a PRIOR-run row
                        # (retitled posting). Today the loser row would never
                        # have been created: drop it and defer the winner to
                        # finalize, which applies it once through save_jobs.
                        logger.info(
                            "Streaming save: url %s exists from a prior run — "
                            "deferring the winner and dropping this run's "
                            "loser row", job.url)
                        conn.execute("DELETE FROM jobs WHERE id = ?",
                                     (winner["row_id"],))
                        conn.commit()
                        state.stats["total_new"] -= 1
                        winner["row_id"] = None
                state.url_owner.pop(winner["job"].url, None)
                state.key_winners[key] = {"job": job,
                                          "row_id": winner["row_id"],
                                          "rank": source_rank}
                state.url_owner[job.url] = key
            state.dropped += 1
            continue

        # Unseen key: a copy touching any PRIOR-run row (same key or same
        # url) is DEFERRED — only the final winner may produce side effects
        # on existing rows, exactly like the old winner-only save.
        prior_contact = conn.execute(
            "SELECT 1 FROM jobs WHERE profile = ? AND "
            "(url = ? OR (normalized_title = ? AND normalized_company = ?)) "
            "LIMIT 1",
            (profile_name, job.url, key[0], key[1])).fetchone()
        state.url_owner[job.url] = key
        if prior_contact:
            state.key_winners[key] = {"job": job, "row_id": None,
                                      "rank": source_rank}
            continue
        survivors.append(job)
        survivor_keys.append(key)

    state.stats["total_jobs"] += len(jobs)
    if survivors:
        batch_stats = save_jobs(conn, survivors, run_id, scorer,
                                profile_name=profile_name)
        state.stats["total_new"] += batch_stats["total_new"]
        state.stats["total_above_threshold"] += batch_stats["total_above_threshold"]
        placeholders = ",".join("?" * len(survivors))
        id_by_url = dict(conn.execute(
            f"SELECT url, id FROM jobs WHERE url IN ({placeholders}) "
            "AND run_id = ?",
            [j.url for j in survivors] + [run_id]).fetchall())
        for job, key in zip(survivors, survivor_keys):
            state.key_winners[key] = {"job": job,
                                      "row_id": id_by_url.get(job.url),
                                      "rank": source_rank}


def finalize_streaming_save(conn, state: StreamingSaveState, run_id: int,
                            scorer, profile_name: str = PROFILE_KEY) -> None:
    """Apply every deferred winner once through save_jobs — the moment the
    scrape is complete and each key's contest is decided. Runs BEFORE the
    scoring consumer is told the run is drained, so anything save_jobs
    inserts here still gets streamed scoring."""
    deferred = [(key, w) for key, w in state.key_winners.items()
                if w["row_id"] is None]
    if not deferred:
        return
    logger.info("Streaming save: applying %d deferred prior-row winners.",
                len(deferred))
    for key, winner in deferred:
        batch_stats = save_jobs(conn, [winner["job"]], run_id, scorer,
                                profile_name=profile_name)
        state.stats["total_new"] += batch_stats["total_new"]
        state.stats["total_above_threshold"] += batch_stats["total_above_threshold"]
        row = conn.execute(
            "SELECT id FROM jobs WHERE url = ? AND run_id = ?",
            (winner["job"].url, run_id)).fetchone()
        if row:
            winner["row_id"] = row["id"]


# ---------------------------------------------------------------------------
# Job saving
# ---------------------------------------------------------------------------

def save_jobs(
    conn: sqlite3.Connection,
    jobs: list,
    run_id: int,
    scorer,
    profile_name: str = PROFILE_KEY,
) -> dict:
    """
    Score each job, save to DB (INSERT OR IGNORE on url), return stats dict.
    """
    from scrapers.base import JobPosting

    total_new = 0
    total_above_threshold = 0
    threshold = scorer.config.get("scoring", {}).get("alert_threshold", 60)

    # Layoff caution: small config-gated penalty for companies flagged by
    # company intel. Shared with reblend_all via engine/scorer.py so a
    # --reblend can never silently drop the penalty.
    layoff_penalty = int(scorer.config.get("scoring", {}).get("layoff_penalty", 0) or 0)
    layoff_companies = load_layoff_companies(conn) if layoff_penalty else set()

    for job in jobs:
        try:
            _score_job_with_layoff(job, scorer, layoff_companies, layoff_penalty)
            score = job.score

            norm_title = _norm_dedup(job.title)
            norm_company = _norm_dedup(job.company)

            # Cross-source dedup: skip if same normalized title+company already exists
            # within this profile, unless the new record has richer data.
            existing = conn.execute(
                "SELECT id, salary_min, salary_max, description FROM jobs "
                "WHERE normalized_title = ? AND normalized_company = ? AND profile = ? LIMIT 1",
                (norm_title, norm_company, profile_name),
            ).fetchone()
            if existing:
                if not existing["salary_min"] and not existing["salary_max"]:
                    if job.salary_min or job.salary_max:
                        # Update the existing row with salary data from this richer source
                        conn.execute(
                            "UPDATE jobs SET salary_min=?, salary_max=?, source=? WHERE id=?",
                            (job.salary_min, job.salary_max, job.source, existing["id"]),
                        )
                        conn.commit()
                # Detail fetches are best-effort, so a row stored during a
                # transient failure (or before the detail-fetch fixes) may be
                # blank — repair it from this scrape. Never overwrites text.
                if (job.description or "").strip() and not (existing["description"] or "").strip():
                    conn.execute(
                        "UPDATE jobs SET description = ? WHERE id = ?",
                        (job.description, existing["id"]),
                    )
                    conn.commit()
                continue  # duplicate — skip INSERT

            cursor = conn.execute(
                """
                INSERT OR IGNORE INTO jobs
                    (title, company, location, url, description,
                     salary_min, salary_max, date_posted, source,
                     score, status, match_explanation, run_id,
                     normalized_title, normalized_company, profile)
                VALUES
                    (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job.title,
                    job.company,
                    job.location,
                    job.url,
                    job.description,
                    job.salary_min,
                    job.salary_max,
                    job.date_posted,
                    job.source,
                    job.score,
                    job.status,
                    job.match_explanation,
                    run_id,
                    norm_title,
                    norm_company,
                    profile_name,
                ),
            )

            if cursor.rowcount > 0:
                total_new += 1

                if score >= threshold:
                    total_above_threshold += 1
            else:
                # INSERT OR IGNORE skipped this URL — check if the new version is better
                dup = conn.execute(
                    "SELECT id, score, llm_score, salary_min, salary_max, description FROM jobs WHERE url = ?",
                    (job.url,),
                ).fetchone()
                if dup:
                    # Once llm_score is set, jobs.score is a 0.4/0.6 blend — a
                    # fresh keyword-only score is not comparable to it and must
                    # never overwrite it (it would re-inflate keyword-heavy junk
                    # the LLM buried). Keyword-vs-keyword comparison only;
                    # rule changes reach blended rows via --reblend.
                    score_improved = (
                        dup["llm_score"] is None
                        and job.score is not None
                        and (dup["score"] is None or job.score > dup["score"])
                    )
                    salary_improved = (job.salary_min or job.salary_max) and (
                        not dup["salary_min"] and not dup["salary_max"]
                    )
                    description_improved = bool((job.description or "").strip()) and not (
                        (dup["description"] or "").strip()
                    )
                    if score_improved or salary_improved or description_improved:
                        # COALESCE guards: match_explanation only changes when
                        # the score did, and description only fills a blank —
                        # a salary- or description-only update must not null
                        # or overwrite the others.
                        conn.execute(
                            """UPDATE jobs
                               SET score=?, match_explanation=COALESCE(?, match_explanation),
                                   salary_min=COALESCE(?, salary_min),
                                   salary_max=COALESCE(?, salary_max),
                                   description=COALESCE(NULLIF(?, ''), description)
                               WHERE id=?""",
                            (
                                job.score if score_improved else dup["score"],
                                job.match_explanation if score_improved else None,
                                job.salary_min,
                                job.salary_max,
                                job.description if description_improved else "",
                                dup["id"],
                            ),
                        )
                        logger.info(
                            "Updated existing job '%s' at '%s': score %s->%s, salary updated=%s, description filled=%s",
                            job.title,
                            job.company,
                            dup["score"],
                            job.score if score_improved else dup["score"],
                            bool(salary_improved),
                            description_improved,
                        )
                        if score_improved and (job.score or 0) >= threshold and (dup["score"] or 0) < threshold:
                            total_above_threshold += 1

        except Exception as e:
            logger.error("Error saving job '%s' at '%s': %s", job.title, job.url, e)
            continue

    conn.commit()

    stats = {
        "total_jobs": len(jobs),
        "total_new": total_new,
        "total_above_threshold": total_above_threshold,
    }
    logger.info(
        "Saved %d new jobs (%d above threshold) out of %d total.",
        total_new,
        total_above_threshold,
        len(jobs),
    )
    return stats


def count_effective_above_threshold(conn: sqlite3.Connection, run_id: int,
                                    threshold: int) -> int:
    """Blended, knockout-gated count of a run's rows at/above threshold.

    save_jobs counts keyword-only scores (the LLM pass hasn't run yet); this
    post-blend recount is the truth the runs table and report should carry.
    """
    return conn.execute(
        f"SELECT COUNT(*) FROM jobs WHERE run_id = ? AND {EFFECTIVE_SCORE_SQL} >= ?",
        (run_id, threshold),
    ).fetchone()[0]


# ---------------------------------------------------------------------------
# Scraper orchestration
# ---------------------------------------------------------------------------

def _urls_with_descriptions(conn: sqlite3.Connection) -> set:
    """URLs already stored with a non-empty description.

    Seeded into config under "_urls_with_descriptions" before run_scrapers so
    detail-fetching scrapers skip re-downloading pages we already have.
    """
    try:
        rows = conn.execute(
            "SELECT url FROM jobs WHERE TRIM(COALESCE(description, '')) != ''"
        ).fetchall()
        return {row["url"] for row in rows}
    except sqlite3.OperationalError:
        return set()


def summarize_scraper_run(counts: dict) -> dict:
    """Summarize per-scraper unique-job counts for run-level visibility.

    ``counts`` maps scraper name -> number of unique jobs it contributed.
    Returns:
      - ``total``: sum of all counts
      - ``empty``: names of scrapers that returned 0 (insertion order preserved)
      - ``all_empty``: True only if scrapers ran but every one returned 0
    """
    total = sum(counts.values())
    empty = [name for name, count in counts.items() if count == 0]
    return {"total": total, "empty": empty, "all_empty": bool(counts) and total == 0}


def detect_source_regressions(
    conn: sqlite3.Connection,
    counts: dict,
    current_run_id: int,
    window: int = 7,
    floor: float = 0.4,
) -> list:
    """Compare this run's per-source yields against the trailing-run average.

    Returns human-readable warnings for sources that usually produce >=5 jobs
    but fell below `floor` (default 40%) of their trailing average.
    """
    import json as _json

    rows = conn.execute(
        "SELECT source_counts FROM runs "
        "WHERE status = 'completed' AND id != ? AND source_counts IS NOT NULL "
        "ORDER BY id DESC LIMIT ?",
        (current_run_id, window),
    ).fetchall()
    history: dict = {}
    for row in rows:
        try:
            for src, n in (_json.loads(row["source_counts"]) or {}).items():
                history.setdefault(src, []).append(n)
        except (ValueError, TypeError):
            continue

    warnings = []
    for src, values in history.items():
        avg = sum(values) / len(values)
        current = counts.get(src, 0)
        if avg >= 5 and current < floor * avg:
            warnings.append(
                f"{src} yield dropped: {current} this run vs trailing avg {avg:.1f} "
                f"(last {len(values)} runs) — check for board changes or blocking."
            )
    return warnings


def apply_scrape_only(args) -> None:
    """Resolve --scrape-only: scrape + keyword-score + save, nothing else.

    Implemented by forcing the two skip flags so the normal-run path
    bypasses LLM scoring and company enrichment. Regex salary extraction,
    the run record, and the JSON report still happen.
    """
    if args.scrape_only:
        args.skip_llm = True
        args.skip_company_intel = True
        args.skip_auto_tailor = True


# Early-return mode flags: each dispatches its own branch in main() and exits.
# attr name -> CLI flag, plus a truthiness test (store_true flags vs valued ones).
_MODE_FLAGS = {
    "dismiss_job": "--dismiss-job",
    "rescore_sample": "--rescore-sample",
    "rescore_all": "--rescore-all",
    "rescore_force": "--rescore-force",
    "reblend": "--reblend",
    "rescore_run": "--rescore-run",
    "backfill_filter": "--backfill-filter",
    "rejudge_filter": "--rejudge-filter",
    "rejudge_filter_all": "--rejudge-filter-all",
    "enrich_companies": "--enrich-companies",
    "dashboard": "--dashboard",
}

# Modes where --dry-run is meaningful. Everywhere else it would be silently
# ignored while the mode does real (billed) work — error instead.
_DRY_RUN_MODES = {"reblend", "rescore_force"}


def _flag_error(msg: str) -> None:
    print(f"main.py: error: {msg}", file=sys.stderr)
    raise SystemExit(2)


def validate_mode_flags(args) -> None:
    """Reject conflicting or orphaned flag combinations before any work runs.

    main() dispatches mode flags as sequential ifs, so without this guard a
    combo like `--reblend --rescore-force` would silently run whichever branch
    comes first — and `--rescore-all --dry-run` would do a real, billed scoring
    pass while the user believes nothing is written.
    """
    active = [attr for attr in _MODE_FLAGS
              if getattr(args, attr) not in (False, None)]

    if len(active) > 1:
        flags = ", ".join(_MODE_FLAGS[a] for a in active)
        _flag_error(f"conflicting mode flags: {flags} — pick exactly one")

    mode = active[0] if active else None

    if args.filter_since_hours is not None and mode not in (
            "backfill_filter", "rejudge_filter", "rejudge_filter_all"):
        _flag_error("--filter-since-hours requires --backfill-filter, "
                    "--rejudge-filter, or --rejudge-filter-all")

    if args.dry_run and mode is not None and mode not in _DRY_RUN_MODES:
        _flag_error(f"--dry-run is not supported with {_MODE_FLAGS[mode]} "
                    "(it would be ignored and the mode would run for real)")


def confirm_rescore_force(count: int, profile: str) -> bool:
    """Interactive gate before the destructive full re-score.

    --rescore-force clears every llm_score for the profile and re-bills the
    whole corpus through the claude CLI, so it must never run by accident:
    non-interactive stdin (cron/pipes) is refused outright, and an interactive
    caller has to type 'yes'.
    """
    if not sys.stdin.isatty():
        logger.error(
            "--rescore-force requires interactive confirmation (it clears all "
            "%d LLM scores for profile '%s' and re-bills the entire corpus). "
            "Run it from a terminal, or use --dry-run to preview.",
            count, profile)
        return False
    try:
        reply = input(
            f"About to CLEAR llm_score for {count} jobs (profile={profile}) and "
            f"re-score them ALL via the claude CLI — this re-bills the entire "
            f"corpus. Type 'yes' to proceed: ")
    except EOFError:
        return False
    return reply.strip().lower() == "yes"


def format_sources(counts: dict) -> str:
    """Comma-joined names of scrapers that contributed >=1 job this run.

    Stored in the runs.sources column; insertion order preserved.
    """
    return ",".join(name.lower() for name, count in counts.items() if count > 0)


def select_auto_tailor_candidates(conn, profile: str, min_score: int,
                                  min_filter_score: float, limit: int):
    """Live, untailored, undismissed 'new' jobs passing BOTH dashboard numbers:
    the effective (knockout-gated) blended score AND the Filter Match judge
    score. filter_score IS NULL (unjudged, or the '{}' filter_json sentinel)
    fails `>= min_filter_score` under SQL three-valued logic — intentionally
    ineligible. auto_tailor_exempt marks the pre-feature grandfather snapshot;
    auto_tailor_attempts >= 2 drops jobs whose auto-tailor keeps failing.
    Manual tailoring from the dashboard still works for all of them."""
    return conn.execute(
        f"""SELECT id, url, title, company FROM jobs
           WHERE profile = ? AND status = 'new'
             AND (dismissed IS NULL OR dismissed = 0)
             AND tailored_at IS NULL
             AND COALESCE(auto_tailor_exempt, 0) = 0
             AND COALESCE(auto_tailor_attempts, 0) < 2
             AND {EFFECTIVE_SCORE_SQL} >= ?
             AND filter_score >= ?
           ORDER BY {EFFECTIVE_SCORE_SQL} DESC
           LIMIT ?""",
        (profile, min_score, min_filter_score, limit),
    ).fetchall()


def _bump_auto_tailor_attempts(conn, job_id: int) -> None:
    """Record a failed auto-tailor attempt; at 2 the gate drops the job
    (auto path only — the dashboard's manual ✂ button still works)."""
    conn.execute(
        "UPDATE jobs SET auto_tailor_attempts = COALESCE(auto_tailor_attempts, 0) + 1 "
        "WHERE id = ?", (job_id,))
    conn.commit()


def _log_auto_tailor_abort(exc: Exception, errors_log: Optional[str],
                           job: Optional[dict] = None) -> None:
    """Batch aborts must not be silent: mirror them into errors.log (the same
    channel run failures use) so a halted queue is visible off the run log.
    job=None means the batch aborted before any job started (pre-flight) —
    naming a candidate there would send triage after a poison pill that
    doesn't exist."""
    if job is None:
        logger.warning("Auto-tailor: batch aborted before start: %s", exc)
    else:
        logger.warning("Auto-tailor: batch aborted at job %d ('%s' @ %s): %s",
                       job["id"], job["title"], job["company"], exc)
    if errors_log:
        import traceback
        from alerts.error_monitor import log_error_to_file
        log_error_to_file(exc, traceback.format_exc(), errors_log)


def _classify_tailor_failure(exc: Exception, _gauth, _GHttpError) -> str:
    """Map a pipeline exception onto the 2026-07-15 failure taxonomy — the
    single source of truth for BOTH the sequential and parallel loops:
      'abort_bump' — ClaudeCLIError: conflates systemic causes with
                     payload-induced ones, so bump the active job (a poison
                     pill self-exempts after 2 runs) AND abort the batch;
      'abort'      — Google auth (missing creds, refresh/transport, HTTP
                     401/403): systemic until re-auth; never burn budget;
      'bump'       — job-specific; bump and continue with other candidates.
    """
    if isinstance(exc, ClaudeCLIError):
        return "abort_bump"
    if isinstance(exc, (FileNotFoundError, _gauth.TransportError, _gauth.RefreshError)):
        return "abort"
    if isinstance(exc, _GHttpError):
        status = getattr(getattr(exc, "resp", None), "status", None)
        if status in (401, 403):
            return "abort"
    return "bump"


def run_auto_tailor(conn, db_path: str, profile: str, at_cfg: dict,
                    errors_log: Optional[str] = None) -> int:
    """Tailor the resume for the top matches, bounded by at_cfg.

    Each pipeline run costs Opus usage via the claude CLI, so max_per_run
    stays small. Failure taxonomy (2026-07-15 spec):
      - job-specific errors: bump auto_tailor_attempts, continue;
      - ClaudeCLIError: bump the ACTIVE job AND abort — the error conflates
        systemic causes (CLI down/logged out) with payload-induced ones (RSS
        guard, non-JSON output), so the bump lets a poison-pill job
        self-exempt after 2 runs while the abort protects the batch;
      - Google auth/transport errors: abort WITHOUT bumping — they fail every
        candidate identically until re-auth and must not burn attempt budget.
    Batch aborts also write to errors.log via alerts.error_monitor."""
    if not settings_store.auto_tailor_enabled(conn, at_cfg):
        if settings_store.get_setting(conn, settings_store.AUTO_TAILOR_KEY) == "0":
            logger.info("Auto-tailor: OFF via dashboard toggle — skipping.")
        else:
            logger.info("Auto-tailor: disabled in config (auto_tailor.enabled) — skipping.")
        return 0

    try:
        import resume_tailor.pipeline  # noqa: F401 — availability check
    except ImportError as exc:
        logger.info("Auto-tailor: resume_tailor unavailable (%s) — skipping.", exc)
        return 0
    import resume_tailor.pipeline as _pipe
    # resume_tailor imported fine, so the google libs it depends on exist.
    import google.auth.exceptions as _gauth
    from googleapiclient.errors import HttpError as _GHttpError

    min_score = int(at_cfg.get("min_score", 60))
    min_filter = float(at_cfg.get("min_filter_score", 60))
    candidates = select_auto_tailor_candidates(
        conn, profile, min_score, min_filter, int(at_cfg.get("max_per_run", 2)))
    if not candidates:
        logger.info(
            "Auto-tailor: no eligible jobs (score >= %s AND filter >= %s, "
            "untailored, non-exempt, attempts < 2).", min_score, min_filter)
        return 0

    workers = min(max(1, int(at_cfg.get("workers", 1))), CLAUDE_CLI_MAX_CONCURRENCY)
    if workers > 1 and len(candidates) > 1:
        return _run_auto_tailor_parallel(
            conn, db_path, candidates, workers, _pipe, _gauth, _GHttpError,
            errors_log)

    done = 0
    for row in candidates:
        job = dict(row)
        try:
            result = _tailor_one_job(_pipe, job, db_path)
            logger.info("Auto-tailor: done — job %d, ATS %s, doc %s", job["id"],
                        result.get("ats_score"), result.get("google_doc_url"))
            done += 1
        except Exception as exc:
            action = _classify_tailor_failure(exc, _gauth, _GHttpError)
            if action == "abort_bump":
                _bump_auto_tailor_attempts(conn, job["id"])
                _log_auto_tailor_abort(exc, errors_log, job)
                break
            if action == "abort":
                _log_auto_tailor_abort(exc, errors_log, job)
                break
            _bump_auto_tailor_attempts(conn, job["id"])
            logger.warning("Auto-tailor: failed for job %d: %s", job["id"], exc)
    return done


def _tailor_one_job(_pipe, job: dict, db_path: str,
                    abort_event: Optional["threading.Event"] = None) -> Optional[dict]:
    """One pipeline run on a worker thread. Progress lines carry the job id —
    concurrent jobs interleave in the log and are ambiguous without it.

    Returns None without running anything when abort_event is set: a freed
    worker dequeues the next work item before the main thread's
    future.cancel() can land (the dequeue wins ~98% of the time with
    staggered failures), so the worker itself must gate on the shared flag
    or a batch abort would still burn a full pipeline of Opus time."""
    if abort_event is not None and abort_event.is_set():
        return None
    logger.info("Auto-tailor: tailoring '%s' @ %s (job %d, this uses Opus)...",
                job["title"], job["company"], job["id"])
    return _pipe.run_tailor_pipeline(
        {"id": job["id"], "url": job["url"],
         "company": job["company"], "title": job["title"]},
        db_path=db_path,
        progress=lambda m, _jid=job["id"]: logger.info(
            "Auto-tailor[job %d]: %s", _jid, m),
    )


def _run_auto_tailor_parallel(conn, db_path: str, candidates, workers: int,
                              _pipe, _gauth, _GHttpError,
                              errors_log: Optional[str]) -> int:
    """Parallel variant of the run_auto_tailor loop (workers >= 2).

    Each candidate's pipeline is independent (own Google Doc, own claude CLI
    calls; token accounting is threading.local and the pipeline opens its own
    sqlite connections), so overlapping them cuts the phase from ~7.5 min per
    job sequential to roughly the slowest job per worker-slot. Same failure
    taxonomy as the sequential loop (via _classify_tailor_failure), adapted
    for concurrency:
      - systemic failures cancel not-yet-started candidates; in-flight ones
        finish and their successes still count;
      - after an abort, NO further failure bumps attempts — failures landing
        during a systemic event share its cause, and burning budget on them
        would let one outage self-exempt several jobs at once;
      - Google auth runs once up front, non-interactively, on the main
        thread: a dead token fails fast instead of blocking the unattended
        run in a browser flow, and workers then take the no-write auth path
        (the token write itself is atomic in google_api for the mid-batch
        expiry case).
    All attempt bumps run on the main thread (conn is not thread-safe).
    """
    from concurrent.futures import as_completed
    from resume_tailor.google_api import GoogleAPIClient

    try:
        GoogleAPIClient().authenticate(allow_interactive=False)
    except Exception as exc:
        # Pre-flight failure is systemic by construction: no pipeline ran,
        # so no job's attempt budget is burned and no job is named.
        _log_auto_tailor_abort(exc, errors_log)
        return 0

    done = 0
    aborted = False
    abort_event = threading.Event()
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="tailor") as executor:
        jobs = [dict(row) for row in candidates]
        futures = {
            executor.submit(_tailor_one_job, _pipe, job, db_path, abort_event): job
            for job in jobs
        }

        def _abort_batch(exc: Exception, job: dict, bump: bool) -> None:
            nonlocal aborted
            aborted = True
            abort_event.set()
            for f in futures:
                f.cancel()
            if bump:
                _bump_auto_tailor_attempts(conn, job["id"])
            _log_auto_tailor_abort(exc, errors_log, job)

        try:
            for future in as_completed(futures):
                job = futures[future]
                if future.cancelled():
                    logger.info("Auto-tailor: job %d cancelled after batch abort.",
                                job["id"])
                    continue
                try:
                    result = future.result()
                except Exception as exc:
                    if aborted:
                        logger.warning(
                            "Auto-tailor: job %d also failed during the batch "
                            "abort (no attempt burned): %s", job["id"], exc)
                        continue
                    action = _classify_tailor_failure(exc, _gauth, _GHttpError)
                    if action == "abort_bump":
                        _abort_batch(exc, job, bump=True)
                    elif action == "abort":
                        _abort_batch(exc, job, bump=False)
                    else:
                        _bump_auto_tailor_attempts(conn, job["id"])
                        logger.warning("Auto-tailor: failed for job %d: %s",
                                       job["id"], exc)
                    continue
                if result is None:  # worker skipped via the abort gate
                    logger.info("Auto-tailor: job %d skipped after batch abort.",
                                job["id"])
                    continue
                logger.info("Auto-tailor: done — job %d, ATS %s, doc %s", job["id"],
                            result.get("ats_score"), result.get("google_doc_url"))
                done += 1
        except BaseException:
            # Ctrl-C or a main-thread failure (e.g. sqlite error in a bump):
            # without this hatch, executor.__exit__ (shutdown(wait=True),
            # no cancel) would run every queued pipeline to completion —
            # ~7.5 min of Opus each — before the exception surfaced.
            abort_event.set()
            for f in futures:
                f.cancel()
            raise
    return done


def run_overlap_scrape_and_score(conn, config: dict, args, run_id: int,
                                 scorer, llm_scorer, score_cap) -> tuple:
    """Scrape with streaming saves while a scoring consumer runs — the
    scoring-behind-scrape overlap. Returns (jobs, source_counts, save_stats,
    streamed_scored, streamed_attempts).

    Failure discipline: if the scrape (or a deferred save error) raises, the
    consumer is STOPPED — not drained — and joined before the exception
    propagates, so a dead run cannot keep spending CLI budget behind a
    failed run record (the old sequential flow spent nothing after a save
    crash). If the consumer itself crashes mid-phase, the true scored count
    is recovered from the DB so the top-up cannot double-spend the cap."""
    reserve = (int(config.get("llm_scoring", {}).get("streaming_reserve", 50) or 0)
               if score_cap else 0)
    drained = threading.Event()
    stop = threading.Event()
    state = StreamingSaveState()
    out = {"scored": None, "attempts": None}

    def _consume() -> None:
        try:
            out["scored"], out["attempts"] = llm_scorer.apply_llm_scores_streaming(
                run_id=run_id, drained=drained, stop=stop, profile=args.profile,
                limit=score_cap, reserve=reserve)
        except Exception:
            logger.error(
                "Streaming scoring consumer failed — the post-scrape top-up "
                "will score this run instead.", exc_info=True)

    consumer = threading.Thread(target=_consume, name="scoring-consumer")
    consumer.start()

    def _on_batch(name: str, batch: list, rank: int) -> None:
        reconcile_save_batch(conn, batch, source_rank=rank, state=state,
                             run_id=run_id, scorer=scorer,
                             profile_name=args.profile)

    try:
        jobs, source_counts = run_scrapers(
            config, known_urls=load_known_linkedin_urls(conn),
            on_batch=_on_batch)
        # Deferred prior-row winners must land BEFORE the drain flag so any
        # row this inserts still gets streamed scoring.
        finalize_streaming_save(conn, state, run_id, scorer, args.profile)
    except BaseException:
        stop.set()
        drained.set()
        consumer.join()
        raise
    drained.set()
    consumer.join()

    scored, attempts = out["scored"], out["attempts"]
    if scored is None:
        row = conn.execute(
            "SELECT COUNT(*) FROM jobs WHERE run_id = ? AND llm_score IS NOT NULL",
            (run_id,)).fetchone()
        scored = attempts = int(row[0])
        logger.warning(
            "Recovered streaming count from the DB after a consumer "
            "failure: %d rows.", scored)

    state.stats["total_jobs"] = len(jobs)
    return jobs, source_counts, state.stats, scored, attempts


def start_company_enrichment(db_path: str, jobs: list) -> Optional[threading.Thread]:
    """Kick off company-intel enrichment on a background thread.

    Enrichment has no in-run consumer until the end-of-run digest: it reads
    only company names from this scrape and writes intel rows on its own
    sqlite connections (no claude CLI calls, so no contention with scoring
    workers), and save-time layoff penalties always come from PRIOR runs'
    intel. Running it across the scoring + tailor phases hides its ~7 min.
    Returns the started (non-daemon) thread, or None when there is nothing
    to enrich. The caller MUST join() before writing the run record so the
    digest and next run see completed intel. Exceptions are contained and
    logged — with the phase off the main thread, a flaky intel source must
    not be able to take down scoring/tailoring."""
    unique_companies = list({j.company for j in jobs if j.company})
    if not unique_companies:
        return None

    logger.info(
        "Enriching %d unique companies with intelligence data (background)...",
        len(unique_companies),
    )

    def _run() -> None:
        try:
            from engine.company_intel import CompanyIntelligence
            intel = CompanyIntelligence(db_path)
            intel.batch_enrich(unique_companies, delay=1.5)
            logger.info("Company enrichment complete.")
        except Exception:
            logger.error("Company enrichment failed (non-fatal).", exc_info=True)

    thread = threading.Thread(target=_run, name="enrich")
    thread.start()
    return thread


def load_known_linkedin_urls(conn: sqlite3.Connection) -> set:
    """Canonical LinkedIn URLs already stored — lets the scraper skip the
    per-card detail fetch for jobs we already have. Returns an empty set on
    any failure: the preload must never block a run."""
    try:
        rows = conn.execute(
            "SELECT url FROM jobs WHERE source='linkedin'"
        ).fetchall()
        return {r[0] for r in rows}
    except sqlite3.Error:
        logger.warning(
            "known-URL preload failed; LinkedIn will scrape without a dedup set",
            exc_info=True,
        )
        return set()


def _linkedin_company_queries(config: dict, keyword_queries: List[str]) -> List[str]:
    """Queries for LinkedIn company-targeted searches: the explicit pinned
    list, so growing search_queries can't silently multiply company-search
    volume; falls back to the keyword queries when unset."""
    return config.get("linkedin_company_search_queries") or list(keyword_queries)


def run_scrapers(config: dict, dry_run: bool = False, known_urls: Optional[set] = None,
                 on_batch=None) -> tuple:
    """
    Instantiate all scrapers, run them with configured queries, and return
    (jobs, counts): the combined deduplicated list of JobPosting objects and
    a dict mapping scraper display name -> unique jobs contributed.

    on_batch(name, jobs, rank), when given, is invoked ON THE MAIN THREAD
    with each scraper's raw batch as that scraper finishes — while slower
    scrapers are still running — so saving/scoring can overlap the scrape.
    rank encodes the old fixed merge order for the reconciling saver
    (LinkedIn keyword 0, other scrapers 1.., LinkedIn company/local 98/99).
    A callback exception is logged, later batches still stream, and the
    first error re-raises after the scrape completes (where a save-phase
    crash surfaced in the sequential flow). The returned (jobs, counts) are
    computed exactly as without the callback.
    """
    from scrapers.linkedin import LinkedInScraper
    from scrapers.greenhouse import GreenhouseScraper

    queries: List[str] = config.get("search_queries", [])
    if not queries:
        logger.warning("No search_queries defined in config.")
        queries = ["Senior Product Manager AI remote"]

    all_jobs = []
    seen_urls: set = set()

    # LinkedIn is deliberately NOT in this list: it runs its own keyword →
    # company → local chain (all against linkedin.com) on a dedicated thread
    # via _run_linkedin_chain below, but its results still merge first.
    linkedin_scraper = LinkedInScraper(config, known_urls=known_urls)
    other_scrapers = [
        ("Greenhouse/Lever", GreenhouseScraper(config)),
    ]

    if config.get("workday_tenants"):
        from scrapers.workday import WorkdayScraper
        other_scrapers.append(("Workday", WorkdayScraper(config)))

    if config.get("eightfold_tenants"):
        from scrapers.eightfold import EightfoldScraper
        other_scrapers.append(("Eightfold", EightfoldScraper(config)))

    if config.get("smartrecruiters_companies"):
        from scrapers.smartrecruiters import SmartRecruitersScraper
        other_scrapers.append(("SmartRecruiters", SmartRecruitersScraper(config)))

    if config.get("successfactors_tenants"):
        from scrapers.successfactors import SuccessFactorsScraper
        other_scrapers.append(("SuccessFactors", SuccessFactorsScraper(config)))

    if config.get("wttj_enabled"):
        from scrapers.wttj import WTTJScraper
        other_scrapers.append(("WTTJ", WTTJScraper(config)))

    if config.get("hn_whoishiring_enabled"):
        from scrapers.hn_whoishiring import HNWhoIsHiringScraper
        other_scrapers.append(("HN Who's Hiring", HNWhoIsHiringScraper(config)))

    # Each scraper targets a different host family, so they run concurrently:
    # per-host politeness (each scraper's internal rate-limit sleeps) rides
    # along into its own thread, and total scrape wall-time drops from the sum
    # of all sources to the slowest single source. LinkedIn's keyword, company,
    # and local passes all hit linkedin.com, so they stay serialized inside the
    # one LinkedIn thread. Results are merged on the main thread afterwards in
    # a fixed order (LinkedIn first, then the other_scrapers list), so URL-dedup
    # winners and per-source counts are identical to the old sequential loop
    # regardless of completion order.
    batch_queue: Optional[queue.Queue] = queue.Queue() if on_batch is not None else None

    def _emit(name: str, rank: int, batch: Optional[list]) -> None:
        if batch_queue is not None and batch:
            batch_queue.put((name, rank, batch))

    def _run_scraper(scraper_name: str, scraper, rank: int) -> Optional[list]:
        logger.info("Starting scraper: %s", scraper_name)
        try:
            result = scraper.scrape_all(queries)
        except Exception as e:
            logger.error("Scraper %s failed: %s", scraper_name, e, exc_info=True)
            return None  # failure sentinel: counts 0, no dead-board warning
        _emit(scraper_name, rank, result)
        return result

    def _run_linkedin_chain() -> tuple:
        keyword_jobs = _run_scraper("LinkedIn", linkedin_scraper, 0)

        # Company-targeted LinkedIn searches (additive — after keyword pass,
        # even when the keyword pass failed, matching the old flow)
        company_jobs = None
        if config.get("linkedin_company_searches"):
            logger.info("Running LinkedIn company-targeted searches...")
            try:
                company_jobs = linkedin_scraper.scrape_companies(
                    _linkedin_company_queries(config, queries)
                )
                _emit("LinkedIn company", 98, company_jobs)
            except Exception as e:
                logger.error("LinkedIn company searches failed: %s", e, exc_info=True)

        # Local-area LinkedIn layer (additive): geo-scoped keyword searches +
        # company-targeted fallback for local companies with no supported ATS.
        local_jobs = None
        if config.get("local_search_queries"):
            logger.info("Running LinkedIn local-area searches...")
            try:
                # Assign only after BOTH passes succeed — the old flow's
                # exception discarded the whole local layer, and a partial
                # merge would record "linkedin local" as healthy in the
                # source-regression baseline while its fallback half is broken.
                combined = linkedin_scraper.scrape_local()
                combined += linkedin_scraper.scrape_local_companies()
                local_jobs = combined
                _emit("LinkedIn local", 99, local_jobs)
            except Exception as e:
                logger.error("LinkedIn local searches failed: %s", e, exc_info=True)

        return keyword_jobs, company_jobs, local_jobs

    # Note: Ctrl-C during this block does not stop in-flight scrapers — the
    # non-daemon worker threads run to completion before the process can exit.
    # Acceptable for the launchd runs; interactive runs should prefer small
    # configs when experimenting.
    on_batch_error: Optional[Exception] = None
    done_sentinel = object()

    def _with_done_marker(fn, *fn_args):
        # Every worker future enqueues exactly one sentinel AFTER its last
        # emit (queue FIFO: a producer's earlier puts dequeue first), so the
        # pump can exit on a sentinel count with no emit left behind — a
        # timeout-poll + done()-check has a lost-batch race.
        try:
            return fn(*fn_args)
        finally:
            if batch_queue is not None:
                batch_queue.put(done_sentinel)

    with ThreadPoolExecutor(
        max_workers=len(other_scrapers) + 1, thread_name_prefix="scraper"
    ) as executor:
        linkedin_future = executor.submit(_with_done_marker, _run_linkedin_chain)
        other_futures = [
            (name, executor.submit(_with_done_marker, _run_scraper, name, s, rank))
            for rank, (name, s) in enumerate(other_scrapers, start=1)
        ]

        if on_batch is not None:
            # Pump batches to the callback on THIS thread as scrapers finish.
            workers_left = 1 + len(other_futures)
            while workers_left:
                item = batch_queue.get()
                if item is done_sentinel:
                    workers_left -= 1
                    continue
                name, rank, batch = item
                try:
                    on_batch(name, batch, rank)
                except Exception as e:
                    if on_batch_error is None:
                        on_batch_error = e
                    logger.error("on_batch callback failed for %s: %s",
                                 name, e, exc_info=True)

    linkedin_keyword_jobs, linkedin_company_jobs, linkedin_local_jobs = (
        linkedin_future.result()
    )

    def _merge(jobs: list) -> int:
        new_count = 0
        for job in jobs:
            if job.url and job.url not in seen_urls:
                seen_urls.add(job.url)
                all_jobs.append(job)
                new_count += 1
        return new_count

    counts: dict = {}
    ordered_results = [("LinkedIn", linkedin_keyword_jobs)] + [
        (name, future.result()) for name, future in other_futures
    ]
    for scraper_name, jobs in ordered_results:
        if jobs is None:  # scraper raised (already logged in its thread)
            counts[scraper_name] = 0
            continue
        try:
            new_count = _merge(jobs)
        except Exception as e:
            # A malformed result list must only zero this one source, like
            # the old loop whose try wrapped scrape AND merge together.
            counts[scraper_name] = 0
            logger.error("Scraper %s failed: %s", scraper_name, e, exc_info=True)
            continue
        counts[scraper_name] = new_count
        if new_count == 0:
            logger.warning(
                "%s returned 0 jobs this run — investigate if this source "
                "usually produces jobs (dead board, anti-bot block, or filter).",
                scraper_name,
            )
        else:
            logger.info("%s returned %d unique jobs.", scraper_name, new_count)

    if linkedin_company_jobs is not None:
        try:
            new_count = _merge(linkedin_company_jobs)
            logger.info("LinkedIn company searches added %d unique jobs.", new_count)
            counts["LinkedIn"] = counts.get("LinkedIn", 0) + new_count
        except Exception as e:
            logger.error("LinkedIn company searches failed: %s", e, exc_info=True)

    if linkedin_local_jobs is not None:
        try:
            new_count = _merge(linkedin_local_jobs)
            logger.info("LinkedIn local searches added %d unique jobs.", new_count)
            counts["LinkedIn Local"] = new_count
        except Exception as e:
            logger.error("LinkedIn local searches failed: %s", e, exc_info=True)

    if counts:
        logger.info(
            "Per-source yields: %s", ", ".join(f"{n}={c}" for n, c in counts.items())
        )
    summary = summarize_scraper_run(counts)
    if summary["all_empty"]:
        logger.error(
            "No jobs scraped from ANY source (%d scrapers, all returned 0) — likely a "
            "network, anti-bot, or board-wide failure. Do not trust this run.",
            len(counts),
        )

    logger.info("Total unique jobs from all scrapers: %d", len(all_jobs))

    # Dedup across sources by title+company (e.g. same job on LinkedIn + Greenhouse)
    all_jobs = dedup_jobs_in_memory(all_jobs)
    logger.info("Total after cross-source dedup: %d", len(all_jobs))

    if on_batch_error is not None:
        # Surface the save-side failure where the sequential flow's
        # save_jobs crash landed: after the scrape, with all yields logged.
        raise on_batch_error

    return all_jobs, counts


# ---------------------------------------------------------------------------
# Dry run output
# ---------------------------------------------------------------------------

def print_dry_run_results(jobs: list) -> None:
    """Print a formatted table of scored jobs to stdout."""
    if not jobs:
        print("\n  No jobs found.\n")
        return

    sorted_jobs = sorted(jobs, key=lambda j: j.score, reverse=True)

    col_widths = {
        "rank": 4,
        "score": 5,
        "title": min(45, max(len(j.title) for j in sorted_jobs)),
        "company": min(25, max(len(j.company) for j in sorted_jobs)),
        "location": min(20, max(len(j.location or "") for j in sorted_jobs)),
        "source": 12,
    }

    separator = "-" * (sum(col_widths.values()) + len(col_widths) * 3 + 2)
    header_fmt = (
        "  {rank:<{rw}}  {score:<{sw}}  {title:<{tw}}  "
        "{company:<{cw}}  {location:<{lw}}  {source:<{sow}}"
    )

    print(f"\n{'=' * 60}")
    print("  JOB SENTINEL — DRY RUN RESULTS")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Total jobs found: {len(sorted_jobs)}")
    above_60 = sum(1 for j in sorted_jobs if j.score >= 60)
    print(f"  Jobs above threshold (60+): {above_60}")
    print(f"{'=' * 60}\n")

    header = header_fmt.format(
        rank="Rank", rw=col_widths["rank"],
        score="Score", sw=col_widths["score"],
        title="Title", tw=col_widths["title"],
        company="Company", cw=col_widths["company"],
        location="Location", lw=col_widths["location"],
        source="Source", sow=col_widths["source"],
    )
    print(header)
    print(f"  {separator}")

    for i, job in enumerate(sorted_jobs, start=1):
        title = (job.title or "")[:col_widths["title"]]
        company = (job.company or "")[:col_widths["company"]]
        location = (job.location or "")[:col_widths["location"]]
        source = (job.source or "")[:col_widths["source"]]

        score_str = str(job.score)
        if job.score >= 80:
            score_display = f"[{score_str:>3}]"
        elif job.score >= 60:
            score_display = f" {score_str:>3} "
        else:
            score_display = f"  {score_str:>2} "

        row = (
            f"  {i:<{col_widths['rank']}}  "
            f"{score_display:<{col_widths['score']}}  "
            f"{title:<{col_widths['title']}}  "
            f"{company:<{col_widths['company']}}  "
            f"{location:<{col_widths['location']}}  "
            f"{source:<{col_widths['source']}}"
        )
        print(row)

        if job.match_explanation:
            explanation = job.match_explanation[:80]
            print(f"  {'':>{col_widths['rank']}}  {'':>{col_widths['score']}}  {explanation}")

        print()

    print(f"  {separator}")
    print(f"\n  [Dry run complete — no data written to database]\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    _check_python_version()
    parser = argparse.ArgumentParser(
        description="Job Sentinel — automated job sourcing engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                        # Full run: scrape + score + save
  python main.py --dry-run              # Scrape and score, print results, no DB write
  python main.py --scrape-only          # Scrape + keyword-score + save (no LLM, no enrichment)
  python main.py --dashboard            # Start the web dashboard
  python main.py --skip-llm             # Skip LLM scoring pass
  python main.py --verbose              # Debug-level logging
  python main.py --dismiss-job 42       # Dismiss job by ID
  python main.py --dismiss-job https://... --profile default  # Dismiss by URL
        """,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Scrape and score but don't write to DB. With --reblend: report "
             "how many scores would change without writing anything. With "
             "--rescore-force: report how many jobs would be cleared and "
             "re-scored, without writing or billing anything",
    )
    parser.add_argument(
        "--scrape-only",
        action="store_true",
        help="Scrape, keyword-score, and save to DB — skip LLM scoring and company enrichment",
    )
    parser.add_argument(
        "--dashboard",
        action="store_true",
        help="Start the FastAPI web dashboard",
    )
    parser.add_argument(
        "--profile",
        default=PROFILE_KEY,
        help="Profile to run (default: from config profile.key)",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to config file (default: config.yaml)",
    )
    parser.add_argument(
        "--db",
        default=os.environ.get("JOB_SENTINEL_DB", "data/jobs.db"),
        help="Path to SQLite database (default: data/jobs.db)",
    )
    parser.add_argument(
        "--skip-company-intel",
        action="store_true",
        help="Skip company intelligence enrichment (faster runs)",
    )
    parser.add_argument(
        "--enrich-companies",
        action="store_true",
        help="Re-fetch company intelligence for all companies in DB and exit",
    )
    parser.add_argument(
        "--skip-llm",
        action="store_true",
        help="Skip LLM scoring pass (keyword-only scoring)",
    )
    parser.add_argument(
        "--skip-auto-tailor",
        action="store_true",
        help="Skip the post-run auto-tailor pass",
    )
    parser.add_argument(
        "--rescore-all",
        action="store_true",
        help="Score all jobs in the DB where llm_score IS NULL (safe to re-run after interruption)",
    )
    parser.add_argument(
        "--rescore-force",
        action="store_true",
        help="Clear all existing LLM scores then re-score everything — use after "
             "changing scoring criteria. Destructive and re-bills the whole corpus, "
             "so it asks for interactive confirmation; combine with --dry-run to preview",
    )
    parser.add_argument(
        "--rescore-sample",
        action="store_true",
        help="Re-score 5 previously-scored jobs with the new model and print old vs new comparison",
    )
    parser.add_argument(
        "--reblend",
        action="store_true",
        help="Recompute every job's score with NO LLM calls: re-run the keyword "
             "scorer (picking up rule changes like the location cap) and re-blend "
             "with each row's existing llm_score at the current weights. Run after "
             "changing keyword rules or blend weights — spends zero subscription budget",
    )
    parser.add_argument(
        "--rescore-run",
        type=int,
        default=None,
        metavar="RUN_ID",
        help="LLM-score only the unscored jobs from one run_id (e.g. after a transient CLI auth "
             "failure left a day's new jobs keyword-only). Bounded to that run; find the id in the runs table.",
    )
    parser.add_argument(
        "--backfill-filter",
        action="store_true",
        help="Compute the Filter Match score for already-scored jobs that lack it, "
             "including the '{}' sentinel written while the feature was disabled "
             "(resumable — written rows leave the target set; prefiltered and "
             "expired/applied/not-interested rows are skipped). Reuses the full "
             "job-scoring path, so it also refreshes llm_score and salary "
             "estimates for the affected rows, not just the filter columns",
    )
    parser.add_argument(
        "--rejudge-filter",
        action="store_true",
        help="Re-run the Filter Match judge (stage 2 only) for live jobs "
             "whose stored judgment predates the current "
             "data/experience_inventory.md. Cheap: no fit re-scoring — run "
             "this after every inventory edit",
    )
    parser.add_argument(
        "--rejudge-filter-all",
        action="store_true",
        help="Like --rejudge-filter but ignores the inventory-hash staleness "
             "check and re-judges every eligible (live, master-basis) job",
    )
    parser.add_argument(
        "--filter-since-hours",
        type=int,
        default=None,
        metavar="HOURS",
        help="With --rejudge-filter or --backfill-filter, only process jobs "
             "created within the last HOURS hours (e.g. 48). Omit to process all "
             "eligible jobs regardless of age",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug-level logging (shorthand for --log-level DEBUG)",
    )
    parser.add_argument(
        "--dismiss-job",
        metavar="URL_OR_ID",
        help="Dismiss (hide) a job by URL or integer ID. Use with --profile.",
    )

    args = parser.parse_args()
    validate_mode_flags(args)
    apply_scrape_only(args)

    # Setup logging (must happen before any logger calls below)
    setup_logging(log_level=args.log_level, verbose=args.verbose)

    # Resolve config path if not explicitly provided
    if args.config is None:
        args.config = "config.yaml"

    logger.info("Running with profile: %s", args.profile)

    # Load config
    config = load_config(args.config)

    # Set DB path env var for dashboard and notifier
    os.environ["JOB_SENTINEL_DB"] = args.db

    errors_log = str(Path("logs") / "errors.log")

    # --- Dismiss-job mode ---
    if args.dismiss_job:
        conn = init_database(args.db)
        target = args.dismiss_job.strip()
        # Try integer ID first, then URL
        if target.isdigit():
            row = conn.execute(
                "SELECT id, title, company FROM jobs WHERE id = ? AND profile = ?",
                (int(target), args.profile),
            ).fetchone()
        else:
            row = conn.execute(
                "SELECT id, title, company FROM jobs WHERE url = ? AND profile = ?",
                (target, args.profile),
            ).fetchone()
        if not row:
            logger.error("No job found for '%s' (profile: %s). Nothing dismissed.", target, args.profile)
            conn.close()
            sys.exit(1)
        conn.execute("UPDATE jobs SET dismissed = 1 WHERE id = ?", (row["id"],))
        conn.commit()
        conn.close()
        logger.info("Dismissed job #%d — %s @ %s", row["id"], row["title"], row["company"])
        return

    # --- Rescore-sample mode ---
    if args.rescore_sample:
        logger.info("RESCORE-SAMPLE mode — comparing old vs new scores for 5 jobs.")
        from engine.llm_scorer import LLMScorer
        llm_scorer = LLMScorer(db_path=args.db)
        llm_scorer.print_rescore_sample(n=5, profile=args.profile)
        return

    # --- Rescore-all mode ---
    if args.rescore_all:
        logger.info("RESCORE-ALL mode — scoring all jobs where llm_score IS NULL (profile: %s).", args.profile)
        from engine.llm_scorer import LLMScorer
        llm_scorer = LLMScorer(db_path=args.db)
        if not llm_scorer.is_available():
            logger.error("claude CLI unavailable — cannot rescore. Is the `claude` CLI installed and logged in? (model=%s)", llm_scorer.model)
            sys.exit(1)
        count = llm_scorer.rescore_all(profile=args.profile)
        logger.info("Rescore complete: %d jobs re-scored.", count)
        return

    # --- Backfill-filter mode ---
    if args.backfill_filter:
        logger.info("BACKFILL-FILTER mode — computing Filter Match for jobs missing it (profile: %s).", args.profile)
        from engine.llm_scorer import LLMScorer
        llm_scorer = LLMScorer(db_path=args.db)
        if not llm_scorer.is_available():
            logger.error("claude CLI unavailable — cannot backfill. Is the `claude` CLI installed and logged in? (model=%s)", llm_scorer.model)
            sys.exit(1)
        count = llm_scorer.backfill_filter(profile=args.profile, since_hours=args.filter_since_hours)
        logger.info("Backfill complete: %d jobs updated.", count)
        return

    # --- Rejudge-filter mode (stage 2 only) ---
    if args.rejudge_filter or args.rejudge_filter_all:
        logger.info("REJUDGE-FILTER mode — re-judging Filter Match vs current inventory (profile: %s).", args.profile)
        from engine.llm_scorer import LLMScorer
        llm_scorer = LLMScorer(db_path=args.db)
        if not llm_scorer.is_available():
            logger.error("claude CLI unavailable — cannot re-judge. Is the `claude` CLI installed and logged in? (model=%s)", llm_scorer.model)
            sys.exit(1)
        count = llm_scorer.rejudge_filter(
            profile=args.profile, force_all=args.rejudge_filter_all,
            since_hours=args.filter_since_hours)
        logger.info("Rejudge complete: %d jobs updated.", count)
        return

    # --- Rescore-run mode (only one run's unscored jobs) ---
    if args.rescore_run is not None:
        logger.info("RESCORE-RUN mode — scoring unscored jobs from run %d (profile: %s).", args.rescore_run, args.profile)
        from engine.llm_scorer import LLMScorer
        llm_scorer = LLMScorer(db_path=args.db)
        if not llm_scorer.is_available():
            logger.error("claude CLI unavailable — cannot rescore. Is the `claude` CLI installed and logged in? (model=%s)", llm_scorer.model)
            sys.exit(1)
        count = llm_scorer.apply_llm_scores_to_db(run_id=args.rescore_run, profile=args.profile)
        logger.info("Rescore-run complete: %d jobs re-scored for run %d.", count, args.rescore_run)
        return

    # --- Reblend mode (no LLM calls) ---
    if args.reblend:
        logger.info("REBLEND mode%s — recomputing scores from stored components, no LLM calls (profile: %s).",
                    " (dry-run)" if args.dry_run else "", args.profile)
        from engine.llm_scorer import LLMScorer
        count = LLMScorer(db_path=args.db).reblend_all(profile=args.profile,
                                                       dry_run=args.dry_run)
        logger.info("Reblend%s complete: %d jobs %s.",
                    " dry-run" if args.dry_run else "", count,
                    "would change" if args.dry_run else "changed")
        return

    # --- Rescore-force mode ---
    if args.rescore_force:
        from engine.llm_scorer import LLMScorer
        llm_scorer = LLMScorer(db_path=args.db)
        pending = llm_scorer.rescore_all(force=True, profile=args.profile, dry_run=True)
        if args.dry_run:
            logger.info("RESCORE-FORCE dry-run: %d jobs would have their LLM score "
                        "cleared and re-scored (profile: %s). Nothing written.",
                        pending, args.profile)
            return
        if not llm_scorer.is_available():
            logger.error("claude CLI unavailable — cannot rescore. Is the `claude` CLI installed and logged in? (model=%s)", llm_scorer.model)
            sys.exit(1)
        if not confirm_rescore_force(pending, args.profile):
            logger.info("Rescore-force aborted — no scores cleared, nothing re-scored.")
            sys.exit(1)
        logger.info("RESCORE-FORCE mode — clearing all LLM scores and re-scoring from scratch (profile: %s).", args.profile)
        count = llm_scorer.rescore_all(force=True, profile=args.profile)
        logger.info("Force-rescore complete: %d jobs re-scored.", count)
        return

    # --- Enrich-companies mode ---
    if args.enrich_companies:
        logger.info("ENRICH-COMPANIES mode — refreshing company intel for all known companies.")
        from engine.company_intel import CompanyIntelligence

        conn = init_database(args.db)
        rows = conn.execute(
            "SELECT DISTINCT company FROM jobs ORDER BY company"
        ).fetchall()
        conn.close()

        companies = [r["company"] for r in rows if r["company"]]
        logger.info("Found %d unique companies to enrich.", len(companies))

        intel = CompanyIntelligence(args.db)
        intel.batch_enrich(companies, delay=2.0)
        logger.info("Company enrichment complete.")
        return

    # --- Dashboard mode ---
    if args.dashboard:
        try:
            import uvicorn
        except ImportError:
            logger.error("uvicorn is required for the dashboard. Run: pip install uvicorn")
            sys.exit(1)

        port = config.get("dashboard", {}).get("port", 8500)
        # Localhost-only by default — the dashboard is unauthenticated and serves
        # resume text / can trigger Opus tailoring, so it must not bind to the LAN.
        host = config.get("dashboard", {}).get("host", "127.0.0.1")
        logger.info("Starting Job Sentinel dashboard on http://%s:%d", host, port)

        project_root = Path(args.config).parent.resolve()
        os.chdir(project_root)

        uvicorn.run(
            "dashboard.app:app",
            host=host,
            port=port,
            reload=False,
            log_level=args.log_level.lower(),
            access_log=False,  # launchd redirect files never rotate; access lines are the growth vector
        )
        return

    # --- Dry run mode ---
    if args.dry_run:
        logger.info("DRY RUN mode — no database writes.")
        from engine.scorer import JobScorer

        scorer = JobScorer(config)
        # NOTE: --dry-run scrapes without a known_urls dedup set (no DB conn here),
        # so it fetches every card's detail; with 14 queries × up to 3 pages it is a
        # materially heavier LinkedIn footprint than the real run, which suppresses
        # ~8K already-known URLs. A heavy dry-run can trip the 429 breaker.
        jobs, _ = run_scrapers(config, dry_run=True)

        for job in jobs:
            score, explanation = scorer.score_and_explain(job)
            job.score = score
            job.match_explanation = explanation

        print_dry_run_results(jobs)
        return

    # --- Normal run: scrape + score + save + LLM score ---
    conn = init_database(args.db)
    _cleanup_db_on_init(
        conn, profile=args.profile,
        retention_days=int(config.get("retention", {}).get("expired_days", 180)),
    )

    # Create a run record
    started_at = datetime.now().isoformat()
    cursor = conn.execute(
        "INSERT INTO runs (started_at, status, profile) VALUES (?, 'running', ?)",
        (started_at, args.profile),
    )
    run_id = cursor.lastrowid
    conn.commit()
    logger.info("Run ID: %d", run_id)

    try:
        from engine.scorer import JobScorer
        from alerts.error_monitor import record_run_result

        scorer = JobScorer(config)

        # Seed scrapers with the URLs already stored WITH a description so
        # the per-job detail fetches (workday/eightfold/smartrecruiters/wttj)
        # skip postings we already have — in steady state that's nearly all
        # of them. Internal key, read by BaseScraper.__init__.
        config["_urls_with_descriptions"] = _urls_with_descriptions(conn)

        # Scoring-behind-scrape: with llm_scoring.overlap_scrape (default on),
        # each scraper's batch is reconcile-saved as it finishes and a
        # streaming consumer scores the growing NULL set DURING the scrape —
        # the old flow's sequential scrape → save → score is the fallback.
        overlap_scoring = (
            not args.skip_llm
            and bool(config.get("llm_scoring", {}).get("overlap_scrape", True))
        )
        llm_scorer = None
        score_cap = config.get("llm_scoring", {}).get("max_jobs_per_run")

        if overlap_scoring:
            from engine.llm_scorer import LLMScorer
            llm_scorer = LLMScorer(db_path=args.db)
            (jobs, source_counts, save_stats,
             streamed, streamed_attempts) = run_overlap_scrape_and_score(
                conn, config, args, run_id, scorer, llm_scorer, score_cap)
        else:
            jobs, source_counts = run_scrapers(
                config, known_urls=load_known_linkedin_urls(conn)
            )
            save_stats = save_jobs(conn, jobs, run_id, scorer, profile_name=args.profile)

        # --- Company intelligence enrichment (background, overlaps scoring
        # and tailor; joined before the run record below) ---
        enrich_thread = None
        if not args.skip_company_intel and jobs:
            enrich_thread = start_company_enrichment(args.db, jobs)

        # --- LLM scoring pass (Improvement #2 & #3) ---
        if not args.skip_llm:
            if overlap_scoring:
                if streamed:
                    logger.info("Streaming pass scored %d jobs during the scrape (%d attempts).",
                                streamed, streamed_attempts)
                # Top-up: best-first among this run's remaining NULL rows —
                # the held-back reserve, reconciled re-NULLs, and anything
                # the streaming pass skipped. Budget counts attempts so the
                # cap bounds CLI calls like the old single LIMIT-N pass.
                remaining_cap = (None if score_cap is None
                                 else max(0, score_cap - streamed_attempts))
                if remaining_cap is None or remaining_cap > 0:
                    llm_scorer.apply_llm_scores_to_db(
                        run_id=run_id, profile=args.profile, limit=remaining_cap)
                if not llm_scorer.is_available():
                    # Parity with the old single pass, which always fell back
                    # to regex salary extraction over the whole run.
                    llm_scorer._extract_salaries_regex_only(run_id=run_id)
                llm_count = conn.execute(
                    "SELECT COUNT(*) FROM jobs WHERE run_id = ? AND llm_score IS NOT NULL",
                    (run_id,)).fetchone()[0]
            else:
                from engine.llm_scorer import LLMScorer
                llm_scorer = LLMScorer(db_path=args.db)
                # Cap LLM calls per run (highest keyword score first) so a high-yield
                # scrape can't exhaust the Claude subscription usage window in one burst.
                llm_count = llm_scorer.apply_llm_scores_to_db(
                    run_id=run_id, profile=args.profile, limit=score_cap)
            if llm_count:
                kw_pct = round(float(config.get("llm_scoring", {}).get("keyword_weight", 0.4)) * 100)
                llm_pct = round(float(config.get("llm_scoring", {}).get("llm_weight", 0.6)) * 100)
                logger.info("LLM re-scored %d jobs (blended: %d%% keyword + %d%% LLM).",
                            llm_count, kw_pct, llm_pct)
            # Self-heal: retry a small bounded batch of live rows a prior run left
            # unscored (transient CLI failures), oldest first — never stale/expired.
            backfilled = llm_scorer.apply_llm_scores_to_db(profile=args.profile, backfill_limit=25)
            if backfilled:
                logger.info("LLM backfill scored %d previously-unscored jobs.", backfilled)
        else:
            logger.info("--skip-llm: skipping LLM scoring pass.")
            # Still do regex salary extraction even when skipping LLM
            from engine.llm_scorer import LLMScorer
            LLMScorer(db_path=args.db)._extract_salaries_regex_only(run_id=run_id)

        # Recount above-threshold on blended, knockout-gated scores — the
        # save-time count was keyword-only (pre-LLM) and overstates junk.
        threshold = int(config.get("scoring", {}).get("alert_threshold", 60))
        blended_above = count_effective_above_threshold(conn, run_id, threshold)
        if blended_above != save_stats["total_above_threshold"]:
            logger.info(
                "Above-threshold recount after LLM blend: %d (save-time keyword count was %d).",
                blended_above, save_stats["total_above_threshold"])
        save_stats["total_above_threshold"] = blended_above

        # --- Auto-tailor top matches ---
        if not args.skip_auto_tailor:
            at_cfg = config.get("auto_tailor") or {}
            try:
                tailored = run_auto_tailor(conn, args.db, args.profile, at_cfg,
                                           errors_log=errors_log)
                if tailored:
                    logger.info("Auto-tailor: %d resume(s) ready in the dashboard/history.", tailored)
            except Exception as e:
                logger.warning("Auto-tailor pass failed (non-fatal): %s", e)

        # Enrichment must be complete before the run record and digest — the
        # digest reads intel, and the next run's layoff penalties read what
        # this run wrote.
        if enrich_thread is not None:
            enrich_thread.join()

        # Update run record
        sources = format_sources(source_counts)
        conn.execute(
            """
            UPDATE runs SET
                completed_at = ?,
                total_scraped = ?,
                total_new = ?,
                total_above_threshold = ?,
                sources = ?,
                source_counts = ?,
                status = 'completed'
            WHERE id = ?
            """,
            (
                datetime.now().isoformat(),
                save_stats["total_jobs"],
                save_stats["total_new"],
                save_stats["total_above_threshold"],
                sources,
                json.dumps(source_counts),
                run_id,
            ),
        )
        conn.commit()

        source_warnings = detect_source_regressions(conn, source_counts, run_id)
        for w in source_warnings:
            logger.warning("SOURCE REGRESSION: %s", w)

        # Record success in run_history
        record_run_result(args.db, "success", run_id=run_id)

        # Save JSON report
        from alerts.notifier import ReportWriter

        report_writer = ReportWriter(config)
        report_dir = str(Path(args.db).parent / "reports")
        report_writer.save_report(jobs, save_stats, report_dir, source_warnings=source_warnings)

        logger.info(
            "Run complete. Scraped: %d | New: %d | Above threshold: %d",
            save_stats["total_jobs"],
            save_stats["total_new"],
            save_stats["total_above_threshold"],
        )

    except Exception as e:
        logger.error("Fatal error during run: %s", e, exc_info=True)

        conn.execute(
            "UPDATE runs SET status = 'failed', completed_at = ? WHERE id = ?",
            (datetime.now().isoformat(), run_id),
        )
        conn.commit()

        # --- Error monitoring: log to errors.log + record failure in run_history ---
        from alerts.error_monitor import handle_run_failure
        handle_run_failure(
            error=e,
            db_path=args.db,
            errors_log=errors_log,
            run_id=run_id,
        )

        conn.close()
        sys.exit(1)

    conn.close()


if __name__ == "__main__":
    main()
