"""
LLM-based job scoring via the local Claude CLI (Sonnet 5).

Uses claude-sonnet-5 through the `claude` CLI to:
  - Rate candidate-job fit on 0-100 scale across 5 dimensions with explanation
  - Extract compensation from job description text

Results are cached in SQLite (llm_score, llm_explanation columns on jobs table).
Final score = 0.4 * keyword_score + 0.6 * llm_score

Falls back gracefully to keyword-only scoring if the claude CLI is unavailable.

Scoring rubric (total 100 pts) — the per-dimension prose is config-driven
(policy.rubric.*); see profile_policy.py. Summary of the five dimensions:
  Role Match      (0-30): target role family fit
  Seniority Match (0-20): target seniority level
  Remote/Location (0-20): remote-friendly, US-based, or local
  Domain Fit      (0-20): target-domain signal in the job description
  Comp Match      (0-10): total comp vs. target
"""
import fcntl
import functools
import json
import logging
import os
import re
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from html import unescape
from typing import Optional, Tuple

import yaml

from claude_cli import run_claude, ClaudeCLIError, ClaudeCLITimeout, is_cli_available
from filter_match import (
    build_filter_json_v2,
    compute_judged_filter_score,
)
from filter_judge import (
    apply_soft_years,
    build_judge_prompt,
    build_judge_system,
    load_inventory,
    parse_judge_response,
)
from local_area import LOCAL_COMMUTER_CITIES, build_local_area_regex
from profile_policy import (
    COMP_CAP_LOW,
    COMP_CAP_MID,
    COMP_TARGET,
    DOMAIN_SIGNAL_RE,
    LOCAL_FULL_COMP,
    LOCAL_PARTIAL_COMP,
    PREFILTER_ENG_TITLE_KEYWORDS,
    PREFILTER_NON_TARGET_ADJACENT_TITLES,
    PREFILTER_NON_TARGET_TITLES,
    PREFILTER_SALES_BD_TITLES,
    PREFILTER_SOLUTIONS_CS_TITLES,
    PREFILTER_TARGET_KEYWORDS,
    PROFILE_KEY,
    RELOCATION_EXCEPTION_COMP,
    RUBRIC_DOMAIN_FIT_BLOCK,
    RUBRIC_LOCAL_ANCHOR,
    RUBRIC_LOCAL_AREA_PROSE,
    RUBRIC_LOCAL_AREA_SHORT,
    RUBRIC_REMOTE_LOCATION_BLOCK,
    RUBRIC_ROLE_MATCH_BLOCK,
    RUBRIC_SENIORITY_MATCH_BLOCK,
)
from salary_rules import (  # noqa: F401 — some re-exported
    MAX_BASE_SALARY,
    NON_US_LOCATIONS,
    extract_salary_regex,
    is_high_comp_exception,
    sanitize_salary_range,
)
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
    before_sleep_log,
)

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-sonnet-5"

# Retry transient CLI failures, but NOT timeouts: for a fixed prompt the
# latency is deterministic, so re-running at the same ceiling just burns the
# subscription window and blocks the global claude semaphore
# (claude_cli.ClaudeCLITimeout exists precisely for this distinction;
# resume_tailor applies the same policy).
_retry_transient_cli = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(min=2, max=30),
    retry=retry_if_exception(
        lambda e: isinstance(e, ClaudeCLIError) and not isinstance(e, ClaudeCLITimeout)
    ),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)

# filter_fields sentinel for "scored, but no usable filter requirements" — an LLM
# parse failure or an empty extraction. The non-NULL filter_json ('{}') lets the row
# leave the --backfill-filter target set (WHERE filter_json IS NULL) after ONE
# attempt instead of being re-scored on every run; NULL score/master/knockout render
# as "—" on the dashboard (filterBadge shows "—" when filter_score is null).
_FILTER_SENTINEL = (None, None, "none", None, "{}")

# Filter Match v2: stage-1 extraction happens in the scoring call; stage-2
# judging happens in _judge_filter against the experience inventory. Flip to
# False to disable both the judge call and all filter writes (sentinel only).
_FILTER_MATCH_ENABLED = True

# Flush scored results to the DB every N completions so a mid-pass crash
# (OOM has happened on this machine) forfeits at most one batch of CLI spend.
_FLUSH_EVERY = 20


def _with_scoring_lock(fn):
    """Cross-process guard around a CLI-spending scoring pass.

    Rows are "claimed" only by the initial SELECT, so two overlapping passes
    (e.g. a manual backfill script alongside the launchd daily run) would both
    score the same rows — double subscription spend. An exclusive flock on
    .scoring.lock next to the DB (data/.scoring.lock in production) serializes
    them: on contention the pass logs a WARNING and returns 0 instead of
    blocking. flock conflicts across open-file-descriptions, so this also
    guards two passes within one process; it is released even on SIGKILL.
    """
    @functools.wraps(fn)
    def wrapper(self, *args, **kwargs):
        if self.db_path == ":memory:":
            # In-memory DBs are process-private — nothing to race against.
            return fn(self, *args, **kwargs)
        lock_path = os.path.join(
            os.path.dirname(os.path.abspath(self.db_path)), ".scoring.lock")
        fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o644)
        try:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                logger.warning(
                    "Another scoring pass holds %s — skipping %s to avoid "
                    "double-scoring (and double-billing) the same rows.",
                    lock_path, fn.__name__)
                return 0
            return fn(self, *args, **kwargs)
        finally:
            os.close(fd)
    return wrapper

# ---------------------------------------------------------------------------
# Config loading — reads config.yaml from project root (one level up from here)
# ---------------------------------------------------------------------------

_DEFAULT_KW_WEIGHT = 0.4
_DEFAULT_LLM_WEIGHT = 0.6


def _load_llm_scoring_config() -> dict:
    """Load llm_scoring section from config.yaml, returning an empty dict on any error."""
    config_path = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
    try:
        with open(config_path, "r") as fh:
            cfg = yaml.safe_load(fh) or {}
        return cfg.get("llm_scoring", {})
    except Exception:
        logger.debug("Could not load config.yaml for llm_scoring — using defaults.")
        return {}


_llm_cfg = _load_llm_scoring_config()
RESUME_SUMMARY: Optional[str] = (_llm_cfg.get("resume_summary") or "").strip() or None
_KW_WEIGHT: float = float(_llm_cfg.get("keyword_weight", _DEFAULT_KW_WEIGHT))
_LLM_WEIGHT: float = float(_llm_cfg.get("llm_weight", _DEFAULT_LLM_WEIGHT))


def _require_resume_summary() -> str:
    """The candidate profile every scoring call is judged against. No silent default."""
    if not RESUME_SUMMARY:
        raise RuntimeError(
            "llm_scoring.resume_summary is not set in config.yaml — LLM scoring would "
            "run against an empty candidate profile. Write a 10-20 line professional "
            "summary (background, skills, target roles, salary floor, location policy) "
            "under llm_scoring.resume_summary and re-run."
        )
    return RESUME_SUMMARY

# Stage-2 judge gate: skip the judge call when the blended score lands below
# this floor. Filter data only matters for jobs the owner will actually see
# (alerts >= 60; auto-tailor needs score AND filter >= 60, inclusive — the
# 2026-07-17 ruling: exactly 60 qualifies everywhere). Gated rows get the
# '{}' sentinel, and backfill_filter applies the same floor so they aren't
# re-billed later.
_JUDGE_MIN_SCORE = int(_llm_cfg.get("judge_min_score", 40))

# Description chars fed to the LLM prompt. The post-parse AI-signal cap must scan
# the SAME window, or it would override the model on evidence the model never saw.
_LLM_DESC_CHARS = 6_000

# Static scoring context (persona + candidate profile + rubric + output
# schema). Sent as the CLI system prompt so consecutive scoring calls share
# an identical prefix (prompt-cache reuse); the volatile job posting goes in
# _FIT_USER_TEMPLATE (the user message). feedback_block changes at most once
# per run, so it stays cache-stable within a pass.
_FIT_SYSTEM_TEMPLATE_RAW = """\
You are a precise job-fit evaluator. The user message contains ONE job posting. Score how well the candidate below fits that job on 5 dimensions and extract its screening requirements. Respond with ONLY a single valid JSON object — no markdown fences, no commentary.

CANDIDATE PROFILE:
{resume}

{feedback_block}
SCORING RUBRIC — score each dimension strictly:

%%ROLE_MATCH_BLOCK%%

%%SENIORITY_MATCH_BLOCK%%

%%REMOTE_LOCATION_BLOCK%%

%%DOMAIN_FIT_BLOCK%%

5. comp_match (0-10): Is total comp likely %%COMP_TARGET_K%%+ based on role/company/explicit salary?
   - 9-10: Explicit salary >= %%COMP_TARGET_K%%, OR FAANG/top-tier AI company (Google, Meta, Apple, Amazon, Microsoft, Nvidia, OpenAI, Anthropic, Scale AI, Databricks, Palantir) at Director+ level
   - 6-8: Explicit salary %%COMP_CAP_LOW_K%%-%%COMP_TARGET_MINUS1_K%%, OR well-funded Series B+ tech company at Senior/Director level with no explicit salary
   - 3-5: No explicit salary AND company/role suggests $150-180K (mid-market SaaS, smaller tech company, fintech IC)
   - 0-2: Explicit salary below %%COMP_CAP_LOW_K%%, non-US comp, government, non-profit, hospital, insurance company, or regulated industry without explicit salary

   IMPORTANT: When salary is not stated, default to 3 unless company tier and role level clearly justify higher. Do NOT default to 6. Hospitals, government agencies, and insurance companies rarely hit %%COMP_TARGET_K%%+ for these roles — score 0-2. When an explicit salary >= %%COMP_TARGET_K%% is provided, you MUST score 9-10 — do not score below 9 for a confirmed high salary.

   LOCAL COMP CALIBRATION: If the job is located in the %%LOCAL_AREA_SHORT%% (see dimension 3), score against the local market instead of %%COMP_TARGET_K%%: explicit salary >= %%LOCAL_FULL_K%% scores 9-10; %%LOCAL_PARTIAL_K%%-%%LOCAL_FULL_MINUS1_K%% scores 6-8; below %%LOCAL_PARTIAL_K%% scores 0-3; no salary stated at a large local employer (health system, manufacturer HQ, bank) defaults to 5.

FILTER EXTRACTION — also extract what an employer's screening layer would check:

- must_have_keywords: the 5-15 terms a recruiter would put in a boolean search —
  skills, tools, domains, certifications EXPLICITLY required or strongly emphasized
  by the job description. Never invent requirements the JD does not state. Each
  term carries "aliases": strict equivalents only — acronym expansions,
  plural/singular forms, exact synonyms a resume might use instead. Aliases are
  never adjacent concepts ("LLMs" is NOT an alias of "Generative AI").
- knockouts: ONLY requirements the JD states as required/must-have that screening
  forms auto-reject on: location/onsite, work authorization, years of experience,
  degree, certification, clearance. Judge each verdict against the CANDIDATE
  PROFILE above: "met" (profile clearly satisfies), "unclear" (not evident from
  the profile), "failed" (profile clearly contradicts).
- Quantified requirements are knockouts, not keywords. A compound requirement
  produces BOTH. Example — JD says "Requires 8+ years of product management with
  Generative AI products":
  "must_have_keywords": [{{"term": "Generative AI", "aliases": ["GenAI", "generative artificial intelligence"]}}, {{"term": "product management", "aliases": ["product manager"]}}],
  "knockouts": [{{"requirement": "8+ years product management experience", "verdict": "met"}}]
- title_variants: acceptable exact-equivalent forms of THIS job's title (e.g.
  "Staff Product Manager" → ["Staff PM"]).
- title_alignment: how the candidate's actual titles compare to this job's title:
  "exact" | "close" | "none".

ALWAYS estimate total compensation: regardless of whether a salary is posted, estimate the realistic US-market TOTAL annual compensation range (base + bonus + equity) for THIS role at THIS company, from company tier, title seniority, and location. Return plain numbers in est_total_comp_min / est_total_comp_max. Calibration anchors: FAANG/top-AI-lab Director ~ 400000-700000; mid-market senior PM ~ 180000-260000; %%LOCAL_ANCHOR%% ~ 150000-220000.

Respond with ONLY a JSON object (no markdown, no extra text):
{{"role_match": <0-30>, "seniority_match": <0-20>, "remote_location": <0-20>, "ai_domain_fit": <0-20>, "comp_match": <0-10>, "fit_score": <sum of all dimensions, 0-100>, "explanation": "<2-3 sentence explanation>", "salary_min": <number or null>, "salary_max": <number or null>, "est_total_comp_min": <number>, "est_total_comp_max": <number>, "filter": {{"must_have_keywords": [{{"term": "<string>", "aliases": ["<string>"]}}], "title_variants": ["<string>"], "title_alignment": "<exact|close|none>", "knockouts": [{{"requirement": "<string>", "verdict": "<met|unclear|failed>"}}]}}}}

Rules for salary_min/salary_max: extract BASE SALARY only, not total compensation. Exclude RSUs, stock grants, bonuses, and equity from salary figures. Convert K→thousands (e.g. %%COMP_TARGET_K%%=220000). Use null if not found.
"""


def _usd(v: int) -> str:
    return f"${v:,}"


def _usd_k(v: int) -> str:
    return f"${v // 1000}K"


def assemble_fit_template(blocks: Optional[dict] = None) -> str:
    """Assemble the fit-scoring system template. `blocks` overrides the four
    config-driven rubric blocks (used by the default-render drift test);
    None uses the owner's configured blocks from profile_policy."""
    b = blocks or {
        "role_match": RUBRIC_ROLE_MATCH_BLOCK,
        "seniority_match": RUBRIC_SENIORITY_MATCH_BLOCK,
        "remote_location": RUBRIC_REMOTE_LOCATION_BLOCK,
        "domain_fit": RUBRIC_DOMAIN_FIT_BLOCK,
    }
    return (
        _FIT_SYSTEM_TEMPLATE_RAW
        .replace("%%ROLE_MATCH_BLOCK%%", b["role_match"])
        .replace("%%SENIORITY_MATCH_BLOCK%%", b["seniority_match"])
        .replace("%%REMOTE_LOCATION_BLOCK%%", b["remote_location"])
        .replace("%%DOMAIN_FIT_BLOCK%%", b["domain_fit"])
        .replace("%%LOCAL_AREA_PROSE%%", RUBRIC_LOCAL_AREA_PROSE)
        .replace("%%LOCAL_AREA_SHORT%%", RUBRIC_LOCAL_AREA_SHORT)
        .replace("%%LOCAL_ANCHOR%%", RUBRIC_LOCAL_ANCHOR)
        .replace("%%EXCEPTION_COMP_K%%", _usd_k(RELOCATION_EXCEPTION_COMP))
        .replace("%%EXCEPTION_COMP%%", _usd(RELOCATION_EXCEPTION_COMP))
        .replace("%%COMP_TARGET_MINUS1_K%%", _usd_k(COMP_TARGET - 1000))
        .replace("%%COMP_TARGET_K%%", _usd_k(COMP_TARGET))
        .replace("%%COMP_CAP_LOW_K%%", _usd_k(COMP_CAP_LOW))
        .replace("%%LOCAL_FULL_MINUS1_K%%", _usd_k(LOCAL_FULL_COMP - 1000))
        .replace("%%LOCAL_FULL_K%%", _usd_k(LOCAL_FULL_COMP))
        .replace("%%LOCAL_PARTIAL_K%%", _usd_k(LOCAL_PARTIAL_COMP))
    )


_FIT_SYSTEM_TEMPLATE = assemble_fit_template()


# Volatile per-job content — the entire user message for a scoring call.
_FIT_USER_TEMPLATE = """\
JOB POSTING:
Title: {title}
Company: {company}
Location: {location}
Salary: {salary_info}
Description:
{description}
"""


# ---------------------------------------------------------------------------
# HTML / text cleaning
# ---------------------------------------------------------------------------

def clean_for_llm(text: str) -> str:
    """Strip HTML tags, decode entities, and normalize whitespace."""
    if not text:
        return ""
    text = unescape(text)                    # decode &amp; &lt; etc. FIRST (handles encoded tags)
    text = re.sub(r"<[^>]+>", " ", text)   # strip tags SECOND
    text = re.sub(r"\s+", " ", text).strip() # normalize whitespace
    return text


def format_feedback_block(more: list, less: list) -> str:
    """Render the user's more/less-like-this examples for the scoring prompt."""
    if not more and not less:
        return ""
    lines = [
        "CANDIDATE FEEDBACK (the candidate rated past postings — use as soft "
        "calibration for role/domain taste, do not override the rubric):"
    ]
    if more:
        lines.append("Wants MORE roles like: " + "; ".join(more))
    if less:
        lines.append("Wants FEWER roles like: " + "; ".join(less))
    return "\n".join(lines) + "\n"


def _load_feedback_examples(db_path: str, profile: str, limit: int = 5):
    """Return (more, less) example lists like 'Title @ Company', newest first."""
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        def _pick(direction):
            rows = conn.execute(
                "SELECT title, company FROM jobs "
                "WHERE profile = ? AND feedback = ? ORDER BY id DESC LIMIT ?",
                (profile, direction, limit),
            ).fetchall()
            return [f"{r['title']} @ {r['company']}" for r in rows]
        more, less = _pick("more"), _pick("less")
        conn.close()
        return more, less
    except sqlite3.OperationalError:
        return [], []


# ---------------------------------------------------------------------------
# Rules-based pre-filter (fast, no LLM needed)
# ---------------------------------------------------------------------------

# Owner policy — target role/domain prefilter patterns, sourced from
# profile_policy (compiled from config in the owner tree, None in a neutral
# tree). A None value means the rule is disabled: every use below is
# None-guarded and simply defers to the LLM instead of applying a cap.
_NON_PM_TITLES = PREFILTER_NON_TARGET_TITLES
_SALES_BD_TITLES = PREFILTER_SALES_BD_TITLES
_SOLUTIONS_CS_TITLES = PREFILTER_SOLUTIONS_CS_TITLES
_ENG_TITLE_KEYWORDS = PREFILTER_ENG_TITLE_KEYWORDS
_NON_PM_ADJACENT_TITLES = PREFILTER_NON_TARGET_ADJACENT_TITLES
_PM_KEYWORDS = PREFILTER_TARGET_KEYWORDS
_AI_SIGNAL_KEYWORDS = DOMAIN_SIGNAL_RE

# Seniority caps — used for post-parse seniority_match cap
# The LLM consistently over-awards Director-level seniority for plain PM/non-PM titles.
_DIRECTOR_TITLE = re.compile(
    r"\b(director|vp\b|svp|evp|head of|chief product|cpo|gm\b|general manager|vice president)\b",
    re.IGNORECASE,
)
_SENIOR_TITLE = re.compile(r"\b(senior|principal|staff|lead\b)\b", re.IGNORECASE)
# Manager-level titles (without Director/VP) that should cap seniority at 11
_MANAGER_TITLE = re.compile(
    r"\b(engineering manager|account executive|account manager|"
    r"product manager|program manager|project manager|"
    r"\bmanager\b)\b",
    re.IGNORECASE,
)

# Owner commuter area — built from profile_policy.LOCAL_CITIES via local_area;
# LLMScorer has no config access, so this module-level pattern is the
# config-less path. Full remote_location points for local jobs.
_LOCAL_AREA_RE = build_local_area_regex(LOCAL_COMMUTER_CITIES)

# Analyst / IC titles — used for post-parse seniority_match cap
_ANALYST_IC_TITLE = re.compile(
    r"\b(analyst|coordinator|specialist)\b",
    re.IGNORECASE,
)

# Non-English language requirements
_LANGUAGE_REQUIRED = re.compile(
    r"\b(japanese|korean|mandarin|cantonese|german|french|spanish|portuguese|"
    r"arabic|hindi|dutch|italian|russian|polish|swedish|norwegian|danish|"
    r"fluent in|native speaker|business proficiency in)\b.*?"
    r"\b(required|mandatory|must|essential|necessary)\b",
    re.IGNORECASE,
)
_LANGUAGE_REQUIRED_ALT = re.compile(
    r"\b(required|mandatory|must have|essential).*?\b"
    r"(japanese|korean|mandarin|cantonese|german|french)\b",
    re.IGNORECASE,
)

# Location gate: any of these anywhere in title, location, or description
# means the job MIGHT be workable from the owner's local area — let it through to scoring.
# Titles count because "(Remote)" suffixes are explicit employer signals even
# when the location names a posting-slot city ("New York, NY"). Hybrid
# passes deliberately; the knockout net (judge stage) catches hidden onsite
# requirements. Keep conservative: the gate exists to save tokens, not to
# make final calls.
_REMOTE_SIGNAL = re.compile(
    r"\b(remote|work from home|wfh|distributed|anywhere|hybrid)\b",
    re.IGNORECASE,
)
# "Tarrytown, NY" / "Aventura, FL" — a named US city+state.
_CITY_STATE = re.compile(r",\s*[A-Z]{2}\b")


def prefilter_job(title: str, location: str, description: str,
                  salary_min: Optional[float] = None,
                  salary_max: Optional[float] = None) -> Tuple[Optional[int], Optional[str]]:
    """
    Fast rules-based pre-filter. Returns (cap_score, reason) if the job should be
    capped, or (None, None) to proceed with LLM scoring.

    Caps are applied in order from most severe to least severe.
    """
    title = title or ""
    location = location or ""
    description = description or ""
    combined_text = f"{title} {description[:2000]}"  # only check start of description

    # 1. Clearly non-PM title (payroll, nurse, accountant…) → cap 5
    if _NON_PM_TITLES and _NON_PM_TITLES.search(title):
        return 5, f"Pre-filter: non-PM title '{title[:60]}'"

    # 2. Engineering title without any PM signal → cap 15
    if (_ENG_TITLE_KEYWORDS and _ENG_TITLE_KEYWORDS.search(title)
            and not (_PM_KEYWORDS and _PM_KEYWORDS.search(title))):
        return 15, f"Pre-filter: engineering title '{title[:60]}'"

    # 3. Non-PM adjacent roles (Business Analyst, Systems Analyst, etc.) → cap 30
    # These are not PM leadership roles; even a perfect remote/comp situation tops out ~30
    if (_NON_PM_ADJACENT_TITLES and _NON_PM_ADJACENT_TITLES.search(title)
            and not (_PM_KEYWORDS and _PM_KEYWORDS.search(title))):
        return 30, f"Pre-filter: non-PM adjacent title '{title[:60]}'"

    # 4. Non-English language fluency required → cap 10
    if _LANGUAGE_REQUIRED.search(combined_text) or _LANGUAGE_REQUIRED_ALT.search(combined_text):
        return 10, f"Pre-filter: non-English language required"

    # 5. Sales/BD titles without any product signal → cap 10
    if (_SALES_BD_TITLES and _SALES_BD_TITLES.search(title)
            and not (_PM_KEYWORDS and _PM_KEYWORDS.search(title))):
        return 10, f"Pre-filter: sales/BD title '{title[:60]}'"

    # 6. Solutions/CS titles without any product signal → cap 20
    if (_SOLUTIONS_CS_TITLES and _SOLUTIONS_CS_TITLES.search(title)
            and not (_PM_KEYWORDS and _PM_KEYWORDS.search(title))):
        return 20, f"Pre-filter: solutions/CS title '{title[:60]}'"

    # 7. Named non-remote, non-local-commuter-area location -> cap 10, no LLM call.
    # The owner is remote-only or local-commuter; a job that names another
    # place and shows zero remote signal is never applied to. Broad strings
    # ("United States", empty) pass through — they may be remote.
    # EXCEPTION (2026-07-13): posted top-of-band >= $300K exempts US onsite
    # roles (is_high_comp_exception — callers pass DB salary with regex
    # fallback); non-US high-comp roles stay gated.
    if (not _REMOTE_SIGNAL.search(title)
            and not _REMOTE_SIGNAL.search(location)
            and not _REMOTE_SIGNAL.search(description)
            and not (_LOCAL_AREA_RE and _LOCAL_AREA_RE.search(location))
            and (_CITY_STATE.search(location) or NON_US_LOCATIONS.search(location))
            and not is_high_comp_exception(salary_min, salary_max, location)):
        return 10, f"Pre-filter: non-remote non-local location '{location[:40]}'"

    # NOTE: Non-US locations are no longer hard-capped here. The rubric scores
    # remote_location 0-4 for non-US, which penalizes appropriately without
    # zeroing out strong role/AI/comp signals (e.g., a PM at Scale AI London).

    return None, None


# ---------------------------------------------------------------------------
# LLM Scorer
# ---------------------------------------------------------------------------

class LLMScorer:
    """
    Scores jobs using the local Claude CLI and caches results in SQLite.

    Usage:
        scorer = LLMScorer(db_path="data/jobs.db")
        if scorer.is_available():
            scored_count = scorer.apply_llm_scores_to_db(run_id=42)
    """

    def __init__(self, db_path: str, model: Optional[str] = None):
        self.db_path = db_path
        self.model = model or _llm_cfg.get("model") or DEFAULT_MODEL
        self._available: Optional[bool] = None  # lazily checked
        self._feedback_block = ""

        # Evidence base for the stage-2 judge (experience inventory, or the
        # master resume as a logged fallback). Empty basis disables judging:
        # rows keep NULL filter fields and are retried once a basis exists.
        self.judge_basis_text, self.judge_basis_sha, self.judge_basis = (
            load_inventory())

        # Per-pass usage accounting. Worker threads call _track_usage
        # concurrently, so mutation is lock-protected; the pass entry points
        # reset at start and log one summary line at the end.
        self._usage_lock = threading.Lock()
        self._usage = self._zero_usage()

    # ------------------------------------------------------------------
    # Availability check
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        """Return True if the `claude` CLI binary is on PATH."""
        if self._available is not None:
            return self._available
        self._available = is_cli_available()
        if not self._available:
            logger.info("`claude` CLI not found on PATH — LLM scoring disabled.")
        return self._available

    # ------------------------------------------------------------------
    # Per-pass usage accounting
    # ------------------------------------------------------------------

    @staticmethod
    def _zero_usage() -> dict:
        return {"calls": 0, "input_tokens": 0, "output_tokens": 0,
                "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0,
                "cost_usd": 0.0}

    def _reset_usage(self) -> None:
        with self._usage_lock:
            self._usage = self._zero_usage()

    def _track_usage(self, result: dict) -> None:
        """Fold one successful run_claude() result into the pass totals."""
        usage = result.get("usage") or {}
        with self._usage_lock:
            self._usage["calls"] += 1
            for key in ("input_tokens", "output_tokens",
                        "cache_read_input_tokens",
                        "cache_creation_input_tokens"):
                self._usage[key] += int(usage.get(key) or 0)
            self._usage["cost_usd"] += float(result.get("cost_usd") or 0.0)

    def _log_usage_summary(self, label: str) -> None:
        """One line per pass: call count, token totals, cache hits, notional $."""
        with self._usage_lock:
            u = dict(self._usage)
        if not u["calls"]:
            return
        logger.info(
            "LLM usage [%s]: %d calls | in=%d (cache_read=%d, cache_write=%d) "
            "| out=%d | notional $%.2f",
            label, u["calls"], u["input_tokens"],
            u["cache_read_input_tokens"], u["cache_creation_input_tokens"],
            u["output_tokens"], u["cost_usd"])

    # ------------------------------------------------------------------
    # Main scoring entry point
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Dimension cap helper (used by both scoring passes)
    # ------------------------------------------------------------------

    def _apply_caps(
        self,
        dims: dict,
        job_title: str,
        job_location: str,
        description: str,
        known_sal_max: Optional[float],
    ) -> dict:
        """Apply rules-based caps to LLM dimension scores in-place and return dims."""
        # remote_location
        if NON_US_LOCATIONS.search(job_location):
            dims["remote_location"] = min(dims["remote_location"], 4)
        elif _LOCAL_AREA_RE and _LOCAL_AREA_RE.search(job_location):
            dims["remote_location"] = 20

        # seniority_match
        if _DIRECTOR_TITLE.search(job_title):
            pass  # allow up to 20
        elif _SENIOR_TITLE.search(job_title):
            dims["seniority_match"] = min(dims["seniority_match"], 14)
        elif _MANAGER_TITLE.search(job_title):
            dims["seniority_match"] = min(dims["seniority_match"], 11)
        elif _ANALYST_IC_TITLE.search(job_title):
            dims["seniority_match"] = min(dims["seniority_match"], 8)
        else:
            dims["seniority_match"] = min(dims["seniority_match"], 11)

        # comp_match
        if known_sal_max:
            if known_sal_max > MAX_BASE_SALARY:
                pass  # implausibly high — likely a data error; leave LLM score untouched
            elif known_sal_max < COMP_CAP_LOW:
                dims["comp_match"] = min(dims["comp_match"], 2)
            elif known_sal_max < COMP_CAP_MID:
                dims["comp_match"] = min(dims["comp_match"], 4)
            elif known_sal_max < COMP_TARGET:
                dims["comp_match"] = min(dims["comp_match"], 6)
            else:  # >= COMP_TARGET and plausible
                dims["comp_match"] = max(dims["comp_match"], 9)

        # ai_domain_fit — scan the same window the LLM prompt saw (_LLM_DESC_CHARS),
        # so the cap only fires when the model genuinely had no AI signal in its input.
        # No signal list configured (neutral tree) → the model's score stands.
        if _AI_SIGNAL_KEYWORDS is not None:
            check_text = f"{job_title} {description[:_LLM_DESC_CHARS]}"
            if not _AI_SIGNAL_KEYWORDS.search(check_text):
                dims["ai_domain_fit"] = min(dims["ai_domain_fit"], 8)

        return dims

    # ------------------------------------------------------------------
    # Filter Match v2 — stage 2: judge extracted requirements
    # ------------------------------------------------------------------

    def _judge_filter(self, job_title: str, filter_raw: dict,
                      salary_min: Optional[float] = None,
                      salary_max: Optional[float] = None) -> Optional[dict]:
        """Stage 2: judge extracted requirements against the inventory.

        Returns the aligned+softened judge dict, or None on any failure
        (missing basis, CLI error after retries, unparseable JSON) — the
        caller leaves filter fields NULL so backfill retries later.
        """
        if not self.judge_basis_text:
            return None
        prompt = build_judge_prompt(
            job_title, filter_raw["title_variants"],
            filter_raw["must_have_keywords"],
            [k["requirement"] for k in filter_raw["knockouts"]],
            posted_top_of_band=max(salary_min or 0, salary_max or 0) or None)
        # Rebuilt per call from the CURRENT basis text (tests swap
        # judge_basis_text after construction); identical content per run, so
        # the CLI's system-prompt cache breakpoint still matches across calls.
        judge_system = build_judge_system(self.judge_basis_text)

        @_retry_transient_cli
        def _invoke():
            return run_claude(prompt, model=self.model,
                              system_prompt=judge_system, timeout=60.0)

        try:
            result = _invoke()
        except ClaudeCLIError as e:
            logger.warning("Judge call failed for '%s': %s", job_title, e)
            return None
        self._track_usage(result)
        judged = parse_judge_response(
            result["text"], filter_raw["must_have_keywords"],
            filter_raw["knockouts"])
        if judged is None:
            logger.warning("Judge response unparseable for '%s'", job_title)
            return None
        judged["knockouts"] = apply_soft_years(judged["knockouts"])
        return judged

    # ------------------------------------------------------------------
    # Per-job worker (called from thread pool)
    # ------------------------------------------------------------------

    def _score_one_job(self, row) -> Optional[tuple]:
        """
        Score a single job row. Returns (job_id, final_score, llm_score,
        explanation, sal_min, sal_max, est_min, est_max, filter_fields) or
        None on failure.
        """
        job_id = row["id"]
        kw_score = row["score"] or 0
        description = clean_for_llm(row["description"] or "")

        regex_sal_min, regex_sal_max = extract_salary_regex(description)

        cap_score, cap_reason = prefilter_job(
            row["title"], row["location"] or "", description,
            salary_min=row["salary_min"] or regex_sal_min,
            salary_max=row["salary_max"] or regex_sal_max,
        )

        if cap_score is not None and cap_score <= 15:
            # Hard skip: clearly irrelevant jobs (engineering titles, non-PM, language barriers).
            # No point spending a Claude CLI call — use the cap directly.
            logger.info(
                "Pre-filter skip (cap=%d): '%s' — %s",
                cap_score, row["title"], cap_reason,
            )
            llm_score = cap_score
            explanation = cap_reason
            llm_sal_min, llm_sal_max = None, None
            est_min, est_max = None, None
            filter_raw = None
        else:
            sal_min_hint = row["salary_min"] or regex_sal_min
            sal_max_hint = row["salary_max"] or regex_sal_max
            llm_score, explanation, llm_sal_min, llm_sal_max, est_min, est_max, dims, filter_raw = self._call_claude(
                row["title"], row["company"], description, temperature=0.1,
                location=row["location"] or "",
                salary_min=sal_min_hint,
                salary_max=sal_max_hint,
            )

            if dims is not None:
                known_sal_max = row["salary_max"] or regex_sal_max or llm_sal_max
                dims = self._apply_caps(
                    dims, row["title"] or "", row["location"] or "",
                    description, known_sal_max,
                )
                llm_score = sum(dims.values())
                dim_str = (
                    f"role={dims['role_match']}/30 "
                    f"seniority={dims['seniority_match']}/20 "
                    f"location={dims['remote_location']}/20 "
                    f"ai={dims['ai_domain_fit']}/20 "
                    f"comp={dims['comp_match']}/10"
                )
                explanation = re.sub(r"^\[role=.*?\] ", f"[{dim_str}] ", explanation or "")
                if not explanation.startswith("["):
                    explanation = f"[{dim_str}] {explanation}"

            # Apply soft cap for higher pre-filter caps (20, 30) — LLM ran but score is ceilinged.
            if cap_score is not None and llm_score is not None and llm_score > cap_score:
                logger.info(
                    "Pre-filter cap applied (cap=%d, llm=%d): '%s'",
                    cap_score, llm_score, row["title"],
                )
                llm_score = cap_score
                if explanation and not explanation.startswith("[Pre-filter"):
                    explanation = f"[Pre-filter cap={cap_score}] {explanation}"

        # Blended final score, computed once — reused by the judge gate below
        # and returned as this job's score.
        blended = (min(100, round(_KW_WEIGHT * kw_score + _LLM_WEIGHT * llm_score))
                   if llm_score is not None else None)

        # Default to the sentinel; a successful judge pass replaces it, a
        # judge FAILURE clears it to all-NULL (row stays in the backfill
        # target set and is retried — never half-written, never fabricated).
        filter_fields = _FILTER_SENTINEL
        if _FILTER_MATCH_ENABLED and filter_raw is not None and filter_raw["must_have_keywords"]:
            if blended is not None and blended < _JUDGE_MIN_SCORE:
                # Score gate: this job can never surface (alerts >= 60,
                # auto-tailor score AND filter >= 60), so a judge call is
                # wasted spend. The
                # sentinel marks it attempted; it renders "—" like an
                # extraction miss.
                logger.info(
                    "Judge gate: blended %d < %d — skipping stage-2 judge "
                    "for '%s'", blended, _JUDGE_MIN_SCORE, row["title"])
            else:
                judged = self._judge_filter(
                    row["title"] or "", filter_raw,
                    salary_min=row["salary_min"] or regex_sal_min,
                    salary_max=row["salary_max"] or regex_sal_max)
                if judged is None:
                    filter_fields = None
                else:
                    fscore, knocked_out, uncapped = compute_judged_filter_score(
                        judged["must_haves"], judged["title_claim"],
                        judged["knockouts"])
                    fjson = build_filter_json_v2(
                        judged["must_haves"], filter_raw["title_variants"],
                        filter_raw["title_alignment"], judged["title_claim"],
                        judged["knockouts"], uncapped,
                        self.judge_basis_sha, self.judge_basis)
                    filter_fields = (fscore, fscore, "master", int(knocked_out), fjson)
        elif _FILTER_MATCH_ENABLED and filter_raw is not None:
            # Valid filter block but no keywords — genuine extraction miss.
            # The sentinel (not NULL) is written so this row won't be re-billed.
            logger.warning(
                "Filter extraction miss: LLM returned no must-have keywords "
                "for '%s' at %s — writing filter sentinel",
                row["title"], row["company"])

        # Include LLM salary if Claude CLI was actually called (cap skipped or cap > 15)
        llm_ran = cap_score is None or cap_score > 15
        sal_min = row["salary_min"] or (llm_sal_min if llm_ran else None) or regex_sal_min
        sal_max = row["salary_max"] or (llm_sal_max if llm_ran else None) or regex_sal_max

        if llm_score is None:
            return None

        return (job_id, blended, llm_score, explanation, sal_min, sal_max,
                est_min, est_max, filter_fields)

    # ------------------------------------------------------------------
    # Main scoring entry point
    # ------------------------------------------------------------------

    @_with_scoring_lock
    def apply_llm_scores_to_db(self, run_id: Optional[int] = None, workers: Optional[int] = None,
                               profile: str = PROFILE_KEY, backfill_limit: Optional[int] = None,
                               limit: Optional[int] = None, ids: Optional[list] = None) -> int:
        """
        Score all unscored jobs in DB (optionally limited to run_id), scoped to profile.
        Uses a thread pool for concurrent Claude CLI calls.
        Returns count of jobs scored by LLM.

        When ``limit`` is set alongside ``run_id``, scores at most that many of the
        run's unscored jobs, HIGHEST keyword score first — so a high-yield run can't
        burn the whole subscription usage window in one burst; the lower-relevance
        tail stays keyword-only and is drained over later runs by ``backfill_limit``.

        When ``backfill_limit`` is set (and ``run_id`` is None), scores at most that
        many genuinely-unscored rows from prior runs, oldest first, skipping expired
        and terminal-status rows — so a transient CLI failure self-heals on later
        runs without an unbounded catch-up.

        When ``ids`` is given, restricts scoring to those job ids (still only the
        NULL-score ones among them) — used by targeted one-off backfills that must
        not drain the whole NULL pool. Composes with the other scopes.
        """
        _require_resume_summary()
        if workers is None:
            workers = int(_llm_cfg.get("workers", 1))
        if not self.is_available():
            logger.info("claude CLI unavailable — skipping LLM scoring pass.")
            self._extract_salaries_regex_only(run_id)
            return 0

        self._reset_usage()
        more, less = _load_feedback_examples(self.db_path, profile)
        self._feedback_block = format_feedback_block(more, less)
        if self._feedback_block:
            logger.info("LLM scoring calibrated with %d 'more' / %d 'less' feedback examples.",
                        len(more), len(less))

        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        q = (
            "SELECT id, title, company, location, description, salary_min, salary_max, score "
            "FROM jobs WHERE llm_score IS NULL AND profile = ?"
        )
        params: list = [profile]
        if ids:
            q += f" AND id IN ({','.join('?' * len(ids))})"
            params.extend(ids)
        if run_id is not None:
            q += " AND run_id = ?"
            params.append(run_id)
            if limit is not None:
                # Best-first cap: score the highest-keyword-score jobs, bounded.
                q += " ORDER BY score DESC LIMIT ?"
                params.append(limit)
        elif backfill_limit is not None:
            # Retry only live failures, oldest first, bounded — never the thousands
            # of stale/expired NULLs (no value re-scoring dead or acted-on listings).
            q += (" AND status NOT IN ('expired', 'applied', 'not_interested') "
                  "ORDER BY created_at ASC LIMIT ?")
            params.append(backfill_limit)
        rows = conn.execute(q, params).fetchall()
        conn.close()

        total = len(rows)
        logger.info("Scoring %d unscored jobs with %d workers.", total, workers)
        count = self._score_rows(rows, workers)
        logger.info("LLM scored %d / %d unscored jobs.", count, total)
        self._log_usage_summary("scoring pass")
        return count

    def _score_rows(self, rows: list, workers: int, stop=None) -> int:
        """Score a batch of SELECTed rows through the worker pool; returns
        how many results actually landed (guard-skipped stale writes are
        excluded — see _write_results)."""
        total = len(rows)
        # Guard values: the scoring inputs as SELECTed. If a row's inputs
        # change mid-call (streaming reconcile swap), the result is stale.
        # Title/company/location are inputs too — a URL-rank swap can change
        # them while description+salary stay byte-identical.
        guards = {row["id"]: (row["description"] or "",
                              row["salary_min"], row["salary_max"],
                              row["title"] or "", row["company"] or "",
                              row["location"] or "")
                  for row in rows}

        # Workers only READ; the calling thread does all WRITES, flushing
        # every _FLUSH_EVERY completions so a mid-pass crash forfeits at most
        # one batch of CLI spend instead of the whole pass.
        pending: list = []
        count = 0
        done = 0

        def _score_gated(row):
            # Worker-side stop gate: future.cancel() loses the dequeue race
            # to a freed worker, so the worker itself must refuse to start a
            # CLI call once the run is dead.
            if stop is not None and stop.is_set():
                return None
            return self._score_one_job(row)

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_score_gated, row): row for row in rows}
            for future in as_completed(futures):
                if stop is not None and stop.is_set():
                    for f in futures:
                        f.cancel()
                if future.cancelled():
                    continue
                done += 1
                try:
                    result = future.result()
                    if result is not None:
                        pending.append(result)
                except Exception as exc:
                    row = futures[future]
                    logger.warning("Error scoring '%s': %s", row["title"], exc)
                if len(pending) >= _FLUSH_EVERY:
                    count += self._write_results(pending, guards=guards)
                    pending = []
                if done % 50 == 0 or done == total:
                    logger.info("Progress: %d / %d scored so far.", done, total)

        count += self._write_results(pending, guards=guards)
        return count

    @_with_scoring_lock
    def apply_llm_scores_streaming(self, run_id: int, drained,
                                   profile: str = PROFILE_KEY,
                                   limit: Optional[int] = None,
                                   reserve: int = 0,
                                   workers: Optional[int] = None,
                                   stop=None) -> tuple:
        """Score a run's NULL rows in waves WHILE the scrape is still saving
        batches. Holds the scoring lock ONCE for the whole phase — repeated
        apply_llm_scores_to_db calls would re-acquire per pass and skip if
        e.g. a dashboard rejudge grabbed the lock between increments.

        Budget: scores at most (limit - reserve) rows; the caller runs the
        best-first top-up over the remaining budget after the final save —
        so when max_jobs_per_run binds, the highest-keyword tail keeps its
        guaranteed slots exactly like the old single best-first pass.

        Budget counts ATTEMPTS, not successes — max_jobs_per_run protects
        the subscription window, and a failure storm must not pull unlimited
        replacement rows into the waves. Returns (scored, attempted).

        Stops when `drained` is set (caller: AFTER the last batch was saved)
        and a subsequent SELECT finds no scorable rows — or immediately when
        `stop` is set (dead run: no further budget may be spent; in-flight
        CLI calls finish, pending ones are cancelled). Rows whose single
        attempt failed or was guard-skipped are left NULL for the caller's
        top-up pass — never retried in-loop, so a poison row can't spin CLI
        spend."""
        _require_resume_summary()
        if workers is None:
            workers = int(_llm_cfg.get("workers", 1))
        if not self.is_available():
            logger.info("claude CLI unavailable — skipping streaming LLM scoring pass "
                        "(the caller's post-scrape fallback handles salary regex).")
            return 0, 0

        self._reset_usage()
        more, less = _load_feedback_examples(self.db_path, profile)
        self._feedback_block = format_feedback_block(more, less)
        if self._feedback_block:
            logger.info("LLM scoring calibrated with %d 'more' / %d 'less' feedback examples.",
                        len(more), len(less))

        logger.info("Streaming LLM scoring started for run %s (limit=%s, reserve=%s, workers=%d).",
                    run_id, limit, reserve, workers)
        scored = 0
        attempted: set = set()
        while True:
            if stop is not None and stop.is_set():
                logger.info("Streaming LLM scoring aborted (run failed).")
                break
            budget = None
            if limit is not None:
                budget = max(0, limit - reserve - len(attempted))
                if budget == 0:
                    break
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            q = ("SELECT id, title, company, location, description, salary_min, "
                 "salary_max, score FROM jobs "
                 "WHERE llm_score IS NULL AND profile = ? AND run_id = ? "
                 "ORDER BY score DESC")
            params: list = [profile, run_id]
            if budget is not None:
                # attempted-but-failed rows still occupy SELECT slots; over-
                # fetch by their count so fresh rows are never starved.
                q += " LIMIT ?"
                params.append(budget + len(attempted))
            rows = conn.execute(q, params).fetchall()
            conn.close()
            rows = [r for r in rows if r["id"] not in attempted]
            if budget is not None:
                rows = rows[:budget]
            if not rows:
                if drained.is_set():
                    break
                time.sleep(1.0)
                continue
            attempted.update(r["id"] for r in rows)
            logger.info("Streaming wave: %d rows.", len(rows))
            scored += self._score_rows(rows, workers, stop=stop)
        logger.info("Streaming LLM scoring finished: %d scored / %d attempted.",
                    scored, len(attempted))
        self._log_usage_summary("streaming scoring pass")
        return scored, len(attempted)

    def _write_results(self, results: list, guards: Optional[dict] = None) -> int:
        """Batch-write scoring results on the calling thread (single writer).

        guards maps job_id -> (description, salary_min, salary_max) as
        SELECTed for scoring. A row whose stored inputs no longer match was
        reconciled to a richer copy mid-call — the stale result is discarded
        (llm_score stays NULL, so a later pass rescores the fresh inputs).
        Returns how many rows were actually written."""
        if not results:
            return 0
        written = 0
        wconn = sqlite3.connect(self.db_path)
        wconn.execute("PRAGMA busy_timeout=30000")
        for (job_id, final_score, llm_score, explanation,
             sal_min, sal_max, est_min, est_max, filter_fields) in results:
            f_score, f_master, f_source, f_knock, f_json = (
                filter_fields if filter_fields is not None
                else (None, None, None, None, None))
            sql = """UPDATE jobs
                   SET score = ?,
                       llm_score = ?,
                       llm_explanation = ?,
                       salary_min = COALESCE(salary_min, ?),
                       salary_max = COALESCE(salary_max, ?),
                       salary_est_min = COALESCE(?, salary_est_min),
                       salary_est_max = COALESCE(?, salary_est_max),
                       filter_score = ?,
                       filter_score_master = ?,
                       filter_source = ?,
                       filter_knockout = ?,
                       filter_json = ?
                   WHERE id = ?"""
            args = [final_score, llm_score, explanation, sal_min, sal_max,
                    est_min, est_max, f_score, f_master, f_source, f_knock,
                    f_json, job_id]
            if guards is not None and job_id in guards:
                g_desc, g_min, g_max, g_title, g_company, g_loc = guards[job_id]
                sql += (" AND COALESCE(description, '') = ?"
                        " AND salary_min IS ? AND salary_max IS ?"
                        " AND COALESCE(title, '') = ?"
                        " AND COALESCE(company, '') = ?"
                        " AND COALESCE(location, '') = ?")
                args.extend([g_desc, g_min, g_max, g_title, g_company, g_loc])
            cur = wconn.execute(sql, args)
            if cur.rowcount > 0:
                written += 1
            else:
                logger.info(
                    "Job %d changed during scoring (reconciled to a richer "
                    "copy) — stale result discarded; a later pass rescores it.",
                    job_id)
        wconn.commit()
        wconn.close()
        return written

    @_with_scoring_lock
    def backfill_filter(self, profile: str = PROFILE_KEY,
                        workers: Optional[int] = None,
                        since_hours: Optional[int] = None) -> int:
        """Compute Filter Match for already-scored jobs that lack it.

        Targets rows with no filter_json (or the `'{}'` sentinel written
        while the feature was disabled), a real description, and a live
        status; re-applies prefilter_job and skips rejects (they would never
        yield filter fields — excluding them keeps re-runs cheap and the scan
        convergent). Resumable: written rows leave the target set. Same
        retry/backoff as normal scoring (tenacity inside _call_claude).

        Convergence: a re-scored sentinel row either gains real filter fields
        (leaving the target set for good) or an extraction miss writes '{}'
        again (idempotent — no infinite churn). Only rows the prefilter
        accepts are attempted, so each run's CLI spend is bounded by genuine
        candidates, not by how many sentinel rows have piled up.

        Rows below the judge gate (jobs.score < _JUDGE_MIN_SCORE) are excluded
        entirely: their filter data could never surface, and re-scoring a
        score-gated sentinel row would re-bill its stage-1 call on every
        backfill run. For never-LLM-scored rows jobs.score is the keyword-only
        score — the best available estimate; a borderline row that later
        blends above the gate re-enters this target set once scored.

        Note: this calls _score_one_job (the full scoring path), so it also
        refreshes llm_score and salary estimates for the affected rows, not
        just the filter columns.

        since_hours: when given, further restricts the target set to rows
        created within the last N hours (an additional filter only — every
        other predicate above still applies). None (default) scopes to the
        full history, same as before this parameter existed.
        """
        _require_resume_summary()
        if workers is None:
            workers = int(_llm_cfg.get("workers", 1))
        if not self.is_available():
            logger.info("claude CLI unavailable — cannot backfill filter scores.")
            return 0

        self._reset_usage()
        more, less = _load_feedback_examples(self.db_path, profile)
        self._feedback_block = format_feedback_block(more, less)

        q = ("SELECT id, title, company, location, description, salary_min, "
             "salary_max, score FROM jobs "
             "WHERE profile = ? AND (filter_json IS NULL OR filter_json = '{}') "
             "AND description IS NOT NULL AND LENGTH(description) > 200 "
             "AND status NOT IN ('expired', 'applied', 'not_interested') "
             "AND score >= ?")
        params = [profile, _JUDGE_MIN_SCORE]
        if since_hours is not None:
            q += " AND created_at >= datetime('now', ?)"
            params.append(f"-{int(since_hours)} hours")

        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(q, params).fetchall()
        conn.close()

        eligible = []
        skipped = 0
        for row in rows:
            desc = clean_for_llm(row["description"] or "")
            regex_sal_min, regex_sal_max = extract_salary_regex(desc)
            cap, _reason = prefilter_job(
                row["title"], row["location"] or "", desc,
                salary_min=row["salary_min"] or regex_sal_min,
                salary_max=row["salary_max"] or regex_sal_max)
            if cap is not None and cap <= 15:
                skipped += 1
                continue
            eligible.append(row)
        logger.info(
            "Backfill-filter: %d candidates (%d prefilter-rejected and skipped).",
            len(eligible), skipped)

        results: list = []
        done = 0
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(self._score_one_job, row): row
                       for row in eligible}
            for future in as_completed(futures):
                done += 1
                try:
                    result = future.result()
                    if result is not None and result[-1] is not None:
                        results.append(result)
                except Exception as exc:
                    row = futures[future]
                    logger.warning("Error backfilling '%s': %s", row["title"], exc)
                if done % 50 == 0 or done == len(eligible):
                    logger.info("Backfill progress: %d / %d.", done, len(eligible))

        self._write_results(results)
        logger.info("Backfill-filter: wrote filter fields for %d jobs.", len(results))
        self._log_usage_summary("backfill-filter pass")
        return len(results)

    @_with_scoring_lock
    def rejudge_filter(self, profile: str = PROFILE_KEY,
                       workers: Optional[int] = None,
                       force_all: bool = False,
                       since_hours: Optional[int] = None,
                       ids: Optional[list] = None) -> int:
        """Stage-2-only re-judge for rows whose judgment is stale.

        Targets live master-basis rows with a real stage-1 extraction in
        filter_json, skipping rows already judged against the current
        inventory (inventory_sha256 match) unless force_all. Tailored rows
        are never touched — their realized literal score is frozen. Sentinel
        rows ('{}', no stage-1 extraction) are never touched either — that's
        backfill_filter's job, not this one's. Writes ONLY the filter
        columns; llm_score and salaries are untouched, which is what makes
        inventory edits cheap: one judge call per row, no fit re-scoring.

        since_hours: when given, further restricts the target set to rows
        created within the last N hours (an additional filter only — every
        other predicate above still applies). None (default) scopes to the
        full history, same as before this parameter existed.

        ids: optional list of job ids — the content-scoped surgical
        re-judge (e.g. after an inventory rule change that affects a known
        row set). Scopes the query; the inventory-sha freshness skip still
        applies, so re-runs are idempotent.
        """
        _require_resume_summary()
        if workers is None:
            workers = int(_llm_cfg.get("workers", 1))
        if not self.is_available():
            logger.info("claude CLI unavailable — cannot re-judge.")
            return 0
        if not self.judge_basis_text:
            logger.error("No experience inventory or master resume cache — "
                         "nothing to judge against.")
            return 0
        self._reset_usage()

        q = ("SELECT id, title, filter_json, salary_min, salary_max, "
             "description FROM jobs "
             "WHERE profile = ? AND filter_json IS NOT NULL "
             "AND filter_json != '{}' AND filter_source = 'master' "
             "AND status NOT IN ('expired', 'applied', 'not_interested')")
        params = [profile]
        if since_hours is not None:
            q += " AND created_at >= datetime('now', ?)"
            params.append(f"-{int(since_hours)} hours")
        if ids:
            q += f" AND id IN ({','.join('?' * len(ids))})"
            params.extend(ids)

        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(q, params).fetchall()
        conn.close()

        targets = []
        skipped_fresh = 0
        for row in rows:
            try:
                blob = json.loads(row["filter_json"])
            except (TypeError, json.JSONDecodeError):
                continue
            must_haves = blob.get("must_haves")
            if not isinstance(must_haves, list) or not must_haves:
                continue
            if not force_all and blob.get("inventory_sha256") == self.judge_basis_sha:
                skipped_fresh += 1
                continue
            filter_raw = {
                "must_have_keywords": [
                    {"term": m.get("term", ""),
                     "aliases": m.get("aliases") or []}
                    for m in must_haves if m.get("term")],
                "knockouts": [
                    {"requirement": k.get("requirement", "")}
                    for k in (blob.get("knockouts") or [])
                    if isinstance(k, dict) and k.get("requirement")],
                "title_variants": blob.get("title_variants") or [],
                "title_alignment": blob.get("title_alignment", "none"),
            }
            regex_sal_min, regex_sal_max = extract_salary_regex(
                row["description"] or "")
            targets.append((row["id"], row["title"] or "",
                            row["salary_min"] or regex_sal_min,
                            row["salary_max"] or regex_sal_max, filter_raw))
        logger.info("Rejudge-filter: %d stale rows (%d already current).",
                    len(targets), skipped_fresh)

        def _one(job_id, title, sal_min, sal_max, filter_raw):
            judged = self._judge_filter(title, filter_raw,
                                        salary_min=sal_min,
                                        salary_max=sal_max)
            if judged is None:
                return None
            fscore, knocked_out, uncapped = compute_judged_filter_score(
                judged["must_haves"], judged["title_claim"],
                judged["knockouts"])
            fjson = build_filter_json_v2(
                judged["must_haves"], filter_raw["title_variants"],
                filter_raw["title_alignment"], judged["title_claim"],
                judged["knockouts"], uncapped,
                self.judge_basis_sha, self.judge_basis)
            return (fscore, fscore, int(knocked_out), fjson, job_id)

        updates = []
        done = 0
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_one, jid, title, smin, smax, fr): jid
                       for jid, title, smin, smax, fr in targets}
            for future in as_completed(futures):
                done += 1
                try:
                    u = future.result()
                    if u is not None:
                        updates.append(u)
                except Exception as exc:
                    logger.warning("Rejudge error (job %s): %s",
                                   futures[future], exc)
                if done % 50 == 0 or done == len(targets):
                    logger.info("Rejudge progress: %d / %d.", done, len(targets))

        if updates:
            wconn = sqlite3.connect(self.db_path)
            wconn.executemany(
                "UPDATE jobs SET filter_score = ?, filter_score_master = ?, "
                "filter_knockout = ?, filter_json = ? WHERE id = ?",
                updates)
            wconn.commit()
            wconn.close()
        logger.info("Rejudge-filter: re-judged %d jobs.", len(updates))
        self._log_usage_summary("rejudge-filter pass")
        return len(updates)

    def rescore_all(self, force: bool = False, profile: str = PROFILE_KEY,
                    dry_run: bool = False) -> int:
        """Score jobs in the DB that don't yet have an LLM score, scoped to profile.

        Args:
            force: When True, clears all existing LLM scores for the profile first
                   so every job is re-evaluated. Use after changing scoring criteria.
                   When False (default), only processes NULL scores — safe to
                   interrupt and resume.
            profile: Only process jobs belonging to this profile.
            dry_run: Report how many jobs the pass would process and return —
                   no writes, no CLI calls, no billing.
        """
        _require_resume_summary()
        conn = sqlite3.connect(self.db_path)
        if dry_run:
            if force:
                count = conn.execute(
                    "SELECT COUNT(*) FROM jobs WHERE profile = ?", (profile,)
                ).fetchone()[0]
            else:
                count = conn.execute(
                    "SELECT COUNT(*) FROM jobs WHERE llm_score IS NULL AND profile = ?",
                    (profile,),
                ).fetchone()[0]
            conn.close()
            logger.info(
                "RESCORE dry-run: %d jobs would be %s (profile=%s). Nothing written.",
                count, "cleared and re-scored" if force else "scored", profile)
            return count
        if force:
            # jobs.score holds the blended value round(_KW*kw + _LLM*llm). Recover
            # the keyword-only baseline BEFORE clearing llm_score, otherwise the
            # rescore would blend the already-blended score into itself (compounding).
            conn.execute(
                "UPDATE jobs SET score = CAST(ROUND((score - ? * llm_score) / ?) AS INTEGER) "
                "WHERE profile = ? AND llm_score IS NOT NULL",
                (_LLM_WEIGHT, _KW_WEIGHT, profile),
            )
            cleared = conn.execute(
                "UPDATE jobs SET llm_score = NULL, llm_explanation = NULL WHERE profile = ?",
                (profile,),
            ).rowcount
            conn.commit()
            logger.info("RESCORE-FORCE: cleared LLM scores for %d jobs (profile=%s).", cleared, profile)
        remaining = conn.execute(
            "SELECT COUNT(*) FROM jobs WHERE llm_score IS NULL AND profile = ?", (profile,)
        ).fetchone()[0]
        conn.close()
        logger.info("RESCORE-ALL: %d jobs with llm_score IS NULL (profile=%s) — starting scoring pass.", remaining, profile)
        return self.apply_llm_scores_to_db(profile=profile)

    def reblend_all(self, profile: str = PROFILE_KEY,
                    config: Optional[dict] = None,
                    dry_run: bool = False) -> int:
        """Recompute jobs.score WITHOUT any LLM calls (profile-scoped).

        Re-runs the keyword scorer against each stored posting — so scorer-rule
        changes such as the location cap take effect — and re-blends it with the
        row's EXISTING llm_score using the current weights. Rows with no
        llm_score keep the pure keyword score. llm_score, the filter columns and
        salary estimates are left untouched. Returns the number of rows whose
        score changed. Applies the same scoring.layoff_penalty as save-time scoring.

        Unlike rescore_all(force=True), this spends zero subscription budget —
        use it after changing keyword rules or blend weights. config defaults to
        the project config.yaml (needed for local_locations / alert_threshold);
        callers may inject a dict for testing.
        """
        from engine.scorer import (JobScorer, apply_layoff_penalty,
                                   load_layoff_companies)
        from scrapers.base import JobPosting

        if config is None:
            config_path = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
            with open(config_path, "r") as fh:
                config = yaml.safe_load(fh) or {}
        kw_scorer = JobScorer(config)

        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        # Same layoff policy as save_jobs — without this, every reblend
        # silently strips the penalty from the whole corpus.
        layoff_penalty = int((config.get("scoring") or {}).get("layoff_penalty", 0) or 0)
        layoff_companies = load_layoff_companies(conn) if layoff_penalty else set()
        rows = conn.execute(
            "SELECT id, title, company, location, url, description, "
            "salary_min, salary_max, score, llm_score FROM jobs WHERE profile = ?",
            (profile,),
        ).fetchall()

        updated = 0
        for row in rows:
            job = JobPosting(
                title=row["title"], company=row["company"],
                location=row["location"] or "", url=row["url"],
                description=row["description"] or "",
                salary_min=row["salary_min"], salary_max=row["salary_max"],
            )
            kw = kw_scorer.score(job)
            kw = apply_layoff_penalty(kw, row["company"], layoff_companies,
                                      layoff_penalty)
            if row["llm_score"] is not None:
                new_score = min(100, round(_KW_WEIGHT * kw + _LLM_WEIGHT * row["llm_score"]))
            else:
                new_score = kw
            if new_score != row["score"]:
                if not dry_run:
                    conn.execute("UPDATE jobs SET score = ? WHERE id = ?",
                                 (new_score, row["id"]))
                updated += 1
        if not dry_run:
            conn.commit()
        conn.close()
        logger.info("REBLEND%s: %d/%d jobs %s (profile=%s) — no LLM calls.",
                    " DRY-RUN" if dry_run else "", updated, len(rows),
                    "would change" if dry_run else "changed", profile)
        return updated

    def print_rescore_sample(self, n: int = 5, profile: str = PROFILE_KEY) -> None:
        """
        Fetch n jobs that already have llm_score (scoped to profile), rescore with new system,
        and print a side-by-side comparison.
        """
        if not self.is_available():
            print("claude CLI unavailable — cannot run sample rescore.")
            return

        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT id, title, company, location, description, salary_min, salary_max, "
            "score, llm_score, llm_explanation "
            "FROM jobs WHERE llm_score IS NOT NULL AND profile = ? ORDER BY llm_score DESC LIMIT ?",
            (profile, n),
        ).fetchall()
        conn.close()

        if not rows:
            print("No previously-scored jobs found for sample comparison.")
            return

        print(f"\n{'=' * 70}")
        print(f"  RESCORE SAMPLE — New ({self.model})")
        print(f"{'=' * 70}\n")

        for row in rows:
            description = clean_for_llm(row["description"] or "")
            regex_sal_min, regex_sal_max = extract_salary_regex(description)
            cap_score, cap_reason = prefilter_job(
                row["title"], row["location"] or "", description,
                salary_min=row["salary_min"] or regex_sal_min,
                salary_max=row["salary_max"] or regex_sal_max)

            if cap_score is not None:
                new_llm = cap_score
                new_expl = cap_reason
            else:
                new_llm, new_expl, _, _, _, _, _, _ = self._call_claude(
                    row["title"], row["company"], description, temperature=0.1,
                    location=row["location"] or "",
                )

            old_kw = round((row["score"] - _LLM_WEIGHT * row["llm_score"]) / _KW_WEIGHT) if row["llm_score"] else row["score"]
            new_final = min(100, round(_KW_WEIGHT * old_kw + _LLM_WEIGHT * new_llm)) if new_llm is not None else "N/A"

            print(f"  Job: {row['title'][:55]} @ {row['company'][:25]}")
            print(f"  Old LLM score : {row['llm_score']:>3}  →  Old final score : {row['score']:>3}")
            print(f"  New LLM score : {str(new_llm):>3}  →  New final score : {str(new_final):>3}")
            print(f"  Old explanation: {(row['llm_explanation'] or '')[:100]}")
            print(f"  New explanation: {(new_expl or '')[:100]}")
            print()

        print(f"{'=' * 70}\n")

    # ------------------------------------------------------------------
    # Claude CLI call
    # ------------------------------------------------------------------

    def _call_claude(
        self,
        title: str,
        company: str,
        description: str,
        temperature: float = 0.1,   # accepted for call-site compatibility; the CLI cannot set sampling
        location: str = "",
        salary_min: Optional[float] = None,
        salary_max: Optional[float] = None,
    ) -> Tuple[Optional[int], Optional[str], Optional[float], Optional[float],
               Optional[float], Optional[float], Optional[dict], Optional[dict]]:
        """Call the Claude CLI and return (llm_score, explanation, sal_min, sal_max,
        est_min, est_max, dims, filter_raw)."""
        if salary_min and salary_max:
            salary_info = f"${salary_min/1000:.0f}K–${salary_max/1000:.0f}K/yr"
        elif salary_max:
            salary_info = f"up to ${salary_max/1000:.0f}K/yr"
        elif salary_min:
            salary_info = f"from ${salary_min/1000:.0f}K/yr"
        else:
            salary_info = "not stated"

        system_prompt = _FIT_SYSTEM_TEMPLATE.format(
            resume=_require_resume_summary(),
            feedback_block=self._feedback_block,
        )
        prompt = _FIT_USER_TEMPLATE.format(
            title=title,
            company=company,
            location=location or "not stated",
            salary_info=salary_info,
            description=(description or "")[:_LLM_DESC_CHARS],
        )

        @_retry_transient_cli
        def _invoke():
            return run_claude(
                prompt, model=self.model, system_prompt=system_prompt, timeout=60.0
            )

        try:
            result = _invoke()
        except ClaudeCLIError as e:
            logger.warning("Claude CLI failed scoring '%s' at %s: %s", title, company, e)
            return None, None, None, None, None, None, None, None

        self._track_usage(result)
        return self._parse_llm_response(result["text"])

    def _parse_llm_response(
        self, raw: str
    ) -> Tuple[Optional[int], Optional[str], Optional[float], Optional[float],
               Optional[float], Optional[float], Optional[dict], Optional[dict]]:
        """
        Parse JSON from LLM response, stripping markdown fences if present.
        Returns (score, explanation, sal_min, sal_max, est_min, est_max, dims_dict,
        filter_raw).

        IMPORTANT: always sums the 5 dimension fields to compute score — the LLM's
        own fit_score field is ignored because models routinely miscalculate it.
        """
        _DIM_KEYS = ["role_match", "seniority_match", "remote_location", "ai_domain_fit", "comp_match"]

        # Remove markdown code fences
        text = re.sub(r"```(?:json)?\s*", "", raw).strip()
        # Find the JSON object. Greedy (not lazy) so nested objects — like the
        # "filter" block's must_have_keywords/knockouts arrays-of-objects — are
        # captured whole instead of truncating at the first inner '}'. Safe
        # because the model is instructed to emit ONLY one JSON object.
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if not m:
            logger.debug("LLM produced no JSON object: %s", raw[:200])
            return None, None, None, None, None, None, None, None
        try:
            data = json.loads(m.group(0))

            dims_present = all(k in data for k in _DIM_KEYS)
            if not dims_present:
                # Missing dimension keys → treat as a scoring failure, not a
                # fabricated 50. Return NULL so the job stays unscored (and is
                # never stored with a made-up mid score).
                logger.debug("LLM response missing dimension keys: %s", raw[:200])
                return None, None, None, None, None, None, None, None
            # Always compute from actual dimension values — never trust fit_score
            dims = {k: max(0, int(data[k])) for k in _DIM_KEYS}
            raw_score = sum(dims.values())

            score = max(0, min(100, int(raw_score)))

            # Build explanation with dimension breakdown
            base_expl = str(data.get("explanation", "")).strip()
            if dims:
                dim_str = (
                    f"role={dims['role_match']}/30 "
                    f"seniority={dims['seniority_match']}/20 "
                    f"location={dims['remote_location']}/20 "
                    f"ai={dims['ai_domain_fit']}/20 "
                    f"comp={dims['comp_match']}/10"
                )
                explanation = f"[{dim_str}] {base_expl}"
            else:
                explanation = base_expl
            explanation = explanation[:600]

            def _to_float(v) -> Optional[float]:
                try:
                    f = float(v)
                    if f > MAX_BASE_SALARY:
                        logger.warning(
                            "LLM salary: discarding $%.0f — exceeds MAX_BASE_SALARY cap", f
                        )
                        return None
                    return f if 30_000 <= f else None
                except (TypeError, ValueError):
                    return None

            sal_min = _to_float(data.get("salary_min"))
            sal_max = _to_float(data.get("salary_max"))

            if sal_min and sal_max:
                sal_min, sal_max = sanitize_salary_range(sal_min, sal_max)

            def _to_est(v) -> Optional[float]:
                try:
                    f = float(v)
                except (TypeError, ValueError):
                    return None
                # Estimates are TOTAL comp — clamp to a sane band, else discard.
                return f if 60_000 <= f <= 1_500_000 else None

            est_min = _to_est(data.get("est_total_comp_min"))
            est_max = _to_est(data.get("est_total_comp_max"))
            if est_min and est_max and est_min > est_max:
                est_min, est_max = est_max, est_min

            filter_raw = self._parse_filter_block(data)

            return (score, explanation, sal_min, sal_max, est_min, est_max,
                    dims, filter_raw)
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.debug("LLM JSON parse error: %s | raw: %s", e, raw[:200])
            return None, None, None, None, None, None, None, None

    def _parse_filter_block(self, data: dict) -> Optional[dict]:
        """Validate/normalize the 'filter' object from a scoring response.

        Returns None (and logs a parse-failure warning) when the block is
        missing or malformed. An empty-but-valid must_have_keywords list is
        NOT handled here — _score_one_job logs that as an extraction miss,
        with the job title for context.
        """
        f = data.get("filter")
        if not isinstance(f, dict) or not isinstance(f.get("must_have_keywords"), list):
            logger.warning(
                "Filter block parse failure: missing or malformed 'filter' object"
            )
            return None
        cleaned = []
        for item in f["must_have_keywords"]:
            if not isinstance(item, dict):
                continue
            # `... or ""` (not .get(key, "")): a present-but-null "term" returns
            # None, and str(None) == "None" would survive the blank guard as a
            # phantom keyword. Coerce null to "" so it's dropped like a blank.
            term = str(item.get("term") or "").strip()
            if not term:
                continue
            aliases = item.get("aliases")
            cleaned.append({
                "term": term,
                "aliases": [str(a).strip() for a in aliases if str(a).strip()]
                if isinstance(aliases, list) else [],
            })
        knockouts = []
        raw_kos = f.get("knockouts")
        for k in (raw_kos if isinstance(raw_kos, list) else []):
            if not isinstance(k, dict):
                continue
            # Same null-coercion guard as "term" above — a present-but-null
            # "requirement" would otherwise survive as the string "None".
            req = str(k.get("requirement") or "").strip()
            if not req:
                continue
            v = str(k.get("verdict", "")).lower()
            knockouts.append({
                "requirement": req,
                "verdict": v if v in ("met", "unclear", "failed") else "unclear",
            })
        variants = f.get("title_variants")
        return {
            "must_have_keywords": cleaned,
            "knockouts": knockouts,
            "title_variants": [str(t).strip() for t in variants if str(t).strip()]
            if isinstance(variants, list) else [],
            "title_alignment": str(f.get("title_alignment", "none")).lower(),
        }

    # ------------------------------------------------------------------
    # Regex-only salary extraction (when claude CLI is unavailable)
    # ------------------------------------------------------------------

    def _extract_salaries_regex_only(self, run_id: Optional[int] = None) -> None:
        """Back-fill salary_min/salary_max from regex on jobs with no salary data."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        q = (
            "SELECT id, description FROM jobs "
            "WHERE salary_min IS NULL AND salary_max IS NULL"
        )
        params: list = []
        if run_id is not None:
            q += " AND run_id = ?"
            params.append(run_id)
        rows = conn.execute(q, params).fetchall()
        conn.close()

        updates = []
        for row in rows:
            sal_min, sal_max = extract_salary_regex(clean_for_llm(row["description"] or ""))
            if sal_min is not None or sal_max is not None:
                updates.append((sal_min, sal_max, row["id"]))

        if updates:
            conn = sqlite3.connect(self.db_path)
            conn.executemany(
                "UPDATE jobs SET salary_min = ?, salary_max = ? WHERE id = ?",
                updates,
            )
            conn.commit()
            conn.close()
            logger.info("Regex salary extraction: updated %d jobs.", len(updates))
