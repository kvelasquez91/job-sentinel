"""Dashboard-independent resume-tailoring pipeline.

The 10-step flow (JD extraction → keyword/gap analysis → Google Docs edits →
layout guard → ATS check → export) as one synchronous function. dashboard/app.py
wraps it in a background thread with task-status bookkeeping; main.py's
auto-tailor step calls it directly. All LLM work happens inside
tailor_engine/jd_extractor via the local `claude` CLI wrapper.
"""
import dataclasses
import json
import logging
import os
import re
import sqlite3
from datetime import datetime
from typing import Callable, Optional

from resume_tailor.jd_extractor import (
    JDExtractionError,
    JDPostingGoneError,
    JobDescription,
    extract_jd,
    get_token_usage as jd_get_token_usage,
    reset_token_usage as jd_reset_token_usage,
)
from resume_tailor.google_api import GoogleAPIClient
from resume_tailor.tailor_engine import (
    extract_keywords,
    gap_analysis,
    generate_edits,
    apply_edits,
    enforce_layout,
    credited_must_haves,
    get_token_usage as engine_get_token_usage,
    reset_token_usage as engine_reset_token_usage,
    set_log_context as engine_set_log_context,
    _SKILL_SUBCATEGORY_LABELS,
    _MASTER_TITLE_LINE,
)
from resume_tailor.layout_guard import build_line_map
from resume_tailor.ats_checker import check_all as ats_check_all, keyword_matches
from resume_tailor.config import (
    MASTER_DOC_ID,
    NAMING_TEMPLATE,
    USER_NAME,
    TAILORED_FOLDER_NAME,
    DOCX_OUTPUT_DIR,
)
from filter_match import (
    MASTER_RESUME_CACHE,
    build_filter_json,
    compute_filter_score,
    evaluate_must_haves,
    resolve_title_tier,
)
from filter_judge import load_inventory
import tailor_diff

logger = logging.getLogger(__name__)


def _load_stored_must_haves(db_path: str, job_id) -> Optional[list]:
    """Stored Filter Match must-haves for grounding the tailor, or None.

    v2 blobs carry judge verdicts + evidence; v1 blobs carry literal present
    flags; both pass through raw. Best-effort by design: NULL, the "{}"
    sentinel, parse failures, and DB errors all return None — the tailor then
    runs ungrounded, exactly as before this feature.
    """
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT filter_json FROM jobs WHERE id = ?", (job_id,)).fetchone()
        conn.close()
        if not row or not row["filter_json"]:
            return None
        stored = json.loads(row["filter_json"])
        return stored.get("must_haves") or None
    except Exception as exc:
        logger.warning(
            "Could not load stored must-haves for job %s: %s", job_id, exc)
        return None


# Same bar Tier 2 applies to generic extraction: anything at or under a short
# paragraph is a nav/error snippet, not a real JD.
_MIN_STORED_DESC_CHARS = 200


def _load_stored_description(db_path: str, job_id) -> Optional[str]:
    """The scraper's sanitized description for this job, or None.

    jobs.description is HTML-stripped centrally (clean_description in
    JobPosting.__post_init__), so it is directly usable as JD raw_text.
    Best-effort: missing row, NULL, sub-paragraph snippets, and DB errors
    all return None — the caller then re-raises the extraction error.
    """
    try:
        conn = sqlite3.connect(db_path)
        row = conn.execute(
            "SELECT description FROM jobs WHERE id = ?", (job_id,)).fetchone()
        conn.close()
        desc = (row[0] or "").strip() if row else ""
        return desc if len(desc) > _MIN_STORED_DESC_CHARS else None
    except Exception as exc:
        logger.warning(
            "Could not load stored description for job %s: %s", job_id, exc)
        return None


def run_tailor_pipeline(
    job: dict,
    db_path: str,
    progress: Optional[Callable[[str], None]] = None,
) -> dict:
    """Run the full tailoring pipeline for one job. Synchronous; raises on failure.

    job: dict with id, url, company, title.
    Any exception raised after the Doc copy exists carries .partial_doc_id so
    callers can tell the user about the orphaned copy.
    """
    progress = progress or (lambda msg: None)
    job_id, job_url = job["id"], job["url"]
    company, title = job.get("company") or "", job.get("title") or ""

    jd_reset_token_usage()
    engine_reset_token_usage()
    engine_set_log_context(job_id)

    # Grounding inputs, read BEFORE any LLM work (and before the post-tailor
    # recompute overwrites the judged blob): the stored Filter Match must-haves
    # and the experience inventory truth base.
    stored_must_haves = _load_stored_must_haves(db_path, job_id)
    inventory_text, _inv_sha, inv_basis = load_inventory()
    if inv_basis != "inventory":
        # The fallback basis IS the master resume text — it adds nothing the
        # tailor doesn't already have, so pass no inventory at all.
        inventory_text = ""

    # Step 1 — Extract job description (raises JDExtractionError). When the
    # posting URL is unreadable (2026-07-20: an overnight Workday window took
    # out 9 LIVE jobs), fall back to the scraper's stored description — the
    # exact text Filter Match judged, so grounding stays consistent. URL-first
    # is deliberate: fresh text when the board is reachable, and the URL
    # attempt is what detects dead postings. JDPostingGoneError therefore
    # propagates un-fallen-back — auto-tailor's suspect/confirm machinery
    # must see it, and a provably-delisted job must never burn a full tailor
    # against stale text.
    progress("Step 1/10: Fetching job description from posting URL...")
    jd_warnings: list = []
    try:
        jd = extract_jd(job_url)
    except JDPostingGoneError:
        raise
    except JDExtractionError as extract_err:
        stored_desc = _load_stored_description(db_path, job_id)
        if not stored_desc:
            raise
        progress("Step 1/10: Posting URL unreadable — using the stored "
                 "description from the last scrape...")
        logger.warning(
            "JD extraction failed for job %s (%s) — falling back to stored "
            "description (%d chars)", job_id, extract_err, len(stored_desc))
        jd = JobDescription(
            title=title,
            company=company,
            location="",
            raw_text=stored_desc,
            source_url=job_url,
            extraction_tier=5,   # stored-DB description (see jd_extractor tiers)
        )
        jd_warnings.append(
            "Tailored from the stored description (posting URL was "
            "unreadable) — text may be stale if the posting changed since "
            "the last scrape.")

    effective_company = company or jd.company or "Unknown"
    effective_title = title or jd.title or "Unknown Role"

    # Step 2 — Keyword extraction (LLM; non-fatal on empty result)
    progress("Step 2/10: Extracting keywords and requirements from job description...")
    jd_analysis = extract_keywords(jd.raw_text)
    exact_title = jd_analysis.exact_job_title or effective_title

    new_doc_id: Optional[str] = None
    try:
        # Step 3 — Authenticate Google API
        progress("Step 3/10: Connecting to Google Docs...")
        client = GoogleAPIClient()
        client.authenticate()

        # Step 4 — Copy master resume
        progress("Step 4/10: Creating a fresh copy of your master resume...")
        doc_title = NAMING_TEMPLATE.format(
            name=USER_NAME, company=effective_company, title=exact_title,
        )
        new_doc_id = client.copy_document(MASTER_DOC_ID, doc_title)

        # Step 5 — Read copy for gap analysis
        progress("Step 5/10: Reading resume content...")
        doc = client.read_document(new_doc_id)
        master_text = client.extract_plain_text(doc)

        # Refresh the master-resume text cache used by the Filter Match score
        # (engine/llm_scorer.py reads it at init). Best-effort.
        try:
            os.makedirs(os.path.dirname(MASTER_RESUME_CACHE), exist_ok=True)
            with open(MASTER_RESUME_CACHE, "w", encoding="utf-8") as fh:
                fh.write(master_text)
        except Exception as cache_err:
            logger.warning("Could not write master resume cache: %s", cache_err)

        # Step 5.5 — Master layout reference: the pristine copy IS the master's
        # layout. One export + line map; the guard compares every edited
        # paragraph against this. Non-fatal: without it the guard is skipped.
        progress("Step 5.5/10: Capturing master layout reference (PDF line map)...")
        try:
            master_map = build_line_map(client.export_as_pdf(new_doc_id))
        except Exception as map_err:
            logger.warning("Master line-map capture failed — layout guard will be skipped: %s", map_err)
            master_map = None

        # Step 6 — Gap analysis (LLM)
        progress("Step 6/10: Comparing your resume against the job requirements...")
        gap = gap_analysis(master_text, jd_analysis,
                           inventory_text=inventory_text,
                           must_haves=stored_must_haves)

        # Step 7 — Generate + validate edits (LLM; may run correction loops)
        progress("Step 7/10: Generating tailored edits to hit keyword targets (15–30s)...")
        edits = generate_edits(master_text, jd_analysis, gap, effective_company,
                               inventory_text=inventory_text,
                               must_haves=stored_must_haves)

        # Step 8 — Apply edits to the Google Doc copy
        progress("Step 8/10: Applying edits to the resume document...")
        apply_edits(new_doc_id, edits, client)

        # Step 8.5 — Layout guard: enforce per-paragraph line counts + no
        # danglers against the master map; repairs then reverts as needed.
        # The guard receives the judge-credited must-haves so repairs preserve
        # them and reverts don't silently destroy them (its dangler rule
        # yields to screening coverage for these terms).
        credited = credited_must_haves(stored_must_haves or [])
        final_pdf = None
        layout_warnings: list = []
        guard_crashed = False
        if master_map is not None:
            progress("Step 8.5/10: Enforcing layout against master (PDF line-map guard)...")
            try:
                final_pdf, layout_warnings = enforce_layout(
                    new_doc_id, edits, doc, master_map, client,
                    credited_items=credited)
            except Exception as guard_err:
                logger.error("Layout guard failed (non-fatal) for job %s: %s",
                             job_id, guard_err)
                guard_crashed = True
                layout_warnings = [f"Layout guard crashed — layout not verified: {guard_err}"]
        else:
            layout_warnings = ["Layout guard skipped — master line map unavailable"]

        # Step 9 — ATS compliance check (reuses the guard's final PDF)
        progress("Step 9/10: Running ATS compliance check (including PDF page count)...")
        updated_doc = client.read_document(new_doc_id)
        pdf_bytes = final_pdf
        if pdf_bytes is None:
            try:
                pdf_bytes = client.export_as_pdf(new_doc_id)
            except Exception as pdf_err:
                logger.warning("Could not export PDF for page-count check: %s", pdf_err)
                pdf_bytes = None
        ats_result = ats_check_all(
            updated_doc, jd_analysis.priority_keywords, exact_title,
            pdf_bytes=pdf_bytes,
            master_word_count=len(master_text.split()) if master_text else None,
        )

        # Keyword match measured on the FINAL document with the same whole-word
        # matcher the ATS check uses — the single source of truth, so the reported
        # count and the ATS keyword warning can never disagree. This reflects the
        # doc after any layout-guard reverts, not the pre-revert edits.keyword_count.
        updated_text = client.extract_plain_text(updated_doc)
        matched_keywords = keyword_matches(updated_text, jd_analysis.priority_keywords)

        # Final-text truth check: a credited must-have absent HERE reads
        # absent to the post-tailor recompute too (same matcher). The engine's
        # pre-apply check can't see layout-guard reverts or substitutions that
        # removed master text — this one can.
        must_have_warnings: list = []
        missing_final: list = []
        if credited:
            evaluated_final = evaluate_must_haves(
                updated_text,
                [{"term": m["term"], "aliases": m.get("aliases", [])}
                 for m in credited])
            missing_final = [m["term"] for m, ev in zip(credited, evaluated_final)
                             if not ev["present"]]
            if missing_final:
                must_have_warnings.append(
                    "Tailored resume is missing must-have terms the Filter "
                    "Match recompute will count as absent: "
                    + ", ".join(missing_final))

        # Ground-truth persistence: per-edit verdicts + badge facts, computed
        # on the FINAL text with the engine's own references (injected).
        # Best-effort like the persistence steps below: a verdict failure
        # must never fail a tailor that already succeeded.
        edits_payload = {**dataclasses.asdict(edits), "master_text": master_text}
        layout_unverified = (master_map is None) or guard_crashed
        try:
            edit_verdicts = tailor_diff.compute_edit_verdicts(
                edits_payload, updated_text,
                _SKILL_SUBCATEGORY_LABELS, _MASTER_TITLE_LINE)
            issue_facts = tailor_diff.build_issue_facts(
                edit_verdicts, missing_final, layout_unverified)
        except Exception as verdict_err:
            logger.warning(
                "Edit-verdict computation failed (non-fatal) for job %s: %s",
                job_id, verdict_err)
            edit_verdicts = []
            issue_facts = None

        # Step 10 — Move to folder + export .docx
        progress("Step 10/10: Exporting resume and saving to Tailored Resumes folder...")
        client.move_to_folder(new_doc_id, TAILORED_FOLDER_NAME)

        os.makedirs(DOCX_OUTPUT_DIR, exist_ok=True)
        safe_company = re.sub(r"[^\w\s-]", "", effective_company).strip()
        safe_title = re.sub(r"[^\w\s-]", "", exact_title).strip()[:50]
        docx_filename = f"{USER_NAME} - {safe_company} - {safe_title}.docx"
        docx_path = os.path.join(DOCX_OUTPUT_DIR, docx_filename)
        client.export_as_docx(new_doc_id, docx_path)

        doc_url = client.get_document_url(new_doc_id)
        completed_at = datetime.now().isoformat()
    except Exception as exc:
        # Let callers surface the orphaned partial Doc copy, if one was made.
        exc.partial_doc_id = new_doc_id
        raise

    jd_usage = jd_get_token_usage()
    engine_usage = engine_get_token_usage()
    total_input = jd_usage["input_tokens"] + engine_usage["input_tokens"]
    total_output = jd_usage["output_tokens"] + engine_usage["output_tokens"]
    est_cost = jd_usage.get("cost_usd", 0.0) + engine_usage.get("cost_usd", 0.0)
    logger.info(
        "Tailor pipeline for job %s: %d input + %d output tokens, est. cost $%.4f",
        job_id, total_input, total_output, est_cost,
    )

    # Update job record (best-effort)
    try:
        conn = sqlite3.connect(db_path)
        with conn:
            conn.execute(
                "UPDATE jobs SET tailored_resume_url = ?, tailored_at = ?, "
                "tailor_issue_json = ? WHERE id = ?",
                (doc_url, completed_at,
                 json.dumps(issue_facts) if issue_facts else None, job_id),
            )
        conn.close()
    except Exception as db_err:
        logger.warning("Could not update tailored_resume_url for job %s: %s", job_id, db_err)

    # Post-tailor Filter Match recompute — same stored denominator, tailored
    # text. Best-effort: never fails the pipeline. filter_score_master is
    # frozen; knockout verdicts describe the candidate and carry over.
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        frow = conn.execute(
            "SELECT filter_json FROM jobs WHERE id = ?", (job_id,)).fetchone()
        conn.close()
        if frow and frow["filter_json"]:
            stored = json.loads(frow["filter_json"])
            evaluated = evaluate_must_haves(
                updated_text,
                [{"term": m["term"], "aliases": m.get("aliases", [])}
                 for m in stored.get("must_haves", [])],
            )
            if evaluated:
                tier = resolve_title_tier(
                    updated_text, exact_title,
                    stored.get("title_variants", []),
                    stored.get("title_alignment", "none"))
                fscore, knocked_out = compute_filter_score(
                    evaluated, tier, stored.get("knockouts", []))
                fjson = build_filter_json(
                    evaluated, stored.get("title_variants", []), tier,
                    stored.get("knockouts", []),
                    stored.get("title_alignment", "none"))
                conn = sqlite3.connect(db_path)
                with conn:
                    conn.execute(
                        "UPDATE jobs SET filter_score = ?, filter_source = 'tailored', "
                        "filter_knockout = ?, filter_json = ? WHERE id = ?",
                        (fscore, int(knocked_out), fjson, job_id))
                conn.close()
                logger.info("Post-tailor Filter Match for job %s: %d (%s)",
                            job_id, fscore, "KO" if knocked_out else "ok")
    except Exception as filter_err:
        logger.warning(
            "Post-tailor filter recompute failed (non-fatal) for job %s: %s",
            job_id, filter_err)

    # Save full tailor result to history table (best-effort)
    try:
        issues_payload = [dataclasses.asdict(i) for i in ats_result.issues]
        conn = sqlite3.connect(db_path)
        with conn:
            conn.execute(
                """INSERT INTO tailor_history
                   (job_id, created_at, google_doc_url, docx_path,
                    ats_score, ats_passed, ats_issues,
                    keywords_matched, keywords_total, keywords_list,
                    edits_json, est_cost, total_input_tokens, total_output_tokens,
                    job_title, company,
                    final_text, final_text_source, warnings_json, edit_verdicts_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    job_id, completed_at, doc_url, docx_path,
                    ats_result.score, int(ats_result.passed),
                    json.dumps(issues_payload),
                    len(matched_keywords),
                    len(jd_analysis.priority_keywords),
                    json.dumps(matched_keywords),
                    json.dumps(edits_payload),
                    round(est_cost, 4), total_input, total_output,
                    exact_title, effective_company,
                    updated_text, "pipeline",
                    json.dumps({"layout": layout_warnings,
                                "must_have": must_have_warnings}),
                    json.dumps(edit_verdicts),
                ),
            )
        conn.close()
        logger.info("Saved tailor history record for job %s", job_id)
    except Exception as hist_err:
        logger.warning("Could not save tailor history for job %s: %s", job_id, hist_err)

    critical_issues = [
        i.description for i in ats_result.issues if i.severity == "critical"
    ]

    return {
        "google_doc_url": doc_url,
        "docx_filename": docx_filename,
        "docx_path": docx_path,
        "ats_score": ats_result.score,
        "keywords_matched": len(matched_keywords),
        "keywords_total": len(jd_analysis.priority_keywords),
        "issues": critical_issues,
        "warnings": jd_warnings + list(ats_result.warnings) + layout_warnings
                    + must_have_warnings,
        "company": effective_company,
        "title": exact_title,
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
        "est_cost": round(est_cost, 4),
        "issue_facts": issue_facts,
    }
