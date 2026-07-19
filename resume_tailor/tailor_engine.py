"""
LLM-powered resume tailoring logic.

4-step chain (all via the local Claude CLI):
  Step 1 — extract_keywords()    : Pull JD keywords/requirements
  Step 2 — gap_analysis()        : Compare master resume vs JD keywords
  Step 3 — generate_edits()      : Generate surgical edit instructions
  Step 4 — validate_keywords()   : Ensure 25-35 keyword density; retry if off

apply_edits() translates the generated edit instructions into Google Docs
batchUpdate requests and applies them to the copied document.

These building blocks are consumed by pipeline.run_tailor_pipeline(), the
single tailoring orchestrator.
"""
import json
import logging
import re
import threading
from dataclasses import dataclass, field
from typing import Optional

from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
    before_sleep_log,
)

from .config import (
    TAILOR_MODEL,
    TAILOR_EDIT_MODEL,
    KEYWORD_MIN,
    KEYWORD_MAX,
    KEYWORD_CORRECTION_ROUNDS,
    FIRST_NAME,
)
from .google_api import GoogleAPIClient
from . import layout_guard as lg
from .layout_guard import EditedParagraph, LineMap, build_line_map, check_layout, normalize

import sys
import os as _os
sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from claude_cli import run_claude, ClaudeCLIError, ClaudeCLITimeout
from filter_match import evaluate_must_haves, phrase_present
from profile_policy import TAILOR_MASTER_TITLE_LINE, TAILOR_SKILL_SUBCATEGORY_LABELS

logger = logging.getLogger(__name__)

# Thread-local token accumulator: tracks usage across all _claude_call invocations
# within a single tailor run (each run executes in its own thread).
_token_acc = threading.local()

# Thread-local job tag for log attribution: concurrent auto-tailor workers
# (workers >= 2) interleave this module's log lines, making a bare
# [layout_reshape] line unattributable and its logged duration read as
# mismatched wall-clock. Each pipeline run stamps its job id here (same
# lifecycle as reset_token_usage — overwritten at run start; thread pools
# reuse threads) and _JobContextFilter prefixes every record on this logger.
_log_ctx = threading.local()


def set_log_context(job_id) -> None:
    """Tag this thread's subsequent tailor_engine log records with the job id.
    Pass None to clear."""
    _log_ctx.job_id = job_id


def get_log_context():
    """Job id set for the current thread, or None."""
    return getattr(_log_ctx, "job_id", None)


class _JobContextFilter(logging.Filter):
    def filter(self, record):
        job_id = get_log_context()
        if job_id is not None:
            record.msg = f"[job {job_id}] {record.msg}"
        return True


logger.addFilter(_JobContextFilter())


def reset_token_usage() -> None:
    """Reset accumulated token counts/cost for the current thread. Call before each run."""
    _token_acc.input_tokens = 0
    _token_acc.output_tokens = 0
    _token_acc.cost_usd = 0.0


def get_token_usage() -> dict:
    """Return accumulated token counts + notional cost for the current thread."""
    return {
        "input_tokens": getattr(_token_acc, "input_tokens", 0),
        "output_tokens": getattr(_token_acc, "output_tokens", 0),
        "cost_usd": getattr(_token_acc, "cost_usd", 0.0),
    }

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class JDAnalysis:
    """
    Structured output from Step 1 (keyword extraction).

    priority_keywords is the ranked list of up to 35 ATS-critical terms that
    drive all downstream gap analysis and edit generation. The order matters —
    the LLM ranks by JD repetition frequency and requirement vs. nice-to-have.
    """

    exact_job_title: str
    hard_skills: list = field(default_factory=list)
    soft_skills: list = field(default_factory=list)
    certifications: list = field(default_factory=list)
    experience_keywords: list = field(default_factory=list)
    priority_keywords: list = field(default_factory=list)  # Ranked, max 35
    years_experience: str = ""
    education_requirements: str = ""


@dataclass
class GapAnalysis:
    """
    Structured output from Step 2 (gap analysis).

    missing_but_relevant contains keywords the candidate has experience with that can
    be substituted into existing bullets — these are the primary edit targets.
    missing_and_irrelevant keywords are skipped to avoid fabricating experience.
    rewrite_candidates identifies up to 2 bullets that would benefit most from
    a full rewrite to better highlight relevant experience for this role.
    """

    already_matched: list = field(default_factory=list)
    missing_but_relevant: list = field(default_factory=list)
    missing_and_irrelevant: list = field(default_factory=list)
    keyword_count: int = 0
    target_additions: int = 0
    rewrite_candidates: list = field(default_factory=list)


@dataclass
class TailoringEdits:
    """
    Structured edit instructions from Step 3 (generate_edits).

    Each field maps to a specific apply_edits() sub-operation:
      title_line_replacement → replaceAllText on the header title line
      summary_replacement    → surgical paragraph replacement in _apply_summary_replacement
      skills_reorder         → replaceAllText per sub-category line in the Skills section
      experience_edits       → replaceAllText per bullet in _apply_experience_edits
      rewritten_bullets      → full bullet rewrites (up to 2) in _apply_rewritten_bullets;
                               anchor is the full original text (label + colon + body)
                               because the bold prefix label may also change.
    """

    title_line_replacement: str = ""
    summary_replacement: str = ""
    skills_reorder: dict = field(default_factory=dict)
    experience_edits: list = field(default_factory=list)
    rewritten_bullets: list = field(default_factory=list)
    keywords_used: list = field(default_factory=list)
    keyword_count: int = 0
    rationale: str = ""


# ---------------------------------------------------------------------------
# Claude CLI helper
# ---------------------------------------------------------------------------

# Default wall-clock budget for a single CLI call. Short analysis steps finish
# well inside this; the Opus edit steps override it (see EDIT_CALL_TIMEOUT) because
# they legitimately run ~100-200s and were being killed at 120s mid-generation.
DEFAULT_CALL_TIMEOUT = 120.0
EDIT_CALL_TIMEOUT = 300.0
# gap_analysis sits between the two: its ~10-11k-char prompts have a healthy
# latency tail past the default (measured successes at 108.5s, 119.7s and 162s;
# the 120s ceiling killed a healthy call at 120.5s), but it never approaches
# the Opus edit steps' 300s.
ANALYSIS_CALL_TIMEOUT = 240.0


@retry(
    # Retry transient CLI errors, but NOT timeouts: a timed-out call has already
    # burned its full wall-clock budget while holding a global-semaphore slot, so
    # a blanket retry here doubles the worst case for EVERY caller — including
    # the 300s Opus edit calls. Callers that can afford it retry timeouts
    # themselves (llm_reshape retries once: measured stalls are per-call flukes
    # — identical prompts run 8s or 120s+ — not prompt-determined latency).
    retry=retry_if_exception(
        lambda e: isinstance(e, ClaudeCLIError) and not isinstance(e, ClaudeCLITimeout)
    ),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=2, max=8),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
def _claude_call_inner(
    prompt: str, temperature: float, model: str = TAILOR_MODEL,
    timeout: float = DEFAULT_CALL_TIMEOUT,
    system_prompt: Optional[str] = None,
) -> tuple:
    """Raw CLI call decorated with retry for transient errors. (temperature is ignored — the CLI cannot set it.)"""
    result = run_claude(prompt, model=model, timeout=timeout,
                        system_prompt=system_prompt)
    return result["text"].strip(), result["usage"], result["cost_usd"]


def _claude_call(
    prompt: str,
    temperature: float = 0.1,
    label: str = "",
    model: str = TAILOR_MODEL,
    timeout: float = DEFAULT_CALL_TIMEOUT,
    system_prompt: Optional[str] = None,
) -> str:
    """
    Call Claude via the local CLI with retry for transient errors.
    Raises ClaudeCLIError once retries are exhausted (timeouts fail fast):
    callers key their failure taxonomy on this exception reaching them —
    swallowing it here let a dead CLI complete a tailor run as a mislabeled
    success (untailored copy with tailored_at set, no attempt bump).
    """
    import time
    tag = f"[{label}] " if label else ""
    logger.info("%sLLM call starting (model=%s, ~%d prompt chars)", tag, model, len(prompt))
    t0 = time.monotonic()
    try:
        text, usage, cost_usd = _claude_call_inner(
            prompt, temperature, model=model, timeout=timeout,
            system_prompt=system_prompt)
        elapsed = time.monotonic() - t0
        logger.info("%sLLM call complete in %.1fs (%d response chars)", tag, elapsed, len(text))
        # Accumulate token usage + notional cost for this run
        if not hasattr(_token_acc, "input_tokens"):
            _token_acc.input_tokens = 0
            _token_acc.output_tokens = 0
            _token_acc.cost_usd = 0.0
        _token_acc.input_tokens += usage.get("input_tokens", 0)
        _token_acc.output_tokens += usage.get("output_tokens", 0)
        _token_acc.cost_usd = getattr(_token_acc, "cost_usd", 0.0) + cost_usd
        return text
    except ClaudeCLIError as exc:
        elapsed = time.monotonic() - t0
        logger.error("%sClaude CLI call failed after %.1fs: %s", tag, elapsed, exc)
        raise


def _parse_json_response(text: str, context: str = "") -> Optional[dict]:
    """
    Extract the first JSON object from an LLM response.
    Retries by stripping markdown fences if direct parse fails.
    """
    if not text:
        return None

    # Strip markdown code fences
    cleaned = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`").strip()

    # Try direct parse
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Try to find the first {...} block
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    logger.warning("Failed to parse JSON from LLM response (%s). Raw: %.200s", context, text)
    return None


# ---------------------------------------------------------------------------
# Step 1 — Keyword extraction
# ---------------------------------------------------------------------------

_KEYWORD_EXTRACTION_PROMPT = """\
You are an expert ATS keyword analyst. Analyze this job description and extract \
the following in JSON format.

CRITICAL RULES:
1. Use the EXACT terms from the posting — not synonyms. "Adobe Creative Cloud" and \
"Adobe Creative Suite" are DIFFERENT to an ATS.
2. For priority_keywords: select only keywords that COULD REALISTICALLY APPEAR in a \
senior product manager's resume. Omit jargon that would never fit naturally in a resume \
bullet or skills list (e.g. legal boilerplate, benefits copy, generic phrases like "team player").
3. Rank priority_keywords by this explicit priority order:
   a. HIGHEST: Terms from the job title and the first paragraph/opening of the JD — these \
signal what the company cares about most.
   b. HIGH: Terms from the requirements/qualifications section — these define the bar for the role.
   c. MEDIUM: Terms from the responsibilities section — these describe day-to-day work.
   d. LOWER: Nice-to-haves, preferred qualifications, or terms that appear only once.
4. Keywords that NAME the company's core product or domain (e.g. "contracts" for a contracts \
platform, "billing" for a billing product) MUST be prioritized over generic industry terms. \
Include keywords that describe what the company BUILDS or what the team DOES, not just \
generic skills like "stakeholder management" or "cross-functional collaboration".
5. If the JD mentions a specific vertical or domain (legal, healthcare, finance, developer tools, \
etc.), include domain-specific terms — these signal domain expertise to the hiring manager and \
are high-signal differentiators.
6. Keep priority_keywords to the 25-35 most impactful terms — quality over quantity.
7. EXCLUDE keywords that are specific to the hiring company's internal products, tools, or \
platforms (e.g., "Azure AI Foundry", "Copilot", "M365", "Salesforce Lightning" if applying \
to Salesforce). These cannot be honestly included in a resume without having worked at that \
company. Focus keyword slots on TRANSFERABLE skills, methodologies, and industry-standard \
technologies that the candidate can legitimately claim.
8. For exact_job_title: copy the FULL title VERBATIM from the posting, including ALL seniority \
qualifiers ("Principal", "Senior", "Staff", "Lead", "Director", "Distinguished", "Associate", \
"Junior"). NEVER omit, abbreviate, or normalize the seniority level. "Principal Product Manager" \
MUST remain "Principal Product Manager" — it MUST NOT become "Product Manager". \
"Senior Director of Product" MUST NOT become "Director of Product".

Return ONLY valid JSON with these keys:
{{
  "exact_job_title": "The FULL, EXACT title from the posting — verbatim, including ALL seniority \
qualifiers (e.g. 'Principal', 'Senior', 'Staff', 'Lead', 'Distinguished'). NEVER shorten or drop \
the seniority level. 'Principal Product Manager' must NOT become 'Product Manager'.",
  "hard_skills": ["Technical skills, tools, languages, frameworks — EXACT wording"],
  "soft_skills": ["Leadership, communication, collaboration terms — EXACT wording"],
  "certifications": ["Any mentioned certifications or credentials"],
  "experience_keywords": ["Specific experience areas — e.g. 'distributed systems'"],
  "priority_keywords": ["The 25-35 most impactful, resume-embeddable keywords ranked by importance — \
EXACT strings from the posting"],
  "years_experience": "Any mentioned experience requirements as a string",
  "education_requirements": "Degree or education mentions as a string"
}}

Job Description:
{jd_text}
"""

_KEYWORD_RETRY_PROMPT = """\
Your previous response was not valid JSON. Respond ONLY with valid JSON, \
no markdown, no explanation. Use this exact structure:
{{"exact_job_title":"...","hard_skills":[...],"soft_skills":[...],\
"certifications":[...],"experience_keywords":[...],"priority_keywords":[...],\
"years_experience":"...","education_requirements":"..."}}

Job Description:
{jd_text}
"""


# ---------------------------------------------------------------------------
# Truth-base system prompt (experience inventory)
# ---------------------------------------------------------------------------

# Static per inventory, sent as the CLI system prompt so consecutive tailor
# calls share an identical cache prefix — the same convention as
# filter_judge.build_judge_system. Per-job content stays in the user prompts.
_TRUTH_SYSTEM_TEMPLATE = """\
You are an expert, strictly truthful resume-tailoring assistant.

The CANDIDATE EXPERIENCE INVENTORY below is the single source of truth for
what the candidate can claim. It is richer than the resume by design —
truthful inventory claims MAY be woven into the resume even when the current
resume text does not mention them.

BINDING RULES:
- The inventory's NOT CLAIMED section is absolute: NEVER state, imply, or
  keyword-match anything listed there, even if other text seems related.
- The BASELINE PRODUCT CRAFT section covers generic product-management craft
  only — it never justifies claiming specific tools, technologies, or domains.
- Never fabricate: every claim in the tailored resume must be defensible in
  an interview from this inventory or the existing resume text.

CANDIDATE EXPERIENCE INVENTORY:
{inventory}"""


def build_truth_system(inventory_text: str) -> Optional[str]:
    """Static system prompt grounding tailor LLM calls in the inventory.

    None when there is no inventory — callers then send no custom system
    prompt and the CLI default applies (legacy behavior).
    """
    if not (inventory_text or "").strip():
        return None
    return _TRUTH_SYSTEM_TEMPLATE.format(inventory=inventory_text.strip())


def _render_must_have_block(must_haves: list) -> str:
    """SCREENING MUST-HAVE TERMS prompt section, or "" when nothing to say.

    Renders the stored Filter Match must-haves — the exact terms the
    post-tailor recompute will literally search the tailored text for.
    Judged items (v2 blob) carry authoritative verdicts and inventory
    evidence; unjudged items (v1 blob) fall back to their literal
    present-in-master flag.
    """
    lines = []
    for item in must_haves or []:
        if not isinstance(item, dict):
            continue
        term = str(item.get("term", "")).strip()
        if not term:
            continue
        aliases = [str(a).strip() for a in (item.get("aliases") or [])
                   if str(a).strip()]
        alias_txt = f" (aliases also count: {', '.join(aliases)})" if aliases else ""
        if "verdict" in item:
            if item.get("verdict") in ("explicit", "evidenced"):
                ev = str(item.get("evidence") or "").strip()
                ev_txt = f' — inventory evidence: "{ev}"' if ev else ""
                lines.append(
                    f'- MUST INCLUDE "{term}"{alias_txt} — judged '
                    f'{item["verdict"]}: truthfully claimable{ev_txt}')
            else:
                lines.append(
                    f'- DO NOT INCLUDE "{term}"{alias_txt} — judged absent: '
                    f'no truthful basis in the inventory')
        elif item.get("present") is True:
            lines.append(
                f'- MUST KEEP "{term}"{alias_txt} — already in the master '
                f'resume; it must survive your edits')
        else:
            lines.append(
                f'- TARGET "{term}"{alias_txt} — include ONLY if the '
                f'inventory truthfully supports it; never fabricate')
    if not lines:
        return ""
    return (
        "SCREENING MUST-HAVE TERMS (recruiter boolean-search screen):\n"
        "After tailoring, the resume is literally re-scored for these exact "
        "terms (whole words; a listed alias counts):\n"
        + "\n".join(lines) + "\n\n"
    )


def _render_missing_must_have_instruction(missing: list) -> str:
    """Correction-prompt block naming credited terms still missing, or ""."""
    if not missing:
        return ""
    lines = []
    for m in missing:
        aliases = [str(a).strip() for a in (m.get("aliases") or [])
                   if str(a).strip()]
        alias_txt = f" (or one alias: {', '.join(aliases)})" if aliases else ""
        lines.append(f'- "{m["term"]}"{alias_txt}')
    return (
        "\nMISSING SCREENING MUST-HAVES — HIGHEST PRIORITY:\n"
        "The employer's screening pass literally searches the final resume "
        "for these terms, and they were verified truthful for this candidate. "
        "Each MUST appear verbatim (whole words — the term itself or one "
        "listed alias) after your edits. Substitute them into the summary, a "
        "skills line, or a genuinely related bullet WITHOUT growing any "
        "line:\n" + "\n".join(lines) + "\n"
    )


def extract_keywords(jd_text: str) -> JDAnalysis:
    """
    Step 1: Use Claude to extract structured keywords from the job description.
    Returns a JDAnalysis dataclass.
    """
    logger.info("Step 1: Extracting keywords from JD (%d chars)", len(jd_text))

    prompt = _KEYWORD_EXTRACTION_PROMPT.format(jd_text=jd_text[:6000])
    raw = _claude_call(prompt, label="extract_keywords")
    data = _parse_json_response(raw, context="keyword_extraction")

    if not data:
        logger.info("Retrying keyword extraction with strict JSON prompt")
        raw = _claude_call(_KEYWORD_RETRY_PROMPT.format(jd_text=jd_text[:6000]), label="extract_keywords_retry")
        data = _parse_json_response(raw, context="keyword_extraction_retry")

    if not data:
        logger.error("Keyword extraction failed after retry — returning empty JDAnalysis")
        return JDAnalysis(exact_job_title="")

    return JDAnalysis(
        exact_job_title=data.get("exact_job_title", ""),
        hard_skills=data.get("hard_skills", []),
        soft_skills=data.get("soft_skills", []),
        certifications=data.get("certifications", []),
        experience_keywords=data.get("experience_keywords", []),
        priority_keywords=data.get("priority_keywords", [])[:35],
        years_experience=data.get("years_experience", ""),
        education_requirements=data.get("education_requirements", ""),
    )


# ---------------------------------------------------------------------------
# Step 2 — Gap analysis
# ---------------------------------------------------------------------------

# Grounding guidance injected into the gap prompt ONLY when the inventory or
# stored must-haves are supplied — so a legacy (ungrounded) call renders a
# byte-identical prompt to the pre-grounding version. Both sentences are
# self-gating ("When a … is provided/appears"), so the paragraph is safe even
# when only one of the two grounding inputs is present.
_GAP_GROUNDING_RULES = f"""\
TRUTH BASE & MUST-HAVES:
- When a CANDIDATE EXPERIENCE INVENTORY is provided (system prompt), judge what \
{FIRST_NAME} "has experience with" against the INVENTORY, not the resume text alone — \
truthful inventory claims missing from the resume are exactly what \
missing_but_relevant is for. The inventory's NOT CLAIMED section is binding: \
those items always belong in missing_and_irrelevant.
- When a SCREENING MUST-HAVE TERMS section appears below, its verdicts are \
authoritative: MUST INCLUDE / MUST KEEP terms belong in already_matched (if \
verbatim in the resume) or missing_but_relevant (name the exact bullet or skill \
line to substitute each into) — NEVER in missing_and_irrelevant. DO NOT INCLUDE \
terms belong in missing_and_irrelevant.

"""

_GAP_ANALYSIS_PROMPT = """\
Compare this resume against the job requirements below.

IMPORTANT — SUBSTITUTION OVER ADDITION:
Your primary job is to identify where JD keywords can REPLACE existing words in the \
resume rather than be added on top. Adding new content causes the resume to overflow \
one page. Look for:
- Existing bullets where a generic word can be swapped for an exact JD keyword
- Skill names that are synonyms of JD terms and can be renamed to match exactly
- Summary phrases that can be reworded to include JD keywords without expanding length

ALSO IDENTIFY REWRITE CANDIDATES:
Identify up to 2 experience bullets that would benefit most from a FULL rewrite (not \
just a keyword swap). A good rewrite candidate is a bullet where {name} clearly has the \
relevant experience but the bullet uses different terminology than the JD — the underlying \
work is a strong match, just framed differently. A full rewrite can reframe, re-emphasize, \
or restructure the bullet to foreground the skills the JD is looking for — based entirely \
on real experience.
DO NOT suggest a rewrite if:
- The bullet is already well-aligned with the JD
- The connection to the JD is only tangential — forced rewrites of loosely related bullets \
produce generic filler that reads as artificial and dilutes the resume's credibility

{grounding_rules}Return ONLY valid JSON with these keys:
{{
  "already_matched": ["Keywords/skills already present in the resume verbatim"],
  "missing_but_relevant": ["JD keywords {name} has experience with — flag the SPECIFIC \
bullet or skill where each can be substituted in without adding words"],
  "missing_and_irrelevant": ["JD keywords {name} genuinely doesn't have \
experience with — DO NOT fabricate these"],
  "keyword_count": <integer: current count of priority_keywords present in resume>,
  "target_additions": <integer: how many more keywords needed to reach 25-35>,
  "rewrite_candidates": [
    {{
      "company": "Company name from resume",
      "bullet_prefix": "The bold label before the colon in the original bullet",
      "original_bullet": "Full original bullet text (bold label + colon + body)",
      "reason": "Why this bullet is a good rewrite candidate for this specific JD"
    }}
  ]
}}

{must_have_block}Resume:
{resume_text}

Priority Keywords (from JD):
{priority_keywords}

Full JD Analysis:
{jd_analysis_json}
"""


def gap_analysis(resume_text: str, jd_analysis: JDAnalysis,
                 inventory_text: str = "",
                 must_haves: Optional[list] = None) -> GapAnalysis:
    """
    Step 2: Compare master resume content against JD keywords.

    inventory_text grounds truthfulness (sent as the cached system prompt);
    must_haves are the stored Filter Match terms — with judge verdicts and
    evidence when the blob was v2. Both optional: omitted, this is the legacy
    resume-only analysis. Returns a GapAnalysis dataclass.
    """
    logger.info("Step 2: Running gap analysis")

    # Grounding guidance shows when there is an inventory OR a renderable
    # must-have block — never for a truthy-but-unrenderable must_haves alone.
    must_have_block = _render_must_have_block(must_haves or [])
    grounded = bool((inventory_text or "").strip() or must_have_block)
    prompt = _GAP_ANALYSIS_PROMPT.format(
        grounding_rules=_GAP_GROUNDING_RULES if grounded else "",
        name=FIRST_NAME,
        resume_text=resume_text[:4000],
        priority_keywords=json.dumps(jd_analysis.priority_keywords, indent=2),
        jd_analysis_json=json.dumps(
            {
                "hard_skills": jd_analysis.hard_skills,
                "soft_skills": jd_analysis.soft_skills,
                "experience_keywords": jd_analysis.experience_keywords,
            },
            indent=2,
        ),
        must_have_block=must_have_block,
    )

    system_prompt = build_truth_system(inventory_text)
    try:
        raw = _claude_call(prompt, label="gap_analysis",
                           timeout=ANALYSIS_CALL_TIMEOUT,
                           system_prompt=system_prompt)
    except ClaudeCLITimeout:
        # One timeout here failed the whole run after the Doc copy existed and
        # bumped the attempt counter — over a per-call CLI stall. Retry once
        # (llm_reshape pattern); a second timeout means the CLI is wedged and
        # must propagate, not degrade into an empty GapAnalysis (that would
        # record a barely-tailored copy as a success).
        logger.warning("gap_analysis timed out — retrying the call once")
        raw = _claude_call(prompt, label="gap_analysis_retry",
                           timeout=ANALYSIS_CALL_TIMEOUT,
                           system_prompt=system_prompt)
    data = _parse_json_response(raw, context="gap_analysis")

    if not data:
        logger.error("Gap analysis failed — returning empty GapAnalysis")
        return GapAnalysis()

    current_count = data.get("keyword_count", 0)
    target_additions = max(0, KEYWORD_MIN - current_count)

    return GapAnalysis(
        already_matched=data.get("already_matched", []),
        missing_but_relevant=data.get("missing_but_relevant", []),
        missing_and_irrelevant=data.get("missing_and_irrelevant", []),
        keyword_count=current_count,
        target_additions=data.get("target_additions", target_additions),
        rewrite_candidates=data.get("rewrite_candidates", [])[:2],
    )


# ---------------------------------------------------------------------------
# Step 3 — Generate edits
# ---------------------------------------------------------------------------

# Injected into the edit prompt ONLY when stored must-haves are supplied, so a
# legacy (ungrounded) edit call renders a byte-identical prompt. The rendered
# SCREENING MUST-HAVE TERMS list (via _render_must_have_block) follows below;
# this block is the standing rule that makes those terms outrank the priority
# keyword list.
_EDIT_MUST_HAVE_RULES = """\
=== SCREENING MUST-HAVE RULES ===
When a SCREENING MUST-HAVE TERMS section is present below, it OUTRANKS the \
priority keyword list. Every term marked MUST INCLUDE or MUST KEEP must appear \
VERBATIM in the final resume text — the term itself or one of its listed \
aliases, as whole words — with the same force as the exact-job-title rule. \
Weave each into the summary, a skills line, or a genuinely related bullet via \
substitution, within the one-page budget. Fluency and factual-accuracy rules \
below still apply — if a term truly cannot be placed truthfully AND fluently, \
leave it out (last resort). NEVER remove a MUST KEEP term while editing. \
NEVER include a DO NOT INCLUDE term.

"""

_GENERATE_EDITS_PROMPT = """\
You are an expert resume writer optimizing for ATS systems.
Generate specific, surgical edits to tailor this resume for the target role.

=== LENGTH CONSTRAINT — THIS IS THE #1 RULE ===
The master resume is {word_count} words. The tailored version MUST have FEWER OR EQUAL
words — hard cap is {max_word_count} words. The resume MUST remain ONE PAGE.

CHARACTER COUNT MATTERS MORE THAN WORD COUNT for page layout. A bullet with longer words
can cause line wrapping and push the resume to a second page even if the word count is equal.

You MUST NOT increase the total word count. Every edit must REPLACE existing words,
not add new ones. If you need to add a keyword, you MUST remove an equal or greater
number of words elsewhere. When in doubt, SHORTEN bullets slightly to make room.

HOW TO STAY WITHIN BUDGET:
- Every edit must be a SUBSTITUTION — swap existing words for JD keywords, net zero or negative.
- NEVER add new bullet points.
- NEVER expand an existing bullet. Prefer shortening slightly to create room for keywords.
- Each bullet replacement (both keyword-swap and full rewrite) MUST NOT exceed the character
  length of the original text. If the replacement is longer, shorten it — trim filler words,
  use abbreviations, or rephrase more concisely. Character count matters more than word count
  for page layout.
- The summary replacement must be the same length or shorter than the current summary.
- Do NOT add skills to the skills section beyond what fits on one line per subcategory.

=== KEYWORD DENSITY RULES ===
1. The EXACT job title "{exact_job_title}" MUST appear in both the title_line_replacement \
AND the professional summary. Use the FULL title — including seniority qualifiers like \
"Principal", "Senior", "Staff", or "Lead". NEVER drop or shorten the seniority level.
2. Target 25-35 keywords from the priority list. Currently at {current_count}. \
Need to add ~{additions_needed} more via SUBSTITUTION, not addition.
3. Use EXACT keyword strings from the JD. Not synonyms.

{must_have_rules}=== OTHER RULES ===
4. NEVER fabricate experience. Only reframe existing experience to align with the JD.
5. NEVER touch the Education section.
6. The bold prefix label before each experience_edit bullet's colon must NEVER be changed \
(use rewritten_bullets if the label needs to change).
7. FLUENCY OVER KEYWORDS — Every edited bullet MUST read as fluent, natural English. \
NEVER force-insert a keyword if it breaks grammar or sounds unnatural (e.g. "enterprise \
buyers satisfaction rating", "enterprise experience adoption", "multi-agent workflows \
Product Owners"). If a keyword cannot be inserted naturally, skip it — one missing keyword \
is far better than broken English.
8. NEVER change factual meaning. If the original says "C-suite executives", do NOT replace \
it with "strategic customers" — these are different audiences. Only substitute terms that \
are genuinely synonymous in this context. Changing who the candidate worked with, what \
they built, or what outcomes they achieved is fabrication.
9. FACTUAL KEYWORD INSERTION — Each keyword MUST be defensible based on the candidate's \
actual work described in that specific bullet. Do not insert a keyword just because it is \
thematically adjacent. BAD EXAMPLES (and why they are wrong):
   - Inserting "SRE practices" into a bullet about call center efficiency improvements: \
SRE (Site Reliability Engineering) is a platform reliability discipline — it is unrelated \
to call center operations and would be immediately flagged as dishonest by any technical interviewer.
   - Inserting "OpenTelemetry" into a 2022 bullet about conversational AI flow optimization: \
OpenTelemetry is a distributed tracing framework — the work described is NLP/dialogue management, \
not distributed systems observability. The connection is fabricated.
   - Inserting "multi-agent workflows" into a bullet about roadmap prioritization meetings: \
multi-agent is an AI architecture term — using it to describe a human meeting process is \
keyword stuffing that misrepresents the work.
   If a keyword cannot be inserted with full factual accuracy and professional credibility, \
SKIP IT.
10. KEYWORD PLACEMENT INTEGRITY — Only inject a keyword into a bullet if the bullet's \
original activity genuinely involved that concept. The keyword must describe what the person \
ACTUALLY DID, not just sound related. Bad example: changing "reporting macros" to "incident \
response macros" when the work was about generating reports, not responding to system incidents. \
Bad example: changing "toolset employee satisfaction" to "API satisfaction" — "API satisfaction" \
is not a real phrase. If a keyword cannot be placed without distorting the meaning of the \
original experience, SKIP that keyword.

=== FULL BULLET REWRITES (up to 2) ===
You may FULLY REWRITE up to 2 experience bullets to better highlight relevant experience \
for this role. Use rewritten_bullets (not experience_edits) for these. Rules:
- Must be based on REAL experience already on the resume — no fabrication
- Can reframe, re-emphasize, or restructure the bullet to foreground JD-relevant skills
- The bold prefix label before the colon CAN change in a rewrite (unlike experience_edits)
- The rewritten bullet MUST be the same character length or shorter than the original.
  Longer text causes line wrapping that pushes the resume to 2 pages. Count characters,
  not just words — a replacement with longer words can overflow even at equal word count.
- Only rewrite bullets where the candidate has relevant experience not well-highlighted
- The gap analysis rewrite_candidates field identifies good targets — prefer those
- A rewritten bullet still counts toward the overall word budget
- PRESERVE ALL SPECIFICS: Rewritten bullets MUST keep specific metrics, numbers, tools, \
and concrete details from the original. Only change the framing, emphasis, and keyword \
alignment — not the substance. "Automated macros and VBA scripts" must NOT become \
"automated integrations". Keep the specifics.
- NEVER replace specific tools or technologies with generic terms. If the original names \
a tool, technology, or process, keep it — even if the JD doesn't use that exact term.
- The rewrite must sound like the candidate wrote it — professional, specific, and grounded \
in real work. Generic corporate jargon ("strategic initiatives", "driving value", \
"cross-functional impact") signals a bot wrote it. Use concrete, specific language instead.

=== SECTIONS YOU MAY EDIT ===
- Professional Summary: Rewrite to lead with the exact job title and align with the \
role. Keep it the SAME LENGTH as the current summary. 2-3 sentences max.
- Skills section: Reorder skills WITHIN each sub-category to front-load JD-matched \
terms. Do not move skills between categories. Do not add so many skills that the line \
wraps to a second line.
- Work Experience bullets: Either (a) swap in exact JD keywords via experience_edits \
(label stays, only post-colon text changes), or (b) fully rewrite up to 2 bullets via \
rewritten_bullets (label may change, full text replaced).

RESPOND IN THIS EXACT JSON FORMAT (no markdown, no extra keys):
{{
  "title_line_replacement": "FULL EXACT JOB TITLE FROM POSTING — include ALL seniority qualifiers (e.g. 'PRINCIPAL PRODUCT MANAGER', not 'PRODUCT MANAGER')",
  "summary_replacement": "New professional summary paragraph text (same length as original)",
  "skills_reorder": {skills_reorder_example},
  "experience_edits": [
    {{
      "company": "EXAMPLE CORP",
      "role": "Role Title",
      "bullet_index": 0,
      "bold_label": "Enterprise Scale & Matrix Leadership",
      "original": "Full bullet text including label",
      "replacement_after_label": "Only the text AFTER the bold label colon — same length as original after-colon text",
      "keywords_added": ["keyword1", "keyword2"]
    }}
  ],
  "rewritten_bullets": [
    {{
      "company": "EXAMPLE CORP",
      "role": "Role Title",
      "bullet_index": 2,
      "original": "Full original bullet text including bold label and colon",
      "rewritten": "Full rewritten bullet text — bold label may differ; must be ≤ original CHARACTER count (not just word count)",
      "keywords_added": ["keyword1", "keyword2"]
    }}
  ],
  "keywords_used": ["full", "list", "of", "JD", "keywords", "now", "present"],
  "keyword_count": 30,
  "rationale": "Brief explanation of tailoring strategy"
}}

{must_have_block}Resume ({word_count} words — final result MUST be ≤ {max_word_count} words; shorten to fit):
{resume_text}

Target Role: {exact_job_title} at {company}

Priority Keywords (ranked):
{priority_keywords}

Gap Analysis:
{gap_analysis_json}
"""

_KEYWORD_CORRECTION_PROMPT = """\
The tailored resume has {count} keywords. Target is 25-35.
{direction_instruction}
{must_have_instruction}
WORD COUNT AND CHARACTER LENGTH CONSTRAINT STILL APPLIES:
- The master resume is {word_count} words. The result MUST have ≤ {max_word_count} words.
- You MUST NOT increase the total word count. Every keyword addition must REPLACE existing
  words — swap them, do not add on top. If you add a keyword, remove equal or more words.
- Prefer SHORTENING bullets slightly to make room for keywords rather than expanding them.
- When adjusting keywords, do not increase the character length of any bullet beyond its
  original length. Longer characters cause line wrapping and push the resume to 2 pages.
- Do NOT add new bullets or expand existing ones.

FLUENCY AND ACCURACY STILL APPLY:
- Every sentence must remain grammatically correct and read as natural English. Do NOT \
force-insert a keyword if it breaks grammar or sounds unnatural — skip it instead.
- Do NOT change factual meaning. Never substitute one audience, technology, or outcome \
for a different one just to include a keyword.
- FACTUAL KEYWORD INSERTION — each keyword must be defensible based on the actual work \
described in that specific bullet. Do not insert a keyword just because it is thematically \
adjacent. If a bullet is about call center efficiency, do NOT insert "SRE practices" — \
that is a platform reliability discipline, not a call center methodology. If in doubt, skip the keyword.

Currently matched: {matched}
Missing high-priority: {missing}

Regenerate the SAME edits JSON structure but with adjusted keyword density.
Respond with ONLY valid JSON — no markdown, no explanation.

Previous edits:
{previous_edits_json}
"""


def _build_skills_reorder_example() -> str:
    """Render the `skills_reorder` example fragment for _GENERATE_EDITS_PROMPT
    from the configured subcategory labels, instead of hardcoding one tree's
    category names into the LLM instructions (those names would steer the
    LLM toward the wrong keys, and apply_edits' `sub in
    _SKILL_SUBCATEGORY_LABELS` filter silently drops every skills_reorder
    edit whose key doesn't match the configured set).

    Reads `_SKILL_SUBCATEGORY_LABELS` fresh on every call (module global,
    same pattern as apply_edits). For a config whose subcategory dict
    preserves the same key order as before (config.yaml dict order is
    stable), the rendered fragment is byte-identical to the previous
    hardcoded block.
    """
    labels = _SKILL_SUBCATEGORY_LABELS
    if not labels:
        # Neutral tree with no configured subcategories yet: fall back to a
        # single illustrative key. skills_reorder keys must match whatever
        # policy.tailor.skill_subcategory_labels ends up configured with —
        # apply_edits silently drops any key not present there.
        return (
            '{\n    "Example Category": ["reordered", "skills", "here"]'
            "  // skills_reorder keys must match this tree's configured"
            " policy.tailor.skill_subcategory_labels keys\n  }"
        )
    lines = ",\n".join(
        f'    {json.dumps(sub)}: ["reordered", "skills", "here"]' for sub in labels
    )
    return "{\n" + lines + "\n  }"


def generate_edits(
    master_resume_text: str,
    jd_analysis: JDAnalysis,
    gap: GapAnalysis,
    company: str,
    inventory_text: str = "",
    must_haves: Optional[list] = None,
) -> TailoringEdits:
    """
    Step 3: Ask the LLM to generate structured edit instructions.
    Runs up to KEYWORD_CORRECTION_ROUNDS correction loops if keyword density is off.

    inventory_text grounds truthfulness (sent as the cached system prompt);
    must_haves are the stored Filter Match terms — credited ones (judge
    explicit/evidenced, or v1 present-in-master) are enforced verbatim through
    the correction loop. Both optional: omitted, this is the legacy edit pass.
    Returns a TailoringEdits dataclass.
    """
    import time
    word_count = len(master_resume_text.split())
    max_word_count = int(word_count * 0.98)
    logger.info(
        "Step 3: Generating tailored edits for '%s' (resume=%d words, budget=%d words)",
        jd_analysis.exact_job_title,
        word_count,
        max_word_count,
    )

    t0 = time.monotonic()
    # Annotate rewrite_candidates with original char counts so the LLM knows
    # exactly how much room it has when writing replacements.
    annotated_candidates = []
    for rc in gap.rewrite_candidates:
        if isinstance(rc, dict):
            orig = rc.get("original_bullet", "")
            annotated_candidates.append({**rc, "original_char_count": len(orig)})
        else:
            annotated_candidates.append(rc)

    # Gate the rules on the RENDERED block, not raw must_haves — a truthy but
    # unrenderable value (e.g. all-junk items) then yields neither the rules
    # nor a term list, never a rules section referencing an absent list.
    must_have_block = _render_must_have_block(must_haves or [])
    prompt = _GENERATE_EDITS_PROMPT.format(
        exact_job_title=jd_analysis.exact_job_title,
        current_count=gap.keyword_count,
        additions_needed=gap.target_additions,
        word_count=word_count,
        max_word_count=max_word_count,
        resume_text=master_resume_text[:4000],
        company=company,
        skills_reorder_example=_build_skills_reorder_example(),
        must_have_rules=_EDIT_MUST_HAVE_RULES if must_have_block else "",
        must_have_block=must_have_block,
        priority_keywords=json.dumps(jd_analysis.priority_keywords, indent=2),
        gap_analysis_json=json.dumps(
            {
                "already_matched": gap.already_matched,
                "missing_but_relevant": gap.missing_but_relevant,
                "rewrite_candidates": annotated_candidates,
            },
            indent=2,
        ),
    )

    truth_system = build_truth_system(inventory_text)
    raw = _claude_call(prompt, temperature=0.2, label="generate_edits",
                       model=TAILOR_EDIT_MODEL, timeout=EDIT_CALL_TIMEOUT,
                       system_prompt=truth_system)
    data = _parse_json_response(raw, context="generate_edits")

    if not data:
        logger.error("generate_edits: LLM returned unparseable response")
        return TailoringEdits()

    edits = _dict_to_tailoring_edits(data)
    _validate_edit_lengths(edits)
    logger.info(
        "Step 3 complete in %.1fs: keyword_count=%d, %d experience edits, %d full rewrites",
        time.monotonic() - t0,
        edits.keyword_count,
        len(edits.experience_edits),
        len(edits.rewritten_bullets),
    )

    # Step 4 (keyword count validation + correction loop)
    edits = _keyword_correction_loop(
        edits,
        jd_analysis.priority_keywords,
        master_resume_text,
        company,
        word_count=word_count,
        max_word_count=max_word_count,
        must_haves=must_haves or [],
        system_prompt=truth_system,
    )

    return edits


def _sanitize_text(value) -> str:
    """Collapse embedded newline/whitespace runs in `value` to single spaces and strip.

    This is the layout guard's ordinal-stability invariant made real: every LLM
    replacement string that gets inserted into the document as paragraph text
    must not contain '\\n', or a replaceAllText call would split one paragraph
    into two and silently shift every ordinal after it (see _doc_paragraphs).
    Returns "" for missing/non-string values so a malformed LLM response can't
    crash the pipeline or leak a literal "None" into the document.
    """
    if isinstance(value, str) and value:
        return " ".join(value.split())
    return ""


def _sanitize_dict_field(item, key: str):
    """Return a copy of dict `item` with item[key] whitespace/newline-sanitized,
    if present and a truthy string. All other keys — notably match anchors like
    'original' and 'bold_label' — are left byte-for-byte untouched. Non-dict
    items, and dicts missing/with a falsy or non-string value for `key`, pass
    through unchanged so a malformed edit entry can't crash the pipeline.
    """
    if not isinstance(item, dict):
        return item
    value = item.get(key)
    if not value or not isinstance(value, str):
        return item
    sanitized = dict(item)
    sanitized[key] = _sanitize_text(value)
    return sanitized


def _sanitize_skills_reorder(skills_reorder) -> dict:
    """Sanitize every skill token in `skills_reorder` via `_sanitize_text`.

    `skills_reorder` is `{subcategory: [skill, ...]}`; apply_edits joins each
    subcategory's list with ", " into a single paragraph line
    (f"{label} {', '.join(new_skills)}") that the layout guard tracks under
    role 'skills:<sub>'. An embedded '\\n' in any skill token would split that
    paragraph and corrupt ordinals — the same failure class the other four
    replacement fields are sanitized against.

    Skills that sanitize to empty are DROPPED (not kept as ""), so the joined
    line can never end up with a stray ", ". Subcategory key order and skill
    order are preserved. Malformed input — a non-dict `skills_reorder`, a
    non-list subcategory value, or a non-string skill (handled for free by
    `_sanitize_text`, which returns "" for non-strings) — is handled
    defensively rather than crashing the pipeline, matching
    `_sanitize_text`/`_sanitize_dict_field`'s style.
    """
    if not isinstance(skills_reorder, dict):
        return {}
    result = {}
    for sub, skills in skills_reorder.items():
        if not isinstance(skills, list):
            continue
        result[sub] = [s for s in (_sanitize_text(skill) for skill in skills) if s]
    return result


def _dict_to_tailoring_edits(data: dict) -> TailoringEdits:
    """The single chokepoint where raw LLM JSON becomes a TailoringEdits.

    Sanitizes exactly the fields that get INSERTED into the document as
    paragraph text (title, summary, each edit's replacement text, and every
    skill in skills_reorder) so they can never carry an embedded newline into
    replaceAllText. Match-anchor fields ('original', 'bold_label', etc.) are
    passed through untouched — they must keep the document's exact existing
    text to match at all.
    """
    return TailoringEdits(
        title_line_replacement=_sanitize_text(data.get("title_line_replacement", "")),
        summary_replacement=_sanitize_text(data.get("summary_replacement", "")),
        skills_reorder=_sanitize_skills_reorder(data.get("skills_reorder", {})),
        experience_edits=[
            _sanitize_dict_field(e, "replacement_after_label")
            for e in data.get("experience_edits", [])
        ],
        rewritten_bullets=[
            _sanitize_dict_field(b, "rewritten")
            for b in data.get("rewritten_bullets", [])[:2]
        ],
        keywords_used=data.get("keywords_used", []),
        keyword_count=data.get("keyword_count", 0),
        rationale=data.get("rationale", ""),
    )


def _validate_edit_lengths(edits: TailoringEdits) -> None:
    """
    Log warnings for any replacement text that exceeds the original by more than 5%.
    Longer replacements cause line wrapping that pushes the resume to a second page.
    """
    _THRESHOLD = 1.05

    for edit in edits.experience_edits:
        if not isinstance(edit, dict):
            continue
        original_full = edit.get("original", "")
        replacement = edit.get("replacement_after_label", "")
        if not original_full or not replacement:
            continue
        # Compare only the post-colon portion (the part that actually changes)
        original_after = original_full.split(":", 1)[1].strip() if ":" in original_full else original_full
        orig_len = len(original_after)
        repl_len = len(replacement)
        if orig_len > 0 and repl_len > orig_len * _THRESHOLD:
            logger.warning(
                "CHAR LENGTH WARNING — experience_edit '%s': replacement is %d chars longer than "
                "original (original=%d, replacement=%d, delta=+%.0f%%). May cause line overflow.",
                edit.get("bold_label", "?"),
                repl_len - orig_len,
                orig_len,
                repl_len,
                100 * (repl_len - orig_len) / orig_len,
            )

    for bullet in edits.rewritten_bullets:
        if not isinstance(bullet, dict):
            continue
        original = bullet.get("original", "")
        rewritten = bullet.get("rewritten", "")
        if not original or not rewritten:
            continue
        orig_len = len(original)
        repl_len = len(rewritten)
        if orig_len > 0 and repl_len > orig_len * _THRESHOLD:
            logger.warning(
                "CHAR LENGTH WARNING — rewritten_bullet '%s...': rewritten is %d chars longer than "
                "original (original=%d, rewritten=%d, delta=+%.0f%%). May cause line overflow.",
                original[:50],
                repl_len - orig_len,
                orig_len,
                repl_len,
                100 * (repl_len - orig_len) / orig_len,
            )

    if edits.summary_replacement:
        # We don't have the original summary here, but we log its length for reference.
        logger.debug(
            "summary_replacement length: %d chars — verify it does not exceed original summary length",
            len(edits.summary_replacement),
        )


# ---------------------------------------------------------------------------
# Step 4 — Keyword count validation loop
# ---------------------------------------------------------------------------

def _projected_text(edits: TailoringEdits, master_text: str = "") -> str:
    """Master text + every replacement fragment — the pre-apply approximation
    of the final document shared by keyword and must-have validation.

    Additions are exact (an edit fragment lands verbatim in the doc); text a
    substitution REMOVES is still counted here, so destruction is only caught
    by the pipeline's post-apply check against the real document text.
    """
    skills_text = " ".join(
        skill
        for skills in edits.skills_reorder.values()
        for skill in (skills if isinstance(skills, list) else [])
    )
    edit_text = " ".join(
        filter(
            None,
            [
                edits.title_line_replacement,
                edits.summary_replacement,
                skills_text,
                " ".join(
                    e.get("replacement_after_label", "")
                    for e in edits.experience_edits
                    if isinstance(e, dict)
                ),
                " ".join(
                    b.get("rewritten", "")
                    for b in edits.rewritten_bullets
                    if isinstance(b, dict)
                ),
            ],
        )
    )
    return master_text + " " + edit_text


def credited_must_haves(must_haves: list) -> list:
    """Items the tailored text is REQUIRED to contain verbatim (term or alias).

    v2 judged items: verdict explicit/evidenced — the judge already verified
    the inventory supports a truthful claim. v1 items (no verdict key):
    present:true — the term is in the master resume, so the tailor must not
    destroy it. Everything else (absent verdicts, present:false) is a
    prompt-level target only, never enforced.
    """
    out = []
    for item in must_haves or []:
        if not isinstance(item, dict) or not str(item.get("term", "")).strip():
            continue
        if "verdict" in item:
            if item.get("verdict") in ("explicit", "evidenced"):
                out.append(item)
        elif item.get("present") is True:
            out.append(item)
    return out


def missing_credited_must_haves(
    edits: TailoringEdits, must_haves: list, master_text: str = ""
) -> list:
    """Credited must-have items absent from the projected tailored text.

    Uses filter_match.evaluate_must_haves — the exact matcher the post-tailor
    recompute runs — so "present" here means the recompute will credit it.
    Returns the full items (term + aliases), not bare strings, so correction
    prompts can offer the aliases too.
    """
    credited = credited_must_haves(must_haves)
    if not credited:
        return []
    projected = _projected_text(edits, master_text)
    evaluated = evaluate_must_haves(
        projected,
        [{"term": m["term"], "aliases": m.get("aliases", [])} for m in credited],
    )
    return [m for m, ev in zip(credited, evaluated) if not ev["present"]]


def validate_keyword_count(
    edits: TailoringEdits, priority_keywords: list, master_text: str = ""
) -> tuple[str, int, list]:
    """
    Check whether the edit instructions hit the 25-35 keyword density target.
    Returns ("OK"|"UNDER"|"OVER", count, matched_keywords).

    Counts keywords across the FULL resume text (master_text unchanged portions
    + proposed edits merged in), mirroring the ATS checker's check_keyword_density()
    which runs against the actual final document.  Both use case-insensitive
    substring matching — the matching method is intentionally identical.
    """
    full_text = _projected_text(edits, master_text).lower()

    matched = [kw for kw in priority_keywords if kw.lower() in full_text]
    count = len(matched)

    if count < KEYWORD_MIN:
        return "UNDER", count, matched
    if count > KEYWORD_MAX:
        return "OVER", count, matched
    return "OK", count, matched


def _keyword_correction_loop(
    edits: TailoringEdits,
    priority_keywords: list,
    resume_text: str,
    company: str,
    word_count: int = 0,
    max_word_count: int = 0,
    must_haves: Optional[list] = None,
    system_prompt: Optional[str] = None,
) -> TailoringEdits:
    """
    Run up to KEYWORD_CORRECTION_ROUNDS LLM correction passes to hit 25-35 keywords
    AND to land every credited screening must-have verbatim.

    Why a loop instead of a single prompt?  The LLM frequently produces an initial
    edit set that lands at 20-24 keywords (cautious) or 36-40 (over-enthusiastic).
    A targeted correction prompt — showing exactly which keywords are matched vs.
    missing and telling it which direction to move — is more reliable than asking
    the initial prompt to self-correct. Each round re-validates the full output
    against the priority keyword list before deciding whether another pass is needed.

    must_haves are the stored Filter Match terms; credited ones (see
    credited_must_haves) must appear verbatim in the projected text or a
    correction round is spent even when density is already in band —
    grounding the tailor in the exact terms the post-tailor recompute checks.
    system_prompt (the inventory truth base) is forwarded to each correction call.
    """
    for round_num in range(1, KEYWORD_CORRECTION_ROUNDS + 1):
        status, count, matched = validate_keyword_count(edits, priority_keywords, master_text=resume_text)
        missing_mh = missing_credited_must_haves(edits, must_haves or [], resume_text)

        # Early exit: density in band AND every credited must-have landed.
        if status == "OK" and not missing_mh:
            logger.info("Keyword count OK: %d (target %d-%d)", count, KEYWORD_MIN, KEYWORD_MAX)
            edits.keyword_count = count
            edits.keywords_used = matched
            return edits

        logger.info(
            "Step 4 keyword correction round %d/%d: count=%d status=%s missing_must_haves=%d",
            round_num,
            KEYWORD_CORRECTION_ROUNDS,
            count,
            status,
            len(missing_mh),
        )

        # Build the set of keywords already present so we can compute what's missing.
        matched_set = {kw.lower() for kw in matched}
        missing = [kw for kw in priority_keywords if kw.lower() not in matched_set]

        # Direction-specific instruction: UNDER means add more via substitution;
        # OVER means trim lower-priority terms to avoid keyword-stuffing detection;
        # OK (only must-haves missing) means hold density and focus on those terms.
        if status == "UNDER":
            direction = (
                f"Weave in more keywords from the priority list via SUBSTITUTION. "
                f"Need at least {KEYWORD_MIN - count} more. Replace generic words, "
                f"do NOT add new bullets or expand existing ones."
            )
        elif status == "OVER":
            direction = (
                f"Remove some lower-priority keywords to avoid stuffing. "
                f"Need to drop {count - KEYWORD_MAX} keywords."
            )
        else:
            direction = (
                "Keyword count is already in range — do NOT change overall keyword "
                "density. Focus ONLY on the missing screening must-haves below."
            )

        correction_prompt = _KEYWORD_CORRECTION_PROMPT.format(
            count=count,
            direction_instruction=direction,
            must_have_instruction=_render_missing_must_have_instruction(missing_mh),
            word_count=word_count or len(resume_text.split()),
            max_word_count=max_word_count or int(len(resume_text.split()) * 0.98),
            matched=json.dumps(matched[:20], indent=2),
            missing=json.dumps(missing[:15], indent=2),
            previous_edits_json=json.dumps(
                {
                    "title_line_replacement": edits.title_line_replacement,
                    "summary_replacement": edits.summary_replacement,
                    "skills_reorder": edits.skills_reorder,
                    "experience_edits": edits.experience_edits,
                    "rewritten_bullets": edits.rewritten_bullets,
                },
                indent=2,
            ),
        )

        raw = _claude_call(correction_prompt, temperature=0.15, label=f"keyword_correction_r{round_num}", model=TAILOR_EDIT_MODEL, timeout=EDIT_CALL_TIMEOUT, system_prompt=system_prompt)
        new_data = _parse_json_response(raw, context=f"keyword_correction_r{round_num}")
        if new_data:
            edits = _dict_to_tailoring_edits(new_data)
            _validate_edit_lengths(edits)

    status, count, matched = validate_keyword_count(edits, priority_keywords, master_text=resume_text)
    missing_mh = missing_credited_must_haves(edits, must_haves or [], resume_text)
    edits.keyword_count = count
    edits.keywords_used = matched
    if status != "OK":
        logger.warning(
            "Keyword count still %s (%d) after %d correction rounds — proceeding with warning",
            status,
            count,
            KEYWORD_CORRECTION_ROUNDS,
        )
    if missing_mh:
        logger.warning(
            "Credited must-have terms still missing after %d correction rounds: %s",
            KEYWORD_CORRECTION_ROUNDS,
            ", ".join(m["term"] for m in missing_mh),
        )

    return edits


# ---------------------------------------------------------------------------
# Apply edits to the Google Doc
# ---------------------------------------------------------------------------

# The master resume's title line — the anchor apply_edits replaces and the
# layout guard's master reference for the "title" paragraph (:1400-1401).
# Single source, sourced from profile_policy (config.yaml policy.tailor.*):
# both consumers pick up the config value automatically through these
# module-level names.
_MASTER_TITLE_LINE = TAILOR_MASTER_TITLE_LINE

_SKILL_SUBCATEGORY_LABELS = dict(TAILOR_SKILL_SUBCATEGORY_LABELS)

# ---------------------------------------------------------------------------
# Layout-guard paragraph tracking
# ---------------------------------------------------------------------------

def _split_label(text: str):
    """Return ("Label:", "body") if text starts with a bold-label pattern
    ("Short Label: body" — ≤60 chars before the colon, no period), else (None, text)."""
    if ":" in text:
        label, rest = text.split(":", 1)
        if 0 < len(label) <= 60 and "." not in label:
            return label + ":", rest.strip()
    return None, text


def _doc_paragraphs(doc: dict) -> list:
    """[{ordinal, text, start, end}] for each paragraph in the doc body.
    Ordinals are stable across our edits: every edit replaces text WITHIN a
    paragraph, never adds/removes one. This holds because every field that
    gets inserted into the document as paragraph text — title_line_replacement,
    summary_replacement, experience_edits[].replacement_after_label,
    rewritten_bullets[].rewritten, and each skill in skills_reorder — is
    newline/whitespace-sanitized in _dict_to_tailoring_edits — the single
    chokepoint where raw LLM JSON becomes TailoringEdits — before it ever
    reaches the document, so replaceAllText can never split one paragraph
    into two."""
    out = []
    for element in doc.get("body", {}).get("content", []):
        paragraph = element.get("paragraph")
        if not paragraph:
            continue
        text = "".join(
            el.get("textRun", {}).get("content", "")
            for el in paragraph.get("elements", [])
        )
        out.append({
            "ordinal": len(out),
            "text": text.strip(),
            "start": element.get("startIndex", 0),
            "end": element.get("endIndex", 0),
        })
    return out


def build_edited_paragraphs(edits: TailoringEdits, pre_edit_doc: dict, post_edit_doc: dict) -> list:
    """Map each tailored edit to a live paragraph ordinal in the post-edit doc.

    Called ONCE per run, right after apply_edits. Matching tries the expected
    post-edit text first, then the master text (covers silent no-op edits),
    then a label-prefix fallback. Unmatched edits are logged and skipped —
    the guard can't enforce what it can't find.
    """
    post = _doc_paragraphs(post_edit_doc)

    def find_post(expected_texts, label=None):
        for rec in post:
            rec_norm = normalize(rec["text"])
            for exp in expected_texts:
                if exp and rec_norm == normalize(exp):
                    return rec
        if label:
            lab = _norm_il(normalize(label))
            for rec in post:
                if _norm_il(normalize(rec["text"])).startswith(lab):
                    return rec
        return None

    paras = []

    def track(role, master_text, rec):
        if rec is None:
            logger.warning("layout guard: could not locate paragraph for %s — not tracked", role)
            return
        paras.append(EditedParagraph(role=role, master_text=master_text, ordinal=rec["ordinal"]))

    # Title (always tracked — the master title line always exists)
    expected_title = (edits.title_line_replacement or "").upper()
    track("title", _MASTER_TITLE_LINE,
          find_post([expected_title, _MASTER_TITLE_LINE], label=None))

    # Summary
    master_summary = _extract_summary_text(pre_edit_doc)
    if master_summary and edits.summary_replacement:
        track("summary", master_summary.strip(),
              find_post([edits.summary_replacement, master_summary]))

    # Skills lines
    master_skill_lines = _extract_skill_line_texts(pre_edit_doc)
    for sub, skills in (edits.skills_reorder or {}).items():
        if not skills or sub not in _SKILL_SUBCATEGORY_LABELS:
            continue
        master_line = master_skill_lines.get(sub, "")
        if not master_line:
            continue
        label = _SKILL_SUBCATEGORY_LABELS[sub]
        new_line = f"{label} {', '.join(skills)}"
        track(f"skills:{sub}", master_line, find_post([new_line, master_line], label=label))

    # Keyword-swap bullets (label fixed)
    for i, e in enumerate(edits.experience_edits or []):
        if not isinstance(e, dict):
            continue
        orig = (e.get("original") or "").strip()
        bold_label = (e.get("bold_label") or "").strip()
        repl = (e.get("replacement_after_label") or "").strip()
        if not orig or not repl:
            continue
        expected_new = f"{bold_label}: {repl}" if bold_label else repl
        track(f"bullet:{e.get('company', '?')}:{bold_label or i}", orig,
              find_post([expected_new, orig], label=f"{bold_label}:" if bold_label else None))

    # Full rewrites (label may change)
    for i, b in enumerate(edits.rewritten_bullets or []):
        if not isinstance(b, dict):
            continue
        orig = (b.get("original") or "").strip()
        new = (b.get("rewritten") or "").strip()
        if not orig or not new:
            continue
        new_label, _ = _split_label(new)
        track(f"rewrite:{b.get('company', '?')}:{i}", orig,
              find_post([new, orig], label=new_label))

    return paras


def refresh_from_doc(paras: list, doc: dict) -> None:
    """Re-read each tracked paragraph's live text + positional range by ordinal.
    Ground truth every round — never trust a remembered/echoed string."""
    plist = _doc_paragraphs(doc)
    for p in paras:
        if 0 <= p.ordinal < len(plist):
            rec = plist[p.ordinal]
            p.tailored_text = rec["text"]
            p.start_index = rec["start"]
            p.end_index = rec["end"]
        else:
            logger.warning("refresh_from_doc: ordinal %d out of range for %s", p.ordinal, p.role)
            p.tailored_text = ""


def apply_paragraph_replacement(doc_id: str, v, new_text: str, client: GoogleAPIClient) -> bool:
    """Replace one paragraph's text, anchored on its LIVE text (v.tailored_text).

    - Sanitizes the replacement (no newlines/tabs — a stray newline would split
      the paragraph and break ordinal stability).
    - If the live text has a bold "Label:" prefix, anchors on the post-label
      body only, so the label's bold run is never inside the replaced range.
      A replacement that altered the label is rejected (reshape must keep it).
    - Reads occurrencesChanged from the batchUpdate response: 0 → returns False
      (silent no-op detected; the reconcile pass will revert positionally).
    """
    new_text = " ".join(new_text.split())
    # Only bullet/skills/rewrite paragraphs carry a bold "Label:" prefix. A colon
    # inside a title or summary is prose, not a label — treating it as one would
    # spuriously reject a valid summary reshape, so gate label detection on role.
    role = getattr(v, "role", "") or ""
    is_labeled = role.startswith(("bullet:", "skills:", "rewrite:"))
    cur_label, cur_body = _split_label(v.tailored_text) if is_labeled else (None, v.tailored_text)
    if cur_label:
        new_label, new_body = _split_label(new_text)
        if new_label != cur_label:
            logger.warning("apply_paragraph_replacement: %s label altered (%r → %r) — rejected",
                           v.role, cur_label, new_label)
            return False
        old_anchor, new_repl = cur_body, new_body
    else:
        old_anchor, new_repl = v.tailored_text, new_text

    if not old_anchor or old_anchor == new_repl:
        return False

    resp = client.batch_update(doc_id, [{
        "replaceAllText": {
            "containsText": {"text": old_anchor, "matchCase": True},
            "replaceText": new_repl,
        }
    }])
    occ = sum(r.get("replaceAllText", {}).get("occurrencesChanged", 0)
              for r in resp.get("replies", []))
    if occ == 0:
        logger.warning("apply_paragraph_replacement: %s anchor not found — silent no-op detected", v.role)
        return False
    if occ > 1:
        logger.warning("apply_paragraph_replacement: %s anchor matched %d times", v.role, occ)
    return True


def _set_paragraph_text(doc_id: str, para, new_text: str,
                        client: GoogleAPIClient) -> None:
    """Positional deleteContentRange + insertText + restyle to new_text.

    Does NOT depend on text matching, so it cannot silently no-op the way
    replaceAllText can. Re-reads the document for fresh indices immediately
    before building the request. Styling is deterministic per role: title is
    fully bold, summary fully non-bold, labeled bullets/skills bold through the
    colon and non-bold after.
    """
    doc = client.read_document(doc_id)
    refresh_from_doc([para], doc)
    if para.end_index <= para.start_index:
        logger.error("_set_paragraph_text: no valid range for %s — skipping", para.role)
        return
    start = para.start_index
    # A bullet's master_text is the LLM-echoed 'original' anchor, which never
    # passed the _dict_to_tailoring_edits sanitize chokepoint — collapse whitespace
    # / newlines here so insertText can't split the paragraph and break ordinals.
    text = " ".join(new_text.split())
    requests = [
        # endIndex - 1 keeps the paragraph's trailing newline (deleting it would
        # merge this paragraph into the next and break ordinal stability).
        {"deleteContentRange": {"range": {"startIndex": start, "endIndex": para.end_index - 1}}},
        {"insertText": {"location": {"index": start}, "text": text}},
    ]
    label, _ = _split_label(text)
    # A colon in a summary/title is prose, not a label — only bullet/skills/rewrite
    # paragraphs get the bold-label/non-bold-body split; everything else is uniform.
    is_labeled = para.role.startswith(("bullet:", "skills:", "rewrite:"))
    if para.role == "title":
        style_spans = [(start, start + len(text), True)]
    elif label and is_labeled:
        style_spans = [(start, start + len(label), True),
                       (start + len(label), start + len(text), False)]
    else:  # summary and any unlabeled paragraph are uniformly non-bold
        style_spans = [(start, start + len(text), False)]
    for s, e, bold in style_spans:
        if e > s:
            requests.append({"updateTextStyle": {
                "range": {"startIndex": s, "endIndex": e},
                "textStyle": {"bold": bold},
                "fields": "bold",
            }})
    client.batch_update(doc_id, requests)
    para.tailored_text = text


def revert_paragraph_to_master(doc_id: str, para, client: GoogleAPIClient) -> None:
    """Guaranteed revert to the paragraph's pre-edit master text (positional)."""
    _set_paragraph_text(doc_id, para, para.master_text, client)


# ---------------------------------------------------------------------------
# Layout guard: shorten-only LLM reshape + round-batched repair loop
# ---------------------------------------------------------------------------

MAX_LAYOUT_ROUNDS = 3

_RESHAPE_PROMPT = """\
Shorten this resume paragraph so it fits within {target_lines} rendered line(s) — \
at most {target_chars} characters total. Rules:
- If it begins with a bold label ending in a colon (e.g. "Product Vision:"), keep \
that label EXACTLY as written — never change or drop it.
- Keep the exact job title (if present), all keywords, tool names, and specific \
metrics/numbers.
{preserve_block}\
- Remove or condense filler words only. The result must read as fluent, natural English.
- Return ONLY the shortened text on a single line — no quotes, no explanation.

{current_text}"""

# Dangler repair is a MINIMAL trim: the paragraph already renders at its master
# line count; only its short last line offends. The grew-style "fit within N
# lines" framing over-shortens (the LLM cuts a whole line's worth), turning a
# cosmetic dangler into a shrank that forces a full revert-to-master.
_RESHAPE_DANGLER_PROMPT = """\
This resume paragraph renders at its correct {target_lines} lines but ends in a \
short {last_line_words}-word last line (a dangler). Remove or condense JUST a few \
words — enough for the last line to fill out. Rules:
- The result MUST stay LONGER than {min_chars} characters (cutting below that \
drops a whole rendered line, which forces this paragraph to be reverted entirely).
- If it begins with a bold label ending in a colon (e.g. "Product Vision:"), keep \
that label EXACTLY as written — never change or drop it.
- Keep the exact job title (if present), all keywords, tool names, and specific \
metrics/numbers.
{preserve_block}\
- The result must read as fluent, natural English.
- Return ONLY the shortened text on a single line — no quotes, no explanation.

{current_text}"""


def _render_preserve_block(preserve_terms: list) -> str:
    if not preserve_terms:
        return ""
    quoted = ", ".join(f'"{t}"' for t in preserve_terms)
    return (f"- SCREENING TERMS — these exact phrases MUST remain in your "
            f"output word-for-word (the employer literally searches for "
            f"them): {quoted}.\n")


def llm_reshape(current_text: str, target_chars: int, target_lines: int,
                kind: str = "grew", min_chars: int = 0,
                last_line_words: int = 0,
                preserve_terms: Optional[list] = None) -> Optional[str]:
    """Ask the LLM to SHORTEN a paragraph to fit its master line count.
    Shorten-only by design: re-expanding a shrunk paragraph without padding is
    incoherent, so shrank/lost paragraphs revert to master instead (spec §4).

    kind="dangler" switches to the minimal-trim prompt with a floor: the
    result is rejected when it cuts to <= min_chars (an over-cut renders one
    line short → shrank → forced revert, destroying the whole edit).
    preserve_terms are credited screening phrases present in current_text;
    a result that drops one (folding-aware, same matcher as the recompute)
    is rejected rather than applied.

    Returns None if the LLM did not actually shorten. A timed-out CLI call is
    retried once, then also treated as None — enforce_layout's round loop and
    reconciliation resolve the un-repaired violation (revert/keep), which
    beats aborting the whole guard into layout_unverified over one stalled
    call. Non-timeout CLI failures still raise ClaudeCLIError (systemic —
    the pipeline's layout-guard wrapper reports them)."""
    preserve_block = _render_preserve_block(preserve_terms or [])
    if kind == "dangler":
        prompt = _RESHAPE_DANGLER_PROMPT.format(
            target_lines=target_lines, min_chars=min_chars,
            last_line_words=last_line_words, preserve_block=preserve_block,
            current_text=current_text)
    else:
        prompt = _RESHAPE_PROMPT.format(
            target_lines=target_lines, target_chars=target_chars,
            preserve_block=preserve_block, current_text=current_text)
    try:
        result = _claude_call(prompt, temperature=0.1, label="layout_reshape")
    except ClaudeCLITimeout:
        # Reshape latency is NOT prompt-determined: identical ~600-char prompts
        # have measured 8s and 120s+ within one run, solo or concurrent — a
        # sporadic per-call stall. A fresh call usually completes normally.
        logger.warning("layout_reshape timed out — retrying the call once")
        try:
            result = _claude_call(prompt, temperature=0.1,
                                  label="layout_reshape_retry")
        except ClaudeCLITimeout:
            logger.warning(
                "layout_reshape timed out twice — treating as no-repair; the "
                "guard's rounds/reconciliation will revert or keep this "
                "paragraph instead")
            return None
    if not result:
        return None
    result = " ".join(result.strip().strip('"').strip("'").split())
    if not result or len(result) >= len(current_text):
        return None
    if min_chars and len(result) <= min_chars:
        logger.warning(
            "layout_reshape over-cut a dangler to %d chars (floor %d) — "
            "rejecting to avoid a shrank→revert cascade", len(result), min_chars)
        return None
    dropped = [t for t in (preserve_terms or []) if not phrase_present(result, t)]
    if dropped:
        logger.warning(
            "layout_reshape dropped credited screening term(s) %s — rejecting",
            dropped)
        return None
    return result


def _credited_variants_in_text(credited_items: list, text: str) -> list:
    """First literally-present variant (term or alias) of each credited item.

    The reshape prompt's preserve list: whichever wording of the credited term
    the paragraph actually carries. Presence uses the recompute's own folding
    matcher, so an "&" spelling of an "and" term counts.
    """
    out = []
    for item in credited_items or []:
        for cand in [str(item.get("term", "")).strip(),
                     *[str(a).strip() for a in (item.get("aliases") or [])]]:
            if cand and phrase_present(text or "", cand):
                out.append(cand)
                break
    return out


def _at_risk_terms(credited_items: list, text: str, master_text: str) -> list:
    """Credited variants present in text whose item the master text does NOT
    satisfy — exactly the terms a revert-to-master would destroy."""
    out = []
    for item in credited_items or []:
        cands = [str(item.get("term", "")).strip(),
                 *[str(a).strip() for a in (item.get("aliases") or [])]]
        cands = [c for c in cands if c]
        if not cands or any(phrase_present(master_text or "", c) for c in cands):
            continue                       # a revert keeps this item satisfied
        hit = next((c for c in cands if phrase_present(text or "", c)), None)
        if hit:
            out.append(hit)
    return out


def _dangler_floor(line_map: LineMap, text: str) -> int:
    """Char floor for a dangler trim: the combined length of the paragraph's
    rendered lines EXCEPT the last (those lines are full), plus joining spaces.
    A result at or under this fits in one fewer line — a shrank. Approximate by
    design; the authoritative gate stays the re-exported PDF measure."""
    span = lg.locate_paragraph(line_map, text)
    if span is None or span[1] <= span[0]:
        return 0
    kept = [normalize(line_map.lines[i].text) for i in range(span[0], span[1])]
    return sum(len(t) for t in kept) + (len(kept) - 1)


def enforce_layout(
    doc_id: str,
    edits: TailoringEdits,
    pre_edit_doc: dict,
    master_map: LineMap,
    client: GoogleAPIClient,
    credited_items: Optional[list] = None,
) -> tuple:
    """PDF-truth layout guard: every edited paragraph must render to the same
    line count as in the master, with no 1-4-word dangling last line.

    Round-batched (≤MAX_LAYOUT_ROUNDS): each round exports the PDF once, re-reads
    the doc (ground truth — anchors are never remembered strings), repairs all
    grew/dangler violations via shorten-only LLM calls, then re-measures next
    round. Anything still violating afterwards — including shrank/lost, which are
    never LLM-repaired — is positionally reverted to the master's original text.

    credited_items (stored Filter Match must-haves the judge credited) soften
    the guard's endgame: repairs must preserve those terms; a residual DANGLER
    whose text carries a credited term the master lacks is KEPT (cosmetic rule
    yields to screening coverage — the post-tailor recompute literally searches
    for these terms); a residual shrank/grew/lost with a recorded page-safe text
    still carrying the terms is positionally RESTORED to that text instead of
    the master. Line-count integrity (page length) is never traded: kept text
    always renders at the master's line count.

    Returns (final_pdf_bytes, warnings). final_pdf_bytes is reused by the ATS
    check so the pipeline never exports the same state twice.
    """
    warnings: list = []
    paras = None
    tailored_pdf = None
    tailored_map = None
    clean = False

    for rnd in range(1, MAX_LAYOUT_ROUNDS + 1):
        tailored_pdf = client.export_as_pdf(doc_id)
        tailored_map = build_line_map(tailored_pdf)
        doc = client.read_document(doc_id)
        if paras is None:
            paras = build_edited_paragraphs(edits, pre_edit_doc, doc)
        refresh_from_doc(paras, doc)

        violations = check_layout(master_map, tailored_map, paras)
        if not violations:
            clean = True
            break

        logger.info("layout guard round %d/%d: %d violation(s): %s",
                    rnd, MAX_LAYOUT_ROUNDS, len(violations),
                    [(v.role, v.kind) for v in violations])
        by_role = {p.role: p for p in paras}
        # A dangler renders at the master's line count — page-safe. Remember the
        # text as the restore candidate should a later repair over-cut it.
        for v in violations:
            if v.kind == "dangler":
                para = by_role.get(v.role)
                if para is not None:
                    para.page_safe_text = v.tailored_text
        repaired_any = False
        for v in violations:
            if v.kind in ("shrank", "lost"):
                continue                       # revert-only cases
            new = llm_reshape(
                v.tailored_text, v.target_chars, v.master_lines,
                kind=v.kind,
                min_chars=(_dangler_floor(tailored_map, v.tailored_text)
                           if v.kind == "dangler" else 0),
                last_line_words=v.last_line_words,
                preserve_terms=_credited_variants_in_text(
                    credited_items or [], v.tailored_text))
            if new and apply_paragraph_replacement(doc_id, v, new, client):
                repaired_any = True
        if not repaired_any:
            break                              # nothing changed — re-measuring won't help

    if not clean:
        # Reconciliation: fresh ground truth, then guaranteed positional
        # reverts — except where reverting would destroy credited screening
        # terms and a page-safe alternative exists (see docstring).
        tailored_pdf = client.export_as_pdf(doc_id)
        tailored_map = build_line_map(tailored_pdf)
        doc = client.read_document(doc_id)
        refresh_from_doc(paras, doc)
        residual = check_layout(master_map, tailored_map, paras)
        if residual:
            by_role = {p.role: p for p in paras}
            kept_roles = set()
            doc_changed = False
            for v in residual:
                para = by_role.get(v.role)
                if para is None:
                    continue
                if v.kind == "dangler":
                    at_risk = _at_risk_terms(
                        credited_items or [], para.tailored_text, para.master_text)
                    if at_risk:
                        kept_roles.add(v.role)
                        msg = (f"Layout guard kept '{v.role}' despite a "
                               f"{v.last_line_words}-word last line — reverting "
                               f"would drop credited screening term(s): "
                               f"{', '.join(at_risk)}")
                        logger.warning(msg)
                        warnings.append(msg)
                        continue
                else:
                    safe_risk = _at_risk_terms(
                        credited_items or [], para.page_safe_text, para.master_text)
                    if safe_risk:
                        _set_paragraph_text(doc_id, para, para.page_safe_text, client)
                        doc_changed = True
                        kept_roles.add(v.role)
                        msg = (f"Layout guard restored '{v.role}' to its last "
                               f"page-safe text instead of master ({v.kind}) — "
                               f"preserving credited screening term(s): "
                               f"{', '.join(safe_risk)}")
                        logger.warning(msg)
                        warnings.append(msg)
                        continue
                revert_paragraph_to_master(doc_id, para, client)
                doc_changed = True
                msg = (f"Layout guard reverted '{v.role}' to master text "
                       f"({v.kind}: master={v.master_lines} lines, "
                       f"tailored={v.tailored_lines}, last line={v.last_line_words} words)")
                logger.warning(msg)
                warnings.append(msg)
            # Final verification export (reused by the ATS check downstream).
            # Skipped when reconciliation changed nothing (kept-only): the
            # reconcile export already reflects the final doc.
            final_map = tailored_map
            if doc_changed:
                tailored_pdf = client.export_as_pdf(doc_id)
                final_map = build_line_map(tailored_pdf)
                doc = client.read_document(doc_id)
                refresh_from_doc(paras, doc)
            still = check_layout(master_map, final_map, paras)
            expected = [v for v in still
                        if v.role in kept_roles and v.kind == "dangler"]
            real = [v for v in still
                    if not (v.role in kept_roles and v.kind == "dangler")]
            if expected:
                logger.info(
                    "layout guard: %d dangler(s) deliberately kept to preserve "
                    "credited screening terms: %s",
                    len(expected), [v.role for v in expected])
            if real:
                msg = (f"Layout guard: {len(real)} violation(s) persist even after "
                       f"revert: {[(v.role, v.kind) for v in real]}")
                logger.error(msg)
                warnings.append(msg)
            if final_map.page_count != 1:
                msg = f"Layout guard: final PDF renders as {final_map.page_count} pages"
                logger.error(msg)
                warnings.append(msg)

    return tailored_pdf, warnings


def _restyle_label_paragraph(doc_id: str, paragraph_text: str, client: GoogleAPIClient) -> None:
    """Bold the 'Label:' prefix and un-bold the body of the paragraph whose FULL
    text equals paragraph_text. Fixes replaceAllText's style inheritance when a
    FULL paragraph (label included) was replaced — the whole replacement inherits
    the bold of the first matched char.

    Matches on the full paragraph text (not just the label prefix), so bullets
    that share a label prefix across companies are disambiguated — the exact
    rewritten paragraph is restyled, never an earlier same-labeled one. No-ops if
    paragraph_text has no bold-label prefix. Generalizes _fix_skills_bold_formatting.
    """
    label, _ = _split_label(paragraph_text)
    if not label:
        return
    target = _norm_il(normalize(paragraph_text))
    doc = client.read_document(doc_id)
    for element in doc.get("body", {}).get("content", []):
        paragraph = element.get("paragraph")
        if not paragraph:
            continue
        text = "".join(el.get("textRun", {}).get("content", "")
                       for el in paragraph.get("elements", [])).strip()
        if _norm_il(normalize(text)) != target:
            continue
        para_start = element["startIndex"]
        para_end = element["endIndex"] - 1          # exclude trailing newline
        label_end = para_start + len(label)
        requests = [{"updateTextStyle": {
            "range": {"startIndex": para_start, "endIndex": min(label_end, para_end)},
            "textStyle": {"bold": True}, "fields": "bold",
        }}]
        if label_end < para_end:
            requests.append({"updateTextStyle": {
                "range": {"startIndex": label_end, "endIndex": para_end},
                "textStyle": {"bold": False}, "fields": "bold",
            }})
        client.batch_update(doc_id, requests)
        return
    logger.warning("_restyle_label_paragraph: no paragraph matches %r", paragraph_text[:60])


def apply_edits(doc_id: str, edits: TailoringEdits, client: GoogleAPIClient) -> None:
    """
    Translate TailoringEdits into Google Docs batchUpdate requests and apply them.

    Edit order:
      1. replaceAllText for header title line and summary (index-safe)
      2. Skills section reordering (replaceAllText per sub-category line)
      3. Experience bullet edits (positional — applied in reverse order)

    The caller is responsible for reading the document first via client.read_document()
    if positional indices are needed. This function uses replaceAllText for all
    section-level edits to avoid index-shifting complexity.
    """
    if edits.title_line_replacement and not _MASTER_TITLE_LINE:
        raise RuntimeError(
            "policy.tailor.master_title_line is not set in config.yaml — the tailor "
            "cannot anchor edits to your master resume. See CUSTOMIZING.md."
        )

    requests_list = []
    request_labels = []

    # 1. Title line in header block
    if edits.title_line_replacement:
        # The master resume has a specific title line; replace it wholesale.
        # replaceAllText is safe across the entire document.
        requests_list.append(
            {
                "replaceAllText": {
                    "containsText": {
                        "text": _MASTER_TITLE_LINE,
                        "matchCase": True,
                    },
                    "replaceText": edits.title_line_replacement.upper(),
                }
            }
        )
        request_labels.append("title")

    # 2. Professional summary — replaceAllText requires the exact current text.
    #    We use a known anchor phrase from the master resume's summary opening.
    #    The caller should pass the actual current summary text if it has changed.
    #    For safety, we look for the opening phrase and replace the whole paragraph
    #    via a single replaceAllText. If the text has drifted, this will be a no-op
    #    and the apply_edits_surgical() path should be used instead.
    if edits.summary_replacement:
        # We'll handle summary replacement in apply_edits_surgical; here we
        # record it but skip the replaceAllText since the summary text is long
        # and fragile to match exactly.
        logger.debug("Summary replacement queued for surgical application")

    # 3. Skills reordering — read the current document to find the FULL text of each
    #    skills subcategory line (label + all items), then replace the entire line.
    #    Matching only the label prefix (e.g. "AI Platforms:") causes the Google Docs
    #    API to replace just that substring, leaving the rest of the original items
    #    appended — producing duplicated content. We must match the full line.
    skills_to_reorder = {
        sub: skills
        for sub, skills in edits.skills_reorder.items()
        if skills and sub in _SKILL_SUBCATEGORY_LABELS
    }
    modified_skill_subcategories: set = set()
    if skills_to_reorder:
        # Read the document once to extract current full skill line texts
        skills_doc = client.read_document(doc_id)
        current_skill_lines = _extract_skill_line_texts(skills_doc)

        for subcategory, new_skills in skills_to_reorder.items():
            label = _SKILL_SUBCATEGORY_LABELS[subcategory]
            new_line = f"{label} {', '.join(new_skills)}"
            current_line = current_skill_lines.get(subcategory, "")
            if not current_line:
                logger.warning(
                    "Could not find current skill line for '%s' in document — skipping", subcategory
                )
                continue
            if current_line == new_line:
                logger.debug("Skills line '%s' unchanged — skipping", subcategory)
                continue
            requests_list.append(
                {
                    "replaceAllText": {
                        "containsText": {"text": current_line, "matchCase": True},
                        "replaceText": new_line,
                    }
                }
            )
            request_labels.append(subcategory)
            modified_skill_subcategories.add(subcategory)

    if requests_list:
        logger.info("Applying %d batch update request(s) to doc %s", len(requests_list), doc_id)
        resp = client.batch_update(doc_id, requests_list)
        _warn_silent_noops("title/skills edit", request_labels, resp)

    # Fix bold bleed-through on skills lines: replaceAllText inherits the formatting
    # of the first character of the matched text. Since skill labels are bold, the
    # entire replacement (label + items) becomes bold. Un-bold the items after the label.
    if modified_skill_subcategories:
        _fix_skills_bold_formatting(doc_id, modified_skill_subcategories, client)

    # 4. Experience bullet edits — keyword-swap edits where only post-colon text changes.
    if edits.experience_edits:
        _apply_experience_edits(doc_id, edits.experience_edits, client)

    # 5. Full bullet rewrites — label may change; anchor is the complete original text.
    if edits.rewritten_bullets:
        _apply_rewritten_bullets(doc_id, edits.rewritten_bullets, client)

    # 6. Summary replacement — surgical: read doc, find the paragraph, replace text
    if edits.summary_replacement:
        _apply_summary_replacement(doc_id, edits.summary_replacement, client)


def _norm_il(s: str) -> str:
    """Normalize uppercase I and lowercase l to a common placeholder.

    In some fonts (and Google Docs exports) uppercase 'I' and lowercase 'l' are
    visually identical. The document may store either character for the 'Al' prefix
    in 'Al Platforms:'. Normalizing both to '|' before comparison lets us match
    regardless of which character is actually present.
    """
    return s.replace("I", "|").replace("l", "|")


def _extract_skill_line_texts(doc: dict) -> dict:
    """
    Walk the document body and return the full text of each skills subcategory line.

    Each skills line looks like: "<Subcategory Label>: item, item, ...". We match
    by checking that the paragraph text starts with one of the configured labels
    (_SKILL_SUBCATEGORY_LABELS — see profile_policy.TAILOR_SKILL_SUBCATEGORY_LABELS).
    Matching is robust to uppercase-I / lowercase-l confusion (visual lookalikes).

    Returns a dict mapping subcategory name → full line text (stripped).
    """
    labels = _SKILL_SUBCATEGORY_LABELS
    result = {}
    for element in doc.get("body", {}).get("content", []):
        paragraph = element.get("paragraph")
        if not paragraph:
            continue
        text = "".join(
            el.get("textRun", {}).get("content", "")
            for el in paragraph.get("elements", [])
        ).strip()
        for subcategory, label in labels.items():
            if subcategory in result:
                continue
            # Use I/l-normalized comparison to handle visual lookalike characters
            if _norm_il(text).startswith(_norm_il(label)):
                logger.debug("_extract_skill_line_texts: matched %r → %r", subcategory, text[:60])
                result[subcategory] = text
                break
    if not result:
        logger.warning(
            "_extract_skill_line_texts: no skill lines found — "
            "labels may not match document text; check for I/l confusion"
        )
    else:
        logger.debug("_extract_skill_line_texts: found %d skill line(s): %s", len(result), list(result))
    return result


def _fix_skills_bold_formatting(
    doc_id: str, modified_subcategories: set, client: GoogleAPIClient
) -> None:
    """
    After skills replaceAllText, un-bold the skill items (text after the label colon+space).

    replaceAllText inherits the formatting of the first character of the matched text.
    Since skill labels are bold, the entire replacement line becomes bold. This function
    re-reads the document, locates each modified skill line by its label prefix, and
    applies updateTextStyle(bold=False) on the range from label_end to paragraph_end.
    Label matching is robust to uppercase-I / lowercase-l confusion (visual lookalikes).
    """
    labels = _SKILL_SUBCATEGORY_LABELS
    logger.info(
        "_fix_skills_bold_formatting: called for subcategories=%s doc=%s",
        modified_subcategories, doc_id
    )
    doc = client.read_document(doc_id)
    unbold_requests = []

    for element in doc.get("body", {}).get("content", []):
        paragraph = element.get("paragraph")
        if not paragraph:
            continue
        text = "".join(
            el.get("textRun", {}).get("content", "")
            for el in paragraph.get("elements", [])
        )
        text_stripped = text.strip()
        for subcategory, label in labels.items():
            if subcategory not in modified_subcategories:
                continue
            # Use I/l-normalized comparison to handle visual lookalike characters
            if _norm_il(text_stripped).startswith(_norm_il(label)):
                logger.info(
                    "Un-bolding skill line for %r: %r (label=%r)",
                    subcategory, text_stripped[:60], label
                )
                # label + one space = where the skill items begin
                label_end_offset = len(label) + 1
                para_start = element["startIndex"]
                # endIndex includes the trailing newline; exclude it from the style range
                para_end = element["endIndex"] - 1
                item_start = para_start + label_end_offset
                if item_start < para_end:
                    unbold_requests.append({
                        "updateTextStyle": {
                            "range": {
                                "startIndex": item_start,
                                "endIndex": para_end,
                            },
                            "textStyle": {"bold": False},
                            "fields": "bold",
                        }
                    })
                break
            else:
                logger.debug(
                    "_fix_skills_bold_formatting: no match for %r — text=%r label=%r",
                    subcategory, text_stripped[:60], label
                )

    if unbold_requests:
        logger.info(
            "Un-bolding skill items in %d line(s) for doc %s",
            len(unbold_requests), doc_id
        )
        client.batch_update(doc_id, unbold_requests)
    else:
        logger.warning(
            "_fix_skills_bold_formatting: no paragraphs matched for subcategories=%s — "
            "skill items may still be fully bold",
            modified_subcategories
        )


def _warn_silent_noops(kind: str, labels: list, resp: dict) -> None:
    """replaceAllText reports batch success even when an anchor matched
    nothing — the reply just omits occurrencesChanged (Google drops
    zero-valued fields). Replies align 1:1 with requests, so a missing or
    zero count IS the silent-no-op signal; surface it per edit."""
    replies = resp.get("replies", []) if isinstance(resp, dict) else []
    for i, label in enumerate(labels):
        reply = replies[i] if i < len(replies) else {}
        occ = (reply or {}).get("replaceAllText", {}).get("occurrencesChanged", 0)
        if occ == 0:
            logger.warning(
                "%s '%s': replaceAllText matched nothing — edit NOT applied "
                "(anchor drifted from document text)", kind, label)
        elif occ > 1:
            logger.warning("%s '%s': anchor matched %d occurrences — all replaced",
                           kind, label, occ)


def _apply_experience_edits(
    doc_id: str, experience_edits: list, client: GoogleAPIClient
) -> None:
    """
    Apply bullet text replacements for work experience.

    For each edit, we use replaceAllText matching the original text after the
    bold label, replacing it with the new text. This avoids positional index math.

    Constraint: the original text must be unique in the document. If duplicates
    exist, replaceAllText will replace all occurrences (acceptable for resumes
    since bullet text is generally unique).
    """
    bullet_requests = []
    request_labels = []
    for edit in experience_edits:
        if not isinstance(edit, dict):
            continue
        bold_label = edit.get("bold_label", "")
        original_full = edit.get("original", "")
        replacement_after = edit.get("replacement_after_label", "")

        if not original_full or not replacement_after:
            logger.debug("Skipping experience edit: missing original or replacement")
            continue

        # Find a unique substring of original_full to avoid too-broad matches.
        # Use the text after the label (which is what changes).
        old_after_colon = original_full.split(":", 1)[1].strip() if ":" in original_full else original_full
        if old_after_colon and old_after_colon != replacement_after:
            bullet_requests.append(
                {
                    "replaceAllText": {
                        "containsText": {"text": old_after_colon, "matchCase": True},
                        "replaceText": replacement_after,
                    }
                }
            )
            request_labels.append(
                bold_label or f"{edit.get('company', '?')} — {edit.get('role', '?')}")

    if bullet_requests:
        logger.info("Applying %d experience bullet edit(s)", len(bullet_requests))
        resp = client.batch_update(doc_id, bullet_requests)
        _warn_silent_noops("experience edit", request_labels, resp)


def _apply_rewritten_bullets(
    doc_id: str, rewritten_bullets: list, client: GoogleAPIClient
) -> None:
    """
    Apply full bullet rewrites for work experience.

    Unlike _apply_experience_edits (which anchors on the post-colon text so the
    bold label is never touched), rewritten bullets may change the bold label too.
    We therefore anchor on the FULL original bullet text (label + colon + body)
    and replace it with the full rewritten text.

    Limited to 2 rewrites per TailoringEdits (enforced in _dict_to_tailoring_edits).
    """
    rewrite_requests = []
    request_labels = []
    for bullet in rewritten_bullets:
        if not isinstance(bullet, dict):
            continue
        original = bullet.get("original", "").strip()
        rewritten = bullet.get("rewritten", "").strip()

        if not original or not rewritten:
            logger.debug("Skipping rewritten bullet: missing original or rewritten text")
            continue

        if original == rewritten:
            logger.debug("Skipping rewritten bullet: original and rewritten are identical")
            continue

        rewrite_requests.append(
            {
                "replaceAllText": {
                    "containsText": {"text": original, "matchCase": True},
                    "replaceText": rewritten,
                }
            }
        )
        request_labels.append(
            f"{bullet.get('company', '?')} — {bullet.get('role', '?')}")
        logger.debug(
            "Queued full rewrite: '%s...' → '%s...'",
            original[:60],
            rewritten[:60],
        )

    if rewrite_requests:
        logger.info("Applying %d full bullet rewrite(s)", len(rewrite_requests))
        resp = client.batch_update(doc_id, rewrite_requests)
        _warn_silent_noops("bullet rewrite", request_labels, resp)

    # replaceAllText inherits the bold of the first matched char (the label),
    # leaving the whole rewritten bullet bold. Restore label-bold/body-plain.
    for bullet in rewritten_bullets:
        if not isinstance(bullet, dict):
            continue
        rewritten = bullet.get("rewritten", "").strip()
        if rewritten:
            _restyle_label_paragraph(doc_id, rewritten, client)


def _apply_summary_replacement(
    doc_id: str, new_summary: str, client: GoogleAPIClient
) -> None:
    """
    Replace the Professional Summary paragraph text.

    Strategy: read the document, find the paragraph after the "PROFESSIONAL SUMMARY"
    header (identified by bold ALL-CAPS text), extract its current text, then use
    replaceAllText to swap it for the new text.
    """
    doc = client.read_document(doc_id)
    current_summary = _extract_summary_text(doc)
    if not current_summary:
        logger.warning("Could not locate Professional Summary paragraph — skipping summary replacement")
        return

    if current_summary.strip() == new_summary.strip():
        logger.debug("Summary unchanged — skipping")
        return

    client.batch_update(
        doc_id,
        [
            {
                "replaceAllText": {
                    "containsText": {"text": current_summary.strip(), "matchCase": True},
                    "replaceText": new_summary.strip(),
                }
            }
        ],
    )
    logger.info("Applied summary replacement (%d chars)", len(new_summary))


def _extract_summary_text(doc: dict) -> str:
    """
    Walk the document body to find the paragraph immediately after the
    "PROFESSIONAL SUMMARY" section header and return its text.

    Section headers in the master resume are bold ALL-CAPS NORMAL_TEXT paragraphs
    (NOT Heading styles — see plan section 1, Google Docs API Representation).
    """
    body_content = doc.get("body", {}).get("content", [])
    after_summary_header = False

    for element in body_content:
        paragraph = element.get("paragraph")
        if not paragraph:
            continue

        text = "".join(
            el.get("textRun", {}).get("content", "")
            for el in paragraph.get("elements", [])
        ).strip()

        if after_summary_header:
            # Skip horizontal rules and empty lines
            if text and text != "\n":
                return text
            continue

        if _is_section_header(paragraph, text) and "SUMMARY" in text.upper():
            after_summary_header = True

    return ""


def _is_section_header(paragraph: dict, text: str) -> bool:
    """
    Detect a section header: bold ALL-CAPS NORMAL_TEXT paragraph.
    (Not styled as HEADING_1/2 — see plan section 1.)
    """
    if not text.strip():
        return False
    if text.strip() != text.strip().upper():
        return False  # Not all-caps
    elements = paragraph.get("elements", [])
    if not elements:
        return False
    first_run = elements[0].get("textRun", {})
    return bool(first_run.get("textStyle", {}).get("bold", False))
