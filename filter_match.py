"""Filter Match: employer-screening survival estimate (pure functions).

Shared by engine/llm_scorer.py (scrape-time, master-resume basis) and
resume_tailor/pipeline.py (post-tailor recompute). Lives at the repo root
because engine/ must not import resume_tailor/ — that package's __init__
pulls in the Google API stack. resume_tailor.ats_checker re-exports
keyword_matches from here so its existing consumers are untouched.

Spec: the 2026-07-08 filter-match score design (private repo notes).
"""
import json
import os
import re
from datetime import datetime

# Score weights. Recruiters lean on boolean keyword search far more than exact
# titles, and the tailor pipeline force-inserts the exact title anyway — so
# coverage dominates. Knockouts gate rather than rank (see KNOCKOUT_CAP).
COVERAGE_POINTS = 75
TITLE_POINTS = {"exact": 10, "close": 5, "none": 0}
KNOCKOUT_POINTS = 15
UNCLEAR_PENALTY = 5
KNOCKOUT_CAP = 15

# Written by resume_tailor.pipeline on every tailor run; read by LLMScorer.
MASTER_RESUME_CACHE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "data", "master_resume.txt"
)


# Standalone "&" ⇄ "and": resume register writes "Experimentation & Funnel
# Analysis" where a stored term says "experimentation and funnel analysis".
# Only a free-standing "&" token folds — "AT&T", "R&D", "Q&A" keep their
# intra-token "&". Both sides of every compare fold to the "and" form.
_AMP_TOKEN_RE = re.compile(r"(?<=\s)&(?=\s)")


def _fold_amp(text: str) -> str:
    return _AMP_TOKEN_RE.sub("and", text)


def _token_present(phrase: str, text_lower: str) -> bool:
    """Whole-token presence of phrase in pre-lowercased text.

    Boundary assertion is alphanumeric look-arounds, NOT \\b — \\b silently
    fails on terms ending in non-word characters ("C++", ".NET"). The optional
    trailing "s" lets a singular extraction match plural resume text ("LLM" →
    "LLMs"); the reverse direction is covered by LLM-provided aliases. Like
    the standalone-"&"→"and" fold, this plural tolerance is intentional and,
    via keyword_matches, is shared with
    resume_tailor.ats_checker.check_keyword_density (the ATS keyword check) —
    changing it here changes ATS keyword-density results too.
    """
    kw = _fold_amp(phrase.lower().strip())
    if not kw:
        return False
    pattern = r"(?<![a-z0-9])" + re.escape(kw) + r"s?(?![a-z0-9])"
    return re.search(pattern, _fold_amp(text_lower)) is not None


def phrase_present(text: str, phrase: str) -> bool:
    """Case-insensitive whole-token presence of phrase in text."""
    return _token_present(phrase, text.lower())


def keyword_matches(text: str, keywords: list) -> list:
    """Return the keywords present in text as whole tokens (order preserved).

    Shared by the ATS keyword check, the tailor pipeline's reported match
    count, and the Filter Match score — one matcher, identical rules.
    """
    text_lower = text.lower()
    return [kw for kw in keywords if _token_present(kw, text_lower)]


def evaluate_must_haves(text: str, must_haves: list) -> list:
    """[{term, aliases}] -> [{term, aliases, present}] against text.

    A must-have counts as present when its term OR any alias matches as a
    whole token. Blank terms are dropped; a missing aliases key is tolerated.
    """
    text_lower = text.lower()
    out = []
    for item in must_haves or []:
        term = str(item.get("term", "")).strip()
        if not term:
            continue
        aliases = [str(a).strip() for a in (item.get("aliases") or [])
                   if str(a).strip()]
        present = any(_token_present(t, text_lower) for t in [term, *aliases])
        out.append({"term": term, "aliases": aliases, "present": present})
    return out


def resolve_title_tier(text: str, jd_title: str, title_variants: list,
                       llm_alignment: str) -> str:
    """'exact' if the JD title or any variant is literally in the resume text;
    else 'close' when the LLM judged the candidate's titles aligned (its
    'exact' claim without literal presence also caps at 'close'); else 'none'.
    """
    text_lower = (text or "").lower()
    candidates = [str(t) for t in [jd_title, *(title_variants or [])]
                  if t and str(t).strip()]
    if any(_token_present(t, text_lower) for t in candidates):
        return "exact"
    if str(llm_alignment or "").lower() in ("exact", "close"):
        return "close"
    return "none"


def compute_filter_score(must_haves: list, title_tier: str,
                         knockouts: list) -> tuple:
    """Return (score 0-100, knocked_out bool). must_haves items carry 'present'.

    Raises ValueError on empty must_haves — no denominator means no score;
    callers leave the filter fields NULL instead (never fabricate from zero).
    """
    if not must_haves:
        raise ValueError("must_haves is empty — leave filter fields NULL")
    present = sum(1 for m in must_haves if m.get("present"))
    coverage = COVERAGE_POINTS * present / len(must_haves)
    title_pts = TITLE_POINTS.get(title_tier, 0)

    verdicts = [str(k.get("verdict", "")).lower() for k in (knockouts or [])]
    knocked_out = "failed" in verdicts
    unclear = verdicts.count("unclear")
    ko_pts = max(0, KNOCKOUT_POINTS - UNCLEAR_PENALTY * unclear)

    score = round(coverage + title_pts + ko_pts)
    if knocked_out:
        # A failed knockout is an auto-reject in real ATS flows — no keyword
        # count rescues it. The boolean is the queryable fact; the capped
        # number keeps the displayed score honest.
        score = min(score, KNOCKOUT_CAP)
    return max(0, min(100, score)), knocked_out


# --- v2: judged ("tailorable ceiling") scoring -----------------------------

# Failed-knockout gate applied to the DASHBOARD score at display/query time.
# jobs.score itself is never mutated (rescore-force inverts the blend from it).
KNOCKOUT_GATE_CAP = 40

# SQL twin of effective_score() for use inside queries (auto-tailor gate).
# Keep in sync with effective_score below and effectiveScore() in
# dashboard/static/index.html. COALESCE mirrors the Python int(score or 0):
# a NULL score must gate to 0, not propagate NULL through SQL's 3-valued logic.
EFFECTIVE_SCORE_SQL = (
    f"(CASE WHEN filter_knockout = 1 AND COALESCE(score, 0) > {KNOCKOUT_GATE_CAP} "
    f"THEN {KNOCKOUT_GATE_CAP} ELSE COALESCE(score, 0) END)"
)


def effective_score(score, knocked_out) -> int:
    """Dashboard/alert-facing score: gated to KNOCKOUT_GATE_CAP on knockout."""
    s = int(score or 0)
    return min(s, KNOCKOUT_GATE_CAP) if knocked_out else s


def compute_judged_filter_score(judged_must_haves: list, title_claim: str,
                                knockouts: list) -> tuple:
    """Ceiling score from judge verdicts: (score, knocked_out, uncapped).

    "explicit" and "evidenced" both count fully — the ceiling assumes a
    truthful tailor closes the wording gap. Same constants and knockout
    semantics as the literal compute_filter_score; the uncapped value is
    returned so the UI can show "15 (90 before knockout cap)".
    Raises ValueError on empty must_haves (no denominator, no score).
    """
    if not judged_must_haves:
        raise ValueError("judged must_haves is empty — leave filter fields NULL")
    credited = sum(1 for m in judged_must_haves
                   if m.get("verdict") in ("explicit", "evidenced"))
    coverage = COVERAGE_POINTS * credited / len(judged_must_haves)
    title_pts = TITLE_POINTS.get(title_claim, 0)

    verdicts = [str(k.get("verdict", "")).lower() for k in (knockouts or [])]
    knocked_out = "failed" in verdicts
    unclear = verdicts.count("unclear")
    ko_pts = max(0, KNOCKOUT_POINTS - UNCLEAR_PENALTY * unclear)

    uncapped = max(0, min(100, round(coverage + title_pts + ko_pts)))
    score = min(uncapped, KNOCKOUT_CAP) if knocked_out else uncapped
    return score, knocked_out, uncapped


def build_filter_json_v2(judged_must_haves: list, title_variants: list,
                         title_alignment: str, title_claim: str,
                         knockouts: list, uncapped_score: int,
                         inventory_sha: str, basis: str) -> str:
    """Serialize the v2 (judged) filter detail blob for jobs.filter_json.

    Keeps term+aliases on every must-have so the post-tailor literal
    recompute (resume_tailor/pipeline.py) keeps working unchanged, and keeps
    title_variants/title_alignment for the same reason. inventory_sha256 is
    the staleness key --rejudge-filter compares against the current file.
    """
    return json.dumps({
        "version": 2,
        "must_haves": judged_must_haves,
        "title_variants": title_variants or [],
        "title_alignment": title_alignment,
        "title_claim": title_claim,
        "knockouts": knockouts or [],
        "uncapped_score": uncapped_score,
        "inventory_sha256": inventory_sha,
        "basis": basis,
        "judged_at": datetime.now().isoformat(timespec="seconds"),
    })


def build_filter_json(evaluated: list, title_variants: list, title_tier: str,
                      knockouts: list, title_alignment: str) -> str:
    """Serialize the filter detail blob stored in jobs.filter_json."""
    return json.dumps({
        "must_haves": evaluated,
        "title_variants": title_variants or [],
        "title_tier": title_tier,
        "title_alignment": title_alignment,
        "knockouts": knockouts or [],
        "computed_at": datetime.now().isoformat(timespec="seconds"),
    })
