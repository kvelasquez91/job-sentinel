"""Filter Match v2 judge: prompt construction + response parsing (pure).

Stage 2 of the two-stage Filter Match design. Stage 1 (engine/llm_scorer.py)
extracts requirements from the JD; this module judges what the candidate can
TRUTHFULLY claim, against the experience inventory. The CLI invocation,
retries, and threading stay in LLMScorer — this module never calls the CLI.

Lives at the repo root because engine/ must not import resume_tailor/.

Spec: the 2026-07-09 filter-match semantic-judge design (private repo notes).
"""
import hashlib
import json
import logging
import os
import re
from typing import Optional

from filter_match import MASTER_RESUME_CACHE

logger = logging.getLogger(__name__)

INVENTORY_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "data",
    "experience_inventory.md")

# Near-miss years requirements soften from "failed" to "unclear" at >= 70%
# of the ask (recruiters flex years for strong candidates; screening forms
# are the reason "failed" still exists at all). Epsilon guards float noise
# on exact-ratio boundaries like 8.4/12.
SOFT_YEARS_THRESHOLD = 0.7
_EPS = 1e-9

JUDGE_SYSTEM = (
    "You are a skeptical employment-screening judge. You decide what a "
    "candidate can TRUTHFULLY claim, based only on the provided inventory. "
    "Respond with ONLY a single valid JSON object — no markdown fences, no "
    "commentary."
)

# Static judge context — persona, rules, response shape, and the candidate
# inventory. Sent as the CLI system prompt so consecutive judge calls share
# an identical prefix (prompt-cache reuse). Per-job content stays in _PROMPT
# (the user message). JUDGE_SYSTEM stays first: callers and test fakes
# identify judge calls by its "screening judge" substring.
_JUDGE_SYSTEM_TEMPLATE = JUDGE_SYSTEM + """

Judge what the candidate can TRUTHFULLY claim against each requirement of
the job in the user message, using ONLY the CANDIDATE INVENTORY below as
evidence.

RULES — apply strictly:
- Verdict per requirement: "explicit" when the inventory states it in nearly
  the same words; "evidenced" when the inventory genuinely demonstrates it,
  such that a truthful resume rewrite could state it; otherwise "absent".
- Every "explicit" or "evidenced" verdict MUST include "evidence": a short
  quote or close paraphrase of the specific inventory line that supports it.
  No evidence means no credit.
- The inventory's NOT CLAIMED section is binding: anything listed there is
  "absent", even if other text seems related.
- The inventory's BASELINE PRODUCT CRAFT section covers generic
  product-management craft only — it never evidences specific tools,
  technologies, or domains.
- Judge in the context of the job title: product-side experience with a
  technology satisfies a product-role requirement but not a hands-on
  engineering requirement.
- Knockouts: judge against the inventory's HARD FACTS: "met" when the facts
  clearly satisfy the requirement, "failed" when they clearly contradict it,
  "unclear" otherwise. Give a one-line "reason". When a knockout is about
  years of experience, also return "required_years" and "candidate_years" as
  numbers; otherwise return null for both.
- title_claim: could the candidate truthfully present their titles as
  equivalent to this job's title? "exact" | "close" | "none".

Return the must_haves array in the SAME ORDER as the numbered REQUIREMENTS
TO JUDGE in the user message — exactly one entry per requirement — and copy
each requirement's term VERBATIM into the "term" field. Do the same for
knockouts.

Respond with ONLY this JSON shape:
{{"must_haves": [{{"term": "<string>", "verdict": "explicit|evidenced|absent", "evidence": "<string or null>"}}], "knockouts": [{{"requirement": "<string>", "verdict": "met|unclear|failed", "reason": "<string>", "required_years": <number or null>, "candidate_years": <number or null>}}], "title_claim": "exact|close|none"}}

CANDIDATE INVENTORY:
{inventory}
"""

# Volatile per-job content — the entire user message for a judge call.
_PROMPT = """\
JOB TITLE: {title}
ACCEPTABLE TITLE VARIANTS: {variants}
POSTED TOP-OF-BAND: {posted_top_of_band}

REQUIREMENTS TO JUDGE:
{requirements}

KNOCKOUT REQUIREMENTS TO JUDGE:
{knockouts}
"""


def load_inventory() -> tuple:
    """Return (text, sha256_hex, basis) for the judge's evidence base.

    Prefers the experience inventory; falls back to the master resume cache
    (logged — coverage reads low against a resume). ("", "", "none") when
    neither exists, so the caller can skip judging rather than fabricate.
    """
    for path, basis in ((INVENTORY_PATH, "inventory"),
                        (MASTER_RESUME_CACHE, "resume_fallback")):
        try:
            with open(path, encoding="utf-8") as fh:
                text = fh.read().strip()
        except OSError:
            continue
        if text:
            if basis == "resume_fallback":
                logger.warning(
                    "Experience inventory missing/empty (%s) — judging "
                    "against master resume text; coverage may read low.",
                    INVENTORY_PATH)
            sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
            return text, sha, basis
    logger.warning(
        "Neither experience inventory (%s) nor master resume cache (%s) is "
        "usable — Filter Match judging disabled until one exists.",
        INVENTORY_PATH, MASTER_RESUME_CACHE)
    return "", "", "none"


def build_judge_system(inventory_text: str) -> str:
    """Render the static judge system prompt (persona + rules + inventory).

    Constant for a given inventory, so every judge call in a run sends an
    identical system prompt — the prompt-cache prefix.
    """
    return _JUDGE_SYSTEM_TEMPLATE.format(inventory=inventory_text)


def build_judge_prompt(job_title: str, title_variants: list,
                       must_have_keywords: list,
                       knockout_requirements: list,
                       posted_top_of_band: Optional[float] = None) -> str:
    """Render the per-job judge user message. must_have_keywords items: {term, aliases}.

    posted_top_of_band: max of the posting's salary band, pre-computed by the
    caller — the judge sees the exact integer the $300K location-exception
    predicate uses (inventory hard-fact carve-out), never a range to parse.
    """
    req_lines = []
    for i, item in enumerate(must_have_keywords, 1):
        aliases = ", ".join(item.get("aliases") or [])
        suffix = f" (also acceptable as: {aliases})" if aliases else ""
        req_lines.append(f"{i}. {item['term']}{suffix}")
    ko_lines = [f"{i}. {req}" for i, req in
                enumerate(knockout_requirements, 1)] or ["(none)"]
    return _PROMPT.format(
        title=job_title or "(unknown)",
        variants=", ".join(title_variants or []) or "(none)",
        posted_top_of_band=(f"${int(posted_top_of_band):,}"
                            if posted_top_of_band else "NONE"),
        requirements="\n".join(req_lines),
        knockouts="\n".join(ko_lines),
    )


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s or "")).strip().lower()


def _align(inputs: list, judged: list, field: str) -> list:
    """Two-pass align `judged` dicts to `inputs` dicts by `field` (the same
    field name — "term" or "requirement" — on both sides).

    Pass 1: exact normalized match, order-preserving, first-available —
    handles reorder. Pass 2: pair remaining unbound inputs with remaining
    unconsumed judged entries positionally — repairs paraphrase without
    ever binding more judged entries than exist. Inputs left over (judge
    returned fewer entries than asked) bind to None — the caller applies
    the no-credit default. Never binds more judged entries than were
    actually returned, so the judge can only grant what it returned.

    Returns a list parallel to `inputs`: each element is the bound judged
    dict, or None if no judged entry was available for it.
    """
    consumed = [False] * len(judged)
    bound = [None] * len(inputs)

    # Pass 1: exact normalized match.
    for i, src in enumerate(inputs):
        target = _norm(src.get(field))
        for j, item in enumerate(judged):
            if not consumed[j] and _norm(item.get(field)) == target:
                bound[i] = item
                consumed[j] = True
                break

    # Pass 2: residual by order.
    unbound_idx = [i for i in range(len(inputs)) if bound[i] is None]
    unconsumed_idx = [j for j in range(len(judged)) if not consumed[j]]
    for i, j in zip(unbound_idx, unconsumed_idx):
        bound[i] = judged[j]
        consumed[j] = True

    return bound


def parse_judge_response(raw: str, must_have_keywords: list,
                         knockouts: list) -> Optional[dict]:
    """Align the judge's JSON to the input requirement lists.

    Two-pass alignment (see `_align`): exact normalized match first
    (handles reorder), then residual-by-order for anything left unbound
    (repairs paraphrase — the judge returned the same count, just renamed
    an entry). Inputs still unbound after both passes — the judge
    genuinely dropped them — get the NO-CREDIT default ("absent" for
    must-haves, "unclear" for knockouts). The matcher never binds more
    judged entries than the judge actually returned, so it can only ever
    grant what was judged, never more. Returns None when no JSON object
    parses.
    """
    text = re.sub(r"```(?:json)?\s*", "", raw or "").strip()
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None

    raw_mh = data.get("must_haves")
    judged_mh = [item for item in (raw_mh if isinstance(raw_mh, list) else [])
                 if isinstance(item, dict)]
    bound_mh = _align(must_have_keywords, judged_mh, "term")

    must_haves = []
    for src, got in zip(must_have_keywords, bound_mh):
        got = got or {}
        verdict = str(got.get("verdict", "")).lower()
        evidence = str(got.get("evidence") or "").strip()
        if verdict not in ("explicit", "evidenced", "absent"):
            verdict = "absent"
        if verdict in ("explicit", "evidenced") and not evidence:
            verdict = "absent"  # no evidence, no credit
        must_haves.append({
            "term": src["term"],
            "aliases": src.get("aliases") or [],
            "verdict": verdict,
            "evidence": evidence if verdict != "absent" else "",
        })

    raw_ko = data.get("knockouts")
    judged_ko = [item for item in (raw_ko if isinstance(raw_ko, list) else [])
                 if isinstance(item, dict)]
    bound_ko = _align(knockouts, judged_ko, "requirement")

    def _num(v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    out_kos = []
    for src, got in zip(knockouts, bound_ko):
        got = got or {}
        req = src.get("requirement", "")
        verdict = str(got.get("verdict", "")).lower()
        if verdict not in ("met", "unclear", "failed"):
            verdict = "unclear"
        out_kos.append({
            "requirement": req,
            "verdict": verdict,
            "reason": str(got.get("reason") or "").strip(),
            "required_years": _num(got.get("required_years")),
            "candidate_years": _num(got.get("candidate_years")),
        })

    claim = str(data.get("title_claim", "")).lower()
    if claim not in ("exact", "close", "none"):
        claim = "none"

    return {"must_haves": must_haves, "knockouts": out_kos,
            "title_claim": claim}


def apply_soft_years(knockouts: list) -> list:
    """Downgrade near-miss years knockouts: failed -> unclear at >= 70%.

    Deterministic, code-side (never trusted to the LLM). Input is not
    mutated — callers may hold the pre-softened list.
    """
    out = []
    for k in knockouts or []:
        k = dict(k)
        req, cand = k.get("required_years"), k.get("candidate_years")
        if (k.get("verdict") == "failed"
                and isinstance(req, (int, float)) and req
                and isinstance(cand, (int, float))
                and cand / req >= SOFT_YEARS_THRESHOLD - _EPS):
            pct = round(100 * cand / req)
            k["verdict"] = "unclear"
            k["reason"] = (f"{k.get('reason', '')} "
                           f"[softened: {pct}% of required years]").strip()
        out.append(k)
    return out
