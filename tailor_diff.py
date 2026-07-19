"""Ground-truth diff + landed-edit verdicts for tailored resumes.

Pure stdlib — no resume_tailor, Google, or LLM imports. Engine references
(the skills label map, the master title line) are INJECTED by callers so
this module can never drift from what the engine actually applied.
Spec: the 2026-07-15 tailor-diff verdicts design (private repo notes).
"""
import difflib
import unicodedata

# Explicit folds NFKC does not cover: Google Docs smart quotes/dashes and
# vertical-tab soft line breaks. NFKC itself folds NBSP/fullwidth/ligatures.
_FOLD_TABLE = str.maketrans({
    '‘': "'", '’': "'",
    '“': '"', '”': '"',
    '–': "-", '—': "-",
    '': "\n",
})

# Shared fallback guard for word-level diffing (readability + bounded cost).
_MIN_WORD_RATIO = 0.4
_MAX_REGION_TOKENS = 300
_CONTEXT_LINES = 2


def _canon(text: str) -> str:
    """Line-preserving canonical form: NFKC, fold table, per-line
    whitespace collapse. Used for alignment AND for display (identical to
    the source modulo quote glyphs and collapsed runs)."""
    text = unicodedata.normalize("NFKC", text or "").translate(_FOLD_TABLE)
    return "\n".join(" ".join(line.split()) for line in text.split("\n"))


def _flat(text: str) -> str:
    """Whole-text canonical form for substring matching (newlines fold too)."""
    return " ".join(_canon(text).split())


def _word_ops(before: str, after: str):
    """[[op, text], …] word-token ops for one changed region, or None to
    signal the whole-block fallback (oversized region or low similarity)."""
    a, b = before.split(), after.split()
    if len(a) > _MAX_REGION_TOKENS or len(b) > _MAX_REGION_TOKENS:
        return None
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    if sm.ratio() < _MIN_WORD_RATIO:
        return None
    ops = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            ops.append(["eq", " ".join(a[i1:i2])])
            continue
        if i1 < i2:
            ops.append(["del", " ".join(a[i1:i2])])
        if j1 < j2:
            ops.append(["ins", " ".join(b[j1:j2])])
    return ops


def compute_diff_blocks(master_text: str, final_text: str) -> list:
    """Two-level diff: line alignment first, then ONE word diff per changed
    region (each side's lines joined — unequal line counts never pose a
    pairing question)."""
    a_lines = _canon(master_text).split("\n")
    b_lines = _canon(final_text).split("\n")
    sm = difflib.SequenceMatcher(None, a_lines, b_lines, autojunk=False)
    blocks = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            lines = a_lines[i1:i2]
            if len(lines) > 2 * _CONTEXT_LINES + 1:
                lines = lines[:_CONTEXT_LINES] + ["⋯"] + lines[-_CONTEXT_LINES:]
            blocks.append({"type": "context", "text": "\n".join(lines)})
            continue
        before = " ".join(" ".join(a_lines[i1:i2]).split())
        after = " ".join(" ".join(b_lines[j1:j2]).split())
        ops = _word_ops(before, after) if before and after else None
        if ops is None:
            blocks.append({"type": "change", "before": before, "after": after})
        else:
            blocks.append({"type": "change", "ops": ops})
    return blocks


VERDICT_LANDED = "landed"
VERDICT_REVERTED = "reverted"      # original present, replacement absent
VERDICT_MODIFIED = "modified"      # neither present (guard repair reworded)
VERDICT_NOT_LANDED = "not_landed"  # replacement absent, no original known


def unwrap_edits(edits_dict) -> tuple:
    """(edits, master_text) from either stored payload shape: the current
    flat shape ({**asdict(edits), "master_text": …}) or the legacy wrapped
    one ({"edits": {...}, "master_text": …}) the dashboard JS already
    tolerates via `edits.edits || edits`."""
    if not isinstance(edits_dict, dict):
        return {}, ""
    inner = edits_dict.get("edits")
    if isinstance(inner, dict):
        master = edits_dict.get("master_text") or inner.get("master_text") or ""
        return inner, master
    return edits_dict, edits_dict.get("master_text") or ""


def _experience_before(e: dict):
    """The post-label original of an experience edit — the exact bytes the
    engine's replaceAllText anchored on. Production payloads store `original`
    WITH its bold label; derive the anchor with the same first-colon split
    _apply_experience_edits uses. An explicit `original_after_label` (the
    spec's key name; no engine version ever wrote it) wins if present.
    None when the payload carries no original at all → not_landed fallback."""
    explicit = e.get("original_after_label")
    if explicit:
        return explicit
    full = e.get("original") or ""
    if ":" in full:
        return full.split(":", 1)[1].strip() or None
    return full or None


def compute_edit_verdicts(edits_dict, final_text, skill_labels,
                          master_title_line) -> list:
    """Deterministic per-edit verdicts by canonicalized substring matching
    against the final doc text. skill_labels / master_title_line are the
    engine's own references (injected — see module docstring)."""
    edits, master_text = unwrap_edits(edits_dict)
    hay = _flat(final_text).lower()

    def present(needle):
        n = _flat(needle or "").lower()
        return bool(n) and n in hay

    def verdict_for(after, before):
        if present(after):
            return VERDICT_LANDED
        if before is None:
            return VERDICT_NOT_LANDED
        return VERDICT_REVERTED if present(before) else VERDICT_MODIFIED

    out = []

    title = edits.get("title_line_replacement") or ""
    if title:
        out.append({"section": "title", "label": "Title",
                    "verdict": verdict_for(title, master_title_line or None),
                    "before": master_title_line or None, "after": title})

    summary = edits.get("summary_replacement") or ""
    if summary:
        # The payload carries no original summary and the master summary is
        # not reliably locatable in plain text — two-state verdict only.
        out.append({"section": "summary", "label": "Summary",
                    "verdict": verdict_for(summary, None),
                    "before": None, "after": summary})

    master_lines = _canon(master_text).split("\n") if master_text else []
    skills_reorder = edits.get("skills_reorder")
    if not isinstance(skills_reorder, dict):
        skills_reorder = {}
    for sub, skills in skills_reorder.items():
        if not skills or sub not in (skill_labels or {}):
            continue  # engine applies only known subcategories — mirror it
        label_txt = skill_labels[sub]
        after_line = f"{label_txt} {', '.join(skills)}"
        prefix = _flat(label_txt).lower()
        original = next((ln for ln in master_lines
                         if ln.lower().startswith(prefix)), None)
        out.append({"section": "skills", "label": sub,
                    "verdict": verdict_for(after_line, original),
                    "before": original, "after": after_line})

    for e in (edits.get("experience_edits") or []):
        if not isinstance(e, dict) or not e.get("replacement_after_label"):
            continue
        before = _experience_before(e)
        out.append({"section": "experience",
                    "label": f"{e.get('company', '')} — {e.get('role', '')}",
                    "verdict": verdict_for(e["replacement_after_label"], before),
                    "before": before, "after": e["replacement_after_label"]})

    for b in (edits.get("rewritten_bullets") or []):
        if not isinstance(b, dict) or not b.get("rewritten"):
            continue
        before = b.get("original") or None
        out.append({"section": "rewrite",
                    "label": f"{b.get('company', '')} — {b.get('role', '')}",
                    "verdict": verdict_for(b["rewritten"], before),
                    "before": before, "after": b["rewritten"]})
    return out


def build_issue_facts(verdicts, missing_terms, layout_unverified):
    """Badge facts, or None when nothing warrants attention. `reverted`
    counts not_landed too (a planned change absent from the final doc,
    whatever the cause). `modified` alone never triggers — a guard repair
    is a successful fix, visible as an amber chip in the panel."""
    reverted = sum(1 for v in verdicts
                   if v["verdict"] in (VERDICT_REVERTED, VERDICT_NOT_LANDED))
    missing = [str(t) for t in (missing_terms or []) if str(t).strip()]
    if not reverted and not missing and not layout_unverified:
        return None
    modified = sum(1 for v in verdicts if v["verdict"] == VERDICT_MODIFIED)
    return {"reverted": reverted, "modified": modified,
            "missing_terms": missing,
            "layout_unverified": bool(layout_unverified)}
