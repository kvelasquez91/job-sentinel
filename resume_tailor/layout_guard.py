"""
PDF-truth layout measurement for the tailor-resume layout guard.

Pure functions only: build a line map from rendered PDF bytes, locate a
paragraph's rendered lines, and detect layout violations against the master.
No Google client, no LLM — orchestration (the repair loop) lives in
tailor_engine.enforce_layout(), which calls into this module.

Spec: the 2026-07-07 tailor-resume layout-guard design (private repo notes).
"""
import io
import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

# A last rendered line of 1..4 words is a dangler (per spec; the old widow fix
# used a 15-char threshold that missed 3-4 word danglers by design).
DANGLER_MAX_WORDS = 4

# Typographic folding: PDF extractors emit ligature glyphs; humans and Google
# Docs emit smart quotes/dashes; the Docs API plaintext may differ from both.
# Fold everything to canonical ASCII so both sides of every compare agree.
_TYPO_TRANS = str.maketrans({
    chr(0x2018): "'", chr(0x2019): "'",   # curly single quotes
    chr(0x201C): '"', chr(0x201D): '"',   # curly double quotes
    chr(0x2013): "-", chr(0x2014): "-",   # en dash, em dash
    chr(0x00A0): " ",                      # non-breaking space
})
_LIGATURES = {
    chr(0xFB00): "ff", chr(0xFB01): "fi", chr(0xFB02): "fl",
    chr(0xFB03): "ffi", chr(0xFB04): "ffl",
}
# Bullet glyphs appear in the PDF text layer but not in Docs-API paragraph text.
_BULLET_CHARS = chr(0x2022) + chr(0x25E6) + chr(0x25AA) + chr(0x2023) + chr(0x2043) + chr(0x25CF)


def normalize(text: str) -> str:
    """Whitespace-collapse and fold typographic variants for stable comparison."""
    for lig, repl in _LIGATURES.items():
        text = text.replace(lig, repl)
    text = text.translate(_TYPO_TRANS)
    tokens = []
    for tok in text.split():
        tok = tok.lstrip(_BULLET_CHARS)
        if tok:
            tokens.append(tok)
    return " ".join(tokens)


@dataclass(frozen=True)
class RenderedLine:
    """One rendered line of text in the PDF, in reading order."""
    text: str
    words: int          # word count of normalize(text)
    top: float          # y-coordinate from the page top (debug/ordering)


@dataclass
class LineMap:
    """Every rendered line of the document, in reading order across pages."""
    lines: list
    page_count: int


@dataclass
class EditedParagraph:
    """One paragraph the tailor changed — INPUT to check_layout().

    Mutable: refresh_from_doc() (tailor_engine) re-reads tailored_text and the
    positional indices from the live document each repair round, keyed by
    `ordinal` (the paragraph's position in the body — stable because every edit
    is a text replacement within a paragraph, never a paragraph add/remove).
    """
    role: str            # "summary" | "title" | "skills:<subcat>" |
                         # "bullet:<company>:<label>" | "rewrite:<company>:<idx>"
    master_text: str     # pre-edit original (revert fallback / master-map lookup)
    tailored_text: str = ""   # live doc text, refreshed every round
    ordinal: int = -1         # index into the doc body's paragraph sequence
    start_index: int = 0      # live positional range (for the positional revert)
    end_index: int = 0
    page_safe_text: str = ""  # last observed text that rendered at the master's
                              # line count (a mere dangler) — the guard's
                              # restore candidate when a repair over-cuts into
                              # a shrank and the text carries credited terms


@dataclass
class LayoutViolation:
    """OUTPUT of check_layout(): an EditedParagraph that failed, plus measurements."""
    role: str
    kind: str            # "grew" | "shrank" | "dangler" | "lost"
    master_lines: int
    tailored_lines: int
    last_line_words: int
    tailored_text: str
    master_text: str
    target_chars: int    # repair hint = len(master_text); the authoritative gate
                         # is re-measuring the re-exported PDF, not this number


def build_line_map(pdf_bytes: bytes) -> LineMap:
    """Extract every rendered text line from the PDF, in reading order.

    Uses pdfplumber's extract_text_lines(), which clusters words into lines by
    baseline; pages are walked in order, and extract_text_lines returns lines
    top-down within a page, so simple concatenation preserves reading order.
    """
    import pdfplumber  # local import: keeps module importable if dep is missing

    lines: list = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        page_count = len(pdf.pages)
        for page in pdf.pages:
            for ln in page.extract_text_lines():
                text = ln.get("text", "").strip()
                if not text:
                    continue
                lines.append(RenderedLine(
                    text=text,
                    words=len(normalize(text).split()),
                    top=float(ln.get("top", 0.0)),
                ))
    return LineMap(lines=lines, page_count=page_count)


def locate_paragraph(line_map: LineMap, text: str) -> Optional[tuple]:
    """Find the contiguous run of rendered lines whose normalized concatenation
    equals normalize(text). Returns inclusive (start_idx, end_idx), or None.

    In a one-column layout a rendered line never mixes two body paragraphs, so
    a paragraph always occupies whole lines; equality of the joined run is the
    correct criterion. O(n) per start with prefix pruning; resumes are ~40 lines.
    """
    target = normalize(text)
    if not target:
        return None
    lines = line_map.lines
    n = len(lines)
    for i in range(n):
        acc = ""
        for j in range(i, n):
            piece = normalize(lines[j].text)
            if not piece:
                break                       # blank rendered line ends any paragraph
            acc = f"{acc} {piece}".strip() if acc else piece
            if acc == target:
                return (i, j)
            if len(acc) >= len(target) or not target.startswith(acc + " "):
                break
    return None


def paragraph_line_count(line_map: LineMap, text: str) -> Optional[int]:
    """Number of rendered lines the paragraph spans, or None if not found."""
    span = locate_paragraph(line_map, text)
    return None if span is None else span[1] - span[0] + 1


def last_line_word_count(line_map: LineMap, text: str) -> Optional[int]:
    """Word count of the paragraph's final rendered line, or None if not found."""
    span = locate_paragraph(line_map, text)
    return None if span is None else line_map.lines[span[1]].words


def check_layout(master_map: LineMap, tailored_map: LineMap, paragraphs: list) -> list:
    """Compare each edited paragraph's rendered layout against the master.

    Violation kinds:
      grew    — spans more rendered lines than in the master
      shrank  — spans fewer (a dropped line shifts everything below it)
      dangler — same line count but the last line has 1..DANGLER_MAX_WORDS words
      lost    — the tailored text can't be located in the tailored map at all

    A paragraph whose master text can't be located is SKIPPED (no reference to
    enforce against); a master that itself ends in a dangler suppresses the
    dangler check (we can't beat the source). Both cases are logged.
    """
    violations = []
    for p in paragraphs:
        m_span = locate_paragraph(master_map, p.master_text)
        if m_span is None:
            logger.warning("layout_guard: %s not found in MASTER map — cannot enforce, skipping", p.role)
            continue
        master_lines = m_span[1] - m_span[0] + 1

        t_span = locate_paragraph(tailored_map, p.tailored_text) if p.tailored_text else None
        if t_span is None:
            violations.append(LayoutViolation(
                role=p.role, kind="lost", master_lines=master_lines,
                tailored_lines=0, last_line_words=0,
                tailored_text=p.tailored_text, master_text=p.master_text,
                target_chars=len(p.master_text)))
            continue

        tailored_lines = t_span[1] - t_span[0] + 1
        last_words = tailored_map.lines[t_span[1]].words

        kind = None
        if tailored_lines > master_lines:
            kind = "grew"
        elif tailored_lines < master_lines:
            kind = "shrank"
        elif tailored_lines >= 2 and last_words <= DANGLER_MAX_WORDS:
            master_last_words = master_map.lines[m_span[1]].words
            if master_last_words <= DANGLER_MAX_WORDS:
                logger.info("layout_guard: %s master itself ends in a %d-word line — dangler check skipped",
                            p.role, master_last_words)
            else:
                kind = "dangler"

        if kind:
            violations.append(LayoutViolation(
                role=p.role, kind=kind, master_lines=master_lines,
                tailored_lines=tailored_lines, last_line_words=last_words,
                tailored_text=p.tailored_text, master_text=p.master_text,
                target_chars=len(p.master_text)))
    return violations
