"""
ATS compliance validation.

Operates on the Google Docs API JSON structure (doc dict from documents().get()).
Checks the resume copy AFTER edits have been applied.

All checks return ATSIssue instances (or None if the check passes).
check_all() runs every check and returns an ATSCheckResult.

NOTE on section header detection:
  The original master resume uses NORMAL_TEXT style throughout — section headers
  are bold ALL-CAPS paragraphs followed by horizontal rules. They are NOT styled
  as HEADING_1/HEADING_2. All detection must use the bold+ALL-CAPS heuristic, not
  namedStyleType. See plan section 1, "Google Docs API Representation".
"""
import io
import logging
import re
from dataclasses import dataclass, field
from typing import Optional

from filter_match import keyword_matches  # shared matcher; re-exported for pipeline/tests
from profile_policy import TAILOR_EXTRA_ATS_HEADERS

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ATSIssue:
    rule: str           # e.g. "SINGLE_COLUMN"
    severity: str       # "critical" | "warning"
    description: str
    location: str       # Human-readable location in the doc
    auto_fixable: bool


@dataclass
class ATSCheckResult:
    passed: bool
    score: int              # 0-100
    issues: list            # List[ATSIssue]
    warnings: list          # List[str] — non-blocking suggestions


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_BASE_STANDARD_HEADERS = {
    "professional summary", "summary", "objective",
    "skills", "technical skills", "core competencies",
    "work experience", "experience", "professional experience",
    "education", "certifications", "projects",
}

# The owner's master resume may use additional section headers beyond the
# generic set above (policy.tailor.extra_ats_headers in config.yaml).
STANDARD_HEADERS = _BASE_STANDARD_HEADERS | {h.lower() for h in TAILOR_EXTRA_ATS_HEADERS}

NON_STANDARD_HEADERS = {
    "my journey", "toolkit", "what i bring", "about me",
    "superpowers", "tech stack", "career highlights",
}

IDEAL_SECTION_ORDER = [
    "summary", "skills", "experience", "education", "certifications"
]

# Expected date format used in the master resume: "Jan 2024" or "Jan 2024 – Present"
_DATE_PATTERNS = {
    "mon_yyyy": re.compile(
        r"\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{4}\b"
    ),
    "mm_slash_yyyy": re.compile(r"\b\d{1,2}/\d{4}\b"),
    # ISO-ish year-month; the month is bounded to 01-12 so a year *range* like
    # "2023-24" (24 is not a month) is not mistaken for a second date format.
    "yyyy_dash_mm": re.compile(r"\b\d{4}-(?:0[1-9]|1[0-2])\b"),
    "full_month_year": re.compile(
        r"\b(January|February|March|April|May|June|July|August|"
        r"September|October|November|December)\s+\d{4}\b"
    ),
}

_EMOJI_RE = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F680-\U0001F6FF"  # transport & map
    "\U0001F900-\U0001F9FF"  # supplemental symbols & pictographs (handshake, brain)
    "\U0001FA70-\U0001FAFF"  # symbols & pictographs extended-A
    "\U00002600-\U000027BF"  # misc symbols
    "\U00002B00-\U00002BFF"  # misc symbols & arrows (stars)
    "\U0000FE00-\U0000FE0F"  # variation selectors
    "]+",
    flags=re.UNICODE,
)

KEYWORD_MIN = 25
KEYWORD_MAX = 35


# ---------------------------------------------------------------------------
# Text extraction helpers
# ---------------------------------------------------------------------------

def _extract_text(paragraph: dict) -> str:
    """Return all text from a paragraph element (concatenated text runs)."""
    return "".join(
        el.get("textRun", {}).get("content", "")
        for el in paragraph.get("elements", [])
    ).strip()


def _extract_all_text_from_content(content: list) -> str:
    """Return all text from a list of body content elements."""
    parts = []
    for element in content:
        paragraph = element.get("paragraph")
        if paragraph:
            parts.append(_extract_text(paragraph))
    return " ".join(parts)


def _is_section_header(paragraph: dict, text: str) -> bool:
    """
    Detect a section header: bold ALL-CAPS NORMAL_TEXT paragraph.
    Does NOT use namedStyleType (see module docstring for why).
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


def _looks_like_header(text: str) -> bool:
    """
    Lenient section-header heuristic: a non-empty ALL-CAPS line.

    Unlike _is_section_header this does NOT require explicit bold on the text run,
    because the original master resume's headers frequently inherit bold from the
    paragraph's named style rather than setting it on textRun.textStyle (see check_section_headers,
    which relies on the same allowance). Used for locating section boundaries when
    we already know the header text we're looking for.
    """
    stripped = text.strip()
    return bool(stripped) and stripped == stripped.upper() and any(c.isalpha() for c in stripped)


def _get_section_order(doc: dict) -> list[str]:
    """Return the list of section header texts found in document order."""
    headers = []
    for element in doc.get("body", {}).get("content", []):
        paragraph = element.get("paragraph")
        if not paragraph:
            continue
        text = _extract_text(paragraph)
        if _is_section_header(paragraph, text):
            headers.append(text.strip().lower())
    return headers


def _extract_full_section_text(doc: dict, section_keyword: str) -> str:
    """
    Return ALL body text within a section — every paragraph from the header whose
    text contains section_keyword up to (but excluding) the next section header.
    The header line itself is not included.

    Header detection is lenient (_looks_like_header, no explicit-bold requirement)
    so an inherited-bold header is still recognized, and the whole section is
    scanned rather than only its first paragraph.
    """
    body_content = doc.get("body", {}).get("content", [])
    in_target_section = False
    parts = []

    for element in body_content:
        paragraph = element.get("paragraph")
        if not paragraph:
            continue
        text = _extract_text(paragraph)

        if in_target_section:
            if _looks_like_header(text) and section_keyword.lower() not in text.lower():
                break  # Reached next section
            parts.append(text)
            continue

        if _looks_like_header(text) and section_keyword.lower() in text.lower():
            in_target_section = True

    return " ".join(parts)


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

def check_single_column(doc: dict) -> Optional[ATSIssue]:
    """Verify no multi-column sections exist (ATS parsers choke on columns)."""
    for element in doc.get("body", {}).get("content", []):
        if "sectionBreak" in element:
            cols = (
                element["sectionBreak"]
                .get("sectionStyle", {})
                .get("columnProperties", [])
            )
            if len(cols) > 1:
                return ATSIssue(
                    rule="SINGLE_COLUMN",
                    severity="critical",
                    description="Multi-column layout detected — ATS parsers read left column only",
                    location=f"Section break at index {element.get('startIndex', '?')}",
                    auto_fixable=True,
                )
    return None


def check_section_headers(doc: dict) -> list[ATSIssue]:
    """
    Verify section headers use standard ATS-friendly names.
    Detects via bold ALL-CAPS heuristic (not namedStyleType).

    Skips the contact header block (name, title, contact info) at the top of the
    document. These are the paragraphs that appear before the first recognized
    standard section header — they are not section headers and must not be flagged.
    """
    body_content = doc.get("body", {}).get("content", [])

    # Find the element index of the first recognized standard section header.
    # Everything before it is the contact block and is exempt from validation.
    # Use a lenient check here (all-caps + known header text, no bold requirement)
    # because the original master resume's section headers may have bold inherited
    # from paragraph style rather than explicitly set in textRun.textStyle, causing _is_section_header
    # to return False and leaving first_standard_idx=None (which skips no elements).
    first_standard_idx: Optional[int] = None
    for i, element in enumerate(body_content):
        paragraph = element.get("paragraph")
        if not paragraph:
            continue
        text = _extract_text(paragraph)
        stripped = text.strip()
        if stripped and stripped == stripped.upper() and stripped.lower() in STANDARD_HEADERS:
            first_standard_idx = i
            logger.debug(
                "check_section_headers: contact block ends before element %d (%r)", i, stripped
            )
            break

    if first_standard_idx is None:
        logger.warning(
            "check_section_headers: no standard header found — contact block will NOT be skipped; "
            "name/title may be flagged as unrecognized headers"
        )

    issues = []
    for i, element in enumerate(body_content):
        # Skip the contact block that precedes the first real section header
        if first_standard_idx is not None and i < first_standard_idx:
            continue
        paragraph = element.get("paragraph")
        if not paragraph:
            continue
        text = _extract_text(paragraph)
        if not _is_section_header(paragraph, text):
            continue
        header_lower = text.lower().strip()
        if header_lower in NON_STANDARD_HEADERS:
            issues.append(
                ATSIssue(
                    rule="STANDARD_HEADERS",
                    severity="critical",
                    description=f"Non-standard section header: '{text}' — rename to an ATS-recognized label",
                    location=f"Index {element.get('startIndex', '?')}",
                    auto_fixable=True,
                )
            )
        elif header_lower and header_lower not in STANDARD_HEADERS:
            # Not in the known-bad list but also not in known-good — warn
            issues.append(
                ATSIssue(
                    rule="STANDARD_HEADERS",
                    severity="warning",
                    description=f"Unrecognized section header: '{text}' — verify ATS compatibility",
                    location=f"Index {element.get('startIndex', '?')}",
                    auto_fixable=False,
                )
            )
    return issues


def check_headers_footers(doc: dict) -> list[ATSIssue]:
    """Ensure no important content is in document headers or footers (ATS ignores them)."""
    issues = []

    for header in doc.get("headers", {}).values():
        text = _extract_all_text_from_content(header.get("content", []))
        if text.strip():
            issues.append(
                ATSIssue(
                    rule="NO_HEADER_FOOTER",
                    severity="critical",
                    description=f"Content in document header: '{text[:60]}…' — ATS parsers ignore headers",
                    location="Document header",
                    auto_fixable=True,
                )
            )

    for footer in doc.get("footers", {}).values():
        text = _extract_all_text_from_content(footer.get("content", []))
        if text.strip():
            issues.append(
                ATSIssue(
                    rule="NO_HEADER_FOOTER",
                    severity="critical",
                    description=f"Content in document footer: '{text[:60]}…' — ATS parsers ignore footers",
                    location="Document footer",
                    auto_fixable=True,
                )
            )

    return issues


def check_date_consistency(doc_text: str) -> Optional[ATSIssue]:
    """
    Verify all dates use the same format.
    The master resume uses "Mon YYYY" (e.g., "Jan 2024") — flag any deviation.
    """
    # Collect every match as a (start, end, format) span. A single date like
    # "May 2024" matches both the abbreviated and full-month patterns (May is the
    # one month whose abbreviation equals its full name); counting it as two
    # formats is a false positive. Dedupe by span so each stretch of text votes
    # for exactly one format before deciding whether formats are truly mixed.
    spans = []
    for fmt_name, pattern in _DATE_PATTERNS.items():
        for m in pattern.finditer(doc_text):
            spans.append((m.start(), m.end(), fmt_name))
    # Longest match first at each position so the most specific pattern wins the span.
    spans.sort(key=lambda s: (s[0], -(s[1] - s[0])))

    found_formats: dict[str, int] = {}
    claimed: list[tuple[int, int]] = []
    for start, end, fmt_name in spans:
        if any(start < c_end and c_start < end for c_start, c_end in claimed):
            continue  # overlaps a span already counted — same date text
        claimed.append((start, end))
        found_formats[fmt_name] = found_formats.get(fmt_name, 0) + 1

    if len(found_formats) > 1:
        return ATSIssue(
            rule="DATE_FORMAT",
            severity="critical",
            description=f"Mixed date formats detected: {found_formats} — standardize to 'Mon YYYY'",
            location="Throughout document",
            auto_fixable=True,
        )
    return None


def check_no_icons(doc_text: str) -> Optional[ATSIssue]:
    """Check for emoji or symbol characters that ATS systems can't parse."""
    matches = _EMOJI_RE.findall(doc_text)
    if matches:
        return ATSIssue(
            rule="NO_ICONS",
            severity="critical",
            description=f"Emoji/icon characters found: {matches[:5]} — remove all non-text characters",
            location="Throughout document",
            auto_fixable=True,
        )
    return None


def check_keyword_density(
    doc_text: str, priority_keywords: list
) -> Optional[ATSIssue]:
    """Verify enough priority keywords are present (sweet spot for ATS match rate).

    The target floor is KEYWORD_MIN, but never more than the number of keywords
    the JD actually yielded — demanding 25 when only 8 exist is a false alarm.
    """
    matched = keyword_matches(doc_text, priority_keywords)
    count = len(matched)
    missing = [kw for kw in priority_keywords if kw not in matched]
    min_required = min(KEYWORD_MIN, len(priority_keywords))

    if count < min_required:
        return ATSIssue(
            rule="KEYWORD_DENSITY",
            severity="warning",
            description=(
                f"Only {count} of {min_required} minimum keywords found. "
                f"Top missing: {missing[:8]}"
            ),
            location="Overall document",
            auto_fixable=False,
        )
    if count > KEYWORD_MAX:
        return ATSIssue(
            rule="KEYWORD_DENSITY",
            severity="warning",
            description=(
                f"{count} keywords found (max {KEYWORD_MAX}). "
                "Risk of keyword-stuffing detection by ATS."
            ),
            location="Overall document",
            auto_fixable=False,
        )
    return None


def check_job_title_in_summary(doc: dict, exact_title: str) -> Optional[ATSIssue]:
    """Verify the exact job title from the posting appears in the Professional Summary."""
    if not exact_title:
        return None
    summary_text = _extract_full_section_text(doc, "summary")
    if not summary_text:
        return ATSIssue(
            rule="JOB_TITLE_MATCH",
            severity="critical",
            description="Professional Summary section not found",
            location="Document body",
            auto_fixable=False,
        )
    if exact_title.lower() not in summary_text.lower():
        return ATSIssue(
            rule="JOB_TITLE_MATCH",
            severity="critical",
            description=(
                f"Exact job title '{exact_title}' not found in Professional Summary. "
                "ATS title-match check will fail."
            ),
            location="Professional Summary",
            auto_fixable=True,
        )
    return None


def check_section_order(doc: dict) -> Optional[ATSIssue]:
    """Warn if sections deviate significantly from ATS-optimal ordering."""
    found = _get_section_order(doc)

    # Map found headers to canonical names for comparison
    def _canonicalize(h: str) -> str:
        if "summary" in h or "objective" in h:
            return "summary"
        if "skill" in h or "competenc" in h:
            return "skills"
        if "experience" in h:
            return "experience"
        if "education" in h:
            return "education"
        if "certif" in h:
            return "certifications"
        return h

    found_canonical = [_canonicalize(h) for h in found]
    # Check that experience comes before education (most important ATS rule)
    try:
        exp_idx = found_canonical.index("experience")
        edu_idx = found_canonical.index("education")
        if edu_idx < exp_idx:
            return ATSIssue(
                rule="SECTION_ORDER",
                severity="warning",
                description="Education appears before Experience — most ATS systems expect Experience first",
                location="Document structure",
                auto_fixable=False,
            )
    except ValueError:
        pass  # One of the sections not found — skip

    return None


def check_inline_tables(doc: dict) -> Optional[ATSIssue]:
    """
    Warn if data tables are used for layout (many ATS parsers skip table content).
    Single-column or single-row tables are layout artifacts (e.g., indentation, text
    boxes) and are NOT flagged — only multi-row × multi-column tables are flagged,
    as those are true data tables that ATS parsers struggle with.
    """
    for element in doc.get("body", {}).get("content", []):
        if "table" not in element:
            continue
        table = element["table"]
        rows = table.get("rows", 0)
        table_rows = table.get("tableRows", [])
        cols = len(table_rows[0].get("tableCells", [])) if table_rows else 0
        # Skip single-column or single-row tables — these are layout/alignment artifacts,
        # not data tables. Tab characters in resumes (company left, date right) can also
        # produce single-row table representations; do not flag these.
        if rows <= 1 or cols <= 1:
            continue
        return ATSIssue(
            rule="NO_TABLES",
            severity="warning",
            description=(
                f"Data table detected ({rows}×{cols}) — many ATS parsers skip table "
                "content entirely. Use plain text with tabs or spaces instead."
            ),
            location=f"Index {element.get('startIndex', '?')}",
            auto_fixable=False,
        )
    return None


def check_contact_info_present(doc_text: str) -> list[ATSIssue]:
    """Verify basic contact fields are present (email, phone)."""
    issues = []
    if not re.search(r"[\w._%+-]+@[\w.-]+\.\w{2,}", doc_text):
        issues.append(
            ATSIssue(
                rule="CONTACT_INFO",
                severity="warning",
                description="No email address found in resume",
                location="Header section",
                auto_fixable=False,
            )
        )
    if not re.search(r"\(?\d{3}\)?[\s.\-]?\d{3}[\s.\-]?\d{4}", doc_text):
        issues.append(
            ATSIssue(
                rule="CONTACT_INFO",
                severity="warning",
                description="No phone number found in resume",
                location="Header section",
                auto_fixable=False,
            )
        )
    return issues


def check_word_count(
    doc_text: str,
    master_word_count: Optional[int] = None,
) -> Optional[ATSIssue]:
    """
    Warn if word count exceeds a one-page estimate.

    Two checks:
    - Absolute: flag if over 525 words (dense 1-page upper bound).
    - Relative: if master_word_count is provided, flag if tailored version is
      more than 10% larger than the master (LLM added content instead of substituting).
    """
    words = len(doc_text.split())

    if master_word_count and words > master_word_count * 1.10:
        overage = words - master_word_count
        pct = round((overage / master_word_count) * 100)
        return ATSIssue(
            rule="ONE_PAGE",
            severity="warning",
            description=(
                f"Tailored resume is {pct}% longer than master ({words} vs "
                f"{master_word_count} words, +{overage}). LLM may have added "
                "content instead of substituting — review for overflow."
            ),
            location="Overall document",
            auto_fixable=False,
        )

    if words > 525:
        return ATSIssue(
            rule="ONE_PAGE",
            severity="warning",
            description=(
                f"Resume may exceed one page ({words} words detected; "
                "target ≤ 475-525 for a dense single page). Review for trimming."
            ),
            location="Overall document",
            auto_fixable=False,
        )
    return None


def check_page_count(pdf_bytes: bytes) -> Optional[ATSIssue]:
    """
    Count the number of pages in the rendered PDF and flag if > 1.
    Uses pypdf for reliable page counting.
    """
    try:
        import pypdf
        reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
        page_count = len(reader.pages)
    except Exception as exc:
        logger.warning("PDF page count check failed: %s", exc)
        return None

    if page_count > 1:
        logger.warning("Tailored resume renders as %d pages (must be 1)", page_count)
        return ATSIssue(
            rule="ONE_PAGE_RENDER",
            severity="critical",
            description=(
                f"Tailored resume renders as {page_count} pages (must be 1 page). "
                "Bullet rewrites may have added characters causing line wraps — shorten bullets."
            ),
            location="Overall document",
            auto_fixable=False,
        )
    return None


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def check_all(
    doc: dict,
    priority_keywords: list,
    exact_title: str = "",
    master_word_count: Optional[int] = None,
    pdf_bytes: Optional[bytes] = None,
) -> ATSCheckResult:
    """
    Run all ATS checks against the document.

    Args:
        doc:               Google Docs API document JSON (from documents().get()).
        priority_keywords: Ranked list of JD keywords (from JDAnalysis.priority_keywords).
        exact_title:       Exact job title from the posting (for JOB_TITLE_MATCH check).
        master_word_count: Word count of the original master resume (for overflow detection).
        pdf_bytes:         PDF export of the tailored doc; if provided, page count is verified
                           as the authoritative one-page check.

    Returns:
        ATSCheckResult with score (0-100), issues, and warnings.
    """
    issues: list[ATSIssue] = []
    warnings: list[str] = []

    # Full document plain text (for text-based checks)
    doc_text = _extract_all_text_from_content(
        doc.get("body", {}).get("content", [])
    )

    # Run all checks
    _add(issues, check_single_column(doc))
    issues.extend(check_section_headers(doc))
    issues.extend(check_headers_footers(doc))
    _add(issues, check_date_consistency(doc_text))
    _add(issues, check_no_icons(doc_text))
    _add(issues, check_keyword_density(doc_text, priority_keywords))
    _add(issues, check_job_title_in_summary(doc, exact_title))
    _add(issues, check_section_order(doc))
    _add(issues, check_inline_tables(doc))
    issues.extend(check_contact_info_present(doc_text))
    _add(issues, check_word_count(doc_text, master_word_count=master_word_count))

    # PDF page count — authoritative one-page check (supersedes word-count estimate)
    if pdf_bytes:
        _add(issues, check_page_count(pdf_bytes))

    # Compute score: start at 100, deduct by severity only. Deductions do NOT
    # depend on issue.auto_fixable — no auto-fixer is wired, so discounting an
    # issue for being "auto-fixable" would credit a repair that never happens.
    deductions = {"critical": 20, "warning": 8}
    score = 100
    for issue in issues:
        score -= deductions.get(issue.severity, 8)
    score = max(0, score)

    # Separate warnings from issues for easier frontend display
    warning_issues = [i for i in issues if i.severity == "warning"]
    critical_issues = [i for i in issues if i.severity == "critical"]
    for w in warning_issues:
        warnings.append(w.description)

    passed = len(critical_issues) == 0

    return ATSCheckResult(
        passed=passed,
        score=score,
        issues=issues,
        warnings=warnings,
    )


def _add(issues: list, issue: Optional[ATSIssue]) -> None:
    """Append issue to list if not None (convenience helper)."""
    if issue is not None:
        issues.append(issue)
