"""Pure unit tests for resume_tailor.ats_checker — no network, no Google, no LLM.

These lock down the ATS compliance linter against four false-positive bugs
(date formats, job-title detection, keyword matching, unearned score credit) and
guard the genuine-failure paths so the fixes don't blind the checker.
"""
import pytest

import resume_tailor.ats_checker as ats
from policy_fixtures import patch_ats_headers


# ---------------------------------------------------------------------------
# Google Docs API fixture helpers
# ---------------------------------------------------------------------------

def para(text, bold=False):
    """Build a Docs-API paragraph body element (one text run)."""
    style = {"bold": True} if bold else {}
    return {"paragraph": {"elements": [{"textRun": {"content": text + "\n", "textStyle": style}}]}}


def make_doc(elements):
    return {"body": {"content": elements}}


# 25 distinct skill tokens that appear verbatim (whole words) in the skills line.
SKILL_TOKENS = [
    "Python", "SQL", "Airflow", "dbt", "Spark", "AWS", "Terraform", "Docker",
    "Kafka", "Snowflake", "Kubernetes", "Redshift", "Tableau", "Looker", "Scala",
    "Java", "Go", "Bash", "Git", "Linux", "Pandas", "NumPy", "Flink", "Hadoop",
    "Presto",
]


def clean_doc(bold_headers=True):
    """A well-formed, ATS-clean resume that should pass every check."""
    b = bold_headers
    return make_doc([
        para("JANE DOE", bold=True),
        para("alex@example.com | (555) 123-4567 | Austin, TX"),
        para("PROFESSIONAL SUMMARY", bold=b),
        para("Senior Data Engineer with 8 years building resilient data platforms."),
        para("CORE COMPETENCIES & TECHNICAL SKILLS", bold=b),
        para(", ".join(SKILL_TOKENS)),
        para("PROFESSIONAL EXPERIENCE", bold=b),
        para("Acme Corp - Senior Data Engineer   May 2020 - Present"),
        para("Built ELT pipelines processing terabytes daily."),
        para("Initech - Data Engineer   Jan 2017 - Apr 2020"),
        para("EDUCATION", bold=b),
        para("B.S. Computer Science, UT Austin   May 2016"),
    ])


# ---------------------------------------------------------------------------
# check_all — integration: a clean doc must score a perfect 100
# ---------------------------------------------------------------------------

def test_clean_doc_scores_100_and_passes(monkeypatch):
    patch_ats_headers(monkeypatch)
    result = ats.check_all(
        clean_doc(), SKILL_TOKENS, exact_title="Senior Data Engineer")
    assert result.passed is True
    assert result.score == 100, [i.description for i in result.issues]


# ---------------------------------------------------------------------------
# Bug 1 — check_date_consistency false positives
# ---------------------------------------------------------------------------

def test_may_date_not_flagged_as_mixed():
    # "May" is the one month whose abbreviation == full name, so it matched both
    # the abbreviated and full-month regexes and looked like two formats.
    assert ats.check_date_consistency("Jan 2021 - May 2020 - Mar 2019") is None


def test_year_range_not_flagged_as_mixed():
    # A fiscal/academic year range ("2023-24") is not an ISO year-month date.
    assert ats.check_date_consistency("Jan 2020 - Present, led FY 2023-24 planning") is None


def test_genuine_mixed_dates_still_flagged():
    # Real inconsistency: "Mon YYYY" alongside "MM/YYYY" must still fail.
    issue = ats.check_date_consistency("Jan 2020 - Present and 03/2024 - Present")
    assert issue is not None
    assert issue.rule == "DATE_FORMAT"


def test_full_month_vs_abbrev_still_flagged():
    # "Jan 2020" (abbrev) mixed with "January 2021" (full) is a real inconsistency.
    issue = ats.check_date_consistency("Jan 2020 and January 2021")
    assert issue is not None
    assert issue.rule == "DATE_FORMAT"


# ---------------------------------------------------------------------------
# Bug 2 — check_job_title_in_summary detection
# ---------------------------------------------------------------------------

def test_title_found_when_summary_header_bold_is_inherited():
    # Header bold inherited from paragraph style (not set on the text run) must
    # still be recognized as the Professional Summary section.
    doc = make_doc([
        para("JANE DOE", bold=True),
        para("PROFESSIONAL SUMMARY", bold=False),
        para("Senior Data Engineer with a decade of platform experience."),
        para("EXPERIENCE", bold=False),
    ])
    assert ats.check_job_title_in_summary(doc, "Senior Data Engineer") is None


def test_title_found_in_later_summary_paragraph():
    # The title may live in the second summary paragraph, not the first line.
    doc = make_doc([
        para("PROFESSIONAL SUMMARY", bold=True),
        para("Results-driven builder of large-scale data platforms."),
        para("Seasoned Senior Data Engineer focused on reliability."),
        para("EXPERIENCE", bold=True),
    ])
    assert ats.check_job_title_in_summary(doc, "Senior Data Engineer") is None


def test_title_missing_from_summary_still_flagged():
    doc = make_doc([
        para("PROFESSIONAL SUMMARY", bold=True),
        para("Generalist engineer who ships."),
        para("EXPERIENCE", bold=True),
    ])
    issue = ats.check_job_title_in_summary(doc, "Senior Data Engineer")
    assert issue is not None
    assert issue.rule == "JOB_TITLE_MATCH"


def test_missing_summary_section_flagged():
    doc = make_doc([
        para("EXPERIENCE", bold=True),
        para("Did things at places."),
    ])
    issue = ats.check_job_title_in_summary(doc, "Senior Data Engineer")
    assert issue is not None
    assert issue.rule == "JOB_TITLE_MATCH"


# ---------------------------------------------------------------------------
# Bug 3 — keyword matching quality and the scaled floor
# ---------------------------------------------------------------------------

def test_keyword_matches_are_whole_word():
    # "AI" must not match inside "maintain"; "Java" must not match "JavaScript".
    matched = ats.keyword_matches("I maintain JavaScript build systems", ["AI", "Java"])
    assert matched == []


def test_single_letter_and_short_keywords_match_as_tokens():
    matched = ats.keyword_matches("Skills: R, Go, and C++ tooling", ["R", "Go", "C++"])
    assert set(matched) == {"R", "Go", "C++"}


def test_keyword_floor_scales_to_short_lists():
    # 4 keywords, all present: the check must not demand a fixed 25.
    assert ats.check_keyword_density("alpha beta gamma delta", ["alpha", "beta", "gamma", "delta"]) is None


def test_keyword_density_warns_below_floor_for_large_lists():
    keywords = [f"kw{i}" for i in range(30)]
    text = " ".join(f"kw{i}" for i in range(10))  # only 10 of 30 present
    issue = ats.check_keyword_density(text, keywords)
    assert issue is not None
    assert issue.rule == "KEYWORD_DENSITY"


def test_empty_keyword_list_no_warning():
    assert ats.check_keyword_density("anything at all", []) is None


# ---------------------------------------------------------------------------
# Bug 4 — dead auto-fix code removed; scoring no longer credits fixability
# ---------------------------------------------------------------------------

def test_build_auto_fix_requests_is_removed():
    assert not hasattr(ats, "build_auto_fix_requests")


def test_single_critical_deducts_full_twenty(monkeypatch):
    # One critical (content in a document header) — the score must drop by the
    # full manual-critical weight (20), not the old discounted auto-fix weight.
    patch_ats_headers(monkeypatch)
    doc = clean_doc()
    doc["headers"] = {"h1": {"content": [para("Jane Doe - Resume")]}}
    result = ats.check_all(doc, SKILL_TOKENS, exact_title="Senior Data Engineer")
    assert result.passed is False
    assert result.score == 80, [i.description for i in result.issues]


# ---------------------------------------------------------------------------
# check_no_icons — emoji / symbol detection
# ---------------------------------------------------------------------------

# Every one of these characters must be caught as a non-text icon. The three
# "newer" emoji (handshake, brain, star) live in Unicode blocks the original
# _EMOJI_RE character class did not cover; the rocket and check mark guard the
# ranges that already worked so a future edit can't silently drop them.
@pytest.mark.parametrize("icon", [
    "🤝",  # U+1F91D handshake  — Supplemental Symbols & Pictographs
    "🧠",  # U+1F9E0 brain      — Supplemental Symbols & Pictographs
    "⭐",  # U+2B50  star        — Miscellaneous Symbols & Arrows
    "🚀",  # U+1F680 rocket      — already covered (regression guard)
    "✔",   # U+2714  check mark  — already covered (regression guard)
])
def test_check_no_icons_flags_emoji(icon):
    issue = ats.check_no_icons(f"Led the team {icon} to deliver results")
    assert issue is not None
    assert issue.rule == "NO_ICONS"


def test_check_no_icons_passes_plain_text():
    assert ats.check_no_icons("Senior Data Engineer with 8 years of experience") is None
