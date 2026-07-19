"""The Filter badge must not misstate what its number means on tailored rows.

2026-07-17: auto-tailor picked a fintech role at exactly the judged ceiling
of 60 — a legitimate `filter_score >= 60` pass — then the post-tailor recompute
overwrote the displayed filter_score with the tailored doc's literal
coverage (53). The badge kept claiming "tailorable ceiling — what a
truthful tailored resume could score", so a sub-60 badge on an
auto-tailored row read as a gate misfire and got reported as a bug.

Pins (same static-file style as test_dashboard_default_sort.py):
- the ceiling copy is no longer applied unconditionally to both sources;
- the tailored branch says what the number actually is and surfaces the
  frozen pre-tailor ceiling (filter_score_master) the selection gates used,
  guarded for legacy tailored rows that predate the master column.
"""
import os
import re

_INDEX = os.path.join(os.path.dirname(__file__), "..",
                      "dashboard", "static", "index.html")


def _filter_badge_body():
    with open(_INDEX, encoding="utf-8") as f:
        html = f.read()
    start = html.index("function filterBadge(job) {")
    end = html.index("\nfunction sourceBadge(", start)
    return html[start:end]


def test_tailored_badge_does_not_claim_ceiling_semantics():
    body = _filter_badge_body()
    # The old single template stamped the ceiling explanation onto tailored
    # rows too ("Filter Match (tailored) — tailorable ceiling — ..."), which
    # is factually wrong there: post-tailor the number is literal coverage.
    assert "Filter Match${src} — tailorable ceiling" not in body, (
        "filterBadge must not describe the post-tailor literal score as a "
        "'tailorable ceiling'; split the tooltip by filter_source")
    # Future-tense ceiling copy belongs to the untailored branch only.
    tailored_claims_could_score = re.search(
        r"\(tailored\)[^'\"`]*could score", body)
    assert not tailored_claims_could_score, (
        "the tailored tooltip must describe achieved literal coverage, not "
        "what a tailored resume 'could score'")


def test_tailored_badge_surfaces_gate_time_ceiling():
    body = _filter_badge_body()
    assert "filter_score_master" in body, (
        "the tailored tooltip must surface the frozen pre-tailor ceiling "
        "(filter_score_master) — the value alerts/auto-tailor actually "
        "gated on — so a post-tailor drop below 60 can't read as a gate "
        "misfire")
    # Legacy tailored rows predate the master column; the ceiling clause
    # must be guarded, not rendered as "ceiling null".
    assert re.search(r"filter_score_master\s*!=\s*null", body), (
        "guard the ceiling clause for tailored rows with a NULL "
        "filter_score_master")


def test_untailored_badge_keeps_ceiling_explanation():
    body = _filter_badge_body()
    assert re.search(
        r"could score against this screen", body), (
        "untailored rows still show the judged ceiling and must keep the "
        "'what a truthful tailored resume could score' explanation")
