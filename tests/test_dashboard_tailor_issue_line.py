"""The amber tailor-issue badge and the tailored-resume panel must agree.

2026-07-18: an auto-tailored job showed the row-header badge (1 edit reverted) while
the expanded TAILORED RESUME panel showed nothing — the panel's issue list
renders only critical ATS issues, and the badge's facts (tailor_issue_json)
surfaced nowhere except a hover tooltip and the View changes modal.

Pins (static-file style, same as test_dashboard_filter_badge.py):
- badge and panel compose wording from ONE shared helper (_issueParts) so
  the two can never drift;
- the shared done-panel shell renders a caller-normalized facts line for
  both the live-completed and the historical panel;
- the live task result forwards issue_facts (app.py's whitelist dropped
  it, leaving the live panel nothing to render).
"""
import importlib
import os

_INDEX = os.path.join(os.path.dirname(__file__), "..",
                      "dashboard", "static", "index.html")


def _html():
    with open(_INDEX, encoding="utf-8") as f:
        return f.read()


def _fn_body(html, name, next_marker):
    start = html.index(name)
    end = html.index(next_marker, start)
    return html[start:end]


def test_badge_and_line_share_parts_helper():
    html = _html()
    assert "function _issueParts(" in html, (
        "wording must live in one helper (_issueParts) shared by the badge "
        "and the panel line, not be duplicated in each")
    badge = _fn_body(html, "function _issueBadge(", "\nfunction ")
    assert "_issueParts(" in badge, (
        "_issueBadge must compose its tooltip from _issueParts")
    line = _fn_body(html, "function _issueFactsLine(", "\nfunction ")
    assert "_issueParts(" in line, (
        "_issueFactsLine must compose from the same _issueParts helper")


def test_done_body_renders_caller_normalized_facts_line():
    html = _html()
    body = _fn_body(html, "function _tailorDoneBody(", "\n// Facts")
    assert "issueFactsHtml" in body, (
        "_tailorDoneBody must accept and render the pre-rendered issue "
        "facts line so every green done panel can show why the row-header "
        "badge fired")


def test_live_completed_panel_passes_issue_facts():
    html = _html()
    body = _fn_body(html, "function tailorResultPanel(",
                    "\n// Shared body")
    assert "issue_facts" in body, (
        "the live-completed panel must render the facts returned by the "
        "task result — the badge fires from the same facts once the row "
        "refreshes, and the panel must not stay blank")


def test_history_panel_passes_issue_facts():
    html = _html()
    body = _fn_body(html, "function _tailorHistoryPanel(",
                    "\n// Minimal fallback")
    assert "tailor_issue_json" in body, (
        "the historical panel must render the job row's stored facts "
        "(tailor_issue_json) — the exact case in the 2026-07-18 report")


def test_worker_forwards_issue_facts_to_live_result():
    app_mod = importlib.import_module("dashboard.app")
    facts = {"reverted": 1, "modified": 0,
             "missing_terms": [], "layout_unverified": False}
    fake = {
        "google_doc_url": "https://docs.google.com/x",
        "docx_filename": "x.docx",
        "ats_score": 92,
        "keywords_matched": 22,
        "keywords_total": 35,
        "issues": [],
        "warnings": [],
        "company": "Jobgether",
        "title": "Staff PM",
        "total_input_tokens": 1,
        "total_output_tokens": 1,
        "est_cost": 0.01,
        "issue_facts": facts,
    }

    def fake_pipeline(job, db_path, progress=None):
        return fake

    with app_mod._tailor_lock:
        app_mod._tailor_tasks["t-issue-facts"] = {
            "task_id": "t-issue-facts", "job_id": 1, "status": "pending",
            "progress": "Starting...", "steps": [], "result": None,
            "error": None, "error_category": None,
            "created_at": "2026-07-18T00:00:00", "completed_at": None,
        }
    orig = app_mod.run_tailor_pipeline
    app_mod.run_tailor_pipeline = fake_pipeline
    try:
        app_mod._tailor_worker("t-issue-facts", 1, "https://x/1",
                               "Jobgether", "Staff PM")
    finally:
        app_mod.run_tailor_pipeline = orig
    with app_mod._tailor_lock:
        task = app_mod._tailor_tasks.pop("t-issue-facts")
    assert task["status"] == "completed"
    assert task["result"]["issue_facts"] == facts, (
        "app.py's result whitelist must forward issue_facts — without it "
        "the live panel has nothing to render")
