from unittest import mock

import pytest

import resume_tailor.tailor_engine as te
from policy_fixtures import patch_tailor_anchors, patch_tailor_master_labels

# Model split: the cheap analysis steps run on Sonnet; only the quality-critical
# edit-generation steps run on Opus. This keeps a single tailor run from spending
# the Claude subscription window on Opus for work Sonnet handles just as well.
CHEAP_MODEL = "claude-sonnet-5"
EDIT_MODEL = "claude-opus-4-8"


def _fake_result(text: str) -> dict:
    return {
        "text": text,
        "usage": {"input_tokens": 10, "output_tokens": 5},
        "cost_usd": 0.0,
        "is_error": False,
        "session_id": None,
    }


def test_claude_call_returns_text_and_accumulates_usage_cost():
    te.reset_token_usage()
    fake = {"text": "hello", "usage": {"input_tokens": 100, "output_tokens": 20},
            "cost_usd": 0.05, "is_error": False, "session_id": None}
    with mock.patch.object(te, "run_claude", return_value=fake) as m:
        out = te._claude_call("prompt", label="t", model="claude-opus-4-8")
    assert out == "hello"
    usage = te.get_token_usage()
    assert usage["input_tokens"] == 100
    assert usage["output_tokens"] == 20
    assert round(usage["cost_usd"], 2) == 0.05
    m.assert_called_once()


def test_tailor_model_split_sonnet_for_analysis_opus_for_edits():
    """Config pins the cost-control contract: cheap steps on Sonnet, edits on Opus."""
    from resume_tailor import config
    assert config.TAILOR_MODEL == CHEAP_MODEL
    assert config.TAILOR_EDIT_MODEL == EDIT_MODEL


def test_keyword_extraction_routes_to_cheap_sonnet_model():
    """A real analysis step (Step 1) must request the cheap Sonnet model."""
    te.reset_token_usage()
    fake = _fake_result('{"exact_job_title": "Senior PM", "priority_keywords": ["AI"]}')
    with mock.patch.object(te, "run_claude", return_value=fake) as m:
        te.extract_keywords("Some job description text")
    assert m.call_args.kwargs["model"] == CHEAP_MODEL


def test_edit_step_routes_to_opus_model():
    """The real edit-generation callsite (generate_edits) must request Opus."""
    te.reset_token_usage()
    jd_analysis = te.JDAnalysis(exact_job_title="Senior PM")
    gap = te.GapAnalysis()
    with mock.patch.object(te, "run_claude", return_value=_fake_result("not json")) as m:
        result = te.generate_edits(
            master_resume_text="Some resume text",
            jd_analysis=jd_analysis,
            gap=gap,
            company="TestCo",
        )
    # _parse_json_response fails on "not json", so generate_edits returns early
    # after exactly one _claude_call -> one run_claude call.
    m.assert_called_once()
    assert m.call_args.kwargs["model"] == EDIT_MODEL
    assert result == te.TailoringEdits()


def test_generate_edits_uses_extended_timeout():
    """The Opus edit step routinely runs ~100-200s; it must get more than the
    default short-call timeout so it is not killed mid-generation."""
    te.reset_token_usage()
    with mock.patch.object(te, "run_claude", return_value=_fake_result("not json")) as m:
        te.generate_edits(
            master_resume_text="Some resume text",
            jd_analysis=te.JDAnalysis(exact_job_title="Senior PM"),
            gap=te.GapAnalysis(),
            company="TestCo",
        )
    assert m.call_args.kwargs["timeout"] >= 240


def test_claude_call_does_not_retry_timeouts():
    """A CLI timeout is deterministic for a fixed prompt — retrying it 3x just
    burns the subscription window and blocks the global semaphore. It must fail
    fast after a single attempt (transient errors still retry) and surface as
    ClaudeCLITimeout, not a swallowed None."""
    te.reset_token_usage()
    with mock.patch.object(
        te, "run_claude", side_effect=te.ClaudeCLITimeout("claude CLI killed: timed out after 300s")
    ) as m:
        with pytest.raises(te.ClaudeCLITimeout):
            te._claude_call("prompt", label="t", model=EDIT_MODEL, timeout=300.0)
    m.assert_called_once()


def test_claude_call_raises_cli_error_after_retries_exhausted(monkeypatch):
    """A CLI failure that survives all retries must RAISE, not return None:
    the swallowed error let the pipeline complete as a mislabeled success
    (untailored copy with tailored_at set, gate satisfied, no attempt bump)."""
    te.reset_token_usage()
    monkeypatch.setattr(te._claude_call_inner.retry, "sleep", lambda s: None)
    calls = {"n": 0}

    def dead_cli(*a, **k):
        calls["n"] += 1
        raise te.ClaudeCLIError("claude CLI killed: exceeded memory ceiling")

    with mock.patch.object(te, "run_claude", side_effect=dead_cli):
        with pytest.raises(te.ClaudeCLIError):
            te._claude_call("prompt", label="t", model=EDIT_MODEL)
    assert calls["n"] == 3   # retry contract unchanged: 3 attempts, then raise


def test_extract_keywords_propagates_cli_error():
    """Step 1 must not degrade a dead CLI into an empty JDAnalysis — callers
    (main.run_auto_tailor, dashboard worker) key their failure taxonomy on
    ClaudeCLIError reaching them."""
    te.reset_token_usage()
    with mock.patch.object(
        te, "run_claude", side_effect=te.ClaudeCLITimeout("timed out after 120s")
    ):
        with pytest.raises(te.ClaudeCLIError):
            te.extract_keywords("Some job description text")


def test_generate_edits_propagates_cli_error():
    """Step 3 must not degrade a dead CLI into empty TailoringEdits — that
    produced an untailored Doc copy recorded as a successful tailor."""
    te.reset_token_usage()
    with mock.patch.object(
        te, "run_claude", side_effect=te.ClaudeCLITimeout("timed out after 300s")
    ):
        with pytest.raises(te.ClaudeCLIError):
            te.generate_edits(
                master_resume_text="Some resume text",
                jd_analysis=te.JDAnalysis(exact_job_title="Senior PM"),
                gap=te.GapAnalysis(),
                company="TestCo",
            )


_GAP_JSON = ('{"already_matched": ["AI"], "missing_but_relevant": ["LLM"], '
             '"keyword_count": 12}')


def test_gap_analysis_uses_extended_timeout():
    """gap_analysis's healthy latency tail crosses the 120s default: measured
    successes at 108.5s, 119.7s and 162s for its ~10-11k-char prompts. The
    default ceiling killed a call at 120.5s — inside the healthy tail, not a
    stall. It needs analysis-scoped headroom like the Opus edit steps got."""
    te.reset_token_usage()
    with mock.patch.object(te, "run_claude", return_value=_fake_result(_GAP_JSON)) as m:
        te.gap_analysis("resume text", te.JDAnalysis(exact_job_title="Senior PM"))
    assert m.call_args.kwargs["timeout"] > te.DEFAULT_CALL_TIMEOUT
    assert m.call_args.kwargs["timeout"] >= 180


def test_gap_analysis_retries_once_on_timeout():
    """A timed-out gap_analysis failed the whole tailor run after the Doc copy
    existed and bumped the attempt counter — over one per-call CLI stall.
    Retry the call once (stalls are per-call flukes; a fresh call usually
    completes normally), like llm_reshape does."""
    te.reset_token_usage()
    with mock.patch.object(
        te, "run_claude",
        side_effect=[te.ClaudeCLITimeout("claude CLI killed: timed out after 240s"),
                     _fake_result(_GAP_JSON)],
    ) as m:
        gap = te.gap_analysis("resume text", te.JDAnalysis(exact_job_title="Senior PM"))
    assert m.call_count == 2
    assert gap.already_matched == ["AI"]
    assert gap.keyword_count == 12


def test_gap_analysis_second_timeout_propagates():
    """Two consecutive timeouts mean the CLI is genuinely wedged — the run must
    fail loudly, not degrade into an empty GapAnalysis (a mislabeled success:
    untailored copy with tailored_at set)."""
    te.reset_token_usage()
    with mock.patch.object(
        te, "run_claude",
        side_effect=te.ClaudeCLITimeout("claude CLI killed: timed out after 240s"),
    ) as m:
        with pytest.raises(te.ClaudeCLITimeout):
            te.gap_analysis("resume text", te.JDAnalysis(exact_job_title="Senior PM"))
    assert m.call_count == 2


def test_gap_analysis_non_timeout_cli_error_not_retried_again(monkeypatch):
    """Non-timeout CLI errors already retry 3x inside _claude_call_inner; the
    timeout-scoped retry must not double that to 6 attempts."""
    te.reset_token_usage()
    monkeypatch.setattr(te._claude_call_inner.retry, "sleep", lambda s: None)
    calls = {"n": 0}

    def dead_cli(*a, **k):
        calls["n"] += 1
        raise te.ClaudeCLIError("claude CLI killed: exceeded memory ceiling")

    with mock.patch.object(te, "run_claude", side_effect=dead_cli):
        with pytest.raises(te.ClaudeCLIError):
            te.gap_analysis("resume text", te.JDAnalysis(exact_job_title="Senior PM"))
    assert calls["n"] == 3


def test_claude_call_still_retries_transient_errors():
    """Non-timeout CLI errors remain retryable."""
    te.reset_token_usage()
    calls = {"n": 0}

    def flaky(*a, **k):
        calls["n"] += 1
        if calls["n"] < 2:
            raise te.ClaudeCLIError("claude CLI killed: exceeded memory ceiling")
        return _fake_result("recovered")

    with mock.patch.object(te, "run_claude", side_effect=flaky):
        out = te._claude_call("prompt", label="t", model=EDIT_MODEL)
    assert out == "recovered"
    assert calls["n"] == 2


# ---------------------------------------------------------------------------
# Job-id log context — concurrent auto-tailor workers (workers >= 2) interleave
# engine log lines; without a per-thread job tag a [layout_reshape] line is
# unattributable and its duration reads as mismatched wall-clock.
# ---------------------------------------------------------------------------

def _engine_records(caplog):
    return [r for r in caplog.records if r.name == "resume_tailor.tailor_engine"]


def test_log_context_tags_engine_records_with_job_id(caplog):
    te.reset_token_usage()
    te.set_log_context(15635)
    try:
        with caplog.at_level("INFO", logger="resume_tailor.tailor_engine"):
            with mock.patch.object(te, "run_claude", return_value=_fake_result("ok")):
                te._claude_call("prompt", label="layout_reshape")
        records = _engine_records(caplog)
        assert records
        assert all(r.getMessage().startswith("[job 15635] ") for r in records)
    finally:
        te.set_log_context(None)


def test_log_context_cleared_leaves_records_untagged(caplog):
    te.reset_token_usage()
    te.set_log_context(None)
    with caplog.at_level("INFO", logger="resume_tailor.tailor_engine"):
        with mock.patch.object(te, "run_claude", return_value=_fake_result("ok")):
            te._claude_call("prompt", label="layout_reshape")
    records = _engine_records(caplog)
    assert records
    assert not any("[job" in r.getMessage() for r in records)


# ---------------------------------------------------------------------------
# _dict_to_tailoring_edits — newline sanitization (layout-guard invariant)
# ---------------------------------------------------------------------------

def test_dict_to_tailoring_edits_sanitizes_newlines_in_replacement_text():
    """LLM replacement text must never contain embedded newlines — applying it
    via replaceAllText would split one paragraph into two, silently shifting
    every ordinal after it for the layout guard's ordinal-based tracking.
    Anchor fields (original, bold_label) must stay byte-for-byte untouched,
    and a malformed edit dict missing the replacement key must not crash."""
    data = {
        "title_line_replacement": "Principal   Product\nManager",
        "summary_replacement": "Leads AI\n\nproduct strategy   across teams.",
        "experience_edits": [
            {
                "company": "ACME",
                "bold_label": "Enterprise Scale",
                "original": "Enterprise Scale: Led delivery of AI platforms across business units.",
                "replacement_after_label": "Led delivery\nof GenAI   platforms across business units.",
            },
            {
                # Malformed: missing replacement_after_label entirely.
                "company": "ACME",
                "bold_label": "Other",
                "original": "Other: something",
            },
        ],
        "rewritten_bullets": [
            {
                "company": "ACME",
                "original": "Product Vision: Set roadmap for conversational AI portfolio at scale.",
                "rewritten": "Platform Vision:\nSet roadmap   for the agent platform\nportfolio at scale.",
            }
        ],
    }

    edits = te._dict_to_tailoring_edits(data)

    assert "\n" not in edits.title_line_replacement
    assert edits.title_line_replacement == "Principal Product Manager"

    assert "\n" not in edits.summary_replacement
    assert edits.summary_replacement == "Leads AI product strategy across teams."

    assert "\n" not in edits.experience_edits[0]["replacement_after_label"]
    assert edits.experience_edits[0]["replacement_after_label"] == (
        "Led delivery of GenAI platforms across business units."
    )
    # Anchor field is a match target for the live doc — must be preserved exactly.
    assert edits.experience_edits[0]["original"] == (
        "Enterprise Scale: Led delivery of AI platforms across business units."
    )

    # Malformed dict (no replacement_after_label) passes through without crashing.
    assert "replacement_after_label" not in edits.experience_edits[1]
    assert edits.experience_edits[1]["original"] == "Other: something"

    assert "\n" not in edits.rewritten_bullets[0]["rewritten"]
    assert edits.rewritten_bullets[0]["rewritten"] == (
        "Platform Vision: Set roadmap for the agent platform portfolio at scale."
    )
    assert edits.rewritten_bullets[0]["original"] == (
        "Product Vision: Set roadmap for conversational AI portfolio at scale."
    )


def test_dict_to_tailoring_edits_sanitizes_newlines_in_skills_reorder():
    """skills_reorder skills get joined into a single paragraph line in apply_edits
    (f"{label} {', '.join(new_skills)}") and tracked by the layout guard under role
    'skills:<sub>' — an embedded '\\n' in any skill token would split that paragraph
    and corrupt ordinals, exactly like the other four sanitized fields. Skills that
    sanitize to empty must be DROPPED (not kept as an empty string), so the joined
    line never contains a stray ', '. Malformed skills_reorder input must not crash."""
    data = {
        "skills_reorder": {
            "AI Platforms": ["GPT-4", "Lang\nChain", "  Claude  "],
        },
    }

    edits = te._dict_to_tailoring_edits(data)

    skills = edits.skills_reorder["AI Platforms"]
    assert skills == ["GPT-4", "Lang Chain", "Claude"]
    assert all("\n" not in s for s in skills)

    # Subcategory key order and skill order are preserved.
    assert list(edits.skills_reorder.keys()) == ["AI Platforms"]

    # Malformed skills_reorder must not crash the pipeline.
    assert te._dict_to_tailoring_edits({"skills_reorder": "not-a-dict"}).skills_reorder == {}
    assert te._dict_to_tailoring_edits({"skills_reorder": {}}).skills_reorder == {}
    assert te._dict_to_tailoring_edits({}).skills_reorder == {}
    assert te._dict_to_tailoring_edits(
        {"skills_reorder": {"Process Design": "not-a-list"}}
    ).skills_reorder == {}


# ---------------------------------------------------------------------------
# Layout-guard paragraph tracking
# ---------------------------------------------------------------------------

def _fake_doc(paragraphs, bold_first=()):
    """Build a minimal Docs-API doc dict. paragraphs: list of paragraph texts
    (without trailing newline). bold_first: indices whose first run is bold
    (section headers)."""
    content, idx = [], 1
    for i, text in enumerate(paragraphs):
        raw = text + "\n"
        content.append({
            "startIndex": idx,
            "endIndex": idx + len(raw),
            "paragraph": {"elements": [{
                "startIndex": idx,
                "endIndex": idx + len(raw),
                "textRun": {"content": raw,
                            "textStyle": {"bold": True} if i in bold_first else {}},
            }]},
        })
        idx += len(raw)
    return {"body": {"content": content}}


_MASTER_PARAS = [
    "JANE DOE",
    te._MASTER_TITLE_LINE,
    "PROFESSIONAL SUMMARY",
    "Results-driven AI Product Leader with 8 years of experience.",
    "CORE COMPETENCIES & TECHNICAL SKILLS",
    "Al Platforms: GPT-4, Claude, LangChain",
    "WORK EXPERIENCE",
    "Enterprise Scale: Led delivery of AI platforms across business units.",
    "Product Vision: Set roadmap for conversational AI portfolio at scale.",
    "EDUCATION",
]

def _master_doc():
    return _fake_doc(_MASTER_PARAS, bold_first=(0, 1, 2, 4, 6, 9))

def _edits():
    return te.TailoringEdits(
        title_line_replacement="Principal Product Manager",
        summary_replacement="Principal Product Manager with 8 years of experience.",
        skills_reorder={"AI Platforms": ["Claude", "GPT-4", "LangChain"]},
        experience_edits=[{
            "company": "ACME", "bold_label": "Enterprise Scale",
            "original": "Enterprise Scale: Led delivery of AI platforms across business units.",
            "replacement_after_label": "Led delivery of GenAI platforms across business units.",
        }],
        rewritten_bullets=[{
            "company": "ACME",
            "original": "Product Vision: Set roadmap for conversational AI portfolio at scale.",
            "rewritten": "Platform Vision: Set roadmap for the agent platform portfolio at scale.",
        }],
    )

def _post_doc():
    paras = list(_MASTER_PARAS)
    paras[1] = "PRINCIPAL PRODUCT MANAGER"
    paras[3] = "Principal Product Manager with 8 years of experience."
    paras[5] = "Al Platforms: Claude, GPT-4, LangChain"
    paras[7] = "Enterprise Scale: Led delivery of GenAI platforms across business units."
    paras[8] = "Platform Vision: Set roadmap for the agent platform portfolio at scale."
    return _fake_doc(paras, bold_first=(0, 1, 2, 4, 6, 9))


def test_split_label():
    assert te._split_label("Enterprise Scale: Led delivery") == ("Enterprise Scale:", "Led delivery")
    assert te._split_label("No label here at all") == (None, "No label here at all")
    # A sentence with a period before the colon is body text, not a label
    assert te._split_label("We shipped. Then: more")[0] is None

def test_build_edited_paragraphs_tracks_all_roles_and_ordinals(monkeypatch):
    # The "AI Platforms" skills line is only tracked when it's a known
    # subcategory label; that map is empty in a neutral tree, so pin the
    # owner-shaped master title line + skill labels.
    patch_tailor_master_labels(monkeypatch)
    paras = te.build_edited_paragraphs(_edits(), _master_doc(), _post_doc())
    by_role = {p.role: p for p in paras}
    assert by_role["title"].ordinal == 1
    assert by_role["summary"].ordinal == 3
    assert by_role["skills:AI Platforms"].ordinal == 5
    assert by_role["bullet:ACME:Enterprise Scale"].ordinal == 7
    assert by_role["rewrite:ACME:0"].ordinal == 8
    assert by_role["summary"].master_text == "Results-driven AI Product Leader with 8 years of experience."
    assert by_role["title"].master_text == te._MASTER_TITLE_LINE

def test_refresh_from_doc_reads_live_text_and_indices():
    paras = te.build_edited_paragraphs(_edits(), _master_doc(), _post_doc())
    doc = _post_doc()
    te.refresh_from_doc(paras, doc)
    summary = next(p for p in paras if p.role == "summary")
    assert summary.tailored_text == "Principal Product Manager with 8 years of experience."
    rec = doc["body"]["content"][3]
    assert (summary.start_index, summary.end_index) == (rec["startIndex"], rec["endIndex"])

def test_refresh_from_doc_out_of_range_ordinal_clears_text():
    p = te.lg.EditedParagraph(role="bullet:X:Y", master_text="m", ordinal=99)
    te.refresh_from_doc([p], _post_doc())
    assert p.tailored_text == ""


# ---------------------------------------------------------------------------
# apply_paragraph_replacement / revert_paragraph_to_master
# ---------------------------------------------------------------------------

class _CaptureClient:
    """Fake GoogleAPIClient capturing batch_update requests."""
    def __init__(self, doc=None, occurrences=1):
        self.requests = []
        self._doc = doc or {"body": {"content": []}}
        self._occ = occurrences
    def batch_update(self, doc_id, requests):
        self.requests.append(requests)
        return {"replies": [
            {"replaceAllText": {"occurrencesChanged": self._occ}}
            if "replaceAllText" in r else {} for r in requests
        ]}
    def read_document(self, doc_id):
        return self._doc

def _violation(tailored_text, role="bullet:ACME:Enterprise Scale"):
    return te.lg.LayoutViolation(
        role=role, kind="grew", master_lines=2, tailored_lines=3, last_line_words=2,
        tailored_text=tailored_text, master_text="", target_chars=100)

def test_apply_replacement_anchors_on_post_label_body():
    client = _CaptureClient()
    v = _violation("Enterprise Scale: Led delivery of GenAI platforms across units and regions")
    ok = te.apply_paragraph_replacement(
        "d", v, "Enterprise Scale: Led delivery of GenAI platforms across units", client)
    assert ok is True
    req = client.requests[0][0]["replaceAllText"]
    # The label is NOT part of the match — its bold run is never touched.
    assert req["containsText"]["text"] == "Led delivery of GenAI platforms across units and regions"
    assert req["replaceText"] == "Led delivery of GenAI platforms across units"

def test_apply_replacement_rejects_altered_label():
    client = _CaptureClient()
    v = _violation("Enterprise Scale: Led delivery of platforms")
    ok = te.apply_paragraph_replacement("d", v, "Enterprise Reach: Led delivery", client)
    assert ok is False
    assert client.requests == []          # nothing applied

def test_apply_replacement_sanitizes_newlines():
    client = _CaptureClient()
    v = _violation("Summary text that is long enough to shorten", role="summary")
    ok = te.apply_paragraph_replacement("d", v, "Shorter summary\nwith a newline", client)
    assert ok is True
    assert client.requests[0][0]["replaceAllText"]["replaceText"] == "Shorter summary with a newline"

def test_apply_replacement_detects_silent_noop():
    client = _CaptureClient(occurrences=0)
    v = _violation("Enterprise Scale: Led delivery of platforms")
    ok = te.apply_paragraph_replacement("d", v, "Enterprise Scale: Led delivery", client)
    assert ok is False                    # occurrencesChanged == 0 → caller must know

def test_apply_replacement_summary_early_colon_not_rejected():
    # Same early-colon shape as the revert case above, but through the
    # replaceAllText path: role="summary" must anchor on the WHOLE tailored
    # text, not a role-blind _split_label's post-colon body.
    client = _CaptureClient()
    v = _violation(
        "Results-driven Leader: 8 years of measurable impact here now.",
        role="summary")
    ok = te.apply_paragraph_replacement(
        "d", v, "Results-driven Leader: 8 years of measurable impact.", client)
    assert ok is True
    req = client.requests[0][0]["replaceAllText"]
    assert req["containsText"]["text"] == "Results-driven Leader: 8 years of measurable impact here now."
    assert req["replaceText"] == "Results-driven Leader: 8 years of measurable impact."

def test_revert_paragraph_is_positional_and_restyles():
    # Paragraph ordinal 0 spans indices [10, 60); master text has a bold label.
    doc = {"body": {"content": [{
        "startIndex": 10, "endIndex": 60,
        "paragraph": {"elements": [{"textRun": {"content": "current text here\n"}}]},
    }]}}
    client = _CaptureClient(doc=doc)
    para = te.lg.EditedParagraph(
        role="bullet:ACME:Enterprise Scale",
        master_text="Enterprise Scale: Led delivery of AI platforms.",
        ordinal=0)
    te.revert_paragraph_to_master("d", para, client)
    reqs = client.requests[0]
    assert reqs[0] == {"deleteContentRange": {"range": {"startIndex": 10, "endIndex": 59}}}
    assert reqs[1] == {"insertText": {"location": {"index": 10},
                                      "text": "Enterprise Scale: Led delivery of AI platforms."}}
    label_len = len("Enterprise Scale:")
    assert reqs[2]["updateTextStyle"]["range"] == {"startIndex": 10, "endIndex": 10 + label_len}
    assert reqs[2]["updateTextStyle"]["textStyle"] == {"bold": True}
    assert reqs[3]["updateTextStyle"]["range"] == {
        "startIndex": 10 + label_len,
        "endIndex": 10 + len("Enterprise Scale: Led delivery of AI platforms.")}
    assert reqs[3]["updateTextStyle"]["textStyle"] == {"bold": False}
    assert para.tailored_text == para.master_text

def test_revert_title_is_fully_bold():
    doc = {"body": {"content": [{
        "startIndex": 5, "endIndex": 40,
        "paragraph": {"elements": [{"textRun": {"content": "OLD TITLE LINE\n"}}]},
    }]}}
    client = _CaptureClient(doc=doc)
    para = te.lg.EditedParagraph(role="title", master_text="MASTER TITLE", ordinal=0)
    te.revert_paragraph_to_master("d", para, client)
    reqs = client.requests[0]
    style_reqs = [r for r in reqs if "updateTextStyle" in r]
    assert len(style_reqs) == 1
    assert style_reqs[0]["updateTextStyle"]["textStyle"] == {"bold": True}


def test_revert_summary_with_early_colon_stays_unbold():
    # A summary whose master text happens to contain an early colon (e.g. a
    # "Results-driven Leader:" opener) must NOT be treated as a labeled bullet
    # — _split_label fires on any short pre-colon prefix regardless of role,
    # so this guards that label detection is gated on role, not text shape.
    master = "Results-driven Leader: 8 years of measurable impact."
    current_text = "current text here"
    start = 10
    end = start + len(current_text) + 1  # +1 for the trailing "\n"
    doc = {"body": {"content": [{
        "startIndex": start, "endIndex": end,
        "paragraph": {"elements": [{"textRun": {"content": current_text + "\n"}}]},
    }]}}
    client = _CaptureClient(doc=doc)
    para = te.lg.EditedParagraph(role="summary", master_text=master, ordinal=0)
    te.revert_paragraph_to_master("d", para, client)
    reqs = client.requests[0]
    style_reqs = [r for r in reqs if "updateTextStyle" in r]
    # Exactly one uniformly non-bold span covering the whole master text — no
    # bold label span for "Results-driven Leader:" should be emitted.
    assert len(style_reqs) == 1
    assert style_reqs[0]["updateTextStyle"]["range"] == {
        "startIndex": start, "endIndex": start + len(master)}
    assert style_reqs[0]["updateTextStyle"]["textStyle"] == {"bold": False}


# ---------------------------------------------------------------------------
# _restyle_label_paragraph
# ---------------------------------------------------------------------------

def test_restyle_targets_correct_paragraph_when_label_recurs():
    # Two paragraphs share the "Enterprise Scale:" label (bullets from two
    # different companies) but have distinct bodies. Only the paragraph whose
    # FULL text matches the rewritten bullet (paragraph B) may be restyled —
    # matching on the label alone would wrongly hit paragraph A, the first in
    # document order, which is the exact bold-bleed bug this fixes.
    label = "Enterprise Scale:"
    text_a = f"{label} original body one"
    text_b = f"{label} the rewritten body two here"
    start_a = 10
    end_a = start_a + len(text_a) + 1        # +1 for the trailing "\n"
    start_b = end_a
    end_b = start_b + len(text_b) + 1
    doc = {"body": {"content": [
        {"startIndex": start_a, "endIndex": end_a,
         "paragraph": {"elements": [{"textRun": {"content": text_a + "\n"}}]}},
        {"startIndex": start_b, "endIndex": end_b,
         "paragraph": {"elements": [{"textRun": {"content": text_b + "\n"}}]}},
    ]}}
    client = _CaptureClient(doc=doc)
    te._restyle_label_paragraph("d", text_b, client)

    assert len(client.requests) == 1          # exactly one batch_update fired
    reqs = client.requests[0]
    assert len(reqs) == 2                     # label-bold + body-unbold
    assert reqs[0]["updateTextStyle"]["range"] == {
        "startIndex": start_b, "endIndex": start_b + len(label)}
    assert reqs[0]["updateTextStyle"]["textStyle"] == {"bold": True}
    assert reqs[1]["updateTextStyle"]["range"] == {
        "startIndex": start_b + len(label), "endIndex": start_b + len(text_b)}
    assert reqs[1]["updateTextStyle"]["textStyle"] == {"bold": False}
    # The restyled range starts at B's start, never A's — the disambiguation
    # the fix exists for.
    assert reqs[0]["updateTextStyle"]["range"]["startIndex"] != start_a


def test_restyle_noop_when_no_label():
    client = _CaptureClient()
    te._restyle_label_paragraph("d", "Plain text without a label prefix", client)
    assert client.requests == []


# ---------------------------------------------------------------------------
# _apply_experience_edits / _apply_rewritten_bullets — silent-no-op detection
# ---------------------------------------------------------------------------
# replaceAllText "succeeds" even when its anchor matches nothing (the batch
# reply just omits occurrencesChanged). Until 2026-07-17 these two paths
# discarded the response, so a drifted anchor dropped the edit with no trace.

_EXP_EDIT = {
    "company": "Acme Corp", "role": "Senior Manager",
    "bold_label": "AI Enablement & Adoption",
    "original": "AI Enablement & Adoption: Orchestrated rollouts that grew utilization",
    "replacement_after_label": "Orchestrated rollouts that grew engagement",
}


def test_apply_experience_edits_warns_when_anchor_matches_nothing(caplog):
    client = _CaptureClient(occurrences=0)
    with caplog.at_level("WARNING", logger="resume_tailor.tailor_engine"):
        te._apply_experience_edits("d", [dict(_EXP_EDIT)], client)
    assert any("AI Enablement & Adoption" in r.message
               and "matched nothing" in r.message for r in caplog.records)


def test_apply_experience_edits_quiet_when_edit_lands(caplog):
    client = _CaptureClient(occurrences=1)
    with caplog.at_level("WARNING", logger="resume_tailor.tailor_engine"):
        te._apply_experience_edits("d", [dict(_EXP_EDIT)], client)
    assert not any("matched nothing" in r.message for r in caplog.records)


def test_apply_experience_edits_warns_on_multiple_matches(caplog):
    client = _CaptureClient(occurrences=2)
    with caplog.at_level("WARNING", logger="resume_tailor.tailor_engine"):
        te._apply_experience_edits("d", [dict(_EXP_EDIT)], client)
    assert any("matched 2" in r.message for r in caplog.records)


def test_apply_rewritten_bullets_warns_when_anchor_matches_nothing(caplog):
    client = _CaptureClient(occurrences=0)
    bullet = {"company": "Acme Corp", "role": "Senior Manager",
              "original": "Old bullet body with no label",
              "rewritten": "New bullet body with no label"}
    with caplog.at_level("WARNING", logger="resume_tailor.tailor_engine"):
        te._apply_rewritten_bullets("d", [bullet], client)
    assert any("matched nothing" in r.message and "Acme Corp" in r.message
               for r in caplog.records)


# ---------------------------------------------------------------------------
# apply_edits — silent-no-op detection on the title/skills batch
# ---------------------------------------------------------------------------
# Same latent gap as the two paths above: apply_edits builds ONE combined
# batchUpdate for the title line and the skills reorder lines, so a drifted
# anchor (e.g. a stale _MASTER_TITLE_LINE) silently no-opped. Replies align
# 1:1 with requests, so warnings must carry the right per-request label.

class _PerRequestOccClient(_CaptureClient):
    """_CaptureClient whose occurrences may be a list — one value per request
    of the batch, so a mixed batch can land one edit and drop another."""
    def batch_update(self, doc_id, requests):
        self.requests.append(requests)
        occs = (self._occ if isinstance(self._occ, list)
                else [self._occ] * len(requests))
        return {"replies": [
            {"replaceAllText": {"occurrencesChanged": occs[i]}}
            if "replaceAllText" in r else {} for i, r in enumerate(requests)
        ]}


def _skills_doc():
    line = "lntegration Skills: Widget A, Widget B, Widget C\n"
    return {"body": {"content": [{
        "startIndex": 1, "endIndex": 1 + len(line),
        "paragraph": {"elements": [{"textRun": {"content": line}}]},
    }]}}


def test_apply_edits_raises_when_title_anchor_unconfigured(monkeypatch):
    # Neutral-tree shape: no policy.tailor.master_title_line configured.
    # apply_edits must refuse to guess an anchor rather than silently no-op.
    monkeypatch.setattr(te, "_MASTER_TITLE_LINE", "")
    edits = te.TailoringEdits(title_line_replacement="Senior AI PM")
    with pytest.raises(RuntimeError, match="master_title_line"):
        te.apply_edits("d", edits, _CaptureClient())


def test_apply_edits_warns_when_title_anchor_matches_nothing(caplog, monkeypatch):
    # A configured (fixture) anchor that simply doesn't match the doc — this
    # pins the warn-on-drifted-anchor path, distinct from the RuntimeError
    # above (which fires only when no anchor is configured at all).
    patch_tailor_anchors(monkeypatch)
    client = _CaptureClient(occurrences=0)
    edits = te.TailoringEdits(title_line_replacement="Senior AI PM")
    with caplog.at_level("WARNING", logger="resume_tailor.tailor_engine"):
        te.apply_edits("d", edits, client)
    assert any("'title'" in r.message and "matched nothing" in r.message
               for r in caplog.records)


def test_apply_edits_labels_skill_line_noop_with_subcategory(caplog, monkeypatch):
    # Mixed batch: title (request 0) lands, skills line (request 1) no-ops.
    # The warning must name the subcategory — proving labels track requests
    # 1:1 even when the skills loop skips entries before queueing this one.
    patch_tailor_anchors(monkeypatch)
    client = _PerRequestOccClient(doc=_skills_doc(), occurrences=[1, 0])
    edits = te.TailoringEdits(
        title_line_replacement="Senior AI PM",
        skills_reorder={"Integration Skills": ["Widget B", "Widget A", "Widget C"]})
    with caplog.at_level("WARNING", logger="resume_tailor.tailor_engine"):
        te.apply_edits("d", edits, client)
    assert any("'Integration Skills'" in r.message and "matched nothing" in r.message
               for r in caplog.records)
    assert not any("'title'" in r.message for r in caplog.records)


def test_apply_edits_quiet_when_title_and_skills_land(caplog, monkeypatch):
    patch_tailor_anchors(monkeypatch)
    client = _CaptureClient(doc=_skills_doc(), occurrences=1)
    edits = te.TailoringEdits(
        title_line_replacement="Senior AI PM",
        skills_reorder={"Integration Skills": ["Widget B", "Widget A", "Widget C"]})
    with caplog.at_level("WARNING", logger="resume_tailor.tailor_engine"):
        te.apply_edits("d", edits, client)
    assert not any("matched nothing" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# llm_reshape / enforce_layout
# ---------------------------------------------------------------------------

def test_llm_reshape_returns_shortened_text():
    with mock.patch.object(te, "_claude_call", return_value='"Enterprise Scale: Led delivery"'):
        out = te.llm_reshape("Enterprise Scale: Led delivery of AI platforms", 30, 1)
    assert out == "Enterprise Scale: Led delivery"

def test_llm_reshape_rejects_longer_output():
    longer = "Enterprise Scale: Led delivery of many more AI platforms than before"
    with mock.patch.object(te, "_claude_call", return_value=longer):
        assert te.llm_reshape("Enterprise Scale: Led delivery", 20, 1) is None

def test_llm_reshape_none_on_llm_failure():
    with mock.patch.object(te, "_claude_call", return_value=None):
        assert te.llm_reshape("text", 10, 1) is None

def test_llm_reshape_dangler_prompt_asks_minimal_cut_with_floor():
    # A dangler already fits its line count — the prompt must ask for a small
    # trim above a stated floor, not the grew-style "fit within N lines" cut
    # that over-shortens into a shrank (the 2026-07-17 revert-cascade incident).
    text = ("Product Management Leadership: Directed a PM team of Product "
            "Owners plus one direct report to integrate LLM capabilities "
            "across platforms at 4M queries per month for the business")
    captured = {}
    def fake_call(prompt, **kw):
        captured["prompt"] = prompt
        return text[:-10]
    with mock.patch.object(te, "_claude_call", side_effect=fake_call):
        te.llm_reshape(text, 251, 3, kind="dangler", min_chars=120,
                       last_line_words=2)
    assert "120" in captured["prompt"]          # the floor is stated
    assert "2-word" in captured["prompt"]       # names the dangling last line
    # The grew-framing must not appear for danglers.
    assert "Shorten this resume paragraph so it fits" not in captured["prompt"]

def test_llm_reshape_rejects_result_at_or_below_floor():
    # Cutting below the floor drops a rendered line (shrank → forced revert),
    # so such a result is rejected rather than applied.
    text = "x" * 200
    with mock.patch.object(te, "_claude_call", return_value="x" * 100):
        assert te.llm_reshape(text, 210, 3, kind="dangler",
                              min_chars=150) is None

def test_llm_reshape_preserve_terms_in_prompt_and_enforced():
    text = ("Product Management Leadership: Directed a PM team plus one "
            "direct report (people management), aligning roadmaps across "
            "platforms at scale for the enterprise business units worldwide")
    captured = {}
    def drops_term(prompt, **kw):
        captured["prompt"] = prompt
        return ("Product Management Leadership: Directed a PM team, aligning "
                "roadmaps across platforms at scale for the enterprise")
    with mock.patch.object(te, "_claude_call", side_effect=drops_term):
        out = te.llm_reshape(text, 250, 3,
                             preserve_terms=["people management"])
    assert "people management" in captured["prompt"]
    assert out is None                          # dropped term → rejected
    def keeps_term(prompt, **kw):
        return ("Product Management Leadership: Directed a PM team plus one "
                "direct report (people management), aligning roadmaps at scale")
    with mock.patch.object(te, "_claude_call", side_effect=keeps_term):
        out = te.llm_reshape(text, 250, 3,
                             preserve_terms=["people management"])
    assert out is not None                      # term kept → accepted

def test_llm_reshape_preserve_terms_fold_ampersand_variant():
    # The doc may carry the "&" form of a preserved "and" term (or vice
    # versa); enforcement uses the same folding matcher as the recompute, so
    # the "&" form counts as preserved.
    text = ("Experimentation & Funnel Analysis: A/B tested conversational "
            "flows for conversion, driving a 20% usage surge at leading "
            "containment and completion rates for the whole business unit")
    def keeps_amp_form(prompt, **kw):
        return ("Experimentation & Funnel Analysis: A/B tested flows for "
                "conversion, driving a 20% usage surge at leading rates")
    with mock.patch.object(te, "_claude_call", side_effect=keeps_amp_form):
        out = te.llm_reshape(text, 250, 3,
                             preserve_terms=["experimentation and funnel analysis"])
    assert out is not None


# ---------------------------------------------------------------------------
# llm_reshape timeout resilience — 2026-07-18 incident: sporadic per-call CLI
# stalls (identical ~600-char prompts measured at 8s and 120s+ in one run,
# solo AND concurrent) hit the 120s kill; the raised ClaudeCLITimeout crashed
# the whole layout guard → layout_unverified, skipping reconciliation reverts.
# ---------------------------------------------------------------------------

def test_llm_reshape_retries_timed_out_call_once():
    calls = []
    def stall_then_succeed(prompt, **kw):
        calls.append(kw.get("label"))
        if len(calls) == 1:
            raise te.ClaudeCLITimeout("claude CLI killed: timed out after 120s")
        return '"Enterprise Scale: Led delivery"'
    with mock.patch.object(te, "_claude_call", side_effect=stall_then_succeed):
        out = te.llm_reshape("Enterprise Scale: Led delivery of AI platforms", 30, 1)
    assert out == "Enterprise Scale: Led delivery"
    assert len(calls) == 2


def test_llm_reshape_second_timeout_degrades_to_none():
    """Two stalls in a row → treat as no-repair (the guard's round loop and
    reconciliation take over), never raise out of the guard."""
    with mock.patch.object(
            te, "_claude_call",
            side_effect=te.ClaudeCLITimeout("claude CLI killed: timed out after 120s")) as m:
        assert te.llm_reshape("some paragraph text to shorten", 10, 1) is None
    assert m.call_count == 2          # exactly one retry, then give up


def test_llm_reshape_non_timeout_cli_error_still_raises():
    """Systemic CLI failures (missing binary, auth) must stay loud — only the
    sporadic-stall timeout degrades gracefully."""
    with mock.patch.object(
            te, "_claude_call",
            side_effect=te.ClaudeCLIError("claude CLI not found on PATH")):
        with pytest.raises(te.ClaudeCLIError):
            te.llm_reshape("some paragraph text to shorten", 10, 1)


class _ScriptedGuardClient(_CaptureClient):
    """export_as_pdf returns sentinel bytes; the test patches build_line_map to
    translate each sentinel into a scripted LineMap."""
    def __init__(self, doc, pdfs):
        super().__init__(doc=doc)
        self._pdfs = list(pdfs)
        self.export_calls = 0
    def export_as_pdf(self, doc_id):
        self.export_calls += 1
        return self._pdfs.pop(0) if len(self._pdfs) > 1 else self._pdfs[0]


def _guard_fixtures():
    """One tracked bullet. Master: 2 lines. Tailored v1: 3 lines (grew).
    Tailored v2 (after reshape): 2 lines (clean)."""
    master_text = ("Enterprise Scale: Led delivery of AI platforms across "
                   "business units and multiple regions worldwide")
    master_map = _lm_engine(
        "Enterprise Scale: Led delivery of AI platforms across",
        "business units and multiple regions worldwide")
    grown_text = ("Enterprise Scale: Led delivery of GenAI platforms across "
                  "business units and multiple regions worldwide at scale")
    grown_map = _lm_engine(
        "Enterprise Scale: Led delivery of GenAI platforms across",
        "business units and multiple regions worldwide at",
        "scale")
    fixed_text = ("Enterprise Scale: Led delivery of GenAI platforms across "
                  "business units and multiple regions worldwide")
    fixed_map = _lm_engine(
        "Enterprise Scale: Led delivery of GenAI platforms across",
        "business units and multiple regions worldwide")
    return master_text, master_map, grown_text, grown_map, fixed_text, fixed_map

def _lm_engine(*texts):
    return te.lg.LineMap(
        lines=[te.lg.RenderedLine(text=t, words=len(te.lg.normalize(t).split()), top=float(i))
               for i, t in enumerate(texts)],
        page_count=1)

def _one_para_doc(text):
    raw = text + "\n"
    return {"body": {"content": [{
        "startIndex": 1, "endIndex": 1 + len(raw),
        "paragraph": {"elements": [{"textRun": {"content": raw}}]},
    }]}}


def test_enforce_layout_repairs_grown_paragraph(monkeypatch):
    master_text, master_map, grown_text, grown_map, fixed_text, fixed_map = _guard_fixtures()
    edits = te.TailoringEdits(experience_edits=[{
        "company": "ACME", "bold_label": "Enterprise Scale",
        "original": master_text,
        "replacement_after_label": grown_text.split(": ", 1)[1],
    }])

    docs = [_one_para_doc(grown_text), _one_para_doc(fixed_text), _one_para_doc(fixed_text)]
    maps = [grown_map, fixed_map, fixed_map]
    client = _ScriptedGuardClient(doc=None, pdfs=[b"pdf1", b"pdf2", b"pdf3"])
    client.read_document = lambda doc_id: docs.pop(0) if len(docs) > 1 else docs[0]
    monkeypatch.setattr(te, "build_line_map", lambda b: maps.pop(0) if len(maps) > 1 else maps[0])
    monkeypatch.setattr(te, "llm_reshape",
                        lambda cur, chars, lines, **kw: fixed_text)

    final_pdf, warnings = te.enforce_layout("d", edits, _one_para_doc(master_text),
                                            master_map, client)
    assert warnings == []                       # repaired, nothing reverted
    assert final_pdf is not None
    applied = [r for batch in client.requests for r in batch if "replaceAllText" in r]
    assert len(applied) == 1                    # one repair replacement

def test_enforce_layout_reverts_unrepairable_paragraph(monkeypatch):
    master_text, master_map, grown_text, grown_map, _, _ = _guard_fixtures()
    edits = te.TailoringEdits(experience_edits=[{
        "company": "ACME", "bold_label": "Enterprise Scale",
        "original": master_text,
        "replacement_after_label": grown_text.split(": ", 1)[1],
    }])
    client = _ScriptedGuardClient(doc=_one_para_doc(grown_text), pdfs=[b"pdf"])
    monkeypatch.setattr(te, "build_line_map", lambda b: grown_map)   # never improves
    monkeypatch.setattr(te, "llm_reshape", lambda cur, chars, lines, **kw: None)  # reshape fails

    final_pdf, warnings = te.enforce_layout("d", edits, _one_para_doc(master_text),
                                            master_map, client)
    assert len(warnings) >= 1
    assert "Reverted" in warnings[0] or "reverted" in warnings[0]
    deletes = [r for batch in client.requests for r in batch if "deleteContentRange" in r]
    inserts = [r for batch in client.requests for r in batch if "insertText" in r]
    assert deletes and inserts                  # positional revert happened
    assert inserts[0]["insertText"]["text"] == master_text


def test_enforce_layout_survives_reshape_timeouts(monkeypatch):
    """The 2026-07-18 incident shape: every reshape CLI call stalls to the
    timeout kill. The guard must complete — reverting the violation to master
    via reconciliation — instead of crashing out of enforcement and leaving
    the doc layout_unverified with violations still in place."""
    master_text, master_map, grown_text, grown_map, _, _ = _guard_fixtures()
    edits = te.TailoringEdits(experience_edits=[{
        "company": "ACME", "bold_label": "Enterprise Scale",
        "original": master_text,
        "replacement_after_label": grown_text.split(": ", 1)[1],
    }])
    client = _ScriptedGuardClient(doc=_one_para_doc(grown_text), pdfs=[b"pdf"])
    monkeypatch.setattr(te, "build_line_map", lambda b: grown_map)   # never improves
    with mock.patch.object(
            te, "_claude_call",
            side_effect=te.ClaudeCLITimeout("claude CLI killed: timed out after 120s")) as m:
        final_pdf, warnings = te.enforce_layout(
            "d", edits, _one_para_doc(master_text), master_map, client)
    assert m.call_count == 2                    # one retry, then no-repair
    assert any("reverted" in w.lower() for w in warnings)
    inserts = [r for batch in client.requests for r in batch if "insertText" in r]
    assert inserts and inserts[0]["insertText"]["text"] == master_text

def _dangler_fixtures():
    """One tracked bullet whose tailored text renders at the master's 3 lines
    but ends in a 2-word dangler, and carries a credited screening term
    ("people management") that the master text lacks."""
    m1 = "Enterprise Scale: Directed a matrixed team of cross-channel Product"
    m2 = "Owners and one direct report to integrate LLM capabilities across many"
    m3 = "internal platforms processing four million employee queries monthly"
    master_text = f"{m1} {m2} {m3}"
    master_map = _lm_engine(m1, m2, m3)

    t1 = "Enterprise Scale: Directed a PM team of cross-channel Product Owners"
    t2 = "plus one direct report (people management), aligning roadmaps to"
    t3 = "integrate capabilities"
    dangler_text = f"{t1} {t2} {t3}"
    dangler_map = _lm_engine(t1, t2, t3)

    s1 = "Enterprise Scale: Directed a PM team of cross-channel Product Owners"
    s2 = "plus one direct report, aligning roadmaps broadly"
    short_text = f"{s1} {s2}"
    short_map = _lm_engine(s1, s2)

    edits = te.TailoringEdits(experience_edits=[{
        "company": "ACME", "bold_label": "Enterprise Scale",
        "original": master_text,
        "replacement_after_label": dangler_text.split(": ", 1)[1],
    }])
    credited = [{"term": "people management", "aliases": ["team management"],
                 "verdict": "explicit"}]
    return (master_text, master_map, dangler_text, dangler_map,
            short_text, short_map, edits, credited)


def test_enforce_layout_passes_dangler_context_to_reshape(monkeypatch):
    (master_text, master_map, dangler_text, dangler_map,
     _, _, edits, credited) = _dangler_fixtures()
    client = _ScriptedGuardClient(doc=_one_para_doc(dangler_text), pdfs=[b"pdf"])
    monkeypatch.setattr(te, "build_line_map", lambda b: dangler_map)
    captured = {}
    def fake_reshape(cur, chars, lines, **kw):
        captured.update(kw, current=cur)
        return None
    monkeypatch.setattr(te, "llm_reshape", fake_reshape)

    te.enforce_layout("d", edits, _one_para_doc(master_text), master_map,
                      client, credited_items=credited)
    assert captured["kind"] == "dangler"
    assert captured["last_line_words"] == 2
    assert captured["preserve_terms"] == ["people management"]
    # Floor sits above the first two (full) rendered lines' combined length —
    # a result at or below it would drop to 2 lines.
    first_two = len(dangler_map.lines[0].text) + len(dangler_map.lines[1].text)
    assert captured["min_chars"] >= first_two
    assert captured["min_chars"] < len(dangler_text)


def test_enforce_layout_keeps_term_carrying_dangler_instead_of_revert(monkeypatch):
    (master_text, master_map, dangler_text, dangler_map,
     _, _, edits, credited) = _dangler_fixtures()
    client = _ScriptedGuardClient(doc=_one_para_doc(dangler_text), pdfs=[b"pdf"])
    monkeypatch.setattr(te, "build_line_map", lambda b: dangler_map)
    monkeypatch.setattr(te, "llm_reshape", lambda cur, chars, lines, **kw: None)

    final_pdf, warnings = te.enforce_layout(
        "d", edits, _one_para_doc(master_text), master_map, client,
        credited_items=credited)
    deletes = [r for batch in client.requests for r in batch
               if "deleteContentRange" in r]
    assert deletes == []                       # never reverted
    assert any("kept" in w and "people management" in w for w in warnings)
    assert not any("persist" in w for w in warnings)
    assert final_pdf is not None
    assert client.export_calls == 2            # loop round + reconcile, no third


def test_enforce_layout_restores_page_safe_text_when_reshape_shrank(monkeypatch):
    (master_text, master_map, dangler_text, dangler_map,
     short_text, short_map, edits, credited) = _dangler_fixtures()
    docs = [_one_para_doc(dangler_text), _one_para_doc(short_text),
            _one_para_doc(short_text), _one_para_doc(dangler_text)]
    maps = [dangler_map, short_map, short_map, dangler_map]
    client = _ScriptedGuardClient(doc=None, pdfs=[b"p1", b"p2", b"p3", b"p4"])
    client.read_document = lambda doc_id: docs.pop(0) if len(docs) > 1 else docs[0]
    monkeypatch.setattr(te, "build_line_map",
                        lambda b: maps.pop(0) if len(maps) > 1 else maps[0])
    # The over-cut repair the real floor would normally reject — force it
    # through to prove the reconcile pass recovers from a shrank aftermath.
    monkeypatch.setattr(te, "llm_reshape", lambda cur, chars, lines, **kw: short_text)

    final_pdf, warnings = te.enforce_layout(
        "d", edits, _one_para_doc(master_text), master_map, client,
        credited_items=credited)
    inserts = [r for batch in client.requests for r in batch if "insertText" in r]
    assert len(inserts) == 1
    # Restored to the recorded page-safe dangler text — NOT the master text.
    assert inserts[0]["insertText"]["text"] == dangler_text
    assert any("restored" in w and "people management" in w for w in warnings)
    assert not any("persist" in w for w in warnings)


def test_enforce_layout_dangler_without_credited_terms_still_reverts(monkeypatch):
    (master_text, master_map, dangler_text, dangler_map,
     _, _, edits, _) = _dangler_fixtures()
    client = _ScriptedGuardClient(doc=_one_para_doc(dangler_text), pdfs=[b"pdf"])
    monkeypatch.setattr(te, "build_line_map", lambda b: dangler_map)
    monkeypatch.setattr(te, "llm_reshape", lambda cur, chars, lines, **kw: None)

    final_pdf, warnings = te.enforce_layout(
        "d", edits, _one_para_doc(master_text), master_map, client)
    inserts = [r for batch in client.requests for r in batch if "insertText" in r]
    assert len(inserts) == 1                   # reverted exactly as before
    assert inserts[0]["insertText"]["text"] == master_text
    assert any("reverted" in w.lower() for w in warnings)


def test_enforce_layout_clean_run_single_export(monkeypatch):
    master_text, master_map, *_ = _guard_fixtures()
    edits = te.TailoringEdits(experience_edits=[{
        "company": "ACME", "bold_label": "Enterprise Scale",
        "original": master_text,
        "replacement_after_label": master_text.split(": ", 1)[1],
    }])
    client = _ScriptedGuardClient(doc=_one_para_doc(master_text), pdfs=[b"pdf"])
    monkeypatch.setattr(te, "build_line_map", lambda b: master_map)

    final_pdf, warnings = te.enforce_layout("d", edits, _one_para_doc(master_text),
                                            master_map, client)
    assert warnings == []
    assert client.export_calls == 1             # no violations → one export, reused


# ---------------------------------------------------------------------------
# Must-have grounding helpers (spec: 2026-07-11-tailor-grounding-design.md)
# ---------------------------------------------------------------------------

def test_projected_text_unions_master_and_all_edit_fragments():
    edits = te.TailoringEdits(
        title_line_replacement="TITLEX",
        summary_replacement="SUMMARYX",
        skills_reorder={"AI Platforms": ["SkillA", "SkillB"]},
        experience_edits=[{"replacement_after_label": "did EDITX"}],
        rewritten_bullets=[{"rewritten": "Label: did REWRITEY"}],
    )
    text = te._projected_text(edits, master_text="MASTERX")
    for token in ("MASTERX", "TITLEX", "SUMMARYX", "SkillA", "SkillB",
                  "EDITX", "REWRITEY"):
        assert token in text


def test_credited_must_haves_v2_explicit_evidenced_only():
    items = [
        {"term": "AI enablement", "aliases": ["AI tooling adoption"],
         "verdict": "explicit", "evidence": "led rollout"},
        {"term": "PLM", "aliases": [], "verdict": "evidenced", "evidence": "e"},
        {"term": "Kubernetes", "aliases": [], "verdict": "absent", "evidence": ""},
    ]
    assert [m["term"] for m in te.credited_must_haves(items)] == [
        "AI enablement", "PLM"]


def test_credited_must_haves_v1_present_true_only():
    items = [
        {"term": "SDLC", "aliases": [], "present": True},
        {"term": "agentification", "aliases": [], "present": False},
    ]
    assert [m["term"] for m in te.credited_must_haves(items)] == ["SDLC"]


def test_credited_must_haves_tolerates_junk():
    assert te.credited_must_haves(None) == []
    assert te.credited_must_haves([]) == []
    assert te.credited_must_haves(
        [{"term": "  ", "verdict": "explicit"}, "nope", {"present": True}]) == []


def test_missing_credited_must_haves_aliases_and_whole_tokens():
    edits = te.TailoringEdits(
        summary_replacement="Drove AI tooling adoption at scale")
    items = [
        {"term": "AI enablement", "aliases": ["AI tooling adoption"],
         "verdict": "explicit", "evidence": "e"},
        {"term": "PLM", "aliases": ["Product Lifecycle Management"],
         "verdict": "evidenced", "evidence": "e"},
        {"term": "Kubernetes", "aliases": [], "verdict": "absent"},
    ]
    missing = te.missing_credited_must_haves(
        edits, items, master_text="Led PLMx efforts")
    # Alias hit satisfies "AI enablement"; "PLMx" is NOT a whole-token "PLM"
    # hit; absent-verdict terms are never enforced.
    assert [m["term"] for m in missing] == ["PLM"]


def test_missing_credited_must_haves_sees_master_text():
    items = [{"term": "SDLC", "aliases": [], "present": True}]
    assert te.missing_credited_must_haves(
        te.TailoringEdits(), items, master_text="Owned SDLC gates") == []
    assert [m["term"] for m in te.missing_credited_must_haves(
        te.TailoringEdits(), items, master_text="nothing here")] == ["SDLC"]


def test_build_truth_system_none_without_inventory():
    assert te.build_truth_system("") is None
    assert te.build_truth_system("   \n") is None


def test_build_truth_system_embeds_inventory_and_binding_rules():
    sys_prompt = te.build_truth_system("## HARD FACTS\n- 8.7 years")
    assert "## HARD FACTS" in sys_prompt
    assert "NOT CLAIMED" in sys_prompt
    assert "BASELINE PRODUCT CRAFT" in sys_prompt
    assert "fabricate" in sys_prompt.lower()


def test_claude_call_passes_system_prompt_to_cli():
    te.reset_token_usage()
    with mock.patch.object(te, "run_claude",
                           return_value=_fake_result("ok")) as m:
        out = te._claude_call("p", label="t", system_prompt="SYS")
    assert out == "ok"
    assert m.call_args.kwargs["system_prompt"] == "SYS"


def test_claude_call_defaults_to_no_system_prompt():
    te.reset_token_usage()
    with mock.patch.object(te, "run_claude",
                           return_value=_fake_result("ok")) as m:
        te._claude_call("p", label="t")
    assert m.call_args.kwargs["system_prompt"] is None


import json as _json


def _fake_gap_response() -> str:
    return _json.dumps({"already_matched": [], "missing_but_relevant": [],
                        "missing_and_irrelevant": [], "keyword_count": 0})


def _capture_claude_call(captured, response):
    def fake_call(prompt, temperature=0.1, label="", model=None, timeout=None,
                  system_prompt=None, **kw):
        captured["prompt"] = prompt
        captured["system_prompt"] = system_prompt
        return response
    return fake_call


def test_gap_analysis_prompt_carries_verdicts_evidence_and_truth_system():
    captured = {}
    jd = te.JDAnalysis(exact_job_title="Director, PM Ops",
                       priority_keywords=["a", "b"])
    mh = [
        {"term": "AI enablement", "aliases": ["AI tooling adoption"],
         "verdict": "explicit", "evidence": "led workforce-wide rollout"},
        {"term": "Kubernetes", "aliases": [], "verdict": "absent",
         "evidence": ""},
    ]
    with mock.patch.object(te, "_claude_call",
                           side_effect=_capture_claude_call(
                               captured, _fake_gap_response())):
        te.gap_analysis("resume text", jd, inventory_text="## HARD FACTS",
                        must_haves=mh)
    p = captured["prompt"]
    assert "SCREENING MUST-HAVE TERMS" in p
    assert 'MUST INCLUDE "AI enablement"' in p
    assert "AI tooling adoption" in p
    assert "led workforce-wide rollout" in p
    assert 'DO NOT INCLUDE "Kubernetes"' in p
    assert captured["system_prompt"] == te.build_truth_system("## HARD FACTS")


def test_gap_analysis_prompt_v1_items_render_keep_and_target():
    captured = {}
    jd = te.JDAnalysis(exact_job_title="PM")
    mh = [
        {"term": "SDLC", "aliases": [], "present": True},
        {"term": "agentification", "aliases": ["AI agent integration"],
         "present": False},
    ]
    with mock.patch.object(te, "_claude_call",
                           side_effect=_capture_claude_call(
                               captured, _fake_gap_response())):
        te.gap_analysis("resume text", jd, must_haves=mh)
    p = captured["prompt"]
    assert 'MUST KEEP "SDLC"' in p
    assert 'TARGET "agentification"' in p
    assert "AI agent integration" in p


def test_gap_analysis_legacy_call_has_no_block_and_no_system_prompt():
    captured = {}
    jd = te.JDAnalysis(exact_job_title="PM")
    with mock.patch.object(te, "_claude_call",
                           side_effect=_capture_claude_call(
                               captured, _fake_gap_response())):
        te.gap_analysis("resume text", jd)
    assert "SCREENING MUST-HAVE TERMS" not in captured["prompt"]
    assert captured["system_prompt"] is None


def _valid_edits_json(summary: str = "S") -> str:
    return _json.dumps({
        "title_line_replacement": "T", "summary_replacement": summary,
        "skills_reorder": {}, "experience_edits": [], "rewritten_bullets": [],
        "keywords_used": [], "keyword_count": 1, "rationale": ""})


def test_generate_edits_prompt_lists_credited_terms_and_rules():
    captured = {}
    mh = [
        {"term": "AI enablement", "aliases": [], "verdict": "explicit",
         "evidence": "e"},
        {"term": "Kubernetes", "aliases": [], "verdict": "absent"},
    ]
    with mock.patch.object(te, "_claude_call",
                           side_effect=_capture_claude_call(
                               captured, "not json")):
        te.generate_edits("resume", te.JDAnalysis(exact_job_title="PM"),
                          te.GapAnalysis(), "ACME",
                          inventory_text="## HARD FACTS", must_haves=mh)
    p = captured["prompt"]
    assert "SCREENING MUST-HAVE RULES" in p
    assert "SCREENING MUST-HAVE TERMS" in p
    assert 'MUST INCLUDE "AI enablement"' in p
    assert 'DO NOT INCLUDE "Kubernetes"' in p
    assert captured["system_prompt"] == te.build_truth_system("## HARD FACTS")


def test_generate_edits_forwards_must_haves_and_system_to_loop():
    seen = {}
    mh = [{"term": "PLM", "aliases": [], "verdict": "evidenced",
           "evidence": "e"}]

    def fake_loop(edits, priority_keywords, resume_text, company,
                  word_count=0, max_word_count=0, must_haves=None,
                  system_prompt=None):
        seen["must_haves"] = must_haves
        seen["system_prompt"] = system_prompt
        return edits

    with mock.patch.object(te, "_claude_call",
                           return_value=_valid_edits_json()), \
         mock.patch.object(te, "_keyword_correction_loop",
                           side_effect=fake_loop):
        te.generate_edits("resume", te.JDAnalysis(exact_job_title="PM"),
                          te.GapAnalysis(), "ACME",
                          inventory_text="## HARD FACTS", must_haves=mh)
    assert seen["must_haves"] == mh
    assert seen["system_prompt"] == te.build_truth_system("## HARD FACTS")


def test_generate_edits_legacy_prompt_has_no_must_have_rules():
    captured = {}
    with mock.patch.object(te, "_claude_call",
                           side_effect=_capture_claude_call(
                               captured, "not json")):
        te.generate_edits("resume", te.JDAnalysis(exact_job_title="PM"),
                          te.GapAnalysis(), "ACME")
    assert "SCREENING MUST-HAVE RULES" not in captured["prompt"]
    assert "SCREENING MUST-HAVE TERMS" not in captured["prompt"]
    assert captured["system_prompt"] is None


def test_correction_loop_fires_on_missing_must_have_at_ok_density(monkeypatch):
    monkeypatch.setattr(te, "KEYWORD_MIN", 1)
    monkeypatch.setattr(te, "KEYWORD_MAX", 5)
    prompts = []

    def fake_call(prompt, temperature=0.1, label="", model=None, timeout=None,
                  system_prompt=None, **kw):
        prompts.append(prompt)
        return _valid_edits_json(summary="Drove PLM adoption")

    with mock.patch.object(te, "_claude_call", side_effect=fake_call):
        edits = te._keyword_correction_loop(
            te.TailoringEdits(summary_replacement="No terms here"),
            ["alpha"], "master alpha text", "ACME",
            must_haves=[{"term": "PLM",
                         "aliases": ["Product Lifecycle Management"],
                         "verdict": "evidenced", "evidence": "e"}])
    assert len(prompts) == 1               # one correction round, then satisfied
    assert "MISSING SCREENING MUST-HAVES" in prompts[0]
    assert '"PLM"' in prompts[0]
    assert "Product Lifecycle Management" in prompts[0]
    assert "PLM" in edits.summary_replacement


def test_correction_loop_no_calls_when_ok_and_credited_present(monkeypatch):
    monkeypatch.setattr(te, "KEYWORD_MIN", 1)
    monkeypatch.setattr(te, "KEYWORD_MAX", 5)
    with mock.patch.object(te, "_claude_call") as cc:
        te._keyword_correction_loop(
            te.TailoringEdits(summary_replacement="Owns PLM today"),
            ["alpha"], "master alpha text", "ACME",
            must_haves=[{"term": "PLM", "aliases": [], "verdict": "explicit",
                         "evidence": "e"}])
    cc.assert_not_called()


def test_correction_loop_warns_and_proceeds_when_term_never_lands(
        monkeypatch, caplog):
    monkeypatch.setattr(te, "KEYWORD_MIN", 1)
    monkeypatch.setattr(te, "KEYWORD_MAX", 5)
    with mock.patch.object(te, "_claude_call",
                           return_value=_valid_edits_json(
                               summary="still nothing")) as cc, \
         caplog.at_level("WARNING"):
        edits = te._keyword_correction_loop(
            te.TailoringEdits(summary_replacement="No terms here"),
            ["alpha"], "master alpha text", "ACME",
            must_haves=[{"term": "PLM", "aliases": [], "verdict": "evidenced",
                         "evidence": "e"}])
    assert cc.call_count == te.KEYWORD_CORRECTION_ROUNDS
    assert edits.summary_replacement == "still nothing"
    assert any("PLM" in r.message for r in caplog.records)


def test_correction_loop_legacy_density_behavior_unchanged(monkeypatch):
    # No must_haves: OK density exits with zero LLM calls, exactly as before.
    monkeypatch.setattr(te, "KEYWORD_MIN", 1)
    monkeypatch.setattr(te, "KEYWORD_MAX", 5)
    with mock.patch.object(te, "_claude_call") as cc:
        edits = te._keyword_correction_loop(
            te.TailoringEdits(), ["alpha"], "master alpha text", "ACME")
    cc.assert_not_called()
    assert edits.keyword_count == 1


def test_legacy_prompt_seams_have_no_placeholder_whitespace_leak():
    """The byte-identical-legacy invariant (prompt-cache prefix + unchanged
    ungrounded behavior) lives entirely at the placeholder seams. This asserts
    each insertion point renders flush when its placeholder is empty — a stray
    space/newline on a placeholder line would break it without tripping the
    marker-absence tests. Complements the byte-identical check verified against
    HEAD during implementation.
    """
    g = te._GAP_ANALYSIS_PROMPT.format(
        grounding_rules="", name="Alex", resume_text="RT", priority_keywords="PK",
        jd_analysis_json="JD", must_have_block="")
    assert "credibility\n\nReturn ONLY valid JSON with these keys:" in g
    assert "}\n\nResume:\nRT" in g

    e = te._GENERATE_EDITS_PROMPT.format(
        exact_job_title="X", current_count=1, additions_needed=2, word_count=10,
        max_word_count=9, resume_text="RT", company="CO", must_have_rules="",
        must_have_block="", priority_keywords="PK", gap_analysis_json="GA",
        skills_reorder_example="{}")
    assert "Not synonyms.\n\n=== OTHER RULES ===" in e
    assert "}\n\nResume (10 words" in e

    c = te._KEYWORD_CORRECTION_PROMPT.format(
        count=3, direction_instruction="DIR", must_have_instruction="",
        word_count=10, max_word_count=9, matched="M", missing="MI",
        previous_edits_json="PE")
    assert "DIR\n\nWORD COUNT AND CHARACTER LENGTH" in c


def test_skills_reorder_example_tracks_configured_subcategory_labels(monkeypatch):
    """The skills_reorder example block shown to the LLM must be built from
    whatever subcategory labels THIS tree has configured, not a hardcoded
    owner set — otherwise the LLM is steered toward keys apply_edits'
    `sub in _SKILL_SUBCATEGORY_LABELS` filter then silently drops."""
    patch_tailor_anchors(monkeypatch)  # -> FIXTURE_SKILL_LABELS (2 entries)
    fragment = te._build_skills_reorder_example()
    assert '"Core Practices"' in fragment
    assert '"Integration Skills"' in fragment
    assert "AI Platforms" not in fragment
    assert "Delivery Toolkit" not in fragment
    assert "Team Leadership" not in fragment
    assert "Process Design" not in fragment

    captured = {}
    with mock.patch.object(te, "_claude_call",
                           side_effect=_capture_claude_call(
                               captured, "not json")):
        te.generate_edits("resume", te.JDAnalysis(exact_job_title="PM"),
                          te.GapAnalysis(), "ACME")
    p = captured["prompt"]
    assert '"Core Practices"' in p
    assert '"Integration Skills"' in p
    assert "AI Platforms" not in p
    assert "Delivery Toolkit" not in p
    assert "Team Leadership" not in p
    assert "Process Design" not in p
