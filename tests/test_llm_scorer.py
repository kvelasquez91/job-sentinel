import json
import logging
import sqlite3
from unittest import mock

import pytest

import engine.llm_scorer as scorer_mod
from engine.llm_scorer import LLMScorer
from policy_fixtures import patch_pm_prefilter

_ENVELOPE_TEXT = (
    '{"role_match":28,"seniority_match":16,"remote_location":20,"ai_domain_fit":18,'
    '"comp_match":9,"fit_score":91,"explanation":"Strong PM/AI fit.",'
    '"salary_min":230000,"salary_max":260000,'
    '"est_total_comp_min":280000,"est_total_comp_max":360000}'
)


def _fake_run(*_args, **_kwargs):
    return {"text": _ENVELOPE_TEXT, "usage": {}, "cost_usd": 0.0,
            "is_error": False, "session_id": None}


_FILTER_BLOCK = (
    '"filter":{"must_have_keywords":['
    '{"term":"Generative AI","aliases":["GenAI"]},'
    '{"term":"Kubernetes","aliases":["K8s"]},'
    '{"term":"roadmap","aliases":[]},'
    '{"term":"missingskill","aliases":[]}],'
    '"title_variants":["Director of AI Product"],'
    '"title_alignment":"close",'
    '"knockouts":[{"requirement":"8+ years product management","verdict":"met"},'
    '{"requirement":"US work authorization","verdict":"unclear"}]}'
)
_ENVELOPE_WITH_FILTER = _ENVELOPE_TEXT[:-1] + "," + _FILTER_BLOCK + "}"


def _fake_run_filter(*_args, **_kwargs):
    return {"text": _ENVELOPE_WITH_FILTER, "usage": {}, "cost_usd": 0.0,
            "is_error": False, "session_id": None}


# Stage-2 judge envelope, matching _FILTER_BLOCK's four must-haves (Generative
# AI, Kubernetes, roadmap, missingskill) and two knockouts (8+ years product
# management, US work authorization) verbatim so parse_judge_response's
# term/requirement alignment finds every entry.
_JUDGE_ENVELOPE_TEXT = json.dumps({
    "must_haves": [
        {"term": "Generative AI", "verdict": "explicit",
         "evidence": "Directed Example Corp's first B2E Generative AI integration"},
        {"term": "Kubernetes", "verdict": "evidenced",
         "evidence": "Led platform migration onto Kubernetes-based infrastructure"},
        {"term": "roadmap", "verdict": "evidenced",
         "evidence": "Owned the AI product roadmap end to end"},
        {"term": "missingskill", "verdict": "absent", "evidence": None},
    ],
    "knockouts": [
        {"requirement": "8+ years product management", "verdict": "met",
         "reason": "8.7 years total PM experience",
         "required_years": 8, "candidate_years": 8.7},
        {"requirement": "US work authorization", "verdict": "met",
         "reason": "US citizen, no sponsorship required",
         "required_years": None, "candidate_years": None},
    ],
    "title_claim": "close",
})


def _fake_run_two_stage(prompt, **kwargs):
    """Stage-1 scoring envelope or stage-2 judge envelope, by system prompt."""
    if "screening judge" in (kwargs.get("system_prompt") or ""):
        return {"text": _JUDGE_ENVELOPE_TEXT, "usage": {}, "cost_usd": 0.0,
                "is_error": False, "session_id": None}
    return {"text": _ENVELOPE_WITH_FILTER, "usage": {}, "cost_usd": 0.0,
            "is_error": False, "session_id": None}


def _row_for_scoring(**overrides):
    row = {"id": 7, "title": "Director of AI Product", "company": "Acme",
           "location": "Remote (US)", "salary_min": None, "salary_max": None,
           "description": "We build AI/ML LLM products.", "score": 50}
    row.update(overrides)
    return row


def test_parse_filter_block():
    s = LLMScorer(db_path=":memory:")
    (*_, filter_raw) = s._parse_llm_response(_ENVELOPE_WITH_FILTER)
    assert len(filter_raw["must_have_keywords"]) == 4
    assert filter_raw["must_have_keywords"][0] == {
        "term": "Generative AI", "aliases": ["GenAI"]}
    assert filter_raw["title_alignment"] == "close"
    assert filter_raw["knockouts"][1]["verdict"] == "unclear"


def test_parse_missing_filter_block_is_none_but_score_survives():
    s = LLMScorer(db_path=":memory:")
    score, *_, dims, filter_raw = s._parse_llm_response(_ENVELOPE_TEXT)
    assert score == 91
    assert filter_raw is None


def test_parse_malformed_filter_block_is_none(caplog):
    bad = _ENVELOPE_TEXT[:-1] + ',"filter":"not an object"}'
    s = LLMScorer(db_path=":memory:")
    import logging
    with caplog.at_level(logging.WARNING):
        *_, filter_raw = s._parse_llm_response(bad)
    assert filter_raw is None
    assert any("parse failure" in r.message.lower() for r in caplog.records)


def test_parse_filter_block_drops_null_term_and_requirement():
    """A present-but-null 'term'/'requirement' must be dropped, not kept as 'None'."""
    data = {"filter": {
        "must_have_keywords": [
            {"term": None, "aliases": ["x"]},
            {"term": "RealSkill", "aliases": []},
        ],
        "title_variants": [],
        "title_alignment": "none",
        "knockouts": [
            {"requirement": None, "verdict": "met"},
            {"requirement": "Real requirement", "verdict": "met"},
        ],
    }}
    s = LLMScorer(db_path=":memory:")
    out = s._parse_filter_block(data)
    assert [m["term"] for m in out["must_have_keywords"]] == ["RealSkill"]
    assert [k["requirement"] for k in out["knockouts"]] == ["Real requirement"]
    # The null entries must NOT survive as the literal string "None".
    assert "None" not in [m["term"] for m in out["must_have_keywords"]]
    assert "None" not in [k["requirement"] for k in out["knockouts"]]


def test_score_one_job_computes_filter_fields():
    s = LLMScorer(db_path=":memory:", model="claude-sonnet-5")
    s.judge_basis_text, s.judge_basis_sha, s.judge_basis = (
        "INVENTORY TEXT", "sha1" * 16, "inventory")
    row = _row_for_scoring()
    with mock.patch.object(scorer_mod, "run_claude",
                           side_effect=_fake_run_two_stage) as m:
        result = s._score_one_job(row)
    filter_fields = result[-1]
    fscore, fmaster, fsource, fknock, fjson = filter_fields
    # Judge fixture (_JUDGE_ENVELOPE_TEXT): Generative AI=explicit,
    # Kubernetes=evidenced, roadmap=evidenced, missingskill=absent -> 3/4
    # credited. coverage = 75 * 3/4 = 56.25. title_claim=close -> +5.
    # Both knockouts "met" -> 0 unclear/failed -> +15.
    # uncapped = round(56.25 + 5 + 15) = round(76.25) = 76; not knocked out.
    assert (fscore, fmaster, fsource, fknock) == (76, 76, "master", 0)
    detail = json.loads(fjson)
    assert detail["version"] == 2
    assert detail["inventory_sha256"] == "sha1" * 16
    assert [mh["verdict"] for mh in detail["must_haves"]] == [
        "explicit", "evidenced", "evidenced", "absent"]
    assert m.call_count == 2  # scoring + judge


def test_judge_call_sends_inventory_in_system_prompt(monkeypatch):
    monkeypatch.setattr(scorer_mod, "_JUDGE_MIN_SCORE", 0)  # gate never blocks
    s = LLMScorer(db_path=":memory:", model="claude-sonnet-5")
    s.judge_basis_text, s.judge_basis_sha, s.judge_basis = (
        "INVENTORY TEXT", "sha1" * 16, "inventory")
    captured = {}

    def _fake(prompt, **kwargs):
        sysp = kwargs.get("system_prompt") or ""
        if "screening judge" in sysp:
            captured["system"] = sysp
            captured["prompt"] = prompt
            return {"text": _JUDGE_ENVELOPE_TEXT, "usage": {}, "cost_usd": 0.0,
                    "is_error": False, "session_id": None}
        return _fake_run_two_stage(prompt, **kwargs)

    with mock.patch.object(scorer_mod, "run_claude", side_effect=_fake):
        s._score_one_job(_row_for_scoring())
    assert "INVENTORY TEXT" in captured["system"]         # static, cacheable
    assert "INVENTORY TEXT" not in captured["prompt"]     # user msg is job-only
    assert "REQUIREMENTS TO JUDGE" in captured["prompt"]


def test_judge_failure_leaves_filter_fields_null():
    def _judge_dies(prompt, **kwargs):
        if "screening judge" in (kwargs.get("system_prompt") or ""):
            raise scorer_mod.ClaudeCLIError("judge exploded")
        return _fake_run_two_stage(prompt, **kwargs)

    s = LLMScorer(db_path=":memory:", model="claude-sonnet-5")
    s.judge_basis_text, s.judge_basis_sha, s.judge_basis = (
        "INVENTORY TEXT", "x", "inventory")
    row = _row_for_scoring()
    with mock.patch.object(scorer_mod, "run_claude", side_effect=_judge_dies):
        result = s._score_one_job(row)
    assert result is not None            # fit score still written
    assert result[-1] is None            # all five filter columns stay NULL


def test_empty_must_haves_writes_sentinel(caplog):
    # A scored job whose LLM yields no usable filter data gets a SENTINEL, not
    # NULL: filter_score/master/knockout stay NULL (renders "—") but filter_json
    # is non-NULL ('{}') so the row leaves the --backfill-filter target set after
    # one attempt instead of being re-scored on every run. No judge call — an
    # extraction miss never reaches stage 2.
    empty = _ENVELOPE_TEXT[:-1] + ',"filter":{"must_have_keywords":[],' \
        '"title_variants":[],"title_alignment":"none","knockouts":[]}}'
    s = LLMScorer(db_path=":memory:", model="claude-sonnet-5")
    s.judge_basis_text, s.judge_basis_sha, s.judge_basis = (
        "INVENTORY TEXT", "sha1" * 16, "inventory")
    row = _row_for_scoring()
    with mock.patch.object(scorer_mod, "run_claude",
                           side_effect=lambda *a, **k: {"text": empty, "usage": {},
                           "cost_usd": 0.0, "is_error": False, "session_id": None}) as m:
        with caplog.at_level(logging.WARNING):
            result = s._score_one_job(row)
    assert result[-1] == (None, None, "none", None, "{}")
    assert any("extraction miss" in r.message.lower() for r in caplog.records)
    assert m.call_count == 1  # no judge call on extraction miss


def test_backfill_reattempts_extraction_miss_idempotently(tmp_path, monkeypatch):
    # The sentinel ('{}', not NULL) marks a row as "attempted" so a first-pass
    # scan can tell it apart from a genuinely-unscored row. Task 5 widens
    # backfill_filter's target SQL to ALSO match '{}' (so sentinels written
    # while the feature was disabled get a real shot once it's re-enabled —
    # see test_backfill_targets_sentinel_rows_too). There is no DB signal
    # that distinguishes a disabled-era sentinel from a genuine extraction
    # miss (both write the identical _FILTER_SENTINEL tuple), so a still-
    # empty row IS re-attempted on the next run — what must hold instead is
    # idempotence: repeated re-attempts keep writing the same sentinel, never
    # corrupt state, and cost stays bounded to prefilter-accepted candidates
    # each run (see backfill_filter's docstring).
    import main as main_mod
    db = tmp_path / "jobs.db"
    conn = main_mod.init_database(str(db))
    conn.execute(
        "INSERT INTO jobs (id, title, company, location, url, description, "
        "score, llm_score, status, profile) "
        "VALUES (9, 'Director of AI Product', 'Acme', 'Remote (US)', "
        "'https://x/9', ?, 50, 80, 'new', 'testuser')",
        ("We build AI/ML LLM products for enterprise GenAI roadmaps. " * 10,))
    conn.commit()
    conn.close()

    empty = _ENVELOPE_TEXT[:-1] + ',"filter":{"must_have_keywords":[],' \
        '"title_variants":[],"title_alignment":"none","knockouts":[]}}'
    empty_run = lambda *a, **k: {"text": empty, "usage": {}, "cost_usd": 0.0,
                                 "is_error": False, "session_id": None}
    s = LLMScorer(db_path=str(db), model="claude-sonnet-5")
    s.judge_basis_text, s.judge_basis_sha, s.judge_basis = (
        "INVENTORY TEXT", "sha1" * 16, "inventory")
    monkeypatch.setattr(scorer_mod, "is_cli_available", lambda: True)

    # First run: writes the sentinel, so filter_json is no longer NULL.
    with mock.patch.object(scorer_mod, "run_claude", side_effect=empty_run) as m1:
        s.backfill_filter(profile="testuser", workers=1)
    assert m1.call_count == 1
    conn = sqlite3.connect(str(db))
    row = conn.execute(
        "SELECT filter_score, filter_source, filter_json FROM jobs WHERE id = 9"
    ).fetchone()
    conn.close()
    assert row == (None, "none", "{}")

    # Second run: still filter_json = '{}', so the widened target SQL picks
    # it up again — but the re-attempt is idempotent: same sentinel, no
    # corruption, no growth.
    with mock.patch.object(scorer_mod, "run_claude", side_effect=empty_run) as m2:
        count2 = s.backfill_filter(profile="testuser", workers=1)
    assert m2.call_count == 1
    assert count2 == 1
    conn = sqlite3.connect(str(db))
    row2 = conn.execute(
        "SELECT filter_score, filter_source, filter_json FROM jobs WHERE id = 9"
    ).fetchone()
    conn.close()
    assert row2 == (None, "none", "{}")


def test_apply_writes_filter_columns(tmp_path, monkeypatch):
    import main as main_mod
    db = tmp_path / "jobs.db"
    conn = main_mod.init_database(str(db))
    conn.execute(
        "INSERT INTO jobs (id, title, company, location, url, description, score, "
        "status, profile) VALUES (7, 'Director of AI Product', 'Acme', "
        "'Remote (US)', 'https://x/7', ?, 50, 'new', 'testuser')",
        ("We build AI/ML LLM products for enterprise GenAI roadmaps. " * 10,),
    )
    conn.commit()
    conn.close()

    s = LLMScorer(db_path=str(db), model="claude-sonnet-5")
    s.judge_basis_text, s.judge_basis_sha, s.judge_basis = (
        "INVENTORY TEXT", "sha1" * 16, "inventory")
    monkeypatch.setattr(scorer_mod, "is_cli_available", lambda: True)
    with mock.patch.object(scorer_mod, "run_claude", side_effect=_fake_run_two_stage):
        s.apply_llm_scores_to_db(profile="testuser", workers=1)

    conn = sqlite3.connect(str(db))
    row = conn.execute(
        "SELECT filter_score, filter_score_master, filter_source, "
        "filter_knockout, filter_json FROM jobs WHERE id = 7").fetchone()
    conn.close()
    assert row[0] == 76 and row[1] == 76 and row[2] == "master" and row[3] == 0
    assert row[4] is not None
    assert json.loads(row[4])["version"] == 2


def test_call_claude_parses_dimensions():
    s = LLMScorer(db_path=":memory:", model="claude-sonnet-4-6")
    with mock.patch.object(scorer_mod, "run_claude", side_effect=_fake_run) as m:
        score, expl, smin, smax, emin, emax, dims, filter_raw = s._call_claude(
            "Director of AI Product", "Acme",
            "We build AI/ML LLM products.", location="Remote (US)",
        )
    assert dims["role_match"] == 28
    assert score == 91
    assert (emin, emax) == (280000.0, 360000.0)
    m.assert_called_once()


def test_is_available_checks_cli_binary(monkeypatch):
    monkeypatch.setattr(scorer_mod, "is_cli_available", lambda: True)
    s = LLMScorer(db_path=":memory:")
    assert s.is_available() is True

    monkeypatch.setattr(scorer_mod, "is_cli_available", lambda: False)
    s2 = LLMScorer(db_path=":memory:")
    assert s2.is_available() is False


def test_high_scorer_makes_single_call_no_self_consistency():
    s = LLMScorer(db_path=":memory:", model="claude-sonnet-4-6")
    row = {
        "id": 1, "title": "Director of AI Product", "company": "Acme",
        "location": "Remote (US)",
        "description": "We build AI/ML LLM products. Comp $230K-$260K.",
        "salary_min": None, "salary_max": None, "score": 80,
    }
    with mock.patch.object(scorer_mod, "run_claude", side_effect=_fake_run) as m:
        result = s._score_one_job(row)
    assert result is not None
    assert m.call_count == 1  # double-call removed


def test_fit_prompt_calibrates_local_comp():
    """Local jobs are scored against the local comp bar: the calibration
    paragraph survives assembly, and its dollar figure tracks the config."""
    import profile_policy
    from engine.llm_scorer import _FIT_SYSTEM_TEMPLATE, _usd_k
    assert "LOCAL COMP CALIBRATION" in _FIT_SYSTEM_TEMPLATE
    assert _usd_k(profile_policy.LOCAL_FULL_COMP) in _FIT_SYSTEM_TEMPLATE
    assert profile_policy._DEFAULTS["policy.comp.local_full"] == 150_000


def test_fit_templates_split_static_from_volatile():
    """Static rubric lives in the system template (the prompt-cache prefix);
    the user template carries ONLY the volatile job posting."""
    from engine.llm_scorer import _FIT_SYSTEM_TEMPLATE, _FIT_USER_TEMPLATE
    assert "SCORING RUBRIC" in _FIT_SYSTEM_TEMPLATE
    assert "FILTER EXTRACTION" in _FIT_SYSTEM_TEMPLATE
    assert "{resume}" in _FIT_SYSTEM_TEMPLATE
    assert "{description}" not in _FIT_SYSTEM_TEMPLATE
    assert "{title}" not in _FIT_SYSTEM_TEMPLATE
    assert "{description}" in _FIT_USER_TEMPLATE
    assert "SCORING RUBRIC" not in _FIT_USER_TEMPLATE


def test_call_claude_sends_rubric_in_system_prompt():
    s = LLMScorer(db_path=":memory:", model="claude-sonnet-5")
    captured = {}

    def _fake(prompt, **kwargs):
        captured["prompt"] = prompt
        captured["system"] = kwargs.get("system_prompt") or ""
        return {"text": _ENVELOPE_TEXT, "usage": {}, "cost_usd": 0.0,
                "is_error": False, "session_id": None}

    with mock.patch.object(scorer_mod, "run_claude", side_effect=_fake):
        s._call_claude("Director of AI Product", "Acme",
                       "We build LLM products.", location="Remote (US)")
    assert "SCORING RUBRIC" in captured["system"]
    assert scorer_mod.RESUME_SUMMARY in captured["system"]        # profile is static context
    assert "SCORING RUBRIC" not in captured["prompt"]     # user msg is job-only
    assert "We build LLM products." in captured["prompt"]
    assert "Director of AI Product" in captured["prompt"]


def test_backfill_filter_targets_and_converges(tmp_path, monkeypatch):
    # Job 4 ("Software Engineer") is skipped only because the engineering-title
    # prefilter caps it at 15; that gate is empty in a neutral tree, so pin the
    # PM-shaped prefilter to keep the target set == {job 1}.
    patch_pm_prefilter(monkeypatch)
    import main as main_mod
    db = tmp_path / "jobs.db"
    conn = main_mod.init_database(str(db))
    good_desc = "We build AI/ML LLM products for enterprise GenAI roadmaps. " * 10
    rows = [
        # (id, title, llm_score, filter_json, status, description)
        (1, "Director of AI Product", 80, None, "new", good_desc),      # target
        (2, "Director of AI Product", 80, '{"x":1}', "new", good_desc), # has filter → skip
        (3, "Director of AI Product", 80, None, "expired", good_desc),  # terminal → skip
        (4, "Software Engineer", 10, None, "new", good_desc),           # prefilter reject → skip
        (5, "Director of AI Product", 80, None, "new", "short"),        # no real desc → skip
    ]
    for (jid, title, llm, fj, status, desc) in rows:
        conn.execute(
            "INSERT INTO jobs (id, title, company, location, url, description, "
            "score, llm_score, filter_json, status, profile) "
            "VALUES (?, ?, 'Acme', 'Remote (US)', ?, ?, 50, ?, ?, ?, 'testuser')",
            (jid, title, f"https://x/{jid}", desc, llm, fj, status))
    conn.commit()
    conn.close()

    s = LLMScorer(db_path=str(db), model="claude-sonnet-5")
    s.judge_basis_text, s.judge_basis_sha, s.judge_basis = (
        "INVENTORY TEXT", "sha1" * 16, "inventory")
    monkeypatch.setattr(scorer_mod, "is_cli_available", lambda: True)
    with mock.patch.object(scorer_mod, "run_claude", side_effect=_fake_run_two_stage) as m:
        count = s.backfill_filter(profile="testuser", workers=1)

    assert count == 1
    assert m.call_count == 2  # job 1: scoring + judge
    conn = sqlite3.connect(str(db))
    got = dict(conn.execute(
        "SELECT id, filter_score FROM jobs WHERE filter_score IS NOT NULL"))
    conn.close()
    assert set(got) == {1}


# ---------------------------------------------------------------------------
# rejudge_filter + backfill sentinel expansion
# ---------------------------------------------------------------------------

def _v2_blob(sha="oldsha"):
    """Stored filter_json v2 blob for rejudge tests.

    The must-have terms and the knockout requirement string are copied
    VERBATIM from _JUDGE_ENVELOPE_TEXT (Generative AI / Kubernetes / roadmap
    / missingskill; "8+ years product management") because
    parse_judge_response aligns the judge's verdicts to the input
    term/requirement strings by normalized-string match — a rejudge only
    reproduces Task 4's fixture math (filter_score 76, see
    test_score_one_job_computes_filter_fields for the derivation) if the
    reconstructed filter_raw asks about the exact same terms the fixture
    judges. The stored verdicts here are irrelevant busywork: rejudge_filter
    discards them and lets the judge re-decide.
    """
    return json.dumps({
        "version": 2,
        "must_haves": [
            {"term": t, "aliases": [], "verdict": "absent", "evidence": ""}
            for t in ("Generative AI", "Kubernetes", "roadmap", "missingskill")
        ],
        "title_variants": ["Director of AI Product"],
        "title_alignment": "close",
        "title_claim": "none",
        "knockouts": [{"requirement": "8+ years product management",
                       "verdict": "unclear", "reason": ""}],
        "uncapped_score": 10, "inventory_sha256": sha, "basis": "inventory",
        "judged_at": "2026-07-09T08:00:00",
    })


def _seed_rejudge_db(tmp_path):
    import main as main_mod
    db = tmp_path / "jobs.db"
    conn = main_mod.init_database(str(db))
    rows = [
        # id 1: stale sha -> re-judged
        ("https://x/1", "new", _v2_blob("oldsha"), "master"),
        # id 2: current sha -> skipped unless force_all
        ("https://x/2", "new", _v2_blob("cursha"), "master"),
        # id 3: tailored -> never re-judged (frozen realized score)
        ("https://x/3", "new", _v2_blob("oldsha"), "tailored"),
        # id 4: sentinel row -> not a rejudge target (backfill's job)
        ("https://x/4", "new", "{}", None),
        # id 5: terminal status -> skipped
        ("https://x/5", "applied", _v2_blob("oldsha"), "master"),
    ]
    for url, status, fjson, fsource in rows:
        conn.execute(
            "INSERT INTO jobs (title, company, url, description, score, "
            "status, profile, llm_score, filter_json, filter_source) "
            "VALUES ('PM', 'Co', ?, ?, 50, ?, 'testuser', 70, ?, ?)",
            (url, "desc " * 60, status, fjson, fsource))
    conn.commit()
    conn.close()
    return db


def test_rejudge_targets_stale_master_rows_only(tmp_path, monkeypatch):
    db = _seed_rejudge_db(tmp_path)
    s = LLMScorer(db_path=str(db))
    s.judge_basis_text, s.judge_basis_sha, s.judge_basis = ("INV", "cursha", "inventory")
    monkeypatch.setattr(scorer_mod, "is_cli_available", lambda: True)
    with mock.patch.object(scorer_mod, "run_claude",
                           side_effect=_fake_run_two_stage) as m:
        count = s.rejudge_filter(profile="testuser")
    assert count == 1
    assert m.call_count == 1  # judge call only — no fit re-scoring
    conn = sqlite3.connect(str(db)); conn.row_factory = sqlite3.Row
    r1 = conn.execute("SELECT * FROM jobs WHERE url='https://x/1'").fetchone()
    # Same fixture math as test_score_one_job_computes_filter_fields: 3/4
    # must-haves credited (56.25) + title_claim close (+5) + both knockouts
    # clean (+15) -> round(76.25) = 76, not knocked out.
    assert r1["filter_score"] == 76
    assert r1["llm_score"] == 70             # untouched — stage 2 only
    assert json.loads(r1["filter_json"])["inventory_sha256"] == "cursha"
    r2 = conn.execute("SELECT filter_json FROM jobs WHERE url='https://x/2'").fetchone()
    assert json.loads(r2["filter_json"])["inventory_sha256"] == "cursha"  # skipped, unchanged sha
    r3 = conn.execute("SELECT filter_source FROM jobs WHERE url='https://x/3'").fetchone()
    assert r3["filter_source"] == "tailored"
    conn.close()


def test_rejudge_force_all_ignores_hash(tmp_path, monkeypatch):
    db = _seed_rejudge_db(tmp_path)
    s = LLMScorer(db_path=str(db))
    s.judge_basis_text, s.judge_basis_sha, s.judge_basis = ("INV", "cursha", "inventory")
    monkeypatch.setattr(scorer_mod, "is_cli_available", lambda: True)
    with mock.patch.object(scorer_mod, "run_claude",
                           side_effect=_fake_run_two_stage):
        count = s.rejudge_filter(profile="testuser", force_all=True)
    assert count == 2  # ids 1 and 2 — still not tailored/sentinel/terminal


def test_rejudge_no_basis_returns_zero(tmp_path, monkeypatch):
    db = _seed_rejudge_db(tmp_path)
    s = LLMScorer(db_path=str(db))
    s.judge_basis_text, s.judge_basis_sha, s.judge_basis = ("", "", "none")
    monkeypatch.setattr(scorer_mod, "is_cli_available", lambda: True)
    with mock.patch.object(scorer_mod, "run_claude",
                           side_effect=_fake_run_two_stage) as m:
        count = s.rejudge_filter(profile="testuser")
    assert count == 0
    assert m.call_count == 0


def test_backfill_targets_sentinel_rows_too(tmp_path, monkeypatch):
    import main as main_mod
    db = tmp_path / "jobs.db"
    conn = main_mod.init_database(str(db))
    conn.execute(
        "INSERT INTO jobs (title, company, url, description, score, status, "
        "profile, llm_score, filter_json) "
        "VALUES ('PM', 'Co', 'https://x/s', ?, 50, 'new', 'testuser', 70, '{}')",
        ("desc " * 60,))
    conn.commit(); conn.close()
    s = LLMScorer(db_path=str(db))
    s.judge_basis_text, s.judge_basis_sha, s.judge_basis = ("INV", "cursha", "inventory")
    monkeypatch.setattr(scorer_mod, "is_cli_available", lambda: True)
    with mock.patch.object(scorer_mod, "run_claude",
                           side_effect=_fake_run_two_stage):
        count = s.backfill_filter(profile="testuser")
    assert count == 1
    conn = sqlite3.connect(str(db))
    fs = conn.execute("SELECT filter_score FROM jobs WHERE url='https://x/s'").fetchone()[0]
    conn.close()
    assert fs == 76


def test_rejudge_since_hours_limits_to_recent(tmp_path, monkeypatch):
    import main as main_mod
    db = tmp_path / "jobs.db"
    conn = main_mod.init_database(str(db))
    # id 1: stale sha, recent created_at -> inside the 48h window, re-judged
    conn.execute(
        "INSERT INTO jobs (title, company, url, description, score, "
        "status, profile, llm_score, filter_json, filter_source, created_at) "
        "VALUES ('PM', 'Co', 'https://x/1', ?, 50, 'new', 'testuser', 70, ?, "
        "'master', datetime('now'))",
        ("desc " * 60, _v2_blob("oldsha")))
    # id 2: stale sha, but created 5 days ago -> outside the window, skipped
    conn.execute(
        "INSERT INTO jobs (title, company, url, description, score, "
        "status, profile, llm_score, filter_json, filter_source, created_at) "
        "VALUES ('PM', 'Co', 'https://x/2', ?, 50, 'new', 'testuser', 70, ?, "
        "'master', datetime('now','-5 days'))",
        ("desc " * 60, _v2_blob("oldsha")))
    conn.commit()
    conn.close()

    s = LLMScorer(db_path=str(db))
    s.judge_basis_text, s.judge_basis_sha, s.judge_basis = ("INV", "cursha", "inventory")
    monkeypatch.setattr(scorer_mod, "is_cli_available", lambda: True)
    with mock.patch.object(scorer_mod, "run_claude",
                           side_effect=_fake_run_two_stage) as m:
        count = s.rejudge_filter(profile="testuser", since_hours=48)
    assert count == 1
    assert m.call_count == 1  # only the recent row's judge call

    conn = sqlite3.connect(str(db)); conn.row_factory = sqlite3.Row
    r1 = conn.execute("SELECT filter_json FROM jobs WHERE url='https://x/1'").fetchone()
    assert json.loads(r1["filter_json"])["inventory_sha256"] == "cursha"  # re-judged
    r2 = conn.execute("SELECT filter_json FROM jobs WHERE url='https://x/2'").fetchone()
    assert json.loads(r2["filter_json"])["inventory_sha256"] == "oldsha"  # untouched, still stale
    conn.close()


def test_backfill_since_hours_limits_to_recent(tmp_path, monkeypatch):
    import main as main_mod
    db = tmp_path / "jobs.db"
    conn = main_mod.init_database(str(db))
    good_desc = "We build AI/ML LLM products for enterprise GenAI roadmaps. " * 10
    # id 1: sentinel/NULL row, recent created_at -> inside the 48h window
    conn.execute(
        "INSERT INTO jobs (title, company, location, url, description, "
        "score, status, profile, created_at) "
        "VALUES ('Director of AI Product', 'Acme', 'Remote (US)', "
        "'https://x/1', ?, 50, 'new', 'testuser', datetime('now'))",
        (good_desc,))
    # id 2: sentinel/NULL row, created 5 days ago -> outside the window
    conn.execute(
        "INSERT INTO jobs (title, company, location, url, description, "
        "score, status, profile, created_at) "
        "VALUES ('Director of AI Product', 'Acme', 'Remote (US)', "
        "'https://x/2', ?, 50, 'new', 'testuser', datetime('now','-5 days'))",
        (good_desc,))
    conn.commit()
    conn.close()

    s = LLMScorer(db_path=str(db), model="claude-sonnet-5")
    s.judge_basis_text, s.judge_basis_sha, s.judge_basis = (
        "INVENTORY TEXT", "sha1" * 16, "inventory")
    monkeypatch.setattr(scorer_mod, "is_cli_available", lambda: True)
    with mock.patch.object(scorer_mod, "run_claude", side_effect=_fake_run_two_stage):
        count = s.backfill_filter(profile="testuser", since_hours=48, workers=1)

    assert count == 1
    conn = sqlite3.connect(str(db))
    got = dict(conn.execute(
        "SELECT url, filter_score FROM jobs WHERE filter_score IS NOT NULL"))
    conn.close()
    assert set(got) == {"https://x/1"}


def test_missing_resume_summary_fails_loudly(monkeypatch):
    monkeypatch.setattr(scorer_mod, "RESUME_SUMMARY", None)
    s = LLMScorer(db_path=":memory:", model="claude-sonnet-5")
    with mock.patch.object(scorer_mod, "run_claude") as rc:
        with pytest.raises(RuntimeError, match="resume_summary"):
            s._call_claude("T", "C", "D", location="Remote")
        rc.assert_not_called()


def test_fit_system_template_byte_identical_to_extraction_baseline():
    """Prompt bytes ARE behavior: drift re-rolls a non-deterministic judge
    (~55% row churn precedent) and breaks the claude CLI prompt-cache prefix.
    Owner-scoped: declare your rubric pin in policy.owner_pins.rubric_sha and
    this guard asserts the assembled template hashes to it; leave the pin empty
    (the neutral/public default) and the guard self-skips — a personalized fork
    that merely overrides a rubric block no longer sees a red suite against a
    sha it can never match. On deliberate rubric changes update the pin in
    policy.owner_pins.rubric_sha IN THE SAME COMMIT and record the re-judge
    decision in the commit message (surgical ids-scoped re-judge, or an explicit
    accept-drift-on-future-rows note)."""
    import hashlib
    import profile_policy
    pin = profile_policy.OWNER_RUBRIC_SHA
    if not pin:
        import pytest
        pytest.skip("no owner rubric sha pin (policy.owner_pins.rubric_sha empty)")
    from engine.llm_scorer import _FIT_SYSTEM_TEMPLATE
    assert hashlib.sha256(_FIT_SYSTEM_TEMPLATE.encode("utf-8")).hexdigest() == pin


def test_default_rubric_blocks_pinned():
    """Pin the NEUTRAL rubric blocks from the defaults table (blocks only, no
    value tokens — value tokens resolve from AMBIENT config and would give a
    different sha per tree). Catches generic-prose drift in both trees' CI.
    On deliberate default changes, update the sha in the same commit."""
    import hashlib
    import profile_policy as pp
    from engine.llm_scorer import assemble_fit_template
    joined = "\n\n".join(pp._DEFAULTS[k] for k in (
        "policy.rubric.role_match_block",
        "policy.rubric.seniority_match_block",
        "policy.rubric.remote_location_block",
        "policy.rubric.domain_fit_block",
    ))
    assert hashlib.sha256(joined.encode("utf-8")).hexdigest() == "bc21dda95fadbb4865c5fb983a46cda61a5727c9f0a4fc16e63ea0aa5b51a893"
    # And the assembled render (whatever tree we're in) must keep the frozen
    # JSON schema key that generic prose is required to preserve:
    assert "ai_domain_fit" in assemble_fit_template()


def test_prefilter_neutral_defaults_defer_to_llm(monkeypatch):
    """Public-tree condition: all prefilter patterns None → no caps fire."""
    import engine.llm_scorer as m
    for attr in ("_NON_PM_TITLES", "_SALES_BD_TITLES", "_SOLUTIONS_CS_TITLES",
                 "_ENG_TITLE_KEYWORDS", "_NON_PM_ADJACENT_TITLES", "_PM_KEYWORDS"):
        monkeypatch.setattr(m, attr, None)
    cap, reason = m.prefilter_job("Staff Accountant", "Remote", "Ledger work.")
    assert cap is None and reason is None
