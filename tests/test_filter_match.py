"""Pure unit tests for filter_match — no network, no Google, no LLM."""
import json

import pytest

import filter_match as fm


# ---------------------------------------------------------------------------
# phrase_present — token boundaries
# ---------------------------------------------------------------------------

def test_phrase_present_whole_word_only():
    assert fm.phrase_present("I maintain JavaScript systems", "AI") is False
    assert fm.phrase_present("I maintain JavaScript systems", "Java") is False
    assert fm.phrase_present("Skills: R, Go, C++, .NET", "C++") is True
    assert fm.phrase_present("Skills: R, Go, C++, .NET", ".NET") is True
    assert fm.phrase_present("Skills: R, Go, C++", "R") is True

def test_phrase_present_case_insensitive():
    assert fm.phrase_present("built LLM products", "llm") is True
    assert fm.phrase_present("built llm products", "LLM") is True

def test_phrase_present_trailing_s_tolerance():
    # Singular extraction matches plural resume text.
    assert fm.phrase_present("shipped LLMs to production", "LLM") is True
    # Multi-word phrases too.
    assert fm.phrase_present("ran A/B tests weekly", "A/B test") is True


def test_phrase_present_standalone_ampersand_equals_and():
    # Resume register writes "&" where the stored term says "and" (the
    # 2026-07-17 incident: "Experimentation & Funnel Analysis" vs
    # "experimentation and funnel analysis"). A standalone "&" token folds
    # to "and" on both sides.
    assert fm.phrase_present(
        "Experimentation & Funnel Analysis: A/B tested flows",
        "experimentation and funnel analysis") is True
    # Reverse direction: term written with "&", text written with "and".
    assert fm.phrase_present(
        "led research and development for the platform",
        "Research & Development") is True

def test_phrase_present_intraword_ampersand_untouched():
    # "&" inside a token is part of the token, never folded.
    assert fm.phrase_present("worked on AT&T network ops", "AT&T") is True
    assert fm.phrase_present("worked on AT&T network ops", "ATandT") is False
    assert fm.phrase_present("ran the R&D team", "R&D") is True
    assert fm.phrase_present("hosted a Q&A session", "Q&A") is True


# ---------------------------------------------------------------------------
# keyword_matches — moved from ats_checker, same semantics + re-export
# ---------------------------------------------------------------------------

def test_keyword_matches_order_preserved():
    matched = fm.keyword_matches("Python and SQL and dbt", ["SQL", "Python", "Spark"])
    assert matched == ["SQL", "Python"]

def test_ats_checker_reexports_same_function():
    from resume_tailor.ats_checker import keyword_matches as ats_km
    assert ats_km is fm.keyword_matches


# ---------------------------------------------------------------------------
# evaluate_must_haves — aliases
# ---------------------------------------------------------------------------

def test_evaluate_must_haves_matches_via_alias():
    text = "Led generative artificial intelligence initiatives"
    items = [{"term": "Generative AI", "aliases": ["GenAI", "generative artificial intelligence"]}]
    out = fm.evaluate_must_haves(text, items)
    assert out == [{"term": "Generative AI",
                    "aliases": ["GenAI", "generative artificial intelligence"],
                    "present": True}]

def test_evaluate_must_haves_ampersand_fold_regression():
    # Pin of the 2026-07-17 auto-tailor miss: the tailored doc literally said
    # "Experimentation & Funnel Analysis" yet the recompute scored the term
    # absent, dropping Filter Match 7 points under the judged ceiling.
    text = ("Experimentation & Funnel Analysis: A/B tested conversational "
            "flows for conversion, driving a 20% usage surge (2022) at "
            "industry-leading containment and completion rates.")
    out = fm.evaluate_must_haves(text, [
        {"term": "experimentation and funnel analysis",
         "aliases": ["A/B testing", "funnel optimization"]},
    ])
    assert out[0]["present"] is True


def test_evaluate_must_haves_absent_and_skips_blank_terms():
    out = fm.evaluate_must_haves("nothing here", [
        {"term": "Kubernetes", "aliases": ["K8s"]},
        {"term": "  ", "aliases": []},          # blank term dropped
        {"term": "SQL"},                          # missing aliases key tolerated
    ])
    assert out == [
        {"term": "Kubernetes", "aliases": ["K8s"], "present": False},
        {"term": "SQL", "aliases": [], "present": False},
    ]


# ---------------------------------------------------------------------------
# resolve_title_tier
# ---------------------------------------------------------------------------

def test_title_tier_exact_when_variant_in_text():
    tier = fm.resolve_title_tier(
        "Seasoned Staff PM with platform scars", "Staff Product Manager",
        ["Staff PM"], "none")
    assert tier == "exact"

def test_title_tier_close_from_llm_alignment():
    tier = fm.resolve_title_tier(
        "Senior Product Manager at Acme", "Principal Product Lead", [], "close")
    assert tier == "close"
    # LLM saying "exact" without literal presence still caps at close.
    tier2 = fm.resolve_title_tier(
        "Senior Product Manager at Acme", "Principal Product Lead", [], "exact")
    assert tier2 == "close"

def test_title_tier_none():
    assert fm.resolve_title_tier("engineer", "Product Manager", [], "none") == "none"


# ---------------------------------------------------------------------------
# compute_filter_score
# ---------------------------------------------------------------------------

def _mh(n_present, n_total):
    return ([{"term": f"t{i}", "aliases": [], "present": True} for i in range(n_present)]
            + [{"term": f"m{i}", "aliases": [], "present": False}
               for i in range(n_total - n_present)])

def test_perfect_score():
    score, ko = fm.compute_filter_score(_mh(10, 10), "exact", [
        {"requirement": "5+ years PM", "verdict": "met"}])
    assert (score, ko) == (100, False)

def test_coverage_math():
    # 6/10 present → 45 coverage; close title → 5; no knockouts → vacuous 15. Total 65.
    score, ko = fm.compute_filter_score(_mh(6, 10), "close", [])
    assert (score, ko) == (65, False)

def test_failed_knockout_caps_at_15():
    score, ko = fm.compute_filter_score(_mh(10, 10), "exact", [
        {"requirement": "Onsite NYC", "verdict": "failed"}])
    assert (score, ko) == (15, True)

def test_unclear_knockouts_deduct_with_floor():
    # 4 unclear → 15 - 20 floors at 0. 10/10 coverage=75 + exact 10 + 0 = 85.
    kos = [{"requirement": f"r{i}", "verdict": "unclear"} for i in range(4)]
    score, ko = fm.compute_filter_score(_mh(10, 10), "exact", kos)
    assert (score, ko) == (85, False)

def test_empty_must_haves_raises():
    with pytest.raises(ValueError):
        fm.compute_filter_score([], "exact", [])


# ---------------------------------------------------------------------------
# build_filter_json
# ---------------------------------------------------------------------------

def test_build_filter_json_round_trips():
    evaluated = _mh(1, 2)
    raw = fm.build_filter_json(evaluated, ["Staff PM"], "exact",
                               [{"requirement": "US work auth", "verdict": "met"}],
                               "close")
    data = json.loads(raw)
    assert data["must_haves"] == evaluated
    assert data["title_variants"] == ["Staff PM"]
    assert data["title_tier"] == "exact"
    assert data["title_alignment"] == "close"
    assert data["knockouts"] == [{"requirement": "US work auth", "verdict": "met"}]
    assert "computed_at" in data


# ---------------------------------------------------------------------------
# v2: judged scoring math, effective score, v2 JSON
# ---------------------------------------------------------------------------
import sqlite3


def _jmh(n_explicit, n_evidenced, n_absent):
    out = []
    for verdict, n in (("explicit", n_explicit), ("evidenced", n_evidenced),
                       ("absent", n_absent)):
        out += [{"term": f"t{verdict}{i}", "aliases": [], "verdict": verdict,
                 "evidence": "e" if verdict != "absent" else ""}
                for i in range(n)]
    return out


def test_judged_score_credits_explicit_and_evidenced_equally():
    # 8 credited of 10 -> coverage 60; close title +5; clean knockouts +15.
    score, ko, uncapped = fm.compute_judged_filter_score(
        _jmh(4, 4, 2), "close", [{"requirement": "x", "verdict": "met"}])
    assert (score, ko, uncapped) == (80, False, 80)


def test_judged_score_failed_knockout_caps_but_keeps_uncapped():
    score, ko, uncapped = fm.compute_judged_filter_score(
        _jmh(10, 0, 0), "exact", [{"requirement": "x", "verdict": "failed"}])
    assert (score, ko, uncapped) == (fm.KNOCKOUT_CAP, True, 100)


def test_judged_score_unclear_deducts_five_each():
    score, ko, uncapped = fm.compute_judged_filter_score(
        _jmh(10, 0, 0), "none",
        [{"requirement": "a", "verdict": "unclear"},
         {"requirement": "b", "verdict": "unclear"}])
    assert (score, ko, uncapped) == (80, False, 80)  # 75 + 0 + (15-10)


def test_judged_score_unclear_knockouts_floor_at_zero():
    # 4 unclear -> 15 - 5*4 = -5 -> floored to 0; coverage 75, title none 0.
    score, ko, uncapped = fm.compute_judged_filter_score(
        _jmh(10, 0, 0), "none",
        [{"requirement": f"r{i}", "verdict": "unclear"} for i in range(4)])
    assert (score, ko, uncapped) == (75, False, 75)


def test_judged_score_empty_must_haves_raises():
    import pytest
    with pytest.raises(ValueError):
        fm.compute_judged_filter_score([], "none", [])


def test_effective_score_gates_only_knocked_out_high_scores():
    assert fm.effective_score(87, True) == fm.KNOCKOUT_GATE_CAP
    assert fm.effective_score(87, False) == 87
    assert fm.effective_score(35, True) == 35
    assert fm.effective_score(None, True) == 0


def test_effective_score_sql_matches_python():
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE jobs (score INTEGER, filter_knockout INTEGER)")
    rows = [(87, 1), (87, 0), (35, 1), (60, None), (None, 1), (None, 0), (None, None)]
    conn.executemany("INSERT INTO jobs VALUES (?, ?)", rows)
    got = [r[0] for r in conn.execute(
        f"SELECT {fm.EFFECTIVE_SCORE_SQL} FROM jobs").fetchall()]
    assert got == [fm.effective_score(s, bool(k)) for s, k in rows]


def test_build_filter_json_v2_round_trips():
    blob = fm.build_filter_json_v2(
        _jmh(1, 1, 1), ["Staff PM"], "close", "close",
        [{"requirement": "8+ years", "verdict": "met", "reason": "8.7y",
          "required_years": 8.0, "candidate_years": 8.7}],
        90, "abc123", "inventory")
    data = json.loads(blob)
    assert data["version"] == 2
    assert data["title_claim"] == "close"
    assert data["title_alignment"] == "close"
    assert data["uncapped_score"] == 90
    assert data["inventory_sha256"] == "abc123"
    assert data["basis"] == "inventory"
    assert data["must_haves"][0]["verdict"] == "explicit"
    assert data["must_haves"][0]["evidence"] == "e"
    assert data["knockouts"][0]["reason"] == "8.7y"
    assert "judged_at" in data
    # Post-tailor literal recompute (resume_tailor/pipeline.py) rebuilds
    # {term, aliases} from this blob — both keys must survive.
    assert all("term" in m and "aliases" in m for m in data["must_haves"])
