"""Pure unit tests for filter_judge — no network, no CLI."""
import json

import filter_judge as fj

# ---------------------------------------------------------------------------
# load_inventory
# ---------------------------------------------------------------------------

def test_load_inventory_prefers_inventory_file(tmp_path, monkeypatch):
    inv = tmp_path / "experience_inventory.md"
    inv.write_text("## HARD FACTS\n- Total: 8.7 years\n", encoding="utf-8")
    monkeypatch.setattr(fj, "INVENTORY_PATH", str(inv))
    text, sha, basis = fj.load_inventory()
    assert "8.7 years" in text
    assert len(sha) == 64
    assert basis == "inventory"


def test_load_inventory_falls_back_to_master_resume(tmp_path, monkeypatch):
    resume = tmp_path / "master_resume.txt"
    resume.write_text("JANE DOE resume text", encoding="utf-8")
    monkeypatch.setattr(fj, "INVENTORY_PATH", str(tmp_path / "missing.md"))
    monkeypatch.setattr(fj, "MASTER_RESUME_CACHE", str(resume))
    text, sha, basis = fj.load_inventory()
    assert "resume text" in text
    assert basis == "resume_fallback"


def test_load_inventory_none_when_both_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(fj, "INVENTORY_PATH", str(tmp_path / "a.md"))
    monkeypatch.setattr(fj, "MASTER_RESUME_CACHE", str(tmp_path / "b.txt"))
    text, sha, basis = fj.load_inventory()
    assert (text, sha, basis) == ("", "", "none")

# ---------------------------------------------------------------------------
# build_judge_prompt
# ---------------------------------------------------------------------------

_MHS = [
    {"term": "RAG", "aliases": ["retrieval-augmented generation"]},
    {"term": "product management", "aliases": []},
]
_KOS = ["8+ years product management experience", "Onsite in Tarrytown, NY"]


def test_prompt_contains_requirements_knockouts_and_title_only():
    p = fj.build_judge_prompt("Director of Product", ["Product Director"],
                              _MHS, _KOS)
    assert "Director of Product" in p
    assert "Product Director" in p
    assert "RAG" in p and "retrieval-augmented generation" in p
    assert "Onsite in Tarrytown, NY" in p
    # Static content moved to the system prompt — the user message is job-only.
    assert "CANDIDATE INVENTORY" not in p
    assert "NOT CLAIMED" not in p


def test_prompt_renders_posted_top_of_band():
    # The judge receives the pre-computed integer — the exact number the
    # Python predicate uses — never a range string to parse.
    p = fj.build_judge_prompt("Director of Product", [], _MHS, _KOS,
                              posted_top_of_band=468_000)
    assert "POSTED TOP-OF-BAND: $468,000" in p


def test_prompt_renders_none_without_salary():
    p = fj.build_judge_prompt("Director of Product", [], _MHS, _KOS)
    assert "POSTED TOP-OF-BAND: NONE" in p


def test_judge_system_carries_persona_rules_shape_and_inventory():
    sysp = fj.build_judge_system("INVENTORY BODY TEXT")
    # Persona first — callers (and test fakes) dispatch on this substring.
    assert sysp.startswith(fj.JUDGE_SYSTEM)
    assert "screening judge" in sysp
    assert "INVENTORY BODY TEXT" in sysp
    # The honesty rails live in the system prompt now.
    assert "NOT CLAIMED" in sysp
    assert "BASELINE PRODUCT CRAFT" in sysp
    assert "evidence" in sysp
    # Response-shape spec (braces rendered, not left as {{ escapes).
    assert '{"must_haves":' in sysp
    assert "{{" not in sysp

# ---------------------------------------------------------------------------
# parse_judge_response
# ---------------------------------------------------------------------------

def _judge_json(**overrides):
    data = {
        "must_haves": [
            {"term": "RAG", "verdict": "evidenced",
             "evidence": "Product owner for internal RAG-based assistant"},
            {"term": "product management", "verdict": "explicit",
             "evidence": "Full Agile Product Owner responsibilities"},
        ],
        "knockouts": [
            {"requirement": "8+ years product management experience",
             "verdict": "met", "reason": "8.7 years total",
             "required_years": 8, "candidate_years": 8.7},
            {"requirement": "Onsite in Tarrytown, NY", "verdict": "failed",
             "reason": "remote-only, not relocating",
             "required_years": None, "candidate_years": None},
        ],
        "title_claim": "close",
    }
    data.update(overrides)
    return json.dumps(data)


def test_parse_happy_path_aligns_and_merges_aliases():
    out = fj.parse_judge_response(_judge_json(), _MHS,
                                  [{"requirement": r} for r in _KOS])
    assert out["title_claim"] == "close"
    assert out["must_haves"][0] == {
        "term": "RAG", "aliases": ["retrieval-augmented generation"],
        "verdict": "evidenced",
        "evidence": "Product owner for internal RAG-based assistant"}
    assert out["knockouts"][1]["verdict"] == "failed"


def test_parse_verdict_without_evidence_downgraded_to_absent():
    raw = _judge_json(must_haves=[
        {"term": "RAG", "verdict": "evidenced", "evidence": None},
        {"term": "product management", "verdict": "explicit", "evidence": ""},
    ])
    out = fj.parse_judge_response(raw, _MHS, [{"requirement": r} for r in _KOS])
    assert [m["verdict"] for m in out["must_haves"]] == ["absent", "absent"]


def test_parse_missing_or_unknown_terms_never_gain_credit():
    # Judge omitted one term and invented a verdict word for the other.
    raw = _judge_json(must_haves=[
        {"term": "RAG", "verdict": "definitely", "evidence": "x"},
    ])
    out = fj.parse_judge_response(raw, _MHS, [{"requirement": r} for r in _KOS])
    assert [m["verdict"] for m in out["must_haves"]] == ["absent", "absent"]


def test_parse_missing_or_bad_knockout_verdict_is_unclear():
    raw = _judge_json(knockouts=[
        {"requirement": "8+ years product management experience",
         "verdict": "maybe", "reason": "?"},
    ])
    out = fj.parse_judge_response(raw, _MHS, [{"requirement": r} for r in _KOS])
    assert [k["verdict"] for k in out["knockouts"]] == ["unclear", "unclear"]


def test_parse_bad_title_claim_is_none():
    out = fj.parse_judge_response(_judge_json(title_claim="perfect"),
                                  _MHS, [{"requirement": r} for r in _KOS])
    assert out["title_claim"] == "none"


def test_parse_malformed_json_returns_none():
    assert fj.parse_judge_response("not json at all", _MHS, []) is None


def test_parse_paraphrased_term_still_credited_by_position():
    # Judge returns the SAME count/order but paraphrases "RAG" as its
    # spelled-out form; "product management" stays verbatim.
    raw = _judge_json(must_haves=[
        {"term": "retrieval-augmented generation", "verdict": "evidenced",
         "evidence": "Product owner for internal RAG-based assistant"},
        {"term": "product management", "verdict": "explicit",
         "evidence": "Full Agile Product Owner responsibilities"},
    ])
    out = fj.parse_judge_response(raw, _MHS, [{"requirement": r} for r in _KOS])
    rag, pm = out["must_haves"]
    # Input strings are preserved even though the judge paraphrased.
    assert rag["term"] == "RAG"
    assert rag["aliases"] == ["retrieval-augmented generation"]
    assert rag["verdict"] == "evidenced"
    assert rag["evidence"] == "Product owner for internal RAG-based assistant"
    assert pm["term"] == "product management"
    assert pm["verdict"] == "explicit"


def test_parse_dropped_entry_stays_absent():
    # Judge drops "RAG" entirely and returns only "product management".
    raw = _judge_json(must_haves=[
        {"term": "product management", "verdict": "explicit",
         "evidence": "Full Agile Product Owner responsibilities"},
    ])
    out = fj.parse_judge_response(raw, _MHS, [{"requirement": r} for r in _KOS])
    rag, pm = out["must_haves"]
    assert rag["term"] == "RAG"
    assert rag["verdict"] == "absent"
    assert rag["evidence"] == ""
    assert pm["term"] == "product management"
    assert pm["verdict"] == "explicit"


def test_parse_reorder_preserves_credit_count():
    # Judge returns both entries verbatim but in reversed order.
    raw = _judge_json(must_haves=[
        {"term": "product management", "verdict": "explicit",
         "evidence": "Full Agile Product Owner responsibilities"},
        {"term": "RAG", "verdict": "evidenced",
         "evidence": "Product owner for internal RAG-based assistant"},
    ])
    out = fj.parse_judge_response(raw, _MHS, [{"requirement": r} for r in _KOS])
    by_term = {m["term"]: m for m in out["must_haves"]}
    assert by_term["RAG"]["verdict"] == "evidenced"
    assert by_term["product management"]["verdict"] == "explicit"
    credited = sum(1 for m in out["must_haves"]
                   if m["verdict"] in ("explicit", "evidenced"))
    assert credited == 2


def test_parse_knockout_paraphrase_aligned_by_position():
    # Judge paraphrases the years knockout requirement string; the
    # onsite requirement stays verbatim.
    raw = _judge_json(knockouts=[
        {"requirement": "at least 8 years of product management",
         "verdict": "met", "reason": "8.7 years total",
         "required_years": 8, "candidate_years": 8.7},
        {"requirement": "Onsite in Tarrytown, NY", "verdict": "failed",
         "reason": "remote-only, not relocating",
         "required_years": None, "candidate_years": None},
    ])
    out = fj.parse_judge_response(raw, _MHS, [{"requirement": r} for r in _KOS])
    years_ko, onsite_ko = out["knockouts"]
    assert years_ko["requirement"] == "8+ years product management experience"
    assert years_ko["verdict"] == "met"
    assert years_ko["required_years"] == 8
    assert years_ko["candidate_years"] == 8.7
    assert onsite_ko["requirement"] == "Onsite in Tarrytown, NY"
    assert onsite_ko["verdict"] == "failed"

# ---------------------------------------------------------------------------
# apply_soft_years
# ---------------------------------------------------------------------------

def _years_ko(verdict, req, cand):
    return {"requirement": f"{req}+ years", "verdict": verdict,
            "reason": "r", "required_years": req, "candidate_years": cand}


def test_soft_years_downgrades_near_miss_failed_to_unclear():
    out = fj.apply_soft_years([_years_ko("failed", 12, 8.7)])  # 72.5%
    assert out[0]["verdict"] == "unclear"
    assert "softened" in out[0]["reason"]


def test_soft_years_leaves_big_miss_failed():
    out = fj.apply_soft_years([_years_ko("failed", 15, 8.7)])  # 58%
    assert out[0]["verdict"] == "failed"


def test_soft_years_ignores_non_years_and_met_verdicts():
    kos = [
        {"requirement": "PhD required", "verdict": "failed", "reason": "BA",
         "required_years": None, "candidate_years": None},
        _years_ko("met", 8, 8.7),
    ]
    out = fj.apply_soft_years(kos)
    assert out[0]["verdict"] == "failed"
    assert out[1]["verdict"] == "met"


def test_soft_years_does_not_mutate_input():
    ko = _years_ko("failed", 12, 8.7)
    fj.apply_soft_years([ko])
    assert ko["verdict"] == "failed"
