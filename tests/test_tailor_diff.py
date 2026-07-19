"""tailor_diff: canonicalization, diff blocks, verdicts, issue facts."""
import tailor_diff as td
from policy_fixtures import FIXTURE_SKILL_LABELS, FIXTURE_TITLE_LINE


# --- _canon / _flat -------------------------------------------------------

def test_canon_folds_google_docs_artifacts():
    # curly quotes (U+201C/U+201D), vertical-tab soft break (U+000B), NBSP
    # (U+00A0) before an en dash (U+2013) -- real codepoints, verified
    # byte-exact against a chr()-built ASCII-only oracle, so the fold table
    # in tailor_diff._FOLD_TABLE is actually exercised (not ASCII lookalikes).
    s = 'Led “GenAI” rollout\x0bacross teams\xa0– fast'
    assert td._canon(s) == 'Led "GenAI" rollout\nacross teams - fast'


def test_canon_folds_curly_apostrophes_and_em_dash():
    # left/right single quotation marks (U+2018/U+2019) -> ', em dash
    # (U+2014) -> - : the remaining _FOLD_TABLE entries not covered above.
    assert td._canon('Rowan’s ‘win’ — done') == "Rowan's 'win' - done"


def test_canon_collapses_whitespace_per_line_only():
    assert td._canon("a\t\t b\n  c   d ") == "a b\nc d"


def test_flat_collapses_newlines_too():
    assert td._flat("ab\nc") == "a b c"


# --- compute_diff_blocks --------------------------------------------------

def test_diff_blocks_identical_texts_collapse_to_context():
    master = "\n".join(f"line {i}" for i in range(10))
    blocks = td.compute_diff_blocks(master, master)
    assert len(blocks) == 1
    assert blocks[0]["type"] == "context"
    assert blocks[0]["text"] == "line 0\nline 1\n⋯\nline 8\nline 9"


def test_diff_blocks_word_ops_for_small_change():
    master = "SUMMARY\nDrove product vision for enterprise automation\nEND"
    final = "SUMMARY\nDrove product vision for GenAI enablement\nEND"
    change = [b for b in td.compute_diff_blocks(master, final)
              if b["type"] == "change"]
    assert len(change) == 1 and "ops" in change[0]
    ops = change[0]["ops"]
    assert ["del", "enterprise automation"] in ops
    assert ["ins", "GenAI enablement"] in ops
    assert any(op[0] == "eq" and "Drove product vision for" in op[1]
               for op in ops)


def test_diff_blocks_region_join_handles_unequal_line_counts():
    # 2 changed lines -> 3 changed lines: one region, one word diff, no
    # line-pairing question. Same words reflowed -> all ops are "eq".
    master = "HEADER\naaa bbb ccc\nddd eee fff\nFOOTER"
    final = "HEADER\naaa bbb\nccc ddd\neee fff\nFOOTER"
    change = [b for b in td.compute_diff_blocks(master, final)
              if b["type"] == "change"]
    assert len(change) == 1
    assert all(op[0] == "eq" for op in change[0]["ops"])


def test_diff_blocks_low_similarity_falls_back_to_block():
    master = "X\ncompletely different original sentence written here\nY"
    final = "X\nnothing shared with that text at all okay then\nY"
    change = [b for b in td.compute_diff_blocks(master, final)
              if b["type"] == "change"][0]
    assert "before" in change and "after" in change and "ops" not in change


def test_diff_blocks_oversized_region_falls_back_to_block():
    big_a = " ".join(f"w{i}" for i in range(350))
    big_b = " ".join(f"w{i}" for i in range(349)) + " zz"
    change = [b for b in td.compute_diff_blocks(f"X\n{big_a}\nY", f"X\n{big_b}\nY")
              if b["type"] == "change"][0]
    assert "before" in change  # block fallback despite high similarity


def test_diff_blocks_pure_insert_and_delete():
    change = [b for b in td.compute_diff_blocks("A\nC", "A\nB\nC")
              if b["type"] == "change"][0]
    assert change["before"] == "" and change["after"] == "B"
    change = [b for b in td.compute_diff_blocks("A\nB\nC", "A\nC")
              if b["type"] == "change"][0]
    assert change["before"] == "B" and change["after"] == ""


def test_diff_blocks_autojunk_disabled():
    # Empirically verified (scratch probe, see task-1-report.md): a 250-word
    # region cycling ["the","quick","brown","fox","jumps"] with ONE word
    # swapped at index 125 fragments badly if autojunk reverts to its
    # difflib default (True). autojunk activates at >=200 elements when a
    # token exceeds 1% frequency -- here each of the 5 tokens sits at 20%.
    #
    # Direct SequenceMatcher comparison on these exact word arrays:
    #   autojunk=True : ('equal',0,125,0,125), ('replace',125,250,125,250)
    #                   -> a 125-word del + a 125-word ins; ratio 0.5 (still
    #                   >= _MIN_WORD_RATIO, so the low-similarity fallback
    #                   would NOT catch this regression either).
    #   autojunk=False: ('equal',0,125,..), ('replace',125,126,..),
    #                   ('equal',126,250,..) -> a clean single-word del/ins;
    #                   ratio 0.996.
    # This test pins the autojunk=False (clean) shape through the real
    # td.compute_diff_blocks code path, so it fails if _word_ops' explicit
    # autojunk=False is ever removed or flipped.
    words = ["the", "quick", "brown", "fox", "jumps"] * 50
    before_words = words.copy()
    after_words = words.copy()
    after_words[125] = "GenAI"
    master = "HEADER\n" + " ".join(before_words) + "\nFOOTER"
    final = "HEADER\n" + " ".join(after_words) + "\nFOOTER"
    change = [b for b in td.compute_diff_blocks(master, final)
              if b["type"] == "change"]
    assert len(change) == 1
    ops = change[0]["ops"]
    assert len(ops) == 4
    assert ["del", "the"] in ops
    assert ["ins", "GenAI"] in ops


# --- compute_edit_verdicts ------------------------------------------------
# The label value is the literal master-doc text ("lntegration" — lowercase
# l standing in for a capital I — is what the document really contains; see
# tailor_engine._SKILL_SUBCATEGORY_LABELS / policy_fixtures.FIXTURE_SKILL_LABELS).
# Never "corrected".
LABELS = FIXTURE_SKILL_LABELS
TITLE_LINE = FIXTURE_TITLE_LINE


def _payload(**over):
    base = {
        "title_line_replacement": "Principal Product Manager",
        "summary_replacement": "Seasoned PM driving GenAI enablement.",
        "skills_reorder": {"Integration Skills": ["LangChain", "MCP", "RAG"]},
        # Production shape (tailor_engine LLM payload): the original is stored
        # WITH its bold label; only the replacement is post-label. The
        # 2026-07-17 incident came from verdicts reading a nonexistent
        # `original_after_label` key off this dict.
        "experience_edits": [{
            "company": "Acme", "role": "Senior PM",
            "bold_label": "Impact",
            "original": "Impact: cut costs 20% via automation",
            "replacement_after_label": "cut costs 20% via agentic automation",
        }],
        "rewritten_bullets": [{
            "company": "Acme", "role": "Senior PM",
            "original": "Ops: ran weekly reviews",
            "rewritten": "Ops: ran weekly AI-enablement reviews",
        }],
        "master_text": (
            f"{FIXTURE_TITLE_LINE}\n"
            "Old summary text.\n"
            "lntegration Skills: Python, SQL\n"
            "Impact: cut costs 20% via automation\n"
            "Ops: ran weekly reviews\n"),
    }
    base.update(over)
    return base


FINAL_ALL_LANDED = (
    "PRINCIPAL PRODUCT MANAGER\n"          # applied .upper()
    "Seasoned PM driving GenAI enablement.\n"
    "lntegration Skills: LangChain, MCP, RAG\n"
    "Impact: cut costs 20% via agentic automation\n"
    "Ops: ran weekly AI-enablement reviews\n")


def test_verdicts_all_landed_including_uppercased_title():
    vs = td.compute_edit_verdicts(_payload(), FINAL_ALL_LANDED, LABELS, TITLE_LINE)
    assert len(vs) == 5
    assert [v["verdict"] for v in vs] == [td.VERDICT_LANDED] * 5
    assert {v["section"] for v in vs} == {
        "title", "summary", "skills", "experience", "rewrite"}


def test_verdicts_reverted_when_master_text_stands():
    final = _payload()["master_text"]  # doc identical to master
    vs = td.compute_edit_verdicts(_payload(), final, LABELS, TITLE_LINE)
    by = {v["section"]: v["verdict"] for v in vs}
    assert by["title"] == td.VERDICT_REVERTED
    assert by["summary"] == td.VERDICT_NOT_LANDED  # no original in payload
    assert by["skills"] == td.VERDICT_REVERTED     # master line via label lookup
    assert by["experience"] == td.VERDICT_REVERTED
    assert by["rewrite"] == td.VERDICT_REVERTED


def test_verdicts_modified_when_neither_side_present():
    vs = td.compute_edit_verdicts(_payload(), "TOTALLY DIFFERENT DOC\n",
                                  LABELS, TITLE_LINE)
    by = {v["section"]: v["verdict"] for v in vs}
    assert by["title"] == td.VERDICT_MODIFIED
    assert by["summary"] == td.VERDICT_NOT_LANDED
    assert by["skills"] == td.VERDICT_MODIFIED
    assert by["experience"] == td.VERDICT_MODIFIED


def test_experience_before_is_post_label_body_from_original():
    # `before` must be the engine's own replaceAllText anchor: the first-colon
    # split of `original` (tailor_engine._apply_experience_edits). Symmetric
    # with `after`, which is also post-label.
    vs = td.compute_edit_verdicts(_payload(), _payload()["master_text"],
                                  LABELS, TITLE_LINE)
    exp = next(v for v in vs if v["section"] == "experience")
    assert exp["before"] == "cut costs 20% via automation"
    assert exp["verdict"] == td.VERDICT_REVERTED


def test_experience_guard_reshape_is_modified_not_not_landed():
    # A guard-reshape shape (the 2026-07-17 incident): the layout guard
    # reworded the landed bullet, so neither the original nor the
    # replacement is verbatim in the final doc. That is
    # `modified` (amber, no badge) — `not_landed` would count as reverted in
    # build_issue_facts and inflate the badge.
    final = "Impact: cut costs 20% with agentic workflows\n"
    vs = td.compute_edit_verdicts(_payload(), final, LABELS, TITLE_LINE)
    exp = next(v for v in vs if v["section"] == "experience")
    assert exp["verdict"] == td.VERDICT_MODIFIED


def test_experience_explicit_original_after_label_takes_precedence():
    # The spec's key name, honored if a payload ever carries it (the fixture
    # shape tests used pre-2026-07-17). More precise than the derived split.
    p = _payload()
    p["experience_edits"] = [{
        "company": "Acme", "role": "Senior PM",
        "original_after_label": "cut costs 20% via automation",
        "replacement_after_label": "cut costs 20% via agentic automation",
    }]
    vs = td.compute_edit_verdicts(p, "cut costs 20% via automation",
                                  LABELS, TITLE_LINE)
    exp = next(v for v in vs if v["section"] == "experience")
    assert exp["before"] == "cut costs 20% via automation"
    assert exp["verdict"] == td.VERDICT_REVERTED


def test_experience_colonless_original_used_whole():
    # No label in `original` → the whole string is the anchor, mirroring the
    # engine's fallback.
    p = _payload()
    p["experience_edits"] = [{
        "company": "Acme", "role": "Senior PM",
        "original": "cut costs 20% via automation",
        "replacement_after_label": "cut costs 20% via agentic automation",
    }]
    vs = td.compute_edit_verdicts(p, "cut costs 20% via automation",
                                  LABELS, TITLE_LINE)
    exp = next(v for v in vs if v["section"] == "experience")
    assert exp["before"] == "cut costs 20% via automation"
    assert exp["verdict"] == td.VERDICT_REVERTED


def test_experience_without_any_original_stays_not_landed():
    p = _payload()
    p["experience_edits"] = [{
        "company": "Acme", "role": "Senior PM",
        "replacement_after_label": "cut costs 20% via agentic automation",
    }]
    vs = td.compute_edit_verdicts(p, "unrelated doc", LABELS, TITLE_LINE)
    exp = next(v for v in vs if v["section"] == "experience")
    assert exp["before"] is None
    assert exp["verdict"] == td.VERDICT_NOT_LANDED


def test_smart_quote_in_doc_does_not_flag_landed_edit():
    p = _payload(summary_replacement="Rowan's GenAI summary.")
    final = "Rowan's GenAI summary."  # curly apostrophe from Google Docs
    vs = td.compute_edit_verdicts(p, final, LABELS, TITLE_LINE)
    assert {v["section"]: v["verdict"] for v in vs}["summary"] == td.VERDICT_LANDED


def test_wrapped_payload_shape_matches_flat():
    flat = _payload()
    master = flat.pop("master_text")
    wrapped = {"edits": flat, "master_text": master}
    a = td.compute_edit_verdicts({**flat, "master_text": master},
                                 FINAL_ALL_LANDED, LABELS, TITLE_LINE)
    b = td.compute_edit_verdicts(wrapped, FINAL_ALL_LANDED, LABELS, TITLE_LINE)
    assert a == b


def test_unknown_skill_subcategory_is_skipped_like_the_engine():
    p = _payload(skills_reorder={"Invented Category": ["X", "Y"]})
    vs = td.compute_edit_verdicts(p, "whatever", LABELS, TITLE_LINE)
    assert all(v["section"] != "skills" for v in vs)


def test_empty_edit_fields_produce_no_verdicts():
    vs = td.compute_edit_verdicts({"master_text": "m"}, "f", LABELS, TITLE_LINE)
    assert vs == []


# --- build_issue_facts ----------------------------------------------------

def _v(verdict):
    return {"section": "experience", "label": "Acme — PM",
            "verdict": verdict, "before": "b", "after": "a"}


def test_facts_none_when_clean():
    assert td.build_issue_facts([_v(td.VERDICT_LANDED)], [], False) is None


def test_facts_modified_alone_is_clean():
    # A guard repair is a successful fix — amber chip in the panel, no badge.
    assert td.build_issue_facts([_v(td.VERDICT_MODIFIED)], [], False) is None


def test_facts_reverted_counts_not_landed_too():
    facts = td.build_issue_facts(
        [_v(td.VERDICT_REVERTED), _v(td.VERDICT_NOT_LANDED),
         _v(td.VERDICT_MODIFIED)], [], False)
    assert facts == {"reverted": 2, "modified": 1,
                     "missing_terms": [], "layout_unverified": False}


def test_facts_missing_terms_and_layout_trigger():
    assert td.build_issue_facts([], ["PLM"], False)["missing_terms"] == ["PLM"]
    assert td.build_issue_facts([], [], True)["layout_unverified"] is True


def test_malformed_skills_reorder_degrades_instead_of_crashing():
    p = _payload(skills_reorder=["not", "a", "dict"])
    vs = td.compute_edit_verdicts(p, "anything", LABELS, TITLE_LINE)
    assert all(v["section"] != "skills" for v in vs)  # skipped, no crash
    assert len(vs) == 4  # title, summary, experience, rewrite still computed


def test_non_dict_edits_payload_yields_no_verdicts():
    assert td.compute_edit_verdicts("not a dict", "final", LABELS, TITLE_LINE) == []
    assert td.compute_edit_verdicts(None, "final", LABELS, TITLE_LINE) == []
