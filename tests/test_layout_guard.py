"""Pure unit tests for resume_tailor.layout_guard — no network, no Google, no LLM."""
import resume_tailor.layout_guard as lg


# ---------------------------------------------------------------------------
# normalize()
# ---------------------------------------------------------------------------

def test_normalize_collapses_whitespace():
    assert lg.normalize("  a\tb\n c  ") == "a b c"

def test_normalize_folds_typographic_variants():
    assert lg.normalize("‘a’ “b”") == "'a' \"b\""
    assert lg.normalize("Jan 2024 – Present — now") == "Jan 2024 - Present - now"
    assert lg.normalize("a b") == "a b"

def test_normalize_expands_ligatures():
    assert lg.normalize("eﬃcient proﬁle workﬂow") == "efficient profile workflow"

def test_normalize_strips_bullet_glyphs():
    # PDF text layers include the bullet glyph; Docs-API paragraph text does not.
    assert lg.normalize("• Led teams") == "Led teams"
    assert lg.normalize("•Led teams") == "Led teams"

def test_normalize_empty():
    assert lg.normalize("") == ""
    assert lg.normalize("•") == ""


# ---------------------------------------------------------------------------
# PDF fixture helper — hand-assembled minimal PDF, no extra dependencies.
# ---------------------------------------------------------------------------

def _make_pdf(pages):
    """Build a minimal valid PDF. `pages` is a list of pages; each page is a
    list of text-line strings drawn top-down (Helvetica 11, 14pt leading).
    Line text must not contain '(', ')' or backslash (PDF string syntax)."""
    n = len(pages)
    page_ids = [3 + 2 * i for i in range(n)]
    content_ids = [4 + 2 * i for i in range(n)]
    font_id = 3 + 2 * n
    kids = " ".join(f"{pid} 0 R" for pid in page_ids)
    objs = [
        (1, "<< /Type /Catalog /Pages 2 0 R >>"),
        (2, f"<< /Type /Pages /Kids [{kids}] /Count {n} >>"),
    ]
    for i, lines in enumerate(pages):
        parts = ["BT /F1 11 Tf"]
        y = 720
        for ln in lines:
            parts.append(f"1 0 0 1 72 {y} Tm ({ln}) Tj")
            y -= 14
        parts.append("ET")
        stream = "\n".join(parts)
        objs.append((page_ids[i],
                     "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
                     f"/Resources << /Font << /F1 {font_id} 0 R >> >> "
                     f"/Contents {content_ids[i]} 0 R >>"))
        objs.append((content_ids[i],
                     f"<< /Length {len(stream)} >>\nstream\n{stream}\nendstream"))
    objs.append((font_id, "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"))
    objs.sort(key=lambda t: t[0])
    out = bytearray(b"%PDF-1.4\n")
    offsets = {}
    for num, body in objs:
        offsets[num] = len(out)
        out += f"{num} 0 obj\n{body}\nendobj\n".encode("latin-1")
    xref_off = len(out)
    count = len(objs) + 1
    out += f"xref\n0 {count}\n".encode()
    out += b"0000000000 65535 f \n"
    for num in sorted(offsets):
        out += f"{offsets[num]:010d} 00000 n \n".encode()
    out += (f"trailer\n<< /Size {count} /Root 1 0 R >>\n"
            f"startxref\n{xref_off}\n%%EOF").encode()
    return bytes(out)


# ---------------------------------------------------------------------------
# build_line_map()
# ---------------------------------------------------------------------------

def test_build_line_map_single_page():
    pdf = _make_pdf([["Hello world on line one", "and a short tail"]])
    m = lg.build_line_map(pdf)
    assert m.page_count == 1
    assert [ln.text for ln in m.lines] == ["Hello world on line one", "and a short tail"]
    assert [ln.words for ln in m.lines] == [5, 4]
    assert m.lines[0].top < m.lines[1].top      # reading order, top-down

def test_build_line_map_two_pages_in_order():
    pdf = _make_pdf([["page one line"], ["page two line"]])
    m = lg.build_line_map(pdf)
    assert m.page_count == 2
    assert [ln.text for ln in m.lines] == ["page one line", "page two line"]


# ---------------------------------------------------------------------------
# locate_paragraph() and friends — synthetic LineMaps, no PDF needed.
# ---------------------------------------------------------------------------

def _lm(*line_texts):
    return lg.LineMap(
        lines=[lg.RenderedLine(text=t, words=len(lg.normalize(t).split()), top=float(i))
               for i, t in enumerate(line_texts)],
        page_count=1,
    )

_WRAPPED = _lm(
    "PROFESSIONAL SUMMARY",
    "Results-driven AI Product Leader with 8 years driving digital",
    "transformation and GenAI integration at Fortune 50 scale across",
    "several teams",
    "CORE COMPETENCIES",
)

def test_locate_single_line():
    assert lg.locate_paragraph(_WRAPPED, "PROFESSIONAL SUMMARY") == (0, 0)

def test_locate_wrapped_paragraph_spans_lines():
    para = ("Results-driven AI Product Leader with 8 years driving digital "
            "transformation and GenAI integration at Fortune 50 scale across "
            "several teams")
    assert lg.locate_paragraph(_WRAPPED, para) == (1, 3)
    assert lg.paragraph_line_count(_WRAPPED, para) == 3
    assert lg.last_line_word_count(_WRAPPED, para) == 2   # "several teams"

def test_locate_not_found_returns_none():
    assert lg.locate_paragraph(_WRAPPED, "text that is not there") is None
    assert lg.paragraph_line_count(_WRAPPED, "nope") is None
    assert lg.last_line_word_count(_WRAPPED, "nope") is None

def test_locate_is_typography_insensitive():
    # Docs text has an em dash + curly quote; the PDF layer has ASCII + ligature.
    pdf_side = _lm("the team’s workﬂow – shipped fast")
    docs_side_text = "the team's workflow - shipped fast"
    assert lg.locate_paragraph(pdf_side, docs_side_text) == (0, 0)

def test_locate_ignores_bullet_glyph_prefix():
    m = _lm("• Enterprise Scale: Led delivery of platforms",
            "across three business units")
    para = "Enterprise Scale: Led delivery of platforms across three business units"
    assert lg.locate_paragraph(m, para) == (0, 1)

def test_locate_empty_text_returns_none():
    assert lg.locate_paragraph(_WRAPPED, "") is None
    assert lg.locate_paragraph(_WRAPPED, "•") is None


# ---------------------------------------------------------------------------
# check_layout()
# ---------------------------------------------------------------------------

_M_TEXT = ("Product Vision: Led roadmap and strategy for the enterprise AI "
           "platform serving many teams across the organization globally")

_MASTER = _lm(
    "Product Vision: Led roadmap and strategy for the enterprise AI",
    "platform serving many teams across the organization globally",
)

def _para(tailored_text, role="bullet:ACME:Product Vision"):
    return lg.EditedParagraph(role=role, master_text=_M_TEXT, tailored_text=tailored_text)

def test_check_layout_exact_match_no_violation():
    tailored = _lm(
        "Product Vision: Led roadmap and planning for the enterprise AI",
        "platform serving many groups across the organization globally",
    )
    text = ("Product Vision: Led roadmap and planning for the enterprise AI "
            "platform serving many groups across the organization globally")
    assert lg.check_layout(_MASTER, tailored, [_para(text)]) == []

def test_check_layout_grew():
    tailored = _lm(
        "Product Vision: Led roadmap and planning for the enterprise AI",
        "platform serving many groups across the organization globally and",
        "several more",
    )
    text = ("Product Vision: Led roadmap and planning for the enterprise AI "
            "platform serving many groups across the organization globally and "
            "several more")
    (v,) = lg.check_layout(_MASTER, tailored, [_para(text)])
    assert (v.kind, v.master_lines, v.tailored_lines) == ("grew", 2, 3)
    assert v.target_chars == len(_M_TEXT)

def test_check_layout_shrank():
    tailored = _lm("Product Vision: Led roadmap and planning briefly")
    (v,) = lg.check_layout(_MASTER, tailored,
                           [_para("Product Vision: Led roadmap and planning briefly")])
    assert (v.kind, v.master_lines, v.tailored_lines) == ("shrank", 2, 1)

def test_check_layout_dangler_at_equal_line_count():
    tailored = _lm(
        "Product Vision: Led roadmap and planning for the enterprise AI platform serving",
        "two words",
    )
    text = ("Product Vision: Led roadmap and planning for the enterprise AI "
            "platform serving two words")
    (v,) = lg.check_layout(_MASTER, tailored, [_para(text)])
    assert (v.kind, v.last_line_words) == ("dangler", 2)

def test_check_layout_five_word_last_line_is_ok():
    tailored = _lm(
        "Product Vision: Led roadmap and planning for the enterprise",
        "AI platform with five words",     # exactly 5 words → not a dangler
    )
    text = ("Product Vision: Led roadmap and planning for the enterprise "
            "AI platform with five words")
    assert lg.check_layout(_MASTER, tailored, [_para(text)]) == []

def test_check_layout_single_line_exempt_from_dangler():
    single_master = _lm("Tools: Jira, Figma")
    tailored = _lm("Tools: Figma, Jira")
    p = lg.EditedParagraph(role="skills:Tools", master_text="Tools: Jira, Figma",
                           tailored_text="Tools: Figma, Jira")
    assert lg.check_layout(single_master, tailored, [p]) == []

def test_check_layout_master_dangler_skips_dangler_check():
    master = _lm("Product Vision: Led roadmap and strategy for enterprise",
                 "AI platforms")                                  # master ends short too
    m_text = "Product Vision: Led roadmap and strategy for enterprise AI platforms"
    tailored = _lm("Product Vision: Led roadmap and planning for enterprise",
                   "ML platforms")
    p = lg.EditedParagraph(role="bullet:ACME:Product Vision", master_text=m_text,
                           tailored_text="Product Vision: Led roadmap and planning for enterprise ML platforms")
    assert lg.check_layout(master, tailored, [p]) == []           # can't beat the source

def test_check_layout_tailored_not_found_is_lost():
    tailored = _lm("completely different content")
    (v,) = lg.check_layout(_MASTER, tailored, [_para("text that is not in the map")])
    assert v.kind == "lost"
    assert v.master_text == _M_TEXT

def test_check_layout_master_not_found_skips():
    p = lg.EditedParagraph(role="bullet:X:Y", master_text="not in master map",
                           tailored_text="whatever")
    assert lg.check_layout(_MASTER, _lm("whatever"), [p]) == []
