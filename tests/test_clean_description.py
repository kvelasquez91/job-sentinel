"""Tests for description HTML sanitization.

Regression: job descriptions were stored verbatim from each source's API.
Some sources return raw HTML (<div>/<p>/<li>), some return ENTITY-ENCODED HTML
(&lt;li&gt;…), and WTTJ's extractor stripped tags BEFORE unescaping — so encoded
tags survived and the dashboard rendered literal "<ul><li>…" text. clean_description
now normalizes every shape to readable plain text, applied centrally in
JobPosting.__post_init__.
"""
from scrapers.base import JobPosting, clean_description


def test_entity_encoded_html_is_decoded_then_stripped():
    """The exact WTTJ failure: entity-encoded list markup must become clean text
    with bullets, not literal tags."""
    raw = ("&lt;ul&gt;&lt;li&gt;6-7 years of experience.&lt;/li&gt;"
           "&lt;li&gt;&lt;p&gt;You can run a discovery conversation.&lt;/p&gt;&lt;/li&gt;&lt;/ul&gt;")
    out = clean_description(raw)
    assert "<" not in out and ">" not in out, out
    assert "6-7 years of experience." in out
    assert "You can run a discovery conversation." in out
    assert "•" in out  # list structure preserved as bullets


def test_raw_html_tags_are_stripped():
    """Greenhouse-style raw HTML."""
    raw = ('<div class="content-intro"><div><strong>About Us</strong></div>'
           '<p>We build a better Internet.</p><ul><li>Ship features</li></ul></div>')
    out = clean_description(raw)
    assert "<" not in out and ">" not in out
    assert "About Us" in out
    assert "We build a better Internet." in out
    assert "Ship features" in out


def test_numeric_and_named_entities_are_decoded():
    raw = "Customer insight.&#xa0;Strong data fluency &amp; rigor &lt;b&gt;bold&lt;/b&gt;"
    out = clean_description(raw)
    assert "\xa0" not in out
    assert "&amp;" not in out and "&#xa0;" not in out
    assert "Strong data fluency & rigor" in out
    assert "<b>" not in out and "bold" in out


def test_plain_text_is_unchanged():
    raw = "Senior PM role. 6+ years experience. Remote-friendly."
    assert clean_description(raw) == raw


def test_idempotent():
    raw = ("&lt;ul&gt;&lt;li&gt;First point.&lt;/li&gt;&lt;li&gt;Second point.&lt;/li&gt;&lt;/ul&gt;"
           "<p>Trailing &amp; paragraph.</p>")
    once = clean_description(raw)
    twice = clean_description(once)
    assert once == twice, f"not idempotent:\n{once!r}\n{twice!r}"


def test_double_encoded_html_is_fully_decoded():
    """Defense-in-depth: even double-encoded tags (&amp;lt;li&amp;gt;) must not
    leak through as literal tags."""
    raw = "&amp;lt;p&amp;gt;Nested encoding.&amp;lt;/p&amp;gt;"
    out = clean_description(raw)
    assert "<" not in out and ">" not in out
    assert "Nested encoding." in out


def test_trailing_truncated_tag_fragment_is_removed():
    """A hard [:N] cap can cut the input mid-tag, leaving an unclosed fragment
    with no '>' (e.g. greenhouse rows ending in '</h2&g' or '<div class=\"title&qu')."""
    assert clean_description("You may be a good fit if you have </h2&g") == \
        "You may be a good fit if you have"
    out = clean_description("What we offer.\n\n<div class=\"title&qu")
    assert "<" not in out
    assert out.startswith("What we offer.")
    # A bare truncated "</" (e.g. "$240,000 USD</") must go too.
    assert clean_description("$187,000 — $240,000 USD</") == "$187,000 — $240,000 USD"


def test_literal_less_than_in_prose_is_preserved():
    """The trailing-fragment strip must not eat legitimate prose that ends with
    a comparison like 'a < b' (space after '<' means it's not a tag)."""
    assert clean_description("Latency stays under a < b threshold") == \
        "Latency stays under a < b threshold"


def test_empty_and_none():
    assert clean_description("") == ""
    assert clean_description(None) == ""


def test_jobposting_sanitizes_description_on_construction():
    """The central choke point: constructing a JobPosting with HTML must store
    clean text, for every source."""
    job = JobPosting(
        title="Senior PM",
        company="Believe",
        location="Remote",
        url="https://example.com/j/1",
        description="<ul><li>6-7 years of experience.</li><li><p>Debug a workflow.</p></li></ul>",
        source="wttj",
    )
    assert "<" not in job.description and ">" not in job.description
    assert "6-7 years of experience." in job.description
    assert "Debug a workflow." in job.description


def test_paired_comparisons_in_prose_preserved():
    """_TAG_RE must only strip real tags (letter after '<'): prose containing
    two comparisons must not have the span between them deleted."""
    assert clean_description("if a < b then x > y") == "if a < b then x > y"
    assert (clean_description("5+ yrs, team of <10, budget >$1M")
            == "5+ yrs, team of <10, budget >$1M")


def test_html_comments_removed():
    out = clean_description("Before<!-- hidden <div> note -->After")
    assert "hidden" not in out
    assert "Before" in out and "After" in out


def test_description_capped_after_cleaning():
    """The length ceiling must apply to CLEANED text (capping raw HTML at the
    scraper let markup overhead eat the budget)."""
    from scrapers.base import DESCRIPTION_MAX_LEN
    words = (DESCRIPTION_MAX_LEN // 5) * 2  # ~2x the ceiling in prose
    raw = "<div><p>" + ("word " * words) + "</p></div>"
    job = JobPosting(
        title="t", company="c", location="l", url="https://x/1", description=raw,
    )
    assert len(job.description) == DESCRIPTION_MAX_LEN
    assert "<" not in job.description


def test_real_sized_descriptions_are_not_truncated():
    """The ceiling is a sanity bound against pathological page dumps, NOT a
    scoring decision: real JDs (max observed in the DB: ~23k chars) must pass
    through whole. Stored text is exactly what the keyword scorer reads on
    reblend, so truncating here silently changes scores — a 5000 cap pushed
    34 live senior-AI roles below the alert line (2026-07-11), and the median
    stored description is ~5.3k, so it bisected the corpus."""
    prose = ("Own the AI roadmap and platform strategy. " * 560).rstrip()
    assert len(prose) > 23000  # at least as long as the longest real JD seen
    job = JobPosting(
        title="t", company="c", location="l", url="https://x/2", description=prose,
    )
    assert job.description == prose  # untruncated, unchanged
