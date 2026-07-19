"""Tests for the keyword scorer (engine/scorer.py).

Word-boundary matching: 'ai' must not match inside 'email'/'training',
'ml' must not match inside 'html'. Plural forms still count ('LLMs',
'product managers'), and hyphenated contexts still match ('AI-powered').
"""
from engine.scorer import JobScorer
from scrapers.base import JobPosting
from policy_fixtures import (
    patch_state_pattern, patch_scorer_keywords, patch_comp_bars,
    patch_relocation_exception,
)


def _job(**overrides):
    defaults = dict(
        title="Developer",
        company="TestCo",
        location="Remote",
        url="https://example.com/job/1",
        description="",
    )
    defaults.update(overrides)
    return JobPosting(**defaults)


def _scorer():
    return JobScorer({"scoring": {"alert_threshold": 60}})


def test_substring_words_earn_no_keyword_points():
    """'email', 'html', 'training', 'maintain' must not trigger ai/ml keywords."""
    scorer = _scorer()
    noise = _job(description="Maintain email templates and html training materials")
    blank = _job(description="")
    assert scorer.score(noise) == scorer.score(blank)


def test_real_ai_keywords_earn_points(monkeypatch):
    patch_scorer_keywords(monkeypatch)
    scorer = _scorer()
    real = _job(description="We build AI and ML products using LLM technology")
    blank = _job(description="")
    assert scorer.score(real) > scorer.score(blank)


def test_plural_keyword_forms_still_match(monkeypatch):
    """'LLMs' (plural) must still count as the 'llm' keyword."""
    patch_scorer_keywords(monkeypatch)
    scorer = _scorer()
    plural = _job(description="Hands-on experience shipping LLMs required")
    blank = _job(description="")
    assert scorer.score(plural) > scorer.score(blank)


def test_multiword_keyword_plural_matches(monkeypatch):
    patch_scorer_keywords(monkeypatch)
    scorer = _scorer()
    multi = _job(description="Looking for experienced product managers")
    blank = _job(description="")
    assert scorer.score(multi) > scorer.score(blank)


def test_hyphenated_context_still_matches(monkeypatch):
    """Hyphen is a word boundary: 'AI-powered' must match the 'ai' keyword."""
    patch_scorer_keywords(monkeypatch)
    scorer = _scorer()
    hyphen = _job(description="An AI-powered analytics platform")
    blank = _job(description="")
    assert scorer.score(hyphen) > scorer.score(blank)


def test_explain_does_not_claim_ai_for_substring_noise():
    scorer = _scorer()
    noise = _job(description="Maintain email templates and html training materials")
    assert "Strong AI/ML signal" not in scorer.explain(noise)


def test_explain_reports_real_ai_keywords(monkeypatch):
    patch_scorer_keywords(monkeypatch)
    scorer = _scorer()
    real = _job(description="We build AI and ML products")
    explanation = scorer.explain(real)
    assert "Strong AI/ML signal" in explanation
    assert "AI" in explanation


# ---------------------------------------------------------------------------
# Local-area scoring
# ---------------------------------------------------------------------------

LOCAL_CONFIG = {
    "scoring": {"alert_threshold": 60},
    "local_locations": ["Springfield", "Riverton", "Cedar Falls"],
}


def _local_scorer(monkeypatch):
    # JobScorer builds its matcher via build_local_area_regex(local_locations)
    # with no explicit state_pattern, so it reads the ambient
    # local_area.LOCAL_STATE_PATTERN default — "" (matcher disabled) in a
    # neutral tree. Pin a real pattern so these local-behavior assertions
    # don't depend on which tree they run in.
    patch_state_pattern(monkeypatch)
    return JobScorer(LOCAL_CONFIG)


def test_local_state_location_scores_like_remote(monkeypatch):
    scorer = _local_scorer(monkeypatch)
    local = _job(location="Springfield, IL",
                 description="AI and ML product work")
    remote = _job(location="Remote",
                  description="AI and ML product work")
    assert scorer.score(local) == scorer.score(remote)


def test_spelled_out_state_is_local(monkeypatch):
    scorer = _local_scorer(monkeypatch)
    assert scorer._is_local("springfield, illinois, united states")
    assert scorer._is_local("cedar falls, il")


def test_local_city_other_states_are_not_local(monkeypatch):
    scorer = _local_scorer(monkeypatch)
    assert not scorer._is_local("springfield, oh")
    assert not scorer._is_local("springfield, missouri")
    assert not scorer._is_local("chicago, il")  # IL but not a listed city


def test_local_comp_calibrated_to_150k(monkeypatch):
    patch_comp_bars(monkeypatch)  # canonical bars — immune to a personalized config
    scorer = _local_scorer(monkeypatch)
    job = _job(location="Springfield, IL", salary_min=140000, salary_max=155000)
    assert scorer._score_compensation(job, "testco", is_local=True) == 20
    assert scorer._score_compensation(job, "testco", is_local=False) == 0


def test_local_midband_comp_partial_points(monkeypatch):
    patch_comp_bars(monkeypatch)  # canonical bars — immune to a personalized config
    scorer = _local_scorer(monkeypatch)
    job = _job(location="Springfield, IL", salary_min=125000, salary_max=135000)
    assert scorer._score_compensation(job, "testco", is_local=True) == 10


def test_multi_city_listing_with_local_not_hard_filtered(monkeypatch):
    scorer = _local_scorer(monkeypatch)
    job = _job(location="Atlanta, GA or Springfield, IL",
               title="Senior Product Manager",
               description="AI product work")
    assert scorer.score(job) > 0


def test_named_non_remote_city_caps_score_not_zeroed():
    """A strong senior AI PM role in a named non-remote city is penalized, not
    zeroed. Remote-only is a soft constraint the LLM blend + Filter Match judge
    far better than a crude city-list zero; previously this returned 0, burying
    genuinely strong roles (e.g. an SVP Product role listed in New York)."""
    scorer = _scorer()
    job = _job(title="Senior Product Manager",
               location="New York, NY",
               description="AI and ML product work")
    score = scorer.score(job)
    assert score > 0            # not zeroed
    assert score <= 40          # capped below the 60 alert threshold


def test_location_cap_is_binding_versus_remote_equivalent(monkeypatch):
    """The cap is what limits the named-city job: the identical role listed
    Remote clears the cap, proving the penalty comes from location alone."""
    patch_scorer_keywords(monkeypatch)
    scorer = _scorer()
    onsite = _job(title="Senior Product Manager", location="New York, NY",
                  description="AI and ML product work")
    remote = _job(title="Senior Product Manager", location="Remote",
                  description="AI and ML product work")
    assert scorer.score(remote) > 40
    assert scorer.score(onsite) <= 40
    assert scorer.score(remote) > scorer.score(onsite)


def test_weak_role_in_named_city_stays_low_under_cap():
    """The cap lifts nothing on its own — a non-PM role in a named city still
    scores low, it just isn't forced to exactly 0."""
    scorer = _scorer()
    job = _job(title="Office Administrator", location="Boston, MA",
               description="Front desk and scheduling duties")
    assert scorer.score(job) <= 40


def test_physical_product_role_without_digital_signal_zeroed(monkeypatch):
    scorer = _local_scorer(monkeypatch)
    job = _job(title="Product Manager - Tire Development",
               location="Springfield, IL",
               description="Own the passenger tire product line roadmap.")
    assert scorer.score(job) == 0


def test_physical_term_with_digital_signal_survives(monkeypatch):
    scorer = _local_scorer(monkeypatch)
    job = _job(title="Product Manager - Tire Digital Services",
               location="Springfield, IL",
               description="Build the AI platform and software for connected tires.")
    assert scorer.score(job) > 0


def test_plural_physical_product_role_zeroed(monkeypatch):
    """Plural forms ('Tires', 'Fasteners') must be caught by the physical-product
    filter, not just the singular."""
    scorer = _local_scorer(monkeypatch)
    job = _job(title="Product Manager - Tires",
               location="Springfield, IL",
               description="Own the passenger tire product line roadmap.")
    assert scorer.score(job) == 0


# ---------------------------------------------------------------------------
# Junior-title hard filter
# ---------------------------------------------------------------------------


def test_associate_director_title_not_zeroed():
    """'Associate Director' is a director-track senior role, not a junior one."""
    scorer = _scorer()
    job = _job(title="Associate Director: PST AI Portfolio Lead (Remote)",
               description="AI product work")
    assert scorer.score(job) > 0


def test_associate_senior_rank_variants_not_zeroed():
    scorer = _scorer()
    for title in ["Associate Director, Product Management",
                  "Associate Directors of Product",
                  "Associate VP, Product",
                  "Associate Vice President of AI",
                  "Associate Principal, Digital Strategy"]:
        job = _job(title=title, description="AI product work")
        assert scorer.score(job) > 0, title


def test_plain_associate_titles_still_zeroed():
    scorer = _scorer()
    for title in ["Associate Product Manager",
                  "Product Associate",
                  "Sales Associates",
                  "Associate, Director's Office"]:
        job = _job(title=title, description="AI product work")
        assert scorer.score(job) == 0, title


def test_internal_and_international_titles_not_zeroed():
    """'intern' must not substring-match 'Internal' or 'International'."""
    scorer = _scorer()
    for title in ["Director of Product - Internal AI",
                  "Head of Internal AI",
                  "Director, Product Management - International"]:
        job = _job(title=title, description="AI product work")
        assert scorer.score(job) > 0, title


def test_true_intern_titles_still_zeroed():
    scorer = _scorer()
    for title in ["Product Management Intern",
                  "Software Engineering Internship",
                  "Interns - AI Team"]:
        job = _job(title=title, description="AI product work")
        assert scorer.score(job) == 0, title


def test_other_junior_signals_still_zeroed():
    scorer = _scorer()
    for title in ["Junior Product Manager",
                  "Project Coordinator",
                  "Entry Level Product Manager",
                  "Entry-Level Analyst"]:
        job = _job(title=title, description="AI product work")
        assert scorer.score(job) == 0, title


# ---------------------------------------------------------------------------
# Remote marker in the title (hard filter 1 + remote-confidence tier)
# ---------------------------------------------------------------------------


def test_remote_title_survives_city_location_cap(monkeypatch):
    """Regression (job 7603): '(Remote)' in the title must count as a remote
    signal even when the location names a mismatch-cap city — the job escapes
    LOCATION_MISMATCH_CAP entirely, not just the old zeroing."""
    patch_scorer_keywords(monkeypatch)
    scorer = _scorer()
    job = _job(
        title="Associate Director, Product Management - Contracting (CTFS) (Remote)",
        location="New York, NY",
        description="Senior product leadership for the Clinical Trial Financial "
                    "Suite, applying AI and ML to contracting workflows.",
    )
    assert scorer.score(job) > 40


def test_city_location_without_remote_marker_still_capped():
    """Without any remote marker the location mismatch cap still applies —
    the title fix must not loosen the gate for genuinely onsite roles."""
    scorer = _scorer()
    job = _job(title="Director, Product Management",
               location="New York, NY",
               description="Onsite product leadership role.")
    assert scorer.score(job) <= 40


def test_remote_title_scores_lower_confidence_than_remote_location():
    """A title-only remote marker earns the mid confidence tier (+10), below
    an explicit remote location (+20) and above no signal at all (+5)."""
    scorer = _scorer()
    desc = "AI product work"
    loc_remote = _job(title="Director of Product", location="Remote",
                      description=desc)
    title_remote = _job(title="Director of Product (Remote)",
                        location="Irvine, CA", description=desc)
    no_signal = _job(title="Director of Product", location="Irvine, CA",
                     description=desc)
    assert scorer.score(loc_remote) - scorer.score(title_remote) == 10
    assert scorer.score(title_remote) - scorer.score(no_signal) == 5


def test_explain_mentions_remote_title():
    scorer = _scorer()
    job = _job(title="Director of Product (Remote)",
               location="Irvine, CA", description="AI product work")
    assert "Title lists the position as remote" in scorer.explain(job)


def test_explain_mentions_local_area(monkeypatch):
    scorer = _local_scorer(monkeypatch)
    job = _job(title="Director of Digital Transformation",
               location="Springfield, IL", description="AI initiatives")
    assert "Springfield" in scorer.explain(job)


def test_no_local_config_means_no_local_behavior():
    scorer = _scorer()  # no local_locations key
    assert not scorer._is_local("springfield, il")


# ---------------------------------------------------------------------------
# $300K location exception (2026-07-13 spec): LOCATION_MISMATCH_CAP is not
# applied when the posted top-of-band reaches $300K (US locations only).
# ---------------------------------------------------------------------------

_ONSITE_AI_DIRECTOR = dict(
    title="Director of Product, AI",
    description="AI ML LLM product leadership. Enterprise automation platform.",
)


def test_location_cap_skipped_for_high_comp_us_job(monkeypatch):
    patch_comp_bars(monkeypatch)            # canonical comp bars
    patch_relocation_exception(monkeypatch)  # canonical $300K exception band
    scorer = _scorer()
    capped = scorer.score(_job(**_ONSITE_AI_DIRECTOR,
                               location="San Francisco, CA",
                               salary_min=200_000, salary_max=250_000))
    uncapped = scorer.score(_job(**_ONSITE_AI_DIRECTOR,
                                 location="San Francisco, CA",
                                 salary_min=305_000, salary_max=385_000))
    assert capped == 40   # LOCATION_MISMATCH_CAP bites
    assert uncapped > 40  # exception lifts the cap; other dims score freely


def test_location_cap_stays_for_high_comp_non_us_job():
    scorer = _scorer()
    j = _job(**_ONSITE_AI_DIRECTOR, location="London",
             salary_min=350_000, salary_max=400_000)
    assert scorer.score(j) <= 40


def test_explain_survives_neutral_keyword_lists(monkeypatch):
    """Public-tree condition: empty keyword lists must not KeyError in explain()
    (it hard-indexed _KEYWORD_PATTERNS['ai'...] — crash-per-job in save_jobs)."""
    import engine.scorer as scorer_mod
    monkeypatch.setattr(scorer_mod, "_KEYWORD_PATTERNS", {})
    s = JobScorer({"scoring": {"alert_threshold": 60}})
    job = _job(title="Director of Operations", location="Remote",
               description="Own the operations roadmap.")
    assert isinstance(s.explain(job), str)
