"""Shared salary plausibility rules: one cap, one range sanitizer.

Wide executive ranges (low end >= $100K) must survive; junk ranges that
start at benefit-level numbers ("$40K-$500K") still get the high end dropped.
"""
from salary_rules import MAX_BASE_SALARY, extract_salary_regex, sanitize_salary_range


def test_cap_is_one_million():
    assert MAX_BASE_SALARY == 1_000_000


def test_wide_exec_range_kept():
    # Netflix-style band: ratio 4.8, both plausible
    assert sanitize_salary_range(190_000, 920_000) == (190_000, 920_000)


def test_wide_range_with_solid_floor_kept():
    # ratio 7 but low end >= $100K -> real executive band, keep it
    assert sanitize_salary_range(100_000, 700_000) == (100_000, 700_000)


def test_junk_wide_range_high_end_dropped():
    # ratio 12.5 with a benefit-level low end -> junk, drop high end
    assert sanitize_salary_range(40_000, 500_000) == (40_000, 40_000)


def test_above_cap_high_end_dropped():
    assert sanitize_salary_range(200_000, 1_200_000) == (200_000, 200_000)


def test_regex_extracts_wide_exec_range():
    lo, hi = extract_salary_regex("Pay range: $190,000 - $920,000 annually")
    assert (lo, hi) == (190_000.0, 920_000.0)


def test_regex_still_strips_401k():
    assert extract_salary_regex("We offer 401(k) matching and dental") == (None, None)


def test_llm_scorer_reexports_shared_rules():
    from engine.llm_scorer import MAX_BASE_SALARY as m, extract_salary_regex as f
    assert m == 1_000_000 and f is extract_salary_regex


# ---------------------------------------------------------------------------
# is_high_comp_exception — the $300K location exception (2026-07-13 spec)
# ---------------------------------------------------------------------------
from salary_rules import (
    LOCATION_EXCEPTION_MIN_COMP,
    NON_US_LOCATIONS,
    is_high_comp_exception,
    is_us_location,
)
from policy_fixtures import patch_relocation_exception


def test_exception_threshold_default_and_alias():
    """The default relocation-exception bar is $300K, and salary_rules' public
    name aliases profile_policy's resolved value (identity, both trees)."""
    import profile_policy
    assert profile_policy._DEFAULTS["policy.comp.relocation_exception"] == 300_000
    assert LOCATION_EXCEPTION_MIN_COMP == profile_policy.RELOCATION_EXCEPTION_COMP


def test_top_of_band_at_threshold_qualifies(monkeypatch):
    patch_relocation_exception(monkeypatch)  # pin the $300K band vs a personalized bar
    # $200K-$300K: the band REACHES $300K — qualifies (top-of-band rule).
    assert is_high_comp_exception(200_000, 300_000, "San Francisco, CA")


def test_just_below_threshold_does_not_qualify(monkeypatch):
    patch_relocation_exception(monkeypatch)  # pin the $300K band vs a personalized bar
    assert not is_high_comp_exception(200_000, 299_999, "San Francisco, CA")


def test_min_only_posting_qualifies(monkeypatch):
    patch_relocation_exception(monkeypatch)  # pin the $300K band vs a personalized bar
    # Single posted number >= threshold, no max.
    assert is_high_comp_exception(310_000, None, "New York, NY")


def test_no_posted_salary_never_qualifies():
    assert not is_high_comp_exception(None, None, "San Francisco, CA")
    assert not is_high_comp_exception(0, 0, "San Francisco, CA")


def test_non_us_location_never_qualifies():
    assert not is_high_comp_exception(350_000, 500_000, "London")
    assert not is_high_comp_exception(350_000, 500_000, "Toronto, ON")
    assert not is_high_comp_exception(350_000, 500_000, "Dubai, UAE")


def test_unlisted_foreign_location_does_not_qualify():
    # Regression (jobs 4154/4316): places absent from the NON_US_LOCATIONS
    # blocklist must still fail the US test — the exception requires positive
    # US evidence, not merely "not on the blocklist".
    assert not is_high_comp_exception(350_000, 500_000, "Ottawa, ON, CA")
    assert not is_high_comp_exception(350_000, 500_000, "Montreal, QC")
    assert not is_high_comp_exception(350_000, 500_000, "Dublin, County Dublin, Ireland")
    assert not is_high_comp_exception(350_000, 500_000, "Zurich, Switzerland")
    assert not is_high_comp_exception(350_000, 500_000, "CAN, ON, Work-at-Home")
    assert not is_high_comp_exception(350_000, 500_000, "Toronto, Ontario, Canada")


def test_unknown_or_empty_location_does_not_qualify():
    # The exception is a carve-out: no positive US evidence -> no exemption.
    # (Flipped 2026-07-14: neither the prefilter gate nor the keyword cap
    # ever consulted the predicate for empty/broad strings, so only the
    # predicate's own contract changes.)
    assert not is_high_comp_exception(None, 400_000, "")
    assert not is_high_comp_exception(None, 400_000, None)
    assert not is_high_comp_exception(None, 400_000, "5 Locations")


def test_us_location_formats_qualify(monkeypatch):
    patch_relocation_exception(monkeypatch)  # pin the $300K band vs a personalized bar
    # Every format below appears verbatim in the live DB on $300K+ rows.
    for loc in [
        "San Francisco, CA",
        "US, CA, Santa Clara",
        "Sunnyvale, us",
        "New York, New York, USA",
        "Los Gatos,California,United States of America",
        "Washington, D.C.",
        "Austin, Texas Metropolitan Area",
        "San Francisco Bay Area",
        "Greater Boston",
        "United States (Remote)",
    ]:
        assert is_high_comp_exception(300_000, 350_000, loc), loc


def test_is_us_location_positive_evidence():
    # State-name evidence outranks blocklist city/country words.
    assert is_us_location("Paris, Texas")
    assert is_us_location("Santa Fe, New Mexico")
    # Canadian province code makes trailing "CA" mean Canada, not California.
    assert not is_us_location("Ottawa, ON, CA")
    assert not is_us_location("")
    assert not is_us_location(None)
    assert not is_us_location("Remote")


def test_non_us_regex_moved_intact():
    # The regex moved from engine/llm_scorer.py — spot-check behavior.
    assert NON_US_LOCATIONS.search("London")
    assert not NON_US_LOCATIONS.search("San Francisco, CA")


# --- quiet flag: over-cap discards log at DEBUG instead of WARNING ----------
# The dashboard re-parses unscored rows on every poll; without quiet the same
# discard WARNING repeats forever (5,290/8,982 log lines on 2026-07-14).

def test_over_cap_discard_warns_by_default(caplog):
    import logging
    with caplog.at_level(logging.DEBUG, logger="salary_rules"):
        assert extract_salary_regex("Total comp up to $1,500,000") == (None, None)
    assert any(r.levelno == logging.WARNING and "MAX_BASE_SALARY" in r.getMessage()
               for r in caplog.records)


def test_quiet_demotes_over_cap_discard_to_debug(caplog):
    import logging
    with caplog.at_level(logging.DEBUG, logger="salary_rules"):
        assert extract_salary_regex(
            "Total comp up to $1,500,000", quiet=True) == (None, None)
    assert not any(r.levelno >= logging.WARNING for r in caplog.records)
    # Still observable at DEBUG for anyone diagnosing a specific row.
    assert any("MAX_BASE_SALARY" in r.getMessage() for r in caplog.records)


def test_quiet_flows_through_to_sanitize_range(caplog):
    # A junk-wide range reaches sanitize_salary_range's WARNING; quiet must
    # cover that log site too, with the same drop-the-high-end result.
    import logging
    with caplog.at_level(logging.DEBUG, logger="salary_rules"):
        lo, hi = extract_salary_regex("Pay: $40K - $500K", quiet=True)
    assert (lo, hi) == (40_000.0, 40_000.0)   # same result, quieter
    assert not any(r.levelno >= logging.WARNING for r in caplog.records)
