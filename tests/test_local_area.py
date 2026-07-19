"""Tests for the shared local-commuter-area location matcher (`local_area`).

This module is the single source of truth for BOTH which cities count as
"local" and how a location string is matched against them, so that
`scrapers/greenhouse.py`, `engine/scorer.py`, and `engine/llm_scorer.py`
can never re-diverge (see `test_three_modules_classify_identically`).
"""
import pathlib
import re

import yaml

from local_area import LOCAL_COMMUTER_CITIES, build_local_area_regex, is_local_commuter_area

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

# A small, stable city subset used to exercise matcher MECHANICS (case-
# insensitivity, gap-width strictness, out-of-state rejection) independent of
# the real/config-driven LOCAL_COMMUTER_CITIES set, which the config-mirroring and
# cross-module-agreement tests below exercise separately.
_FIXTURE_CITIES = ("Springfield", "Riverton", "Fairview", "Lakewood", "Cedar Falls")
_STATE_PATTERN = r"illinois|il\b"


def _is_local_fixture(location, cities=_FIXTURE_CITIES):
    """Same contract as ``is_local_commuter_area``, but pinned to
    ``_STATE_PATTERN`` instead of threading through the ambient
    ``local_area.LOCAL_STATE_PATTERN`` default — ``is_local_commuter_area``
    itself takes no ``state_pattern`` argument, so a test calling it directly
    with non-empty location/cities would silently depend on which tree it
    runs in (a real state pattern here, "" — matcher disabled — in a neutral
    tree). Mirrors is_local_commuter_area's own empty-input short-circuit."""
    if not location or not cities:
        return False
    pat = build_local_area_regex(cities, state_pattern=_STATE_PATTERN)
    return bool(pat.search(location)) if pat else False


# --- build_local_area_regex ---------------------------------------------------

def test_build_returns_none_for_empty_cities():
    """No cities → no matcher → callers treat everything as non-local."""
    assert build_local_area_regex([], state_pattern=_STATE_PATTERN) is None
    assert build_local_area_regex(["", "   "], state_pattern=_STATE_PATTERN) is None


def test_build_returns_compiled_pattern():
    pat = build_local_area_regex(["Springfield"], state_pattern=_STATE_PATTERN)
    assert isinstance(pat, re.Pattern)


# --- is_local_commuter_area: positives -------------------------------------------------

def test_matches_city_directly_followed_by_state_context():
    for loc in [
        "Springfield, IL",
        "Springfield, Illinois, United States",
        "Cedar Falls, IL",
        "Fairview, IL - 203 E. Main Street (29319)",
        "riverton, il",          # case-insensitive
        "RIVERTON, IL",
    ]:
        assert _is_local_fixture(loc), loc


# --- is_local_commuter_area: strict gap (the tightening) -------------------------------

def test_strict_gap_rejects_indirect_state_context():
    """A city must be *directly* followed by the state — not separated by an
    arbitrary run of characters (the old loose ``[\\s\\S]{0,24}`` behavior)."""
    for loc in ["Fairview County, IL", "Springfield area, IL", "Lakewood Mill Rd, IL"]:
        assert not _is_local_fixture(loc), loc


def test_rejects_out_of_state_even_when_state_token_appears_nearby():
    """The loose gap produced false positives on out-of-state jobs whenever an
    'il' token happened to appear within 24 chars; the strict gap must not."""
    for loc in [
        "Springfield, OH",
        "Chicago, IL",                    # IL but not a listed city
        "Springfield, MO",
        "Springfield, OH (near IL)",
        "Lakewood, OH - some IL clients",
    ]:
        assert not _is_local_fixture(loc), loc


def test_empty_inputs_are_not_local():
    assert not is_local_commuter_area("", _FIXTURE_CITIES)
    assert not is_local_commuter_area("Springfield, IL", [])


# --- canonical city set -------------------------------------------------------

def test_canonical_cities_mirror_config_local_locations():
    """LOCAL_COMMUTER_CITIES is the config-less fallback for LLMScorer; it must
    stay in lockstep with config.yaml's `local_locations` so the LLM scorer
    classifies the same cities as JobScorer/the scrapers (which read config)."""
    cfg = yaml.safe_load((_REPO_ROOT / "config.yaml").read_text()) or {}
    assert {c.lower() for c in LOCAL_COMMUTER_CITIES} == {
        c.lower() for c in (cfg.get("local_locations") or [])
    }


# --- consumers all route through this one helper (anti-drift) -----------------

def test_greenhouse_reexports_shared_is_local_commuter_area():
    """The scraper must not keep its own copy of the matcher — the name it
    exports (and that smartrecruiters/workday import from it) IS this one."""
    import local_area
    import scrapers.greenhouse as gh

    assert gh.is_local_commuter_area is local_area.is_local_commuter_area


def test_llm_scorer_city_set_aligned_to_canonical():
    """LLMScorer's `_LOCAL_AREA_RE` must be the exact same compiled matcher
    everyone else gets for the canonical city set — not just an equivalent one.
    `build_local_area_regex` is `lru_cache`d on the normalized (cities,
    state_pattern) tuple, so calling it again with `LOCAL_COMMUTER_CITIES` returns
    the SAME object `_LOCAL_AREA_RE` already is (or both are None, if no
    cities are configured)."""
    from engine.llm_scorer import _LOCAL_AREA_RE

    expected = build_local_area_regex(LOCAL_COMMUTER_CITIES)
    assert _LOCAL_AREA_RE is expected or (_LOCAL_AREA_RE is None and expected is None)


# A battery spanning normal formats, the strict-gap cases, out-of-state false
# positives, and the previously-divergent city-set members. Every consumer must
# return the SAME verdict on every one of these.
_AGREEMENT_CASES = [
    "Springfield, IL",
    "Springfield, Illinois, United States",
    "Cedar Falls, IL",
    "Fairview, IL - 203 E. Main Street (29319)",
    "Millbrook, IL",
    "Riverton, IL",
    "Springfield, OH",
    "Chicago, IL",
    "Austin, TX",
    "Remote",
    "",
    "Fairview County, IL",
    "Springfield area, IL",
    "Lakewood Mill Rd, IL",
    "Springfield, OH (near IL)",
    "Lakewood, OH - some IL clients",
    "Danville, IL",
    "Meadowview, IL",
    "Milltown, IL",
]


def test_three_modules_classify_identically():
    """Regression guard: greenhouse's `is_local_commuter_area`, scorer's `_is_local`,
    and llm_scorer's `_LOCAL_AREA_RE` must agree on every case when handed the
    same city set. Fails loudly the moment any of them re-diverges."""
    from engine.llm_scorer import _LOCAL_AREA_RE
    from engine.scorer import JobScorer
    from scrapers.greenhouse import is_local_commuter_area as gh_is_local_commuter_area

    scorer = JobScorer(
        {"local_locations": list(LOCAL_COMMUTER_CITIES), "scoring": {"alert_threshold": 60}}
    )
    for loc in _AGREEMENT_CASES:
        gh = gh_is_local_commuter_area(loc, LOCAL_COMMUTER_CITIES)
        sc = bool(scorer._is_local(loc.lower()))
        llm = bool(_LOCAL_AREA_RE.search(loc)) if _LOCAL_AREA_RE else False
        assert gh == sc == llm, (
            f"divergence on {loc!r}: greenhouse={gh} scorer={sc} llm={llm}"
        )


# --- neutral-default safety ---------------------------------------------------

def test_empty_state_pattern_disables_matching():
    """A blank state pattern must NOT compile to an empty alternation (which
    would match every city regardless of state) — it disables local matching."""
    assert build_local_area_regex(["Springfield"], state_pattern="") is None
    assert build_local_area_regex(["Springfield"], state_pattern="   ") is None


def test_cities_with_default_empty_pattern_do_not_match():
    """With profile_policy's neutral default ('' state pattern), even a listed
    city never matches — guards the public-tree fresh-clone condition."""
    import profile_policy
    if profile_policy.LOCAL_STATE_PATTERN.strip():
        import pytest
        pytest.skip("owner config defines a state pattern")
    assert not is_local_commuter_area("Springfield, IL", ["Springfield"])
