"""Shared PM-shaped prefilter fixture for tests that exercise engine.llm_scorer's
rules-based pre-filter and caps.

engine/llm_scorer.py now sources its seven prefilter/cap regexes from
profile_policy, which compiles them from config (owner tree) or leaves them
None (neutral/public tree — rule disabled, defer to the LLM). Tests that
assert specific prefilter behavior need the PM-shaped patterns regardless of
which tree they're running in, so they patch engine.llm_scorer's module
attributes directly via patch_pm_prefilter(monkeypatch).

The patterns below are copied verbatim from engine/llm_scorer.py as of the
commit that moved them to profile_policy (Task 6) — they must stay in sync
with the owner's config-sourced values for these tests to remain meaningful.
"""
import re

# Engineering/non-PM title keywords
_ENG_TITLE_KEYWORDS = re.compile(
    r"\b(software engineer|sre|devops|data engineer|cuda|kernel|"
    r"backend engineer|frontend engineer|full[ -]?stack|"
    r"ml engineer|machine learning engineer|infrastructure engineer|"
    r"platform engineer|systems engineer|security engineer|"
    r"network engineer|database administrator|dba|qa engineer|"
    r"test engineer|embedded engineer|firmware|hardware engineer)\b",
    re.IGNORECASE,
)
_PM_KEYWORDS = re.compile(r"\b(product|manager|management|director|vp|head)\b", re.IGNORECASE)

# Non-PM adjacent roles — relevant enough to score but should be capped low
# These pass LLM scoring but hit a ceiling since they're not PM leadership roles
_NON_PM_ADJACENT_TITLES = re.compile(
    r"\b(business analyst|systems analyst|data analyst|"
    r"analyst ii|analyst 2|analyst iii|analyst 3|"
    r"^analyst\b|coordinator|specialist)\b",
    re.IGNORECASE,
)

# AI/ML signal keywords — used for post-parse ai_domain_fit cap
_AI_SIGNAL_KEYWORDS = re.compile(
    r"\b(ai\b|ml\b|llm|llms|genai|nlp|machine.?learning|artificial.?intelligence|"
    r"generative\s+ai|generative\s+model|foundation\s+model|large\s+language|"
    r"automation|intelligent|predictive\s+model|neural\s+net|deep\s+learn|"
    r"chatbot|conversational\s+ai|natural\s+language\s+processing|data\s+science|"
    r"computer\s+vision|reinforcement\s+learning|vector\s+search|embedding|"
    r"rag\b|retrieval.augmented|copilot|agentic)\b",
    re.IGNORECASE,
)

# Clearly non-PM roles
_NON_PM_TITLES = re.compile(
    r"\b(payroll|accountant|accounting|recruiter|recruiting|talent acquisition|"
    r"legal counsel|attorney|lawyer|nurse|physician|doctor|pharmacist|"
    r"dental|physical therapist|social worker|financial advisor|"
    r"insurance agent|real estate agent|truck driver|warehouse)\b",
    re.IGNORECASE,
)

# Sales/BD titles — not PM roles. Exclude if "product" is also in the title
# (e.g. "Sales Product Manager" is still a PM role).
_SALES_BD_TITLES = re.compile(
    r"\b(account executive|account manager|"
    r"business development (manager|director|representative)|"
    r"sales (director|manager|representative|engineer)|"
    r"vp of sales|chief revenue officer|revenue operations manager)\b",
    re.IGNORECASE,
)

# Solutions/Customer Success — field/post-sales roles, not PM.
# Exclude if "product" appears in title (e.g. "Product Solutions Architect").
_SOLUTIONS_CS_TITLES = re.compile(
    r"\b(solutions engineer|customer success manager|"
    r"implementation (manager|engineer)|"
    r"field (engineer|application engineer)|"
    r"technical account manager)\b",
    re.IGNORECASE,
)

PM_PREFILTER_PATTERNS = {
    "_NON_PM_TITLES": _NON_PM_TITLES,
    "_SALES_BD_TITLES": _SALES_BD_TITLES,
    "_SOLUTIONS_CS_TITLES": _SOLUTIONS_CS_TITLES,
    "_ENG_TITLE_KEYWORDS": _ENG_TITLE_KEYWORDS,
    "_NON_PM_ADJACENT_TITLES": _NON_PM_ADJACENT_TITLES,
    "_PM_KEYWORDS": _PM_KEYWORDS,
    "_AI_SIGNAL_KEYWORDS": _AI_SIGNAL_KEYWORDS,
}


def patch_pm_prefilter(monkeypatch):
    """Give llm_scorer the PM-shaped prefilter regardless of ambient config."""
    import engine.llm_scorer as m
    for attr, pat in PM_PREFILTER_PATTERNS.items():
        monkeypatch.setattr(m, attr, pat)


# ---------------------------------------------------------------------------
# Tailor anchors (Task 9) — resume_tailor.tailor_engine sources its title-line
# and skill-subcategory-label anchors from profile_policy (config.yaml
# policy.tailor.*), which is blank/empty in a neutral tree. Tests that need a
# configured anchor regardless of which tree they're running in patch the
# module attributes directly via patch_tailor_anchors(monkeypatch).
# ---------------------------------------------------------------------------

FIXTURE_TITLE_LINE = "SENIOR FIXTURE ROLE | EXAMPLE DOMAIN"

# A 2-entry stand-in for tailor_engine._SKILL_SUBCATEGORY_LABELS. The second
# entry mimics the invented-but-real-shaped skills-line quirk documented on
# FIXTURE_MASTER_SKILL_LABELS below (a document may store a visual-lookalike
# capital-I as lowercase-l) so tests exercising that I/l-normalization path
# still test something real.
FIXTURE_SKILL_LABELS = {
    "Core Practices": "Core Practices:",
    "Integration Skills": "lntegration Skills:",  # 'I' stored as 'l' — see tailor_engine._norm_il
}


def patch_tailor_anchors(monkeypatch):
    """Give tailor_engine a configured title/skill-label anchor regardless of
    ambient config, so tests pass the same way in the owner tree and a
    neutral tree (where profile_policy.TAILOR_MASTER_TITLE_LINE is "")."""
    import resume_tailor.tailor_engine as m
    monkeypatch.setattr(m, "_MASTER_TITLE_LINE", FIXTURE_TITLE_LINE)
    monkeypatch.setattr(m, "_SKILL_SUBCATEGORY_LABELS", FIXTURE_SKILL_LABELS)


# ---------------------------------------------------------------------------
# Local-area state pattern (Task 10) — local_area.build_local_area_regex()
# falls back to a module-level LOCAL_STATE_PATTERN name when a caller doesn't
# pass state_pattern explicitly. That name is bound once, at import time, by
# local_area.py's `from profile_policy import LOCAL_CITIES, LOCAL_STATE_PATTERN`
# — so it lives in *local_area*'s namespace, not profile_policy's. Every
# caller that doesn't thread state_pattern through explicitly (JobScorer,
# scrapers/*.is_local_commuter_area, engine.llm_scorer's module-level
# _LOCAL_AREA_RE) reads local_area.LOCAL_STATE_PATTERN. In the owner's tree
# that's a real state pattern; in a neutral/public tree profile_policy (and
# therefore local_area) resolves it to "" and build_local_area_regex()
# returns None (Task 3's blank-pattern-disables-matching sentinel) — so any
# test asserting *local* behavior through one of those callers needs this
# patched regardless of which tree it runs in. Patching profile_policy's copy
# alone would NOT work: local_area already captured its own module-level name
# at import time and never re-reads profile_policy afterward.
# ---------------------------------------------------------------------------


def patch_state_pattern(monkeypatch, pattern=r"illinois|il\b"):
    """Give local_area's ambient state pattern a real value regardless of
    ambient config, so tests exercising *_is_local*/`is_local_commuter_area`/
    `_LOCAL_AREA_RE` through their default (no explicit state_pattern) code
    path behave the same in the owner tree and a neutral tree.

    Clears `_compile`'s lru_cache before patching, as insurance against a
    stale entry from an earlier test. No teardown clear is needed: `_compile`
    is keyed on `(cities, resolved_pattern)`, so an entry cached under one
    pattern can never be returned for a lookup under a different (ambient or
    patched) pattern — `monkeypatch.setattr`'s own automatic teardown, which
    restores `LOCAL_STATE_PATTERN` to its pre-test value, is sufficient for
    later tests to resolve their own correct cache key. (pytest's
    `monkeypatch` fixture has no public `addfinalizer` of its own — only the
    `request` fixture does — so a teardown-side clear would need every caller
    to thread `request` through as well, for a safety net this method's
    keying already makes redundant.)"""
    import local_area
    import profile_policy
    local_area._compile.cache_clear()
    monkeypatch.setattr(profile_policy, "LOCAL_STATE_PATTERN", pattern)
    monkeypatch.setattr(local_area, "LOCAL_STATE_PATTERN", pattern)


# ---------------------------------------------------------------------------
# Scraper title gates (Task 11) — the scraper modules source their title/role
# gates from profile_policy (config.yaml policy.title_gates.* / policy.wttj.* /
# policy.hn.*), which is EMPTY in a neutral tree: greenhouse's baseline gate
# rejects every title (so smartrecruiters/successfactors, which reuse it, fall
# back to profile target_titles alone), WTTJ yields nothing (empty keyword
# list; empty default query set), and HN matches nothing (HN_ROLE_RE is None).
# Tests exercising the parsing/filter MECHANICS need the PM-shaped gates
# regardless of which tree they run in, so they patch the module attributes
# directly. The values below are copied verbatim from the scraper modules as
# of the commit that moved them to profile_policy (Task "scraper geo/title/role
# gates from profile_policy") — mirroring the patch_pm_prefilter pattern.
# ---------------------------------------------------------------------------

# scrapers.greenhouse.PRODUCT_TITLE_KEYWORDS / AI_TITLE_KEYWORDS. greenhouse's
# _is_relevant_title / _title_passes_filter read these module globals at call
# time; smartrecruiters.py and successfactors.py import _title_passes_filter
# from greenhouse, so patching greenhouse's globals fixes all three scrapers.
GREENHOUSE_PRODUCT_TITLE_KEYWORDS = [
    "product manager", " pm ", "pm,", "(pm)", "product lead",
    "director of product", "vp product", "vp of product", "head of product",
    "head of ai", "director ai", "director of ai", "chief product", "chief ai",
    "principal product", "group product", "senior product",
]
GREENHOUSE_AI_TITLE_KEYWORDS = [
    "ai", "ml", "llm", "genai", "generative", "nlp", "machine learning",
    "artificial intelligence", "automation", "conversational",
]


def patch_greenhouse_gates(monkeypatch):
    """Give greenhouse (and smartrecruiters/successfactors, which reuse its
    _title_passes_filter) the PM-shaped baseline title gate regardless of
    ambient config, so tests exercising the title filter pass the same way in
    the owner tree and a neutral tree (where PRODUCT_TITLE_KEYWORDS is [])."""
    import scrapers.greenhouse as m
    monkeypatch.setattr(m, "PRODUCT_TITLE_KEYWORDS", GREENHOUSE_PRODUCT_TITLE_KEYWORDS)
    monkeypatch.setattr(m, "AI_TITLE_KEYWORDS", GREENHOUSE_AI_TITLE_KEYWORDS)


# scrapers.wttj._TITLE_KEYWORDS / _DEFAULT_WTTJ_QUERIES (imported under those
# underscore aliases from profile_policy.WTTJ_TITLE_KEYWORDS / WTTJ_DEFAULT_QUERIES).
WTTJ_TITLE_KEYWORDS = [
    "product manager", "product lead", "lead product manager",
    "head of product", "director of product", "product director",
    "vp product", "vp of product", "chief product", "cpo",
    "principal product", "group product", "staff product",
    "head of ai", "director of ai", "vp of ai", "vp ai", "ai product",
    "head of machine learning", "head of data",
]
WTTJ_DEFAULT_QUERIES = [
    "AI Product Manager", "Senior Product Manager", "Product Manager",
    "Head of Product", "Director of Product", "Head of AI",
    "Principal Product Manager", "GenAI Product",
]


def patch_wttj_gates(monkeypatch):
    """Give the WTTJ scraper its PM-shaped title keywords and default query set
    regardless of ambient config. Patch BEFORE constructing WTTJScraper: the
    scraper binds self.queries = config['wttj_queries'] or _DEFAULT_WTTJ_QUERIES
    in __init__, while _relevant() reads _TITLE_KEYWORDS at call time."""
    import scrapers.wttj as m
    monkeypatch.setattr(m, "_TITLE_KEYWORDS", WTTJ_TITLE_KEYWORDS)
    monkeypatch.setattr(m, "_DEFAULT_WTTJ_QUERIES", WTTJ_DEFAULT_QUERIES)


# scrapers.hn_whoishiring._ROLE_RE (imported from profile_policy.HN_ROLE_RE,
# which is None in a neutral tree → every comment fails the role gate).
HN_ROLE_RE = re.compile(
    r"(group product manager|principal product manager|senior product manager|"
    r"product manager|product lead|head of product|director of product|"
    r"director, product|product director|vp,? ?product|vp of product|"
    r"head of ai|ai product)",
    re.IGNORECASE,
)


def patch_hn_gates(monkeypatch):
    """Give the HN scraper its PM-shaped role gate regardless of ambient config
    (neutral tree: HN_ROLE_RE is None → nothing matches). HN_REMOTE_ONLY is
    already True by default, but pin it too so the fixture is self-contained."""
    import scrapers.hn_whoishiring as m
    monkeypatch.setattr(m, "_ROLE_RE", HN_ROLE_RE)
    monkeypatch.setattr(m, "HN_REMOTE_ONLY", True)


# ---------------------------------------------------------------------------
# Keyword scorer (Task 11) — engine.scorer sources PRIMARY_KEYWORDS /
# SECONDARY_KEYWORDS from profile_policy (config.yaml policy.keywords.*), empty
# in a neutral tree so every job earns 0 keyword points. _KEYWORD_PATTERNS is
# built ONCE at import from those lists, so it must be rebuilt in lock-step.
# Values copied verbatim from engine/scorer.py as of the commit that moved them
# to profile_policy ("scorer keywords/companies/comp bars from profile_policy").
# ---------------------------------------------------------------------------

SCORER_PRIMARY_KEYWORDS = [
    "ai", "ml", "llm", "genai", "nlp", "product manager", "conversational ai",
]
SCORER_SECONDARY_KEYWORDS = [
    "kubernetes", "terraform", "google cloud", "digital transformation",
    "automation", "enterprise", "chatbot", "agile",
]


def patch_scorer_keywords(monkeypatch):
    """Give engine.scorer its PM/AI-shaped keyword lists AND the matching
    compiled-pattern dict (built at import from those lists) regardless of
    ambient config, so keyword-point and location-cap deltas are identical in
    the owner tree and a neutral tree (where the lists are [])."""
    import engine.scorer as m
    monkeypatch.setattr(m, "PRIMARY_KEYWORDS", SCORER_PRIMARY_KEYWORDS)
    monkeypatch.setattr(m, "SECONDARY_KEYWORDS", SCORER_SECONDARY_KEYWORDS)
    monkeypatch.setattr(m, "_KEYWORD_PATTERNS", {
        kw: m._kw_regex(kw)
        for kw in SCORER_PRIMARY_KEYWORDS + SCORER_SECONDARY_KEYWORDS
    })


# ---------------------------------------------------------------------------
# ATS standard headers (Task 11) — resume_tailor.ats_checker builds
# STANDARD_HEADERS from a generic base set plus profile_policy.TAILOR_EXTRA_ATS_
# HEADERS (config.yaml policy.tailor.extra_ats_headers), empty in a neutral
# tree. The clean-resume fixture uses a combined "core competencies & technical
# skills" header the generic set doesn't carry, so tests that expect it to be
# standard patch STANDARD_HEADERS the same way the module composes it.
# ---------------------------------------------------------------------------

FIXTURE_EXTRA_ATS_HEADERS = ["core competencies & technical skills"]


def patch_ats_headers(monkeypatch):
    """Add the owner-style combined ATS section header to ats_checker's
    STANDARD_HEADERS regardless of ambient config, recomposing the set exactly
    as the module does (_BASE_STANDARD_HEADERS | lowercased extras)."""
    import resume_tailor.ats_checker as m
    monkeypatch.setattr(
        m, "STANDARD_HEADERS",
        m._BASE_STANDARD_HEADERS | {h.lower() for h in FIXTURE_EXTRA_ATS_HEADERS})


# ---------------------------------------------------------------------------
# Comp bars + the $300K relocation exception (owner-pins fix) — engine.scorer
# binds the comp bars as module globals at import (from profile_policy), and
# salary_rules.is_high_comp_exception reads LOCATION_EXCEPTION_MIN_COMP as its
# module global at call time. Both resolve from config.yaml's policy.comp.*
# (owner tree) or _DEFAULTS (neutral tree) — but a PERSONALIZED config sets its
# own dollar bars, which would move the bands these behavior tests assert
# against. Any test pinning an exact comp-point result or an exact
# exception-band boundary therefore patches the canonical fixture values (the
# profile_policy _DEFAULTS / shipped example) so it passes identically in the
# owner tree, a neutral tree, AND a personalized tree.
# ---------------------------------------------------------------------------

FIXTURE_COMP_TARGET = 220_000
FIXTURE_LOCAL_FULL_COMP = 150_000
FIXTURE_LOCAL_PARTIAL_COMP = 120_000
FIXTURE_REMOTE_PARTIAL_COMP = 180_000
FIXTURE_RELOCATION_EXCEPTION_COMP = 300_000


def patch_comp_bars(monkeypatch):
    """Pin engine.scorer's four comp bars to the canonical fixture values
    regardless of ambient config, so _score_compensation point assertions
    (full=20 / partial=10 / 0) are identical in the owner tree, a neutral tree,
    and a personalized tree. _score_compensation reads these as module globals,
    so patch engine.scorer's copies (not profile_policy's)."""
    import engine.scorer as m
    monkeypatch.setattr(m, "COMP_TARGET", FIXTURE_COMP_TARGET)
    monkeypatch.setattr(m, "LOCAL_FULL_COMP", FIXTURE_LOCAL_FULL_COMP)
    monkeypatch.setattr(m, "LOCAL_PARTIAL_COMP", FIXTURE_LOCAL_PARTIAL_COMP)
    monkeypatch.setattr(m, "REMOTE_PARTIAL_COMP", FIXTURE_REMOTE_PARTIAL_COMP)


def patch_relocation_exception(monkeypatch):
    """Pin salary_rules.LOCATION_EXCEPTION_MIN_COMP to the canonical $300K
    fixture threshold regardless of ambient config. is_high_comp_exception reads
    this module global at call time, so patching salary_rules' copy alone covers
    every caller — salary_rules directly, engine.scorer.score (imported
    function), and engine.llm_scorer.prefilter_job. A personalized tree sets its
    own policy.comp.relocation_exception, which would otherwise move the band
    these exception tests assert against."""
    import salary_rules as m
    monkeypatch.setattr(m, "LOCATION_EXCEPTION_MIN_COMP", FIXTURE_RELOCATION_EXCEPTION_COMP)


# ---------------------------------------------------------------------------
# Tailor master anchors, owner-shaped (Task 11) — a few tailor tests assert
# against a real master resume's multi-subcategory skills layout (a skills
# line stored with a visual-lookalike "Al" quirk) and the master title line.
# The 2-entry FIXTURE_SKILL_LABELS above deliberately omits "AI Platforms"
# (its consumers assert that key is dropped), so those tests need the fuller
# owner-shaped label map instead. Values are invented, same-shaped stand-ins
# for the pre-refactor resume_tailor.tailor_engine module constants (not the
# owner's real resume headers).
#
# NOTE: resume_tailor.pipeline imports _MASTER_TITLE_LINE / _SKILL_SUBCATEGORY_
# LABELS into ITS OWN namespace, so a pipeline test must patch the pipeline
# module's copies (patch_pipeline_title_line), not tailor_engine's.
# ---------------------------------------------------------------------------

FIXTURE_MASTER_SKILL_LABELS = {
    "AI Platforms": "Al Platforms:",  # doc stores 'Al' (capital A + lowercase l)
    "Delivery Toolkit": "Delivery Toolkit:",
    "Team Leadership": "Team Leadership:",
    "Process Design": "Process Design:",
}


def patch_tailor_master_labels(monkeypatch, title_line=FIXTURE_TITLE_LINE):
    """Give tailor_engine the owner-shaped master title line and multi-entry
    skill-subcategory label map (including "AI Platforms") regardless of
    ambient config, so tests exercising the full skills-layout tracking pass
    the same way in the owner tree and a neutral tree."""
    import resume_tailor.tailor_engine as m
    monkeypatch.setattr(m, "_MASTER_TITLE_LINE", title_line)
    monkeypatch.setattr(m, "_SKILL_SUBCATEGORY_LABELS", FIXTURE_MASTER_SKILL_LABELS)


def patch_pipeline_title_line(monkeypatch, title_line=FIXTURE_TITLE_LINE):
    """Give resume_tailor.pipeline a non-empty master title line regardless of
    ambient config. The pipeline passes its OWN _MASTER_TITLE_LINE into
    tailor_diff.compute_edit_verdicts; with the neutral "" it collapses the
    title's before-text to None, so a not-landed title is scored not_landed
    instead of modified. A real title line keeps the owner-tree verdict."""
    import resume_tailor.pipeline as m
    monkeypatch.setattr(m, "_MASTER_TITLE_LINE", title_line)
