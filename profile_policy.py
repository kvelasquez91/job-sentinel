"""Owner profile policy — a pure LOADER, not a values file.

Every owner-specific value (keywords, comp bars, geography, title gates,
rubric prose, prefilter patterns, tailor anchors) is read from config.yaml's
`policy:` / `profile:` / `dashboard:` sections, falling back per-key to the
generic `_DEFAULTS` below. The tracked tree therefore carries NO owner policy:
an absent or example config yields a neutral, functional configuration
(empty ATS/scraper title gates make those scrapers rely solely on config
`target_titles`, and empty wttj/hn config leaves those sources dormant; empty
prefilter patterns defer to the LLM; empty keyword lists contribute no keyword
score). See CUSTOMIZING.md in the public tree for the schema.

Mechanics live in the modules that import from here; do not add behavior.
"""
import os
import re
from typing import Optional

import yaml


def _load_config() -> dict:
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}
    except Exception:
        return {}


# Generic, functional defaults. Keys are dotted config paths. Comp defaults
# keep reasonable tech-market bars (non-identifying; the setup interview
# overwrites them). Empty gate/pattern/keyword values mean "feature dormant
# until configured" — dedicated tests pin these semantics.
_RUBRIC_ROLE_MATCH_DEFAULT = """\
1. role_match (0-30): Does this job match the candidate's target role family (see CANDIDATE PROFILE above)?
   - 25-30: Title and focus are squarely in the candidate's target role family at their target level
   - 15-24: Adjacent role — same discipline, different specialization
   - 5-14: Loosely related — overlapping skills but a different discipline
   - 0-4: A different profession entirely

   IMPORTANT: Judge the ROLE being hired for, not whether the candidate could stretch into it. When title and description disagree, weigh the description."""

_RUBRIC_SENIORITY_DEFAULT = """\
2. seniority_match (0-20): Does the job's seniority match the level the candidate targets (see CANDIDATE PROFILE)?

   Score based on the ACTUAL TITLE of the role being posted, not the candidate's seniority:
   - 18-20: At or above the candidate's target seniority
   - 12-17: One step below the candidate's target
   - 8-11: Two steps below the target
   - 4-7: Mid-level IC roles when the candidate targets senior or leadership roles
   - 0-3: Junior, associate, entry-level, new-grad, or intern roles

   IMPORTANT: A bare "Manager" title with no seniority qualifier is mid-level, not leadership."""

_RUBRIC_REMOTE_LOCATION_DEFAULT = """\
3. remote_location (0-20): Is this remote-friendly, US-based, or in the candidate's local commuter area?
   - 18-20: Fully remote (US), remote-first, OR located in/near %%LOCAL_AREA_PROSE%%
   - 12-17: Hybrid US (NYC, SF, etc.) or remote with US restriction
   - 5-11: In-office only but major US city
   - 0-4: Non-US location (Japan, UK, Germany, India, Singapore, Dubai, UAE, Canada, Australia, etc.) or in-office non-US
   EXCEPTION: an in-office/hybrid US role whose posted salary (see the Salary line) has a top-of-band >= %%EXCEPTION_COMP%% scores 12-17, not 5-11 — the candidate will relocate within the US for %%EXCEPTION_COMP_K%%+ comp. Never apply this to non-US locations."""

_RUBRIC_DOMAIN_FIT_DEFAULT = """\
4. ai_domain_fit (0-20): Does the JOB DESCRIPTION explicitly involve the candidate's target domain (see CANDIDATE PROFILE)?

   CRITICAL RULE: Score this dimension solely on the JOB DESCRIPTION's content — NOT on the candidate's background. If the description shows no explicit signal of the candidate's target domain, the score MUST be 8 or below regardless of the candidate's qualifications. (The JSON key stays "ai_domain_fit" for schema compatibility; it measures domain fit generally.)

   - 17-20: The candidate's target domain is central to the role
   - 11-16: The description explicitly involves the domain as a significant part of the role
   - 5-10: Same broad industry, but no explicit signal of the target domain
   - 0-4: An unrelated domain with no connection to the candidate's target"""

_DEFAULTS = {
    "profile.key": "default",
    # Owner-repo CI pins (optional; blank/false = disabled = the neutral
    # default). A fork maintainer opts in once their rubric bytes and
    # resume_summary are tuned — see CUSTOMIZING.md "Owner pins". These gate
    # owner-only guard tests; personalization never requires setting them, so
    # a personalized fork keeps a green suite with both left at their defaults.
    "policy.owner_pins.rubric_sha": "",
    "policy.owner_pins.summary_mentions_relocation": False,
    "policy.geography.local_state_pattern": "",
    "policy.geography.linkedin_local_geo": "",
    "policy.geography.rubric_local_area_prose": "the candidate's local commuter area (see profile)",
    "policy.geography.rubric_local_area_short": "the candidate's local commuter area",
    "policy.geography.rubric_local_anchor": "local-market senior role",
    "policy.comp.target": 220_000,
    "policy.comp.local_full": 150_000,
    "policy.comp.local_partial": 120_000,
    "policy.comp.remote_partial": 180_000,
    "policy.comp.cap_low": 170_000,
    "policy.comp.cap_mid": 200_000,
    "policy.comp.relocation_exception": 300_000,
    "policy.keywords.primary": [],
    "policy.keywords.secondary": [],
    "policy.companies.high_paying": [],
    "policy.companies.priority": [],
    "policy.title_gates.baseline": [],
    "policy.title_gates.domain": [],
    "policy.wttj.title_keywords": [],
    "policy.wttj.default_queries": [],
    "policy.hn.role_pattern": "",
    "policy.hn.remote_only": True,
    "policy.rubric.role_match_block": _RUBRIC_ROLE_MATCH_DEFAULT,
    "policy.rubric.seniority_match_block": _RUBRIC_SENIORITY_DEFAULT,
    "policy.rubric.remote_location_block": _RUBRIC_REMOTE_LOCATION_DEFAULT,
    "policy.rubric.domain_fit_block": _RUBRIC_DOMAIN_FIT_DEFAULT,
    "policy.prefilter.non_target_titles": "",
    "policy.prefilter.sales_bd_titles": "",
    "policy.prefilter.solutions_cs_titles": "",
    "policy.prefilter.eng_title_keywords": "",
    "policy.prefilter.non_target_adjacent_titles": "",
    "policy.prefilter.target_keywords": "",
    "policy.prefilter.domain_signal_pattern": "",
    "policy.tailor.master_title_line": "",
    "policy.tailor.skill_subcategory_labels": {},
    "policy.tailor.extra_ats_headers": [],
    "dashboard.page_title": "Job Sentinel — Opportunities",
    "dashboard.comp_tiers": [150_000, 200_000, 250_000],
    "dashboard.profiles": {},
}


def _resolve(cfg: dict, dotted: str):
    """Nested config lookup by dotted path; falls back to _DEFAULTS[dotted]."""
    node = cfg
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return _DEFAULTS[dotted]
        node = node[part]
    return _DEFAULTS[dotted] if node is None else node


def _compile_or_none(pattern: str) -> Optional[re.Pattern]:
    """Compile a policy regex; empty/blank string means 'rule disabled'."""
    pattern = (pattern or "").strip()
    return re.compile(pattern, re.IGNORECASE) if pattern else None


_cfg = _load_config()

# --- Identity ---------------------------------------------------------------
PROFILE_KEY: str = str(_resolve(_cfg, "profile.key"))

# --- Owner-repo CI pins (optional) ------------------------------------------
# rubric_sha: when non-empty, the byte-identity guard test asserts the
# assembled fit-scoring system template hashes to THIS value (a fork maintainer
# pins their own rubric bytes once tuned); empty → the guard self-skips. See
# CUSTOMIZING.md "Owner pins".
OWNER_RUBRIC_SHA: str = str(_resolve(_cfg, "policy.owner_pins.rubric_sha"))
# summary_mentions_relocation: when True, the owner-lint test asserts the
# configured resume_summary literally contains the formatted relocation bar;
# False → that lint self-skips (the neutral default).
OWNER_SUMMARY_LINT: bool = bool(_resolve(_cfg, "policy.owner_pins.summary_mentions_relocation"))

# --- Geography --------------------------------------------------------------
LOCAL_CITIES: tuple = tuple(
    str(c).strip() for c in (_cfg.get("local_locations") or []) if str(c).strip()
)
LOCAL_STATE_PATTERN = str(_resolve(_cfg, "policy.geography.local_state_pattern"))
LINKEDIN_LOCAL_GEO = str(_resolve(_cfg, "policy.geography.linkedin_local_geo"))
RUBRIC_LOCAL_AREA_PROSE = str(_resolve(_cfg, "policy.geography.rubric_local_area_prose"))
RUBRIC_LOCAL_AREA_SHORT = str(_resolve(_cfg, "policy.geography.rubric_local_area_short"))
RUBRIC_LOCAL_ANCHOR = str(_resolve(_cfg, "policy.geography.rubric_local_anchor"))

# --- Compensation (USD/year) -------------------------------------------------
COMP_TARGET = int(_resolve(_cfg, "policy.comp.target"))
LOCAL_FULL_COMP = int(_resolve(_cfg, "policy.comp.local_full"))
LOCAL_PARTIAL_COMP = int(_resolve(_cfg, "policy.comp.local_partial"))
REMOTE_PARTIAL_COMP = int(_resolve(_cfg, "policy.comp.remote_partial"))
COMP_CAP_LOW = int(_resolve(_cfg, "policy.comp.cap_low"))
COMP_CAP_MID = int(_resolve(_cfg, "policy.comp.cap_mid"))
RELOCATION_EXCEPTION_COMP = int(_resolve(_cfg, "policy.comp.relocation_exception"))

# --- Profession: keyword scoring (engine/scorer.py) --------------------------
PRIMARY_KEYWORDS = list(_resolve(_cfg, "policy.keywords.primary"))
SECONDARY_KEYWORDS = list(_resolve(_cfg, "policy.keywords.secondary"))
HIGH_PAYING_COMPANIES = set(_resolve(_cfg, "policy.companies.high_paying"))
PRIORITY_COMPANIES = set(_resolve(_cfg, "policy.companies.priority"))

# --- Profession: scraper title gates -----------------------------------------
PRODUCT_TITLE_KEYWORDS = list(_resolve(_cfg, "policy.title_gates.baseline"))
AI_TITLE_KEYWORDS = list(_resolve(_cfg, "policy.title_gates.domain"))
WTTJ_TITLE_KEYWORDS = list(_resolve(_cfg, "policy.wttj.title_keywords"))
WTTJ_DEFAULT_QUERIES = list(_resolve(_cfg, "policy.wttj.default_queries"))
HN_ROLE_RE = _compile_or_none(str(_resolve(_cfg, "policy.hn.role_pattern")))
HN_REMOTE_ONLY = bool(_resolve(_cfg, "policy.hn.remote_only"))

# --- LLM rubric blocks (engine/llm_scorer.py splices these) -------------------
RUBRIC_ROLE_MATCH_BLOCK = str(_resolve(_cfg, "policy.rubric.role_match_block"))
RUBRIC_SENIORITY_MATCH_BLOCK = str(_resolve(_cfg, "policy.rubric.seniority_match_block"))
RUBRIC_REMOTE_LOCATION_BLOCK = str(_resolve(_cfg, "policy.rubric.remote_location_block"))
RUBRIC_DOMAIN_FIT_BLOCK = str(_resolve(_cfg, "policy.rubric.domain_fit_block"))

# --- Rules-based prefilter patterns (engine/llm_scorer.py) --------------------
PREFILTER_NON_TARGET_TITLES = _compile_or_none(str(_resolve(_cfg, "policy.prefilter.non_target_titles")))
PREFILTER_SALES_BD_TITLES = _compile_or_none(str(_resolve(_cfg, "policy.prefilter.sales_bd_titles")))
PREFILTER_SOLUTIONS_CS_TITLES = _compile_or_none(str(_resolve(_cfg, "policy.prefilter.solutions_cs_titles")))
PREFILTER_ENG_TITLE_KEYWORDS = _compile_or_none(str(_resolve(_cfg, "policy.prefilter.eng_title_keywords")))
PREFILTER_NON_TARGET_ADJACENT_TITLES = _compile_or_none(str(_resolve(_cfg, "policy.prefilter.non_target_adjacent_titles")))
PREFILTER_TARGET_KEYWORDS = _compile_or_none(str(_resolve(_cfg, "policy.prefilter.target_keywords")))
DOMAIN_SIGNAL_RE = _compile_or_none(str(_resolve(_cfg, "policy.prefilter.domain_signal_pattern")))

# --- Tailor anchors (resume_tailor/) -----------------------------------------
TAILOR_MASTER_TITLE_LINE = str(_resolve(_cfg, "policy.tailor.master_title_line"))
TAILOR_SKILL_SUBCATEGORY_LABELS = dict(_resolve(_cfg, "policy.tailor.skill_subcategory_labels"))
TAILOR_EXTRA_ATS_HEADERS = list(_resolve(_cfg, "policy.tailor.extra_ats_headers"))

# --- Dashboard UI values (dashboard/app.py serves these) ----------------------
DASHBOARD_PAGE_TITLE = str(_resolve(_cfg, "dashboard.page_title"))
DASHBOARD_COMP_TIERS = [int(v) for v in _resolve(_cfg, "dashboard.comp_tiers")]
DASHBOARD_PROFILES = dict(_resolve(_cfg, "dashboard.profiles"))
