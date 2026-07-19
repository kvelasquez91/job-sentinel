"""
Job scoring engine. Scores each JobPosting 0-100 based on:
- Keyword relevance (configured policy keywords)
- Seniority match
- Compensation estimate
- Remote confidence
"""
import logging
import re
import sqlite3
from typing import List, Tuple

from local_area import build_local_area_regex
from profile_policy import (
    COMP_TARGET, HIGH_PAYING_COMPANIES, LOCAL_FULL_COMP, LOCAL_PARTIAL_COMP,
    PRIMARY_KEYWORDS, PRIORITY_COMPANIES, REMOTE_PARTIAL_COMP, SECONDARY_KEYWORDS,
)
from salary_rules import is_high_comp_exception
from scrapers.base import JobPosting

logger = logging.getLogger(__name__)

# HIGH_PAYING_COMPANIES and PRIORITY_COMPANIES (company-name policy lists) now
# live in profile_policy.py — see the import above.

# A named non-remote city caps the keyword score here rather than zeroing it:
# remote-only is a soft constraint the LLM blend + Filter Match judge far better
# than a crude city list, and a genuinely strong role/seniority/AI signal should
# survive to be weighed. The cap sits below alert_threshold (60) so keyword-only
# mode still won't surface an onsite role on its own.
# EXCEPTION (2026-07-13): posted top-of-band >= $300K exempts US onsite roles
# from this cap — see salary_rules.is_high_comp_exception (the US-only test is
# inside the predicate, so this branch stays safe under any blend weights).
LOCATION_MISMATCH_CAP = 40

# --- Layoff-penalty policy (shared by save-time scoring AND reblend) ---
# Both writers of jobs.score must apply the identical penalty, otherwise a
# --reblend silently strips it from the corpus (2026-07-12 audit finding).


def normalize_company_for_intel(name: str) -> str:
    """Company-intel name normalization: punctuation stripped, lowercased,
    whitespace NOT collapsed — must match engine/company_intel.py."""
    return re.sub(r"[^\w\s]", "", (name or "").lower()).strip()


def load_layoff_companies(conn) -> set:
    """Normalized names of companies flagged has_recent_layoffs=1.

    Returns an empty set when the company_insights table doesn't exist
    (fresh DBs, minimal test fixtures).
    """
    try:
        rows = conn.execute(
            "SELECT company_name_normalized FROM company_insights "
            "WHERE has_recent_layoffs = 1"
        ).fetchall()
        return {r[0] for r in rows}
    except sqlite3.OperationalError:
        return set()


def apply_layoff_penalty(score: int, company: str, layoff_companies: set,
                         penalty: int) -> int:
    """Config-gated caution penalty for layoff-flagged companies (0 disables)."""
    if penalty and score > 0 and normalize_company_for_intel(company) in layoff_companies:
        return max(0, score - penalty)
    return score

# Junior/entry level title signals - return 0 immediately
JUNIOR_SIGNALS = [
    "associate", "junior", "entry level", "entry-level", "intern", "internship",
    "coordinator",
]

# PRIMARY_KEYWORDS (10 pts each, max 30 pts) and SECONDARY_KEYWORDS (5 pts
# each, max 10 pts) now live in profile_policy.py — see the import above.

# Physical product-line roles at manufacturers ("Product Manager - Tires") are
# not digital product roles. Titles matching these terms are zeroed UNLESS the
# posting shows a digital signal somewhere. Light heuristic — the LLM pass is
# the real filter.
_PHYSICAL_PRODUCT_TITLE = re.compile(
    r"\b(tire|tyre|rubber|machinery|extrusion|molding|casting|textile|apparel|"
    r"flooring|fastener|bearing|compressor|turbine|powertrain|chassis|"
    r"packaging line|assembly line)s?\b"
)
_DIGITAL_SIGNALS = re.compile(
    r"\b(digital|software|data|ai|ml|llm|platform|app|cloud|saas|analytics|"
    r"automation|e-?commerce|iot|connected)\b"
)


def _kw_regex(kw: str) -> re.Pattern:
    """Whole-word matcher for a lowercase keyword, allowing an optional plural 's'.

    Word boundaries stop substring false positives ('ai' inside 'email',
    'ml' inside 'html'); the optional 's' keeps plurals matching ('LLMs',
    'chatbots', 'product managers').
    """
    return re.compile(r"\b" + re.escape(kw) + r"s?\b")


_KEYWORD_PATTERNS = {kw: _kw_regex(kw) for kw in PRIMARY_KEYWORDS + SECONDARY_KEYWORDS}

# Word-boundary matchers so 'intern' can't fire inside 'Internal'/'International';
# the optional plural keeps 'Interns'/'Associates' caught.
_JUNIOR_SIGNAL_PATTERNS = {s: _kw_regex(s) for s in JUNIOR_SIGNALS}

# 'Associate Director/VP/Principal' titles are director-track senior roles, not
# junior ones — the 'associate' signal must not fire on them.
_SENIOR_ASSOCIATE_RE = re.compile(
    r"\bassociate\s+(director|vp|vice president|principal)s?\b"
)


def is_junior_title(title_lower: str) -> bool:
    """True when the title carries a junior/entry-level signal."""
    for signal in JUNIOR_SIGNALS:
        if not _JUNIOR_SIGNAL_PATTERNS[signal].search(title_lower):
            continue
        if signal == "associate" and _SENIOR_ASSOCIATE_RE.search(title_lower):
            continue
        return True
    return False

# Seniority signals mapped to point values
SENIORITY_TIERS = [
    # (keywords, points)
    (["director", "vp", "vice president", "head of", "chief", "svp", "evp", "cpo"], 20),
    (["senior manager", "senior pm", "sr. pm", "sr pm", "lead", "principal", "staff"], 15),
    (["manager", " pm ", "pm,", "(pm)", "product manager"], 10),
]

_SENIOR_TITLE_RE = re.compile(r"\b(senior|principal|staff|lead|sr)\b")


def seniority_bucket(title: str) -> str:
    """Coarse seniority tier for comp benchmarking: director|senior|mid|other.

    NOT the same as SENIORITY_TIERS scoring: tier 2 there requires phrases like
    "senior manager"/"senior pm" and would misfile "Senior Product Manager" as
    mid — for benchmarking, any senior/principal/staff/lead modifier counts.
    """
    title_lower = (title or "").lower()
    if any(kw in title_lower for kw in SENIORITY_TIERS[0][0]):
        return "director"
    if _SENIOR_TITLE_RE.search(title_lower):
        return "senior"
    if any(kw in title_lower for kw in SENIORITY_TIERS[2][0]):
        return "mid"
    return "other"


class JobScorer:
    """Scores job postings against the owner profile (profile_policy + config)."""

    def __init__(self, config: dict):
        self.config = config
        self.alert_threshold = config.get("scoring", {}).get("alert_threshold", 60)
        self.logger = logging.getLogger(self.__class__.__name__)

        # Configured local commuter area: city names from config, matched only
        # when directly followed by a state context (see profile_policy's
        # LOCAL_STATE_PATTERN) so an out-of-state namesake city stays non-local.
        # Uses the shared matcher in local_area.py (also used by
        # scrapers.is_local_commuter_area and llm_scorer's local-area check) so the
        # three never drift apart; returns None when no cities are configured
        # (→ _is_local stays False).
        self._local_re = build_local_area_regex(config.get("local_locations") or [])

    def _is_local(self, location_lower: str) -> bool:
        """True when the location is in the configured local commuter area."""
        if not self._local_re or not location_lower:
            return False
        return bool(self._local_re.search(location_lower))

    def score(self, job: JobPosting) -> int:
        """
        Score a job posting 0-100.
        Returns 0 immediately for jobs that fail hard filters.
        """
        title_lower = (job.title or "").lower()
        description_lower = (job.description or "").lower()
        location_lower = (job.location or "").lower()
        company_lower = (job.company or "").lower()
        combined_text = title_lower + " " + description_lower

        # Ceiling applied at return time. A named non-remote city lowers it
        # (see LOCATION_MISMATCH_CAP) instead of zeroing the score outright.
        hard_cap = 100

        # Location filter: Not remote and names a specific city → cap, not zero.
        # The title counts too: "(Remote)" in a title is an explicit employer
        # signal even when the location names a city (posting-slot HQ).
        remote_terms = ["remote", "work from home", "distributed", "wfh", "anywhere"]
        is_remote = any(term in location_lower for term in remote_terms) or \
                    any(term in description_lower for term in remote_terms) or \
                    any(term in title_lower for term in remote_terms)

        is_local = self._is_local(location_lower)
        if not is_remote and not is_local:
            # Check if location contains a specific city (not just broad regions)
            # "United States" or empty location is OK to pass through
            non_remote_indicators = [
                "san francisco", "new york", "seattle", "austin", "boston",
                "chicago", "los angeles", "denver", "atlanta", "miami",
                "washington", "portland", "phoenix", "toronto", "london",
                "paris", "berlin", "singapore", "sf,", "nyc",
            ]
            if (any(city in location_lower for city in non_remote_indicators)
                    and not is_high_comp_exception(
                        job.salary_min, job.salary_max, job.location)):
                hard_cap = LOCATION_MISMATCH_CAP

        # Hard filter 2: Clearly junior role
        if is_junior_title(title_lower):
            return 0

        # Hard filter 3: physical product-line role with no digital signal anywhere
        if _PHYSICAL_PRODUCT_TITLE.search(title_lower) and not _DIGITAL_SIGNALS.search(
            combined_text
        ):
            return 0

        total_score = 0

        # --- Keyword matching (max 40 pts) ---
        primary_pts = 0
        for kw in PRIMARY_KEYWORDS:
            if _KEYWORD_PATTERNS[kw].search(combined_text):
                primary_pts += 10
                if primary_pts >= 30:
                    break
        total_score += min(primary_pts, 30)

        secondary_pts = 0
        for kw in SECONDARY_KEYWORDS:
            if _KEYWORD_PATTERNS[kw].search(combined_text):
                secondary_pts += 5
                if secondary_pts >= 10:
                    break
        total_score += min(secondary_pts, 10)

        # --- Seniority match (max 20 pts) ---
        seniority_pts = 0
        for keywords, pts in SENIORITY_TIERS:
            if any(kw in title_lower for kw in keywords):
                seniority_pts = pts
                break
        total_score += seniority_pts

        # --- Compensation estimate (max 20 pts) ---
        comp_pts = self._score_compensation(job, company_lower, is_local)
        total_score += comp_pts

        # --- Priority company bonus (+10 pts) ---
        if any(pc in company_lower for pc in PRIORITY_COMPANIES):
            total_score += 10

        # --- Remote/local confidence (max 20 pts) ---
        # A job in the configured commuter area is as good as remote.
        if "remote" in location_lower or is_local:
            total_score += 20
        elif "hybrid" in location_lower or (
            (any(term in description_lower for term in remote_terms) or
             any(term in title_lower for term in remote_terms)) and
            "remote" not in location_lower
        ):
            # Title/description markers are lower confidence than a remote
            # location: they pass the hard filter but earn the mid tier.
            total_score += 10
        else:
            total_score += 5

        return min(total_score, hard_cap)

    def _score_compensation(self, job: JobPosting, company_lower: str,
                            is_local: bool = False) -> int:
        """Score the compensation component (0-20 pts).

        Local jobs calibrate to the local market bars in profile_policy.
        """
        full_target = LOCAL_FULL_COMP if is_local else COMP_TARGET
        partial_target = LOCAL_PARTIAL_COMP if is_local else REMOTE_PARTIAL_COMP
        # Direct salary data
        if job.salary_min is not None or job.salary_max is not None:
            max_sal = max(
                v for v in [job.salary_min, job.salary_max] if v is not None
            )
            min_sal = min(
                v for v in [job.salary_min, job.salary_max] if v is not None
            )
            if max_sal >= full_target:
                return 20
            if min_sal >= partial_target:
                return 10
            return 0

        # No salary data - use company reputation
        # Normalize company name for matching
        company_normalized = company_lower.strip()
        for high_payer in HIGH_PAYING_COMPANIES:
            if high_payer in company_normalized:
                return 15

        return 5

    def explain(self, job: JobPosting) -> str:
        """
        Generate a 2-3 sentence human-readable explanation of why this job matches.
        """
        title_lower = (job.title or "").lower()
        description_lower = (job.description or "").lower()
        combined = title_lower + " " + description_lower

        reasons: List[str] = []

        # Check AI/ML match
        ai_found = [
            kw.upper()
            for kw in ["ai", "ml", "llm", "genai", "nlp"]
            if kw in _KEYWORD_PATTERNS and _KEYWORD_PATTERNS[kw].search(combined)
        ]
        if ai_found:
            reasons.append(f"Strong AI/ML signal with keywords: {', '.join(ai_found[:3])}")

        # Check seniority
        for keywords, _ in SENIORITY_TIERS:
            matching = [kw for kw in keywords if kw in title_lower]
            if matching:
                reasons.append(f"Title indicates senior-level role ({job.title})")
                break

        # Check compensation
        if job.salary_min and job.salary_max and job.salary_min != job.salary_max:
            reasons.append(
                f"Salary range ${job.salary_min:,.0f}–${job.salary_max:,.0f} meets compensation target"
            )
        elif job.salary_min or job.salary_max:
            sal = job.salary_min or job.salary_max
            reasons.append(f"Salary ${sal:,.0f} meets compensation target")
        else:
            company_lower = (job.company or "").lower()
            for high_payer in HIGH_PAYING_COMPANIES:
                if high_payer in company_lower:
                    reasons.append(
                        f"{job.company} is a top-tier company known for strong compensation packages"
                    )
                    break

        # Check remote / local
        location_lower_ex = (job.location or "").lower()
        if "remote" in location_lower_ex:
            reasons.append("Explicitly listed as remote position")
        elif self._is_local(location_lower_ex):
            reasons.append(f"Located in the configured local commuter area ({job.location})")
        elif "remote" in title_lower:
            reasons.append("Title lists the position as remote")
        elif any(t in description_lower for t in ["remote", "work from home", "distributed"]):
            reasons.append("Description mentions remote work flexibility")

        # Check priority company
        company_lower_ex = (job.company or "").lower()
        if any(pc in company_lower_ex for pc in PRIORITY_COMPANIES):
            reasons.insert(0, f"{job.company} is a priority target company")

        if not reasons:
            reasons.append(f"Matches configured search criteria at {job.company}")

        return " ".join(reasons[:3])

    def score_and_explain(self, job: JobPosting) -> Tuple[int, str]:
        """Score a job and generate explanation in one call."""
        s = self.score(job)
        explanation = self.explain(job) if s > 0 else "Job did not meet minimum criteria."
        return s, explanation
