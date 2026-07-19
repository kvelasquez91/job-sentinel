"""Shared salary plausibility rules.

Single source of truth for the max-plausible-salary cap and range sanitizing,
used by the scrapers (linkedin, greenhouse, workday, smartrecruiters,
successfactors, eightfold) and engine/llm_scorer.py — same root-level-module
pattern as local_area.py.

MAX_BASE_SALARY: values above this are treated as data errors (equity totals,
misparsed numbers), not salaries. $1M keeps wide executive bands (Netflix
posts up to ~$920K) while still rejecting "$1.5M" junk.

sanitize_salary_range: a range wider than 5x is junk ONLY when its low end is
below $100K (junk ranges start at benefit-level numbers like $40K); real
executive bands ($190K-$920K) start high and are kept intact.
"""
import logging
import re
from typing import Optional, Tuple

from profile_policy import RELOCATION_EXCEPTION_COMP as LOCATION_EXCEPTION_MIN_COMP

logger = logging.getLogger(__name__)

MAX_BASE_SALARY = 1_000_000
_WIDE_RANGE_SOLID_FLOOR = 100_000
_SUSPICIOUS_RATIO = 5


def sanitize_salary_range(
    v1: float, v2: float, *, quiet: bool = False
) -> Tuple[Optional[float], Optional[float]]:
    """Clamp an extracted (low, high) salary pair to plausible values.

    quiet=True logs discards at DEBUG instead of WARNING — for re-read paths
    (the dashboard gate recomputes on every poll) where the discard was
    already reported when the row was first scraped/scored.
    """
    _log = logger.debug if quiet else logger.warning
    lo, hi = min(v1, v2), max(v1, v2)
    if hi > MAX_BASE_SALARY:
        _log(
            "sanitize_salary_range: discarding $%.0f–$%.0f high end — exceeds "
            "MAX_BASE_SALARY cap", lo, hi,
        )
        return (lo, lo) if lo <= MAX_BASE_SALARY else (None, None)
    if lo > 0 and hi / lo > _SUSPICIOUS_RATIO and lo < _WIDE_RANGE_SOLID_FLOOR:
        _log(
            "sanitize_salary_range: suspicious range $%.0f–$%.0f (ratio %.1f, "
            "low end under $%d), dropping high end",
            lo, hi, hi / lo, _WIDE_RANGE_SOLID_FLOOR,
        )
        return lo, lo
    return lo, hi


def extract_salary_regex(
    text: str, *, quiet: bool = False
) -> Tuple[Optional[float], Optional[float]]:
    """
    Fast regex-based salary extraction from description text.
    Handles patterns like: $180K-$250K, $180,000 to $250,000, $220K, $200,000
    Returns (salary_min, salary_max) or (None, None) if not found.
    quiet=True demotes discard logs to DEBUG (see sanitize_salary_range).
    """
    if not text:
        return None, None

    # Strip 401(k) / 401k references before parsing to avoid confusing them
    # with salary figures (e.g. "401(k) matching" → "$401K" false positive).
    text = re.sub(r"\b401\s*\([Kk]\)", "", text)
    text = re.sub(r"\b401[Kk]\b", "", text)

    def _parse_amount(s: str) -> Optional[float]:
        s = s.strip().replace(",", "")
        mult = 1.0
        if s.lower().endswith("k"):
            mult = 1000.0
            s = s[:-1]
        elif s.lower().endswith("m"):
            mult = 1_000_000.0
            s = s[:-1]
        try:
            val = float(s) * mult
            if val > MAX_BASE_SALARY:
                (logger.debug if quiet else logger.warning)(
                    "extract_salary_regex: discarding $%.0f — exceeds MAX_BASE_SALARY cap", val
                )
                return None
            if 30_000 <= val:
                return val
        except ValueError:
            pass
        return None

    # Range patterns: $180K–$250K  /  $180,000 to $250,000  /  $180K - $250K
    range_pat = re.compile(
        r"\$\s*([\d,]+\.?\d*\s*[KkMm]?)\s*(?:[-–—]|to)\s*\$?\s*([\d,]+\.?\d*\s*[KkMm]?)",
        re.IGNORECASE,
    )
    for m in range_pat.finditer(text[:5_000]):
        v1 = _parse_amount(m.group(1))
        v2 = _parse_amount(m.group(2))
        if v1 and v2:
            return sanitize_salary_range(v1, v2, quiet=quiet)

    # Single value: $220,000 or $220K
    single_pat = re.compile(
        r"\$\s*([\d,]+\.?\d*\s*[KkMm]?)",
        re.IGNORECASE,
    )
    for m in single_pat.finditer(text[:5_000]):
        val = _parse_amount(m.group(1))
        if val:
            return val, val

    return None, None


# Non-US locations. Owned here (moved from engine/llm_scorer.py 2026-07-13).
# Used by llm_scorer for the prefilter location gate and the post-parse
# remote_location clamp, and below as one negative signal inside
# is_us_location. It is deliberately NOT the US test for the $300K exception:
# a ~35-entry blocklist can never enumerate the world (Ottawa/Zurich/Dublin
# slipped through it on 2026-07-14) — the exception requires positive US
# evidence instead.
NON_US_LOCATIONS = re.compile(
    r"\b(japan|tokyo|osaka|united kingdom|uk\b|london|germany|berlin|munich|"
    r"india|bangalore|bengaluru|hyderabad|mumbai|singapore|australia|sydney|"
    r"melbourne|canada|toronto|vancouver|france|paris|netherlands|amsterdam|"
    r"china|beijing|shanghai|korea|seoul|brazil|sao paulo|mexico|dubai|uae)\b",
    re.IGNORECASE,
)

# --- Positive-evidence US location test (2026-07-14) -----------------------
# Country signal: "United States (Remote)", "New York, New York, USA",
# "Sunnyvale, us", "US, CA, Santa Clara" (all live DB formats).
_US_COUNTRY = re.compile(
    r"\b(?:united states(?: of america)?|usa|us)\b|u\.s\.",
    re.IGNORECASE,
)

# Full state names + DC. Checked BEFORE the non-US guards so "Santa Fe,
# New Mexico" survives NON_US's \bmexico\b and "Paris, Texas" survives
# \bparis\b. ("Georgia" the country is a known, accepted ambiguity.)
_US_STATE_NAME = re.compile(
    r"\b(alabama|alaska|arizona|arkansas|california|colorado|connecticut|"
    r"delaware|florida|georgia|hawaii|idaho|illinois|indiana|iowa|kansas|"
    r"kentucky|louisiana|maine|maryland|massachusetts|michigan|minnesota|"
    r"mississippi|missouri|montana|nebraska|nevada|new hampshire|new jersey|"
    r"new mexico|new york|north carolina|north dakota|ohio|oklahoma|oregon|"
    r"pennsylvania|rhode island|south carolina|south dakota|tennessee|texas|"
    r"utah|vermont|virginia|washington|west virginia|wisconsin|wyoming|"
    r"district of columbia|puerto rico)\b|d\.c\.",
    re.IGNORECASE,
)

# Canadian province codes plus "CAN": in "Ottawa, ON, CA" the trailing "CA"
# is the COUNTRY, disambiguated by the province code before it. Uppercase-only
# on purpose — a case-insensitive \bon\b would fire constantly.
_CANADA_SIGNAL = re.compile(
    r"\b(AB|BC|MB|NB|NL|NS|NT|NU|ON|PE|QC|SK|YT|CAN)\b"
)

# US state/territory postal codes. Uppercase-only so English words (in, or,
# me, ok, hi, de, la) never match. Checked AFTER the non-US guards because
# two-letter ISO country codes collide with state codes (DE, IN, CA, ...).
_US_STATE_CODE = re.compile(
    r"\b(A[LKZR]|C[AOT]|DE|FL|GA|HI|I[DLNA]|K[SY]|LA|M[EDAINSOT]|"
    r"N[EVHJMYCD]|O[HKR]|PA|RI|S[CD]|T[NX]|UT|V[TA]|W[AVIY]|DC|PR|GU|VI|MP|AS)\b"
)

# Major-US-metro strings with no state token at all: "Greater Boston",
# "San Francisco Bay Area", "Miami-Fort Lauderdale Area" (live DB formats).
# Safe only because the non-US guards run first ("London, ON" never gets here).
_US_METRO = re.compile(
    r"\b(san francisco|bay area|silicon valley|new york|nyc|seattle|austin|"
    r"boston|chicago|houston|los angeles|denver|atlanta|miami|washington|"
    r"portland|phoenix|san jose|san diego)\b",
    re.IGNORECASE,
)


def is_us_location(location: Optional[str]) -> bool:
    """True only on positive evidence that a location string is in the US.

    Ambiguous/unknown strings ("", "5 Locations", "Remote", unlisted foreign
    cities) return False — callers that treat US-ness as a carve-out (the
    $300K exception) must default to NOT qualifying.
    """
    loc = (location or "").strip()
    if not loc:
        return False
    if _US_COUNTRY.search(loc) or _US_STATE_NAME.search(loc):
        return True
    if _CANADA_SIGNAL.search(loc) or NON_US_LOCATIONS.search(loc):
        return False
    return bool(_US_STATE_CODE.search(loc) or _US_METRO.search(loc))


# The $300K location exception (2026-07-13): a job whose POSTED top-of-band
# salary reaches this is exempt from the remote-or-local rule — onsite/hybrid/
# relocation anywhere in the US is acceptable. Posted salary only (never LLM
# estimates); non-US locations never qualify, and since 2026-07-14 the US
# test needs positive evidence (is_us_location) — a location that is merely
# absent from NON_US_LOCATIONS does not qualify. Value owned by
# profile_policy.RELOCATION_EXCEPTION_COMP (imported above); this comment
# documents the rule this module enforces.


def is_high_comp_exception(salary_min: Optional[float],
                           salary_max: Optional[float],
                           location: Optional[str]) -> bool:
    """THE definition of the $300K location exception.

    Every enforcement layer (prefilter step 7, keyword LOCATION_MISMATCH_CAP)
    calls this — the US-only test lives inside so no consumer is safe merely
    by blend-weight coincidence.
    """
    if max(salary_min or 0, salary_max or 0) < LOCATION_EXCEPTION_MIN_COMP:
        return False
    return is_us_location(location)
