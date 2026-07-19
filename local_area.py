"""Shared local-commuter-area location matching.

Single source of truth for the two things that used to be copied (and had
drifted) across three modules:

  1. **Which cities count as local** — ``LOCAL_COMMUTER_CITIES``, an alias of
     ``profile_policy.LOCAL_CITIES`` (itself read from config.yaml's
     ``local_locations`` at import time — the cities now COME from config via
     profile_policy, not a copy kept in sync with it).
  2. **How a location string is matched against them** — ``build_local_area_regex``.

Consumers:

  * ``scrapers/greenhouse.py`` — ``is_local_commuter_area`` gates the local-title
    filter-broadening (whether to surface a candidate job).
  * ``engine/scorer.py`` — ``JobScorer._is_local`` drives local comp
    calibration ($150K vs $220K) and remote-equivalent location scoring.
  * ``engine/llm_scorer.py`` — ``_LOCAL_AREA_RE`` caps the LLM
    ``remote_location`` dimension at full points for local jobs.

Keeping all three on this one helper prevents the classic failure where a job
is treated as local by one module but not another.

Gap width — deliberately STRICT (``[,\\s]+``, i.e. the city must be *directly*
followed by the state)::

    \\b(city)\\b[,\\s]+(?:state pattern)

An earlier copy in the scrapers used a loose ``[\\s\\S]{0,24}`` gap (up to 24
arbitrary characters between city and state). That was not merely "broader
recall": it produced genuine out-of-state false positives — e.g.
``"Springfield, OR (near IL)"`` and ``"Anderson, IN - some IL clients"`` matched
as local, directly contradicting the intent that ``"Springfield, OR"`` is NOT
local when the configured state is "IL". The strict gap is used everywhere for
consistency; if you ever need to broaden it, change it HERE so all three
consumers move together.
"""
import re
from functools import lru_cache
from typing import Iterable, Optional

from profile_policy import LOCAL_CITIES, LOCAL_STATE_PATTERN

# Historical alias kept for existing importers (llm_scorer, scrapers, tests).
# The live source is config.yaml's `local_locations`, read once by profile_policy.
LOCAL_COMMUTER_CITIES = LOCAL_CITIES


@lru_cache(maxsize=None)
def _compile(cities: tuple, state_pattern: str) -> Optional[re.Pattern]:
    """Compile (and cache) the matcher for a normalized tuple of cities."""
    if not cities:
        return None
    if not (state_pattern or "").strip():
        return None
    city_alt = "|".join(re.escape(c) for c in cities)
    return re.compile(
        r"\b(" + city_alt + r")\b[,\s]+(?:" + state_pattern + r")",
        re.IGNORECASE,
    )


def build_local_area_regex(cities: Iterable[str],
                           state_pattern: Optional[str] = None) -> Optional[re.Pattern]:
    """Compile the local-commuter-area matcher for ``cities``.

    Returns ``None`` when ``cities`` is empty (after dropping blanks) so callers
    can treat "no configured cities" as "no local behavior". Returns ``None``
    also when ``state_pattern`` (or its ``profile_policy.LOCAL_STATE_PATTERN``
    fallback) is blank — a blank state pattern must never compile to an empty
    alternation that would match any city regardless of state, so it disables
    local matching entirely rather than matching everywhere. Otherwise the
    compiled pattern is case-insensitive and matches a listed city only when it
    is directly followed by the configured state context (e.g. "IL" /
    "Illinois" for a "Springfield, IL" setup; defaults to
    ``profile_policy.LOCAL_STATE_PATTERN`` when ``state_pattern`` is not given).
    """
    norm = tuple(str(c).strip() for c in cities if str(c).strip())
    resolved = LOCAL_STATE_PATTERN if state_pattern is None else state_pattern
    return _compile(norm, resolved)


def is_local_commuter_area(location: str, cities: Iterable[str]) -> bool:
    """True when ``location`` names one of ``cities`` with a configured-state context.

    ``"Springfield, OR"`` / ``"Portland, IL"`` are NOT local (wrong state /
    unlisted city), and neither is ``"Sangamon County, IL"`` (city not
    directly followed by the state — see the module docstring on gap width).
    """
    if not location or not cities:
        return False
    pat = build_local_area_regex(cities)
    return bool(pat.search(location)) if pat else False
