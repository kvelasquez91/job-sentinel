"""One-off surgical re-judge (2026-07-13; scoped 2026-07-14): high-comp location exception.

Re-judges ONLY the currently-KO'd rows that BOTH (a) have posted top-of-band comp
>= $300K and (b) actually failed a LOCATION knockout — the only rows the inventory
carve-out can flip. Rows KO'd purely on non-location grounds (years, degree) are
skipped: the carve-out cannot change them (they stay KO'd), so re-judging them is
pure churn/spend. Selection reads the stored judge verdicts in filter_json, so a
row with a compound failure (location AND years) is still re-judged and correctly
stays KO'd on the surviving non-location knockout.

NEVER use main.py --rejudge-filter for this: the inventory edit marks ~every judged
row stale and a corpus-wide re-roll churns ~55% of rows.

Spec: the 2026-07-13 high-comp location-exception design (private repo notes).
Run:  venv/bin/python scripts/rejudge_high_comp_location.py
"""
import json
import logging
import os
import sqlite3
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from engine.llm_scorer import LLMScorer  # noqa: E402
from salary_rules import LOCATION_EXCEPTION_MIN_COMP  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
DB = os.path.join(os.path.dirname(__file__), "..", "data", "jobs.db")

# Location-flavored knockout requirement terms (lowercased substring match).
_LOCATION_TERMS = (
    "onsite", "on-site", "on site", "in-office", "in office", "in the office",
    "office presence", "relocat", "based in", "must reside", "must be located",
    "must live", "hybrid", "commut", "located in", "on-location",
)

conn = sqlite3.connect(DB)
conn.row_factory = sqlite3.Row
rows = conn.execute(
    "SELECT id, filter_json FROM jobs WHERE filter_knockout = 1 "
    "AND MAX(COALESCE(salary_min,0), COALESCE(salary_max,0)) >= ? "
    "AND status NOT IN ('expired','applied','not_interested') "
    "ORDER BY id", (LOCATION_EXCEPTION_MIN_COMP,)).fetchall()
conn.close()

ids = []
skipped_non_location = 0
for r in rows:
    try:
        blob = json.loads(r["filter_json"] or "{}")
    except (TypeError, json.JSONDecodeError):
        continue
    failed_location = any(
        k.get("verdict") == "failed"
        and any(t in (k.get("requirement") or "").lower() for t in _LOCATION_TERMS)
        for k in (blob.get("knockouts") or []))
    if failed_location:
        ids.append(r["id"])
    else:
        skipped_non_location += 1

print(f"{len(rows)} KO'd rows with qualifying comp: "
      f"{len(ids)} have a failed LOCATION knockout (re-judging); "
      f"{skipped_non_location} are non-location KOs (left KO'd, not re-judged)")
if not ids:
    sys.exit(0)

n = LLMScorer(db_path=DB).rejudge_filter(ids=ids)
print(f"re-judged {n} rows")
