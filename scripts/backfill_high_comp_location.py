"""One-off backfill (2026-07-13; retargeted 2026-07-14): high-comp location exception.

Scores ONLY the location-killed stubs whose posted top-of-band qualifies for the
$300K exception (is_high_comp_exception), selected by id — NOT the whole NULL-score
pool. Rationale: the live DB also holds unrelated backlog NULLs, so a pool-wide
drain (apply_llm_scores_to_db(backfill_limit=...)) would spend LLM on those too.
We select qualifying stubs with THE predicate (is_high_comp_exception, per-field
DB-or-regex salary — identical to the prefilter), clear exactly those, and score
exactly those via apply_llm_scores_to_db(ids=...).

Non-qualifying location stubs are left untouched (they stay correctly killed; the
prefilter would only re-stub them at cap 10). The scoring pass runs the stage-2
judge inline against the updated inventory carve-out.

Spec: the 2026-07-13 high-comp location-exception design (private repo notes).
Run:  venv/bin/python scripts/backfill_high_comp_location.py
"""
import logging
import os
import sqlite3
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from engine.llm_scorer import LLMScorer, clean_for_llm  # noqa: E402
from salary_rules import extract_salary_regex, is_high_comp_exception  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
DB = os.path.join(os.path.dirname(__file__), "..", "data", "jobs.db")

conn = sqlite3.connect(DB)
conn.row_factory = sqlite3.Row
stubs = conn.execute(
    "SELECT id, salary_min, salary_max, location, description, status FROM jobs "
    "WHERE llm_explanation LIKE 'Pre-filter: non-remote non-local location%' "
    "AND status NOT IN ('expired', 'applied', 'not_interested')"
).fetchall()

qualifying = []
for r in stubs:
    desc = clean_for_llm(r["description"] or "")
    regex_min, regex_max = extract_salary_regex(desc)
    smin = r["salary_min"] or regex_min
    smax = r["salary_max"] or regex_max
    if is_high_comp_exception(smin, smax, r["location"]):
        qualifying.append(r["id"])

print(f"{len(stubs)} live location stubs; {len(qualifying)} qualify for the $300K exception")
if not qualifying:
    conn.close()
    sys.exit(0)

conn.execute(
    f"UPDATE jobs SET llm_score = NULL, llm_explanation = NULL "
    f"WHERE id IN ({','.join('?' * len(qualifying))})", qualifying)
conn.commit()
conn.close()
print(f"cleared {len(qualifying)} qualifying stubs; scoring them by id")

written = LLMScorer(db_path=DB).apply_llm_scores_to_db(ids=qualifying)
print(f"backfill scored {written} rows (LLM-scored + judged inline)")
