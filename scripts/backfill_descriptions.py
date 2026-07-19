#!/usr/bin/env python3
"""One-time backfill of missing job descriptions for the list-API scrapers.

Workday, Eightfold, and SmartRecruiters all return jobs from a LIST/search API
that carries no description; the full text lives on a per-job DETAIL endpoint
that the scrapers previously never called (see the scraper fixes). Every job
already stored from these sources therefore has an empty description ("No
description available" in the dashboard).

This script walks the empty-description rows for those three sources, fetches
each job's detail record through the (now fixed) scraper machinery, and writes
the description back to the DB. For those default sources it is idempotent —
re-running only touches rows that are still empty — and best-effort: a job
whose detail fetch fails (WAF, 403, expired posting) is left empty and counted
as a skip.

A fourth, OPT-IN backfiller, `wttj`, is different: it re-fetches EVERY stored
WTTJ row (not just empty ones) to upgrade requirements-only descriptions to
the full JD, so it is excluded from the no-arg default — name it explicitly.

Every description is normalized through scrapers.base.clean_description before
writing: the scraper path is sanitized centrally by JobPosting.__post_init__,
but this script does direct SQL UPDATEs and must apply the same guarantee.

It updates the description TEXT only. It does not re-score jobs or re-extract
salary; run the normal scoring/reblend pass afterwards if you want the new
description text reflected in scores.

Usage:
    ./venv/bin/python scripts/backfill_descriptions.py           # the 3 list-API sources
    ./venv/bin/python scripts/backfill_descriptions.py workday   # one source
    ./venv/bin/python scripts/backfill_descriptions.py wttj      # opt-in: refresh ALL wttj rows
    ./venv/bin/python scripts/backfill_descriptions.py workday --ids 14217 14279
        # only these rows — use for fresh stubs so the walk doesn't re-hit the
        # hundreds of permanently-dead (delisted) empty rows
"""
import argparse
import os
import sqlite3
import sys
import time
from urllib.parse import urlparse, parse_qs

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import main as jobsentinel_main
from scrapers.base import DESCRIPTION_MAX_LEN, clean_description
from scrapers.workday import WorkdayScraper
from scrapers.eightfold import EightfoldScraper
from scrapers.smartrecruiters import SmartRecruitersScraper
from scrapers.wttj import WTTJScraper

# Anchored to the repo root: a cwd-relative default would silently CREATE a
# fresh empty DB when the script is run from another directory.
DB_PATH = os.environ.get("JOB_SENTINEL_DB") or os.path.join(REPO_ROOT, "data", "jobs.db")
POLITE_DELAY = 0.25  # seconds between detail fetches, per source


# Optional id allowlist (--ids); empty means every empty-description row.
_ONLY_IDS: list = []


def _empty_rows(conn, source):
    q = ("SELECT id, url FROM jobs "
         "WHERE source = ? AND (description IS NULL OR TRIM(description) = '')")
    params = [source]
    if _ONLY_IDS:
        q += f" AND id IN ({','.join('?' * len(_ONLY_IDS))})"
        params.extend(_ONLY_IDS)
    return conn.execute(q, params).fetchall()


def _save(conn, job_id, description):
    conn.execute("UPDATE jobs SET description = ? WHERE id = ?", (description, job_id))


# ---------------------------------------------------------------------------
# Workday
# ---------------------------------------------------------------------------
def backfill_workday(conn, config):
    tenants = {t["tenant_url"]: t for t in (config.get("workday_tenants") or [])}
    sc = WorkdayScraper(config)

    rows = _empty_rows(conn, "workday")
    print(f"[workday] {len(rows)} empty rows")
    filled = skipped = 0
    established = set()

    for i, (job_id, url) in enumerate(rows, 1):
        host = urlparse(url).netloc
        tenant = tenants.get(host)
        if not tenant or "/en-US" not in url:
            skipped += 1
            continue
        external_path = url.split("/en-US", 1)[1]
        # Seed session cookies once per tenant (Cloudflare clearance / XSRF).
        if host not in established:
            try:
                sc._establish_session_for_tenant(tenant)
            except Exception:
                pass
            established.add(host)
        # The fetcher returns raw HTML; the scraper path is cleaned by
        # JobPosting.__post_init__, so this direct-SQL path must clean too.
        raw = sc._fetch_description(tenant, external_path)
        desc = clean_description(raw)[:DESCRIPTION_MAX_LEN] if raw else ""
        if desc:
            _save(conn, job_id, desc)
            filled += 1
        else:
            skipped += 1
        if i % 50 == 0:
            conn.commit()
            print(f"[workday] {i}/{len(rows)}  filled={filled} skipped={skipped}")
        time.sleep(POLITE_DELAY)

    conn.commit()
    print(f"[workday] done: filled={filled} skipped={skipped}")


# ---------------------------------------------------------------------------
# Eightfold
# ---------------------------------------------------------------------------
def backfill_eightfold(conn, config):
    tenant_list = list(config.get("eightfold_tenants") or [])
    by_subdomain = {t.get("subdomain"): t for t in tenant_list if t.get("subdomain")}
    by_host = {urlparse(t["base_url"]).netloc: t for t in tenant_list if t.get("base_url")}
    sc = EightfoldScraper(config)

    rows = _empty_rows(conn, "eightfold")
    print(f"[eightfold] {len(rows)} empty rows")
    filled = skipped = 0

    for i, (job_id, url) in enumerate(rows, 1):
        parsed = urlparse(url)
        qs = parse_qs(parsed.query)
        # pid is either ?pid=... (subdomain form) or the last path segment for
        # canonical URLs — both /jobs/{pid} and Netflix's /careers/job/{pid}.
        pid = (qs.get("pid") or [None])[0]
        if not pid:
            last = parsed.path.rstrip("/").rsplit("/", 1)[-1]
            if last.isdigit():
                pid = last
        subdomain = parsed.netloc.split(".")[0]
        tenant = by_subdomain.get(subdomain) or by_host.get(parsed.netloc)
        if not tenant or not pid:
            skipped += 1
            continue
        raw = sc._fetch_description(tenant, pid)
        desc = clean_description(raw)[:DESCRIPTION_MAX_LEN] if raw else ""
        if desc:
            _save(conn, job_id, desc)
            filled += 1
        else:
            skipped += 1
        if i % 50 == 0:
            conn.commit()
            print(f"[eightfold] {i}/{len(rows)}  filled={filled} skipped={skipped}")
        time.sleep(POLITE_DELAY)

    conn.commit()
    print(f"[eightfold] done: filled={filled} skipped={skipped}")


# ---------------------------------------------------------------------------
# SmartRecruiters
# ---------------------------------------------------------------------------
def backfill_smartrecruiters(conn, config):
    sc = SmartRecruitersScraper(config)
    rows = _empty_rows(conn, "smartrecruiters")
    print(f"[smartrecruiters] {len(rows)} empty rows")
    filled = skipped = 0

    for i, (job_id, url) in enumerate(rows, 1):
        # The stored URL IS the detail API URL (the posting's `ref`).
        job_ad = sc._fetch_job_ad(url)
        sections = (job_ad or {}).get("sections") or {}
        description = ""
        for key in ("jobDescription", "qualifications", "additionalInformation"):
            text = (sections.get(key) or {}).get("text") or ""
            if text:
                description = f"{description} {text}".strip()
        # The sections carry raw HTML; the scraper path is cleaned by
        # JobPosting.__post_init__, so this direct-SQL path must clean too.
        description = clean_description(description)[:DESCRIPTION_MAX_LEN]
        if description:
            _save(conn, job_id, description)
            filled += 1
        else:
            skipped += 1
        if i % 25 == 0:
            conn.commit()
            print(f"[smartrecruiters] {i}/{len(rows)}  filled={filled} skipped={skipped}")
        time.sleep(POLITE_DELAY)

    conn.commit()
    print(f"[smartrecruiters] done: filled={filled} skipped={skipped}")


# ---------------------------------------------------------------------------
# Welcome to the Jungle
# ---------------------------------------------------------------------------
def backfill_wttj(conn, config):
    """Re-fetch WTTJ descriptions from the public REST API.

    Older rows hold only the Algolia `profile` (requirements) because the www.*
    job page was WAF-blocked; the api.* endpoint now yields the full JD. Re-fetch
    EVERY wttj row (not just empty ones) so requirements-only rows are upgraded.
    """
    sc = WTTJScraper(config)
    rows = conn.execute("SELECT id, url, description FROM jobs WHERE source='wttj'").fetchall()
    print(f"[wttj] {len(rows)} rows")
    filled = skipped = 0

    for i, (job_id, url, current) in enumerate(rows, 1):
        org_slug = slug = None
        if "/companies/" in url and "/jobs/" in url:
            org_slug = url.split("/companies/", 1)[1].split("/jobs/", 1)[0]
            slug = url.split("/jobs/", 1)[1].split("?", 1)[0].split("#", 1)[0]
        if not (org_slug and slug):
            skipped += 1
            continue
        raw = sc._fetch_description(org_slug, slug)
        desc = clean_description(raw)[:DESCRIPTION_MAX_LEN] if raw else ""
        if desc and desc != (current or ""):
            _save(conn, job_id, desc)
            filled += 1
        else:
            skipped += 1
        if i % 25 == 0:
            conn.commit()
            print(f"[wttj] {i}/{len(rows)}  filled={filled} skipped={skipped}")
        time.sleep(POLITE_DELAY)

    conn.commit()
    print(f"[wttj] done: filled={filled} skipped={skipped}")


BACKFILLERS = {
    "workday": backfill_workday,
    "eightfold": backfill_eightfold,
    "smartrecruiters": backfill_smartrecruiters,
    "wttj": backfill_wttj,
}

# wttj is deliberately NOT in the default: its backfiller re-fetches and may
# rewrite EVERY stored wttj row, so it must be an explicit opt-in.
DEFAULT_SOURCES = ("workday", "eightfold", "smartrecruiters")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("sources", nargs="*", choices=list(BACKFILLERS),
                    help="Sources to backfill (default: workday eightfold "
                         "smartrecruiters; 'wttj' refreshes ALL wttj rows and "
                         "runs only when named explicitly).")
    ap.add_argument("--ids", type=int, nargs="+", default=[],
                    help="Only backfill these job ids (still only empty-"
                         "description rows among them).")
    args = ap.parse_args()
    sources = args.sources or list(DEFAULT_SOURCES)
    global _ONLY_IDS
    _ONLY_IDS = args.ids

    config = jobsentinel_main.load_config()
    conn = sqlite3.connect(DB_PATH)
    try:
        for src in sources:
            BACKFILLERS[src](conn, config)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
