"""seniority_bucket + /api/jobs/{id}/comps market benchmarks."""
import asyncio
import importlib

import pytest
from fastapi import HTTPException

import main as main_mod
from engine.scorer import seniority_bucket
from local_area import build_local_area_regex

app_mod = importlib.import_module("dashboard.app")

# Explicit fixture cities AND an explicit state pattern so locality
# classification is config-agnostic: the share package ships config.yaml
# with `local_locations: []` (and an empty local_state_pattern), which
# makes the module-level dashboard.app._LOCAL_RE None at import time.
# Patching the compiled matcher keeps the comps locality split tested
# regardless of config — an explicit state_pattern is required here (not
# just explicit cities), since a blank ambient pattern disables matching
# outright (Task 3).
_FIXTURE_LOCAL_RE = build_local_area_regex(
    ("Springfield", "Riverton", "Fairview"), state_pattern=r"illinois|il\b")


def test_seniority_bucket_tiers():
    assert seniority_bucket("Director of Product, AI") == "director"
    assert seniority_bucket("VP Product") == "director"
    assert seniority_bucket("Senior Product Manager") == "senior"
    assert seniority_bucket("Principal Product Manager") == "senior"
    assert seniority_bucket("Product Manager") == "mid"
    assert seniority_bucket("Barista") == "other"


def _seed(tmp_path):
    db = tmp_path / "jobs.db"
    conn = main_mod.init_database(str(db))
    rows = [
        ("Director of Product", "Remote", 300000),   # the subject job (id 1)
        ("Director of AI",      "Remote", 280000),
        ("Director, Platform",  "Remote (US)", 320000),
        ("Head of Product",     "Remote", 260000),
        ("Director Digital",    "Remote", 340000),
        ("Senior Product Manager", "Remote", 180000),   # different bucket — excluded
        ("Director of Product", "Springfield, IL", 170000),  # local — excluded for a Remote job
    ]
    conn.executemany(
        "INSERT INTO jobs (title, company, url, location, salary_max, score, status, profile) "
        "VALUES (?, 'Co', 'https://x/' || ?, ?, ?, 50, 'new', 'testuser')",
        [(t, i, loc, sal) for i, (t, loc, sal) in enumerate(rows)],
    )
    conn.commit()
    conn.close()
    return db


def test_comps_same_bucket_same_locality(tmp_path, monkeypatch):
    monkeypatch.setattr(app_mod, "DB_PATH", str(_seed(tmp_path)))
    monkeypatch.setattr(app_mod, "_LOCAL_RE", _FIXTURE_LOCAL_RE)
    comps = asyncio.run(app_mod.get_job_comps(job_id=1, profile="testuser"))
    assert comps["bucket"] == "director"
    assert comps["local"] is False
    assert comps["count"] == 4          # 4 remote director comps, subject excluded
    assert comps["median"] == 300000    # median of [260, 280, 320, 340]k
    assert comps["p25"] is not None and comps["p75"] is not None


def test_comps_nonexistent_job_returns_404(tmp_path, monkeypatch):
    monkeypatch.setattr(app_mod, "DB_PATH", str(_seed(tmp_path)))
    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(app_mod.get_job_comps(job_id=999999, profile="testuser"))
    assert exc_info.value.status_code == 404
