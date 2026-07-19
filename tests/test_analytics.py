"""/api/analytics: weekly flow, comp distribution, source yield."""
import asyncio
import importlib

import main as main_mod

app_mod = importlib.import_module("dashboard.app")


def _seed(tmp_path):
    db = tmp_path / "jobs.db"
    conn = main_mod.init_database(str(db))
    conn.executemany(
        "INSERT INTO jobs (title, company, url, source, score, status, profile, "
        "salary_max, salary_est_max, created_at) VALUES (?, 'Co', ?, ?, ?, 'new', 'testuser', ?, ?, datetime('now'))",
        [
            ("A", "u1", "linkedin", 90, 240000, None),
            ("B", "u2", "linkedin", 50, None, 180000),   # estimate fills in
            ("C", "u3", "hn",       85, 130000, None),
            ("D", "u4", "greenhouse", 40, None, None),   # no comp — not bucketed
        ],
    )
    conn.commit()
    conn.close()
    return db


def test_analytics_shapes(tmp_path, monkeypatch):
    monkeypatch.setattr(app_mod, "DB_PATH", str(_seed(tmp_path)))
    # Buckets now derive from DASHBOARD_COMP_TIERS; pin the tiers so the
    # label assertions stay stable and tree-independent (any config).
    monkeypatch.setattr(app_mod, "DASHBOARD_COMP_TIERS", [150_000, 200_000, 250_000])
    a = asyncio.run(app_mod.get_analytics(profile="testuser"))
    assert a["comp_buckets"] == {"<150k": 1, "150-200k": 1, "200-250k": 1, "250k+": 0}
    assert a["weekly"][-1]["new_jobs"] == 4
    assert a["weekly"][-1]["high_matches"] == 2
    by_src = {s["source"]: s for s in a["sources_14d"]}
    assert by_src["linkedin"]["jobs"] == 2 and by_src["linkedin"]["high"] == 1
    assert by_src["hn"]["high"] == 1
