"""/api/jobs must be slim (no description) and not truncate at 500 rows."""
import asyncio
import importlib

import main as main_mod
from policy_fixtures import patch_pm_prefilter

app_mod = importlib.import_module("dashboard.app")


def _seed(tmp_path, n):
    db = tmp_path / "jobs.db"
    conn = main_mod.init_database(str(db))
    conn.executemany(
        "INSERT INTO jobs (title, company, url, description, score, status, profile) "
        "VALUES (?, ?, ?, ?, ?, 'new', 'testuser')",
        [(f"Job {i}", "Co", f"https://x/{i}", "long description " * 50, 50)
         for i in range(n)],
    )
    conn.commit()
    conn.close()
    return db


def _get_jobs(**kw):
    args = dict(min_score=None, status=None, sort_by="score",
                limit=2000, profile="testuser", show_dismissed=False)
    args.update(kw)
    return asyncio.run(app_mod.get_jobs(**args))


def test_list_endpoint_omits_description_and_returns_all(tmp_path, monkeypatch):
    monkeypatch.setattr(app_mod, "DB_PATH", str(_seed(tmp_path, 600)))
    jobs = _get_jobs()
    assert len(jobs) == 600
    assert "description" not in jobs[0]
    assert "title" in jobs[0] and "salary_max" in jobs[0]


def test_detail_endpoint_still_has_description(tmp_path, monkeypatch):
    monkeypatch.setattr(app_mod, "DB_PATH", str(_seed(tmp_path, 1)))
    job = asyncio.run(app_mod.get_job(job_id=1))
    assert "long description" in job["description"]


def test_list_omits_filter_json_but_keeps_filter_scalars(tmp_path, monkeypatch):
    db = tmp_path / "jobs.db"
    conn = main_mod.init_database(str(db))
    conn.execute(
        "INSERT INTO jobs (title, company, url, description, score, status, profile, "
        "filter_score, filter_score_master, filter_source, filter_knockout, filter_json) "
        "VALUES ('Job', 'Co', 'https://x/1', 'desc', 50, 'new', 'testuser', "
        "76, 60, 'tailored', 0, '{\"must_haves\": []}')")
    conn.commit()
    conn.close()
    monkeypatch.setattr(app_mod, "DB_PATH", str(db))

    jobs = _get_jobs()
    assert jobs[0]["filter_score"] == 76
    assert jobs[0]["filter_score_master"] == 60
    assert jobs[0]["filter_source"] == "tailored"
    assert jobs[0]["filter_knockout"] == 0
    assert "filter_json" not in jobs[0]           # slim list, like description

    detail = asyncio.run(app_mod.get_job(jobs[0]["id"]))
    assert detail["filter_json"] == '{"must_haves": []}'


def test_filter_gate_marks_prefilter_skipped_rows(tmp_path, monkeypatch):
    """Rows the prefilter would hard-skip (never scored) get a filter_gate
    reason so the UI can tell 'onsite/off-target' apart from 'not scored yet'."""
    # The off-target gate ('Senior Software Engineer') and the soft-cap case
    # ('Business Analyst', cap 30) rely on the PM-shaped prefilter patterns,
    # which are empty in a neutral tree.
    patch_pm_prefilter(monkeypatch)
    db = tmp_path / "jobs.db"
    conn = main_mod.init_database(str(db))
    desc = "We build enterprise AI products for large teams. " * 30  # no remote signal
    conn.executemany(
        "INSERT INTO jobs (title, company, url, location, description, score, "
        "status, profile, filter_score) "
        "VALUES (?, 'Co', ?, ?, ?, 50, 'new', 'testuser', ?)",
        [
            # location-gated: PM title, named non-remote city, no remote in desc
            ("Senior Product Manager", "https://x/onsite", "New York, NY", desc, None),
            # title-gated: engineering title (skipped regardless of remote)
            ("Senior Software Engineer", "https://x/eng", "Remote", desc, None),
            # not gated: PM title, remote — genuinely unscored
            ("Senior Product Manager", "https://x/remote", "Remote", desc, None),
            # NOT hard-skipped: cap 30 (non-PM-adjacent) still gets an LLM score,
            # so it must NOT be marked gated even though it's unscored + onsite.
            ("Business Analyst", "https://x/softcap", "New York, NY", desc, None),
            # already scored: no gate regardless of location
            ("Senior Product Manager", "https://x/scored", "New York, NY", desc, 80),
        ],
    )
    conn.commit()
    conn.close()
    monkeypatch.setattr(app_mod, "DB_PATH", str(db))

    jobs = {j["url"]: j for j in _get_jobs()}
    assert jobs["https://x/onsite"]["filter_gate"]["kind"] == "onsite"
    assert jobs["https://x/eng"]["filter_gate"]["kind"] == "off-target"
    assert jobs["https://x/remote"]["filter_gate"] is None
    assert jobs["https://x/softcap"]["filter_gate"] is None   # cap 30 > 15
    assert jobs["https://x/scored"]["filter_gate"] is None


# --- _filter_gate caching: repeat polls must not re-parse unchanged rows ----
# The frontend polls /api/jobs every 5 minutes; before caching, every poll
# re-ran extract_salary_regex on every unscored row, repeating the same
# over-cap discard WARNINGs forever (5,290/8,982 log lines on 2026-07-14).

_GATED_DESC = "We build enterprise AI products for large teams. " * 30


def _seed_gated(tmp_path, location="New York, NY", desc=_GATED_DESC):
    """One unscored PM row the prefilter hard-skips (onsite, no remote)."""
    db = tmp_path / "jobs.db"
    conn = main_mod.init_database(str(db))
    conn.execute(
        "INSERT INTO jobs (title, company, url, location, description, score, "
        "status, profile, filter_score) "
        "VALUES ('Senior Product Manager', 'Co', 'https://x/1', ?, ?, 50, "
        "'new', 'testuser', NULL)",
        (location, desc),
    )
    conn.commit()
    conn.close()
    return db


def _count_salary_parses(monkeypatch):
    calls = []
    real = app_mod.extract_salary_regex
    monkeypatch.setattr(
        app_mod, "extract_salary_regex",
        lambda text, **kw: (calls.append(1), real(text, **kw))[1])
    return calls


def test_filter_gate_cached_across_polls(tmp_path, monkeypatch):
    monkeypatch.setattr(app_mod, "DB_PATH", str(_seed_gated(tmp_path)))
    app_mod._FILTER_GATE_CACHE.clear()
    calls = _count_salary_parses(monkeypatch)

    first = _get_jobs()
    assert first[0]["filter_gate"]["kind"] == "onsite"
    n = len(calls)
    assert n > 0                      # first poll really computed the gate

    second = _get_jobs()
    assert len(calls) == n            # repeat poll served from cache
    assert second[0]["filter_gate"] == first[0]["filter_gate"]


def test_filter_gate_cache_invalidates_on_content_change(tmp_path, monkeypatch):
    db = _seed_gated(tmp_path)
    monkeypatch.setattr(app_mod, "DB_PATH", str(db))
    app_mod._FILTER_GATE_CACHE.clear()

    assert _get_jobs()[0]["filter_gate"]["kind"] == "onsite"

    # Row content changes (location goes remote) -> stale entry must not stick.
    import sqlite3
    conn = sqlite3.connect(str(db))
    conn.execute("UPDATE jobs SET location = 'Remote' WHERE id = 1")
    conn.commit()
    conn.close()

    assert _get_jobs()[0]["filter_gate"] is None


def test_filter_gate_scoring_bypasses_cache(tmp_path, monkeypatch):
    """Once a row gets a filter_score, the gate is None even with a stale
    cached verdict for that id."""
    db = _seed_gated(tmp_path)
    monkeypatch.setattr(app_mod, "DB_PATH", str(db))
    app_mod._FILTER_GATE_CACHE.clear()

    assert _get_jobs()[0]["filter_gate"]["kind"] == "onsite"

    import sqlite3
    conn = sqlite3.connect(str(db))
    conn.execute("UPDATE jobs SET filter_score = 80 WHERE id = 1")
    conn.commit()
    conn.close()

    assert _get_jobs()[0]["filter_gate"] is None


def test_dashboard_gate_does_not_warn_on_over_cap_salary(tmp_path, monkeypatch, caplog):
    """The dashboard path is a re-read, not a discovery — over-cap discards
    there must not log WARNINGs (the scoring/scraping paths still do)."""
    import logging
    desc = _GATED_DESC + " Total comp up to $1,500,000."
    monkeypatch.setattr(app_mod, "DB_PATH", str(_seed_gated(tmp_path, desc=desc)))
    app_mod._FILTER_GATE_CACHE.clear()

    with caplog.at_level(logging.DEBUG, logger="salary_rules"):
        _get_jobs()
    assert not any(r.levelno >= logging.WARNING for r in caplog.records)
