"""Auto-tailor: dual-gate candidate selection + bounded execution."""
import main as main_mod

import google.auth.exceptions as gauth_exc

import settings_store
from claude_cli import ClaudeCLIError


# Gate config used across execution tests (mirrors shipped config.yaml).
_CFG = {"enabled": True, "min_score": 60, "min_filter_score": 60, "max_per_run": 2}


def _db(tmp_path):
    conn = main_mod.init_database(str(tmp_path / "jobs.db"))
    rows = [
        # (title, url, score, filter_score, status, dismissed, tailored_at)
        ("Winner A",   "u1", 95, 88.0, "new", 0, None),
        ("Winner B",   "u2", 90, 75.0, "new", 0, None),
        ("Third",      "u3", 88, 70.0, "new", 0, None),
        ("Low score",  "u4", 55, 90.0, "new", 0, None),        # score < 60
        ("Low filter", "u5", 90, 55.0, "new", 0, None),        # filter < 60
        ("Unjudged",   "u6", 90, None, "new", 0, None),        # NULL filter
        ("Applied",    "u7", 99, 99.0, "applied", 0, None),    # not 'new'
        ("Dismissed",  "u8", 99, 99.0, "new", 1, None),        # dismissed
        ("Done",       "u9", 99, 99.0, "new", 0, "2026-07-01"),  # tailored
    ]
    conn.executemany(
        "INSERT INTO jobs (title, url, score, filter_score, status, dismissed, "
        "tailored_at, company, profile) VALUES (?, ?, ?, ?, ?, ?, ?, 'Co', 'testuser')",
        rows,
    )
    conn.commit()
    return conn


def _select(conn, limit=10):
    return main_mod.select_auto_tailor_candidates(
        conn, "testuser", min_score=60, min_filter_score=60, limit=limit)


def _attempts(conn, url):
    return conn.execute(
        "SELECT COALESCE(auto_tailor_attempts, 0) FROM jobs WHERE url = ?",
        (url,)).fetchone()[0]


class _FakeResp:
    """Minimal stub of the httplib2 response googleapiclient.HttpError wraps."""
    def __init__(self, status):
        self.status = status
        self.reason = "stub"


def test_dual_gate_selects_top_eligible_bounded(tmp_path):
    conn = _db(tmp_path)
    got = _select(conn, limit=2)
    assert [r["title"] for r in got] == ["Winner A", "Winner B"]


def test_dual_gate_excludes_low_score_low_filter_and_unjudged(tmp_path):
    conn = _db(tmp_path)
    got = _select(conn)
    assert [r["title"] for r in got] == ["Winner A", "Winner B", "Third"]


def test_boundary_exactly_60_is_eligible(tmp_path):
    """Both gates are INCLUSIVE floors: score 60 + filter 60.0 qualifies.

    Pins the owner's 2026-07-17 ruling (">= 60, everywhere") after an
    auto-tailored job at exactly the judged ceiling — selected at judged
    ceiling exactly 60.0, then misread as a misfire when the post-tailor
    recompute displayed 53. Guards against anyone "fixing" the gate to
    strict >."""
    conn = _db(tmp_path)
    conn.execute(
        "INSERT INTO jobs (title, url, score, filter_score, status, dismissed, "
        "company, profile) VALUES ('Boundary', 'u11', 60, 60.0, 'new', 0, 'Co', 'testuser')")
    conn.commit()
    assert "Boundary" in [r["title"] for r in _select(conn)]


def test_sentinel_row_excluded(tmp_path):
    # Pins the verified invariant: the '{}' sentinel lives in filter_json with
    # NULL numeric fields, so no TEXT value ever reaches the numeric gate.
    conn = _db(tmp_path)
    conn.execute(
        "INSERT INTO jobs (title, url, score, filter_score, filter_json, status, "
        "company, profile) VALUES ('Sentinel', 'u10', 90, NULL, '{}', 'new', 'Co', 'testuser')")
    conn.commit()
    assert "Sentinel" not in [r["title"] for r in _select(conn)]


def test_exempt_and_attempts_excluded(tmp_path):
    conn = _db(tmp_path)
    conn.executemany(
        "INSERT INTO jobs (title, url, score, filter_score, status, company, "
        "profile, auto_tailor_exempt, auto_tailor_attempts) "
        "VALUES (?, ?, 96, 96.0, 'new', 'Co', 'testuser', ?, ?)",
        [
            ("Grandfathered", "u11", 1, 0),   # exempt -> out
            ("Failed twice",  "u12", None, 2),  # attempts cap -> out
            ("Failed once",   "u13", None, 1),  # still eligible
        ])
    conn.commit()
    titles = [r["title"] for r in _select(conn)]
    assert "Grandfathered" not in titles
    assert "Failed twice" not in titles
    assert titles[0] == "Failed once"   # 96 effective outranks Winner A's 95


def test_knocked_out_jobs_excluded(tmp_path):
    conn = _db(tmp_path)
    # High raw score AND high filter, but a failed knockout gates the
    # effective score to 40 — must not qualify.
    conn.execute(
        "INSERT INTO jobs (title, url, score, filter_score, filter_knockout, "
        "status, company, profile) "
        "VALUES ('KO', 'u14', 95, 99.0, 1, 'new', 'Co', 'testuser')")
    conn.commit()
    assert "KO" not in [r["title"] for r in _select(conn)]


def test_tailor_identity_env_overrides(monkeypatch):
    import importlib
    import dotenv
    # resume_tailor/config.py does `from dotenv import load_dotenv as _load_dotenv`
    # and calls it with override=True at import time. A real .env file (as
    # instructed by share/SETUP.md) would otherwise clobber the monkeypatched
    # env vars below on reload. Patching the `dotenv` module's attribute works
    # because each importlib.reload() re-executes the from-import, re-binding
    # `_load_dotenv` to whatever `dotenv.load_dotenv` currently is.
    monkeypatch.setattr(dotenv, "load_dotenv", lambda *a, **k: False)
    monkeypatch.setenv("MASTER_RESUME_DOC_ID", "DOC123")
    monkeypatch.setenv("TAILOR_USER_NAME", "Ada Lovelace")
    import resume_tailor.config as rt_config
    importlib.reload(rt_config)
    assert rt_config.MASTER_DOC_ID == "DOC123"
    assert rt_config.USER_NAME == "Ada Lovelace"
    assert rt_config.FIRST_NAME == "Ada"
    monkeypatch.delenv("MASTER_RESUME_DOC_ID")
    monkeypatch.delenv("TAILOR_USER_NAME")
    importlib.reload(rt_config)  # restore module state for other tests


def test_job_failure_bumps_attempts_and_continues(tmp_path, monkeypatch):
    conn = _db(tmp_path)
    calls = []

    def fake_pipeline(job, db_path, progress=None):
        calls.append(job["title"])
        if job["title"] == "Winner A":
            raise RuntimeError("transient")  # job-specific: bump + continue
        return {"ats_score": 90, "google_doc_url": "https://d/x"}

    import resume_tailor.pipeline as pipe
    monkeypatch.setattr(pipe, "run_tailor_pipeline", fake_pipeline)
    done = main_mod.run_auto_tailor(conn, str(tmp_path / "jobs.db"), "testuser", _CFG)
    assert calls == ["Winner A", "Winner B"]
    assert done == 1
    assert _attempts(conn, "u1") == 1
    assert _attempts(conn, "u2") == 0


def test_claude_cli_error_bumps_active_job_and_aborts(tmp_path, monkeypatch):
    """Poison-pill guard: the active job self-exempts after 2 runs, but the
    batch aborts so a down CLI isn't hammered."""
    conn = _db(tmp_path)
    errors_log = tmp_path / "errors.log"
    calls = []

    def fake_pipeline(job, db_path, progress=None):
        calls.append(job["title"])
        raise ClaudeCLIError("claude CLI produced no output (exit=137)")

    import resume_tailor.pipeline as pipe
    monkeypatch.setattr(pipe, "run_tailor_pipeline", fake_pipeline)
    done = main_mod.run_auto_tailor(
        conn, str(tmp_path / "jobs.db"), "testuser", _CFG, errors_log=str(errors_log))
    assert calls == ["Winner A"]          # aborted after the first job
    assert done == 0
    assert _attempts(conn, "u1") == 1     # active job bumped
    assert _attempts(conn, "u2") == 0     # rest of queue untouched
    assert "ClaudeCLIError" in errors_log.read_text()


def test_cli_failure_in_real_pipeline_bumps_and_aborts(tmp_path, monkeypatch):
    """2026-07-15 final review, end-to-end: with the REAL pipeline (not the
    fake-pipeline harness) and a dead Claude CLI, the failure must surface as
    ClaudeCLIError — bump the active job, abort the batch, log to errors.log —
    instead of completing as a mislabeled success that sets tailored_at and
    silently removes the job from the gate."""
    import dataclasses
    from types import SimpleNamespace
    import resume_tailor.pipeline as pipe
    import resume_tailor.tailor_engine as te

    conn = _db(tmp_path)
    errors_log = tmp_path / "errors.log"

    @dataclasses.dataclass
    class FakeEdits:
        keyword_count: int = 0

    class FakeClient:
        def authenticate(self): pass
        def copy_document(self, master_id, title): return "doc-123"
        def read_document(self, doc_id): return {"body": {}}
        def extract_plain_text(self, doc): return "resume text"
        def export_as_pdf(self, doc_id): return b"%PDF"
        def move_to_folder(self, doc_id, folder): pass
        def export_as_docx(self, doc_id, path):
            with open(path, "wb") as f:
                f.write(b"docx")
        def get_document_url(self, doc_id): return f"https://docs.google.com/{doc_id}"

    # Stub everything AROUND the LLM; extract_keywords stays REAL so the
    # engine's CLI call sits on the executed path.
    monkeypatch.setattr(pipe, "extract_jd", lambda url: SimpleNamespace(
        raw_text="JD text", company="Co", title="T"))
    monkeypatch.setattr(pipe, "GoogleAPIClient", FakeClient)
    monkeypatch.setattr(pipe, "gap_analysis", lambda master, jd, **kw: {"gaps": []})
    monkeypatch.setattr(pipe, "generate_edits", lambda m, j, g, c, **kw: FakeEdits())
    monkeypatch.setattr(pipe, "apply_edits", lambda doc_id, edits, client: None)
    monkeypatch.setattr(pipe, "build_line_map", lambda pdf: object())
    monkeypatch.setattr(pipe, "enforce_layout",
                        lambda doc_id, edits, doc, master_map, client: (b"%PDF", []))
    monkeypatch.setattr(pipe, "ats_check_all", lambda *a, **k: SimpleNamespace(
        score=88, passed=True, issues=[], warnings=[]))
    monkeypatch.setattr(pipe, "DOCX_OUTPUT_DIR", str(tmp_path))
    monkeypatch.setattr(pipe, "MASTER_RESUME_CACHE", str(tmp_path / "c.txt"))
    monkeypatch.setattr(pipe, "load_inventory", lambda: ("", "", "none"))

    def dead_cli(*a, **k):
        raise te.ClaudeCLITimeout("claude CLI killed: timed out after 120s")
    monkeypatch.setattr(te, "run_claude", dead_cli)

    done = main_mod.run_auto_tailor(
        conn, str(tmp_path / "jobs.db"), "testuser", _CFG, errors_log=str(errors_log))

    assert done == 0
    assert _attempts(conn, "u1") == 1     # active job bumped (poison-pill guard)
    assert _attempts(conn, "u2") == 0     # batch aborted before job 2
    tailored = conn.execute(
        "SELECT COUNT(*) FROM jobs WHERE tailored_at IS NOT NULL "
        "AND url IN ('u1', 'u2')").fetchone()[0]
    assert tailored == 0                  # neither candidate mislabeled as tailored
    assert "ClaudeCLI" in errors_log.read_text()


def test_google_auth_error_aborts_without_bump(tmp_path, monkeypatch):
    conn = _db(tmp_path)
    errors_log = tmp_path / "errors.log"
    calls = []

    def fake_pipeline(job, db_path, progress=None):
        calls.append(job["title"])
        raise gauth_exc.RefreshError("token expired")

    import resume_tailor.pipeline as pipe
    monkeypatch.setattr(pipe, "run_tailor_pipeline", fake_pipeline)
    done = main_mod.run_auto_tailor(
        conn, str(tmp_path / "jobs.db"), "testuser", _CFG, errors_log=str(errors_log))
    assert calls == ["Winner A"]
    assert done == 0
    assert _attempts(conn, "u1") == 0     # systemic: no attempt burned
    assert "RefreshError" in errors_log.read_text()


def test_missing_google_creds_aborts_without_bump(tmp_path, monkeypatch):
    conn = _db(tmp_path)
    calls = []

    def fake_pipeline(job, db_path, progress=None):
        calls.append(job["title"])
        raise FileNotFoundError("client_secret.json missing")

    import resume_tailor.pipeline as pipe
    monkeypatch.setattr(pipe, "run_tailor_pipeline", fake_pipeline)
    done = main_mod.run_auto_tailor(conn, str(tmp_path / "jobs.db"), "testuser", _CFG)
    assert calls == ["Winner A"]
    assert done == 0
    assert _attempts(conn, "u1") == 0


def test_google_http_401_aborts_without_bump(tmp_path, monkeypatch):
    from googleapiclient.errors import HttpError
    conn = _db(tmp_path)
    errors_log = tmp_path / "errors.log"
    calls = []

    def fake_pipeline(job, db_path, progress=None):
        calls.append(job["title"])
        raise HttpError(resp=_FakeResp(401), content=b"unauthorized")

    import resume_tailor.pipeline as pipe
    monkeypatch.setattr(pipe, "run_tailor_pipeline", fake_pipeline)
    done = main_mod.run_auto_tailor(
        conn, str(tmp_path / "jobs.db"), "testuser", _CFG, errors_log=str(errors_log))
    assert calls == ["Winner A"]       # auth is systemic: batch aborted
    assert done == 0
    assert _attempts(conn, "u1") == 0  # no attempt burned
    assert "HttpError" in errors_log.read_text()


def test_google_http_500_bumps_and_continues(tmp_path, monkeypatch):
    from googleapiclient.errors import HttpError
    conn = _db(tmp_path)
    calls = []

    def fake_pipeline(job, db_path, progress=None):
        calls.append(job["title"])
        if job["title"] == "Winner A":
            raise HttpError(resp=_FakeResp(500), content=b"boom")
        return {"ats_score": 90, "google_doc_url": "https://d/x"}

    import resume_tailor.pipeline as pipe
    monkeypatch.setattr(pipe, "run_tailor_pipeline", fake_pipeline)
    done = main_mod.run_auto_tailor(conn, str(tmp_path / "jobs.db"), "testuser", _CFG)
    assert calls == ["Winner A", "Winner B"]   # 5xx is job-specific: continued
    assert done == 1
    assert _attempts(conn, "u1") == 1          # attempt burned on the failing job
    assert _attempts(conn, "u2") == 0


def test_toggle_off_in_db_skips_batch(tmp_path, monkeypatch):
    conn = _db(tmp_path)
    settings_store.set_setting(conn, settings_store.AUTO_TAILOR_KEY, "0")

    def boom(job, db_path, progress=None):
        raise AssertionError("pipeline must not run when the toggle is off")

    import resume_tailor.pipeline as pipe
    monkeypatch.setattr(pipe, "run_tailor_pipeline", boom)
    # config says enabled — the DB toggle must win
    assert main_mod.run_auto_tailor(
        conn, str(tmp_path / "jobs.db"), "testuser", _CFG) == 0


def test_second_failure_drops_job_from_next_selection(tmp_path, monkeypatch):
    conn = _db(tmp_path)

    def fake_pipeline(job, db_path, progress=None):
        raise RuntimeError("always fails")

    import resume_tailor.pipeline as pipe
    monkeypatch.setattr(pipe, "run_tailor_pipeline", fake_pipeline)
    for _ in range(2):
        main_mod.run_auto_tailor(conn, str(tmp_path / "jobs.db"), "testuser", _CFG)
    assert _attempts(conn, "u1") == 2
    assert _attempts(conn, "u2") == 2
    # Both burned out; next selection moves past them.
    assert [r["title"] for r in _select(conn, limit=2)] == ["Third"]


# ---------------------------------------------------------------------------
# Parallel auto-tailor (perf): auto_tailor.workers >= 2 runs independent
# candidates concurrently (each pipeline is its own Google Doc + its own
# claude CLI calls; token accounting is threading.local and every sqlite
# access in the pipeline opens its own connection). workers defaults to 1,
# which is the untouched sequential loop above — the shipped config opts in.
# Taxonomy adaptations for the parallel path:
#   - systemic errors cancel not-yet-started candidates; in-flight ones finish;
#   - concurrent ClaudeCLIErrors in one batch are ONE outage: at most one
#     bump, so an outage can't self-exempt N jobs at once;
#   - Google auth runs once up front on the main thread (token.json refresh
#     is not atomic under concurrent authenticate() calls).
# ---------------------------------------------------------------------------
import threading
import time as _time

_CFG_PAR = {"enabled": True, "min_score": 60, "min_filter_score": 60,
            "max_per_run": 3, "workers": 2}


def _ok_preflight(monkeypatch):
    """Neutralize the parallel path's up-front Google auth (no creds in CI)."""
    import resume_tailor.google_api as gapi

    class _OkClient:
        def authenticate(self, allow_interactive=True):
            pass

    monkeypatch.setattr(gapi, "GoogleAPIClient", _OkClient)


def test_parallel_tailor_runs_concurrently(tmp_path, monkeypatch):
    """With workers=2, two candidates must be in flight at once: each waits
    on a 2-party barrier that deadlocks (times out) under sequential
    execution."""
    conn = _db(tmp_path)
    _ok_preflight(monkeypatch)
    barrier = threading.Barrier(2, timeout=5)

    def fake_pipeline(job, db_path, progress=None):
        barrier.wait()
        return {"ats_score": 90, "google_doc_url": "https://d/x"}

    import resume_tailor.pipeline as pipe
    monkeypatch.setattr(pipe, "run_tailor_pipeline", fake_pipeline)
    done = main_mod.run_auto_tailor(
        conn, str(tmp_path / "jobs.db"), "testuser", {**_CFG_PAR, "max_per_run": 2})
    assert done == 2
    assert _attempts(conn, "u1") == 0
    assert _attempts(conn, "u2") == 0


def test_parallel_cli_outage_bumps_at_most_one_and_cancels_pending(tmp_path, monkeypatch):
    """Two in-flight workers both hitting ClaudeCLIError is one CLI outage,
    not two poison pills: at most ONE attempt bump total, and the queued
    third candidate never starts."""
    conn = _db(tmp_path)
    _ok_preflight(monkeypatch)
    errors_log = tmp_path / "errors.log"
    release = threading.Barrier(2, timeout=5)

    def fake_pipeline(job, db_path, progress=None):
        # The first two candidates fail together at the barrier; the queued
        # third is normally cancelled, but cancellation is best-effort — if
        # it does start, it hits the same outage (no barrier: parties=2).
        if job["title"] != "Third":
            release.wait()
        raise ClaudeCLIError("claude CLI produced no output (exit=137)")

    import resume_tailor.pipeline as pipe
    monkeypatch.setattr(pipe, "run_tailor_pipeline", fake_pipeline)
    done = main_mod.run_auto_tailor(
        conn, str(tmp_path / "jobs.db"), "testuser", _CFG_PAR,
        errors_log=str(errors_log))
    assert done == 0
    total_bumps = (_attempts(conn, "u1") + _attempts(conn, "u2")
                   + _attempts(conn, "u3"))
    assert total_bumps == 1, (
        "concurrent CLI failures are one outage — exactly one bump total")
    assert "ClaudeCLIError" in errors_log.read_text()


def test_parallel_auth_error_aborts_without_bumps(tmp_path, monkeypatch):
    """Google auth failure mid-batch is systemic: no attempt budget burned on
    ANY candidate, pending candidates cancelled."""
    conn = _db(tmp_path)
    _ok_preflight(monkeypatch)
    errors_log = tmp_path / "errors.log"
    release = threading.Barrier(2, timeout=5)

    def fake_pipeline(job, db_path, progress=None):
        if job["title"] != "Third":   # see CLI-outage test on cancellation races
            release.wait()
        raise gauth_exc.RefreshError("token expired")

    import resume_tailor.pipeline as pipe
    monkeypatch.setattr(pipe, "run_tailor_pipeline", fake_pipeline)
    done = main_mod.run_auto_tailor(
        conn, str(tmp_path / "jobs.db"), "testuser", _CFG_PAR,
        errors_log=str(errors_log))
    assert done == 0
    assert _attempts(conn, "u1") == 0
    assert _attempts(conn, "u2") == 0
    assert _attempts(conn, "u3") == 0
    assert "RefreshError" in errors_log.read_text()


def test_parallel_job_failure_bumps_and_others_continue(tmp_path, monkeypatch, caplog):
    """A job-specific failure in one worker must not disturb the others, and
    parallel progress lines must carry the job id (interleaved logs are
    ambiguous without it)."""
    import logging as _logging
    conn = _db(tmp_path)
    _ok_preflight(monkeypatch)

    def fake_pipeline(job, db_path, progress=None):
        if progress:
            progress("Step 1/10: probing")
        if job["title"] == "Winner A":
            raise RuntimeError("job-specific")
        return {"ats_score": 90, "google_doc_url": "https://d/x"}

    import resume_tailor.pipeline as pipe
    monkeypatch.setattr(pipe, "run_tailor_pipeline", fake_pipeline)
    with caplog.at_level(_logging.INFO, logger="job_sentinel"):
        done = main_mod.run_auto_tailor(
            conn, str(tmp_path / "jobs.db"), "testuser", _CFG_PAR)
    assert done == 2
    assert _attempts(conn, "u1") == 1
    assert _attempts(conn, "u2") == 0
    assert _attempts(conn, "u3") == 0
    assert any("[job " in r.message for r in caplog.records), (
        "parallel progress lines must be tagged with the job id")


def test_parallel_preflight_auth_failure_aborts_before_any_pipeline(tmp_path, monkeypatch):
    """workers>=2 authenticates Google once on the main thread BEFORE
    submitting — concurrent authenticate() calls race on the non-atomic
    token.json refresh write. A dead refresh token must abort the batch with
    zero pipeline starts and zero bumps."""
    conn = _db(tmp_path)
    errors_log = tmp_path / "errors.log"
    import resume_tailor.google_api as gapi

    class _DeadClient:
        def authenticate(self, allow_interactive=True):
            raise gauth_exc.RefreshError("invalid_grant")

    monkeypatch.setattr(gapi, "GoogleAPIClient", _DeadClient)

    def boom(job, db_path, progress=None):
        raise AssertionError("pipeline must not start when pre-flight auth fails")

    import resume_tailor.pipeline as pipe
    monkeypatch.setattr(pipe, "run_tailor_pipeline", boom)
    done = main_mod.run_auto_tailor(
        conn, str(tmp_path / "jobs.db"), "testuser", _CFG_PAR,
        errors_log=str(errors_log))
    assert done == 0
    assert _attempts(conn, "u1") == 0
    assert _attempts(conn, "u2") == 0
    assert "RefreshError" in errors_log.read_text()


def test_parallel_workers_clamped_to_cli_max(tmp_path, monkeypatch):
    """auto_tailor.workers must never exceed the claude CLI's process-wide
    concurrency cap — beyond it, threads just queue on the semaphore."""
    conn = _db(tmp_path)
    _ok_preflight(monkeypatch)
    monkeypatch.setattr(main_mod, "CLAUDE_CLI_MAX_CONCURRENCY", 2)
    active = {"n": 0, "max": 0}
    lock = threading.Lock()
    pair = threading.Barrier(2, timeout=5)

    def fake_pipeline(job, db_path, progress=None):
        with lock:
            active["n"] += 1
            active["max"] = max(active["max"], active["n"])
        if job["title"] != "Third":
            pair.wait()          # proves two ARE concurrent (parallel path)
        _time.sleep(0.05)        # window for an over-subscribed third worker
        with lock:
            active["n"] -= 1
        return {"ats_score": 90, "google_doc_url": "https://d/x"}

    import resume_tailor.pipeline as pipe
    monkeypatch.setattr(pipe, "run_tailor_pipeline", fake_pipeline)
    done = main_mod.run_auto_tailor(
        conn, str(tmp_path / "jobs.db"), "testuser", {**_CFG_PAR, "workers": 8})
    assert done == 3
    assert active["max"] == 2, "workers=8 must clamp to the patched CLI cap of 2"


def test_parallel_no_bumps_after_systemic_abort(tmp_path, monkeypatch):
    """Once the batch is aborted by a systemic failure, in-flight siblings
    failing with ANY error class must not burn attempt budget — during a
    shared outage their failures are infrastructure, not poison pills."""
    conn = _db(tmp_path)
    _ok_preflight(monkeypatch)
    errors_log = tmp_path / "errors.log"

    def fake_pipeline(job, db_path, progress=None):
        if job["title"] == "Winner A":
            raise gauth_exc.RefreshError("token expired")  # systemic, fast
        # Sibling in flight during the abort: generous sleep so the abort is
        # processed first, then a job-shaped failure caused by the outage.
        _time.sleep(0.5)
        raise RuntimeError("batchUpdate failed (same outage)")

    import resume_tailor.pipeline as pipe
    monkeypatch.setattr(pipe, "run_tailor_pipeline", fake_pipeline)
    done = main_mod.run_auto_tailor(
        conn, str(tmp_path / "jobs.db"), "testuser", _CFG_PAR,
        errors_log=str(errors_log))
    assert done == 0
    assert _attempts(conn, "u1") == 0
    assert _attempts(conn, "u2") == 0, "post-abort sibling failure must not bump"
    assert _attempts(conn, "u3") == 0


def test_tailor_one_job_gates_on_abort_event():
    """future.cancel() usually loses the dequeue race to a freed worker
    (measured 491/500 with staggered failures), so the worker itself must
    gate on the shared abort flag before doing any work."""
    from types import SimpleNamespace
    calls = []
    fake_pipe = SimpleNamespace(
        run_tailor_pipeline=lambda *a, **k: calls.append(1))
    ev = threading.Event()
    ev.set()
    job = {"id": 1, "url": "u", "company": "C", "title": "T"}
    assert main_mod._tailor_one_job(fake_pipe, job, "db", abort_event=ev) is None
    assert calls == []


def test_parallel_mainthread_crash_propagates_promptly(tmp_path, monkeypatch):
    """An exception escaping the result loop (e.g. sqlite failure in a bump)
    must propagate — with the escape hatch cancelling the queue rather than
    executor.__exit__ running every queued pipeline to completion first."""
    import sqlite3 as _sqlite3
    import pytest as _pytest
    conn = _db(tmp_path)
    _ok_preflight(monkeypatch)

    def fake_pipeline(job, db_path, progress=None):
        if job["title"] == "Winner A":
            raise RuntimeError("job failure that triggers the bump")
        _time.sleep(0.2)
        return {"ats_score": 90, "google_doc_url": "https://d/x"}

    import resume_tailor.pipeline as pipe
    monkeypatch.setattr(pipe, "run_tailor_pipeline", fake_pipeline)

    def broken_bump(conn_, job_id):
        raise _sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(main_mod, "_bump_auto_tailor_attempts", broken_bump)
    with _pytest.raises(_sqlite3.OperationalError):
        main_mod.run_auto_tailor(
            conn, str(tmp_path / "jobs.db"), "testuser", _CFG_PAR)
