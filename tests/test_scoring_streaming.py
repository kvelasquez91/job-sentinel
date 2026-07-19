"""Streaming LLM scoring: score a run's NULL rows in waves WHILE the scrape
is still saving batches, under ONE scoring-lock hold, with an ATTEMPTS-based
budget (max_jobs_per_run protects the subscription window even in a failure
storm), a reserve for the caller's best-first top-up, a stale-write guard
covering every scoring input a reconcile swap can change, and a stop event
that aborts a dead run's spend."""
import threading
import time

import main as main_mod
from engine.llm_scorer import LLMScorer


def _mkdb(tmp_path):
    conn = main_mod.init_database(str(tmp_path / "jobs.db"))
    return conn, str(tmp_path / "jobs.db")


def _insert(conn, n, run_id=9, start=0):
    for i in range(start, start + n):
        conn.execute(
            "INSERT INTO jobs (title, company, url, description, source, "
            "status, score, run_id, profile) "
            "VALUES (?, 'Co', ?, 'd', 't', 'new', ?, ?, 'testuser')",
            (f"J{i}", f"https://x/{i}", 50 + i, run_id))
    conn.commit()


def _fake_result(row):
    return (row["id"], 61.0, 70.0, "llm", None, None, None, None, None)


def _prep(monkeypatch, score_fn=None):
    monkeypatch.setattr("engine.llm_scorer._require_resume_summary", lambda: None)
    monkeypatch.setattr("engine.llm_scorer._load_feedback_examples",
                        lambda db, profile: ([], []))
    monkeypatch.setattr(LLMScorer, "is_available", lambda self: True)
    monkeypatch.setattr(
        LLMScorer, "_score_one_job",
        score_fn or (lambda self, row: _fake_result(row)))


def _null_count(conn):
    return conn.execute(
        "SELECT COUNT(*) FROM jobs WHERE llm_score IS NULL").fetchone()[0]


def test_streaming_scores_rows_saved_after_start_and_drains(tmp_path, monkeypatch):
    conn, db = _mkdb(tmp_path)
    _insert(conn, 3)
    scorer = LLMScorer(db_path=db)
    first_wave = threading.Event()

    def score_fn(self, row):
        first_wave.set()
        return _fake_result(row)

    _prep(monkeypatch, score_fn)
    drained = threading.Event()
    out = {}

    t = threading.Thread(target=lambda: out.setdefault(
        "r", scorer.apply_llm_scores_streaming(run_id=9, drained=drained, profile="testuser")))
    t.start()
    assert first_wave.wait(timeout=10), "consumer never started a wave"
    _insert(conn, 2, start=3)  # land while the consumer is already running
    drained.set()
    t.join(timeout=10)
    assert not t.is_alive()
    assert out["r"] == (5, 5)
    assert _null_count(conn) == 0


def test_streaming_honors_reserve_budget(tmp_path, monkeypatch):
    conn, db = _mkdb(tmp_path)
    _insert(conn, 6)
    scorer = LLMScorer(db_path=db)
    _prep(monkeypatch)
    drained = threading.Event()
    drained.set()

    scored, attempted = scorer.apply_llm_scores_streaming(
        run_id=9, drained=drained, limit=4, reserve=2, profile="testuser")
    assert (scored, attempted) == (2, 2), "streaming must stop at limit - reserve"
    assert _null_count(conn) == 4


def test_streaming_budget_counts_attempts_not_successes(tmp_path, monkeypatch):
    """A failure storm must not pull unlimited replacement rows: the cap
    bounds CLI ATTEMPTS, exactly like the old single LIMIT-N select."""
    conn, db = _mkdb(tmp_path)
    _insert(conn, 10)
    scorer = LLMScorer(db_path=db)
    attempts = []

    def failing(self, row):
        attempts.append(row["id"])
        return None  # CLI failure — row stays NULL

    _prep(monkeypatch, failing)
    drained = threading.Event()
    drained.set()

    scored, attempted = scorer.apply_llm_scores_streaming(
        run_id=9, drained=drained, limit=5, reserve=0, profile="testuser")
    assert scored == 0
    assert attempted == 5, "attempts, not successes, must consume the budget"
    assert len(attempts) == 5
    assert _null_count(conn) == 10


def test_streaming_stop_aborts_dead_run_spend(tmp_path, monkeypatch):
    """When the scrape fails, stop must end the streaming pass at the next
    boundary instead of draining the whole saved set."""
    conn, db = _mkdb(tmp_path)
    _insert(conn, 8)
    scorer = LLMScorer(db_path=db)
    started = threading.Event()
    release = threading.Event()
    calls = []

    def slow(self, row):
        started.set()
        calls.append(row["id"])
        assert release.wait(timeout=10)
        return _fake_result(row)

    _prep(monkeypatch, slow)
    drained = threading.Event()
    stop = threading.Event()
    out = {}
    t = threading.Thread(target=lambda: out.setdefault(
        "r", scorer.apply_llm_scores_streaming(
            run_id=9, drained=drained, stop=stop, workers=2, profile="testuser")))
    t.start()
    assert started.wait(timeout=10)
    stop.set()
    drained.set()
    release.set()
    t.join(timeout=10)
    assert not t.is_alive()
    scored, attempted = out["r"]
    assert _null_count(conn) >= 6, (
        f"stop must abort before draining the set (scored={scored})")


def test_write_results_skips_row_whose_inputs_changed(tmp_path, monkeypatch):
    """The reconcile swap can replace a row's description — or ONLY its
    title/location (URL-rank swap) — while the thin copy is mid-CLI-call;
    the stale result must not land."""
    conn, db = _mkdb(tmp_path)
    _insert(conn, 3)
    scorer = LLMScorer(db_path=db)
    rows = conn.execute(
        "SELECT id, title, company, location, description, salary_min, "
        "salary_max FROM jobs ORDER BY id").fetchall()

    conn.execute("UPDATE jobs SET description='swapped rich text' WHERE id=?",
                 (rows[0]["id"],))
    conn.execute("UPDATE jobs SET title='Retitled PM' WHERE id=?",
                 (rows[1]["id"],))
    conn.commit()

    results = [(r["id"], 61.0, 70.0, "llm", None, None, None, None, None)
               for r in rows]
    guards = {r["id"]: (r["description"] or "", r["salary_min"],
                        r["salary_max"], r["title"] or "", r["company"] or "",
                        r["location"] or "")
              for r in rows}
    written = scorer._write_results(results, guards=guards)
    assert written == 1
    got = dict(conn.execute(
        "SELECT id, llm_score FROM jobs ORDER BY id").fetchall())
    assert got[rows[0]["id"]] is None, "description change must void the result"
    assert got[rows[1]["id"]] is None, "title change must void the result"
    assert got[rows[2]["id"]] == 70.0


def test_streaming_holds_one_lock_for_whole_phase(tmp_path, monkeypatch):
    """A dashboard rejudge (or any second pass) during the streaming window
    must hit lock contention — proving the lock is not released between
    waves."""
    conn, db = _mkdb(tmp_path)
    _insert(conn, 1)
    scorer = LLMScorer(db_path=db)
    in_wave = threading.Event()
    gate = threading.Event()

    def slow_score(self, row):
        in_wave.set()
        assert gate.wait(timeout=10)
        return _fake_result(row)

    _prep(monkeypatch, slow_score)
    drained = threading.Event()
    out = {}
    t = threading.Thread(target=lambda: out.setdefault(
        "r", scorer.apply_llm_scores_streaming(run_id=9, drained=drained, profile="testuser")))
    t.start()
    assert in_wave.wait(timeout=10), "consumer never reached a wave"

    other = LLMScorer(db_path=db)
    _prep(monkeypatch)
    assert other.apply_llm_scores_to_db(run_id=9, profile="testuser") == 0, (
        "a concurrent pass must skip on lock contention")

    gate.set()
    drained.set()
    t.join(timeout=10)
    assert out["r"] == (1, 1)
