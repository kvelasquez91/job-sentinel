"""Streaming reconcile-save: batches are saved as scrapers finish, and
same-run collisions are resolved to EXACTLY the state today's
dedup_jobs_in_memory + save_jobs would produce — completeness-wins with
identity swap (url/source/content follow the winner), source-rank URL ties,
prior-run fill upgrades, and llm_score re-NULL when a scored row's inputs
change so the consumer rescores it."""
import sqlite3

import main as main_mod
from scrapers.base import JobPosting


class _StubScorer:
    """Deterministic keyword scorer with today's config surface."""
    config = {"scoring": {"alert_threshold": 60}}

    def score_and_explain(self, job):
        return 50, "kw"


def _mkdb(tmp_path, name="jobs.db"):
    return main_mod.init_database(str(tmp_path / name))


def _job(title="AI PM", company="Acme", url="https://a/1", desc="short",
         salary=None, source="greenhouse", location="Remote"):
    return JobPosting(
        title=title, company=company, location=location, url=url,
        description=desc, salary_min=salary, salary_max=salary, source=source,
    )


# Trailing space stripped up front: JobPosting.__post_init__ normalizes
# descriptions via clean_description, and these tests compare stored text.
RICH_DESC = ("long description " * 120).strip()  # completeness ~3.0 via len/500


def _row(conn, key="https://%"):
    return conn.execute(
        "SELECT * FROM jobs WHERE url LIKE ? ORDER BY id", (key,)).fetchall()


def _new_state():
    return main_mod.StreamingSaveState()


def _save(conn, state, jobs, rank, run_id=7):
    return main_mod.reconcile_save_batch(
        conn, jobs, source_rank=rank, state=state, run_id=run_id,
        scorer=_StubScorer(), profile_name="testuser")


def test_richer_later_copy_swaps_row_identity(tmp_path):
    conn = _mkdb(tmp_path)
    state = _new_state()
    _save(conn, state, [_job(url="https://a/thin", desc="short")], rank=1)
    _save(conn, state, [_job(url="https://b/rich", desc=RICH_DESC,
                             salary=180000, source="linkedin")], rank=0)
    rows = _row(conn)
    assert len(rows) == 1, "same title+company must stay one row"
    row = dict(rows[0])
    assert row["url"] == "https://b/rich"
    assert row["source"] == "linkedin"
    assert row["description"] == RICH_DESC
    assert row["salary_min"] == 180000
    assert state.stats["total_new"] == 1


def test_swap_renulls_scoring_of_already_scored_row(tmp_path):
    conn = _mkdb(tmp_path)
    state = _new_state()
    _save(conn, state, [_job(url="https://a/thin", desc="short")], rank=1)
    conn.execute(
        "UPDATE jobs SET llm_score=80, llm_explanation='x', filter_score=70, "
        "filter_json='{}', salary_est_min=1, salary_est_max=2 "
        "WHERE url='https://a/thin'")
    conn.commit()
    _save(conn, state, [_job(url="https://b/rich", desc=RICH_DESC)], rank=0)
    row = dict(_row(conn)[0])
    assert row["url"] == "https://b/rich"
    assert row["llm_score"] is None, "score computed on the thin copy must not survive the swap"
    assert row["llm_explanation"] is None
    assert row["filter_score"] is None
    assert row["filter_json"] is None
    assert row["salary_est_min"] is None


def test_thinner_later_copy_is_dropped(tmp_path):
    conn = _mkdb(tmp_path)
    state = _new_state()
    _save(conn, state, [_job(url="https://a/rich", desc=RICH_DESC, salary=1)], rank=1)
    _save(conn, state, [_job(url="https://b/thin", desc="short")], rank=0)
    rows = _row(conn)
    assert len(rows) == 1
    assert dict(rows[0])["url"] == "https://a/rich"
    assert state.dropped == 1


def test_equal_completeness_within_batch_keeps_first(tmp_path):
    """Within one batch, ties keep the earlier copy — batch order IS the
    old merge order there (cross-batch ties go to the lower rank instead;
    see test_completeness_tie_across_batches_follows_fixed_order)."""
    conn = _mkdb(tmp_path)
    state = _new_state()
    _save(conn, state, [
        _job(url="https://a/first", desc="same"),
        _job(url="https://b/second", desc="same"),
    ], rank=1)
    rows = _row(conn)
    assert len(rows) == 1
    assert dict(rows[0])["url"] == "https://a/first"


def test_same_url_across_sources_lower_rank_wins(tmp_path):
    conn = _mkdb(tmp_path)
    state = _new_state()
    shared = "https://shared/job"
    _save(conn, state, [_job(title="T1", company="C1", url=shared,
                             desc="workday text", source="workday")], rank=3)
    _save(conn, state, [_job(title="T2", company="C2", url=shared,
                             desc="linkedin text", source="linkedin")], rank=0)
    rows = _row(conn, "https://shared/%")
    assert len(rows) == 1
    assert dict(rows[0])["source"] == "linkedin", "fixed merge order (LinkedIn first) must win URL ties"

    conn2 = _mkdb(tmp_path, "jobs2.db")
    state2 = _new_state()
    _save(conn2, state2, [_job(title="T2", company="C2", url=shared,
                               desc="linkedin text", source="linkedin")], rank=0)
    _save(conn2, state2, [_job(title="T1", company="C1", url=shared,
                               desc="workday text", source="workday")], rank=3)
    assert dict(_row(conn2, "https://shared/%")[0])["source"] == "linkedin"


def test_streaming_matches_batch_semantics_golden(tmp_path):
    """Golden parity: the same multi-source corpus produces an identical jobs
    table through (a) today's dedup_jobs_in_memory + one save_jobs call and
    (b) arrival-order reconcile-save with a shuffled arrival order."""
    tie_li = ("li tie " * 260).strip()
    tie_wd = ("wd tie " * 280).strip()
    corpus = {
        0: [_job(title="A", company="X", url="https://li/a", desc="li text " * 60,
                 salary=150000, source="linkedin"),
            _job(title="D", company="W", url="https://li/d", desc=tie_li,
                 source="linkedin")],
        1: [_job(title="A", company="X", url="https://gh/a", desc="gh", source="greenhouse"),
            _job(title="B", company="Y", url="https://gh/b", desc="gh b " * 80,
                 source="greenhouse"),
            # within-batch duplicate pair: completeness must pick the second
            _job(title="E", company="V", url="https://gh/e1", desc="e thin",
                 source="greenhouse"),
            _job(title="E", company="V", url="https://gh/e2", desc="e rich " * 90,
                 source="greenhouse"),
            # copy touching a prior-run row (pre-seeded in both DBs)
            _job(title="F", company="U", url="https://gh/f", desc="f fill " * 70,
                 source="greenhouse")],
        3: [_job(title="B", company="Y", url="https://wd/b", desc="wd", source="workday"),
            _job(title="C", company="Z", url="https://wd/c", desc="wd c", source="workday"),
            # cross-batch completeness TIE with rank 0's D — fixed order wins
            _job(title="D", company="W", url="https://wd/d", desc=tie_wd,
                 source="workday")],
    }

    def _seed_prior(c):
        c.execute(
            "INSERT INTO jobs (title, company, url, description, source, "
            "status, llm_score, normalized_title, normalized_company, profile) "
            "VALUES ('F', 'U', 'https://prior/f', '', 'eightfold', 'new', 55, "
            "?, ?, 'testuser')",
            (main_mod._norm_dedup("F"), main_mod._norm_dedup("U")))
        c.commit()

    conn_a = _mkdb(tmp_path, "a.db")
    _seed_prior(conn_a)
    ordered = corpus[0] + corpus[1] + corpus[3]  # fixed merge order
    merged = main_mod.dedup_jobs_in_memory(list(ordered))
    main_mod.save_jobs(conn_a, merged, run_id=7, scorer=_StubScorer(),
                       profile_name="testuser")

    conn_b = _mkdb(tmp_path, "b.db")
    _seed_prior(conn_b)
    state = _new_state()
    for rank in (3, 1, 0):  # reversed arrival order
        _save(conn_b, state, corpus[rank], rank=rank)
    main_mod.finalize_streaming_save(conn_b, state, run_id=7,
                                     scorer=_StubScorer(), profile_name="testuser")

    q = ("SELECT title, company, url, source, description, salary_min, "
         "salary_max, score FROM jobs ORDER BY normalized_title, normalized_company")
    assert [tuple(r) for r in conn_a.execute(q)] == \
           [tuple(r) for r in conn_b.execute(q)]
    assert state.stats["total_new"] == 5  # A, B, C, D, E (F filled a prior row)


# ---------------------------------------------------------------------------
# Review-driven parity fixes: within-batch dupes, completeness-tie rank
# order, prior-row copies deferred to a final winner-only resolution, and
# normalized columns following an identity swap.
# ---------------------------------------------------------------------------


def test_within_batch_duplicate_resolved_by_completeness(tmp_path):
    """Two same-key copies in ONE batch must resolve like dedup_jobs_in_memory
    (completeness-wins full identity), not first-wins with blank-fill."""
    conn = _mkdb(tmp_path)
    state = _new_state()
    _save(conn, state, [
        _job(url="https://a/thin", desc="short"),
        _job(url="https://a/rich", desc=RICH_DESC, salary=150000),
    ], rank=1)
    rows = _row(conn)
    assert len(rows) == 1
    row = dict(rows[0])
    assert row["url"] == "https://a/rich"
    assert row["description"] == RICH_DESC


def test_completeness_tie_across_batches_follows_fixed_order(tmp_path):
    """Equal completeness (both descriptions past the cap) must keep the copy
    from the earlier FIXED-merge-order source (LinkedIn rank 0), no matter
    which batch arrived first — the old flow's incumbent was fixed-order."""
    conn = _mkdb(tmp_path)
    state = _new_state()
    tie_a = ("greenhouse text " * 120).strip()
    tie_b = ("linkedin text " * 130).strip()
    _save(conn, state, [_job(url="https://gh/x", desc=tie_a,
                             source="greenhouse")], rank=1)
    _save(conn, state, [_job(url="https://li/x", desc=tie_b,
                             source="linkedin")], rank=0)
    row = dict(_row(conn)[0])
    assert row["source"] == "linkedin", "tie must go to the lower rank (old fixed order)"
    assert row["url"] == "https://li/x"

    # And the mirror case: lower rank arrives FIRST and must stay the winner.
    conn2 = _mkdb(tmp_path, "jobs2.db")
    state2 = _new_state()
    _save(conn2, state2, [_job(url="https://li/x", desc=tie_b,
                               source="linkedin")], rank=0)
    _save(conn2, state2, [_job(url="https://gh/x", desc=tie_a,
                               source="greenhouse")], rank=1)
    assert dict(_row(conn2)[0])["source"] == "linkedin"


def test_prior_row_copies_deferred_to_winner_only_resolution(tmp_path):
    """Copies matching a PRIOR-run row must produce NO writes until the end
    of the scrape, and then exactly one save_jobs application of the FINAL
    winner — a thin copy's fill must never touch the row at all when a
    richer copy arrives later (the old flow's loser never reached save_jobs)."""
    conn = _mkdb(tmp_path)
    conn.execute(
        "INSERT INTO jobs (title, company, url, description, source, status, "
        "llm_score, salary_min, normalized_title, normalized_company, profile) "
        "VALUES ('AI PM', 'Acme', 'https://prior/x', '', 'workday', 'new', 77, "
        "NULL, ?, ?, 'testuser')",
        (main_mod._norm_dedup("AI PM"), main_mod._norm_dedup("Acme")))
    conn.commit()
    state = _new_state()
    _save(conn, state, [_job(url="https://a/thin", desc="thin fill",
                             salary=99000)], rank=1)
    mid = dict(_row(conn, "https://prior/%")[0])
    assert mid["description"] == "", "deferred copy must not touch the prior row mid-scrape"
    assert mid["salary_min"] is None

    # Richer on BOTH axes so it beats the salaried thin copy (salary counts
    # 4.0 toward completeness): only the winner's fields may ever land.
    _save(conn, state, [_job(url="https://b/rich", desc=RICH_DESC,
                             salary=150000)], rank=0)
    main_mod.finalize_streaming_save(conn, state, run_id=7,
                                     scorer=_StubScorer(), profile_name="testuser")
    rows = _row(conn, "https://prior/%")
    assert len(rows) == 1
    row = dict(rows[0])
    assert row["description"] == RICH_DESC, "final winner's fill must apply"
    assert row["salary_min"] == 150000, "the winner's salary lands — never the loser's 99000"
    assert row["llm_score"] == 77
    assert len(_row(conn, "https://a/%")) == 0 and len(_row(conn, "https://b/%")) == 0


def test_retitled_prior_url_does_not_lose_rich_copy(tmp_path):
    """A thin copy whose URL exists from a prior run under a DIFFERENT key is
    deferred; when a richer same-key copy wins, the rich copy must INSERT as
    its own row and the prior row must stay untouched — exactly the old flow
    where the thin loser never reached save_jobs."""
    conn = _mkdb(tmp_path)
    conn.execute(
        "INSERT INTO jobs (title, company, url, description, source, status, "
        "score, normalized_title, normalized_company, profile) "
        "VALUES ('Staff PM', 'Acme', 'https://prior/u', 'old text', 'workday', "
        "'new', 42, ?, ?, 'testuser')",
        (main_mod._norm_dedup("Staff PM"), main_mod._norm_dedup("Acme")))
    conn.commit()
    state = _new_state()
    _save(conn, state, [_job(title="Senior PM", url="https://prior/u",
                             desc="thin")], rank=1)
    _save(conn, state, [_job(title="Senior PM", url="https://new/rich",
                             desc=RICH_DESC)], rank=0)
    main_mod.finalize_streaming_save(conn, state, run_id=7,
                                     scorer=_StubScorer(), profile_name="testuser")

    prior = dict(_row(conn, "https://prior/%")[0])
    assert prior["description"] == "old text", "prior row must be untouched"
    assert prior["score"] == 42
    rich = _row(conn, "https://new/%")
    assert len(rich) == 1 and dict(rich[0])["description"] == RICH_DESC
    assert state.stats["total_new"] == 1


def test_swap_updates_normalized_columns(tmp_path):
    """A URL-contest swap can change the key; the stored normalized columns
    must follow, or every later run's dedup matches the dead key."""
    conn = _mkdb(tmp_path)
    state = _new_state()
    shared = "https://shared/u"
    _save(conn, state, [_job(title="Staff PM", company="Acme", url=shared,
                             desc="a", source="workday")], rank=3)
    _save(conn, state, [_job(title="Senior PM", company="Acme", url=shared,
                             desc="b", source="linkedin")], rank=0)
    row = dict(_row(conn, "https://shared/%")[0])
    assert row["title"] == "Senior PM"
    assert row["normalized_title"] == main_mod._norm_dedup("Senior PM")


# ---------------------------------------------------------------------------
# Orchestration wiring (run_overlap_scrape_and_score): failure discipline and
# crash recovery — the parts only real launchd runs would otherwise exercise.
# ---------------------------------------------------------------------------
import threading
from types import SimpleNamespace


class _FakeStreamer:
    """Stands in for LLMScorer: records the events it was handed."""

    def __init__(self, result=(0, 0), crash=False):
        self.result = result
        self.crash = crash
        self.saw_stop = None
        self.started = threading.Event()

    def apply_llm_scores_streaming(self, run_id, drained, stop=None,
                                   profile="testuser", limit=None, reserve=0):
        self.started.set()
        if self.crash:
            raise RuntimeError("consumer crash")
        assert drained.wait(timeout=10), "orchestrator never signalled drain"
        self.saw_stop = bool(stop is not None and stop.is_set())
        return self.result


def _args():
    return SimpleNamespace(profile="testuser")


def test_overlap_failure_stops_consumer_before_raising(tmp_path, monkeypatch):
    conn = _mkdb(tmp_path)
    streamer = _FakeStreamer()

    def exploding_run_scrapers(config, known_urls=None, on_batch=None):
        raise RuntimeError("scrape infra down")

    monkeypatch.setattr(main_mod, "run_scrapers", exploding_run_scrapers)
    import pytest as _pytest
    with _pytest.raises(RuntimeError, match="scrape infra down"):
        main_mod.run_overlap_scrape_and_score(
            conn, {}, _args(), run_id=7, scorer=_StubScorer(),
            llm_scorer=streamer, score_cap=None)
    assert streamer.saw_stop is True, (
        "a dead run must STOP the consumer (no drain-to-completion spend)")


def test_overlap_consumer_crash_recovers_count_from_db(tmp_path, monkeypatch):
    conn = _mkdb(tmp_path)
    for i in range(3):
        conn.execute(
            "INSERT INTO jobs (title, company, url, source, status, run_id, "
            "llm_score, profile) VALUES (?, 'Co', ?, 't', 'new', 7, 70, 'testuser')",
            (f"T{i}", f"https://x/{i}"))
    conn.commit()
    streamer = _FakeStreamer(crash=True)

    monkeypatch.setattr(main_mod, "run_scrapers",
                        lambda config, known_urls=None, on_batch=None: ([], {}))
    jobs, counts, stats, scored, attempts = main_mod.run_overlap_scrape_and_score(
        conn, {}, _args(), run_id=7, scorer=_StubScorer(),
        llm_scorer=streamer, score_cap=200)
    assert scored == 3, "crash recovery must read the true count from the DB"
    assert attempts == 3
