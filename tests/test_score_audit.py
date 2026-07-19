"""Every change to jobs.score is recorded in score_audit via a DB trigger,
regardless of which code path (or human) wrote it."""
import main as main_mod


def test_score_update_writes_audit_row(tmp_path):
    conn = main_mod.init_database(str(tmp_path / "jobs.db"))
    conn.execute("INSERT INTO jobs (title, company, url, score, profile) "
                 "VALUES ('T', 'C', 'u1', 50, 'testuser')")
    conn.execute("UPDATE jobs SET score = 60 WHERE url = 'u1'")
    conn.commit()
    rows = conn.execute(
        "SELECT job_id, old_score, new_score, changed_at FROM score_audit"
    ).fetchall()
    assert len(rows) == 1
    assert (rows[0]["old_score"], rows[0]["new_score"]) == (50, 60)
    assert rows[0]["changed_at"] is not None


def test_no_audit_row_when_score_unchanged(tmp_path):
    conn = main_mod.init_database(str(tmp_path / "jobs.db"))
    conn.execute("INSERT INTO jobs (title, company, url, score, profile) "
                 "VALUES ('T', 'C', 'u1', 50, 'testuser')")
    conn.execute("UPDATE jobs SET score = 50, status = 'expired' WHERE url = 'u1'")
    conn.execute("UPDATE jobs SET dismissed = 1 WHERE url = 'u1'")
    conn.commit()
    assert conn.execute("SELECT COUNT(*) FROM score_audit").fetchone()[0] == 0


def test_init_database_is_idempotent_with_trigger(tmp_path):
    path = str(tmp_path / "jobs.db")
    main_mod.init_database(path).close()
    conn = main_mod.init_database(path)  # second init: must not raise
    n = conn.execute("SELECT COUNT(*) FROM sqlite_master "
                     "WHERE type='trigger' AND name='trg_jobs_score_audit'"
                     ).fetchone()[0]
    assert n == 1
