"""A stored row with an empty description gets repaired by a later scrape.

Regression: detail fetches are best-effort, so a transient failure could store
a job blank — and neither branch of save_jobs' duplicate handling (the
cross-source title+company dedup that `continue`s before INSERT, nor the
INSERT-OR-IGNORE URL branch) ever updated description. A blank row stayed
"No description available" forever unless manually backfilled.
"""
import main as main_mod
from engine.scorer import JobScorer
from scrapers.base import JobPosting


def _conn(tmp_path):
    return main_mod.init_database(str(tmp_path / "jobs.db"))


def _scorer():
    return JobScorer({"scoring": {"alert_threshold": 60}})


def _job(desc, title="Senior PM", url="https://x/j/1"):
    return JobPosting(
        title=title, company="Acme", location="Remote",
        url=url, description=desc, source="workday",
    )


def test_rescrape_fills_blank_description_via_dedup_branch(tmp_path):
    """Same title+company re-scraped (the common next-run case) hits the
    cross-source dedup branch — it must fill an empty stored description."""
    conn = _conn(tmp_path)
    scorer = _scorer()
    main_mod.save_jobs(conn, [_job("")], 1, scorer)
    main_mod.save_jobs(conn, [_job("Full JD text.")], 2, scorer)

    desc = conn.execute("SELECT description FROM jobs").fetchone()[0]
    assert desc == "Full JD text."


def test_url_dup_with_changed_title_fills_blank_description(tmp_path):
    """Same URL but a retitled posting lands in the INSERT-OR-IGNORE branch —
    it must also fill an empty stored description."""
    conn = _conn(tmp_path)
    scorer = _scorer()
    main_mod.save_jobs(conn, [_job("", url="https://x/j/9")], 1, scorer)
    main_mod.save_jobs(
        conn,
        [_job("Full JD text.", title="Sr Product Manager, Platform",
              url="https://x/j/9")],
        2, scorer,
    )

    desc = conn.execute(
        "SELECT description FROM jobs WHERE url = 'https://x/j/9'"
    ).fetchone()[0]
    assert desc == "Full JD text."


def test_existing_description_never_overwritten(tmp_path):
    conn = _conn(tmp_path)
    scorer = _scorer()
    main_mod.save_jobs(conn, [_job("Original text.")], 1, scorer)
    main_mod.save_jobs(conn, [_job("Different text.")], 2, scorer)

    desc = conn.execute("SELECT description FROM jobs").fetchone()[0]
    assert desc == "Original text."


def test_match_explanation_survives_description_only_update(tmp_path):
    """The URL-dup UPDATE must not null match_explanation when only the
    description improved."""
    conn = _conn(tmp_path)
    scorer = _scorer()
    main_mod.save_jobs(conn, [_job("", url="https://x/j/9")], 1, scorer)
    before = conn.execute(
        "SELECT match_explanation FROM jobs WHERE url = 'https://x/j/9'"
    ).fetchone()[0]

    # Force the URL-dup branch (different normalized title) with an equal
    # score so only description_improved triggers the UPDATE.
    job2 = _job("Full JD text.", title="Sr Product Manager, Platform",
                url="https://x/j/9")
    main_mod.save_jobs(conn, [job2], 2, scorer)

    row = conn.execute(
        "SELECT description, match_explanation FROM jobs WHERE url = 'https://x/j/9'"
    ).fetchone()
    assert row["description"] == "Full JD text."
    assert row["match_explanation"] is not None
    if before:
        assert row["match_explanation"] != ""
