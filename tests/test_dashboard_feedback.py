"""job feedback: PATCH /feedback stores more/less/NULL."""
import asyncio
import importlib

import pytest
from fastapi import HTTPException

import main as main_mod

app_mod = importlib.import_module("dashboard.app")


def _seed(tmp_path):
    db = tmp_path / "jobs.db"
    conn = main_mod.init_database(str(db))
    conn.execute("INSERT INTO jobs (title, company, url, score, status, profile) "
                 "VALUES ('PM', 'Co', 'https://x/1', 90, 'new', 'testuser')")
    conn.commit()
    conn.close()
    return db


def test_feedback_set_and_clear(tmp_path, monkeypatch):
    monkeypatch.setattr(app_mod, "DB_PATH", str(_seed(tmp_path)))
    row = asyncio.run(app_mod.update_job_feedback(
        job_id=1, body=app_mod.FeedbackUpdate(direction="more")))
    assert row["feedback"] == "more"
    row = asyncio.run(app_mod.update_job_feedback(
        job_id=1, body=app_mod.FeedbackUpdate(direction=None)))
    assert row["feedback"] is None


def test_feedback_rejects_junk(tmp_path, monkeypatch):
    monkeypatch.setattr(app_mod, "DB_PATH", str(_seed(tmp_path)))
    with pytest.raises(HTTPException):
        asyncio.run(app_mod.update_job_feedback(
            job_id=1, body=app_mod.FeedbackUpdate(direction="meh")))


def test_feedback_nonexistent_job_returns_404(tmp_path, monkeypatch):
    monkeypatch.setattr(app_mod, "DB_PATH", str(_seed(tmp_path)))
    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(app_mod.update_job_feedback(
            job_id=999, body=app_mod.FeedbackUpdate(direction="more")))
    assert exc_info.value.status_code == 404
