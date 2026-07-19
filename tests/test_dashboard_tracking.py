"""Application-tracking fields: PATCH /tracking + applied_at auto-stamp."""
import asyncio
import importlib

import pytest
from fastapi import HTTPException

import main as main_mod

app_mod = importlib.import_module("dashboard.app")


def _seed(tmp_path):
    db = tmp_path / "jobs.db"
    conn = main_mod.init_database(str(db))
    conn.execute(
        "INSERT INTO jobs (title, company, url, score, status, profile) "
        "VALUES ('PM', 'Co', 'https://x/1', 90, 'new', 'testuser')")
    conn.commit()
    conn.close()
    return db


def test_tracking_update_sets_fields(tmp_path, monkeypatch):
    monkeypatch.setattr(app_mod, "DB_PATH", str(_seed(tmp_path)))
    body = app_mod.TrackingUpdate(stage="interview", notes="Panel on Friday",
                                  next_action_date="2026-07-10")
    row = asyncio.run(app_mod.update_job_tracking(job_id=1, body=body))
    assert row["stage"] == "interview"
    assert row["notes"] == "Panel on Friday"
    assert row["next_action_date"] == "2026-07-10"


def test_tracking_empty_string_clears_field(tmp_path, monkeypatch):
    monkeypatch.setattr(app_mod, "DB_PATH", str(_seed(tmp_path)))
    asyncio.run(app_mod.update_job_tracking(job_id=1, body=app_mod.TrackingUpdate(notes="x")))
    row = asyncio.run(app_mod.update_job_tracking(job_id=1, body=app_mod.TrackingUpdate(notes="")))
    assert row["notes"] is None


def test_tracking_rejects_bad_stage(tmp_path, monkeypatch):
    monkeypatch.setattr(app_mod, "DB_PATH", str(_seed(tmp_path)))
    with pytest.raises(HTTPException):
        asyncio.run(app_mod.update_job_tracking(
            job_id=1, body=app_mod.TrackingUpdate(stage="vibing")))


def test_tracking_rejects_bad_offer_json(tmp_path, monkeypatch):
    monkeypatch.setattr(app_mod, "DB_PATH", str(_seed(tmp_path)))
    with pytest.raises(HTTPException):
        asyncio.run(app_mod.update_job_tracking(
            job_id=1, body=app_mod.TrackingUpdate(offer_json="{not json")))


def test_status_applied_stamps_applied_at_once(tmp_path, monkeypatch):
    monkeypatch.setattr(app_mod, "DB_PATH", str(_seed(tmp_path)))
    row1 = asyncio.run(app_mod.update_job_status(
        job_id=1, body=app_mod.JobStatusUpdate(status="applied")))
    assert row1["applied_at"] is not None
    # Flip away and back — the original stamp must survive.
    asyncio.run(app_mod.update_job_status(job_id=1, body=app_mod.JobStatusUpdate(status="new")))
    row2 = asyncio.run(app_mod.update_job_status(
        job_id=1, body=app_mod.JobStatusUpdate(status="applied")))
    assert row2["applied_at"] == row1["applied_at"]


def test_tracking_nonexistent_job_returns_404(tmp_path, monkeypatch):
    monkeypatch.setattr(app_mod, "DB_PATH", str(_seed(tmp_path)))
    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(app_mod.update_job_tracking(
            job_id=999, body=app_mod.TrackingUpdate(stage="interview")))
    assert exc_info.value.status_code == 404


def test_tracking_partial_update_preserves_other_fields(tmp_path, monkeypatch):
    monkeypatch.setattr(app_mod, "DB_PATH", str(_seed(tmp_path)))
    asyncio.run(app_mod.update_job_tracking(
        job_id=1, body=app_mod.TrackingUpdate(notes="Keep me", next_action_date="2026-08-01")))
    row = asyncio.run(app_mod.update_job_tracking(
        job_id=1, body=app_mod.TrackingUpdate(stage="interview")))
    assert row["stage"] == "interview"
    assert row["notes"] == "Keep me"
    assert row["next_action_date"] == "2026-08-01"
