"""Tests for report writing + retention in alerts/notifier.py."""
import json
from datetime import datetime, timedelta
from pathlib import Path

from alerts.notifier import ReportWriter
from scrapers.base import JobPosting


def _jobs():
    return [
        JobPosting(
            title="Senior PM",
            company="TestCo",
            location="Remote",
            url="https://example.com/job/1",
            description="desc",
        )
    ]


def test_save_report_writes_valid_json(tmp_path):
    writer = ReportWriter({})
    path = writer.save_report(_jobs(), {"total_jobs": 1}, str(tmp_path))
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    assert data["stats"] == {"total_jobs": 1}
    assert data["jobs"][0]["title"] == "Senior PM"


def test_save_report_prunes_reports_older_than_90_days(tmp_path):
    old = tmp_path / "report_2020-01-01.json"
    old.write_text("{}")
    recent_date = (datetime.now() - timedelta(days=5)).strftime("%Y-%m-%d")
    recent = tmp_path / f"report_{recent_date}.json"
    recent.write_text("{}")
    unrelated = tmp_path / "notes.json"
    unrelated.write_text("{}")
    malformed = tmp_path / "report_not-a-date.json"
    malformed.write_text("{}")

    writer = ReportWriter({})
    writer.save_report(_jobs(), {"total_jobs": 1}, str(tmp_path))

    assert not old.exists()          # >90 days old — pruned
    assert recent.exists()           # 5 days old — kept
    assert unrelated.exists()        # not a report file — untouched
    assert malformed.exists()        # unparseable date — untouched
