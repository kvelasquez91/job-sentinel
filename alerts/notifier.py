"""
Run report writer for Job Sentinel.

Saves a JSON report of each run to data/reports/. Email digest delivery was
removed — results are consumed via the dashboard and these JSON reports.
"""
import json
import logging
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

from scrapers.base import JobPosting

logger = logging.getLogger(__name__)


class ReportWriter:
    """Saves JSON run reports for Job Sentinel."""

    def __init__(self, config: dict):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

    def save_report(self, jobs: List[JobPosting], run_stats: dict, output_dir: str,
                    source_warnings: Optional[List[str]] = None) -> str:
        """
        Save a JSON report of all jobs and stats to output_dir/report_{YYYY-MM-DD}.json.
        Returns the path to the saved report.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        today = datetime.now().strftime("%Y-%m-%d")
        report_path = output_path / f"report_{today}.json"

        # Build report data
        report = {
            "generated_at": datetime.now().isoformat(),
            "stats": run_stats,
            "source_warnings": source_warnings or [],
            "jobs": [
                {
                    "title": j.title,
                    "company": j.company,
                    "location": j.location,
                    "url": j.url,
                    "description": j.description[:1000] if j.description else "",
                    "salary_min": j.salary_min,
                    "salary_max": j.salary_max,
                    "date_posted": j.date_posted,
                    "source": j.source,
                    "score": j.score,
                    "status": j.status,
                    "match_explanation": j.match_explanation,
                }
                for j in sorted(jobs, key=lambda x: x.score, reverse=True)
            ],
        }

        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        self.logger.info("Report saved to %s", report_path)
        self._prune_old_reports(output_path)
        return str(report_path)

    def _prune_old_reports(self, output_path: Path, max_age_days: int = 90) -> int:
        """Delete report_YYYY-MM-DD.json files older than max_age_days.

        The date is parsed from the FILENAME (not mtime) so copied/backed-up
        files can't dodge or trigger pruning incorrectly. Files that don't
        match the report naming pattern are never touched.
        Returns the number of files deleted.
        """
        cutoff = datetime.now() - timedelta(days=max_age_days)
        deleted = 0
        for f in output_path.glob("report_*.json"):
            m = re.fullmatch(r"report_(\d{4}-\d{2}-\d{2})\.json", f.name)
            if not m:
                continue
            try:
                file_date = datetime.strptime(m.group(1), "%Y-%m-%d")
            except ValueError:
                continue
            if file_date < cutoff:
                try:
                    f.unlink()
                    deleted += 1
                except OSError as exc:
                    self.logger.warning("Could not prune old report %s: %s", f, exc)
        if deleted:
            self.logger.info(
                "Pruned %d report(s) older than %d days from %s.",
                deleted, max_age_days, output_path,
            )
        return deleted
