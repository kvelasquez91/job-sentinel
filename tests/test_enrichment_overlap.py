"""Company enrichment runs on a background thread overlapping the scoring and
tailor phases (perf: it was a ~7-min serial phase with no in-run consumer —
it reads only company names from the scrape and writes intel on its own
sqlite connections; layoff penalties at save time always come from PRIOR
runs' intel). The caller joins before the run record/digest."""
import threading
import time

import main as main_mod

from scrapers.base import JobPosting


def _job(company):
    return JobPosting(title="T", company=company, location="Remote",
                      url=f"https://x/{company or 'none'}", description="d",
                      source="test")


class _Recorder:
    def __init__(self):
        self.db_path = None
        self.batches = []

    def make_cls(self):
        rec = self

        class _FakeIntel:
            def __init__(self, db_path):
                rec.db_path = db_path

            def batch_enrich(self, companies, delay=1.5):
                rec.batches.append((sorted(companies), delay))
                return {}

        return _FakeIntel


def test_returns_none_without_companies():
    assert main_mod.start_company_enrichment("db", []) is None
    assert main_mod.start_company_enrichment("db", [_job(""), _job(None)]) is None


def test_enriches_unique_companies_in_background(monkeypatch):
    import engine.company_intel as ci
    rec = _Recorder()
    monkeypatch.setattr(ci, "CompanyIntelligence", rec.make_cls())

    jobs = [_job("Acme"), _job("Acme"), _job("Globex"), _job("")]
    t = main_mod.start_company_enrichment("some.db", jobs)
    assert t is not None
    t.join(timeout=5)
    assert not t.is_alive()
    assert rec.db_path == "some.db"
    assert rec.batches == [(["Acme", "Globex"], 1.5)]


def test_enrichment_exception_is_contained(monkeypatch, caplog):
    import logging
    import engine.company_intel as ci

    class _Boom:
        def __init__(self, db_path):
            pass

        def batch_enrich(self, companies, delay=1.5):
            raise RuntimeError("intel provider down")

    monkeypatch.setattr(ci, "CompanyIntelligence", _Boom)
    with caplog.at_level(logging.ERROR, logger="job_sentinel"):
        t = main_mod.start_company_enrichment("db", [_job("Acme")])
        assert t is not None
        t.join(timeout=5)
    assert not t.is_alive()
    assert any("enrichment" in r.message.lower() for r in caplog.records)


def test_runs_concurrently_with_caller(monkeypatch):
    """start_company_enrichment must return while enrichment is still
    running — an inline call would deadlock this test."""
    import engine.company_intel as ci
    release = threading.Event()

    class _Waiting:
        def __init__(self, db_path):
            pass

        def batch_enrich(self, companies, delay=1.5):
            assert release.wait(timeout=5), "caller never released the event"

    monkeypatch.setattr(ci, "CompanyIntelligence", _Waiting)
    t = main_mod.start_company_enrichment("db", [_job("Acme")])
    assert t is not None
    assert t.is_alive(), "enrichment must still be in flight after start returns"
    release.set()  # only reachable because start() did not block
    t.join(timeout=5)
    assert not t.is_alive()
