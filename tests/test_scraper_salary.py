"""Eightfold / SmartRecruiters / SuccessFactors must extract posted salaries."""
from scrapers.eightfold import EightfoldScraper
from scrapers.smartrecruiters import SmartRecruitersScraper
from scrapers.successfactors import SuccessFactorsScraper
from policy_fixtures import patch_greenhouse_gates

CFG = {
    "scraping": {"max_results_per_source": 5, "rate_limit_seconds": 0},
    "search_queries": ["Senior Product Manager AI remote"],
    "profile": {"target_titles": ["Senior Product Manager AI"]},
    "eightfold_tenants": [{"company": "Netflix", "base_url": "https://x", "domain": "x.com"}],
    "smartrecruiters_companies": [{"company": "Bosch", "company_id": "BoschGroup"}],
    "successfactors_tenants": [{"company": "Milliken", "base_url": "https://careers.milliken.com"}],
}


def test_eightfold_keeps_description_and_extracts_salary():
    s = EightfoldScraper(CFG)
    pos = {
        "id": 1, "name": "Senior Product Manager, ML",
        "city": "Remote", "state": "", "country": "US",
        "job_description": "<p>Own the ML roadmap.</p><p>Range: $200,000 - $250,000</p>",
    }
    job = s._parse_position(pos, CFG["eightfold_tenants"][0])
    assert "Own the ML roadmap." in job.description
    assert (job.salary_min, job.salary_max) == (200_000.0, 250_000.0)


def test_smartrecruiters_extracts_salary_from_sections(monkeypatch):
    patch_greenhouse_gates(monkeypatch)
    s = SmartRecruitersScraper(CFG)
    posting = {
        "id": "9", "name": "Senior Product Manager",
        "location": {"city": "Remote", "country": "US", "remote": True},
        "releasedDate": "2026-07-01T00:00:00Z",
        "ref": "https://jobs.smartrecruiters.com/x/9",
        "jobAd": {"sections": {"jobDescription": {"text": "Base pay $180K - $220K plus bonus."}}},
    }
    job = s._parse_posting(posting, "Bosch")
    assert (job.salary_min, job.salary_max) == (180_000.0, 220_000.0)


def test_successfactors_extracts_salary_from_description(monkeypatch):
    patch_greenhouse_gates(monkeypatch)
    s = SuccessFactorsScraper(CFG)
    search_html = """
    <table><tr class="data-row">
      <td><a class="jobTitle-link" href="/job/pm-1">Senior Product Manager</a></td>
      <td><span class="jobLocation">Springfield, IL</span></td>
      <td><span class="jobDate">Jul 1, 2026</span></td>
    </tr></table>"""
    pages = iter([search_html, None])  # page 1, then stop

    monkeypatch.setattr(s, "_get_html", lambda url: next(pages))
    monkeypatch.setattr(
        s, "_fetch_description",
        lambda url: "Lead digital products. Salary $150,000 to $175,000 DOE.",
    )
    monkeypatch.setattr("time.sleep", lambda *_: None)

    jobs = s._scrape_tenant(CFG["successfactors_tenants"][0])
    assert len(jobs) == 1
    assert (jobs[0].salary_min, jobs[0].salary_max) == (150_000.0, 175_000.0)
