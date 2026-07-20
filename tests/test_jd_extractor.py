"""Workday Tier-1 extractor: cxs URL construction and response parsing.

2026-07-20 incident context: the extractor guessed the site token from the
URL path segment before /job/ — which on our own scraped URLs is the LOCALE
(nvidia.wd5.myworkdayjobs.com/en-US/job/... → site "en-US") — and hardcoded
wd5 in the API hostname (Mastercard is wd1). The cxs call therefore failed
for every configured tenant and extraction silently leaned on Tier 2, which
the overnight window then took down too. Correct site tokens live in
config.yaml workday_tenants (the same source scrapers/workday.py builds its
verified detail URLs from).
"""
import pytest

import resume_tailor.jd_extractor as jd_ex


class _Resp:
    """Minimal stand-in for requests.Response."""

    def __init__(self, payload=None, status_code=200):
        self._payload = payload if payload is not None else {}
        self.status_code = status_code
        self.text = ""

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise jd_ex.requests.HTTPError(f"HTTP {self.status_code}")


def _capture_get(monkeypatch, responses=None):
    """Stub _get, recording every URL; unknown URLs answer 404."""
    responses = responses or {}
    calls = []

    def fake_get(url, **kwargs):
        calls.append(url)
        return responses.get(url, _Resp(status_code=404))

    monkeypatch.setattr(jd_ex, "_get", fake_get)
    return calls


def _set_site_map(monkeypatch, mapping):
    monkeypatch.setattr(jd_ex, "_workday_site_map", lambda: mapping)


def test_get_merges_caller_headers_with_defaults(monkeypatch):
    """_get(url, headers={...}) must MERGE with the default browser headers,
    not raise. The old signature forwarded **kwargs into requests.get(...,
    headers=HEADERS, **kwargs) → instant TypeError ('multiple values for
    headers') on every Tier-1 call that adds an Accept header — Workday's cxs
    candidates and Eightfold's API were never actually requested; the blanket
    except swallowed the crash."""
    seen = {}

    def fake_requests_get(url, headers=None, timeout=None, **kw):
        seen["headers"] = headers
        return _Resp()

    monkeypatch.setattr(jd_ex.requests, "get", fake_requests_get)
    jd_ex._get("https://x.example/api", headers={"Accept": "application/json"})

    assert seen["headers"]["Accept"] == "application/json"      # caller's wins
    assert "User-Agent" in seen["headers"]                      # defaults kept


def test_workday_uses_config_site_token_and_real_hostname(monkeypatch):
    """A configured tenant must hit the scraper-shaped cxs detail URL: the
    tenant's real hostname (wd1, not a hardcoded wd5) and its config
    site_path (not the locale segment the pretty URL carries)."""
    _set_site_map(monkeypatch, {
        "mastercard.wd1.myworkdayjobs.com": ("mastercard", "CorporateCareers"),
    })
    api = ("https://mastercard.wd1.myworkdayjobs.com/wday/cxs/mastercard/"
           "CorporateCareers/job/OFallon-Missouri/Manager-Product-Management_R-240")
    calls = _capture_get(monkeypatch, {api: _Resp({"jobPostingInfo": {
        "title": "Manager, Product Management",
        "location": "O'Fallon, Missouri",
        # cxs detail serves the description as a plain HTML STRING (the shape
        # scrapers/workday.py parses), not the legacy {"content": ...} dict.
        "jobDescription": "<p>Own the roadmap for tokenization.</p>",
    }})})

    jd = jd_ex._extract_workday(
        "https://mastercard.wd1.myworkdayjobs.com/en-US/job/OFallon-Missouri/"
        "Manager-Product-Management_R-240")

    assert calls[0] == api
    assert jd is not None
    assert jd.raw_text == "Own the roadmap for tokenization."
    assert jd.title == "Manager, Product Management"
    assert jd.location == "O'Fallon, Missouri"
    assert jd.extraction_tier == 1


def test_workday_unknown_tenant_keeps_path_guess_on_real_hostname(monkeypatch):
    """Tenants not in config keep the historical guess (path segment before
    /job/ + /jobPostingDetails), but on the URL's real hostname — never a
    hardcoded wd5."""
    _set_site_map(monkeypatch, {})
    calls = _capture_get(monkeypatch)

    jd = jd_ex._extract_workday(
        "https://acme.wd3.myworkdayjobs.com/SomeSite/job/City/Role_R-1")

    assert jd is None
    assert calls[0] == ("https://acme.wd3.myworkdayjobs.com/wday/cxs/acme/"
                        "SomeSite/job/City/Role_R-1/jobPostingDetails")
    assert not any("wd5" in c for c in calls if "wd3" not in c)


def test_workday_config_site_failure_falls_back_to_guess(monkeypatch):
    """If the config-based endpoint answers non-200 (stale site_path, window
    flake), the historical guess still runs as the second candidate."""
    _set_site_map(monkeypatch, {
        "acme.wd3.myworkdayjobs.com": ("acme", "StaleSite"),
    })
    guess = ("https://acme.wd3.myworkdayjobs.com/wday/cxs/acme/"
             "SomeSite/job/City/Role_R-1/jobPostingDetails")
    calls = _capture_get(monkeypatch, {guess: _Resp({"jobPostingInfo": {
        "title": "Role",
        "jobDescription": {"content": "<p>Legacy dict shape.</p>"},
    }})})

    jd = jd_ex._extract_workday(
        "https://acme.wd3.myworkdayjobs.com/SomeSite/job/City/Role_R-1")

    assert calls[0] == ("https://acme.wd3.myworkdayjobs.com/wday/cxs/acme/"
                        "StaleSite/job/City/Role_R-1")
    assert calls[1] == guess
    # Legacy {"content": ...} responses must keep parsing too.
    assert jd is not None
    assert jd.raw_text == "Legacy dict shape."


def test_workday_site_map_reads_config_yaml(tmp_path, monkeypatch):
    """The site map comes from config.yaml workday_tenants: tenant_url keys
    the map; company_slug defaults to company.lower() exactly as
    WorkdayScraper._detail_url does."""
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        "workday_tenants:\n"
        "  - company: NVIDIA\n"
        "    tenant_url: nvidia.wd5.myworkdayjobs.com\n"
        "    company_slug: nvidia\n"
        "    site_path: NVIDIAExternalCareerSite\n"
        "  - company: Acme\n"
        "    tenant_url: acme.wd1.myworkdayjobs.com\n"
        "    site_path: External\n"
    )
    monkeypatch.setattr(jd_ex, "_WORKDAY_CONFIG_PATH", str(cfg))
    monkeypatch.setattr(jd_ex, "_workday_site_map_cache", None)

    site_map = jd_ex._workday_site_map()

    assert site_map["nvidia.wd5.myworkdayjobs.com"] == (
        "nvidia", "NVIDIAExternalCareerSite")
    assert site_map["acme.wd1.myworkdayjobs.com"] == ("acme", "External")


def test_workday_site_map_empty_when_config_missing(tmp_path, monkeypatch):
    """A shared/fresh checkout without config.yaml must not break Tier 1 —
    the extractor just runs guess-only, as before."""
    monkeypatch.setattr(jd_ex, "_WORKDAY_CONFIG_PATH", str(tmp_path / "nope.yaml"))
    monkeypatch.setattr(jd_ex, "_workday_site_map_cache", None)

    assert jd_ex._workday_site_map() == {}
