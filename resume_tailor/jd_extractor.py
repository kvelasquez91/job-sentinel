"""
Job description extraction from URLs.

Tiered strategy:
  Tier 1 — Platform-specific extractor (Greenhouse, Lever, Ashby, Workday,
            LinkedIn, SmartRecruiters, Eightfold, iCIMS)
  Tier 2 — Generic readability extraction via trafilatura
  Tier 3 — LLM cleanup pass (strips nav/footer noise from Tier 2 output)
  Tier 4 — Raises JDExtractionError; caller should prompt user to paste manually

Platform detection uses URL hostname pattern matching.
"""
import json
import logging
import re
import threading
from dataclasses import dataclass, field
from typing import Optional
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    before_sleep_log,
)

import sys as _sys
import os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from claude_cli import run_claude, ClaudeCLIError

from .config import TAILOR_MODEL

logger = logging.getLogger(__name__)

# Thread-local token accumulator for LLM cleanup calls in this module.
_token_acc = threading.local()


def reset_token_usage() -> None:
    """Reset accumulated token counts for the current thread."""
    _token_acc.input_tokens = 0
    _token_acc.output_tokens = 0
    _token_acc.cost_usd = 0.0


def get_token_usage() -> dict:
    """Return accumulated token counts for the current thread."""
    return {
        "input_tokens": getattr(_token_acc, "input_tokens", 0),
        "output_tokens": getattr(_token_acc, "output_tokens", 0),
        "cost_usd": getattr(_token_acc, "cost_usd", 0.0),
    }

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/121.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/json,*/*",
}

REQUEST_TIMEOUT = 15  # seconds


class JDExtractionError(Exception):
    """Raised when all extraction tiers fail."""


@dataclass
class JobDescription:
    title: str
    company: str
    location: str
    raw_text: str
    requirements: list = field(default_factory=list)
    responsibilities: list = field(default_factory=list)
    qualifications: list = field(default_factory=list)
    keywords: list = field(default_factory=list)
    source_url: str = ""
    extraction_tier: int = 0  # 1-4; useful for debugging


# ---------------------------------------------------------------------------
# Platform detection
# ---------------------------------------------------------------------------

_PLATFORM_MATCHERS = {
    "greenhouse":      [r"boards\.greenhouse\.io", r"job-boards\.greenhouse\.io"],
    "lever":           [r"jobs\.lever\.co"],
    "ashby":           [r"jobs\.ashbyhq\.com"],
    "workday":         [r"\.myworkdayjobs\.com", r"\.wd\d+\.myworkdayjobs\.com"],
    "linkedin":        [r"linkedin\.com/jobs"],
    "smartrecruiters": [r"jobs\.smartrecruiters\.com"],
    "icims":           [r"\.icims\.com"],
    "eightfold":       [r"\.eightfold\.ai", r"explore\.jobs\."],
}


def _detect_platform(url: str) -> Optional[str]:
    """Return the ATS platform name for the URL, or None if unknown."""
    for platform, patterns in _PLATFORM_MATCHERS.items():
        for pattern in patterns:
            if re.search(pattern, url, re.IGNORECASE):
                return platform
    return None


# ---------------------------------------------------------------------------
# Tier 1 — Platform-specific extractors
# ---------------------------------------------------------------------------

@retry(
    retry=retry_if_exception_type(requests.RequestException),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
def _get(url: str, **kwargs) -> requests.Response:
    return requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT, **kwargs)


def _clean_html(html: str) -> str:
    """Strip HTML tags and collapse whitespace."""
    soup = BeautifulSoup(html, "lxml")
    text = soup.get_text(separator="\n")
    # Collapse runs of blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _extract_greenhouse(url: str) -> Optional[JobDescription]:
    """
    Greenhouse board URL formats:
      https://boards.greenhouse.io/{board_token}/jobs/{job_id}
      https://job-boards.greenhouse.io/{board_token}/jobs/{job_id}

    The public JSON API returns the full JD in the `content` field.
    """
    match = re.search(
        r"(?:boards|job-boards)\.greenhouse\.io/([^/]+)/jobs/(\d+)", url
    )
    if not match:
        return None

    board_token, job_id = match.group(1), match.group(2)
    api_url = f"https://boards-api.greenhouse.io/v1/boards/{board_token}/jobs/{job_id}"
    logger.debug("Greenhouse API: %s", api_url)

    try:
        resp = _get(api_url)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        logger.warning("Greenhouse API request failed for %s: %s", url, exc)
        return None

    content_html = data.get("content", "")
    title = data.get("title", "")
    location = data.get("location", {}).get("name", "") if isinstance(data.get("location"), dict) else ""
    company = board_token.replace("-", " ").title()

    return JobDescription(
        title=title,
        company=company,
        location=location,
        raw_text=_clean_html(content_html) if content_html else "",
        source_url=url,
        extraction_tier=1,
    )


def _extract_lever(url: str) -> Optional[JobDescription]:
    """
    Lever URL format: https://jobs.lever.co/{company}/{job_id}
    Public JSON API: https://api.lever.co/v0/postings/{company}/{job_id}
    """
    match = re.search(r"jobs\.lever\.co/([^/]+)/([^/?#]+)", url)
    if not match:
        return None

    company_slug, job_id = match.group(1), match.group(2)
    api_url = f"https://api.lever.co/v0/postings/{company_slug}/{job_id}"
    logger.debug("Lever API: %s", api_url)

    try:
        resp = _get(api_url)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        logger.warning("Lever API request failed for %s: %s", url, exc)
        return None

    # Lever response: text (object with description/lists), categories
    text_sections = data.get("text", "")
    lists = data.get("lists", [])
    all_text_parts = [str(text_sections)] if text_sections else []
    for lst in lists:
        all_text_parts.append(lst.get("text", "") + "\n" + lst.get("content", ""))

    categories = data.get("categories", {})
    location = categories.get("location", "") or categories.get("allLocations", [""])[0]

    return JobDescription(
        title=data.get("text", {}).get("title", "") if isinstance(data.get("text"), dict) else data.get("title", ""),
        company=company_slug.replace("-", " ").title(),
        location=location,
        raw_text=_clean_html("\n\n".join(all_text_parts)),
        source_url=url,
        extraction_tier=1,
    )


def _extract_ashby(url: str) -> Optional[JobDescription]:
    """
    Ashby URL: https://jobs.ashbyhq.com/{company}/{job_id}
    Embeds __NEXT_DATA__ JSON in the page with descriptionHtml.
    """
    try:
        resp = _get(url)
        resp.raise_for_status()
    except Exception as exc:
        logger.warning("Ashby fetch failed for %s: %s", url, exc)
        return None

    soup = BeautifulSoup(resp.text, "lxml")
    next_data_tag = soup.find("script", id="__NEXT_DATA__")
    if not next_data_tag:
        return None

    try:
        next_data = json.loads(next_data_tag.string)
    except (json.JSONDecodeError, TypeError):
        logger.warning("Failed to parse __NEXT_DATA__ for %s", url)
        return None

    # Navigate the Next.js page props tree (structure varies by Ashby version)
    props = next_data.get("props", {}).get("pageProps", {})
    job = props.get("jobPosting") or props.get("job") or {}

    description_html = (
        job.get("descriptionHtml")
        or job.get("description")
        or ""
    )
    title = job.get("title") or job.get("jobTitle") or ""
    location = job.get("locationName") or job.get("location") or ""

    # Company slug from URL
    match = re.search(r"jobs\.ashbyhq\.com/([^/]+)", url)
    company = match.group(1).replace("-", " ").title() if match else ""

    return JobDescription(
        title=title,
        company=company,
        location=location,
        raw_text=_clean_html(description_html) if description_html else "",
        source_url=url,
        extraction_tier=1,
    )


def _extract_workday(url: str) -> Optional[JobDescription]:
    """
    Workday URL: https://{company}.wd5.myworkdayjobs.com/.../{job_id}
    Internal CXS API returns JSON:
      GET /wday/cxs/{company}/{site}/job/{externalId}/jobPostingDetails
    The job path in the URL is usually the slug used in the API.
    Falls back to HTML parsing if the JSON API is unreachable.
    """
    # Extract company/tenant from the hostname
    hostname_match = re.search(r"([\w-]+)\.wd\d+\.myworkdayjobs\.com", url)
    if not hostname_match:
        return None

    company_slug = hostname_match.group(1)

    # Try to extract job path from URL; Workday URLs look like:
    # /External_Career_Site/job/Remote-USA/Senior-PM_JR-123456
    path = urlparse(url).path
    path_parts = [p for p in path.split("/") if p]

    # Attempt the CXS API with the full path structure
    # Pattern: /wday/cxs/{tenant}/{site}/job/{job-path}/jobPostingDetails
    api_candidates = []
    for i, part in enumerate(path_parts):
        if part.lower() in ("job", "jobs"):
            site = path_parts[i - 1] if i > 0 else "External_Career_Site"
            job_path = "/".join(path_parts[i:])
            api_url = (
                f"https://{company_slug}.wd5.myworkdayjobs.com"
                f"/wday/cxs/{company_slug}/{site}/{job_path}/jobPostingDetails"
            )
            api_candidates.append(api_url)
            break

    for api_url in api_candidates:
        try:
            resp = _get(api_url, headers={**HEADERS, "Accept": "application/json"})
            if resp.status_code == 200:
                data = resp.json()
                details = data.get("jobPostingInfo", {})
                desc_html = details.get("jobDescription", {}).get("content", "")
                title = details.get("title") or details.get("jobTitle") or ""
                location = details.get("primaryLocation", {}).get("descriptor") or ""
                if desc_html or title:
                    return JobDescription(
                        title=title,
                        company=company_slug.replace("-", " ").title(),
                        location=location,
                        raw_text=_clean_html(desc_html),
                        source_url=url,
                        extraction_tier=1,
                    )
        except Exception as exc:
            logger.debug("Workday CXS API failed for %s: %s", api_url, exc)

    # HTML fallback — parse the page directly
    try:
        resp = _get(url)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "lxml")
        # Workday renders via React; look for JSON in script tags
        for script in soup.find_all("script"):
            if script.string and "jobPostingInfo" in script.string:
                try:
                    # Extract JSON blob from inline script
                    json_match = re.search(r"\{.*jobPostingInfo.*\}", script.string, re.DOTALL)
                    if json_match:
                        data = json.loads(json_match.group(0))
                        desc = data.get("jobPostingInfo", {}).get("jobDescription", {}).get("content", "")
                        if desc:
                            return JobDescription(
                                title=data.get("jobPostingInfo", {}).get("title", ""),
                                company=company_slug.replace("-", " ").title(),
                                location="",
                                raw_text=_clean_html(desc),
                                source_url=url,
                                extraction_tier=1,
                            )
                except (json.JSONDecodeError, ValueError):
                    pass
    except Exception as exc:
        logger.warning("Workday HTML fallback failed for %s: %s", url, exc)

    return None


def _extract_linkedin(url: str) -> Optional[JobDescription]:
    """
    LinkedIn public job posting. JD is in div.description__text.
    LinkedIn aggressively blocks scrapers — this may 403; Tier 2 is the fallback.
    """
    try:
        resp = _get(url)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "lxml")

        desc_div = soup.find("div", class_="description__text") or \
                   soup.find("div", {"class": re.compile(r"description")})

        title_tag = soup.find("h1", class_=re.compile(r"title")) or \
                    soup.find("h1")
        company_tag = soup.find("a", class_=re.compile(r"company")) or \
                      soup.find("span", class_=re.compile(r"company"))

        title = title_tag.get_text(strip=True) if title_tag else ""
        company = company_tag.get_text(strip=True) if company_tag else ""
        raw_text = _clean_html(str(desc_div)) if desc_div else ""

        if raw_text:
            return JobDescription(
                title=title,
                company=company,
                location="",
                raw_text=raw_text,
                source_url=url,
                extraction_tier=1,
            )
    except Exception as exc:
        logger.warning("LinkedIn extraction failed for %s: %s", url, exc)

    return None


def _extract_smartrecruiters(url: str) -> Optional[JobDescription]:
    """
    SmartRecruiters URL: https://jobs.smartrecruiters.com/{company}/{job_id}
    Public API: GET https://api.smartrecruiters.com/v1/companies/{company}/postings/{id}
    """
    match = re.search(r"jobs\.smartrecruiters\.com/([^/]+)/([^/?#]+)", url)
    if not match:
        return None

    company_id, posting_id = match.group(1), match.group(2)
    api_url = f"https://api.smartrecruiters.com/v1/companies/{company_id}/postings/{posting_id}"
    logger.debug("SmartRecruiters API: %s", api_url)

    try:
        resp = _get(api_url)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        logger.warning("SmartRecruiters API failed for %s: %s", url, exc)
        return None

    sections = data.get("jobAd", {}).get("sections", {})
    desc_parts = []
    for key in ("companyDescription", "jobDescription", "qualifications", "additionalInformation"):
        content = sections.get(key, {}).get("text", "")
        if content:
            desc_parts.append(content)

    location = ""
    loc_data = data.get("location")
    if isinstance(loc_data, dict):
        location = loc_data.get("city", "") or loc_data.get("country", "")

    return JobDescription(
        title=data.get("name", ""),
        company=data.get("company", {}).get("name", company_id),
        location=location,
        raw_text=_clean_html("\n\n".join(desc_parts)),
        source_url=url,
        extraction_tier=1,
    )


def _extract_eightfold(url: str) -> Optional[JobDescription]:
    """
    Eightfold.ai job pages load via XHR.
    API pattern: GET /api/apply/v2/jobs/{job_id}
    """
    parsed = urlparse(url)
    # Job ID is typically the last numeric segment
    job_id_match = re.search(r"/(\d+)(?:[/?#]|$)", parsed.path)
    if not job_id_match:
        return None

    job_id = job_id_match.group(1)
    api_url = f"{parsed.scheme}://{parsed.netloc}/api/apply/v2/jobs/{job_id}"
    logger.debug("Eightfold API: %s", api_url)

    try:
        resp = _get(api_url, headers={**HEADERS, "Accept": "application/json"})
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        logger.warning("Eightfold API failed for %s: %s", url, exc)
        return None

    job = data.get("data", data)
    desc = job.get("description") or job.get("job_description") or ""
    title = job.get("name") or job.get("title") or ""
    location = job.get("location") or ""
    if isinstance(location, dict):
        location = location.get("name") or ""
    company = parsed.netloc.split(".")[0].title()

    return JobDescription(
        title=title,
        company=company,
        location=location,
        raw_text=_clean_html(desc) if desc else "",
        source_url=url,
        extraction_tier=1,
    )


def _extract_icims(url: str) -> Optional[JobDescription]:
    """
    iCIMS job pages often embed the JD in an iframe.
    Attempt to fetch the iframe URL if found.
    """
    try:
        resp = _get(url)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "lxml")

        # Look for the iframe with job content
        iframe = soup.find("iframe", src=re.compile(r"icims\.com"))
        if iframe and iframe.get("src"):
            iframe_url = iframe["src"]
            if iframe_url.startswith("//"):
                iframe_url = "https:" + iframe_url
            resp2 = _get(iframe_url)
            soup = BeautifulSoup(resp2.text, "lxml")

        # iCIMS job description is usually in div#iCIMS_Content_Wrapper or .iCIMS-Job-Description
        desc_div = (
            soup.find(id="iCIMS_Content_Wrapper")
            or soup.find("div", class_=re.compile(r"iCIMS|job.desc", re.I))
            or soup.find("div", id=re.compile(r"job.desc|description", re.I))
        )
        title_tag = soup.find("h1") or soup.find("title")
        title = title_tag.get_text(strip=True) if title_tag else ""

        if desc_div:
            return JobDescription(
                title=title,
                company="",
                location="",
                raw_text=_clean_html(str(desc_div)),
                source_url=url,
                extraction_tier=1,
            )
    except Exception as exc:
        logger.warning("iCIMS extraction failed for %s: %s", url, exc)

    return None


_TIER1_EXTRACTORS = {
    "greenhouse":      _extract_greenhouse,
    "lever":           _extract_lever,
    "ashby":           _extract_ashby,
    "workday":         _extract_workday,
    "linkedin":        _extract_linkedin,
    "smartrecruiters": _extract_smartrecruiters,
    "eightfold":       _extract_eightfold,
    "icims":           _extract_icims,
}


# ---------------------------------------------------------------------------
# Tier 2 — Generic readability via trafilatura
# ---------------------------------------------------------------------------

def _extract_generic(url: str) -> Optional[str]:
    """
    Use trafilatura to extract the main article content from any page.
    Returns cleaned plain text, or None if nothing useful was extracted.
    """
    try:
        import trafilatura  # optional dependency; install with: pip install trafilatura

        downloaded = trafilatura.fetch_url(url)
        if not downloaded:
            return None
        text = trafilatura.extract(
            downloaded,
            include_tables=True,
            include_links=False,
            favor_recall=True,
            no_fallback=False,
        )
        return text
    except ImportError:
        logger.warning(
            "trafilatura not installed — skipping Tier 2 generic extraction. "
            "Install with: pip install trafilatura"
        )
        return None
    except Exception as exc:
        logger.warning("trafilatura extraction failed for %s: %s", url, exc)
        return None


# ---------------------------------------------------------------------------
# Tier 3 — LLM cleanup
# ---------------------------------------------------------------------------

_LLM_CLEANUP_PROMPT = """\
Extract ONLY the job description from the following page content.
Return the job title, company name, location, and the full job description
including requirements, responsibilities, and qualifications.
Strip any navigation, footer, or unrelated content.
Respond with plain text — no markdown, no JSON.

Page content:
{raw_text}
"""


@retry(
    retry=retry_if_exception_type(ClaudeCLIError),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=2, max=8),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
def _llm_cleanup_inner(prompt: str) -> tuple:
    """Raw CLI call for LLM cleanup, retried on transient errors."""
    result = run_claude(prompt, model=TAILOR_MODEL, timeout=120.0)
    return result["text"].strip(), result["usage"], result["cost_usd"]


def _llm_cleanup(raw_text: str) -> Optional[str]:
    """
    Pass noisy extracted text through Claude for cleanup.
    Retries on transient errors (rate limits, timeouts, 5xx); silently skips on auth failure.
    Returns cleaned text, or None if the API call ultimately fails.
    """
    prompt = _LLM_CLEANUP_PROMPT.format(raw_text=raw_text[:8000])  # avoid token overflow
    try:
        text, usage, cost_usd = _llm_cleanup_inner(prompt)
        if not hasattr(_token_acc, "input_tokens"):
            _token_acc.input_tokens = 0
            _token_acc.output_tokens = 0
            _token_acc.cost_usd = 0.0
        _token_acc.input_tokens += usage.get("input_tokens", 0)
        _token_acc.output_tokens += usage.get("output_tokens", 0)
        _token_acc.cost_usd = getattr(_token_acc, "cost_usd", 0.0) + cost_usd
        return text
    except Exception as exc:
        logger.warning("LLM cleanup failed after retries: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def extract_jd(url: str) -> JobDescription:
    """
    Main entry point. Attempts extraction in tier order:
      1. Platform-specific extractor
      2. Generic readability (trafilatura)
      3. LLM cleanup of Tier 2 output
      4. Raises JDExtractionError

    Returns a populated JobDescription on success.
    """
    logger.info("Extracting JD from %s", url)

    # Tier 1
    platform = _detect_platform(url)
    if platform:
        extractor = _TIER1_EXTRACTORS.get(platform)
        if extractor:
            logger.debug("Using Tier 1 extractor: %s", platform)
            result = extractor(url)
            if result and result.raw_text.strip():
                logger.info(
                    "Tier 1 (%s) extraction succeeded: %d chars",
                    platform,
                    len(result.raw_text),
                )
                return result
            logger.info("Tier 1 (%s) returned empty result, falling through", platform)

    # Tier 2 — generic readability extraction
    logger.debug("Trying Tier 2 generic extraction")
    generic_text = _extract_generic(url)
    # 200-char minimum filters out pages that returned only a nav/error snippet —
    # anything shorter than a short paragraph is not a real JD.
    if generic_text and len(generic_text.strip()) > 200:
        logger.info("Tier 2 generic extraction succeeded: %d chars", len(generic_text))

        # Tier 3 — LLM cleanup of generic text
        logger.debug("Running Tier 3 LLM cleanup on generic text")
        cleaned = _llm_cleanup(generic_text)
        final_text = cleaned if cleaned and len(cleaned.strip()) > 100 else generic_text
        tier = 3 if (cleaned and len(cleaned.strip()) > 100) else 2

        return JobDescription(
            title="",
            company="",
            location="",
            raw_text=final_text,
            source_url=url,
            extraction_tier=tier,
        )

    # Tier 4 — failed
    raise JDExtractionError(
        f"All extraction tiers failed for URL: {url}\n"
        "Hint: set extraction_tier=4 and supply manual_jd_text to proceed."
    )


def extract_jd_from_text(text: str, source_url: str = "") -> JobDescription:
    """
    Tier 4: User-supplied manual paste. Wraps the raw text in a JobDescription
    so the rest of the pipeline can proceed identically.
    """
    return JobDescription(
        title="",
        company="",
        location="",
        raw_text=text.strip(),
        source_url=source_url,
        extraction_tier=4,
    )
