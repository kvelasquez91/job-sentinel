"""
resume_tailor — Resume tailoring module for job-sentinel.

Orchestrates the full pipeline:
  1. Extract job description from URL
  2. Copy master resume in Google Docs
  3. LLM-powered keyword extraction & gap analysis
  4. Generate and apply surgical edits to the copy
  5. ATS compliance validation
  6. Export as .docx and move to "Tailored Resumes" folder
"""
from .config import (
    MASTER_DOC_ID,
    TAILORED_FOLDER_NAME,
    NAMING_TEMPLATE,
    USER_NAME,
)
from .google_api import GoogleAPIClient
from .jd_extractor import extract_jd, JobDescription
from .ats_checker import check_all, ATSCheckResult

__all__ = [
    "MASTER_DOC_ID",
    "TAILORED_FOLDER_NAME",
    "NAMING_TEMPLATE",
    "USER_NAME",
    "GoogleAPIClient",
    "extract_jd",
    "JobDescription",
    "check_all",
    "ATSCheckResult",
]
