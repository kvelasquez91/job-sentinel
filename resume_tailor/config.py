"""
Resume tailor configuration.

All secrets and credential files live in resume_tailor/config/ (gitignored).
One-time setup: python scripts/setup_google_auth.py

Environment variables (all optional except where noted — fall back to the
defaults below; identity values have no baked-in defaults and MUST be set
via .env for a working tailor pipeline):
  GOOGLE_CLIENT_SECRET_PATH  Path to client_secret.json (OAuth Desktop App creds).
  GOOGLE_TOKEN_PATH          Path to token.json (written by setup_google_auth.py).
  TAILOR_DOCX_DIR            Directory where exported .docx files are saved.
  MASTER_RESUME_DOC_ID       Google Doc ID of YOUR master resume. No default — blank
                             until set.
  TAILOR_USER_NAME           Your full name (used in tailored-doc titles/filenames).
                             No default — falls back to "Your Name" until set.

Tuning constants:
  KEYWORD_MIN / KEYWORD_MAX         Target keyword density range for ATS (25-35).
  KEYWORD_CORRECTION_ROUNDS         How many LLM retry passes to attempt if density is off.
  ONE_PAGE_WORD_BUDGET              Conservative word-count ceiling for a one-page resume.
"""
import os

# Load .env before reading any env vars.
# override=True is required because Claude Desktop launches the server with
# ANTHROPIC_API_KEY="" (explicitly empty), so a plain load_dotenv() would
# silently skip it.  With override=True the value from .env wins.
try:
    from dotenv import load_dotenv as _load_dotenv
    _load_dotenv(override=True)
except ImportError:
    pass  # python-dotenv not installed; rely on the shell environment

# ---------------------------------------------------------------------------
# Master resume
# ---------------------------------------------------------------------------

MASTER_DOC_ID = os.environ.get("MASTER_RESUME_DOC_ID", "")

# ---------------------------------------------------------------------------
# Output naming
# ---------------------------------------------------------------------------

TAILORED_FOLDER_NAME = "Tailored Resumes"

# Format args: name, company, title
# e.g. "Jane Doe - ExampleCorp - Senior Widget Engineer"
NAMING_TEMPLATE = "{name} - {company} - {title}"

USER_NAME = os.environ.get("TAILOR_USER_NAME", "Your Name")
FIRST_NAME = USER_NAME.split()[0] if USER_NAME.strip() else "the candidate"

# ---------------------------------------------------------------------------
# File paths (relative to this module's directory)
# ---------------------------------------------------------------------------

_MODULE_DIR = os.path.dirname(__file__)
_CONFIG_DIR = os.path.join(_MODULE_DIR, "config")

# client_secret.json downloaded from Google Cloud Console
CLIENT_SECRET_PATH = os.environ.get(
    "GOOGLE_CLIENT_SECRET_PATH",
    os.path.join(_CONFIG_DIR, "client_secret.json"),
)

# OAuth token (created automatically on first auth; refresh token persisted here)
TOKEN_PATH = os.environ.get(
    "GOOGLE_TOKEN_PATH",
    os.path.join(_CONFIG_DIR, "token.json"),
)

# .docx exports land here during processing
DOCX_OUTPUT_DIR = os.environ.get(
    "TAILOR_DOCX_DIR",
    os.path.join(_MODULE_DIR, "..", "data", "tailored_resumes"),
)

# ---------------------------------------------------------------------------
# OAuth scopes
# ---------------------------------------------------------------------------

OAUTH_SCOPES = [
    "https://www.googleapis.com/auth/documents",
    "https://www.googleapis.com/auth/drive.file",
    "https://www.googleapis.com/auth/drive",
]

# ---------------------------------------------------------------------------
# LLM (Claude CLI)
# ---------------------------------------------------------------------------

# Tailor steps split across two models via the local `claude` CLI (subscription
# billing). The analysis steps run on Sonnet; only the quality-critical edit
# generation runs on Opus. This keeps one tailor run from spending the whole
# subscription usage window on Opus for work Sonnet handles just as well.
TAILOR_MODEL = "claude-sonnet-5"      # keyword extraction, gap analysis, JD cleanup, layout reshape
TAILOR_EDIT_MODEL = os.environ.get("TAILOR_EDIT_MODEL", "claude-opus-4-8")   # edit generation + correction loops

# Keyword density targets for ATS
KEYWORD_MIN = 25
KEYWORD_MAX = 35
KEYWORD_CORRECTION_ROUNDS = 2  # Max LLM retry loops to hit density target

# Approximate word budget for a one-page resume (conservative)
ONE_PAGE_WORD_BUDGET = 500
