"""
Base classes for job scrapers.
"""
import logging
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from html import unescape
from typing import List, Optional

logger = logging.getLogger(__name__)

# --- Description HTML sanitization -------------------------------------------
# Job descriptions arrive from many APIs in inconsistent shapes: some are plain
# text, some are raw HTML (<div>/<p>/<li>), and some are ENTITY-ENCODED HTML
# (&lt;li&gt;…). Rendered verbatim in the dashboard these show up as literal
# tags. clean_description() normalizes them ALL to readable plain text, and is
# applied centrally in JobPosting.__post_init__ so every scraper — current and
# future — stores clean text. It is idempotent (safe to re-apply) and preserves
# list/paragraph breaks so the dashboard's bullet/paragraph parser still works.
_LI_OPEN_RE = re.compile(r"(?i)<li[^>]*>")
_BR_RE = re.compile(r"(?i)<br\s*/?>")
_BLOCK_CLOSE_RE = re.compile(
    r"(?i)</(p|div|li|ul|ol|h[1-6]|tr|section|article|blockquote)\s*>"
)
_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
# Real tags start with a letter after "<" or "</". The anchor keeps prose
# comparisons intact: without it, "if a < b then x > y" loses everything
# between the brackets.
_TAG_RE = re.compile(r"</?[a-zA-Z][^>]*>")

# Sanity ceiling applied AFTER cleaning (in JobPosting.__post_init__), so
# markup overhead never eats the budget. This is a bound against pathological
# inputs (a scraper storing a whole page dump), NOT a scoring decision:
#  - stored text is exactly what the keyword scorer reads (both at scrape time
#    and on --reblend), so the ceiling must clear every real JD — max observed
#    in the DB is ~23k chars and the MEDIAN is ~5.3k. A 5000 cap here bisected
#    the corpus and pushed 34 live senior-AI roles below the alert line.
#  - it does NOT bound LLM spend: the LLM prompt has its own tighter cap
#    (_LLM_DESC_CHARS in engine/llm_scorer.py).
# If prefix-scoring is ever wanted as length normalization, implement it as an
# explicit [:N] inside engine/scorer.py where it is visible as scoring policy.
DESCRIPTION_MAX_LEN = 30_000


def clean_description(raw: Optional[str]) -> str:
    """Strip HTML to readable plain text, preserving list/paragraph breaks.

    Handles entity-encoded HTML by fully decoding entities BEFORE stripping
    tags (so "&lt;li&gt;" becomes a bullet, not the literal text "<li>").
    Idempotent: re-applying to already-clean text returns it unchanged.
    """
    if not raw:
        return ""

    # Fully decode entities first — repeatedly, to handle double-encoding
    # (&amp;lt; → &lt; → <). Bounded: each pass strictly shrinks the entity set.
    text, prev = raw, None
    while text != prev:
        prev = text
        text = unescape(text)

    # Preserve structure the dashboard renders: list items → bullet lines,
    # line breaks and block-element ends → newlines.
    text = _LI_OPEN_RE.sub("\n• ", text)
    text = _BR_RE.sub("\n", text)
    text = _BLOCK_CLOSE_RE.sub("\n", text)
    # Strip comments first (they may contain tags), then any remaining tags.
    text = _COMMENT_RE.sub(" ", text)
    text = _TAG_RE.sub(" ", text)
    # A hard character cap upstream can truncate the input mid-tag, leaving a
    # dangling fragment with no closing ">" that the tag regex can't match:
    # "<div class=…", "</h2&g", or a bare "</" at the very end. Drop it. The
    # letter-after-"<" anchor avoids eating legitimate prose like "a < b".
    text = re.sub(r"</?[a-zA-Z][^>]*$|</\s*$", "", text)
    text = text.replace("\xa0", " ")  # non-breaking space → normal space

    # Normalize whitespace: collapse runs of spaces/tabs, trim each line, and
    # cap consecutive blank lines at one.
    out: List[str] = []
    blanks = 0
    for line in text.split("\n"):
        line = re.sub(r"[ \t]+", " ", line).strip()
        if line:
            out.append(line)
            blanks = 0
        elif out:
            blanks += 1
            if blanks == 1:
                out.append("")
    return "\n".join(out).strip()


@dataclass
class JobPosting:
    """Represents a single job posting."""
    title: str
    company: str
    location: str
    url: str
    description: str
    salary_min: Optional[float] = None
    salary_max: Optional[float] = None
    date_posted: Optional[str] = None
    source: str = ""
    score: int = 0
    status: str = "new"
    match_explanation: str = ""

    def __post_init__(self):
        # Central sanitization: every scraper builds a JobPosting, so cleaning
        # the description here guarantees no raw/entity-encoded HTML reaches the
        # DB, the LLM scorer, or the dashboard — regardless of the source.
        # The length cap lives here too, after cleaning, so markup overhead
        # never counts against the budget.
        self.description = clean_description(self.description)[:DESCRIPTION_MAX_LEN]


class BaseScraper(ABC):
    """Abstract base class for all job scrapers."""

    # Short source label used in the detail-fetch run summary (matches the
    # JobPosting.source value). Falls back to the class name if unset.
    source_name: str = ""

    # Detail-fetch failures must be visible at production log level (INFO):
    # the first DETAIL_FAILURE_WARN_LIMIT failures per run log at WARNING
    # (enough to see WHAT is failing), the rest at DEBUG, and the scrape_all
    # summary always carries the full count. Above DETAIL_FAILURE_ERROR_RATE
    # the summary escalates to ERROR so a systemic failure (403 blocking,
    # endpoint change) lands in logs/errors.log instead of vanishing.
    DETAIL_FAILURE_WARN_LIMIT = 5
    DETAIL_FAILURE_ERROR_RATE = 0.5

    def __init__(self, config: dict):
        self.config = config
        self.scraping_config = config.get("scraping", {})
        self.max_results = self.scraping_config.get("max_results_per_source", 50)
        self.rate_limit = self.scraping_config.get("rate_limit_seconds", 2)
        # URLs already stored with a non-empty description (seeded by main.py
        # under the internal "_urls_with_descriptions" config key). Scrapers
        # that fetch per-job detail records skip the fetch for these — in
        # steady-state daily runs nearly every posting is already stored, and
        # re-downloading its detail page buys nothing.
        self.known_description_urls: set = set(
            config.get("_urls_with_descriptions") or ()
        )
        # Per-run detail-fetch counters (network attempts only — cache hits
        # and known-URL skips don't count). attempts = successes + failures.
        self._detail_fetch_attempts = 0
        self._detail_fetch_failures = 0
        self.logger = logging.getLogger(self.__class__.__name__)

    def _count_detail_fetch_success(self) -> None:
        """Record one successful per-job detail fetch."""
        self._detail_fetch_attempts += 1

    def _log_detail_fetch_failure(self, msg: str, *args) -> None:
        """Record one failed per-job detail fetch and log it.

        The first DETAIL_FAILURE_WARN_LIMIT failures per run log at WARNING;
        the rest at DEBUG (per-job WARNINGs would drown the log when a whole
        tenant is blocked). The scrape_all summary reports the full count.
        """
        self._detail_fetch_attempts += 1
        self._detail_fetch_failures += 1
        if self._detail_fetch_failures <= self.DETAIL_FAILURE_WARN_LIMIT:
            self.logger.warning(msg, *args)
            if self._detail_fetch_failures == self.DETAIL_FAILURE_WARN_LIMIT:
                self.logger.warning(
                    "Further detail-fetch failures this run will log at DEBUG; "
                    "the end-of-run summary carries the totals."
                )
        else:
            self.logger.debug(msg, *args)

    def _log_detail_fetch_summary(self) -> None:
        """Emit the per-run detail-fetch summary line.

        Called at the end of scrape_all. Silent when no detail fetch was
        attempted (scrapers without detail endpoints, or steady-state runs
        where every URL was already stored with a description).
        """
        attempts = self._detail_fetch_attempts
        failures = self._detail_fetch_failures
        if attempts == 0:
            return
        name = self.source_name or self.__class__.__name__
        if failures == 0:
            self.logger.info("%s: 0/%d detail fetches failed", name, attempts)
        elif failures / attempts > self.DETAIL_FAILURE_ERROR_RATE:
            self.logger.error(
                "%s: %d/%d detail fetches failed — possible systemic blocking "
                "or endpoint change", name, failures, attempts,
            )
        else:
            self.logger.warning(
                "%s: %d/%d detail fetches failed", name, failures, attempts
            )

    @abstractmethod
    def scrape(self, query: str) -> List[JobPosting]:
        """Scrape jobs for a given query. Returns list of JobPosting objects."""
        pass

    def scrape_all(self, queries: List[str]) -> List[JobPosting]:
        """
        Scrape jobs for all queries, deduplicate by URL, respecting rate limiting.
        Per-query exceptions are caught and logged so one bad query doesn't stop the batch.
        """
        all_jobs: List[JobPosting] = []
        seen_urls: set = set()

        for i, query in enumerate(queries):
            try:
                self.logger.info("Scraping query %d/%d: %s", i + 1, len(queries), query)
                jobs = self.scrape(query)
                new_jobs = 0
                for job in jobs:
                    if job.url and job.url not in seen_urls:
                        seen_urls.add(job.url)
                        all_jobs.append(job)
                        new_jobs += 1
                self.logger.info("Query '%s' returned %d new jobs (%d total)", query, new_jobs, len(jobs))
            except Exception as exc:
                self.logger.error("Error scraping query '%s': %s", query, exc, exc_info=True)

            # Rate limit between queries (skip sleep after last query)
            if i < len(queries) - 1:
                time.sleep(self.rate_limit)

        self.logger.info("Total unique jobs scraped: %d", len(all_jobs))
        self._log_detail_fetch_summary()
        return all_jobs

    def _normalize_location(self, location: str) -> str:
        """Normalize location string by stripping whitespace and standardizing common patterns."""
        if not location:
            return ""
        loc = location.strip()
        # Normalize common remote patterns
        loc_lower = loc.lower()
        if any(term in loc_lower for term in ["remote", "work from home", "wfh", "distributed", "anywhere"]):
            return "Remote"
        return loc

    def _is_remote(self, job: "JobPosting") -> bool:
        """
        Check if a job is remote by examining location and description.
        Returns True if remote indicators are found.
        """
        remote_terms = ["remote", "work from home", "wfh", "distributed", "anywhere"]
        location_lower = (job.location or "").lower()
        description_lower = (job.description or "").lower()

        for term in remote_terms:
            if term in location_lower or term in description_lower:
                return True
        return False
