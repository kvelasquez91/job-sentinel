"""
Hacker News "Ask HN: Who is hiring?" scraper.

Uses the free Algolia HN Search API (no auth, no bot detection):
  Story lookup:  GET https://hn.algolia.com/api/v1/search_by_date
                 ?query=Ask HN: Who is hiring?&tags=story,author_whoishiring
  Comments:      GET https://hn.algolia.com/api/v1/search_by_date
                 ?tags=comment,story_{id}&hitsPerPage=100&page=N

Each TOP-LEVEL comment is one job posting, conventionally formatted
"Company | Role | Location | Salary | ...". We keep comments that mention a
product-leadership role AND remote work, and extract salary with the shared
regex. Threads are monthly; URL-dedup in the pipeline makes daily runs cheap.
"""
import logging
import re
import time
from html import unescape
from typing import Iterator, List, Optional

import requests
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from profile_policy import HN_ROLE_RE as _ROLE_RE, HN_REMOTE_ONLY
from salary_rules import extract_salary_regex
from .base import BaseScraper, JobPosting

logger = logging.getLogger(__name__)

ALGOLIA_BASE = "https://hn.algolia.com/api/v1"
MAX_COMMENT_PAGES = 25  # 100 comments/page. The comment query returns posts AND replies
# (newest-first), so size the cap off total comment volume, not just top-level post count.

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/121.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json",
}


def _strip_html(raw: str) -> str:
    """Comment HTML -> text, turning <p> into newlines so line 1 stays line 1."""
    text = unescape(raw or "")
    text = re.sub(r"<p>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"[ \t]+", " ", text).strip()


class HNWhoIsHiringScraper(BaseScraper):
    """Scrapes the latest monthly HN Who's Hiring thread via Algolia."""

    def __init__(self, config: dict):
        super().__init__(config)
        self.session = requests.Session()
        self.session.headers.update(HEADERS)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((requests.ConnectionError, requests.Timeout)),
        reraise=True,
    )
    def _get_json(self, url: str, params: Optional[dict] = None) -> Optional[dict]:
        try:
            resp = self.session.get(url, params=params, timeout=20)
            if resp.status_code != 200:
                self.logger.warning("HN Algolia returned %d for %s", resp.status_code, url)
                return None
            return resp.json()
        except (requests.ConnectionError, requests.Timeout):
            raise
        except Exception as e:
            self.logger.warning("Unexpected error from HN Algolia %s: %s", url, e)
            return None

    def _find_latest_story(self) -> Optional[str]:
        """Return the objectID of the newest 'Ask HN: Who is hiring?' story."""
        data = self._get_json(
            f"{ALGOLIA_BASE}/search_by_date",
            {"query": "Ask HN: Who is hiring?",
             "tags": "story,author_whoishiring", "hitsPerPage": 5},
        )
        for hit in (data or {}).get("hits", []):
            if (hit.get("title") or "").lower().startswith("ask hn: who is hiring"):
                self.logger.info("HN: latest thread '%s' (id %s)", hit["title"], hit["objectID"])
                return str(hit["objectID"])
        self.logger.warning("HN: could not find a Who is Hiring story.")
        return None

    def _iter_comments(self, story_id: str) -> Iterator[dict]:
        total_pages = 0
        for page in range(MAX_COMMENT_PAGES):
            data = self._get_json(
                f"{ALGOLIA_BASE}/search_by_date",
                {"tags": f"comment,story_{story_id}", "hitsPerPage": 100, "page": page},
            )
            hits = (data or {}).get("hits") or []
            if not hits:
                return
            yield from hits
            total_pages = data.get("nbPages") or 0
            if page + 1 >= total_pages:
                return
            time.sleep(self.rate_limit)
        # Reached the page cap with pages still remaining — surface it, never truncate silently.
        self.logger.warning(
            "HN: hit MAX_COMMENT_PAGES=%d cap for story %s with nbPages=%d — some older "
            "top-level postings may be unscanned this run.",
            MAX_COMMENT_PAGES, story_id, total_pages,
        )

    def _parse_comment(self, comment: dict, story_id: str) -> Optional[JobPosting]:
        # Top-level comments only — replies are discussion, not postings.
        if str(comment.get("parent_id")) != str(story_id):
            return None
        text = _strip_html(comment.get("comment_text") or "")
        if not text:
            return None
        role_match = _ROLE_RE.search(text) if _ROLE_RE else None
        if not role_match:
            return None
        # Remote-only is owner policy (profile_policy.HN_REMOTE_ONLY); local
        # coverage comes from the ATS scrapers.
        if HN_REMOTE_ONLY and "remote" not in text.lower():
            return None

        first_line = text.split("\n", 1)[0]
        segments = [s.strip() for s in first_line.split("|") if s.strip()]
        company = (segments[0] if segments else "HN poster")[:80]
        role_segment = next((s for s in segments if _ROLE_RE.search(s)), None)
        title = (role_segment or role_match.group(0).title())[:120]

        sal_min, sal_max = extract_salary_regex(text)
        object_id = comment.get("objectID")
        if not object_id:
            return None

        return JobPosting(
            title=title,
            company=company,
            location="Remote",
            url=f"https://news.ycombinator.com/item?id={object_id}",
            description=text[:5000],
            salary_min=sal_min,
            salary_max=sal_max,
            date_posted=(comment.get("created_at") or "")[:10] or None,
            source="hn",
        )

    def scrape(self, query: str) -> List[JobPosting]:
        """One pass over the latest thread; only the first configured query
        triggers work (SmartRecruiters pattern — the thread isn't keyword-
        searchable in a useful way, we filter locally)."""
        first_query = (self.config.get("search_queries") or [""])[0]
        if query != first_query:
            return []

        story_id = self._find_latest_story()
        if not story_id:
            return []

        jobs: List[JobPosting] = []
        seen_urls: set = set()
        for comment in self._iter_comments(story_id):
            if len(jobs) >= self.max_results:
                break
            try:
                job = self._parse_comment(comment, story_id)
            except Exception as e:
                self.logger.debug("HN: error parsing comment: %s", e)
                continue
            if job and job.url not in seen_urls:
                seen_urls.add(job.url)
                jobs.append(job)

        self.logger.info("HN Who's Hiring: %d relevant jobs", len(jobs))
        return jobs
