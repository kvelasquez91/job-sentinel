"""
Company intelligence module for Job Sentinel.
Fetches financial health and culture signals for hiring companies.
Caches results in SQLite company_insights table (TTL: 48h).

Free data sources used:
  - Yahoo Finance (yfinance) — stock trends for public companies
  - Wikipedia REST API      — headcount estimates
  - Google News RSS         — general news sentiment
  - Google News RSS         — layoff-specific signals
  - Glassdoor (via Google search snippet) — ratings (overall, culture, WLB, management)
"""
import json
import logging
import re
import sqlite3
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Ticker map for well-known companies (supplemented by yfinance search)
# ---------------------------------------------------------------------------
KNOWN_TICKERS: Dict[str, str] = {
    "google": "GOOGL",
    "alphabet": "GOOGL",
    "meta": "META",
    "facebook": "META",
    "microsoft": "MSFT",
    "amazon": "AMZN",
    "apple": "AAPL",
    "netflix": "NFLX",
    "salesforce": "CRM",
    "nvidia": "NVDA",
    "palantir": "PLTR",
    "snowflake": "SNOW",
    "intel": "INTC",
    "adobe": "ADBE",
    "oracle": "ORCL",
    "ibm": "IBM",
    "uber": "UBER",
    "lyft": "LYFT",
    "airbnb": "ABNB",
    "doordash": "DASH",
    "coinbase": "COIN",
    "spotify": "SPOT",
    "shopify": "SHOP",
    "twilio": "TWLO",
    "zendesk": "ZEN",
    "workday": "WDAY",
    "servicenow": "NOW",
    "datadog": "DDOG",
    "cloudflare": "NET",
    "mongodb": "MDB",
    "elastic": "ESTC",
    "okta": "OKTA",
    "zoom": "ZM",
    "slack": "WORK",
    "dropbox": "DBX",
    "box": "BOX",
    "github": "MSFT",  # owned by Microsoft
    # New public companies
    "tsmc": "TSM",
    "taiwan semiconductor": "TSM",
    "asml": "ASML",
    "broadcom": "AVGO",
    "arista networks": "ANET",
    "palo alto networks": "PANW",
    "visa": "V",
    "mastercard": "MA",
    "jpmorgan": "JPM",
    "jpmorgan chase": "JPM",
    "eli lilly": "LLY",
    "lilly": "LLY",
    "unitedhealth": "UNH",
    "unitedhealth group": "UNH",
}

CACHE_TTL_HOURS = 48

# Google search URL for Glassdoor snippet scraping
_GOOGLE_SEARCH_URL = "https://www.google.com/search"
# Headers that mimic a real browser for search scraping
_SEARCH_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

# Sentiment keyword sets
_NEGATIVE_WORDS = {
    "layoff", "layoffs", "laid", "fired", "bankrupt", "bankruptcy",
    "loss", "losses", "cuts", "cutting", "lawsuit", "fine", "fraud",
    "scandal", "crash", "plunge", "declining", "restructuring",
    "downsizing", "closure", "shutting", "shutdown",
}
_POSITIVE_WORDS = {
    "funding", "raises", "raised", "series", "ipo", "profit", "profits",
    "record", "growth", "launch", "expansion", "acquires", "acquired",
    "partnership", "hiring", "milestone", "breakthrough", "investment",
    "revenue", "valuation",
}


def _normalize(name: str) -> str:
    """Normalize a company name for use as a cache key."""
    return re.sub(r"[^\w\s]", "", name.lower()).strip()


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class CompanyIntelligence:
    """Fetches and caches company financial health and culture signals."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._session = requests.Session()
        self._session.headers.update({
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept-Language": "en-US,en;q=0.9",
        })

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_or_fetch(self, company_name: str) -> Dict[str, Any]:
        """Return cached insights or fetch fresh data for a company."""
        normalized = _normalize(company_name)

        cached = self._get_cached(normalized)
        if cached:
            logger.debug("Cache hit: %s", company_name)
            return cached

        logger.info("Fetching company intel: %s", company_name)
        insights = self._fetch_all(company_name, normalized)
        self._save(insights)
        return insights

    def batch_enrich(
        self, companies: List[str], delay: float = 1.5
    ) -> Dict[str, Dict[str, Any]]:
        """
        Enrich a list of companies with rate limiting.
        Returns {company_name: insights_dict}.
        """
        results: Dict[str, Dict[str, Any]] = {}
        seen: set = set()
        unique = [c for c in companies if c and c not in seen and not seen.add(c)]  # type: ignore[func-returns-value]

        for i, company in enumerate(unique):
            results[company] = self.get_or_fetch(company)
            if i < len(unique) - 1:
                time.sleep(delay)

        return results

    # ------------------------------------------------------------------
    # Cache
    # ------------------------------------------------------------------

    def _get_cached(self, normalized: str) -> Optional[Dict[str, Any]]:
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                """SELECT * FROM company_insights
                   WHERE company_name_normalized = ?
                     AND fetched_at > datetime('now', '-' || cache_ttl_hours || ' hours')""",
                (normalized,),
            ).fetchone()
            conn.close()
            return dict(row) if row else None
        except Exception as e:
            logger.warning("Cache lookup error for %s: %s", normalized, e)
            return None

    def _save(self, insights: Dict[str, Any]) -> None:
        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute("PRAGMA busy_timeout=30000")
            conn.execute(
                """INSERT OR REPLACE INTO company_insights
                   (company_name, company_name_normalized, is_public, stock_ticker,
                    stock_trend, stock_change_30d, headcount_estimate, headcount_trend,
                    has_recent_layoffs, layoff_details, recent_news_sentiment,
                    recent_headlines, glassdoor_overall, glassdoor_culture,
                    glassdoor_wlb, glassdoor_management,
                    health_score, health_summary, health_flags,
                    data_sources, fetched_at, cache_ttl_hours)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    insights["company_name"],
                    insights["company_name_normalized"],
                    insights["is_public"],
                    insights["stock_ticker"],
                    insights["stock_trend"],
                    insights["stock_change_30d"],
                    insights["headcount_estimate"],
                    insights["headcount_trend"],
                    insights["has_recent_layoffs"],
                    insights["layoff_details"],
                    insights["recent_news_sentiment"],
                    insights["recent_headlines"],
                    insights.get("glassdoor_overall"),
                    insights.get("glassdoor_culture"),
                    insights.get("glassdoor_wlb"),
                    insights.get("glassdoor_management"),
                    insights["health_score"],
                    insights["health_summary"],
                    insights["health_flags"],
                    insights["data_sources"],
                    insights["fetched_at"],
                    insights["cache_ttl_hours"],
                ),
            )
            conn.commit()
            conn.close()
        except Exception as e:
            logger.warning(
                "Failed to save company insights for %s: %s",
                insights.get("company_name"), e,
            )

    # ------------------------------------------------------------------
    # Data fetching
    # ------------------------------------------------------------------

    def _fetch_all(self, company_name: str, normalized: str) -> Dict[str, Any]:
        """Fetch all signals and return a complete insights dict."""
        insights: Dict[str, Any] = {
            "company_name": company_name,
            "company_name_normalized": normalized,
            "is_public": 0,
            "stock_ticker": None,
            "stock_trend": "unknown",
            "stock_change_30d": None,
            "headcount_estimate": None,
            "headcount_trend": "unknown",
            "has_recent_layoffs": 0,
            "layoff_details": "[]",
            "recent_news_sentiment": "neutral",
            "recent_headlines": "[]",
            "glassdoor_overall": None,
            "glassdoor_culture": None,
            "glassdoor_wlb": None,
            "glassdoor_management": None,
            "health_score": 50,
            "health_summary": "Limited data available",
            "health_flags": "[]",
            "data_sources": "[]",
            "fetched_at": datetime.now().isoformat(),
            "cache_ttl_hours": CACHE_TTL_HOURS,
        }
        sources: List[str] = []

        # 1. Stock data (yfinance)
        ticker = self._resolve_ticker(normalized, company_name)
        if ticker:
            stock = self._fetch_stock(ticker)
            if stock:
                insights.update(stock)
                sources.append("yahoo_finance")

        # 2. Wikipedia headcount
        wiki = self._fetch_wikipedia(company_name)
        if wiki:
            insights.update(wiki)
            sources.append("wikipedia")

        # 3. General news sentiment (Google News RSS)
        news = self._fetch_news_sentiment(company_name)
        if news:
            insights.update(news)
            sources.append("google_news")

        # 4. Layoff signals (targeted Google News RSS query)
        layoffs = self._fetch_layoff_signals(company_name)
        if layoffs:
            insights.update(layoffs)
            sources.append("layoff_news")

        # 5. Glassdoor ratings (via Google search snippet)
        glassdoor = self._fetch_glassdoor(company_name)
        if glassdoor:
            insights.update(glassdoor)
            sources.append("glassdoor")

        insights["data_sources"] = json.dumps(sources)

        score, summary, flags = self._compute_health(insights)
        insights["health_score"] = score
        insights["health_summary"] = summary
        insights["health_flags"] = json.dumps(flags)
        return insights

    def _resolve_ticker(self, normalized: str, company_name: str) -> Optional[str]:
        """Find a stock ticker for a company, or None if private/unknown."""
        # Hardcoded map
        if normalized in KNOWN_TICKERS:
            return KNOWN_TICKERS[normalized]
        # Try first word (e.g. "stripe inc" → "stripe")
        first = normalized.split()[0] if normalized.split() else ""
        if first and first in KNOWN_TICKERS:
            return KNOWN_TICKERS[first]

        # yfinance search as fallback
        try:
            import yfinance as yf  # optional dependency
            results = yf.Search(company_name, max_results=1, news_count=0)
            quotes = getattr(results, "quotes", [])
            if quotes:
                sym = quotes[0].get("symbol", "")
                exch = quotes[0].get("exchange", "")
                # Only trust major US exchanges to avoid false matches
                if sym and exch in ("NMS", "NYQ", "NGM", "PCX", "BATS"):
                    return sym
        except Exception:
            pass
        return None

    def _fetch_stock(self, ticker: str) -> Optional[Dict[str, Any]]:
        """Return 30-day stock change stats via yfinance."""
        try:
            import yfinance as yf
            hist = yf.Ticker(ticker).history(period="1mo")
            if hist.empty or len(hist) < 2:
                return None
            start_price = hist["Close"].iloc[0]
            end_price = hist["Close"].iloc[-1]
            change_pct = round(((end_price - start_price) / start_price) * 100, 2)
            if change_pct > 5:
                trend = "up"
            elif change_pct < -5:
                trend = "down"
            else:
                trend = "stable"
            return {
                "is_public": 1,
                "stock_ticker": ticker,
                "stock_trend": trend,
                "stock_change_30d": change_pct,
            }
        except Exception as e:
            logger.warning("yfinance error for %s: %s", ticker, e)
            return None

    # Wikipedia requires a descriptive User-Agent; browser UAs get blocked/empty responses.
    _WIKI_HEADERS = {
        "User-Agent": "JobSentinel/1.0 (job-sentinel-app; https://github.com/local/job-sentinel)",
        "Accept": "application/json",
    }

    def _fetch_wikipedia(self, company_name: str) -> Optional[Dict[str, Any]]:
        """Extract headcount from Wikipedia infobox via MediaWiki API."""
        try:
            api = "https://en.wikipedia.org/w/api.php"
            # Step 1: search for article
            search_resp = self._session.get(
                api,
                params={
                    "action": "opensearch",
                    "search": company_name,
                    "limit": 3,
                    "format": "json",
                },
                headers=self._WIKI_HEADERS,
                timeout=10,
            )
            if not search_resp.content or search_resp.status_code != 200:
                logger.debug(
                    "Wikipedia search empty response for %s (status %d)",
                    company_name, search_resp.status_code,
                )
                return None
            search_data = search_resp.json()
            titles: List[str] = search_data[1] if len(search_data) > 1 else []
            if not titles:
                return None

            # Prefer an exact or "Inc/Corp" match
            page_title = titles[0]
            cn_lower = company_name.lower()
            for t in titles:
                tl = t.lower()
                if cn_lower in tl and any(
                    w in tl for w in ("inc", "corp", "llc", "ltd", "limited")
                ):
                    page_title = t
                    break

            # Step 2: fetch wikitext of section 0 (intro + infobox)
            content_resp = self._session.get(
                api,
                params={
                    "action": "query",
                    "titles": page_title,
                    "prop": "revisions",
                    "rvprop": "content",
                    "rvslots": "main",
                    "rvsection": "0",
                    "format": "json",
                },
                headers=self._WIKI_HEADERS,
                timeout=10,
            )
            if not content_resp.content or content_resp.status_code != 200:
                logger.debug(
                    "Wikipedia content empty response for %s (status %d)",
                    company_name, content_resp.status_code,
                )
                return None
            pages = content_resp.json().get("query", {}).get("pages", {})
            page = next(iter(pages.values()), {})
            wikitext = (
                page.get("revisions", [{}])[0]
                .get("slots", {})
                .get("main", {})
                .get("*", "")
            )

            result: Dict[str, Any] = {}
            emp_match = re.search(
                r"\|\s*num_employees\s*=\s*([\d,]+)",
                wikitext,
                re.IGNORECASE,
            )
            if emp_match:
                digits = re.sub(r"[^\d]", "", emp_match.group(1))
                if digits:
                    result["headcount_estimate"] = f"{int(digits):,}"
            return result or None

        except Exception as e:
            logger.warning("Wikipedia fetch failed for %s: %s", company_name, e)
            return None

    def _fetch_news_sentiment(self, company_name: str) -> Optional[Dict[str, Any]]:
        """Fetch recent headlines via Google News RSS and classify sentiment."""
        try:
            query = f'"{company_name}"'
            url = (
                "https://news.google.com/rss/search"
                f"?q={requests.utils.quote(query)}&hl=en-US&gl=US&ceid=US:en"
            )
            resp = self._session.get(url, timeout=12)
            if resp.status_code != 200:
                return None

            soup = BeautifulSoup(resp.content, "lxml-xml")
            items = soup.find_all("item")[:12]
            if not items:
                return None

            headlines: List[str] = []
            signals: List[str] = []

            for item in items:
                title_el = item.find("title")
                if not title_el:
                    continue
                title = title_el.get_text(strip=True)
                headlines.append(title)

                words = set(re.findall(r"\w+", title.lower()))
                if words & _NEGATIVE_WORDS:
                    signals.append("negative")
                elif words & _POSITIVE_WORDS:
                    signals.append("positive")
                else:
                    signals.append("neutral")

            neg = signals.count("negative")
            pos = signals.count("positive")
            if neg >= 2 and neg > pos:
                sentiment = "negative"
            elif pos >= 2 and pos > neg:
                sentiment = "positive"
            elif neg > 0 or pos > 0:
                sentiment = "mixed"
            else:
                sentiment = "neutral"

            return {
                "recent_news_sentiment": sentiment,
                "recent_headlines": json.dumps(headlines[:5]),
            }
        except Exception as e:
            logger.warning("News fetch failed for %s: %s", company_name, e)
            return None

    def _fetch_layoff_signals(self, company_name: str) -> Optional[Dict[str, Any]]:
        """Search Google News RSS specifically for layoff news."""
        try:
            query = f'"{company_name}" layoffs OR "laid off" OR "job cuts"'
            url = (
                "https://news.google.com/rss/search"
                f"?q={requests.utils.quote(query)}&hl=en-US&gl=US&ceid=US:en"
            )
            resp = self._session.get(url, timeout=12)
            if resp.status_code != 200:
                return None

            soup = BeautifulSoup(resp.content, "lxml-xml")
            items = soup.find_all("item")[:10]
            if not items:
                return None

            layoff_hits: List[Dict[str, str]] = []
            company_first = company_name.lower().split()[0]

            for item in items:
                title_el = item.find("title")
                pub_el = item.find("pubDate")
                if not title_el:
                    continue
                title = title_el.get_text(strip=True)
                pub_date = pub_el.get_text(strip=True) if pub_el else ""
                title_lower = title.lower()

                if company_first in title_lower and any(
                    w in title_lower
                    for w in ("layoff", "laid off", "job cut", "cut jobs", "fired")
                ):
                    layoff_hits.append({"headline": title, "date": pub_date})

            if not layoff_hits:
                return None

            return {
                "has_recent_layoffs": 1,
                "layoff_details": json.dumps(layoff_hits[:3]),
            }
        except Exception as e:
            logger.warning("Layoff check failed for %s: %s", company_name, e)
            return None

    def _fetch_glassdoor(self, company_name: str) -> Optional[Dict[str, Any]]:
        """
        Attempt to scrape Glassdoor ratings for a company.

        Strategy:
          1. Google search: site:glassdoor.com "{company}" reviews
             Parse the star rating from the search result snippet.
          2. If that yields nothing, try a direct Glassdoor search page.

        Returns dict with keys: glassdoor_overall, glassdoor_culture,
        glassdoor_wlb, glassdoor_management (all floats or None).
        Returns None if nothing is found.
        """
        # --- Strategy 1: Google search snippet ---
        try:
            query = f'site:glassdoor.com "{company_name}" reviews'
            resp = self._session.get(
                _GOOGLE_SEARCH_URL,
                params={"q": query, "num": 5, "hl": "en"},
                headers=_SEARCH_HEADERS,
                timeout=10,
            )
            if resp.status_code == 200:
                result = self._parse_glassdoor_from_google(resp.text, company_name)
                if result:
                    logger.debug(
                        "Glassdoor via Google snippet for %s: overall=%.1f",
                        company_name, result.get("glassdoor_overall", 0),
                    )
                    return result
        except Exception as e:
            logger.debug("Glassdoor Google search failed for %s: %s", company_name, e)

        # --- Strategy 2: Glassdoor search API ---
        try:
            slug = company_name.lower().replace(" ", "-").replace(",", "")
            url = (
                f"https://www.glassdoor.com/Reviews/{slug}-reviews-SRCH_KE0,{len(slug)}.htm"
            )
            resp = self._session.get(url, headers=_SEARCH_HEADERS, timeout=12)
            if resp.status_code == 200:
                result = self._parse_glassdoor_page(resp.text)
                if result:
                    logger.debug(
                        "Glassdoor direct page for %s: overall=%.1f",
                        company_name, result.get("glassdoor_overall", 0),
                    )
                    return result
        except Exception as e:
            logger.debug("Glassdoor direct page failed for %s: %s", company_name, e)

        return None

    def _parse_glassdoor_from_google(
        self, html: str, company_name: str
    ) -> Optional[Dict[str, Any]]:
        """Extract Glassdoor rating from a Google search results page."""
        soup = BeautifulSoup(html, "lxml")
        text = soup.get_text(" ", strip=True)

        # Look for rating patterns like "4.2 stars", "Rating: 3.8", "4.2/5"
        rating_pat = re.compile(
            r"(?:glassdoor[^\d]{0,30}|rating[:\s]+|overall[:\s]+)"
            r"(\d\.\d)\s*(?:/\s*5|stars?|out)",
            re.IGNORECASE,
        )
        m = rating_pat.search(text)
        if not m:
            # Simpler fallback: look for a standalone X.X pattern near "glassdoor"
            gl_idx = text.lower().find("glassdoor")
            if gl_idx >= 0:
                snippet = text[gl_idx: gl_idx + 200]
                m2 = re.search(r"\b([1-5]\.\d)\b", snippet)
                if m2:
                    try:
                        overall = float(m2.group(1))
                        if 1.0 <= overall <= 5.0:
                            return {"glassdoor_overall": overall}
                    except ValueError:
                        pass
            return None

        try:
            overall = float(m.group(1))
            if 1.0 <= overall <= 5.0:
                return {"glassdoor_overall": overall}
        except ValueError:
            pass
        return None

    def _parse_glassdoor_page(self, html: str) -> Optional[Dict[str, Any]]:
        """Extract ratings from a Glassdoor reviews page."""
        soup = BeautifulSoup(html, "lxml")
        result: Dict[str, Any] = {}

        # Overall rating — commonly in a span/div with class containing "rating" or "ratingNum"
        for sel in [
            '[class*="ratingNum"]',
            '[class*="rating-num"]',
            '[data-test="rating-info-overall"]',
            '[class*="overallRating"]',
        ]:
            el = soup.select_one(sel)
            if el:
                try:
                    val = float(el.get_text(strip=True))
                    if 1.0 <= val <= 5.0:
                        result["glassdoor_overall"] = val
                        break
                except ValueError:
                    pass

        # Sub-ratings via text search (Glassdoor embeds them as JSON or labeled text)
        text = soup.get_text(" ", strip=True)

        def _find_sub(pattern: str) -> Optional[float]:
            m = re.search(pattern, text, re.IGNORECASE)
            if m:
                try:
                    v = float(m.group(1))
                    return v if 1.0 <= v <= 5.0 else None
                except ValueError:
                    pass
            return None

        culture = _find_sub(r"culture\s*(?:&|and)\s*values?[^\d]{0,20}(\d\.\d)")
        wlb = _find_sub(r"work[- ]life\s*balance[^\d]{0,20}(\d\.\d)")
        mgmt = _find_sub(r"(?:senior\s*)?management[^\d]{0,20}(\d\.\d)")

        if culture:
            result["glassdoor_culture"] = culture
        if wlb:
            result["glassdoor_wlb"] = wlb
        if mgmt:
            result["glassdoor_management"] = mgmt

        return result if result else None

    # ------------------------------------------------------------------
    # Health score
    # ------------------------------------------------------------------

    def _compute_health(
        self, insights: Dict[str, Any]
    ) -> Tuple[int, str, List[Dict[str, str]]]:
        """Return (score 0-100, summary text, list of flag dicts)."""
        score = 50
        flags: List[Dict[str, str]] = []

        # ---- Layoffs: big red flag ----
        if insights.get("has_recent_layoffs"):
            score -= 25
            try:
                details = json.loads(insights.get("layoff_details", "[]"))
                headline = details[0]["headline"] if details else "Recent layoffs reported"
            except Exception:
                headline = "Recent layoffs reported"
            short = headline[:80] + ("..." if len(headline) > 80 else "")
            flags.append({"type": "red", "label": "Layoffs", "text": short})

        # ---- Stock performance ----
        change = insights.get("stock_change_30d")
        if change is not None:
            if change >= 20:
                score += 15
                flags.append({"type": "green", "label": "Stock", "text": f"+{change:.1f}% (30d) — strong uptrend"})
            elif change >= 5:
                score += 8
                flags.append({"type": "green", "label": "Stock", "text": f"+{change:.1f}% (30d)"})
            elif change <= -20:
                score -= 15
                flags.append({"type": "red", "label": "Stock", "text": f"{change:.1f}% (30d) — sharp decline"})
            elif change <= -5:
                score -= 8
                flags.append({"type": "yellow", "label": "Stock", "text": f"{change:.1f}% (30d)"})
            else:
                flags.append({"type": "gray", "label": "Stock", "text": f"{change:+.1f}% (30d) — stable"})

        # ---- News sentiment ----
        sentiment = insights.get("recent_news_sentiment", "neutral")
        if sentiment == "positive":
            score += 10
            flags.append({"type": "green", "label": "News", "text": "Positive recent coverage"})
        elif sentiment == "negative":
            score -= 12
            flags.append({"type": "red", "label": "News", "text": "Negative recent coverage"})
        elif sentiment == "mixed":
            flags.append({"type": "yellow", "label": "News", "text": "Mixed recent coverage"})
        else:
            flags.append({"type": "gray", "label": "News", "text": "Neutral / no coverage"})

        # ---- Glassdoor ratings ----
        gd_overall = insights.get("glassdoor_overall")
        if gd_overall is not None:
            if gd_overall >= 4.0:
                score += 10
                flags.append({
                    "type": "green",
                    "label": "Glassdoor",
                    "text": f"{gd_overall:.1f}/5.0 — highly rated employer",
                })
            elif gd_overall >= 3.5:
                score += 5
                flags.append({
                    "type": "green",
                    "label": "Glassdoor",
                    "text": f"{gd_overall:.1f}/5.0 — above-average employer",
                })
            elif gd_overall >= 3.0:
                flags.append({
                    "type": "yellow",
                    "label": "Glassdoor",
                    "text": f"{gd_overall:.1f}/5.0 — mixed reviews",
                })
            else:
                score -= 10
                flags.append({
                    "type": "red",
                    "label": "Glassdoor",
                    "text": f"{gd_overall:.1f}/5.0 — poor employer rating",
                })

            # Add sub-ratings as info flags
            gd_culture = insights.get("glassdoor_culture")
            gd_wlb = insights.get("glassdoor_wlb")
            gd_mgmt = insights.get("glassdoor_management")
            sub_parts: List[str] = []
            if gd_culture:
                sub_parts.append(f"Culture {gd_culture:.1f}")
            if gd_wlb:
                sub_parts.append(f"WLB {gd_wlb:.1f}")
            if gd_mgmt:
                sub_parts.append(f"Mgmt {gd_mgmt:.1f}")
            if sub_parts:
                flags.append({
                    "type": "gray",
                    "label": "GD Details",
                    "text": " · ".join(sub_parts),
                })

        # ---- Headcount (data signal only, no score impact) ----
        headcount = insights.get("headcount_estimate")
        if headcount:
            flags.append({"type": "gray", "label": "Headcount", "text": f"~{headcount} employees"})

        score = max(0, min(100, score))

        if score >= 70:
            summary = "Strong signals — healthy company outlook"
        elif score >= 50:
            summary = "Generally stable — minor considerations"
        elif score >= 30:
            summary = "Mixed signals — research recommended"
        else:
            summary = "Caution — significant risk factors present"

        return score, summary, flags
