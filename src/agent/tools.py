from __future__ import annotations

import html
import re
import unicodedata
from datetime import datetime, timedelta
from email.utils import parsedate_to_datetime
from typing import Any, Dict, List, Tuple
from urllib.parse import quote_plus, urlparse

import feedparser

from utils.logger import get_logger

logger = get_logger("AgentTools")

GOOGLE_NEWS_RSS_SEARCH_BASE = "https://news.google.com/rss/search?q={query}&hl=vi&gl=VN&ceid=VN:vi"
DEFAULT_NEWS_LOOKBACK_DAYS = 7
DEFAULT_MAX_RAW_ITEMS_PER_QUERY = 20
DEFAULT_MAX_NEWS_ITEMS = 5


def clean_html(raw_html: str) -> str:
    return html.unescape(re.sub(r"<.*?>", "", str(raw_html))).replace("\xa0", " ").strip()


def tool_search_google_news(ticker: str, run_id: str | None = None) -> Dict[str, Any]:
    _ = run_id
    ticker = str(ticker or "").upper().strip()
    keywords = _news_keywords_for_ticker(ticker)
    queries = _google_news_queries_for_ticker(ticker)
    logger.info("News scan started | ticker=%s | source=Google News RSS | keywords=%s", ticker, keywords)

    raw_items: List[Dict[str, str]] = []
    errors: List[str] = []
    sources: List[Dict[str, Any]] = []

    for query in queries:
        url = _google_news_url(query)
        logger.info("Google News RSS fetch started | url=%s", url)
        try:
            feed = feedparser.parse(url)
            entries = list(getattr(feed, "entries", []) or [])
            parse_error = getattr(feed, "bozo_exception", None) if getattr(feed, "bozo", False) else None
            if parse_error:
                raise parse_error

            sources.append({"source": "Google News RSS", "query": query, "url": url, "status": "OK", "items": len(entries)})
            logger.info("Google News RSS fetch completed | url=%s | status=OK | raw_items=%s", url, len(entries))
            for entry in entries[:DEFAULT_MAX_RAW_ITEMS_PER_QUERY]:
                raw_items.append(_entry_to_item(entry, query))
        except Exception as exc:
            errors.append(f"{query}: {type(exc).__name__}: {exc}")
            sources.append({"source": "Google News RSS", "query": query, "url": url, "status": "ERROR", "error": str(exc)})
            logger.warning("Google News RSS fetch failed | url=%s | error_type=%s | error=%s", url, type(exc).__name__, exc)

    raw_items = _filter_by_lookback(_dedupe_items(raw_items), DEFAULT_NEWS_LOOKBACK_DAYS)
    matched_items = _match_items(raw_items, keywords)[:DEFAULT_MAX_NEWS_ITEMS]
    status = "NEWS_FOUND_GOOGLE_RSS" if matched_items else ("NO_NEWS_WITH_ERRORS" if errors else "NO_NEWS")
    context = _news_context(matched_items)

    logger.info(
        "News scan completed | ticker=%s | status=%s | matched_items=%s | rss_errors=%s",
        ticker,
        status,
        len(matched_items),
        len(errors),
    )
    return {
        "news_context": context,
        "news_found": bool(matched_items),
        "news_items_count": len(matched_items),
        "news_items": matched_items,
        "evidence_level": _evidence_level(matched_items),
        "shock_type": "EVENT_DRIVEN" if matched_items else "NO_NEWS",
        "news_status": status,
        "news_sources": sources,
        "news_errors": errors,
        "news_keywords": keywords,
        "google_news_queries": queries,
        "google_news_used": True,
    }


def _entry_to_item(entry: Any, query: str) -> Dict[str, str]:
    source = entry.get("source")
    publisher = clean_html(source.get("title", "")) if isinstance(source, dict) else ""
    link = entry.get("link", "")
    return {
        "title": clean_html(entry.get("title", "")),
        "published": entry.get("published", entry.get("updated", "UNKNOWN_DATE")),
        "link": link,
        "summary": clean_html(entry.get("summary", ""))[:350],
        "publisher": publisher,
        "source_domain": _source_domain(link, publisher),
        "query": query,
    }


def _match_items(raw_items: List[Dict[str, str]], keywords: List[str]) -> List[Dict[str, Any]]:
    matched: List[Dict[str, Any]] = []
    for item in raw_items:
        is_match, matched_keywords = _match_news_item(item, keywords)
        logger.debug("News match check | matched=%s | matched_keywords=%s | title=%s", is_match, matched_keywords, item["title"])
        if not is_match:
            continue
        matched.append({**item, "matched_keywords": matched_keywords, "source": "Google News RSS"})
    return matched


def _news_context(items: List[Dict[str, Any]]) -> str:
    if not items:
        return "NO_NEWS"
    return "\n\n".join(
        f"[{item['published']}] {item['title']}\n"
        f"Summary: {item['summary']}\n"
        f"Publisher: {item.get('publisher', '')}\n"
        f"Matched keywords: {', '.join(item['matched_keywords'])}"
        for item in items
    )


def _news_keywords_for_ticker(ticker: str) -> List[str]:
    keyword_map = {
        "VIC": ["VIC", "Vingroup", "VinFast", "VFS", "Vinhomes", "VHM", "Vincom Retail", "VRE", "VPL", "cổ phiếu VIC"],
        "VHM": ["VHM", "Vinhomes", "Vingroup", "cổ phiếu VHM"],
        "VRE": ["VRE", "Vincom Retail", "Vingroup", "cổ phiếu VRE"],
        "VPL": ["VPL", "Vinpearl", "Vingroup", "cổ phiếu VPL"],
    }
    return _dedupe_preserve_order(keyword_map.get(ticker, [ticker, "Vingroup"]))


def _google_news_queries_for_ticker(ticker: str) -> List[str]:
    query_map = {
        "VIC": ['"VIC" "Vingroup"', '"cổ phiếu VIC"', '"Vingroup" "chứng khoán"', '"VinFast" "Vingroup"'],
        "VHM": ['"VHM" "Vinhomes"', '"Vinhomes" "cổ phiếu"'],
        "VRE": ['"VRE" "Vincom Retail"', '"Vincom Retail" "cổ phiếu"'],
        "VPL": ['"VPL" "Vinpearl"', '"Vinpearl" "Vingroup"'],
    }
    return query_map.get(ticker, [f'"{ticker}" "Vingroup"', f'"{ticker}" "cổ phiếu"'])


def _google_news_url(query: str) -> str:
    return GOOGLE_NEWS_RSS_SEARCH_BASE.format(query=quote_plus(query))


def _match_news_item(item: Dict[str, str], keywords: List[str]) -> Tuple[bool, List[str]]:
    content = f"{item.get('title', '')} {item.get('summary', '')}"
    content_lower = content.lower()
    content_normalized = _normalize_text(content)
    matched_keywords = [
        keyword
        for keyword in keywords
        if keyword.lower() in content_lower or _normalize_text(keyword) in content_normalized
    ]
    return bool(matched_keywords), matched_keywords


def _filter_by_lookback(items: List[Dict[str, str]], lookback_days: int) -> List[Dict[str, str]]:
    cutoff = datetime.now().astimezone() - timedelta(days=lookback_days)
    filtered = []
    for item in items:
        published_dt = _parse_published_datetime(item.get("published", ""))
        if published_dt is None or published_dt >= cutoff:
            filtered.append(item)
    return filtered


def _parse_published_datetime(value: str) -> datetime | None:
    try:
        parsed = parsedate_to_datetime(str(value))
    except (TypeError, ValueError, IndexError, OverflowError):
        return None
    return parsed.astimezone() if parsed.tzinfo is None else parsed


def _dedupe_items(items: List[Dict[str, str]]) -> List[Dict[str, str]]:
    deduped = []
    seen = set()
    for item in items:
        key = _normalize_text(item.get("title", "")) + "|" + str(item.get("link", "")).lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def _source_domain(link: str, publisher: str = "") -> str:
    domain = urlparse(str(link or "")).netloc.lower().replace("www.", "")
    if domain and domain != "news.google.com":
        return domain
    return _normalize_text(publisher).replace(" ", "_") if publisher else domain


def _evidence_level(news_items: List[Dict[str, Any]]) -> str:
    if not news_items:
        return "NONE"
    if len(news_items) == 1:
        return "LOW"
    if len(news_items) <= 3:
        return "MEDIUM"
    return "HIGH"


def _normalize_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", str(value))
    return "".join(ch for ch in normalized if not unicodedata.combining(ch)).lower()


def _dedupe_preserve_order(values: List[str]) -> List[str]:
    seen = set()
    output = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        output.append(value)
    return output
