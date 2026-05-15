from __future__ import annotations

import html
import json
import re
import unicodedata
from datetime import datetime, timedelta
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple
from urllib.parse import quote_plus, urlparse

import feedparser

from utils.logger import get_logger

logger = get_logger("AgentTools")

NEWS_DEBUG_DIR = Path("data/news_debug")
GOOGLE_NEWS_RSS_SEARCH_BASE = "https://news.google.com/rss/search?q={query}&hl=vi&gl=VN&ceid=VN:vi"
DEFAULT_NEWS_LOOKBACK_DAYS = 7
DEFAULT_MAX_RAW_NEWS_ITEMS_PER_QUERY = 20
DEFAULT_MAX_MATCHED_NEWS_ITEMS = 10


def clean_html(raw_html: str) -> str:
    cleanr = re.compile("<.*?>")
    return html.unescape(re.sub(cleanr, "", str(raw_html))).replace("\xa0", " ").strip()


def tool_search_google_news(ticker: str, run_id: str | None = None) -> Dict[str, Any]:
    ticker = str(ticker or "").upper().strip()
    keywords = _news_keywords_for_ticker(ticker)
    google_queries = _google_news_queries_for_ticker(ticker)
    logger.info(
        "News scan started | ticker=%s | source=Google News RSS | keywords=%s",
        ticker,
        keywords,
    )

    raw_news_items: List[Dict[str, str]] = []
    news_sources: List[Dict[str, Any]] = []
    news_errors: List[str] = []

    _fetch_google_news_rss_sources(
        urls=[_google_news_url(query) for query in google_queries],
        news_sources=news_sources,
        raw_news_items=raw_news_items,
        news_errors=news_errors,
        queries=google_queries,
        max_items_per_query=DEFAULT_MAX_RAW_NEWS_ITEMS_PER_QUERY,
    )

    raw_news_items = _dedupe_raw_news_items(raw_news_items)
    raw_news_items = _filter_by_lookback(raw_news_items, DEFAULT_NEWS_LOOKBACK_DAYS)
    news_items, matched_news_items = _filter_news_items(
        ticker,
        raw_news_items,
        keywords,
        max_matched_items=DEFAULT_MAX_MATCHED_NEWS_ITEMS,
    )

    status = _news_scan_status(matched_items=len(news_items), error_count=len(news_errors))
    debug_path = _write_news_debug_artifact(
        ticker=ticker,
        run_id=run_id,
        news_sources=news_sources,
        keywords=keywords,
        google_news_queries=google_queries,
        raw_news_items=raw_news_items,
        matched_news_items=matched_news_items,
        status=status,
        news_errors_count=len(news_errors),
    )

    if not news_items:
        logger.info(
            "News scan completed | ticker=%s | status=%s | raw_news_items=%s | matched_news_items=0 | news_errors=%s",
            ticker,
            status,
            len(raw_news_items),
            len(news_errors),
        )
        return {
            "news_context": "NO_NEWS",
            "news_found": False,
            "news_items_count": 0,
            "news_items": [],
            "evidence_level": "NONE",
            "shock_type": "NO_NEWS",
            "news_errors": news_errors,
            "news_sources": news_sources,
            "raw_news_items_count": len(raw_news_items),
            "matched_news_items_count": 0,
            "news_errors_count": len(news_errors),
            "news_status": status,
            "news_debug_path": str(debug_path),
            "news_keywords": keywords,
            "google_news_queries": google_queries,
            "google_news_used": True,
        }

    context = "\n\n".join(
        [
            f"[{item['published']}] {item['title']}\n"
            f"Summary: {item['summary']}\n"
            f"Source: {item['source']}\n"
            f"Matched keywords: {', '.join(item['matched_keywords'])}"
            for item in news_items[:5]
        ]
    )
    evidence_level = _evidence_level(news_items)
    logger.info(
        "News scan completed | ticker=%s | status=%s | raw_news_items=%s | matched_news_items=%s | news_errors=%s | evidence_level=%s",
        ticker,
        status,
        len(raw_news_items),
        len(news_items),
        len(news_errors),
        evidence_level,
    )
    return {
        "news_context": context,
        "news_found": True,
        "news_items_count": len(news_items),
        "news_items": news_items[:5],
        "evidence_level": evidence_level,
        "shock_type": "EVENT_DRIVEN",
        "news_errors": news_errors,
        "news_sources": news_sources,
        "raw_news_items_count": len(raw_news_items),
        "matched_news_items_count": len(news_items),
        "news_errors_count": len(news_errors),
        "news_status": status,
        "news_debug_path": str(debug_path),
        "news_keywords": keywords,
        "google_news_queries": google_queries,
        "google_news_used": True,
    }


def _fetch_google_news_rss_sources(
    urls: List[str],
    news_sources: List[Dict[str, Any]],
    raw_news_items: List[Dict[str, str]],
    news_errors: List[str],
    queries: List[str] | None = None,
    max_items_per_query: int | None = None,
) -> None:
    queries = queries or []
    for index, url in enumerate(urls):
        query = queries[index] if index < len(queries) else None
        logger.info("Google News RSS fetch started | url=%s", url)
        try:
            feed = feedparser.parse(url)
            entries = list(getattr(feed, "entries", []) or [])
            http_status = feed.get("status")
            content_type = (feed.get("headers") or {}).get("content-type") if feed.get("headers") else None
            parse_error = getattr(feed, "bozo_exception", None) if getattr(feed, "bozo", False) else None

            source_record = {
                "source": "Google News RSS",
                "url": url,
                "query": query,
                "raw_news_items_count": len(entries),
                "http_status": http_status,
                "content_type": content_type,
            }

            if parse_error:
                error_text = str(parse_error)
                error_type = type(parse_error).__name__
                news_errors.append(f"Google News RSS | {url}: {error_type}: {error_text}")
                source_record.update(
                    {
                        "status": "ERROR",
                        "error": error_text,
                        "error_type": error_type,
                    }
                )
                logger.warning(
                    "Google News RSS fetch failed | url=%s | error_type=%s | error=%s | http_status=%s | content_type=%s",
                    url,
                    error_type,
                    error_text,
                    http_status,
                    content_type,
                )
            else:
                source_record.update({"status": "OK", "error": None})
                logger.info(
                    "Google News RSS fetch completed | url=%s | status=OK | raw_news_items=%s | http_status=%s | content_type=%s",
                    url,
                    len(entries),
                    http_status,
                    content_type,
                )

            news_sources.append(source_record)
            for entry in entries[: max_items_per_query or len(entries)]:
                raw_news_items.append(_news_entry_to_debug_item(entry, url, query))
        except Exception as exc:
            error_text = str(exc)
            error_type = type(exc).__name__
            news_errors.append(f"Google News RSS | {url}: {error_type}: {error_text}")
            news_sources.append(
                {
                    "source": "Google News RSS",
                    "url": url,
                    "query": query,
                    "status": "ERROR",
                    "raw_news_items_count": 0,
                    "error": error_text,
                    "error_type": error_type,
                    "http_status": None,
                    "content_type": None,
                }
            )
            logger.warning(
                "Google News RSS fetch failed | url=%s | error_type=%s | error=%s",
                url,
                error_type,
                error_text,
            )


def _filter_news_items(
    ticker: str,
    raw_news_items: List[Dict[str, str]],
    keywords: List[str],
    max_matched_items: int = DEFAULT_MAX_MATCHED_NEWS_ITEMS,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    news_items: List[Dict[str, Any]] = []
    matched_news_items: List[Dict[str, Any]] = []
    seen_keys = set()
    for item in raw_news_items:
        matched, matched_keywords = _match_news_item(item, keywords)
        logger.debug(
            "News match check | ticker=%s | matched=%s | matched_keywords=%s | title=%s",
            ticker,
            matched,
            matched_keywords,
            item["title"],
        )
        if not matched:
            continue
        dedupe_key = _news_dedupe_key(item)
        if dedupe_key in seen_keys:
            continue
        seen_keys.add(dedupe_key)
        news_items.append(
            {
                "published": item["published"],
                "title": item["title"],
                "summary": item["summary"][:350],
                "source": item.get("source", "Google News RSS"),
                "publisher": item.get("publisher", ""),
                "source_domain": item.get("source_domain", ""),
                "link": item["link"],
                "matched_keywords": matched_keywords,
            }
        )
        matched_news_items.append(
            {
                "title": item["title"],
                "published": item["published"],
                "link": item["link"],
                "source": item.get("source", "Google News RSS"),
                "publisher": item.get("publisher", ""),
                "source_domain": item.get("source_domain", ""),
                "matched_keywords": matched_keywords,
            }
        )
        if len(news_items) >= max_matched_items:
            break
    return news_items, matched_news_items


def _news_keywords_for_ticker(ticker: str) -> List[str]:
    keyword_map = {
        "VIC": [
            "VIC",
            "Vingroup",
            "VinGroup",
            "T\u1eadp \u0111o\u00e0n Vingroup",
            "Tap doan Vingroup",
            "h\u1ecd Vingroup",
            "ho Vingroup",
            "VinFast",
            "VFS",
            "Vinhomes",
            "VHM",
            "Vincom Retail",
            "VRE",
            "VPL",
            "c\u1ed5 phi\u1ebfu VIC",
            "co phieu VIC",
        ],
        "VHM": ["VHM", "Vinhomes", "Vingroup", "c\u1ed5 phi\u1ebfu VHM", "co phieu VHM"],
        "VRE": ["VRE", "Vincom Retail", "Vingroup", "c\u1ed5 phi\u1ebfu VRE", "co phieu VRE"],
        "VPL": ["VPL", "Vinpearl", "Vingroup", "c\u1ed5 phi\u1ebfu VPL", "co phieu VPL"],
    }
    keywords = keyword_map.get(ticker, [ticker, "Vingroup"])
    return _dedupe_preserve_order(keywords)


def _google_news_queries_for_ticker(ticker: str) -> List[str]:
    query_map = {
        "VIC": [
            '"VIC" "Vingroup"',
            '"c\u1ed5 phi\u1ebfu VIC"',
            '"Vingroup" "ch\u1ee9ng kho\u00e1n"',
            '"VinFast" "Vingroup"',
        ],
        "VHM": ['"VHM" "Vinhomes"', '"Vinhomes" "c\u1ed5 phi\u1ebfu"'],
        "VRE": ['"VRE" "Vincom Retail"', '"Vincom Retail" "c\u1ed5 phi\u1ebfu"'],
        "VPL": ['"VPL" "Vinpearl"', '"Vinpearl" "Vingroup"'],
    }
    return query_map.get(ticker, [f'"{ticker}" "Vingroup"', f'"{ticker}" "c\u1ed5 phi\u1ebfu"'])


def _google_news_url(query: str) -> str:
    return GOOGLE_NEWS_RSS_SEARCH_BASE.format(query=quote_plus(query))


def _news_entry_to_debug_item(entry: Any, url: str, query: str | None = None) -> Dict[str, str]:
    publisher = ""
    source = entry.get("source")
    if isinstance(source, dict):
        publisher = clean_html(source.get("title", ""))
    source_domain = _source_domain(entry.get("link", ""), publisher)
    return {
        "title": clean_html(entry.get("title", "")),
        "published": entry.get("published", entry.get("updated", "UNKNOWN_DATE")),
        "link": entry.get("link", ""),
        "summary": clean_html(entry.get("summary", ""))[:500],
        "source": "Google News RSS",
        "publisher": publisher,
        "source_domain": source_domain,
        "source_url": url,
        "query": query or "",
    }


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


def _normalize_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", str(value))
    ascii_text = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    return ascii_text.lower()


def _news_scan_status(matched_items: int, error_count: int) -> str:
    if matched_items > 0:
        return "NEWS_FOUND_GOOGLE_NEWS"
    if error_count > 0:
        return "GOOGLE_NEWS_ERROR"
    return "NO_NEWS"


def _write_news_debug_artifact(
    ticker: str,
    run_id: str | None,
    news_sources: List[Dict[str, Any]],
    keywords: List[str],
    google_news_queries: List[str],
    raw_news_items: List[Dict[str, str]],
    matched_news_items: List[Dict[str, Any]],
    status: str,
    news_errors_count: int,
) -> Path:
    NEWS_DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    run_date = datetime.now().strftime("%Y-%m-%d")
    debug_id = run_id or datetime.now().strftime("%H%M%S")
    debug_path = NEWS_DEBUG_DIR / f"{ticker}_news_debug_{run_date}_{debug_id}.json"
    payload = {
        "ticker": ticker,
        "run_date": run_date,
        "run_id": run_id,
        "news_sources": news_sources,
        "google_news_queries": google_news_queries,
        "keywords": keywords,
        "raw_news_items": raw_news_items,
        "matched_news_items": matched_news_items,
        "summary": {
            "raw_news_items_count": len(raw_news_items),
            "matched_news_items_count": len(matched_news_items),
            "news_errors_count": news_errors_count,
            "status": status,
            "google_news_used": True,
        },
    }
    with debug_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    logger.info("News debug artifact saved | path=%s", debug_path)
    return debug_path


def _dedupe_raw_news_items(raw_news_items: List[Dict[str, str]]) -> List[Dict[str, str]]:
    deduped = []
    seen = set()
    for item in raw_news_items:
        key = _news_dedupe_key(item)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def _filter_by_lookback(raw_news_items: List[Dict[str, str]], lookback_days: int) -> List[Dict[str, str]]:
    cutoff = datetime.now().astimezone() - timedelta(days=lookback_days)
    filtered = []
    for item in raw_news_items:
        published_dt = _parse_published_datetime(item.get("published", ""))
        if published_dt is None or published_dt >= cutoff:
            filtered.append(item)
    return filtered


def _parse_published_datetime(value: str) -> datetime | None:
    try:
        parsed = parsedate_to_datetime(str(value))
    except (TypeError, ValueError, IndexError, OverflowError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.astimezone()
    return parsed


def _news_dedupe_key(item: Dict[str, str]) -> str:
    title = _normalize_title_for_dedupe(item.get("title", ""))
    link = str(item.get("link", "")).strip().lower()
    return f"{title}|{link}"


def _normalize_title_for_dedupe(title: str) -> str:
    title = clean_html(title)
    title = re.sub(r"\s+-\s+[^-]{2,80}$", "", title)
    title = re.sub(r"\s+", " ", title).strip()
    return _normalize_text(title)


def _source_domain(link: str, publisher: str = "") -> str:
    parsed = urlparse(str(link or ""))
    domain = parsed.netloc.lower().replace("www.", "")
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
    domains = {
        str(item.get("source_domain") or item.get("publisher") or item.get("source") or "").lower()
        for item in news_items
    }
    domains.discard("")
    return "HIGH" if len(domains) >= 2 else "MEDIUM"


def _dedupe_preserve_order(values: List[str]) -> List[str]:
    seen = set()
    deduped = []
    for value in values:
        if value not in seen:
            seen.add(value)
            deduped.append(value)
    return deduped
