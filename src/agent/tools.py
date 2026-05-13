from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List

import feedparser
import yaml

from utils.logger import get_logger

logger = get_logger("AgentTools")


def clean_html(raw_html: str) -> str:
    cleanr = re.compile("<.*?>")
    return re.sub(cleanr, "", str(raw_html)).strip()


def tool_search_vietstock_news(ticker: str) -> Dict[str, Any]:
    logger.info("News scan started | ticker=%s | source=Vietstock RSS", ticker)
    rss_urls = [
        "https://vietstock.vn/rss/doanh-nghiep.rss",
        "https://vietstock.vn/rss/chung-khoan.rss",
        "https://vietstock.vn/rss/vi-mo.rss",
    ]
    keywords = [
        ticker,
        "VN-Index",
        "lai suat",
        "lãi suất",
        "khoi ngoai",
        "khối ngoại",
        "ngan hang nha nuoc",
        "ngân hàng nhà nước",
        "Fed",
        "Vingroup",
    ]

    all_entries: List[Any] = []
    errors: List[str] = []
    for url in rss_urls:
        try:
            feed = feedparser.parse(url)
            if getattr(feed, "bozo", False):
                errors.append(f"{url}: {getattr(feed, 'bozo_exception', 'parse error')}")
            all_entries.extend(feed.entries)
        except Exception as exc:
            errors.append(f"{url}: {exc}")

    news_items: List[Dict[str, str]] = []
    for entry in all_entries:
        title = clean_html(entry.get("title", ""))
        summary = clean_html(entry.get("summary", ""))
        content = f"{title} {summary}".lower()
        if any(keyword.lower() in content for keyword in keywords):
            news_items.append(
                {
                    "published": entry.get("published", "UNKNOWN_DATE"),
                    "title": title,
                    "summary": summary[:350],
                    "source": "Vietstock RSS",
                    "link": entry.get("link", ""),
                }
            )

    if not news_items:
        context = "NO_NEWS"
        logger.info(
            "News scan completed | ticker=%s | status=NO_NEWS | matched_items=0 | rss_errors=%s",
            ticker,
            len(errors),
        )
        return {
            "news_context": context,
            "news_found": False,
            "news_items_count": 0,
            "news_items": [],
            "evidence_level": "NONE",
            "shock_type": "NO_NEWS",
            "rss_errors": errors,
        }

    context = "\n\n".join(
        [
            f"[{item['published']}] {item['title']}\nSummary: {item['summary']}\nSource: {item['source']}"
            for item in news_items[:5]
        ]
    )
    evidence_level = "HIGH" if len(news_items) >= 4 else "MEDIUM" if len(news_items) >= 2 else "LOW"
    logger.info(
        "News scan completed | ticker=%s | status=NEWS_FOUND | matched_items=%s | evidence_level=%s",
        ticker,
        len(news_items),
        evidence_level,
    )
    return {
        "news_context": context,
        "news_found": True,
        "news_items_count": len(news_items),
        "news_items": news_items[:5],
        "evidence_level": evidence_level,
        "shock_type": "EVENT_DRIVEN",
        "rss_errors": errors,
    }


def tool_adjust_model_hyperparams(adjustment_type: str) -> str:
    logger.info("Model config adjustment requested | strategy=%s", adjustment_type)

    config_path = Path("configs/model_config.yaml")
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    params = config["lightgbm_params"]
    lr = params.get("learning_rate", 0.05)
    md = params.get("max_depth", 4)
    ne = params.get("n_estimators", 100)

    if adjustment_type == "FIX_OVERFITTING":
        params["max_depth"] = max(md - 1, 2)
        params["learning_rate"] = max(lr - 0.01, 0.01)
        params["n_estimators"] = max(ne - 20, 30)
        params["num_leaves"] = min(2**params["max_depth"] - 1, 31)
        msg = (
            "Adjusted config for overfitting risk: "
            f"max_depth={params['max_depth']}, learning_rate={params['learning_rate']:.3f}, "
            f"n_estimators={params['n_estimators']}"
        )
    elif adjustment_type == "FIX_UNDERFITTING":
        params["max_depth"] = min(md + 1, 8)
        params["learning_rate"] = min(lr + 0.02, 0.2)
        params["n_estimators"] = min(ne + 30, 300)
        params["num_leaves"] = min(2**params["max_depth"] - 1, 63)
        msg = (
            "Adjusted config for underfitting risk: "
            f"max_depth={params['max_depth']}, learning_rate={params['learning_rate']:.3f}, "
            f"n_estimators={params['n_estimators']}"
        )
    elif adjustment_type == "ADAPT_SHOCK":
        params["learning_rate"] = min(lr + 0.05, 0.3)
        params["n_estimators"] = max(ne - 30, 20)
        msg = (
            "Adjusted config for event-driven regime: "
            f"learning_rate={params['learning_rate']:.3f}, n_estimators={params['n_estimators']}"
        )
    else:
        msg = f"No config adjustment applied for unknown strategy: {adjustment_type}"

    with config_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)

    logger.info("Model config adjustment completed | strategy=%s", adjustment_type)
    return msg
