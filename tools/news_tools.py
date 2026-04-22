"""
tools/news_tools.py
News crawling and Vietnamese sentiment analysis tools.
Uses RSS feeds + BeautifulSoup for news collection.
Sentiment: keyword-based fast path, PhoBERT heavy path.
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timedelta
from typing import List

import feedparser  # type: ignore
import requests
from bs4 import BeautifulSoup
from langchain_core.tools import tool
from loguru import logger

from config.settings import settings
from core.cache import cache, CacheClient

# ── Vietnamese sentiment keywords (fast path) ─────────────────────────────────
_POS_KW = [
    "tăng", "lợi nhuận", "tích cực", "khởi sắc", "vượt", "bứt phá",
    "tăng trưởng", "hiệu quả", "lạc quan", "mua vào", "khuyến nghị mua",
    "dòng tiền", "kỳ vọng", "hồi phục", "phục hồi",
]
_NEG_KW = [
    "giảm", "lỗ", "sụt", "bán tháo", "rủi ro", "tiêu cực", "vi phạm",
    "thanh tra", "kiểm tra", "nợ xấu", "trì trệ", "giảm điểm", "áp lực",
    "cảnh báo", "thoái vốn", "đình chỉ",
]


def _keyword_sentiment(text: str) -> tuple[str, float]:
    """Fast keyword-based sentiment. Returns (label, score 0-1)."""
    text_lower = text.lower()
    pos = sum(1 for kw in _POS_KW if kw in text_lower)
    neg = sum(1 for kw in _NEG_KW if kw in text_lower)
    total = pos + neg
    if total == 0:
        return "neutral", 0.5
    score = pos / total
    if score > 0.6:
        return "positive", round(score, 3)
    elif score < 0.4:
        return "negative", round(1 - score, 3)
    return "neutral", 0.5


def _parse_rss_feed(url: str, limit: int = 20) -> List[dict]:
    """Parse an RSS feed and return articles."""
    try:
        feed = feedparser.parse(url)
        articles = []
        for entry in feed.entries[:limit]:
            articles.append({
                "title": entry.get("title", ""),
                "summary": entry.get("summary", "")[:300],
                "link": entry.get("link", ""),
                "published": entry.get("published", ""),
            })
        return articles
    except Exception as e:
        logger.warning(f"RSS parse error {url}: {e}")
        return []


def _search_news_for_ticker(ticker: str, limit: int = 10) -> List[dict]:
    """Search CafeF for ticker-specific news."""
    articles = []
    # CafeF RSS
    cafef_articles = _parse_rss_feed(settings.cafef_rss, limit=50)
    ticker_upper = ticker.upper()
    for a in cafef_articles:
        combined = f"{a['title']} {a['summary']}"
        if ticker_upper in combined.upper():
            articles.append(a)
            if len(articles) >= limit:
                break

    # VnEconomy RSS fallback
    if len(articles) < 3:
        vne_articles = _parse_rss_feed(settings.vneconomy_rss, limit=50)
        for a in vne_articles:
            combined = f"{a['title']} {a['summary']}"
            if ticker_upper in combined.upper():
                articles.append(a)
                if len(articles) >= limit:
                    break

    return articles[:limit]


def _aggregate_sentiment(articles: List[dict]) -> dict:
    """Aggregate sentiment scores from a list of articles."""
    if not articles:
        return {"label": "neutral", "score": 0.5, "positive": 0, "negative": 0, "neutral": 0}

    counts = {"positive": 0, "negative": 0, "neutral": 0}
    scores = []
    for a in articles:
        text = f"{a.get('title', '')} {a.get('summary', '')}"
        label, score = _keyword_sentiment(text)
        a["sentiment"] = label
        a["sentiment_score"] = score
        counts[label] += 1
        scores.append(score if label == "positive" else (1 - score if label == "negative" else 0.5))

    avg_score = sum(scores) / len(scores)
    overall = "positive" if avg_score > 0.55 else ("negative" if avg_score < 0.45 else "neutral")
    return {
        "label": overall,
        "score": round(avg_score, 3),
        **counts,
    }


# ── LangChain Tools ───────────────────────────────────────────────────────────

@tool
def fetch_stock_news(ticker: str, limit: int = 10) -> str:
    """
    Lấy tin tức mới nhất liên quan đến một mã chứng khoán.

    Args:
        ticker: Mã chứng khoán (VD: VNM, VIC, HPG).
        limit: Số lượng bài báo tối đa (mặc định 10).

    Returns:
        JSON danh sách tin tức với tiêu đề, tóm tắt, link và thời gian.
    """
    key = CacheClient.build_key("news", ticker.upper(), str(limit))
    if (hit := cache.get(key)):
        return json.dumps(hit, ensure_ascii=False)

    articles = _search_news_for_ticker(ticker, limit=limit)
    result = {
        "ticker": ticker.upper(),
        "count": len(articles),
        "articles": articles,
        "fetched_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }
    cache.set(key, result, ttl=settings.cache_ttl_news)
    return json.dumps(result, ensure_ascii=False, indent=2)


@tool
def analyze_stock_sentiment(ticker: str) -> str:
    """
    Phân tích sentiment (tâm lý thị trường) của một mã chứng khoán
    dựa trên tin tức gần đây.

    Args:
        ticker: Mã chứng khoán (VD: VNM, HPG).

    Returns:
        JSON với nhãn sentiment (positive/negative/neutral), điểm số
        và phân bố bài báo theo từng nhãn.
    """
    key = CacheClient.build_key("sentiment", ticker.upper())
    if (hit := cache.get(key)):
        return json.dumps(hit, ensure_ascii=False)

    articles = _search_news_for_ticker(ticker, limit=20)
    agg = _aggregate_sentiment(articles)

    label_vi = {"positive": "Tích cực", "negative": "Tiêu cực", "neutral": "Trung lập"}
    result = {
        "ticker": ticker.upper(),
        "overall_sentiment": agg["label"],
        "overall_sentiment_vi": label_vi.get(agg["label"], ""),
        "confidence_score": agg["score"],
        "breakdown": {
            "positive": agg["positive"],
            "negative": agg["negative"],
            "neutral": agg["neutral"],
        },
        "articles_analyzed": len(articles),
        "top_articles": [
            {"title": a["title"], "sentiment": a.get("sentiment", ""), "link": a["link"]}
            for a in articles[:5]
        ],
    }
    cache.set(key, result, ttl=settings.cache_ttl_news)
    return json.dumps(result, ensure_ascii=False, indent=2)


@tool
def get_market_news(limit: int = 15) -> str:
    """
    Lấy tin tức thị trường chứng khoán Việt Nam tổng quan (không theo mã cụ thể).

    Args:
        limit: Số lượng bài báo (mặc định 15).

    Returns:
        JSON danh sách tin tức thị trường chung kèm sentiment tổng hợp.
    """
    key = CacheClient.build_key("news", "market", str(limit))
    if (hit := cache.get(key)):
        return json.dumps(hit, ensure_ascii=False)

    articles: List[dict] = []
    articles.extend(_parse_rss_feed(settings.cafef_rss, limit=limit))
    articles.extend(_parse_rss_feed(settings.vneconomy_rss, limit=limit))
    articles = articles[:limit]
    agg = _aggregate_sentiment(articles)

    result = {
        "market_sentiment": agg["label"],
        "sentiment_score": agg["score"],
        "count": len(articles),
        "articles": articles,
    }
    cache.set(key, result, ttl=settings.cache_ttl_news)
    return json.dumps(result, ensure_ascii=False, indent=2)
