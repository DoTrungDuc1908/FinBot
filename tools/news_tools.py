"""
tools/news_tools.py
News crawling and Vietnamese sentiment analysis tools.
Uses RSS feeds + BeautifulSoup for news collection.
Sentiment: Tích hợp LLM (Fast Model) với Structured Output để đọc hiểu ngữ cảnh tài chính.
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timedelta
from typing import List

import feedparser  # type: ignore
import requests
from bs4 import BeautifulSoup
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from config.settings import settings
from core.cache import cache, CacheClient
from core.llm import get_fast_llm  # Sử dụng Fast LLM để chấm điểm nhanh


# ── 1. Định nghĩa Pydantic Schema cho LLM (THÊM MỚI) ──────────────────────────

class ArticleSentiment(BaseModel):
    article_id: int = Field(description="ID của bài báo trong danh sách cung cấp")
    sentiment: str = Field(description="Chỉ được chọn 1 trong 3: 'positive', 'negative', 'neutral'")
    reasoning: str = Field(description="Giải thích cực kỳ ngắn gọn (1 câu) tại sao chọn sentiment này dựa trên ngữ cảnh tài chính")

class BatchSentimentResponse(BaseModel):
    results: List[ArticleSentiment] = Field(description="Danh sách kết quả phân tích sentiment cho toàn bộ bài báo")


# ── 2. Các hàm Cào dữ liệu (Giữ nguyên logic) ─────────────────────────────────

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def _fetch_rss_content(url: str) -> bytes:
    logger.debug(f"Đang tải RSS từ: {url}")
    timeout_val = getattr(settings, "request_timeout", 10)
    response = requests.get(url, timeout=timeout_val)
    response.raise_for_status()
    return response.content

def _parse_rss_feed(url: str, limit: int = 15) -> List[dict]:
    try:
        raw_content = _fetch_rss_content(url)
        feed = feedparser.parse(raw_content)
        
        articles = []
        for entry in feed.entries[:limit]:
            summary = entry.get("summary", "")
            if summary:
                soup = BeautifulSoup(summary, "html.parser")
                summary = soup.get_text(separator=" ", strip=True)
            
            articles.append({
                "title": entry.get("title", ""),
                "link": entry.get("link", ""),
                "summary": summary,
                "published": entry.get("published", "")
            })
        return articles
    except Exception as e:
        logger.exception(f"LỖI NGHIÊM TRỌNG khi parse RSS từ {url}:")
        return []

def _search_news_for_ticker(ticker: str, limit: int = 5) -> List[dict]:
    try:
        all_articles = []
        all_articles.extend(_parse_rss_feed(settings.cafef_rss, limit=20))
        all_articles.extend(_parse_rss_feed(settings.vneconomy_rss, limit=20))
        
        ticker_lower = ticker.lower()
        filtered = []
        for a in all_articles:
            if ticker_lower in a["title"].lower() or ticker_lower in a["summary"].lower():
                filtered.append(a)
                
        return filtered[:limit]
    except Exception as e:
        logger.exception(f"LỖI NGHIÊM TRỌNG khi lọc tin tức cho mã {ticker}:")
        return []


# ── 3. Phân tích Sentiment bằng LLM (THAY THẾ HOÀN TOÀN TỪ KHÓA) ──────────────

def _aggregate_sentiment_with_llm(articles: List[dict]) -> dict:
    """Sử dụng LLM để đọc và chấm điểm sentiment bằng Manual JSON Parsing."""
    if not articles:
        return {"label": "neutral", "score": 0.5, "positive": 0, "negative": 0, "neutral": 0}

    articles_text = ""
    for idx, a in enumerate(articles):
        articles_text += f"[ID: {idx}] Tiêu đề: {a.get('title', '')}\nTóm tắt: {a.get('summary', '')[:300]}\n---\n"

    system_prompt = f"""Bạn là chuyên gia phân tích tâm lý thị trường chứng khoán Việt Nam.
Nhiệm vụ: Đọc danh sách các bài báo dưới đây và đánh giá tác động (sentiment) của chúng đến cổ phiếu/thị trường.
Quy tắc ngữ cảnh: 
- "Giảm lỗ", "thoát lỗ", "kế hoạch vượt khó" -> 'positive'
- "Cảnh báo", "kiểm soát", "hủy niêm yết" -> 'negative'

BẮT BUỘC TRẢ VỀ ĐỊNH DẠNG JSON. KHÔNG VIẾT THÊM BẤT KỲ CHỮ NÀO KHÁC TRƯỚC HAY SAU JSON.
Cấu trúc JSON bắt buộc:
{{
  "results": [
    {{
      "article_id": 0,
      "sentiment": "positive",
      "reasoning": "Lý do cực kỳ ngắn gọn..."
    }}
  ]
}}

Danh sách bài báo:
{articles_text}"""

    try:
        llm = get_fast_llm(model_name=settings.nvidia_eval_model)
        raw_output = llm.invoke([HumanMessage(content=system_prompt)]).content.strip()
        
        import re
        import json
        
        sentiment_map = {}
        
        # 1. Thử bóc tách JSON bằng Regex
        match = re.search(r'\{.*\}', raw_output, re.DOTALL)
        
        if match:
            json_str = match.group(0)
            parsed_json = json.loads(json_str)
            for item in parsed_json.get("results", []):
                if "article_id" in item:
                    sentiment_map[item["article_id"]] = item
        else:
            # 2. LỚP PHÒNG VỆ MỚI: Xử lý khi LLM cứng đầu trả về Text
            logger.warning("Không tìm thấy JSON, kích hoạt đọc hiểu Text thuần từ LLM.")
            text_lower = raw_output.lower()
            
            guessed_label = "neutral"
            if "positive" in text_lower or "tích cực" in text_lower:
                guessed_label = "positive"
            elif "negative" in text_lower or "tiêu cực" in text_lower:
                guessed_label = "negative"
            
            cleaned_reason = raw_output.replace("*", "").strip()[:200] + "..."
            for idx in range(len(articles)):
                sentiment_map[idx] = {
                    "article_id": idx,
                    "sentiment": guessed_label,
                    "reasoning": cleaned_reason
                }
                
    except Exception as e:
        logger.error(f"Lỗi phân giải JSON Sentiment: {e}. Raw Output: {raw_output}")
        sentiment_map = {}

    # --- PHẦN TÍNH ĐIỂM VÀ RETURN ---
    counts = {"positive": 0, "negative": 0, "neutral": 0}
    scores = []

    for idx, a in enumerate(articles):
        if idx in sentiment_map:
            res = sentiment_map[idx]
            label = res.get("sentiment", "neutral").lower()
            if label not in ["positive", "negative", "neutral"]: label = "neutral"
            
            a["sentiment"] = label
            a["reasoning"] = res.get("reasoning", "Dựa trên ngữ cảnh chung.")
        else:
            a["sentiment"] = "neutral"
            a["reasoning"] = "Dữ liệu chưa được phân tích."
            label = "neutral"
            
        counts[label] += 1
        scores.append(1.0 if label == "positive" else (0.0 if label == "negative" else 0.5))

    avg_score = sum(scores) / len(scores) if scores else 0.5
    overall = "positive" if avg_score > 0.55 else ("negative" if avg_score < 0.45 else "neutral")
    
    return {
        "label": overall,
        "score": round(avg_score, 3),
        **counts,
    }


# ── 4. LangChain Tools ────────────────────────────────────────────────────────

@tool
def fetch_stock_news(ticker: str, limit: int = 10) -> str:
    """Lấy tin tức mới nhất liên quan đến một mã chứng khoán (chỉ lấy tin, không phân tích)."""
    try:
        key = CacheClient.build_key("news", ticker.upper(), str(limit))
        if (hit := cache.get(key)): return json.dumps(hit, ensure_ascii=False)

        articles = _search_news_for_ticker(ticker, limit=limit)
        result = {
            "ticker": ticker.upper(),
            "count": len(articles),
            "articles": articles,
            "fetched_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        }
        cache.set(key, result, ttl=settings.cache_ttl_news)
        return json.dumps(result, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.exception(f"Lỗi Tool fetch_stock_news cho {ticker}:")
        return json.dumps({"error": f"Lỗi trích xuất tin tức: {str(e)}"}, ensure_ascii=False)

@tool
def analyze_stock_sentiment(ticker: str, start_date: str = None, end_date: str = None) -> str:
    """Phân tích sentiment tin tức bằng LLM, có hỗ trợ lọc theo ngày. Trả về JSON chứa Markdown và Dữ liệu thô."""
    import json # Đảm bảo import json
    try:
        articles = _search_news_for_ticker(ticker, limit=50)
        
        if start_date and end_date:
            import pandas as pd
            start_dt = pd.to_datetime(start_date, utc=True).tz_localize(None)
            end_dt = pd.to_datetime(end_date, utc=True).tz_localize(None)
            filtered_articles = []
            for a in articles:
                try:
                    pub_dt = pd.to_datetime(a["published"], utc=True).tz_localize(None)
                    if start_dt <= pub_dt <= end_dt: filtered_articles.append(a)
                except:
                    filtered_articles.append(a)
            articles = filtered_articles[:15]
        else:
            articles = articles[:15]

        # Gọi hàm aggregate bằng LLM
        agg = _aggregate_sentiment_with_llm(articles)

        # 1. Build Markdown báo cáo cho LLM (Advisor Agent) đọc
        md = f"### 📰 Báo cáo Tin tức & Tâm lý: **{ticker.upper()}**\n"
        if start_date: md += f"*(Giai đoạn: {start_date} đến {end_date})*\n\n"
        
        sentiment_icon = "🟢 TÍCH CỰC" if agg['label'] == 'positive' else "🔴 TIÊU CỰC" if agg['label'] == 'negative' else "⚪ TRUNG LẬP"
        md += f"**Trạng thái chung:** {sentiment_icon} (Điểm: {agg['score']:.2f}/1.0)\n"
        md += f"**Thống kê:** {len(articles)} bài báo ({agg['positive']} Tích cực, {agg['negative']} Tiêu cực, {agg['neutral']} Trung lập)\n\n"
        md += "---\n\n"
        
        if not articles:
            md += "*Không tìm thấy tin tức nào trong giai đoạn này.*\n"
        
        for a in articles[:5]:
            icon = "🟢" if a.get('sentiment') == 'positive' else "🔴" if a.get('sentiment') == 'negative' else "⚪"
            md += f"#### {icon} [{a['title']}]({a['link']})\n"
            md += f"> {a.get('summary', '')}\n\n"
            md += f"**📅 Ngày:** {a['published'][:16]} | **Tác động:** {a.get('sentiment', '').upper()}\n"
            if "reasoning" in a:
                md += f"**🧠 AI Đánh giá:** *{a['reasoning']}*\n\n"
            else:
                md += "\n"

        # 2. THAY ĐỔI QUAN TRỌNG: Đóng gói JSON trả về
        result_payload = {
            "markdown_report": md,
            "raw_data": {
                "ticker": ticker.upper(),
                "overall_sentiment": agg['label'],
                "score": agg['score'],
                "stats": {
                    "pos": agg['positive'], 
                    "neg": agg['negative'], 
                    "neu": agg['neutral']
                },
                "articles": articles[:10] # Chỉ lấy tối đa 10 bài báo trả về cho Frontend hiển thị UI
            }
        }
        
        return json.dumps(result_payload, ensure_ascii=False)
        
    except Exception as e:
        logger.exception(f"Lỗi Tool analyze_stock_sentiment cho {ticker}:")
        # Đảm bảo format lỗi cũng tuân thủ cấu trúc JSON
        error_payload = {
            "markdown_report": f"⚠️ Lỗi phân tích tâm lý: {str(e)}",
            "raw_data": None
        }
        return json.dumps(error_payload, ensure_ascii=False)

@tool
def get_market_news(limit: int = 15) -> str:
    """Lấy tin tức thị trường chung và đánh giá sentiment bằng LLM."""
    try:
        key = CacheClient.build_key("news", "market", "llm_eval", str(limit))
        if (hit := cache.get(key)): return json.dumps(hit, ensure_ascii=False)

        articles: List[dict] = []
        articles.extend(_parse_rss_feed(settings.cafef_rss, limit=limit))
        articles.extend(_parse_rss_feed(settings.vneconomy_rss, limit=limit))
        articles = articles[:limit]
        
        # Đánh giá bằng LLM
        agg = _aggregate_sentiment_with_llm(articles)

        result = {
            "market_sentiment": agg["label"],
            "sentiment_score": agg["score"],
            "count": len(articles),
            "articles": articles,
        }
        cache.set(key, result, ttl=settings.cache_ttl_news)
        return json.dumps(result, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.exception("Lỗi Tool get_market_news:")
        return json.dumps({"error": f"Lỗi lấy tin thị trường: {str(e)}"}, ensure_ascii=False)