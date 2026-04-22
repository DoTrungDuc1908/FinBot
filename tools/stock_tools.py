"""
tools/stock_tools.py
LangChain tools for stock data retrieval using vnstock + TCBS.
Covers: company info, price history, and timeframe-based filtering.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import requests
from langchain_core.tools import tool
from loguru import logger

from config.settings import settings
from core.cache import cached, cache, CacheClient

# ── Period aliases ────────────────────────────────────────────────────────────
PERIOD_ALIASES: dict[str, int] = {
    "1w": 7,
    "1m": 30,
    "3m": 90,
    "6m": 180,
    "1y": 365,
    "ytd": 0,  # handled separately
}


def _resolve_dates(period: str | None, start: str | None, end: str | None):
    """Resolve start/end dates from period alias or explicit strings."""
    today = datetime.now()
    end_dt = datetime.strptime(end, "%Y-%m-%d") if end else today
    if period:
        period = period.lower().strip()
        if period == "ytd":
            start_dt = datetime(today.year, 1, 1)
        else:
            days = PERIOD_ALIASES.get(period, 90)
            start_dt = today - timedelta(days=days)
    elif start:
        start_dt = datetime.strptime(start, "%Y-%m-%d")
    else:
        start_dt = today - timedelta(days=90)
    return start_dt.strftime("%Y-%m-%d"), end_dt.strftime("%Y-%m-%d")


# ── vnstock wrapper ───────────────────────────────────────────────────────────

def _fetch_price_vnstock(ticker: str, start: str, end: str) -> list[dict]:
    """Fetch OHLCV from vnstock library."""
    try:
        import pandas as pd
        from vnstock import Quote  # type: ignore
        
        # vnstock v3 sử dụng class Quote thay cho hàm stock_historical_data
        quote = Quote(symbol=ticker.upper())
        df = quote.history(
            start=start,
            end=end,
            interval="1D"
        )
        
        if df is None or df.empty:
            return []
            
        # vnstock v3 mặc định trả về các cột: time, open, high, low, close, volume
        # Ta chỉ cần đổi tên cột 'time' thành 'date'
        df = df.rename(columns={"time": "date"})
        
        # Format định dạng ngày tháng
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
        
        return df[["date", "open", "high", "low", "close", "volume"]].to_dict(orient="records")
    except Exception as e:
        logger.error(f"vnstock fetch error for {ticker}: {e}")
        return []


def _fetch_company_tcbs(ticker: str) -> dict:
    """Fetch company overview from TCBS."""
    url = f"{settings.tcbs_base_url}/stock-insight/v2/stock/ticker/{ticker.upper()}/company"
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        return {
            "ticker": ticker.upper(),
            "company_name": data.get("companyName", ""),
            "industry": data.get("industry", ""),
            "exchange": data.get("exchange", ""),
            "market_cap": data.get("marketCap", 0),
            "website": data.get("website", ""),
            "description": data.get("businessStrategy", "")[:500],
        }
    except Exception as e:
        logger.warning(f"TCBS company fetch failed for {ticker}: {e}")
        return {}


# ── LangChain Tools ───────────────────────────────────────────────────────────

@tool
def get_company_info(ticker: str) -> str:
    """
    Lấy thông tin doanh nghiệp theo mã chứng khoán.
    Trả về: tên công ty, ngành, sàn, vốn hóa, website, mô tả.

    Args:
        ticker: Mã chứng khoán Việt Nam (VD: VNM, VIC, HPG).
    """
    key = CacheClient.build_key("company", ticker.upper())
    cached_val = cache.get(key)
    if cached_val:
        logger.debug(f"Cache HIT: company info {ticker}")
        return json.dumps(cached_val, ensure_ascii=False)

    info = _fetch_company_tcbs(ticker)
    if not info:
        # Fallback: return minimal structure
        info = {"ticker": ticker.upper(), "error": "Không tìm thấy dữ liệu doanh nghiệp"}

    cache.set(key, info, ttl=settings.cache_ttl_company)
    return json.dumps(info, ensure_ascii=False, indent=2)


@tool
def get_price_history(
    ticker: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    period: Optional[str] = None,
) -> str:
    """
    Lấy dữ liệu giá lịch sử (OHLCV) của một mã chứng khoán.

    Args:
        ticker: Mã chứng khoán (VD: VNM, VIC).
        start_date: Ngày bắt đầu định dạng YYYY-MM-DD (tùy chọn).
        end_date: Ngày kết thúc định dạng YYYY-MM-DD (tùy chọn).
        period: Khung thời gian: '1w', '1m', '3m', '6m', '1y', 'ytd'.
                Nếu có period, start_date/end_date sẽ bị bỏ qua.

    Returns:
        JSON danh sách ngày giao dịch với open, high, low, close, volume.
    """
    start, end = _resolve_dates(period, start_date, end_date)
    key = CacheClient.build_key("price", ticker.upper(), start, end)
    cached_val = cache.get(key)
    if cached_val:
        logger.debug(f"Cache HIT: price {ticker} {start}-{end}")
        return json.dumps(cached_val, ensure_ascii=False)

    records = _fetch_price_vnstock(ticker, start, end)
    result = {
        "ticker": ticker.upper(),
        "start": start,
        "end": end,
        "count": len(records),
        "data": records,
    }
    ttl = settings.cache_ttl_price if not period or period in ("1w",) else 3600
    cache.set(key, result, ttl=ttl)
    return json.dumps(result, ensure_ascii=False, indent=2)


@tool
def get_market_overview() -> str:
    """
    Lấy tổng quan thị trường chứng khoán Việt Nam:
    VN-Index, HNX-Index, UPCOM-Index (điểm số và % thay đổi hôm nay).
    """
    key = CacheClient.build_key("market", "overview")
    cached_val = cache.get(key)
    if cached_val:
        return json.dumps(cached_val, ensure_ascii=False)

    url = f"{settings.tcbs_base_url}/stock-insight/v2/market/index"
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        raw = resp.json()
        indices = [
            {
                "index": item.get("indexId", ""),
                "value": item.get("indexValue", 0),
                "change": item.get("indexChange", 0),
                "pct_change": item.get("percentChange", 0),
                "volume": item.get("totalVolume", 0),
            }
            for item in raw.get("data", [])
        ]
        result = {"indices": indices, "as_of": datetime.now().strftime("%Y-%m-%d %H:%M")}
        cache.set(key, result, ttl=120)
        return json.dumps(result, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"Market overview fetch error: {e}")
        return json.dumps({"error": str(e)}, ensure_ascii=False)
