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

PERIOD_ALIASES: dict[str, int] = {
    "1w": 7,
    "1m": 30,
    "3m": 90,
    "6m": 180,
    "1y": 365,
    "ytd": 0,
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



def _fetch_price_vnstock(ticker: str, start: str, end: str) -> list[dict]:
    """Fetch OHLCV from vnstock library."""
    try:
        import pandas as pd
        from vnstock import Quote  # type: ignore
        
        quote = Quote(symbol=ticker.upper())
        df = quote.history(
            start=start,
            end=end,
            interval="1D"
        )
        
        if df is None or df.empty:
            return []
            
        df = df.rename(columns={"time": "date"})
        
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
        
        return df[["date", "open", "high", "low", "close", "volume"]].to_dict(orient="records")
    except Exception as e:
        logger.exception(f"LỖI NGHIÊM TRỌNG khi lấy dữ liệu giá vnstock cho {ticker}:")
        return []


def _fetch_company_tcbs(ticker: str) -> dict:
    """Fetch company overview from TCBS."""
    url = f"{settings.tcbs_base_url}/stock-insight/v2/stock/ticker/{ticker.upper()}/company"
    try:
        timeout_val = getattr(settings, "request_timeout", 10)
        resp = requests.get(url, timeout=timeout_val)
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
        logger.exception(f"LỖI NGHIÊM TRỌNG khi lấy thông tin công ty từ TCBS cho {ticker}:")
        return {}



@tool
def get_company_info(ticker: str) -> str:
    """
    Lấy thông tin doanh nghiệp theo mã chứng khoán.
    Trả về: tên công ty, ngành, sàn, vốn hóa, website, mô tả.

    Args:
        ticker: Mã chứng khoán Việt Nam (VD: VNM, VIC, HPG).
    """
    try:
        key = CacheClient.build_key("company", ticker.upper())
        cached_val = cache.get(key)
        if cached_val:
            logger.debug(f"Cache HIT: company info {ticker}")
            return json.dumps(cached_val, ensure_ascii=False)

        info = _fetch_company_tcbs(ticker)
        if not info:
            info = {"ticker": ticker.upper(), "error": "Không tìm thấy dữ liệu doanh nghiệp từ nguồn API."}

        cache.set(key, info, ttl=settings.cache_ttl_company)
        return json.dumps(info, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.exception(f"Lỗi Tool get_company_info cho mã {ticker}:")
        return json.dumps({"error": f"Lỗi lấy thông tin công ty: {str(e)}"}, ensure_ascii=False)


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
    """
    try:
        if start_date and end_date:
            start = start_date
            end = end_date
        else:
            start, end = _resolve_dates(period, start_date, end_date)

        import pandas as pd
        if pd.to_datetime(start) > pd.to_datetime(end):
            start, end = end, start 

        key = CacheClient.build_key("price", ticker.upper(), start, end)
        cached_val = cache.get(key)
        if cached_val:
            logger.debug(f"Cache HIT: price {ticker} {start}-{end}")
            return json.dumps(cached_val, ensure_ascii=False)

        records = _fetch_price_vnstock(ticker, start, end)
        
        if not records:
            return json.dumps({"error": f"Không có dữ liệu giá giao dịch cho mã {ticker} từ {start} đến {end}."}, ensure_ascii=False)

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
    except Exception as e:
        logger.exception(f"Lỗi Tool get_price_history cho mã {ticker}:")
        return json.dumps({"error": f"Lỗi lấy dữ liệu giá lịch sử: {str(e)}"}, ensure_ascii=False)


@tool
def get_market_overview() -> str:
    """
    Lấy tổng quan thị trường chứng khoán Việt Nam:
    VN-Index, HNX-Index, UPCOM-Index (điểm số và % thay đổi hôm nay).
    """
    try:
        key = CacheClient.build_key("market", "overview")
        cached_val = cache.get(key)
        if cached_val:
            return json.dumps(cached_val, ensure_ascii=False)

        url = f"{settings.tcbs_base_url}/stock-insight/v2/market/index"
        timeout_val = getattr(settings, "request_timeout", 10)
        resp = requests.get(url, timeout=timeout_val)
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
        logger.exception("LỖI Tool get_market_overview:")
        return json.dumps({"error": f"Hệ thống không thể kết nối tới nguồn cấp dữ liệu thị trường (Lỗi: {str(e)})"}, ensure_ascii=False)
    
def _fetch_fundamentals_tcbs(ticker: str) -> dict:
    """Lấy chỉ số tài chính cơ bản từ TCBS."""
    url = f"{settings.tcbs_base_url}/stock-insight/v1/stock/financial-analysis?ticker={ticker.upper()}&yearly=0"
    try:
        timeout_val = getattr(settings, "request_timeout", 10)
        resp = requests.get(url, timeout=timeout_val)
        resp.raise_for_status()
        data = resp.json()
        
        if "data" in data and len(data["data"]) > 0:
            latest = data["data"][0]
            return {
                "period": f"Q{latest.get('quarter')}/{latest.get('year')}",
                "eps": latest.get("eps", 0),
                "pe": latest.get("pe", 0),
                "pb": latest.get("pb", 0),
                "roe": round(latest.get("roe", 0) * 100, 2),
                "roa": round(latest.get("roa", 0) * 100, 2),
                "debt_to_equity": round(latest.get("debtEquity", 0), 2),
            }
        return {}
    except Exception as e:
        logger.warning(f"Lỗi lấy TCBS Fundamentals cho {ticker}: {e}")
        return {}

@tool
def get_financial_fundamentals(ticker: str) -> str:
    """
    Lấy các chỉ số tài chính và định giá cơ bản (P/E, P/B, ROE, ROA, EPS) của một công ty.
    Sử dụng tool này đầu tiên khi người dùng hỏi về "định giá", "tình hình tài chính", "lợi nhuận" hoặc "chỉ số cơ bản".

    Args:
        ticker: Mã chứng khoán (VD: VNM, VIC).
    """
    try:
        key = CacheClient.build_key("fundamentals", ticker.upper())
        if (hit := cache.get(key)):
            return json.dumps(hit, ensure_ascii=False)

        data = _fetch_fundamentals_tcbs(ticker)
        if not data:
            return json.dumps({"error": f"Không tìm thấy số liệu tài chính cơ bản cho {ticker}."}, ensure_ascii=False)

        md_report = f"### 📊 Số liệu Tài chính Cơ bản ({ticker.upper()} - {data['period']})\n"
        md_report += f"- **EPS (Lợi nhuận trên mỗi CP):** {data['eps']:,} VNĐ\n"
        md_report += f"- **P/E (Hệ số Giá/Lợi nhuận):** {data['pe']}\n"
        md_report += f"- **P/B (Hệ số Giá/Sổ sách):** {data['pb']}\n"
        md_report += f"- **ROE (Tỷ suất LN/Vốn CSH):** {data['roe']}%\n"
        md_report += f"- **ROA (Tỷ suất LN/Tổng Tài sản):** {data['roa']}%\n"
        md_report += f"- **Tỷ lệ Nợ/Vốn CSH:** {data['debt_to_equity']} lần\n"

        result = {
            "ticker": ticker.upper(),
            "data": data,
            "markdown": md_report
        }
        
        cache.set(key, result, ttl=86400)
        return json.dumps(result, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.exception(f"Lỗi Tool get_financial_fundamentals cho {ticker}:")
        return json.dumps({"error": f"Lỗi truy xuất dữ liệu: {str(e)}"}, ensure_ascii=False)