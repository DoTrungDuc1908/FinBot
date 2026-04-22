"""
tools/technical_tools.py
LangChain tools for technical analysis indicators.
All calculations are done locally with pandas-ta — no LLM token cost.
Supports: SMA, EMA, RSI, MACD, Bollinger Bands.
"""
from __future__ import annotations

import json
from typing import Optional

import pandas as pd
import pandas_ta as ta  # type: ignore
from langchain_core.tools import tool
from loguru import logger

from core.cache import cache, CacheClient
from tools.stock_tools import _fetch_price_vnstock, _resolve_dates


def _load_price_df(ticker: str, period: str = "6m") -> pd.DataFrame | None:
    """Load OHLCV data as a DataFrame."""
    start, end = _resolve_dates(period, None, None)
    records = _fetch_price_vnstock(ticker, start, end)
    if not records:
        return None
    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _signal_sma(price: float, sma: float) -> str:
    if price > sma * 1.02:
        return "Giá đang TRÊN SMA → xu hướng tăng"
    elif price < sma * 0.98:
        return "Giá đang DƯỚI SMA → xu hướng giảm"
    return "Giá gần SMA → trung tính / tích lũy"


def _signal_rsi(rsi: float) -> str:
    if rsi >= 70:
        return "RSI quá mua (overbought) → cân nhắc chốt lời"
    elif rsi <= 30:
        return "RSI quá bán (oversold) → cân nhắc mua vào"
    elif rsi >= 60:
        return "RSI tăng mạnh, momentum tích cực"
    elif rsi <= 40:
        return "RSI yếu, áp lực bán còn đó"
    return "RSI trung tính (40–60)"


# ── LangChain Tools ───────────────────────────────────────────────────────────

@tool
def calculate_sma(
    ticker: str,
    window: int = 20,
    period: str = "6m",
) -> str:
    """
    Tính Simple Moving Average (SMA) cho mã chứng khoán.

    Args:
        ticker: Mã chứng khoán (VD: VNM, VIC).
        window: Số phiên giao dịch để tính SMA (VD: 10, 20, 50, 200).
        period: Khung thời gian dữ liệu: '1m', '3m', '6m', '1y'. Mặc định '6m'.

    Returns:
        JSON với giá trị SMA gần nhất, giá hiện tại, tín hiệu kỹ thuật và lịch sử SMA.
    """
    cache_key = CacheClient.build_key("sma", ticker.upper(), str(window), period)
    if (hit := cache.get(cache_key)):
        return json.dumps(hit, ensure_ascii=False)

    df = _load_price_df(ticker, period)
    if df is None or len(df) < window:
        return json.dumps({"error": f"Không đủ dữ liệu để tính SMA({window})"})

    df[f"sma_{window}"] = ta.sma(df["close"], length=window)
    last = df.dropna(subset=[f"sma_{window}"]).iloc[-1]
    history = (
        df[[f"sma_{window}"]].dropna().tail(30)
        .reset_index()
        .rename(columns={"date": "date", f"sma_{window}": "sma"})
        .assign(date=lambda x: x["date"].dt.strftime("%Y-%m-%d"))
        .to_dict(orient="records")
    )
    current_price = float(df["close"].iloc[-1])
    sma_val = float(last[f"sma_{window}"])
    result = {
        "ticker": ticker.upper(),
        "indicator": f"SMA({window})",
        "current_price": current_price,
        "sma_value": round(sma_val, 2),
        "signal": _signal_sma(current_price, sma_val),
        "history_30d": history,
    }
    cache.set(cache_key, result, ttl=300)
    return json.dumps(result, ensure_ascii=False, indent=2)


@tool
def calculate_rsi(
    ticker: str,
    window: int = 14,
    period: str = "6m",
) -> str:
    """
    Tính Relative Strength Index (RSI) cho mã chứng khoán.

    Args:
        ticker: Mã chứng khoán (VD: VNM, HPG).
        window: Số phiên để tính RSI. Mặc định 14 (tiêu chuẩn).
        period: Khung thời gian dữ liệu: '1m', '3m', '6m', '1y'. Mặc định '6m'.

    Returns:
        JSON với giá trị RSI hiện tại, tín hiệu và lịch sử 30 ngày.
    """
    cache_key = CacheClient.build_key("rsi", ticker.upper(), str(window), period)
    if (hit := cache.get(cache_key)):
        return json.dumps(hit, ensure_ascii=False)

    df = _load_price_df(ticker, period)
    if df is None or len(df) < window + 1:
        return json.dumps({"error": f"Không đủ dữ liệu để tính RSI({window})"})

    df[f"rsi_{window}"] = ta.rsi(df["close"], length=window)
    history = (
        df[[f"rsi_{window}"]].dropna().tail(30)
        .reset_index()
        .rename(columns={"date": "date", f"rsi_{window}": "rsi"})
        .assign(date=lambda x: x["date"].dt.strftime("%Y-%m-%d"))
        .to_dict(orient="records")
    )
    last_rsi = float(df[f"rsi_{window}"].dropna().iloc[-1])
    result = {
        "ticker": ticker.upper(),
        "indicator": f"RSI({window})",
        "current_price": float(df["close"].iloc[-1]),
        "rsi_value": round(last_rsi, 2),
        "signal": _signal_rsi(last_rsi),
        "overbought_level": 70,
        "oversold_level": 30,
        "history_30d": history,
    }
    cache.set(cache_key, result, ttl=300)
    return json.dumps(result, ensure_ascii=False, indent=2)


@tool
def calculate_macd(
    ticker: str,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
    period: str = "1y",
) -> str:
    """
    Tính MACD (Moving Average Convergence Divergence) cho mã chứng khoán.

    Args:
        ticker: Mã chứng khoán.
        fast: EMA nhanh (mặc định 12).
        slow: EMA chậm (mặc định 26).
        signal: Signal line (mặc định 9).
        period: Khung thời gian: '6m' hoặc '1y'. Mặc định '1y'.

    Returns:
        JSON với MACD, signal line, histogram và tín hiệu kỹ thuật.
    """
    cache_key = CacheClient.build_key("macd", ticker.upper(), str(fast), str(slow), period)
    if (hit := cache.get(cache_key)):
        return json.dumps(hit, ensure_ascii=False)

    df = _load_price_df(ticker, period)
    if df is None or len(df) < slow + signal:
        return json.dumps({"error": "Không đủ dữ liệu để tính MACD"})

    macd_df = ta.macd(df["close"], fast=fast, slow=slow, signal=signal)
    col_macd = f"MACD_{fast}_{slow}_{signal}"
    col_hist = f"MACDh_{fast}_{slow}_{signal}"
    col_sig = f"MACDs_{fast}_{slow}_{signal}"

    if macd_df is None or col_macd not in macd_df.columns:
        return json.dumps({"error": "Lỗi tính MACD"})

    last = macd_df.dropna().iloc[-1]
    macd_val = float(last[col_macd])
    sig_val = float(last[col_sig])
    hist_val = float(last[col_hist])

    if macd_val > sig_val and hist_val > 0:
        interpretation = "MACD cắt lên trên Signal → tín hiệu MUA"
    elif macd_val < sig_val and hist_val < 0:
        interpretation = "MACD cắt xuống dưới Signal → tín hiệu BÁN"
    else:
        interpretation = "MACD đang hội tụ → chờ tín hiệu rõ hơn"

    result = {
        "ticker": ticker.upper(),
        "indicator": f"MACD({fast},{slow},{signal})",
        "macd": round(macd_val, 4),
        "signal_line": round(sig_val, 4),
        "histogram": round(hist_val, 4),
        "signal": interpretation,
    }
    cache.set(cache_key, result, ttl=300)
    return json.dumps(result, ensure_ascii=False, indent=2)


@tool
def calculate_bollinger_bands(
    ticker: str,
    window: int = 20,
    std: float = 2.0,
    period: str = "6m",
) -> str:
    """
    Tính Bollinger Bands cho mã chứng khoán.

    Args:
        ticker: Mã chứng khoán.
        window: Số phiên (mặc định 20).
        std: Hệ số độ lệch chuẩn (mặc định 2.0).
        period: Khung thời gian dữ liệu.

    Returns:
        JSON với upper band, middle (SMA), lower band và vị trí giá hiện tại.
    """
    cache_key = CacheClient.build_key("bbands", ticker.upper(), str(window), period)
    if (hit := cache.get(cache_key)):
        return json.dumps(hit, ensure_ascii=False)

    df = _load_price_df(ticker, period)
    if df is None or len(df) < window:
        return json.dumps({"error": f"Không đủ dữ liệu để tính Bollinger Bands({window})"})

    bb = ta.bbands(df["close"], length=window, std=std)
    if bb is None:
        return json.dumps({"error": "Lỗi tính Bollinger Bands"})

    col_upper = f"BBU_{window}_{std}"
    col_mid = f"BBM_{window}_{std}"
    col_lower = f"BBL_{window}_{std}"

    last = bb.dropna().iloc[-1]
    price = float(df["close"].iloc[-1])
    upper = float(last[col_upper])
    mid = float(last[col_mid])
    lower = float(last[col_lower])

    if price > upper:
        pos = "Giá TRÊN upper band → overbought"
    elif price < lower:
        pos = "Giá DƯỚI lower band → oversold"
    else:
        pct = (price - lower) / (upper - lower) * 100
        pos = f"Giá trong band, vị trí {pct:.0f}% từ lower→upper"

    result = {
        "ticker": ticker.upper(),
        "indicator": f"BB({window},{std})",
        "current_price": price,
        "upper_band": round(upper, 2),
        "middle_band": round(mid, 2),
        "lower_band": round(lower, 2),
        "band_width": round((upper - lower) / mid * 100, 2),
        "position": pos,
    }
    cache.set(cache_key, result, ttl=300)
    return json.dumps(result, ensure_ascii=False, indent=2)
