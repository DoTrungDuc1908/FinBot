"""
tools/technical_tools.py
LangChain tools for technical analysis indicators.
All calculations are done locally with pandas-ta — no LLM token cost.
Supports: SMA, RSI, MACD, Bollinger Bands with Semantic Insights.
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


def _load_price_df(ticker: str, period: str = "1y", start_date: str = None, end_date: str = None, interval: str = "1D") -> pd.DataFrame | None:
    """Load OHLCV data as a DataFrame with explicit dates (ignoring period)."""
    from datetime import datetime, timedelta
    
    today = datetime.now()
    end = end_date if end_date else today.strftime("%Y-%m-%d")
    start = start_date if start_date else (today - timedelta(days=365)).strftime("%Y-%m-%d")

    try:
        fetch_start_dt = datetime.strptime(start, "%Y-%m-%d") - timedelta(days=60)
        fetch_start = fetch_start_dt.strftime("%Y-%m-%d")
    except Exception:
        fetch_start = start

    from vnstock import Quote
    quote = Quote(symbol=ticker.upper())
    df = quote.history(start=fetch_start, end=end, interval=interval)
    
    if df is None or df.empty:
        return None
        
    df = df.rename(columns={"time": "date"})
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _signal_sma(price: float, sma: float) -> str:
    if price > sma * 1.02: return "Giá đang TRÊN SMA → xu hướng tăng"
    elif price < sma * 0.98: return "Giá đang DƯỚI SMA → xu hướng giảm"
    return "Giá gần SMA → trung tính / tích lũy"

def _signal_rsi(rsi: float) -> str:
    if rsi >= 70: return "RSI quá mua (overbought) → cân nhắc chốt lời"
    elif rsi <= 30: return "RSI quá bán (oversold) → cân nhắc mua vào"
    elif rsi >= 60: return "RSI tăng mạnh, momentum tích cực"
    elif rsi <= 40: return "RSI yếu, áp lực bán còn đó"
    return "RSI trung tính (40–60)"



def _generate_sma_insights(df: pd.DataFrame, window: int) -> str:
    sma_col = f"sma_{window}"
    total_days = len(df)
    
    days_above = len(df[df["close"] > df[sma_col]])
    win_rate = (days_above / total_days) * 100
    
    current_price = df["close"].iloc[-1]
    current_sma = df[sma_col].iloc[-1]
    prev_sma = df[sma_col].iloc[-2]
    
    trend_now = "TĂNG TÍCH CỰC" if current_price > current_sma and current_sma > prev_sma else \
                "HỒI PHỤC KỸ THUẬT" if current_price > current_sma and current_sma <= prev_sma else \
                "GIẢM TIÊU CỰC" if current_price < current_sma and current_sma < prev_sma else \
                "ĐIỀU CHỈNH/TÍCH LŨY"
    
    return (f"📊 THỐNG KÊ SMA({window}) TRONG {total_days} PHIÊN:\n"
            f"- Lịch sử: Giá nằm TRÊN đường SMA {days_above} phiên (chiếm {win_rate:.1f}% thời gian).\n"
            f"- Trạng thái hiện tại: Giá đang ở mức {current_price}, SMA ở mức {current_sma:.2f}.\n"
            f"- Đánh giá xu hướng: {trend_now}.")

def _generate_rsi_insights(df: pd.DataFrame, window: int) -> str:
    rsi_col = f"rsi_{window}"
    total_days = len(df)
    
    overbought_count = len(df[df[rsi_col] >= 70])
    oversold_count = len(df[df[rsi_col] <= 30])
    
    current_rsi = df[rsi_col].iloc[-1]
    prev_rsi_avg = df[rsi_col].iloc[-4:-1].mean()
    
    momentum = "Đang dốc lên (Dòng tiền vào)" if current_rsi > prev_rsi_avg else "Đang cắm xuống (Áp lực chốt lời)"
    
    warning = ""
    if current_rsi >= 70: warning = "⚠️ RỦI RO ĐU ĐỈNH: Cổ phiếu đã vào vùng quá mua."
    elif current_rsi <= 30: warning = "💡 CƠ HỘI BẮT ĐÁY: Cổ phiếu đã bị bán tháo quá mức."
    else: warning = "Vùng giá trung tính, chưa có tín hiệu cực đoan."

    return (f"📊 THỐNG KÊ RSI({window}) TRONG {total_days} PHIÊN:\n"
            f"- Lịch sử: {overbought_count} lần chạm Quá Mua (>=70) và {oversold_count} lần chạm Quá Bán (<=30).\n"
            f"- Trạng thái hiện tại: RSI = {current_rsi:.1f} ({momentum}).\n"
            f"- Đánh giá xu hướng: {warning}")

def _generate_macd_insights(df: pd.DataFrame, fast: int, slow: int, signal: int, col_macd: str, col_sig: str) -> str:
    total_days = len(df)
    col_hist = df.columns[df.columns.str.contains('MACDh')][0]
    
    diff = df[col_macd] - df[col_sig]
    crossovers = (diff * diff.shift(1) < 0).sum()
    
    current_macd = df[col_macd].iloc[-1]
    current_sig = df[col_sig].iloc[-1]
    current_hist = df[col_hist].iloc[-1]
    prev_hist = df[col_hist].iloc[-2]
    
    trend = ""
    if current_macd > current_sig:
        if current_hist > prev_hist: trend = "Xu hướng TĂNG ĐƯỢC CỦNG CỐ (Histogram mở rộng dương)."
        else: trend = "Xu hướng TĂNG SUY YẾU (Histogram thu hẹp, cẩn trọng đảo chiều)."
    else:
        if current_hist < prev_hist: trend = "Xu hướng GIẢM ĐANG MẠNH LÊN (Histogram mở rộng âm)."
        else: trend = "Áp lực GIẢM CẠN KIỆT (Histogram thu hẹp, có thể sắp tạo đáy)."

    return (f"📊 THỐNG KÊ MACD({fast},{slow},{signal}) TRONG {total_days} PHIÊN:\n"
            f"- Lịch sử: Tính dao động cao, đã xảy ra {crossovers} lần giao cắt.\n"
            f"- Trạng thái hiện tại: MACD ({current_macd:.3f}), Signal ({current_sig:.3f}).\n"
            f"- Đánh giá xu hướng: {trend}")

def _generate_bb_insights(df: pd.DataFrame, window: int, std: float, col_upper: str, col_lower: str) -> str:
    total_days = len(df)
    
    above_upper = len(df[df["close"] > df[col_upper]])
    below_lower = len(df[df["close"] < df[col_lower]])
    
    current_price = df["close"].iloc[-1]
    current_upper = df[col_upper].iloc[-1]
    current_lower = df[col_lower].iloc[-1]
    
    current_bw = (current_upper - current_lower) / current_price
    prev_bw = (df[col_upper].iloc[-2] - df[col_lower].iloc[-2]) / df["close"].iloc[-2]
    
    volatility = "MỞ RỘNG (Biến động mạnh/Sắp có xu hướng rõ ràng)" if current_bw > prev_bw * 1.05 else \
                 "THU HẸP (Tích lũy nén chặt/Sắp có bứt phá)" if current_bw < prev_bw * 0.95 else \
                 "ĐIỀU HÒA (Biến động ổn định)"

    if current_price >= current_upper:
        position = "Giá chạm/vượt CẠNH TRÊN (Upper Band)"
        trend = "Quá mua (Overbought) - Rủi ro chốt lời hoặc cổ phiếu đang vào siêu sóng tăng."
    elif current_price <= current_lower:
        position = "Giá chạm/thủng CẠNH DƯỚI (Lower Band)"
        trend = "Quá bán (Oversold) - Cơ hội hồi phục hoặc cổ phiếu đang rơi rụng mạnh."
    else:
        percent_b = (current_price - current_lower) / (current_upper - current_lower)
        if percent_b > 0.5:
            position = "Nửa TRÊN của dải Bollinger"
            trend = "Tích cực - Lực cầu đang chiếm ưu thế nhẹ."
        else:
            position = "Nửa DƯỚI của dải Bollinger"
            trend = "Tiêu cực - Áp lực bán đang lấn lướt."

    return (f"📊 THỐNG KÊ BOLLINGER BANDS({window}, {std}) TRONG {total_days} PHIÊN:\n"
            f"- Lịch sử: {above_upper} lần bứt dải trên và {below_lower} lần thủng dải dưới.\n"
            f"- Độ biến động: Dải Bollinger đang {volatility}.\n"
            f"- Vị trí hiện tại: {position}.\n"
            f"- Đánh giá xu hướng: {trend}")


@tool
def calculate_sma(ticker: str, window: int = 14, period: str = "1y", start_date: str = None, end_date: str = None, interval: str = "1D", full_data: bool = False) -> str:
    """Tính SMA. Trả về JSON chứa tín hiệu và insight dài hạn."""
    try:
        cache_key = CacheClient.build_key("sma", ticker.upper(), str(window), period, str(full_data))
        if (hit := cache.get(cache_key)): return json.dumps(hit, ensure_ascii=False)

        df = _load_price_df(ticker, period, start_date, end_date, interval)
        if df is None or len(df) < window:
            return json.dumps({"error": f"Không đủ dữ liệu tính SMA({window})"})

        df[f"sma_{window}"] = ta.sma(df["close"], length=window)
        df_valid = df.dropna(subset=[f"sma_{window}"])
        actual_start = start_date if start_date else _resolve_dates(period, None, None)[0]
        
        actual_start_dt = pd.to_datetime(actual_start)
        df_valid = df_valid[df_valid.index >= actual_start_dt]
        
        if df_valid.empty:
            return json.dumps({"error": f"Không có dữ liệu giao dịch hợp lệ từ ngày {actual_start}."}, ensure_ascii=False)
        
        last = df_valid.iloc[-1]
        current_price = float(last["close"])
        sma_val = float(last[f"sma_{window}"])

        result = {
            "ticker": ticker.upper(),
            "indicator": f"SMA({window})",
            "current_price": current_price,
            "sma_value": round(sma_val, 2),
            "signal": _signal_sma(current_price, sma_val),
            "historical_insights": _generate_sma_insights(df_valid, window),
        }

        if full_data:
            result["history_data"] = (
                df_valid[[f"sma_{window}", "close"]].reset_index()
                .rename(columns={"date": "date", f"sma_{window}": "sma"})
                .assign(date=lambda x: x["date"].dt.strftime("%Y-%m-%d"))
                .to_dict(orient="records")
            )
        else:
            result["history_data"] = "Bị ẩn. Hãy dùng historical_insights để phân tích."

        cache.set(cache_key, result, ttl=300)
        return json.dumps(result, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.exception(f"Lỗi Tool calculate_sma cho {ticker}:")
        return json.dumps({"error": f"Lỗi tính toán SMA: {str(e)}"}, ensure_ascii=False)


@tool
def calculate_rsi(ticker: str, window: int = 14, period: str = "1y", start_date: str = None, end_date: str = None, interval: str = "1D", full_data: bool = False) -> str:
    """Tính RSI. Trả về JSON chứa tín hiệu và insight dài hạn."""
    try:
        cache_key = CacheClient.build_key("rsi", ticker.upper(), str(window), period, str(full_data))
        if (hit := cache.get(cache_key)): return json.dumps(hit, ensure_ascii=False)

        df = _load_price_df(ticker, period, start_date, end_date, interval)
        if df is None or len(df) < window + 1:
            return json.dumps({"error": f"Không đủ dữ liệu tính RSI({window})"})

        df[f"rsi_{window}"] = ta.rsi(df["close"], length=window)
        df_valid = df.dropna(subset=[f"rsi_{window}"])
        actual_start, _ = _resolve_dates(period, start_date, end_date)
        actual_start = start_date if start_date else _resolve_dates(period, None, None)[0]
        
        actual_start_dt = pd.to_datetime(actual_start)
        df_valid = df_valid[df_valid.index >= actual_start_dt]
        
        if df_valid.empty:
            return json.dumps({"error": f"Không có dữ liệu giao dịch hợp lệ từ ngày {actual_start}."}, ensure_ascii=False)
        
        last = df_valid.iloc[-1]
        last_rsi = float(last[f"rsi_{window}"])

        result = {
            "ticker": ticker.upper(),
            "indicator": f"RSI({window})",
            "current_price": float(last["close"]),
            "rsi_value": round(last_rsi, 2),
            "signal": _signal_rsi(last_rsi),
            "historical_insights": _generate_rsi_insights(df_valid, window),
        }

        if full_data:
            result["history_data"] = (
                df_valid[[f"rsi_{window}", "close"]].reset_index()
                .rename(columns={"date": "date", f"rsi_{window}": "rsi"})
                .assign(date=lambda x: x["date"].dt.strftime("%Y-%m-%d"))
                .to_dict(orient="records")
            )
        else:
            result["history_data"] = "Bị ẩn. Hãy dùng historical_insights để phân tích."

        cache.set(cache_key, result, ttl=300)
        return json.dumps(result, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.exception(f"Lỗi Tool calculate_rsi cho {ticker}:")
        return json.dumps({"error": f"Lỗi tính toán RSI: {str(e)}"}, ensure_ascii=False)


@tool
def calculate_macd(ticker: str, fast: int = 12, slow: int = 26, signal: int = 9, period: str = "1y", start_date: str = None, end_date: str = None, interval: str = "1D", full_data: bool = False) -> str:
    """Tính MACD. Trả về JSON chứa tín hiệu và insight dài hạn."""
    try:
        cache_key = CacheClient.build_key("macd", ticker.upper(), str(fast), str(slow), period, str(full_data))
        if (hit := cache.get(cache_key)): return json.dumps(hit, ensure_ascii=False)

        df = _load_price_df(ticker, period, start_date, end_date, interval)
        if df is None or len(df) < slow + signal:
            return json.dumps({"error": "Không đủ dữ liệu để tính MACD"})

        macd_df = ta.macd(df["close"], fast=fast, slow=slow, signal=signal)
        if macd_df is None or macd_df.empty:
            return json.dumps({"error": "Lỗi tính MACD"})

        col_macd, col_hist, col_sig = macd_df.columns[0], macd_df.columns[1], macd_df.columns[2]

        df = pd.concat([df, macd_df], axis=1)
        df_valid = df.dropna(subset=[col_macd])
        actual_start = start_date if start_date else _resolve_dates(period, None, None)[0]
        
        actual_start_dt = pd.to_datetime(actual_start)
        df_valid = df_valid[df_valid.index >= actual_start_dt]
        
        if df_valid.empty:
            return json.dumps({"error": f"Không có dữ liệu giao dịch hợp lệ từ ngày {actual_start}."}, ensure_ascii=False)
        
        last = df_valid.iloc[-1]
        macd_val, sig_val, hist_val = float(last[col_macd]), float(last[col_sig]), float(last[col_hist])

        if macd_val > sig_val and hist_val > 0: interp = "MACD cắt lên trên Signal → tín hiệu MUA"
        elif macd_val < sig_val and hist_val < 0: interp = "MACD cắt xuống dưới Signal → tín hiệu BÁN"
        else: interp = "MACD đang hội tụ → chờ tín hiệu rõ hơn"

        result = {
            "ticker": ticker.upper(),
            "indicator": f"MACD({fast},{slow},{signal})",
            "macd": round(macd_val, 4),
            "signal_line": round(sig_val, 4),
            "histogram": round(hist_val, 4),
            "signal": interp,
            "historical_insights": _generate_macd_insights(df_valid, fast, slow, signal, col_macd, col_sig),
        }

        if full_data:
            result["history_data"] = (
                df_valid[[col_macd, col_sig, col_hist, "close"]].reset_index()
                .rename(columns={col_macd: "macd", col_sig: "signal_line", col_hist: "histogram"})
                .assign(date=lambda x: x["date"].dt.strftime("%Y-%m-%d"))
                .to_dict(orient="records")
            )
        else:
            result["history_data"] = "Bị ẩn. Hãy dùng historical_insights để phân tích."

        cache.set(cache_key, result, ttl=300)
        return json.dumps(result, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.exception(f"Lỗi Tool calculate_macd cho {ticker}:")
        return json.dumps({"error": f"Lỗi tính toán MACD: {str(e)}"}, ensure_ascii=False)


@tool
def calculate_bollinger_bands(ticker: str, window: int = 20, std: float = 2.0, period: str = "1y", start_date: str = None, end_date: str = None, interval: str = "1D", full_data: bool = False) -> str:
    """Tính Bollinger Bands. Trả về JSON chứa tín hiệu và insight dài hạn."""
    try:
        cache_key = CacheClient.build_key("bbands", ticker.upper(), str(window), period, str(full_data))
        if (hit := cache.get(cache_key)): return json.dumps(hit, ensure_ascii=False)

        df = _load_price_df(ticker, period, start_date, end_date, interval)
        if df is None or len(df) < window:
            return json.dumps({"error": f"Không đủ dữ liệu tính Bollinger Bands({window})"})

        bb = ta.bbands(df["close"], length=window, std=std)
        if bb is None or bb.empty: return json.dumps({"error": "Lỗi tính Bollinger Bands"})

        col_lower, col_mid, col_upper = bb.columns[0], bb.columns[1], bb.columns[2]
        
        df = pd.concat([df, bb], axis=1)
        df_valid = df.dropna(subset=[col_upper])
        actual_start = start_date if start_date else _resolve_dates(period, None, None)[0]
        
        actual_start_dt = pd.to_datetime(actual_start)
        df_valid = df_valid[df_valid.index >= actual_start_dt]
        
        if df_valid.empty:
            return json.dumps({"error": f"Không có dữ liệu giao dịch hợp lệ từ ngày {actual_start}."}, ensure_ascii=False)

        last = df_valid.iloc[-1]
        price, upper, mid, lower = float(last["close"]), float(last[col_upper]), float(last[col_mid]), float(last[col_lower])

        if price > upper: pos = "Giá TRÊN upper band → overbought"
        elif price < lower: pos = "Giá DƯỚI lower band → oversold"
        else: pos = f"Giá trong band, vị trí {(price - lower) / (upper - lower) * 100:.0f}% từ dưới lên"

        result = {
            "ticker": ticker.upper(),
            "indicator": f"BB({window},{std})",
            "current_price": price,
            "upper_band": round(upper, 2),
            "middle_band": round(mid, 2),
            "lower_band": round(lower, 2),
            "band_width": round((upper - lower) / mid * 100, 2),
            "position": pos,
            "historical_insights": _generate_bb_insights(df_valid, window, std, col_upper, col_lower),
        }

        if full_data:
            result["history_data"] = (
                df_valid[[col_upper, col_mid, col_lower, "close"]].reset_index()
                .rename(columns={col_upper: "upper_band", col_mid: "middle_band", col_lower: "lower_band"})
                .assign(date=lambda x: x["date"].dt.strftime("%Y-%m-%d"))
                .to_dict(orient="records")
            )
        else:
            result["history_data"] = "Bị ẩn. Hãy dùng historical_insights để phân tích."

        cache.set(cache_key, result, ttl=300)
        return json.dumps(result, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.exception(f"Lỗi Tool calculate_bollinger_bands cho {ticker}:")
        return json.dumps({"error": f"Lỗi tính toán Bollinger Bands: {str(e)}"}, ensure_ascii=False)