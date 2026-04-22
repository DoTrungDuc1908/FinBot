"""
api/app.py
FastAPI application — REST API gateway for FinBot.
"""
from __future__ import annotations

from typing import AsyncGenerator, Optional
import time

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field
from loguru import logger

# from config.settings import Settings
from agents.supervisor import run_supervisor
from core.cache import cache

# ── App setup ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title="FinBot — AI Tư Vấn Đầu Tư Việt Nam",
    description="Hệ thống multi-agent AI hỗ trợ phân tích và tư vấn đầu tư chứng khoán Việt Nam",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Request / Response Models ─────────────────────────────────────────────────

class ChatRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=2000, description="Câu hỏi của người dùng")
    risk_profile: Optional[str] = Field("trung bình", description="Khẩu vị rủi ro: thấp / trung bình / cao")
    session_id: Optional[str] = Field(None, description="Session ID để theo dõi hội thoại")


class ChatResponse(BaseModel):
    answer: str
    session_id: Optional[str]
    latency_ms: int


class HealthResponse(BaseModel):
    status: str
    redis: str
    version: str


# ── Health Check ──────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Kiểm tra trạng thái hệ thống."""
    redis_ok = cache.ping()
    return HealthResponse(
        status="ok",
        redis="connected" if redis_ok else "disconnected",
        version="1.0.0",
    )


# ── Chat Endpoint ─────────────────────────────────────────────────────────────

@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(request: ChatRequest):
    """
    Gửi câu hỏi đến hệ thống multi-agent FinBot.

    **Ví dụ câu hỏi:**
    - "Thông tin công ty VNM?"
    - "Phân tích RSI 14 ngày của HPG trong 3 tháng gần nhất"
    - "Có nên mua VIC không? Khẩu vị rủi ro của tôi là thấp"
    - "Tình hình tài chính của TCB trong năm 2023"
    """
    start = time.perf_counter()
    question = request.question
    if request.risk_profile and request.risk_profile != "trung bình":
        question = f"{question} (khẩu vị rủi ro: {request.risk_profile})"

    try:
        logger.info(f"[{request.session_id}] Q: {request.question[:80]}")
        answer = run_supervisor(question)
        latency = int((time.perf_counter() - start) * 1000)
        logger.info(f"[{request.session_id}] Done in {latency}ms")
        return ChatResponse(
            answer=answer,
            session_id=request.session_id,
            latency_ms=latency,
        )
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=f"Lỗi hệ thống: {str(e)}")


# ── Quick Lookup Endpoints ────────────────────────────────────────────────────

@app.get("/stock/{ticker}", tags=["Stock"])
async def get_stock_info(ticker: str):
    """Lấy thông tin nhanh về một mã chứng khoán."""
    from tools.stock_tools import get_company_info
    try:
        result = get_company_info.invoke({"ticker": ticker})
        return JSONResponse(content={"ticker": ticker.upper(), "data": result})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stock/{ticker}/price", tags=["Stock"])
async def get_stock_price(
    ticker: str,
    period: Optional[str] = "3m",
    start: Optional[str] = None,
    end: Optional[str] = None,
):
    """Lấy lịch sử giá OHLCV. Period: 1w, 1m, 3m, 6m, 1y, ytd."""
    from tools.stock_tools import get_price_history
    try:
        result = get_price_history.invoke({
            "ticker": ticker,
            "period": period,
            "start_date": start,
            "end_date": end,
        })
        return JSONResponse(content={"ticker": ticker.upper(), "data": result})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stock/{ticker}/technical", tags=["Technical"])
async def get_technical(ticker: str, indicator: str = "rsi", window: int = 14, period: str = "6m"):
    """
    Tính chỉ số kỹ thuật. indicator: sma / rsi / macd / bbands.
    """
    from tools.technical_tools import calculate_sma, calculate_rsi, calculate_macd, calculate_bollinger_bands
    tool_map = {
        "sma": (calculate_sma, {"ticker": ticker, "window": window, "period": period}),
        "rsi": (calculate_rsi, {"ticker": ticker, "window": window, "period": period}),
        "macd": (calculate_macd, {"ticker": ticker, "period": period}),
        "bbands": (calculate_bollinger_bands, {"ticker": ticker, "window": window, "period": period}),
    }
    if indicator.lower() not in tool_map:
        raise HTTPException(status_code=400, detail=f"Indicator không hợp lệ. Chọn: {list(tool_map.keys())}")
    tool_fn, kwargs = tool_map[indicator.lower()]
    try:
        result = tool_fn.invoke(kwargs)
        return JSONResponse(content={"ticker": ticker.upper(), "indicator": indicator, "data": result})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stock/{ticker}/sentiment", tags=["Sentiment"])
async def get_stock_sentiment(ticker: str):
    """Phân tích sentiment tin tức cho một mã cổ phiếu."""
    from tools.news_tools import analyze_stock_sentiment
    try:
        result = analyze_stock_sentiment.invoke({"ticker": ticker})
        return JSONResponse(content={"ticker": ticker.upper(), "data": result})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/market", tags=["Market"])
async def get_market():
    """Tổng quan thị trường: VN-Index, HNX, UPCOM."""
    from tools.stock_tools import get_market_overview
    try:
        result = get_market_overview.invoke({})
        return JSONResponse(content={"data": result})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Report Ingestion ──────────────────────────────────────────────────────────

class IngestRequest(BaseModel):
    ticker: str
    pdf_path: str
    report_type: str = Field("financial", description="'financial' hoặc 'analyst'")


@app.post("/ingest/report", tags=["Admin"])
async def ingest_report(request: IngestRequest):
    """Nạp báo cáo PDF vào vector store."""
    from tools.rag_tools import ingest_pdf
    try:
        count = ingest_pdf(request.pdf_path, request.ticker, request.report_type)
        return {"message": f"Đã nạp {count} chunks từ báo cáo", "ticker": request.ticker}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
