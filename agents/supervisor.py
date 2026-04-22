"""
agents/supervisor.py
LangGraph Supervisor — intent classification, parallel dispatch, response synthesis.
"""
from __future__ import annotations

import asyncio
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END
from loguru import logger

from core.llm import get_fast_llm, get_llm
from agents.stock_info_agent import run_stock_info_agent
from agents.technical_agent import run_technical_agent
from agents.sentiment_agent import run_sentiment_agent
from agents.report_rag_agent import run_report_rag_agent
from agents.advisor_agent import synthesize_investment_advice

# ── Intent labels ─────────────────────────────────────────────────────────────
INTENTS = {
    "stock_info": "Thông tin công ty, giá cổ phiếu, lịch sử giá, OHLCV, volume, sàn, vốn hóa",
    "technical": "Phân tích kỹ thuật, SMA, RSI, MACD, Bollinger Bands, xu hướng, tín hiệu kỹ thuật",
    "sentiment": "Tin tức, sentiment, tâm lý thị trường, bài báo, sự kiện, thông tin gần đây",
    "report_rag": "Báo cáo tài chính, BCTC, doanh thu, lợi nhuận, EPS, ROE, báo cáo CTCK, khuyến nghị",
    "advisor": "Tư vấn đầu tư, nên mua, nên bán, có nên đầu tư, đánh giá tổng thể, tiềm năng",
    "market": "Thị trường chung, VN-Index, HNX, tổng quan thị trường",
}

CLASSIFY_PROMPT = """Bạn là bộ phân loại intent cho hệ thống tư vấn đầu tư.
Phân loại câu hỏi vào MỘT HOẶC NHIỀU loại sau (dùng dấu phẩy):
- stock_info: {stock_info}
- technical: {technical}
- sentiment: {sentiment}
- report_rag: {report_rag}
- advisor: {advisor}
- market: {market}

Câu hỏi: {{question}}

Chỉ trả về các nhãn, cách nhau bằng dấu phẩy. Ví dụ: technical,sentiment
Không giải thích thêm.""".format(**INTENTS)

TICKER_PATTERN = re.compile(r'\b([A-Z]{2,5})\b')
VN_TICKERS = {
    "VNM", "VIC", "VHM", "VRE", "HPG", "TCB", "VPB", "MBB", "BID", "CTG",
    "VCB", "FPT", "MSN", "MWG", "GVR", "SAB", "PLX", "GAS", "POW", "PVD",
    "SSI", "VND", "HDB", "TPB", "ACB", "STB", "SHB", "EIB", "KDC", "DGC",
    "DBC", "REE", "PNJ", "KBC", "NVL", "PDR", "DXG", "KDH", "VPI", "AGR",
}


def extract_ticker(text: str) -> str | None:
    """Extract Vietnamese stock ticker from user text."""
    matches = TICKER_PATTERN.findall(text.upper())
    for m in matches:
        if m in VN_TICKERS:
            return m
    # Return first 2-5 uppercase word if any
    for m in matches:
        if 2 <= len(m) <= 5 and m not in {"SMA", "RSI", "EMA", "VNĐ", "USD", "ROE", "ROA", "EPS"}:
            return m
    return None


def extract_risk_profile(text: str) -> str:
    """Extract risk profile from user text."""
    text_lower = text.lower()
    if any(k in text_lower for k in ["thấp", "an toàn", "bảo thủ", "ít rủi ro"]):
        return "thấp"
    if any(k in text_lower for k in ["cao", "rủi ro cao", "tích cực", "mạo hiểm"]):
        return "cao"
    return "trung bình"


# ── LangGraph State ───────────────────────────────────────────────────────────

class SupervisorState(TypedDict):
    question: str
    ticker: str | None
    risk_profile: str
    intents: list[str]
    stock_info_result: str
    technical_result: str
    sentiment_result: str
    report_rag_result: str
    final_answer: str


# ── Graph Nodes ───────────────────────────────────────────────────────────────

def classify_node(state: SupervisorState) -> SupervisorState:
    """Classify user intent using fast LLM."""
    question = state["question"]
    state["ticker"] = extract_ticker(question)
    state["risk_profile"] = extract_risk_profile(question)

    prompt = CLASSIFY_PROMPT.replace("{question}", question)
    llm = get_fast_llm()
    try:
        raw = llm.invoke([HumanMessage(content=prompt)]).content.strip().lower()
        intents = [i.strip() for i in raw.split(",") if i.strip() in INTENTS]
    except Exception as e:
        logger.warning(f"Intent classification failed: {e}, defaulting to stock_info")
        intents = ["stock_info"]

    if not intents:
        intents = ["stock_info"]

    # If advisor is requested, ensure we have enough context
    if "advisor" in intents:
        for needed in ["stock_info", "technical", "sentiment"]:
            if needed not in intents:
                intents.append(needed)

    state["intents"] = intents
    logger.info(f"Intents: {intents} | Ticker: {state['ticker']}")
    return state


def dispatch_node(state: SupervisorState) -> SupervisorState:
    """Dispatch to sub-agents in parallel using ThreadPoolExecutor."""
    intents = state["intents"]
    question = state["question"]
    ticker = state["ticker"]
    ticker_prefix = f"[{ticker}] " if ticker else ""

    tasks: dict[str, tuple] = {}
    if "stock_info" in intents or "market" in intents:
        tasks["stock_info"] = (run_stock_info_agent, f"{ticker_prefix}{question}")
    if "technical" in intents:
        tasks["technical"] = (run_technical_agent, f"{ticker_prefix}{question}")
    if "sentiment" in intents:
        tasks["sentiment"] = (run_sentiment_agent, f"{ticker_prefix}{question}")
    if "report_rag" in intents:
        tasks["report_rag"] = (run_report_rag_agent, f"{ticker_prefix}{question}")

    results: dict[str, str] = {
        "stock_info": "", "technical": "", "sentiment": "", "report_rag": ""
    }

    def _run(key: str, fn, q: str) -> tuple[str, str]:
        try:
            return key, fn(q)
        except Exception as e:
            logger.error(f"Agent {key} error: {e}")
            return key, f"[Lỗi: {e}]"

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(_run, k, fn, q): k for k, (fn, q) in tasks.items()}
        for future in futures:
            key, output = future.result(timeout=60)
            results[key] = output

    state["stock_info_result"] = results["stock_info"]
    state["technical_result"] = results["technical"]
    state["sentiment_result"] = results["sentiment"]
    state["report_rag_result"] = results["report_rag"]
    return state


def synthesize_node(state: SupervisorState) -> SupervisorState:
    """Synthesize final answer from sub-agent outputs."""
    intents = state["intents"]

    # If advisor → structured investment report
    if "advisor" in intents and state["ticker"]:
        answer = synthesize_investment_advice(
            ticker=state["ticker"],
            stock_info=state["stock_info_result"],
            technical_analysis=state["technical_result"],
            sentiment=state["sentiment_result"],
            report_summary=state["report_rag_result"],
            risk_profile=state["risk_profile"],
        )
        state["final_answer"] = answer
        return state

    # Single-intent: return the relevant result directly
    if len(intents) == 1:
        key = intents[0]
        mapping = {
            "stock_info": state["stock_info_result"],
            "market": state["stock_info_result"],
            "technical": state["technical_result"],
            "sentiment": state["sentiment_result"],
            "report_rag": state["report_rag_result"],
        }
        state["final_answer"] = mapping.get(key, "Không có kết quả.")
        return state

    # Multi-intent: combine with LLM
    parts = []
    if state["stock_info_result"]:
        parts.append(f"**Thông tin cổ phiếu:**\n{state['stock_info_result']}")
    if state["technical_result"]:
        parts.append(f"**Phân tích kỹ thuật:**\n{state['technical_result']}")
    if state["sentiment_result"]:
        parts.append(f"**Tin tức & Sentiment:**\n{state['sentiment_result']}")
    if state["report_rag_result"]:
        parts.append(f"**Báo cáo tài chính:**\n{state['report_rag_result']}")

    combined = "\n\n".join(parts)
    synthesis_prompt = (
        f"Câu hỏi người dùng: {state['question']}\n\n"
        f"Thông tin thu thập được:\n{combined}\n\n"
        "Hãy tổng hợp thành câu trả lời hoàn chỉnh, ngắn gọn bằng tiếng Việt."
    )
    llm = get_llm(temperature=0.1)
    chain = llm | StrOutputParser()
    state["final_answer"] = chain.invoke([HumanMessage(content=synthesis_prompt)])
    return state


# ── Build LangGraph ───────────────────────────────────────────────────────────

def _build_graph() -> Any:
    graph = StateGraph(SupervisorState)
    graph.add_node("classify", classify_node)
    graph.add_node("dispatch", dispatch_node)
    graph.add_node("synthesize", synthesize_node)

    graph.set_entry_point("classify")
    graph.add_edge("classify", "dispatch")
    graph.add_edge("dispatch", "synthesize")
    graph.add_edge("synthesize", END)

    return graph.compile()


_graph = None


def get_supervisor() -> Any:
    global _graph
    if _graph is None:
        _graph = _build_graph()
    return _graph


def run_supervisor(question: str) -> str:
    """
    Main entry point. Run the full supervisor pipeline.

    Args:
        question: User's natural language question in Vietnamese.

    Returns:
        Final answer string.
    """
    supervisor = get_supervisor()
    initial_state: SupervisorState = {
        "question": question,
        "ticker": None,
        "risk_profile": "trung bình",
        "intents": [],
        "stock_info_result": "",
        "technical_result": "",
        "sentiment_result": "",
        "report_rag_result": "",
        "final_answer": "",
    }
    result = supervisor.invoke(initial_state)
    return result.get("final_answer", "Xin lỗi, tôi không thể xử lý câu hỏi này.")
