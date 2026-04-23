"""
agents/supervisor.py
LangGraph Supervisor — intent classification (Structured Output), parallel dispatch (Async), response synthesis.
"""
from __future__ import annotations

import asyncio
import re
from typing import Any, TypedDict, List
from pydantic import BaseModel, Field
from aiolimiter import AsyncLimiter
import json

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.output_parsers import JsonOutputParser
from langgraph.graph import StateGraph, END
from loguru import logger

from core.llm import get_fast_llm, get_llm
from config.settings import settings
from agents.stock_info_agent import run_stock_info_agent
from agents.technical_agent import run_technical_agent
from agents.sentiment_agent import run_sentiment_agent
from agents.report_rag_agent import run_report_rag_agent
from agents.advisor_agent import synthesize_investment_advice

class IntentClassification(BaseModel):
    selected_agents: List[str] = Field(
        description="Danh sách các agent cần gọi. Các giá trị hợp lệ bắt buộc lấy từ: stock_info, technical, sentiment, report_rag, advisor, market"
    )

INTENTS = {
    "stock_info": "Thông tin công ty, giá cổ phiếu, lịch sử giá, OHLCV, volume, sàn, vốn hóa",
    "technical": "Phân tích kỹ thuật, SMA, RSI, MACD, Bollinger Bands, xu hướng, tín hiệu kỹ thuật",
    "sentiment": "Tin tức, sentiment, tâm lý thị trường, bài báo, sự kiện, thông tin gần đây",
    "report_rag": "Báo cáo tài chính, BCTC, doanh thu, lợi nhuận, EPS, ROE, báo cáo CTCK, khuyến nghị",
    "advisor": "Tư vấn đầu tư, nên mua, nên bán, có nên đầu tư, đánh giá tổng thể, tiềm năng",
    "market": "Thị trường chung, VN-Index, HNX, tổng quan thị trường",
}

CLASSIFY_SYSTEM = """Bạn là bộ phân loại intent cho hệ thống tư vấn đầu tư.
Nhiệm vụ của bạn là phân loại câu hỏi của người dùng vào MỘT HOẶC NHIỀU danh mục sau đây:
- stock_info: {stock_info}
- technical: {technical}
- sentiment: {sentiment}
- report_rag: {report_rag}
- advisor: {advisor}
- market: {market}
""".format(**INTENTS)

TICKER_PATTERN = re.compile(r'\b([A-Z]{2,5})\b')
VN_TICKERS = {
    "VNM", "VIC", "VHM", "VRE", "HPG", "TCB", "VPB", "MBB", "BID", "CTG",
    "VCB", "FPT", "MSN", "MWG", "GVR", "SAB", "PLX", "GAS", "POW", "PVD",
    "SSI", "VND", "HDB", "TPB", "ACB", "STB", "SHB", "EIB", "KDC", "DGC",
    "DBC", "REE", "PNJ", "KBC", "NVL", "PDR", "DXG", "KDH", "VPI", "AGR",
}

EXCLUDE_TICKERS = {
    "SMA", "RSI", "EMA", "VNĐ", "VND", "USD", "ROE", "ROA", "EPS", "PE", "PB",
    "XU", "CÓ", "LÀ", "VÀ", "MÀ", "THÌ", "TIN", "TỨC", "GIÁ", "MÃ", "CHỈ", "SỐ",
    "MUA", "BÁN", "GIỮ", "XEM", "HAY", "NÊN", "LÃI", "LỖ", "CÁC"
}

def extract_ticker(text: str) -> str | None:
    matches = TICKER_PATTERN.findall(text.upper())
    for m in matches:
        if m in VN_TICKERS: return m
    for m in matches:
        if len(m) == 3 and m not in EXCLUDE_TICKERS: return m
    return None

def extract_risk_profile(text: str) -> str:
    text_lower = text.lower()
    if any(k in text_lower for k in ["thấp", "an toàn", "bảo thủ", "ít rủi ro"]): return "thấp"
    if any(k in text_lower for k in ["cao", "rủi ro cao", "tích cực", "mạo hiểm"]): return "cao"
    return "trung bình"


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
    chart_metadata: dict
    start_date: str | None
    end_date: str | None
    interval: str
    news_metadata: dict



def classify_node(state: SupervisorState) -> SupervisorState:
    """Classify user intent using fast LLM with Manual JSON Parsing."""
    question = state["question"]
    state["ticker"] = extract_ticker(question)
    state["risk_profile"] = extract_risk_profile(question)

    json_instructions = """
BẮT BUỘC TRẢ VỀ ĐỊNH DẠNG JSON. KHÔNG VIẾT THÊM BẤT KỲ CHỮ NÀO KHÁC TRƯỚC HAY SAU JSON.
Cấu trúc JSON bắt buộc:
{
  "selected_agents": ["stock_info", "technical"] 
}
"""
    prompt = CLASSIFY_SYSTEM + json_instructions + f"\n\nCâu hỏi: {question}"
    llm = get_fast_llm(model_name=settings.nvidia_router_model)
    
    try:
        raw_output = llm.invoke([HumanMessage(content=prompt)]).content.strip()
        
        if raw_output.startswith("```json"):
            raw_output = raw_output[7:]
        if raw_output.endswith("```"):
            raw_output = raw_output[:-3]
        raw_output = raw_output.strip()

        parsed_json = json.loads(raw_output)
        
        raw_intents = parsed_json.get("selected_agents", [])
        if isinstance(raw_intents, list):
            intents = [i.strip() for i in raw_intents if i.strip() in INTENTS]
        else:
            intents = ["stock_info"]
            
    except Exception as e:
        logger.warning(f"Intent classification failed: {e}. Raw Output: {raw_output}")
        intents = ["stock_info"]

    if not intents:
        intents = ["stock_info"]

    if "advisor" in intents:
        for needed in ["stock_info", "technical", "sentiment"]:
            if needed not in intents: intents.append(needed)

    state["intents"] = intents
    logger.info(f"Intents: {intents} | Ticker: {state['ticker']}")
    return state


rate_limiter = AsyncLimiter(max_rate=settings.llm_rate_limit, time_period=1)

async def dispatch_node(state: SupervisorState) -> SupervisorState:
    """Dispatch to sub-agents in PARALLEL with Async/Await and Rate Limiting."""
    import json
    
    intents = state["intents"]
    question = state["question"]
    ticker = state["ticker"]
    ticker_prefix = f"[{ticker}] " if ticker else ""

    time_context = ""
    if state.get("start_date") and state.get("end_date"):
        interval = state.get("interval", "1D")
        time_context = f"\n[LƯU Ý: Chỉ lấy dữ liệu, tin tức và phân tích trong khoảng thời gian từ {state['start_date']} đến {state['end_date']}, khung {interval}]"
    
    full_query = f"{ticker_prefix}{question}{time_context}"

    tasks: dict[str, tuple] = {}
    if "stock_info" in intents or "market" in intents:
        tasks["stock_info"] = (run_stock_info_agent, full_query)
    if "technical" in intents:
        tasks["technical"] = (run_technical_agent, full_query)
    if "sentiment" in intents:
        tasks["sentiment"] = (run_sentiment_agent, full_query)
    if "report_rag" in intents:
        tasks["report_rag"] = (run_report_rag_agent, full_query)

    results: dict[str, str] = {
        "stock_info": "", "technical": "", "sentiment": "", "report_rag": ""
    }

    async def _run_bounded(key: str, fn, q: str):
        async with rate_limiter:
            try:
                logger.info(f"Chạy Agent Song Song: {key}...")
                output = await asyncio.to_thread(fn, q)
                return key, output
            except Exception as e:
                logger.error(f"Agent {key} error: {e}")
                return key, f"[Thông báo: Hiện không thể lấy dữ liệu phần này do hệ thống bận. Lỗi: {e}]"

    coroutines = [_run_bounded(key, fn, q_task) for key, (fn, q_task) in tasks.items()]
    completed_results = await asyncio.gather(*coroutines)

    for key, output in completed_results:
        results[key] = output

    state["stock_info_result"] = results["stock_info"]
    state["technical_result"] = results["technical"]
    state["report_rag_result"] = results["report_rag"]
    
    sentiment_raw = results["sentiment"]
    try:
        if sentiment_raw and sentiment_raw.strip().startswith("{"):
            data = json.loads(sentiment_raw)
            state["sentiment_result"] = data.get("markdown_report", "")
            state["news_metadata"] = data.get("raw_data", None)
        else:
            state["sentiment_result"] = sentiment_raw
            state["news_metadata"] = None
    except Exception as e:
        logger.warning(f"Lỗi parse JSON Sentiment trong dispatch_node: {e}")
        state["sentiment_result"] = sentiment_raw
        state["news_metadata"] = None
    
    return state


async def synthesize_node(state: SupervisorState) -> SupervisorState:
    """Synthesize final answer asynchronously."""
    intents = state["intents"]
    ticker = state["ticker"]

    if ticker and any(i in intents for i in ["stock_info", "technical", "advisor"]):
        state["chart_metadata"] = {
            "ticker": ticker.upper(),
            "render_chart": True,
            "default_indicator": "sma" if "technical" in intents else "price",
            "default_period": "6m",
            "start_date": state.get("start_date"), 
            "end_date": state.get("end_date"),
            "interval": state.get("interval", "1D")
        }
    else:
        state["chart_metadata"] = {"render_chart": False}

    if "advisor" in intents and state["ticker"]:
        answer = await asyncio.to_thread(
            synthesize_investment_advice,
            ticker=state["ticker"],
            stock_info=state["stock_info_result"],
            technical_analysis=state["technical_result"],
            sentiment=state["sentiment_result"],
            report_summary=state["report_rag_result"],
            risk_profile=state["risk_profile"],
        )
        state["final_answer"] = answer
        return state

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

    parts = []
    if state["stock_info_result"]: parts.append(f"**Thông tin cổ phiếu:**\n{state['stock_info_result']}")
    if state["technical_result"]: parts.append(f"**Phân tích kỹ thuật:**\n{state['technical_result']}")
    if state["sentiment_result"]: parts.append(f"**Tin tức & Sentiment:**\n{state['sentiment_result']}")
    if state["report_rag_result"]: parts.append(f"**Báo cáo tài chính:**\n{state['report_rag_result']}")

    combined = "\n\n".join(parts)
    synthesis_prompt = (
        f"Câu hỏi người dùng: {state['question']}\n\n"
        f"Thông tin thu thập được:\n{combined}\n\n"
        "Hãy tổng hợp thành câu trả lời hoàn chỉnh, ngắn gọn bằng tiếng Việt."
    )
    
    try:
        llm = get_llm(temperature=0.1, model_name=settings.nvidia_advisor_model)
        chain = llm | StrOutputParser()
        state["final_answer"] = await chain.ainvoke([HumanMessage(content=synthesis_prompt)])
    except Exception as e:
        logger.error(f"Lỗi tổng hợp dữ liệu tại Synthesize Node: {e}")
        state["final_answer"] = "Hệ thống tổng hợp ngôn ngữ đang quá tải. Dưới đây là thông tin thô thu thập được:\n\n" + combined
    
    return state



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

async def run_supervisor(question: str, start_date: str = None, end_date: str = None, interval: str = "1D") -> dict:
    supervisor = get_supervisor()
    initial_state: SupervisorState = {
        "question": question,
        "ticker": None,
        "risk_profile": "trung bình",
        "start_date": start_date,
        "end_date": end_date,
        "interval": interval,
        "intents": [],
        "stock_info_result": "",
        "technical_result": "",
        "sentiment_result": "",
        "report_rag_result": "",
        "final_answer": "",
        "chart_metadata": {}, 
    }
    
    try:
        result = await supervisor.ainvoke(initial_state)
        return {
            "answer": result.get("final_answer", "Xin lỗi, tôi không thể xử lý câu hỏi này."),
            "chart_metadata": result.get("chart_metadata", {"render_chart": False}),
            "news_metadata": result.get("news_metadata")
        }
    except Exception as e:
        logger.error(f"Fatal lỗi ở Supervisor: {e}")
        return {
            "answer": f"Hệ thống điều phối trung tâm hiện đang gặp sự cố. Vui lòng thử lại sau. Lỗi: {str(e)}",
            "chart_metadata": {"render_chart": False}
        }