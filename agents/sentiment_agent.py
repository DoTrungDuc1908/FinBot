"""
agents/sentiment_agent.py
Agent phân tích sentiment tin tức và tâm lý thị trường.
"""
from langchain.agents import create_agent
from langchain_core.prompts import ChatPromptTemplate

from core.llm import get_llm
from tools.news_tools import fetch_stock_news, analyze_stock_sentiment, get_market_news

SENTIMENT_SYSTEM = """Bạn là chuyên gia phân tích tin tức và tâm lý thị trường chứng khoán Việt Nam.

Nhiệm vụ:
- Thu thập và phân tích tin tức liên quan đến mã cổ phiếu hoặc thị trường chung.
- Đánh giá sentiment (tâm lý): Tích cực / Tiêu cực / Trung lập.
- Tóm tắt các sự kiện, thông tin quan trọng ảnh hưởng đến cổ phiếu.

Quy tắc:
1. Luôn dùng tools để lấy tin tức thực tế.
2. Phân tích xem tin tức có tác động ngắn hạn hay dài hạn.
3. Chỉ ra các rủi ro hoặc cơ hội từ tin tức.
4. Trình bày tóm tắt ngắn gọn (3-5 điểm chính).

Trả lời bằng tiếng Việt, khách quan và có nguồn trích dẫn."""

# Danh sách các công cụ
_TOOLS = [fetch_stock_news, analyze_stock_sentiment, get_market_news]

def create_sentiment_agent():
    # Sử dụng temperature thấp để đảm bảo tính khách quan
    llm = get_llm(temperature=0.1)
    
    # create_react_agent từ langgraph là cách tiêu chuẩn mới 
    # thay thế cho AgentExecutor truyền thống
    return create_agent(
        model=llm,
        tools=_TOOLS,
        system_prompt=SENTIMENT_SYSTEM
    )
    return agent

_agent = None

def get_sentiment_agent():
    global _agent
    if _agent is None:
        _agent = create_sentiment_agent()
    return _agent

def run_sentiment_agent(query: str) -> str:
    agent = get_sentiment_agent()
    # Cách invoke của LangGraph hơi khác một chút
    inputs = {"messages": [("user", query)]}
    result = agent.invoke(inputs)
    
    # Lấy phản hồi cuối cùng từ danh sách tin nhắn
    return result["messages"][-1].content