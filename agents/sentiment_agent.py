"""
agents/sentiment_agent.py
Agent phân tích sentiment tin tức và tâm lý thị trường.
"""
from langchain.agents import create_agent
from loguru import logger  # THÊM MỚI: Import thư viện ghi log

from core.llm import get_llm
from config.settings import settings
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
    llm = get_llm(temperature=0.0, model_name=settings.nvidia_agent_model)
    
    return create_agent(
        model=llm,
        tools=_TOOLS,
        system_prompt=SENTIMENT_SYSTEM
    )

_agent = None

def get_sentiment_agent():
    global _agent
    if _agent is None:
        _agent = create_sentiment_agent()
    return _agent

def run_sentiment_agent(query: str) -> str:
    """Thực thi Sentiment Agent để đánh giá tâm lý thị trường/cổ phiếu."""
    
    try:
        logger.debug(f"Đang thực thi Sentiment Agent phân tích tin tức: '{query}'")
        agent = get_sentiment_agent()
        
        inputs = {"messages": [("user", query)]}
        result = agent.invoke(inputs)
        
        logger.success("Phân tích sentiment tin tức thành công.")
        
        return result["messages"][-1].content
        
    except Exception as e:
        
        logger.error(f"Lỗi khi chạy Sentiment Agent: {str(e)}")
        
        return (
            f"**Lưu ý về Tin tức:** Hệ thống thu thập tin tức và phân tích tâm lý thị trường "
            f"hiện đang bị gián đoạn. Tạm thời không thể đánh giá tác động của truyền thông lên mã này.\n"
            f"*(Chi tiết lỗi: {str(e)})*"
        )