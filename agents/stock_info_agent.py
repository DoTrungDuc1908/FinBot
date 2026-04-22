"""
agents/stock_info_agent.py
Agent chuyên tra cứu thông tin cổ phiếu và lịch sử giá.
"""
from langchain.agents import create_agent # Cập nhật từ template mới
from core.llm import get_llm
from tools.stock_tools import get_company_info, get_price_history, get_market_overview

STOCK_INFO_SYSTEM = """Bạn là chuyên gia tra cứu thông tin chứng khoán Việt Nam.
Nhiệm vụ của bạn là cung cấp thông tin chính xác và đầy đủ về:
- Thông tin doanh nghiệp (tên, ngành, sàn niêm yết, vốn hóa)
- Dữ liệu giá lịch sử OHLCV

Quy tắc:
1. Luôn sử dụng tools để lấy dữ liệu thực tế, không tự tạo số liệu.
2. Khi người dùng hỏi về khoảng thời gian, hãy xác định rõ start_date/end_date hoặc period.
3. Trình bày số liệu rõ ràng, có đơn vị (VNĐ, tỷ đồng, cổ phiếu).
4. Nếu không tìm thấy dữ liệu, hãy thông báo rõ ràng.

Trả lời bằng tiếng Việt, ngắn gọn và chính xác."""

_TOOLS = [get_company_info, get_price_history, get_market_overview]

def create_stock_info_agent():
    """Khởi tạo Stock Info Agent bằng hàm create_agent mới."""
    llm = get_llm(temperature=0.0)
    
    # Sử dụng create_agent thay thế cho create_tool_calling_agent và AgentExecutor
    return create_agent(
        model=llm,
        tools=_TOOLS,
        system_prompt=STOCK_INFO_SYSTEM
    )

# Singleton
_agent = None

def get_stock_info_agent():
    global _agent
    if _agent is None:
        _agent = create_stock_info_agent()
    return _agent

def run_stock_info_agent(query: str) -> str:
    """Run the stock info agent with a natural language query."""
    agent = get_stock_info_agent()
    
    # Sử dụng cấu trúc invoke mới với danh sách messages
    result = agent.invoke({
        "messages": [("user", query)]
    })
    
    # Trích xuất nội dung phản hồi từ tin nhắn cuối cùng trong kết quả
    return result["messages"][-1].content