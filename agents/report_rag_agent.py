"""
agents/report_rag_agent.py
Agent RAG tổng hợp báo cáo tài chính và báo cáo phân tích CTCK.
"""
from langchain.agents import create_agent # Sử dụng hàm tạo agent tiêu chuẩn mới
from core.llm import get_llm
from tools.rag_tools import search_financial_reports, search_analyst_reports, list_available_reports

RAG_SYSTEM = """Bạn là chuyên gia phân tích báo cáo tài chính và báo cáo CTCK Việt Nam.

Nguồn thông tin bạn có quyền truy cập:
- Báo cáo tài chính (BCTC): Doanh thu, lợi nhuận, tài sản, nợ, EPS, ROE, ROA, P/E.
- Báo cáo phân tích CTCK: SSI Research, VCSC, MBS, BSC — khuyến nghị, giá mục tiêu.

Quy tắc:
1. Luôn dùng tools để tìm kiếm thông tin, không tự tạo số liệu.
2. Trích dẫn nguồn rõ ràng (tên file báo cáo).
3. Nếu không tìm thấy dữ liệu, thông báo rõ ràng thay vì đoán mò.
4. So sánh các chỉ số với trung bình ngành nếu có dữ liệu.
5. Phân tích xu hướng theo thời gian nếu có nhiều kỳ dữ liệu.

Trả lời bằng tiếng Việt, có cấu trúc rõ ràng với các mục tiêu đề."""

# Danh sách các công cụ RAG
_TOOLS = [search_financial_reports, search_analyst_reports, list_available_reports]

def create_report_rag_agent():
    """Khởi tạo Report RAG Agent bằng hàm create_agent hiện đại."""
    # Giữ temperature thấp để đảm bảo dữ liệu trích xuất chính xác
    llm = get_llm(temperature=0.1)
    
    # create_agent xử lý logic tool-calling thay cho AgentExecutor
    return create_agent(
        model=llm,
        tools=_TOOLS,
        system_prompt=RAG_SYSTEM
    )

# Singleton Pattern
_agent = None

def get_report_rag_agent():
    global _agent
    if _agent is None:
        _agent = create_report_rag_agent()
    return _agent

def run_report_rag_agent(query: str) -> str:
    """Thực thi RAG agent để tra cứu báo cáo tài chính."""
    agent = get_report_rag_agent()
    
    # Cấu trúc input mới sử dụng "messages" thay vì "input"
    result = agent.invoke({
        "messages": [("user", query)]
    })
    
    # Trích xuất phản hồi cuối cùng từ danh sách messages trả về
    return result["messages"][-1].content