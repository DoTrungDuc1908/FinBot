from langchain.agents import create_agent
from loguru import logger

from core.llm import get_llm
from config.settings import settings
from tools.rag_tools import search_financial_reports, search_analyst_reports, list_available_reports

from tools.stock_tools import get_financial_fundamentals 

RAG_SYSTEM = """Bạn là chuyên gia phân tích báo cáo tài chính và báo cáo CTCK Việt Nam.

Quy trình làm việc BẮT BUỘC:
1. LUÔN sử dụng công cụ `get_financial_fundamentals` ĐẦU TIÊN để lấy các chỉ số nền tảng (P/E, ROE, EPS...).
2. NẾU BƯỚC 1 KHÔNG CÓ DỮ LIỆU (hoặc dữ liệu rỗng), BẠN BẮT BUỘC phải sử dụng `search_financial_reports` để tìm kiếm thông tin "Báo cáo Kết quả Kinh doanh", "Doanh thu", "Lợi nhuận" từ cơ sở dữ liệu nội bộ (RAG).
3. Đọc hiểu các số liệu hàng/cột trong Markdown và đưa ra đánh giá: Doanh nghiệp đang tăng trưởng hay đi lùi? Lợi nhuận qua các năm biến động ra sao?

Quy tắc:
- Trích dẫn nguồn rõ ràng (từ API hoặc từ tên file báo cáo RAG).
- Phân tích sâu sắc sự thay đổi của các con số, không chỉ liệt kê.
- Trình bày kết quả dưới dạng thẻ Markdown, tô đậm các con số quan trọng.

Trả lời bằng tiếng Việt, logic, mạch lạc."""

_TOOLS = [
    get_financial_fundamentals, 
    search_financial_reports, 
    search_analyst_reports, 
    list_available_reports
]

def create_report_rag_agent():
    llm = get_llm(temperature=0.1, model_name=settings.nvidia_advisor_model)
    return create_agent(
        model=llm,
        tools=_TOOLS,
        system_prompt=RAG_SYSTEM
    )

_agent = None

def get_report_rag_agent():
    global _agent
    if _agent is None:
        _agent = create_report_rag_agent()
    return _agent

def run_report_rag_agent(query: str) -> str:
    try:
        logger.debug(f"Đang thực thi RAG truy xuất báo cáo tài chính: '{query}'")
        agent = get_report_rag_agent()
        result = agent.invoke({
            "messages": [("user", query)]
        })
        logger.success("Truy xuất dữ liệu báo cáo tài chính thành công.")
        return result["messages"][-1].content
    except Exception as e:
        logger.error(f"Lỗi khi chạy Report RAG Agent: {str(e)}")
        return f"**Thông báo:** Quá trình tra cứu dữ liệu cơ bản hiện đang bị gián đoạn. *(Lỗi: {str(e)})*"