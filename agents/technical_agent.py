"""
agents/technical_agent.py
Agent phân tích kỹ thuật: SMA, RSI, MACD, Bollinger Bands.
"""
from langchain.agents import create_agent # Sử dụng phương thức tạo agent mới
from loguru import logger # THÊM MỚI: Import thư viện ghi log

from core.llm import get_llm
from config.settings import settings
from tools.technical_tools import (
    calculate_sma,
    calculate_rsi,
    calculate_macd,
    calculate_bollinger_bands,
)

TECHNICAL_SYSTEM = """Bạn là chuyên gia phân tích kỹ thuật chứng khoán Việt Nam.
Nhiệm vụ: Tính toán và diễn giải các chỉ số kỹ thuật cho cổ phiếu.

Chỉ số bạn có thể tính:
- SMA (Simple Moving Average): Trung bình động đơn giản
- RSI (Relative Strength Index): Chỉ số sức mạnh tương đối
- MACD: Moving Average Convergence Divergence  
- Bollinger Bands: Dải Bollinger

Quy tắc diễn giải:
- RSI > 70: Quá mua → tín hiệu bán
- RSI < 30: Quá bán → tín hiệu mua
- Giá > SMA: Xu hướng tăng
- Giá < SMA: Xu hướng giảm
- MACD cắt lên Signal: Tín hiệu mua
- MACD cắt xuống Signal: Tín hiệu bán

Quy tắc:
1. Dùng window_size mà người dùng yêu cầu, nếu không có dùng mặc định.
2. Diễn giải tín hiệu bằng ngôn ngữ dễ hiểu.
3. Kết hợp nhiều chỉ số để đưa ra đánh giá toàn diện.
4. KHÔNG đưa ra khuyến nghị mua/bán cuối cùng, chỉ phân tích kỹ thuật.

Trả lời bằng tiếng Việt."""

# Danh sách các công cụ tính toán kỹ thuật
_TOOLS = [calculate_sma, calculate_rsi, calculate_macd, calculate_bollinger_bands]

def create_technical_agent():
    """Khởi tạo Technical Agent bằng hàm create_agent mới."""
    llm = get_llm(temperature=0.0, model_name=settings.nvidia_agent_model)
    
    # create_agent thay thế cho create_tool_calling_agent và AgentExecutor
    return create_agent(
        model=llm,
        tools=_TOOLS,
        system_prompt=TECHNICAL_SYSTEM
    )

# Singleton Pattern để tránh khởi tạo lại nhiều lần
_agent = None

def get_technical_agent():
    global _agent
    if _agent is None:
        _agent = create_technical_agent()
    return _agent

def run_technical_agent(query: str) -> str:
    """Thực thi Technical Agent với đầu vào ngôn ngữ tự nhiên."""
    # THÊM MỚI: Bọc toàn bộ quá trình tính toán trong try...except
    try:
        logger.debug(f"Đang thực thi Technical Agent phân tích kỹ thuật: '{query}'")
        agent = get_technical_agent()
        
        # Cấu trúc input mới theo định dạng tin nhắn (messages list)
        result = agent.invoke({
            "messages": [("user", query)]
        })
        
        logger.success("Phân tích kỹ thuật thành công.")
        # Trích xuất nội dung phản hồi từ tin nhắn cuối cùng trong chuỗi hội thoại
        return result["messages"][-1].content
        
    except Exception as e:
        # Ghi log lỗi chi tiết ra console
        logger.error(f"Lỗi khi chạy Technical Agent: {str(e)}")
        
        # Cơ chế dự phòng (Graceful fallback)
        return (
            f"⚠️ **Thông tin Kỹ thuật:** Hệ thống tính toán chỉ số kỹ thuật "
            f"hiện đang bị gián đoạn do quá tải xử lý. Tạm thời không thể đưa ra nhận định xu hướng giá lúc này.\n"
            f"*(Chi tiết lỗi: {str(e)})*"
        )