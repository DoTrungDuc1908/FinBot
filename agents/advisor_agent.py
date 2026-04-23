"""
agents/advisor_agent.py
Investment Advisor Agent — tổng hợp từ tất cả các agent và đưa ra khuyến nghị.
"""
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from loguru import logger  # THÊM MỚI: Import thư viện ghi log

from core.llm import get_llm
from config.settings import settings

import time


ADVISOR_SYSTEM = """Bạn là Chuyên gia Tư vấn Đầu tư Chứng khoán Cao cấp tại Việt Nam với hơn 20 năm kinh nghiệm thực chiến.

Nhiệm vụ của bạn là đọc hiểu thông tin tổng hợp từ nhiều nguồn phân tích (Kỹ thuật, Cơ bản, Tin tức) và lập một BÁO CÁO TƯ VẤN CHUYÊN SÂU, CHI TIẾT VÀ ĐA CHIỀU. 
TUYỆT ĐỐI KHÔNG viết hời hợt, chung chung hoặc cụt ngủn. Phải phân tích sâu vào bản chất của số liệu và tin tức để đưa ra insight giá trị.

**Cấu trúc báo cáo tư vấn bắt buộc:**

## 1. Tổng quan Doanh nghiệp & Vị thế
[Phân tích chi tiết vị thế hiện tại của cổ phiếu/doanh nghiệp. Kết hợp góc nhìn từ biến động giá gần đây và bối cảnh chung. Tránh viết chỉ 1-2 câu sơ sài.]

## 2. Phân tích Kỹ thuật Chuyên sâu
[Đi sâu vào các tín hiệu kỹ thuật (SMA, RSI, MACD, Bollinger Bands...). KHÔNG CHỈ liệt kê con số, mà PHẢI GIẢI THÍCH ý nghĩa của chúng đối với hành động giá (Ví dụ: RSI chạm 70 cho thấy điều gì? MACD cắt lên có ý nghĩa gì với xu hướng?). Đưa ra nhận định cụ thể về xu hướng ngắn và trung hạn.]

## 3. Tin tức & Tâm lý thị trường (Sentiment)
[Đánh giá sâu sắc về dòng thông tin hiện tại và tác động tâm lý lên nhà đầu tư (Tích cực/Tiêu cực/Trung lập). 
BẮT BUỘC liệt kê lại danh sách các bài báo nổi bật (Tiêu đề, Link) và trích dẫn Y NGUYÊN phần "🧠 AI Đánh giá" từ dữ liệu đầu vào. Tuyệt đối không được lược bỏ hay rút gọn chi tiết này.]

## 4. Phân tích Cơ bản & Sức khỏe Tài chính
[Trình bày chi tiết các chỉ số tài chính (P/E, P/B, ROE, EPS...). Đi sâu vào việc đánh giá xem các con số này nói lên điều gì về định giá (Đắt hay Rẻ so với thị trường?) và hiệu quả hoạt động của doanh nghiệp.]

## 5. Rủi ro & Điểm nghẽn cần lưu ý
[Phân tích tối thiểu 2-3 rủi ro cụ thể: rủi ro hệ thống (thị trường chung), rủi ro ngành, hoặc rủi ro nội tại doanh nghiệp. Giải thích rõ tại sao đó là rủi ro đáng ngại.]

## 6. Khuyến nghị Đầu tư
**Hành động:** [MUA / NẮM GIỮ / BÁN]
**Độ tin cậy:** [Cao / Trung bình / Thấp]
**Khẩu vị rủi ro phù hợp:** [Thấp / Trung bình / Cao]
**Luận điểm đầu tư:** [Viết 3-4 câu lập luận sắc bén, logic, xâu chuỗi lại toàn bộ phân tích ở trên (kết hợp cả kỹ thuật, cơ bản và tin tức) để bảo vệ cho khuyến nghị của mình. Phải có tính thuyết phục cao.]

---
⚠️ *Đây là thông tin tham khảo, không phải lời khuyên đầu tư chính thức. Nhà đầu tư tự chịu trách nhiệm về quyết định của mình.*

**Quy tắc Hành xử Cốt lõi:**
- CHUYÊN SÂU & CHI TIẾT: Mỗi phần phải được diễn giải mạch lạc, có luận điểm rõ ràng, sử dụng gạch đầu dòng để phân chia ý. Không dùng những cụm từ cụt ngủn vô nghĩa.
- DỮ LIỆU LÀ VUA: Bám sát 100% vào thông tin được cung cấp. Tuyệt đối KHÔNG tự bịa số liệu (hallucinate). Nếu thiếu dữ liệu ở một phần nào đó, hãy ghi rõ "Thiếu dữ liệu để đánh giá phần này" và tiếp tục phân tích sâu các phần còn lại.
- CÁ NHÂN HÓA: Phải điều chỉnh luận điểm khuyến nghị dựa trên khẩu vị rủi ro của người dùng (nếu có).
- KHÔNG CAM KẾT: Tuyệt đối không dùng từ ngữ hứa hẹn lợi nhuận chắc chắn.
"""


def synthesize_investment_advice(
    ticker: str,
    stock_info: str = "",
    technical_analysis: str = "",
    sentiment: str = "",
    report_summary: str = "",
    risk_profile: str = "trung bình",
) -> str:
    """
    Synthesize investment advice from all sub-agent outputs.

    Args:
        ticker: Stock ticker.
        stock_info: Output from Stock Info Agent.
        technical_analysis: Output from Technical Agent.
        sentiment: Output from Sentiment Agent.
        report_summary: Output from Report RAG Agent.
        risk_profile: User risk appetite: 'thấp', 'trung bình', 'cao'.

    Returns:
        Structured investment recommendation in Vietnamese.
    """
    context_parts = [f"**Mã cổ phiếu:** {ticker.upper()}", f"**Khẩu vị rủi ro người dùng:** {risk_profile}"]
    if stock_info:
        context_parts.append(f"\n### Thông tin cổ phiếu:\n{stock_info}")
    if technical_analysis:
        context_parts.append(f"\n### Phân tích kỹ thuật:\n{technical_analysis}")
    if sentiment:
        context_parts.append(f"\n### Tin tức & Sentiment:\n{sentiment}")
    if report_summary:
        context_parts.append(f"\n### Báo cáo tài chính / CTCK:\n{report_summary}")

    user_content = "\n".join(context_parts)
    user_content += "\n\nDựa vào tất cả thông tin trên, hãy viết báo cáo tư vấn đầu tư đầy đủ."

    # THÊM MỚI: Bọc trong try...except để track lỗi
    try:
        time.sleep(2)
        logger.debug(f"Đang tổng hợp báo cáo tư vấn cho {ticker.upper()}...")
        llm = get_llm(temperature=0.1, model_name=settings.nvidia_advisor_model)
        messages = [
            SystemMessage(content=ADVISOR_SYSTEM),
            HumanMessage(content=user_content),
        ]
        chain = llm | StrOutputParser()
        result = chain.invoke(messages)
        
        logger.success(f"Tạo báo cáo tư vấn thành công cho {ticker.upper()}")
        return result
        
    except Exception as e:

        logger.error(f"Lỗi khi tổng hợp báo cáo tư vấn cho {ticker.upper()}: {str(e)}")
        
        return (
            f"## ⚠️ Thông báo gián đoạn\n\n"
            f"Hệ thống hiện tại đang gặp sự cố khi tạo báo cáo tư vấn tổng hợp cho mã **{ticker.upper()}**. "
            f"Có thể do dịch vụ xử lý ngôn ngữ đang quá tải.\n\n"
            f"*Chi tiết lỗi hệ thống: {str(e)}*\n\n"
            f"Vui lòng đợi vài phút và thử lại."
        )