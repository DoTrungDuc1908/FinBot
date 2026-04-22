"""
agents/advisor_agent.py
Investment Advisor Agent — tổng hợp từ tất cả các agent và đưa ra khuyến nghị.
"""
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser

from core.llm import get_llm

ADVISOR_SYSTEM = """Bạn là chuyên gia tư vấn đầu tư chứng khoán tại Việt Nam với hơn 20 năm kinh nghiệm.

Bạn nhận được thông tin tổng hợp từ nhiều nguồn phân tích, hãy đưa ra đánh giá toàn diện.

**Cấu trúc báo cáo tư vấn:**

## 1. Tóm tắt đánh giá
[Một đoạn ngắn tổng kết về cổ phiếu]

## 2. Phân tích kỹ thuật
[Tóm tắt các tín hiệu kỹ thuật (SMA, RSI, MACD,...)]

## 3. Tin tức & Tâm lý thị trường  
[Sentiment, sự kiện quan trọng]

## 4. Phân tích cơ bản (nếu có dữ liệu)
[Chỉ số tài chính, đánh giá CTCK]

## 5. Rủi ro cần lưu ý
[Liệt kê các rủi ro chính]

## 6. Khuyến nghị
**Hành động:** [MUA / NẮM GIỮ / BÁN]
**Độ tin cậy:** [Cao / Trung bình / Thấp]
**Lý do:** [1-3 câu lý do chính]

---
⚠️ *Đây là thông tin tham khảo, không phải lời khuyên đầu tư chính thức. Nhà đầu tư tự chịu trách nhiệm về quyết định của mình.*

**Quy tắc:**
- Khách quan, dựa trên dữ liệu được cung cấp.
- Không hứa hẹn lợi nhuận chắc chắn.
- Luôn nhắc đến rủi ro.
- Cá nhân hóa theo khẩu vị rủi ro nếu được cung cấp."""


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

    llm = get_llm(temperature=0.2)
    messages = [
        SystemMessage(content=ADVISOR_SYSTEM),
        HumanMessage(content=user_content),
    ]
    chain = llm | StrOutputParser()
    return chain.invoke(messages)
