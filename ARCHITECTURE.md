# FinBot – Kiến Trúc Hệ Thống Multi-Agent Tài Chính Việt Nam

> **Phiên bản:** 1.0 | **Cập nhật:** 2026-04-18  
> **Mục tiêu:** Hệ thống AI tư vấn đầu tư thị trường Việt Nam — nhanh, tiết kiệm token, chính xác cao.

---

## 1. Tổng Quan Kiến Trúc

```
┌──────────────────────────────────────────────────────────────────┐
│                         USER INTERFACE                           │
│              (Web Chat / REST API / Telegram Bot)                │
└─────────────────────────────┬────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│                    SUPERVISOR AGENT (Orchestrator)               │
│   - Intent Classification  - Task Routing  - Response Synthesis  │
└───┬──────────┬──────────┬──────────┬──────────┬─────────────────┘
    │          │          │          │          │
    ▼          ▼          ▼          ▼          ▼
┌───────┐ ┌───────┐ ┌───────┐ ┌───────┐ ┌──────────┐
│Stock  │ │Tech   │ │News & │ │Report │ │Investment│
│Info   │ │Analysis│ │Senti- │ │RAG    │ │Advisor   │
│Agent  │ │Agent  │ │ment   │ │Agent  │ │Agent     │
│       │ │       │ │Agent  │ │       │ │          │
└───┬───┘ └───┬───┘ └───┬───┘ └───┬───┘ └────┬─────┘
    │          │          │          │          │
    └──────────┴──────────┴──────────┴──────────┘
                              │
                    ┌─────────▼─────────┐
                    │    DATA LAYER     │
                    │  (APIs, DBs, RAG) │
                    └───────────────────┘
```

---

## 2. Các Thành Phần Chính

### 2.1 Supervisor Agent (Điều Phối Trung Tâm)

**Vai trò:** Nhận câu hỏi từ người dùng, phân loại ý định, định tuyến đến các sub-agent phù hợp, tổng hợp và trả kết quả.

**Cơ chế hoạt động:**
- Dùng **LLM nhỏ / prompt classification** để phân loại intent (tránh gọi LLM lớn cho routing).
- Hỗ trợ **parallel dispatch**: khi câu hỏi yêu cầu nhiều agent, gọi song song để tiết kiệm thời gian.
- Áp dụng **ReAct / Plan-and-Execute** pattern để xử lý câu hỏi phức tạp nhiều bước.

**Công nghệ:** LangGraph / LangChain Supervisor | LLM: Llama 3.1 (NVIDIA NIM)

---

### 2.2 Stock Info Agent

**Chức năng:**
- Tra cứu thông tin doanh nghiệp theo mã chứng khoán (ticker): tên, ngành, vốn hóa, ban lãnh đạo, sàn niêm yết.
- Truy xuất dữ liệu giá lịch sử: OHLCV (Open, High, Low, Close, Volume).
- Lọc theo ngày cụ thể hoặc khung thời gian (`3 tháng gần nhất`, `2023-01-01 đến 2023-06-30`).

**Công cụ (Tools):**
| Tool | Mô tả | Nguồn dữ liệu |
|------|-------|---------------|
| `get_company_info(ticker)` | Thông tin doanh nghiệp | TCBS API / SSI API |
| `get_price_history(ticker, start, end)` | Dữ liệu OHLCV | vnstock / HOSE/HNX feed |
| `get_price_by_timeframe(ticker, period)` | Lọc theo khung thời gian | vnstock |

**Tối ưu:** Cache Redis (TTL 5 phút cho giá realtime, 24h cho thông tin công ty).

---

### 2.3 Technical Analysis Agent

**Chức năng:**
- Tính toán các chỉ số kỹ thuật theo `window_size` do người dùng yêu cầu.
- Hỗ trợ: SMA, EMA, RSI, MACD, Bollinger Bands.
- Diễn giải tín hiệu kỹ thuật (vùng mua/bán, xu hướng).

**Công cụ:**
| Tool | Mô tả |
|------|-------|
| `calculate_sma(ticker, window, period)` | Simple Moving Average |
| `calculate_rsi(ticker, window, period)` | Relative Strength Index |
| `calculate_macd(ticker, period)` | MACD + Signal Line |
| `interpret_signals(indicators)` | Diễn giải tín hiệu tổng hợp |

**Công nghệ:** pandas-ta / ta-lib | Tính toán local, không cần LLM → **tiết kiệm token tối đa**.

---

### 2.4 News & Sentiment Agent

**Chức năng:**
- Thu thập tin tức mới nhất về mã cổ phiếu, ngành, thị trường chung.
- Phân tích sentiment: Tích cực / Tiêu cực / Trung lập.
- Đánh giá sentiment theo nhóm cổ phiếu (VN30, ngân hàng, bất động sản, ...).

**Công cụ:**
| Tool | Mô tả | Nguồn |
|------|-------|-------|
| `fetch_stock_news(ticker, limit)` | Tin tức theo mã | CafeF, VnEconomy, TCBS News |
| `analyze_sentiment(texts)` | Phân tích sentiment | PhoBERT / FinBERT-Vietnamese |
| `get_market_sentiment(group)` | Sentiment theo nhóm | Tổng hợp nhiều nguồn |

**Tối ưu:** Dùng model embedding nhỏ (PhoBERT) để phân tích sentiment, không gọi LLM lớn.

---

### 2.5 Report RAG Agent

**Chức năng:**
- Truy cập và tổng hợp thông tin từ:
  - Báo cáo tài chính (BCTC quý, năm): doanh thu, lợi nhuận, EPS, P/E.
  - Báo cáo phân tích của công ty chứng khoán (SSI, VCSC, MBS, ...).
- Trả lời câu hỏi cụ thể từ tài liệu (RAG pipeline).

**Kiến trúc RAG:**
```
[PDF / HTML Reports]
        │
        ▼
[Document Chunking & Embedding]  ←  NVIDIA Embedding / BGE-M3
        │
        ▼
[Vector Store]  ←  ChromaDB / Qdrant
        │
   User Query ──► [Retriever] ──► Top-K Chunks
                                        │
                                        ▼
                              [LLM Synthesis]  ←  Llama 3.1
```

**Công cụ:**
| Tool | Mô tả |
|------|-------|
| `search_financial_reports(ticker, query)` | Tìm kiếm BCTC |
| `search_analyst_reports(ticker, query)` | Tìm báo cáo phân tích |
| `summarize_financials(ticker, period)` | Tóm tắt chỉ số tài chính |

---

### 2.6 Investment Advisor Agent

**Chức năng:**
- Tổng hợp output từ tất cả các agent khác.
- Đưa ra khuyến nghị đầu tư: Mua / Nắm giữ / Bán.
- Cá nhân hóa tư vấn theo khẩu vị rủi ro của người dùng (thấp / trung bình / cao).
- Cảnh báo rủi ro, đa dạng hóa danh mục.

**Input:** Kết quả từ Stock Info + Technical Analysis + Sentiment + RAG.  
**Output:** Báo cáo tư vấn có cấu trúc với luận điểm rõ ràng.

---

## 3. Data Layer

### 3.1 Nguồn Dữ Liệu

| Loại | Nguồn | Giao thức |
|------|-------|-----------|
| Giá cổ phiếu realtime | TCBS, SSI, HNX | REST API |
| Dữ liệu lịch sử | vnstock library | Python SDK |
| Tin tức | CafeF, VnEconomy, VietStock | RSS / Web Scraping |
| Báo cáo tài chính | HOSE/HNX disclosure, Vietstock | PDF / HTML |
| Báo cáo CTCK | SSI Research, VCSC, MBS | PDF ingestion |

### 3.2 Cache & Storage

```
┌─────────────────────────────────────────────┐
│              STORAGE LAYER                  │
├──────────────┬──────────────┬───────────────┤
│  Redis Cache │  PostgreSQL  │  Vector Store │
│  (Realtime)  │  (History)   │  (ChromaDB)   │
│  TTL: 5m-24h │  OHLCV Data  │  Reports/News │
└──────────────┴──────────────┴───────────────┘
```

---

## 4. Luồng Xử Lý Câu Hỏi (Request Flow)

### Ví dụ: *"Phân tích kỹ thuật VNM với RSI 14 ngày và cho biết tin tức gần đây"*

```
1. User gửi câu hỏi
         │
2. Supervisor phân loại intent
   → Cần: [TechnicalAnalysis, NewsSentiment]
         │
3. Dispatch SONG SONG:
   ├── Technical Agent: RSI(VNM, window=14)
   └── News Agent: fetch_news(VNM) + sentiment()
         │
4. Thu thập kết quả từ cả 2 agent
         │
5. Investment Advisor tổng hợp
         │
6. Trả về câu trả lời có cấu trúc
```

**Thời gian ước tính:** < 5 giây (nhờ parallel dispatch + cache).

---

## 5. Chiến Lược Tối Ưu

### 5.1 Tiết Kiệm Token
| Chiến lược | Mô tả |
|-----------|-------|
| **Local computation** | Tính SMA/RSI bằng pandas, không gọi LLM |
| **Structured output** | Agent trả JSON, Supervisor không cần parse văn bản |
| **Prompt compression** | Chỉ truyền dữ liệu cần thiết vào context LLM |
| **Small model routing** | Dùng model nhỏ/regex để classify intent |
| **Sentiment model local** | PhoBERT chạy local, không gọi LLM API |

### 5.2 Tăng Tốc Độ
| Chiến lược | Mô tả |
|-----------|-------|
| **Parallel agent calls** | Gọi nhiều agent đồng thời (asyncio) |
| **Redis caching** | Cache giá, tin tức, kết quả tính toán |
| **Streaming response** | Stream token LLM về UI ngay khi có |
| **Pre-computed indicators** | Tính sẵn SMA/RSI cho các window phổ biến |

### 5.3 Độ Chính Xác
| Chiến lược | Mô tả |
|-----------|-------|
| **RAG với reranking** | Dùng cross-encoder reranker cho retrieval |
| **Source citation** | Luôn trích dẫn nguồn dữ liệu |
| **Hallucination guard** | Kiểm tra output LLM với dữ liệu thực tế |
| **Confidence scoring** | Gắn độ tin cậy cho mỗi khuyến nghị |

---

## 6. Tech Stack

| Thành phần | Công nghệ |
|-----------|-----------|
| **LLM** | Llama 3.1 70B (NVIDIA NIM) |
| **Embedding** | NVIDIA NV-Embed / BGE-M3 |
| **Sentiment** | PhoBERT-base (Vietnamese) |
| **Agent Framework** | LangGraph |
| **Vector DB** | ChromaDB / Qdrant |
| **Cache** | Redis |
| **Database** | PostgreSQL |
| **Stock Data** | vnstock, TCBS API, SSI API |
| **API Server** | FastAPI + Uvicorn |
| **Frontend** | React / Streamlit |
| **Container** | Docker Compose |

---

## 7. Cấu Trúc Thư Mục Dự Án

```
FinBot/
├── agents/
│   ├── supervisor.py          # Orchestrator chính
│   ├── stock_info_agent.py    # Stock info & price history
│   ├── technical_agent.py     # SMA, RSI, MACD
│   ├── sentiment_agent.py     # News & sentiment
│   ├── report_rag_agent.py    # RAG báo cáo tài chính
│   └── advisor_agent.py       # Investment advisor
├── tools/
│   ├── stock_tools.py         # vnstock wrappers
│   ├── technical_tools.py     # pandas-ta indicators
│   ├── news_tools.py          # News scrapers
│   └── rag_tools.py           # Vector search tools
├── data/
│   ├── ingestion/             # ETL pipelines
│   └── reports/               # PDF báo cáo
├── core/
│   ├── llm.py                 # LLM client (NVIDIA NIM)
│   ├── embeddings.py          # Embedding client
│   ├── cache.py               # Redis cache
│   └── vector_store.py        # ChromaDB client
├── api/
│   └── app.py                 # FastAPI server
├── config/
│   └── settings.py            # Cấu hình hệ thống
├── ARCHITECTURE.md            # File này
└── docker-compose.yml
```

---

## 8. Lộ Trình Phát Triển

| Giai đoạn | Nội dung | Thời gian |
|-----------|---------|-----------|
| **Phase 1** | Stock Info Agent + Technical Agent + API cơ bản | 1-2 tuần |
| **Phase 2** | News Sentiment Agent + Redis Cache | 1-2 tuần |
| **Phase 3** | Report RAG Agent + Vector DB | 2-3 tuần |
| **Phase 4** | Investment Advisor + Supervisor hoàn chỉnh | 1-2 tuần |
| **Phase 5** | Tối ưu hiệu năng + UI + Production deploy | 1-2 tuần |

---

## 9. Rủi Ro & Giới Hạn

| Rủi Ro | Biện Pháp |
|--------|-----------|
| API nguồn dữ liệu bị giới hạn rate | Implement retry + multiple providers |
| Dữ liệu báo cáo không cập nhật | Lịch cron job tự động crawl |
| LLM hallucination trong tư vấn | Disclaimer + source citation bắt buộc |
| Chi phí LLM cao khi traffic lớn | Cache aggressive + dùng model nhỏ khi có thể |

---

> ⚠️ **Disclaimer:** Hệ thống cung cấp thông tin tham khảo, không phải lời khuyên đầu tư chính thức. Nhà đầu tư tự chịu trách nhiệm về quyết định của mình.
