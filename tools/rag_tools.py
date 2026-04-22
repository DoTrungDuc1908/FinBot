"""
tools/rag_tools.py
LangChain tools for RAG-based retrieval of financial reports and analyst reports.
Uses ChromaDB vector store with NVIDIA embeddings.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

from langchain_core.documents import Document
from langchain_core.tools import tool
from loguru import logger

from config.settings import settings
from core.cache import cache, CacheClient
from core.vector_store import similarity_search, add_documents


# ── Document Ingestion ────────────────────────────────────────────────────────

def ingest_pdf(pdf_path: str, ticker: str, report_type: str = "financial") -> int:
    """
    Ingest a PDF report into the vector store.

    Args:
        pdf_path: Absolute path to the PDF file.
        ticker: Stock ticker this report belongs to.
        report_type: 'financial' or 'analyst'.

    Returns:
        Number of chunks ingested.
    """
    try:
        from pypdf import PdfReader  # type: ignore
        reader = PdfReader(pdf_path)
        full_text = ""
        for page in reader.pages:
            full_text += page.extract_text() or ""

        # Chunk text into ~500 word pieces with 50-word overlap
        words = full_text.split()
        chunk_size = 500
        overlap = 50
        chunks = []
        i = 0
        while i < len(words):
            chunk = " ".join(words[i: i + chunk_size])
            chunks.append(chunk)
            i += chunk_size - overlap

        docs = [
            Document(
                page_content=chunk,
                metadata={
                    "ticker": ticker.upper(),
                    "report_type": report_type,
                    "source": Path(pdf_path).name,
                    "chunk_index": idx,
                },
            )
            for idx, chunk in enumerate(chunks)
        ]
        collection = settings.chroma_collection_reports
        add_documents(collection, docs)
        logger.info(f"Ingested {len(docs)} chunks from {pdf_path} for {ticker}")
        return len(docs)
    except Exception as e:
        logger.error(f"PDF ingestion error: {e}")
        return 0


def ingest_text(text: str, ticker: str, source: str, report_type: str = "analyst") -> int:
    """Ingest raw text into the vector store."""
    words = text.split()
    chunk_size = 400
    overlap = 40
    chunks, i = [], 0
    while i < len(words):
        chunks.append(" ".join(words[i: i + chunk_size]))
        i += chunk_size - overlap

    docs = [
        Document(
            page_content=chunk,
            metadata={"ticker": ticker.upper(), "report_type": report_type, "source": source, "chunk_index": idx},
        )
        for idx, chunk in enumerate(chunks)
    ]
    add_documents(settings.chroma_collection_reports, docs)
    return len(docs)


# ── LangChain Tools ───────────────────────────────────────────────────────────

@tool
def search_financial_reports(ticker: str, query: str, k: int = 5) -> str:
    """
    Tìm kiếm thông tin trong báo cáo tài chính (BCTC) của một công ty.
    Sử dụng semantic search để tìm đoạn văn liên quan nhất đến câu hỏi.

    Args:
        ticker: Mã chứng khoán (VD: VNM, VIC).
        query: Câu hỏi hoặc từ khóa cần tìm (VD: 'doanh thu 2023', 'lợi nhuận ròng').
        k: Số đoạn văn trả về (mặc định 5).

    Returns:
        JSON danh sách các đoạn văn liên quan từ báo cáo tài chính.
    """
    cache_key = CacheClient.build_key("rag_fin", ticker.upper(), CacheClient.hash_key(query))
    if (hit := cache.get(cache_key)):
        return json.dumps(hit, ensure_ascii=False)

    try:
        docs = similarity_search(
            collection_name=settings.chroma_collection_reports,
            query=f"{ticker} {query}",
            k=k,
            filter={"ticker": ticker.upper(), "report_type": "financial"},
        )
        results = [
            {
                "content": doc.page_content,
                "source": doc.metadata.get("source", ""),
                "ticker": doc.metadata.get("ticker", ""),
                "chunk": doc.metadata.get("chunk_index", 0),
            }
            for doc in docs
        ]
        result = {"ticker": ticker.upper(), "query": query, "count": len(results), "results": results}
        cache.set(cache_key, result, ttl=3600)
        return json.dumps(result, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"RAG search error: {e}")
        return json.dumps({"error": str(e), "ticker": ticker, "query": query})


@tool
def search_analyst_reports(ticker: str, query: str, k: int = 5) -> str:
    """
    Tìm kiếm thông tin trong báo cáo phân tích của các công ty chứng khoán
    (SSI Research, VCSC, MBS, BSC, ...).

    Args:
        ticker: Mã chứng khoán (VD: VNM, VIC).
        query: Câu hỏi hoặc nội dung cần tìm (VD: 'khuyến nghị', 'giá mục tiêu').
        k: Số đoạn văn trả về (mặc định 5).

    Returns:
        JSON danh sách các đoạn văn từ báo cáo phân tích của CTCK.
    """
    cache_key = CacheClient.build_key("rag_ana", ticker.upper(), CacheClient.hash_key(query))
    if (hit := cache.get(cache_key)):
        return json.dumps(hit, ensure_ascii=False)

    try:
        docs = similarity_search(
            collection_name=settings.chroma_collection_reports,
            query=f"{ticker} {query}",
            k=k,
            filter={"ticker": ticker.upper(), "report_type": "analyst"},
        )
        results = [
            {
                "content": doc.page_content,
                "source": doc.metadata.get("source", ""),
                "ticker": doc.metadata.get("ticker", ""),
            }
            for doc in docs
        ]
        result = {"ticker": ticker.upper(), "query": query, "count": len(results), "results": results}
        cache.set(cache_key, result, ttl=3600)
        return json.dumps(result, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"Analyst RAG search error: {e}")
        return json.dumps({"error": str(e), "ticker": ticker, "query": query})


@tool
def list_available_reports(ticker: Optional[str] = None) -> str:
    """
    Liệt kê các báo cáo có sẵn trong hệ thống.

    Args:
        ticker: Lọc theo mã chứng khoán (tùy chọn). Nếu không có, trả về tất cả.

    Returns:
        JSON danh sách các báo cáo đã được nạp vào hệ thống.
    """
    try:
        filter_dict = {"ticker": ticker.upper()} if ticker else None
        docs = similarity_search(
            collection_name=settings.chroma_collection_reports,
            query="báo cáo tài chính phân tích",
            k=50,
            filter=filter_dict,
        )
        sources = list({
            f"{doc.metadata.get('source', '')} ({doc.metadata.get('report_type', '')})"
            for doc in docs
        })
        return json.dumps({"available_reports": sources, "count": len(sources)}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)})
