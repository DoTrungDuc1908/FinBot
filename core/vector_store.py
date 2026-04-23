"""
core/vector_store.py
ChromaDB vector store client.
Manages two collections: financial_reports and stock_news.
"""
from functools import lru_cache
from typing import List, Optional

import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document
from loguru import logger

from config.settings import settings
from core.embeddings import get_embeddings
import requests


@lru_cache(maxsize=1)
def _get_chroma_client():
    return chromadb.PersistentClient(path="./chroma_data")


def get_vector_store(collection_name: str) -> Chroma:
    """
    Return a LangChain Chroma vector store for the given collection.

    Args:
        collection_name: One of CHROMA_COLLECTION_REPORTS or CHROMA_COLLECTION_NEWS.
    """
    return Chroma(
        client=_get_chroma_client(),
        collection_name=collection_name,
        embedding_function=get_embeddings(),
    )


def get_reports_store() -> Chroma:
    return get_vector_store(settings.chroma_collection_reports)


def get_news_store() -> Chroma:
    return get_vector_store(settings.chroma_collection_news)





def _get_nvidia_embedding_direct(text: str) -> List[float]:
    """Gọi trực tiếp API NVIDIA để lấy embedding, tránh lỗi định dạng mảng."""
    url = f"{settings.nvidia_base_url}/embeddings"
    headers = {
        "Authorization": f"Bearer {settings.nvidia_api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "input": text,
        "model": "nvidia/nv-embedqa-e5-v5",
        "input_type": "query"
    }
    
    response = requests.post(url, headers=headers, json=payload)
    if response.status_code != 200:
        raise Exception(f"NVIDIA API Error {response.status_code}: {response.text}")
    
    return response.json()["data"][0]["embedding"]

def add_documents(collection_name: str, docs: List[Document]) -> int:
    """Nạp tài liệu vào ChromaDB bằng cách gọi trực tiếp API Embedding."""
    if not docs:
        return 0
        
    store = get_vector_store(collection_name)
    logger.info(f"Đang nạp {len(docs)} đoạn văn bản vào {collection_name} qua Direct API...")
    
    success_count = 0
    for idx, doc in enumerate(docs):
        try:
            vector = _get_nvidia_embedding_direct(doc.page_content)
            
            store.add_texts(
                texts=[doc.page_content],
                metadatas=[doc.metadata],
                embeddings=[vector]
            )
            success_count += 1
        except Exception as e:
            logger.error(f"Thất bại tại đoạn {idx} (Ticker: {doc.metadata.get('ticker')}): {e}")
            continue
            
    logger.success(f"Hoàn tất! Đã nạp thành công {success_count}/{len(docs)} đoạn văn bản.")
    return success_count


def similarity_search(
    collection_name: str,
    query: str,
    k: int = 5,
    filter: Optional[dict] = None,
) -> List[Document]:
    """
    Retrieve top-k similar documents from the collection.

    Args:
        collection_name: ChromaDB collection name.
        query: The search query string.
        k: Number of top results to return.
        filter: Optional metadata filter dict.
    """
    store = get_vector_store(collection_name)
    return store.similarity_search(query, k=k, filter=filter)
