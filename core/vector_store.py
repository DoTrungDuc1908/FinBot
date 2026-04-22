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


@lru_cache(maxsize=1)
def _get_chroma_client() -> chromadb.HttpClient:
    """Return a singleton ChromaDB HTTP client."""
    client = chromadb.HttpClient(
        host=settings.chroma_host,
        port=settings.chroma_port,
    )
    logger.info(f"Connected to ChromaDB at {settings.chroma_host}:{settings.chroma_port}")
    return client


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


def add_documents(collection_name: str, docs: List[Document]) -> None:
    """Add documents to a ChromaDB collection."""
    store = get_vector_store(collection_name)
    store.add_documents(docs)
    logger.info(f"Added {len(docs)} documents to collection '{collection_name}'")


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
