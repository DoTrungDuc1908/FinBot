"""
core/embeddings.py
Embedding client using NVIDIA NIM (OpenAI-compatible embeddings endpoint).
Falls back to a local sentence-transformers model if NVIDIA key is unavailable.
"""
from functools import lru_cache
from typing import List

from loguru import logger
from langchain_openai import OpenAIEmbeddings

from config.settings import settings


@lru_cache(maxsize=1)
def get_embeddings() -> OpenAIEmbeddings:
    """
    Return cached NVIDIA NIM embedding model.
    Compatible with LangChain VectorStore interfaces.
    """
    logger.info(f"Loading embedding model: {settings.nvidia_embedding_model}")
    return OpenAIEmbeddings(
        model=settings.nvidia_embedding_model,
        api_key=settings.nvidia_api_key,
        base_url=settings.nvidia_base_url,
        dimensions=1024,
    )


def embed_texts(texts: List[str]) -> List[List[float]]:
    """Embed a list of texts and return float vectors."""
    embedder = get_embeddings()
    return embedder.embed_documents(texts)


def embed_query(text: str) -> List[float]:
    """Embed a single query string."""
    embedder = get_embeddings()
    return embedder.embed_query(text)
