"""
config/settings.py
Centralized configuration using pydantic-settings.
All values are loaded from .env file.
"""
from functools import lru_cache
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    nvidia_api_key: str = Field(default="", alias="NVIDIA_API_KEY")
    nvidia_base_url: str = Field(
        default="https://integrate.api.nvidia.com/v1", alias="NVIDIA_BASE_URL"
    )
    nvidia_embedding_model: str = Field(
        default="nvidia/nv-embedqa-e5-v5", alias="NVIDIA_EMBEDDING_MODEL"
    )
    
    nvidia_llm_model: str = Field(default="meta/llama-3.1-70b-instruct", alias="NVIDIA_LLM_MODEL")
    
    nvidia_router_model: str = Field(default="meta/llama-3.1-8b-instruct", alias="NVIDIA_ROUTER_MODEL")
    
    nvidia_agent_model: str = Field(default="meta/llama-3.1-70b-instruct", alias="NVIDIA_AGENT_MODEL")

    nvidia_sentiment_model: str = Field(default="qwen/qwen3-next-80b-a3b-thinking", alias="NVIDIA_SENTIMENT_MODEL")
    
    nvidia_advisor_model: str = Field(default="meta/llama-3.1-70b-instruct", alias="NVIDIA_ADVISOR_MODEL")
    
    nvidia_eval_model: str = Field(default="meta/llama-3.1-8b-instruct", alias="NVIDIA_EVAL_MODEL")

    redis_host: str = Field(default="localhost", alias="REDIS_HOST")
    redis_port: int = Field(default=6379, alias="REDIS_PORT")
    redis_db: int = Field(default=0, alias="REDIS_DB")
    redis_password: str = Field(default="", alias="REDIS_PASSWORD")

    postgres_host: str = Field(default="localhost", alias="POSTGRES_HOST")
    postgres_port: int = Field(default=5432, alias="POSTGRES_PORT")
    postgres_db: str = Field(default="finbot", alias="POSTGRES_DB")
    postgres_user: str = Field(default="finbot_user", alias="POSTGRES_USER")
    postgres_password: str = Field(default="finbot_pass", alias="POSTGRES_PASSWORD")

    @property
    def postgres_url(self) -> str:
        return (
            f"postgresql+psycopg2://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    chroma_host: str = Field(default="localhost", alias="CHROMA_HOST")
    chroma_port: int = Field(default=8000, alias="CHROMA_PORT")
    chroma_collection_reports: str = Field(
        default="financial_reports", alias="CHROMA_COLLECTION_REPORTS"
    )
    chroma_collection_news: str = Field(default="stock_news", alias="CHROMA_COLLECTION_NEWS")

    api_host: str = Field(default="0.0.0.0", alias="API_HOST")
    api_port: int = Field(default=8080, alias="API_PORT")
    api_debug: bool = Field(default=False, alias="API_DEBUG")

    tcbs_base_url: str = Field(
        default="https://apipubaws.tcbs.com.vn", alias="TCBS_BASE_URL"
    )
    ssi_base_url: str = Field(
        default="https://fc-data.ssi.com.vn", alias="SSI_BASE_URL"
    )

    cafef_rss: str = Field(
        default="https://cafef.vn/thi-truong-chung-khoan.rss", alias="CAFEF_RSS"
    )
    vneconomy_rss: str = Field(
        default="https://vneconomy.vn/chung-khoan.rss", alias="VNECONOMY_RSS"
    )

    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    cache_ttl_price: int = Field(default=300, alias="CACHE_TTL_PRICE")
    cache_ttl_company: int = Field(default=86400, alias="CACHE_TTL_COMPANY")
    cache_ttl_news: int = Field(default=1800, alias="CACHE_TTL_NEWS")
    max_tokens_per_request: int = Field(default=4096, alias="MAX_TOKENS_PER_REQUEST")

    request_timeout: int = Field(default=10, alias="REQUEST_TIMEOUT") 
    
    max_retries: int = Field(default=3, alias="MAX_RETRIES")
    
    enable_parallel_agents: bool = Field(default=True, alias="ENABLE_PARALLEL_AGENTS")
    llm_rate_limit: int = Field(default=1, alias="LLM_RATE_LIMIT")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached settings singleton."""
    return Settings()


settings = get_settings()