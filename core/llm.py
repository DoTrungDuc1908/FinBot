"""
core/llm.py
LLM client factory targeting NVIDIA NIM (OpenAI-compatible endpoint).
"""
from functools import lru_cache
from langchain_openai import ChatOpenAI
from config.settings import settings

class SingleToolChatOpenAI(ChatOpenAI):
    """
    Custom wrapper cho ChatOpenAI để tắt tính năng Parallel Tool Calling.
    Bắt buộc áp dụng cho các model Llama/NVIDIA NIM không hỗ trợ tính năng này.
    """
    def bind_tools(self, tools, **kwargs):
        # Ép buộc tắt parallel tool calls ở mọi agent sử dụng LLM này
        kwargs["parallel_tool_calls"] = False
        return super().bind_tools(tools, **kwargs)

@lru_cache(maxsize=1)
def get_llm(temperature: float = 0.1, max_tokens: int | None = None) -> ChatOpenAI:
    """
    Return a cached ChatOpenAI instance pointed at NVIDIA NIM.
    Uses low temperature by default for factual financial answers.
    """
    # Sử dụng class custom thay vì ChatOpenAI gốc
    return SingleToolChatOpenAI(
        model=settings.nvidia_llm_model,
        api_key=settings.nvidia_api_key,
        base_url=settings.nvidia_base_url,
        temperature=temperature,
        max_tokens=max_tokens or settings.max_tokens_per_request,
        streaming=True,
    )

def get_fast_llm() -> ChatOpenAI:
    """
    Lightweight LLM for intent classification / routing tasks.
    Uses llama-3.1-8b if available, otherwise falls back to main model.
    """
    fast_model = "meta/llama-3.1-8b-instruct"
    # Tương tự, áp dụng cho cả fast_llm
    return SingleToolChatOpenAI(
        model=fast_model,
        api_key=settings.nvidia_api_key,
        base_url=settings.nvidia_base_url,
        temperature=0.0,
        max_tokens=256,
        streaming=False,
    )