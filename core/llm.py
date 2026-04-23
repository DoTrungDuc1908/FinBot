"""
core/llm.py
LLM client factory targeting NVIDIA NIM (OpenAI-compatible endpoint).
Hỗ trợ Dynamic Model Selection (Đổi model theo Agent).
"""
from functools import lru_cache
from langchain_openai import ChatOpenAI
from config.settings import settings

class SingleToolChatOpenAI(ChatOpenAI):
    def bind_tools(self, tools, **kwargs):
        kwargs["parallel_tool_calls"] = False
        return super().bind_tools(tools, **kwargs)

@lru_cache(maxsize=5)
def get_llm(temperature: float = 0.1, max_tokens: int | None = None, model_name: str | None = None) -> ChatOpenAI:
    """
    Khởi tạo LLM cho các Agent cơ bản và phức tạp.
    """
    target_model = model_name or settings.nvidia_llm_model
    return SingleToolChatOpenAI(
        model=target_model,
        api_key=settings.nvidia_api_key,
        base_url=settings.nvidia_base_url,
        temperature=temperature,
        max_tokens=max_tokens or settings.max_tokens_per_request,
        streaming=True,
    )

def get_fast_llm(max_tokens: int | None = 2048, model_name: str | None = None) -> ChatOpenAI:
    """
    Khởi tạo LLM tốc độ cao cho Router và Data Evaluation.
    """
    target_model = model_name or settings.nvidia_router_model
    return SingleToolChatOpenAI(
        model=target_model,
        api_key=settings.nvidia_api_key,
        base_url=settings.nvidia_base_url,
        temperature=0.0,
        max_tokens=max_tokens,
        streaming=False,
    )