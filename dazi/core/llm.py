"""LLM client wrapper — wraps OpenAI (or compatible) API via langchain-openai."""

from langchain_openai import ChatOpenAI

from dazi.core.config import OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL


def create_llm(
    model: str | None = None,
    temperature: float = 0.0,
    streaming: bool = True,
    base_url: str | None = None,
    api_key: str | None = None,
) -> ChatOpenAI:
    """Create a ChatOpenAI instance with project defaults.

    Args:
        model: Override model name. Defaults to OPENAI_MODEL env var.
        temperature: Model temperature.
        streaming: Enable streaming responses.
        base_url: Override API base URL. Defaults to OPENAI_BASE_URL env var.
        api_key: Override API key. Defaults to OPENAI_API_KEY env var.

    Returns:
        Configured ChatOpenAI instance.
    """
    kwargs: dict = {
        "model": model or OPENAI_MODEL,
        "temperature": temperature,
        "streaming": streaming,
    }
    if api_key:
        kwargs["api_key"] = api_key
    elif OPENAI_API_KEY:
        kwargs["api_key"] = OPENAI_API_KEY
    effective_base_url = base_url or OPENAI_BASE_URL
    if effective_base_url:
        kwargs["base_url"] = effective_base_url

    return ChatOpenAI(**kwargs)
