"""Token counting and context window management.

KEY CONCEPTS:
  1. Token usage is tracked per-turn to decide when to compact
  2. AUTOCOMPACT_BUFFER_TOKENS = 13,000 — the safety margin before context limit
  3. Context window varies by model (200K for Anthropic, 128K for GPT-4o)
  4. Token estimation uses tiktoken for OpenAI models, ~4 chars/token fallback
"""

from __future__ import annotations

import tiktoken
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage


# ─────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────

AUTOCOMPACT_BUFFER_TOKENS = 13_000

WARNING_THRESHOLD_BUFFER_TOKENS = 20_000

MAX_CONSECUTIVE_COMPACT_FAILURES = 3

MAX_OUTPUT_TOKENS_FOR_SUMMARY = 4_000

# OpenAI models have different limits
MODEL_CONTEXT_LIMITS: dict[str, int] = {
    "gpt-4o": 128_000,
    "gpt-4o-mini": 128_000,
    "gpt-4-turbo": 128_000,
    "gpt-4": 8_192,
    "gpt-3.5-turbo": 16_385,
    "claude-sonnet-4-20250514": 200_000,
    "claude-opus-4-20250514": 200_000,
    "claude-haiku-4-5-20251001": 200_000,
}

# Default context window if model not in lookup
DEFAULT_CONTEXT_WINDOW = 128_000

# Characters per token for rough estimation
CHARS_PER_TOKEN = 4

# Message overhead tokens (role, formatting)
MESSAGE_OVERHEAD_TOKENS = 4


# ─────────────────────────────────────────────────────────
# TOKEN COUNTING
# ─────────────────────────────────────────────────────────

def _get_encoding(model: str) -> tiktoken.Encoding | None:
    """Get tiktoken encoding for a model.

    We use tiktoken for OpenAI models, fallback for others.
    """
    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        # Try cl100k_base as fallback for newer OpenAI models
        try:
            return tiktoken.get_encoding("cl100k_base")
        except Exception:
            return None


def count_text_tokens(text: str, model: str = "") -> int:
    """Count tokens in a text string.

    Uses tiktoken for accuracy, falls back to ~4 chars/token.
    """
    if not text:
        return 0

    encoding = _get_encoding(model) if model else None
    if encoding:
        return len(encoding.encode(text))

    # Fallback: ~4 chars per token
    return len(text) // CHARS_PER_TOKEN


def count_message_tokens(message: BaseMessage, model: str = "") -> int:
    """Estimate tokens for a single message.

    tokens = message_overhead + content_tokens + tool_call_tokens
    """
    # Content tokens
    content = ""
    if isinstance(message.content, str):
        content = message.content
    elif isinstance(message.content, list):
        # Multi-part content (e.g., with images)
        for part in message.content:
            if isinstance(part, dict) and "text" in part:
                content += part["text"]
    content_tokens = count_text_tokens(content, model)

    # Tool call tokens (from AIMessage)
    tool_tokens = 0
    if isinstance(message, AIMessage) and hasattr(message, "tool_calls") and message.tool_calls:
        for tc in message.tool_calls:
            tool_tokens += count_text_tokens(tc.get("name", ""), model)
            tool_tokens += count_text_tokens(str(tc.get("args", {})), model)

    # Tool result tokens (from ToolMessage)
    if isinstance(message, ToolMessage):
        tool_tokens += count_text_tokens(message.name or "", model)

    return MESSAGE_OVERHEAD_TOKENS + content_tokens + tool_tokens


def count_messages_tokens(messages: list[BaseMessage], model: str = "") -> int:
    """Count total tokens across all messages.

    This is the main function used by should_auto_compact() to
    determine if the context window is filling up.
    """
    total = 0
    for msg in messages:
        total += count_message_tokens(msg, model)
    return total


# ─────────────────────────────────────────────────────────
# CONTEXT WINDOW MANAGEMENT
# ─────────────────────────────────────────────────────────

def get_context_window(model: str) -> int:
    """Get the context window size for a model.

    Falls back to DEFAULT_CONTEXT_WINDOW if model not in lookup.
    """
    # Try exact match first
    if not model or model in MODEL_CONTEXT_LIMITS:
        return MODEL_CONTEXT_LIMITS.get(model, DEFAULT_CONTEXT_WINDOW)

    # Try prefix match (e.g., "gpt-4o-2024-08-06" matches "gpt-4o")
    for key, limit in MODEL_CONTEXT_LIMITS.items():
        if model.startswith(key):
            return limit

    return DEFAULT_CONTEXT_WINDOW


def get_compact_threshold(model: str) -> int:
    """Get the token threshold that triggers auto-compact.

    threshold = context_window - AUTOCOMPACT_BUFFER_TOKENS
    """
    return get_context_window(model) - AUTOCOMPACT_BUFFER_TOKENS


def get_warning_threshold(model: str) -> int:
    """Get the token threshold for the warning state."""
    return get_context_window(model) - WARNING_THRESHOLD_BUFFER_TOKENS


def should_auto_compact(
    messages: list[BaseMessage],
    model: str = "",
    consecutive_failures: int = 0,
) -> bool:
    """Check if auto-compact should be triggered.

    Returns True if:
    1. Token count exceeds compact threshold
    2. Circuit breaker hasn't tripped (failures < MAX)
    """
    if consecutive_failures >= MAX_CONSECUTIVE_COMPACT_FAILURES:
        return False

    token_count = count_messages_tokens(messages, model)
    threshold = get_compact_threshold(model)
    return token_count >= threshold


def get_token_warning_state(
    messages: list[BaseMessage],
    model: str = "",
) -> str:
    """Get the current token warning state.

    Returns: "ok" | "warning" | "compact" | "error"
    """
    token_count = count_messages_tokens(messages, model)
    window = get_context_window(model)

    if token_count >= window - AUTOCOMPACT_BUFFER_TOKENS:
        return "compact"
    if token_count >= window - WARNING_THRESHOLD_BUFFER_TOKENS:
        return "warning"
    return "ok"
