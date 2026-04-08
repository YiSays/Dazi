"""Context compression: micro-compact, full compact, and message grouping.

KEY CONCEPTS:
  1. Micro-compact: clear old tool results, keep tool calls (fast, no LLM)
  2. Full compact: LLM summarizes old messages, replaces with summary
  3. API-round grouping: messages grouped by Human->AI->Tool sequences
  4. Compact boundary marker: separates summary from recent messages
  5. Circuit breaker: stop retrying after 3 consecutive failures
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage

from dazi.core.tokenizer import (
    AUTOCOMPACT_BUFFER_TOKENS,
    MAX_OUTPUT_TOKENS_FOR_SUMMARY,
    count_messages_tokens,
    count_text_tokens,
    get_compact_threshold,
    get_context_window,
)


# ─────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────

# The boundary marker between summary and recent messages
COMPACT_BOUNDARY = "<!-- COMPACT_BOUNDARY -->"

# Replacement text for cleared tool results
CLEARED_TOOL_RESULT = "[Old tool result content cleared]"

# Tools whose results can be micro-compacted
COMPACTABLE_TOOLS = frozenset({
    "file_reader", "shell_exec", "grep", "glob",
    "web_search", "web_fetch", "memory_read", "memory_search",
})

# Default number of recent rounds to preserve during compact
DEFAULT_KEEP_RECENT_ROUNDS = 3

# Maximum tokens for the compact summary output
COMPACT_MAX_OUTPUT_TOKENS = MAX_OUTPUT_TOKENS_FOR_SUMMARY


# ─────────────────────────────────────────────────────────
# DATA CLASSES
# ─────────────────────────────────────────────────────────

@dataclass
class CompactResult:
    """Result of a compact operation."""
    messages: list[BaseMessage]
    tokens_before: int
    tokens_after: int
    method: str  # "micro", "full", "none"
    summary: str = ""
    rounds_removed: int = 0
    tool_results_cleared: int = 0


# ─────────────────────────────────────────────────────────
# MESSAGE GROUPING
# ─────────────────────────────────────────────────────────

def group_messages_by_round(messages: list[BaseMessage]) -> list[list[BaseMessage]]:
    """Group messages into API rounds.

    An API round starts with a HumanMessage and includes all subsequent
    AI responses and tool messages until the next HumanMessage.

    This is used for:
    1. Micro-compact: identify old rounds to clear tool results from
    2. Full compact: identify old rounds to summarize
    3. Keeping recent rounds intact

    Args:
        messages: Message list to group.

    Returns:
        List of message groups (rounds). SystemMessages are attached to
        the first round.
    """
    if not messages:
        return []

    rounds: list[list[BaseMessage]] = []
    current_round: list[BaseMessage] = []

    for msg in messages:
        # SystemMessages go at the start of the first round
        if isinstance(msg, SystemMessage):
            if current_round:
                current_round.insert(0, msg)
            else:
                current_round.append(msg)
            continue

        # HumanMessage starts a new round (unless it's the very first)
        if isinstance(msg, HumanMessage):
            if current_round and not all(isinstance(m, SystemMessage) for m in current_round):
                rounds.append(current_round)
                current_round = []
            current_round.append(msg)
            continue

        # AIMessage and ToolMessage continue the current round
        current_round.append(msg)

    if current_round:
        rounds.append(current_round)

    return rounds


# ─────────────────────────────────────────────────────────
# MICRO-COMPACT
# ─────────────────────────────────────────────────────────

def micro_compact(
    messages: list[BaseMessage],
    keep_recent_rounds: int = DEFAULT_KEEP_RECENT_ROUNDS,
    model: str = "",
) -> CompactResult:
    """Clear old tool results without LLM call.

    Strategy:
    1. Group messages by API rounds
    2. Skip the last N rounds (keep them intact)
    3. For older rounds, replace tool results from compactable tools
       with the CLEARED_TOOL_RESULT marker
    4. Keep tool calls intact (so you know what happened)

    This is fast — no LLM call needed. Typically saves 40-60% of
    tool result tokens.

    Args:
        messages: Current message list.
        keep_recent_rounds: Number of recent rounds to preserve.
        model: Model name for token counting.

    Returns:
        CompactResult with cleared tool results.
    """
    tokens_before = count_messages_tokens(messages, model)

    rounds = group_messages_by_round(messages)
    if len(rounds) <= keep_recent_rounds:
        return CompactResult(
            messages=messages,
            tokens_before=tokens_before,
            tokens_after=tokens_before,
            method="none",
        )

    # Identify rounds to compact (all except last N)
    old_rounds = rounds[:-keep_recent_rounds]
    recent_rounds = rounds[-keep_recent_rounds:]

    tool_results_cleared = 0
    new_old_rounds: list[list[BaseMessage]] = []

    for round_msgs in old_rounds:
        new_msgs: list[BaseMessage] = []
        for msg in round_msgs:
            if isinstance(msg, ToolMessage) and msg.name in COMPACTABLE_TOOLS:
                # Replace content with marker, keep the message structure
                new_msg = ToolMessage(
                    content=CLEARED_TOOL_RESULT,
                    tool_call_id=msg.tool_call_id,
                    name=msg.name,
                )
                new_msgs.append(new_msg)
                tool_results_cleared += 1
            else:
                new_msgs.append(msg)
        new_old_rounds.append(new_msgs)

    # Reassemble
    result_messages: list[BaseMessage] = []
    for round_msgs in new_old_rounds:
        result_messages.extend(round_msgs)
    for round_msgs in recent_rounds:
        result_messages.extend(round_msgs)

    tokens_after = count_messages_tokens(result_messages, model)

    return CompactResult(
        messages=result_messages,
        tokens_before=tokens_before,
        tokens_after=tokens_after,
        method="micro",
        tool_results_cleared=tool_results_cleared,
    )


# ─────────────────────────────────────────────────────────
# FULL COMPACT
# ─────────────────────────────────────────────────────────

COMPACT_PROMPT = """\
Summarize the following conversation concisely. Preserve:

1. Key decisions and conclusions reached
2. Important facts, preferences, and constraints mentioned
3. The current state of any ongoing work
4. Any errors encountered and how they were resolved
5. Tool usage outcomes (file contents read, command outputs)

Rules:
- Be specific, not generic. Include actual values, names, and paths.
- Preserve the chronological flow of the conversation.
- Don't include greetings, acknowledgments, or filler.
- Use bullet points for clarity.
- Keep under {max_tokens} tokens.

Conversation to summarize:
{conversation}
"""

COMPACT_SUMMARY_PREFIX = "[Conversation Summary — auto-compact]"
COMPACT_SUMMARY_SUFFIX = (
    "Messages above this line have been summarized. "
    "Continue the conversation naturally — the summary preserves key context."
)


async def full_compact(
    messages: list[BaseMessage],
    llm: Any,
    keep_recent_rounds: int = DEFAULT_KEEP_RECENT_ROUNDS,
    model: str = "",
) -> CompactResult:
    """Full compact: summarize old messages with LLM.

    Strategy:
    1. Group messages by API rounds
    2. Split: old rounds → summarize, recent rounds → keep intact
    3. Send old rounds to LLM with summarization prompt
    4. Replace old rounds with SystemMessage containing summary
    5. Add COMPACT_BOUNDARY marker between summary and recent messages

    Args:
        messages: Current message list.
        llm: LLM instance for summarization.
        keep_recent_rounds: Number of recent rounds to preserve.
        model: Model name for token counting.

    Returns:
        CompactResult with summarized old messages.
    """
    tokens_before = count_messages_tokens(messages, model)

    rounds = group_messages_by_round(messages)

    if len(rounds) <= keep_recent_rounds:
        return CompactResult(
            messages=messages,
            tokens_before=tokens_before,
            tokens_after=tokens_before,
            method="none",
        )

    old_rounds = rounds[:-keep_recent_rounds]
    recent_rounds = rounds[-keep_recent_rounds:]
    rounds_removed = len(old_rounds)

    # Format old conversation for summarization
    old_messages: list[BaseMessage] = []
    for round_msgs in old_rounds:
        old_messages.extend(round_msgs)

    conversation_text = _format_for_summarization(old_messages)

    # Create summarization prompt
    prompt = COMPACT_PROMPT.format(
        max_tokens=COMPACT_MAX_OUTPUT_TOKENS,
        conversation=conversation_text,
    )

    # Call LLM for summary
    try:
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        summary = response.content if hasattr(response, "content") else str(response)
    except Exception as e:
        # Compact failed — return original messages
        return CompactResult(
            messages=messages,
            tokens_before=tokens_before,
            tokens_after=tokens_before,
            method="none",
            summary=f"Compact failed: {e}",
        )

    # Build summary message
    summary_text = (
        f"{COMPACT_SUMMARY_PREFIX}\n\n{summary}\n\n"
        f"{COMPACT_SUMMARY_SUFFIX}\n{COMPACT_BOUNDARY}"
    )

    summary_message = SystemMessage(content=summary_text)

    # Reassemble: summary + recent rounds
    result_messages: list[BaseMessage] = [summary_message]
    for round_msgs in recent_rounds:
        result_messages.extend(round_msgs)

    tokens_after = count_messages_tokens(result_messages, model)

    return CompactResult(
        messages=result_messages,
        tokens_before=tokens_before,
        tokens_after=tokens_after,
        method="full",
        summary=summary,
        rounds_removed=rounds_removed,
    )


def _format_for_summarization(messages: list[BaseMessage]) -> str:
    """Format messages into text for the summarization prompt.

    Strips images, keeps text content.
    """
    parts: list[str] = []
    for msg in messages:
        if isinstance(msg, SystemMessage):
            role = "System"
        elif isinstance(msg, HumanMessage):
            role = "User"
        elif isinstance(msg, AIMessage):
            role = "Assistant"
        elif isinstance(msg, ToolMessage):
            role = f"Tool({msg.name or 'unknown'})"
        else:
            role = "Unknown"

        content = ""
        if isinstance(msg.content, str):
            content = msg.content
        elif isinstance(msg.content, list):
            for part_item in msg.content:
                if isinstance(part_item, dict) and "text" in part_item:
                    content += part_item["text"]

        # Truncate very long tool results
        if isinstance(msg, ToolMessage) and len(content) > 2000:
            content = content[:2000] + "\n... (truncated)"

        if content:
            parts.append(f"[{role}]: {content}")

        # Include tool call names for AI messages
        if isinstance(msg, AIMessage) and hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                parts.append(f"[Assistant called {tc['name']}({str(tc.get('args', {}))[:200]})]")

    return "\n\n".join(parts)


# ─────────────────────────────────────────────────────────
# AUTO-COMPACT ORCHESTRATOR
# ─────────────────────────────────────────────────────────

async def auto_compact(
    messages: list[BaseMessage],
    llm: Any,
    model: str = "",
    consecutive_failures: int = 0,
) -> CompactResult:
    """Orchestrate auto-compaction: try micro, then full if needed.

    Strategy:
    1. Check if compaction needed (token count > threshold)
    2. Circuit breaker: skip if too many consecutive failures
    3. Try micro-compact first (fast, no LLM call)
    4. If still over threshold, try full compact (LLM summarization)
    5. Track consecutive failures for circuit breaker

    Args:
        messages: Current message list.
        llm: LLM instance for full compact.
        model: Model name for token counting.
        consecutive_failures: Number of consecutive compact failures.

    Returns:
        CompactResult from whichever method was used.
    """
    from dazi.core.tokenizer import should_auto_compact

    # Circuit breaker check
    if consecutive_failures >= 3:
        return CompactResult(
            messages=messages,
            tokens_before=count_messages_tokens(messages, model),
            tokens_after=count_messages_tokens(messages, model),
            method="none",
            summary="Circuit breaker active — too many consecutive failures",
        )

    # Check if compaction needed
    if not should_auto_compact(messages, model, consecutive_failures):
        return CompactResult(
            messages=messages,
            tokens_before=count_messages_tokens(messages, model),
            tokens_after=count_messages_tokens(messages, model),
            method="none",
        )

    # Try micro-compact first (fast)
    micro_result = micro_compact(messages, model=model)

    if micro_result.method == "none":
        return micro_result

    # Check if micro was enough
    threshold = get_compact_threshold(model)
    if micro_result.tokens_after < threshold:
        return micro_result

    # Still over threshold — try full compact
    full_result = await full_compact(
        micro_result.messages, llm, model=model,
    )

    return full_result


# ─────────────────────────────────────────────────────────
# MANUAL COMPACT — user-triggered /compact
# ─────────────────────────────────────────────────────────

async def manual_compact(
    messages: list[BaseMessage],
    llm: Any,
    model: str = "",
) -> CompactResult:
    """User-triggered full compact.

    Uses a smaller buffer for more aggressive compaction.
    """
    if len(messages) <= 2:
        return CompactResult(
            messages=messages,
            tokens_before=count_messages_tokens(messages, model),
            tokens_after=count_messages_tokens(messages, model),
            method="none",
            summary="Not enough messages to compact",
        )

    return await full_compact(messages, llm, model=model)
