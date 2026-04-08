"""Concurrent tool execution — batching tools by safety level."""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import ToolMessage

from dazi.core.base import ToolSafety


# ─────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────

MAX_CONCURRENT = int(os.getenv("DAZI_MAX_CONCURRENT", "5"))


# ─────────────────────────────────────────────────────────
# TOOL CALL BATCH
# ─────────────────────────────────────────────────────────

@dataclass
class ToolCallBatch:
    """A batch of tool calls categorized by concurrency safety."""
    parallel: list[dict] = field(default_factory=list)
    serial: list[dict] = field(default_factory=list)


def partition_tool_calls(
    tool_calls: list[dict],
    tool_meta: dict[str, Any],
) -> ToolCallBatch:
    """Split tool calls into parallel-safe and serial batches.

    Rules:
      - SAFE tools: run in parallel (file_reader, calculator, glob, grep)
      - WRITE tools: run serially (file_writer)
      - DESTRUCTIVE tools: run serially (shell_exec)
      - Same tool with same args: deduplicate

    Args:
        tool_calls: List of tool call dicts with 'name' and 'args'.
        tool_meta: Dict mapping tool_name -> DaziTool metadata.

    Returns:
        ToolCallBatch with parallel and serial lists.
    """
    parallel: list[dict] = []
    serial: list[dict] = []
    seen_keys: set[str] = set()

    for tc in tool_calls:
        name = tc["name"]
        args = tc.get("args", {})
        dedup_key = f"{name}:{sorted(args.items())}"

        # Deduplicate identical calls
        if dedup_key in seen_keys:
            continue
        seen_keys.add(dedup_key)

        meta = tool_meta.get(name)
        if meta and meta.is_concurrency_safe:
            parallel.append(tc)
        else:
            serial.append(tc)

    return ToolCallBatch(parallel=parallel, serial=serial)


# ─────────────────────────────────────────────────────────
# CONCURRENT EXECUTION
# ─────────────────────────────────────────────────────────

async def execute_tool(
    tool_call: dict,
    tools: list[Any],
) -> ToolMessage:
    """Execute a single tool call and return a ToolMessage."""
    tool_name = tool_call["name"]
    tool_args = tool_call.get("args", {})
    tool_id = tool_call["id"]

    tool = next((t for t in tools if t.name == tool_name), None)
    if tool is None:
        return ToolMessage(
            content=f"Error: Tool '{tool_name}' not found",
            tool_call_id=tool_id,
        )

    try:
        result = tool.invoke(tool_args)
        return ToolMessage(content=str(result), tool_call_id=tool_id)
    except Exception as e:
        return ToolMessage(
            content=f"Error executing '{tool_name}': {e}",
            tool_call_id=tool_id,
        )


async def execute_tools_concurrent(
    tool_calls: list[dict],
    tools: list[Any],
    tool_meta: dict[str, Any],
    max_concurrent: int = MAX_CONCURRENT,
) -> list[ToolMessage]:
    """Execute tool calls with concurrency batching.

    1. Partition into parallel-safe and serial batches
    2. Run parallel batch with semaphore-limited concurrency
    3. Run serial batch one at a time
    4. Return all ToolMessages in original order

    Args:
        tool_calls: Tool calls from the LLM response.
        tools: Available StructuredTool instances.
        tool_meta: Dict mapping tool_name -> DaziTool.
        max_concurrent: Max parallel tool executions.

    Returns:
        List of ToolMessages in the same order as tool_calls.
    """
    if not tool_calls:
        return []

    batch = partition_tool_calls(tool_calls, tool_meta)
    semaphore = asyncio.Semaphore(max_concurrent)

    # Build ordered execution list: (index, tool_call, is_parallel)
    ordered: list[tuple[int, dict, bool]] = []
    tc_index_map: dict[str, int] = {}  # tool_call_id -> original index

    for i, tc in enumerate(tool_calls):
        tc_id = tc["id"]
        tc_index_map[tc_id] = i
        if tc in batch.parallel:
            ordered.append((i, tc, True))
        elif tc in batch.serial:
            ordered.append((i, tc, False))

    # Track results by tool_call_id
    results_map: dict[str, ToolMessage] = {}

    # Execute parallel batch
    async def _run_parallel(tc: dict):
        async with semaphore:
            msg = await execute_tool(tc, tools)
            results_map[tc["id"]] = msg

    parallel_tasks = []
    for _, tc, is_parallel in ordered:
        if is_parallel:
            parallel_tasks.append(asyncio.create_task(_run_parallel(tc)))

    if parallel_tasks:
        await asyncio.gather(*parallel_tasks)

    # Execute serial batch
    for _, tc, is_parallel in ordered:
        if not is_parallel:
            msg = await execute_tool(tc, tools)
            results_map[tc["id"]] = msg

    # Reassemble in original order
    results: list[ToolMessage] = []
    for tc in tool_calls:
        if tc["id"] in results_map:
            results.append(results_map[tc["id"]])

    return results


def execute_tools_sync(
    tool_calls: list[dict],
    tools: list[Any],
    tool_meta: dict[str, Any],
) -> list[ToolMessage]:
    """Synchronous version — executes all tools serially.

    Used as fallback when async is not available (e.g. in LangGraph sync nodes).
    """
    if not tool_calls:
        return []

    results: list[ToolMessage] = []
    for tc in tool_calls:
        results.append(asyncio.run(execute_tool(tc, tools)))
    return results
