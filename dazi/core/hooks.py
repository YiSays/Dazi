"""Hook system — pre/post tool execution events.

We implement 5 core events that cover the tool execution lifecycle:
    validate -> pre-hook -> permission check -> execute -> post-hook -> result
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Coroutine

from dazi.core.permissions import PermissionBehavior


# ─────────────────────────────────────────────────────────
# HOOK EVENTS
# ─────────────────────────────────────────────────────────

class HookEvent(str, Enum):
    """Core hook events in the tool execution lifecycle.

    Events:
      PRE_TOOL_USE         — before tool execution, can modify input
      POST_TOOL_USE        — after successful tool execution, can modify output
      POST_TOOL_USE_FAILURE — after tool execution failure
      USER_PROMPT_SUBMIT   — when user submits a prompt (before LLM call)
      SESSION_START         — when agent session begins
    """
    PRE_TOOL_USE = "pre_tool_use"
    POST_TOOL_USE = "post_tool_use"
    POST_TOOL_USE_FAILURE = "post_tool_use_failure"
    USER_PROMPT_SUBMIT = "user_prompt_submit"
    SESSION_START = "session_start"


# ─────────────────────────────────────────────────────────
# HOOK RESULT
# ─────────────────────────────────────────────────────────

@dataclass
class HookResult:
    """Result from a hook execution.

    Hooks can:
      - Modify tool input before execution (modified_input)
      - Modify tool output after execution (modified_output)
      - Override permission behavior (permission_override)
      - Block execution entirely (should_block)
    """
    modified_input: dict[str, Any] | None = None
    modified_output: str | None = None
    permission_override: PermissionBehavior | None = None
    should_block: bool = False
    block_reason: str = ""

    @staticmethod
    def merge(*results: HookResult) -> HookResult:
        """Merge multiple hook results — later results override earlier ones."""
        merged = HookResult()
        for r in results:
            if r.modified_input is not None:
                merged.modified_input = r.modified_input
            if r.modified_output is not None:
                merged.modified_output = r.modified_output
            if r.permission_override is not None:
                merged.permission_override = r.permission_override
            if r.should_block:
                merged.should_block = True
                merged.block_reason = r.block_reason
        return merged


# ─────────────────────────────────────────────────────────
# HOOK HANDLER
# ─────────────────────────────────────────────────────────

HookHandler = Callable[..., Coroutine[Any, Any, HookResult]]


# ─────────────────────────────────────────────────────────
# HOOK REGISTRY
# ─────────────────────────────────────────────────────────

class HookRegistry:
    """Registry for hook handlers.

    Hooks are fired in priority order (lower priority = runs first).
    Multiple hooks can be registered for the same event.
    Results are merged — later hooks can override earlier ones.
    """

    def __init__(self):
        self._hooks: dict[HookEvent, list[tuple[int, HookHandler]]] = {}

    def register(
        self,
        event: HookEvent,
        handler: HookHandler,
        priority: int = 0,
    ) -> None:
        """Register a hook handler.

        Args:
            event: The event to listen for.
            handler: Async function that returns HookResult.
            priority: Lower runs first. Default 0.
        """
        if event not in self._hooks:
            self._hooks[event] = []
        self._hooks[event].append((priority, handler))
        # Sort by priority
        self._hooks[event].sort(key=lambda x: x[0])

    def unregister(self, event: HookEvent, handler: HookHandler) -> bool:
        """Remove a specific handler. Returns True if found and removed."""
        if event not in self._hooks:
            return False
        original_len = len(self._hooks[event])
        self._hooks[event] = [
            (p, h) for p, h in self._hooks[event] if h is not handler
        ]
        return len(self._hooks[event]) < original_len

    def clear(self, event: HookEvent | None = None) -> None:
        """Clear hooks. If event is None, clears all hooks."""
        if event is None:
            self._hooks.clear()
        elif event in self._hooks:
            del self._hooks[event]

    def get_handlers(self, event: HookEvent) -> list[HookHandler]:
        """Get handlers for an event, sorted by priority."""
        return [h for _, h in self._hooks.get(event, [])]

    async def fire(self, event: HookEvent, **kwargs: Any) -> HookResult:
        """Fire all handlers for an event, merge results.

        Runs hooks sequentially, merges results (later hooks override earlier ones).

        Args:
            event: The event to fire.
            **kwargs: Context passed to handlers (tool_name, tool_args, etc.)

        Returns:
            Merged HookResult from all handlers.
        """
        handlers = self.get_handlers(event)
        if not handlers:
            return HookResult()

        results = []
        for handler in handlers:
            try:
                result = await handler(**kwargs)
                results.append(result)
            except Exception as e:
                # Hooks should not crash the agent — log and continue
                print(f"  [hook] Error in {event.value} handler: {e}")
                results.append(HookResult())

        return HookResult.merge(*results)

    def list_hooks(self) -> dict[str, list[int]]:
        """List all registered hooks for debugging."""
        return {
            event.value: [p for p, _ in handlers]
            for event, handlers in self._hooks.items()
        }
