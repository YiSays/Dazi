"""Teammate lifecycle — in-process teammate spawning via asyncio.Task.

KEY CONCEPTS:
  1. Each teammate runs in its own asyncio.Task (context isolation)
  2. Fresh message history per teammate (not shared with parent)
  3. Scoped tool set (from spawn parameters)
  4. Team-aware system prompt (includes team name, member name)
  5. Independent abort signal (can be cancelled without affecting parent)
  6. run_func parameter for testability (tests pass a simple coroutine)
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Coroutine


# ─────────────────────────────────────────────────────────
# TEAMMATE STATUS
# ─────────────────────────────────────────────────────────


class TeammateStatus(str, Enum):
    """Teammate lifecycle states."""
    SPAWNING = "spawning"
    ACTIVE = "active"
    IDLE = "idle"
    SHUTTING_DOWN = "shutting_down"
    COMPLETED = "completed"
    FAILED = "failed"


# ─────────────────────────────────────────────────────────
# TEAMMATE HANDLE
# ─────────────────────────────────────────────────────────


@dataclass
class TeammateHandle:
    """Tracks a spawned teammate's runtime state."""
    name: str                          # "frontend", "backend"
    agent_id: str                      # "frontend@web-dev"
    team_name: str                     # "web-dev"
    status: TeammateStatus = TeammateStatus.SPAWNING
    task: asyncio.Task[Any] | None = None          # The asyncio.Task
    abort_signal: asyncio.Event = field(default_factory=asyncio.Event)


# ─────────────────────────────────────────────────────────
# TEAMMATE RUNNER
# ─────────────────────────────────────────────────────────


class TeammateRunner:
    """Manages teammate lifecycle: spawn, run, shutdown.

    Each teammate runs in its own asyncio.Task with:
    - Fresh message history (context isolation)
    - Scoped tool set (from spawn parameters)
    - Team-aware system prompt (includes team name, member name)
    - Independent abort signal (can be cancelled without affecting parent)
    """

    def __init__(self) -> None:
        self._teammates: dict[str, TeammateHandle] = {}  # keyed by agent_id

    def spawn(
        self,
        team_name: str,
        member_name: str,
        agent_type: str = "general-purpose",
        tools: list[Any] | None = None,
        system_prompt: str = "",
        initial_task: str = "",
        run_func: Callable[..., Coroutine[Any, Any, None]] | None = None,
    ) -> asyncio.Task[Any]:
        """Spawn a teammate as an asyncio.Task.

        Args:
            team_name: Team the teammate belongs to.
            member_name: Name of the teammate (e.g., "frontend").
            agent_type: Type of agent (for tool scoping).
            tools: List of tools available to this teammate.
            system_prompt: Custom system prompt for the teammate.
            initial_task: First task/instruction for the teammate.
            run_func: Optional custom run function for testing.
                      If None, uses default _run() which simulates work.

        Returns:
            The asyncio.Task running the teammate.
        """
        agent_id = f"{member_name}@{team_name}"

        handle = TeammateHandle(
            name=member_name,
            agent_id=agent_id,
            team_name=team_name,
            status=TeammateStatus.SPAWNING,
        )

        task = asyncio.create_task(
            self._run(handle, tools, system_prompt, initial_task, run_func),
            name=f"teammate-{agent_id}",
        )
        handle.task = task
        self._teammates[agent_id] = handle

        return task

    async def _run(
        self,
        handle: TeammateHandle,
        tools: list[Any] | None,
        system_prompt: str,
        initial_task: str,
        run_func: Callable[..., Coroutine[Any, Any, None]] | None,
    ) -> None:
        """Main teammate loop.

        In this stage, the loop is simplified:
        - No actual LLM calls (full autonomy is Stage 15)
        - Processes the initial_task (if run_func provided) or just yields
        - For testing: run_func replaces the default behavior
        """
        handle.status = TeammateStatus.ACTIVE

        try:
            if run_func is not None:
                # Test hook: run the provided coroutine
                await run_func(handle)
            else:
                # Default: simulate work by yielding control
                # Full agent loop (LangGraph invocation) comes in Stage 15
                await asyncio.sleep(0)
                handle.status = TeammateStatus.IDLE

            if handle.status != TeammateStatus.SHUTTING_DOWN:
                handle.status = TeammateStatus.COMPLETED

        except asyncio.CancelledError:
            handle.status = TeammateStatus.COMPLETED
        except Exception:
            handle.status = TeammateStatus.FAILED

    async def shutdown(self, team_name: str, member_name: str) -> bool:
        """Gracefully shut down a teammate.

        1. Find the teammate handle
        2. Set status to SHUTTING_DOWN
        3. Cancel the asyncio.Task
        4. Wait for cancellation (with timeout)
        5. Remove from tracking
        """
        agent_id = f"{member_name}@{team_name}"
        handle = self._teammates.get(agent_id)

        if handle is None or handle.task is None:
            return False

        handle.status = TeammateStatus.SHUTTING_DOWN
        handle.task.cancel()

        try:
            await asyncio.wait_for(handle.task, timeout=5.0)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            pass

        handle.status = TeammateStatus.COMPLETED
        del self._teammates[agent_id]
        return True

    async def shutdown_all(self, team_name: str) -> int:
        """Shutdown all teammates for a team.

        Returns:
            Count of teammates shut down.
        """
        to_shutdown = [
            (h.team_name, h.name)
            for h in self._teammates.values()
            if h.team_name == team_name
        ]

        count = 0
        for tm_name, mb_name in to_shutdown:
            if await self.shutdown(tm_name, mb_name):
                count += 1

        return count

    def get_handle(self, agent_id: str) -> TeammateHandle | None:
        """Get a teammate handle by agent_id."""
        return self._teammates.get(agent_id)

    def list_handles(self) -> list[TeammateHandle]:
        """List all tracked teammate handles."""
        return list(self._teammates.values())

    def list_handles_for_team(self, team_name: str) -> list[TeammateHandle]:
        """List teammate handles for a specific team."""
        return [h for h in self._teammates.values() if h.team_name == team_name]

    def reset(self) -> None:
        """Clear all tracked teammates. For testing only."""
        self._teammates.clear()
