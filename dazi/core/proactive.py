"""Proactive mode state management.

Tracks active/paused/inactive state for tick-based autonomous operation.

The proactive tick system sends periodic <tick> messages that keep the agent
"awake" between turns, allowing autonomous investigation and work without
waiting for user input.

State machine:
    INACTIVE ──activate()──> ACTIVE ──pause()──> PAUSED
    INACTIVE <──deactivate()< ACTIVE <──resume()── PAUSED

Usage:
    from dazi.core.proactive import proactive_manager, format_tick
    proactive_manager.activate()
    if proactive_manager.should_generate_tick():
        msg = format_tick()  # "<tick>14:30:00</tick>"
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Callable

from dazi.core.base import DaziTool, ToolSafety


class ProactiveSource(str, Enum):
    """How proactive mode was activated."""
    COMMAND = "command"    # /proactive on
    ENV = "env"            # DAZI_PROACTIVE=1


class ProactiveState(str, Enum):
    """Current proactive state."""
    INACTIVE = "inactive"
    ACTIVE = "active"
    PAUSED = "paused"


@dataclass
class ProactiveManager:
    """Track proactive mode state: active/inactive/paused."""
    _active: bool = False
    _paused: bool = False
    _activation_count: int = 0
    _first_tick_pending: bool = False
    _source: ProactiveSource | None = None
    _last_tick_time: str | None = None
    _subscribers: list[Callable[[ProactiveState, ProactiveState], None]] = field(
        default_factory=list
    )

    @property
    def state(self) -> ProactiveState:
        if not self._active:
            return ProactiveState.INACTIVE
        if self._paused:
            return ProactiveState.PAUSED
        return ProactiveState.ACTIVE

    def is_proactive_active(self) -> bool:
        """Check if proactive mode is enabled (active or paused)."""
        return self._active

    def is_proactive_paused(self) -> bool:
        """Check if proactive mode is temporarily paused."""
        return self._paused

    def should_generate_tick(self) -> bool:
        """Whether a tick should be generated."""
        return self._active and not self._paused

    def activate(self, source: ProactiveSource = ProactiveSource.COMMAND) -> None:
        """Activate proactive mode.

        Sets first_tick_pending so the first wake-up triggers a greeting.
        """
        old = self.state
        self._active = True
        self._paused = False
        self._activation_count += 1
        self._first_tick_pending = True
        self._source = source
        self._last_tick_time = None
        self._notify(old, self.state)

    def deactivate(self) -> None:
        """Deactivate proactive mode completely."""
        old = self.state
        self._active = False
        self._paused = False
        self._first_tick_pending = False
        self._source = None
        self._last_tick_time = None
        self._notify(old, self.state)

    def pause(self) -> None:
        """Pause tick generation.

        Called when user presses Escape or Ctrl+C to regain control.
        """
        if self._active and not self._paused:
            old = self.state
            self._paused = True
            self._notify(old, self.state)

    def resume(self) -> None:
        """Resume tick generation.

        Called when user submits input after pausing.
        Re-sets first_tick_pending so agent re-greets after resume.
        """
        if self._active and self._paused:
            old = self.state
            self._paused = False
            self._first_tick_pending = True
            self._notify(old, self.state)

    def mark_tick_sent(self) -> None:
        """Record that a tick was generated."""
        self._first_tick_pending = False
        self._last_tick_time = datetime.now().strftime("%H:%M:%S")

    @property
    def is_first_tick(self) -> bool:
        """Whether the next tick will be the first one (activation or resume)."""
        return self._first_tick_pending

    @property
    def activation_count(self) -> int:
        return self._activation_count

    @property
    def source(self) -> ProactiveSource | None:
        return self._source

    @property
    def last_tick_time(self) -> str | None:
        return self._last_tick_time

    def subscribe(
        self, callback: Callable[[ProactiveState, ProactiveState], None]
    ) -> None:
        """Subscribe to state changes. Callback receives (old_state, new_state)."""
        self._subscribers.append(callback)

    def _notify(
        self, old_state: ProactiveState, new_state: ProactiveState
    ) -> None:
        for cb in self._subscribers:
            try:
                cb(old_state, new_state)
            except Exception:
                pass

    def reset(self) -> None:
        """Reset all state (for testing)."""
        self._active = False
        self._paused = False
        self._activation_count = 0
        self._first_tick_pending = False
        self._source = None
        self._last_tick_time = None
        self._subscribers.clear()


def format_tick() -> str:
    """Format a tick message with local timestamp.

    Format: <tick>HH:MM:SS</tick>
    """
    now = datetime.now().strftime("%H:%M:%S")
    return f"<tick>{now}</tick>"


# ─────────────────────────────────────────────────────────
# SLEEP TOOL
# ─────────────────────────────────────────────────────────

import asyncio
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field


class SleepInput(BaseModel):
    seconds: float = Field(description="Number of seconds to sleep (0.1 to 300)", ge=0.1, le=300)


async def sleep_func(seconds: float) -> str:
    """Non-blocking sleep for pacing between actions."""
    await asyncio.sleep(seconds)
    return f"Slept for {seconds:.1f}s"


sleep_tool = StructuredTool.from_function(
    func=lambda **kwargs: "",
    coroutine=sleep_func,
    name="sleep",
    description="Non-blocking sleep for pacing between actions. Use instead of shell_exec('sleep N'). Range: 0.1 to 300 seconds.",
    args_schema=SleepInput,
)
sleep_meta = DaziTool(name="sleep", description="Non-blocking sleep for pacing.", safety=ToolSafety.SAFE)
