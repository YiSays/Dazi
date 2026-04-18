"""Autonomous teammate coordination — scan-claim-execute cycle.

KEY CONCEPTS:
  1. Teammates autonomously scan the shared task board for available work
  2. Claiming is atomic: set status=IN_PROGRESS and owner=agent_name
  3. Faster agents naturally claim more tasks (workload rebalancing)
  4. When no tasks are available, agents go idle and notify the leader
  5. The cycle respects task dependencies (blocked_by)
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from textwrap import dedent
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from dazi.base import DaziTool, ToolSafety
from dazi.mailbox import Mailbox
from dazi.protocols import create_idle_notification
from dazi.task_store import Task, TaskStatus, TaskStore
from dazi.teammate import TeammateHandle, TeammateRunner

# ─────────────────────────────────────────────────────────
# AUTONOMOUS CONFIGURATION
# ─────────────────────────────────────────────────────────


@dataclass
class AutonomousConfig:
    """Configuration for autonomous teammate behavior.

    Attributes:
        max_tasks_per_agent: Maximum tasks a single agent can claim.
                             Prevents task hoarding.
        claim_delay: Seconds to wait between claim scans.
                     Prevents aggressive polling.
        idle_timeout: Seconds with no work before sending idle notification.
        max_turns_per_task: Maximum agent turns per task execution.
                            Prevents infinite agent loops.
    """

    max_tasks_per_agent: int = 10
    claim_delay: float = 1.0
    idle_timeout: float = 30.0
    max_turns_per_task: int = 50


# ─────────────────────────────────────────────────────────
# AUTONOMOUS TEAMMATE
# ─────────────────────────────────────────────────────────


class AutonomousTeammate(TeammateRunner):
    """Extended TeammateRunner with autonomous task claiming.

    Lifecycle: idle -> check inbox -> scan TaskList -> claim -> execute -> report -> idle

    This extends the base TeammateRunner by adding:
    - scan_tasks(): find available work on the task board
    - execute_claimed_task(): run work and update status
    - run_autonomous_cycle(): main autonomous loop
    """

    def __init__(self) -> None:
        super().__init__()
        self._tasks_claimed: dict[str, int] = {}  # agent_id -> count

    # ── Task Scanning ────────────────────────────────────

    def scan_tasks(
        self,
        task_store: TaskStore,
        agent_name: str,
        max_tasks: int = 10,
    ) -> Task | None:
        """Scan task board for available work and return the first claimable task.

        Finds the first task that is:
        1. Status PENDING
        2. No active blockers (all blocked_by tasks are COMPLETED)
        3. Not already claimed by this agent beyond max_tasks limit

        Does NOT claim the task — call claim_task() separately for atomicity.

        Args:
            task_store: Task store to scan.
            agent_name: Name of the agent looking for work.
            max_tasks: Max tasks this agent is allowed to claim.

        Returns:
            The first available Task, or None if no work available.
        """
        # Check if agent has reached task limit
        claimed_count = self._tasks_claimed.get(agent_name, 0)
        if claimed_count >= max_tasks:
            return None

        tasks = task_store.list_all()

        for task in tasks:
            if task.status != TaskStatus.PENDING:
                continue

            # Check for active blockers
            active_blockers = task_store.get_active_blockers(task.id)
            if active_blockers:
                continue

            # Found an unblocked, pending task
            return task

        return None

    # ── Task Claiming ────────────────────────────────────

    def claim_task(
        self,
        task_store: TaskStore,
        task: Task,
        agent_name: str,
    ) -> Task | None:
        """Atomically claim a task for an agent.

        Sets status=IN_PROGRESS and owner=agent_name.

        Args:
            task_store: Task store to update.
            task: The task to claim.
            agent_name: Name of the claiming agent.

        Returns:
            The updated task, or None if claim failed (task no longer pending).
        """
        # Re-read to ensure task is still pending (atomicity)
        fresh = task_store.get(task.id)
        if fresh is None or fresh.status != TaskStatus.PENDING:
            return None

        updated = task_store.update(
            task.id,
            status=TaskStatus.IN_PROGRESS,
            owner=agent_name,
        )

        if updated is not None:
            self._tasks_claimed[agent_name] = self._tasks_claimed.get(agent_name, 0) + 1

        return updated

    # ── Task Execution ───────────────────────────────────

    async def execute_claimed_task(
        self,
        task_store: TaskStore,
        task: Task,
        run_func: Callable[..., Coroutine[Any, Any, str]],
    ) -> str:
        """Execute a claimed task using the provided run function.

        Wraps the run function in try/except to always update task status.
        On success: status=COMPLETED. On failure: status back to PENDING, owner cleared.

        Args:
            task_store: Task store to update.
            task: The task to execute.
            run_func: Async function that performs the work. Should return a result string.

        Returns:
            Result string from run_func, or error message.
        """
        try:
            result = await run_func(task)
            task_store.update(task.id, status=TaskStatus.COMPLETED)
            return result
        except asyncio.CancelledError:
            # Task was cancelled — reset so another agent can pick it up
            task_store.update(task.id, status=TaskStatus.PENDING, owner=None)
            raise
        except Exception as e:
            # Task failed — reset so another agent can retry
            task_store.update(task.id, status=TaskStatus.PENDING, owner=None)
            # Decrement claim count since this task is being released
            if task.owner in self._tasks_claimed:
                self._tasks_claimed[task.owner] = max(0, self._tasks_claimed[task.owner] - 1)
            return f"Task {task.id} failed: {e}"

    # ── Autonomous Cycle ─────────────────────────────────

    async def run_autonomous_cycle(
        self,
        team_name: str,
        agent_name: str,
        task_store: TaskStore,
        mailbox: Mailbox | None = None,
        config: AutonomousConfig | None = None,
        run_func: Callable[..., Coroutine[Any, Any, str]] | None = None,
    ) -> None:
        """Main autonomous loop: scan -> claim -> execute -> report -> idle.

        Runs continuously until:
        - abort_signal is set (external shutdown)
        - asyncio.Task is cancelled

        When no work is found for idle_timeout seconds, sends idle notification.

        Args:
            team_name: Team this agent belongs to.
            agent_name: Name of this agent.
            task_store: Shared task store to scan.
            mailbox: Optional mailbox for sending idle notifications.
            config: Autonomous behavior configuration.
            run_func: Async function for executing claimed tasks.
                     Receives a Task, returns a result string.
        """
        if config is None:
            config = AutonomousConfig()
        if run_func is None:
            run_func = _default_run_func

        agent_id = f"{agent_name}@{team_name}"
        idle_since: float | None = None

        while True:
            # Check if we've been asked to shut down
            handle = self.get_handle(agent_id)
            if handle and handle.abort_signal.is_set():
                break

            # Try to find and claim work
            task = self.scan_tasks(task_store, agent_name, config.max_tasks_per_agent)

            if task is not None:
                claimed = self.claim_task(task_store, task, agent_name)
                if claimed is not None:
                    idle_since = None  # Reset idle timer
                    await self.execute_claimed_task(task_store, claimed, run_func)
                    continue

            # No work found — track idle time
            import time

            now = time.monotonic()

            if idle_since is None:
                idle_since = now

            # Send idle notification after timeout
            if (now - idle_since) >= config.idle_timeout:
                if mailbox is not None:
                    notification = create_idle_notification(
                        from_agent=agent_name,
                        idle_reason="no_pending_work",
                    )
                    await mailbox.send(team_name, notification)
                idle_since = now  # Reset to avoid spamming

            # Wait before next scan
            await asyncio.sleep(config.claim_delay)

    # ── Convenience Spawn ────────────────────────────────

    def spawn_autonomous(
        self,
        team_name: str,
        member_name: str,
        task_store: TaskStore,
        mailbox: Mailbox | None = None,
        config: AutonomousConfig | None = None,
        run_func: Callable[..., Coroutine[Any, Any, str]] | None = None,
        agent_type: str = "general-purpose",
    ) -> asyncio.Task[Any]:
        """Spawn an autonomous teammate that runs the scan-claim-execute cycle.

        Convenience method that combines spawn() with run_autonomous_cycle().

        Args:
            team_name: Team to spawn agent into.
            member_name: Name for the new teammate.
            task_store: Shared task store to scan.
            mailbox: Optional mailbox for idle notifications.
            config: Autonomous behavior configuration.
            run_func: Async function for task execution.
            agent_type: Type of agent (for tool scoping).

        Returns:
            The asyncio.Task running the autonomous cycle.
        """

        async def _autonomous_run(handle: TeammateHandle) -> None:
            await self.run_autonomous_cycle(
                team_name=team_name,
                agent_name=member_name,
                task_store=task_store,
                mailbox=mailbox,
                config=config,
                run_func=run_func,
            )

        return self.spawn(
            team_name=team_name,
            member_name=member_name,
            agent_type=agent_type,
            run_func=_autonomous_run,
        )

    # ── Reset ────────────────────────────────────────────

    def reset(self) -> None:
        """Clear all tracked teammates and claim counts. For testing only."""
        super().reset() if hasattr(super(), "reset") else None
        self._tasks_claimed.clear()


# ─────────────────────────────────────────────────────────
# DEFAULT RUN FUNCTION
# ─────────────────────────────────────────────────────────


async def _default_run_func(task: Task) -> str:
    """Default task execution function.

    In production, this would invoke the LLM agent.
    For testing and demonstration, it simply marks the task as done.
    """
    await asyncio.sleep(0.1)  # Simulate work
    return f"Completed task {task.id}: {task.subject}"


# ─────────────────────────────────────────────────────────
# DELEGATE TASK TOOL
# ─────────────────────────────────────────────────────────

SUB_AGENT_PROMPT = """\
You are a sub-agent of Dazi. You have been delegated a specific task.
Complete the task using the available tools. Be thorough and concise.

When done, provide a summary of what you found or accomplished.
Do NOT ask for user input — make decisions autonomously within your scope."""


class DelegateTaskInput(BaseModel):
    task: str = Field(description="The task for the sub-agent to complete")
    max_turns: int = Field(default=5, description="Maximum number of LLM turns")
    allowed_tools: list[str] | None = Field(
        default=None,
        description="Optional list of tool names the sub-agent can use. None = all parent tools.",
    )


def delegate_task(task: str, max_turns: int = 5, allowed_tools: list[str] | None = None) -> str:
    """Delegate a task to a sub-agent with fresh context."""
    from dazi.filesystem import calculator_tool, file_reader_tool, shell_exec_tool
    from dazi.llm import create_llm

    all_tools = [file_reader_tool, calculator_tool, shell_exec_tool]
    all_tool_names = {t.name for t in all_tools}

    if allowed_tools:
        sub_tools = [t for t in all_tools if t.name in allowed_tools]
        blocked = set(allowed_tools) - all_tool_names
        if blocked:
            return f"Error: Unknown tools requested: {blocked}"
    else:
        sub_tools = all_tools

    llm = create_llm().bind_tools(sub_tools)
    messages = [
        HumanMessage(content=f"{SUB_AGENT_PROMPT}\n\nYour task: {task}"),
    ]

    turn_count = 0
    while turn_count < max_turns:
        turn_count += 1
        try:
            response = llm.invoke(messages)
            messages.append(response)

            if not response.tool_calls:
                break

            for tc in response.tool_calls:
                tool = next((t for t in sub_tools if t.name == tc["name"]), None)
                if tool:
                    result = tool.invoke(tc["args"])
                else:
                    result = f"Error: Tool '{tc['name']}' not available"
                messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))

        except Exception as e:
            messages.append(SystemMessage(content=f"Sub-agent error: {e}"))
            break

    final_msg = ""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            break
        if hasattr(msg, "content") and isinstance(msg.content, str) and msg.content:
            final_msg = msg.content
            if "error" in final_msg.lower() and isinstance(msg, SystemMessage):
                final_msg = f"Sub-agent error: {msg.content}"
                break

    turn_info = f" (completed in {turn_count} turn{'s' if turn_count != 1 else ''})"
    if not final_msg:
        return f"Sub-agent did not produce a response{turn_info}."
    return f"[Sub-agent{turn_info}]\n{final_msg}"


delegate_task_tool = StructuredTool.from_function(
    func=delegate_task,
    name="delegate_task",
    description=dedent("""\
        Delegate a task to a sub-agent with fresh context.
        The sub-agent starts with no knowledge of the parent conversation.
        Use for: research tasks, independent analysis, parallel work.
        The sub-agent returns a summary when done.""").strip(),
    args_schema=DelegateTaskInput,
)
delegate_task_meta = DaziTool(
    name="delegate_task", description="Delegate a task to a sub-agent.", safety=ToolSafety.SAFE
)


# ─────────────────────────────────────────────────────────
# SPAWN AGENT TOOL (autonomous version — final)
# ─────────────────────────────────────────────────────────


class AgentSpawnInput(BaseModel):
    team_name: str = Field(description="Name of the team to spawn the agent into")
    member_name: str = Field(description="Name for the new teammate (e.g., 'frontend', 'backend')")
    agent_type: str = Field(
        default="general-purpose",
        description="Type of agent: 'general-purpose', 'explore', or 'plan'",
    )
    initial_task: str = Field(
        default="", description="Optional initial task description for the teammate"
    )


async def spawn_agent_func(
    team_name: str,
    member_name: str,
    agent_type: str = "general-purpose",
    initial_task: str = "",
) -> str:
    """Spawn an autonomous teammate that scans for and claims tasks."""
    from dazi._singletons import autonomous_teammate, team_manager
    from dazi.config import DATA_DIR
    from dazi.team import TeamMember
    from dazi.task_store import TaskStore

    TASKS_DIR = DATA_DIR / "tasks"

    config = AutonomousConfig()
    task_store = TaskStore(TASKS_DIR, list_id=team_name)

    agent_id = f"{member_name}@{team_name}"
    team_manager.add_member(team_name, TeamMember(
        name=member_name,
        agent_id=agent_id,
        agent_type=agent_type,
    ))

    task = autonomous_teammate.spawn_autonomous(
        team_name=team_name,
        member_name=member_name,
        task_store=task_store,
        config=config,
        agent_type=agent_type,
    )

    return (
        f"Spawned autonomous teammate '{member_name}' in team '{team_name}' "
        f"(type: {agent_type}, task: {task.get_name()})"
        + (f"\nInitial task: {initial_task}" if initial_task else "")
    )


spawn_agent_tool = StructuredTool.from_function(
    func=lambda **kwargs: "",
    coroutine=spawn_agent_func,
    name="spawn_agent",
    description=(
        "Spawn an autonomous teammate that will scan the task board, "
        "claim available work, and execute it. Requires a team to exist."
    ),
    args_schema=AgentSpawnInput,
)
spawn_agent_meta = DaziTool(
    name="spawn_agent", description="Spawn an autonomous teammate.", safety=ToolSafety.WRITE
)
