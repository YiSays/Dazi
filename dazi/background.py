"""Background task management with asyncio subprocess execution.

WHAT THIS MODULE PROVIDES:
  - BackgroundTask: in-memory dataclass for tracking a background subprocess
  - BackgroundTaskManager: submit/check/cancel/collect lifecycle
  - File-based output: stdout/stderr appended to <output_dir>/<task_id>.output
  - Notification dedup: collect_completed() returns unnotified tasks only
"""

from __future__ import annotations

import asyncio
import os
import secrets
import signal
import time
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from dazi.base import DaziTool, ToolSafety

# ─────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────

TASK_ID_CHARS = "0123456789abcdef"
TASK_ID_LENGTH = 8
TASK_PREFIX = "bash"

GRACEFUL_SHUTDOWN_TIMEOUT = 5.0  # seconds before SIGKILL


# ─────────────────────────────────────────────────────────
# BACKGROUND TASK STATUS
# ─────────────────────────────────────────────────────────


class BackgroundTaskStatus(StrEnum):
    """Background task lifecycle.

    Lifecycle: pending -> running -> (completed | failed | killed)
    """

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    KILLED = "killed"


# ─────────────────────────────────────────────────────────
# BACKGROUND TASK DATACLASS
# ─────────────────────────────────────────────────────────


@dataclass
class BackgroundTask:
    """In-memory background task representation.

    Key differences from persistent Task (task_store.py):
      - In-memory only (no JSON file for state metadata)
      - Output written to file (<output_dir>/<id>.output)
      - Tracks live asyncio.subprocess.Process
      - Has notified flag for deduplication
    """

    id: str
    command: str
    status: BackgroundTaskStatus = BackgroundTaskStatus.PENDING
    description: str = ""
    output_file: Path = field(default_factory=Path)
    output_offset: int = 0
    pid: int | None = None
    process: asyncio.subprocess.Process | None = field(default=None, repr=False)
    started_at: float | None = None
    completed_at: float | None = None
    exit_code: int | None = None
    error: str | None = None
    notified: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict (excludes process — not JSON-serializable)."""
        return {
            "id": self.id,
            "command": self.command,
            "status": self.status.value,
            "description": self.description,
            "output_file": str(self.output_file),
            "output_offset": self.output_offset,
            "pid": self.pid,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "exit_code": self.exit_code,
            "error": self.error,
            "notified": self.notified,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BackgroundTask:
        """Deserialize from dict (process=None)."""
        return cls(
            id=data["id"],
            command=data["command"],
            status=BackgroundTaskStatus(data.get("status", "pending")),
            description=data.get("description", ""),
            output_file=Path(data.get("output_file", "")),
            output_offset=data.get("output_offset", 0),
            pid=data.get("pid"),
            process=None,  # Cannot deserialize a live process
            started_at=data.get("started_at"),
            completed_at=data.get("completed_at"),
            exit_code=data.get("exit_code"),
            error=data.get("error"),
            notified=data.get("notified", False),
        )

    @property
    def duration_seconds(self) -> float | None:
        """Compute how long the task has been running (or ran)."""
        if self.started_at is None:
            return None
        end = self.completed_at or time.time()
        return end - self.started_at

    @property
    def is_terminal(self) -> bool:
        """Whether the task is in a terminal state."""
        return self.status in (
            BackgroundTaskStatus.COMPLETED,
            BackgroundTaskStatus.FAILED,
            BackgroundTaskStatus.KILLED,
        )


# ─────────────────────────────────────────────────────────
# BACKGROUND TASK MANAGER
# ─────────────────────────────────────────────────────────


class BackgroundTaskManager:
    """Manages background tasks: submit, track, cancel, collect notifications.

    Storage:
      - Tasks stored in self._tasks: dict[str, BackgroundTask] (in-memory)
      - Output written to <output_dir>/<task_id>.output (append-only file)
    """

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self._tasks: dict[str, BackgroundTask] = {}
        self._monitor_tasks: dict[str, asyncio.Task] = {}

    # ─────────────────────────────────────────────────────
    # TASK ID GENERATION
    # ─────────────────────────────────────────────────────

    def _generate_task_id(self) -> str:
        """Generate unique task ID: prefix + 8 random hex chars."""
        random_part = "".join(secrets.choice(TASK_ID_CHARS) for _ in range(TASK_ID_LENGTH))
        return f"{TASK_PREFIX}_{random_part}"

    # ─────────────────────────────────────────────────────
    # SUBMIT — non-blocking task launch
    # ─────────────────────────────────────────────────────

    async def submit(self, command: str, description: str = "") -> str:
        """Submit a command for background execution. Returns task_id immediately.

        1. Generate task ID
        2. Create BackgroundTask with status=PENDING
        3. Create output file
        4. Launch asyncio.create_subprocess_shell()
        5. Start _monitor_task() as asyncio.Task
        6. Return task_id (non-blocking)
        """
        task_id = self._generate_task_id()
        output_file = self.output_dir / f"{task_id}.output"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        task = BackgroundTask(
            id=task_id,
            command=command,
            description=description,
            output_file=output_file,
        )
        self._tasks[task_id] = task

        # Spawn subprocess
        process = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            start_new_session=True,  # Create new process group for clean kill
        )

        task.status = BackgroundTaskStatus.RUNNING
        task.pid = process.pid
        task.process = process
        task.started_at = time.time()

        # Start monitor + output writer as background asyncio tasks
        monitor = asyncio.create_task(self._monitor_task(task_id, process))
        self._monitor_tasks[task_id] = monitor

        return task_id

    # ─────────────────────────────────────────────────────
    # CHECK — poll task status
    # ─────────────────────────────────────────────────────

    async def check(self, task_id: str) -> BackgroundTask | None:
        """Get current status of a background task.

        Returns None if task_id not found.
        """
        return self._tasks.get(task_id)

    def check_sync(self, task_id: str) -> BackgroundTask | None:
        """Synchronous version of check (for REPL display)."""
        return self._tasks.get(task_id)

    # ─────────────────────────────────────────────────────
    # CANCEL — graceful then force kill
    # ─────────────────────────────────────────────────────

    async def cancel(self, task_id: str) -> bool:
        """Cancel a running background task.

        1. Send SIGTERM to process group (graceful)
        2. Wait up to GRACEFUL_SHUTDOWN_TIMEOUT seconds
        3. If still alive, send SIGKILL (force)
        4. Update status to KILLED
        """
        task = self._tasks.get(task_id)
        if task is None:
            return False

        if task.is_terminal:
            return False  # Already done

        process = task.process
        if process is None or process.returncode is not None:
            # Process already exited
            if not task.is_terminal:
                task.status = BackgroundTaskStatus.KILLED
                task.completed_at = time.time()
            return True

        try:
            # Send SIGTERM to entire process group
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            # Process already dead
            pass

        # Wait for graceful shutdown
        try:
            await asyncio.wait_for(process.wait(), timeout=GRACEFUL_SHUTDOWN_TIMEOUT)
        except TimeoutError:
            # Force kill
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
            try:
                await process.wait()
            except Exception:
                pass

        task.status = BackgroundTaskStatus.KILLED
        task.completed_at = time.time()
        task.exit_code = process.returncode
        return True

    # ─────────────────────────────────────────────────────
    # COLLECT COMPLETED — notification dedup
    # ─────────────────────────────────────────────────────

    def collect_completed(self) -> list[BackgroundTask]:
        """Collect tasks that completed but haven't been notified yet.

        Returns tasks where status is terminal and notified=False.
        Sets notified=True on each returned task (marks as delivered).

        Called after each graph cycle in the REPL loop.
        """
        completed = []
        for task in self._tasks.values():
            if task.is_terminal and not task.notified:
                task.notified = True
                completed.append(task)
        return completed

    # ─────────────────────────────────────────────────────
    # OUTPUT READING
    # ─────────────────────────────────────────────────────

    def get_output(self, task_id: str, offset: int = 0) -> str:
        """Read output from a task's output file.

        Args:
            task_id: The task ID
            offset: Byte offset to start reading from (default 0)

        Returns:
            Output string (may be empty if file doesn't exist yet).
        """
        task = self._tasks.get(task_id)
        if task is None:
            return ""

        output_file = task.output_file
        if not output_file.exists():
            return ""

        try:
            content = output_file.read_text(encoding="utf-8", errors="replace")
            return content[offset:]
        except Exception:
            return ""

    def get_output_tail(self, task_id: str, lines: int = 20) -> str:
        """Get the last N lines of output from a task.

        Args:
            task_id: The task ID
            lines: Number of recent lines to return (default 20)

        Returns:
            Last N lines as a string.
        """
        output = self.get_output(task_id)
        if not output:
            return ""

        all_lines = output.splitlines()
        return "\n".join(all_lines[-lines:]) if all_lines else ""

    # ─────────────────────────────────────────────────────
    # LIST ALL
    # ─────────────────────────────────────────────────────

    def list_all(self) -> list[BackgroundTask]:
        """List all tracked tasks, sorted by start time (newest first)."""
        tasks = list(self._tasks.values())
        tasks.sort(key=lambda t: t.started_at or 0, reverse=True)
        return tasks

    def list_active(self) -> list[BackgroundTask]:
        """List only non-terminal (running/pending) tasks."""
        return [t for t in self._tasks.values() if not t.is_terminal]

    # ─────────────────────────────────────────────────────
    # INTERNAL: OUTPUT WRITER
    # ─────────────────────────────────────────────────────

    async def _write_output(self, task_id: str, process: asyncio.subprocess.Process) -> None:
        """Continuously read stdout/stderr and append to output file.

        This runs as an asyncio.Task to prevent pipe buffer deadlocks.
        Without this, a subprocess writing lots of output would block
        when the OS pipe buffer fills up.
        """
        task = self._tasks.get(task_id)
        if task is None:
            return

        try:
            if process.stdout is None:
                return

            while True:
                chunk = await process.stdout.read(4096)
                if not chunk:
                    break

                # Append to output file
                try:
                    with task.output_file.open("ab") as f:
                        f.write(chunk)
                except Exception:
                    pass
        except Exception:
            pass

    # ─────────────────────────────────────────────────────
    # INTERNAL: MONITOR
    # ─────────────────────────────────────────────────────

    async def _monitor_task(self, task_id: str, process: asyncio.subprocess.Process) -> None:
        """Monitor a subprocess until completion.

        Runs as an asyncio.Task spawned by submit().
        Sets status to COMPLETED/FAILED on exit.
        """
        task = self._tasks.get(task_id)
        if task is None:
            return

        try:
            # Run output writer and process wait concurrently
            await asyncio.gather(
                self._write_output(task_id, process),
                process.wait(),
            )

            task.completed_at = time.time()
            task.exit_code = process.returncode

            if task.status == BackgroundTaskStatus.KILLED:
                # Already killed, don't override status
                return

            if process.returncode == 0:
                task.status = BackgroundTaskStatus.COMPLETED
            else:
                task.status = BackgroundTaskStatus.FAILED
                task.error = f"Process exited with code {process.returncode}"
        except asyncio.CancelledError:
            # Monitor was cancelled (e.g., during reset)
            if not task.is_terminal:
                task.status = BackgroundTaskStatus.KILLED
                task.completed_at = time.time()
        except Exception as e:
            if not task.is_terminal:
                task.status = BackgroundTaskStatus.FAILED
                task.completed_at = time.time()
                task.error = str(e)
        finally:
            # Clean up process reference
            task.process = None
            self._monitor_tasks.pop(task_id, None)

    # ─────────────────────────────────────────────────────
    # RESET — for testing only
    # ─────────────────────────────────────────────────────

    def reset(self) -> None:
        """Cancel all running tasks and clear state.

        For testing only. Cancels all monitor asyncio.Tasks.
        """
        # Cancel all monitor tasks
        for task_id, monitor in self._monitor_tasks.items():
            monitor.cancel()
        self._monitor_tasks.clear()

        # Kill any still-running processes
        for task in self._tasks.values():
            if not task.is_terminal and task.process is not None:
                try:
                    os.killpg(os.getpgid(task.process.pid), signal.SIGKILL)
                except (ProcessLookupError, PermissionError, OSError):
                    pass

        self._tasks.clear()


# ─────────────────────────────────────────────────────────
# BACKGROUND TOOLS
# ─────────────────────────────────────────────────────────


class RunBackgroundInput(BaseModel):
    command: str = Field(description="The shell command to execute in the background")
    description: str = Field(
        default="", description="Brief description of what this task does (for display purposes)"
    )


async def run_background(command: str, description: str = "") -> str:
    """Execute a shell command in the background (non-blocking)."""
    from dazi._singletons import background_manager

    task_id = await background_manager.submit(command, description)
    return (
        f"Background task started: {task_id}\n"
        f"Command: {command}\n"
        f"Description: {description or '(none)'}\n"
        f"Use check_background('{task_id}') to monitor progress.\n"
        f"The system will notify you when this task completes."
    )


run_background_tool = StructuredTool.from_function(
    func=run_background,
    name="run_background",
    description=(
        "Execute a shell command in the background. "
        "Returns immediately with a task_id. Use check_background to monitor."
    ),
    args_schema=RunBackgroundInput,
)
run_background_meta = DaziTool(
    name="run_background",
    description="Run a shell command in the background.",
    safety=ToolSafety.WRITE,
)


class CheckBackgroundInput(BaseModel):
    task_id: str = Field(
        description="The ID of the background task to check (e.g., 'bash_a1b2c3d4')"
    )
    tail: int = Field(default=20, description="Number of recent output lines to show (default 20)")


def check_background(task_id: str, tail: int = 20) -> str:
    """Check the status and recent output of a background task."""
    from dazi._singletons import background_manager

    task = background_manager.check_sync(task_id)
    if task is None:
        return f"Error: Background task '{task_id}' not found."

    status_str = f"[{task.status.value}]"
    pid_str = f"PID: {task.pid}" if task.pid else "PID: -"
    duration_str = (
        f"Duration: {task.duration_seconds:.1f}s" if task.duration_seconds else "Duration: -"
    )

    lines = [
        f"Background task {task.id}: {status_str}",
        f"Command: {task.command}",
        pid_str,
        duration_str,
    ]

    if task.exit_code is not None:
        lines.append(f"Exit code: {task.exit_code}")
    if task.error:
        lines.append(f"Error: {task.error}")

    output = background_manager.get_output_tail(task_id, lines=tail)
    if output:
        output_lines = output.splitlines()
        lines.append(f"Output (last {len(output_lines)} lines):")
        lines.append("```")
        lines.append(output)
        lines.append("```")

    return "\n".join(lines)


check_background_tool = StructuredTool.from_function(
    func=check_background,
    name="check_background",
    description="Check the status and recent output of a background task.",
    args_schema=CheckBackgroundInput,
)
check_background_meta = DaziTool(
    name="check_background", description="Check background task status.", safety=ToolSafety.SAFE
)


class CancelBackgroundInput(BaseModel):
    task_id: str = Field(description="The ID of the background task to cancel")


async def cancel_background(task_id: str) -> str:
    """Cancel a running background task."""
    from dazi._singletons import background_manager

    task = background_manager.check_sync(task_id)
    if task is None:
        return f"Error: Background task '{task_id}' not found."

    if task.is_terminal:
        return (
            f"Background task {task_id} is already in terminal state: {task.status.value}.\n"
            f"Cannot cancel a task that has already completed, failed, or been killed."
        )

    success = await background_manager.cancel(task_id)
    if success:
        return f"Background task {task_id} cancelled.\nCommand was: {task.command}"
    return f"Error: Failed to cancel background task {task_id}."


cancel_background_tool = StructuredTool.from_function(
    func=cancel_background,
    name="cancel_background",
    description="Cancel a running background task. Sends SIGTERM, then SIGKILL if needed.",
    args_schema=CancelBackgroundInput,
)
cancel_background_meta = DaziTool(
    name="cancel_background",
    description="Cancel a running background task.",
    safety=ToolSafety.DESTRUCTIVE,
)
