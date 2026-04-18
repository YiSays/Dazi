"""Persistent task storage with file-based JSON per task.

KEY CONCEPTS:
  1. Each task is a separate JSON file (prevents total corruption from single-file failure)
  2. IDs derived from max existing file — reset when all tasks removed
  3. Bidirectional dependencies: add_block() updates both tasks
  4. Task lifecycle: pending -> in_progress -> completed
  5. status="deleted" is a special sentinel that removes the task file (not a real status)
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from textwrap import dedent
from typing import Any

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field, field_validator

from dazi.base import DaziTool, ToolSafety

# ─────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────

DEFAULT_LIST_ID = "default"


# ─────────────────────────────────────────────────────────
# TASK STATUS
# ─────────────────────────────────────────────────────────
# Note: "deleted" is NOT a real status — it triggers file removal on update


class TaskStatus(StrEnum):
    """Task lifecycle states."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"


# ─────────────────────────────────────────────────────────
# TASK DATACLASS
# ─────────────────────────────────────────────────────────


@dataclass
class Task:
    """A single task on the task board.

    Storage: one JSON file per task at <tasks_dir>/<list_id>/<task_id>.json
    """

    id: int
    subject: str
    description: str
    status: TaskStatus = TaskStatus.PENDING
    active_form: str = ""
    owner: str | None = None
    blocks: list[int] = field(default_factory=list)
    blocked_by: list[int] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict[str, Any]:
        """Serialize task to a JSON-compatible dict."""
        return {
            "id": self.id,
            "subject": self.subject,
            "description": self.description,
            "status": self.status.value,
            "active_form": self.active_form,
            "owner": self.owner,
            "blocks": self.blocks,
            "blocked_by": self.blocked_by,
            "metadata": self.metadata,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Task:
        """Deserialize a task from a dict."""
        return cls(
            id=data["id"],
            subject=data["subject"],
            description=data["description"],
            status=TaskStatus(data["status"]),
            active_form=data.get("active_form", ""),
            owner=data.get("owner"),
            blocks=data.get("blocks", []),
            blocked_by=data.get("blocked_by", []),
            metadata=data.get("metadata", {}),
            created_at=data.get("created_at", datetime.now().isoformat()),
        )


# ─────────────────────────────────────────────────────────
# TASK STORE
# ─────────────────────────────────────────────────────────


class TaskStore:
    """Persistent task storage with file-based JSON per task.

    Storage layout:
        <tasks_base_dir>/<list_id>/
        ├── 1.json         — task files
        ├── 2.json
        └── ...
    """

    def __init__(self, tasks_base_dir: Path, list_id: str = DEFAULT_LIST_ID) -> None:
        self.tasks_base_dir = tasks_base_dir
        self.list_id = list_id
        self._list_dir = tasks_base_dir / list_id
        self._lock = threading.Lock()

    # ── ID Generation ────────────────────────────────────
    # IDs are max(existing_file_ids) + 1.
    # When all task files are removed, IDs reset to 1.

    def _next_id(self) -> int:
        """Get the next task ID from the max existing task ID.

        If no tasks exist, starts from 1. This means IDs reset when
        all tasks are deleted or completed.
        """
        self._list_dir.mkdir(parents=True, exist_ok=True)

        max_id = 0
        for path in self._list_dir.glob("*.json"):
            try:
                max_id = max(max_id, int(path.stem))
            except ValueError:
                continue

        return max_id + 1

    # ── CRUD Operations ──────────────────────────────────

    def create(
        self,
        subject: str,
        description: str,
        active_form: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> Task:
        """Create a new task with status 'pending'."""
        with self._lock:
            task_id = self._next_id()
            task = Task(
                id=task_id,
                subject=subject,
                description=description,
                active_form=active_form,
                metadata=metadata or {},
            )
            self._write_task(task)
            return task

    def get(self, task_id: int) -> Task | None:
        """Retrieve a task by ID."""
        path = self._list_dir / f"{task_id}.json"
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return Task.from_dict(data)
        except (json.JSONDecodeError, KeyError):
            return None

    def update(self, task_id: int, **kwargs: Any) -> Task | None:
        """Update fields on an existing task.

        Allowed fields: subject, description, active_form, status, owner, metadata
        """
        with self._lock:
            task = self._get_unlocked(task_id)
            if task is None:
                return None

            allowed_fields = {
                "subject",
                "description",
                "active_form",
                "status",
                "owner",
                "metadata",
            }
            for key, value in kwargs.items():
                if key in allowed_fields:
                    setattr(task, key, value)

            self._write_task(task)
            return task

    def delete(self, task_id: int) -> bool:
        """Delete a task by removing its file.

        Note: Does NOT cascade (other tasks' blocks/blocked_by may still reference it).
        """
        with self._lock:
            path = self._list_dir / f"{task_id}.json"
            if path.exists():
                path.unlink()
                return True
            return False

    def list_all(self) -> list[Task]:
        """List all tasks, sorted by ID ascending."""
        if not self._list_dir.exists():
            return []

        tasks = []
        for path in self._list_dir.glob("*.json"):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                tasks.append(Task.from_dict(data))
            except (json.JSONDecodeError, KeyError):
                continue

        tasks.sort(key=lambda t: t.id)
        return tasks

    # ── Dependency Management ────────────────────────────
    # Bidirectional: both tasks are updated to maintain consistency.
    # All operations acquire self._lock to prevent read-modify-write races
    # when parallel tool calls modify overlapping task files.

    def _get_unlocked(self, task_id: int) -> Task | None:
        """Read a task without acquiring the lock (caller must hold it)."""
        path = self._list_dir / f"{task_id}.json"
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return Task.from_dict(data)
        except (json.JSONDecodeError, KeyError):
            return None

    def add_block(self, task_id: int, blocked_task_id: int) -> tuple[Task | None, Task | None]:
        """Mark task_id as blocking blocked_task_id (bidirectional update).

        Adds blocked_task_id to task.blocks and task_id to blocked_task.blocked_by.
        """
        with self._lock:
            task = self._get_unlocked(task_id)
            blocked_task = self._get_unlocked(blocked_task_id)
            if task is None or blocked_task is None:
                return None, None

            if blocked_task_id not in task.blocks:
                task.blocks.append(blocked_task_id)
                self._write_task(task)

            if task_id not in blocked_task.blocked_by:
                blocked_task.blocked_by.append(task_id)
                self._write_task(blocked_task)

            return task, blocked_task

    def add_blocked_by(
        self, task_id: int, blocking_task_id: int
    ) -> tuple[Task | None, Task | None]:
        """Mark task_id as blocked by blocking_task_id (bidirectional update).

        Adds blocking_task_id to task.blocked_by and task_id to blocking_task.blocks.
        """
        with self._lock:
            task = self._get_unlocked(task_id)
            blocking_task = self._get_unlocked(blocking_task_id)
            if task is None or blocking_task is None:
                return None, None

            if blocking_task_id not in task.blocked_by:
                task.blocked_by.append(blocking_task_id)
                self._write_task(task)

            if task_id not in blocking_task.blocks:
                blocking_task.blocks.append(task_id)
                self._write_task(blocking_task)

            return task, blocking_task

    def get_active_blockers(self, task_id: int) -> list[int]:
        """Get task IDs still blocking this task (excluding completed ones)."""
        task = self.get(task_id)
        if task is None:
            return []

        active = []
        for blocker_id in task.blocked_by:
            blocker = self.get(blocker_id)
            if blocker and blocker.status != TaskStatus.COMPLETED:
                active.append(blocker_id)
        return active

    # ── Internal Helpers ─────────────────────────────────

    def _write_task(self, task: Task) -> None:
        """Write a task to its JSON file."""
        self._list_dir.mkdir(parents=True, exist_ok=True)
        path = self._list_dir / f"{task.id}.json"
        path.write_text(json.dumps(task.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")

    def reset(self) -> None:
        """Remove all tasks. For testing only."""
        if self._list_dir.exists():
            for path in self._list_dir.glob("*.json"):
                path.unlink()


# ─────────────────────────────────────────────────────────
# TASK TOOLS
# ─────────────────────────────────────────────────────────


class TaskCreateInput(BaseModel):
    subject: str = Field(description="A brief title for the task")
    description: str = Field(description="Detailed description of what needs to be done")
    activeForm: str = Field(
        default="",
        description="Present continuous form for progress display (e.g., 'Running tests')",
    )
    metadata: dict[str, object] | None = Field(
        default=None,
        description="Optional arbitrary key-value pairs to attach to the task",
    )


def task_create(
    subject: str, description: str, activeForm: str = "", metadata: dict[str, object] | None = None
) -> str:
    """Create a new task on the task board."""
    from dazi._singletons import get_active_task_store

    store = get_active_task_store()
    task = store.create(
        subject=subject, description=description, active_form=activeForm, metadata=metadata or {}
    )
    return (
        f"Task created: #{task.id}\n"
        f"Subject: {task.subject}\n"
        f"Status: {task.status.value}\n"
        f"Use task_update with taskId={task.id} to change status."
    )


task_create_tool = StructuredTool.from_function(
    func=task_create,
    name="task_create",
    description="Create a new task on the task board. Tasks start in 'pending' status.",
    args_schema=TaskCreateInput,
)
task_create_meta = DaziTool(
    name="task_create", description="Create a new task.", safety=ToolSafety.SAFE
)


class TaskUpdateInput(BaseModel):
    taskId: str = Field(description="The ID of the task to update")
    status: str | None = Field(
        default=None,
        description="New status: pending, in_progress, completed, or 'deleted' to remove",
    )
    subject: str | None = Field(default=None, description="New subject for the task")
    description: str | None = Field(default=None, description="New description")
    activeForm: str | None = Field(
        default=None, description="Present continuous form for progress display"
    )
    owner: str | None = Field(default=None, description="New owner for the task")
    addBlocks: list[str] | None = Field(
        default=None, description="Task IDs that this task should block"
    )
    addBlockedBy: list[str] | None = Field(
        default=None, description="Task IDs that should block this task"
    )
    metadata: dict[str, object] | None = Field(
        default=None,
        description="Metadata keys to merge into the task. Set a key to null to delete it.",
    )

    @field_validator("addBlocks", "addBlockedBy", mode="before")
    @classmethod
    def coerce_list_fields(cls, v: Any) -> list[str] | None:
        """Coerce string representations of lists into actual lists.

        LLM tool calls may serialize lists as strings (e.g., '["19"]' instead of ["19"]).
        """
        if v is None:
            return v
        if isinstance(v, list):
            return v
        if isinstance(v, str):
            try:
                parsed = json.loads(v)
                if isinstance(parsed, list):
                    return [str(item) for item in parsed]
            except (json.JSONDecodeError, ValueError):
                pass
            return [v]
        return v

    @field_validator("status")
    @classmethod
    def validate_status(cls, v: str | None) -> str | None:
        if v is None:
            return v
        valid = {"pending", "in_progress", "completed", "deleted"}
        if v not in valid:
            raise ValueError(f"Invalid status '{v}'. Must be one of: {', '.join(sorted(valid))}")
        return v

    @field_validator("taskId")
    @classmethod
    def validate_task_id(cls, v: str) -> str:
        try:
            int(v)
        except (ValueError, TypeError):
            raise ValueError("taskId must be a valid integer")
        return v


def task_update(
    taskId: str,
    status: str | None = None,
    subject: str | None = None,
    description: str | None = None,
    activeForm: str | None = None,
    owner: str | None = None,
    addBlocks: list[str] | None = None,
    addBlockedBy: list[str] | None = None,
    metadata: dict[str, object] | None = None,
) -> str:
    """Update a task's status, fields, or dependencies."""
    from dazi._singletons import get_active_task_store

    store = get_active_task_store()

    try:
        tid = int(taskId)
    except (ValueError, TypeError):
        return f"Error: Invalid taskId '{taskId}'. Must be an integer."

    if status == "deleted":
        if store.delete(tid):
            return f"Task #{tid} deleted."
        return f"Error: Task #{tid} not found."

    update_kwargs: dict = {}
    if status is not None:
        update_kwargs["status"] = TaskStatus(status)
    if subject is not None:
        update_kwargs["subject"] = subject
    if description is not None:
        update_kwargs["description"] = description
    if activeForm is not None:
        update_kwargs["active_form"] = activeForm
    if owner is not None:
        update_kwargs["owner"] = owner
    if metadata is not None:
        update_kwargs["metadata"] = metadata

    updated_fields: list[str] = list(update_kwargs.keys())

    if update_kwargs:
        task = store.update(tid, **update_kwargs)
        if task is None:
            return f"Error: Task #{tid} not found."
    else:
        task = store.get(tid)
        if task is None:
            return f"Error: Task #{tid} not found."

    if addBlocks:
        for blocked_id_str in addBlocks:
            try:
                store.add_block(tid, int(blocked_id_str))
            except (ValueError, TypeError):
                pass
        updated_fields.append("addBlocks")

    if addBlockedBy:
        for blocking_id_str in addBlockedBy:
            try:
                store.add_blocked_by(tid, int(blocking_id_str))
            except (ValueError, TypeError):
                pass
        updated_fields.append("addBlockedBy")

    task = store.get(tid)
    if task is None:
        return f"Error: Task #{tid} not found after update."

    status_change = ""
    if status and status != "deleted":
        status_change = f"\nStatus: {task.status.value}"

    fields_str = ", ".join(updated_fields) if updated_fields else "none"
    return (
        f"Task #{task.id} updated ({fields_str}).\n"
        f"Subject: {task.subject}{status_change}\n"
        f"Blocks: {task.blocks or 'none'}\n"
        f"Blocked by: {task.blocked_by or 'none'}"
    )


task_update_tool = StructuredTool.from_function(
    func=task_update,
    name="task_update",
    description=dedent("""\
        Update a task's status, fields, or dependencies.
        Status transitions: pending -> in_progress -> completed.
        Use status='deleted' to remove a task entirely.
        Use addBlocks/addBlockedBy to set up dependency chains.""").strip(),
    args_schema=TaskUpdateInput,
)
task_update_meta = DaziTool(
    name="task_update",
    description="Update a task's status or dependencies.",
    safety=ToolSafety.SAFE,
)


class TaskListInput(BaseModel):
    pass


def task_list() -> str:
    """List all tasks with summary info."""
    from dazi._singletons import get_active_task_store

    store = get_active_task_store()
    tasks = store.list_all()
    if not tasks:
        return "No tasks found."

    lines = []
    for task in tasks:
        active_blockers = store.get_active_blockers(task.id)
        owner_str = f" (owner: {task.owner})" if task.owner else ""
        blocked_str = f" [blocked by: {active_blockers}]" if active_blockers else ""
        lines.append(f"  #{task.id} [{task.status.value}]{owner_str} {task.subject}{blocked_str}")

    return f"Tasks ({len(tasks)}):\n" + "\n".join(lines)


task_list_tool = StructuredTool.from_function(
    func=task_list,
    name="task_list",
    description="List all tasks on the task board. Shows ID, status, owner, and active blockers.",
    args_schema=TaskListInput,
)
task_list_meta = DaziTool(name="task_list", description="List all tasks.", safety=ToolSafety.SAFE)


class TaskGetInput(BaseModel):
    taskId: str = Field(description="The ID of the task to retrieve")

    @field_validator("taskId")
    @classmethod
    def validate_task_id(cls, v: str) -> str:
        try:
            int(v)
        except (ValueError, TypeError):
            raise ValueError("taskId must be a valid integer")
        return v


def task_get(taskId: str) -> str:
    """Get full details of a specific task."""
    from dazi._singletons import get_active_task_store

    store = get_active_task_store()

    try:
        tid = int(taskId)
    except (ValueError, TypeError):
        return f"Error: Invalid taskId '{taskId}'. Must be an integer."

    task = store.get(tid)
    if task is None:
        return f"Task not found: #{taskId}"

    lines = [
        f"Task #{task.id}: {task.subject}",
        f"Status: {task.status.value}",
        f"Description: {task.description}",
    ]
    if task.active_form:
        lines.append(f"Active form: {task.active_form}")
    if task.owner:
        lines.append(f"Owner: {task.owner}")
    if task.blocks:
        lines.append(f"Blocks: {task.blocks}")
    if task.blocked_by:
        lines.append(f"Blocked by: {task.blocked_by}")
    if task.metadata:
        lines.append(f"Metadata: {task.metadata}")
    lines.append(f"Created: {task.created_at}")
    return "\n".join(lines)


task_get_tool = StructuredTool.from_function(
    func=task_get,
    name="task_get",
    description="Get full details of a specific task by ID.",
    args_schema=TaskGetInput,
)
task_get_meta = DaziTool(
    name="task_get", description="Get task details by ID.", safety=ToolSafety.SAFE
)
