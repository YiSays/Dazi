"""Tests for dazi.task_store — Task dataclass, TaskStatus enum, TaskStore CRUD & dependencies."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from dazi.task_store import (
    Task,
    TaskCreateInput,
    TaskGetInput,
    TaskListInput,
    TaskStatus,
    TaskStore,
    TaskUpdateInput,
    task_create,
    task_create_tool,
    task_get,
    task_list,
    task_list_tool,
    task_update,
    task_update_tool,
)
from tests.helpers.mock_singletons import patch_singletons

# ─────────────────────────────────────────────────────────
# TaskStatus enum
# ─────────────────────────────────────────────────────────


class TestTaskStatus:
    def test_pending_value(self):
        assert TaskStatus.PENDING.value == "pending"

    def test_in_progress_value(self):
        assert TaskStatus.IN_PROGRESS.value == "in_progress"

    def test_completed_value(self):
        assert TaskStatus.COMPLETED.value == "completed"

    def test_is_string_enum(self):
        assert isinstance(TaskStatus.PENDING, str)


# ─────────────────────────────────────────────────────────
# Task dataclass
# ─────────────────────────────────────────────────────────


class TestTask:
    def test_to_dict_roundtrip(self):
        task = Task(
            id=1,
            subject="Write tests",
            description="Write unit tests for task_store",
            status=TaskStatus.IN_PROGRESS,
            active_form="Writing tests",
            owner="alice",
            blocks=[2, 3],
            blocked_by=[4],
            metadata={"priority": "high"},
            created_at="2025-01-01T00:00:00",
        )
        d = task.to_dict()
        restored = Task.from_dict(d)
        assert restored.id == task.id
        assert restored.subject == task.subject
        assert restored.description == task.description
        assert restored.status == task.status
        assert restored.active_form == task.active_form
        assert restored.owner == task.owner
        assert restored.blocks == task.blocks
        assert restored.blocked_by == task.blocked_by
        assert restored.metadata == task.metadata
        assert restored.created_at == task.created_at

    def test_to_dict_status_is_string(self):
        task = Task(id=1, subject="s", description="d")
        d = task.to_dict()
        assert isinstance(d["status"], str)
        assert d["status"] == "pending"

    def test_from_dict_defaults(self):
        d = {"id": 5, "subject": "s", "description": "d", "status": "pending"}
        task = Task.from_dict(d)
        assert task.active_form == ""
        assert task.owner is None
        assert task.blocks == []
        assert task.blocked_by == []
        assert task.metadata == {}

    def test_from_dict_missing_status_raises(self):
        with pytest.raises(KeyError):
            Task.from_dict({"id": 1, "subject": "s", "description": "d"})


# ─────────────────────────────────────────────────────────
# TaskStore — CRUD
# ─────────────────────────────────────────────────────────


class TestTaskStoreCreate:
    def test_first_task_gets_id_1(self, mock_task_store):
        task = mock_task_store.create("Task A", "First task")
        assert task.id == 1
        assert task.status == TaskStatus.PENDING

    def test_sequential_ids(self, mock_task_store):
        t1 = mock_task_store.create("A", "a")
        t2 = mock_task_store.create("B", "b")
        t3 = mock_task_store.create("C", "c")
        assert t1.id == 1
        assert t2.id == 2
        assert t3.id == 3

    def test_create_with_metadata(self, mock_task_store):
        task = mock_task_store.create("T", "d", metadata={"x": 1})
        assert task.metadata == {"x": 1}

    def test_create_with_active_form(self, mock_task_store):
        task = mock_task_store.create("T", "d", active_form="Testing")
        assert task.active_form == "Testing"

    def test_create_persists_to_disk(self, mock_task_store):
        task = mock_task_store.create("T", "d")
        path = mock_task_store._list_dir / f"{task.id}.json"
        assert path.exists()
        data = json.loads(path.read_text())
        assert data["subject"] == "T"


class TestTaskStoreGet:
    def test_get_existing(self, mock_task_store):
        created = mock_task_store.create("X", "y")
        fetched = mock_task_store.get(created.id)
        assert fetched is not None
        assert fetched.subject == "X"

    def test_get_nonexistent_returns_none(self, mock_task_store):
        assert mock_task_store.get(999) is None


class TestTaskStoreUpdate:
    def test_update_subject(self, mock_task_store):
        task = mock_task_store.create("Old", "desc")
        updated = mock_task_store.update(task.id, subject="New")
        assert updated is not None
        assert updated.subject == "New"

    def test_update_status(self, mock_task_store):
        task = mock_task_store.create("T", "d")
        updated = mock_task_store.update(task.id, status=TaskStatus.IN_PROGRESS)
        assert updated.status == TaskStatus.IN_PROGRESS

    def test_update_nonexistent_returns_none(self, mock_task_store):
        assert mock_task_store.update(999, subject="X") is None


class TestTaskStoreDelete:
    def test_delete_existing_returns_true(self, mock_task_store):
        task = mock_task_store.create("T", "d")
        assert mock_task_store.delete(task.id) is True
        assert mock_task_store.get(task.id) is None

    def test_delete_nonexistent_returns_false(self, mock_task_store):
        assert mock_task_store.delete(999) is False

    def test_ids_reset_after_full_deletion(self, mock_task_store):
        t1 = mock_task_store.create("A", "a")
        t2 = mock_task_store.create("B", "b")
        t3 = mock_task_store.create("C", "c")
        mock_task_store.delete(t1.id)
        mock_task_store.delete(t2.id)
        mock_task_store.delete(t3.id)
        t_new = mock_task_store.create("D", "d")
        assert t_new.id == 1

    def test_ids_continue_after_partial_deletion(self, mock_task_store):
        mock_task_store.create("A", "a")
        t2 = mock_task_store.create("B", "b")
        mock_task_store.create("C", "c")
        mock_task_store.delete(t2.id)
        t_new = mock_task_store.create("D", "d")
        assert t_new.id == 4


class TestTaskStoreListAll:
    def test_list_all_sorted_by_id(self, mock_task_store):
        mock_task_store.create("C", "c")
        mock_task_store.create("A", "a")
        mock_task_store.create("B", "b")
        tasks = mock_task_store.list_all()
        assert [t.subject for t in tasks] == ["C", "A", "B"]
        assert [t.id for t in tasks] == [1, 2, 3]

    def test_list_all_empty(self, mock_task_store):
        assert mock_task_store.list_all() == []


# ─────────────────────────────────────────────────────────
# TaskStore — Dependencies
# ─────────────────────────────────────────────────────────


class TestTaskStoreDependencies:
    def test_add_block_bidirectional(self, mock_task_store):
        t1 = mock_task_store.create("Blocker", "d")
        t2 = mock_task_store.create("Blocked", "d")
        a, b = mock_task_store.add_block(t1.id, t2.id)
        assert t2.id in a.blocks
        assert t1.id in b.blocked_by

    def test_add_block_idempotent(self, mock_task_store):
        t1 = mock_task_store.create("A", "d")
        t2 = mock_task_store.create("B", "d")
        mock_task_store.add_block(t1.id, t2.id)
        mock_task_store.add_block(t1.id, t2.id)
        t1_check = mock_task_store.get(t1.id)
        assert t1_check.blocks.count(t2.id) == 1

    def test_add_block_nonexistent_returns_none_tuple(self, mock_task_store):
        a, b = mock_task_store.add_block(999, 998)
        assert a is None
        assert b is None

    def test_add_blocked_by_reverse(self, mock_task_store):
        t1 = mock_task_store.create("A", "d")
        t2 = mock_task_store.create("B", "d")
        a, b = mock_task_store.add_blocked_by(t2.id, t1.id)
        assert t1.id in a.blocked_by
        assert t2.id in b.blocks

    def test_get_active_blockers_excludes_completed(self, mock_task_store):
        t1 = mock_task_store.create("Blocker", "d")
        t2 = mock_task_store.create("Blocked", "d")
        t3 = mock_task_store.create("Another blocker", "d")
        mock_task_store.add_block(t1.id, t2.id)
        mock_task_store.add_block(t3.id, t2.id)
        # Complete t1
        mock_task_store.update(t1.id, status=TaskStatus.COMPLETED)
        active = mock_task_store.get_active_blockers(t2.id)
        assert t1.id not in active
        assert t3.id in active

    def test_get_active_blockers_nonexistent(self, mock_task_store):
        assert mock_task_store.get_active_blockers(999) == []


# ─────────────────────────────────────────────────────────
# TaskStore — Reset
# ─────────────────────────────────────────────────────────


class TestTaskStoreReset:
    def test_reset_removes_all_tasks(self, mock_task_store):
        mock_task_store.create("A", "a")
        mock_task_store.create("B", "b")
        mock_task_store.reset()
        assert mock_task_store.list_all() == []

    def test_reset_on_empty(self, mock_task_store):
        mock_task_store.reset()  # should not raise
        assert mock_task_store.list_all() == []


# ─────────────────────────────────────────────────────────
# TaskStore — Edge cases / error handling
# ─────────────────────────────────────────────────────────


class TestTaskStoreEdgeCases:
    def test_next_id_ignores_non_numeric_filenames(self, tmp_path):
        store = TaskStore(tmp_path / "tasks", list_id="test")
        store._list_dir.mkdir(parents=True, exist_ok=True)
        # Create a non-numeric file
        (store._list_dir / "badfile.json").write_text("{}")
        (store._list_dir / "3.json").write_text("{}")
        task_id = store._next_id()
        assert task_id == 4

    def test_get_returns_none_on_invalid_json(self, tmp_path):
        store = TaskStore(tmp_path / "tasks", list_id="test")
        store._list_dir.mkdir(parents=True, exist_ok=True)
        (store._list_dir / "1.json").write_text("not valid json {{{")
        assert store.get(1) is None

    def test_get_returns_none_on_missing_keys(self, tmp_path):
        store = TaskStore(tmp_path / "tasks", list_id="test")
        store._list_dir.mkdir(parents=True, exist_ok=True)
        (store._list_dir / "1.json").write_text('{"id": 1}')
        assert store.get(1) is None

    def test_list_all_skips_invalid_files(self, tmp_path):
        store = TaskStore(tmp_path / "tasks", list_id="test")
        store._list_dir.mkdir(parents=True, exist_ok=True)
        # Create a valid task
        store.create("Valid", "d")
        # Create an invalid file
        (store._list_dir / "bad.json").write_text("not json")
        tasks = store.list_all()
        assert len(tasks) == 1
        assert tasks[0].subject == "Valid"

    def test_list_all_skips_files_with_missing_keys(self, tmp_path):
        store = TaskStore(tmp_path / "tasks", list_id="test")
        store._list_dir.mkdir(parents=True, exist_ok=True)
        store.create("Valid", "d")
        (store._list_dir / "incomplete.json").write_text('{"id": 2}')
        tasks = store.list_all()
        assert len(tasks) == 1

    def test_add_blocked_by_one_nonexistent(self, mock_task_store):
        t1 = mock_task_store.create("A", "d")
        a, b = mock_task_store.add_blocked_by(t1.id, 999)
        assert a is None
        assert b is None

    def test_update_ignores_unknown_fields(self, mock_task_store):
        task = mock_task_store.create("T", "d")
        updated = mock_task_store.update(task.id, subject="New", unknown_field="val")
        assert updated is not None
        assert updated.subject == "New"
        assert not hasattr(updated, "unknown_field")

    def test_update_all_allowed_fields(self, mock_task_store):
        task = mock_task_store.create("T", "d")
        updated = mock_task_store.update(
            task.id,
            subject="New Subj",
            description="New Desc",
            active_form="Doing",
            status=TaskStatus.IN_PROGRESS,
            owner="bob",
            metadata={"key": "val"},
        )
        assert updated.subject == "New Subj"
        assert updated.description == "New Desc"
        assert updated.active_form == "Doing"
        assert updated.status == TaskStatus.IN_PROGRESS
        assert updated.owner == "bob"
        assert updated.metadata == {"key": "val"}


# ─────────────────────────────────────────────────────────
# Pydantic Validators
# ─────────────────────────────────────────────────────────


class TestTaskUpdateInputValidators:
    def test_validate_status_valid_values(self):
        for status in ("pending", "in_progress", "completed", "deleted"):
            inp = TaskUpdateInput(taskId="1", status=status)
            assert inp.status == status

    def test_validate_status_none_is_ok(self):
        inp = TaskUpdateInput(taskId="1", status=None)
        assert inp.status is None

    def test_validate_status_invalid_raises(self):
        with pytest.raises(ValidationError):
            TaskUpdateInput(taskId="1", status="invalid_status")

    def test_validate_task_id_valid(self):
        inp = TaskUpdateInput(taskId="42")
        assert inp.taskId == "42"

    def test_validate_task_id_invalid_raises(self):
        with pytest.raises(ValidationError):
            TaskUpdateInput(taskId="not_a_number")

    def test_validate_task_id_empty_raises(self):
        with pytest.raises(ValidationError):
            TaskUpdateInput(taskId="")


class TestTaskGetInputValidator:
    def test_validate_task_id_valid(self):
        inp = TaskGetInput(taskId="42")
        assert inp.taskId == "42"

    def test_validate_task_id_invalid_raises(self):
        with pytest.raises(ValidationError):
            TaskGetInput(taskId="abc")


class TestTaskCreateInput:
    def test_defaults(self):
        inp = TaskCreateInput(subject="S", description="D")
        assert inp.activeForm == ""
        assert inp.metadata is None

    def test_with_all_fields(self):
        inp = TaskCreateInput(subject="S", description="D", activeForm="Doing", metadata={"k": "v"})
        assert inp.activeForm == "Doing"
        assert inp.metadata == {"k": "v"}


class TestTaskListInput:
    def test_creates_empty(self):
        TaskListInput()  # just ensure no validation error


# ─────────────────────────────────────────────────────────
# Tool functions (task_create, task_update, task_list, task_get)
# ─────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _patch_singletons(monkeypatch, tmp_path):
    """Patch singletons for all tool function tests."""
    patch_singletons(monkeypatch, tmp_path)


class TestTaskCreateFunction:
    def test_basic_create(self):
        result = task_create(subject="My task", description="Do something")
        assert "Task created: #1" in result
        assert "Subject: My task" in result
        assert "Status: pending" in result
        assert "task_update" in result

    def test_create_with_active_form(self):
        result = task_create(subject="T", description="D", activeForm="Working")
        assert "Task created" in result

    def test_create_with_metadata(self):
        result = task_create(subject="T", description="D", metadata={"p": 1})
        assert "Task created" in result

    def test_create_second_task(self):
        task_create(subject="First", description="D")
        result = task_create(subject="Second", description="D")
        assert "#2" in result


class TestTaskUpdateFunction:
    def test_update_status(self):
        task_create(subject="T", description="D")
        result = task_update(taskId="1", status="in_progress")
        assert "Task #1 updated" in result
        assert "Status: in_progress" in result

    def test_update_subject(self):
        task_create(subject="Old", description="D")
        result = task_update(taskId="1", subject="New")
        assert "Subject: New" in result

    def test_update_description(self):
        task_create(subject="T", description="Old desc")
        result = task_update(taskId="1", description="New desc")
        assert "Task #1 updated" in result

    def test_update_active_form(self):
        task_create(subject="T", description="D")
        result = task_update(taskId="1", activeForm="Doing things")
        assert "Task #1 updated" in result

    def test_update_owner(self):
        task_create(subject="T", description="D")
        result = task_update(taskId="1", owner="alice")
        assert "Task #1 updated" in result

    def test_update_metadata(self):
        task_create(subject="T", description="D")
        result = task_update(taskId="1", metadata={"priority": "high"})
        assert "Task #1 updated" in result

    def test_delete_task(self):
        task_create(subject="T", description="D")
        result = task_update(taskId="1", status="deleted")
        assert "Task #1 deleted" in result

    def test_delete_nonexistent_task(self):
        result = task_update(taskId="999", status="deleted")
        assert "Error" in result
        assert "not found" in result

    def test_update_nonexistent_task(self):
        result = task_update(taskId="999", status="in_progress")
        assert "Error" in result
        assert "not found" in result

    def test_update_invalid_task_id(self):
        result = task_update(taskId="abc", status="in_progress")
        assert "Error" in result
        assert "Invalid taskId" in result

    def test_update_no_changes(self):
        task_create(subject="T", description="D")
        result = task_update(taskId="1")
        assert "Task #1 updated (none)" in result
        assert "Subject: T" in result

    def test_update_no_changes_nonexistent_task(self):
        result = task_update(taskId="999")
        assert "Error" in result
        assert "Task #999 not found" in result

    def test_task_gone_after_update(self, monkeypatch):
        """Cover the 'not found after update' path (line 455)."""
        from dazi._singletons import task_store as real_store

        # Create a task, then make the store's get return None on re-fetch
        task_create(subject="T", description="D")

        # Patch _get_unlocked so update() succeeds (first call), then patch get()
        # so the re-fetch after addBlockedBy returns None.
        original_get = real_store.get
        call_count = 0

        def flaky_get(tid):
            nonlocal call_count
            call_count += 1
            if call_count <= 1:
                return original_get(tid)
            return None

        # Patch the public get() — the re-fetch at the end of task_update uses this.
        # update() now uses _get_unlocked internally so it isn't affected.
        monkeypatch.setattr(real_store, "get", flaky_get)

        # Use addBlockedBy to trigger the re-fetch path after update
        result = task_update(taskId="1", addBlockedBy=["999"])
        # addBlockedBy with nonexistent ID is silently skipped,
        # then the re-fetch get() returns None (call_count > 1)
        assert "not found after update" in result

    def test_update_with_add_blocks(self):
        task_create(subject="Blocker", description="D")
        task_create(subject="Blocked", description="D")
        result = task_update(taskId="1", addBlocks=["2"])
        assert "addBlocks" in result
        assert "Blocks:" in result

    def test_update_with_add_blocked_by(self):
        task_create(subject="Blocker", description="D")
        task_create(subject="Blocked", description="D")
        result = task_update(taskId="2", addBlockedBy=["1"])
        assert "addBlockedBy" in result
        assert "Blocked by:" in result

    def test_update_add_blocks_invalid_id(self):
        task_create(subject="T", description="D")
        result = task_update(taskId="1", addBlocks=["not_a_number"])
        assert "addBlocks" in result

    def test_update_add_blocked_by_invalid_id(self):
        task_create(subject="T", description="D")
        result = task_update(taskId="1", addBlockedBy=["not_a_number"])
        # Should not crash, just skip the invalid ID
        assert "addBlockedBy" in result

    def test_update_multiple_fields(self):
        task_create(subject="T", description="D")
        result = task_update(
            taskId="1", status="in_progress", subject="New Subject", activeForm="Working"
        )
        assert "in_progress" in result
        assert "New Subject" in result

    def test_update_deleted_status_no_status_change_line(self):
        task_create(subject="T", description="D")
        # deleted status triggers delete, not update
        result = task_update(taskId="1", status="deleted")
        assert "deleted" in result
        # Should NOT have a "Status: deleted" line (it was removed)
        assert "Status: deleted" not in result

    def test_update_with_blocks_and_blocked_by(self):
        task_create(subject="A", description="D")
        task_create(subject="B", description="D")
        task_create(subject="C", description="D")
        result = task_update(taskId="1", addBlocks=["2"], addBlockedBy=["3"])
        assert "addBlocks" in result
        assert "addBlockedBy" in result


class TestTaskListFunction:
    def test_empty_list(self):
        result = task_list()
        assert result == "No tasks found."

    def test_list_with_tasks(self):
        task_create(subject="Task A", description="D")
        result = task_list()
        assert "Tasks (1):" in result
        assert "#1" in result
        assert "Task A" in result

    def test_list_with_owner(self):
        task_create(subject="T", description="D")
        # Manually set owner via update
        task_update(taskId="1", owner="alice")
        result = task_list()
        assert "(owner: alice)" in result

    def test_list_with_active_blockers(self):
        task_create(subject="Blocker", description="D")
        task_create(subject="Blocked", description="D")
        task_update(taskId="2", addBlockedBy=["1"])
        result = task_list()
        assert "[blocked by:" in result


class TestTaskGetFunction:
    def test_get_existing_task(self):
        task_create(subject="My Task", description="A description")
        result = task_get(taskId="1")
        assert "Task #1: My Task" in result
        assert "Status: pending" in result
        assert "Description: A description" in result
        assert "Created:" in result

    def test_get_nonexistent_task(self):
        result = task_get(taskId="999")
        assert "Task not found: #999" in result

    def test_get_invalid_task_id(self):
        result = task_get(taskId="abc")
        assert "Error" in result
        assert "Invalid taskId" in result

    def test_get_with_active_form(self):
        task_create(subject="T", description="D", activeForm="Working")
        result = task_get(taskId="1")
        assert "Active form: Working" in result

    def test_get_with_owner(self):
        task_create(subject="T", description="D")
        task_update(taskId="1", owner="bob")
        result = task_get(taskId="1")
        assert "Owner: bob" in result

    def test_get_with_blocks(self):
        task_create(subject="A", description="D")
        task_create(subject="B", description="D")
        task_update(taskId="1", addBlocks=["2"])
        result = task_get(taskId="1")
        assert "Blocks:" in result

    def test_get_with_blocked_by(self):
        task_create(subject="A", description="D")
        task_create(subject="B", description="D")
        task_update(taskId="2", addBlockedBy=["1"])
        result = task_get(taskId="2")
        assert "Blocked by:" in result

    def test_get_with_metadata(self):
        task_create(subject="T", description="D", metadata={"p": "high"})
        result = task_get(taskId="1")
        assert "Metadata:" in result


# ─────────────────────────────────────────────────────────
# StructuredTool adapter tests
# ─────────────────────────────────────────────────────────


class TestTaskUpdateToolAdapter:
    """Exercise the StructuredTool wrappers, not just the bare Python functions.

    These tests validate that the Pydantic schema + StructuredTool adapter layer
    correctly handles inputs — including edge-case types that LLM callers may produce.
    """

    def test_structured_tool_add_blocked_by(self):
        task_create(subject="Blocker", description="D")
        task_create(subject="Blocked", description="D")
        result = task_update_tool.invoke({"taskId": "2", "addBlockedBy": ["1"]})
        assert "addBlockedBy" in result
        assert "Blocked by:" in result

    def test_structured_tool_add_blocks(self):
        task_create(subject="Blocker", description="D")
        task_create(subject="Blocked", description="D")
        result = task_update_tool.invoke({"taskId": "1", "addBlocks": ["2"]})
        assert "addBlocks" in result
        assert "Blocks:" in result

    def test_structured_tool_string_list_coercion(self):
        """LLM may pass addBlockedBy as a JSON-encoded string '["1"]' instead of ["1"]."""
        task_create(subject="Blocker", description="D")
        task_create(subject="Blocked", description="D")
        result = task_update_tool.invoke({"taskId": "2", "addBlockedBy": '["1"]'})
        assert "addBlockedBy" in result
        assert "Blocked by:" in result

    def test_structured_tool_string_list_coercion_add_blocks(self):
        """LLM may pass addBlocks as a JSON-encoded string '["2"]' instead of ["2"]."""
        task_create(subject="Blocker", description="D")
        task_create(subject="Blocked", description="D")
        result = task_update_tool.invoke({"taskId": "1", "addBlocks": '["2"]'})
        assert "addBlocks" in result
        assert "Blocks:" in result

    def test_structured_tool_end_to_end(self):
        """Create tasks via tool, add deps via tool, verify list shows them."""
        task_create_tool.invoke({"subject": "Setup", "description": "Install deps"})
        task_create_tool.invoke({"subject": "Build", "description": "Build feature"})
        task_update_tool.invoke({"taskId": "1", "addBlocks": ["2"]})
        list_result = task_list_tool.invoke({})
        assert "blocked by:" in list_result


# ─────────────────────────────────────────────────────────
# Team-aware tool function routing
# ─────────────────────────────────────────────────────────


class TestToolFunctionsTeamRouting:
    """Verify tool functions route to the correct store based on team context."""

    @pytest.fixture(autouse=True)
    def _patch_singletons(self, monkeypatch, tmp_path):
        patch_singletons(monkeypatch, tmp_path)

    @pytest.fixture(autouse=True)
    def _reset_team_context(self):
        """Clean up team context between tests."""
        import dazi._singletons as _s

        _s.active_team_name = None
        _s.team_task_store = None
        yield
        _s.active_team_name = None
        _s.team_task_store = None

    def test_create_goes_to_default_when_no_team(self):
        result = task_create(subject="Default task", description="D")
        assert "Task created: #1" in result

    def test_list_empty_when_no_team(self):
        assert "No tasks found." in task_list()

    def test_create_goes_to_team_store_when_active(self, tmp_path):
        import dazi._singletons as _s
        from dazi.task_store import TaskStore

        team_dir = tmp_path / ".dazi" / "tasks" / "my-team"
        team_store = TaskStore(team_dir, list_id="default")
        _s.active_team_name = "my-team"
        _s.team_task_store = team_store

        task_create(subject="Team task", description="D")

        from dazi._singletons import task_store as default_store

        assert default_store.list_all() == []
        assert len(team_store.list_all()) == 1
        assert team_store.list_all()[0].subject == "Team task"

    def test_list_shows_team_tasks_when_active(self, tmp_path):
        import dazi._singletons as _s
        from dazi.task_store import TaskStore

        team_dir = tmp_path / ".dazi" / "tasks" / "my-team"
        team_store = TaskStore(team_dir, list_id="default")
        team_store.create("Team-only task", "D")
        _s.active_team_name = "my-team"
        _s.team_task_store = team_store

        assert "Team-only task" in task_list()

    def test_get_reads_from_team_store(self, tmp_path):
        import dazi._singletons as _s
        from dazi.task_store import TaskStore

        team_dir = tmp_path / ".dazi" / "tasks" / "my-team"
        team_store = TaskStore(team_dir, list_id="default")
        team_store.create("Found task", "Details here")
        _s.active_team_name = "my-team"
        _s.team_task_store = team_store

        result = task_get(taskId="1")
        assert "Found task" in result
        assert "Details here" in result

    def test_update_modifies_team_store(self, tmp_path):
        import dazi._singletons as _s
        from dazi.task_store import TaskStore

        team_dir = tmp_path / ".dazi" / "tasks" / "my-team"
        team_store = TaskStore(team_dir, list_id="default")
        _s.active_team_name = "my-team"
        _s.team_task_store = team_store

        task_create(subject="Original", description="D")
        result = task_update(taskId="1", status="in_progress")
        assert "in_progress" in result
        assert team_store.get(1).status.value == "in_progress"

    def test_returns_to_default_after_deactivation(self, tmp_path):
        import dazi._singletons as _s
        from dazi.task_store import TaskStore

        team_dir = tmp_path / ".dazi" / "tasks" / "my-team"
        team_store = TaskStore(team_dir, list_id="default")
        _s.active_team_name = "my-team"
        _s.team_task_store = team_store

        task_create(subject="Team task", description="D")

        _s.active_team_name = None
        _s.team_task_store = None

        result = task_create(subject="Default task", description="D")
        assert "Task created: #1" in result
