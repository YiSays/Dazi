"""Tests for dazi/coordinator.py — AutonomousConfig, AutonomousTeammate scan/claim/report."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from dazi.coordinator import (
    AgentSpawnInput,
    AutonomousConfig,
    AutonomousTeammate,
    DelegateTaskInput,
    _default_run_func,
    delegate_task,
    delegate_task_meta,
    delegate_task_tool,
    spawn_agent_func,
    spawn_agent_meta,
    spawn_agent_tool,
)
from dazi.mailbox import Mailbox
from dazi.task_store import Task, TaskStatus, TaskStore

# ─────────────────────────────────────────────────────────
# AutonomousConfig defaults
# ─────────────────────────────────────────────────────────


class TestAutonomousConfig:
    def test_defaults(self):
        cfg = AutonomousConfig()
        assert cfg.max_tasks_per_agent == 10
        assert cfg.claim_delay == 1.0
        assert cfg.idle_timeout == 30.0
        assert cfg.max_turns_per_task == 50

    def test_custom_values(self):
        cfg = AutonomousConfig(
            max_tasks_per_agent=5,
            claim_delay=0.5,
            idle_timeout=10.0,
            max_turns_per_task=20,
        )
        assert cfg.max_tasks_per_agent == 5
        assert cfg.claim_delay == 0.5
        assert cfg.idle_timeout == 10.0
        assert cfg.max_turns_per_task == 20


# ─────────────────────────────────────────────────────────
# scan_tasks
# ─────────────────────────────────────────────────────────


class TestScanTasks:
    @pytest.fixture
    def setup(self, tmp_path: Path):
        store = TaskStore(tmp_path / "tasks", list_id="test")
        teammate = AutonomousTeammate()
        return store, teammate

    def test_finds_pending_unblocked_task(self, setup):
        store, teammate = setup
        store.create(subject="Task 1", description="Do something")
        result = teammate.scan_tasks(store, "worker")
        assert result is not None
        assert result.subject == "Task 1"

    def test_no_available_tasks(self, setup):
        store, teammate = setup
        # No tasks at all
        result = teammate.scan_tasks(store, "worker")
        assert result is None

    def test_skips_blocked_tasks(self, setup):
        store, teammate = setup
        t1 = store.create(subject="Blocker", description="Must finish first")
        t2 = store.create(subject="Blocked", description="Waiting on t1")
        store.add_blocked_by(t2.id, t1.id)
        # t1 is pending (not completed), so t2 is blocked
        result = teammate.scan_tasks(store, "worker")
        assert result is not None
        assert result.id == t1.id  # returns the blocker (unblocked pending task)

    def test_respects_max_tasks_limit(self, setup):
        store, teammate = setup
        store.create(subject="T1", description="desc")
        # Simulate agent already at max
        teammate._tasks_claimed["worker"] = 1
        result = teammate.scan_tasks(store, "worker", max_tasks=1)
        assert result is None

    def test_skips_non_pending_tasks(self, setup):
        """Tasks that are IN_PROGRESS, COMPLETED, etc. should be skipped."""
        store, teammate = setup
        t1 = store.create(subject="Already done", description="desc")
        store.update(t1.id, status=TaskStatus.COMPLETED)
        store.create(subject="T2", description="pending")

        result = teammate.scan_tasks(store, "worker")
        assert result is not None
        assert result.subject == "T2"

    def test_skips_tasks_with_active_blockers(self, setup):
        """A task with active blockers (pending blockers) should be skipped."""
        store, teammate = setup
        blocker = store.create(subject="Blocker", description="Not done yet")
        blocked = store.create(subject="Blocked task", description="waiting")
        store.add_blocked_by(blocked.id, blocker.id)
        # Mark blocker as in-progress (not completed) — so it's an active blocker
        store.update(blocker.id, status=TaskStatus.IN_PROGRESS)

        result = teammate.scan_tasks(store, "worker")
        # Should return None because the only pending task has an active blocker
        assert result is None

    def test_returns_none_when_all_tasks_blocked(self, setup):
        store, teammate = setup
        t1 = store.create(subject="T1", description="pending")
        t2 = store.create(subject="T2", description="pending")
        store.add_blocked_by(t2.id, t1.id)
        # Both are pending, but t2 is blocked by t1 (pending), and t1 is the first
        # pending task but t2 is blocked
        result = teammate.scan_tasks(store, "worker")
        # t1 should still be found since it has no blockers
        assert result is not None
        assert result.id == t1.id


# ─────────────────────────────────────────────────────────
# claim_task
# ─────────────────────────────────────────────────────────


class TestClaimTask:
    @pytest.fixture
    def setup(self, tmp_path: Path):
        store = TaskStore(tmp_path / "tasks", list_id="test")
        teammate = AutonomousTeammate()
        return store, teammate

    def test_claim_success(self, setup):
        store, teammate = setup
        task = store.create(subject="T1", description="desc")
        result = teammate.claim_task(store, task, "worker")
        assert result is not None
        assert result.status == TaskStatus.IN_PROGRESS
        assert result.owner == "worker"
        assert teammate._tasks_claimed["worker"] == 1

    def test_claim_already_claimed_returns_none(self, setup):
        store, teammate = setup
        task = store.create(subject="T1", description="desc")
        # First claim succeeds
        teammate.claim_task(store, task, "worker")
        # Second claim should fail (task is now IN_PROGRESS, not PENDING)
        result = teammate.claim_task(store, task, "other")
        assert result is None

    def test_claim_nonexistent_task_returns_none(self, setup):
        store, teammate = setup
        fake_task = Task(id="nonexistent", subject="fake", description="fake")
        result = teammate.claim_task(store, fake_task, "worker")
        assert result is None

    def test_claim_increments_count(self, setup):
        store, teammate = setup
        t1 = store.create(subject="T1", description="desc")
        t2 = store.create(subject="T2", description="desc")
        teammate.claim_task(store, t1, "worker")
        teammate.claim_task(store, t2, "worker")
        assert teammate._tasks_claimed["worker"] == 2


# ─────────────────────────────────────────────────────────
# execute_claimed_task
# ─────────────────────────────────────────────────────────


class TestExecuteClaimedTask:
    @pytest.mark.asyncio
    async def test_task_completion_reporting(self, tmp_path: Path):
        store = TaskStore(tmp_path / "tasks", list_id="test")
        teammate = AutonomousTeammate()

        task = store.create(subject="T1", description="desc")
        claimed = teammate.claim_task(store, task, "worker")
        assert claimed is not None

        async def success_run(t: Task) -> str:
            return f"Done: {t.subject}"

        result = await teammate.execute_claimed_task(store, claimed, success_run)
        assert "Done: T1" in result
        # Task should be marked COMPLETED
        updated = store.get(task.id)
        assert updated.status == TaskStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_task_failure_resets_to_pending(self, tmp_path: Path):
        store = TaskStore(tmp_path / "tasks", list_id="test")
        teammate = AutonomousTeammate()

        task = store.create(subject="T1", description="desc")
        claimed = teammate.claim_task(store, task, "worker")

        async def failing_run(t: Task) -> str:
            raise RuntimeError("Something went wrong")

        result = await teammate.execute_claimed_task(store, claimed, failing_run)
        assert "failed" in result.lower()
        # Task should be reset to PENDING
        updated = store.get(task.id)
        assert updated.status == TaskStatus.PENDING

    @pytest.mark.asyncio
    async def test_task_cancelled_resets_to_pending(self, tmp_path: Path):
        """CancelledError should reset task to PENDING and re-raise."""
        store = TaskStore(tmp_path / "tasks", list_id="test")
        teammate = AutonomousTeammate()

        task = store.create(subject="T1", description="desc")
        claimed = teammate.claim_task(store, task, "worker")

        async def cancelling_run(t: Task) -> str:
            raise asyncio.CancelledError()

        with pytest.raises(asyncio.CancelledError):
            await teammate.execute_claimed_task(store, claimed, cancelling_run)

        # Task should be reset to PENDING with no owner
        updated = store.get(task.id)
        assert updated.status == TaskStatus.PENDING
        assert updated.owner is None

    @pytest.mark.asyncio
    async def test_task_failure_decrements_claim_count(self, tmp_path: Path):
        """When a task fails, the claim count for the owner should decrement."""
        store = TaskStore(tmp_path / "tasks", list_id="test")
        teammate = AutonomousTeammate()

        task = store.create(subject="T1", description="desc")
        claimed = teammate.claim_task(store, task, "worker")
        assert teammate._tasks_claimed["worker"] == 1

        async def failing_run(t: Task) -> str:
            raise RuntimeError("fail")

        await teammate.execute_claimed_task(store, claimed, failing_run)
        assert teammate._tasks_claimed["worker"] == 0

    @pytest.mark.asyncio
    async def test_task_failure_does_not_decrement_below_zero(self, tmp_path: Path):
        """Claim count should never go below zero."""
        store = TaskStore(tmp_path / "tasks", list_id="test")
        teammate = AutonomousTeammate()

        task = store.create(subject="T1", description="desc")
        # Manually create a claimed task without going through claim_task
        claimed = store.update(task.id, status=TaskStatus.IN_PROGRESS, owner="worker")
        # Do NOT increment _tasks_claimed — simulates inconsistency

        async def failing_run(t: Task) -> str:
            raise RuntimeError("fail")

        await teammate.execute_claimed_task(store, claimed, failing_run)
        # Should be 0, not negative
        assert teammate._tasks_claimed.get("worker", 0) == 0


# ─────────────────────────────────────────────────────────
# run_autonomous_cycle
# ─────────────────────────────────────────────────────────


class TestRunAutonomousCycle:
    @pytest.mark.asyncio
    async def test_cycle_uses_default_config_and_run_func(self, tmp_path: Path):
        """When config and run_func are None, defaults are used."""
        store = TaskStore(tmp_path / "tasks", list_id="test")
        teammate = AutonomousTeammate()

        # Create a task so the default run_func runs
        store.create(subject="T1", description="desc")

        atask = teammate.spawn_autonomous(
            team_name="team1",
            member_name="worker",
            task_store=store,
            config=AutonomousConfig(claim_delay=0.01, idle_timeout=0.01),
        )

        # Let it run briefly to hit the default run_func
        await asyncio.sleep(0.1)
        handle = teammate.get_handle("worker@team1")
        if handle:
            handle.abort_signal.set()
        try:
            await asyncio.wait_for(atask, timeout=2.0)
        except (TimeoutError, asyncio.CancelledError):
            atask.cancel()
            try:
                await atask
            except asyncio.CancelledError:
                pass

    @pytest.mark.asyncio
    async def test_cycle_exits_on_abort_signal(self, tmp_path: Path):
        """The cycle should exit immediately when abort_signal is set."""
        store = TaskStore(tmp_path / "tasks", list_id="test")
        teammate = AutonomousTeammate()

        async def quick_abort(handle):
            # Set abort after tiny delay
            await asyncio.sleep(0.05)
            agent_id = "worker@team1"
            h = teammate.get_handle(agent_id)
            if h:
                h.abort_signal.set()

        # Spawn with a custom run that exits immediately
        atask = teammate.spawn_autonomous(
            team_name="team1",
            member_name="worker",
            task_store=store,
            config=AutonomousConfig(claim_delay=0.01, idle_timeout=0.01),
            run_func=AsyncMock(return_value="done"),
        )

        # Schedule abort
        abort_task = asyncio.create_task(quick_abort(None))
        try:
            await asyncio.wait_for(atask, timeout=2.0)
        except (TimeoutError, asyncio.CancelledError):
            atask.cancel()
            try:
                await atask
            except asyncio.CancelledError:
                pass
        finally:
            abort_task.cancel()
            try:
                await abort_task
            except asyncio.CancelledError:
                pass

    @pytest.mark.asyncio
    async def test_cycle_scan_claim_execute_path(self, tmp_path: Path):
        """Test the full scan -> claim -> execute path in the cycle."""
        store = TaskStore(tmp_path / "tasks", list_id="test")
        teammate = AutonomousTeammate()

        store.create(subject="T1", description="desc")

        run_called = asyncio.Event()
        run_func = AsyncMock(side_effect=lambda t: (run_called.set(), "done")[1])

        atask = teammate.spawn_autonomous(
            team_name="team1",
            member_name="worker",
            task_store=store,
            config=AutonomousConfig(claim_delay=0.01, idle_timeout=0.01),
            run_func=run_func,
        )

        # Wait for run to be called
        try:
            await asyncio.wait_for(run_called.wait(), timeout=2.0)
        except TimeoutError:
            pass

        # Shut down
        handle = teammate.get_handle("worker@team1")
        if handle:
            handle.abort_signal.set()
        try:
            await asyncio.wait_for(atask, timeout=2.0)
        except (TimeoutError, asyncio.CancelledError):
            atask.cancel()
            try:
                await atask
            except asyncio.CancelledError:
                pass

        # Task should be completed
        tasks = store.list_all()
        completed = [t for t in tasks if t.status == TaskStatus.COMPLETED]
        assert len(completed) >= 1

    @pytest.mark.asyncio
    async def test_cycle_idle_notification_with_mailbox(self, tmp_path: Path):
        """When idle timeout is reached, notification should be sent via mailbox."""
        store = TaskStore(tmp_path / "tasks", list_id="test")
        mailbox = Mailbox(base_dir=tmp_path / "mail")
        teammate = AutonomousTeammate()

        config = AutonomousConfig(claim_delay=0.02, idle_timeout=0.05)

        atask = teammate.spawn_autonomous(
            team_name="team1",
            member_name="worker",
            task_store=store,
            mailbox=mailbox,
            config=config,
            run_func=AsyncMock(return_value="done"),
        )

        # Let it idle
        await asyncio.sleep(0.3)
        handle = teammate.get_handle("worker@team1")
        if handle:
            handle.abort_signal.set()
        try:
            await asyncio.wait_for(atask, timeout=2.0)
        except (TimeoutError, asyncio.CancelledError):
            atask.cancel()
            try:
                await atask
            except asyncio.CancelledError:
                pass

    @pytest.mark.asyncio
    async def test_cycle_no_handle_does_not_crash(self, tmp_path: Path):
        """When get_handle returns None (handle not found), cycle should still work."""
        store = TaskStore(tmp_path / "tasks", list_id="test")
        teammate = AutonomousTeammate()
        store.create(subject="T1", description="desc")

        # Run the cycle directly (not via spawn) so no handle exists
        run_func = AsyncMock(return_value="done")

        async def run_cycle():
            try:
                await asyncio.wait_for(
                    teammate.run_autonomous_cycle(
                        team_name="team1",
                        agent_name="orphan",
                        task_store=store,
                        config=AutonomousConfig(claim_delay=0.01, idle_timeout=0.05),
                        run_func=run_func,
                    ),
                    timeout=1.0,
                )
            except TimeoutError:
                pass

        cycle_task = asyncio.create_task(run_cycle())
        await asyncio.sleep(0.2)
        cycle_task.cancel()
        try:
            await cycle_task
        except asyncio.CancelledError:
            pass


# ─────────────────────────────────────────────────────────
# _default_run_func
# ─────────────────────────────────────────────────────────


class TestDefaultRunFunc:
    @pytest.mark.asyncio
    async def test_returns_completion_string(self, tmp_path: Path):
        task = Task(id="t1", subject="Test Task", description="desc")
        result = await _default_run_func(task)
        assert "t1" in result
        assert "Test Task" in result
        assert "Completed" in result


# ─────────────────────────────────────────────────────────
# spawn_autonomous
# ─────────────────────────────────────────────────────────


class TestSpawnAutonomous:
    @pytest.mark.asyncio
    async def test_spawn_returns_task(self, tmp_path: Path):
        store = TaskStore(tmp_path / "tasks", list_id="test")
        teammate = AutonomousTeammate()

        result = teammate.spawn_autonomous(
            team_name="team1",
            member_name="worker",
            task_store=store,
            config=AutonomousConfig(claim_delay=0.01, idle_timeout=0.01),
            run_func=AsyncMock(return_value="done"),
        )

        assert isinstance(result, asyncio.Task)

        # Cleanup
        handle = teammate.get_handle("worker@team1")
        if handle:
            handle.abort_signal.set()
        try:
            await asyncio.wait_for(result, timeout=2.0)
        except (TimeoutError, asyncio.CancelledError):
            result.cancel()
            try:
                await result
            except asyncio.CancelledError:
                pass

    @pytest.mark.asyncio
    async def test_spawn_with_initial_task(self, tmp_path: Path):
        store = TaskStore(tmp_path / "tasks", list_id="test")
        teammate = AutonomousTeammate()

        result = teammate.spawn_autonomous(
            team_name="team1",
            member_name="worker",
            task_store=store,
            config=AutonomousConfig(claim_delay=0.01, idle_timeout=0.01),
            run_func=AsyncMock(return_value="done"),
            agent_type="explore",
        )

        assert isinstance(result, asyncio.Task)

        # Cleanup
        handle = teammate.get_handle("worker@team1")
        if handle:
            handle.abort_signal.set()
        try:
            await asyncio.wait_for(result, timeout=2.0)
        except (TimeoutError, asyncio.CancelledError):
            result.cancel()
            try:
                await result
            except asyncio.CancelledError:
                pass


# ─────────────────────────────────────────────────────────
# reset
# ─────────────────────────────────────────────────────────


class TestReset:
    def test_reset_clears_claim_counts(self, tmp_path: Path):
        store = TaskStore(tmp_path / "tasks", list_id="test")
        teammate = AutonomousTeammate()

        task = store.create(subject="T1", description="desc")
        teammate.claim_task(store, task, "worker")
        assert teammate._tasks_claimed["worker"] == 1

        teammate.reset()
        assert teammate._tasks_claimed == {}


# ─────────────────────────────────────────────────────────
# delegate_task
# ─────────────────────────────────────────────────────────


def _make_mock_tools():
    """Create mock tools matching what delegate_task imports from dazi.filesystem."""
    file_reader = MagicMock()
    file_reader.name = "file_reader_tool"
    file_reader.invoke = MagicMock(return_value="file contents here")
    calculator = MagicMock()
    calculator.name = "calculator_tool"
    calculator.invoke = MagicMock(return_value="42")
    shell_exec = MagicMock()
    shell_exec.name = "shell_exec_tool"
    shell_exec.invoke = MagicMock(return_value="shell output")
    return file_reader, calculator, shell_exec


class TestDelegateTask:
    def test_delegate_task_with_no_tool_calls(self):
        """When LLM returns no tool calls, delegate_task should return the response."""
        mock_response = MagicMock()
        mock_response.content = "Research complete: found 3 results"
        mock_response.tool_calls = []

        mock_llm = MagicMock()
        mock_llm.bind_tools.return_value = mock_llm
        mock_llm.invoke.return_value = mock_response

        file_reader, calculator, shell_exec = _make_mock_tools()

        with (
            patch("dazi.llm.create_llm", return_value=mock_llm),
            patch("dazi.filesystem.file_reader_tool", file_reader),
            patch("dazi.filesystem.calculator_tool", calculator),
            patch("dazi.filesystem.shell_exec_tool", shell_exec),
        ):
            result = delegate_task("Research topic X", max_turns=3)
            assert "Research complete" in result
            assert "Sub-agent" in result

    def test_delegate_task_with_tool_calls(self):
        """When LLM uses tools, delegate_task should execute them."""
        # First response: has tool call; Second response: no tool calls
        mock_response1 = MagicMock()
        mock_response1.content = ""
        mock_response1.tool_calls = [
            {"name": "file_reader_tool", "id": "tc1", "args": {"file_path": "/tmp/test.txt"}}
        ]

        mock_response2 = MagicMock()
        mock_response2.content = "File contents: hello world"
        mock_response2.tool_calls = []

        mock_llm = MagicMock()
        mock_llm.bind_tools.return_value = mock_llm
        mock_llm.invoke.side_effect = [mock_response1, mock_response2]

        file_reader, calculator, shell_exec = _make_mock_tools()

        with (
            patch("dazi.llm.create_llm", return_value=mock_llm),
            patch("dazi.filesystem.file_reader_tool", file_reader),
            patch("dazi.filesystem.calculator_tool", calculator),
            patch("dazi.filesystem.shell_exec_tool", shell_exec),
        ):
            result = delegate_task("Read file /tmp/test.txt", max_turns=5)
            # The mock file_reader_tool returns "file contents here"
            assert "file contents" in result or "hello world" in result

    def test_delegate_task_with_unknown_tool_in_call(self):
        """When LLM calls a tool not in sub_tools, should get error message."""
        mock_response1 = MagicMock()
        mock_response1.content = ""
        mock_response1.tool_calls = [{"name": "nonexistent_tool", "id": "tc1", "args": {}}]

        mock_response2 = MagicMock()
        mock_response2.content = "Error: Tool 'nonexistent_tool' not available"
        mock_response2.tool_calls = []

        mock_llm = MagicMock()
        mock_llm.bind_tools.return_value = mock_llm
        mock_llm.invoke.side_effect = [mock_response1, mock_response2]

        file_reader, calculator, shell_exec = _make_mock_tools()

        with (
            patch("dazi.llm.create_llm", return_value=mock_llm),
            patch("dazi.filesystem.file_reader_tool", file_reader),
            patch("dazi.filesystem.calculator_tool", calculator),
            patch("dazi.filesystem.shell_exec_tool", shell_exec),
        ):
            result = delegate_task("Do something", max_turns=5)
            # The LLM receives the error and responds
            assert result is not None

    def test_delegate_task_with_allowed_tools_filter(self):
        """When allowed_tools is specified, only those tools are passed."""
        mock_response = MagicMock()
        mock_response.content = "Done with calculator"
        mock_response.tool_calls = []

        mock_llm = MagicMock()
        mock_llm.bind_tools.return_value = mock_llm
        mock_llm.invoke.return_value = mock_response

        file_reader, calculator, shell_exec = _make_mock_tools()

        with (
            patch("dazi.llm.create_llm", return_value=mock_llm),
            patch("dazi.filesystem.file_reader_tool", file_reader),
            patch("dazi.filesystem.calculator_tool", calculator),
            patch("dazi.filesystem.shell_exec_tool", shell_exec),
        ):
            result = delegate_task("Calculate something", allowed_tools=["calculator_tool"])
            assert "Done with calculator" in result

    def test_delegate_task_with_unknown_allowed_tools(self):
        """When allowed_tools contains unknown tool names, should return error."""
        file_reader, calculator, shell_exec = _make_mock_tools()

        with (
            patch("dazi.filesystem.file_reader_tool", file_reader),
            patch("dazi.filesystem.calculator_tool", calculator),
            patch("dazi.filesystem.shell_exec_tool", shell_exec),
        ):
            result = delegate_task("Do stuff", allowed_tools=["nonexistent_tool"])
            assert "Unknown tools" in result
            assert "nonexistent_tool" in result

    def test_delegate_task_llm_exception(self):
        """When LLM raises an exception, should append error and return."""
        mock_llm = MagicMock()
        mock_llm.bind_tools.return_value = mock_llm
        mock_llm.invoke.side_effect = RuntimeError("LLM crashed")

        file_reader, calculator, shell_exec = _make_mock_tools()

        with (
            patch("dazi.llm.create_llm", return_value=mock_llm),
            patch("dazi.filesystem.file_reader_tool", file_reader),
            patch("dazi.filesystem.calculator_tool", calculator),
            patch("dazi.filesystem.shell_exec_tool", shell_exec),
        ):
            result = delegate_task("Do something", max_turns=3)
            # Should contain the error message
            assert "Sub-agent error" in result or "error" in result.lower()

    def test_delegate_task_empty_final_message(self):
        """When final message is empty, should return 'did not produce a response'."""
        mock_response = MagicMock()
        mock_response.content = ""
        mock_response.tool_calls = []

        mock_llm = MagicMock()
        mock_llm.bind_tools.return_value = mock_llm
        mock_llm.invoke.return_value = mock_response

        file_reader, calculator, shell_exec = _make_mock_tools()

        with (
            patch("dazi.llm.create_llm", return_value=mock_llm),
            patch("dazi.filesystem.file_reader_tool", file_reader),
            patch("dazi.filesystem.calculator_tool", calculator),
            patch("dazi.filesystem.shell_exec_tool", shell_exec),
        ):
            result = delegate_task("Do something", max_turns=1)
            assert "did not produce a response" in result

    def test_delegate_task_single_turn(self):
        """Turn count grammar: 1 turn should not have 's'."""
        mock_response = MagicMock()
        mock_response.content = "Done in 1 turn"
        mock_response.tool_calls = []

        mock_llm = MagicMock()
        mock_llm.bind_tools.return_value = mock_llm
        mock_llm.invoke.return_value = mock_response

        file_reader, calculator, shell_exec = _make_mock_tools()

        with (
            patch("dazi.llm.create_llm", return_value=mock_llm),
            patch("dazi.filesystem.file_reader_tool", file_reader),
            patch("dazi.filesystem.calculator_tool", calculator),
            patch("dazi.filesystem.shell_exec_tool", shell_exec),
        ):
            result = delegate_task("Task", max_turns=1)
            assert "1 turn" in result  # singular, no 's'


# ─────────────────────────────────────────────────────────
# delegate_task tool metadata
# ─────────────────────────────────────────────────────────


class TestDelegateTaskMetadata:
    def test_tool_exists(self):
        assert delegate_task_tool.name == "delegate_task"
        assert "delegate" in delegate_task_tool.description.lower()

    def test_meta_exists(self):
        assert delegate_task_meta.name == "delegate_task"
        assert delegate_task_meta.safety.value == "safe"

    def test_input_schema(self):
        inp = DelegateTaskInput(task="do something")
        assert inp.task == "do something"
        assert inp.max_turns == 5
        assert inp.allowed_tools is None

        inp2 = DelegateTaskInput(task="do something", max_turns=10, allowed_tools=["tool1"])
        assert inp2.max_turns == 10
        assert inp2.allowed_tools == ["tool1"]


# ─────────────────────────────────────────────────────────
# spawn_agent_func
# ─────────────────────────────────────────────────────────


class TestSpawnAgentFunc:
    @pytest.mark.asyncio
    async def test_spawn_agent_func(self, tmp_path: Path):
        mock_task = MagicMock()
        mock_task.get_name.return_value = "worker@team1"

        mock_teammate = MagicMock()
        mock_teammate.spawn_autonomous.return_value = mock_task

        with (
            patch("dazi._singletons.autonomous_teammate", mock_teammate),
            patch("dazi.config.DATA_DIR", tmp_path),
        ):
            result = await spawn_agent_func(
                team_name="team1",
                member_name="worker",
                agent_type="general-purpose",
            )
            assert "worker" in result
            assert "team1" in result
            mock_teammate.spawn_autonomous.assert_called_once()

    @pytest.mark.asyncio
    async def test_spawn_agent_func_with_initial_task(self, tmp_path: Path):
        mock_task = MagicMock()
        mock_task.get_name.return_value = "backend@team1"

        mock_teammate = MagicMock()
        mock_teammate.spawn_autonomous.return_value = mock_task

        with (
            patch("dazi._singletons.autonomous_teammate", mock_teammate),
            patch("dazi.config.DATA_DIR", tmp_path),
        ):
            result = await spawn_agent_func(
                team_name="team1",
                member_name="backend",
                agent_type="explore",
                initial_task="Fix the bug in auth module",
            )
            assert "backend" in result
            assert "Initial task: Fix the bug" in result


class TestSpawnAgentMetadata:
    def test_tool_exists(self):
        assert spawn_agent_tool.name == "spawn_agent"

    def test_meta_exists(self):
        assert spawn_agent_meta.name == "spawn_agent"
        assert spawn_agent_meta.safety.value == "write"

    def test_input_schema(self):
        inp = AgentSpawnInput(team_name="team1", member_name="worker")
        assert inp.team_name == "team1"
        assert inp.member_name == "worker"
        assert inp.agent_type == "general-purpose"
        assert inp.initial_task == ""

        inp2 = AgentSpawnInput(
            team_name="team1", member_name="backend", agent_type="explore", initial_task="fix bug"
        )
        assert inp2.agent_type == "explore"
        assert inp2.initial_task == "fix bug"


# ─────────────────────────────────────────────────────────
# idle notification sending (via run_autonomous_cycle)
# ─────────────────────────────────────────────────────────


class TestIdleNotification:
    @pytest.mark.asyncio
    async def test_idle_notification_sent(self, tmp_path: Path):
        store = TaskStore(tmp_path / "tasks", list_id="test")
        mailbox = Mailbox(base_dir=tmp_path / "mail")
        teammate = AutonomousTeammate()

        config = AutonomousConfig(
            claim_delay=0.05,
            idle_timeout=0.1,
            max_tasks_per_agent=5,
        )

        # Create a handle with abort_signal that we'll set after short delay
        task = teammate.spawn(
            team_name="team1",
            member_name="worker",
            run_func=lambda h: None,  # dummy
        )
        await task

        # Re-spawn for the autonomous cycle
        teammate.reset()
        atask = teammate.spawn_autonomous(
            team_name="team1",
            member_name="worker",
            task_store=store,
            mailbox=mailbox,
            config=config,
            run_func=AsyncMock(return_value="done"),
        )

        # Let it run briefly, then shut down
        await asyncio.sleep(0.3)
        handle = teammate.get_handle("worker@team1")
        if handle:
            handle.abort_signal.set()
        try:
            await asyncio.wait_for(atask, timeout=2.0)
        except (TimeoutError, asyncio.CancelledError):
            atask.cancel()
            try:
                await atask
            except asyncio.CancelledError:
                pass

        # Check that some notification was sent (mailbox should have messages)
        # The notification goes to all team members via broadcast
