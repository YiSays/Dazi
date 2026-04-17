"""Tests for dazi/repl_display.py — Rich table/panel display helpers."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from rich.console import Console

from dazi.background import BackgroundTask, BackgroundTaskStatus
from dazi.dazimd import DaziMdFile
from dazi.mcp_client import (
    MCPResource,
    MCPServerConfig,
    MCPServerConnection,
    MCPServerStatus,
    MCPServerTool,
)
from dazi.permissions import PermissionBehavior, PermissionRule
from dazi.skills import Skill
from dazi.task_store import TaskStatus, TaskStore
from tests.helpers.mock_singletons import patch_singletons


@pytest.fixture(autouse=True)
def _patch(monkeypatch, tmp_path: Path):
    patch_singletons(monkeypatch, tmp_path)
    # Also patch module-level references that were captured at import time
    import dazi.repl_display as _mod
    from dazi._singletons import (
        background_manager,
        mcp_manager,
        memory_store,
        skill_registry,
        task_store,
    )

    monkeypatch.setattr(_mod, "background_manager", background_manager)
    monkeypatch.setattr(_mod, "mcp_manager", mcp_manager)
    monkeypatch.setattr(_mod, "memory_store", memory_store)
    monkeypatch.setattr(_mod, "skill_registry", skill_registry)
    monkeypatch.setattr(_mod, "task_store", task_store)


def _console():
    return Console(file=MagicMock(), force_terminal=False)


# ─────────────────────────────────────────────────────────
# get_mode_badge
# ─────────────────────────────────────────────────────────


class TestGetModeBadge:
    def test_plan_mode(self):
        import dazi.repl_display as mod

        badge = mod.get_mode_badge("plan")
        assert len(badge) == 2
        assert badge[1][1] == " (Shift+Tab to switch)"

    def test_execute_mode(self):
        import dazi.repl_display as mod

        badge = mod.get_mode_badge("execute")
        assert len(badge) == 2
        assert badge[1][1] == " (Shift+Tab to switch)"

    def test_plan_mode_constant(self):
        import dazi.repl_display as mod

        badge = mod.get_mode_badge(mod.PLAN_MODE)
        assert len(badge) == 2


# ─────────────────────────────────────────────────────────
# list_rules_table
# ─────────────────────────────────────────────────────────


class TestListRulesTable:
    def test_empty_rules(self):
        import dazi.repl_display as mod

        with patch.object(mod, "_get_effective_rules", return_value=[]):
            mod.list_rules_table()

    def test_with_rules(self):
        import dazi.repl_display as mod

        rules = [
            PermissionRule(behavior=PermissionBehavior.ALLOW, tool_name="git", source="cli"),
            PermissionRule(
                behavior=PermissionBehavior.DENY,
                tool_name="rm",
                pattern="*",
                source="settings",
            ),
            PermissionRule(behavior=PermissionBehavior.ASK, tool_name="npm", source="cli"),
        ]
        with patch.object(mod, "_get_effective_rules", return_value=rules):
            mod.list_rules_table()


# ─────────────────────────────────────────────────────────
# list_memories_table
# ─────────────────────────────────────────────────────────


class TestListMemoriesTable:
    def test_no_memories(self):
        import dazi.repl_display as mod
        from dazi._singletons import memory_store

        memory_store.list_all = MagicMock(return_value=[])
        mod.list_memories_table()

    def test_with_memories(self):
        import dazi.repl_display as mod
        from dazi._singletons import memory_store
        from dazi.memory import MemoryCategory, MemoryEntry

        entry = MagicMock(spec=MemoryEntry)
        entry.id = "mem1"
        entry.category = MemoryCategory.USER
        entry.description = "Short desc"
        entry.content = "some content"
        entry.created_at = "2025-01-15T10:00:00"
        memory_store.list_all = MagicMock(return_value=[entry])
        mod.list_memories_table()

    def test_long_description_truncated(self):
        import dazi.repl_display as mod
        from dazi._singletons import memory_store
        from dazi.memory import MemoryCategory, MemoryEntry

        entry = MagicMock(spec=MemoryEntry)
        entry.id = "mem2"
        entry.category = MemoryCategory.PROJECT
        entry.description = "x" * 100  # > 60 chars triggers truncation
        entry.content = "some content"
        entry.created_at = "2025-01-15T10:00:00"
        memory_store.list_all = MagicMock(return_value=[entry])
        mod.list_memories_table()


# ─────────────────────────────────────────────────────────
# show_dazimd_files
# ─────────────────────────────────────────────────────────


class TestShowDazimdFiles:
    def test_empty(self):
        import dazi.repl_display as mod

        mod.show_dazimd_files([])

    def test_with_files(self):
        import dazi.repl_display as mod

        files = [
            DaziMdFile(path=Path("/project/DAZI.md"), priority=300, content="# Rules\nBe nice."),
            DaziMdFile(
                path=Path("/project/DAZI.local.md"), priority=400, content="# Local\nSecrets."
            ),
        ]
        mod.show_dazimd_files(files)


# ─────────────────────────────────────────────────────────
# show_token_info
# ─────────────────────────────────────────────────────────


class TestShowTokenInfo:
    def test_empty_messages(self):
        import dazi.repl_display as mod

        with (
            patch.object(mod, "_get_model_name", return_value="gpt-4o"),
            patch.object(mod, "get_context_window", return_value=128000),
        ):
            mod.show_token_info([])

    def test_with_messages(self):
        from langchain_core.messages import AIMessage, HumanMessage

        import dazi.repl_display as mod

        messages = [HumanMessage(content="Hello"), AIMessage(content="Hi")]
        with (
            patch.object(mod, "_get_model_name", return_value="gpt-4o"),
            patch.object(mod, "count_messages_tokens", return_value=500),
            patch.object(mod, "get_context_window", return_value=128000),
            patch.object(mod, "get_token_warning_state", return_value="ok"),
        ):
            mod.show_token_info(messages)

    def test_warning_state_warning(self):
        from langchain_core.messages import HumanMessage

        import dazi.repl_display as mod

        messages = [HumanMessage(content="test")]
        with (
            patch.object(mod, "_get_model_name", return_value="gpt-4o"),
            patch.object(mod, "count_messages_tokens", return_value=115000),
            patch.object(mod, "get_context_window", return_value=128000),
            patch.object(mod, "get_token_warning_state", return_value="warning"),
        ):
            mod.show_token_info(messages)

    def test_warning_state_compact(self):
        from langchain_core.messages import HumanMessage

        import dazi.repl_display as mod

        messages = [HumanMessage(content="test")]
        with (
            patch.object(mod, "_get_model_name", return_value="gpt-4o"),
            patch.object(mod, "count_messages_tokens", return_value=120000),
            patch.object(mod, "get_context_window", return_value=128000),
            patch.object(mod, "get_token_warning_state", return_value="compact"),
        ):
            mod.show_token_info(messages)

    def test_warning_state_error(self):
        from langchain_core.messages import HumanMessage

        import dazi.repl_display as mod

        messages = [HumanMessage(content="test")]
        with (
            patch.object(mod, "_get_model_name", return_value="gpt-4o"),
            patch.object(mod, "count_messages_tokens", return_value=130000),
            patch.object(mod, "get_context_window", return_value=128000),
            patch.object(mod, "get_token_warning_state", return_value="error"),
        ):
            mod.show_token_info(messages)

    def test_zero_context_window(self):
        from langchain_core.messages import HumanMessage

        import dazi.repl_display as mod

        messages = [HumanMessage(content="test")]
        with (
            patch.object(mod, "_get_model_name", return_value="gpt-4o"),
            patch.object(mod, "count_messages_tokens", return_value=100),
            patch.object(mod, "get_context_window", return_value=0),
            patch.object(mod, "get_token_warning_state", return_value="ok"),
        ):
            mod.show_token_info(messages)


# ─────────────────────────────────────────────────────────
# list_tasks_table
# ─────────────────────────────────────────────────────────


class TestListTasksTable:
    def _make_store(self, tmp_path: Path) -> TaskStore:
        return TaskStore(tmp_path / "tasks", list_id="test")

    def test_no_tasks(self, tmp_path: Path):
        import dazi.repl_display as mod

        store = self._make_store(tmp_path)
        mod.list_tasks_table(
            active_team_name=None,
            default_task_store=store,
            team_task_store=None,
        )

    def test_with_tasks(self, tmp_path: Path):
        import dazi.repl_display as mod

        store = self._make_store(tmp_path)
        store.create(subject="Task one", description="desc")
        t2 = store.create(subject="Task two", description="desc")
        store.update(t2.id, status=TaskStatus.IN_PROGRESS, owner="agent")
        mod.list_tasks_table(
            active_team_name=None,
            default_task_store=store,
            team_task_store=None,
        )

    def test_team_store_used(self, tmp_path: Path):
        import dazi.repl_display as mod

        team_store = self._make_store(tmp_path / "team")
        team_store.create(subject="Team task", description="desc")
        default_store = self._make_store(tmp_path / "default")
        mod.list_tasks_table(
            active_team_name="my-team",
            default_task_store=default_store,
            team_task_store=team_store,
        )

    def test_with_blockers(self, tmp_path: Path):
        import dazi.repl_display as mod

        store = self._make_store(tmp_path)
        t1 = store.create(subject="Blocked task", description="desc")
        t2 = store.create(subject="Blocker", description="desc")
        store.add_block(t2.id, t1.id)
        mod.list_tasks_table(
            active_team_name=None,
            default_task_store=store,
            team_task_store=None,
        )


# ─────────────────────────────────────────────────────────
# show_task_detail
# ─────────────────────────────────────────────────────────


class TestShowTaskDetail:
    def _make_store(self, tmp_path: Path) -> TaskStore:
        return TaskStore(tmp_path / "tasks", list_id="test")

    def test_not_found(self, tmp_path: Path):
        import dazi.repl_display as mod

        store = self._make_store(tmp_path)
        mod.show_task_detail(
            999,
            active_team_name=None,
            default_task_store=store,
            team_task_store=None,
        )

    def test_found_minimal(self, tmp_path: Path):
        import dazi.repl_display as mod

        store = self._make_store(tmp_path)
        task = store.create(subject="Test task", description="A description")
        mod.show_task_detail(
            task.id,
            active_team_name=None,
            default_task_store=store,
            team_task_store=None,
        )

    def test_found_full(self, tmp_path: Path):
        import dazi.repl_display as mod

        store = self._make_store(tmp_path)
        # Create a blocker task and a blocked task
        blocker = store.create(subject="Blocker", description="b")
        task = store.create(
            subject="Full task",
            description="Full desc",
            active_form="Implementing feature X",
        )
        store.add_block(blocker.id, task.id)
        store.update(
            task.id, status=TaskStatus.IN_PROGRESS, owner="claude", metadata={"priority": "high"}
        )
        mod.show_task_detail(
            task.id,
            active_team_name=None,
            default_task_store=store,
            team_task_store=None,
        )

    def test_team_store_used(self, tmp_path: Path):
        import dazi.repl_display as mod

        team_store = self._make_store(tmp_path / "team")
        default_store = self._make_store(tmp_path / "default")
        task = team_store.create(subject="Team task", description="desc")
        mod.show_task_detail(
            task.id,
            active_team_name="team",
            default_task_store=default_store,
            team_task_store=team_store,
        )

    def test_task_with_blocks(self, tmp_path: Path):
        """Show a task that blocks others (has non-empty blocks list)."""
        import dazi.repl_display as mod

        store = self._make_store(tmp_path)
        blocker = store.create(subject="Blocker task", description="Blocks other tasks")
        store.create(subject="Dependent task", description="Depends on blocker")
        store.add_block(blocker.id, 2)  # blocker blocks task 2 -> blocker.blocks=[2]
        mod.show_task_detail(
            blocker.id,
            active_team_name=None,
            default_task_store=store,
            team_task_store=None,
        )


# ─────────────────────────────────────────────────────────
# show_background_tasks_table
# ─────────────────────────────────────────────────────────


class TestShowBackgroundTasksTable:
    def test_no_tasks(self):
        import dazi.repl_display as mod
        from dazi._singletons import background_manager

        background_manager.list_all = MagicMock(return_value=[])
        mod.show_background_tasks_table()

    def test_with_tasks(self):
        import dazi.repl_display as mod
        from dazi._singletons import background_manager

        tasks = [
            BackgroundTask(
                id="abc12345",
                command="echo hello world this is a long command",
                status=BackgroundTaskStatus.RUNNING,
                pid=12345,
                exit_code=None,
            ),
            BackgroundTask(
                id="def67890",
                command="ls -la",
                status=BackgroundTaskStatus.COMPLETED,
                pid=12346,
                exit_code=0,
            ),
            BackgroundTask(
                id="ghi11111",
                command="cat nonexistent",
                status=BackgroundTaskStatus.FAILED,
                pid=12347,
                exit_code=1,
                error="file not found",
            ),
            BackgroundTask(
                id="jkl22222",
                command="sleep 999",
                status=BackgroundTaskStatus.KILLED,
            ),
        ]
        # Set started_at/completed_at so duration_seconds has a value
        import time

        for t in tasks:
            t.started_at = time.time() - 10
        tasks[1].completed_at = time.time()
        tasks[2].completed_at = time.time()

        background_manager.list_all = MagicMock(return_value=tasks)
        mod.show_background_tasks_table()


# ─────────────────────────────────────────────────────────
# show_background_task_detail
# ─────────────────────────────────────────────────────────


class TestShowBackgroundTaskDetail:
    def test_not_found(self):
        import dazi.repl_display as mod
        from dazi._singletons import background_manager

        background_manager.check_sync = MagicMock(return_value=None)
        mod.show_background_task_detail("nonexistent")

    def test_running_task(self):
        import dazi.repl_display as mod
        from dazi._singletons import background_manager

        task = BackgroundTask(
            id="run12345",
            command="python server.py",
            status=BackgroundTaskStatus.RUNNING,
            description="Start dev server",
            pid=9999,
        )
        import time

        task.started_at = time.time() - 5
        background_manager.check_sync = MagicMock(return_value=task)
        background_manager.get_output_tail = MagicMock(return_value="Listening on port 8000")
        mod.show_background_task_detail("run12345")

    def test_completed_task_with_output(self):
        import dazi.repl_display as mod
        from dazi._singletons import background_manager

        task = BackgroundTask(
            id="done12345",
            command="npm test",
            status=BackgroundTaskStatus.COMPLETED,
            exit_code=0,
        )
        import time

        task.started_at = time.time() - 30
        task.completed_at = time.time()
        background_manager.check_sync = MagicMock(return_value=task)
        background_manager.get_output_tail = MagicMock(return_value="All tests passed!")
        mod.show_background_task_detail("done12345")

    def test_failed_task_with_error(self):
        import dazi.repl_display as mod
        from dazi._singletons import background_manager

        task = BackgroundTask(
            id="fail12345",
            command="npm test",
            status=BackgroundTaskStatus.FAILED,
            exit_code=1,
            error="AssertionError: expected 200 got 500",
        )
        import time

        task.started_at = time.time() - 10
        task.completed_at = time.time()
        background_manager.check_sync = MagicMock(return_value=task)
        background_manager.get_output_tail = MagicMock(return_value=None)
        mod.show_background_task_detail("fail12345")

    def test_pending_task(self):
        import dazi.repl_display as mod
        from dazi._singletons import background_manager

        task = BackgroundTask(
            id="pend12345",
            command="sleep 100",
            status=BackgroundTaskStatus.PENDING,
        )
        background_manager.check_sync = MagicMock(return_value=task)
        background_manager.get_output_tail = MagicMock(return_value=None)
        mod.show_background_task_detail("pend12345")


# ─────────────────────────────────────────────────────────
# show_mcp_servers_table
# ─────────────────────────────────────────────────────────


class TestShowMCPServersTable:
    def test_no_servers(self):
        import dazi.repl_display as mod
        from dazi._singletons import mcp_manager

        mcp_manager.list_servers = MagicMock(return_value=[])
        mod.show_mcp_servers_table()

    def test_with_servers(self):
        import dazi.repl_display as mod
        from dazi._singletons import mcp_manager

        servers = [
            {
                "name": "filesystem",
                "status": "connected",
                "tool_count": 5,
                "resource_count": 2,
                "command": "npx @modelcontextprotocol/server-filesystem /tmp",
            },
            {
                "name": "broken",
                "status": "error",
                "tool_count": 0,
                "resource_count": 0,
                "command": "python broken_server.py",
            },
            {
                "name": "pending",
                "status": "disconnected",
                "tool_count": 3,
                "resource_count": 1,
                "command": "npx some-server",
            },
        ]
        mcp_manager.list_servers = MagicMock(return_value=servers)
        mod.show_mcp_servers_table()


# ─────────────────────────────────────────────────────────
# show_mcp_server_detail
# ─────────────────────────────────────────────────────────


class TestShowMCPServerDetail:
    def test_not_found(self):
        import dazi.repl_display as mod
        from dazi._singletons import mcp_manager

        mcp_manager.get_server = MagicMock(return_value=None)
        mod.show_mcp_server_detail("nonexistent")

    def test_connected_with_tools_and_resources(self):
        import dazi.repl_display as mod
        from dazi._singletons import mcp_manager

        config = MCPServerConfig(
            name="filesystem",
            command="npx",
            args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
        )
        conn = MCPServerConnection(
            config=config,
            status=MCPServerStatus.CONNECTED,
            tools=[
                MCPServerTool(
                    server_name="filesystem",
                    name="read_file",
                    qualified_name="mcp__filesystem__read_file",
                    description="Read a file from disk",
                    input_schema={"type": "object"},
                    is_read_only=True,
                ),
                MCPServerTool(
                    server_name="filesystem",
                    name="write_file",
                    qualified_name="mcp__filesystem__write_file",
                    description=(
                        "Write content to a file on disk, this is a longer "
                        "description that exceeds sixty chars"
                    ),
                    input_schema={"type": "object"},
                    is_read_only=False,
                ),
            ],
            resources=[
                MCPResource(
                    server_name="filesystem",
                    uri="file:///tmp/test.txt",
                    name="test.txt",
                    mime_type="text/plain",
                ),
            ],
        )
        mcp_manager.get_server = MagicMock(return_value=conn)
        mod.show_mcp_server_detail("filesystem")

    def test_connected_no_tools_no_resources(self):
        import dazi.repl_display as mod
        from dazi._singletons import mcp_manager

        config = MCPServerConfig(name="empty", command="python", args=[])
        conn = MCPServerConnection(
            config=config,
            status=MCPServerStatus.CONNECTED,
            tools=[],
            resources=[],
        )
        mcp_manager.get_server = MagicMock(return_value=conn)
        mod.show_mcp_server_detail("empty")

    def test_error_status(self):
        import dazi.repl_display as mod
        from dazi._singletons import mcp_manager

        config = MCPServerConfig(name="broken", command="python", args=["bad.py"])
        conn = MCPServerConnection(
            config=config,
            status=MCPServerStatus.ERROR,
            error="Connection refused",
            tools=[],
        )
        mcp_manager.get_server = MagicMock(return_value=conn)
        mod.show_mcp_server_detail("broken")

    def test_config_with_none_args(self):
        import dazi.repl_display as mod
        from dazi._singletons import mcp_manager

        config = MCPServerConfig(name="noargs", command="bash", args=None)
        conn = MCPServerConnection(
            config=config,
            status=MCPServerStatus.CONNECTED,
            tools=[],
        )
        mcp_manager.get_server = MagicMock(return_value=conn)
        mod.show_mcp_server_detail("noargs")


# ─────────────────────────────────────────────────────────
# show_skills_table
# ─────────────────────────────────────────────────────────


class TestShowSkillsTable:
    def test_no_skills(self):
        import dazi.repl_display as mod
        from dazi._singletons import skill_registry

        skill_registry.list_all = MagicMock(return_value=[])
        mod.show_skills_table()

    def test_with_skills(self):
        import dazi.repl_display as mod
        from dazi._singletons import skill_registry

        skills = [
            Skill(
                name="commit",
                description="Create a git commit",
                prompt="Commit the changes",
                is_bundled=True,
                user_invocable=True,
                source_path=None,
            ),
            Skill(
                name="custom-skill",
                description="A custom skill with a longer description that should be truncated",
                prompt="Custom prompt",
                is_bundled=False,
                user_invocable=False,
                source_path=Path("/project/.dazi/skills/custom-skill"),
            ),
            Skill(
                name="no-source",
                description="No source path skill",
                prompt="prompt",
                is_bundled=False,
                user_invocable=True,
                source_path=None,
            ),
        ]
        skill_registry.list_all = MagicMock(return_value=skills)
        mod.show_skills_table()


# ─────────────────────────────────────────────────────────
# show_skill_detail
# ─────────────────────────────────────────────────────────


class TestShowSkillDetail:
    def test_not_found(self):
        import dazi.repl_display as mod
        from dazi._singletons import skill_registry

        skill_registry.get = MagicMock(return_value=None)
        mod.show_skill_detail("nonexistent")

    def test_found_minimal(self):
        import dazi.repl_display as mod
        from dazi._singletons import skill_registry

        skill = Skill(
            name="minimal",
            description="A minimal skill",
            prompt="Do the thing",
            version="1.0",
        )
        skill_registry.get = MagicMock(return_value=skill)
        mod.show_skill_detail("minimal")

    def test_found_full(self):
        import dazi.repl_display as mod
        from dazi._singletons import skill_registry

        skill = Skill(
            name="full-skill",
            description="A fully specified skill",
            prompt=(
                "## Instructions\n\nFollow these steps:\n"
                "1. Read the code\n2. Fix the bugs\n3. Write tests"
            ),
            version="2.0",
            argument_hint="[file-path]",
            arguments=["file-path", "description"],
            user_invocable=True,
            when_to_use="When you need to fix bugs",
            allowed_tools=["file_reader", "file_writer"],
            model="claude-3-5-sonnet",
            effort="high",
            is_bundled=True,
            source_path=None,
        )
        skill_registry.get = MagicMock(return_value=skill)
        mod.show_skill_detail("full-skill")

    def test_found_non_bundled_with_source(self):
        import dazi.repl_display as mod
        from dazi._singletons import skill_registry

        skill = Skill(
            name="project-skill",
            description="A project-level skill",
            prompt="prompt",
            version="1.0",
            is_bundled=False,
            source_path=Path("/project/.dazi/skills/project-skill"),
        )
        skill_registry.get = MagicMock(return_value=skill)
        mod.show_skill_detail("project-skill")


# ─────────────────────────────────────────────────────────
# render_user_panel / render_dazi_panel
# ─────────────────────────────────────────────────────────


class TestRenderPanels:
    def test_render_user_panel(self):
        import dazi.repl_display as mod

        con = _console()
        mod.render_user_panel("Hello, world!", con)

    def test_render_dazi_panel(self):
        import dazi.repl_display as mod

        con = _console()
        mod.render_dazi_panel("I am Dazi.", con)


# ─────────────────────────────────────────────────────────
# add_demo_hook
# ─────────────────────────────────────────────────────────


class TestAddDemoHook:
    @pytest.mark.asyncio
    async def test_registers_hook_and_calls_inner(self):
        import dazi.graph as _graph_mod
        import dazi.repl_display as mod

        registered_fn = None

        def capture_register(event, fn, priority=None):
            nonlocal registered_fn
            registered_fn = fn

        original_register = _graph_mod.hook_registry.register
        _graph_mod.hook_registry.register = MagicMock(side_effect=capture_register)
        try:
            mod.add_demo_hook()
            assert registered_fn is not None
            # Call the inner async hook to cover lines 567-569
            result = await registered_fn(tool_name="test_tool", tool_args={"key": "val"})
            assert result is not None
        finally:
            _graph_mod.hook_registry.register = original_register


# ─────────────────────────────────────────────────────────
# print_ascii_banner
# ─────────────────────────────────────────────────────────


class TestPrintAsciiBanner:
    @patch("dazi.config.PROJECT_ROOT", Path("/home/user/project"))
    def test_banner(self):
        import dazi.repl_display as mod

        con = _console()
        mod.print_ascii_banner(con, version="0.1.0")

    @patch("dazi.config.PROJECT_ROOT", Path("/home/user/project"))
    def test_banner_different_version(self):
        import dazi.repl_display as mod

        con = _console()
        mod.print_ascii_banner(con, version="1.2.3")

    @patch("dazi.config.PROJECT_ROOT", Path("/home/user/project"))
    def test_banner_midnight(self):
        """Test hour < 6 branch: 'Late night coding'."""
        from datetime import datetime

        import dazi.repl_display as mod

        mock_now = datetime(2025, 6, 15, 3, 30)
        con = _console()
        with patch("datetime.datetime") as mock_dt:
            mock_dt.now.return_value = mock_now
            mod.print_ascii_banner(con, version="0.1.0")

    @patch("dazi.config.PROJECT_ROOT", Path("/home/user/project"))
    def test_banner_afternoon(self):
        """Test 12 <= hour < 17 branch: 'Good afternoon'."""
        from datetime import datetime

        import dazi.repl_display as mod

        mock_now = datetime(2025, 6, 15, 14, 30)
        con = _console()
        with patch("datetime.datetime") as mock_dt:
            mock_dt.now.return_value = mock_now
            mod.print_ascii_banner(con, version="0.1.0")

    @patch("dazi.config.PROJECT_ROOT", Path("/home/user/project"))
    def test_banner_evening(self):
        """Test 17 <= hour < 21 branch: 'Good evening'."""
        from datetime import datetime

        import dazi.repl_display as mod

        mock_now = datetime(2025, 6, 15, 19, 30)
        con = _console()
        with patch("datetime.datetime") as mock_dt:
            mock_dt.now.return_value = mock_now
            mod.print_ascii_banner(con, version="0.1.0")

    @patch("dazi.config.PROJECT_ROOT", Path("/home/user/project"))
    def test_banner_late_night(self):
        """Test hour >= 21 branch: 'Late night coding'."""
        from datetime import datetime

        import dazi.repl_display as mod

        mock_now = datetime(2025, 6, 15, 23, 30)
        con = _console()
        with patch("datetime.datetime") as mock_dt:
            mock_dt.now.return_value = mock_now
            mod.print_ascii_banner(con, version="0.1.0")


# ─────────────────────────────────────────────────────────
# print_welcome_message
# ─────────────────────────────────────────────────────────


class TestPrintWelcomeMessage:
    def test_basic_welcome(self):
        import dazi.repl_display as mod
        import dazi.repl_teams as _teams

        original_team_name = _teams.active_team_name
        original_team_store = _teams.team_task_store
        _teams.active_team_name = None
        _teams.team_task_store = None
        try:
            with (
                patch.object(mod, "_get_model_name", return_value="gpt-4o"),
                patch("dazi.tokenizer.get_context_window", return_value=128000),
            ):
                con = _console()
                mod.print_welcome_message(
                    con,
                    skill_count=5,
                    team_count=2,
                    dazimd_files=[Path("/project/DAZI.md")],
                )
        finally:
            _teams.active_team_name = original_team_name
            _teams.team_task_store = original_team_store

    def test_welcome_with_team_active(self, tmp_path: Path):
        import dazi.repl_display as mod
        import dazi.repl_teams as _teams

        team_store = TaskStore(tmp_path / "team_tasks", list_id="team1")
        original_team_name = _teams.active_team_name
        original_team_store = _teams.team_task_store
        _teams.active_team_name = "my-team"
        _teams.team_task_store = team_store
        try:
            with (
                patch.object(mod, "_get_model_name", return_value="gpt-4o"),
                patch("dazi.tokenizer.get_context_window", return_value=128000),
            ):
                con = _console()
                mod.print_welcome_message(
                    con,
                    skill_count=10,
                    team_count=3,
                    dazimd_files=[],
                )
        finally:
            _teams.active_team_name = original_team_name
            _teams.team_task_store = original_team_store
