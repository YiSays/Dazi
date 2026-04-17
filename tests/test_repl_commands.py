"""Tests for dazi/repl_commands.py — handle_command dispatch."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from rich.console import Console

from tests.helpers.mock_singletons import patch_singletons


@pytest.fixture(autouse=True)
def _patch(monkeypatch, tmp_path: Path):
    patch_singletons(monkeypatch, tmp_path)


def _console():
    return Console(file=MagicMock(), force_terminal=False)


def _state(**overrides):
    s = {"mode": "execute", "messages": []}
    s.update(overrides)
    return s


def _session():
    return MagicMock()


# ─────────────────────────────────────────────────────────
# Commands
# ─────────────────────────────────────────────────────────


class TestHandleCommand:
    """All handle_command tests — it's async so every test must await."""

    @pytest.mark.asyncio
    async def test_quit(self):
        import dazi.repl_commands as mod

        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr(mod, "_teams", MagicMock(active_team_name=None))
        result = await mod.handle_command(
            "/quit",
            state=_state(),
            session=_session(),
            console=_console(),
            dazimd_files=[],
            print_welcome_fn=MagicMock(),
        )
        assert result == "break"

    @pytest.mark.asyncio
    async def test_help(self):
        import dazi.repl_commands as mod

        result = await mod.handle_command(
            "/help",
            state=_state(),
            session=_session(),
            console=_console(),
            dazimd_files=[],
            print_welcome_fn=MagicMock(),
        )
        assert result == "continue"

    @pytest.mark.asyncio
    async def test_cost(self, monkeypatch):
        import dazi.repl_commands as mod

        monkeypatch.setattr(
            mod, "cost_tracker", MagicMock(format_summary=MagicMock(return_value="$0.01"))
        )
        result = await mod.handle_command(
            "/cost",
            state=_state(),
            session=_session(),
            console=_console(),
            dazimd_files=[],
            print_welcome_fn=MagicMock(),
        )
        assert result == "continue"

    @pytest.mark.asyncio
    async def test_cost_last(self, monkeypatch):
        import dazi.repl_commands as mod

        monkeypatch.setattr(
            mod, "cost_tracker", MagicMock(format_last_session=MagicMock(return_value="$0.05"))
        )
        result = await mod.handle_command(
            "/cost last",
            state=_state(),
            session=_session(),
            console=_console(),
            dazimd_files=[],
            print_welcome_fn=MagicMock(),
        )
        assert result == "continue"

    @pytest.mark.asyncio
    async def test_settings(self, monkeypatch):
        import dazi.repl_commands as mod

        s = MagicMock()
        s.model = "gpt-4o"
        s.api_base_url = "https://api.example.com"
        s.default_mode = "default"
        s.auto_compact = True
        s.auto_memory = True
        s.max_concurrent_tools = 5
        s.allow_rules = []
        s.deny_rules = []
        s.env = {}
        monkeypatch.setattr(
            mod,
            "settings_manager",
            MagicMock(
                settings=s,
                source_map={"model": "user"},
                user_path="/u/.dazi/settings.json",
                project_path="/p/.dazi/settings.json",
            ),
        )
        result = await mod.handle_command(
            "/settings",
            state=_state(),
            session=_session(),
            console=_console(),
            dazimd_files=[],
            print_welcome_fn=MagicMock(),
        )
        assert result == "continue"

    @pytest.mark.asyncio
    async def test_onboard(self, monkeypatch):
        import dazi.repl_commands as mod

        monkeypatch.setattr("dazi.onboard.run_onboarding", MagicMock())
        monkeypatch.setattr(mod, "settings_manager", MagicMock())
        result = await mod.handle_command(
            "/onboard",
            state=_state(),
            session=_session(),
            console=_console(),
            dazimd_files=[],
            print_welcome_fn=MagicMock(),
        )
        assert result == "continue"

    @pytest.mark.asyncio
    async def test_mcp(self, monkeypatch):
        import dazi.repl_commands as mod

        monkeypatch.setattr(mod, "show_mcp_servers_table", MagicMock())
        result = await mod.handle_command(
            "/mcp",
            state=_state(),
            session=_session(),
            console=_console(),
            dazimd_files=[],
            print_welcome_fn=MagicMock(),
        )
        assert result == "continue"

    @pytest.mark.asyncio
    async def test_clear(self, monkeypatch):
        import os

        import dazi.repl_commands as mod

        mock_system = MagicMock()
        monkeypatch.setattr(os, "name", "posix")
        monkeypatch.setattr(os, "system", mock_system)
        result = await mod.handle_command(
            "/clear",
            state=_state(),
            session=_session(),
            console=_console(),
            dazimd_files=[],
            print_welcome_fn=MagicMock(),
        )
        assert result == "continue"
        mock_system.assert_called_once_with("clear")

    @pytest.mark.asyncio
    async def test_plan(self):
        import dazi.repl_commands as mod

        state = _state(mode="execute")
        result = await mod.handle_command(
            "/plan",
            state=state,
            session=_session(),
            console=_console(),
            dazimd_files=[],
            print_welcome_fn=MagicMock(),
        )
        assert result == "continue"
        assert state["mode"] == "plan"

    @pytest.mark.asyncio
    async def test_plan_already(self):
        import dazi.repl_commands as mod

        state = _state(mode="plan")
        result = await mod.handle_command(
            "/plan",
            state=state,
            session=_session(),
            console=_console(),
            dazimd_files=[],
            print_welcome_fn=MagicMock(),
        )
        assert result == "continue"

    @pytest.mark.asyncio
    async def test_go(self, monkeypatch, tmp_path):
        import dazi.repl_commands as mod

        pf = tmp_path / "plan.md"
        pf.write_text("# Plan")
        monkeypatch.setattr(mod, "PLAN_FILE", pf)
        state = _state(mode="plan")
        result = await mod.handle_command(
            "/go",
            state=state,
            session=_session(),
            console=_console(),
            dazimd_files=[],
            print_welcome_fn=MagicMock(),
        )
        assert result == "continue"
        assert state["mode"] == "execute"

    @pytest.mark.asyncio
    async def test_go_not_in_plan(self):
        import dazi.repl_commands as mod

        state = _state(mode="execute")
        result = await mod.handle_command(
            "/go",
            state=state,
            session=_session(),
            console=_console(),
            dazimd_files=[],
            print_welcome_fn=MagicMock(),
        )
        assert result == "continue"

    @pytest.mark.asyncio
    async def test_show(self, monkeypatch, tmp_path):
        import dazi.repl_commands as mod

        pf = tmp_path / "plan.md"
        pf.write_text("# P")
        monkeypatch.setattr(mod, "PLAN_FILE", pf)
        result = await mod.handle_command(
            "/show",
            state=_state(),
            session=_session(),
            console=_console(),
            dazimd_files=[],
            print_welcome_fn=MagicMock(),
        )
        assert result == "continue"

    @pytest.mark.asyncio
    async def test_show_no_plan(self, monkeypatch, tmp_path):
        import dazi.repl_commands as mod

        monkeypatch.setattr(mod, "PLAN_FILE", tmp_path / "no.md")
        result = await mod.handle_command(
            "/show",
            state=_state(),
            session=_session(),
            console=_console(),
            dazimd_files=[],
            print_welcome_fn=MagicMock(),
        )
        assert result == "continue"

    @pytest.mark.asyncio
    async def test_tools(self, monkeypatch):
        import dazi.repl_commands as mod

        mt = MagicMock()
        mt.name = "file_reader"
        mm = MagicMock()
        mm.safety.value = "safe"
        mm.is_concurrency_safe = True
        mm.description = "Read"
        mm.name = "file_reader"
        monkeypatch.setattr(mod, "EXECUTE_MODE_TOOLS", [mt])
        monkeypatch.setattr(mod, "EXECUTE_MODE_META", {"file_reader": mm})
        result = await mod.handle_command(
            "/tools",
            state=_state(),
            session=_session(),
            console=_console(),
            dazimd_files=[],
            print_welcome_fn=MagicMock(),
        )
        assert result == "continue"

    @pytest.mark.asyncio
    async def test_rules(self, monkeypatch):
        import dazi.repl_commands as mod

        monkeypatch.setattr(mod, "list_rules_table", MagicMock())
        result = await mod.handle_command(
            "/rules",
            state=_state(),
            session=_session(),
            console=_console(),
            dazimd_files=[],
            print_welcome_fn=MagicMock(),
        )
        assert result == "continue"

    @pytest.mark.asyncio
    async def test_allow(self, monkeypatch):
        import dazi.repl_commands as mod

        r = MagicMock()
        r.tool_name = "file_reader"
        r.pattern = "/tmp/*"
        monkeypatch.setattr(mod, "parse_rule", MagicMock(return_value=r))
        monkeypatch.setattr(mod, "permission_rules", [])
        result = await mod.handle_command(
            "/allow file_reader /tmp/*",
            state=_state(),
            session=_session(),
            console=_console(),
            dazimd_files=[],
            print_welcome_fn=MagicMock(),
        )
        assert result == "continue"

    @pytest.mark.asyncio
    async def test_allow_invalid(self, monkeypatch):
        import dazi.repl_commands as mod

        monkeypatch.setattr(mod, "parse_rule", MagicMock(side_effect=ValueError("bad")))
        result = await mod.handle_command(
            "/allow x",
            state=_state(),
            session=_session(),
            console=_console(),
            dazimd_files=[],
            print_welcome_fn=MagicMock(),
        )
        assert result == "continue"

    @pytest.mark.asyncio
    async def test_deny(self, monkeypatch):
        import dazi.repl_commands as mod

        r = MagicMock()
        r.tool_name = "shell_exec"
        r.pattern = "rm *"
        monkeypatch.setattr(mod, "parse_rule", MagicMock(return_value=r))
        monkeypatch.setattr(mod, "permission_rules", [])
        result = await mod.handle_command(
            "/deny shell_exec rm *",
            state=_state(),
            session=_session(),
            console=_console(),
            dazimd_files=[],
            print_welcome_fn=MagicMock(),
        )
        assert result == "continue"

    @pytest.mark.asyncio
    async def test_hooks(self, monkeypatch):
        import dazi.repl_commands as mod

        monkeypatch.setattr(mod, "hook_registry", MagicMock(list_hooks=MagicMock(return_value={})))
        result = await mod.handle_command(
            "/hooks",
            state=_state(),
            session=_session(),
            console=_console(),
            dazimd_files=[],
            print_welcome_fn=MagicMock(),
        )
        assert result == "continue"

    @pytest.mark.asyncio
    async def test_hook_demo(self, monkeypatch):
        import dazi.repl_commands as mod

        monkeypatch.setattr(mod, "add_demo_hook", MagicMock())
        result = await mod.handle_command(
            "/hook",
            state=_state(),
            session=_session(),
            console=_console(),
            dazimd_files=[],
            print_welcome_fn=MagicMock(),
        )
        assert result == "continue"

    @pytest.mark.asyncio
    async def test_remember(self, monkeypatch):
        import dazi.repl_commands as mod

        monkeypatch.setattr(mod, "memory_store", MagicMock())
        result = await mod.handle_command(
            "/remember test memory",
            state=_state(),
            session=_session(),
            console=_console(),
            dazimd_files=[],
            print_welcome_fn=MagicMock(),
        )
        assert result == "continue"

    @pytest.mark.asyncio
    async def test_remember_empty(self):
        import dazi.repl_commands as mod

        result = await mod.handle_command(
            "/remember ",
            state=_state(),
            session=_session(),
            console=_console(),
            dazimd_files=[],
            print_welcome_fn=MagicMock(),
        )
        assert result == "continue"

    @pytest.mark.asyncio
    async def test_forget(self, monkeypatch):
        import dazi.repl_commands as mod

        e = MagicMock()
        e.id = "abc123"
        monkeypatch.setattr(mod, "memory_store", MagicMock(list_all=MagicMock(return_value=[e])))
        result = await mod.handle_command(
            "/forget abc",
            state=_state(),
            session=_session(),
            console=_console(),
            dazimd_files=[],
            print_welcome_fn=MagicMock(),
        )
        assert result == "continue"

    @pytest.mark.asyncio
    async def test_forget_not_found(self, monkeypatch):
        import dazi.repl_commands as mod

        monkeypatch.setattr(
            mod,
            "memory_store",
            MagicMock(
                list_all=MagicMock(return_value=[]),
                delete=MagicMock(return_value=False),
            ),
        )
        result = await mod.handle_command(
            "/forget xyz",
            state=_state(),
            session=_session(),
            console=_console(),
            dazimd_files=[],
            print_welcome_fn=MagicMock(),
        )
        assert result == "continue"

    @pytest.mark.asyncio
    async def test_memories(self, monkeypatch):
        import dazi.repl_commands as mod

        monkeypatch.setattr(mod, "list_memories_table", MagicMock())
        result = await mod.handle_command(
            "/memories",
            state=_state(),
            session=_session(),
            console=_console(),
            dazimd_files=[],
            print_welcome_fn=MagicMock(),
        )
        assert result == "continue"

    @pytest.mark.asyncio
    async def test_dazimd(self, monkeypatch):
        import dazi.repl_commands as mod

        monkeypatch.setattr(mod, "show_dazimd_files", MagicMock())
        result = await mod.handle_command(
            "/dazimd",
            state=_state(),
            session=_session(),
            console=_console(),
            dazimd_files=[],
            print_welcome_fn=MagicMock(),
        )
        assert result == "continue"

    @pytest.mark.asyncio
    async def test_reindex(self, monkeypatch):
        import dazi.repl_commands as mod

        monkeypatch.setattr(mod, "memory_store", MagicMock())
        result = await mod.handle_command(
            "/reindex",
            state=_state(),
            session=_session(),
            console=_console(),
            dazimd_files=[],
            print_welcome_fn=MagicMock(),
        )
        assert result == "continue"

    @pytest.mark.asyncio
    async def test_tokens(self, monkeypatch):
        import dazi.repl_commands as mod

        monkeypatch.setattr(mod, "show_token_info", MagicMock())
        result = await mod.handle_command(
            "/tokens",
            state=_state(),
            session=_session(),
            console=_console(),
            dazimd_files=[],
            print_welcome_fn=MagicMock(),
        )
        assert result == "continue"

    @pytest.mark.asyncio
    async def test_tasks(self, monkeypatch):
        import dazi.repl_commands as mod

        monkeypatch.setattr(mod, "list_tasks_table", MagicMock())
        result = await mod.handle_command(
            "/tasks",
            state=_state(),
            session=_session(),
            console=_console(),
            dazimd_files=[],
            print_welcome_fn=MagicMock(),
        )
        assert result == "continue"

    @pytest.mark.asyncio
    async def test_task(self, monkeypatch):
        import dazi.repl_commands as mod

        monkeypatch.setattr(mod, "show_task_detail", MagicMock())
        result = await mod.handle_command(
            "/task 1",
            state=_state(),
            session=_session(),
            console=_console(),
            dazimd_files=[],
            print_welcome_fn=MagicMock(),
        )
        assert result == "continue"

    @pytest.mark.asyncio
    async def test_task_invalid(self):
        import dazi.repl_commands as mod

        result = await mod.handle_command(
            "/task abc",
            state=_state(),
            session=_session(),
            console=_console(),
            dazimd_files=[],
            print_welcome_fn=MagicMock(),
        )
        assert result == "continue"

    @pytest.mark.asyncio
    async def test_bg(self, monkeypatch):
        import dazi.repl_commands as mod

        monkeypatch.setattr(mod, "show_background_tasks_table", MagicMock())
        result = await mod.handle_command(
            "/bg",
            state=_state(),
            session=_session(),
            console=_console(),
            dazimd_files=[],
            print_welcome_fn=MagicMock(),
        )
        assert result == "continue"

    @pytest.mark.asyncio
    async def test_bg_detail(self, monkeypatch):
        import dazi.repl_commands as mod

        monkeypatch.setattr(mod, "show_background_task_detail", MagicMock())
        result = await mod.handle_command(
            "/bg 123",
            state=_state(),
            session=_session(),
            console=_console(),
            dazimd_files=[],
            print_welcome_fn=MagicMock(),
        )
        assert result == "continue"

    @pytest.mark.asyncio
    async def test_skills(self, monkeypatch):
        import dazi.repl_commands as mod

        monkeypatch.setattr(mod, "show_skills_table", MagicMock())
        result = await mod.handle_command(
            "/skills",
            state=_state(),
            session=_session(),
            console=_console(),
            dazimd_files=[],
            print_welcome_fn=MagicMock(),
        )
        assert result == "continue"

    @pytest.mark.asyncio
    async def test_skill(self, monkeypatch):
        import dazi.repl_commands as mod

        monkeypatch.setattr(mod, "show_skill_detail", MagicMock())
        result = await mod.handle_command(
            "/skill myskill",
            state=_state(),
            session=_session(),
            console=_console(),
            dazimd_files=[],
            print_welcome_fn=MagicMock(),
        )
        assert result == "continue"

    @pytest.mark.asyncio
    async def test_teams(self, monkeypatch):
        import dazi.repl_commands as mod

        monkeypatch.setattr(mod, "_teams", MagicMock())
        result = await mod.handle_command(
            "/teams",
            state=_state(),
            session=_session(),
            console=_console(),
            dazimd_files=[],
            print_welcome_fn=MagicMock(),
        )
        assert result == "continue"

    @pytest.mark.asyncio
    async def test_team_create(self, monkeypatch):
        import dazi.repl_commands as mod

        t = MagicMock()
        t.name = "myteam"
        monkeypatch.setattr(mod, "_teams", MagicMock())
        monkeypatch.setattr(
            mod,
            "team_manager",
            MagicMock(
                create_team=MagicMock(return_value=t),
                _config_path=MagicMock(return_value="/p"),
                _task_dir=MagicMock(return_value="/p"),
            ),
        )
        result = await mod.handle_command(
            "/team create myteam",
            state=_state(),
            session=_session(),
            console=_console(),
            dazimd_files=[],
            print_welcome_fn=MagicMock(),
        )
        assert result == "continue"

    @pytest.mark.asyncio
    async def test_team_create_empty(self):
        import dazi.repl_commands as mod

        result = await mod.handle_command(
            "/team create ",
            state=_state(),
            session=_session(),
            console=_console(),
            dazimd_files=[],
            print_welcome_fn=MagicMock(),
        )
        assert result == "continue"

    @pytest.mark.asyncio
    async def test_team_leave_active(self, monkeypatch):
        import dazi.repl_commands as mod

        monkeypatch.setattr(mod, "_teams", MagicMock(active_team_name="t"))
        result = await mod.handle_command(
            "/team leave",
            state=_state(),
            session=_session(),
            console=_console(),
            dazimd_files=[],
            print_welcome_fn=MagicMock(),
        )
        assert result == "continue"

    @pytest.mark.asyncio
    async def test_team_leave_none(self, monkeypatch):
        import dazi.repl_commands as mod

        monkeypatch.setattr(mod, "_teams", MagicMock(active_team_name=None))
        result = await mod.handle_command(
            "/team leave",
            state=_state(),
            session=_session(),
            console=_console(),
            dazimd_files=[],
            print_welcome_fn=MagicMock(),
        )
        assert result == "continue"

    @pytest.mark.asyncio
    async def test_team_delete(self, monkeypatch):
        import dazi.repl_commands as mod

        monkeypatch.setattr(mod, "_teams", MagicMock(active_team_name=None))
        monkeypatch.setattr(
            mod, "team_manager", MagicMock(delete_team=MagicMock(return_value=True))
        )
        result = await mod.handle_command(
            "/team delete myteam",
            state=_state(),
            session=_session(),
            console=_console(),
            dazimd_files=[],
            print_welcome_fn=MagicMock(),
        )
        assert result == "continue"

    @pytest.mark.asyncio
    async def test_proactive_on(self, monkeypatch):
        import dazi.repl_commands as mod

        monkeypatch.setattr(mod, "proactive_manager", MagicMock())
        monkeypatch.setattr(mod, "_update_proactive_prompt", MagicMock())
        result = await mod.handle_command(
            "/proactive on",
            state=_state(),
            session=_session(),
            console=_console(),
            dazimd_files=[],
            print_welcome_fn=MagicMock(),
        )
        assert result == "continue"

    @pytest.mark.asyncio
    async def test_proactive_off(self, monkeypatch):
        import dazi.repl_commands as mod

        monkeypatch.setattr(mod, "proactive_manager", MagicMock())
        monkeypatch.setattr(mod, "_update_proactive_prompt", MagicMock())
        result = await mod.handle_command(
            "/proactive off",
            state=_state(),
            session=_session(),
            console=_console(),
            dazimd_files=[],
            print_welcome_fn=MagicMock(),
        )
        assert result == "continue"

    @pytest.mark.asyncio
    async def test_proactive_status(self, monkeypatch):
        import dazi.repl_commands as mod

        pm = MagicMock()
        pm.state.value = "active"
        pm.source.value = "command"
        pm.activation_count = 3
        pm.is_first_tick = True
        pm.last_tick_time = "10:00"
        monkeypatch.setattr(mod, "proactive_manager", pm)
        result = await mod.handle_command(
            "/proactive",
            state=_state(),
            session=_session(),
            console=_console(),
            dazimd_files=[],
            print_welcome_fn=MagicMock(),
        )
        assert result == "continue"

    @pytest.mark.asyncio
    async def test_autonomous(self, monkeypatch):
        import dazi.repl_commands as mod

        monkeypatch.setattr(
            mod, "autonomous_teammate", MagicMock(list_handles=MagicMock(return_value=[]))
        )
        result = await mod.handle_command(
            "/autonomous",
            state=_state(),
            session=_session(),
            console=_console(),
            dazimd_files=[],
            print_welcome_fn=MagicMock(),
        )
        assert result == "continue"

    @pytest.mark.asyncio
    async def test_worktree_list(self, monkeypatch):
        import dazi.repl_commands as mod

        monkeypatch.setattr(mod, "worktree_manager", MagicMock(list_all=MagicMock(return_value=[])))
        result = await mod.handle_command(
            "/worktree",
            state=_state(),
            session=_session(),
            console=_console(),
            dazimd_files=[],
            print_welcome_fn=MagicMock(),
        )
        assert result == "continue"

    @pytest.mark.asyncio
    async def test_worktree_create_error(self, monkeypatch):
        import dazi.repl_commands as mod

        monkeypatch.setattr(
            mod,
            "worktree_manager",
            MagicMock(
                create=MagicMock(side_effect=ValueError("exists")),
            ),
        )
        result = await mod.handle_command(
            "/worktree create mywt",
            state=_state(),
            session=_session(),
            console=_console(),
            dazimd_files=[],
            print_welcome_fn=MagicMock(),
        )
        assert result == "continue"

    @pytest.mark.asyncio
    async def test_worktree_finish_not_found(self, monkeypatch):
        import dazi.repl_commands as mod

        monkeypatch.setattr(
            mod,
            "worktree_manager",
            MagicMock(
                sanitize_agent_name=MagicMock(return_value="mywt"),
                get=MagicMock(return_value=None),
            ),
        )
        result = await mod.handle_command(
            "/worktree finish mywt",
            state=_state(),
            session=_session(),
            console=_console(),
            dazimd_files=[],
            print_welcome_fn=MagicMock(),
        )
        assert result == "continue"

    @pytest.mark.asyncio
    async def test_inbox(self, monkeypatch):
        import dazi.repl_commands as mod

        monkeypatch.setattr(mod, "_teams", MagicMock(show_inbox=AsyncMock()))
        result = await mod.handle_command(
            "/inbox",
            state=_state(),
            session=_session(),
            console=_console(),
            dazimd_files=[],
            print_welcome_fn=MagicMock(),
        )
        assert result == "continue"

    @pytest.mark.asyncio
    async def test_send(self, monkeypatch):
        import dazi.repl_commands as mod

        monkeypatch.setattr(mod, "_teams", MagicMock(send_repl_message=AsyncMock()))
        result = await mod.handle_command(
            "/send agent1 hello",
            state=_state(),
            session=_session(),
            console=_console(),
            dazimd_files=[],
            print_welcome_fn=MagicMock(),
        )
        assert result == "continue"

    @pytest.mark.asyncio
    async def test_send_usage(self):
        import dazi.repl_commands as mod

        result = await mod.handle_command(
            "/send agent1",
            state=_state(),
            session=_session(),
            console=_console(),
            dazimd_files=[],
            print_welcome_fn=MagicMock(),
        )
        assert result == "continue"

    @pytest.mark.asyncio
    async def test_broadcast(self, monkeypatch):
        import dazi.repl_commands as mod

        monkeypatch.setattr(mod, "_teams", MagicMock(broadcast_repl_message=AsyncMock()))
        result = await mod.handle_command(
            "/broadcast hello",
            state=_state(),
            session=_session(),
            console=_console(),
            dazimd_files=[],
            print_welcome_fn=MagicMock(),
        )
        assert result == "continue"

    @pytest.mark.asyncio
    async def test_shutdown(self, monkeypatch):
        import dazi.repl_commands as mod

        monkeypatch.setattr(mod, "_teams", MagicMock(send_shutdown_request=AsyncMock()))
        result = await mod.handle_command(
            "/shutdown agent1",
            state=_state(),
            session=_session(),
            console=_console(),
            dazimd_files=[],
            print_welcome_fn=MagicMock(),
        )
        assert result == "continue"

    @pytest.mark.asyncio
    async def test_compact_few_messages(self):
        import dazi.repl_commands as mod

        result = await mod.handle_command(
            "/compact",
            state=_state(messages=[MagicMock()]),
            session=_session(),
            console=_console(),
            dazimd_files=[],
            print_welcome_fn=MagicMock(),
        )
        assert result == "continue"

    @pytest.mark.asyncio
    async def test_not_a_command(self):
        import dazi.repl_commands as mod

        result = await mod.handle_command(
            "hello world",
            state=_state(),
            session=_session(),
            console=_console(),
            dazimd_files=[],
            print_welcome_fn=MagicMock(),
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_mcp_connect(self, monkeypatch):
        import dazi.repl_commands as mod
        from dazi.mcp_client import MCPServerStatus

        c = MagicMock()
        c.status = MCPServerStatus.CONNECTED
        c.tools = []
        monkeypatch.setattr(
            mod,
            "mcp_manager",
            MagicMock(
                get_server=MagicMock(return_value=c),
                connect_server=AsyncMock(return_value=True),
            ),
        )
        monkeypatch.setattr(mod, "rebuild_tool_lists", MagicMock())
        result = await mod.handle_command(
            "/mcp connect srv",
            state=_state(),
            session=_session(),
            console=_console(),
            dazimd_files=[],
            print_welcome_fn=MagicMock(),
        )
        assert result == "continue"

    @pytest.mark.asyncio
    async def test_mcp_disconnect(self, monkeypatch):
        import dazi.repl_commands as mod

        monkeypatch.setattr(mod, "mcp_manager", MagicMock(disconnect_server=AsyncMock()))
        monkeypatch.setattr(mod, "rebuild_tool_lists", MagicMock())
        result = await mod.handle_command(
            "/mcp disconnect srv",
            state=_state(),
            session=_session(),
            console=_console(),
            dazimd_files=[],
            print_welcome_fn=MagicMock(),
        )
        assert result == "continue"

    @pytest.mark.asyncio
    async def test_reload(self, monkeypatch):
        import dazi.repl_commands as mod

        monkeypatch.setattr(mod, "settings_manager", MagicMock())
        monkeypatch.setattr(mod, "skill_registry", MagicMock(reload=MagicMock(return_value=5)))
        monkeypatch.setattr(mod, "mcp_manager", MagicMock(disconnect_all=AsyncMock()))
        monkeypatch.setattr(mod, "connect_mcp_servers", AsyncMock())
        result = await mod.handle_command(
            "/reload",
            state=_state(),
            session=_session(),
            console=_console(),
            dazimd_files=[],
            print_welcome_fn=MagicMock(),
        )
        assert result == "continue"

    @pytest.mark.asyncio
    async def test_mcp_server_cmd(self, monkeypatch):
        import dazi.repl_commands as mod

        t = MagicMock()
        t.qualified_name = "tool1"
        s = MagicMock()
        s.status.value = "connected"
        s.tools = [t]
        monkeypatch.setattr(
            mod,
            "mcp_manager",
            MagicMock(
                get_server=MagicMock(return_value=s),
                call_tool=AsyncMock(return_value="ok"),
            ),
        )
        result = await mod.handle_command(
            "/testserver",
            state=_state(),
            session=_session(),
            console=_console(),
            dazimd_files=[],
            print_welcome_fn=MagicMock(),
        )
        assert result == "continue"

    @pytest.mark.asyncio
    async def test_skill_invocation(self, monkeypatch):
        import dazi.repl_commands as mod

        sk = MagicMock()
        sk.user_invocable = True
        monkeypatch.setattr(
            mod,
            "skill_registry",
            MagicMock(
                has_skill=MagicMock(return_value=True),
                get=MagicMock(return_value=sk),
                expand_skill=MagicMock(return_value="expanded"),
            ),
        )
        monkeypatch.setattr("dazi.graph.run_graph_turn", AsyncMock())
        result = await mod.handle_command(
            "/mysk arg",
            state=_state(messages=[]),
            session=_session(),
            console=_console(),
            dazimd_files=[],
            print_welcome_fn=MagicMock(),
        )
        assert result == "continue"

    # ── Additional coverage for missing branches ──

    @pytest.mark.asyncio
    async def test_mcp_connect_success(self, monkeypatch):
        """Lines 195-200: successful mcp connect."""
        import dazi.repl_commands as mod

        conn = MagicMock()
        conn.status.value = "connected"
        conn.tools = [MagicMock()]
        mock_mm = MagicMock()
        mock_mm.get_server.return_value = conn
        mock_mm.connect_server = AsyncMock(return_value=True)
        mock_rebuild = MagicMock()
        monkeypatch.setattr(mod, "mcp_manager", mock_mm)
        monkeypatch.setattr(mod, "rebuild_tool_lists", mock_rebuild)
        result = await mod.handle_command(
            "/mcp connect myserver",
            state=_state(),
            session=_session(),
            console=_console(),
            dazimd_files=[],
            print_welcome_fn=MagicMock(),
        )
        assert result == "continue"
        mock_rebuild.assert_called_once()

    @pytest.mark.asyncio
    async def test_mcp_connect_failed(self, monkeypatch):
        """Lines 202-203: failed mcp connect."""
        import dazi.repl_commands as mod

        conn = MagicMock()
        conn.error = "timeout"
        mock_mm = MagicMock()
        mock_mm.get_server.return_value = conn
        mock_mm.connect_server = AsyncMock(return_value=False)
        mock_rebuild = MagicMock()
        monkeypatch.setattr(mod, "mcp_manager", mock_mm)
        monkeypatch.setattr(mod, "rebuild_tool_lists", mock_rebuild)
        result = await mod.handle_command(
            "/mcp connect badserver",
            state=_state(),
            session=_session(),
            console=_console(),
            dazimd_files=[],
            print_welcome_fn=MagicMock(),
        )
        assert result == "continue"

    @pytest.mark.asyncio
    async def test_mcp_disconnect_no_name(self, monkeypatch):
        """Lines 213-215: mcp disconnect without name."""
        import dazi.repl_commands as mod

        monkeypatch.setattr(mod, "mcp_manager", MagicMock())
        result = await mod.handle_command(
            "/mcp disconnect",
            state=_state(),
            session=_session(),
            console=_console(),
            dazimd_files=[],
            print_welcome_fn=MagicMock(),
        )
        assert result == "continue"

    @pytest.mark.asyncio
    async def test_mcp_detail(self, monkeypatch):
        """Lines 215: mcp detail show."""
        import dazi.repl_commands as mod

        mock_show = MagicMock()
        monkeypatch.setattr(mod, "mcp_manager", MagicMock())
        monkeypatch.setattr(mod, "show_mcp_server_detail", mock_show)
        result = await mod.handle_command(
            "/mcp someserver",
            state=_state(),
            session=_session(),
            console=_console(),
            dazimd_files=[],
            print_welcome_fn=MagicMock(),
        )
        assert result == "continue"
        mock_show.assert_called_once_with("someserver")

    @pytest.mark.asyncio
    async def test_deny_invalid_rule(self, monkeypatch):
        """Lines 326-327: deny with invalid rule."""
        import dazi.repl_commands as mod

        mock_parse = MagicMock(side_effect=ValueError("bad rule"))
        monkeypatch.setattr(mod, "parse_rule", mock_parse)
        monkeypatch.setattr(mod, "permission_rules", [])
        result = await mod.handle_command(
            "/deny badrule",
            state=_state(),
            session=_session(),
            console=_console(),
            dazimd_files=[],
            print_welcome_fn=MagicMock(),
        )
        assert result == "continue"

    @pytest.mark.asyncio
    async def test_hooks_with_entries(self, monkeypatch):
        """Lines 336-337: hooks with registered entries."""
        import dazi.repl_commands as mod

        mock_registry = MagicMock(list_hooks=MagicMock(return_value={"on_start": [1, 2]}))
        monkeypatch.setattr(mod, "hook_registry", mock_registry)
        result = await mod.handle_command(
            "/hooks",
            state=_state(),
            session=_session(),
            console=_console(),
            dazimd_files=[],
            print_welcome_fn=MagicMock(),
        )
        assert result == "continue"

    @pytest.mark.asyncio
    async def test_forget_with_multiple(self, monkeypatch):
        """Lines 362-363, 370, 375: forget with multiple matches and confirm."""
        import dazi.repl_commands as mod

        mock_mem = MagicMock()
        mock_mem.list_all.return_value = [MagicMock(id="a1"), MagicMock(id="a2")]
        mock_mem.delete.return_value = True
        mock_mem.get.return_value = MagicMock()
        monkeypatch.setattr(mod, "memory_store", mock_mem)
        result = await mod.handle_command(
            "/forget test",
            state=_state(),
            session=_session(),
            console=_console(),
            dazimd_files=[],
            print_welcome_fn=MagicMock(),
        )
        assert result == "continue"

    @pytest.mark.asyncio
    async def test_compact_with_enough_messages(self, monkeypatch):
        """Lines 403-419: compact with enough messages."""
        import dazi.repl_commands as mod

        mock_compact_result = MagicMock()
        mock_compact_result.method = "summarize"
        mock_compact_result.tokens_before = 1000
        mock_compact_result.tokens_after = 400
        mock_compact_result.messages = []
        mock_compact_result.summary = "summarized"
        monkeypatch.setattr(mod, "manual_compact", AsyncMock(return_value=mock_compact_result))
        monkeypatch.setattr(mod, "_get_model_name", MagicMock(return_value="gpt-4o"))
        monkeypatch.setattr(mod, "_get_llm", MagicMock())
        monkeypatch.setattr(mod, "count_messages_tokens", MagicMock(return_value=1000))
        msgs = [MagicMock(), MagicMock(), MagicMock()]
        result = await mod.handle_command(
            "/compact",
            state=_state(messages=msgs),
            session=_session(),
            console=_console(),
            dazimd_files=[],
            print_welcome_fn=MagicMock(),
        )
        assert result == "continue"

    @pytest.mark.asyncio
    async def test_compact_none_method(self, monkeypatch):
        """Lines 417-418: compact with no compaction needed."""
        import dazi.repl_commands as mod

        mock_compact_result = MagicMock()
        mock_compact_result.method = "none"
        mock_compact_result.tokens_before = 500
        mock_compact_result.tokens_after = 500
        mock_compact_result.messages = []
        mock_compact_result.summary = ""
        monkeypatch.setattr(mod, "manual_compact", AsyncMock(return_value=mock_compact_result))
        monkeypatch.setattr(mod, "_get_model_name", MagicMock(return_value="gpt-4o"))
        monkeypatch.setattr(mod, "_get_llm", MagicMock())
        monkeypatch.setattr(mod, "count_messages_tokens", MagicMock(return_value=500))
        msgs = [MagicMock(), MagicMock()]
        result = await mod.handle_command(
            "/compact",
            state=_state(messages=msgs),
            session=_session(),
            console=_console(),
            dazimd_files=[],
            print_welcome_fn=MagicMock(),
        )
        assert result == "continue"

    @pytest.mark.asyncio
    async def test_team_create_error(self, monkeypatch):
        """Lines 487-488: team create exception."""
        import dazi.repl_commands as mod

        mock_tm = MagicMock(create_team=MagicMock(side_effect=Exception("fail")))
        monkeypatch.setattr(mod, "team_manager", mock_tm)
        result = await mod.handle_command(
            "/team create badteam",
            state=_state(),
            session=_session(),
            console=_console(),
            dazimd_files=[],
            print_welcome_fn=MagicMock(),
        )
        assert result == "continue"

    @pytest.mark.asyncio
    async def test_team_delete_not_active(self, monkeypatch):
        """Lines 502, 507: team delete when not active team, team not found."""
        import dazi.repl_commands as mod

        mock_tm = MagicMock(delete_team=MagicMock(return_value=False))
        mock_teams = MagicMock(active_team_name=None)
        monkeypatch.setattr(mod, "team_manager", mock_tm)
        monkeypatch.setattr(mod, "_teams", mock_teams)
        result = await mod.handle_command(
            "/team delete ghost",
            state=_state(),
            session=_session(),
            console=_console(),
            dazimd_files=[],
            print_welcome_fn=MagicMock(),
        )
        assert result == "continue"

    @pytest.mark.asyncio
    async def test_team_delete_error(self, monkeypatch):
        """Lines 508-509: team delete exception."""
        import dazi.repl_commands as mod

        mock_tm = MagicMock(delete_team=MagicMock(side_effect=Exception("err")))
        mock_teams = MagicMock(active_team_name="t")
        monkeypatch.setattr(mod, "team_manager", mock_tm)
        monkeypatch.setattr(mod, "_teams", mock_teams)
        result = await mod.handle_command(
            "/team delete t",
            state=_state(),
            session=_session(),
            console=_console(),
            dazimd_files=[],
            print_welcome_fn=MagicMock(),
        )
        assert result == "continue"

    @pytest.mark.asyncio
    async def test_team_delete_no_name(self, monkeypatch):
        """Lines 499-500: team delete without name."""
        import dazi.repl_commands as mod

        result = await mod.handle_command(
            "/team delete",
            state=_state(),
            session=_session(),
            console=_console(),
            dazimd_files=[],
            print_welcome_fn=MagicMock(),
        )
        assert result == "continue"

    @pytest.mark.asyncio
    async def test_team_activate(self, monkeypatch):
        """Lines 512-515: team activate by name."""
        import dazi.repl_commands as mod

        mock_teams = MagicMock()
        monkeypatch.setattr(mod, "_teams", mock_teams)
        result = await mod.handle_command(
            "/team myteam",
            state=_state(),
            session=_session(),
            console=_console(),
            dazimd_files=[],
            print_welcome_fn=MagicMock(),
        )
        assert result == "continue"
        mock_teams.activate_team.assert_called_once_with("myteam")

    @pytest.mark.asyncio
    async def test_proactive_status_active(self, monkeypatch):
        """Lines 541, 543-548: proactive status when active."""
        import dazi.repl_commands as mod

        mock_pm = MagicMock()
        mock_pm.state.value = "active"
        mock_pm.source.value = "command"
        mock_pm.activation_count = 5
        mock_pm.is_first_tick = False
        mock_pm.last_tick_time = "2024-01-01"
        monkeypatch.setattr(mod, "proactive_manager", mock_pm)
        monkeypatch.setattr(mod, "_update_proactive_prompt", MagicMock())
        result = await mod.handle_command(
            "/proactive",
            state=_state(),
            session=_session(),
            console=_console(),
            dazimd_files=[],
            print_welcome_fn=MagicMock(),
        )
        assert result == "continue"

    @pytest.mark.asyncio
    async def test_autonomous_with_handles(self, monkeypatch):
        """Lines 557-565: autonomous with running handles."""
        import dazi.repl_commands as mod

        h = MagicMock()
        h.name = "agent1"
        h.team_name = "team1"
        h.status.value = "running"
        mock_at = MagicMock(list_handles=MagicMock(return_value=[h]), _tasks_claimed={"agent1": 3})
        monkeypatch.setattr(mod, "autonomous_teammate", mock_at)
        result = await mod.handle_command(
            "/autonomous",
            state=_state(),
            session=_session(),
            console=_console(),
            dazimd_files=[],
            print_welcome_fn=MagicMock(),
        )
        assert result == "continue"

    @pytest.mark.asyncio
    async def test_worktree_create_success(self, monkeypatch):
        """Lines 578-579: worktree create success."""
        import dazi.repl_commands as mod

        wt = MagicMock(path="/tmp/wt", branch="feature")
        mock_wm = MagicMock(create=MagicMock(return_value=wt))
        monkeypatch.setattr(mod, "worktree_manager", mock_wm)
        result = await mod.handle_command(
            "/worktree create mywt",
            state=_state(),
            session=_session(),
            console=_console(),
            dazimd_files=[],
            print_welcome_fn=MagicMock(),
        )
        assert result == "continue"

    @pytest.mark.asyncio
    async def test_worktree_finish_keep(self, monkeypatch):
        """Lines 594-599: worktree finish --keep."""
        import dazi.repl_commands as mod

        wt = MagicMock(path="/tmp/wt")
        mock_wm = MagicMock()
        mock_wm.sanitize_agent_name.return_value = "mywt"
        mock_wm.get.return_value = wt
        mock_wm.keep.return_value = "feature"
        monkeypatch.setattr(mod, "worktree_manager", mock_wm)
        result = await mod.handle_command(
            "/worktree finish mywt --keep",
            state=_state(),
            session=_session(),
            console=_console(),
            dazimd_files=[],
            print_welcome_fn=MagicMock(),
        )
        assert result == "continue"

    @pytest.mark.asyncio
    async def test_worktree_finish_remove_clean(self, monkeypatch):
        """Lines 600, 608-612: worktree finish --remove clean."""
        import dazi.repl_commands as mod

        wt = MagicMock(path="/tmp/wt")
        mock_wm = MagicMock()
        mock_wm.sanitize_agent_name.return_value = "mywt"
        mock_wm.get.return_value = wt
        mock_wm.has_uncommitted_changes.return_value = False
        mock_wm.remove.return_value = True
        monkeypatch.setattr(mod, "worktree_manager", mock_wm)
        result = await mod.handle_command(
            "/worktree finish mywt --remove",
            state=_state(),
            session=_session(),
            console=_console(),
            dazimd_files=[],
            print_welcome_fn=MagicMock(),
        )
        assert result == "continue"

    @pytest.mark.asyncio
    async def test_worktree_finish_remove_dirty(self, monkeypatch):
        """Lines 601-607: worktree finish --remove with uncommitted changes."""
        import dazi.repl_commands as mod

        wt = MagicMock(path="/tmp/wt")
        mock_wm = MagicMock()
        mock_wm.sanitize_agent_name.return_value = "mywt"
        mock_wm.get.return_value = wt
        mock_wm.has_uncommitted_changes.return_value = True
        monkeypatch.setattr(mod, "worktree_manager", mock_wm)
        result = await mod.handle_command(
            "/worktree finish mywt --remove",
            state=_state(),
            session=_session(),
            console=_console(),
            dazimd_files=[],
            print_welcome_fn=MagicMock(),
        )
        assert result == "continue"

    @pytest.mark.asyncio
    async def test_worktree_finish_remove_failed(self, monkeypatch):
        """Lines 611-612: worktree finish --remove failed."""
        import dazi.repl_commands as mod

        wt = MagicMock(path="/tmp/wt")
        mock_wm = MagicMock()
        mock_wm.sanitize_agent_name.return_value = "mywt"
        mock_wm.get.return_value = wt
        mock_wm.has_uncommitted_changes.return_value = False
        mock_wm.remove.return_value = False
        monkeypatch.setattr(mod, "worktree_manager", mock_wm)
        result = await mod.handle_command(
            "/worktree finish mywt --remove",
            state=_state(),
            session=_session(),
            console=_console(),
            dazimd_files=[],
            print_welcome_fn=MagicMock(),
        )
        assert result == "continue"

    @pytest.mark.asyncio
    async def test_worktree_finish_invalid_flag(self, monkeypatch):
        """Line 594: worktree finish with invalid flag defaults to keep."""
        import dazi.repl_commands as mod

        wt = MagicMock(path="/tmp/wt")
        mock_wm = MagicMock()
        mock_wm.sanitize_agent_name.return_value = "mywt"
        mock_wm.get.return_value = wt
        mock_wm.keep.return_value = "feature"
        monkeypatch.setattr(mod, "worktree_manager", mock_wm)
        result = await mod.handle_command(
            "/worktree finish mywt --invalid",
            state=_state(),
            session=_session(),
            console=_console(),
            dazimd_files=[],
            print_welcome_fn=MagicMock(),
        )
        assert result == "continue"

    @pytest.mark.asyncio
    async def test_worktree_list_with_items(self, monkeypatch):
        """Lines 618-630: worktree list with items."""
        import dazi.repl_commands as mod

        wt = MagicMock(agent_name="a1", branch="f1", path="/tmp/wt", id="wt1")
        mock_wm = MagicMock(
            list_all=MagicMock(return_value=[wt]),
            has_uncommitted_changes=MagicMock(return_value=True),
        )
        monkeypatch.setattr(mod, "worktree_manager", mock_wm)
        result = await mod.handle_command(
            "/worktree",
            state=_state(),
            session=_session(),
            console=_console(),
            dazimd_files=[],
            print_welcome_fn=MagicMock(),
        )
        assert result == "continue"

    @pytest.mark.asyncio
    async def test_broadcast_empty(self, monkeypatch):
        """Line 652: broadcast with empty message."""
        import dazi.repl_commands as mod

        mock_teams = MagicMock()
        monkeypatch.setattr(mod, "_teams", mock_teams)
        result = await mod.handle_command(
            "/broadcast ",
            state=_state(),
            session=_session(),
            console=_console(),
            dazimd_files=[],
            print_welcome_fn=MagicMock(),
        )
        assert result == "continue"

    @pytest.mark.asyncio
    async def test_shutdown_no_agent(self, monkeypatch):
        """Line 661: shutdown without agent name."""
        import dazi.repl_commands as mod

        mock_teams = MagicMock()
        monkeypatch.setattr(mod, "_teams", mock_teams)
        result = await mod.handle_command(
            "/shutdown ",
            state=_state(),
            session=_session(),
            console=_console(),
            dazimd_files=[],
            print_welcome_fn=MagicMock(),
        )
        assert result == "continue"

    @pytest.mark.asyncio
    async def test_mcp_server_call_error(self, monkeypatch):
        """Lines 677-678: MCP server slash command with error."""
        import dazi.repl_commands as mod

        t = MagicMock()
        t.qualified_name = "svr.tool"
        s = MagicMock()
        s.status.value = "connected"
        s.tools = [t]
        mock_mm = MagicMock(
            get_server=MagicMock(return_value=s),
            call_tool=AsyncMock(side_effect=Exception("fail")),
        )
        monkeypatch.setattr(mod, "mcp_manager", mock_mm)
        result = await mod.handle_command(
            "/svr",
            state=_state(),
            session=_session(),
            console=_console(),
            dazimd_files=[],
            print_welcome_fn=MagicMock(),
        )
        assert result == "continue"

    @pytest.mark.asyncio
    async def test_mcp_server_multiple_tools(self, monkeypatch):
        """Lines 680-684: MCP server with multiple tools."""
        import dazi.repl_commands as mod

        t1 = MagicMock(name="tool1", description="desc1")
        t2 = MagicMock(name="tool2", description="desc2")
        s = MagicMock()
        s.status.value = "connected"
        s.tools = [t1, t2]
        mock_mm = MagicMock(get_server=MagicMock(return_value=s))
        monkeypatch.setattr(mod, "mcp_manager", mock_mm)
        result = await mod.handle_command(
            "/svr",
            state=_state(),
            session=_session(),
            console=_console(),
            dazimd_files=[],
            print_welcome_fn=MagicMock(),
        )
        assert result == "continue"

    @pytest.mark.asyncio
    async def test_unknown_slash_not_server_not_skill(self, monkeypatch):
        """Lines 699-700: unknown slash command that's not a server or skill."""
        import dazi.repl_commands as mod

        monkeypatch.setattr(mod, "mcp_manager", MagicMock(get_server=MagicMock(return_value=None)))
        monkeypatch.setattr(
            mod, "skill_registry", MagicMock(has_skill=MagicMock(return_value=False))
        )
        result = await mod.handle_command(
            "/unknown",
            state=_state(messages=[]),
            session=_session(),
            console=_console(),
            dazimd_files=[],
            print_welcome_fn=MagicMock(),
        )
        assert result is None
