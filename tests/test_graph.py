"""Tests for dazi/graph.py — routing, rules, _consume_stream, nodes, and run_graph_turn."""

from __future__ import annotations

import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from tests.helpers.mock_singletons import patch_singletons


@pytest.fixture(autouse=True)
def _patch(monkeypatch, tmp_path: Path):
    patch_singletons(monkeypatch, tmp_path)


# ─────────────────────────────────────────────────────────
# _get_effective_rules
# ─────────────────────────────────────────────────────────


class TestGetEffectiveRules:
    def test_combines_settings_and_cli_rules(self, monkeypatch):
        import dazi.graph as graph_mod
        from dazi.permissions import PermissionBehavior, PermissionRule

        # Setup settings_manager mock to return a rule
        settings_rule = PermissionRule(
            behavior=PermissionBehavior.ALLOW,
            tool_name="file_reader",
            source="settings",
        )
        sm = MagicMock()
        sm.get_permission_rules.return_value = [settings_rule]
        monkeypatch.setattr(graph_mod, "settings_manager", sm)

        # Add a CLI rule
        cli_rule = PermissionRule(
            behavior=PermissionBehavior.DENY,
            tool_name="shell_exec",
            source="cli",
        )
        monkeypatch.setattr(graph_mod, "permission_rules", [cli_rule])

        result = graph_mod._get_effective_rules()
        assert len(result) == 2
        assert result[0].source == "settings"
        assert result[1].source == "cli"

    def test_empty_when_no_rules(self, monkeypatch):
        import dazi.graph as graph_mod

        sm = MagicMock()
        sm.get_permission_rules.return_value = []
        monkeypatch.setattr(graph_mod, "settings_manager", sm)
        monkeypatch.setattr(graph_mod, "permission_rules", [])

        result = graph_mod._get_effective_rules()
        assert result == []


# ─────────────────────────────────────────────────────────
# should_continue routing
# ─────────────────────────────────────────────────────────


class TestShouldContinue:
    def test_ai_message_with_tool_calls_routes_to_check_permissions(self):

        from dazi.graph import should_continue

        ai_msg = AIMessage(
            content="",
            tool_calls=[{"id": "tc1", "name": "file_reader", "args": {"file_path": "/tmp/x"}}],
        )
        state = {"messages": [HumanMessage(content="hi"), ai_msg]}
        result = should_continue(state)
        assert result == "check_permissions"

    def test_ai_message_without_tool_calls_routes_to_end(self):
        from langgraph.graph import END

        from dazi.graph import should_continue

        ai_msg = AIMessage(content="Hello! How can I help?")
        state = {"messages": [HumanMessage(content="hi"), ai_msg]}
        result = should_continue(state)
        assert result == END


# ─────────────────────────────────────────────────────────
# has_allowed_tools routing
# ─────────────────────────────────────────────────────────


class TestHasAllowedTools:
    def test_allowed_tools_routes_to_execute(self):
        from dazi.graph import has_allowed_tools

        state = {
            "messages": [],
            "allowed_tool_ids": ["tc1", "tc2"],
        }
        result = has_allowed_tools(state)
        assert result == "execute_tools"

    def test_empty_allowed_tools_routes_to_call_llm(self):
        from dazi.graph import has_allowed_tools

        state = {
            "messages": [],
            "allowed_tool_ids": [],
        }
        result = has_allowed_tools(state)
        assert result == "call_llm"

    def test_missing_allowed_tools_routes_to_call_llm(self):
        from dazi.graph import has_allowed_tools

        state = {"messages": []}
        result = has_allowed_tools(state)
        assert result == "call_llm"


# ─────────────────────────────────────────────────────────
# _build_full_tool_lists
# ─────────────────────────────────────────────────────────


class TestBuildFullToolLists:
    def test_returns_plan_and_execute_lists(self, monkeypatch):
        import dazi.graph as graph_mod

        # Ensure mcp_manager has no tools (clean state)
        mm = MagicMock()
        mm.build_langchain_tools.return_value = []
        monkeypatch.setattr(graph_mod, "mcp_manager", mm)

        execute_tools, plan_tools = graph_mod._build_full_tool_lists()
        assert len(execute_tools) > 0
        assert len(plan_tools) > 0
        # Execute should have more tools than plan
        assert len(execute_tools) >= len(plan_tools)


# ─────────────────────────────────────────────────────────
# _consume_stream
# ─────────────────────────────────────────────────────────


async def _async_events(*events: dict):
    """Helper: async generator yielding event dicts."""
    for e in events:
        yield e


class TestConsumeStream:
    @pytest.fixture(autouse=True)
    def _setup(self, monkeypatch):
        self.mock_console = MagicMock()
        monkeypatch.setattr("dazi.graph.console", self.mock_console)
        self.mock_render_panel = MagicMock()
        monkeypatch.setattr("dazi.repl_display.render_dazi_panel", self.mock_render_panel)
        self.mock_render_thinking_panel = MagicMock()
        monkeypatch.setattr(
            "dazi.repl_display.render_thinking_panel", self.mock_render_thinking_panel
        )
        self.mock_print_tool_call = MagicMock()
        monkeypatch.setattr("dazi.graph._print_tool_call_compact", self.mock_print_tool_call)
        self.mock_print_tool_result = MagicMock()
        monkeypatch.setattr("dazi.graph._print_tool_result_compact", self.mock_print_tool_result)

    @pytest.mark.asyncio
    async def test_accumulates_text_and_renders_panel(self):
        from dazi.graph import _consume_stream

        events = [
            {"event": "on_chat_model_stream", "data": {"chunk": SimpleNamespace(content="Hello ")}},
            {"event": "on_chat_model_stream", "data": {"chunk": SimpleNamespace(content="world")}},
            {"event": "on_chat_model_end", "data": {"output": SimpleNamespace(tool_calls=[])}},
        ]
        spinner = MagicMock()
        await _consume_stream(_async_events(*events), spinner=spinner)

        self.mock_render_panel.assert_called_once_with("Hello world", self.mock_console)
        spinner.update_label.assert_any_call("Responding...")

    @pytest.mark.asyncio
    async def test_tool_calls_printed_compact(self):
        from dazi.graph import _consume_stream

        tc = {"name": "file_reader", "args": {"path": "/tmp/x"}}
        events = [
            {"event": "on_chat_model_end", "data": {"output": SimpleNamespace(tool_calls=[tc])}},
        ]
        spinner = MagicMock()
        await _consume_stream(_async_events(*events), spinner=spinner)

        self.mock_print_tool_call.assert_called_once_with(tc)
        spinner.update_label.assert_any_call("Executing tools...")

    @pytest.mark.asyncio
    async def test_tool_end_renders_result(self):
        from dazi.graph import _consume_stream

        events = [
            {"event": "on_tool_end", "data": {"output": SimpleNamespace(content="file contents")}},
        ]
        spinner = MagicMock()
        await _consume_stream(_async_events(*events), spinner=spinner)

        self.mock_print_tool_result.assert_called_once_with("file contents", is_error=False)

    @pytest.mark.asyncio
    async def test_tool_end_truncates_long_output(self):
        from dazi.graph import _consume_stream

        events = [
            {"event": "on_tool_end", "data": {"output": SimpleNamespace(content="x" * 1000)}},
        ]
        spinner = MagicMock()
        await _consume_stream(_async_events(*events), spinner=spinner)

        content = self.mock_print_tool_result.call_args[0][0]
        assert "... (truncated)" in content
        assert len(content) < 1000

    @pytest.mark.asyncio
    async def test_error_tool_results(self):
        from dazi.graph import _consume_stream

        for prefix in ("DENIED", "BLOCKED", "REQUIRES"):
            self.mock_print_tool_result.reset_mock()
            events = [
                {
                    "event": "on_tool_end",
                    "data": {"output": SimpleNamespace(content=f"{prefix}: error")},
                },
            ]
            spinner = MagicMock()
            await _consume_stream(_async_events(*events), spinner=spinner)

            self.mock_print_tool_result.assert_called_once_with(f"{prefix}: error", is_error=True)

    @pytest.mark.asyncio
    async def test_edge_case_no_model_end_still_renders(self):
        from dazi.graph import _consume_stream

        events = [
            {"event": "on_chat_model_stream", "data": {"chunk": SimpleNamespace(content="Hello")}},
        ]
        spinner = MagicMock()
        await _consume_stream(_async_events(*events), spinner=spinner)

        self.mock_render_panel.assert_called_once_with("Hello", self.mock_console)

    @pytest.mark.asyncio
    async def test_empty_stream_no_output(self):
        from dazi.graph import _consume_stream

        spinner = MagicMock()
        await _consume_stream(_async_events(), spinner=spinner)

        self.mock_render_panel.assert_not_called()
        self.mock_print_tool_call.assert_not_called()
        self.mock_print_tool_result.assert_not_called()

    @pytest.mark.asyncio
    async def test_thinking_chunks_accumulated_and_rendered(self):
        from dazi.graph import _consume_stream

        events = [
            {
                "event": "on_chat_model_stream",
                "data": {
                    "chunk": SimpleNamespace(
                        content="",
                        additional_kwargs={"reasoning_content": "Let me consider..."},
                    )
                },
            },
            {
                "event": "on_chat_model_stream",
                "data": {
                    "chunk": SimpleNamespace(
                        content="",
                        additional_kwargs={"reasoning_content": " then decide."},
                    )
                },
            },
            {
                "event": "on_chat_model_stream",
                "data": {"chunk": SimpleNamespace(content="The answer.")},
            },
            {"event": "on_chat_model_end", "data": {"output": SimpleNamespace(tool_calls=[])}},
        ]
        spinner = MagicMock()
        await _consume_stream(_async_events(*events), spinner=spinner)

        self.mock_render_thinking_panel.assert_called_once_with(
            "Let me consider... then decide.", self.mock_console
        )
        self.mock_render_panel.assert_called_once_with("The answer.", self.mock_console)
        spinner.update_label.assert_any_call("Thinking...")

    @pytest.mark.asyncio
    async def test_thinking_with_tool_calls_shows_thinking_no_dazi_panel(self):
        from dazi.graph import _consume_stream

        tc = {"name": "file_reader", "args": {"path": "/tmp/x"}}
        events = [
            {
                "event": "on_chat_model_stream",
                "data": {
                    "chunk": SimpleNamespace(
                        content="",
                        additional_kwargs={"reasoning_content": "I should read the file."},
                    )
                },
            },
            {"event": "on_chat_model_end", "data": {"output": SimpleNamespace(tool_calls=[tc])}},
        ]
        spinner = MagicMock()
        await _consume_stream(_async_events(*events), spinner=spinner)

        self.mock_render_thinking_panel.assert_called_once()
        self.mock_render_panel.assert_not_called()
        self.mock_print_tool_call.assert_called_once_with(tc)


# ─────────────────────────────────────────────────────────
# rebuild_tool_lists
# ─────────────────────────────────────────────────────────


class TestRebuildToolLists:
    def test_rebuild_updates_globals(self, monkeypatch):
        import dazi.graph as graph_mod

        old_exec = list(graph_mod.EXECUTE_TOOLS_FULL)
        monkeypatch.setattr(graph_mod, "EXECUTE_TOOLS_FULL", old_exec)
        monkeypatch.setattr(graph_mod, "PLAN_TOOLS_FULL", list(graph_mod.PLAN_TOOLS_FULL))

        graph_mod.rebuild_tool_lists()
        assert graph_mod.EXECUTE_TOOLS_FULL is not old_exec


# ─────────────────────────────────────────────────────────
# connect_mcp_servers
# ─────────────────────────────────────────────────────────


class TestConnectMCPServers:
    @pytest.mark.asyncio
    async def test_no_servers_configured(self, monkeypatch):
        import dazi.graph as graph_mod

        sm = MagicMock()
        sm.get_mcp_servers.return_value = {}
        monkeypatch.setattr(graph_mod, "settings_manager", sm)

        mock_console = MagicMock()
        monkeypatch.setattr(graph_mod, "console", mock_console)

        await graph_mod.connect_mcp_servers()

        mock_console.print.assert_called_once_with(
            "[dim]No MCP servers configured in settings.[/dim]"
        )

    @pytest.mark.asyncio
    async def test_servers_connect_successfully(self, monkeypatch):
        import dazi.graph as graph_mod

        sm = MagicMock()
        sm.get_mcp_servers.return_value = {"test_server": {"command": "test", "args": []}}
        monkeypatch.setattr(graph_mod, "settings_manager", sm)

        mock_mcp = MagicMock()
        mock_conn = MagicMock()
        mock_conn.tools = ["tool1", "tool2"]
        mock_mcp.add_server = MagicMock()
        mock_mcp.connect_all = AsyncMock(return_value={"test_server": True})
        mock_mcp.get_server = MagicMock(return_value=mock_conn)
        mock_mcp.get_tools = MagicMock(return_value=["tool1", "tool2"])
        mock_mcp.build_langchain_tools = MagicMock(return_value=[])
        monkeypatch.setattr(graph_mod, "mcp_manager", mock_mcp)

        mock_console = MagicMock()
        monkeypatch.setattr(graph_mod, "console", mock_console)

        await graph_mod.connect_mcp_servers()

        # Should have printed connection messages
        calls = [str(c) for c in mock_console.print.call_args_list]
        assert any("test_server" in c for c in calls)

    @pytest.mark.asyncio
    async def test_server_config_invalid(self, monkeypatch):
        import dazi.graph as graph_mod

        sm = MagicMock()
        sm.get_mcp_servers.return_value = {"bad_server": {"invalid": "config"}}
        monkeypatch.setattr(graph_mod, "settings_manager", sm)

        mock_mcp = MagicMock()
        mock_mcp.connect_all = AsyncMock(return_value={})
        mock_mcp.get_tools = MagicMock(return_value=[])
        monkeypatch.setattr(graph_mod, "mcp_manager", mock_mcp)

        mock_console = MagicMock()
        monkeypatch.setattr(graph_mod, "console", mock_console)

        await graph_mod.connect_mcp_servers()

        calls = [str(c) for c in mock_console.print.call_args_list]
        assert any("Warning" in c for c in calls)

    @pytest.mark.asyncio
    async def test_server_connect_fails(self, monkeypatch):
        import dazi.graph as graph_mod

        sm = MagicMock()
        sm.get_mcp_servers.return_value = {"fail_server": {"command": "test", "args": []}}
        monkeypatch.setattr(graph_mod, "settings_manager", sm)

        mock_mcp = MagicMock()
        mock_conn = MagicMock()
        mock_conn.error = "connection refused"
        mock_mcp.add_server = MagicMock()
        mock_mcp.connect_all = AsyncMock(return_value={"fail_server": False})
        mock_mcp.get_server = MagicMock(return_value=mock_conn)
        mock_mcp.get_tools = MagicMock(return_value=[])
        monkeypatch.setattr(graph_mod, "mcp_manager", mock_mcp)

        mock_console = MagicMock()
        monkeypatch.setattr(graph_mod, "console", mock_console)

        await graph_mod.connect_mcp_servers()

        calls = [str(c) for c in mock_console.print.call_args_list]
        assert any("connection refused" in c for c in calls)


# ─────────────────────────────────────────────────────────
# check_compact
# ─────────────────────────────────────────────────────────


class TestCheckCompact:
    @pytest.mark.asyncio
    async def test_too_few_messages_returns_empty(self):
        from dazi.graph import check_compact

        state = {"messages": [HumanMessage(content="hi")]}
        result = await check_compact(state)
        assert result == {"messages": []}

    @pytest.mark.asyncio
    async def test_no_auto_compact_needed(self, monkeypatch):
        import dazi.graph as graph_mod

        monkeypatch.setattr(graph_mod, "should_auto_compact", MagicMock(return_value=False))
        monkeypatch.setattr(graph_mod, "_get_model_name", MagicMock(return_value="gpt-4o"))

        from dazi.graph import check_compact

        state = {"messages": [HumanMessage(content="hi")] * 5}
        result = await check_compact(state)
        assert result == {"messages": []}

    @pytest.mark.asyncio
    async def test_compact_succeeds(self, monkeypatch):
        import dazi.graph as graph_mod

        monkeypatch.setattr(
            graph_mod,
            "should_auto_compact",
            MagicMock(return_value=True),
        )
        monkeypatch.setattr(graph_mod, "_get_model_name", MagicMock(return_value="gpt-4o"))
        monkeypatch.setattr(graph_mod, "count_messages_tokens", MagicMock(return_value=100_000))
        monkeypatch.setattr(
            graph_mod,
            "get_context_window",
            MagicMock(return_value=128_000),
        )
        monkeypatch.setattr(graph_mod, "_get_llm", MagicMock())

        from types import SimpleNamespace as SN

        compact_result = SN(
            method="summarize",
            tokens_before=100_000,
            tokens_after=20_000,
            tool_results_cleared=5,
            rounds_removed=3,
            summary="Summarized successfully",
            messages=[HumanMessage(content="summarized")],
        )
        monkeypatch.setattr(graph_mod, "auto_compact", AsyncMock(return_value=compact_result))

        mock_console = MagicMock()
        monkeypatch.setattr(graph_mod, "console", mock_console)

        from dazi.graph import check_compact

        state = {"messages": [HumanMessage(content="hi")] * 5}
        result = await check_compact(state)
        assert result["messages"] == [HumanMessage(content="summarized")]
        assert graph_mod.consecutive_compact_failures == 0

    @pytest.mark.asyncio
    async def test_compact_fails_increment_counter(self, monkeypatch):
        import dazi.graph as graph_mod

        monkeypatch.setattr(
            graph_mod,
            "should_auto_compact",
            MagicMock(return_value=True),
        )
        monkeypatch.setattr(graph_mod, "_get_model_name", MagicMock(return_value="gpt-4o"))
        monkeypatch.setattr(graph_mod, "count_messages_tokens", MagicMock(return_value=100_000))
        monkeypatch.setattr(
            graph_mod,
            "get_context_window",
            MagicMock(return_value=128_000),
        )
        monkeypatch.setattr(graph_mod, "_get_llm", MagicMock())

        from types import SimpleNamespace as SN

        compact_result = SN(
            method="none",
            tokens_before=100_000,
            tokens_after=100_000,
            tool_results_cleared=None,
            rounds_removed=None,
            summary="Compact failed: too long",
            messages=[],
        )
        monkeypatch.setattr(graph_mod, "auto_compact", AsyncMock(return_value=compact_result))

        mock_console = MagicMock()
        monkeypatch.setattr(graph_mod, "console", mock_console)

        from dazi.graph import check_compact

        state = {"messages": [HumanMessage(content="hi")] * 5}
        result = await check_compact(state)
        assert result == {"messages": []}
        assert graph_mod.consecutive_compact_failures == 1

    @pytest.mark.asyncio
    async def test_compact_method_none_no_failure(self, monkeypatch):
        import dazi.graph as graph_mod

        monkeypatch.setattr(
            graph_mod,
            "should_auto_compact",
            MagicMock(return_value=True),
        )
        monkeypatch.setattr(graph_mod, "_get_model_name", MagicMock(return_value="gpt-4o"))
        monkeypatch.setattr(graph_mod, "count_messages_tokens", MagicMock(return_value=100_000))
        monkeypatch.setattr(
            graph_mod,
            "get_context_window",
            MagicMock(return_value=128_000),
        )
        monkeypatch.setattr(graph_mod, "_get_llm", MagicMock())

        from types import SimpleNamespace as SN

        compact_result = SN(
            method="none",
            tokens_before=100_000,
            tokens_after=100_000,
            tool_results_cleared=None,
            rounds_removed=None,
            summary="Skipped, no action needed",
            messages=[],
        )
        monkeypatch.setattr(graph_mod, "auto_compact", AsyncMock(return_value=compact_result))

        from dazi.graph import check_compact

        # Reset global counter since previous tests may have incremented it
        graph_mod.consecutive_compact_failures = 0

        state = {"messages": [HumanMessage(content="hi")] * 5}
        result = await check_compact(state)
        assert result == {"messages": []}
        # Counter should NOT increment when summary doesn't contain "failed"
        assert graph_mod.consecutive_compact_failures == 0


# ─────────────────────────────────────────────────────────
# call_llm
# ─────────────────────────────────────────────────────────


class TestCallLLM:
    def _make_mock_llm(self, response=None):
        """Create a mock LLM with astream returning an async iterator."""
        mock_llm = MagicMock()
        mock_llm.bind_tools = MagicMock(return_value=mock_llm)

        captured = {"messages": None}

        async def _astream(messages):
            captured["messages"] = messages
            if response is not None:
                yield response

        mock_llm.astream = _astream
        mock_llm._captured = captured
        return mock_llm

    @pytest.mark.asyncio
    async def test_call_llm_basic(self, monkeypatch):
        import dazi.graph as graph_mod

        ai_response = AIMessage(content="Hello!")
        ai_response.response_metadata = {
            "token_usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }
        mock_llm = self._make_mock_llm(ai_response)
        monkeypatch.setattr(graph_mod, "_get_llm", MagicMock(return_value=mock_llm))
        monkeypatch.setattr(graph_mod, "get_memory_content", MagicMock(return_value=""))
        monkeypatch.setattr(graph_mod, "get_skills_content", MagicMock(return_value=""))
        monkeypatch.setattr(graph_mod, "_get_effective_rules", MagicMock(return_value=[]))

        mock_prompt_builder = MagicMock()
        mock_prompt_builder.build.return_value = "You are a helpful assistant."
        monkeypatch.setattr(graph_mod, "prompt_builder", mock_prompt_builder)

        monkeypatch.setattr(graph_mod, "cost_tracker", MagicMock())

        from dazi.graph import call_llm

        state = {
            "messages": [HumanMessage(content="hello")],
            "mode": "execute",
        }
        result = await call_llm(state)
        assert len(result["messages"]) == 1
        assert result["messages"][0].content == "Hello!"

    @pytest.mark.asyncio
    async def test_call_llm_prepends_system_message(self, monkeypatch):
        import dazi.graph as graph_mod

        captured_messages = []
        ai_response = AIMessage(content="Hi!")
        ai_response.response_metadata = {"token_usage": {}}

        async def _astream(messages):
            captured_messages.extend(messages)
            yield ai_response

        mock_llm = MagicMock()
        mock_llm.bind_tools = MagicMock(return_value=mock_llm)
        mock_llm.astream = _astream

        monkeypatch.setattr(graph_mod, "_get_llm", MagicMock(return_value=mock_llm))
        monkeypatch.setattr(graph_mod, "get_memory_content", MagicMock(return_value=""))
        monkeypatch.setattr(graph_mod, "get_skills_content", MagicMock(return_value=""))
        monkeypatch.setattr(graph_mod, "_get_effective_rules", MagicMock(return_value=[]))

        mock_prompt_builder = MagicMock()
        mock_prompt_builder.build.return_value = "System prompt here"
        monkeypatch.setattr(graph_mod, "prompt_builder", mock_prompt_builder)

        monkeypatch.setattr(graph_mod, "cost_tracker", MagicMock())

        from dazi.graph import call_llm

        state = {
            "messages": [HumanMessage(content="hello")],
            "mode": "execute",
        }
        result = await call_llm(state)
        # The LLM should have been called with a SystemMessage prepended
        assert isinstance(captured_messages[0], SystemMessage)
        assert len(result["messages"]) == 1

    @pytest.mark.asyncio
    async def test_call_llm_replaces_existing_system_message(self, monkeypatch):
        import dazi.graph as graph_mod

        ai_response = AIMessage(content="Hi!")
        ai_response.response_metadata = {"token_usage": {}}
        mock_llm = self._make_mock_llm(ai_response)
        monkeypatch.setattr(graph_mod, "_get_llm", MagicMock(return_value=mock_llm))
        monkeypatch.setattr(graph_mod, "get_memory_content", MagicMock(return_value=""))
        monkeypatch.setattr(graph_mod, "get_skills_content", MagicMock(return_value=""))
        monkeypatch.setattr(graph_mod, "_get_effective_rules", MagicMock(return_value=[]))

        mock_prompt_builder = MagicMock()
        mock_prompt_builder.build.return_value = "New system prompt"
        monkeypatch.setattr(graph_mod, "prompt_builder", mock_prompt_builder)

        monkeypatch.setattr(graph_mod, "cost_tracker", MagicMock())

        from dazi.graph import call_llm

        state = {
            "messages": [
                SystemMessage(content="Old system prompt"),
                HumanMessage(content="hello"),
            ],
            "mode": "execute",
        }
        await call_llm(state)
        call_args = mock_llm._captured["messages"]
        assert isinstance(call_args[0], SystemMessage)
        assert call_args[0].content == "New system prompt"
        # Old system message should be replaced (only one SystemMessage)
        sys_count = sum(1 for m in call_args if isinstance(m, SystemMessage))
        assert sys_count == 1

    @pytest.mark.asyncio
    async def test_call_llm_empty_chunks(self, monkeypatch):
        import dazi.graph as graph_mod

        mock_llm = self._make_mock_llm(None)
        monkeypatch.setattr(graph_mod, "_get_llm", MagicMock(return_value=mock_llm))
        monkeypatch.setattr(graph_mod, "get_memory_content", MagicMock(return_value=""))
        monkeypatch.setattr(graph_mod, "get_skills_content", MagicMock(return_value=""))
        monkeypatch.setattr(graph_mod, "_get_effective_rules", MagicMock(return_value=[]))

        mock_prompt_builder = MagicMock()
        mock_prompt_builder.build.return_value = "System prompt"
        monkeypatch.setattr(graph_mod, "prompt_builder", mock_prompt_builder)

        from dazi.graph import call_llm

        state = {
            "messages": [HumanMessage(content="hello")],
            "mode": "execute",
        }
        result = await call_llm(state)
        assert result == {"messages": []}

    @pytest.mark.asyncio
    async def test_call_llm_plan_mode_uses_plan_tools(self, monkeypatch):
        import dazi.graph as graph_mod

        ai_response = AIMessage(content="Plan response")
        ai_response.response_metadata = {"token_usage": {}}
        mock_llm = self._make_mock_llm(ai_response)
        monkeypatch.setattr(graph_mod, "_get_llm", MagicMock(return_value=mock_llm))
        monkeypatch.setattr(graph_mod, "get_memory_content", MagicMock(return_value=""))
        monkeypatch.setattr(graph_mod, "get_skills_content", MagicMock(return_value=""))
        monkeypatch.setattr(graph_mod, "_get_effective_rules", MagicMock(return_value=[]))

        mock_prompt_builder = MagicMock()
        mock_prompt_builder.build.return_value = "Plan system prompt"
        monkeypatch.setattr(graph_mod, "prompt_builder", mock_prompt_builder)

        from dazi.graph import call_llm

        state = {
            "messages": [HumanMessage(content="hello")],
            "mode": "plan",
        }
        await call_llm(state)
        # bind_tools should have been called with PLAN_TOOLS_FULL
        mock_llm.bind_tools.assert_called_once_with(graph_mod.PLAN_TOOLS_FULL)

    @pytest.mark.asyncio
    async def test_call_llm_records_cost(self, monkeypatch):
        import dazi.graph as graph_mod

        ai_response = AIMessage(content="Hello!")
        ai_response.response_metadata = {
            "token_usage": {"prompt_tokens": 100, "completion_tokens": 50},
        }
        mock_llm = self._make_mock_llm(ai_response)
        monkeypatch.setattr(graph_mod, "_get_llm", MagicMock(return_value=mock_llm))
        monkeypatch.setattr(graph_mod, "get_memory_content", MagicMock(return_value=""))
        monkeypatch.setattr(graph_mod, "get_skills_content", MagicMock(return_value=""))
        monkeypatch.setattr(graph_mod, "_get_effective_rules", MagicMock(return_value=[]))

        mock_prompt_builder = MagicMock()
        mock_prompt_builder.build.return_value = "System prompt"
        monkeypatch.setattr(graph_mod, "prompt_builder", mock_prompt_builder)

        mock_cost_tracker = MagicMock()
        mock_settings = MagicMock()
        mock_settings.get_model_name.return_value = "gpt-4o"
        monkeypatch.setattr(graph_mod, "cost_tracker", mock_cost_tracker)
        monkeypatch.setattr(graph_mod, "settings_manager", mock_settings)

        from dazi.graph import call_llm

        state = {
            "messages": [HumanMessage(content="hello")],
            "mode": "execute",
        }
        await call_llm(state)
        mock_cost_tracker.record_usage.assert_called_once_with(
            model="gpt-4o",
            input_tokens=100,
            output_tokens=50,
        )

    @pytest.mark.asyncio
    async def test_call_llm_no_token_usage(self, monkeypatch):
        import dazi.graph as graph_mod

        ai_response = AIMessage(content="Hello!")
        ai_response.response_metadata = {"token_usage": None}
        mock_llm = self._make_mock_llm(ai_response)
        monkeypatch.setattr(graph_mod, "_get_llm", MagicMock(return_value=mock_llm))
        monkeypatch.setattr(graph_mod, "get_memory_content", MagicMock(return_value=""))
        monkeypatch.setattr(graph_mod, "get_skills_content", MagicMock(return_value=""))
        monkeypatch.setattr(graph_mod, "_get_effective_rules", MagicMock(return_value=[]))

        mock_prompt_builder = MagicMock()
        mock_prompt_builder.build.return_value = "System prompt"
        monkeypatch.setattr(graph_mod, "prompt_builder", mock_prompt_builder)

        mock_cost_tracker = MagicMock()
        monkeypatch.setattr(graph_mod, "cost_tracker", mock_cost_tracker)

        from dazi.graph import call_llm

        state = {
            "messages": [HumanMessage(content="hello")],
            "mode": "execute",
        }
        await call_llm(state)
        mock_cost_tracker.record_usage.assert_not_called()

    @pytest.mark.asyncio
    async def test_call_llm_has_plan_flag(self, monkeypatch):
        import dazi.graph as graph_mod

        ai_response = AIMessage(content="Plan response")
        ai_response.response_metadata = {"token_usage": {}}
        mock_llm = self._make_mock_llm(ai_response)
        monkeypatch.setattr(graph_mod, "_get_llm", MagicMock(return_value=mock_llm))
        monkeypatch.setattr(graph_mod, "get_memory_content", MagicMock(return_value=""))
        monkeypatch.setattr(graph_mod, "get_skills_content", MagicMock(return_value=""))
        monkeypatch.setattr(graph_mod, "_get_effective_rules", MagicMock(return_value=[]))

        mock_prompt_builder = MagicMock()
        mock_prompt_builder.build.return_value = "System prompt"
        monkeypatch.setattr(graph_mod, "prompt_builder", mock_prompt_builder)

        from dazi.graph import call_llm

        # With a plan file existing
        monkeypatch.setattr(graph_mod, "PLAN_FILE", MagicMock(exists=MagicMock(return_value=True)))

        state = {
            "messages": [HumanMessage(content="hello")],
            "mode": "execute",
        }
        await call_llm(state)
        # build should have been called with has_plan=True
        mock_prompt_builder.build.assert_called()
        kwargs = mock_prompt_builder.build.call_args[1]
        assert kwargs["has_plan"] is True


# ─────────────────────────────────────────────────────────
# check_permissions
# ─────────────────────────────────────────────────────────


class TestCheckPermissions:
    @pytest.mark.asyncio
    async def test_no_tool_calls_returns_empty(self):
        from dazi.graph import check_permissions

        state = {"messages": [AIMessage(content="No tools")], "mode": "execute"}
        result = await check_permissions(state)
        assert result == {"messages": []}

    @pytest.mark.asyncio
    async def test_non_ai_message_returns_empty(self):
        from dazi.graph import check_permissions

        state = {"messages": [HumanMessage(content="hello")]}
        result = await check_permissions(state)
        assert result == {"messages": []}

    @pytest.mark.asyncio
    async def test_all_allowed(self, monkeypatch):
        import dazi.graph as graph_mod
        from dazi.permissions import PermissionBehavior, PermissionResult

        monkeypatch.setattr(graph_mod, "_get_effective_rules", MagicMock(return_value=[]))
        monkeypatch.setattr(
            graph_mod,
            "check_permission",
            MagicMock(
                return_value=PermissionResult(behavior=PermissionBehavior.ALLOW, reason="ok"),
            ),
        )
        monkeypatch.setattr(
            graph_mod,
            "hook_registry",
            MagicMock(
                fire=AsyncMock(
                    return_value=MagicMock(
                        should_block=False,
                        permission_override=None,
                        modified_input=None,
                    )
                )
            ),
        )

        from dazi.graph import check_permissions

        ai_msg = AIMessage(
            content="",
            tool_calls=[{"id": "tc1", "name": "file_reader", "args": {"path": "/tmp/x"}}],
        )
        state = {"messages": [ai_msg], "mode": "execute"}
        result = await check_permissions(state)
        assert result["allowed_tool_ids"] == ["tc1"]
        assert result["messages"] == []

    @pytest.mark.asyncio
    async def test_all_denied(self, monkeypatch):
        import dazi.graph as graph_mod
        from dazi.permissions import PermissionBehavior, PermissionResult

        monkeypatch.setattr(graph_mod, "_get_effective_rules", MagicMock(return_value=[]))
        monkeypatch.setattr(
            graph_mod,
            "check_permission",
            MagicMock(
                return_value=PermissionResult(behavior=PermissionBehavior.DENY, reason="unsafe"),
            ),
        )
        monkeypatch.setattr(
            graph_mod,
            "hook_registry",
            MagicMock(
                fire=AsyncMock(
                    return_value=MagicMock(
                        should_block=False,
                        permission_override=None,
                        modified_input=None,
                    )
                )
            ),
        )

        from dazi.graph import check_permissions

        ai_msg = AIMessage(
            content="",
            tool_calls=[{"id": "tc1", "name": "shell_exec", "args": {"command": "rm -rf /"}}],
        )
        state = {"messages": [ai_msg], "mode": "execute"}
        result = await check_permissions(state)
        assert result["allowed_tool_ids"] == []
        assert len(result["messages"]) == 1
        assert "DENIED" in result["messages"][0].content

    @pytest.mark.asyncio
    async def test_hook_blocks_tool(self, monkeypatch):
        import dazi.graph as graph_mod

        hook_result = MagicMock(
            should_block=True,
            block_reason="not allowed by policy",
            permission_override=None,
            modified_input=None,
        )
        mock_registry = MagicMock(fire=AsyncMock(return_value=hook_result))
        monkeypatch.setattr(graph_mod, "hook_registry", mock_registry)

        from dazi.graph import check_permissions

        ai_msg = AIMessage(
            content="",
            tool_calls=[{"id": "tc1", "name": "file_reader", "args": {"path": "/etc/passwd"}}],
        )
        state = {"messages": [ai_msg], "mode": "execute"}
        result = await check_permissions(state)
        assert result["allowed_tool_ids"] == []
        assert "BLOCKED" in result["messages"][0].content

    @pytest.mark.asyncio
    async def test_hook_overrides_permission(self, monkeypatch):
        import dazi.graph as graph_mod
        from dazi.permissions import PermissionBehavior

        hook_result = MagicMock(
            should_block=False,
            permission_override=PermissionBehavior.ALLOW,
            modified_input=None,
        )
        mock_registry = MagicMock(fire=AsyncMock(return_value=hook_result))
        monkeypatch.setattr(graph_mod, "hook_registry", mock_registry)

        from dazi.graph import check_permissions

        ai_msg = AIMessage(
            content="",
            tool_calls=[{"id": "tc1", "name": "shell_exec", "args": {"command": "ls"}}],
        )
        state = {"messages": [ai_msg], "mode": "execute"}
        result = await check_permissions(state)
        assert result["allowed_tool_ids"] == ["tc1"]

    @pytest.mark.asyncio
    async def test_hook_modifies_input(self, monkeypatch):
        import dazi.graph as graph_mod
        from dazi.permissions import PermissionBehavior, PermissionResult

        hook_result = MagicMock(
            should_block=False,
            permission_override=None,
            modified_input={"path": "/safe/path"},
        )
        mock_registry = MagicMock(fire=AsyncMock(return_value=hook_result))
        monkeypatch.setattr(graph_mod, "hook_registry", mock_registry)
        monkeypatch.setattr(
            graph_mod,
            "check_permission",
            MagicMock(
                return_value=PermissionResult(behavior=PermissionBehavior.ALLOW, reason="ok"),
            ),
        )

        from dazi.graph import check_permissions

        ai_msg = AIMessage(
            content="",
            tool_calls=[{"id": "tc1", "name": "file_reader", "args": {"path": "/unsafe"}}],
        )
        state = {"messages": [ai_msg], "mode": "execute"}
        result = await check_permissions(state)
        assert result["allowed_tool_ids"] == ["tc1"]
        # Tool call args should be modified
        assert ai_msg.tool_calls[0]["args"] == {"path": "/safe/path"}

    @pytest.mark.asyncio
    async def test_mcp_tool_safety_derived(self, monkeypatch):
        import dazi.graph as graph_mod
        from dazi.permissions import PermissionBehavior, PermissionMode, PermissionResult

        monkeypatch.setattr(graph_mod, "_get_effective_rules", MagicMock(return_value=[]))

        # No meta for "custom_mcp_tool" — should fall through to MCP manager
        mock_mcp_tool_info = MagicMock()
        mock_mcp_tool_info.is_read_only = True
        mock_mcp = MagicMock()
        mock_mcp.get_tool = MagicMock(return_value=mock_mcp_tool_info)
        monkeypatch.setattr(graph_mod, "mcp_manager", mock_mcp)

        monkeypatch.setattr(
            graph_mod,
            "check_permission",
            MagicMock(
                return_value=PermissionResult(behavior=PermissionBehavior.ALLOW, reason="safe"),
            ),
        )
        monkeypatch.setattr(
            graph_mod,
            "hook_registry",
            MagicMock(
                fire=AsyncMock(
                    return_value=MagicMock(
                        should_block=False,
                        permission_override=None,
                        modified_input=None,
                    )
                )
            ),
        )

        from dazi.graph import check_permissions

        ai_msg = AIMessage(
            content="",
            tool_calls=[{"id": "tc1", "name": "custom_mcp_tool", "args": {}}],
        )
        state = {"messages": [ai_msg], "mode": "execute"}
        result = await check_permissions(state)
        assert result["allowed_tool_ids"] == ["tc1"]
        # check_permission should have been called with safety="safe"
        graph_mod.check_permission.assert_called_once()
        call_args = graph_mod.check_permission.call_args
        assert call_args[0][3] == PermissionMode.DEFAULT
        assert call_args[0][4] == "safe"

    @pytest.mark.asyncio
    async def test_mcp_tool_unknown_is_destructive(self, monkeypatch):
        import dazi.graph as graph_mod
        from dazi.permissions import PermissionBehavior, PermissionResult

        monkeypatch.setattr(graph_mod, "_get_effective_rules", MagicMock(return_value=[]))
        mock_mcp = MagicMock()
        mock_mcp.get_tool = MagicMock(return_value=None)
        monkeypatch.setattr(graph_mod, "mcp_manager", mock_mcp)

        monkeypatch.setattr(
            graph_mod,
            "check_permission",
            MagicMock(
                return_value=PermissionResult(behavior=PermissionBehavior.DENY, reason="unknown"),
            ),
        )
        monkeypatch.setattr(
            graph_mod,
            "hook_registry",
            MagicMock(
                fire=AsyncMock(
                    return_value=MagicMock(
                        should_block=False,
                        permission_override=None,
                        modified_input=None,
                    )
                )
            ),
        )

        from dazi.graph import check_permissions

        ai_msg = AIMessage(
            content="",
            tool_calls=[{"id": "tc1", "name": "unknown_tool", "args": {}}],
        )
        state = {"messages": [ai_msg], "mode": "execute"}
        result = await check_permissions(state)  # noqa: F841
        graph_mod.check_permission.assert_called_once()
        assert graph_mod.check_permission.call_args[0][4] == "destructive"

    @pytest.mark.asyncio
    async def test_ask_permission_allow(self, monkeypatch):
        import dazi.graph as graph_mod
        from dazi.permissions import PermissionBehavior, PermissionResult

        monkeypatch.setattr(graph_mod, "_get_effective_rules", MagicMock(return_value=[]))
        monkeypatch.setattr(
            graph_mod,
            "check_permission",
            MagicMock(
                return_value=PermissionResult(
                    behavior=PermissionBehavior.ASK, reason="review needed"
                ),
            ),
        )
        monkeypatch.setattr(
            graph_mod,
            "hook_registry",
            MagicMock(
                fire=AsyncMock(
                    return_value=MagicMock(
                        should_block=False,
                        permission_override=None,
                        modified_input=None,
                    )
                )
            ),
        )
        monkeypatch.setattr(
            graph_mod,
            "derive_permission_pattern",
            MagicMock(return_value="file_reader *"),
        )

        # Mock the interrupt() function to return a decision
        decisions = {"tc1": {"action": "allow"}}
        monkeypatch.setattr(graph_mod, "interrupt", MagicMock(return_value=decisions))

        from dazi.graph import check_permissions

        ai_msg = AIMessage(
            content="",
            tool_calls=[{"id": "tc1", "name": "file_reader", "args": {"path": "/tmp/x"}}],
        )
        state = {"messages": [ai_msg], "mode": "execute"}
        result = await check_permissions(state)
        assert result["allowed_tool_ids"] == ["tc1"]
        # Should have added an ALLOW rule to module-level permission_rules
        assert len(graph_mod.permission_rules) == 1
        assert graph_mod.permission_rules[0].behavior == PermissionBehavior.ALLOW

    @pytest.mark.asyncio
    async def test_ask_permission_deny(self, monkeypatch):
        import dazi.graph as graph_mod
        from dazi.permissions import PermissionBehavior, PermissionResult

        monkeypatch.setattr(graph_mod, "_get_effective_rules", MagicMock(return_value=[]))
        monkeypatch.setattr(
            graph_mod,
            "check_permission",
            MagicMock(
                return_value=PermissionResult(
                    behavior=PermissionBehavior.ASK, reason="review needed"
                ),
            ),
        )
        monkeypatch.setattr(
            graph_mod,
            "hook_registry",
            MagicMock(
                fire=AsyncMock(
                    return_value=MagicMock(
                        should_block=False,
                        permission_override=None,
                        modified_input=None,
                    )
                )
            ),
        )

        decisions = {"tc1": {"action": "deny"}}
        monkeypatch.setattr(graph_mod, "interrupt", MagicMock(return_value=decisions))

        from dazi.graph import check_permissions

        ai_msg = AIMessage(
            content="",
            tool_calls=[{"id": "tc1", "name": "shell_exec", "args": {"command": "rm -rf /"}}],
        )
        state = {"messages": [ai_msg], "mode": "execute"}
        result = await check_permissions(state)
        assert result["allowed_tool_ids"] == []
        assert len(result["messages"]) == 1
        assert "DENIED by user" in result["messages"][0].content

    @pytest.mark.asyncio
    async def test_ask_permission_skip(self, monkeypatch):
        import dazi.graph as graph_mod
        from dazi.permissions import PermissionBehavior, PermissionResult

        monkeypatch.setattr(graph_mod, "_get_effective_rules", MagicMock(return_value=[]))
        monkeypatch.setattr(
            graph_mod,
            "check_permission",
            MagicMock(
                return_value=PermissionResult(
                    behavior=PermissionBehavior.ASK, reason="review needed"
                ),
            ),
        )
        monkeypatch.setattr(
            graph_mod,
            "hook_registry",
            MagicMock(
                fire=AsyncMock(
                    return_value=MagicMock(
                        should_block=False,
                        permission_override=None,
                        modified_input=None,
                    )
                )
            ),
        )

        decisions = {"tc1": {"action": "skip", "message": "Not now"}}
        monkeypatch.setattr(graph_mod, "interrupt", MagicMock(return_value=decisions))

        from dazi.graph import check_permissions

        ai_msg = AIMessage(
            content="",
            tool_calls=[{"id": "tc1", "name": "shell_exec", "args": {"command": "ls"}}],
        )
        state = {"messages": [ai_msg], "mode": "execute"}
        result = await check_permissions(state)
        assert result["allowed_tool_ids"] == []
        assert "SKIPPED" in result["messages"][0].content

    @pytest.mark.asyncio
    async def test_ask_permission_legacy_string_format(self, monkeypatch):
        import dazi.graph as graph_mod
        from dazi.permissions import PermissionBehavior, PermissionResult

        monkeypatch.setattr(graph_mod, "_get_effective_rules", MagicMock(return_value=[]))
        monkeypatch.setattr(
            graph_mod,
            "check_permission",
            MagicMock(
                return_value=PermissionResult(behavior=PermissionBehavior.ASK, reason="review"),
            ),
        )
        monkeypatch.setattr(
            graph_mod,
            "hook_registry",
            MagicMock(
                fire=AsyncMock(
                    return_value=MagicMock(
                        should_block=False,
                        permission_override=None,
                        modified_input=None,
                    )
                )
            ),
        )
        monkeypatch.setattr(
            graph_mod,
            "derive_permission_pattern",
            MagicMock(return_value="*"),
        )

        # Legacy string format
        decisions = {"tc1": "allow"}
        monkeypatch.setattr(graph_mod, "interrupt", MagicMock(return_value=decisions))

        from dazi.graph import check_permissions

        ai_msg = AIMessage(
            content="",
            tool_calls=[{"id": "tc1", "name": "file_reader", "args": {}}],
        )
        state = {"messages": [ai_msg], "mode": "execute"}
        result = await check_permissions(state)
        assert result["allowed_tool_ids"] == ["tc1"]

    @pytest.mark.asyncio
    async def test_plan_mode_uses_plan_meta(self, monkeypatch):
        import dazi.graph as graph_mod
        from dazi.permissions import PermissionBehavior, PermissionMode, PermissionResult

        monkeypatch.setattr(graph_mod, "_get_effective_rules", MagicMock(return_value=[]))
        monkeypatch.setattr(
            graph_mod,
            "check_permission",
            MagicMock(
                return_value=PermissionResult(behavior=PermissionBehavior.ALLOW, reason="ok"),
            ),
        )
        monkeypatch.setattr(
            graph_mod,
            "hook_registry",
            MagicMock(
                fire=AsyncMock(
                    return_value=MagicMock(
                        should_block=False,
                        permission_override=None,
                        modified_input=None,
                    )
                )
            ),
        )

        from dazi.graph import check_permissions

        ai_msg = AIMessage(
            content="",
            tool_calls=[{"id": "tc1", "name": "file_reader", "args": {"path": "/tmp/x"}}],
        )
        state = {"messages": [ai_msg], "mode": "plan"}
        result = await check_permissions(state)  # noqa: F841
        graph_mod.check_permission.assert_called_once()
        call_args = graph_mod.check_permission.call_args
        assert call_args[0][3] == PermissionMode.PLAN


# ─────────────────────────────────────────────────────────
# execute_tools
# ─────────────────────────────────────────────────────────


class TestExecuteTools:
    @pytest.mark.asyncio
    async def test_no_tool_calls_returns_empty(self):
        from dazi.graph import execute_tools

        state = {"messages": [AIMessage(content="no tools")]}
        result = await execute_tools(state)
        assert result == {"messages": []}

    @pytest.mark.asyncio
    async def test_no_allowed_ids_returns_empty(self):
        from dazi.graph import execute_tools

        ai_msg = AIMessage(
            content="",
            tool_calls=[{"id": "tc1", "name": "file_reader", "args": {}}],
        )
        state = {"messages": [ai_msg], "allowed_tool_ids": [], "mode": "execute"}
        result = await execute_tools(state)
        assert result == {"messages": []}

    @pytest.mark.asyncio
    async def test_non_ai_message_returns_empty(self):
        from dazi.graph import execute_tools

        state = {"messages": [HumanMessage(content="hi")], "allowed_tool_ids": ["tc1"]}
        result = await execute_tools(state)
        assert result == {"messages": []}

    @pytest.mark.asyncio
    async def test_executes_allowed_tools(self, monkeypatch):
        import dazi.graph as graph_mod

        tool_msg = ToolMessage(content="file contents", tool_call_id="tc1")
        monkeypatch.setattr(
            graph_mod,
            "execute_tools_concurrent",
            AsyncMock(return_value=[tool_msg]),
        )
        monkeypatch.setattr(
            graph_mod,
            "hook_registry",
            MagicMock(fire=AsyncMock(return_value=MagicMock(modified_output=None))),
        )

        from dazi.graph import execute_tools

        ai_msg = AIMessage(
            content="",
            tool_calls=[
                {"id": "tc1", "name": "file_reader", "args": {"path": "/tmp/x"}},
                {"id": "tc2", "name": "file_reader", "args": {"path": "/tmp/y"}},
            ],
        )
        state = {"messages": [ai_msg], "allowed_tool_ids": ["tc1"], "mode": "execute"}
        result = await execute_tools(state)
        assert len(result["messages"]) == 1

    @pytest.mark.asyncio
    async def test_hook_modifies_tool_output(self, monkeypatch):
        import dazi.graph as graph_mod

        tool_msg = ToolMessage(content="original output", tool_call_id="tc1")
        monkeypatch.setattr(
            graph_mod,
            "execute_tools_concurrent",
            AsyncMock(return_value=[tool_msg]),
        )
        monkeypatch.setattr(
            graph_mod,
            "hook_registry",
            MagicMock(fire=AsyncMock(return_value=MagicMock(modified_output="modified output"))),
        )

        from dazi.graph import execute_tools

        ai_msg = AIMessage(
            content="",
            tool_calls=[{"id": "tc1", "name": "file_reader", "args": {}}],
        )
        state = {"messages": [ai_msg], "allowed_tool_ids": ["tc1"], "mode": "execute"}
        result = await execute_tools(state)
        assert result["messages"][0].content == "modified output"

    @pytest.mark.asyncio
    async def test_plan_mode_uses_plan_tools(self, monkeypatch):
        import dazi.graph as graph_mod

        tool_msg = ToolMessage(content="result", tool_call_id="tc1")
        mock_concurrent = AsyncMock(return_value=[tool_msg])
        monkeypatch.setattr(graph_mod, "execute_tools_concurrent", mock_concurrent)
        monkeypatch.setattr(
            graph_mod,
            "hook_registry",
            MagicMock(fire=AsyncMock(return_value=MagicMock(modified_output=None))),
        )

        from dazi.graph import execute_tools

        ai_msg = AIMessage(
            content="",
            tool_calls=[{"id": "tc1", "name": "file_reader", "args": {}}],
        )
        state = {"messages": [ai_msg], "allowed_tool_ids": ["tc1"], "mode": "plan"}
        await execute_tools(state)
        # Should use PLAN_TOOLS_FULL
        mock_concurrent.assert_called_once()
        call_args = mock_concurrent.call_args
        assert call_args[0][1] is graph_mod.PLAN_TOOLS_FULL


# ─────────────────────────────────────────────────────────
# SpinnerManager
# ─────────────────────────────────────────────────────────


class TestSpinnerManager:
    def test_start_stop(self, monkeypatch):
        import dazi.graph as graph_mod

        mock_live = MagicMock()
        monkeypatch.setattr(graph_mod, "Live", MagicMock(return_value=mock_live))
        monkeypatch.setattr(graph_mod, "Spinner", MagicMock(return_value=MagicMock()))

        from dazi.graph import SpinnerManager

        spinner = SpinnerManager()
        spinner.start()
        mock_live.start.assert_called_once()

        spinner.start()  # Double start should be no-op
        assert mock_live.start.call_count == 1

        spinner.stop()
        mock_live.stop.assert_called_once()

        spinner.stop()  # Double stop should be no-op
        assert mock_live.stop.call_count == 1

    def test_update_label(self, monkeypatch):
        import dazi.graph as graph_mod

        mock_live = MagicMock()
        monkeypatch.setattr(graph_mod, "Live", MagicMock(return_value=mock_live))
        mock_spinner = MagicMock()
        monkeypatch.setattr(graph_mod, "Spinner", MagicMock(return_value=mock_spinner))

        from dazi.graph import SpinnerManager

        spinner = SpinnerManager()
        spinner.start()
        spinner.update_label("New label")
        mock_live.update.assert_called_once_with(mock_spinner)


# ─────────────────────────────────────────────────────────
# _print_tool_call_compact
# ─────────────────────────────────────────────────────────


class TestPrintToolCallCompact:
    def test_short_args(self, monkeypatch):
        from dazi.graph import _print_tool_call_compact

        mock_console = MagicMock()
        monkeypatch.setattr("dazi.graph.console", mock_console)

        _print_tool_call_compact({"name": "file_reader", "args": {"path": "/tmp/x"}})
        mock_console.print.assert_called_once()
        assert "file_reader" in str(mock_console.print.call_args)

    def test_long_args_truncated(self, monkeypatch):
        from dazi.graph import _print_tool_call_compact

        mock_console = MagicMock()
        monkeypatch.setattr("dazi.graph.console", mock_console)

        long_args = {"data": "x" * 200}
        _print_tool_call_compact({"name": "tool", "args": long_args})
        output = str(mock_console.print.call_args)
        assert "..." in output


# ─────────────────────────────────────────────────────────
# _print_tool_result_compact
# ─────────────────────────────────────────────────────────


class TestPrintToolResultCompact:
    def test_normal_result(self, monkeypatch):
        from dazi.graph import _print_tool_result_compact

        mock_console = MagicMock()
        monkeypatch.setattr("dazi.graph.console", mock_console)

        _print_tool_result_compact("file contents here")
        mock_console.print.assert_called_once()

    def test_error_result(self, monkeypatch):
        from dazi.graph import _print_tool_result_compact

        mock_console = MagicMock()
        monkeypatch.setattr("dazi.graph.console", mock_console)

        _print_tool_result_compact("DENIED: not allowed", is_error=True)
        output = str(mock_console.print.call_args)
        assert "red" in output

    def test_long_result_truncated(self, monkeypatch):
        from dazi.graph import _print_tool_result_compact

        mock_console = MagicMock()
        monkeypatch.setattr("dazi.graph.console", mock_console)

        _print_tool_result_compact("x" * 200)
        output = str(mock_console.print.call_args)
        assert "..." in output


# ─────────────────────────────────────────────────────────
# _watch_esc
# ─────────────────────────────────────────────────────────


class TestWatchEsc:
    def test_non_esc_bytes_ignored(self, monkeypatch):
        """Test that non-ESC bytes are ignored and the function terminates cleanly."""
        import dazi.graph as graph_mod

        done_event = threading.Event()
        cancel_fn = MagicMock()

        # Simulate stdin returning a non-ESC byte then set done
        read_index = [0]

        def mock_read(fd, n):
            idx = read_index[0]
            read_index[0] += 1
            if idx == 0:
                return b"a"  # Non-ESC byte
            done_event.set()
            raise OSError("no data")

        monkeypatch.setattr("os.read", mock_read)

        # First select: data available (non-ESC)
        # Second select: set done_event
        select_results = [
            ([True], [], []),  # Data available
        ]
        select_idx = [0]

        def mock_select(r, w, x, timeout):
            idx = select_idx[0]
            if idx < len(select_results):
                result = select_results[idx]
                select_idx[0] += 1
                return result
            done_event.set()
            return ([], [], [])

        monkeypatch.setattr("select.select", mock_select)
        monkeypatch.setattr("termios.tcgetattr", MagicMock(return_value=[0] * 6))
        monkeypatch.setattr("termios.tcsetattr", MagicMock())
        monkeypatch.setattr("tty.setcbreak", MagicMock())

        # Mock sys.stdin to have fileno
        mock_stdin = MagicMock()
        mock_stdin.fileno.return_value = 0
        monkeypatch.setattr("sys.stdin", mock_stdin)

        loop = MagicMock()
        graph_mod._watch_esc(loop, cancel_fn, done_event)
        # Non-ESC byte should not trigger cancel
        cancel_fn.assert_not_called()

    def test_standalone_esc_cancels(self, monkeypatch):
        """Test that standalone ESC (not escape sequence) cancels the stream."""
        import dazi.graph as graph_mod

        done_event = threading.Event()
        cancel_fn = MagicMock()

        read_index = [0]

        def mock_read(fd, n):
            idx = read_index[0]
            read_index[0] += 1
            if idx == 0:
                return b"\x1b"  # ESC byte
            done_event.set()
            raise OSError("no data")

        monkeypatch.setattr("os.read", mock_read)

        # First select: data available (ESC)
        # Second select (ESC follow-up): no data = standalone ESC -> cancel
        select_results = [
            ([True], [], []),  # ESC byte available
            ([], [], []),  # No follow-up = standalone ESC
        ]
        select_idx = [0]

        def mock_select(r, w, x, timeout):
            idx = select_idx[0]
            if idx < len(select_results):
                result = select_results[idx]
                select_idx[0] += 1
                return result
            done_event.set()
            return ([], [], [])

        monkeypatch.setattr("select.select", mock_select)
        monkeypatch.setattr("termios.tcgetattr", MagicMock(return_value=[0] * 6))
        monkeypatch.setattr("termios.tcsetattr", MagicMock())
        monkeypatch.setattr("tty.setcbreak", MagicMock())

        mock_stdin = MagicMock()
        mock_stdin.fileno.return_value = 0
        monkeypatch.setattr("sys.stdin", mock_stdin)

        loop = MagicMock()
        graph_mod._watch_esc(loop, cancel_fn, done_event)
        loop.call_soon_threadsafe.assert_called_once_with(cancel_fn)

    def test_escape_sequence_consumed(self, monkeypatch):
        """Test that escape sequences (not standalone ESC) are consumed."""
        import dazi.graph as graph_mod

        done_event = threading.Event()
        cancel_fn = MagicMock()

        mock_stdin_data = [b"\x1b", b"[", b"A"]  # Up arrow sequence
        read_index = [0]

        def mock_read(fd, n):
            idx = read_index[0]
            if idx >= len(mock_stdin_data):
                done_event.set()
                raise OSError("no data")
            val = mock_stdin_data[idx]
            read_index[0] += 1
            return val

        monkeypatch.setattr("os.read", mock_read)

        # First select: data available (ESC)
        # Second select (ESC follow-up): data available = escape sequence
        # Third select (inner loop): no data
        # Fourth select (outer loop): set done
        select_results = [
            ([True], [], []),  # ESC byte available
            ([True], [], []),  # Follow-up byte available (escape sequence)
            ([], [], []),  # Inner loop: no more bytes
            ([], [], []),  # Outer loop: done_event set
        ]
        select_idx = [0]

        def mock_select(r, w, x, timeout):
            idx = select_idx[0]
            if idx < len(select_results):
                result = select_results[idx]
                select_idx[0] += 1
                return result
            done_event.set()
            return ([], [], [])

        monkeypatch.setattr("select.select", mock_select)
        monkeypatch.setattr("termios.tcgetattr", MagicMock(return_value=[0] * 6))
        monkeypatch.setattr("termios.tcsetattr", MagicMock())
        monkeypatch.setattr("tty.setcbreak", MagicMock())

        mock_stdin = MagicMock()
        mock_stdin.fileno.return_value = 0
        monkeypatch.setattr("sys.stdin", mock_stdin)

        loop = MagicMock()
        graph_mod._watch_esc(loop, cancel_fn, done_event)
        cancel_fn.assert_not_called()


# ─────────────────────────────────────────────────────────
# _stream_and_display
# ─────────────────────────────────────────────────────────


class TestStreamAndDisplay:
    @pytest.mark.asyncio
    async def test_normal_stream(self, monkeypatch):
        import dazi.graph as graph_mod

        monkeypatch.setattr(graph_mod, "_consume_stream", AsyncMock())
        monkeypatch.setattr("sys.stdin.isatty", MagicMock(return_value=False))

        mock_console = MagicMock()
        monkeypatch.setattr(graph_mod, "console", mock_console)

        from dazi.graph import _stream_and_display

        async def _fake_stream():
            yield

        spinner = MagicMock()
        await _stream_and_display(_fake_stream(), spinner=spinner)
        graph_mod._consume_stream.assert_called_once()

    @pytest.mark.asyncio
    async def test_cancel_raises_keyboard_interrupt(self, monkeypatch):
        import dazi.graph as graph_mod

        async def _cancel_consume(*args, **kwargs):
            import asyncio

            raise asyncio.CancelledError

        monkeypatch.setattr(graph_mod, "_consume_stream", _cancel_consume)
        monkeypatch.setattr("sys.stdin.isatty", MagicMock(return_value=False))

        mock_console = MagicMock()
        monkeypatch.setattr(graph_mod, "console", mock_console)

        from dazi.graph import _stream_and_display

        async def _fake_stream():
            yield

        spinner = MagicMock()
        with pytest.raises(KeyboardInterrupt):
            await _stream_and_display(_fake_stream(), spinner=spinner)

        mock_console.print.assert_called_once_with("\n[dim]Generation cancelled.[/dim]")


# ─────────────────────────────────────────────────────────
# display_background_notifications
# ─────────────────────────────────────────────────────────


class TestDisplayBackgroundNotifications:
    def test_completed_task(self, monkeypatch):
        import dazi.graph as graph_mod
        from dazi.background import BackgroundTask, BackgroundTaskStatus

        task = BackgroundTask(
            id="bash_abc123",
            command="echo hello",
            status=BackgroundTaskStatus.COMPLETED,
            exit_code=0,
            started_at=1000.0,
            completed_at=1001.5,
        )

        mock_bg = MagicMock()
        mock_bg.get_output_tail = MagicMock(return_value="hello\n")
        monkeypatch.setattr(graph_mod, "background_manager", mock_bg)

        mock_console = MagicMock()
        monkeypatch.setattr(graph_mod, "console", mock_console)

        from dazi.graph import display_background_notifications

        msgs = display_background_notifications([task])
        assert len(msgs) == 1
        assert isinstance(msgs[0], HumanMessage)
        assert "bash_abc123" in msgs[0].content
        mock_console.print.assert_called_once()

    def test_failed_task(self, monkeypatch):
        import dazi.graph as graph_mod
        from dazi.background import BackgroundTask, BackgroundTaskStatus

        task = BackgroundTask(
            id="bash_fail",
            command="exit 1",
            status=BackgroundTaskStatus.FAILED,
            exit_code=1,
            error="command failed",
            started_at=1000.0,
            completed_at=1000.5,
        )

        mock_bg = MagicMock()
        mock_bg.get_output_tail = MagicMock(return_value="")
        monkeypatch.setattr(graph_mod, "background_manager", mock_bg)

        mock_console = MagicMock()
        monkeypatch.setattr(graph_mod, "console", mock_console)

        from dazi.graph import display_background_notifications

        msgs = display_background_notifications([task])
        assert len(msgs) == 1
        assert "failed" in msgs[0].content.lower()

    def test_killed_task(self, monkeypatch):
        import dazi.graph as graph_mod
        from dazi.background import BackgroundTask, BackgroundTaskStatus

        task = BackgroundTask(
            id="bash_kill",
            command="sleep 100",
            status=BackgroundTaskStatus.KILLED,
        )

        mock_bg = MagicMock()
        mock_bg.get_output_tail = MagicMock(return_value="")
        monkeypatch.setattr(graph_mod, "background_manager", mock_bg)

        mock_console = MagicMock()
        monkeypatch.setattr(graph_mod, "console", mock_console)

        from dazi.graph import display_background_notifications

        msgs = display_background_notifications([task])
        assert len(msgs) == 1
        assert "killed" in msgs[0].content.lower()

    def test_multiple_tasks(self, monkeypatch):
        import dazi.graph as graph_mod
        from dazi.background import BackgroundTask, BackgroundTaskStatus

        task1 = BackgroundTask(
            id="bash_1",
            command="echo a",
            status=BackgroundTaskStatus.COMPLETED,
            exit_code=0,
        )
        task2 = BackgroundTask(
            id="bash_2",
            command="echo b",
            status=BackgroundTaskStatus.COMPLETED,
            exit_code=0,
        )

        mock_bg = MagicMock()
        mock_bg.get_output_tail = MagicMock(return_value="output\n")
        monkeypatch.setattr(graph_mod, "background_manager", mock_bg)

        mock_console = MagicMock()
        monkeypatch.setattr(graph_mod, "console", mock_console)

        from dazi.graph import display_background_notifications

        msgs = display_background_notifications([task1, task2])
        assert len(msgs) == 2

    def test_empty_list(self):
        from dazi.graph import display_background_notifications

        msgs = display_background_notifications([])
        assert msgs == []


# ─────────────────────────────────────────────────────────
# run_graph_turn
# ─────────────────────────────────────────────────────────


class TestRunGraphTurn:
    @pytest.mark.asyncio
    async def test_basic_turn_no_interrupts(self, monkeypatch):
        import dazi.graph as graph_mod

        # Mock the graph app
        mock_app = MagicMock()
        mock_final_values = {
            "messages": [AIMessage(content="Hello!")],
            "mode": "execute",
        }
        mock_state = MagicMock()
        mock_state.next = []
        mock_state.values = mock_final_values
        mock_app.astream_events = AsyncMock(return_value=_async_events())
        mock_app.get_state = MagicMock(return_value=mock_state)
        monkeypatch.setattr(graph_mod, "app", mock_app)

        # Mock background manager
        mock_bg = MagicMock()
        mock_bg.collect_completed = MagicMock(return_value=[])
        monkeypatch.setattr(graph_mod, "background_manager", mock_bg)

        # Mock _stream_and_display
        monkeypatch.setattr(graph_mod, "_stream_and_display", AsyncMock())

        from dazi.graph import run_graph_turn

        result = await run_graph_turn(
            messages=[HumanMessage(content="hi")],
            state={"mode": "execute", "messages": []},
            session=MagicMock(),
        )
        assert "messages" in result
        mock_app.astream_events.assert_called_once()

    @pytest.mark.asyncio
    async def test_turn_with_interrupt_and_resume(self, monkeypatch):
        from langgraph.errors import GraphInterrupt

        import dazi.graph as graph_mod

        # Mock the graph app
        mock_final_values = {
            "messages": [AIMessage(content="Done")],
            "mode": "execute",
        }
        mock_state_with_interrupt = MagicMock()
        mock_task = MagicMock()
        mock_interrupt_obj = MagicMock()
        mock_interrupt_obj.value = {
            "ask_tools": [
                {
                    "tool_call_id": "tc1",
                    "tool_name": "file_reader",
                    "tool_args": {},
                    "reason": "check",
                }
            ]
        }
        mock_task.interrupts = [mock_interrupt_obj]
        mock_state_with_interrupt.next = ["check_permissions"]
        mock_state_with_interrupt.tasks = [mock_task]

        mock_state_no_next = MagicMock()
        mock_state_no_next.next = []
        mock_state_no_next.values = mock_final_values
        mock_state_no_next.tasks = []

        call_count = [0]

        def mock_get_state(config):
            call_count[0] += 1
            if call_count[0] <= 2:
                return mock_state_with_interrupt
            return mock_state_no_next

        def mock_astream_events(*args, **kwargs):
            raise GraphInterrupt("interrupted")

        mock_app = MagicMock()
        mock_app.astream_events = mock_astream_events
        mock_app.get_state = MagicMock(side_effect=mock_get_state)
        monkeypatch.setattr(graph_mod, "app", mock_app)

        # Mock prompt_permission_decisions
        monkeypatch.setattr(
            graph_mod,
            "prompt_permission_decisions",
            AsyncMock(return_value={"tc1": {"action": "allow"}}),
        )

        # Mock background
        mock_bg = MagicMock()
        mock_bg.collect_completed = MagicMock(return_value=[])
        monkeypatch.setattr(graph_mod, "background_manager", mock_bg)

        monkeypatch.setattr(graph_mod, "_stream_and_display", AsyncMock())

        from dazi.graph import run_graph_turn

        result = await run_graph_turn(
            messages=[HumanMessage(content="hi")],
            state={"mode": "execute", "messages": []},
            session=MagicMock(),
        )
        assert "messages" in result

    @pytest.mark.asyncio
    async def test_turn_with_background_notifications(self, monkeypatch):
        import dazi.graph as graph_mod
        from dazi.background import BackgroundTask, BackgroundTaskStatus

        # Mock the graph app
        mock_app = MagicMock()
        mock_final_values = {
            "messages": [AIMessage(content="Done")],
            "mode": "execute",
        }
        mock_state = MagicMock()
        mock_state.next = []
        mock_state.values = mock_final_values
        mock_app.astream_events = AsyncMock(return_value=_async_events())
        mock_app.get_state = MagicMock(return_value=mock_state)
        monkeypatch.setattr(graph_mod, "app", mock_app)

        # Mock background manager with completed tasks
        task = BackgroundTask(
            id="bash_123",
            command="echo done",
            status=BackgroundTaskStatus.COMPLETED,
            exit_code=0,
            started_at=1000.0,
            completed_at=1001.0,
        )
        mock_bg = MagicMock()
        mock_bg.collect_completed = MagicMock(return_value=[task])
        mock_bg.get_output_tail = MagicMock(return_value="done\n")
        monkeypatch.setattr(graph_mod, "background_manager", mock_bg)

        mock_console = MagicMock()
        monkeypatch.setattr(graph_mod, "console", mock_console)

        monkeypatch.setattr(graph_mod, "_stream_and_display", AsyncMock())

        from dazi.graph import run_graph_turn

        repl_state = {"mode": "execute", "messages": []}
        result = await run_graph_turn(
            messages=[HumanMessage(content="hi")],
            state=repl_state,
            session=MagicMock(),
        )
        assert "messages" in result
        # REPL state should have been updated with notification messages
        assert len(repl_state["messages"]) > 0

    @pytest.mark.asyncio
    async def test_interrupt_no_ask_tools_breaks_loop(self, monkeypatch):
        from langgraph.errors import GraphInterrupt

        import dazi.graph as graph_mod

        mock_app = MagicMock()
        mock_final_values = {
            "messages": [AIMessage(content="Done")],
            "mode": "execute",
        }
        mock_state_with_other_interrupt = MagicMock()
        mock_task = MagicMock()
        mock_interrupt_obj = MagicMock()
        mock_interrupt_obj.value = {"other_key": "value"}
        mock_task.interrupts = [mock_interrupt_obj]
        mock_state_with_other_interrupt.next = ["some_node"]
        mock_state_with_other_interrupt.tasks = [mock_task]

        mock_state_no_next = MagicMock()
        mock_state_no_next.next = []
        mock_state_no_next.values = mock_final_values
        mock_state_no_next.tasks = []

        call_count = [0]

        def mock_get_state(config):
            call_count[0] += 1
            if call_count[0] == 1:
                return mock_state_with_other_interrupt
            return mock_state_no_next

        mock_app.astream_events = AsyncMock(
            side_effect=[GraphInterrupt("interrupted"), _async_events()],
        )
        mock_app.get_state = MagicMock(side_effect=mock_get_state)
        monkeypatch.setattr(graph_mod, "app", mock_app)

        mock_bg = MagicMock()
        mock_bg.collect_completed = MagicMock(return_value=[])
        monkeypatch.setattr(graph_mod, "background_manager", mock_bg)

        monkeypatch.setattr(graph_mod, "_stream_and_display", AsyncMock())

        from dazi.graph import run_graph_turn

        result = await run_graph_turn(
            messages=[HumanMessage(content="hi")],
            state={"mode": "execute", "messages": []},
            session=MagicMock(),
        )
        assert "messages" in result
