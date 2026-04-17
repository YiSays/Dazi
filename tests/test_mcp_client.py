"""Tests for dazi/mcp_client.py."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from dazi.mcp_client import (
    MAX_MCP_DESCRIPTION_LENGTH,
    MCPConnectionError,
    MCPManager,
    MCPServerConfig,
    MCPServerConnection,
    MCPServerStatus,
    MCPServerTool,
    MCPToolError,
    _build_mcp_tool_name,
    _clear_task_cancellation,
    _convert_schema_to_pydantic,
    _force_cleanup_stale_scopes,
    _map_json_type,
    _normalize_name,
    _parse_mcp_tool_name,
    list_mcp_resources_func,
    list_mcp_servers_func,
)
from tests.helpers.mock_singletons import patch_singletons


# Apply singleton patches before importing modules that use singletons
@pytest.fixture(autouse=True)
def _patch(monkeypatch, tmp_path: Path):
    patch_singletons(monkeypatch, tmp_path)


# ─────────────────────────────────────────────────────────
# Name normalization
# ─────────────────────────────────────────────────────────


class TestNormalizeName:
    def test_lowercases(self):
        assert _normalize_name("FileSystem") == "filesystem"

    def test_replaces_special_chars(self):
        assert _normalize_name("my-server.v2") == "my-server_v2"
        assert _normalize_name("hello world") == "hello_world"

    def test_strips_non_alphanumeric_except_underscore_dash(self):
        assert _normalize_name("a!@#b") == "a___b"


class TestBuildMcpToolName:
    def test_builds_qualified_name(self):
        result = _build_mcp_tool_name("filesystem", "read_file")
        assert result == "mcp__filesystem__read_file"

    def test_normalizes_names(self):
        result = _build_mcp_tool_name("My Server", "Some Tool")
        assert result == "mcp__my_server__some_tool"


class TestParseMcpToolName:
    def test_parses_qualified_name(self):
        server, tool = _parse_mcp_tool_name("mcp__filesystem__read_file")
        assert server == "filesystem"
        assert tool == "read_file"

    def test_raises_for_non_mcp_name(self):
        with pytest.raises(ValueError, match="Not an MCP tool name"):
            _parse_mcp_tool_name("regular_tool")

    def test_raises_for_invalid_format(self):
        with pytest.raises(ValueError, match="Invalid MCP tool name format"):
            _parse_mcp_tool_name("mcp__noserverpart")


# ─────────────────────────────────────────────────────────
# Exceptions
# ─────────────────────────────────────────────────────────


class TestMCPToolError:
    def test_message_format(self):
        err = MCPToolError("myserver", "mytool", "something went wrong")
        assert err.server_name == "myserver"
        assert err.tool_name == "mytool"
        assert err.message == "something went wrong"
        assert "myserver/mytool" in str(err)
        assert "something went wrong" in str(err)


class TestMCPConnectionError:
    def test_message_format(self):
        err = MCPConnectionError("myserver", "connection refused")
        assert err.server_name == "myserver"
        assert err.message == "connection refused"
        assert "myserver" in str(err)
        assert "connection refused" in str(err)


# ─────────────────────────────────────────────────────────
# MCPServerStatus
# ─────────────────────────────────────────────────────────


class TestMCPServerStatus:
    def test_values(self):
        assert MCPServerStatus.DISCONNECTED == "disconnected"
        assert MCPServerStatus.CONNECTING == "connecting"
        assert MCPServerStatus.CONNECTED == "connected"
        assert MCPServerStatus.ERROR == "error"


# ─────────────────────────────────────────────────────────
# MCPServerConfig
# ─────────────────────────────────────────────────────────


class TestMCPServerConfig:
    def test_from_dict(self):
        data = {
            "command": "npx",
            "args": ["-y", "@mcp/server"],
            "env": {"KEY": "value"},
            "description": "Test server",
        }
        config = MCPServerConfig.from_dict("test", data)
        assert config.name == "test"
        assert config.command == "npx"
        assert config.args == ["-y", "@mcp/server"]
        assert config.env == {"KEY": "value"}
        assert config.description == "Test server"

    def test_from_dict_minimal(self):
        data = {"command": "node"}
        config = MCPServerConfig.from_dict("minimal", data)
        assert config.name == "minimal"
        assert config.command == "node"
        assert config.args == []
        assert config.env is None
        assert config.description == ""

    def test_to_dict_full(self):
        config = MCPServerConfig(
            name="test",
            command="npx",
            args=["-y", "@mcp/server"],
            env={"KEY": "value"},
            description="My server",
        )
        d = config.to_dict()
        assert d["command"] == "npx"
        assert d["args"] == ["-y", "@mcp/server"]
        assert d["env"] == {"KEY": "value"}
        assert d["description"] == "My server"

    def test_to_dict_omits_empty_fields(self):
        config = MCPServerConfig(name="test", command="node")
        d = config.to_dict()
        assert "args" not in d
        assert "env" not in d
        assert "description" not in d


# ─────────────────────────────────────────────────────────
# MCPServerConnection (dataclass)
# ─────────────────────────────────────────────────────────


class TestMCPServerConnection:
    def test_defaults(self):
        config = MCPServerConfig(name="test", command="node")
        conn = MCPServerConnection(config=config)
        assert conn.config == config
        assert conn.status == MCPServerStatus.DISCONNECTED
        assert conn.tools == []
        assert conn.resources == []
        assert conn.error is None
        assert conn._session is None
        assert conn._stdio_cm is None
        assert conn._cm_stack is None


# ─────────────────────────────────────────────────────────
# Schema conversion
# ─────────────────────────────────────────────────────────


class TestMapJsonType:
    def test_string(self):
        assert _map_json_type({"type": "string"}) is str

    def test_integer(self):
        assert _map_json_type({"type": "integer"}) is int

    def test_number(self):
        assert _map_json_type({"type": "number"}) is float

    def test_boolean(self):
        assert _map_json_type({"type": "boolean"}) is bool

    def test_array(self):
        assert _map_json_type({"type": "array"}) is list

    def test_object(self):
        assert _map_json_type({"type": "object"}) is dict

    def test_fallback_to_string(self):
        assert _map_json_type({"type": "unknown_type"}) is str

    def test_default_is_string(self):
        assert _map_json_type({}) is str


class TestConvertSchemaToPydantic:
    def test_required_fields(self):
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
            "required": ["name", "age"],
        }
        model = _convert_schema_to_pydantic("test_tool", schema)
        assert model.__name__ == "MCPInput_test_tool"
        assert hasattr(model.model_fields["name"], "default")
        assert hasattr(model.model_fields["age"], "default")

    def test_optional_fields(self):
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
        }
        model = _convert_schema_to_pydantic("test_tool", schema)
        # Optional field should have default=None
        assert model.model_fields["name"].default is None

    def test_empty_schema(self):
        schema = {}
        model = _convert_schema_to_pydantic("test_tool", schema)
        # Should have input_data with dict default
        assert "input_data" in model.model_fields

    def test_no_properties(self):
        schema = {"type": "object"}
        model = _convert_schema_to_pydantic("test_tool", schema)
        assert "input_data" in model.model_fields

    def test_mixed_types(self):
        schema = {
            "properties": {
                "flag": {"type": "boolean"},
                "count": {"type": "number"},
                "items": {"type": "array"},
                "meta": {"type": "object"},
            },
            "required": ["flag"],
        }
        model = _convert_schema_to_pydantic("test_tool", schema)
        assert "flag" in model.model_fields
        assert "count" in model.model_fields
        assert "items" in model.model_fields
        assert "meta" in model.model_fields


# ─────────────────────────────────────────────────────────
# MCPManager initial state
# ─────────────────────────────────────────────────────────


class TestMCPManagerInitialState:
    def test_no_servers(self):
        mgr = MCPManager()
        assert mgr.list_servers() == []

    def test_no_tools(self):
        mgr = MCPManager()
        assert mgr.get_tools() == []

    def test_get_tool_returns_none(self):
        mgr = MCPManager()
        assert mgr.get_tool("mcp__test__tool") is None

    def test_get_server_returns_none(self):
        mgr = MCPManager()
        assert mgr.get_server("nonexistent") is None


# ─────────────────────────────────────────────────────────
# MCPManager server management
# ─────────────────────────────────────────────────────────


class TestMCPManagerServerManagement:
    def test_add_server(self):
        mgr = MCPManager()
        config = MCPServerConfig(name="test", command="node")
        mgr.add_server(config)
        assert mgr.get_server("test") is not None
        assert mgr.get_server("test").config == config

    def test_remove_server(self):
        mgr = MCPManager()
        config = MCPServerConfig(name="test", command="node")
        mgr.add_server(config)
        mgr.remove_server("test")
        assert mgr.get_server("test") is None

    def test_remove_nonexistent_server(self):
        mgr = MCPManager()
        # Should not raise
        mgr.remove_server("nonexistent")

    def test_remove_server_clears_tool_map(self):
        mgr = MCPManager()
        config = MCPServerConfig(name="test", command="node")
        mgr.add_server(config)
        # Manually add a tool to the tool_map for this server
        tool = MCPServerTool(
            server_name="test",
            name="my_tool",
            qualified_name="mcp__test__my_tool",
            description="desc",
            input_schema={},
        )
        mgr._tool_map["mcp__test__my_tool"] = tool
        mgr._servers["test"].tools = [tool]

        mgr.remove_server("test")
        assert mgr.get_tool("mcp__test__my_tool") is None

    def test_reset(self):
        mgr = MCPManager()
        config = MCPServerConfig(name="test", command="node")
        mgr.add_server(config)
        mgr.reset()
        assert mgr.list_servers() == []
        assert mgr.get_tools() == []

    def test_list_servers(self):
        mgr = MCPManager()
        config1 = MCPServerConfig(name="server1", command="node")
        config2 = MCPServerConfig(name="server2", command="npx", args=["-y", "pkg"])
        mgr.add_server(config1)
        mgr.add_server(config2)

        servers = mgr.list_servers()
        assert len(servers) == 2
        names = {s["name"] for s in servers}
        assert names == {"server1", "server2"}

    def test_list_servers_with_error(self):
        mgr = MCPManager()
        config = MCPServerConfig(name="err", command="node")
        mgr.add_server(config)
        mgr._servers["err"].error = "Connection failed"

        servers = mgr.list_servers()
        assert servers[0]["error"] == "Connection failed"


# ─────────────────────────────────────────────────────────
# MCPManager connection lifecycle
# ─────────────────────────────────────────────────────────


class TestMCPManagerConnect:
    @pytest.mark.asyncio
    async def test_connect_server_not_found(self):
        mgr = MCPManager()
        result = await mgr.connect_server("nonexistent")
        assert result is False

    @pytest.mark.asyncio
    async def test_connect_all(self):
        mgr = MCPManager()
        config = MCPServerConfig(name="test", command="node")
        mgr.add_server(config)

        with patch.object(mgr, "connect_server", new_callable=AsyncMock, return_value=True):
            results = await mgr.connect_all()
        assert results == {"test": True}

    @pytest.mark.asyncio
    async def test_connect_all_mixed_results(self):
        mgr = MCPManager()
        mgr.add_server(MCPServerConfig(name="good", command="node"))
        mgr.add_server(MCPServerConfig(name="bad", command="node"))

        async def _connect(name):
            return name == "good"

        with patch.object(mgr, "connect_server", new_callable=AsyncMock, side_effect=_connect):
            results = await mgr.connect_all()
        assert results == {"good": True, "bad": False}

    @pytest.mark.asyncio
    async def test_connect_server_success(self):
        mgr = MCPManager()
        config = MCPServerConfig(name="test", command="node", args=["-y", "pkg"])
        mgr.add_server(config)

        # Mock stdio_client and ClientSession
        mock_stdio_cm = AsyncMock()
        mock_read = MagicMock()
        mock_write = MagicMock()
        mock_stdio_cm.__aenter__ = AsyncMock(return_value=(mock_read, mock_write))
        mock_stdio_cm.__aexit__ = AsyncMock(return_value=False)

        mock_session_cm = AsyncMock()
        mock_session = AsyncMock()
        mock_session_cm.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session_cm.__aexit__ = AsyncMock(return_value=False)

        mock_list_tools = MagicMock()
        mock_list_tools.tools = []
        mock_session.list_tools = AsyncMock(return_value=mock_list_tools)
        mock_session.initialize = AsyncMock()
        mock_session.list_resources = AsyncMock(return_value=MagicMock(resources=None))

        with (
            patch("mcp.client.stdio.stdio_client", return_value=mock_stdio_cm),
            patch("mcp.ClientSession", return_value=mock_session_cm),
        ):
            result = await mgr.connect_server("test")

        assert result is True
        assert mgr._servers["test"].status == MCPServerStatus.CONNECTED

    @pytest.mark.asyncio
    async def test_connect_server_failure(self):
        mgr = MCPManager()
        config = MCPServerConfig(name="test", command="node")
        mgr.add_server(config)

        with (
            patch(
                "mcp.client.stdio.stdio_client",
                side_effect=Exception("spawn failed"),
            ),
            patch("dazi.mcp_client._force_cleanup_stale_scopes"),
        ):
            result = await mgr.connect_server("test")

        assert result is False
        assert mgr._servers["test"].status == MCPServerStatus.ERROR
        assert "spawn failed" in mgr._servers["test"].error


class TestMCPManagerDisconnect:
    @pytest.mark.asyncio
    async def test_disconnect_nonexistent(self):
        mgr = MCPManager()
        # Should not raise
        await mgr.disconnect_server("nonexistent")

    @pytest.mark.asyncio
    async def test_disconnect_already_disconnected(self):
        mgr = MCPManager()
        config = MCPServerConfig(name="test", command="node")
        mgr.add_server(config)
        # Default status is DISCONNECTED
        await mgr.disconnect_server("test")
        assert mgr._servers["test"].status == MCPServerStatus.DISCONNECTED

    @pytest.mark.asyncio
    async def test_disconnect_connected(self):
        mgr = MCPManager()
        config = MCPServerConfig(name="test", command="node")
        mgr.add_server(config)
        conn = mgr._servers["test"]
        conn.status = MCPServerStatus.CONNECTED

        # Manually add tools and resources
        tool = MCPServerTool(
            server_name="test",
            name="my_tool",
            qualified_name="mcp__test__my_tool",
            description="desc",
            input_schema={},
        )
        conn.tools = [tool]
        conn.resources = []
        mgr._tool_map["mcp__test__my_tool"] = tool

        mock_session_cm = AsyncMock()
        mock_session_cm.__aexit__ = AsyncMock(return_value=False)
        conn._cm_stack = mock_session_cm

        mock_stdio_cm = AsyncMock()
        mock_stdio_cm.__aexit__ = AsyncMock(return_value=False)
        conn._stdio_cm = mock_stdio_cm

        with patch("dazi.mcp_client._force_cleanup_stale_scopes"):
            await mgr.disconnect_server("test")

        assert conn.status == MCPServerStatus.DISCONNECTED
        assert conn.tools == []
        assert conn.resources == []
        assert mgr.get_tool("mcp__test__my_tool") is None

    @pytest.mark.asyncio
    async def test_disconnect_all(self):
        mgr = MCPManager()
        mgr.add_server(MCPServerConfig(name="s1", command="node"))
        mgr.add_server(MCPServerConfig(name="s2", command="node"))
        mgr._servers["s1"].status = MCPServerStatus.CONNECTED
        mgr._servers["s2"].status = MCPServerStatus.CONNECTED

        with patch("dazi.mcp_client._force_cleanup_stale_scopes"):
            await mgr.disconnect_all()

        assert mgr._servers["s1"].status == MCPServerStatus.DISCONNECTED
        assert mgr._servers["s2"].status == MCPServerStatus.DISCONNECTED


class TestCleanupConnection:
    @pytest.mark.asyncio
    async def test_cleanup_both_none(self):
        mgr = MCPManager()
        conn = MCPServerConnection(config=MagicMock())
        with patch("dazi.mcp_client._force_cleanup_stale_scopes"):
            await mgr._cleanup_connection(conn)
        assert conn._session is None
        assert conn._cm_stack is None
        assert conn._stdio_cm is None

    @pytest.mark.asyncio
    async def test_cleanup_cm_stack_raises(self):
        mgr = MCPManager()
        conn = MCPServerConnection(config=MagicMock())
        mock_cm_stack = AsyncMock()
        mock_cm_stack.__aexit__ = AsyncMock(side_effect=Exception("session error"))
        conn._cm_stack = mock_cm_stack

        with patch("dazi.mcp_client._force_cleanup_stale_scopes"):
            await mgr._cleanup_connection(conn)

        assert conn._session is None
        assert conn._cm_stack is None

    @pytest.mark.asyncio
    async def test_cleanup_stdio_raises(self):
        mgr = MCPManager()
        conn = MCPServerConnection(config=MagicMock())
        mock_stdio = AsyncMock()
        mock_stdio.__aexit__ = AsyncMock(side_effect=Exception("stdio error"))
        conn._stdio_cm = mock_stdio

        with patch("dazi.mcp_client._force_cleanup_stale_scopes"):
            await mgr._cleanup_connection(conn)

        assert conn._stdio_cm is None

    @pytest.mark.asyncio
    async def test_cleanup_both_raise(self):
        mgr = MCPManager()
        conn = MCPServerConnection(config=MagicMock())
        mock_cm_stack = AsyncMock()
        mock_cm_stack.__aexit__ = AsyncMock(side_effect=Exception("err1"))
        conn._cm_stack = mock_cm_stack

        mock_stdio = AsyncMock()
        mock_stdio.__aexit__ = AsyncMock(side_effect=Exception("err2"))
        conn._stdio_cm = mock_stdio

        with patch("dazi.mcp_client._force_cleanup_stale_scopes"):
            await mgr._cleanup_connection(conn)

        assert conn._session is None
        assert conn._cm_stack is None
        assert conn._stdio_cm is None

    @pytest.mark.asyncio
    async def test_cleanup_success(self):
        mgr = MCPManager()
        conn = MCPServerConnection(config=MagicMock())
        mock_cm_stack = AsyncMock()
        mock_cm_stack.__aexit__ = AsyncMock(return_value=False)
        conn._cm_stack = mock_cm_stack

        mock_stdio = AsyncMock()
        mock_stdio.__aexit__ = AsyncMock(return_value=False)
        conn._stdio_cm = mock_stdio

        with patch("dazi.mcp_client._force_cleanup_stale_scopes"):
            await mgr._cleanup_connection(conn)

        mock_cm_stack.__aexit__.assert_awaited_once()
        mock_stdio.__aexit__.assert_awaited_once()
        assert conn._session is None
        assert conn._cm_stack is None
        assert conn._stdio_cm is None


# ─────────────────────────────────────────────────────────
# Tool discovery
# ─────────────────────────────────────────────────────────


class TestDiscoverTools:
    @pytest.mark.asyncio
    async def test_no_session(self):
        mgr = MCPManager()
        conn = MCPServerConnection(config=MagicMock())
        # _session is None — should return early
        await mgr._discover_tools("test", conn)
        assert conn.tools == []

    @pytest.mark.asyncio
    async def test_discovers_tools(self):
        mgr = MCPManager()
        conn = MCPServerConnection(config=MagicMock())
        mock_session = AsyncMock()

        # Create mock MCP tool objects
        mock_tool1 = MagicMock()
        mock_tool1.name = "read_file"
        mock_tool1.description = "Read a file"
        mock_tool1.annotations = None
        mock_tool1.inputSchema = {"type": "object", "properties": {"path": {"type": "string"}}}

        mock_tool2 = MagicMock()
        mock_tool2.name = "write_file"
        mock_tool2.description = "Write a file"
        mock_tool2.annotations = MagicMock(readOnlyHint=False)
        mock_tool2.inputSchema = {}

        mock_result = MagicMock()
        mock_result.tools = [mock_tool1, mock_tool2]
        mock_session.list_tools = AsyncMock(return_value=mock_result)

        conn._session = mock_session
        await mgr._discover_tools("myserver", conn)

        assert len(conn.tools) == 2
        # Check qualified names
        assert conn.tools[0].qualified_name == "mcp__myserver__read_file"
        assert conn.tools[1].qualified_name == "mcp__myserver__write_file"
        # Check read-only defaults
        assert conn.tools[0].is_read_only is True  # annotations is None -> default True
        assert conn.tools[1].is_read_only is False  # annotations.readOnlyHint = False
        # Check tool_map
        assert mgr.get_tool("mcp__myserver__read_file") is not None

    @pytest.mark.asyncio
    async def test_discover_tools_long_description(self):
        mgr = MCPManager()
        conn = MCPServerConnection(config=MagicMock())
        mock_session = AsyncMock()

        mock_tool = MagicMock()
        mock_tool.name = "tool"
        mock_tool.description = "x" * (MAX_MCP_DESCRIPTION_LENGTH + 100)
        mock_tool.annotations = None
        mock_tool.inputSchema = {}

        mock_result = MagicMock()
        mock_result.tools = [mock_tool]
        mock_session.list_tools = AsyncMock(return_value=mock_result)
        conn._session = mock_session

        await mgr._discover_tools("server", conn)
        assert len(conn.tools[0].description) == MAX_MCP_DESCRIPTION_LENGTH

    @pytest.mark.asyncio
    async def test_discover_tools_input_schema_model_dump(self):
        mgr = MCPManager()
        conn = MCPServerConnection(config=MagicMock())
        mock_session = AsyncMock()

        # inputSchema has model_dump (pydantic model)
        mock_schema = MagicMock()
        mock_schema.model_dump.return_value = {"type": "object", "properties": {}}

        mock_tool = MagicMock()
        mock_tool.name = "tool"
        mock_tool.description = "desc"
        mock_tool.annotations = None
        mock_tool.inputSchema = mock_schema

        mock_result = MagicMock()
        mock_result.tools = [mock_tool]
        mock_session.list_tools = AsyncMock(return_value=mock_result)
        conn._session = mock_session

        await mgr._discover_tools("server", conn)
        assert conn.tools[0].input_schema == {"type": "object", "properties": {}}

    @pytest.mark.asyncio
    async def test_discover_tools_input_schema_dict(self):
        mgr = MCPManager()
        conn = MCPServerConnection(config=MagicMock())
        mock_session = AsyncMock()

        # inputSchema is a regular dict
        mock_tool = MagicMock()
        mock_tool.name = "tool"
        mock_tool.description = "desc"
        mock_tool.annotations = None
        mock_tool.inputSchema = {"type": "object", "properties": {"x": {"type": "string"}}}

        mock_result = MagicMock()
        mock_result.tools = [mock_tool]
        mock_session.list_tools = AsyncMock(return_value=mock_result)
        conn._session = mock_session

        await mgr._discover_tools("server", conn)
        assert conn.tools[0].input_schema == {
            "type": "object",
            "properties": {"x": {"type": "string"}},
        }

    @pytest.mark.asyncio
    async def test_discover_tools_input_schema_none(self):
        mgr = MCPManager()
        conn = MCPServerConnection(config=MagicMock())
        mock_session = AsyncMock()

        mock_tool = MagicMock()
        mock_tool.name = "tool"
        mock_tool.description = "desc"
        mock_tool.annotations = None
        mock_tool.inputSchema = None

        mock_result = MagicMock()
        mock_result.tools = [mock_tool]
        mock_session.list_tools = AsyncMock(return_value=mock_result)
        conn._session = mock_session

        await mgr._discover_tools("server", conn)
        assert conn.tools[0].input_schema == {}

    @pytest.mark.asyncio
    async def test_discover_tools_exception(self):
        mgr = MCPManager()
        conn = MCPServerConnection(config=MagicMock())
        mock_session = AsyncMock()
        mock_session.list_tools = AsyncMock(side_effect=Exception("list failed"))
        conn._session = mock_session

        # Should not raise, just log
        await mgr._discover_tools("server", conn)
        assert conn.tools == []


class TestDiscoverResources:
    @pytest.mark.asyncio
    async def test_no_session(self):
        mgr = MCPManager()
        conn = MCPServerConnection(config=MagicMock())
        await mgr._discover_resources("test", conn)
        assert conn.resources == []

    @pytest.mark.asyncio
    async def test_discovers_resources(self):
        mgr = MCPManager()
        conn = MCPServerConnection(config=MagicMock())
        mock_session = AsyncMock()

        mock_resource = MagicMock()
        mock_resource.uri = "file:///tmp/test"
        mock_resource.name = "test.txt"
        mock_resource.mimeType = "text/plain"
        mock_resource.description = "A test file"

        mock_result = MagicMock()
        mock_result.resources = [mock_resource]
        mock_session.list_resources = AsyncMock(return_value=mock_result)
        conn._session = mock_session

        await mgr._discover_resources("server", conn)
        assert len(conn.resources) == 1
        assert conn.resources[0].uri == "file:///tmp/test"
        assert conn.resources[0].name == "test.txt"

    @pytest.mark.asyncio
    async def test_discovers_resources_empty_list(self):
        mgr = MCPManager()
        conn = MCPServerConnection(config=MagicMock())
        mock_session = AsyncMock()

        mock_result = MagicMock()
        mock_result.resources = []
        mock_session.list_resources = AsyncMock(return_value=mock_result)
        conn._session = mock_session

        await mgr._discover_resources("server", conn)
        assert conn.resources == []

    @pytest.mark.asyncio
    async def test_discovers_resources_none_list(self):
        mgr = MCPManager()
        conn = MCPServerConnection(config=MagicMock())
        mock_session = AsyncMock()

        mock_result = MagicMock()
        mock_result.resources = None
        mock_session.list_resources = AsyncMock(return_value=mock_result)
        conn._session = mock_session

        await mgr._discover_resources("server", conn)
        assert conn.resources == []

    @pytest.mark.asyncio
    async def test_discovers_resources_optional_attrs(self):
        mgr = MCPManager()
        conn = MCPServerConnection(config=MagicMock())
        mock_session = AsyncMock()

        mock_resource = MagicMock()
        mock_resource.uri = "file:///tmp/test"
        mock_resource.name = None
        mock_resource.mimeType = None
        mock_resource.description = None

        mock_result = MagicMock()
        mock_result.resources = [mock_resource]
        mock_session.list_resources = AsyncMock(return_value=mock_result)
        conn._session = mock_session

        await mgr._discover_resources("server", conn)
        assert conn.resources[0].name == ""
        assert conn.resources[0].mime_type == ""
        assert conn.resources[0].description == ""

    @pytest.mark.asyncio
    async def test_discover_resources_exception(self):
        mgr = MCPManager()
        conn = MCPServerConnection(config=MagicMock())
        mock_session = AsyncMock()
        mock_session.list_resources = AsyncMock(side_effect=Exception("list failed"))
        conn._session = mock_session

        # Should not raise
        await mgr._discover_resources("server", conn)
        assert conn.resources == []


# ─────────────────────────────────────────────────────────
# Tool execution
# ─────────────────────────────────────────────────────────


class TestCallTool:
    @pytest.mark.asyncio
    async def test_server_not_registered(self):
        mgr = MCPManager()
        with pytest.raises(MCPConnectionError, match="not registered"):
            await mgr.call_tool("mcp__missing__tool", {})

    @pytest.mark.asyncio
    async def test_server_not_connected(self):
        mgr = MCPManager()
        config = MCPServerConfig(name="test", command="node")
        mgr.add_server(config)
        # Default status is DISCONNECTED
        with pytest.raises(MCPConnectionError, match="not connected"):
            await mgr.call_tool("mcp__test__tool", {})

    @pytest.mark.asyncio
    async def test_no_session(self):
        mgr = MCPManager()
        config = MCPServerConfig(name="test", command="node")
        mgr.add_server(config)
        conn = mgr._servers["test"]
        conn.status = MCPServerStatus.CONNECTED
        conn._session = None

        with pytest.raises(MCPConnectionError, match="no active session"):
            await mgr.call_tool("mcp__test__tool", {})

    @pytest.mark.asyncio
    async def test_call_success(self):
        mgr = MCPManager()
        config = MCPServerConfig(name="test", command="node")
        mgr.add_server(config)
        conn = mgr._servers["test"]
        conn.status = MCPServerStatus.CONNECTED

        mock_session = AsyncMock()
        mock_content = MagicMock()
        mock_content.text = "result text"
        mock_call_result = MagicMock()
        mock_call_result.isError = False
        mock_call_result.content = [mock_content]
        mock_session.call_tool = AsyncMock(return_value=mock_call_result)
        conn._session = mock_session

        result = await mgr.call_tool("mcp__test__my_tool", {"key": "value"})
        assert result == "result text"
        mock_session.call_tool.assert_awaited_once_with("my_tool", arguments={"key": "value"})

    @pytest.mark.asyncio
    async def test_call_error_response(self):
        mgr = MCPManager()
        config = MCPServerConfig(name="test", command="node")
        mgr.add_server(config)
        conn = mgr._servers["test"]
        conn.status = MCPServerStatus.CONNECTED

        mock_session = AsyncMock()
        mock_content = MagicMock()
        mock_content.text = "error message"
        mock_call_result = MagicMock()
        mock_call_result.isError = True
        mock_call_result.content = [mock_content]
        mock_session.call_tool = AsyncMock(return_value=mock_call_result)
        conn._session = mock_session

        with pytest.raises(MCPToolError, match="error message"):
            await mgr.call_tool("mcp__test__my_tool", {})

    @pytest.mark.asyncio
    async def test_call_exception(self):
        mgr = MCPManager()
        config = MCPServerConfig(name="test", command="node")
        mgr.add_server(config)
        conn = mgr._servers["test"]
        conn.status = MCPServerStatus.CONNECTED

        mock_session = AsyncMock()
        mock_session.call_tool = AsyncMock(side_effect=RuntimeError("boom"))
        conn._session = mock_session

        with pytest.raises(MCPToolError, match="boom"):
            await mgr.call_tool("mcp__test__my_tool", {})

    @pytest.mark.asyncio
    async def test_server_in_error_state(self):
        mgr = MCPManager()
        config = MCPServerConfig(name="test", command="node")
        mgr.add_server(config)
        conn = mgr._servers["test"]
        conn.status = MCPServerStatus.ERROR

        with pytest.raises(MCPConnectionError, match="not connected"):
            await mgr.call_tool("mcp__test__tool", {})


# ─────────────────────────────────────────────────────────
# _extract_text_from_content
# ─────────────────────────────────────────────────────────


class TestExtractTextFromContent:
    def test_text_content(self):
        mgr = MCPManager()
        item = MagicMock()
        item.text = "hello world"
        result = mgr._extract_text_from_content([item])
        assert result == "hello world"

    def test_image_content(self):
        mgr = MCPManager()
        item = MagicMock(spec=["data", "mimeType"])
        item.data = "base64data"
        item.mimeType = "image/png"
        result = mgr._extract_text_from_content([item])
        assert result == "[Image: image/png]"

    def test_image_content_unknown_mime(self):
        mgr = MCPManager()
        item = MagicMock()
        del item.text
        item.data = "base64data"
        # mimeType not set — getattr returns 'unknown'
        type(item).mimeType = MagicMock()  # make it raise AttributeError
        item.mimeType = "unknown"
        result = mgr._extract_text_from_content([item])
        assert result == "[Image: unknown]"

    def test_embedded_resource_with_text(self):
        mgr = MCPManager()

        # Use a simple class to precisely control hasattr behavior
        class FakeResource:
            text = "resource text"

        class FakeItem:
            resource = FakeResource()

        result = mgr._extract_text_from_content([FakeItem()])
        assert result == "resource text"

    def test_embedded_resource_with_uri(self):
        mgr = MCPManager()

        class FakeResource:
            uri = "file:///path"

        class FakeItem:
            resource = FakeResource()

        result = mgr._extract_text_from_content([FakeItem()])
        assert result == "[Resource: file:///path]"

    def test_embedded_resource_fallback(self):
        mgr = MCPManager()

        class FakeResource:
            pass

        class FakeItem:
            resource = FakeResource()

        result = mgr._extract_text_from_content([FakeItem()])
        assert "FakeItem" in result or "FakeResource" in result

    def test_empty_content(self):
        mgr = MCPManager()
        result = mgr._extract_text_from_content([])
        assert result == ""

    def test_unknown_content_type(self):
        mgr = MCPManager()
        item = object()  # no text, no data, no resource
        result = mgr._extract_text_from_content([item])
        assert str(item) in result

    def test_multiple_items(self):
        mgr = MCPManager()
        item1 = MagicMock()
        item1.text = "line1"
        item2 = MagicMock()
        item2.text = "line2"
        result = mgr._extract_text_from_content([item1, item2])
        assert result == "line1\nline2"


# ─────────────────────────────────────────────────────────
# Resource access
# ─────────────────────────────────────────────────────────


class TestGetResources:
    def test_no_servers(self):
        mgr = MCPManager()
        assert mgr.get_resources() == []

    def test_only_connected(self):
        mgr = MCPManager()
        mgr.add_server(MCPServerConfig(name="s1", command="node"))
        mgr.add_server(MCPServerConfig(name="s2", command="node"))
        mgr._servers["s1"].status = MCPServerStatus.CONNECTED
        mgr._servers["s2"].status = MCPServerStatus.DISCONNECTED

        res1 = MagicMock()
        res1.server_name = "s1"
        mgr._servers["s1"].resources = [res1]

        resources = mgr.get_resources()
        assert len(resources) == 1
        assert resources[0].server_name == "s1"


class TestReadResource:
    @pytest.mark.asyncio
    async def test_server_not_registered(self):
        mgr = MCPManager()
        with pytest.raises(MCPConnectionError, match="not registered"):
            await mgr.read_resource("missing", "file:///path")

    @pytest.mark.asyncio
    async def test_server_not_connected(self):
        mgr = MCPManager()
        mgr.add_server(MCPServerConfig(name="test", command="node"))
        with pytest.raises(MCPConnectionError, match="not connected"):
            await mgr.read_resource("test", "file:///path")

    @pytest.mark.asyncio
    async def test_no_session(self):
        mgr = MCPManager()
        mgr.add_server(MCPServerConfig(name="test", command="node"))
        mgr._servers["test"].status = MCPServerStatus.CONNECTED
        with pytest.raises(MCPConnectionError, match="no active session"):
            await mgr.read_resource("test", "file:///path")

    @pytest.mark.asyncio
    async def test_read_success(self):
        mgr = MCPManager()
        mgr.add_server(MCPServerConfig(name="test", command="node"))
        conn = mgr._servers["test"]
        conn.status = MCPServerStatus.CONNECTED

        mock_session = AsyncMock()
        mock_content = MagicMock()
        mock_content.text = "resource data"
        mock_result = MagicMock()
        mock_result.contents = [mock_content]
        mock_session.read_resource = AsyncMock(return_value=mock_result)
        conn._session = mock_session

        result = await mgr.read_resource("test", "file:///path")
        assert result == "resource data"

    @pytest.mark.asyncio
    async def test_read_exception(self):
        mgr = MCPManager()
        mgr.add_server(MCPServerConfig(name="test", command="node"))
        conn = mgr._servers["test"]
        conn.status = MCPServerStatus.CONNECTED

        mock_session = AsyncMock()
        mock_session.read_resource = AsyncMock(side_effect=Exception("read failed"))
        conn._session = mock_session

        with pytest.raises(MCPToolError, match="read failed"):
            await mgr.read_resource("test", "file:///path")


# ─────────────────────────────────────────────────────────
# build_langchain_tools
# ─────────────────────────────────────────────────────────


class TestBuildLangchainTools:
    def test_no_tools(self):
        mgr = MCPManager()
        tools = mgr.build_langchain_tools()
        assert tools == []

    def test_builds_tools(self):
        mgr = MCPManager()
        tool = MCPServerTool(
            server_name="server",
            name="my_tool",
            qualified_name="mcp__server__my_tool",
            description="A test tool",
            input_schema={
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
            is_read_only=True,
        )
        mgr._tool_map["mcp__server__my_tool"] = tool

        tools = mgr.build_langchain_tools()
        assert len(tools) == 1
        assert tools[0].name == "mcp__server__my_tool"
        assert tools[0].description == "A test tool"
        assert "mcp" in tools[0].tags
        assert tools[0].metadata["mcp_server_name"] == "server"
        assert tools[0].metadata["mcp_is_read_only"] is True

    def test_build_tool_no_properties(self):
        mgr = MCPManager()
        tool = MCPServerTool(
            server_name="server",
            name="no_params",
            qualified_name="mcp__server__no_params",
            description="No params",
            input_schema={},
            is_read_only=True,
        )
        mgr._tool_map["mcp__server__no_params"] = tool

        tools = mgr.build_langchain_tools()
        assert len(tools) == 1

    def test_build_tool_handles_exception(self):
        """Tool that fails schema conversion is skipped."""
        mgr = MCPManager()
        tool = MCPServerTool(
            server_name="server",
            name="bad_tool",
            qualified_name="mcp__server__bad_tool",
            description="Bad",
            input_schema="not a dict",
            is_read_only=True,
        )
        mgr._tool_map["mcp__server__bad_tool"] = tool

        # Should not raise, should log warning and skip
        tools = mgr.build_langchain_tools()
        assert len(tools) == 0


# ─────────────────────────────────────────────────────────
# list_mcp_servers_func / list_mcp_resources_func
# ─────────────────────────────────────────────────────────


class TestListMcpServersFunc:
    def test_no_servers(self):
        # mcp_manager is patched via autouse fixture with empty MCPManager
        result = list_mcp_servers_func()
        assert "No MCP servers configured" in result

    def test_with_servers(self):
        from dazi._singletons import mcp_manager

        mcp_manager.add_server(MCPServerConfig(name="srv1", command="node"))
        mcp_manager._servers["srv1"].status = MCPServerStatus.CONNECTED
        mcp_manager._servers["srv1"].tools = [MagicMock()]

        result = list_mcp_servers_func()
        assert "srv1" in result
        assert "+" in result  # connected icon

    def test_with_error_server(self):
        from dazi._singletons import mcp_manager

        mcp_manager.add_server(MCPServerConfig(name="err", command="node"))
        mcp_manager._servers["err"].status = MCPServerStatus.ERROR
        mcp_manager._servers["err"].error = "fail"

        result = list_mcp_servers_func()
        assert "err" in result
        assert "!" in result  # error icon
        assert "fail" in result

    def test_with_resources(self):
        from dazi._singletons import mcp_manager

        mcp_manager.add_server(MCPServerConfig(name="srv", command="node"))
        mcp_manager._servers["srv"].status = MCPServerStatus.CONNECTED
        mcp_manager._servers["srv"].resources = [MagicMock()]

        result = list_mcp_servers_func()
        assert "resources" in result


class TestListMcpResourcesFunc:
    def test_no_resources(self):
        result = list_mcp_resources_func()
        assert "No MCP resources" in result

    def test_no_resources_for_server(self):
        result = list_mcp_resources_func(server_name="missing")
        assert "No resources found for server 'missing'" in result

    def test_with_resources(self):
        from dazi._singletons import mcp_manager

        mcp_manager.add_server(MCPServerConfig(name="srv", command="node"))
        mcp_manager._servers["srv"].status = MCPServerStatus.CONNECTED
        from dazi.mcp_client import MCPResource

        mcp_manager._servers["srv"].resources = [
            MCPResource(
                server_name="srv",
                uri="file:///test",
                name="test.txt",
                mime_type="text/plain",
                description="A file",
            )
        ]

        result = list_mcp_resources_func()
        assert "file:///test" in result
        assert "test.txt" in result
        assert "text/plain" in result
        assert "A file" in result

    def test_filtered_by_server(self):
        from dazi._singletons import mcp_manager

        mcp_manager.add_server(MCPServerConfig(name="s1", command="node"))
        mcp_manager.add_server(MCPServerConfig(name="s2", command="node"))
        mcp_manager._servers["s1"].status = MCPServerStatus.CONNECTED
        mcp_manager._servers["s2"].status = MCPServerStatus.CONNECTED

        from dazi.mcp_client import MCPResource

        mcp_manager._servers["s1"].resources = [MCPResource(server_name="s1", uri="file:///a")]
        mcp_manager._servers["s2"].resources = [MCPResource(server_name="s2", uri="file:///b")]

        result = list_mcp_resources_func(server_name="s1")
        assert "file:///a" in result
        assert "file:///b" not in result

    def test_resource_no_optional_fields(self):
        from dazi._singletons import mcp_manager

        mcp_manager.add_server(MCPServerConfig(name="srv", command="node"))
        mcp_manager._servers["srv"].status = MCPServerStatus.CONNECTED

        from dazi.mcp_client import MCPResource

        mcp_manager._servers["srv"].resources = [MCPResource(server_name="srv", uri="file:///test")]

        result = list_mcp_resources_func()
        assert "file:///test" in result


# ─────────────────────────────────────────────────────────
# read_mcp_resource_func
# ─────────────────────────────────────────────────────────


class TestReadMcpResourceFunc:
    @pytest.mark.asyncio
    async def test_success(self):
        from dazi._singletons import mcp_manager

        mcp_manager.add_server(MCPServerConfig(name="srv", command="node"))
        conn = mcp_manager._servers["srv"]
        conn.status = MCPServerStatus.CONNECTED

        mock_session = AsyncMock()
        mock_content = MagicMock()
        mock_content.text = "data"
        mock_result = MagicMock()
        mock_result.contents = [mock_content]
        mock_session.read_resource = AsyncMock(return_value=mock_result)
        conn._session = mock_session

        from dazi.mcp_client import read_mcp_resource_func

        result = await read_mcp_resource_func("srv", "file:///path")
        assert result == "data"

    @pytest.mark.asyncio
    async def test_empty_content(self):
        from dazi._singletons import mcp_manager

        mcp_manager.add_server(MCPServerConfig(name="srv", command="node"))
        conn = mcp_manager._servers["srv"]
        conn.status = MCPServerStatus.CONNECTED

        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.contents = []
        mock_session.read_resource = AsyncMock(return_value=mock_result)
        conn._session = mock_session

        from dazi.mcp_client import read_mcp_resource_func

        result = await read_mcp_resource_func("srv", "file:///path")
        assert "empty content" in result.lower()

    @pytest.mark.asyncio
    async def test_error(self):

        from dazi.mcp_client import read_mcp_resource_func

        result = await read_mcp_resource_func("missing", "file:///path")
        assert "Error" in result


# ─────────────────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────────────────


class TestClearTaskCancellation:
    def test_no_current_task(self):
        with patch("dazi.mcp_client.asyncio.current_task", return_value=None):
            _clear_task_cancellation()

    def test_clears_cancellation(self):
        mock_task = MagicMock()
        mock_task.cancelling.return_value = 2
        mock_task.cancelling.side_effect = [2, 1, 0]
        with patch("dazi.mcp_client.asyncio.current_task", return_value=mock_task):
            _clear_task_cancellation()
        assert mock_task.uncancel.call_count == 2

    def test_handles_attribute_error(self):
        mock_task = MagicMock(spec=[])  # no cancelling attribute
        with patch("dazi.mcp_client.asyncio.current_task", return_value=mock_task):
            # Should not raise
            _clear_task_cancellation()


class TestForceCleanupStaleScopes:
    def test_no_current_task(self):
        with patch("dazi.mcp_client.asyncio.current_task", return_value=None):
            _force_cleanup_stale_scopes()

    def test_no_task_states(self):
        """Patch anyio._backends._asyncio._task_states to return empty dict."""
        mock_task = MagicMock()
        mock_task_states = MagicMock()
        mock_task_states.get.return_value = None
        with (
            patch("dazi.mcp_client.asyncio.current_task", return_value=mock_task),
            patch("anyio._backends._asyncio._task_states", mock_task_states),
        ):
            _force_cleanup_stale_scopes()


# ─────────────────────────────────────────────────────────
# Module-level tool definitions (line coverage)
# ─────────────────────────────────────────────────────────


class TestModuleToolDefinitions:
    def test_list_mcp_servers_tool_exists(self):
        from dazi.mcp_client import list_mcp_servers_meta, list_mcp_servers_tool

        assert list_mcp_servers_tool.name == "list_mcp_servers"
        assert list_mcp_servers_meta.name == "list_mcp_servers"
        assert list_mcp_servers_meta.safety.value == "safe"

    def test_list_mcp_resources_tool_exists(self):
        from dazi.mcp_client import list_mcp_resources_meta, list_mcp_resources_tool

        assert list_mcp_resources_tool.name == "list_mcp_resources"
        assert list_mcp_resources_meta.name == "list_mcp_resources"
        assert list_mcp_resources_meta.safety.value == "safe"

    def test_read_mcp_resource_tool_exists(self):
        from dazi.mcp_client import read_mcp_resource_meta, read_mcp_resource_tool

        assert read_mcp_resource_tool.name == "read_mcp_resource"
        assert read_mcp_resource_meta.name == "read_mcp_resource"
        assert read_mcp_resource_meta.safety.value == "safe"

    def test_mcp_resource_dataclass(self):
        from dazi.mcp_client import MCPResource

        res = MCPResource(
            server_name="srv",
            uri="file:///path",
            name="test.txt",
            mime_type="text/plain",
            description="A test",
        )
        assert res.server_name == "srv"
        assert res.uri == "file:///path"
        assert res.name == "test.txt"
        assert res.mime_type == "text/plain"
        assert res.description == "A test"

    def test_mcp_resource_defaults(self):
        from dazi.mcp_client import MCPResource

        res = MCPResource(server_name="srv", uri="file:///path")
        assert res.name == ""
        assert res.mime_type == ""
        assert res.description == ""


# ─────────────────────────────────────────────────────────
# MCPServerTool dataclass
# ─────────────────────────────────────────────────────────


class TestMCPServerTool:
    def test_defaults(self):
        tool = MCPServerTool(
            server_name="srv",
            name="tool",
            qualified_name="mcp__srv__tool",
            description="desc",
            input_schema={},
        )
        assert tool.is_read_only is False

    def test_read_only_true(self):
        tool = MCPServerTool(
            server_name="srv",
            name="tool",
            qualified_name="mcp__srv__tool",
            description="desc",
            input_schema={},
            is_read_only=True,
        )
        assert tool.is_read_only is True
