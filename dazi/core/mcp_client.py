"""MCP (Model Context Protocol) client — tool discovery, schema normalization, execution routing.

KEY CONCEPTS:
  1. MCP servers expose tools via JSON-RPC over stdio (subprocess stdin/stdout)
  2. The mcp Python SDK handles protocol negotiation, message framing, etc.
  3. Tool naming convention: mcp__<server>__<tool> (qualified name for routing)
  4. MCP tools use passthrough permission behavior
  5. Schema normalization: MCP JSON Schema -> Pydantic model for LangChain tools
  6. Connection lifecycle: stdio transport -> initialize -> list_tools -> ready
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field, create_model

from dazi.core.base import DaziTool, ToolSafety

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────

MAX_MCP_DESCRIPTION_LENGTH = 2048

MCP_TOOL_PREFIX = "mcp__"
MCP_NAME_SEPARATOR = "__"


# ─────────────────────────────────────────────────────────
# EXCEPTIONS
# ─────────────────────────────────────────────────────────


class MCPToolError(Exception):
    """Error from an MCP tool call (server returned isError: true)."""

    def __init__(self, server_name: str, tool_name: str, message: str):
        self.server_name = server_name
        self.tool_name = tool_name
        self.message = message
        super().__init__(f"MCP tool '{server_name}/{tool_name}' failed: {message}")


class MCPConnectionError(Exception):
    """Error connecting to an MCP server."""

    def __init__(self, server_name: str, message: str):
        self.server_name = server_name
        self.message = message
        super().__init__(f"MCP server '{server_name}' connection error: {message}")


# ─────────────────────────────────────────────────────────
# DATA MODELS
# ─────────────────────────────────────────────────────────


class MCPServerStatus(str, Enum):
    """MCP server connection status."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"


@dataclass
class MCPServerConfig:
    """MCP server configuration — parsed from settings.json mcpServers.

    Supports stdio transport: command + args + env.

    Example config dict:
        {
            "type": "stdio",
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
            "env": {"GITHUB_TOKEN": "..."},
            "description": "File system access"
        }
    """
    name: str
    command: str
    args: list[str] = field(default_factory=list)
    env: dict[str, str] | None = None
    description: str = ""

    @classmethod
    def from_dict(cls, name: str, data: dict) -> MCPServerConfig:
        """Parse from JSON config dict."""
        return cls(
            name=name,
            command=data["command"],
            args=data.get("args", []),
            env=data.get("env"),
            description=data.get("description", ""),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize for display/debugging."""
        result: dict[str, Any] = {"command": self.command}
        if self.args:
            result["args"] = self.args
        if self.env:
            result["env"] = self.env
        if self.description:
            result["description"] = self.description
        return result


@dataclass
class MCPServerTool:
    """Tool discovered from an MCP server.

    Each MCP server tool gets a qualified name: mcp__<server>__<tool>
    """
    server_name: str       # Original server name (e.g., "filesystem")
    name: str              # Original tool name (e.g., "read_file")
    qualified_name: str    # mcp__filesystem__read_file
    description: str
    input_schema: dict     # JSON Schema dict
    is_read_only: bool = False


@dataclass
class MCPResource:
    """Resource from an MCP server."""
    server_name: str
    uri: str
    name: str = ""
    mime_type: str = ""
    description: str = ""


@dataclass
class MCPServerConnection:
    """Live connection to an MCP server.

    Tracks status, discovered tools/resources, and the underlying SDK objects.
    """
    config: MCPServerConfig
    status: MCPServerStatus = MCPServerStatus.DISCONNECTED
    tools: list[MCPServerTool] = field(default_factory=list)
    resources: list[MCPResource] = field(default_factory=list)
    error: str | None = None
    # SDK objects (not serialized)
    _session: Any = field(default=None, repr=False)
    _stdio_cm: Any = field(default=None, repr=False)
    _cm_stack: Any = field(default=None, repr=False)


# ─────────────────────────────────────────────────────────
# NAME NORMALIZATION
# ─────────────────────────────────────────────────────────

_INVALID_NAME_CHARS = re.compile(r"[^a-zA-Z0-9_-]")


def _normalize_name(name: str) -> str:
    """Normalize a name for use in qualified tool names.

    Replaces non-alphanumeric chars (except _ and -) with underscores.
    """
    return _INVALID_NAME_CHARS.sub("_", name).lower()


def _build_mcp_tool_name(server_name: str, tool_name: str) -> str:
    """Build qualified MCP tool name: mcp__<server>__<tool>."""
    normalized_server = _normalize_name(server_name)
    normalized_tool = _normalize_name(tool_name)
    return f"{MCP_TOOL_PREFIX}{normalized_server}{MCP_NAME_SEPARATOR}{normalized_tool}"


def _parse_mcp_tool_name(qualified_name: str) -> tuple[str, str]:
    """Parse qualified name back into server name and tool name.

    Returns: (server_name, tool_name)
    Raises ValueError if the name doesn't start with "mcp__".
    """
    if not qualified_name.startswith(MCP_TOOL_PREFIX):
        raise ValueError(f"Not an MCP tool name: {qualified_name}")

    # Remove prefix, split on separator
    rest = qualified_name[len(MCP_TOOL_PREFIX):]
    parts = rest.split(MCP_NAME_SEPARATOR, 1)
    if len(parts) != 2:
        raise ValueError(f"Invalid MCP tool name format: {qualified_name}")

    return parts[0], parts[1]


# ─────────────────────────────────────────────────────────
# SCHEMA CONVERSION — JSON Schema -> Pydantic model
# ─────────────────────────────────────────────────────────


def _convert_schema_to_pydantic(tool_name: str, schema: dict) -> type[BaseModel]:
    """Convert an MCP tool's JSON Schema to a Pydantic model for LangChain.

    Handles: string, integer, number, boolean, array, object, $ref.
    Optional fields (not in "required") get default=None.
    """
    properties = schema.get("properties", {})
    required = set(schema.get("required", []))

    fields_dict: dict[str, Any] = {}

    for prop_name, prop_schema in properties.items():
        pydantic_type = _map_json_type(prop_schema)
        is_required = prop_name in required

        if is_required:
            fields_dict[prop_name] = (pydantic_type, Field(...))
        else:
            fields_dict[prop_name] = (pydantic_type | None, Field(default=None))

    # Handle empty schema (no properties) — accept any kwargs
    if not fields_dict:
        fields_dict["input_data"] = (dict, Field(default_factory=dict))

    model = create_model(
        f"MCPInput_{tool_name}",
        **fields_dict,
    )
    return model


def _map_json_type(prop_schema: dict) -> type:
    """Map a JSON Schema type to a Python type."""
    json_type = prop_schema.get("type", "string")

    type_map = {
        "string": str,
        "integer": int,
        "number": float,
        "boolean": bool,
    }

    if json_type in type_map:
        return type_map[json_type]
    elif json_type == "array":
        # Arrays are typed as list[Any]
        return list
    elif json_type == "object":
        # Nested objects are typed as dict
        return dict
    else:
        # Fallback to string
        return str


# ─────────────────────────────────────────────────────────
# MCP MANAGER — connection lifecycle + tool routing
# ─────────────────────────────────────────────────────────


class MCPManager:
    """Manages MCP server connections, tool discovery, and execution routing.

    Uses the official mcp Python SDK for protocol handling.
    Pattern: configure servers -> connect -> discover tools -> route calls.
    """

    def __init__(self) -> None:
        self._servers: dict[str, MCPServerConnection] = {}
        self._tool_map: dict[str, MCPServerTool] = {}  # qualified_name -> tool

    # ─────────────────────────────────────────────────
    # SERVER MANAGEMENT
    # ─────────────────────────────────────────────────

    def add_server(self, config: MCPServerConfig) -> None:
        """Register an MCP server configuration (does not connect)."""
        conn = MCPServerConnection(config=config)
        self._servers[config.name] = conn
        logger.info(f"MCP server registered: {config.name} ({config.command})")

    def remove_server(self, name: str) -> None:
        """Remove a server configuration and disconnect if connected."""
        if name in self._servers:
            # Remove tools from tool map
            conn = self._servers[name]
            for tool in conn.tools:
                self._tool_map.pop(tool.qualified_name, None)
            del self._servers[name]
            logger.info(f"MCP server removed: {name}")

    # ─────────────────────────────────────────────────
    # CONNECTION LIFECYCLE
    # ─────────────────────────────────────────────────

    async def connect_all(self) -> dict[str, bool]:
        """Connect to all registered servers.

        Returns dict of server_name -> success (bool).
        """
        results = {}
        for name in self._servers:
            results[name] = await self.connect_server(name)
        return results

    async def connect_server(self, name: str) -> bool:
        """Connect to a specific MCP server via stdio.

        Flow: spawn subprocess -> initialize -> list_tools -> list_resources

        Returns True on success, False on failure (error stored in connection).
        """
        if name not in self._servers:
            logger.warning(f"MCP server not found: {name}")
            return False

        conn = self._servers[name]
        config = conn.config
        conn.status = MCPServerStatus.CONNECTING
        conn.error = None

        try:
            from mcp.client.stdio import StdioServerParameters, stdio_client

            # Build server parameters
            server_params = StdioServerParameters(
                command=config.command,
                args=config.args,
                env=config.env,
            )

            # Connect via stdio transport
            stdio_cm = stdio_client(server_params)
            read_stream, write_stream = await stdio_cm.__aenter__()
            conn._stdio_cm = stdio_cm

            from mcp import ClientSession

            # Create and initialize session
            session_cm = ClientSession(read_stream, write_stream)
            session = await session_cm.__aenter__()
            conn._cm_stack = session_cm
            conn._session = session

            # Initialize the MCP connection
            await session.initialize()

            conn.status = MCPServerStatus.CONNECTED
            logger.info(f"MCP server connected: {name}")

            # Discover tools
            await self._discover_tools(name, conn)

            # Discover resources
            await self._discover_resources(name, conn)

            return True

        except Exception as e:
            conn.status = MCPServerStatus.ERROR
            conn.error = str(e)
            # Clean up partial connections
            await self._cleanup_connection(conn)
            logger.error(f"MCP server '{name}' connection failed: {e}")
            return False

    async def disconnect_server(self, name: str) -> None:
        """Disconnect from an MCP server and clean up."""
        if name not in self._servers:
            return

        conn = self._servers[name]
        if conn.status == MCPServerStatus.DISCONNECTED:
            return

        await self._cleanup_connection(conn)
        conn.status = MCPServerStatus.DISCONNECTED
        # Remove tools from tool map
        for tool in conn.tools:
            self._tool_map.pop(tool.qualified_name, None)
        conn.tools = []
        conn.resources = []
        logger.info(f"MCP server disconnected: {name}")

    async def disconnect_all(self) -> None:
        """Disconnect all servers."""
        for name in list(self._servers.keys()):
            await self.disconnect_server(name)

    async def _cleanup_connection(self, conn: MCPServerConnection) -> None:
        """Clean up SDK session and stdio transport."""
        try:
            if conn._cm_stack is not None:
                await conn._cm_stack.__aexit__(None, None, None)
        except Exception as e:
            logger.debug(f"Error closing session: {e}")
        finally:
            conn._session = None
            conn._cm_stack = None

        try:
            if conn._stdio_cm is not None:
                await conn._stdio_cm.__aexit__(None, None, None)
        except Exception as e:
            logger.debug(f"Error closing stdio transport: {e}")
        finally:
            conn._stdio_cm = None

    # ─────────────────────────────────────────────────
    # TOOL DISCOVERY
    # ─────────────────────────────────────────────────

    async def _discover_tools(self, name: str, conn: MCPServerConnection) -> None:
        """Discover tools from a connected MCP server.

        Calls tools/list and populates conn.tools + self._tool_map.
        """
        session = conn._session
        if session is None:
            return

        try:
            result = await session.list_tools()
            tools: list[MCPServerTool] = []

            for mcp_tool in result.tools:
                # Build qualified name
                qualified_name = _build_mcp_tool_name(name, mcp_tool.name)

                # Check annotations for read-only hint
                is_read_only = False
                if mcp_tool.annotations is not None:
                    # mcp.types.Annotations has audience field
                    # readOnlyHint is in the MCP spec but SDK exposes via annotations
                    is_read_only = getattr(mcp_tool.annotations, "audience", None) is None

                tool = MCPServerTool(
                    server_name=name,
                    name=mcp_tool.name,
                    qualified_name=qualified_name,
                    description=(mcp_tool.description or "")[:MAX_MCP_DESCRIPTION_LENGTH],
                    input_schema=(
                        mcp_tool.inputSchema.model_dump()
                        if hasattr(mcp_tool.inputSchema, "model_dump")
                        else dict(mcp_tool.inputSchema) if mcp_tool.inputSchema
                        else {}
                    ),
                    is_read_only=is_read_only,
                )
                tools.append(tool)
                self._tool_map[qualified_name] = tool

            conn.tools = tools
            logger.info(f"MCP server '{name}': discovered {len(tools)} tools")

        except Exception as e:
            logger.error(f"Failed to discover tools from '{name}': {e}")

    async def _discover_resources(self, name: str, conn: MCPServerConnection) -> None:
        """Discover resources from a connected MCP server.

        Calls resources/list and populates conn.resources.
        """
        session = conn._session
        if session is None:
            return

        try:
            result = await session.list_resources()
            resources: list[MCPResource] = []

            for resource in (result.resources or []):
                res = MCPResource(
                    server_name=name,
                    uri=resource.uri,
                    name=resource.name or "",
                    mime_type=resource.mimeType or "",
                    description=resource.description or "",
                )
                resources.append(res)

            conn.resources = resources
            if resources:
                logger.info(f"MCP server '{name}': discovered {len(resources)} resources")

        except Exception as e:
            # Resources are optional — log but don't fail
            logger.debug(f"Failed to discover resources from '{name}': {e}")

    # ─────────────────────────────────────────────────
    # TOOL EXECUTION
    # ─────────────────────────────────────────────────

    async def call_tool(self, qualified_name: str, arguments: dict) -> str:
        """Route a tool call to the appropriate MCP server.

        Flow: parse name -> find server -> session.call_tool() -> process result
        """
        # Parse qualified name to find server and tool
        server_name, tool_name = _parse_mcp_tool_name(qualified_name)

        # Look up the server connection
        conn = self._servers.get(server_name)
        if conn is None:
            raise MCPConnectionError(server_name, f"Server '{server_name}' is not registered")

        if conn.status != MCPServerStatus.CONNECTED:
            raise MCPConnectionError(
                server_name,
                f"Server '{server_name}' is not connected (status: {conn.status.value})",
            )

        if conn._session is None:
            raise MCPConnectionError(server_name, f"Server '{server_name}' has no active session")

        try:
            result = await conn._session.call_tool(tool_name, arguments=arguments)

            # Check for error response
            if result.isError:
                # Extract error text from content
                error_text = self._extract_text_from_content(result.content)
                raise MCPToolError(server_name, tool_name, error_text)

            # Extract text from content items
            return self._extract_text_from_content(result.content)

        except MCPToolError:
            raise
        except MCPConnectionError:
            raise
        except Exception as e:
            raise MCPToolError(server_name, tool_name, str(e))

    def _extract_text_from_content(self, content: list) -> str:
        """Extract text from MCP CallToolResult content items.

        Handles TextContent, ImageContent, EmbeddedResource, etc.
        """
        texts = []
        for item in content:
            # TextContent: {type: "text", text: "..."}
            if hasattr(item, "text"):
                texts.append(item.text)
            # ImageContent: {type: "image", data: "...", mimeType: "..."}
            elif hasattr(item, "data"):
                texts.append(f"[Image: {getattr(item, 'mimeType', 'unknown')}]")
            # EmbeddedResource: {type: "resource", resource: {...}}
            elif hasattr(item, "resource"):
                resource = item.resource
                if hasattr(resource, "text"):
                    texts.append(resource.text)
                elif hasattr(resource, "uri"):
                    texts.append(f"[Resource: {resource.uri}]")
                else:
                    texts.append(str(item))
            else:
                texts.append(str(item))

        return "\n".join(texts) if texts else ""

    # ─────────────────────────────────────────────────
    # RESOURCE ACCESS
    # ─────────────────────────────────────────────────

    def get_resources(self) -> list[MCPResource]:
        """Get all resources from all connected servers."""
        resources = []
        for conn in self._servers.values():
            if conn.status == MCPServerStatus.CONNECTED:
                resources.extend(conn.resources)
        return resources

    async def read_resource(self, server_name: str, uri: str) -> str:
        """Read a specific MCP resource."""
        conn = self._servers.get(server_name)
        if conn is None:
            raise MCPConnectionError(server_name, f"Server '{server_name}' is not registered")

        if conn.status != MCPServerStatus.CONNECTED:
            raise MCPConnectionError(
                server_name,
                f"Server '{server_name}' is not connected (status: {conn.status.value})",
            )

        if conn._session is None:
            raise MCPConnectionError(server_name, f"Server '{server_name}' has no active session")

        try:
            result = await conn._session.read_resource(uri)
            return self._extract_text_from_content(result.contents)
        except Exception as e:
            raise MCPToolError(server_name, uri, str(e))

    # ─────────────────────────────────────────────────
    # TOOL ACCESS
    # ─────────────────────────────────────────────────

    def get_tools(self) -> list[MCPServerTool]:
        """Get all discovered tools from all connected servers."""
        return list(self._tool_map.values())

    def get_tool(self, qualified_name: str) -> MCPServerTool | None:
        """Look up a tool by its qualified name."""
        return self._tool_map.get(qualified_name)

    def get_server(self, name: str) -> MCPServerConnection | None:
        """Get server connection by name."""
        return self._servers.get(name)

    def list_servers(self) -> list[dict]:
        """List all registered servers with their status."""
        result = []
        for name, conn in self._servers.items():
            result.append({
                "name": name,
                "status": conn.status.value,
                "tool_count": len(conn.tools),
                "resource_count": len(conn.resources),
                "command": conn.config.command,
                "error": conn.error,
            })
        return result

    # ─────────────────────────────────────────────────
    # LANGCHAIN TOOL BUILDING
    # ─────────────────────────────────────────────────

    def build_langchain_tools(self) -> list[StructuredTool]:
        """Build LangChain StructuredTool list from all discovered MCP tools.

        Each MCP tool becomes a StructuredTool with:
          - name: qualified_name (e.g., mcp__filesystem__read_file)
          - description: MCP tool description (truncated to 2048 chars)
          - args_schema: Pydantic model converted from JSON Schema
          - coroutine: calls MCPManager.call_tool()
        """
        tools = []
        for mcp_tool in self.get_tools():
            try:
                # Build Pydantic model from JSON Schema
                args_schema = _convert_schema_to_pydantic(
                    mcp_tool.qualified_name, mcp_tool.input_schema
                )

                # Create the async function that routes to the MCP server
                async def _call_mcp_tool(
                    arguments: dict,
                    _qualified_name: str = mcp_tool.qualified_name,
                ) -> str:
                    return await self.call_tool(_qualified_name, arguments)

                tool = StructuredTool.from_function(
                    func=lambda _: "",
                    coroutine=_call_mcp_tool,
                    name=mcp_tool.qualified_name,
                    description=mcp_tool.description,
                    args_schema=args_schema,
                )

                # Tag the tool with MCP metadata for permission checking
                # Use tags + metadata (Pydantic model fields) since we can't
                # set arbitrary attributes on StructuredTool.
                tool.tags = ["mcp"]
                tool.metadata = {
                    "mcp_server_name": mcp_tool.server_name,
                    "mcp_tool_name": mcp_tool.name,
                    "mcp_is_read_only": mcp_tool.is_read_only,
                }

                tools.append(tool)
            except Exception as e:
                logger.warning(
                    f"Failed to build LangChain tool for {mcp_tool.qualified_name}: {e}"
                )
                continue

        return tools

    # ─────────────────────────────────────────────────
    # RESET — for testing
    # ─────────────────────────────────────────────────

    def reset(self) -> None:
        """Reset all state. Does not disconnect — for testing only."""
        self._servers.clear()
        self._tool_map.clear()


# ─────────────────────────────────────────────────────────
# MCP MANAGEMENT TOOLS
# ─────────────────────────────────────────────────────────


def list_mcp_servers_func() -> str:
    """List all registered MCP servers with their status."""
    from dazi.core._singletons import mcp_manager

    servers = mcp_manager.list_servers()
    if not servers:
        return "No MCP servers configured. Add servers via settings.json mcpServers field."

    lines = ["MCP Servers:"]
    for s in servers:
        status_icon = {"connected": "+", "disconnected": "-", "connecting": "...", "error": "!"}
        icon = status_icon.get(s["status"], "?")
        line = f"  [{icon}] {s['name']} ({s['status']})"
        if s["tool_count"]:
            line += f" — {s['tool_count']} tools"
        if s["resource_count"]:
            line += f", {s['resource_count']} resources"
        if s["error"]:
            line += f" — ERROR: {s['error']}"
        lines.append(line)

    lines.append("\nUse /mcp to see details. MCP server tools are prefixed with 'mcp__'.")
    return "\n".join(lines)


def list_mcp_resources_func(server_name: str = "") -> str:
    """List available resources from MCP servers."""
    from dazi.core._singletons import mcp_manager

    all_resources = mcp_manager.get_resources()
    if server_name:
        all_resources = [r for r in all_resources if r.server_name == server_name]

    if not all_resources:
        if server_name:
            return f"No resources found for server '{server_name}'."
        return "No MCP resources available from connected servers."

    lines = ["MCP Resources:"]
    for r in all_resources:
        line = f"  [{r.server_name}] {r.uri}"
        if r.name:
            line += f" — {r.name}"
        if r.mime_type:
            line += f" ({r.mime_type})"
        if r.description:
            line += f"\n    {r.description}"
        lines.append(line)

    return "\n".join(lines)


async def read_mcp_resource_func(server_name: str, uri: str) -> str:
    """Read a specific MCP resource by server name and URI."""
    from dazi.core._singletons import mcp_manager

    try:
        content = await mcp_manager.read_resource(server_name, uri)
        return content or f"Resource {uri} returned empty content."
    except Exception as e:
        return f"Error reading resource: {e}"


list_mcp_servers_tool = StructuredTool.from_function(
    func=list_mcp_servers_func,
    name="list_mcp_servers",
    description="List all registered MCP servers with their connection status, tool count, and resource count. Use this to discover available MCP tools and server health.",
)

list_mcp_servers_meta = DaziTool(
    name="list_mcp_servers",
    description="List all registered MCP servers.",
    safety=ToolSafety.SAFE,
)

list_mcp_resources_tool = StructuredTool.from_function(
    func=list_mcp_resources_func,
    name="list_mcp_resources",
    description="List available resources from connected MCP servers. Optionally filter by server name.",
)

list_mcp_resources_meta = DaziTool(
    name="list_mcp_resources",
    description="List available MCP resources.",
    safety=ToolSafety.SAFE,
)

read_mcp_resource_tool = StructuredTool.from_function(
    func=lambda **kwargs: "",
    coroutine=read_mcp_resource_func,
    name="read_mcp_resource",
    description="Read a specific MCP resource by server name and URI. Returns the resource content as text.",
)

read_mcp_resource_meta = DaziTool(
    name="read_mcp_resource",
    description="Read an MCP resource.",
    safety=ToolSafety.SAFE,
)
