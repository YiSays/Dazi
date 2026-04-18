"""Test helpers for mocking _singletons and common test utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage


def patch_singletons(monkeypatch, tmp_path: Path) -> None:
    """Patch all dazi._singletons attributes with tmp_path-based instances.

    Also patches module-level references in production modules that capture
    singleton values at import time (e.g., ``from dazi._singletons import X``).
    """
    from dazi.background import BackgroundTaskManager
    from dazi.coordinator import AutonomousTeammate
    from dazi.cost_tracker import CostTracker
    from dazi.mailbox import Mailbox
    from dazi.mcp_client import MCPManager
    from dazi.memory import MemoryStore
    from dazi.permission_bridge import PermissionBridge
    from dazi.proactive import ProactiveManager
    from dazi.settings import SettingsManager
    from dazi.skills import SkillRegistry
    from dazi.task_store import TaskStore
    from dazi.team import TeamManager
    from dazi.teammate import TeammateRunner
    from dazi.worktree import WorktreeManager

    data_dir = tmp_path / ".dazi"
    data_dir.mkdir(exist_ok=True)

    instances = {
        "memory_store": MemoryStore(data_dir / "memory"),
        "settings_manager": SettingsManager(project_root=data_dir, user_dir=data_dir),
        "cost_tracker": CostTracker(data_dir),
        "mcp_manager": MCPManager(),
        "skill_registry": SkillRegistry(),
        "proactive_manager": ProactiveManager(),
        "task_store": TaskStore(data_dir / "tasks", list_id="default"),
        "team_task_store": None,
        "background_manager": BackgroundTaskManager(data_dir / "background"),
        "team_manager": TeamManager(),
        "teammate_runner": TeammateRunner(),
        "mailbox": Mailbox(),
        "permission_bridge": PermissionBridge(mailbox=MagicMock()),
        "autonomous_teammate": AutonomousTeammate(),
        "worktree_manager": WorktreeManager(),
    }

    for name, instance in instances.items():
        monkeypatch.setattr(f"dazi._singletons.{name}", instance)

    # Patch module-level references captured at import time in production code.
    # These modules do ``from dazi._singletons import X`` at the top level,
    # so monkeypatch must also update their local references.
    _MODULE_SINGLETON_MAP: dict[str, list[str]] = {
        "dazi.graph": [
            "background_manager",
            "cost_tracker",
            "mcp_manager",
            "settings_manager",
        ],
        "dazi.lifecycle": [
            "autonomous_teammate",
            "background_manager",
            "cost_tracker",
            "mcp_manager",
            "proactive_manager",
            "settings_manager",
            "skill_registry",
            "teammate_runner",
            "worktree_manager",
        ],
        "dazi.repl_display": [
            "background_manager",
            "mcp_manager",
            "memory_store",
            "skill_registry",
            "task_store",
        ],
        "dazi.repl_commands": [
            "autonomous_teammate",
            "cost_tracker",
            "mcp_manager",
            "memory_store",
            "proactive_manager",
            "settings_manager",
            "skill_registry",
            "task_store",
            "team_manager",
            "worktree_manager",
        ],
        "dazi.llm": [
            "memory_store",
            "settings_manager",
            "skill_registry",
        ],
        "dazi.repl_teams": [
            "mailbox",
            "team_manager",
            "team_task_store",
        ],
        "dazi.main": [
            "autonomous_teammate",
            "background_manager",
            "cost_tracker",
            "mcp_manager",
            "memory_store",
            "proactive_manager",
            "settings_manager",
            "task_store",
            "team_manager",
            "worktree_manager",
        ],
    }

    for module_name, names in _MODULE_SINGLETON_MAP.items():
        for name in names:
            if name in instances:
                monkeypatch.setattr(f"{module_name}.{name}", instances[name])


def create_mock_llm(response_text: str = "Mock response") -> AsyncMock:
    """Create a mock LLM that behaves like ChatOpenAI."""
    mock_llm = AsyncMock()
    mock_response = AIMessage(content=response_text)
    mock_response.response_metadata = {"token_usage": {"prompt_tokens": 10, "completion_tokens": 5}}
    mock_llm.ainvoke = AsyncMock(return_value=mock_response)
    mock_llm.astream = AsyncMock(return_value=iter([mock_response]))
    mock_llm.bind_tools = MagicMock(return_value=mock_llm)
    return mock_llm


def create_mock_tool(name: str, result: str = "mock result") -> MagicMock:
    """Create a mock StructuredTool."""
    mock_tool = MagicMock()
    mock_tool.name = name
    mock_tool.invoke = MagicMock(return_value=result)
    mock_tool.coroutine = AsyncMock(return_value=result)
    return mock_tool


def make_messages(pairs: list[tuple[str, str]]) -> list[Any]:
    """Create a list of BaseMessage from (type, text) tuples.

    Types: "human", "ai", "system", "tool"
    """
    type_map = {
        "human": HumanMessage,
        "ai": AIMessage,
        "system": SystemMessage,
        "tool": ToolMessage,
    }
    messages = []
    for msg_type, text in pairs:
        cls = type_map.get(msg_type, HumanMessage)
        if msg_type == "tool":
            messages.append(cls(content=text, tool_call_id="test"))
        else:
            messages.append(cls(content=text))
    return messages
