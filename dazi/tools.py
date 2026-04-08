"""Dazi tools — public API for all tool definitions.

Tool definitions live in their corresponding core modules.
This module imports them, assembles mode-filtered tool lists,
and re-exports singletons from the centralized registry.
"""

from __future__ import annotations

from dazi.core.base import DaziTool

# ── Singletons, constants, and context ──
from dazi.core._singletons import (  # noqa: F401
    active_team_name,
    background_manager,
    cost_tracker,
    current_agent_name,
    mailbox,
    memory_store,
    mcp_manager,
    permission_bridge,
    proactive_manager,
    autonomous_teammate,
    settings_manager,
    skill_registry,
    task_store,
    team_manager,
    teammate_runner,
    worktree_manager,
    # Path constants
    BACKGROUND_DIR,
    MEMORY_DIR,
    PLAN_FILE,
    TASKS_DIR,
    TEAMS_DIR,
)

# ── Tool imports from core modules ──
from dazi.core.filesystem import (
    calculator_tool, calculator_meta,
    file_reader_tool, file_reader_meta,
    file_writer_tool, file_writer_meta,
    shell_exec_tool, shell_exec_meta,
    plan_writer_tool, plan_writer_meta,
)
from dazi.core.coordinator import (
    delegate_task_tool, delegate_task_meta,
    spawn_agent_tool, spawn_agent_meta,
    AgentSpawnInput,
)
from dazi.core.memory import (
    memory_write_tool, memory_write_meta,
    memory_read_tool, memory_read_meta,
    memory_search_tool, memory_search_meta,
)
from dazi.core.task_store import (
    task_create_tool, task_create_meta,
    task_update_tool, task_update_meta,
    task_list_tool, task_list_meta,
    task_get_tool, task_get_meta,
)
from dazi.core.background import (
    run_background_tool, run_background_meta,
    check_background_tool, check_background_meta,
    cancel_background_tool, cancel_background_meta,
)
from dazi.core.mcp_client import (
    list_mcp_servers_tool, list_mcp_servers_meta,
    list_mcp_resources_tool, list_mcp_resources_meta,
    read_mcp_resource_tool, read_mcp_resource_meta,
)
from dazi.core.skills import skill_tool, skill_tool_meta
from dazi.core.team import (
    create_team_tool, create_team_meta,
    delete_team_tool, delete_team_meta,
    list_teams_tool, list_teams_meta,
    show_team_tool, show_team_meta,
)
from dazi.core.mailbox import (
    send_message_tool, send_message_meta,
    check_inbox_tool, check_inbox_meta,
    send_idle_notification,
)
from dazi.core.permission_bridge import (
    request_permission_tool, request_permission_meta,
)
from dazi.core.proactive import sleep_tool, sleep_meta
from dazi.core.worktree import (
    create_worktree_tool, create_worktree_meta,
    finish_worktree_tool, finish_worktree_meta,
    list_worktrees_tool, list_worktrees_meta,
)

# ── All tool metadata ──
ALL_TOOL_META: dict[str, DaziTool] = {
    t.name: t for t in [
        # Filesystem
        calculator_meta, file_reader_meta, file_writer_meta,
        shell_exec_meta, plan_writer_meta,
        # Delegate + spawn
        delegate_task_meta, spawn_agent_meta,
        # Memory
        memory_write_meta, memory_read_meta, memory_search_meta,
        # Tasks
        task_create_meta, task_update_meta, task_list_meta, task_get_meta,
        # Background
        run_background_meta, check_background_meta, cancel_background_meta,
        # MCP
        list_mcp_servers_meta, list_mcp_resources_meta, read_mcp_resource_meta,
        # Skills
        skill_tool_meta,
        # Teams
        create_team_meta, delete_team_meta, list_teams_meta, show_team_meta,
        # Messaging
        send_message_meta, check_inbox_meta,
        # Permission
        request_permission_meta,
        # Proactive
        sleep_meta,
        # Worktree
        create_worktree_meta, finish_worktree_meta, list_worktrees_meta,
    ]
}

# ── Plan mode tools (SAFE only) ──
PLAN_MODE_TOOLS = [
    # Filesystem (safe subset)
    file_reader_tool, shell_exec_tool, plan_writer_tool, calculator_tool,
    # Memory
    memory_write_tool, memory_read_tool, memory_search_tool,
    # Tasks
    task_create_tool, task_update_tool, task_list_tool, task_get_tool,
    # Background (read-only)
    run_background_tool, check_background_tool,
    # Skills
    skill_tool,
    # Teams (read-only)
    list_teams_tool, show_team_tool,
    # Messaging (read-only)
    check_inbox_tool,
    # Proactive
    sleep_tool,
    # Worktree (read-only)
    list_worktrees_tool,
]
PLAN_MODE_META: dict[str, DaziTool] = {t.name: t for t in [
    file_reader_meta, shell_exec_meta, plan_writer_meta, calculator_meta,
    memory_write_meta, memory_read_meta, memory_search_meta,
    task_create_meta, task_update_meta, task_list_meta, task_get_meta,
    run_background_meta, check_background_meta,
    skill_tool_meta,
    list_teams_meta, show_team_meta,
    check_inbox_meta,
    sleep_meta,
    list_worktrees_meta,
]}

# ── Execute mode tools (all tools) ──
EXECUTE_MODE_TOOLS = [
    # Filesystem (all)
    calculator_tool, file_reader_tool, file_writer_tool, shell_exec_tool,
    # Delegate + spawn
    delegate_task_tool, spawn_agent_tool,
    # Plan writer
    plan_writer_tool,
    # Memory
    memory_write_tool, memory_read_tool, memory_search_tool,
    # Tasks
    task_create_tool, task_update_tool, task_list_tool, task_get_tool,
    # Background (all)
    run_background_tool, check_background_tool, cancel_background_tool,
    # Skills
    skill_tool,
    # Teams (all)
    create_team_tool, delete_team_tool, list_teams_tool, show_team_tool,
    # Messaging (all)
    send_message_tool, check_inbox_tool,
    # Permission
    request_permission_tool,
    # Proactive
    sleep_tool,
    # Worktree (all)
    create_worktree_tool, finish_worktree_tool, list_worktrees_tool,
]
EXECUTE_MODE_META: dict[str, DaziTool] = {
    t.name: t for t in [
        calculator_meta, file_reader_meta, file_writer_meta,
        shell_exec_meta, delegate_task_meta, spawn_agent_meta,
        plan_writer_meta,
        memory_write_meta, memory_read_meta, memory_search_meta,
        task_create_meta, task_update_meta, task_list_meta, task_get_meta,
        run_background_meta, check_background_meta, cancel_background_meta,
        skill_tool_meta,
        create_team_meta, delete_team_meta, list_teams_meta, show_team_meta,
        send_message_meta, check_inbox_meta,
        request_permission_meta,
        sleep_meta,
        create_worktree_meta, finish_worktree_meta, list_worktrees_meta,
    ]
}
