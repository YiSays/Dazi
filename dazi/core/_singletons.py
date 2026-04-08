"""Centralized singleton instances and mutable context for all core modules.

Singletons are created here in dependency order to avoid circular imports.
Core modules import from this file when they need shared state.
"""

from __future__ import annotations

from dazi.core.config import DATA_DIR
from dazi.core.memory import MemoryStore
from dazi.core.task_store import TaskStore
from dazi.core.background import BackgroundTaskManager
from dazi.core.settings import SettingsManager
from dazi.core.cost_tracker import CostTracker
from dazi.core.mcp_client import MCPManager
from dazi.core.skills import SkillRegistry
from dazi.core.team import TeamManager
from dazi.core.teammate import TeammateRunner
from dazi.core.mailbox import Mailbox
from dazi.core.permission_bridge import PermissionBridge
from dazi.core.proactive import ProactiveManager
from dazi.core.coordinator import AutonomousTeammate
from dazi.core.worktree import WorktreeManager

# ── Path constants ──
MEMORY_DIR = DATA_DIR / "memory"
TASKS_DIR = DATA_DIR / "tasks"
BACKGROUND_DIR = DATA_DIR / "background"
TEAMS_DIR = DATA_DIR / "teams"
PLAN_DIR = DATA_DIR / "plans"
PLAN_FILE = PLAN_DIR / "plan.md"

# ── Singletons (dependency order) ──
# 1. No dependencies
memory_store = MemoryStore(MEMORY_DIR)
settings_manager = SettingsManager()
cost_tracker = CostTracker(DATA_DIR)
mcp_manager = MCPManager()
skill_registry = SkillRegistry()
proactive_manager = ProactiveManager()

# 2. Task store (default, no deps)
task_store = TaskStore(TASKS_DIR, list_id="default")

# 3. Background manager (no deps)
background_manager = BackgroundTaskManager(BACKGROUND_DIR)

# 4. Team + teammate (no deps)
team_manager = TeamManager()
teammate_runner = TeammateRunner()

# 5. Mailbox (no deps)
mailbox = Mailbox()

# 6. Permission bridge (depends on mailbox)
permission_bridge = PermissionBridge(mailbox=mailbox)

# 7. Autonomous teammate
autonomous_teammate = AutonomousTeammate()

# 8. Worktree manager (no deps)
worktree_manager = WorktreeManager()

# ── Mutable context (set by main.py) ──
active_team_name: str | None = None
current_agent_name: str | None = None
