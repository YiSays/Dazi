"""Team management — team config, member registry, file-based persistence.

KEY CONCEPTS:
  1. Team config is a JSON file at .dazi/teams/{name}/config.json (under project root)
  2. Task directory per team at .dazi/tasks/{name}/ (under project root)
  3. Team members tracked with agent_id format: "member-name@team-name"
  4. Team names are sanitized (non-alphanumeric → hyphens)
  5. delete_team fails if any active members exist
"""

from __future__ import annotations

import dataclasses
import json
import re
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from dazi.core.base import DaziTool, ToolSafety
from dazi.core.config import DATA_DIR


# ─────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────
TEAM_LEAD_NAME = "team-lead"
TEAM_CONFIG_FILENAME = "config.json"


# ─────────────────────────────────────────────────────────
# TEAM MEMBER
# ─────────────────────────────────────────────────────────
@dataclass
class TeamMember:
    """A member of a team.

    Each member has a unique agent_id in format "name@team".
    """
    name: str                                    # "frontend", "backend"
    agent_id: str                                # "frontend@web-dev"
    agent_type: str = "general-purpose"          # agent type for tool scoping
    status: str = "active"                       # "active" | "idle" | "completed"
    joined_at: str = field(default_factory=lambda: datetime.now().isoformat())


# ─────────────────────────────────────────────────────────
# TEAM CONFIG
# ─────────────────────────────────────────────────────────
@dataclass
class TeamConfig:
    """Team configuration with member list.

    Stored as JSON at .dazi/teams/{name}/config.json (under project root).
    """
    name: str
    description: str = ""
    members: list[TeamMember] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TeamConfig:
        """Deserialize from dict (loaded from JSON)."""
        members = [TeamMember(**m) for m in data.get("members", [])]
        return cls(
            name=data["name"],
            description=data.get("description", ""),
            members=members,
            created_at=data.get("created_at", ""),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict (for JSON storage)."""
        return {
            "name": self.name,
            "description": self.description,
            "members": [dataclasses.asdict(m) for m in self.members],
            "created_at": self.created_at,
        }


# ─────────────────────────────────────────────────────────
# TEAM MANAGER
# ─────────────────────────────────────────────────────────
class TeamError(Exception):
    """Raised for team operation errors."""
    pass


class TeamManager:
    """Manages team lifecycle: create, get, list, add/remove members, delete.

    Config stored at: <base_dir>/teams/<sanitized_name>/config.json
    Task directory:   <base_dir>/tasks/<sanitized_name>/
    """

    def __init__(self, base_dir: Path | None = None) -> None:
        """Initialize TeamManager.

        Args:
            base_dir: Root directory for team data. Defaults to DATA_DIR (.dazi/ under project root).
        """
        self.base_dir = base_dir or DATA_DIR
        self.teams_dir = self.base_dir / "teams"
        self.tasks_base_dir = self.base_dir / "tasks"

    @staticmethod
    def _sanitize_name(name: str) -> str:
        """Sanitize team name for filesystem use.

        Replace non-alphanumeric characters with hyphens, lowercase,
        strip leading/trailing hyphens, collapse consecutive hyphens.
        """
        sanitized = re.sub(r"[^a-z0-9]", "-", name.lower())
        sanitized = re.sub(r"-+", "-", sanitized)
        sanitized = sanitized.strip("-")
        return sanitized or "unnamed-team"

    # ── Path Helpers ──

    def _config_path(self, team_name: str) -> Path:
        """Get path to team config.json."""
        return self.teams_dir / self._sanitize_name(team_name) / TEAM_CONFIG_FILENAME

    def _task_dir(self, team_name: str) -> Path:
        """Get path to team task directory."""
        return self.tasks_base_dir / self._sanitize_name(team_name)

    def _team_dir(self, team_name: str) -> Path:
        """Get path to team directory."""
        return self.teams_dir / self._sanitize_name(team_name)

    # ── CRUD Operations ──

    def create_team(self, name: str, description: str = "") -> TeamConfig:
        """Create a new team with config file and task directory.

        Args:
            name: Team name (will be sanitized for filesystem).
            description: Optional team description.

        Returns:
            The created TeamConfig.

        Raises:
            TeamError: If team already exists.
        """
        sanitized = self._sanitize_name(name)

        if self.team_exists(name):
            raise TeamError(f"Team '{name}' already exists (sanitized: {sanitized})")

        team_dir = self.teams_dir / sanitized
        task_dir = self.tasks_base_dir / sanitized

        # Create directories
        team_dir.mkdir(parents=True, exist_ok=True)
        task_dir.mkdir(parents=True, exist_ok=True)

        # Create team config
        config = TeamConfig(
            name=name,
            description=description,
            members=[],
            created_at=datetime.now().isoformat(),
        )

        # Write config file
        config_path = team_dir / TEAM_CONFIG_FILENAME
        config_path.write_text(
            json.dumps(config.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        return config

    def get_team(self, name: str) -> TeamConfig | None:
        """Get a team by name.

        """
        config_path = self._config_path(name)
        if not config_path.exists():
            return None

        try:
            data = json.loads(config_path.read_text(encoding="utf-8"))
            return TeamConfig.from_dict(data)
        except (json.JSONDecodeError, KeyError) as e:
            raise TeamError(f"Corrupted team config for '{name}': {e}")

    def list_teams(self) -> list[TeamConfig]:
        """List all teams.

        """
        teams: list[TeamConfig] = []

        if not self.teams_dir.exists():
            return teams

        for team_dir in sorted(self.teams_dir.iterdir()):
            if team_dir.is_dir():
                config_path = team_dir / TEAM_CONFIG_FILENAME
                if config_path.exists():
                    try:
                        data = json.loads(config_path.read_text(encoding="utf-8"))
                        teams.append(TeamConfig.from_dict(data))
                    except (json.JSONDecodeError, KeyError):
                        continue  # Skip corrupted configs

        return teams

    def add_member(self, team_name: str, member: TeamMember) -> TeamMember:
        """Add a member to a team.

        Raises:
            TeamError: If team doesn't exist or member already in team.
        """
        config = self.get_team(team_name)
        if config is None:
            raise TeamError(f"Team '{team_name}' not found")

        # Check for duplicate agent_id
        for existing in config.members:
            if existing.agent_id == member.agent_id:
                raise TeamError(f"Member '{member.agent_id}' already in team '{team_name}'")

        config.members.append(member)
        self._write_config(config)
        return member

    def remove_member(self, team_name: str, agent_id: str) -> bool:
        """Remove a member from a team.

        """
        config = self.get_team(team_name)
        if config is None:
            return False

        original_len = len(config.members)
        config.members = [m for m in config.members if m.agent_id != agent_id]

        if len(config.members) == original_len:
            return False  # Member not found

        self._write_config(config)
        return True

    def get_member(self, team_name: str, agent_id: str) -> TeamMember | None:
        """Get a specific member from a team."""
        config = self.get_team(team_name)
        if config is None:
            return None

        for m in config.members:
            if m.agent_id == agent_id:
                return m
        return None

    def update_member_status(self, team_name: str, agent_id: str, status: str) -> bool:
        """Update a member's status.

        """
        config = self.get_team(team_name)
        if config is None:
            return False

        for m in config.members:
            if m.agent_id == agent_id:
                m.status = status
                self._write_config(config)
                return True

        return False  # Member not found

    def delete_team(self, team_name: str) -> bool:
        """Delete a team (config + task directory).

        Safety: Fails if any active members exist.

        Raises:
            TeamError: If team has active members.
        """
        config = self.get_team(team_name)
        if config is None:
            return False

        # Check for active members
        active_members = [m for m in config.members if m.status not in ("completed", "idle")]
        if active_members:
            names = ", ".join(m.name for m in active_members)
            raise TeamError(
                f"Cannot delete team '{team_name}': {len(active_members)} active member(s): {names}. "
                f"Shut down all teammates before deleting."
            )

        sanitized = self._sanitize_name(team_name)
        team_dir = self.teams_dir / sanitized
        task_dir = self.tasks_base_dir / sanitized

        # Remove directories
        if team_dir.exists():
            shutil.rmtree(team_dir)
        if task_dir.exists():
            shutil.rmtree(task_dir)

        return True

    def team_exists(self, name: str) -> bool:
        """Check if a team exists."""
        return self._config_path(name).exists()

    # ── Internal ──

    def _write_config(self, config: TeamConfig) -> None:
        """Write team config to disk.

        """
        config_path = self._config_path(config.name)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(
            json.dumps(config.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )


# ─────────────────────────────────────────────────────────
# TEAM TOOLS
# ─────────────────────────────────────────────────────────

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field


class CreateTeamInput(BaseModel):
    name: str = Field(description="Name for the team (e.g., 'web-dev', 'research')")
    description: str = Field(default="", description="Optional team description/purpose")


async def create_team_func(name: str, description: str = "") -> str:
    """Create a new team with a shared task board."""
    from dazi.core._singletons import team_manager

    try:
        team = team_manager.create_team(name, description)
        sanitized = team_manager._sanitize_name(name)
        return (
            f"Team created: {team.name}\n"
            f"Description: {team.description}\n"
            f"Config: {team_manager._config_path(name)}\n"
            f"Task directory: {team_manager._task_dir(name)}\n"
            f"Use show_team to view team details."
        )
    except TeamError as e:
        return f"Error creating team '{name}': {e}"


create_team_tool = StructuredTool.from_function(
    func=lambda **kwargs: "",
    coroutine=create_team_func,
    name="create_team",
    description="Create a new agent team with a shared task board.",
    args_schema=CreateTeamInput,
)
create_team_meta = DaziTool(name="create_team", description="Create a new agent team.", safety=ToolSafety.WRITE)


class DeleteTeamInput(BaseModel):
    name: str = Field(description="Name of the team to delete")


async def delete_team_func(name: str) -> str:
    """Delete a team and its task directory."""
    from dazi.core._singletons import team_manager

    try:
        if not team_manager.team_exists(name):
            return f"Team '{name}' not found."
        if team_manager.delete_team(name):
            return f"Team '{name}' deleted successfully."
        return f"Failed to delete team '{name}'."
    except TeamError as e:
        return f"Error deleting team '{name}': {e}"


delete_team_tool = StructuredTool.from_function(
    func=lambda **kwargs: "",
    coroutine=delete_team_func,
    name="delete_team",
    description="Delete a team and clean up its config and task directory. Fails if any active members exist.",
    args_schema=DeleteTeamInput,
)
delete_team_meta = DaziTool(name="delete_team", description="Delete a team.", safety=ToolSafety.DESTRUCTIVE)


class ListTeamsInput(BaseModel):
    pass


async def list_teams_func() -> str:
    """List all teams with member counts and status."""
    from dazi.core._singletons import team_manager

    teams = team_manager.list_teams()
    if not teams:
        return "No teams exist. Use create_team to create one."

    lines = ["Teams:", ""]
    for t in teams:
        active = sum(1 for m in t.members if m.status == "active")
        idle = sum(1 for m in t.members if m.status == "idle")
        completed = sum(1 for m in t.members if m.status == "completed")
        lines.append(f"  * {t.name} — {t.description or '(no description)'}")
        lines.append(f"    Members: {len(t.members)} total ({active} active, {idle} idle, {completed} completed)")
        lines.append(f"    Created: {t.created_at[:10] if t.created_at else 'unknown'}")
        lines.append("")

    return "\n".join(lines)


list_teams_tool = StructuredTool.from_function(
    func=lambda **kwargs: "",
    coroutine=list_teams_func,
    name="list_teams",
    description="List all teams with their member counts and status.",
    args_schema=ListTeamsInput,
)
list_teams_meta = DaziTool(name="list_teams", description="List all teams.", safety=ToolSafety.SAFE)


class ShowTeamInput(BaseModel):
    name: str = Field(description="Name of the team to show")


async def show_team_func(name: str) -> str:
    """Show detailed team info including members and their status."""
    from dazi.core._singletons import team_manager

    team = team_manager.get_team(name)
    if team is None:
        return f"Team '{name}' not found."

    lines = [
        f"Team: {team.name}",
        f"Description: {team.description or '(no description)'}",
        f"Created: {team.created_at}",
        f"Members ({len(team.members)}):",
    ]

    if team.members:
        for m in team.members:
            status_icon = {"active": "+", "idle": "=", "completed": "✓"}.get(m.status, "?")
            lines.append(f"  [{status_icon}] {m.name} ({m.agent_id}) — {m.status}")
    else:
        lines.append("  (no members)")

    lines.append(f"\nConfig: {team_manager._config_path(name)}")
    lines.append(f"Tasks:  {team_manager._task_dir(name)}")

    return "\n".join(lines)


show_team_tool = StructuredTool.from_function(
    func=lambda **kwargs: "",
    coroutine=show_team_func,
    name="show_team",
    description="Show detailed information about a team including all members and their current status.",
    args_schema=ShowTeamInput,
)
show_team_meta = DaziTool(name="show_team", description="Show detailed team info.", safety=ToolSafety.SAFE)
