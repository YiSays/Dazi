"""File-based mailbox system for inter-agent messaging.

KEY CONCEPTS:
  1. Each agent has a JSON inbox file at
     .dazi/teams/{team}/inboxes/{agent}.json (under project root)
  2. Messages are stored as a JSON array of message dicts
  3. File locking (fcntl.flock) prevents concurrent write corruption
  4. Broadcast (*) writes to every team member's inbox except sender
  5. Messages have types: text, shutdown_request, shutdown_response,
    permission_request, permission_response, plan_approval_request,
    plan_approval_response, idle_notification
"""

from __future__ import annotations

import fcntl
import json
import re
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from dazi.base import DaziTool, ToolSafety


@dataclass
class Message:
    """A single mailbox message.

    Stored in .dazi/teams/{team}/inboxes/{agent}.json (under project root).
    """

    id: str
    from_agent: str
    to_agent: str  # Agent name or "*" for broadcast
    text: str
    timestamp: str  # ISO 8601
    summary: str = ""
    msg_type: str = "text"  # text, shutdown_request, shutdown_response, etc.
    read: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-compatible dict."""
        return {
            "id": self.id,
            "from_agent": self.from_agent,
            "to_agent": self.to_agent,
            "text": self.text,
            "timestamp": self.timestamp,
            "summary": self.summary,
            "msg_type": self.msg_type,
            "read": self.read,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Message:
        """Deserialize from dict (with backward compatibility for missing fields)."""
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            from_agent=data.get("from_agent", ""),
            to_agent=data.get("to_agent", ""),
            text=data.get("text", ""),
            timestamp=data.get("timestamp", datetime.now(UTC).isoformat()),
            summary=data.get("summary", ""),
            msg_type=data.get("msg_type", "text"),
            read=data.get("read", False),
            metadata=data.get("metadata", {}),
        )


class Mailbox:
    """File-based per-agent inbox with POSIX file locking.

    Storage layout:
      <base_dir>/teams/<sanitized_team>/inboxes/<agent_name>.json

    Each JSON file is an array of Message dicts.
    Uses fcntl.flock() for exclusive access on POSIX systems.

    Args:
        base_dir: Base directory for all mailbox data.
                 Defaults to DATA_DIR (.dazi/ under project root).
    """

    def __init__(self, base_dir: Path | None = None) -> None:
        from dazi.config import DATA_DIR

        self.base_dir = base_dir or DATA_DIR

    # ── Path helpers ──

    @staticmethod
    def _sanitize_name(name: str) -> str:
        """Sanitize team/agent name for filesystem safety."""
        sanitized = re.sub(r"[^a-zA-Z0-9]", "-", name.lower())
        sanitized = re.sub(r"-+", "-", sanitized)
        return sanitized.strip("-")

    def _inbox_path(self, team_name: str, agent_name: str) -> Path:
        """Get path to an agent's inbox file."""
        team_dir = self.base_dir / "teams" / self._sanitize_name(team_name)
        return team_dir / "inboxes" / f"{self._sanitize_name(agent_name)}.json"

    def _ensure_inbox_dir(self, team_name: str) -> Path:
        """Ensure the inboxes directory exists for a team.

        Returns the inboxes directory path.
        """
        inbox_dir = self.base_dir / "teams" / self._sanitize_name(team_name) / "inboxes"
        inbox_dir.mkdir(parents=True, exist_ok=True)
        return inbox_dir

    # ── File I/O with locking ──

    def _read_inbox(self, path: Path) -> list[Message]:
        """Read all messages from an inbox file (with shared read lock)."""
        if not path.exists():
            return []

        with path.open() as f:
            fcntl.flock(f, fcntl.LOCK_SH)
            try:
                data = json.load(f)
                return [Message.from_dict(m) for m in data]
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)

    def _write_inbox(self, path: Path, messages: list[Message]) -> None:
        """Write messages to an inbox file (with exclusive write lock)."""
        path.parent.mkdir(parents=True, exist_ok=True)
        data = [m.to_dict() for m in messages]

        with path.open("w") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            try:
                json.dump(data, f, indent=2)
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)

    # ── Summary derivation ──

    @staticmethod
    def _derive_summary(text: str) -> str:
        """Generate a 5-10 word summary from message text.

        Truncates to first 10 words, max ~60 chars.
        """
        words = text.split()[:10]
        summary = " ".join(words)
        if len(summary) > 60:
            summary = summary[:57] + "..."
        return summary

    # ── Public API ──

    async def send(
        self,
        team_name: str,
        message: Message,
        team_members: list[str] | None = None,
    ) -> list[str]:
        """Send a message to one or more agents.

        If message.to_agent == "*": broadcast to all team members (except sender).
        Otherwise: deliver to a single agent's inbox.

        Args:
            team_name: Team the message belongs to.
            message: The Message to send.
            team_members: List of member names for broadcast resolution.
                         If None, reads team config to discover members.

        Returns:
            List of agent names that received the message.

        """
        if not message.summary and message.text:
            message.summary = self._derive_summary(message.text)
        if not message.id:
            message.id = str(uuid.uuid4())
        if not message.timestamp:
            message.timestamp = datetime.now(UTC).isoformat()

        recipients: list[str] = []

        if message.to_agent == "*":
            # Broadcast: resolve member list
            if team_members is None:
                team_members = self._get_team_members(team_name)

            for member_name in team_members:
                if member_name != message.from_agent:
                    recipients.append(member_name)
                    inbox_path = self._inbox_path(team_name, member_name)
                    existing = self._read_inbox(inbox_path)
                    existing.append(message)
                    self._write_inbox(inbox_path, existing)
        else:
            # Direct message
            recipients.append(message.to_agent)
            inbox_path = self._inbox_path(team_name, message.to_agent)
            existing = self._read_inbox(inbox_path)
            existing.append(message)
            self._write_inbox(inbox_path, existing)

        return recipients

    async def receive(
        self,
        team_name: str,
        agent_name: str,
        unread_only: bool = True,
        limit: int = 20,
    ) -> list[Message]:
        """Read messages from an agent's inbox.

        Args:
            team_name: Team the agent belongs to.
            agent_name: Agent name (not agent_id).
            unread_only: If True, only return unread messages.
            limit: Max messages to return.

        Returns:
            List of Message objects, newest first.

        """
        inbox_path = self._inbox_path(team_name, agent_name)
        messages = self._read_inbox(inbox_path)

        if unread_only:
            messages = [m for m in messages if not m.read]

        # Newest first
        messages.reverse()
        return messages[:limit]

    async def mark_read(
        self,
        team_name: str,
        agent_name: str,
        message_ids: list[str] | None = None,
    ) -> int:
        """Mark messages as read.

        Args:
            team_name: Team the agent belongs to.
            agent_name: Agent name.
            message_ids: Specific message IDs to mark. If None, mark ALL.

        Returns:
            Count of messages marked as read.

        """
        inbox_path = self._inbox_path(team_name, agent_name)
        messages = self._read_inbox(inbox_path)

        count = 0
        for msg in messages:
            if message_ids is None or msg.id in message_ids:
                if not msg.read:
                    msg.read = True
                    count += 1

        if count > 0:
            self._write_inbox(inbox_path, messages)

        return count

    async def purge(self, team_name: str, agent_name: str) -> int:
        """Delete all messages for an agent.

        Returns count of messages deleted. Used during team deletion cleanup.
        """
        inbox_path = self._inbox_path(team_name, agent_name)
        messages = self._read_inbox(inbox_path)
        count = len(messages)

        if count > 0 and inbox_path.exists():
            inbox_path.unlink()

        return count

    # ── Team member resolution ──

    def _get_team_members(self, team_name: str) -> list[str]:
        """Get all member names for a team by reading config.

        Falls back to scanning inbox directory if config is not available.
        """
        try:
            from dazi.team import TeamManager

            tm = TeamManager(base_dir=self.base_dir)
            team = tm.get_team(team_name)
            if team and team.members:
                return [m.name for m in team.members]
        except Exception:
            pass

        # Fallback: scan inbox directory for existing inboxes
        inbox_dir = self.base_dir / "teams" / self._sanitize_name(team_name) / "inboxes"
        if inbox_dir.exists():
            return [p.stem for p in inbox_dir.glob("*.json")]

        return []


# ─────────────────────────────────────────────────────────
# MESSAGING TOOLS
# ─────────────────────────────────────────────────────────


class SendMessageInput(BaseModel):
    to: str = Field(description="Recipient agent name or '*' for broadcast to all team members")
    message: str = Field(description="Message content to send")
    summary: str = Field(default="", description="Brief 5-10 word summary for preview")


async def send_message_func(to: str, message: str, summary: str = "") -> str:
    """Send a message to a teammate or broadcast to all."""
    from dazi._singletons import active_team_name, current_agent_name, mailbox, team_manager
    from dazi.protocols import create_text_message

    if not active_team_name:
        return "Error: no active team. Use /team <name> to activate a team first."

    if not current_agent_name:
        return "Error: agent identity not set. Cannot send messages."

    if to == current_agent_name:
        return f"Error: cannot send a message to yourself ({current_agent_name})."

    team_members = None
    if to == "*":
        team = team_manager.get_team(active_team_name)
        if team:
            team_members = [m.name for m in team.members]
        if not team_members:
            return f"Error: no team members found for '{active_team_name}'."

    msg = create_text_message(
        from_agent=current_agent_name, to_agent=to, text=message, summary=summary
    )

    recipients = await mailbox.send(
        team_name=active_team_name, message=msg, team_members=team_members
    )

    if to == "*":
        return f"Broadcast sent to {len(recipients)} teammate(s): {', '.join(recipients)}"
    else:
        if recipients:
            return f"Message sent to {to}."
        else:
            return f"Error: could not deliver message to '{to}'."


send_message_tool = StructuredTool.from_function(
    func=lambda **kwargs: "",
    coroutine=send_message_func,
    name="send_message",
    description="Send a message to a teammate or broadcast to all team members.",
    args_schema=SendMessageInput,
)
send_message_meta = DaziTool(
    name="send_message", description="Send a message to a teammate.", safety=ToolSafety.WRITE
)


class CheckInboxInput(BaseModel):
    unread_only: bool = Field(default=True, description="Only show unread messages")
    limit: int = Field(default=20, description="Max messages to return")


async def check_inbox_func(unread_only: bool = True, limit: int = 20) -> str:
    """Check the current agent's inbox for messages."""
    from dazi._singletons import active_team_name, current_agent_name, mailbox

    if not active_team_name:
        return "Error: no active team. Use /team <name> to activate a team first."

    if not current_agent_name:
        return "Error: agent identity not set."

    messages = await mailbox.receive(
        team_name=active_team_name,
        agent_name=current_agent_name,
        unread_only=unread_only,
        limit=limit,
    )

    if not messages:
        return "No unread messages." if unread_only else "No messages."

    msg_ids = [m.id for m in messages]
    await mailbox.mark_read(
        team_name=active_team_name, agent_name=current_agent_name, message_ids=msg_ids
    )

    lines = [f"Inbox ({len(messages)} message(s)):", ""]
    for msg in messages:
        type_tag = f"[{msg.msg_type}] " if msg.msg_type != "text" else ""
        time_short = msg.timestamp[:19] if msg.timestamp else "unknown"
        lines.append(f"  From: {msg.from_agent} | {time_short}")
        lines.append(f"  {type_tag}{msg.text[:200]}")
        if msg.metadata:
            meta_keys = list(msg.metadata.keys())
            if meta_keys:
                lines.append(f"  Metadata: {', '.join(meta_keys)}")
        lines.append("")

    return "\n".join(lines)


check_inbox_tool = StructuredTool.from_function(
    func=lambda **kwargs: "",
    coroutine=check_inbox_func,
    name="check_inbox",
    description="Check your inbox for new messages. Returns unread messages by default.",
    args_schema=CheckInboxInput,
)
check_inbox_meta = DaziTool(
    name="check_inbox", description="Check your inbox for messages.", safety=ToolSafety.SAFE
)


async def send_idle_notification(
    agent_name: str,
    team_name: str,
    completed_task_id: str | None = None,
    idle_reason: str = "no_pending_work",
    summary: str = "",
) -> None:
    """Send an idle notification to all teammates."""
    from dazi._singletons import mailbox, team_manager
    from dazi.protocols import create_idle_notification

    team = team_manager.get_team(team_name)
    team_members = [m.name for m in team.members] if team else None

    msg = create_idle_notification(
        from_agent=agent_name,
        completed_task_id=completed_task_id,
        idle_reason=idle_reason,
        summary=summary,
    )

    await mailbox.send(team_name=team_name, message=msg, team_members=team_members)
