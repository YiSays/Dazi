"""Tests for dazi/repl_teams.py — team state management and inbox REPL helpers."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from rich.console import Console

from tests.helpers.mock_singletons import patch_singletons


@pytest.fixture(autouse=True)
def _patch(monkeypatch, tmp_path: Path):
    patch_singletons(monkeypatch, tmp_path)


@pytest.fixture(autouse=True)
def _reset_module_globals():
    """Reset module-level globals before each test."""
    import dazi.repl_teams as mod

    mod.active_team_name = None
    mod.current_agent_name = None
    mod.team_task_store = None
    yield
    mod.active_team_name = None
    mod.current_agent_name = None
    mod.team_task_store = None


def _make_team(
    name: str = "test-team",
    description: str = "A test team",
    members: list | None = None,
    created_at: str | None = None,
):
    """Create a TeamConfig-like object."""
    from dazi.team import TeamConfig

    return TeamConfig(
        name=name,
        description=description,
        members=members or [],
        created_at=created_at or datetime.now().isoformat(),
    )


def _make_member(
    name: str = "agent1",
    agent_id: str = "agent1@test-team",
    agent_type: str = "general-purpose",
    status: str = "active",
):
    """Create a TeamMember object."""
    from dazi.team import TeamMember

    return TeamMember(
        name=name,
        agent_id=agent_id,
        agent_type=agent_type,
        status=status,
    )


def _make_message(
    msg_id: str = "msg-1",
    from_agent: str = "agent1",
    to_agent: str = "team-lead",
    text: str = "Hello",
    timestamp: str = "2026-01-01T12:00:00",
    summary: str = "Hello summary",
    msg_type: str = "text",
):
    """Create a Message object."""
    from dazi.mailbox import Message

    return Message(
        id=msg_id,
        from_agent=from_agent,
        to_agent=to_agent,
        text=text,
        timestamp=timestamp,
        summary=summary,
        msg_type=msg_type,
    )


# ─────────────────────────────────────────────────────────
# _require_team
# ─────────────────────────────────────────────────────────


class TestRequireTeam:
    def test_no_active_team(self, monkeypatch):
        import dazi.repl_teams as mod

        console = Console(file=MagicMock(), force_terminal=False)
        monkeypatch.setattr(mod, "console", console)
        mod.active_team_name = None
        mod.current_agent_name = "team-lead"

        result = mod._require_team()
        assert result is False

    def test_no_agent_identity(self, monkeypatch):
        import dazi.repl_teams as mod

        console = Console(file=MagicMock(), force_terminal=False)
        monkeypatch.setattr(mod, "console", console)
        mod.active_team_name = "test-team"
        mod.current_agent_name = None

        result = mod._require_team()
        assert result is False

    def test_both_set(self):
        import dazi.repl_teams as mod

        mod.active_team_name = "test-team"
        mod.current_agent_name = "team-lead"

        result = mod._require_team()
        assert result is True


# ─────────────────────────────────────────────────────────
# show_teams_table
# ─────────────────────────────────────────────────────────


class TestShowTeamsTable:
    def test_no_teams(self, monkeypatch):
        import dazi.repl_teams as mod
        from dazi._singletons import team_manager

        console = Console(file=MagicMock(), force_terminal=False)
        monkeypatch.setattr(mod, "console", console)
        monkeypatch.setattr(mod, "team_manager", team_manager)
        monkeypatch.setattr(team_manager, "list_teams", MagicMock(return_value=[]))

        mod.show_teams_table()

    def test_with_teams(self, monkeypatch):
        import dazi.repl_teams as mod
        from dazi._singletons import team_manager

        console = Console(file=MagicMock(), force_terminal=False)
        monkeypatch.setattr(mod, "console", console)
        monkeypatch.setattr(mod, "team_manager", team_manager)

        m1 = _make_member(name="agent1", status="active")
        m2 = _make_member(name="agent2", status="idle")
        team = _make_team(
            name="web-dev",
            description="A longer description that should be truncated because it is really long",
            members=[m1, m2],
        )
        monkeypatch.setattr(team_manager, "list_teams", MagicMock(return_value=[team]))

        mod.show_teams_table()

    def test_long_description_truncation(self, monkeypatch):
        import dazi.repl_teams as mod
        from dazi._singletons import team_manager

        console = Console(file=MagicMock(), force_terminal=False)
        monkeypatch.setattr(mod, "console", console)
        monkeypatch.setattr(mod, "team_manager", team_manager)

        team = _make_team(description="x" * 100)
        monkeypatch.setattr(team_manager, "list_teams", MagicMock(return_value=[team]))

        mod.show_teams_table()

    def test_team_with_various_member_statuses(self, monkeypatch):
        import dazi.repl_teams as mod
        from dazi._singletons import team_manager

        console = Console(file=MagicMock(), force_terminal=False)
        monkeypatch.setattr(mod, "console", console)
        monkeypatch.setattr(mod, "team_manager", team_manager)

        members = [
            _make_member(name="a", status="active"),
            _make_member(name="b", status="idle"),
            _make_member(name="c", status="completed"),
        ]
        team = _make_team(members=members)
        monkeypatch.setattr(team_manager, "list_teams", MagicMock(return_value=[team]))

        mod.show_teams_table()


# ─────────────────────────────────────────────────────────
# show_team_detail
# ─────────────────────────────────────────────────────────


class TestShowTeamDetail:
    def test_team_not_found(self, monkeypatch):
        import dazi.repl_teams as mod
        from dazi._singletons import team_manager

        console = Console(file=MagicMock(), force_terminal=False)
        monkeypatch.setattr(mod, "console", console)
        monkeypatch.setattr(mod, "team_manager", team_manager)
        monkeypatch.setattr(team_manager, "get_team", MagicMock(return_value=None))

        mod.show_team_detail("nonexistent")

    def test_team_found_with_members(self, monkeypatch):
        import dazi.repl_teams as mod
        from dazi._singletons import team_manager

        console = Console(file=MagicMock(), force_terminal=False)
        monkeypatch.setattr(mod, "console", console)
        monkeypatch.setattr(mod, "team_manager", team_manager)

        m1 = _make_member(name="agent1", status="active")
        m2 = _make_member(name="agent2", status="idle")
        m3 = _make_member(name="agent3", status="completed")
        team = _make_team(members=[m1, m2, m3])
        monkeypatch.setattr(team_manager, "get_team", MagicMock(return_value=team))
        monkeypatch.setattr(
            team_manager, "_config_path", MagicMock(return_value=Path("/fake/config"))
        )
        monkeypatch.setattr(team_manager, "_task_dir", MagicMock(return_value=Path("/fake/tasks")))

        mod.show_team_detail("test-team")

    def test_team_found_no_members(self, monkeypatch):
        import dazi.repl_teams as mod
        from dazi._singletons import team_manager

        console = Console(file=MagicMock(), force_terminal=False)
        monkeypatch.setattr(mod, "console", console)
        monkeypatch.setattr(mod, "team_manager", team_manager)

        team = _make_team(members=[])
        monkeypatch.setattr(team_manager, "get_team", MagicMock(return_value=team))
        monkeypatch.setattr(
            team_manager, "_config_path", MagicMock(return_value=Path("/fake/config"))
        )
        monkeypatch.setattr(team_manager, "_task_dir", MagicMock(return_value=Path("/fake/tasks")))

        mod.show_team_detail("test-team")

    def test_team_with_all_known_statuses(self, monkeypatch):
        """Test that each known status (active, idle, completed) renders without error."""
        import dazi.repl_teams as mod
        from dazi._singletons import team_manager

        console = Console(file=MagicMock(), force_terminal=False)
        monkeypatch.setattr(mod, "console", console)
        monkeypatch.setattr(mod, "team_manager", team_manager)

        # Test each known status individually
        for status in ("active", "idle", "completed"):
            m = _make_member(name=f"agent-{status}", status=status)
            team = _make_team(members=[m])
            monkeypatch.setattr(team_manager, "get_team", MagicMock(return_value=team))
            monkeypatch.setattr(
                team_manager, "_config_path", MagicMock(return_value=Path("/fake/config"))
            )
            monkeypatch.setattr(
                team_manager, "_task_dir", MagicMock(return_value=Path("/fake/tasks"))
            )
            mod.show_team_detail("test-team")


# ─────────────────────────────────────────────────────────
# activate_team
# ─────────────────────────────────────────────────────────


class TestActivateTeam:
    def test_team_not_found(self, monkeypatch):
        import dazi.repl_teams as mod
        from dazi._singletons import team_manager

        console = Console(file=MagicMock(), force_terminal=False)
        monkeypatch.setattr(mod, "console", console)
        monkeypatch.setattr(mod, "team_manager", team_manager)
        monkeypatch.setattr(team_manager, "get_team", MagicMock(return_value=None))

        mod.activate_team("nonexistent")

        assert mod.active_team_name is None

    def test_successful_activation_no_members(self, monkeypatch):
        import dazi._singletons as _singletons_mod
        import dazi.repl_teams as mod
        from dazi._singletons import team_manager
        from dazi.team import TEAM_LEAD_NAME

        console = Console(file=MagicMock(), force_terminal=False)
        monkeypatch.setattr(mod, "console", console)
        monkeypatch.setattr(mod, "team_manager", team_manager)

        team = _make_team(name="web-dev", members=[])
        monkeypatch.setattr(team_manager, "get_team", MagicMock(return_value=team))
        monkeypatch.setattr(team_manager, "_sanitize_name", MagicMock(return_value="web-dev"))
        monkeypatch.setattr(team_manager, "teams_dir", Path("/fake/.dazi/teams"))

        fake_tasks_dir = Path("/fake/.dazi/tasks")
        monkeypatch.setattr(mod, "TASKS_DIR", fake_tasks_dir)

        mock_mailbox = MagicMock()
        monkeypatch.setattr(mod, "mailbox", mock_mailbox)

        with patch("dazi.repl_teams.TaskStore") as MockTaskStore:
            mock_store = MagicMock()
            MockTaskStore.return_value = mock_store
            mod.activate_team("web-dev")

        assert mod.active_team_name == "web-dev"
        assert mod.current_agent_name == TEAM_LEAD_NAME
        assert _singletons_mod.active_team_name == "web-dev"
        assert _singletons_mod.current_agent_name == TEAM_LEAD_NAME
        mock_mailbox._ensure_inbox_dir.assert_called_once_with("web-dev")
        MockTaskStore.assert_called_once_with(fake_tasks_dir, list_id="web-dev")
        assert mod.team_task_store is mock_store

    def test_successful_activation_with_members(self, monkeypatch):
        import dazi.repl_teams as mod
        from dazi._singletons import team_manager

        console = Console(file=MagicMock(), force_terminal=False)
        monkeypatch.setattr(mod, "console", console)
        monkeypatch.setattr(mod, "team_manager", team_manager)

        m1 = _make_member(name="frontend")
        m2 = _make_member(name="backend")
        team = _make_team(name="web-dev", members=[m1, m2])
        monkeypatch.setattr(team_manager, "get_team", MagicMock(return_value=team))
        monkeypatch.setattr(team_manager, "_sanitize_name", MagicMock(return_value="web-dev"))
        monkeypatch.setattr(team_manager, "teams_dir", Path("/fake/.dazi/teams"))

        mock_mailbox = MagicMock()
        monkeypatch.setattr(mod, "mailbox", mock_mailbox)

        with patch("dazi.repl_teams.TaskStore"):
            mod.activate_team("web-dev")

        assert mod.active_team_name == "web-dev"


# ─────────────────────────────────────────────────────────
# deactivate_team
# ─────────────────────────────────────────────────────────


class TestDeactivateTeam:
    def test_deactivate_when_active(self, monkeypatch):
        import dazi._singletons as _singletons_mod
        import dazi.repl_teams as mod

        console = Console(file=MagicMock(), force_terminal=False)
        monkeypatch.setattr(mod, "console", console)

        mod.active_team_name = "test-team"
        mod.current_agent_name = "team-lead"
        mod.team_task_store = MagicMock()

        mod.deactivate_team()

        assert mod.active_team_name is None
        assert mod.current_agent_name is None
        assert mod.team_task_store is None
        assert _singletons_mod.active_team_name is None
        assert _singletons_mod.current_agent_name is None

    def test_deactivate_when_not_active(self, monkeypatch):
        import dazi._singletons as _singletons_mod
        import dazi.repl_teams as mod

        console = Console(file=MagicMock(), force_terminal=False)
        monkeypatch.setattr(mod, "console", console)

        mod.active_team_name = None
        mod.current_agent_name = None
        mod.team_task_store = None

        mod.deactivate_team()

        assert mod.active_team_name is None
        assert _singletons_mod.active_team_name is None


# ─────────────────────────────────────────────────────────
# show_inbox
# ─────────────────────────────────────────────────────────


class TestShowInbox:
    @pytest.mark.asyncio
    async def test_no_active_team(self, monkeypatch):
        import dazi.repl_teams as mod

        console = Console(file=MagicMock(), force_terminal=False)
        monkeypatch.setattr(mod, "console", console)
        mod.active_team_name = None

        await mod.show_inbox()

    @pytest.mark.asyncio
    async def test_no_agent_identity(self, monkeypatch):
        import dazi.repl_teams as mod

        console = Console(file=MagicMock(), force_terminal=False)
        monkeypatch.setattr(mod, "console", console)
        mod.active_team_name = "test-team"
        mod.current_agent_name = None

        await mod.show_inbox()

    @pytest.mark.asyncio
    async def test_no_unread_messages(self, monkeypatch):
        import dazi.repl_teams as mod

        console = Console(file=MagicMock(), force_terminal=False)
        monkeypatch.setattr(mod, "console", console)

        mock_mailbox = AsyncMock()
        mock_mailbox.receive = AsyncMock(return_value=[])
        monkeypatch.setattr(mod, "mailbox", mock_mailbox)

        mod.active_team_name = "test-team"
        mod.current_agent_name = "team-lead"

        await mod.show_inbox()

        mock_mailbox.receive.assert_called_once_with(
            team_name="test-team",
            agent_name="team-lead",
            unread_only=True,
            limit=20,
        )

    @pytest.mark.asyncio
    async def test_with_messages(self, monkeypatch):
        import dazi.repl_teams as mod

        console = Console(file=MagicMock(), force_terminal=False)
        monkeypatch.setattr(mod, "console", console)

        msg1 = _make_message(msg_id="m1", from_agent="agent1", text="Hello world", summary="Hi")
        msg2 = _make_message(msg_id="m2", from_agent="agent2", msg_type="shutdown_request")

        mock_mailbox = AsyncMock()
        mock_mailbox.receive = AsyncMock(return_value=[msg1, msg2])
        mock_mailbox.mark_read = AsyncMock()
        monkeypatch.setattr(mod, "mailbox", mock_mailbox)

        mod.active_team_name = "test-team"
        mod.current_agent_name = "team-lead"

        await mod.show_inbox()

        mock_mailbox.mark_read.assert_called_once_with(
            team_name="test-team",
            agent_name="team-lead",
            message_ids=["m1", "m2"],
        )

    @pytest.mark.asyncio
    async def test_peek_at_other_agent(self, monkeypatch):
        import dazi.repl_teams as mod

        console = Console(file=MagicMock(), force_terminal=False)
        monkeypatch.setattr(mod, "console", console)

        msg = _make_message(from_agent="agent2")
        mock_mailbox = AsyncMock()
        mock_mailbox.receive = AsyncMock(return_value=[msg])
        mock_mailbox.mark_read = AsyncMock()
        monkeypatch.setattr(mod, "mailbox", mock_mailbox)

        mod.active_team_name = "test-team"
        mod.current_agent_name = "team-lead"

        await mod.show_inbox(agent_name="agent1")

        mock_mailbox.receive.assert_called_once_with(
            team_name="test-team",
            agent_name="agent1",
            unread_only=True,
            limit=20,
        )

    @pytest.mark.asyncio
    async def test_message_no_summary(self, monkeypatch):
        import dazi.repl_teams as mod

        console = Console(file=MagicMock(), force_terminal=False)
        monkeypatch.setattr(mod, "console", console)

        msg = _make_message(
            summary="", text="This is the full text that should be used as fallback"
        )
        mock_mailbox = AsyncMock()
        mock_mailbox.receive = AsyncMock(return_value=[msg])
        mock_mailbox.mark_read = AsyncMock()
        monkeypatch.setattr(mod, "mailbox", mock_mailbox)

        mod.active_team_name = "test-team"
        mod.current_agent_name = "team-lead"

        await mod.show_inbox()

    @pytest.mark.asyncio
    async def test_message_no_timestamp(self, monkeypatch):
        import dazi.repl_teams as mod

        console = Console(file=MagicMock(), force_terminal=False)
        monkeypatch.setattr(mod, "console", console)

        msg = _make_message(timestamp="")
        mock_mailbox = AsyncMock()
        mock_mailbox.receive = AsyncMock(return_value=[msg])
        mock_mailbox.mark_read = AsyncMock()
        monkeypatch.setattr(mod, "mailbox", mock_mailbox)

        mod.active_team_name = "test-team"
        mod.current_agent_name = "team-lead"

        await mod.show_inbox()


# ─────────────────────────────────────────────────────────
# send_repl_message
# ─────────────────────────────────────────────────────────


class TestSendReplMessage:
    @pytest.mark.asyncio
    async def test_no_active_team(self):
        import dazi.repl_teams as mod

        mod.active_team_name = None
        mod.current_agent_name = None

        await mod.send_repl_message("agent1", "hello")

    @pytest.mark.asyncio
    async def test_send_to_self(self):
        import dazi.repl_teams as mod

        mod.active_team_name = "test-team"
        mod.current_agent_name = "team-lead"

        await mod.send_repl_message("team-lead", "hello")

    @pytest.mark.asyncio
    async def test_successful_send(self, monkeypatch):
        import dazi.repl_teams as mod

        mock_mailbox = AsyncMock()
        mock_mailbox.send = AsyncMock(return_value=["agent1"])
        monkeypatch.setattr(mod, "mailbox", mock_mailbox)

        mod.active_team_name = "test-team"
        mod.current_agent_name = "team-lead"

        mock_create = MagicMock(return_value=_make_message())
        monkeypatch.setattr("dazi.protocols.create_text_message", mock_create)

        await mod.send_repl_message("agent1", "hello")

        mock_create.assert_called_once_with(
            from_agent="team-lead",
            to_agent="agent1",
            text="hello",
        )
        mock_mailbox.send.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_fails(self, monkeypatch):
        import dazi.repl_teams as mod

        mock_mailbox = AsyncMock()
        mock_mailbox.send = AsyncMock(return_value=[])
        monkeypatch.setattr(mod, "mailbox", mock_mailbox)

        mod.active_team_name = "test-team"
        mod.current_agent_name = "team-lead"

        mock_create = MagicMock(return_value=_make_message())
        monkeypatch.setattr("dazi.protocols.create_text_message", mock_create)

        await mod.send_repl_message("agent1", "hello")


# ─────────────────────────────────────────────────────────
# broadcast_repl_message
# ─────────────────────────────────────────────────────────


class TestBroadcastReplMessage:
    @pytest.mark.asyncio
    async def test_no_active_team(self):
        import dazi.repl_teams as mod

        mod.active_team_name = None
        mod.current_agent_name = None

        await mod.broadcast_repl_message("hello everyone")

    @pytest.mark.asyncio
    async def test_team_not_found(self, monkeypatch):
        import dazi.repl_teams as mod
        from dazi._singletons import team_manager

        monkeypatch.setattr(mod, "team_manager", team_manager)
        monkeypatch.setattr(team_manager, "get_team", MagicMock(return_value=None))

        mod.active_team_name = "test-team"
        mod.current_agent_name = "team-lead"

        await mod.broadcast_repl_message("hello everyone")

    @pytest.mark.asyncio
    async def test_team_no_members(self, monkeypatch):
        import dazi.repl_teams as mod
        from dazi._singletons import team_manager

        monkeypatch.setattr(mod, "team_manager", team_manager)

        team = _make_team(members=[])
        monkeypatch.setattr(team_manager, "get_team", MagicMock(return_value=team))

        mod.active_team_name = "test-team"
        mod.current_agent_name = "team-lead"

        await mod.broadcast_repl_message("hello everyone")

    @pytest.mark.asyncio
    async def test_successful_broadcast(self, monkeypatch):
        import dazi.repl_teams as mod
        from dazi._singletons import team_manager

        monkeypatch.setattr(mod, "team_manager", team_manager)

        m1 = _make_member(name="frontend")
        m2 = _make_member(name="backend")
        team = _make_team(members=[m1, m2])
        monkeypatch.setattr(team_manager, "get_team", MagicMock(return_value=team))

        mock_mailbox = AsyncMock()
        mock_mailbox.send = AsyncMock(return_value=["frontend", "backend"])
        monkeypatch.setattr(mod, "mailbox", mock_mailbox)

        mock_create = MagicMock(return_value=_make_message())
        monkeypatch.setattr("dazi.protocols.create_text_message", mock_create)

        mod.active_team_name = "test-team"
        mod.current_agent_name = "team-lead"

        await mod.broadcast_repl_message("hello everyone")

        mock_create.assert_called_once_with(
            from_agent="team-lead",
            to_agent="*",
            text="hello everyone",
        )
        mock_mailbox.send.assert_called_once_with(
            team_name="test-team",
            message=mock_create.return_value,
            team_members=["frontend", "backend"],
        )


# ─────────────────────────────────────────────────────────
# send_shutdown_request
# ─────────────────────────────────────────────────────────


class TestSendShutdownRequest:
    @pytest.mark.asyncio
    async def test_no_active_team(self):
        import dazi.repl_teams as mod

        mod.active_team_name = None
        mod.current_agent_name = None

        await mod.send_shutdown_request("agent1")

    @pytest.mark.asyncio
    async def test_successful_shutdown(self, monkeypatch):
        import dazi.repl_teams as mod

        mock_mailbox = AsyncMock()
        mock_mailbox.send = AsyncMock(return_value=["agent1"])
        monkeypatch.setattr(mod, "mailbox", mock_mailbox)

        mod.active_team_name = "test-team"
        mod.current_agent_name = "team-lead"

        mock_create = MagicMock(return_value=_make_message(msg_type="shutdown_request"))
        monkeypatch.setattr("dazi.protocols.create_shutdown_request", mock_create)

        await mod.send_shutdown_request("agent1")

        mock_create.assert_called_once_with(
            from_agent="team-lead",
            to_agent="agent1",
        )
        mock_mailbox.send.assert_called_once()

    @pytest.mark.asyncio
    async def test_shutdown_fails(self, monkeypatch):
        import dazi.repl_teams as mod

        mock_mailbox = AsyncMock()
        mock_mailbox.send = AsyncMock(return_value=[])
        monkeypatch.setattr(mod, "mailbox", mock_mailbox)

        mod.active_team_name = "test-team"
        mod.current_agent_name = "team-lead"

        mock_create = MagicMock(return_value=_make_message(msg_type="shutdown_request"))
        monkeypatch.setattr("dazi.protocols.create_shutdown_request", mock_create)

        await mod.send_shutdown_request("agent1")
