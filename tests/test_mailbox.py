"""Tests for dazi/mailbox.py — Message, Mailbox (send, receive, mark_read, purge)."""

from __future__ import annotations

from pathlib import Path

import pytest

from dazi.mailbox import Mailbox, Message

# ─────────────────────────────────────────────────────────
# Message serialization
# ─────────────────────────────────────────────────────────


class TestMessage:
    def test_to_dict_roundtrip(self):
        msg = Message(
            id="msg-1",
            from_agent="alice",
            to_agent="bob",
            text="Hello",
            timestamp="2025-01-01T00:00:00Z",
            summary="A greeting",
            msg_type="text",
        )
        d = msg.to_dict()
        assert d["id"] == "msg-1"
        assert d["from_agent"] == "alice"
        assert d["text"] == "Hello"

        restored = Message.from_dict(d)
        assert restored.id == msg.id
        assert restored.from_agent == msg.from_agent
        assert restored.text == msg.text

    def test_from_dict_defaults(self):
        msg = Message.from_dict({"id": "x"})
        assert msg.from_agent == ""
        assert msg.to_agent == ""
        assert msg.msg_type == "text"
        assert msg.read is False
        assert msg.metadata == {}


# ─────────────────────────────────────────────────────────
# Mailbox helpers
# ─────────────────────────────────────────────────────────


class TestMailboxHelpers:
    def test_sanitize_name_lowercases_and_replaces_specials(self):
        assert Mailbox._sanitize_name("My Team!") == "my-team"

    def test_sanitize_name_collapses_dashes(self):
        assert Mailbox._sanitize_name("a---b") == "a-b"

    def test_sanitize_name_strips_dashes(self):
        assert Mailbox._sanitize_name("-hello-") == "hello"

    def test_sanitize_name_all_special(self):
        assert Mailbox._sanitize_name("!!!") == ""

    def test_derive_summary_short_text(self):
        result = Mailbox._derive_summary("Hello world this is a test")
        assert result == "Hello world this is a test"

    def test_derive_summary_truncates_long_text(self):
        words = " ".join(f"word{i}" for i in range(20))
        result = Mailbox._derive_summary(words)
        assert len(result) <= 60

    def test_derive_summary_ten_word_limit(self):
        text = " ".join(f"word{i}" for i in range(15))
        result = Mailbox._derive_summary(text)
        # Should be at most 10 words
        assert len(result.split()) <= 10


# ─────────────────────────────────────────────────────────
# Mailbox send / receive
# ─────────────────────────────────────────────────────────


class TestMailboxSendReceive:
    @pytest.fixture
    def mailbox(self, tmp_path: Path) -> Mailbox:
        return Mailbox(base_dir=tmp_path)

    @pytest.mark.asyncio
    async def test_send_direct_message(self, mailbox: Mailbox):
        msg = Message(
            id="m1",
            from_agent="alice",
            to_agent="bob",
            text="Hello Bob",
            timestamp="2025-01-01T00:00:00Z",
        )
        recipients = await mailbox.send("team1", msg)
        assert recipients == ["bob"]

    @pytest.mark.asyncio
    async def test_send_broadcast(self, mailbox: Mailbox):
        msg = Message(
            id="m2",
            from_agent="alice",
            to_agent="*",
            text="Hello all",
            timestamp="2025-01-01T00:00:00Z",
        )
        recipients = await mailbox.send("team1", msg, team_members=["alice", "bob", "carol"])
        assert "bob" in recipients
        assert "carol" in recipients
        assert "alice" not in recipients  # sender excluded

    @pytest.mark.asyncio
    async def test_receive_unread_only(self, mailbox: Mailbox):
        msg = Message(
            id="m3",
            from_agent="alice",
            to_agent="bob",
            text="Hi",
            timestamp="2025-01-01T00:00:00Z",
        )
        await mailbox.send("team1", msg)
        received = await mailbox.receive("team1", "bob", unread_only=True)
        assert len(received) == 1
        assert received[0].text == "Hi"

    @pytest.mark.asyncio
    async def test_receive_all_includes_read(self, mailbox: Mailbox):
        msg = Message(
            id="m4",
            from_agent="alice",
            to_agent="bob",
            text="Hi",
            timestamp="2025-01-01T00:00:00Z",
            read=True,
        )
        await mailbox.send("team1", msg)
        received = await mailbox.receive("team1", "bob", unread_only=False)
        assert len(received) == 1

    @pytest.mark.asyncio
    async def test_receive_empty_inbox(self, mailbox: Mailbox):
        received = await mailbox.receive("team1", "nobody", unread_only=True)
        assert received == []

    @pytest.mark.asyncio
    async def test_receive_respects_limit(self, mailbox: Mailbox):
        for i in range(5):
            msg = Message(
                id=f"m{i}",
                from_agent="alice",
                to_agent="bob",
                text=f"msg {i}",
                timestamp="2025-01-01T00:00:00Z",
            )
            await mailbox.send("team1", msg)
        received = await mailbox.receive("team1", "bob", unread_only=False, limit=2)
        assert len(received) == 2


# ─────────────────────────────────────────────────────────
# Mailbox mark_read / purge
# ─────────────────────────────────────────────────────────


class TestMailboxMarkReadPurge:
    @pytest.fixture
    def mailbox(self, tmp_path: Path) -> Mailbox:
        return Mailbox(base_dir=tmp_path)

    @pytest.mark.asyncio
    async def test_mark_read_specific_ids(self, mailbox: Mailbox):
        msg1 = Message(
            id="r1",
            from_agent="a",
            to_agent="b",
            text="1",
            timestamp="2025-01-01T00:00:00Z",
        )
        msg2 = Message(
            id="r2",
            from_agent="a",
            to_agent="b",
            text="2",
            timestamp="2025-01-01T00:00:00Z",
        )
        await mailbox.send("team1", msg1)
        await mailbox.send("team1", msg2)
        count = await mailbox.mark_read("team1", "b", message_ids=["r1"])
        assert count == 1

    @pytest.mark.asyncio
    async def test_mark_read_all(self, mailbox: Mailbox):
        for i in range(3):
            msg = Message(
                id=f"a{i}",
                from_agent="x",
                to_agent="y",
                text=f"msg{i}",
                timestamp="2025-01-01T00:00:00Z",
            )
            await mailbox.send("team1", msg)
        count = await mailbox.mark_read("team1", "y")
        assert count == 3

    @pytest.mark.asyncio
    async def test_purge_with_messages(self, mailbox: Mailbox):
        msg = Message(
            id="p1",
            from_agent="a",
            to_agent="b",
            text="bye",
            timestamp="2025-01-01T00:00:00Z",
        )
        await mailbox.send("team1", msg)
        count = await mailbox.purge("team1", "b")
        assert count == 1
        # After purge, inbox should be empty
        received = await mailbox.receive("team1", "b", unread_only=False)
        assert received == []

    @pytest.mark.asyncio
    async def test_purge_empty_inbox(self, mailbox: Mailbox):
        count = await mailbox.purge("team1", "nobody")
        assert count == 0


# ─────────────────────────────────────────────────────────
# Standalone tool functions
# ─────────────────────────────────────────────────────────


class TestSendMessageFunc:
    @pytest.mark.asyncio
    async def test_no_active_team(self, monkeypatch):

        import dazi.mailbox as mod

        monkeypatch.setattr("dazi._singletons.active_team_name", None)
        monkeypatch.setattr("dazi._singletons.current_agent_name", "worker")
        result = await mod.send_message_func(to="leader", message="hi")
        assert "no active team" in result

    @pytest.mark.asyncio
    async def test_no_agent_identity(self, monkeypatch):

        import dazi.mailbox as mod

        monkeypatch.setattr("dazi._singletons.active_team_name", "team1")
        monkeypatch.setattr("dazi._singletons.current_agent_name", None)
        result = await mod.send_message_func(to="leader", message="hi")
        assert "agent identity not set" in result

    @pytest.mark.asyncio
    async def test_send_to_self(self, monkeypatch):

        import dazi.mailbox as mod

        monkeypatch.setattr("dazi._singletons.active_team_name", "team1")
        monkeypatch.setattr("dazi._singletons.current_agent_name", "worker")
        result = await mod.send_message_func(to="worker", message="hi")
        assert "yourself" in result

    @pytest.mark.asyncio
    async def test_broadcast_no_members(self, monkeypatch):
        from unittest.mock import MagicMock

        import dazi.mailbox as mod

        mock_tm = MagicMock(get_team=MagicMock(return_value=MagicMock(members=[])))
        monkeypatch.setattr("dazi._singletons.active_team_name", "team1")
        monkeypatch.setattr("dazi._singletons.current_agent_name", "worker")
        monkeypatch.setattr("dazi._singletons.team_manager", mock_tm)
        result = await mod.send_message_func(to="*", message="hi")
        assert "no team members" in result

    @pytest.mark.asyncio
    async def test_direct_send_success(self, monkeypatch):
        from unittest.mock import AsyncMock, MagicMock

        import dazi.mailbox as mod

        monkeypatch.setattr("dazi._singletons.active_team_name", "team1")
        monkeypatch.setattr("dazi._singletons.current_agent_name", "worker")
        monkeypatch.setattr(
            "dazi._singletons.mailbox", MagicMock(send=AsyncMock(return_value=["leader"]))
        )
        monkeypatch.setattr("dazi._singletons.team_manager", MagicMock())
        result = await mod.send_message_func(to="leader", message="hello")
        assert "sent to leader" in result

    @pytest.mark.asyncio
    async def test_direct_send_fail(self, monkeypatch):
        from unittest.mock import AsyncMock, MagicMock

        import dazi.mailbox as mod

        monkeypatch.setattr("dazi._singletons.active_team_name", "team1")
        monkeypatch.setattr("dazi._singletons.current_agent_name", "worker")
        monkeypatch.setattr("dazi._singletons.mailbox", MagicMock(send=AsyncMock(return_value=[])))
        monkeypatch.setattr("dazi._singletons.team_manager", MagicMock())
        result = await mod.send_message_func(to="leader", message="hello")
        assert "could not deliver" in result

    @pytest.mark.asyncio
    async def test_broadcast_success(self, monkeypatch):
        from unittest.mock import AsyncMock, MagicMock

        import dazi.mailbox as mod

        m1 = MagicMock(name="leader")
        m2 = MagicMock(name="backend")
        monkeypatch.setattr("dazi._singletons.active_team_name", "team1")
        monkeypatch.setattr("dazi._singletons.current_agent_name", "worker")
        monkeypatch.setattr(
            "dazi._singletons.team_manager",
            MagicMock(
                get_team=MagicMock(
                    return_value=MagicMock(members=[m1, m2, MagicMock(name="worker")])
                )
            ),
        )
        monkeypatch.setattr(
            "dazi._singletons.mailbox",
            MagicMock(send=AsyncMock(return_value=["leader", "backend"])),
        )
        result = await mod.send_message_func(to="*", message="hi")
        assert "Broadcast sent to 2" in result


class TestCheckInboxFunc:
    @pytest.mark.asyncio
    async def test_no_active_team(self, monkeypatch):

        import dazi.mailbox as mod

        monkeypatch.setattr("dazi._singletons.active_team_name", None)
        result = await mod.check_inbox_func()
        assert "no active team" in result

    @pytest.mark.asyncio
    async def test_no_agent_identity(self, monkeypatch):

        import dazi.mailbox as mod

        monkeypatch.setattr("dazi._singletons.active_team_name", "team1")
        monkeypatch.setattr("dazi._singletons.current_agent_name", None)
        result = await mod.check_inbox_func()
        assert "agent identity not set" in result

    @pytest.mark.asyncio
    async def test_empty_inbox(self, monkeypatch):
        from unittest.mock import AsyncMock, MagicMock

        import dazi.mailbox as mod

        monkeypatch.setattr("dazi._singletons.active_team_name", "team1")
        monkeypatch.setattr("dazi._singletons.current_agent_name", "worker")
        monkeypatch.setattr(
            "dazi._singletons.mailbox",
            MagicMock(
                receive=AsyncMock(return_value=[]),
                mark_read=AsyncMock(return_value=0),
            ),
        )
        result = await mod.check_inbox_func()
        assert "No unread" in result

    @pytest.mark.asyncio
    async def test_with_messages(self, monkeypatch):
        from unittest.mock import AsyncMock, MagicMock

        import dazi.mailbox as mod

        msg = MagicMock(
            id="m1",
            msg_type="text",
            from_agent="leader",
            timestamp="2025-01-01T00:00:00Z",
            text="hello world",
            metadata={},
        )
        monkeypatch.setattr("dazi._singletons.active_team_name", "team1")
        monkeypatch.setattr("dazi._singletons.current_agent_name", "worker")
        monkeypatch.setattr(
            "dazi._singletons.mailbox",
            MagicMock(
                receive=AsyncMock(return_value=[msg]),
                mark_read=AsyncMock(return_value=1),
            ),
        )
        result = await mod.check_inbox_func()
        assert "Inbox (1" in result
        assert "leader" in result

    @pytest.mark.asyncio
    async def test_message_with_metadata(self, monkeypatch):
        from unittest.mock import AsyncMock, MagicMock

        import dazi.mailbox as mod

        msg = MagicMock(
            id="m1",
            msg_type="permission_request",
            from_agent="leader",
            timestamp="2025-01-01T00:00:00Z",
            text="check",
            metadata={"request_id": "r1"},
        )
        monkeypatch.setattr("dazi._singletons.active_team_name", "team1")
        monkeypatch.setattr("dazi._singletons.current_agent_name", "worker")
        monkeypatch.setattr(
            "dazi._singletons.mailbox",
            MagicMock(
                receive=AsyncMock(return_value=[msg]),
                mark_read=AsyncMock(return_value=1),
            ),
        )
        result = await mod.check_inbox_func()
        assert "Metadata: request_id" in result
        assert "[permission_request]" in result

    @pytest.mark.asyncio
    async def test_all_messages(self, monkeypatch):
        from unittest.mock import AsyncMock, MagicMock

        import dazi.mailbox as mod

        msg = MagicMock(
            id="m1",
            msg_type="text",
            from_agent="leader",
            timestamp="2025-01-01T00:00:00Z",
            text="hi",
            metadata={},
        )
        monkeypatch.setattr("dazi._singletons.active_team_name", "team1")
        monkeypatch.setattr("dazi._singletons.current_agent_name", "worker")
        monkeypatch.setattr(
            "dazi._singletons.mailbox",
            MagicMock(
                receive=AsyncMock(return_value=[msg]),
                mark_read=AsyncMock(return_value=1),
            ),
        )
        result = await mod.check_inbox_func(unread_only=False)
        assert "No messages" not in result


class TestSendIdleNotification:
    @pytest.mark.asyncio
    async def test_with_team(self, monkeypatch):
        from unittest.mock import AsyncMock, MagicMock

        import dazi.mailbox as mod

        m = MagicMock(name="leader")
        monkeypatch.setattr("dazi._singletons.mailbox", MagicMock(send=AsyncMock(return_value=[m])))
        monkeypatch.setattr(
            "dazi._singletons.team_manager",
            MagicMock(
                get_team=MagicMock(return_value=MagicMock(members=[m, MagicMock(name="backend")])),
            ),
        )
        await mod.send_idle_notification(
            agent_name="worker",
            team_name="team1",
            completed_task_id="1",
            idle_reason="done",
        )

    @pytest.mark.asyncio
    async def test_no_team(self, monkeypatch):
        from unittest.mock import AsyncMock, MagicMock

        import dazi.mailbox as mod

        monkeypatch.setattr("dazi._singletons.mailbox", MagicMock(send=AsyncMock(return_value=[])))
        monkeypatch.setattr(
            "dazi._singletons.team_manager", MagicMock(get_team=MagicMock(return_value=None))
        )
        await mod.send_idle_notification(agent_name="worker", team_name="team1")
