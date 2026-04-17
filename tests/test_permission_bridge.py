"""Tests for dazi/permission_bridge.py — PermissionBridge evaluate_request, request_permission."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from dazi.mailbox import Mailbox, Message
from dazi.permission_bridge import PermissionBridge, PermissionRequestResult

# ─────────────────────────────────────────────────────────
# _find_matching_response
# ─────────────────────────────────────────────────────────


class TestFindMatchingResponse:
    def test_finds_matching_response(self):
        bridge = PermissionBridge()
        messages = [
            Message(
                id="r1",
                from_agent="leader",
                to_agent="worker",
                text="ok",
                timestamp="2025-01-01T00:00:00Z",
                msg_type="permission_response",
                metadata={"request_id": "req-1", "approved": True},
            ),
        ]
        result = bridge._find_matching_response(messages, "req-1")
        assert result is not None
        assert result.metadata["approved"] is True

    def test_wrong_type_returns_none(self):
        bridge = PermissionBridge()
        messages = [
            Message(
                id="r2",
                from_agent="leader",
                to_agent="worker",
                text="not a perm response",
                timestamp="2025-01-01T00:00:00Z",
                msg_type="text",
                metadata={"request_id": "req-1"},
            ),
        ]
        result = bridge._find_matching_response(messages, "req-1")
        assert result is None

    def test_wrong_id_returns_none(self):
        bridge = PermissionBridge()
        messages = [
            Message(
                id="r3",
                from_agent="leader",
                to_agent="worker",
                text="ok",
                timestamp="2025-01-01T00:00:00Z",
                msg_type="permission_response",
                metadata={"request_id": "req-999"},
            ),
        ]
        result = bridge._find_matching_response(messages, "req-1")
        assert result is None

    def test_empty_messages_returns_none(self):
        bridge = PermissionBridge()
        result = bridge._find_matching_response([], "req-1")
        assert result is None


# ─────────────────────────────────────────────────────────
# evaluate_request
# ─────────────────────────────────────────────────────────


class TestEvaluateRequest:
    @pytest.fixture
    def bridge_with_mailbox(self, tmp_path: Path) -> PermissionBridge:
        mailbox = Mailbox(base_dir=tmp_path)
        return PermissionBridge(mailbox=mailbox)

    @pytest.mark.asyncio
    async def test_evaluate_allowed(self, bridge_with_mailbox: PermissionBridge):
        from dazi.permissions import PermissionBehavior, PermissionRule

        rules = [
            PermissionRule(
                behavior=PermissionBehavior.ALLOW,
                tool_name="file_reader",
                source="cli",
            ),
        ]
        result = await bridge_with_mailbox.evaluate_request(
            leader_agent="team-lead",
            tool_name="file_reader",
            tool_args={"file_path": "/tmp/test.txt"},
            rules=rules,
            request_id="req-1",
            requester_agent="worker",
            team_name="team1",
        )
        assert result is True

    @pytest.mark.asyncio
    async def test_evaluate_denied(self, bridge_with_mailbox: PermissionBridge):
        from dazi.permissions import PermissionBehavior, PermissionRule

        rules = [
            PermissionRule(
                behavior=PermissionBehavior.DENY,
                tool_name="shell_exec",
                source="cli",
            ),
        ]
        result = await bridge_with_mailbox.evaluate_request(
            leader_agent="team-lead",
            tool_name="shell_exec",
            tool_args={"command": "rm -rf /"},
            rules=rules,
            request_id="req-2",
            requester_agent="worker",
            team_name="team1",
        )
        assert result is False

    @pytest.mark.asyncio
    async def test_evaluate_ask_denied_by_default(self, bridge_with_mailbox: PermissionBridge):
        from dazi.permissions import PermissionBehavior, PermissionRule

        rules = [
            PermissionRule(
                behavior=PermissionBehavior.ASK,
                tool_name="file_writer",
                source="cli",
            ),
        ]
        result = await bridge_with_mailbox.evaluate_request(
            leader_agent="team-lead",
            tool_name="file_writer",
            tool_args={"file_path": "/tmp/x"},
            rules=rules,
            request_id="req-3",
            requester_agent="worker",
            team_name="team1",
        )
        # ASK in bridge mode is treated as denied (no interactive prompt)
        assert result is False

    @pytest.mark.asyncio
    async def test_evaluate_sends_response_via_mailbox(self, bridge_with_mailbox: PermissionBridge):
        from dazi.permissions import PermissionBehavior, PermissionRule

        rules = [
            PermissionRule(
                behavior=PermissionBehavior.ALLOW,
                tool_name="file_reader",
                source="cli",
            ),
        ]
        await bridge_with_mailbox.evaluate_request(
            leader_agent="team-lead",
            tool_name="file_reader",
            tool_args={},
            rules=rules,
            request_id="req-4",
            requester_agent="worker",
            team_name="team1",
        )
        # Check that a response was delivered to the worker's inbox
        msgs = await bridge_with_mailbox.mailbox.receive("team1", "worker", unread_only=True)
        assert len(msgs) >= 1
        resp = msgs[0]
        assert resp.msg_type == "permission_response"
        assert resp.metadata.get("request_id") == "req-4"
        assert resp.metadata.get("approved") is True


# ─────────────────────────────────────────────────────────
# request_permission
# ─────────────────────────────────────────────────────────


class TestRequestPermission:
    @pytest.mark.asyncio
    async def test_request_permission_approved(self, tmp_path: Path):
        mailbox = Mailbox(base_dir=tmp_path)
        bridge = PermissionBridge(mailbox=mailbox)

        # Simulate: request permission, then immediately evaluate and respond
        async def _responder():
            # Wait for the request to arrive at leader's inbox
            await asyncio.sleep(0.2)
            leader_msgs = await mailbox.receive("team1", "team-lead", unread_only=True)
            if leader_msgs:
                msg = leader_msgs[0]
                request_id = msg.metadata.get("request_id", "")
                tool_name = msg.metadata.get("tool_name", "")
                await bridge.evaluate_request(
                    leader_agent="team-lead",
                    tool_name=tool_name,
                    tool_args={},
                    rules=[],
                    request_id=request_id,
                    requester_agent="worker",
                    team_name="team1",
                )

        responder_task = asyncio.create_task(_responder())
        result = await bridge.request_permission(
            from_agent="worker",
            tool_name="file_reader",
            tool_args={},
            team_name="team1",
            timeout=5.0,
            poll_interval=0.1,
        )
        await responder_task
        assert isinstance(result, PermissionRequestResult)
        assert result.approved is True

    @pytest.mark.asyncio
    async def test_request_permission_denied(self, tmp_path: Path):
        mailbox = Mailbox(base_dir=tmp_path)
        bridge = PermissionBridge(mailbox=mailbox)

        from dazi.permissions import PermissionBehavior, PermissionRule

        deny_rules = [
            PermissionRule(
                behavior=PermissionBehavior.DENY,
                tool_name="shell_exec",
                source="cli",
            ),
        ]

        async def _responder():
            await asyncio.sleep(0.2)
            leader_msgs = await mailbox.receive("team1", "team-lead", unread_only=True)
            if leader_msgs:
                msg = leader_msgs[0]
                request_id = msg.metadata.get("request_id", "")
                tool_name = msg.metadata.get("tool_name", "")
                await bridge.evaluate_request(
                    leader_agent="team-lead",
                    tool_name=tool_name,
                    tool_args={},
                    rules=deny_rules,
                    request_id=request_id,
                    requester_agent="worker",
                    team_name="team1",
                )

        responder_task = asyncio.create_task(_responder())
        result = await bridge.request_permission(
            from_agent="worker",
            tool_name="shell_exec",
            tool_args={},
            team_name="team1",
            timeout=5.0,
            poll_interval=0.1,
        )
        await responder_task
        assert result.approved is False

    @pytest.mark.asyncio
    async def test_request_permission_timeout(self, tmp_path: Path):
        mailbox = Mailbox(base_dir=tmp_path)
        bridge = PermissionBridge(mailbox=mailbox)

        with pytest.raises(TimeoutError):
            await bridge.request_permission(
                from_agent="worker",
                tool_name="file_reader",
                tool_args={},
                team_name="team1",
                timeout=0.3,
                poll_interval=0.1,
            )


# ─────────────────────────────────────────────────────────
# request_permission_func (standalone tool function)
# ─────────────────────────────────────────────────────────


class TestRequestPermissionFunc:
    @pytest.mark.asyncio
    async def test_no_active_team(self, monkeypatch):

        import dazi.permission_bridge as mod

        monkeypatch.setattr("dazi._singletons.active_team_name", None)
        monkeypatch.setattr("dazi._singletons.current_agent_name", "worker")

        result = await mod.request_permission_func(tool_name="file_reader")
        assert "Error: no active team" in result

    @pytest.mark.asyncio
    async def test_no_agent_identity(self, monkeypatch):

        import dazi.permission_bridge as mod

        monkeypatch.setattr("dazi._singletons.active_team_name", "team1")
        monkeypatch.setattr("dazi._singletons.current_agent_name", None)

        result = await mod.request_permission_func(tool_name="file_reader")
        assert "Error: agent identity not set" in result

    @pytest.mark.asyncio
    async def test_leader_cannot_request_own_permission(self, monkeypatch):

        import dazi.permission_bridge as mod

        monkeypatch.setattr("dazi._singletons.active_team_name", "team1")
        monkeypatch.setattr("dazi._singletons.current_agent_name", "team-lead")

        result = await mod.request_permission_func(tool_name="file_reader")
        assert "team leader does not need to request permission" in result

    @pytest.mark.asyncio
    async def test_approved(self, monkeypatch):
        from unittest.mock import AsyncMock, MagicMock

        import dazi.permission_bridge as mod

        mock_result = MagicMock()
        mock_result.approved = True
        mock_result.request_id = "r1"
        mock_result.reason = "ok"
        mock_bridge = AsyncMock()
        mock_bridge.request_permission = AsyncMock(return_value=mock_result)
        monkeypatch.setattr("dazi._singletons.active_team_name", "team1")
        monkeypatch.setattr("dazi._singletons.current_agent_name", "worker")
        monkeypatch.setattr("dazi._singletons.permission_bridge", mock_bridge)

        result = await mod.request_permission_func(tool_name="file_reader")
        assert "APPROVED" in result

    @pytest.mark.asyncio
    async def test_denied(self, monkeypatch):
        from unittest.mock import AsyncMock, MagicMock

        import dazi.permission_bridge as mod

        mock_result = MagicMock()
        mock_result.approved = False
        mock_result.request_id = "r1"
        mock_result.reason = "denied"
        mock_bridge = AsyncMock()
        mock_bridge.request_permission = AsyncMock(return_value=mock_result)
        monkeypatch.setattr("dazi._singletons.active_team_name", "team1")
        monkeypatch.setattr("dazi._singletons.current_agent_name", "worker")
        monkeypatch.setattr("dazi._singletons.permission_bridge", mock_bridge)

        result = await mod.request_permission_func(tool_name="shell_exec")
        assert "DENIED" in result

    @pytest.mark.asyncio
    async def test_timeout(self, monkeypatch):
        from unittest.mock import AsyncMock

        import dazi.permission_bridge as mod

        mock_bridge = AsyncMock()
        mock_bridge.request_permission = AsyncMock(side_effect=TimeoutError("timed out"))
        monkeypatch.setattr("dazi._singletons.active_team_name", "team1")
        monkeypatch.setattr("dazi._singletons.current_agent_name", "worker")
        monkeypatch.setattr("dazi._singletons.permission_bridge", mock_bridge)

        result = await mod.request_permission_func(tool_name="file_reader")
        assert "timed out" in result
