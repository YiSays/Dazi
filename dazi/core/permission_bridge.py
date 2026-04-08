"""Leader-mediated permission bridge.

KEY CONCEPTS:
  1. Teammates don't have direct access to permission rules
  2. Teammates send permission_request to leader's mailbox
  3. Leader evaluates the request against its rules and responds
  4. PermissionBridge automates the send-wait-receive cycle

FLOW:
  Teammate: request_permission(tool_name, tool_args, team_name)
    -> creates permission_request message (via protocols.py)
    -> sends to leader's inbox via Mailbox
    -> polls own inbox for permission_response
    -> returns PermissionRequestResult

  Leader: evaluate_request(leader_agent, tool_name, tool_args, rules, ...)
    -> checks tool against permission rules (via shared/permissions.py)
    -> creates permission_response message (approved/denied)
    -> sends to requester's inbox via Mailbox
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from dazi.core.base import DaziTool, ToolSafety
from dazi.core.mailbox import Mailbox, Message
from dazi.core.permissions import PermissionBehavior, check_permission
from dazi.core.protocols import (
    create_permission_request,
    create_permission_response,
)
from dazi.core.team import TEAM_LEAD_NAME


# ─────────────────────────────────────────────────────────
# RESULT
# ─────────────────────────────────────────────────────────


@dataclass
class PermissionRequestResult:
    """Result of a permission request through the bridge."""

    approved: bool
    reason: str
    request_id: str


class PermissionBridge:
    """Leader-mediated permission coordination.

    Provides two roles:
    1. Teammate side: request_permission() — send request, wait for response
    2. Leader side: evaluate_request() — check rules, send response

    The bridge uses the Mailbox for message transport.
    """

    def __init__(self, mailbox: Mailbox | None = None) -> None:
        self.mailbox = mailbox or Mailbox()

    async def request_permission(
        self,
        from_agent: str,
        tool_name: str,
        tool_args: dict[str, Any],
        team_name: str,
        timeout: float = 30.0,
        poll_interval: float = 0.5,
    ) -> PermissionRequestResult:
        """Teammate requests permission from the team leader.

        Flow:
        1. Create permission_request message
        2. Send to leader's inbox
        3. Poll own inbox for matching permission_response
        4. Return result or timeout

        Args:
            from_agent: Teammate's name (e.g., "frontend").
            tool_name: Name of the tool needing permission.
            tool_args: Tool arguments.
            team_name: Team context.
            timeout: Max seconds to wait for response.
            poll_interval: Seconds between inbox polls.

        Returns:
            PermissionRequestResult with approved/rejected status.

        Raises:
            TimeoutError: If leader doesn't respond within timeout.
        """
        # 1. Create the request message
        request_msg = create_permission_request(
            from_agent=from_agent,
            tool_name=tool_name,
            tool_args=tool_args,
        )
        request_id = request_msg.metadata["request_id"]

        # 2. Send to leader's inbox
        await self.mailbox.send(
            team_name=team_name,
            message=request_msg,
            team_members=[TEAM_LEAD_NAME],
        )

        # 3. Poll own inbox for matching response
        elapsed = 0.0
        while elapsed < timeout:
            await asyncio.sleep(poll_interval)
            elapsed += poll_interval

            messages = await self.mailbox.receive(
                team_name=team_name,
                agent_name=from_agent,
                unread_only=True,
            )

            response = self._find_matching_response(messages, request_id)
            if response is not None:
                # Mark as read
                await self.mailbox.mark_read(
                    team_name=team_name,
                    agent_name=from_agent,
                    message_ids=[response.id],
                )

                approved = response.metadata.get("approved", False)
                reason = response.metadata.get("reason", "")

                return PermissionRequestResult(
                    approved=approved,
                    reason=reason,
                    request_id=request_id,
                )

        raise TimeoutError(
            f"Permission request timed out after {timeout}s "
            f"(request_id={request_id}, tool={tool_name})"
        )

    async def evaluate_request(
        self,
        leader_agent: str,
        tool_name: str,
        tool_args: dict[str, Any],
        rules: list[Any],
        request_id: str,
        requester_agent: str,
        team_name: str,
    ) -> bool:
        """Leader evaluates a permission request and sends response.

        Flow:
        1. Check tool against leader's permission rules
        2. Create permission_response message (approved/denied)
        3. Send to requester's inbox

        Args:
            leader_agent: Leader's name (typically TEAM_LEAD_NAME).
            tool_name: Tool being requested.
            tool_args: Tool arguments.
            rules: Leader's permission rules (list of PermissionRule).
            request_id: ID of the request being responded to.
            requester_agent: Name of the requesting teammate.
            team_name: Team context.

        Returns:
            True if request was approved.
        """
        # 1. Evaluate against rules
        # Use "safe" safety so no-matching = allow (bridge delegates to rules only)
        # If the leader has a deny rule, it will match and return DENY
        result = check_permission(
            tool_name=tool_name,
            tool_args=tool_args,
            rules=rules,
            tool_safety="safe",
        )

        approved = result.behavior == PermissionBehavior.ALLOW
        reason = ""

        if approved:
            reason = "Approved by team leader"
        elif result.behavior == PermissionBehavior.DENY:
            reason = f"Denied by team leader: {result.reason or 'rule mismatch'}"
        else:
            # ASK → deny by default in bridge mode (no interactive prompt)
            reason = f"Requires approval (ask): {result.reason or 'no matching rule'}"

        # 2. Send response to requester
        response_msg = create_permission_response(
            from_agent=leader_agent,
            to_agent=requester_agent,
            request_id=request_id,
            approved=approved,
            reason=reason,
        )

        await self.mailbox.send(
            team_name=team_name,
            message=response_msg,
            team_members=[requester_agent],
        )

        return approved

    def _find_matching_response(
        self,
        messages: list[Message],
        request_id: str,
    ) -> Message | None:
        """Find a permission_response message matching a request_id.

        Scans messages for msg_type == "permission_response" with matching
        metadata["request_id"].

        """
        for msg in messages:
            if msg.msg_type == "permission_response":
                if msg.metadata.get("request_id") == request_id:
                    return msg
        return None


# ─────────────────────────────────────────────────────────
# PERMISSION REQUEST TOOL
# ─────────────────────────────────────────────────────────

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field


class RequestPermissionInput(BaseModel):
    tool_name: str = Field(description="Name of the tool requesting permission")
    tool_args: dict = Field(default_factory=dict, description="Tool arguments as a dict")


async def request_permission_func(tool_name: str, tool_args: dict | None = None) -> str:
    """Request permission from the team leader for a tool call."""
    from dazi.core._singletons import permission_bridge, active_team_name, current_agent_name
    from dazi.core.team import TEAM_LEAD_NAME

    if not active_team_name:
        return "Error: no active team. Cannot request permission without a team."

    if not current_agent_name:
        return "Error: agent identity not set."

    if current_agent_name == TEAM_LEAD_NAME:
        return "Error: the team leader does not need to request permission from itself."

    if tool_args is None:
        tool_args = {}

    try:
        result = await permission_bridge.request_permission(
            from_agent=current_agent_name,
            tool_name=tool_name,
            tool_args=tool_args,
            team_name=active_team_name,
            timeout=30.0,
        )

        if result.approved:
            return (
                f"Permission APPROVED for {tool_name} "
                f"(request_id: {result.request_id}).\n"
                f"Reason: {result.reason}"
            )
        else:
            return (
                f"Permission DENIED for {tool_name} "
                f"(request_id: {result.request_id}).\n"
                f"Reason: {result.reason}"
            )

    except TimeoutError as e:
        return f"Error: {e}"


request_permission_tool = StructuredTool.from_function(
    func=lambda **kwargs: "",
    coroutine=request_permission_func,
    name="request_permission",
    description="Request permission from the team leader to use a tool. Only available for teammates (not the team leader).",
    args_schema=RequestPermissionInput,
)
request_permission_meta = DaziTool(name="request_permission", description="Request tool permission from team leader.", safety=ToolSafety.WRITE)
