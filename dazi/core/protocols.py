"""Protocol message factories for inter-agent communication.

KEY CONCEPTS:
  1. Protocol messages have specific msg_type values beyond plain "text"
  2. Each factory produces a Message with the correct structure
  3. Idle notifications fire when a teammate finishes work
  4. Shutdown is a request/response protocol
  5. Permission coordination uses request/response with request_id correlation
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from dazi.core.mailbox import Message
from dazi.core.team import TEAM_LEAD_NAME


# ─────────────────────────────────────────────────────────
# HELPER
# ─────────────────────────────────────────────────────────


def _now() -> str:
    """Return current UTC time as ISO 8601 string."""
    return datetime.now(timezone.utc).isoformat()


def _new_id() -> str:
    """Generate a new UUID4 string."""
    return str(uuid.uuid4())


# ─────────────────────────────────────────────────────────
# TEXT MESSAGE
# ─────────────────────────────────────────────────────────


def create_text_message(
    from_agent: str,
    to_agent: str,
    text: str,
    summary: str = "",
) -> Message:
    """Create a standard text message.

    Args:
        from_agent: Sender's name.
        to_agent: Recipient name or "*" for broadcast.
        text: Message body.
        summary: Optional summary. Auto-derived from text if empty.
    """
    return Message(
        id=_new_id(),
        from_agent=from_agent,
        to_agent=to_agent,
        text=text,
        timestamp=_now(),
        summary=summary,
        msg_type="text",
    )


# ─────────────────────────────────────────────────────────
# SHUTDOWN PROTOCOL
# ─────────────────────────────────────────────────────────


def create_shutdown_request(
    from_agent: str,
    to_agent: str,
    reason: str = "",
) -> Message:
    """Create a shutdown request message.

    The receiver should respond with create_shutdown_response.

    Args:
        from_agent: Agent requesting shutdown (usually team-lead).
        to_agent: Agent being asked to shut down.
        reason: Optional reason for the shutdown request.
    """
    return Message(
        id=_new_id(),
        from_agent=from_agent,
        to_agent=to_agent,
        text=f"Shutdown request from {from_agent}" + (f": {reason}" if reason else ""),
        timestamp=_now(),
        summary=f"Shutdown request from {from_agent}",
        msg_type="shutdown_request",
        metadata={
            "request_id": _new_id(),
            "reason": reason,
        },
    )


def create_shutdown_response(
    from_agent: str,
    to_agent: str,
    request_id: str,
    approve: bool,
    reason: str = "",
) -> Message:
    """Create a shutdown response message.

    Args:
        from_agent: Agent responding to shutdown request.
        to_agent: Agent that sent the shutdown request.
        request_id: The request_id from the shutdown_request.
        approve: True to accept shutdown, False to reject.
        reason: Optional reason (especially for rejection).
    """
    return Message(
        id=_new_id(),
        from_agent=from_agent,
        to_agent=to_agent,
        text=f"Shutdown {'approved' if approve else 'rejected'}" + (f": {reason}" if reason else ""),
        timestamp=_now(),
        summary=f"Shutdown {'approved' if approve else 'rejected'} by {from_agent}",
        msg_type="shutdown_response",
        metadata={
            "request_id": request_id,
            "approve": approve,
            "reason": reason,
        },
    )


# ─────────────────────────────────────────────────────────
# PERMISSION PROTOCOL
# ─────────────────────────────────────────────────────────


def create_permission_request(
    from_agent: str,
    tool_name: str,
    tool_args: dict[str, Any],
    reason: str = "",
) -> Message:
    """Create a permission request to the team leader.

    Always sent to TEAM_LEAD_NAME. Contains tool details for leader to evaluate.

    Args:
        from_agent: Teammate requesting permission.
        tool_name: Name of the tool needing permission.
        tool_args: Tool arguments (JSON-serializable dict).
        reason: Optional explanation for why the tool is needed.
    """
    return Message(
        id=_new_id(),
        from_agent=from_agent,
        to_agent=TEAM_LEAD_NAME,
        text=f"Permission request from {from_agent}: {tool_name}" + (f" — {reason}" if reason else ""),
        timestamp=_now(),
        summary=f"Permission request: {tool_name}",
        msg_type="permission_request",
        metadata={
            "request_id": _new_id(),
            "tool_name": tool_name,
            "tool_args": tool_args,
            "reason": reason,
        },
    )


def create_permission_response(
    from_agent: str,
    to_agent: str,
    request_id: str,
    approved: bool,
    reason: str = "",
) -> Message:
    """Create a permission response from the leader.

    Args:
        from_agent: Leader's name (usually TEAM_LEAD_NAME).
        to_agent: Teammate that requested permission.
        request_id: The request_id from the permission_request.
        approved: True if tool use is allowed, False if denied.
        reason: Optional explanation (especially for denial).
    """
    return Message(
        id=_new_id(),
        from_agent=from_agent,
        to_agent=to_agent,
        text=f"Permission {'approved' if approved else 'denied'} for {to_agent}" + (f": {reason}" if reason else ""),
        timestamp=_now(),
        summary=f"Permission {'approved' if approved else 'denied'}",
        msg_type="permission_response",
        metadata={
            "request_id": request_id,
            "approved": approved,
            "reason": reason,
        },
    )


# ─────────────────────────────────────────────────────────
# PLAN APPROVAL PROTOCOL
# ─────────────────────────────────────────────────────────


def create_plan_approval_request(
    from_agent: str,
    to_agent: str,
    plan_text: str,
) -> Message:
    """Create a plan approval request.

    A teammate sends this to the leader when it has a plan that needs review.

    Args:
        from_agent: Teammate submitting the plan.
        to_agent: Leader's name (usually TEAM_LEAD_NAME).
        plan_text: The plan content to be reviewed.
    """
    return Message(
        id=_new_id(),
        from_agent=from_agent,
        to_agent=to_agent,
        text=plan_text,
        timestamp=_now(),
        summary=f"Plan approval request from {from_agent}",
        msg_type="plan_approval_request",
        metadata={
            "request_id": _new_id(),
        },
    )


def create_plan_approval_response(
    from_agent: str,
    to_agent: str,
    request_id: str,
    approve: bool,
    feedback: str = "",
) -> Message:
    """Create a plan approval response.

    Args:
        from_agent: Leader's name.
        to_agent: Teammate that submitted the plan.
        request_id: The request_id from the plan_approval_request.
        approve: True to approve the plan, False to request changes.
        feedback: Optional feedback or requested changes.
    """
    return Message(
        id=_new_id(),
        from_agent=from_agent,
        to_agent=to_agent,
        text=f"Plan {'approved' if approve else 'changes requested'}" + (f": {feedback}" if feedback else ""),
        timestamp=_now(),
        summary=f"Plan {'approved' if approve else 'changes requested'}",
        msg_type="plan_approval_response",
        metadata={
            "request_id": request_id,
            "approve": approve,
            "feedback": feedback,
        },
    )


# ─────────────────────────────────────────────────────────
# IDLE NOTIFICATION
# ─────────────────────────────────────────────────────────


def create_idle_notification(
    from_agent: str,
    completed_task_id: str | None = None,
    idle_reason: str = "no_pending_work",
    summary: str = "",
) -> Message:
    """Create an idle notification.

    Always broadcast to all teammates so everyone (especially the leader)
    knows this agent is available.

    Args:
        from_agent: Agent going idle.
        completed_task_id: Optional ID of the task that was just completed.
        idle_reason: Why the agent is idle. Values: "no_pending_work",
                     "interrupted", "failed".
        summary: Optional summary of what was accomplished.
    """
    return Message(
        id=_new_id(),
        from_agent=from_agent,
        to_agent="*",
        text=f"{from_agent} is now idle" + (f" (reason: {idle_reason})" if idle_reason != "no_pending_work" else ""),
        timestamp=_now(),
        summary=summary or f"{from_agent} idle ({idle_reason})",
        msg_type="idle_notification",
        metadata={
            "completedTaskId": completed_task_id,
            "idleReason": idle_reason,
        },
    )
