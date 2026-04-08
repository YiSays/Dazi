"""Permission system — rule-based tool access control.

The tool execution lifecycle is:
    validate -> pre-hook -> permission check -> execute -> post-hook -> result

The permission check gates every tool call. Rules come from three sources
(priority order): CLI flags > settings.json > project DAZI.md.

Rule matching patterns:
  - Exact:    "git push"       — matches only "git push"
  - Prefix:   "npm:*"          — matches "npm install", "npm run build"
  - Wildcard: "git *"          — matches "git add", "git commit", etc.
  - Glob:     "/tmp/*"         — matches file paths under /tmp/
"""

from __future__ import annotations

import fnmatch
import re
import shlex
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# ─────────────────────────────────────────────────────────
# PERMISSION MODES
# ─────────────────────────────────────────────────────────


class PermissionMode(str, Enum):
    """How the agent handles tool permissions.

    4 core modes that cover the key behaviors.
    """

    DEFAULT = "default"
    PLAN = "plan"
    ACCEPT_EDITS = "acceptEdits"
    BYPASS = "bypassPermissions"


# ─────────────────────────────────────────────────────────
# PERMISSION BEHAVIORS
# ─────────────────────────────────────────────────────────


class PermissionBehavior(str, Enum):
    """What happens when a permission rule matches."""

    ALLOW = "allow"
    DENY = "deny"
    ASK = "ask"


# ─────────────────────────────────────────────────────────
# PERMISSION RULE
# ─────────────────────────────────────────────────────────


@dataclass
class PermissionRule:
    """A single permission rule.

    Fields:
      behavior: "allow", "deny", or "ask"
      tool_name: tool name or pattern
      pattern: argument pattern (glob-style)
      source: "cli" | "settings" | "project"  (priority order)
    """

    behavior: PermissionBehavior
    tool_name: str | None = None
    pattern: str | None = None
    source: str = "cli"  # cli > settings > project

    # Matching type derived from tool_name format:
    #   "exact match"    — no special chars
    #   "prefix:*"       — colon-separated prefix
    #   "wildcard *"     — ends with *
    #   "/glob/*"        — path glob
    _match_type: str = field(default="", init=False, repr=False)

    def __post_init__(self):
        if self.tool_name:
            if self.tool_name.endswith("*"):
                self._match_type = "wildcard"
            elif ":" in self.tool_name:
                self._match_type = "prefix"
            else:
                self._match_type = "exact"

    def matches_tool(self, tool_name: str) -> bool:
        """Check if this rule matches a given tool name."""
        if self.tool_name is None:
            return True  # Rule with no tool_name matches all tools

        if self._match_type == "exact":
            return tool_name == self.tool_name

        if self._match_type == "prefix":
            prefix = self.tool_name.rstrip(":")
            return (
                tool_name == prefix
                or tool_name.startswith(prefix + ":")
                or tool_name.startswith(prefix + " ")
            )

        if self._match_type == "wildcard":
            base = self.tool_name.rstrip("*").rstrip()
            if not base:
                return True  # "*" matches all tools
            return (
                tool_name == base
                or tool_name.startswith(base + " ")
                or tool_name.startswith(base + ":")
            )

        return False

    def matches_args(self, tool_args: dict[str, Any]) -> bool:
        """Check if this rule's pattern matches the tool arguments.

        For shell commands, we match against the command string.
        For file tools, we match against file_path using glob.
        """
        if self.pattern is None:
            return True  # No pattern = matches any arguments

        # Check file_path argument (for file_reader, file_writer)
        file_path = tool_args.get("file_path", "")
        if file_path and self.pattern:
            return fnmatch.fnmatch(file_path, self.pattern)

        # Check command argument (for shell_exec)
        command = tool_args.get("command", "")
        if command and self.pattern:
            # Parse command and match the first word or full command
            return _shell_command_matches(command, self.pattern)

        return True


def _shell_command_matches(command: str, pattern: str) -> bool:
    """Check if a shell command matches a pattern.

    Patterns:
      - "*"              — match all commands
      - "git push"       — exact command match
      - "git *"          — any git subcommand
      - "npm:*"          — any npm subcommand
      - "rm *"           — any rm with args
    """
    # Wildcard * matches everything
    if pattern.strip() == "*":
        return True

    try:
        parts = shlex.split(command)
    except ValueError:
        parts = command.split()

    if not parts:
        return False

    # Exact match
    if pattern == command:
        return True

    # Wildcard pattern: "git *" matches "git push origin main"
    if pattern.endswith("*"):
        base = pattern.rstrip("*").rstrip()
        return " ".join(parts).startswith(base + " ") or (
            len(parts) > 0 and parts[0] == base
        )

    # Prefix pattern: "npm:" matches "npm install"
    if pattern.endswith(":"):
        base = pattern.rstrip(":")
        return parts[0] == base

    return " ".join(parts[: len(pattern.split())]) == pattern


# ─────────────────────────────────────────────────────────
# PERMISSION CHECKER
# ─────────────────────────────────────────────────────────

# Source priority (higher = checked first)
SOURCE_PRIORITY = {"cli": 3, "settings": 2, "project": 1}

# Mode defaults — what happens when no rule matches
MODE_DEFAULTS: dict[PermissionMode, dict[str, PermissionBehavior]] = {
    PermissionMode.DEFAULT: {
        "safe": PermissionBehavior.ALLOW,
        "write": PermissionBehavior.ASK,
        "destructive": PermissionBehavior.ASK,
    },
    PermissionMode.PLAN: {
        "safe": PermissionBehavior.ALLOW,
        "write": PermissionBehavior.DENY,
        "destructive": PermissionBehavior.DENY,
    },
    PermissionMode.ACCEPT_EDITS: {
        "safe": PermissionBehavior.ALLOW,
        "write": PermissionBehavior.ALLOW,
        "destructive": PermissionBehavior.ASK,
    },
    PermissionMode.BYPASS: {
        "safe": PermissionBehavior.ALLOW,
        "write": PermissionBehavior.ALLOW,
        "destructive": PermissionBehavior.ALLOW,
    },
}


@dataclass
class PermissionResult:
    """Result of a permission check."""

    behavior: PermissionBehavior
    matched_rule: PermissionRule | None = None
    reason: str = ""


def check_permission(
    tool_name: str,
    tool_args: dict[str, Any],
    rules: list[PermissionRule],
    mode: PermissionMode = PermissionMode.DEFAULT,
    tool_safety: str = "destructive",  # "safe", "write", "destructive"
) -> PermissionResult:
    """Check if a tool call is allowed.

    Priority order:
    1. Bypass mode — everything allowed
    2. Check rules in source priority order (cli > settings > project)
    3. Check rules within same source (deny > allow)
    4. Fall back to mode defaults

    Args:
        tool_name: Name of the tool being called.
        tool_args: Arguments passed to the tool.
        rules: List of permission rules to check.
        mode: Current permission mode.
        tool_safety: Safety level of the tool ("safe", "write", "destructive").

    Returns:
        PermissionResult with behavior and optional matched rule.
    """
    # 1. Bypass mode — everything allowed
    if mode == PermissionMode.BYPASS:
        return PermissionResult(
            behavior=PermissionBehavior.ALLOW,
            reason="Bypass mode — all tools allowed",
        )

    # 1b. Plan mode — deny write and destructive tools regardless of rules
    # In plan mode, write/destructive tools are stripped from the tool set entirely.
    # We enforce this at the permission level as well.
    if mode == PermissionMode.PLAN and tool_safety in ("write", "destructive"):
        return PermissionResult(
            behavior=PermissionBehavior.DENY,
            reason=f"Plan mode blocks {tool_safety} tools",
        )

    # 2. Find all matching rules grouped by source priority
    # Within a source: deny > allow > ask
    # Across sources: highest-priority source that has ANY match wins
    best_match: PermissionRule | None = None
    best_priority = -1

    for rule in rules:
        if rule.matches_tool(tool_name) and rule.matches_args(tool_args):
            rule_priority = SOURCE_PRIORITY.get(rule.source, 0)
            # Higher-priority source always wins
            if rule_priority > best_priority:
                best_match = rule
                best_priority = rule_priority
            # Same source: deny > allow > ask
            elif rule_priority == best_priority and best_match is not None:
                behavior_rank = {
                    PermissionBehavior.DENY: 3,
                    PermissionBehavior.ASK: 2,
                    PermissionBehavior.ALLOW: 1,
                }
                if behavior_rank.get(rule.behavior, 0) > behavior_rank.get(
                    best_match.behavior, 0
                ):
                    best_match = rule

    # 4. Apply matched rule
    if best_match:
        if best_match.behavior == PermissionBehavior.DENY:
            return PermissionResult(
                behavior=PermissionBehavior.DENY,
                matched_rule=best_match,
                reason=f"Denied by rule: {best_match.tool_name or '*'} (source: {best_match.source})",
            )
        elif best_match.behavior == PermissionBehavior.ALLOW:
            return PermissionResult(
                behavior=PermissionBehavior.ALLOW,
                matched_rule=best_match,
                reason=f"Allowed by rule: {best_match.tool_name or '*'} (source: {best_match.source})",
            )
        else:
            return PermissionResult(
                behavior=PermissionBehavior.ASK,
                matched_rule=best_match,
                reason=f"Requires approval: {best_match.tool_name or '*'} (source: {best_match.source})",
            )

    # 5. No rule matched — use mode defaults
    mode_defaults = MODE_DEFAULTS.get(mode, MODE_DEFAULTS[PermissionMode.DEFAULT])
    default_behavior = mode_defaults.get(tool_safety, PermissionBehavior.ASK)

    return PermissionResult(
        behavior=default_behavior,
        reason=f"No rule matched. Mode default for {tool_safety}: {default_behavior.value}",
    )


# ─────────────────────────────────────────────────────────
# RULE PARSER — parse string rules into PermissionRule
# ─────────────────────────────────────────────────────────


def parse_rule(rule_str: str, source: str = "cli") -> PermissionRule:
    """Parse a permission rule string into a PermissionRule.

    Formats:
      - "allow file_reader"                              — allow tool
      - "deny file_writer /etc/*"                       — deny with arg pattern
      - "allow shell_exec git *"                        — allow git commands
      - "deny shell_exec rm *"                          — deny rm commands
    """
    parts = rule_str.strip().split(None, 2)

    if len(parts) < 2:
        raise ValueError(
            f"Invalid rule format: '{rule_str}'. Expected: '<behavior> <tool> [pattern]'"
        )

    behavior_str = parts[0].lower()
    tool_name = parts[1]
    pattern = parts[2] if len(parts) > 2 else None

    try:
        behavior = PermissionBehavior(behavior_str)
    except ValueError:
        raise ValueError(
            f"Invalid behavior: '{behavior_str}'. Expected: allow, deny, ask"
        )

    return PermissionRule(
        behavior=behavior,
        tool_name=tool_name,
        pattern=pattern,
        source=source,
    )


def parse_rules(rule_strings: list[str], source: str = "cli") -> list[PermissionRule]:
    """Parse multiple rule strings into PermissionRule objects."""
    return [parse_rule(r, source) for r in rule_strings]


# ─────────────────────────────────────────────────────────
# PERMISSION PROMPT HELPERS — used by REPL interrupt handling
# ─────────────────────────────────────────────────────────


def derive_permission_pattern(tool_name: str, tool_args: dict) -> str | None:
    """Derive a permission pattern from tool args for auto-allow rules.

    Returns a pattern string or None:
      - file tools: directory glob from file_path (e.g., "/tmp/*")
      - shell tools: command prefix (e.g., "git *")
      - other tools: None (match all args)
    """
    from pathlib import Path

    file_path = tool_args.get("file_path", "")
    if file_path:
        parent = str(Path(file_path).parent)
        if not parent.endswith("/"):
            parent += "/"
        return parent + "*"

    command = tool_args.get("command", "")
    if command:
        parts = command.split(None, 1)
        return parts[0] + " *" if parts else "*"

    return None


async def prompt_permission_decisions(
    ask_tools: list[dict],
    session: Any,
    console: Any,
) -> dict[str, str]:
    """Prompt user for permission decisions on ASK tools.

    Renders a Rich Panel for each tool and collects allow/deny decisions
    via prompt_toolkit FormattedText prompt.

    Args:
        ask_tools: List of dicts with tool_call_id, tool_name, tool_args, reason.
        session: prompt_toolkit PromptSession instance.
        console: Rich Console instance.

    Returns:
        Dict mapping tool_call_id -> "allow" | "deny".
    """
    from prompt_toolkit.formatted_text import FormattedText
    from rich.panel import Panel

    decisions: dict[str, str] = {}

    for ask in ask_tools:
        pattern = derive_permission_pattern(ask["tool_name"], ask["tool_args"])
        pattern_display = (
            f"\n[dim]Rule to add: allow {ask['tool_name']} {pattern}[/dim]"
            if pattern
            else ""
        )
        console.print(
            Panel(
                f"[bold]Tool:[/bold] {ask['tool_name']}\n"
                f"[bold]Args:[/bold] {ask['tool_args']}\n"
                f"[bold]Reason:[/bold] {ask['reason']}"
                f"{pattern_display}",
                title="[yellow]Permission Required[/yellow]",
                border_style="yellow",
            )
        )
        answer = await session.prompt_async(
            FormattedText([("bold fg:yellow", " ALLOW? [y/N]: ")])
        )
        decisions[ask["tool_call_id"]] = (
            "allow" if answer.strip().lower() in ("y", "yes") else "deny"
        )

    return decisions
