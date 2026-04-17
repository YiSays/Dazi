"""Slash-command completer, /help display, and PromptSession configuration."""

from __future__ import annotations

from dataclasses import dataclass, field

from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.completion import CompleteEvent, Completer, Completion
from prompt_toolkit.document import Document
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.shortcuts.prompt import CompleteStyle
from rich.console import Console
from rich.table import Table

# ─────────────────────────────────────────────────────────
# COMMAND REGISTRY
# ─────────────────────────────────────────────────────────


@dataclass
class CommandEntry:
    """Metadata for a slash command."""

    name: str  # e.g. "/mcp"
    usage: str  # e.g. "/mcp [last]"
    description: str  # one-line help
    category: str  # group heading
    subcommands: list[CommandEntry] = field(default_factory=list)


COMMAND_REGISTRY: list[CommandEntry] = [
    # Core
    CommandEntry("/quit", "/quit", "Exit Dazi", "Core"),
    CommandEntry("/clear", "/clear", "Clear conversation history", "Core"),
    CommandEntry("/cost", "/cost [last]", "Show session cost", "Core"),
    CommandEntry("/settings", "/settings", "Show current settings", "Core"),
    CommandEntry("/reload", "/reload", "Reload settings, skills, and MCP servers", "Core"),
    CommandEntry("/onboard", "/onboard", "Re-run setup wizard", "Core"),
    CommandEntry("/help", "/help", "Show this help message", "Core"),
    # Mode
    CommandEntry("/plan", "/plan", "Switch to plan mode (read-only tools)", "Mode"),
    CommandEntry("/go", "/go", "Switch to execute mode (all tools)", "Mode"),
    CommandEntry("/show", "/show", "Display the plan file", "Mode"),
    CommandEntry("/tools", "/tools", "List available tools for current mode", "Mode"),
    # Permissions
    CommandEntry("/rules", "/rules", "List permission rules", "Permissions"),
    CommandEntry("/allow", "/allow <rule>", "Add an allow rule", "Permissions"),
    CommandEntry("/deny", "/deny <rule>", "Add a deny rule", "Permissions"),
    # Hooks
    CommandEntry("/hooks", "/hooks", "List registered hooks", "Hooks"),
    CommandEntry("/hook", "/hook", "Add a demo logging hook", "Hooks"),
    # Memory
    CommandEntry("/remember", "/remember <content>", "Store a memory entry", "Memory"),
    CommandEntry("/forget", "/forget <id>", "Delete a memory entry", "Memory"),
    CommandEntry("/memories", "/memories", "List all memories", "Memory"),
    CommandEntry("/reindex", "/reindex", "Rebuild memory index", "Memory"),
    # Tasks
    CommandEntry("/tasks", "/tasks", "List tasks on the board", "Tasks"),
    CommandEntry("/task", "/task <id>", "Show task detail", "Tasks"),
    CommandEntry("/bg", "/bg [task_id]", "List or inspect background tasks", "Tasks"),
    # MCP
    CommandEntry(
        "/mcp",
        "/mcp",
        "List MCP servers",
        "MCP",
        subcommands=[
            CommandEntry("/mcp connect", "/mcp connect <name>", "Connect to an MCP server", "MCP"),
            CommandEntry(
                "/mcp disconnect", "/mcp disconnect <name>", "Disconnect an MCP server", "MCP"
            ),
        ],
    ),
    # Skills
    CommandEntry("/skills", "/skills", "List all loaded skills", "Skills"),
    CommandEntry("/skill", "/skill <name>", "Show skill detail", "Skills"),
    # Teams
    CommandEntry("/teams", "/teams", "List all teams", "Teams"),
    CommandEntry(
        "/team",
        "/team <name>",
        "Activate or view a team",
        "Teams",
        subcommands=[
            CommandEntry("/team create", "/team create <name>", "Create a new team", "Teams"),
            CommandEntry("/team delete", "/team delete <name>", "Delete a team", "Teams"),
            CommandEntry("/team leave", "/team leave", "Leave active team", "Teams"),
        ],
    ),
    # Communications
    CommandEntry("/inbox", "/inbox [agent]", "Check inbox messages", "Communications"),
    CommandEntry("/send", "/send <agent> <msg>", "Send message to a teammate", "Communications"),
    CommandEntry("/broadcast", "/broadcast <msg>", "Broadcast to all teammates", "Communications"),
    CommandEntry(
        "/shutdown", "/shutdown <agent>", "Send shutdown request to a teammate", "Communications"
    ),
    # Proactive
    CommandEntry("/proactive", "/proactive [on|off]", "Show or toggle proactive mode", "Proactive"),
    CommandEntry("/autonomous", "/autonomous", "List autonomous teammates", "Proactive"),
    # Worktree
    CommandEntry(
        "/worktree",
        "/worktree",
        "List active worktrees",
        "Worktree",
        subcommands=[
            CommandEntry(
                "/worktree create", "/worktree create <name>", "Create a new worktree", "Worktree"
            ),
            CommandEntry(
                "/worktree finish",
                "/worktree finish <name> [--keep|--remove]",
                "Finish a worktree",
                "Worktree",
            ),
        ],
    ),
    # Context
    CommandEntry("/dazimd", "/dazimd", "Show loaded DAZI.md files", "Context"),
    CommandEntry("/compact", "/compact", "Manually compact conversation tokens", "Context"),
    CommandEntry("/tokens", "/tokens", "Show token usage information", "Context"),
]


def _build_skill_commands() -> list[CommandEntry]:
    """Build CommandEntry list for dynamically loaded user-invocable skills."""
    from dazi._singletons import skill_registry

    entries: list[CommandEntry] = []
    for skill in skill_registry.list_user_invocable():
        arg_hint = f" {skill.argument_hint}" if skill.argument_hint else ""
        desc = skill.description or f"Invoke skill: {skill.name}"
        entries.append(
            CommandEntry(
                name=f"/{skill.name}",
                usage=f"/{skill.name}{arg_hint}",
                description=desc,
                category="Skills (custom)",
            )
        )
    return entries


def _build_mcp_commands() -> list[CommandEntry]:
    """Build CommandEntry list for connected MCP servers."""
    from dazi._singletons import mcp_manager

    entries: list[CommandEntry] = []
    for server in mcp_manager.list_servers():
        if server["status"] != "connected":
            continue
        name = server["name"]
        tool_count = server["tool_count"]
        desc = f"MCP server ({tool_count} tool{'s' if tool_count != 1 else ''})"
        entries.append(
            CommandEntry(
                name=f"/{name}",
                usage=f"/{name}",
                description=desc,
                category="MCP Servers",
            )
        )
    return entries


# ─────────────────────────────────────────────────────────
# COMPLETER
# ─────────────────────────────────────────────────────────


class SlashCommandCompleter(Completer):
    """Completer for Dazi slash commands.

    Activates on Tab when input starts with '/'. Supports sub-command
    completion for /mcp, /team, /worktree. Dynamically includes
    user-invocable skills.
    """

    def __init__(self) -> None:
        self._subcommand_map: dict[str, list[CommandEntry]] = {}
        for entry in COMMAND_REGISTRY:
            if entry.subcommands:
                self._subcommand_map[entry.name] = entry.subcommands

    def get_completions(self, document: Document, complete_event: CompleteEvent):
        text_before = document.text_before_cursor
        if not text_before or not text_before.startswith("/"):
            return

        tokens = text_before.split()
        if not tokens:
            return

        first_token = tokens[0]

        # Sub-command completion
        if first_token in self._subcommand_map:
            subs = self._subcommand_map[first_token]
            if len(tokens) >= 2:
                partial = tokens[-1]
                for sub in subs:
                    token_part = sub.name.split()[-1]
                    if token_part.startswith(partial):
                        yield Completion(
                            text=token_part,
                            start_position=-len(partial),
                            display=sub.name,
                            display_meta=sub.description,
                        )
                return
            else:
                for sub in subs:
                    yield Completion(
                        text=" " + sub.name.split()[-1],
                        start_position=0,
                        display=sub.name,
                        display_meta=sub.description,
                    )
                return

        # Top-level command completion
        partial = first_token
        for entry in COMMAND_REGISTRY:
            if entry.name.startswith(partial):
                yield Completion(
                    text=entry.name,
                    start_position=-len(partial),
                    display=entry.name,
                    display_meta=f"[{entry.category}] {entry.description}",
                )
        for entry in _build_skill_commands():
            if entry.name.startswith(partial):
                yield Completion(
                    text=entry.name,
                    start_position=-len(partial),
                    display=entry.name,
                    display_meta=f"[{entry.category}] {entry.description}",
                )
        for entry in _build_mcp_commands():
            if entry.name.startswith(partial):
                yield Completion(
                    text=entry.name,
                    start_position=-len(partial),
                    display=entry.name,
                    display_meta=f"[{entry.category}] {entry.description}",
                )


# ─────────────────────────────────────────────────────────
# /help DISPLAY
# ─────────────────────────────────────────────────────────

_CATEGORY_ORDER = [
    "Core",
    "Mode",
    "Permissions",
    "Hooks",
    "Memory",
    "Tasks",
    "MCP",
    "MCP Servers",
    "Skills",
    "Skills (custom)",
    "Teams",
    "Communications",
    "Proactive",
    "Worktree",
    "Context",
]


def print_help(console: Console) -> None:
    """Display all slash commands grouped by category in Rich tables."""
    all_commands = list(COMMAND_REGISTRY) + _build_skill_commands() + _build_mcp_commands()

    categories: dict[str, list[CommandEntry]] = {}
    for cmd in all_commands:
        categories.setdefault(cmd.category, []).append(cmd)

    for cat_name in _CATEGORY_ORDER:
        entries = categories.pop(cat_name, None)
        if not entries:
            continue

        table = Table(title=cat_name, show_header=True, header_style="bold cyan")
        table.add_column("Command", style="bold green", min_width=30)
        table.add_column("Description")

        for entry in entries:
            table.add_row(entry.usage, entry.description)
            for sub in entry.subcommands:
                table.add_row(f"  {sub.usage}", sub.description)

        console.print(table)
        console.print()

    # Any remaining unknown categories
    for cat_name in sorted(categories):
        entries = categories[cat_name]
        table = Table(title=cat_name, show_header=True, header_style="bold cyan")
        table.add_column("Command", style="bold green", min_width=30)
        table.add_column("Description")
        for entry in entries:
            table.add_row(entry.usage, entry.description)
        console.print(table)
        console.print()


# ─────────────────────────────────────────────────────────
# KEY BINDINGS + SESSION CONFIG
# ─────────────────────────────────────────────────────────


def _build_repl_key_bindings(state: dict) -> KeyBindings:
    """Custom key bindings for the Dazi REPL."""
    from dazi.graph import EXECUTE_MODE

    kb = KeyBindings()

    @kb.add("c-q")
    def _quit(event):
        event.current_buffer.text = "/quit"
        event.current_buffer.validate_and_handle()

    @kb.add("s-tab")
    def _toggle_mode(event):
        cmd = "/plan" if state["mode"] == EXECUTE_MODE else "/go"
        event.current_buffer.text = cmd
        event.current_buffer.validate_and_handle()

    # Double ESC → clear input (uses prompt_toolkit's built-in sequence matching)
    @kb.add("escape", "escape")
    def _double_esc(event):
        event.current_buffer.text = ""

    # Double Ctrl+C → submit /quit to exit REPL
    @kb.add("c-c", "c-c")
    def _double_ctrl_c(event):
        event.current_buffer.text = "/quit"
        event.current_buffer.validate_and_handle()

    return kb


def get_prompt_session_kwargs(state: dict) -> dict:
    """Return kwargs for PromptSession with all enhancements enabled."""
    from dazi.config import DATA_DIR

    return {
        "history": FileHistory(DATA_DIR / "chat" / "history"),
        "completer": SlashCommandCompleter(),
        "complete_style": CompleteStyle.COLUMN,
        "auto_suggest": AutoSuggestFromHistory(),
        "complete_while_typing": False,
        "key_bindings": _build_repl_key_bindings(state),
        "enable_history_search": True,
        "search_ignore_case": True,
    }
