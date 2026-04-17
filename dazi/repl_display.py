"""Rich table/panel display helpers for the Dazi REPL."""

from __future__ import annotations

from rich.align import Align
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

import dazi.graph as _graph_mod
from dazi._singletons import (
    background_manager,
    mcp_manager,
    memory_store,
    skill_registry,
    task_store,
)
from dazi.background import BackgroundTaskStatus
from dazi.dazimd import DaziMdFile, merge_dazimd_content
from dazi.graph import PLAN_MODE, _get_effective_rules
from dazi.hooks import HookEvent, HookResult
from dazi.llm import _get_model_name
from dazi.permissions import PermissionBehavior
from dazi.task_store import TaskStore
from dazi.theme import BORDER, PROMPT
from dazi.tokenizer import (
    count_messages_tokens,
    get_context_window,
    get_token_warning_state,
)

console = Console()


# ─────────────────────────────────────────────────────────
# MODE + RULES
# ─────────────────────────────────────────────────────────


def get_mode_badge(mode: str) -> list[tuple[str, str]]:
    if mode == PLAN_MODE:
        return [
            (PROMPT["mode_plan"], "PLAN MODE"),
            (PROMPT["dim"], " (Shift+Tab to switch)"),
        ]
    return [
        (PROMPT["mode_execute"], "EXEC MODE"),
        (PROMPT["dim"], " (Shift+Tab to switch)"),
    ]


def list_rules_table() -> None:
    table = Table(title="Permission Rules")
    table.add_column("Behavior", style="bold")
    table.add_column("Tool", style="cyan")
    table.add_column("Pattern", style="dim")
    table.add_column("Source", style="dim")

    all_rules = _get_effective_rules()
    for rule in all_rules:
        behavior_color = {
            PermissionBehavior.ALLOW: "[green]ALLOW[/green]",
            PermissionBehavior.DENY: "[red]DENY[/red]",
            PermissionBehavior.ASK: "[yellow]ASK[/yellow]",
        }
        table.add_row(
            behavior_color.get(rule.behavior, str(rule.behavior)),
            rule.tool_name or "*",
            rule.pattern or "*",
            rule.source,
        )
    console.print(table)


# ─────────────────────────────────────────────────────────
# MEMORIES
# ─────────────────────────────────────────────────────────


def list_memories_table() -> None:
    entries = memory_store.list_all()
    if not entries:
        console.print("[dim]No memories stored.[/dim]")
        return

    table = Table(title=f"Memories ({len(entries)} entries)")
    table.add_column("ID", style="cyan")
    table.add_column("Category")
    table.add_column("Description")
    table.add_column("Created", style="dim")

    for entry in entries:
        desc = entry.description or entry.content[:60]
        if len(desc) > 60:
            desc = desc[:57] + "..."
        table.add_row(entry.id, entry.category.value, desc, entry.created_at[:10])
    console.print(table)


# ─────────────────────────────────────────────────────────
# DAZI.MD
# ─────────────────────────────────────────────────────────


def show_dazimd_files(dazimd_files: list[DaziMdFile]) -> None:
    if not dazimd_files:
        console.print("[dim]No DAZI.md files loaded.[/dim]")
        return

    table = Table(title="Loaded DAZI.md Files")
    table.add_column("Path", style="cyan")
    table.add_column("Priority")
    table.add_column("Size", style="dim")

    for f in dazimd_files:
        table.add_row(str(f.path), str(f.priority), f"{len(f.content)} chars")
    console.print(table)

    merged = merge_dazimd_content(dazimd_files)
    if merged:
        console.print(
            Panel(
                Markdown(merged),
                title="Merged DAZI.md Content",
                border_style=BORDER["primary"],
            )
        )


# ─────────────────────────────────────────────────────────
# TOKENS
# ─────────────────────────────────────────────────────────


def show_token_info(messages: list) -> None:
    """Display token usage information."""
    model = _get_model_name()
    token_count = count_messages_tokens(messages, model) if messages else 0
    context_window = get_context_window(model)
    threshold = context_window - 13_000
    warning_state = get_token_warning_state(messages, model) if messages else "ok"

    pct = (token_count / context_window * 100) if context_window > 0 else 0

    state_colors = {
        "ok": "green",
        "warning": "yellow",
        "compact": "red",
        "error": "bold red",
    }
    state_color = state_colors.get(warning_state, "white")

    table = Table(title="Token Usage")
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")

    table.add_row("Model", model)
    table.add_row("Context Window", f"{context_window:,} tokens")
    table.add_row("Compact Threshold", f"{threshold:,} tokens")
    table.add_row("Current Usage", f"{token_count:,} tokens ({pct:.1f}%)")
    table.add_row("Warning State", f"[{state_color}]{warning_state}[/{state_color}]")
    table.add_row("Message Count", str(len(messages)))
    table.add_row("Compact Failures", f"{_graph_mod.consecutive_compact_failures}/3")

    console.print(table)


# ─────────────────────────────────────────────────────────
# TASKS
# ─────────────────────────────────────────────────────────


def list_tasks_table(
    *,
    active_team_name: str | None,
    default_task_store: TaskStore,
    team_task_store: TaskStore | None,
) -> None:
    """Display all tasks in a formatted table (team-scoped or default)."""
    store = team_task_store if active_team_name else default_task_store
    tasks = store.list_all()
    if not tasks:
        console.print("[dim]No tasks on the board.[/dim]")
        return

    table = Table(title=f"Task Board ({len(tasks)} tasks)")
    table.add_column("ID", style="cyan", justify="right")
    table.add_column("Status")
    table.add_column("Subject")
    table.add_column("Owner", style="dim")
    table.add_column("Blocked By", style="dim")

    status_colors = {
        "pending": "[white]pending[/white]",
        "in_progress": "[yellow]in_progress[/yellow]",
        "completed": "[green]completed[/green]",
    }

    for task in tasks:
        active_blockers = store.get_active_blockers(task.id)
        table.add_row(
            str(task.id),
            status_colors.get(task.status.value, task.status.value),
            task.subject,
            task.owner or "-",
            ", ".join(str(b) for b in active_blockers) if active_blockers else "-",
        )
    console.print(table)


def show_task_detail(
    task_id: int,
    *,
    active_team_name: str | None,
    default_task_store: TaskStore,
    team_task_store: TaskStore | None,
) -> None:
    """Display full details of a single task (team-scoped or default)."""
    store = team_task_store if active_team_name else default_task_store
    task = store.get(task_id)
    if task is None:
        console.print(f"[red]Task #{task_id} not found.[/red]")
        return

    lines = [
        f"[bold]#{task.id}[/bold] [{task.status.value}] [bold]{task.subject}[/bold]",
        "",
        task.description,
    ]
    if task.active_form:
        lines.append(f"\n[dim]Active form: {task.active_form}[/dim]")
    if task.owner:
        lines.append(f"[dim]Owner: {task.owner}[/dim]")
    if task.blocks:
        lines.append(f"[dim]Blocks: {task.blocks}[/dim]")
    if task.blocked_by:
        lines.append(f"[dim]Blocked by: {task.blocked_by}[/dim]")
    if task.metadata:
        lines.append(f"[dim]Metadata: {task.metadata}[/dim]")
    lines.append(f"[dim]Created: {task.created_at}[/dim]")

    console.print(
        Panel(
            "\n".join(lines),
            title=f"Task #{task.id}",
            border_style=BORDER["info"],
        )
    )


# ─────────────────────────────────────────────────────────
# BACKGROUND TASKS
# ─────────────────────────────────────────────────────────


def show_background_tasks_table() -> None:
    """Display all background tasks in a Rich table."""
    tasks = background_manager.list_all()
    if not tasks:
        console.print("[dim]No background tasks.[/dim]")
        return

    table = Table(title=f"Background Tasks ({len(tasks)})")
    table.add_column("ID", style="cyan")
    table.add_column("Status")
    table.add_column("Command", max_width=50)
    table.add_column("PID", style="dim", justify="right")
    table.add_column("Duration", style="dim", justify="right")
    table.add_column("Exit", style="dim", justify="right")

    status_colors = {
        "pending": "[white]pending[/white]",
        "running": "[yellow]running[/yellow]",
        "completed": "[green]completed[/green]",
        "failed": "[red]failed[/red]",
        "killed": "[bold red]killed[/bold red]",
    }

    for task in tasks:
        cmd = task.command[:50] + ("..." if len(task.command) > 50 else "")
        pid_str = str(task.pid) if task.pid else "-"
        dur_str = f"{task.duration_seconds:.1f}s" if task.duration_seconds else "-"
        exit_str = str(task.exit_code) if task.exit_code is not None else "-"
        table.add_row(
            task.id,
            status_colors.get(task.status.value, task.status.value),
            cmd,
            pid_str,
            dur_str,
            exit_str,
        )
    console.print(table)


def show_background_task_detail(task_id: str) -> None:
    """Show full details of a background task including output."""
    task = background_manager.check_sync(task_id)
    if task is None:
        console.print(f"[red]Background task '{task_id}' not found.[/red]")
        return

    lines = [
        f"[bold]{task.id}[/bold] [{task.status.value}]",
        f"[bold]Command:[/bold] {task.command}",
    ]
    if task.description:
        lines.append(f"[bold]Description:[/bold] {task.description}")
    if task.pid:
        lines.append(f"[dim]PID: {task.pid}[/dim]")
    if task.duration_seconds:
        lines.append(f"[dim]Duration: {task.duration_seconds:.1f}s[/dim]")
    if task.exit_code is not None:
        lines.append(f"[dim]Exit code: {task.exit_code}[/dim]")
    if task.error:
        lines.append(f"[red]Error: {task.error}[/red]")

    # Show output
    output = background_manager.get_output_tail(task_id, lines=30)
    if output:
        lines.append("\n[bold]Output (last 30 lines):[/bold]")
        lines.append("```")
        lines.append(output)
        lines.append("```")

    status_style = {
        BackgroundTaskStatus.RUNNING: "yellow",
        BackgroundTaskStatus.COMPLETED: "green",
        BackgroundTaskStatus.FAILED: "red",
        BackgroundTaskStatus.KILLED: "bold red",
        BackgroundTaskStatus.PENDING: "white",
    }
    border = status_style.get(task.status, "blue")

    console.print(
        Panel(
            "\n".join(lines),
            title=f"Background Task: {task.id}",
            border_style=border,
        )
    )


# ─────────────────────────────────────────────────────────
# MCP SERVERS
# ─────────────────────────────────────────────────────────


def show_mcp_servers_table() -> None:
    """Show all MCP servers in a Rich table."""
    servers = mcp_manager.list_servers()
    if not servers:
        console.print("[dim]No MCP servers configured.[/dim]")
        console.print(
            "[dim]Add servers via settings.json: "
            '{"mcpServers": {"name": {"command": "...", "args": [...]}}}[/dim]'
        )
        return

    table = Table(title="MCP Servers")
    table.add_column("Name", style="cyan")
    table.add_column("Status")
    table.add_column("Tools", justify="right")
    table.add_column("Resources", justify="right")
    table.add_column("Command", style="dim")

    for s in servers:
        status = s["status"]
        if status == "connected":
            status_str = f"[green]{status}[/green]"
        elif status == "error":
            status_str = f"[red]{status}[/red]"
        else:
            status_str = f"[dim]{status}[/dim]"

        table.add_row(
            s["name"],
            status_str,
            str(s["tool_count"]),
            str(s["resource_count"]),
            s["command"],
        )

    console.print(table)
    console.print(
        "[dim]Commands: /mcp <name> for details, /mcp connect <name>, /mcp disconnect <name>[/dim]"
    )


def show_mcp_server_detail(server_name: str) -> None:
    """Show detailed info for a specific MCP server."""
    conn = mcp_manager.get_server(server_name)
    if conn is None:
        console.print(f"[red]Server '{server_name}' not found.[/red]")
        return

    from dazi.mcp_client import MCPServerStatus

    # Server info panel
    config_text = (
        f"Command: {conn.config.command}\n"
        f"Args: {' '.join(conn.config.args) if conn.config.args else '(none)'}\n"
        f"Status: {conn.status.value}"
    )
    if conn.error:
        config_text += f"\nError: {conn.error}"
    console.print(
        Panel(
            config_text,
            title=f"[cyan]{server_name}[/cyan]",
            border_style=BORDER["primary"],
        )
    )

    # Tools
    if conn.tools:
        tool_table = Table(title=f"Tools ({len(conn.tools)})", show_lines=False)
        tool_table.add_column("Qualified Name", style="cyan")
        tool_table.add_column("Original", style="dim")
        tool_table.add_column("Read-Only")
        tool_table.add_column("Description", max_width=60)
        for t in conn.tools:
            ro = "[green]yes[/green]" if t.is_read_only else "[dim]no[/dim]"
            desc = (
                t.description[:60] + "..." if len(t.description) > 60 else t.description
            )
            tool_table.add_row(t.qualified_name, t.name, ro, desc)
        console.print(tool_table)
    elif conn.status == MCPServerStatus.CONNECTED:
        console.print("[dim]No tools discovered.[/dim]")

    # Resources
    if conn.resources:
        res_table = Table(title=f"Resources ({len(conn.resources)})", show_lines=False)
        res_table.add_column("URI", style="cyan")
        res_table.add_column("Name")
        res_table.add_column("MIME Type", style="dim")
        for r in conn.resources:
            res_table.add_row(str(r.uri), r.name, r.mime_type)
        console.print(res_table)


# ─────────────────────────────────────────────────────────
# SKILLS
# ─────────────────────────────────────────────────────────


def show_skills_table() -> None:
    """Show all registered skills in a Rich table."""
    skills = skill_registry.list_all()
    if not skills:
        console.print("[dim]No skills loaded.[/dim]")
        return

    table = Table(title=f"Skills ({len(skills)})")
    table.add_column("Name", style="cyan")
    table.add_column("Description")
    table.add_column("Source", style="dim")
    table.add_column("Invocable")

    for s in skills:
        source = (
            "bundled"
            if s.is_bundled
            else str(s.source_path.parent.name) if s.source_path else "unknown"
        )
        invocable = "[green]yes[/green]" if s.user_invocable else "[dim]no[/dim]"
        desc = s.description[:50] + "..." if len(s.description) > 50 else s.description
        table.add_row(f"/{s.name}", desc, source, invocable)

    console.print(table)


def show_skill_detail(skill_name: str) -> None:
    """Show detailed information about a specific skill."""
    skill = skill_registry.get(skill_name)
    if skill is None:
        console.print(f"[red]Skill '{skill_name}' not found.[/red]")
        return

    lines = [
        f"[bold]/{skill.name}[/bold]",
        f"Description: {skill.description}",
        f"Version: {skill.version}",
        f"User-invocable: {'yes' if skill.user_invocable else 'no'}",
    ]
    if skill.argument_hint:
        lines.append(f"Argument hint: {skill.argument_hint}")
    if skill.arguments:
        lines.append(f"Arguments: {', '.join(skill.arguments)}")
    if skill.when_to_use:
        lines.append(f"When to use: {skill.when_to_use}")
    if skill.allowed_tools:
        lines.append(f"Allowed tools: {', '.join(skill.allowed_tools)}")
    if skill.model:
        lines.append(f"Model override: {skill.model}")
    if skill.effort:
        lines.append(f"Effort: {skill.effort}")
    source = (
        "bundled"
        if skill.is_bundled
        else str(skill.source_path) if skill.source_path else "unknown"
    )
    lines.append(f"Source: {source}")

    console.print(
        Panel(
            "\n".join(lines),
            title=f"Skill: {skill.name}",
            border_style=BORDER["info"],
        )
    )

    # Show the prompt template
    console.print(
        Panel(
            Markdown(skill.prompt),
            title="Prompt Template",
            border_style=BORDER["primary"],
        )
    )


# ─────────────────────────────────────────────────────────
# USER MESSAGE
# ─────────────────────────────────────────────────────────


def render_user_panel(user_input: str, console: Console) -> None:
    """Render user input as a right-aligned chat-bubble Panel."""
    from dazi.theme import BORDER, RICH

    panel = Panel(
        Markdown(user_input, justify="right"),
        title=f"[{RICH['user_title']}]YOU[/{RICH['user_title']}]",
        title_align="right",
        border_style=BORDER["user"],
        padding=(0, 1),
    )
    console.print(Align.right(panel))


def render_dazi_panel(text: str, console: Console) -> None:
    """Render AI response as a left-aligned Panel."""
    from dazi.theme import BORDER, RICH

    console.print(
        Panel(
            Markdown(text),
            title=f"[{RICH['primary']}]DAZI[/{RICH['primary']}]",
            title_align="left",
            border_style=BORDER["info"],
            padding=(0, 1),
        )
    )


def render_thinking_panel(text: str, console: Console) -> None:
    """Render AI thinking/reasoning as a dim panel.

    Display is truncated at 500 chars for readability.
    Full text is preserved in AIMessage.additional_kwargs for API round-trips.
    """
    from dazi.theme import BORDER, RICH

    display = text[:500] + "\n..." if len(text) > 500 else text

    console.print(
        Panel(
            Text(display, style="dim"),
            title=f"[{RICH['dim']}]THINKING[/{RICH['dim']}]",
            # title_align="left",
            border_style=BORDER["dim"],
            padding=(0, 1),
        )
    )


# ─────────────────────────────────────────────────────────
# HOOKS (demo)
# ─────────────────────────────────────────────────────────


def add_demo_hook() -> None:
    async def logging_hook(
        tool_name: str = "", tool_args: dict | None = None, **kwargs
    ) -> HookResult:
        args_display = str(tool_args or {})[:100]
        console.print(f"  [dim][hook] pre_tool_use: {tool_name}({args_display})[/dim]")
        return HookResult()

    _graph_mod.hook_registry.register(
        HookEvent.PRE_TOOL_USE, logging_hook, priority=100
    )
    console.print("[green]Registered logging hook (priority=100).[/green]")


# ─────────────────────────────────────────────────────────
# BANNER / WELCOME
# ─────────────────────────────────────────────────────────


def print_ascii_banner(console: Console, *, version: str) -> None:
    """Print DAZI ASCII art banner in a single Panel with two-column layout."""
    from datetime import datetime

    block_lines = [
        "██████╗   █████╗  ███████╗ ██████╗",
        "██╔══██╗ ██╔══██╗ ╚══███╔╝ ╚═██╔═╝",
        "██║  ██║ ███████║   ███╔╝    ██║",
        "██║  ██║ ██╔══██║  ███╔╝     ██║",
        "██████╔╝ ██║  ██║ ███████╗ ██████╗",
        "╚═════╝  ╚═╝  ╚═╝ ╚══════╝ ╚═════╝",
        "https://github.com/YiSays/DAZI",
    ]

    now = datetime.now()
    hour = now.hour
    if hour < 6:
        greeting = "Late night coding"
    elif hour < 12:
        greeting = "Good morning"
    elif hour < 17:
        greeting = "Good afternoon"
    elif hour < 21:
        greeting = "Good evening"
    else:
        greeting = "Late night coding"

    from dazi.config import PROJECT_ROOT

    messages = (
        f"[bold magenta]{greeting}! 💡[/bold magenta]\n\n"
        f"[dim]{now:%A, %B %d, %Y — %H:%M}[/dim]\n"
        f"[dim]{str(PROJECT_ROOT).replace(str(PROJECT_ROOT.home()), '~')}[/dim]\n"
        "[dim]Type / for commands, Tab for autocompletion.[/dim]\n"
        "[dim]Ctrl+Q to quit.[/dim]"
    )

    inner = Table(
        show_header=False,
        show_edge=False,
        expand=True,
        padding=(0, 1),
        border_style="cyan",
    )
    inner.add_column(ratio=1)
    inner.add_column(ratio=1)
    inner.add_row(
        Align.center(
            "[bold cyan]" + "\n".join(block_lines) + "[/bold cyan]", vertical="middle"
        ),
        Align.center(messages, vertical="middle"),
    )

    console.print()
    console.print(
        Panel(
            inner,
            title="[cyan]  DAZI  [/cyan]",
            subtitle=(
                f"[italic cyan]  Develop Autonomously, "
                f"Zero Interruption. v{version}[/italic cyan]  "
            ),
            border_style=BORDER["brand"],
            padding=(1, 1),
        )
    )
    console.print()


def print_welcome_message(
    console: Console,
    *,
    skill_count: int,
    team_count: int,
    dazimd_files: list,
) -> None:
    """Print the welcome message with command list and session stats."""
    import dazi.repl_teams as _teams
    from dazi._singletons import (
        background_manager,
        memory_store,
    )
    from dazi.llm import _get_model_name
    from dazi.tokenizer import get_context_window

    # console.print("[dim]Type /help for commands. Tab to autocomplete. Ctrl+Q to quit.[/dim]")

    model = _get_model_name()
    _cur_task_store = _teams.team_task_store if _teams.active_team_name else task_store
    all_servers = mcp_manager.list_servers()
    mcp_total = len(all_servers)
    mcp_connected = sum(1 for s in all_servers if s["status"] == "connected")
    console.print(
        f"[dim]Model: {model} | Context: {get_context_window(model):,} tokens | "
        f"MCP: {mcp_connected}/{mcp_total} | "
        f"Memories: {len(memory_store.list_all())} | Tasks: {len(_cur_task_store.list_all())} | "
        f"Background: {len(background_manager.list_active())} active | Skills: {skill_count} | "
        f"Teams: {team_count} | "
        f"DAZI.md: {len(dazimd_files)} file(s)[/dim]"
    )
