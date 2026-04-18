"""Team state management and inbox/messaging REPL helpers."""

from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from dazi._singletons import TASKS_DIR, mailbox, team_manager
from dazi.task_store import TaskStore
from dazi.team import TEAM_LEAD_NAME
from dazi.theme import BORDER

# Module-level team state (session-level, not per-conversation-turn)
active_team_name: str | None = None
current_agent_name: str | None = None
team_task_store: TaskStore | None = None

console = Console()


def _require_team() -> bool:
    """Return True if an active team and agent identity are set."""
    if not active_team_name:
        console.print("[red]No active team. Use /team <name> to activate a team first.[/red]")
        return False
    if not current_agent_name:
        console.print("[red]Agent identity not set.[/red]")
        return False
    return True


def show_teams_table() -> None:
    """Show all teams in a Rich table."""
    teams = team_manager.list_teams()
    if not teams:
        console.print("[dim]No teams exist. Use /team create <name> to create one.[/dim]")
        return

    table = Table(title=f"Teams ({len(teams)})")
    table.add_column("Name", style="cyan")
    table.add_column("Description")
    table.add_column("Members", justify="right")
    table.add_column("Active", justify="right")
    table.add_column("Created", style="dim")
    table.add_column("Status", style="dim")

    for t in teams:
        active = sum(1 for m in t.members if m.status == "active")
        idle = sum(1 for m in t.members if m.status == "idle")
        completed = sum(1 for m in t.members if m.status == "completed")
        status = f"{active}A/{idle}I/{completed}C"
        created = t.created_at[:10] if t.created_at else "?"
        desc = t.description[:40] + "..." if len(t.description) > 40 else t.description
        table.add_row(t.name, desc, str(len(t.members)), str(active), created, status)

    console.print(table)
    console.print("[dim]Commands: /team <name> to activate, /team delete <name> to remove[/dim]")


def show_team_detail(team_name: str) -> None:
    """Show detailed information about a specific team."""
    team = team_manager.get_team(team_name)
    if team is None:
        console.print(f"[red]Team '{team_name}' not found.[/red]")
        return

    lines = [
        f"[bold]{team.name}[/bold]",
        f"Description: {team.description or '(no description)'}",
        f"Created: {team.created_at}",
        f"Members ({len(team.members)}):",
    ]

    if team.members:
        status_icons = {
            "active": "[green]+[/green] active",
            "idle": "[yellow]=[/yellow] idle",
            "completed": "[dim]✓[/dim] completed",
        }
        for m in team.members:
            icon = status_icons.get(m.status, f"[?]{m.status}[/?]")
            lines.append(f"  {icon}  [bold]{m.name}[/bold] ({m.agent_id}) — {m.agent_type}")
    else:
        lines.append("  (no members)")

    lines.append(f"\nConfig: {team_manager._config_path(team_name)}")
    lines.append(f"Tasks:  {team_manager._task_dir(team_name)}")

    console.print(
        Panel(
            "\n".join(lines),
            title=f"Team: {team.name}",
            border_style=BORDER["info"],
        )
    )


def activate_team(name: str) -> None:
    """Switch to a team context, creating a team-scoped task store."""
    global active_team_name, current_agent_name, team_task_store
    import dazi._singletons as _singletons

    team = team_manager.get_team(name)
    if team is None:
        console.print(f"[red]Team '{name}' not found.[/red]")
        return

    active_team_name = name
    current_agent_name = TEAM_LEAD_NAME  # REPL user is always the leader

    # Sync with tools module
    _singletons.active_team_name = name
    _singletons.current_agent_name = TEAM_LEAD_NAME

    # Ensure inbox directory exists
    mailbox._ensure_inbox_dir(name)

    team_task_store = TaskStore(TASKS_DIR, list_id=name)
    _singletons.team_task_store = team_task_store
    task_dir = team_task_store._list_dir
    console.print(f"[green]Switched to team: {name} (as team-lead)[/green]")
    console.print(f"[dim]Task board: {task_dir}[/dim]")
    console.print(
        f"[dim]Inbox: "
        f"{team_manager.teams_dir / team_manager._sanitize_name(name)}"
        f"/inboxes/{TEAM_LEAD_NAME}.json"
        "[/dim]"
    )
    if team.members:
        console.print(f"[dim]Members: {', '.join(m.name for m in team.members)}[/dim]")


def deactivate_team() -> None:
    """Deactivate the current team, returning to the default task store."""
    global active_team_name, current_agent_name, team_task_store
    import dazi._singletons as _singletons

    if active_team_name:
        console.print(f"[dim]Left team: {active_team_name}[/dim]")
    active_team_name = None
    current_agent_name = None
    team_task_store = None

    # Sync with tools module
    _singletons.active_team_name = None
    _singletons.current_agent_name = None
    _singletons.team_task_store = None


# ─────────────────────────────────────────────────────────
# INBOX REPL HELPERS
# ─────────────────────────────────────────────────────────


async def show_inbox(agent_name: str | None = None) -> None:
    """Show inbox messages for an agent."""
    if not active_team_name:
        console.print("[red]No active team. Use /team <name> to activate a team first.[/red]")
        return

    target = agent_name or current_agent_name
    if not target:
        console.print("[red]Agent identity not set.[/red]")
        return

    messages = await mailbox.receive(
        team_name=active_team_name,
        agent_name=target,
        unread_only=True,
        limit=20,
    )

    if not messages:
        label = f"{agent_name}'s" if agent_name else "your"
        console.print(f"[dim]No unread messages in {label} inbox.[/dim]")
        return

    # Mark as read
    msg_ids = [m.id for m in messages]
    await mailbox.mark_read(
        team_name=active_team_name,
        agent_name=target,
        message_ids=msg_ids,
    )

    title = f"Inbox: {target}" + (" (peek)" if agent_name else "")
    table = Table(title=title)
    table.add_column("From", style="cyan")
    table.add_column("Type", style="bold")
    table.add_column("Summary", max_width=50)
    table.add_column("Time", style="dim")

    for msg in messages:
        type_tag = msg.msg_type if msg.msg_type != "text" else ""
        time_short = msg.timestamp[:19] if msg.timestamp else "?"
        summary = msg.summary or msg.text[:50]
        table.add_row(msg.from_agent, type_tag, summary, time_short)

    console.print(table)
    console.print(f"[dim]{len(messages)} message(s) marked as read.[/dim]")


async def send_repl_message(agent_name: str, text: str) -> None:
    """Send a message to a specific teammate via REPL."""
    if not _require_team():
        return

    if agent_name == current_agent_name:
        console.print(f"[red]Cannot send a message to yourself ({current_agent_name}).[/red]")
        return

    from dazi.protocols import create_text_message

    msg = create_text_message(
        from_agent=current_agent_name,
        to_agent=agent_name,
        text=text,
    )

    recipients = await mailbox.send(
        team_name=active_team_name,
        message=msg,
    )

    if recipients:
        console.print(f"[green]Message sent to {agent_name}.[/green]")
    else:
        console.print(f"[red]Could not deliver message to '{agent_name}'.[/red]")


async def broadcast_repl_message(text: str) -> None:
    """Broadcast a message to all teammates via REPL."""
    if not _require_team():
        return

    team = team_manager.get_team(active_team_name)
    if not team or not team.members:
        console.print(f"[red]No team members found for '{active_team_name}'.[/red]")
        return

    from dazi.protocols import create_text_message

    msg = create_text_message(
        from_agent=current_agent_name,
        to_agent="*",
        text=text,
    )

    team_members = [m.name for m in team.members]
    recipients = await mailbox.send(
        team_name=active_team_name,
        message=msg,
        team_members=team_members,
    )

    console.print(
        f"[green]Broadcast sent to {len(recipients)} teammate(s): {', '.join(recipients)}[/green]"
    )


async def send_shutdown_request(agent_name: str) -> None:
    """Send a shutdown request to a specific teammate via REPL."""
    if not _require_team():
        return

    from dazi.protocols import create_shutdown_request

    msg = create_shutdown_request(
        from_agent=current_agent_name,
        to_agent=agent_name,
    )

    recipients = await mailbox.send(
        team_name=active_team_name,
        message=msg,
    )

    if recipients:
        console.print(f"[yellow]Shutdown request sent to {agent_name}.[/yellow]")
    else:
        console.print(f"[red]Could not send shutdown request to '{agent_name}'.[/red]")
