"""Startup and shutdown lifecycle for the Dazi REPL."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from rich.console import Console

from dazi._singletons import (
    autonomous_teammate,
    background_manager,
    cost_tracker,
    mcp_manager,
    proactive_manager,
    settings_manager,
    skill_registry,
    teammate_runner,
    worktree_manager,
)
from dazi.dazimd import DaziMdFile, discover_dazimd_files, merge_dazimd_content
from dazi.prompt_builder import prompt_builder


# ─────────────────────────────────────────────────────────
# SUBSYSTEM LOAD RESULT
# ─────────────────────────────────────────────────────────


@dataclass
class SubsystemLoadResult:
    """Result from loading/reloading all subsystems."""

    dazimd_files: list[DaziMdFile]
    skill_count: int


def load_dazimd(*, console: Console) -> list[DaziMdFile]:
    """Discover and load DAZI.md files at startup."""
    project_root = Path.cwd()
    files = discover_dazimd_files(project_root=project_root, cwd=project_root)

    if files:
        merged = merge_dazimd_content(files)
        prompt_builder.set_dazimd_content(merged, files=files)
        console.print(f"[dim]Loaded {len(files)} DAZI.md file(s):[/dim]")
        for f in files:
            console.print(f"  [dim]{f.path} (priority: {f.priority})[/dim]")
    else:
        prompt_builder.set_dazimd_content("", files=[])
        console.print("[dim]No DAZI.md files found.[/dim]")

    return files


async def load_subsystems(*, console: Console) -> SubsystemLoadResult:
    """Load/reload all runtime subsystems: DAZI.md, settings, skills, MCP.

    Safe to call at startup (initial load) and at runtime (/reload, /onboard).
    MCP disconnect is a no-op when nothing is connected.
    """
    # DAZI.md — prints "[dim]Loaded N file(s)[/dim]" internally
    dazimd_files = load_dazimd(console=console)

    # Settings — 3-layer merge (DEFAULT → USER → PROJECT)
    settings_manager.reload()

    # Skills
    skill_count = skill_registry.reload()

    # MCP servers — disconnect is no-op on first call
    await mcp_manager.disconnect_all()

    from dazi.graph import connect_mcp_servers

    await connect_mcp_servers()

    return SubsystemLoadResult(dazimd_files=dazimd_files, skill_count=skill_count)


async def cleanup_on_exit(
    *,
    console: Console,
    active_team_name: str | None = None,
    say_goodbye: bool = False,
) -> None:
    """Shutdown all subsystems on REPL exit.

    Consolidates the cleanup sequence shared by /quit and the finally block.
    """
    proactive_manager.deactivate()

    for handle in autonomous_teammate.list_handles():
        await autonomous_teammate.shutdown(handle.team_name, handle.name)

    active_worktrees = worktree_manager.list_all()
    for wt in active_worktrees:
        try:
            worktree_manager.remove(wt.id, force=True)
        except Exception:
            pass
    if active_worktrees:
        console.print(f"[dim]Cleaned up {len(active_worktrees)} worktree(s).[/dim]")

    if active_team_name:
        count = await teammate_runner.shutdown_all(active_team_name)
        if count:
            console.print(
                f"[dim]Shut down {count} remaining teammate(s) for team '{active_team_name}'.[/dim]"
            )

    active = background_manager.list_active()
    if active:
        console.print(f"[dim]Cancelling {len(active)} remaining background task(s)...[/dim]")
        for task in active:
            await background_manager.cancel(task.id)

    await mcp_manager.disconnect_all()
    cost_tracker.save()

    if say_goodbye:
        console.print("[dim]Goodbye![/dim]")
