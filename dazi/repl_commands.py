"""Slash command dispatch for the Dazi REPL.

All built-in slash commands are handled here. The REPL loop calls
`handle_command()` and acts on the result:
  - returns "continue" → command was handled, continue the loop
  - returns "break"    → /quit was issued, exit the loop
  - returns None       → not a built-in command, treat as regular input
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

import dazi.graph as _graph_mod
import dazi.repl_teams as _teams
from dazi._singletons import (
    PLAN_FILE,
    autonomous_teammate,
    cost_tracker,
    mcp_manager,
    memory_store,
    proactive_manager,
    settings_manager,
    skill_registry,
    task_store,
    team_manager,
    worktree_manager,
)
from dazi.compact import manual_compact
from dazi.graph import (
    EXECUTE_MODE,
    PLAN_MODE,
    hook_registry,
    permission_rules,
    rebuild_tool_lists,
)
from dazi.llm import _get_llm, _get_model_name
from dazi.memory import MemoryEntry
from dazi.permissions import parse_rule
from dazi.proactive import ProactiveSource
from dazi.prompt_builder import _update_proactive_prompt
from dazi.registry import (
    EXECUTE_MODE_META,
    EXECUTE_MODE_TOOLS,
    PLAN_MODE_META,
    PLAN_MODE_TOOLS,
)
from dazi.repl_completer import print_help
from dazi.repl_display import (
    add_demo_hook,
    list_memories_table,
    list_rules_table,
    list_tasks_table,
    show_background_task_detail,
    show_background_tasks_table,
    show_dazimd_files,
    show_mcp_server_detail,
    show_mcp_servers_table,
    show_skill_detail,
    show_skills_table,
    show_task_detail,
    show_token_info,
)
from dazi.theme import BORDER
from dazi.tokenizer import count_messages_tokens

if TYPE_CHECKING:
    from prompt_toolkit import PromptSession


async def handle_command(
    cmd: str,
    *,
    state: dict,
    session: PromptSession,
    console: Console,
    print_welcome_fn: Callable[[], None],
) -> str | None:
    """Dispatch a slash command.

    Returns:
        "continue" — command handled, continue loop
        "break"    — exit the loop
        None       — not a slash command, treat as regular input
    """
    # ── /quit ──
    if cmd == "/quit":
        from dazi.lifecycle import cleanup_on_exit

        await cleanup_on_exit(
            console=console,
            active_team_name=_teams.active_team_name,
            say_goodbye=True,
        )
        return "break"

    # ── /help ──
    if cmd == "/help":
        print_help(console)
        return "continue"

    # ── /cost ──
    if cmd == "/cost" or cmd.startswith("/cost "):
        if cmd == "/cost last":
            console.print(
                Panel(
                    cost_tracker.format_last_session(),
                    title="Previous Session",
                    border_style=BORDER["primary"],
                )
            )
        else:
            console.print(
                Panel(
                    cost_tracker.format_summary(),
                    title="Session Cost",
                    border_style=BORDER["success"],
                )
            )
        return "continue"

    # ── /settings ──
    if cmd == "/settings":
        s = settings_manager.settings
        sm = settings_manager.source_map
        table = Table(title="Settings")
        table.add_column("Field", style="bold")
        table.add_column("Value")
        table.add_column("Source", style="dim")
        for f_name, f_value in [
            ("model", s.model),
            ("api_base_url", str(s.api_base_url)),
            ("default_mode", s.default_mode),
            ("auto_compact", str(s.auto_compact)),
            ("auto_memory", str(s.auto_memory)),
            ("max_concurrent_tools", str(s.max_concurrent_tools)),
            ("allow_rules", f"{len(s.allow_rules)} rule(s)"),
            ("deny_rules", f"{len(s.deny_rules)} rule(s)"),
            ("env", f"{len(s.env)} var(s)"),
        ]:
            source = sm.get(f_name, "default")
            source_color = {"user": "yellow", "project": "green"}.get(source, "dim")
            table.add_row(
                f_name,
                str(f_value),
                f"[{source_color}]{source}[/{source_color}]",
            )
        console.print(table)
        console.print(f"\n[dim]User:   {settings_manager.user_path}[/dim]")
        console.print(f"[dim]Project: {settings_manager.project_path}[/dim]")
        return "continue"

    # ── /reload ──
    if cmd == "/reload":
        from dazi.lifecycle import load_subsystems

        await load_subsystems(console=console)
        console.print("[green]Reloaded: DAZI.md, settings, skills, MCP servers.[/green]")
        return "continue"

    # ── /onboard ──
    if cmd == "/onboard":
        from dazi.lifecycle import load_subsystems
        from dazi.onboard import run_onboarding

        run_onboarding(console)
        await load_subsystems(console=console)
        return "continue"

    # ── /mcp ──
    if cmd == "/mcp":
        show_mcp_servers_table()
        return "continue"

    if cmd.startswith("/mcp "):
        mcp_arg = cmd[5:].strip()
        if mcp_arg.startswith("connect "):
            server_name = mcp_arg[8:].strip()
            if server_name:
                from dazi.mcp_client import MCPServerStatus

                conn = mcp_manager.get_server(server_name)
                if conn and conn.status == MCPServerStatus.CONNECTED:
                    console.print(f"[yellow]{server_name} is already connected.[/yellow]")
                else:
                    console.print(f"[dim]Connecting to {server_name}...[/dim]")
                    success = await mcp_manager.connect_server(server_name)
                    if success:
                        c = mcp_manager.get_server(server_name)
                        console.print(f"[green]+ {server_name} ({len(c.tools)} tools)[/green]")
                        rebuild_tool_lists()
                    else:
                        err = mcp_manager.get_server(server_name)
                        console.print(f"[red]Failed: {err.error if err else 'unknown error'}[/red]")
            else:
                console.print("[red]Usage: /mcp connect <server_name>[/red]")
        elif mcp_arg.startswith("disconnect "):
            server_name = mcp_arg[10:].strip()
            if server_name:
                await mcp_manager.disconnect_server(server_name)
                console.print(f"[dim]Disconnected {server_name}.[/dim]")
                rebuild_tool_lists()
            else:
                console.print("[red]Usage: /mcp disconnect <server_name>[/red]")
        else:
            show_mcp_server_detail(mcp_arg)
        return "continue"

    # ── /clear ──
    if cmd == "/clear":
        state["messages"] = []
        _graph_mod.consecutive_compact_failures = 0
        import os as _os

        _os.system("clear" if _os.name != "nt" else "cls")
        print_welcome_fn()
        console.print("[dim]Cleared. Memories, tasks, and background tasks persist.[/dim]")
        return "continue"

    # ── /plan ──
    if cmd == "/plan":
        if state["mode"] == PLAN_MODE:
            console.print("[blue]Already in plan mode.[/blue]")
            return "continue"
        if PLAN_FILE.exists():
            PLAN_FILE.unlink()
        state["mode"] = PLAN_MODE
        console.print(
            Panel(
                "[bold blue]PLAN MODE[/bold blue]\n"
                "Read-only tools + plan_writer + memory tools "
                "+ task tools + background tools enabled.\n"
                "Type /go to exit plan mode.",
                border_style=BORDER["primary"],
            )
        )
        return "continue"

    # ── /go ──
    if cmd == "/go":
        if state["mode"] == EXECUTE_MODE:
            console.print("[yellow]Not in plan mode.[/yellow]")
            return "continue"
        state["mode"] = EXECUTE_MODE
        if PLAN_FILE.exists():
            plan_content = PLAN_FILE.read_text(encoding="utf-8")
            console.print(
                Panel(
                    Markdown(plan_content),
                    title="[bold green]Plan[/bold green]",
                    border_style=BORDER["success"],
                )
            )
        console.print("[bold green]EXECUTE MODE[/bold green] -- all tools enabled.")
        return "continue"

    # ── /show ──
    if cmd == "/show":
        if not PLAN_FILE.exists():
            console.print("[dim]No plan file found.[/dim]")
            return "continue"
        console.print(
            Panel(
                Markdown(PLAN_FILE.read_text(encoding="utf-8")),
                title="Plan File",
                border_style=BORDER["primary"],
            )
        )
        return "continue"

    # ── /tools ──
    if cmd == "/tools":
        mode = state["mode"]
        meta_dict = PLAN_MODE_META if mode == PLAN_MODE else EXECUTE_MODE_META
        tools_list = PLAN_MODE_TOOLS if mode == PLAN_MODE else EXECUTE_MODE_TOOLS
        safety_tags = {
            "safe": "[green]safe[/green]",
            "write": "[yellow]write[/yellow]",
            "destructive": "[red]destructive[/red]",
        }
        console.print(f"\n[bold]Tools ({mode} mode):[/bold]")
        for tool in tools_list:
            meta = meta_dict.get(tool.name)
            if meta:
                tag = safety_tags.get(meta.safety.value, meta.safety.value)
                concurrent = (
                    "[dim]parallel[/dim]" if meta.is_concurrency_safe else "[dim]serial[/dim]"
                )
                console.print(f"  * {meta.name} -- {meta.description} ({tag}, {concurrent})")
        return "continue"

    # ── /rules ──
    if cmd == "/rules":
        list_rules_table()
        return "continue"

    # ── /allow <rule> ──
    if cmd.startswith("/allow "):
        rule_str = cmd[7:].strip()
        try:
            rule = parse_rule(f"allow {rule_str}", "cli")
            permission_rules.append(rule)
            console.print(
                f"[green]Added rule: ALLOW {rule.tool_name or '*'} {rule.pattern or ''}[/green]"
            )
        except ValueError as e:
            console.print(f"[red]Invalid rule: {e}[/red]")
        return "continue"

    # ── /deny <rule> ──
    if cmd.startswith("/deny "):
        rule_str = cmd[6:].strip()
        try:
            rule = parse_rule(f"deny {rule_str}", "cli")
            permission_rules.append(rule)
            console.print(
                f"[red]Added rule: DENY {rule.tool_name or '*'} {rule.pattern or ''}[/red]"
            )
        except ValueError as e:
            console.print(f"[red]Invalid rule: {e}[/red]")
        return "continue"

    # ── /hooks ──
    if cmd == "/hooks":
        hooks = hook_registry.list_hooks()
        if not hooks:
            console.print("[dim]No hooks registered.[/dim]")
        else:
            for event, priorities in hooks.items():
                console.print(f"  {event}: {len(priorities)} handler(s), priorities: {priorities}")
        return "continue"

    # ── /hook (demo) ──
    if cmd == "/hook":
        add_demo_hook()
        return "continue"

    # ── /remember <content> ──
    if cmd.startswith("/remember "):
        content = cmd[10:].strip()
        if not content:
            console.print("[red]Usage: /remember <content>[/red]")
            return "continue"
        from dazi.memory import MemoryCategory

        memory_entry = MemoryEntry(content=content, category=MemoryCategory("user"))
        memory_store.write(memory_entry)
        console.print(f"[green]Remembered: {memory_entry.id}[/green]")
        return "continue"

    # ── /forget <id> ──
    if cmd.startswith("/forget "):
        mem_id = cmd[8:].strip()
        if not mem_id:
            console.print("[red]Usage: /forget <memory-id>[/red]")
            return "continue"
        entries = memory_store.list_all()
        matches = [e for e in entries if e.id.startswith(mem_id)]
        if len(matches) == 1:
            memory_store.delete(matches[0].id)
            console.print(f"[green]Forgotten: {matches[0].id}[/green]")
        elif len(matches) > 1:
            console.print(
                f"[yellow]Multiple matches: {[e.id for e in matches]}. Be more specific.[/yellow]"
            )
        else:
            if memory_store.delete(mem_id):
                console.print(f"[green]Forgotten: {mem_id}[/green]")
            else:
                console.print(f"[red]Memory not found: {mem_id}[/red]")
        return "continue"

    # ── /memories ──
    if cmd == "/memories":
        list_memories_table()
        return "continue"

    # ── /dazimd ──
    if cmd == "/dazimd":
        show_dazimd_files()
        return "continue"

    # ── /reindex ──
    if cmd == "/reindex":
        memory_store.rebuild_index()
        console.print("[green]Memory index rebuilt.[/green]")
        return "continue"

    # ── /compact ──
    if cmd == "/compact":
        msgs = state.get("messages", [])
        if len(msgs) < 2:
            console.print("[dim]Not enough messages to compact.[/dim]")
            return "continue"

        model = _get_model_name()
        tokens_before = count_messages_tokens(msgs, model)
        console.print(f"[dim]Compressing... ({tokens_before:,} tokens)[/dim]")

        result = await manual_compact(msgs, _get_llm(), model=model)

        if result.method != "none":
            saved = result.tokens_before - result.tokens_after
            console.print(
                f"[green]Compacted ({result.method}):[/green] "
                f"{result.tokens_before:,} -> {result.tokens_after:,} tokens "
                f"(saved {saved:,})"
            )
            state["messages"] = result.messages
        else:
            console.print(f"[dim]{result.summary or 'No compaction needed.'}[/dim]")
        return "continue"

    # ── /tokens ──
    if cmd == "/tokens":
        show_token_info(state.get("messages", []))
        return "continue"

    # ── /tasks ──
    if cmd == "/tasks":
        list_tasks_table(
            active_team_name=_teams.active_team_name,
            default_task_store=task_store,
            team_task_store=_teams.team_task_store,
        )
        return "continue"

    # ── /task <id> ──
    if cmd.startswith("/task "):
        task_id_str = cmd[6:].strip()
        try:
            task_id = int(task_id_str)
            show_task_detail(
                task_id,
                active_team_name=_teams.active_team_name,
                default_task_store=task_store,
                team_task_store=_teams.team_task_store,
            )
        except ValueError:
            console.print("[red]Usage: /task <id>[/red]")
        return "continue"

    # ── /bg ──
    if cmd == "/bg":
        show_background_tasks_table()
        return "continue"

    if cmd.startswith("/bg "):
        show_background_task_detail(cmd[4:].strip())
        return "continue"

    # ── /skills ──
    if cmd == "/skills":
        show_skills_table()
        return "continue"

    # ── /skill <name> ──
    if cmd.startswith("/skill "):
        show_skill_detail(cmd[7:].strip())
        return "continue"

    # ── /teams ──
    if cmd == "/teams":
        _teams.show_teams_table()
        return "continue"

    # ── /team <name> ──
    if cmd.startswith("/team "):
        team_arg = cmd[6:].strip()
        if team_arg.startswith("create "):
            team_name = team_arg[7:].strip()
            if not team_name:
                console.print("[red]Usage: /team create <name>[/red]")
                return "continue"
            try:
                team = team_manager.create_team(team_name)
                console.print(f"[green]Team created: {team.name}[/green]")
                console.print(f"[dim]Config: {team_manager._config_path(team_name)}[/dim]")
                console.print(f"[dim]Tasks:  {team_manager._task_dir(team_name)}[/dim]")
            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")
            return "continue"
        elif team_arg == "leave":
            if _teams.active_team_name:
                _teams.deactivate_team()
            else:
                console.print("[dim]No active team to leave.[/dim]")
            return "continue"
        elif team_arg.startswith("delete "):
            team_name = team_arg[7:].strip()
            if not team_name:
                console.print("[red]Usage: /team delete <name>[/red]")
                return "continue"
            if _teams.active_team_name == team_name:
                _teams.deactivate_team()
            try:
                if team_manager.delete_team(team_name):
                    console.print(f"[green]Team deleted: {team_name}[/green]")
                else:
                    console.print(f"[red]Team '{team_name}' not found.[/red]")
            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")
            return "continue"
        else:
            if team_arg:
                _teams.activate_team(team_arg)
            else:
                _teams.show_teams_table()
            return "continue"

    # ── /proactive ──
    if cmd == "/proactive" or cmd.startswith("/proactive "):
        proactive_arg = cmd[10:].strip() if len(cmd) > 10 else ""
        if proactive_arg == "on":
            proactive_manager.activate(source=ProactiveSource.COMMAND)
            _update_proactive_prompt()
            console.print(
                Panel(
                    "[bold green]PROACTIVE MODE ON[/bold green]\n"
                    "The agent will wake up periodically via tick prompts.\n"
                    "Use /proactive off to stop. Ctrl+C pauses ticks.",
                    border_style=BORDER["success"],
                )
            )
        elif proactive_arg == "off":
            proactive_manager.deactivate()
            _update_proactive_prompt()
            console.print("[dim]Proactive mode off.[/dim]")
        else:
            state_desc = proactive_manager.state.value
            source_desc = proactive_manager.source.value if proactive_manager.source else "none"
            count = proactive_manager.activation_count
            if state_desc == "inactive":
                console.print(f"[dim]Proactive: {state_desc} | Use /proactive on to activate[/dim]")
            else:
                first = "yes" if proactive_manager.is_first_tick else "no"
                last_tick = proactive_manager.last_tick_time or "never"
                console.print(
                    f"[bold]Proactive:[/bold] {state_desc} | Source: {source_desc} | "
                    f"Activations: {count} | First tick: {first} | Last tick: {last_tick}"
                )
        return "continue"

    # ── /autonomous ──
    if cmd == "/autonomous" or cmd.startswith("/autonomous "):
        autonomous_handles = autonomous_teammate.list_handles()
        if not autonomous_handles:
            console.print("[dim]No autonomous teammates running.[/dim]")
        else:
            table = Table(title="Autonomous Teammates", show_lines=True)
            table.add_column("Name", style="cyan")
            table.add_column("Team", style="green")
            table.add_column("Status", style="yellow")
            table.add_column("Tasks Claimed", style="magenta")
            for h in autonomous_handles:
                claimed = autonomous_teammate._tasks_claimed.get(h.name, 0)
                table.add_row(h.name, h.team_name, h.status.value, str(claimed))
            console.print(table)
        return "continue"

    # ── /worktree ──
    if cmd == "/worktree" or cmd.startswith("/worktree "):
        wt_arg = cmd[10:].strip() if len(cmd) > 10 else ""
        if wt_arg.startswith("create "):
            wt_name = wt_arg[7:].strip()
            if not wt_name:
                console.print("[red]Usage: /worktree create <name>[/red]")
                return "continue"
            try:
                wt = worktree_manager.create(wt_name)
                console.print(f"[green]Created worktree:[/green] {wt.path} on branch {wt.branch}")
                console.print(f"[dim]Use /worktree finish {wt_name} when done.[/dim]")
            except (ValueError, RuntimeError) as e:
                console.print(f"[red]Error: {e}[/red]")
        elif wt_arg.startswith("finish "):
            finish_parts = wt_arg[7:].strip().split()
            if not finish_parts:
                console.print("[red]Usage: /worktree finish <name> [--keep|--remove][/red]")
                return "continue"
            finish_name = finish_parts[0]
            action_flag = finish_parts[1] if len(finish_parts) > 1 else "--keep"
            slug = worktree_manager.sanitize_agent_name(finish_name)
            wt = worktree_manager.get(slug)
            if wt is None:
                console.print(f"[red]No worktree found for '{finish_name}'.[/red]")
                return "continue"
            action = action_flag[2:] if action_flag in ("--keep", "--remove") else "keep"
            if action == "keep":
                branch = worktree_manager.keep(slug)
                console.print(
                    f"[green]Kept worktree:[/green] branch '{branch}' preserved at {wt.path}"
                )
            elif action == "remove":
                if worktree_manager.has_uncommitted_changes(slug):
                    console.print(
                        "[yellow]Worktree has uncommitted changes. "
                        "Use --remove with --force or keep instead.[/yellow]"
                    )
                    worktree_manager.remove(slug, force=False)
                else:
                    removed = worktree_manager.remove(slug, force=True)
                    if removed:
                        console.print(f"[green]Removed worktree:[/green] {finish_name}")
                    else:
                        console.print("[red]Failed to remove worktree.[/red]")
        else:
            active_worktrees = worktree_manager.list_all()
            if not active_worktrees:
                console.print("[dim]No active worktrees.[/dim]")
            else:
                table = Table(title=f"Worktrees ({len(active_worktrees)})", show_lines=True)
                table.add_column("Name", style="cyan")
                table.add_column("Branch", style="green")
                table.add_column("Path", style="dim")
                table.add_column("Dirty")
                for wt in active_worktrees:
                    dirty = (
                        "[yellow]yes[/yellow]"
                        if worktree_manager.has_uncommitted_changes(wt.id)
                        else "[green]no[/green]"
                    )
                    table.add_row(wt.agent_name, wt.branch, str(wt.path), dirty)
                console.print(table)
        return "continue"

    # ── /inbox ──
    if cmd == "/inbox" or cmd.startswith("/inbox "):
        inbox_agent = cmd[7:].strip() if cmd.startswith("/inbox ") else None
        await _teams.show_inbox(inbox_agent)
        return "continue"

    # ── /send ──
    if cmd.startswith("/send "):
        parts = cmd[6:].strip().split(None, 1)
        if len(parts) < 2:
            console.print("[red]Usage: /send <agent-name> <message>[/red]")
        else:
            await _teams.send_repl_message(parts[0], parts[1])
        return "continue"

    # ── /broadcast ──
    if cmd.startswith("/broadcast "):
        msg_text = cmd[11:].strip()
        if not msg_text:
            console.print("[red]Usage: /broadcast <message>[/red]")
        else:
            await _teams.broadcast_repl_message(msg_text)
        return "continue"

    # ── /shutdown ──
    if cmd.startswith("/shutdown "):
        agent = cmd[10:].strip()
        if not agent:
            console.print("[red]Usage: /shutdown <agent-name>[/red]")
        else:
            await _teams.send_shutdown_request(agent)
        return "continue"

    # ── Dynamic MCP server slash commands ──
    if cmd.startswith("/"):
        parts = cmd.split(None, 1)
        server_name = parts[0][1:]  # strip leading /
        server = mcp_manager.get_server(server_name)
        if server and server.status.value == "connected":
            tools = server.tools
            if len(tools) == 1:
                try:
                    result = await mcp_manager.call_tool(tools[0].qualified_name, {})
                    console.print(Panel(result, title=server_name, border_style=BORDER["primary"]))
                except Exception as e:
                    console.print(f"[red]Error: {e}[/red]")
                return "continue"
            elif len(tools) > 1:
                console.print(f"[bold]{server_name}[/bold] has {len(tools)} tools:")
                for t in tools:
                    console.print(f"  /{server_name} {t.name} — {t.description[:80]}")
                return "continue"

    # ── Slash command expansion (skill invocation) ──
    if cmd.startswith("/"):
        from langchain_core.messages import HumanMessage, SystemMessage

        from dazi.graph import run_graph_turn

        parts = cmd.split(None, 1)
        skill_name = parts[0][1:]
        skill_args = parts[1] if len(parts) > 1 else ""

        if skill_registry.has_skill(skill_name):
            skill = skill_registry.get(skill_name)
            if not skill.user_invocable:
                console.print(f"[red]Skill '{skill_name}' is not user-invocable.[/red]")
                return "continue"

            expanded = skill_registry.expand_skill(skill_name, skill_args)
            console.print(f"[dim]Expanded skill: /{skill_name}[/dim]")

            messages = state.get("messages", [])
            messages = [m for m in messages if not isinstance(m, SystemMessage)]
            messages.append(HumanMessage(content=expanded))
            state["messages"] = messages

            await run_graph_turn(
                messages=messages,
                state=state,
                session=session,
                status_label=f"Thinking... (skill: {skill_name})",
            )
            return "continue"

    # Not a built-in command
    return None
