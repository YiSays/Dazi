"""Dazi — coding assistant REPL."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from prompt_toolkit import PromptSession
    from prompt_toolkit.formatted_text import FormattedText

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.errors import GraphInterrupt
from rich.console import Console

import dazi.repl_teams as _teams
from dazi import __version__
from dazi._singletons import (
    BACKGROUND_DIR,
    MEMORY_DIR,
    TASKS_DIR,
    autonomous_teammate,
    background_manager,
    cost_tracker,
    mcp_manager,
    memory_store,
    proactive_manager,
    settings_manager,
    task_store,
    team_manager,
    worktree_manager,
)
from dazi.config import DATA_DIR
from dazi.graph import (
    EXECUTE_MODE,
    permission_rules,
    run_graph_turn,
)
from dazi.lifecycle import cleanup_on_exit, load_subsystems
from dazi.llm import _get_model_name
from dazi.proactive import ProactiveSource, format_tick
from dazi.prompt_builder import _update_proactive_prompt
from dazi.repl_commands import handle_command
from dazi.repl_display import get_mode_badge, print_ascii_banner, print_welcome_message
from dazi.tokenizer import (
    count_messages_tokens,
    get_context_window,
    get_token_warning_state,
)

console = Console()


# ─────────────────────────────────────────────────────────
# REPL LOOP
# ─────────────────────────────────────────────────────────


async def _prompt_with_background_watcher(
    session: PromptSession,
    formatted_text: FormattedText,
    state: dict,
) -> str | None:
    """Await user input, racing against background task completion.

    Returns the user input string, or None if a background task completed
    while the user was idle (notifications already displayed).
    """
    import asyncio

    from dazi.graph import display_background_notifications

    bg_event = background_manager.completion_event
    prompt_coro = session.prompt_async(formatted_text)
    completion_coro = bg_event.wait()

    done, pending = await asyncio.wait(
        [asyncio.ensure_future(prompt_coro), asyncio.ensure_future(completion_coro)],
        return_when=asyncio.FIRST_COMPLETED,
    )

    # Cancel whichever didn't finish.
    # NOTE: If the user was mid-typing when a completion fires, their partial
    # input is discarded. A future improvement could preserve the draft.
    for task in pending:
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass

    # Figure out which won
    for task in done:
        if task.done() and not task.cancelled():
            try:
                result = task.result()
            except Exception:
                continue

            # Check if this was the prompt (returns string) or the event (returns True)
            if isinstance(result, str):
                return result

    # Background completion won — display notifications
    completed = background_manager.collect_completed()
    if completed:
        display_background_notifications(completed)
    return None


async def run_repl() -> None:
    import asyncio

    from prompt_toolkit import PromptSession
    from prompt_toolkit.formatted_text import FormattedText

    from dazi.repl_completer import get_prompt_session_kwargs
    from dazi.theme import PROMPT as _P

    # Onboarding: check if required settings are present
    s = settings_manager.settings
    if not s.api_key or not s.api_base_url or not s.model:
        from dazi.onboard import run_onboarding

        run_onboarding(console)
        settings_manager.reload()

    # Ensure directories exist
    MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    TASKS_DIR.mkdir(parents=True, exist_ok=True)
    BACKGROUND_DIR.mkdir(parents=True, exist_ok=True)

    # Load subsystems (DAZI.md, settings, skills, MCP)
    result = await load_subsystems(console=console)
    skill_count = result.skill_count

    # Count existing teams
    team_count = len(team_manager.list_teams())

    # Check for proactive env var activation
    import os as _os

    if _os.getenv("DAZI_PROACTIVE", "").lower() in ("1", "true", "on"):
        proactive_manager.activate(source=ProactiveSource.ENV)
        _update_proactive_prompt()
        console.print("[dim]Proactive mode activated via DAZI_PROACTIVE env var.[/dim]")

    def print_welcome():
        print_welcome_message(
            console, skill_count=skill_count, team_count=team_count
        )

    print_ascii_banner(console, version=__version__)
    print_welcome()

    # Create .dazi directory if it doesn't exist
    (DATA_DIR / "chat").mkdir(parents=True, exist_ok=True)
    state: dict = {"mode": EXECUTE_MODE, "messages": []}
    session = PromptSession(**get_prompt_session_kwargs(state))

    try:
        while True:
            try:
                # ── Build status bar ──
                mode_badge = get_mode_badge(state["mode"])
                rule_count = len(permission_rules)
                mem_count = len(memory_store.list_all())
                active_store = _teams.team_task_store if _teams.active_team_name else task_store
                tsk_count = len(active_store.list_all())
                bg_active = len(background_manager.list_active())

                model = _get_model_name()
                msgs = state.get("messages", [])
                display_msgs = [m for m in msgs if not isinstance(m, SystemMessage)]
                token_count = count_messages_tokens(display_msgs, model) if display_msgs else 0
                context_window = get_context_window(model)
                token_pct = (token_count / context_window * 100) if context_window > 0 else 0

                warning_state = (
                    get_token_warning_state(display_msgs, model) if display_msgs else "ok"
                )
                token_style = {
                    "ok": _P["token_ok"],
                    "warning": _P["token_warning"],
                    "compact": _P["token_compact"],
                }.get(warning_state, "fg:white")

                # Build dot-separated segments
                segments: list[tuple[str, str]] = [
                    *mode_badge,
                    ("", " "),
                    (token_style, f"{token_pct:.0f}%"),
                ]

                # Special badges (PROACTIVE, AUTONOMOUS, WT)
                if proactive_manager.is_proactive_active():
                    badge = "PAUSED" if proactive_manager.is_proactive_paused() else "ACTIVE"
                    segments += [(_P["separator"], " \u00b7 "), (_P["mode_plan"], f"PRO:{badge}")]
                autonomous_handles = autonomous_teammate.list_handles()
                if autonomous_handles:
                    active_count = len(
                        [
                            h
                            for h in autonomous_handles
                            if h.status.value in ("active", "idle", "spawning")
                        ]
                    )
                    segments += [
                        (_P["separator"], " \u00b7 "),
                        (_P["primary"], f"AUTO:{active_count}"),
                    ]
                active_worktrees = worktree_manager.list_all()
                if active_worktrees:
                    segments += [
                        (_P["separator"], " \u00b7 "),
                        (_P["primary"], f"WT:{len(active_worktrees)}"),
                    ]

                # Optional info items
                optional_items: list[str] = []
                if rule_count:
                    optional_items.append(f"{rule_count} rules")
                if mem_count:
                    optional_items.append(f"{mem_count} mem")
                if tsk_count:
                    optional_items.append(f"{tsk_count} tasks")
                if bg_active:
                    optional_items.append(f"{bg_active} bg")
                mcp_tools_count = len(mcp_manager.get_tools())
                if mcp_tools_count:
                    optional_items.append(f"{mcp_tools_count} mcp")
                if _teams.active_team_name:
                    optional_items.append(_teams.active_team_name)
                optional_items.append(cost_tracker.format_cost())

                for item in optional_items:
                    segments += [(_P["separator"], " \u00b7 "), (_P["dim"], item)]

                # Prompt line
                segments += [("", "\n")]
                if _teams.active_team_name:
                    segments += [
                        (_P["primary"], _teams.active_team_name),
                        (_P["separator"], " "),
                    ]
                segments += [
                    ("fg:#ff8c00", "\u276f "),
                ]

                user_input = await _prompt_with_background_watcher(
                    session, FormattedText(segments), state
                )
                if user_input is None:
                    # Background completion fired — notifications already displayed.
                    # Re-render the prompt for next input.
                    continue
                if not user_input.strip():
                    continue

                # ── Clear prompt_toolkit output and re-render as chat bubble ──
                from dazi.repl_display import render_user_panel
                from dazi.terminal import clear_lines, count_prompt_lines

                n_lines = count_prompt_lines(segments, user_input, console.width)
                clear_lines(n_lines)
                render_user_panel(user_input, console)

                cmd = user_input.strip()

                # ── Try built-in slash commands ──
                result = await handle_command(
                    cmd,
                    state=state,
                    session=session,
                    console=console,
                    print_welcome_fn=print_welcome,
                )
                if result == "break":
                    break
                if result == "continue":
                    continue

                # ── Regular input: send to graph ──
                messages = state.get("messages", [])
                messages = [m for m in messages if not isinstance(m, SystemMessage)]
                messages.append(HumanMessage(content=user_input))
                state["messages"] = messages

                # Resume proactive on user input
                if proactive_manager.is_proactive_active():
                    proactive_manager.resume()
                _update_proactive_prompt()

                await run_graph_turn(
                    messages=messages,
                    state=state,
                    session=session,
                    status_label=f"Thinking... ({state['mode']} mode)",
                )

                # ── Proactive tick injection ──
                while proactive_manager.should_generate_tick():
                    await asyncio.sleep(0)
                    proactive_manager.mark_tick_sent()

                    tick_content = format_tick()
                    tick_msg = HumanMessage(
                        content=tick_content,
                        additional_kwargs={"is_meta": True, "is_tick": True},
                    )

                    _update_proactive_prompt()
                    tick_messages = state.get("messages", [])
                    tick_messages = [m for m in tick_messages if not isinstance(m, SystemMessage)]
                    tick_messages.append(tick_msg)

                    await run_graph_turn(
                        messages=tick_messages,
                        state=state,
                        session=session,
                        status_label="Thinking... (tick)",
                        label_suffix=" (tick)",
                    )

            except GraphInterrupt:
                console.print(
                    "\n[yellow]Interrupt escaped graph — this should not happen.[/yellow]"
                )
            except KeyboardInterrupt:
                if proactive_manager.is_proactive_active():
                    proactive_manager.pause()
                    console.print(
                        "\n[dim]Proactive mode paused. "
                        "Submit input to resume. Use /quit to exit.[/dim]"
                    )
                # If during model streaming, the cancellation message was already printed
                # in _stream_and_display, so just continue to the next prompt.
            except asyncio.CancelledError:
                import traceback as _tb

                console.print("\n[red]CancelledError — traceback:[/red]")
                _tb.print_exc()
                console.print("[dim]Shutting down...[/dim]")
                break
            except Exception as e:
                console.print(f"\n[red]Error: {e}[/red]")

    finally:
        await cleanup_on_exit(console=console, active_team_name=_teams.active_team_name)


if __name__ == "__main__":
    import asyncio

    asyncio.run(run_repl())
