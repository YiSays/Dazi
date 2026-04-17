"""First-time onboarding wizard for Dazi.

Runs automatically when required settings (api_key, model, api_base_url) are
missing. Can be re-triggered via /onboard command.
"""

from __future__ import annotations

import json
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, Prompt

from dazi.settings import DaziSettings
from dazi.tokenizer import MODEL_CONTEXT_LIMITS, get_context_window

# ─────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────


def _mask_key(key: str, visible: int = 4) -> str:
    """Show last `visible` chars of a key, mask the rest."""
    if len(key) <= visible:
        return key
    return "*" * (len(key) - visible) + key[-visible:]


def _print_header(console: Console, *, is_rerun: bool) -> None:
    """Print onboarding header panel."""
    if is_rerun:
        title = "[bold cyan]DAZI Settings[/bold cyan]"
        subtitle = "Update your configuration"
    else:
        title = "[bold cyan]Welcome to DAZI[/bold cyan]"
        subtitle = (
            "Develop Autonomously, Zero Interruption.\n"
            "Let's set up your configuration. "
            "Required fields cannot be skipped on first run."
        )
    console.print()
    console.print(
        Panel(
            f"[bold]{subtitle}[/bold]",
            title=title,
            border_style="cyan",
            padding=(1, 2),
        )
    )
    console.print()


def _prompt_value(
    console: Console,
    label: str,
    *,
    default: str | None = None,
    required: bool = False,
    password: bool = False,
) -> str | None:
    """Prompt user for a value with default display.

    Returns None if skipped (only possible when not required).
    """
    default_display = default if default else None
    prompt_text = f"  {label}"
    if default_display:
        hint = _mask_key(default_display) if password else default_display
        prompt_text += f" [{hint}]"

    while True:
        try:
            if password:
                # Don't pass default to Prompt.ask — it would print the raw value.
                # Handle empty input manually to return the existing default.
                value = Prompt.ask(prompt_text, password=password)
            else:
                value = Prompt.ask(
                    prompt_text,
                    default=default_display or "",
                )
        except (KeyboardInterrupt, EOFError):
            if required:
                console.print("[yellow]  This field is required.[/yellow]")
                continue
            console.print("[dim]  Skipped.[/dim]")
            return None

        value = value.strip()
        if not value:
            if default:
                return default
            if required:
                console.print("[yellow]  This field is required.[/yellow]")
                continue
            return None
        return value


# ─────────────────────────────────────────────────────────
# STEP FUNCTIONS
# ─────────────────────────────────────────────────────────


def _step_api_key(console: Console, current: DaziSettings, *, required: bool) -> str | None:
    """Step 1: API Key."""
    console.print("[bold cyan]1. API Key[/bold cyan]")
    console.print("[dim]  Your OpenAI-compatible API key.[/dim]")
    value = _prompt_value(
        console,
        "API key",
        default=current.api_key,
        required=required,
        password=True,
    )
    if value:
        console.print("[green]  ✓[/green] set")
    return value


def _step_model(console: Console, current: DaziSettings, *, required: bool) -> str | None:
    """Step 2: Model Name."""
    console.print()
    console.print("[bold cyan]2. Model Name[/bold cyan]")
    known = ", ".join(sorted(MODEL_CONTEXT_LIMITS.keys()))
    console.print(f"[dim]  Known models: {known}[/dim]")
    value = _prompt_value(
        console,
        "Model name",
        default=current.model,
        required=required,
    )
    if value:
        console.print(f"[green]  ✓[/green] {value}")
    return value


def _step_base_url(console: Console, current: DaziSettings, *, required: bool) -> str | None:
    """Step 3: Base URL."""
    console.print()
    console.print("[bold cyan]3. API Base URL[/bold cyan]")
    console.print("[dim]  OpenAI-compatible endpoint (e.g., https://api.openai.com/v1).[/dim]")

    while True:
        value = _prompt_value(
            console,
            "Base URL",
            default=current.api_base_url,
            required=required,
        )
        if not value:
            return None
        if value.startswith(("http://", "https://")):
            console.print(f"[green]  ✓[/green] {value}")
            return value
        console.print("[yellow]  URL must start with http:// or https://[/yellow]")


def _step_token_window(console: Console, current: DaziSettings, model: str | None) -> int | None:
    """Step 4: Token Window override (optional)."""
    console.print()
    console.print("[bold cyan]4. Token Window[/bold cyan] (optional)")

    detected = get_context_window(model or "")
    existing = current.context_window
    effective = existing if existing is not None else detected

    console.print(f"[dim]  Auto-detected: {detected:,} tokens[/dim]")
    if existing is not None:
        console.print(f"[dim]  Current override: {existing:,} tokens[/dim]")

    value = _prompt_value(
        console,
        f"Token window (Enter to keep {effective:,})",
        default=str(effective),
        required=False,
    )
    if not value:
        return existing
    try:
        tokens = int(value.replace(",", ""))
        if tokens > 0:
            console.print(f"[green]  ✓[/green] {tokens:,} tokens")
            return tokens
    except ValueError:
        pass
    console.print(f"[yellow]  Invalid number. Using {effective:,}.[/yellow]")
    return existing


def _step_mcp(console: Console, current: DaziSettings) -> dict[str, dict]:
    """Step 5: MCP Servers via JSON paste."""
    console.print()
    console.print("[bold cyan]5. MCP Servers[/bold cyan] (optional)")

    existing = current.mcp_servers or {}
    if existing:
        names = ", ".join(existing.keys())
        console.print(f"[dim]  Existing servers: {names}[/dim]")

    console.print("[dim]  Paste a JSON object in mcpServers format. Example:[/dim]")
    example = {
        "my-server": {
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
            "env": {"API_KEY": "your-key"},
        },
    }
    console.print(f"[dim]  {json.dumps(example, indent=2)}[/dim]")

    if not Confirm.ask("  Configure MCP servers?", default=False):
        return existing

    console.print("[dim]  Paste JSON (press Enter twice to finish):[/dim]")
    lines: list[str] = []
    empty_count = 0
    while True:
        try:
            line = input()
        except EOFError:
            break
        if not line.strip():
            empty_count += 1
            if empty_count >= 2:
                break
            lines.append(line)
        else:
            empty_count = 0
            lines.append(line)

    raw = "\n".join(lines).strip()
    if not raw:
        return existing

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as e:
        console.print(f"[red]  Invalid JSON: {e}[/red]")
        console.print("[yellow]  Keeping existing MCP config.[/yellow]")
        return existing

    if not isinstance(parsed, dict):
        console.print("[red]  JSON must be an object (dict).[/red]")
        return existing

    # Unwrap standard MCP format: {"mcpServers": {...}}
    if "mcpServers" in parsed and isinstance(parsed.get("mcpServers"), dict):
        parsed = parsed["mcpServers"]
    elif "mcp_servers" in parsed and isinstance(parsed.get("mcp_servers"), dict):
        parsed = parsed["mcp_servers"]

    merged = {**existing, **parsed}
    console.print(f"[green]  ✓[/green] {len(merged)} server(s) configured")
    return merged


def _step_dazimd(console: Console, *, dazimd_path: Path | None = None) -> None:
    """Step 6: DAZI.md template creation."""
    console.print()
    console.print("[bold cyan]6. DAZI.md[/bold cyan] (optional)")
    console.print("[dim]  Project-specific instructions loaded into every conversation.[/dim]")

    dazimd_path = dazimd_path or Path.home() / ".dazi" / "DAZI.md"
    if dazimd_path.exists():
        console.print(f"[dim]  Already exists: {dazimd_path}[/dim]")
        if not Confirm.ask("  Overwrite?", default=False):
            return

    if not Confirm.ask("  Create template DAZI.md?", default=True):
        return

    template = """\
# DAZI.md — Your Global Instructions

<!-- This file is loaded as the lowest-priority instruction layer -->
<!-- for every DAZI session. Project-level files override this. -->
<!-- Priority: DAZI.local.md (400) > DAZI.md (300) > this file (100) -->
<!-- Supports @include directives to compose from other files. -->
<!-- Example: @include ~/dazi/rules/coding-style.md -->
<!-- -->
<!-- HTML comments (like this) are invisible to the LLM. -->
<!-- Visible text below is injected into the system prompt. -->
<!-- Keep visible text concise — every token counts. -->

My name is DAZI (Develop Autonomously, Zero Interruption).
I am a terminal-based AI coding assistant built to help with software
engineering tasks: writing and editing code, debugging, refactoring,
running commands, and coordinating multi-agent teams.

## File Locations
Project-level (inside `.dazi/` under project root):
- `settings.json` — project settings
- `plans/plan.md` — plan mode output
- `tasks/default/*.json` — task files
- `teams/<name>/config.json` — team configuration
- `teams/<name>/inboxes/*.json` — agent mailboxes
- `memory/MEMORY.md` — memory index
- `memory/*.md` — individual memory files
- `skills/<name>/SKILL.md` — project skill definitions
- `background/*.output` — background task output
- `worktrees/<slug>/` — git worktree directories
User-level (under `~/.dazi/`):
- `settings.json` — global user settings
- `DAZI.md` — this file (global instructions)
- `skills/<name>/SKILL.md` — user skill definitions
Project root:
- `DAZI.md` — project instructions (priority 300)
- `DAZI.local.md` — private project instructions (priority 400)
- `.env` — environment variables

## About the User
<!-- Tell DAZI about yourself so responses are tailored -->
<!-- e.g., "Senior Python developer working on web backends" -->

## Coding Style
<!-- e.g., "Prefer functional style over OOP" -->
<!-- e.g., "Use type hints everywhere. No Any without a comment." -->

## Tools and Environment
<!-- e.g., "Always use uv for Python package management" -->
<!-- e.g., "Use ruff for linting and formatting" -->

## Communication Preferences
<!-- e.g., "When I paste an error, always try to fix it — don't just explain" -->
<!-- e.g., "Show me the diff summary, not the full file content" -->

## Conventions
<!-- e.g., "Use snake_case for Python, camelCase for TypeScript" -->
<!-- e.g., "Keep functions under 30 lines" -->
"""
    dazimd_path.write_text(template, encoding="utf-8")
    console.print(f"[green]  ✓[/green] Created {dazimd_path}")


# ─────────────────────────────────────────────────────────
# MAIN ENTRY
# ─────────────────────────────────────────────────────────


def run_onboarding(console: Console) -> None:
    """Run the onboarding wizard.

    Reads current settings, walks through each step, and saves
    the result to ~/.dazi/settings.json.
    """
    from dazi._singletons import settings_manager

    current = settings_manager.settings
    is_rerun = bool(current.api_key and current.api_base_url and current.model)

    _print_header(console, is_rerun=is_rerun)

    required = not is_rerun

    # Required steps
    api_key = _step_api_key(console, current, required=required)
    model = _step_model(console, current, required=required)
    base_url = _step_base_url(console, current, required=required)

    # Optional steps
    context_window = _step_token_window(console, current, model)
    mcp_servers = _step_mcp(console, current)
    _step_dazimd(console)

    # Build settings — preserve existing fields for non-asked items
    new_settings = DaziSettings(
        api_key=api_key or current.api_key,
        model=model or current.model,
        api_base_url=base_url or current.api_base_url,
        default_mode=current.default_mode,
        allow_rules=current.allow_rules,
        deny_rules=current.deny_rules,
        env=current.env,
        auto_compact=current.auto_compact,
        auto_memory=current.auto_memory,
        max_concurrent_tools=current.max_concurrent_tools,
        mcp_servers=mcp_servers,
        context_window=context_window,
    )

    # Summary
    console.print()
    console.print("[bold]─── Configuration Summary ───[/bold]")
    console.print(f"  Model:        {new_settings.model}")
    console.print(f"  API Key:      {_mask_key(new_settings.api_key or '')}")
    console.print(f"  Base URL:     {new_settings.api_base_url}")
    cw = new_settings.context_window or get_context_window(new_settings.model or "")
    console.print(f"  Token Window: {cw:,} tokens")
    if new_settings.mcp_servers:
        console.print(f"  MCP Servers:  {', '.join(new_settings.mcp_servers.keys())}")
    console.print()

    # Save
    settings_manager.save_user_settings(new_settings)
    console.print("[green]✓ Settings saved to ~/.dazi/settings.json[/green]")
    console.print("[dim]Starting DAZI...[/dim]")
    console.print()
