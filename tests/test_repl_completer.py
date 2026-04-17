"""Tests for the slash-command completer, /help display, and key bindings."""

from __future__ import annotations

from io import StringIO
from unittest.mock import MagicMock

from prompt_toolkit.completion import CompleteEvent
from prompt_toolkit.document import Document
from prompt_toolkit.keys import Keys
from rich.console import Console

from dazi.repl_completer import (
    COMMAND_REGISTRY,
    SlashCommandCompleter,
    _build_repl_key_bindings,
    print_help,
)


def _make_event() -> CompleteEvent:
    return CompleteEvent(completion_requested=True)


def _complete(completer: SlashCommandCompleter, text: str) -> list[str]:
    """Helper: return all completion texts for a given input."""
    doc = Document(text, len(text))
    return [c.text for c in completer.get_completions(doc, _make_event())]


def _complete_display(completer: SlashCommandCompleter, text: str) -> list[str]:
    """Helper: return all display strings for a given input."""
    doc = Document(text, len(text))
    results = []
    for c in completer.get_completions(doc, _make_event()):
        # display may be str or FormattedText
        if isinstance(c.display, str):
            results.append(c.display)
        else:
            # FormattedText: extract plain text from (style, text) tuples
            results.append("".join(part[1] for part in c.display))
    return results


class TestSlashCommandCompleter:
    def setup_method(self):
        self.completer = SlashCommandCompleter()

    def test_no_completion_for_plain_text(self):
        assert _complete(self.completer, "hello") == []

    def test_no_completion_for_empty(self):
        assert _complete(self.completer, "") == []

    def test_completes_from_slash(self):
        results = _complete_display(self.completer, "/")
        assert "/help" in results
        assert "/quit" in results
        assert "/plan" in results

    def test_completes_partial_command(self):
        results = _complete(self.completer, "/qu")
        assert "/quit" in results

    def test_completes_partial_cost(self):
        results = _complete(self.completer, "/co")
        texts = "".join(results)
        assert "st" in texts  # /cost
        assert "mpact" in texts  # /compact

    def test_no_completions_after_full_match(self):
        # "/quit" is a complete command with no subcommands — yields itself
        results = _complete(self.completer, "/quit")
        assert "/quit" in results
        assert len(results) == 1

    def test_mcp_subcommands_on_space(self):
        results = _complete_display(self.completer, "/mcp ")
        assert any("connect" in r for r in results)
        assert any("disconnect" in r for r in results)

    def test_mcp_partial_subcommand(self):
        results = _complete(self.completer, "/mcp con")
        assert any("nect" in r for r in results)

    def test_team_subcommands(self):
        results = _complete_display(self.completer, "/team ")
        assert any("create" in r for r in results)
        assert any("delete" in r for r in results)
        assert any("leave" in r for r in results)

    def test_worktree_subcommands(self):
        results = _complete_display(self.completer, "/worktree ")
        assert any("create" in r for r in results)
        assert any("finish" in r for r in results)

    def test_display_meta_includes_category(self):
        doc = Document("/quit", 5)
        completions = list(self.completer.get_completions(doc, _make_event()))
        assert len(completions) == 1
        meta = completions[0].display_meta
        meta_str = "".join(part[1] for part in meta) if not isinstance(meta, str) else meta
        assert "Core" in meta_str


class TestCommandRegistry:
    REQUIRED_COMMANDS = (
        "quit",
        "clear",
        "cost",
        "settings",
        "reload",
        "help",
        "plan",
        "go",
        "show",
        "tools",
        "rules",
        "allow",
        "deny",
        "hooks",
        "hook",
        "remember",
        "forget",
        "memories",
        "reindex",
        "tasks",
        "task",
        "bg",
        "mcp",
        "skills",
        "skill",
        "teams",
        "team",
        "inbox",
        "send",
        "broadcast",
        "shutdown",
        "proactive",
        "autonomous",
        "worktree",
        "dazimd",
        "compact",
        "tokens",
    )

    def test_registry_covers_all_commands(self):
        registry_names = {c.name.lstrip("/") for c in COMMAND_REGISTRY}
        for cmd in self.REQUIRED_COMMANDS:
            assert cmd in registry_names, f"/{cmd} missing from COMMAND_REGISTRY"

    def test_no_duplicate_names(self):
        names = [c.name for c in COMMAND_REGISTRY]
        assert len(names) == len(set(names)), "Duplicate command names in registry"

    def test_subcommands_have_parent(self):
        parents_with_subs = {c.name for c in COMMAND_REGISTRY if c.subcommands}
        assert "/mcp" in parents_with_subs
        assert "/team" in parents_with_subs
        assert "/worktree" in parents_with_subs

    def test_all_entries_have_category(self):
        for entry in COMMAND_REGISTRY:
            assert entry.category, f"{entry.name} missing category"
            for sub in entry.subcommands:
                assert sub.category, f"{sub.name} missing category"


class TestPrintHelp:
    def test_print_help_runs_without_error(self):
        output = StringIO()
        console = Console(file=output, force_terminal=False)
        print_help(console)
        rendered = output.getvalue()
        assert "Core" in rendered
        assert "Mode" in rendered
        assert "/help" in rendered

    def test_help_includes_all_categories(self):
        output = StringIO()
        console = Console(file=output, force_terminal=False)
        print_help(console)
        rendered = output.getvalue()
        for category in ["Core", "Mode", "Permissions", "MCP", "Teams", "Context"]:
            assert category in rendered


# ─────────────────────────────────────────────────────────
# Key bindings
# ─────────────────────────────────────────────────────────


def _find_handler(kb, *keys):
    """Find a binding handler by its key sequence."""
    for binding in kb.bindings:
        if tuple(binding.keys) == tuple(keys):
            return binding.handler
    return None


def _mock_event():
    """Create a mock event with a buffer."""
    buf = MagicMock()
    buf.text = "some input"
    event = MagicMock()
    event.current_buffer = buf
    return event, buf


class TestReplKeyBindings:
    def test_ctrl_q_submits_quit(self):
        kb = _build_repl_key_bindings({"mode": "execute"})
        handler = _find_handler(kb, Keys.ControlQ)
        assert handler is not None

        event, buf = _mock_event()
        handler(event)
        assert buf.text == "/quit"
        buf.validate_and_handle.assert_called_once()

    def test_shift_tab_toggles_to_plan_in_execute_mode(self):
        kb = _build_repl_key_bindings({"mode": "execute"})
        handler = _find_handler(kb, Keys.BackTab)
        assert handler is not None

        event, buf = _mock_event()
        handler(event)
        assert buf.text == "/plan"
        buf.validate_and_handle.assert_called_once()

    def test_shift_tab_toggles_to_go_in_plan_mode(self):
        kb = _build_repl_key_bindings({"mode": "plan"})
        handler = _find_handler(kb, Keys.BackTab)
        assert handler is not None

        event, buf = _mock_event()
        handler(event)
        assert buf.text == "/go"
        buf.validate_and_handle.assert_called_once()

    def test_double_esc_clears_input(self):
        kb = _build_repl_key_bindings({"mode": "execute"})
        handler = _find_handler(kb, Keys.Escape, Keys.Escape)
        assert handler is not None

        event, buf = _mock_event()
        handler(event)
        assert buf.text == ""

    def test_double_ctrl_c_submits_quit(self):
        kb = _build_repl_key_bindings({"mode": "execute"})
        handler = _find_handler(kb, Keys.ControlC, Keys.ControlC)
        assert handler is not None

        event, buf = _mock_event()
        handler(event)
        assert buf.text == "/quit"
        buf.validate_and_handle.assert_called_once()

    def test_all_bindings_registered(self):
        kb = _build_repl_key_bindings({"mode": "execute"})
        key_sequences = [tuple(b.keys) for b in kb.bindings]
        assert (Keys.ControlQ,) in key_sequences
        assert (Keys.BackTab,) in key_sequences
        assert (Keys.Escape, Keys.Escape) in key_sequences
        assert (Keys.ControlC, Keys.ControlC) in key_sequences
