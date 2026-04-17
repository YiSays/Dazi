"""Tests for dazi/onboard.py — onboarding wizard steps and run_onboarding."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from rich.console import Console

from dazi.settings import DaziSettings
from tests.helpers.mock_singletons import patch_singletons


@pytest.fixture(autouse=True)
def _patch(monkeypatch, tmp_path: Path):
    patch_singletons(monkeypatch, tmp_path)


def _console() -> Console:
    return Console(file=MagicMock(), force_terminal=False)


def _settings(**overrides) -> DaziSettings:
    defaults: dict = {
        "api_key": None,
        "model": None,
        "api_base_url": None,
        "default_mode": "default",
        "allow_rules": [],
        "deny_rules": [],
        "env": {},
        "auto_compact": True,
        "auto_memory": True,
        "max_concurrent_tools": 5,
        "mcp_servers": {},
        "context_window": None,
    }
    defaults.update(overrides)
    return DaziSettings(**defaults)


# ─────────────────────────────────────────────────────────
# _mask_key
# ─────────────────────────────────────────────────────────


class TestMaskKey:
    def test_short_key_returned_as_is(self):
        from dazi.onboard import _mask_key

        assert _mask_key("abc") == "abc"

    def test_exact_length_returned_as_is(self):
        from dazi.onboard import _mask_key

        assert _mask_key("abcd") == "abcd"

    def test_long_key_masked(self):
        from dazi.onboard import _mask_key

        # "sk-1234567890" = 13 chars, show last 4 -> 9 asterisks
        assert _mask_key("sk-1234567890") == "*********7890"

    def test_custom_visible(self):
        from dazi.onboard import _mask_key

        assert _mask_key("abcdefgh", visible=2) == "******gh"


# ─────────────────────────────────────────────────────────
# _print_header
# ─────────────────────────────────────────────────────────


class TestPrintHeader:
    def test_first_run(self):
        from dazi.onboard import _print_header

        console = _console()
        _print_header(console, is_rerun=False)
        # No exception, console.print was called
        console.file.write.assert_called()

    def test_rerun(self):
        from dazi.onboard import _print_header

        console = _console()
        _print_header(console, is_rerun=True)
        console.file.write.assert_called()


# ─────────────────────────────────────────────────────────
# _prompt_value
# ─────────────────────────────────────────────────────────


class TestPromptValue:
    @patch("dazi.onboard.Prompt")
    def test_returns_user_input(self, mock_prompt_cls):
        from dazi.onboard import _prompt_value

        mock_prompt_cls.ask.return_value = "my-value"
        console = _console()

        result = _prompt_value(console, "Label")
        assert result == "my-value"

    @patch("dazi.onboard.Prompt")
    def test_strips_whitespace(self, mock_prompt_cls):
        from dazi.onboard import _prompt_value

        mock_prompt_cls.ask.return_value = "  my-value  "
        console = _console()

        result = _prompt_value(console, "Label")
        assert result == "my-value"

    @patch("dazi.onboard.Prompt")
    def test_empty_input_returns_default(self, mock_prompt_cls):
        from dazi.onboard import _prompt_value

        mock_prompt_cls.ask.return_value = ""
        console = _console()

        result = _prompt_value(console, "Label", default="fallback")
        assert result == "fallback"

    @patch("dazi.onboard.Prompt")
    def test_empty_input_no_default_returns_none(self, mock_prompt_cls):
        from dazi.onboard import _prompt_value

        mock_prompt_cls.ask.return_value = ""
        console = _console()

        result = _prompt_value(console, "Label", required=False)
        assert result is None

    @patch("dazi.onboard.Prompt")
    def test_required_empty_repeats(self, mock_prompt_cls):
        from dazi.onboard import _prompt_value

        mock_prompt_cls.ask.side_effect = ["", "valid"]
        console = _console()

        result = _prompt_value(console, "Label", required=True)
        assert result == "valid"
        assert mock_prompt_cls.ask.call_count == 2

    @patch("dazi.onboard.Prompt")
    def test_keyboard_interrupt_required_repeats(self, mock_prompt_cls):
        from dazi.onboard import _prompt_value

        mock_prompt_cls.ask.side_effect = [KeyboardInterrupt, "valid"]
        console = _console()

        result = _prompt_value(console, "Label", required=True)
        assert result == "valid"

    @patch("dazi.onboard.Prompt")
    def test_keyboard_interrupt_optional_returns_none(self, mock_prompt_cls):
        from dazi.onboard import _prompt_value

        mock_prompt_cls.ask.side_effect = [KeyboardInterrupt]
        console = _console()

        result = _prompt_value(console, "Label", required=False)
        assert result is None

    @patch("dazi.onboard.Prompt")
    def test_eof_error_optional_returns_none(self, mock_prompt_cls):
        from dazi.onboard import _prompt_value

        mock_prompt_cls.ask.side_effect = [EOFError]
        console = _console()

        result = _prompt_value(console, "Label", required=False)
        assert result is None

    @patch("dazi.onboard.Prompt")
    def test_password_does_not_pass_default(self, mock_prompt_cls):
        from dazi.onboard import _prompt_value

        mock_prompt_cls.ask.return_value = ""
        console = _console()

        result = _prompt_value(console, "Label", default="secret", password=True)
        # For password fields, default is not passed to Prompt.ask, but
        # empty input returns the default manually
        assert result == "secret"
        # Verify default was NOT passed to Prompt.ask (would be second positional arg)
        call_kwargs = mock_prompt_cls.ask.call_args[1]
        assert "default" not in call_kwargs


# ─────────────────────────────────────────────────────────
# _step_api_key
# ─────────────────────────────────────────────────────────


class TestStepApiKey:
    @patch("dazi.onboard._prompt_value")
    def test_returns_value(self, mock_prompt):
        from dazi.onboard import _step_api_key

        mock_prompt.return_value = "sk-newkey"
        console = _console()
        current = _settings(api_key="sk-oldkey")

        result = _step_api_key(console, current, required=False)
        assert result == "sk-newkey"

    @patch("dazi.onboard._prompt_value")
    def test_returns_none_when_skipped(self, mock_prompt):
        from dazi.onboard import _step_api_key

        mock_prompt.return_value = None
        console = _console()
        current = _settings()

        result = _step_api_key(console, current, required=False)
        assert result is None


# ─────────────────────────────────────────────────────────
# _step_model
# ─────────────────────────────────────────────────────────


class TestStepModel:
    @patch("dazi.onboard._prompt_value")
    def test_returns_model(self, mock_prompt):
        from dazi.onboard import _step_model

        mock_prompt.return_value = "gpt-4o"
        console = _console()
        current = _settings()

        result = _step_model(console, current, required=True)
        assert result == "gpt-4o"

    @patch("dazi.onboard._prompt_value")
    def test_returns_none_when_skipped(self, mock_prompt):
        from dazi.onboard import _step_model

        mock_prompt.return_value = None
        console = _console()
        current = _settings()

        result = _step_model(console, current, required=False)
        assert result is None


# ─────────────────────────────────────────────────────────
# _step_base_url
# ─────────────────────────────────────────────────────────


class TestStepBaseUrl:
    @patch("dazi.onboard._prompt_value")
    def test_valid_https_url(self, mock_prompt):
        from dazi.onboard import _step_base_url

        mock_prompt.return_value = "https://api.openai.com/v1"
        console = _console()
        current = _settings()

        result = _step_base_url(console, current, required=True)
        assert result == "https://api.openai.com/v1"

    @patch("dazi.onboard._prompt_value")
    def test_valid_http_url(self, mock_prompt):
        from dazi.onboard import _step_base_url

        mock_prompt.return_value = "http://localhost:8080/v1"
        console = _console()
        current = _settings()

        result = _step_base_url(console, current, required=True)
        assert result == "http://localhost:8080/v1"

    @patch("dazi.onboard._prompt_value")
    def test_invalid_url_rejected_then_accepted(self, mock_prompt):
        from dazi.onboard import _step_base_url

        mock_prompt.side_effect = ["ftp://bad", "https://api.openai.com/v1"]
        console = _console()
        current = _settings()

        result = _step_base_url(console, current, required=True)
        assert result == "https://api.openai.com/v1"
        assert mock_prompt.call_count == 2

    @patch("dazi.onboard._prompt_value")
    def test_skip_returns_none(self, mock_prompt):
        from dazi.onboard import _step_base_url

        mock_prompt.return_value = None
        console = _console()
        current = _settings()

        result = _step_base_url(console, current, required=False)
        assert result is None


# ─────────────────────────────────────────────────────────
# _step_token_window
# ─────────────────────────────────────────────────────────


class TestStepTokenWindow:
    """_step_token_window calls get_context_window which hits settings_manager.
    We patch it to return a predictable int so f-string formatting works."""

    @patch("dazi.onboard.get_context_window", return_value=128000)
    @patch("dazi.onboard._prompt_value")
    def test_returns_user_value(self, mock_prompt, mock_gcw):
        from dazi.onboard import _step_token_window

        mock_prompt.return_value = "200000"
        console = _console()
        current = _settings()

        result = _step_token_window(console, current, "claude-sonnet-4-20250514")
        assert result == 200000

    @patch("dazi.onboard.get_context_window", return_value=128000)
    @patch("dazi.onboard._prompt_value")
    def test_empty_keeps_existing(self, mock_prompt, mock_gcw):
        from dazi.onboard import _step_token_window

        mock_prompt.return_value = None
        console = _console()
        current = _settings(context_window=99999)

        result = _step_token_window(console, current, "gpt-4o")
        assert result == 99999

    @patch("dazi.onboard.get_context_window", return_value=128000)
    @patch("dazi.onboard._prompt_value")
    def test_empty_no_existing_returns_none(self, mock_prompt, mock_gcw):
        from dazi.onboard import _step_token_window

        mock_prompt.return_value = None
        console = _console()
        current = _settings()

        result = _step_token_window(console, current, "gpt-4o")
        assert result is None

    @patch("dazi.onboard.get_context_window", return_value=128000)
    @patch("dazi.onboard._prompt_value")
    def test_comma_separated_number(self, mock_prompt, mock_gcw):
        from dazi.onboard import _step_token_window

        mock_prompt.return_value = "128,000"
        console = _console()
        current = _settings()

        result = _step_token_window(console, current, "gpt-4o")
        assert result == 128000

    @patch("dazi.onboard.get_context_window", return_value=128000)
    @patch("dazi.onboard._prompt_value")
    def test_invalid_number_keeps_existing(self, mock_prompt, mock_gcw):
        from dazi.onboard import _step_token_window

        mock_prompt.return_value = "not-a-number"
        console = _console()
        current = _settings(context_window=50000)

        result = _step_token_window(console, current, "gpt-4o")
        assert result == 50000

    @patch("dazi.onboard.get_context_window", return_value=128000)
    @patch("dazi.onboard._prompt_value")
    def test_invalid_number_no_existing_returns_none(self, mock_prompt, mock_gcw):
        from dazi.onboard import _step_token_window

        mock_prompt.return_value = "not-a-number"
        console = _console()
        current = _settings()

        result = _step_token_window(console, current, "gpt-4o")
        assert result is None

    @patch("dazi.onboard.get_context_window", return_value=128000)
    @patch("dazi.onboard._prompt_value")
    def test_negative_number_keeps_existing(self, mock_prompt, mock_gcw):
        from dazi.onboard import _step_token_window

        mock_prompt.return_value = "-100"
        console = _console()
        current = _settings(context_window=8000)

        result = _step_token_window(console, current, "gpt-4o")
        assert result == 8000

    @patch("dazi.onboard.get_context_window", return_value=128000)
    @patch("dazi.onboard._prompt_value")
    def test_zero_keeps_existing(self, mock_prompt, mock_gcw):
        from dazi.onboard import _step_token_window

        mock_prompt.return_value = "0"
        console = _console()
        current = _settings(context_window=8000)

        result = _step_token_window(console, current, "gpt-4o")
        assert result == 8000


# ─────────────────────────────────────────────────────────
# _step_mcp
# ─────────────────────────────────────────────────────────


class TestStepMcp:
    @patch("dazi.onboard.Confirm")
    @patch("builtins.input")
    def test_skip_returns_existing(self, mock_input, mock_confirm_cls):
        from dazi.onboard import _step_mcp

        mock_confirm_cls.ask.return_value = False
        console = _console()
        current = _settings(mcp_servers={"old": {"command": "echo"}})

        result = _step_mcp(console, current)
        assert result == {"old": {"command": "echo"}}

    @patch("dazi.onboard.Confirm")
    @patch("builtins.input")
    def test_empty_paste_returns_existing(self, mock_input, mock_confirm_cls):
        from dazi.onboard import _step_mcp

        mock_confirm_cls.ask.return_value = True
        # Two empty lines to terminate
        mock_input.side_effect = ["", ""]
        console = _console()
        current = _settings(mcp_servers={"old": {"command": "echo"}})

        result = _step_mcp(console, current)
        assert result == {"old": {"command": "echo"}}

    @patch("dazi.onboard.Confirm")
    @patch("builtins.input")
    def test_valid_json_merges(self, mock_input, mock_confirm_cls):
        from dazi.onboard import _step_mcp

        mock_confirm_cls.ask.return_value = True
        new_json = json.dumps({"new-server": {"command": "npx", "args": []}})
        mock_input.side_effect = [new_json, "", ""]
        console = _console()
        current = _settings(mcp_servers={"old": {"command": "echo"}})

        result = _step_mcp(console, current)
        assert "old" in result
        assert "new-server" in result

    @patch("dazi.onboard.Confirm")
    @patch("builtins.input")
    def test_invalid_json_returns_existing(self, mock_input, mock_confirm_cls):
        from dazi.onboard import _step_mcp

        mock_confirm_cls.ask.return_value = True
        mock_input.side_effect = ["{bad json}", "", ""]
        console = _console()
        current = _settings(mcp_servers={"old": {"command": "echo"}})

        result = _step_mcp(console, current)
        assert result == {"old": {"command": "echo"}}

    @patch("dazi.onboard.Confirm")
    @patch("builtins.input")
    def test_non_dict_json_returns_existing(self, mock_input, mock_confirm_cls):
        from dazi.onboard import _step_mcp

        mock_confirm_cls.ask.return_value = True
        mock_input.side_effect = ["[1,2,3]", "", ""]
        console = _console()
        current = _settings(mcp_servers={"old": {"command": "echo"}})

        result = _step_mcp(console, current)
        assert result == {"old": {"command": "echo"}}

    @patch("dazi.onboard.Confirm")
    @patch("builtins.input")
    def test_no_existing_and_skip(self, mock_input, mock_confirm_cls):
        from dazi.onboard import _step_mcp

        mock_confirm_cls.ask.return_value = False
        console = _console()
        current = _settings()

        result = _step_mcp(console, current)
        assert result == {}

    @patch("dazi.onboard.Confirm")
    @patch("builtins.input")
    def test_eof_returns_existing(self, mock_input, mock_confirm_cls):
        from dazi.onboard import _step_mcp

        mock_confirm_cls.ask.return_value = True
        mock_input.side_effect = [EOFError]
        console = _console()
        current = _settings(mcp_servers={"old": {"command": "echo"}})

        result = _step_mcp(console, current)
        assert result == {"old": {"command": "echo"}}

    @patch("dazi.onboard.Confirm")
    @patch("builtins.input")
    def test_multiline_json(self, mock_input, mock_confirm_cls):
        from dazi.onboard import _step_mcp

        mock_confirm_cls.ask.return_value = True
        mock_input.side_effect = ['{"srv": {"command": "npx"}}', "", ""]
        console = _console()
        current = _settings()

        result = _step_mcp(console, current)
        assert result == {"srv": {"command": "npx"}}


# ─────────────────────────────────────────────────────────
# _step_dazimd
# ─────────────────────────────────────────────────────────


class TestStepDazimd:
    @patch("dazi.onboard.Confirm")
    def test_create_new(self, mock_confirm_cls, tmp_path, monkeypatch):
        from dazi.onboard import _step_dazimd

        # Mock Path.home() to use tmp_path
        fake_home = tmp_path / "home"
        fake_home.mkdir()
        (fake_home / ".dazi").mkdir()
        monkeypatch.setattr("pathlib.Path.home", lambda: fake_home)

        mock_confirm_cls.ask.return_value = True
        console = _console()

        _step_dazimd(console)

        dazimd_path = fake_home / ".dazi" / "DAZI.md"
        assert dazimd_path.exists()
        content = dazimd_path.read_text()
        assert "Develop Autonomously" in content
        assert "File Locations" in content

    @patch("dazi.onboard.Confirm")
    def test_decline_create(self, mock_confirm_cls, tmp_path, monkeypatch):
        from dazi.onboard import _step_dazimd

        fake_home = tmp_path / "home"
        fake_home.mkdir()
        (fake_home / ".dazi").mkdir()
        monkeypatch.setattr("pathlib.Path.home", lambda: fake_home)

        mock_confirm_cls.ask.return_value = False
        console = _console()

        _step_dazimd(console)

        dazimd_path = fake_home / ".dazi" / "DAZI.md"
        assert not dazimd_path.exists()

    @patch("dazi.onboard.Confirm")
    def test_existing_decline_overwrite(self, mock_confirm_cls, tmp_path, monkeypatch):
        from dazi.onboard import _step_dazimd

        fake_home = tmp_path / "home"
        fake_home.mkdir()
        dazi_dir = fake_home / ".dazi"
        dazi_dir.mkdir()
        dazimd_path = dazi_dir / "DAZI.md"
        dazimd_path.write_text("original content")
        monkeypatch.setattr("pathlib.Path.home", lambda: fake_home)

        # First call: Overwrite? -> False
        mock_confirm_cls.ask.return_value = False
        console = _console()

        _step_dazimd(console)

        assert dazimd_path.read_text() == "original content"

    @patch("dazi.onboard.Confirm")
    def test_existing_accept_overwrite(self, mock_confirm_cls, tmp_path, monkeypatch):
        from dazi.onboard import _step_dazimd

        fake_home = tmp_path / "home"
        fake_home.mkdir()
        dazi_dir = fake_home / ".dazi"
        dazi_dir.mkdir()
        dazimd_path = dazi_dir / "DAZI.md"
        dazimd_path.write_text("original content")
        monkeypatch.setattr("pathlib.Path.home", lambda: fake_home)

        # Overwrite? -> True, then Create template? -> True
        mock_confirm_cls.ask.side_effect = [True, True]
        console = _console()

        _step_dazimd(console)

        content = dazimd_path.read_text()
        assert "Develop Autonomously" in content
        assert "original content" not in content


# ─────────────────────────────────────────────────────────
# run_onboarding
# ─────────────────────────────────────────────────────────


class TestRunOnboarding:
    @patch("dazi.onboard._step_dazimd")
    @patch("dazi.onboard._step_mcp")
    @patch("dazi.onboard._step_token_window")
    @patch("dazi.onboard._step_base_url")
    @patch("dazi.onboard._step_model")
    @patch("dazi.onboard._step_api_key")
    def test_first_run_sets_all_required(
        self, mock_api, mock_model, mock_url, mock_tw, mock_mcp, mock_dazimd, monkeypatch
    ):
        from dazi.onboard import run_onboarding

        console = _console()

        # No existing settings -> first run, required=True
        settings_manager = MagicMock()
        settings_manager.settings = _settings()
        settings_manager.save_user_settings = MagicMock()
        monkeypatch.setattr("dazi._singletons.settings_manager", settings_manager)

        mock_api.return_value = "sk-test123"
        mock_model.return_value = "gpt-4o"
        mock_url.return_value = "https://api.openai.com/v1"
        mock_tw.return_value = 128000
        mock_mcp.return_value = {}

        run_onboarding(console)

        settings_manager.save_user_settings.assert_called_once()
        saved = settings_manager.save_user_settings.call_args[0][0]
        assert saved.api_key == "sk-test123"
        assert saved.model == "gpt-4o"
        assert saved.api_base_url == "https://api.openai.com/v1"
        assert saved.context_window == 128000

    @patch("dazi.onboard._step_dazimd")
    @patch("dazi.onboard._step_mcp")
    @patch("dazi.onboard._step_token_window")
    @patch("dazi.onboard._step_base_url")
    @patch("dazi.onboard._step_model")
    @patch("dazi.onboard._step_api_key")
    def test_rerun_preserves_existing(
        self, mock_api, mock_model, mock_url, mock_tw, mock_mcp, mock_dazimd, monkeypatch
    ):
        from dazi.onboard import run_onboarding

        console = _console()

        # Existing settings -> rerun, required=False
        settings_manager = MagicMock()
        settings_manager.settings = _settings(
            api_key="sk-existing",
            model="gpt-4o-mini",
            api_base_url="https://api.openai.com/v1",
            context_window=64000,
            default_mode="plan",
            auto_compact=False,
        )
        settings_manager.save_user_settings = MagicMock()
        monkeypatch.setattr("dazi._singletons.settings_manager", settings_manager)

        # User skips all steps
        mock_api.return_value = None
        mock_model.return_value = None
        mock_url.return_value = None
        mock_tw.return_value = 64000
        mock_mcp.return_value = {}

        run_onboarding(console)

        saved = settings_manager.save_user_settings.call_args[0][0]
        assert saved.api_key == "sk-existing"
        assert saved.model == "gpt-4o-mini"
        assert saved.api_base_url == "https://api.openai.com/v1"
        assert saved.default_mode == "plan"
        assert saved.auto_compact is False

    @patch("dazi.onboard._step_dazimd")
    @patch("dazi.onboard._step_mcp")
    @patch("dazi.onboard._step_token_window")
    @patch("dazi.onboard._step_base_url")
    @patch("dazi.onboard._step_model")
    @patch("dazi.onboard._step_api_key")
    def test_rerun_updates_some_fields(
        self, mock_api, mock_model, mock_url, mock_tw, mock_mcp, mock_dazimd, monkeypatch
    ):
        from dazi.onboard import run_onboarding

        console = _console()

        settings_manager = MagicMock()
        settings_manager.settings = _settings(
            api_key="sk-existing",
            model="gpt-4o-mini",
            api_base_url="https://api.openai.com/v1",
        )
        settings_manager.save_user_settings = MagicMock()
        monkeypatch.setattr("dazi._singletons.settings_manager", settings_manager)

        # User updates api_key and model, keeps url
        mock_api.return_value = "sk-new-key"
        mock_model.return_value = "claude-sonnet-4-20250514"
        mock_url.return_value = None
        mock_tw.return_value = 200000
        mock_mcp.return_value = {"my-server": {"command": "npx"}}

        run_onboarding(console)

        saved = settings_manager.save_user_settings.call_args[0][0]
        assert saved.api_key == "sk-new-key"
        assert saved.model == "claude-sonnet-4-20250514"
        assert saved.api_base_url == "https://api.openai.com/v1"
        assert saved.mcp_servers == {"my-server": {"command": "npx"}}

    @patch("dazi.onboard._step_dazimd")
    @patch("dazi.onboard._step_mcp")
    @patch("dazi.onboard._step_token_window")
    @patch("dazi.onboard._step_base_url")
    @patch("dazi.onboard._step_model")
    @patch("dazi.onboard._step_api_key")
    def test_rerun_prints_mcp_servers(
        self, mock_api, mock_model, mock_url, mock_tw, mock_mcp, mock_dazimd, monkeypatch
    ):
        from dazi.onboard import run_onboarding

        console = _console()

        settings_manager = MagicMock()
        settings_manager.settings = _settings(
            api_key="sk-existing",
            model="gpt-4o",
            api_base_url="https://api.openai.com/v1",
        )
        settings_manager.save_user_settings = MagicMock()
        monkeypatch.setattr("dazi._singletons.settings_manager", settings_manager)

        mock_api.return_value = "sk-existing"
        mock_model.return_value = "gpt-4o"
        mock_url.return_value = "https://api.openai.com/v1"
        mock_tw.return_value = 128000
        mock_mcp.return_value = {"server-a": {}, "server-b": {}}

        run_onboarding(console)

        settings_manager.save_user_settings.assert_called_once()

    @patch("dazi.onboard._step_dazimd")
    @patch("dazi.onboard._step_mcp")
    @patch("dazi.onboard._step_token_window")
    @patch("dazi.onboard._step_base_url")
    @patch("dazi.onboard._step_model")
    @patch("dazi.onboard._step_api_key")
    def test_preserves_env_and_rules(
        self, mock_api, mock_model, mock_url, mock_tw, mock_mcp, mock_dazimd, monkeypatch
    ):
        from dazi.onboard import run_onboarding

        console = _console()

        settings_manager = MagicMock()
        settings_manager.settings = _settings(
            api_key="sk-test",
            model="gpt-4o",
            api_base_url="https://api.openai.com/v1",
            env={"MY_VAR": "value"},
            allow_rules=["allow-read"],
            deny_rules=["deny-write"],
        )
        settings_manager.save_user_settings = MagicMock()
        monkeypatch.setattr("dazi._singletons.settings_manager", settings_manager)

        mock_api.return_value = "sk-test"
        mock_model.return_value = "gpt-4o"
        mock_url.return_value = "https://api.openai.com/v1"
        mock_tw.return_value = None
        mock_mcp.return_value = {}

        run_onboarding(console)

        saved = settings_manager.save_user_settings.call_args[0][0]
        assert saved.env == {"MY_VAR": "value"}
        assert saved.allow_rules == ["allow-read"]
        assert saved.deny_rules == ["deny-write"]
