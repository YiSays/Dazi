"""Tests for dazi/lifecycle.py — load_dazimd, cleanup_on_exit."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from rich.console import Console

from tests.helpers.mock_singletons import patch_singletons


@pytest.fixture(autouse=True)
def _patch(monkeypatch, tmp_path: Path):
    patch_singletons(monkeypatch, tmp_path)


def _mock_deps(monkeypatch, mod, **overrides):
    """Patch all singletons on the lifecycle module with sensible defaults."""
    defaults = dict(
        proactive_manager=MagicMock(),
        autonomous_teammate=MagicMock(list_handles=MagicMock(return_value=[])),
        worktree_manager=MagicMock(list_all=MagicMock(return_value=[])),
        teammate_runner=MagicMock(shutdown_all=AsyncMock(return_value=0)),
        background_manager=MagicMock(list_active=MagicMock(return_value=[])),
        mcp_manager=MagicMock(disconnect_all=AsyncMock()),
        cost_tracker=MagicMock(),
    )
    defaults.update(overrides)
    for name, obj in defaults.items():
        monkeypatch.setattr(mod, name, obj)


class TestLoadDazimd:
    def test_with_files(self, monkeypatch, tmp_path):
        import dazi.lifecycle as mod

        console = Console(file=MagicMock(), force_terminal=False)
        dazi_md = tmp_path / "DAZI.md"
        dazi_md.write_text("# Test\n")

        mock_file = MagicMock()
        mock_file.path = str(dazi_md)
        mock_file.priority = 10

        monkeypatch.setattr(mod, "discover_dazimd_files", MagicMock(return_value=[mock_file]))
        monkeypatch.setattr(mod, "merge_dazimd_content", MagicMock(return_value="merged"))
        mock_pb = MagicMock()
        monkeypatch.setattr(mod, "prompt_builder", mock_pb)
        monkeypatch.setattr("pathlib.Path.cwd", MagicMock(return_value=tmp_path))

        result = mod.load_dazimd(console=console)

        assert result == [mock_file]
        mock_pb.set_dazimd_content.assert_called_once_with("merged")

    def test_no_files(self, monkeypatch):
        import dazi.lifecycle as mod

        console = Console(file=MagicMock(), force_terminal=False)

        monkeypatch.setattr(mod, "discover_dazimd_files", MagicMock(return_value=[]))
        mock_pb = MagicMock()
        monkeypatch.setattr(mod, "prompt_builder", mock_pb)
        monkeypatch.setattr("pathlib.Path.cwd", MagicMock(return_value=Path("/tmp")))

        result = mod.load_dazimd(console=console)

        assert result == []
        mock_pb.set_dazimd_content.assert_not_called()


class TestCleanupOnExit:
    @pytest.mark.asyncio
    async def test_basic_cleanup(self, monkeypatch):
        import dazi.lifecycle as mod

        console = Console(file=MagicMock(), force_terminal=False)
        _mock_deps(monkeypatch, mod)

        await mod.cleanup_on_exit(console=console)

    @pytest.mark.asyncio
    async def test_with_autonomous_teammates(self, monkeypatch):
        import dazi.lifecycle as mod

        console = Console(file=MagicMock(), force_terminal=False)
        handle = MagicMock()
        handle.team_name = "team1"
        handle.name = "agent1"
        _mock_deps(
            monkeypatch,
            mod,
            autonomous_teammate=MagicMock(
                list_handles=MagicMock(return_value=[handle]),
                shutdown=AsyncMock(),
            ),
        )

        await mod.cleanup_on_exit(console=console)

    @pytest.mark.asyncio
    async def test_with_worktrees(self, monkeypatch):
        import dazi.lifecycle as mod

        console = Console(file=MagicMock(), force_terminal=False)
        wt = MagicMock()
        wt.id = "wt1"
        _mock_deps(
            monkeypatch,
            mod,
            worktree_manager=MagicMock(list_all=MagicMock(return_value=[wt]), remove=MagicMock()),
        )

        await mod.cleanup_on_exit(console=console)

    @pytest.mark.asyncio
    async def test_with_worktree_remove_raises(self, monkeypatch):
        import dazi.lifecycle as mod

        console = Console(file=MagicMock(), force_terminal=False)
        wt = MagicMock()
        wt.id = "wt1"
        _mock_deps(
            monkeypatch,
            mod,
            worktree_manager=MagicMock(
                list_all=MagicMock(return_value=[wt]),
                remove=MagicMock(side_effect=Exception("fail")),
            ),
        )

        await mod.cleanup_on_exit(console=console)

    @pytest.mark.asyncio
    async def test_with_active_team(self, monkeypatch):
        import dazi.lifecycle as mod

        console = Console(file=MagicMock(), force_terminal=False)
        _mock_deps(
            monkeypatch,
            mod,
            teammate_runner=MagicMock(shutdown_all=AsyncMock(return_value=3)),
        )

        await mod.cleanup_on_exit(console=console, active_team_name="my-team")

    @pytest.mark.asyncio
    async def test_with_background_tasks(self, monkeypatch):
        import dazi.lifecycle as mod

        console = Console(file=MagicMock(), force_terminal=False)
        bg_task = MagicMock()
        bg_task.id = "bg1"
        _mock_deps(
            monkeypatch,
            mod,
            background_manager=MagicMock(
                list_active=MagicMock(return_value=[bg_task]), cancel=AsyncMock()
            ),
        )

        await mod.cleanup_on_exit(console=console)

    @pytest.mark.asyncio
    async def test_with_goodbye(self, monkeypatch):
        import dazi.lifecycle as mod

        console = Console(file=MagicMock(), force_terminal=False)
        _mock_deps(monkeypatch, mod)

        await mod.cleanup_on_exit(console=console, say_goodbye=True)
