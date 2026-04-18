"""Tests for dazi/lifecycle.py — load_dazimd, load_subsystems, cleanup_on_exit."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, call

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
        mock_pb.set_dazimd_content.assert_called_once_with("merged", files=[mock_file])

    def test_no_files(self, monkeypatch):
        import dazi.lifecycle as mod

        console = Console(file=MagicMock(), force_terminal=False)

        monkeypatch.setattr(mod, "discover_dazimd_files", MagicMock(return_value=[]))
        mock_pb = MagicMock()
        monkeypatch.setattr(mod, "prompt_builder", mock_pb)
        monkeypatch.setattr("pathlib.Path.cwd", MagicMock(return_value=Path("/tmp")))

        result = mod.load_dazimd(console=console)

        assert result == []
        mock_pb.set_dazimd_content.assert_called_once_with("", files=[])


class TestLoadSubsystems:
    """Tests for the shared load_subsystems() function."""

    @pytest.mark.asyncio
    async def test_calls_all_subsystems(self, monkeypatch):
        """load_subsystems calls DAZI.md, settings, skills, and MCP in order."""
        import dazi.lifecycle as mod

        console = Console(file=MagicMock(), force_terminal=False)

        mock_file = MagicMock()
        monkeypatch.setattr(mod, "load_dazimd", MagicMock(return_value=[mock_file]))
        mock_settings = MagicMock()
        monkeypatch.setattr(mod, "settings_manager", mock_settings)
        mock_skills = MagicMock()
        mock_skills.reload.return_value = 7
        monkeypatch.setattr(mod, "skill_registry", mock_skills)
        mock_mcp = MagicMock(disconnect_all=AsyncMock())
        monkeypatch.setattr(mod, "mcp_manager", mock_mcp)
        monkeypatch.setattr("dazi.graph.connect_mcp_servers", AsyncMock())

        result = await mod.load_subsystems(console=console)

        assert result.dazimd_files == [mock_file]
        assert result.skill_count == 7
        mod.load_dazimd.assert_called_once_with(console=console)
        mock_settings.reload.assert_called_once()
        mock_skills.reload.assert_called_once()
        mock_mcp.disconnect_all.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_no_dazimd_files(self, monkeypatch):
        """Returns empty dazimd_files when no DAZI.md files found."""
        import dazi.lifecycle as mod

        console = Console(file=MagicMock(), force_terminal=False)

        monkeypatch.setattr(mod, "load_dazimd", MagicMock(return_value=[]))
        monkeypatch.setattr(mod, "settings_manager", MagicMock())
        monkeypatch.setattr(mod, "skill_registry", MagicMock(reload=MagicMock(return_value=0)))
        monkeypatch.setattr(mod, "mcp_manager", MagicMock(disconnect_all=AsyncMock()))
        monkeypatch.setattr("dazi.graph.connect_mcp_servers", AsyncMock())

        result = await mod.load_subsystems(console=console)

        assert result.dazimd_files == []
        assert result.skill_count == 0

    @pytest.mark.asyncio
    async def test_returns_subsystem_load_result(self, monkeypatch):
        """Returns a SubsystemLoadResult dataclass."""
        import dazi.lifecycle as mod

        console = Console(file=MagicMock(), force_terminal=False)

        files = [MagicMock(), MagicMock()]
        monkeypatch.setattr(mod, "load_dazimd", MagicMock(return_value=files))
        monkeypatch.setattr(mod, "settings_manager", MagicMock())
        monkeypatch.setattr(mod, "skill_registry", MagicMock(reload=MagicMock(return_value=3)))
        monkeypatch.setattr(mod, "mcp_manager", MagicMock(disconnect_all=AsyncMock()))
        monkeypatch.setattr("dazi.graph.connect_mcp_servers", AsyncMock())

        result = await mod.load_subsystems(console=console)

        assert isinstance(result, mod.SubsystemLoadResult)
        assert result.dazimd_files == files
        assert result.skill_count == 3

    @pytest.mark.asyncio
    async def test_order_dazimd_before_settings(self, monkeypatch):
        """DAZI.md is loaded before settings reload."""
        import dazi.lifecycle as mod

        console = Console(file=MagicMock(), force_terminal=False)
        call_order = []

        monkeypatch.setattr(mod, "load_dazimd", MagicMock(side_effect=lambda **kw: call_order.append("dazimd") or []))
        mock_settings = MagicMock()
        mock_settings.reload.side_effect = lambda: call_order.append("settings")
        monkeypatch.setattr(mod, "settings_manager", mock_settings)
        monkeypatch.setattr(mod, "skill_registry", MagicMock(reload=MagicMock(side_effect=lambda: call_order.append("skills") or 0)))
        monkeypatch.setattr(mod, "mcp_manager", MagicMock(disconnect_all=AsyncMock(side_effect=lambda: call_order.append("mcp_disconnect"))))
        monkeypatch.setattr("dazi.graph.connect_mcp_servers", AsyncMock(side_effect=lambda: call_order.append("mcp_connect")))

        await mod.load_subsystems(console=console)

        assert call_order == ["dazimd", "settings", "skills", "mcp_disconnect", "mcp_connect"]


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
