"""Tests for dazi/worktree.py — sanitize_agent_name, validate_slug, WorktreeConfig,
WorktreeManager listing, create/remove/keep/cleanup, tool functions."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from dazi.worktree import (
    Worktree,
    WorktreeConfig,
    WorktreeManager,
    create_worktree_func,
    finish_worktree_func,
    list_worktrees_func,
)

# ─────────────────────────────────────────────────────────
# sanitize_agent_name
# ─────────────────────────────────────────────────────────


class TestSanitizeAgentName:
    def test_replaces_spaces_with_dashes(self):
        mgr = WorktreeManager()
        assert mgr.sanitize_agent_name("my agent") == "my-agent"

    def test_lowercases(self):
        mgr = WorktreeManager()
        assert mgr.sanitize_agent_name("MyAgent") == "myagent"

    def test_strips_leading_trailing_dashes(self):
        mgr = WorktreeManager()
        assert mgr.sanitize_agent_name("-hello-") == "hello"

    def test_max_length_respected_in_validate(self):
        mgr = WorktreeManager(config=WorktreeConfig(max_slug_length=10))
        slug = "a" * 11
        with pytest.raises(ValueError, match="too long"):
            mgr.validate_slug(slug)


# ─────────────────────────────────────────────────────────
# validate_slug
# ─────────────────────────────────────────────────────────


class TestValidateSlug:
    def test_valid_slug(self):
        mgr = WorktreeManager()
        mgr.validate_slug("my-agent-123")  # should not raise

    def test_empty_slug_rejected(self):
        mgr = WorktreeManager()
        with pytest.raises(ValueError, match="empty"):
            mgr.validate_slug("")

    def test_path_traversal_dotdot_rejected(self):
        mgr = WorktreeManager()
        with pytest.raises(ValueError, match=r"[.][.]"):
            mgr.validate_slug("..")

    def test_path_traversal_dot_rejected(self):
        mgr = WorktreeManager()
        with pytest.raises(ValueError, match=r'[.]"'):
            mgr.validate_slug(".")

    def test_special_chars_rejected(self):
        mgr = WorktreeManager()
        with pytest.raises(ValueError, match="Invalid"):
            mgr.validate_slug("agent!@#")

    def test_slash_segment_rejected(self):
        mgr = WorktreeManager()
        with pytest.raises(ValueError):
            mgr.validate_slug("foo/../bar")

    def test_empty_segment_rejected(self):
        mgr = WorktreeManager()
        with pytest.raises(ValueError, match="empty segments"):
            mgr.validate_slug("foo//bar")

    def test_slash_segments_valid(self):
        mgr = WorktreeManager()
        mgr.validate_slug("foo/bar-baz")  # multi-segment valid slug


# ─────────────────────────────────────────────────────────
# WorktreeConfig defaults
# ─────────────────────────────────────────────────────────


class TestWorktreeConfig:
    def test_defaults(self):
        cfg = WorktreeConfig()
        assert cfg.worktree_base == ".dazi/worktrees"
        assert cfg.branch_prefix == "agent-"
        assert cfg.max_slug_length == 64
        assert cfg.stale_cutoff_days == 30


# ─────────────────────────────────────────────────────────
# WorktreeManager listing
# ─────────────────────────────────────────────────────────


class TestWorktreeManagerList:
    def test_list_empty(self):
        mgr = WorktreeManager()
        assert mgr.list_all() == []

    def test_list_with_entries(self):
        mgr = WorktreeManager()
        wt = Worktree(
            id="test",
            path="/tmp/test",
            branch="agent-test",
            agent_name="test",
            created_at="2025-01-01T00:00:00Z",
            original_cwd="/tmp",
            original_branch="main",
        )
        mgr._worktrees["test"] = wt
        result = mgr.list_all()
        assert len(result) == 1
        assert result[0].id == "test"

    def test_get_existing(self):
        mgr = WorktreeManager()
        wt = Worktree(
            id="x",
            path="/tmp/x",
            branch="agent-x",
            agent_name="x",
            created_at="2025-01-01T00:00:00Z",
            original_cwd="/tmp",
            original_branch="main",
        )
        mgr._worktrees["x"] = wt
        assert mgr.get("x") is wt

    def test_get_nonexistent(self):
        mgr = WorktreeManager()
        assert mgr.get("nobody") is None


# ─────────────────────────────────────────────────────────
# WorktreeManager create/remove with mocked git
# ─────────────────────────────────────────────────────────


class TestWorktreeManagerCreateRemove:
    def test_create_with_mocked_git(self):
        mgr = WorktreeManager()
        mock_result = type("R", (), {"returncode": 0, "stdout": "/tmp/repo\n", "stderr": ""})()
        mock_branch = type("R", (), {"returncode": 0, "stdout": "main\n", "stderr": ""})()
        mock_create = type("R", (), {"returncode": 0, "stdout": "", "stderr": ""})()

        with patch(
            "dazi.worktree.subprocess.run", side_effect=[mock_result, mock_branch, mock_create]
        ):
            wt = mgr.create("frontend")
            assert wt.id == "frontend"
            assert wt.branch == "agent-frontend"
            assert wt.agent_name == "frontend"

    def test_remove_with_mocked_git(self):
        mgr = WorktreeManager()
        wt = Worktree(
            id="test",
            path="/tmp/test",
            branch="agent-test",
            agent_name="test",
            created_at="2025-01-01T00:00:00Z",
            original_cwd="/tmp",
            original_branch="main",
        )
        mgr._worktrees["test"] = wt

        mock_status = type("R", (), {"returncode": 0, "stdout": "", "stderr": ""})()
        mock_remove = type("R", (), {"returncode": 0, "stdout": "", "stderr": ""})()
        mock_branch_del = type("R", (), {"returncode": 0, "stdout": "", "stderr": ""})()

        with patch(
            "dazi.worktree.subprocess.run", side_effect=[mock_status, mock_remove, mock_branch_del]
        ):
            result = mgr.remove("test", force=True)
            assert result is True
            assert mgr.get("test") is None


# ─────────────────────────────────────────────────────────
# _get_repo_root error paths
# ─────────────────────────────────────────────────────────


class TestGetRepoRoot:
    def test_not_a_git_repo(self):
        mgr = WorktreeManager()
        mock_result = type("R", (), {"returncode": 128, "stdout": "", "stderr": "not a repo"})()
        with patch("dazi.worktree.subprocess.run", return_value=mock_result):
            with pytest.raises(RuntimeError, match="Not a git repository"):
                mgr._get_repo_root()

    def test_git_not_found(self):
        mgr = WorktreeManager()
        with patch("dazi.worktree.subprocess.run", side_effect=FileNotFoundError):
            with pytest.raises(RuntimeError, match="git not found"):
                mgr._get_repo_root()


# ─────────────────────────────────────────────────────────
# _get_current_branch error paths
# ─────────────────────────────────────────────────────────


class TestGetCurrentBranch:
    def test_cannot_determine_branch(self):
        mgr = WorktreeManager()
        mock_result = type("R", (), {"returncode": 128, "stdout": "", "stderr": "error"})()
        with patch("dazi.worktree.subprocess.run", return_value=mock_result):
            with pytest.raises(RuntimeError, match="Cannot determine branch"):
                mgr._get_current_branch()

    def test_git_not_found(self):
        mgr = WorktreeManager()
        with patch("dazi.worktree.subprocess.run", side_effect=FileNotFoundError):
            with pytest.raises(RuntimeError, match="git not found"):
                mgr._get_current_branch()


# ─────────────────────────────────────────────────────────
# create — additional paths
# ─────────────────────────────────────────────────────────


def _mock_subprocess(returncode=0, stdout="", stderr=""):
    return type("R", (), {"returncode": returncode, "stdout": stdout, "stderr": stderr})()


class TestWorktreeManagerCreateAdditional:
    def test_create_already_exists(self):
        mgr = WorktreeManager()
        wt = Worktree(
            id="dup",
            path="/tmp/dup",
            branch="agent-dup",
            agent_name="dup",
            created_at="2025-01-01T00:00:00Z",
            original_cwd="/tmp",
            original_branch="main",
        )
        mgr._worktrees["dup"] = wt
        with pytest.raises(ValueError, match="already exists"):
            mgr.create("dup")

    def test_create_with_base_branch(self):
        mgr = WorktreeManager()
        side_effects = [
            _mock_subprocess(0, "/tmp/repo\n"),
            _mock_subprocess(0, "main\n"),
            _mock_subprocess(0),
        ]
        with patch("dazi.worktree.subprocess.run", side_effect=side_effects) as mock_run:
            with patch.object(Path, "cwd", return_value=Path("/cwd")):
                wt = mgr.create("my-agent", base_branch="develop")
        assert wt.branch == "agent-my-agent"
        # Check that base_branch was appended to the git command
        create_cmd = mock_run.call_args_list[2][0][0]
        assert "develop" in create_cmd

    def test_create_git_worktree_add_fails(self):
        mgr = WorktreeManager()
        side_effects = [
            _mock_subprocess(0, "/tmp/repo\n"),
            _mock_subprocess(0, "main\n"),
            _mock_subprocess(1, "", "worktree add failed"),
        ]
        with patch("dazi.worktree.subprocess.run", side_effect=side_effects):
            with pytest.raises(RuntimeError, match="Failed to create worktree"):
                mgr.create("bad")

    def test_create_mkdir_oserror_caught(self):
        mgr = WorktreeManager()
        side_effects = [
            _mock_subprocess(0, "/tmp/repo\n"),
            _mock_subprocess(0, "main\n"),
            _mock_subprocess(0),
        ]
        with patch("dazi.worktree.subprocess.run", side_effect=side_effects):
            with patch.object(Path, "cwd", return_value=Path("/cwd")):
                with patch.object(Path, "mkdir", side_effect=OSError("permission denied")):
                    wt = mgr.create("mkdir-fail")
        assert wt.id == "mkdir-fail"


# ─────────────────────────────────────────────────────────
# remove — additional paths
# ─────────────────────────────────────────────────────────


class TestWorktreeManagerRemoveAdditional:
    def test_remove_nonexistent_returns_false(self):
        mgr = WorktreeManager()
        assert mgr.remove("ghost") is False

    def test_remove_dirty_without_force_raises(self):
        mgr = WorktreeManager()
        wt = Worktree(
            id="dirty",
            path="/tmp/dirty",
            branch="agent-dirty",
            agent_name="dirty",
            created_at="2025-01-01T00:00:00Z",
            original_cwd="/tmp",
            original_branch="main",
        )
        mgr._worktrees["dirty"] = wt
        mock_dirty = _mock_subprocess(0, " M file.txt\n")
        with patch("dazi.worktree.subprocess.run", return_value=mock_dirty):
            with pytest.raises(RuntimeError, match="uncommitted changes"):
                mgr.remove("dirty", force=False)

    def test_remove_fallback_to_shutil(self):
        import shutil

        mgr = WorktreeManager()
        mock_path = MagicMock(spec=Path)
        mock_path.__str__ = lambda self: "/tmp/fail-remove"
        mock_path.exists.return_value = True
        wt = Worktree(
            id="fail-remove",
            path=mock_path,
            branch="agent-fail-remove",
            agent_name="fail-remove",
            created_at="2025-01-01T00:00:00Z",
            original_cwd="/tmp",
            original_branch="main",
        )
        mgr._worktrees["fail-remove"] = wt

        with patch("dazi.worktree.subprocess.run", side_effect=Exception("git fail")):
            with patch.object(shutil, "rmtree") as mock_rmtree:
                mgr.remove("fail-remove", force=True)
        mock_rmtree.assert_called_once_with(mock_path, ignore_errors=True)

    def test_remove_branch_delete_exception_ignored(self):
        mgr = WorktreeManager()
        wt = Worktree(
            id="branch-fail",
            path="/tmp/branch-fail",
            branch="agent-branch-fail",
            agent_name="branch-fail",
            created_at="2025-01-01T00:00:00Z",
            original_cwd="/tmp",
            original_branch="main",
        )
        mgr._worktrees["branch-fail"] = wt
        clean_status = _mock_subprocess(0)
        remove_ok = _mock_subprocess(0)
        with patch(
            "dazi.worktree.subprocess.run",
            side_effect=[clean_status, remove_ok, Exception("branch del fail")],
        ):
            result = mgr.remove("branch-fail", force=True)
        assert result is True
        assert mgr.get("branch-fail") is None


# ─────────────────────────────────────────────────────────
# keep
# ─────────────────────────────────────────────────────────


class TestWorktreeManagerKeep:
    def test_keep_existing(self):
        mgr = WorktreeManager()
        wt = Worktree(
            id="keep-me",
            path="/tmp/keep-me",
            branch="agent-keep-me",
            agent_name="keep-me",
            created_at="2025-01-01T00:00:00Z",
            original_cwd="/tmp",
            original_branch="main",
        )
        mgr._worktrees["keep-me"] = wt
        branch = mgr.keep("keep-me")
        assert branch == "agent-keep-me"
        assert mgr.get("keep-me") is None

    def test_keep_nonexistent_raises(self):
        mgr = WorktreeManager()
        with pytest.raises(KeyError, match="No worktree found"):
            mgr.keep("ghost")


# ─────────────────────────────────────────────────────────
# has_uncommitted_changes
# ─────────────────────────────────────────────────────────


class TestHasUncommittedChanges:
    def test_clean_worktree(self):
        mgr = WorktreeManager()
        wt = Worktree(
            id="clean",
            path="/tmp/clean",
            branch="agent-clean",
            agent_name="clean",
            created_at="2025-01-01T00:00:00Z",
            original_cwd="/tmp",
            original_branch="main",
        )
        mgr._worktrees["clean"] = wt
        mock_result = _mock_subprocess(0, "")
        with patch("dazi.worktree.subprocess.run", return_value=mock_result):
            assert mgr.has_uncommitted_changes("clean") is False

    def test_dirty_worktree(self):
        mgr = WorktreeManager()
        wt = Worktree(
            id="dirty",
            path="/tmp/dirty",
            branch="agent-dirty",
            agent_name="dirty",
            created_at="2025-01-01T00:00:00Z",
            original_cwd="/tmp",
            original_branch="main",
        )
        mgr._worktrees["dirty"] = wt
        mock_result = _mock_subprocess(0, " M changed.txt\n")
        with patch("dazi.worktree.subprocess.run", return_value=mock_result):
            assert mgr.has_uncommitted_changes("dirty") is True

    def test_git_command_fails_returns_true(self):
        mgr = WorktreeManager()
        wt = Worktree(
            id="fail",
            path="/tmp/fail",
            branch="agent-fail",
            agent_name="fail",
            created_at="2025-01-01T00:00:00Z",
            original_cwd="/tmp",
            original_branch="main",
        )
        mgr._worktrees["fail"] = wt
        mock_result = _mock_subprocess(1, "", "error")
        with patch("dazi.worktree.subprocess.run", return_value=mock_result):
            assert mgr.has_uncommitted_changes("fail") is True

    def test_exception_returns_true(self):
        mgr = WorktreeManager()
        wt = Worktree(
            id="except",
            path="/tmp/except",
            branch="agent-except",
            agent_name="except",
            created_at="2025-01-01T00:00:00Z",
            original_cwd="/tmp",
            original_branch="main",
        )
        mgr._worktrees["except"] = wt
        with patch("dazi.worktree.subprocess.run", side_effect=TimeoutError):
            assert mgr.has_uncommitted_changes("except") is True

    def test_nonexistent_worktree_returns_false(self):
        mgr = WorktreeManager()
        assert mgr.has_uncommitted_changes("nobody") is False


# ─────────────────────────────────────────────────────────
# cleanup_stale
# ─────────────────────────────────────────────────────────


class TestCleanupStale:
    def _make_wt(self, wt_id: str, created_at: str) -> Worktree:
        return Worktree(
            id=wt_id,
            path=f"/tmp/{wt_id}",
            branch=f"agent-{wt_id}",
            agent_name=wt_id,
            created_at=created_at,
            original_cwd="/tmp",
            original_branch="main",
        )

    def test_no_stale(self):
        mgr = WorktreeManager()
        recent = datetime.now(UTC) - timedelta(days=1)
        mgr._worktrees["recent"] = self._make_wt("recent", recent.isoformat())
        with patch.object(mgr, "has_uncommitted_changes", return_value=False):
            count = mgr.cleanup_stale(cutoff_days=30)
        assert count == 0

    def test_stale_clean_removed(self):
        mgr = WorktreeManager()
        old = datetime.now(UTC) - timedelta(days=100)
        mgr._worktrees["old"] = self._make_wt("old", old.isoformat())
        with patch.object(mgr, "has_uncommitted_changes", return_value=False):
            with patch.object(mgr, "remove", return_value=True) as mock_remove:
                count = mgr.cleanup_stale(cutoff_days=30)
        assert count == 1
        mock_remove.assert_called_once_with("old", force=True)

    def test_stale_dirty_not_removed(self):
        mgr = WorktreeManager()
        old = datetime.now(UTC) - timedelta(days=100)
        mgr._worktrees["old-dirty"] = self._make_wt("old-dirty", old.isoformat())
        with patch.object(mgr, "has_uncommitted_changes", return_value=True):
            count = mgr.cleanup_stale(cutoff_days=30)
        assert count == 0

    def test_stale_remove_exception_skipped(self):
        mgr = WorktreeManager()
        old = datetime.now(UTC) - timedelta(days=100)
        mgr._worktrees["old-fail"] = self._make_wt("old-fail", old.isoformat())
        with patch.object(mgr, "has_uncommitted_changes", return_value=False):
            with patch.object(mgr, "remove", side_effect=RuntimeError("boom")):
                count = mgr.cleanup_stale(cutoff_days=30)
        assert count == 0

    def test_bad_timestamp_skipped(self):
        mgr = WorktreeManager()
        mgr._worktrees["bad-ts"] = self._make_wt("bad-ts", "not-a-timestamp")
        with patch.object(mgr, "has_uncommitted_changes", return_value=False):
            count = mgr.cleanup_stale(cutoff_days=30)
        assert count == 0

    def test_uses_config_default_cutoff(self):
        mgr = WorktreeManager(config=WorktreeConfig(stale_cutoff_days=7))
        old = datetime.now(UTC) - timedelta(days=20)
        mgr._worktrees["old"] = self._make_wt("old", old.isoformat())
        with patch.object(mgr, "has_uncommitted_changes", return_value=False):
            with patch.object(mgr, "remove", return_value=True):
                count = mgr.cleanup_stale()
        assert count == 1

    def test_none_worktree_during_iteration(self):
        mgr = WorktreeManager()
        old = datetime.now(UTC) - timedelta(days=100)
        mgr._worktrees["old"] = self._make_wt("old", old.isoformat())
        with patch.object(mgr, "has_uncommitted_changes", return_value=False):
            # Simulate worktree removed mid-iteration
            with patch.dict(mgr._worktrees, {}, clear=True):
                count = mgr.cleanup_stale(cutoff_days=30)
        assert count == 0


# ─────────────────────────────────────────────────────────
# reset
# ─────────────────────────────────────────────────────────


class TestReset:
    def test_reset_clears_all(self):
        mgr = WorktreeManager()
        wt = Worktree(
            id="r",
            path="/tmp/r",
            branch="agent-r",
            agent_name="r",
            created_at="2025-01-01T00:00:00Z",
            original_cwd="/tmp",
            original_branch="main",
        )
        mgr._worktrees["r"] = wt
        mgr.reset()
        assert mgr.list_all() == []


# ─────────────────────────────────────────────────────────
# create_worktree_func (async tool function)
# ─────────────────────────────────────────────────────────


class TestCreateWorktreeFunc:
    @pytest.mark.asyncio
    async def test_creates_worktree(self):
        mgr = WorktreeManager()
        mock_wt = Worktree(
            id="agent",
            path="/tmp/agent",
            branch="agent-agent",
            agent_name="agent",
            created_at="2025-01-01T00:00:00Z",
            original_cwd="/tmp",
            original_branch="main",
        )
        with patch("dazi.worktree.subprocess.run", return_value=_mock_subprocess(0)):
            with patch.object(Path, "cwd", return_value=Path("/cwd")):
                with patch("dazi._singletons.worktree_manager", mgr):
                    with patch.object(mgr, "create", return_value=mock_wt) as mock_create:
                        result = await create_worktree_func("agent")
        mock_create.assert_called_once_with("agent", base_branch=None)
        assert "Created worktree" in result
        assert "/tmp/agent" in result

    @pytest.mark.asyncio
    async def test_creates_with_base_branch(self):
        mgr = WorktreeManager()
        mock_wt = Worktree(
            id="feat",
            path="/tmp/feat",
            branch="agent-feat",
            agent_name="feat",
            created_at="2025-01-01T00:00:00Z",
            original_cwd="/tmp",
            original_branch="main",
        )
        with patch("dazi._singletons.worktree_manager", mgr):
            with patch.object(mgr, "create", return_value=mock_wt) as mock_create:
                result = await create_worktree_func("feat", base_branch="develop")
        mock_create.assert_called_once_with("feat", base_branch="develop")
        assert "Created worktree" in result


# ─────────────────────────────────────────────────────────
# finish_worktree_func (async tool function)
# ─────────────────────────────────────────────────────────


class TestFinishWorktreeFunc:
    @pytest.mark.asyncio
    async def test_finish_keep(self):
        mgr = WorktreeManager()
        wt = Worktree(
            id="keep",
            path="/tmp/keep",
            branch="agent-keep",
            agent_name="keep",
            created_at="2025-01-01T00:00:00Z",
            original_cwd="/tmp",
            original_branch="main",
        )
        with patch("dazi._singletons.worktree_manager", mgr):
            with patch.object(mgr, "get", return_value=wt):
                with patch.object(mgr, "keep", return_value="agent-keep"):
                    result = await finish_worktree_func("keep", action="keep")
        assert "Kept worktree" in result
        assert "agent-keep" in result

    @pytest.mark.asyncio
    async def test_finish_remove_success(self):
        mgr = WorktreeManager()
        wt = Worktree(
            id="rem",
            path="/tmp/rem",
            branch="agent-rem",
            agent_name="rem",
            created_at="2025-01-01T00:00:00Z",
            original_cwd="/tmp",
            original_branch="main",
        )
        with patch("dazi._singletons.worktree_manager", mgr):
            with patch.object(mgr, "get", return_value=wt):
                with patch.object(mgr, "has_uncommitted_changes", return_value=False):
                    with patch.object(mgr, "remove", return_value=True):
                        result = await finish_worktree_func("rem", action="remove")
        assert "Removed worktree" in result

    @pytest.mark.asyncio
    async def test_finish_remove_dirty_no_force(self):
        mgr = WorktreeManager()
        wt = Worktree(
            id="dirty",
            path="/tmp/dirty",
            branch="agent-dirty",
            agent_name="dirty",
            created_at="2025-01-01T00:00:00Z",
            original_cwd="/tmp",
            original_branch="main",
        )
        with patch("dazi._singletons.worktree_manager", mgr):
            with patch.object(mgr, "get", return_value=wt):
                with patch.object(mgr, "has_uncommitted_changes", return_value=True):
                    result = await finish_worktree_func("dirty", action="remove", force=False)
        assert "uncommitted changes" in result

    @pytest.mark.asyncio
    async def test_finish_remove_dirty_with_force(self):
        mgr = WorktreeManager()
        wt = Worktree(
            id="force",
            path="/tmp/force",
            branch="agent-force",
            agent_name="force",
            created_at="2025-01-01T00:00:00Z",
            original_cwd="/tmp",
            original_branch="main",
        )
        with patch("dazi._singletons.worktree_manager", mgr):
            with patch.object(mgr, "get", return_value=wt):
                with patch.object(mgr, "has_uncommitted_changes", return_value=True):
                    with patch.object(mgr, "remove", return_value=True):
                        result = await finish_worktree_func("force", action="remove", force=True)
        assert "Removed worktree" in result

    @pytest.mark.asyncio
    async def test_finish_remove_failed(self):
        mgr = WorktreeManager()
        wt = Worktree(
            id="fail",
            path="/tmp/fail",
            branch="agent-fail",
            agent_name="fail",
            created_at="2025-01-01T00:00:00Z",
            original_cwd="/tmp",
            original_branch="main",
        )
        with patch("dazi._singletons.worktree_manager", mgr):
            with patch.object(mgr, "get", return_value=wt):
                with patch.object(mgr, "has_uncommitted_changes", return_value=False):
                    with patch.object(mgr, "remove", return_value=False):
                        result = await finish_worktree_func("fail", action="remove")
        assert "Failed to remove" in result

    @pytest.mark.asyncio
    async def test_finish_unknown_action(self):
        mgr = WorktreeManager()
        wt = Worktree(
            id="x",
            path="/tmp/x",
            branch="agent-x",
            agent_name="x",
            created_at="2025-01-01T00:00:00Z",
            original_cwd="/tmp",
            original_branch="main",
        )
        with patch("dazi._singletons.worktree_manager", mgr):
            with patch.object(mgr, "get", return_value=wt):
                result = await finish_worktree_func("x", action="explode")
        assert "Unknown action" in result

    @pytest.mark.asyncio
    async def test_finish_nonexistent_agent(self):
        mgr = WorktreeManager()
        with patch("dazi._singletons.worktree_manager", mgr):
            with patch.object(mgr, "get", return_value=None):
                result = await finish_worktree_func("nobody", action="keep")
        assert "No worktree found" in result


# ─────────────────────────────────────────────────────────
# list_worktrees_func (async tool function)
# ─────────────────────────────────────────────────────────


class TestListWorktreesFunc:
    @pytest.mark.asyncio
    async def test_list_empty(self):
        mgr = WorktreeManager()
        with patch("dazi._singletons.worktree_manager", mgr):
            result = await list_worktrees_func()
        assert result == "No active worktrees."

    @pytest.mark.asyncio
    async def test_list_with_worktrees(self):
        mgr = WorktreeManager()
        wt = Worktree(
            id="a1",
            path="/tmp/a1",
            branch="agent-a1",
            agent_name="a1",
            created_at="2025-01-01T00:00:00Z",
            original_cwd="/tmp",
            original_branch="main",
        )
        mgr._worktrees["a1"] = wt
        with patch("dazi._singletons.worktree_manager", mgr):
            with patch.object(mgr, "has_uncommitted_changes", return_value=False):
                result = await list_worktrees_func()
        assert "Active worktrees:" in result
        assert "a1" in result

    @pytest.mark.asyncio
    async def test_list_with_dirty_worktree(self):
        mgr = WorktreeManager()
        wt = Worktree(
            id="dirty",
            path="/tmp/dirty",
            branch="agent-dirty",
            agent_name="dirty",
            created_at="2025-01-01T00:00:00Z",
            original_cwd="/tmp",
            original_branch="main",
        )
        mgr._worktrees["dirty"] = wt
        with patch("dazi._singletons.worktree_manager", mgr):
            with patch.object(mgr, "has_uncommitted_changes", return_value=True):
                result = await list_worktrees_func()
        assert "(dirty)" in result
