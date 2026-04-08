"""Git worktree isolation — create, keep, remove lifecycle.

KEY CONCEPTS:
  1. Git worktrees give each agent its own working directory on a separate branch
  2. Parallel agents can edit the same files without merge conflicts
  3. Lifecycle: create → agent works → keep (preserve branch) or remove (clean up)
  4. Safety: refuses to remove worktrees with uncommitted changes unless forced
  5. Slug validation prevents path traversal attacks
"""

from __future__ import annotations

import os
import re
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from dazi.core.base import DaziTool, ToolSafety


# ─────────────────────────────────────────────────────────
# WORKTREE DATACLASS
# ─────────────────────────────────────────────────────────


@dataclass
class Worktree:
    """Represents a tracked git worktree.

    Each worktree is a full checkout on a separate branch.
    Edits in one worktree don't affect another.
    """

    id: str                  # Sanitized agent name (slug)
    path: Path               # Absolute path to worktree directory
    branch: str              # Branch name (e.g., "agent-frontend")
    agent_name: str          # Original agent name (before sanitization)
    created_at: str          # ISO timestamp
    original_cwd: str        # CWD before entering worktree
    original_branch: str     # Branch before creating worktree


# ─────────────────────────────────────────────────────────
# WORKTREE CONFIG
# ─────────────────────────────────────────────────────────


@dataclass
class WorktreeConfig:
    """Configuration for worktree operations.

    Attributes:
        worktree_base: Relative path (from repo root) for worktree storage.
        branch_prefix: Prefix for auto-generated branch names.
        max_slug_length: Maximum characters for worktree slug.
        stale_cutoff_days: Days before stale worktree cleanup.
    """

    worktree_base: str = ".dazi/worktrees"
    branch_prefix: str = "agent-"
    max_slug_length: int = 64
    stale_cutoff_days: int = 30


# ─────────────────────────────────────────────────────────
# WORKTREE MANAGER
# ─────────────────────────────────────────────────────────


class WorktreeManager:
    """Git worktree lifecycle management.

    All git operations use subprocess.run() with explicit arguments.
    No gitpython dependency.
    """

    VALID_SLUG_SEGMENT = re.compile(r"^[a-zA-Z0-9._-]+$")

    def __init__(self, config: WorktreeConfig | None = None) -> None:
        self._config = config or WorktreeConfig()
        self._worktrees: dict[str, Worktree] = {}  # id -> Worktree

    # ── Slug Validation ─────────────────────────────────

    def validate_slug(self, slug: str) -> None:
        """Validate worktree slug to prevent path traversal.

        Rules:
          1. Total length must not exceed max_slug_length
          2. Each segment (split by "/") must match VALID_SLUG_SEGMENT
          3. No segment can be "." or ".."
          4. Total length must be > 0

        Raises:
            ValueError: If slug is invalid.
        """
        if not slug:
            raise ValueError("Worktree slug cannot be empty.")

        if len(slug) > self._config.max_slug_length:
            raise ValueError(
                f"Worktree slug too long: {len(slug)} chars "
                f"(max {self._config.max_slug_length})"
            )

        segments = slug.split("/")
        for segment in segments:
            if not segment:
                raise ValueError("Worktree slug cannot contain empty segments.")
            if segment == ".":
                raise ValueError(
                    'Worktree slug cannot contain "." segments (path traversal).'
                )
            if segment == "..":
                raise ValueError(
                    'Worktree slug cannot contain ".." segments (path traversal).'
                )
            if not self.VALID_SLUG_SEGMENT.match(segment):
                raise ValueError(
                    f"Invalid worktree slug segment: '{segment}'. "
                    f"Only alphanumeric, dots, underscores, and dashes allowed."
                )

    def sanitize_agent_name(self, agent_name: str) -> str:
        """Sanitize agent name into a valid worktree slug.

        Replaces spaces and special characters with dashes,
        lowercases, and strips leading/trailing dashes.
        """
        # Replace non-alphanumeric (except dots, underscores, dashes) with dash
        slug = re.sub(r"[^a-zA-Z0-9._-]", "-", agent_name)
        # Collapse multiple dashes
        slug = re.sub(r"-+", "-", slug)
        # Strip leading/trailing dashes
        slug = slug.strip("-")
        # Lowercase for consistency
        slug = slug.lower()
        return slug

    # ── Repository Detection ────────────────────────────

    def _get_repo_root(self, cwd: Path | None = None) -> Path:
        """Find git repo root via 'git rev-parse --show-toplevel'.

        Returns:
            Absolute Path to repo root.

        Raises:
            RuntimeError: If not inside a git repository.
        """
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--show-toplevel"],
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode != 0:
                raise RuntimeError(f"Not a git repository: {result.stderr.strip()}")
            return Path(result.stdout.strip()).resolve()
        except FileNotFoundError:
            raise RuntimeError("git not found. Is git installed?")

    def _get_current_branch(self, cwd: Path | None = None) -> str:
        """Get current branch name via 'git rev-parse --abbrev-ref HEAD'.

        Returns:
            Branch name string.

        Raises:
            RuntimeError: If git command fails.
        """
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode != 0:
                raise RuntimeError(f"Cannot determine branch: {result.stderr.strip()}")
            return result.stdout.strip()
        except FileNotFoundError:
            raise RuntimeError("git not found. Is git installed?")

    # ── Worktree Creation ───────────────────────────────

    def create(
        self,
        agent_name: str,
        base_branch: str | None = None,
    ) -> Worktree:
        """Create a git worktree for isolated agent work.

        Steps:
        1. Sanitize agent_name -> slug
        2. Validate slug
        3. Get repo root and current branch
        4. Construct path: repo_root / worktree_base / slug
        5. Construct branch: branch_prefix + slug
        6. Run: git worktree add <path> -b <branch> [base]
        7. Store Worktree in _worktrees dict
        8. Return Worktree

        Args:
            agent_name: Name for the agent/worktree (e.g., "frontend").
            base_branch: Optional base branch. Defaults to current HEAD.

        Returns:
            Worktree with path, branch, and tracking info.

        Raises:
            ValueError: If slug is invalid or worktree already exists.
            RuntimeError: If git command fails.
        """
        slug = self.sanitize_agent_name(agent_name)
        self.validate_slug(slug)

        # Prevent double-creation
        if slug in self._worktrees:
            raise ValueError(
                f"Worktree already exists for '{agent_name}' (slug: {slug}). "
                f"Use finish_worktree first."
            )

        repo_root = self._get_repo_root()
        current_branch = self._get_current_branch()
        original_cwd = str(Path.cwd())

        worktree_path = repo_root / self._config.worktree_base / slug
        branch_name = f"{self._config.branch_prefix}{slug}"

        # Ensure worktree base directory exists
        try:
            worktree_path.parent.mkdir(parents=True, exist_ok=True)
        except OSError:
            pass  # Will be created by git worktree add if needed

        # Build git worktree add command
        cmd = [
            "git", "worktree", "add",
            str(worktree_path),
            "-b", branch_name,
        ]
        if base_branch:
            cmd.append(base_branch)

        result = subprocess.run(
            cmd,
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=30,
        )

        if result.returncode != 0:
            raise RuntimeError(
                f"Failed to create worktree: {result.stderr.strip()}"
            )

        worktree = Worktree(
            id=slug,
            path=worktree_path,
            branch=branch_name,
            agent_name=agent_name,
            created_at=datetime.now(timezone.utc).isoformat(),
            original_cwd=original_cwd,
            original_branch=current_branch,
        )

        self._worktrees[slug] = worktree
        return worktree

    # ── Worktree Removal ────────────────────────────────

    def remove(self, worktree_id: str, force: bool = False) -> bool:
        """Remove a worktree and its branch.

        Safety checks (unless force=True):
        1. Check for uncommitted changes
        2. If changes exist, raise RuntimeError
        3. Run: git worktree remove --force <path>
        4. Run: git branch -D <branch>
        5. Remove from _worktrees dict

        Args:
            worktree_id: The worktree slug to remove.
            force: If True, remove even with uncommitted changes.

        Returns:
            True on success.

        Raises:
            RuntimeError: If worktree has uncommitted changes and force=False.
            KeyError: If worktree_id not found.
        """
        wt = self._worktrees.get(worktree_id)
        if wt is None:
            return False

        # Safety: refuse if dirty unless forced
        if not force and self.has_uncommitted_changes(worktree_id):
            raise RuntimeError(
                f"Worktree '{worktree_id}' has uncommitted changes. "
                f"Use force=True to discard, or keep() to preserve the branch."
            )

        # Remove worktree directory
        try:
            subprocess.run(
                ["git", "worktree", "remove", "--force", str(wt.path)],
                capture_output=True,
                text=True,
                timeout=30,
            )
        except Exception:
            # Fallback: rm -rf
            if wt.path.exists():
                import shutil
                shutil.rmtree(wt.path, ignore_errors=True)

        # Delete the branch
        try:
            subprocess.run(
                ["git", "branch", "-D", wt.branch],
                capture_output=True,
                text=True,
                timeout=10,
            )
        except Exception:
            pass  # Branch may already be gone

        del self._worktrees[worktree_id]
        return True

    # ── Worktree Keep ───────────────────────────────────

    def keep(self, worktree_id: str) -> str:
        """Keep the worktree branch but stop tracking it.

        Returns the branch name for the caller to reference later.
        Removes from _worktrees tracking but leaves filesystem intact.

        Args:
            worktree_id: The worktree slug to keep.

        Returns:
            The branch name (e.g., "agent-frontend").

        Raises:
            KeyError: If worktree_id not found.
        """
        wt = self._worktrees.get(worktree_id)
        if wt is None:
            raise KeyError(f"No worktree found for '{worktree_id}'")

        branch = wt.branch
        del self._worktrees[worktree_id]
        return branch

    # ── Change Detection ────────────────────────────────

    def has_uncommitted_changes(self, worktree_id: str) -> bool:
        """Check for uncommitted changes in a worktree.

        Uses: git -C <path> status --porcelain
        Returns True if any output lines exist (uncommitted changes).
        Fails closed: returns True if git command fails.
        """
        wt = self._worktrees.get(worktree_id)
        if wt is None:
            return False

        try:
            result = subprocess.run(
                ["git", "-C", str(wt.path), "status", "--porcelain"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            # Fail closed: if git fails, assume changes exist
            if result.returncode != 0:
                return True
            return bool(result.stdout.strip())
        except Exception:
            # Fail closed on any exception
            return True

    # ── Listing ─────────────────────────────────────────

    def list_all(self) -> list[Worktree]:
        """List all tracked worktrees."""
        return list(self._worktrees.values())

    def get(self, worktree_id: str) -> Worktree | None:
        """Get a tracked worktree by ID (slug)."""
        return self._worktrees.get(worktree_id)

    # ── Stale Cleanup ───────────────────────────────────

    def cleanup_stale(self, cutoff_days: int | None = None) -> int:
        """Remove worktrees older than cutoff_days that have no changes.

        Only removes worktrees that are:
        1. Older than cutoff_days
        2. Have no uncommitted changes

        Args:
            cutoff_days: Days threshold. Defaults to config.stale_cutoff_days.

        Returns:
            Count of removed worktrees.
        """
        if cutoff_days is None:
            cutoff_days = self._config.stale_cutoff_days

        now = time.time()
        cutoff_seconds = cutoff_days * 86400
        removed = 0

        to_check = list(self._worktrees.keys())
        for wt_id in to_check:
            wt = self._worktrees.get(wt_id)
            if wt is None:
                continue

            # Parse created_at timestamp
            try:
                created = datetime.fromisoformat(wt.created_at)
                age_seconds = (now - created.timestamp())
            except (ValueError, OSError):
                continue

            if age_seconds < cutoff_seconds:
                continue

            # Only remove if clean
            if not self.has_uncommitted_changes(wt_id):
                try:
                    self.remove(wt_id, force=True)
                    removed += 1
                except Exception:
                    pass

        return removed

    # ── Reset ───────────────────────────────────────────

    def reset(self) -> None:
        """Clear all tracked worktrees. For testing only."""
        self._worktrees.clear()


# ─────────────────────────────────────────────────────────
# WORKTREE TOOLS
# ─────────────────────────────────────────────────────────

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field


class CreateWorktreeInput(BaseModel):
    agent_name: str = Field(description="Name for the agent/worktree. Alphanumeric, dots, underscores, and dashes only.")
    base_branch: str | None = Field(default=None, description="Optional base branch for the worktree. Defaults to current HEAD.")


async def create_worktree_func(agent_name: str, base_branch: str | None = None) -> str:
    """Create a git worktree for isolated agent work."""
    from dazi.core._singletons import worktree_manager

    wt = worktree_manager.create(agent_name, base_branch=base_branch)
    return (
        f"Created worktree at {wt.path} on branch {wt.branch}.\n"
        f"Agent '{agent_name}' can now work in isolation. "
        f"Use finish_worktree to clean up when done."
    )


create_worktree_tool = StructuredTool.from_function(
    func=lambda **kwargs: "",
    coroutine=create_worktree_func,
    name="create_worktree",
    description="Create a git worktree for isolated agent work. Each worktree is a full checkout on a separate branch.",
    args_schema=CreateWorktreeInput,
)
create_worktree_meta = DaziTool(name="create_worktree", description="Create a git worktree for isolated agent work.", safety=ToolSafety.WRITE)


class FinishWorktreeInput(BaseModel):
    agent_name: str = Field(description="Name of the agent whose worktree to finish.")
    action: str = Field(description='"keep" to preserve branch for manual merge, "remove" to delete entirely.')
    force: bool = Field(default=False, description="Required when action is 'remove' and worktree has uncommitted changes.")


async def finish_worktree_func(agent_name: str, action: str = "keep", force: bool = False) -> str:
    """Finish a worktree: keep or remove."""
    from dazi.core._singletons import worktree_manager

    slug = worktree_manager.sanitize_agent_name(agent_name)
    wt = worktree_manager.get(slug)
    if wt is None:
        return f"No worktree found for agent '{agent_name}'."

    if action == "keep":
        branch = worktree_manager.keep(slug)
        return f"Kept worktree. Branch '{branch}' preserved at {wt.path}.\nYou can merge changes from this branch."
    elif action == "remove":
        if worktree_manager.has_uncommitted_changes(slug) and not force:
            return "Worktree has uncommitted changes. Use force=true to discard, or action='keep' to preserve."
        removed = worktree_manager.remove(slug, force=force)
        return f"Removed worktree for '{agent_name}'." if removed else "Failed to remove worktree."
    else:
        return f"Unknown action '{action}'. Use 'keep' or 'remove'."


finish_worktree_tool = StructuredTool.from_function(
    func=lambda **kwargs: "",
    coroutine=finish_worktree_func,
    name="finish_worktree",
    description="Finish a worktree: keep the branch for manual merge, or remove entirely.",
    args_schema=FinishWorktreeInput,
)
finish_worktree_meta = DaziTool(name="finish_worktree", description="Finish a worktree: keep or remove.", safety=ToolSafety.WRITE)


class ListWorktreesInput(BaseModel):
    pass


async def list_worktrees_func() -> str:
    """List all active worktrees."""
    from dazi.core._singletons import worktree_manager

    worktrees = worktree_manager.list_all()
    if not worktrees:
        return "No active worktrees."
    lines = ["Active worktrees:"]
    for wt in worktrees:
        dirty = " (dirty)" if worktree_manager.has_uncommitted_changes(wt.id) else ""
        lines.append(f"  {wt.agent_name}: {wt.path} [{wt.branch}]{dirty}")
    return "\n".join(lines)


list_worktrees_tool = StructuredTool.from_function(
    func=lambda: "No active worktrees.",
    coroutine=list_worktrees_func,
    name="list_worktrees",
    description="List all active git worktrees with dirty status.",
    args_schema=ListWorktreesInput,
)
list_worktrees_meta = DaziTool(name="list_worktrees", description="List all active git worktrees.", safety=ToolSafety.SAFE)
