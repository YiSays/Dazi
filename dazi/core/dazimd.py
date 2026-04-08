"""DAZI.md hierarchical loading with @include support.

DAZI.md files are loaded from multiple locations with a priority
chain. Higher-priority files override lower-priority ones. The system supports
@include directives for composing files from other paths.

Priority chain (highest to lowest):
  1. DAZI.local.md in project root (private, project-specific, not checked in)
  2. DAZI.md in project root (checked-in, shared with team)
  3. ~/.dazi/DAZI.md (user's private global instructions)

Key behaviors:
  - Duplicate content across files is deduplicated
  - @include directives are resolved recursively
  - Loading is triggered at session start and on file watcher events
  - Content is injected into the static section of the system prompt (before DYNAMIC_BOUNDARY)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


# ─────────────────────────────────────────────────────────
# DAZI.md FILE
# ─────────────────────────────────────────────────────────

@dataclass
class DaziMdFile:
    """A discovered DAZI.md file with its priority."""
    path: Path
    priority: int
    content: str = ""


# ─────────────────────────────────────────────────────────
# PRIORITY LEVELS
# ─────────────────────────────────────────────────────────

# Priority constants (higher = more important)
PRIORITY_LOCAL = 400      # DAZI.local.md — private, project-specific
PRIORITY_PROJECT = 300    # DAZI.md — checked-in, shared with team
PRIORITY_GLOBAL = 100     # ~/.dazi/DAZI.md — user global


# ─────────────────────────────────────────────────────────
# @include PATTERN
# ─────────────────────────────────────────────────────────

INCLUDE_PATTERN = re.compile(r"^@include\s+(.+)$", re.MULTILINE)


# ─────────────────────────────────────────────────────────
# DISCOVERY
# ─────────────────────────────────────────────────────────

def discover_dazimd_files(
    project_root: Path | None = None,
    cwd: Path | None = None,
) -> list[DaziMdFile]:
    """Discover DAZI.md files by priority chain.

    Searches in this order (highest to lowest priority):
    1. <project_root>/DAZI.local.md
    2. <project_root>/DAZI.md
    3. ~/.dazi/DAZI.md

    Args:
        project_root: Project root directory. Defaults to cwd.
        cwd: Current working directory. Defaults to Path.cwd().

    Returns:
        List of DaziMdFile objects sorted by priority (highest first).
    """
    root = project_root or cwd or Path.cwd()
    files: list[DaziMdFile] = []

    def _try_load(path: Path, priority: int) -> None:
        """Try to load a DAZI.md file if it exists."""
        if path.exists() and path.is_file():
            try:
                content = path.read_text(encoding="utf-8")
                resolved = resolve_includes(content, path.parent)
                files.append(DaziMdFile(
                    path=path,
                    priority=priority,
                    content=resolved,
                ))
            except Exception:
                pass  # Skip files we can't read

    # 1. DAZI.local.md — highest priority
    _try_load(root / "DAZI.local.md", PRIORITY_LOCAL)

    # 2. DAZI.md in project root
    _try_load(root / "DAZI.md", PRIORITY_PROJECT)

    # 3. ~/.dazi/DAZI.md — user global (lowest priority)
    global_md = Path.home() / ".dazi" / "DAZI.md"
    _try_load(global_md, PRIORITY_GLOBAL)

    # Sort by priority descending (highest first)
    files.sort(key=lambda f: f.priority, reverse=True)
    return files


# ─────────────────────────────────────────────────────────
# @include RESOLUTION
# ─────────────────────────────────────────────────────────

def resolve_includes(content: str, base_path: Path) -> str:
    """Process @include directives in DAZI.md content.

    @include directives allow composing DAZI.md from other files:
        @include .dazi/rules/coding-style.md
        @include ~/dazi/global-rules.md
        @include ./relative/path.md

    Includes are resolved recursively. Circular includes are detected and stopped.

    Args:
        content: The content potentially containing @include directives.
        base_path: Directory to resolve relative paths from.

    Returns:
        Content with all @include directives resolved.
    """
    return _resolve_includes_inner(content, base_path, set())


def _resolve_includes_inner(
    content: str,
    base_path: Path,
    seen: set[str],
) -> str:
    """Recursive @include resolution with circular detection."""

    def replace_include(match: re.Match) -> str:
        include_path_str = match.group(1).strip()

        # Expand ~ to home directory
        if include_path_str.startswith("~"):
            include_path = Path.home() / include_path_str[1:].lstrip("/")
        elif include_path_str.startswith("/"):
            include_path = Path(include_path_str)
        else:
            include_path = base_path / include_path_str

        # Normalize for circular detection
        resolved_key = str(include_path.resolve())

        if resolved_key in seen:
            return f"[Circular include detected: {include_path_str}]"

        if not include_path.exists():
            return f"[Include not found: {include_path_str}]"

        try:
            included_content = include_path.read_text(encoding="utf-8")
            # Recursively resolve nested includes
            return _resolve_includes_inner(
                included_content,
                include_path.parent,
                seen | {resolved_key},
            )
        except Exception as e:
            return f"[Include error: {e}]"

    return INCLUDE_PATTERN.sub(replace_include, content)


# ─────────────────────────────────────────────────────────
# MERGE LOADED FILES
# ─────────────────────────────────────────────────────────

def merge_dazimd_content(files: list[DaziMdFile]) -> str:
    """Merge multiple DAZI.md files into a single content string.

    Files are already sorted by priority (highest first). Content from
    higher-priority files comes first. Duplicate lines are removed.

    Args:
        files: List of DaziMdFile objects sorted by priority.

    Returns:
        Merged content string with deduplication.
    """
    if not files:
        return ""

    seen_lines: set[str] = set()
    merged_parts: list[str] = []

    for f in files:
        if not f.content.strip():
            continue

        # Track which file the content came from
        if len(files) > 1:
            merged_parts.append(f"<!-- From: {f.path} (priority: {f.priority}) -->")

        for line in f.content.splitlines():
            stripped = line.strip()
            # Skip empty lines that we've already seen (dedup blank lines)
            if not stripped:
                continue
            if stripped in seen_lines:
                continue
            seen_lines.add(stripped)
            merged_parts.append(line)

    return "\n".join(merged_parts)
