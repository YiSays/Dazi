"""Memory system — persistent storage with index management.

Provides a memory store for persisting information across conversations.
Each memory is stored as an individual .md file with YAML frontmatter,
organized by category and searchable by keyword relevance.

Architecture:
    memories/
    ├── MEMORY.md          — index file (max 200 lines, max 25KB)
    ├── user_preferences.md   — individual memory file
    ├── project_notes.md
    └── feedback_notes.md

Each memory file has YAML frontmatter with metadata and a content body.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from dazi.core.base import DaziTool, ToolSafety


# ─────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────

MAX_ENTRYPOINT_LINES = 200       # MEMORY.md max lines
MAX_MEMORY_INDEX_SIZE = 25_000   # MEMORY.md max bytes


# ─────────────────────────────────────────────────────────
# MEMORY CATEGORIES
# ─────────────────────────────────────────────────────────

class MemoryCategory(str, Enum):
    """Types of memories in Dazi's memory taxonomy."""
    USER = "user"              # User preferences and identity
    FEEDBACK = "feedback"      # Behavioral guidance (what to do/avoid)
    PROJECT = "project"        # Project-specific knowledge
    REFERENCE = "reference"    # External pointers and resources


# ─────────────────────────────────────────────────────────
# MEMORY ENTRY
# ─────────────────────────────────────────────────────────

@dataclass
class MemoryEntry:
    """A single memory stored as an individual .md file.

    Storage format (markdown with YAML frontmatter):
    ```markdown
    ---
    id: abc123
    category: user
    created_at: 2025-04-03T12:34:56
    tags: [preference, programming]
    ---
    User prefers functional programming.
    ```
    """
    content: str
    category: MemoryCategory = MemoryCategory.USER
    id: str = field(default_factory=lambda: _generate_id())
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    tags: list[str] = field(default_factory=list)
    description: str = ""  # One-line summary for MEMORY.md index

    @classmethod
    def from_markdown(cls, content: str) -> MemoryEntry:
        """Parse a memory .md file into a MemoryEntry.

        Handles YAML frontmatter format:
        ```
        ---
        id: abc123
        category: user
        created_at: 2025-04-03T12:34:56
        tags: [preference]
        description: User prefers functional programming
        ---
        Body content here.
        ```
        """
        # Split frontmatter from body
        if content.startswith("---"):
            parts = content.split("---", 2)
            if len(parts) >= 3:
                frontmatter_text = parts[1].strip()
                body = parts[2].strip()
                meta = _parse_frontmatter(frontmatter_text)
                return cls(
                    id=meta.get("id", _generate_id()),
                    category=MemoryCategory(meta.get("category", "user")),
                    content=body,
                    created_at=meta.get("created_at", datetime.now().isoformat()),
                    tags=meta.get("tags", []),
                    description=meta.get("description", ""),
                )
        # No frontmatter — entire content is the body
        return cls(content=content)

    def to_markdown(self) -> str:
        """Serialize a MemoryEntry to markdown with YAML frontmatter."""
        desc = self.description or self.content[:100]
        tags_str = str(self.tags) if self.tags else "[]"
        return f"""\
---
id: {self.id}
category: {self.category.value}
created_at: {self.created_at}
tags: {tags_str}
description: {desc}
---
{self.content}"""


def _generate_id() -> str:
    """Generate a short unique ID for a memory entry."""
    return datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]


def _parse_frontmatter(text: str) -> dict[str, Any]:
    """Parse simple YAML-like frontmatter into a dict.

    Handles: key: value, key: [item1, item2]
    This is NOT a full YAML parser — just enough for our format.
    """
    result: dict[str, Any] = {}
    for line in text.splitlines():
        line = line.strip()
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        key = key.strip()
        value = value.strip()
        # Handle list values: [item1, item2]
        if value.startswith("[") and value.endswith("]"):
            inner = value[1:-1]
            result[key] = [item.strip().strip("'\"") for item in inner.split(",") if item.strip()]
        else:
            # Strip quotes
            value = value.strip("'\"")
            result[key] = value
    return result


# ─────────────────────────────────────────────────────────
# MEMORY STORE
# ─────────────────────────────────────────────────────────

class MemoryStore:
    """Persistent memory storage with index management.

    Stores individual memories as .md files in a memory directory.
    Maintains a MEMORY.md index for quick access and LLM context injection.

    Storage layout:
        <memory_dir>/
        ├── MEMORY.md              — index file
        ├── <id>.md                — individual memory files
        └── ...

    The index file has format:
        # Memory Index
        <entry> — <description>
    """

    def __init__(self, memory_dir: Path) -> None:
        self.memory_dir = memory_dir
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self._index_path = self.memory_dir / "MEMORY.md"

    @property
    def index_path(self) -> Path:
        return self._index_path

    # ── CRUD ──

    def write(self, entry: MemoryEntry) -> Path:
        """Write a memory entry to disk.

        Returns the path to the written file.
        """
        file_path = self.memory_dir / f"{entry.id}.md"
        file_path.write_text(entry.to_markdown(), encoding="utf-8")
        self.rebuild_index()
        return file_path

    def read(self, memory_id: str) -> MemoryEntry | None:
        """Read a memory entry by ID."""
        file_path = self.memory_dir / f"{memory_id}.md"
        if not file_path.exists():
            return None
        try:
            content = file_path.read_text(encoding="utf-8")
            return MemoryEntry.from_markdown(content)
        except Exception:
            return None

    def delete(self, memory_id: str) -> bool:
        """Delete a memory entry by ID.

        Returns True if deleted, False if not found.
        """
        file_path = self.memory_dir / f"{memory_id}.md"
        if file_path.exists():
            file_path.unlink()
            self.rebuild_index()
            return True
        return False

    def list_all(self) -> list[MemoryEntry]:
        """List all memory entries."""
        entries: list[MemoryEntry] = []
        for file_path in sorted(self.memory_dir.glob("*.md")):
            if file_path.name == "MEMORY.md":
                continue
            try:
                content = file_path.read_text(encoding="utf-8")
                entry = MemoryEntry.from_markdown(content)
                entries.append(entry)
            except Exception:
                continue
        return entries

    # ── SEARCH ──

    def find_relevant(
        self,
        query: str,
        limit: int = 5,
    ) -> list[MemoryEntry]:
        """Find memories relevant to a query.

        Uses keyword-based relevance scoring:
        1. Split query into terms
        2. Score each memory by term overlap
        3. Boost matches in category name and tags
        4. Return top-N by score
        """
        all_entries = self.list_all()
        if not all_entries or not query.strip():
            return []

        # Tokenize query into terms
        query_terms = set(_tokenize(query.lower()))
        if not query_terms:
            return []

        scored: list[tuple[MemoryEntry, float]] = []
        for entry in all_entries:
            score = _compute_relevance(entry, query_terms)
            if score > 0:
                scored.append((entry, score))

        # Sort by score descending
        scored.sort(key=lambda x: x[1], reverse=True)
        return [entry for entry, _ in scored[:limit]]

    # ── INDEX MANAGEMENT ──

    def rebuild_index(self) -> str:
        """Rebuild the MEMORY.md index file.

        The index has constraints:
          - MAX_ENTRYPOINT_LINES = 200 lines max
          - MAX_MEMORY_INDEX_SIZE = 25_000 bytes max

        Format:
            # Memory Index
            Last updated: 2025-04-03T12:34:56

            ## User
            - [id](memories/id.md) — description

            ## Feedback
            - [id](memories/id.md) — description
        """
        entries = self.list_all()

        # Group by category
        by_category: dict[str, list[MemoryEntry]] = {}
        for entry in entries:
            cat = entry.category.value
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append(entry)

        # Build index content
        lines: list[str] = [
            "# Memory Index",
            f"Last updated: {datetime.now().isoformat()}",
            "",
        ]

        # Category order
        category_order = ["user", "feedback", "project", "reference"]

        for cat_name in category_order:
            cat_entries = by_category.get(cat_name, [])
            if not cat_entries:
                continue

            lines.append(f"## {cat_name.title()}")
            for entry in cat_entries:
                desc = entry.description or entry.content[:80]
                if len(desc) > 80:
                    desc = desc[:77] + "..."
                lines.append(f"- [{entry.id}]({entry.id}.md) — {desc}")
            lines.append("")

        content = "\n".join(lines)

        # Enforce size limits
        if len(content) > MAX_MEMORY_INDEX_SIZE:
            content = content[:MAX_MEMORY_INDEX_SIZE - 50] + "\n\n... (index truncated)"

        content_lines = content.splitlines()
        if len(content_lines) > MAX_ENTRYPOINT_LINES:
            content = "\n".join(content_lines[:MAX_ENTRYPOINT_LINES - 2]) + "\n\n... (index truncated)"

        self._index_path.write_text(content, encoding="utf-8")
        return content

    def get_index_content(self) -> str:
        """Read the current MEMORY.md index content."""
        if self._index_path.exists():
            return self._index_path.read_text(encoding="utf-8")
        return ""


# ─────────────────────────────────────────────────────────
# RELEVANCE SCORING
# ─────────────────────────────────────────────────────────

def _tokenize(text: str) -> list[str]:
    """Split text into search terms."""
    # Lowercase and split on non-alphanumeric, filter empty
    return [t for t in re.split(r"[^a-z0-9]+", text.lower()) if t]


def _compute_relevance(entry: MemoryEntry, query_terms: set[str]) -> float:
    """Compute relevance score for a memory entry against query terms.

    Scoring:
      - Term match in content: +1 per term
      - Term match in category: +2 per term
      - Term match in tags: +2 per term
      - Term match in description: +1.5 per term
      - Normalized by content length (avoid bias toward long memories)
    """
    content_lower = entry.content.lower()
    content_terms = set(_tokenize(content_lower))
    description_lower = (entry.description or "").lower()
    desc_terms = set(_tokenize(description_lower))
    tag_terms = set(t.lower() for t in entry.tags)
    cat_terms = set(_tokenize(entry.category.value))

    score = 0.0

    for term in query_terms:
        # Content match
        if term in content_terms:
            score += 1.0
        # Description match (stronger signal)
        if term in desc_terms:
            score += 1.5
        # Tag match
        if term in tag_terms:
            score += 2.0
        # Category match
        if term in cat_terms:
            score += 2.0
        # Substring match in content (weaker signal)
        if term in content_lower:
            score += 0.5

    # Normalize by content length to avoid bias toward long memories
    if score > 0:
        score /= max(1, len(content_terms) / 20)

    return score


# ─────────────────────────────────────────────────────────
# MEMORY TOOLS
# ─────────────────────────────────────────────────────────

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field, field_validator

VALID_CATEGORIES = {c.value for c in MemoryCategory}


class MemoryWriteInput(BaseModel):
    content: str = Field(description="The information to remember")
    category: str = Field(
        default="user",
        description=f"Memory category: {', '.join(sorted(VALID_CATEGORIES))}",
    )
    description: str = Field(
        default="",
        description="Optional one-line summary for the memory index",
    )
    tags: list[str] = Field(
        default_factory=list,
        description="Optional tags for search relevance",
    )

    @field_validator("category")
    @classmethod
    def validate_category(cls, v: str) -> str:
        if v not in VALID_CATEGORIES:
            raise ValueError(f"Invalid category: '{v}'. Must be one of: {', '.join(sorted(VALID_CATEGORIES))}")
        return v


def memory_write(content: str, category: str = "user", description: str = "", tags: list[str] | None = None) -> str:
    """Store a memory for future recall."""
    from dazi.core._singletons import memory_store

    entry = MemoryEntry(
        content=content,
        category=MemoryCategory(category),
        description=description,
        tags=tags or [],
    )
    path = memory_store.write(entry)
    return f"Memory stored: {entry.id}\nCategory: {category}\nPath: {path}"


memory_write_tool = StructuredTool.from_function(
    func=memory_write,
    name="memory_write",
    description="Store a memory for future recall. Memories persist across conversations. Categories: user (preferences), feedback (guidance), project (knowledge), reference (pointers). Use tags to improve search relevance.",
    args_schema=MemoryWriteInput,
)

memory_write_meta = DaziTool(
    name="memory_write",
    description="Store a memory for future recall.",
    safety=ToolSafety.SAFE,
)


class MemoryReadInput(BaseModel):
    memory_id: str = Field(description="The ID of the memory to read")


def memory_read(memory_id: str) -> str:
    """Read a specific memory by ID."""
    from dazi.core._singletons import memory_store

    entry = memory_store.read(memory_id)
    if entry is None:
        return f"Memory not found: {memory_id}"
    return f"ID: {entry.id}\nCategory: {entry.category.value}\nCreated: {entry.created_at}\nTags: {entry.tags}\n\n{entry.content}"


memory_read_tool = StructuredTool.from_function(
    func=memory_read,
    name="memory_read",
    description="Read a specific memory by its ID.",
    args_schema=MemoryReadInput,
)

memory_read_meta = DaziTool(
    name="memory_read",
    description="Read a memory by ID.",
    safety=ToolSafety.SAFE,
)


class MemorySearchInput(BaseModel):
    query: str = Field(description="Search query to find relevant memories")
    limit: int = Field(default=5, description="Maximum number of results")


def memory_search(query: str, limit: int = 5) -> str:
    """Search memories by relevance to a query."""
    from dazi.core._singletons import memory_store

    results = memory_store.find_relevant(query, limit=limit)
    if not results:
        return "No relevant memories found."
    lines = []
    for entry in results:
        desc = entry.description or entry.content[:80]
        lines.append(f"[{entry.id}] ({entry.category.value}) {desc}")
    return f"Found {len(results)} memories:\n" + "\n".join(lines)


memory_search_tool = StructuredTool.from_function(
    func=memory_search,
    name="memory_search",
    description="Search memories by relevance to a query. Returns the most relevant memories.",
    args_schema=MemorySearchInput,
)

memory_search_meta = DaziTool(
    name="memory_search",
    description="Search memories by relevance.",
    safety=ToolSafety.SAFE,
)
