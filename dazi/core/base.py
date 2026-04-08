"""Shared tool interface for Dazi.

Every tool has a name, description, and safety classification. Tool metadata
is used for mode filtering (plan mode restricts to SAFE tools) and for
permission checks. LangChain StructuredTool instances handle the actual
LLM-facing tool interface; DaziTool instances carry our internal metadata.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ToolSafety(Enum):
    """Tool safety classification.

    SAFE       = read-only, concurrency-safe (e.g. Calculator, FileRead)
    WRITE      = not read-only, not concurrency-safe (e.g. FileWrite)
    DESTRUCTIVE = irreversible operations (e.g. ShellExec, FileDelete)
    """
    SAFE = "safe"
    WRITE = "write"
    DESTRUCTIVE = "destructive"


@dataclass
class DaziTool:
    """Base tool definition with metadata for mode filtering and permissions.

    Defaults are conservative: tools are destructive, not concurrency-safe,
    and not read-only unless explicitly marked otherwise.
    """
    name: str
    description: str  # Short description for the LLM
    safety: ToolSafety = ToolSafety.DESTRUCTIVE  # Conservative default
    enabled: bool = True

    def __post_init__(self):
        # Derive flags from safety level
        self.is_concurrency_safe = self.safety == ToolSafety.SAFE
        self.is_read_only = self.safety in (ToolSafety.SAFE,)
