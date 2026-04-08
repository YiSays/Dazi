"""Basic filesystem and utility tools.

Tools for reading/writing files, executing shell commands,
evaluating expressions, and writing plan files.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from textwrap import dedent

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field, field_validator

from dazi.config import DATA_DIR
from dazi.base import DaziTool, ToolSafety


# ─────────────────────────────────────────────────────────
# FILE READER — safe, always available
# ─────────────────────────────────────────────────────────


class FileReaderInput(BaseModel):
    file_path: str = Field(description="Absolute path to the file to read. Always read a file before editing it.")
    offset: int = Field(
        default=0, description="Line number to start reading from (0-indexed)"
    )
    limit: int = Field(default=2000, description="Maximum number of lines to read")

    @field_validator("file_path")
    @classmethod
    def validate_absolute_path(cls, v: str) -> str:
        if not v.startswith("/"):
            raise ValueError("file_path must be an absolute path")
        return v


def file_reader(file_path: str, offset: int = 0, limit: int = 2000) -> str:
    """Read a file with line numbers."""
    path = Path(file_path)
    if not path.exists():
        return f"Error: File not found: {file_path}"
    if not path.is_file():
        return f"Error: Not a file: {file_path}"
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        selected = lines[offset : offset + limit]
        if offset > 0 or len(selected) < len(lines):
            header = f"Lines {offset + 1}-{offset + len(selected)} of {len(lines)}\n"
        else:
            header = f"Total lines: {len(lines)}\n"
        numbered = "\n".join(
            f"{offset + i + 1:>6}\t{line}" for i, line in enumerate(selected)
        )
        return header + numbered
    except Exception as e:
        return f"Error reading file: {e}"


file_reader_tool = StructuredTool.from_function(
    func=file_reader,
    name="file_reader",
    description="Read a file from disk. Returns content with line numbers. ALWAYS read a file before editing it — never edit blindly.",
    args_schema=FileReaderInput,
)

file_reader_meta = DaziTool(
    name="file_reader",
    description="Read a file from disk.",
    safety=ToolSafety.SAFE,
)


# ─────────────────────────────────────────────────────────
# SHELL EXEC
# ─────────────────────────────────────────────────────────


class ShellExecInput(BaseModel):
    command: str = Field(description="The shell command to execute")
    timeout: int = Field(default=30, description="Timeout in seconds (max 120)")

    @field_validator("timeout")
    @classmethod
    def validate_timeout(cls, v: int) -> int:
        if v > 120:
            raise ValueError("Timeout cannot exceed 120 seconds")
        return v


def shell_exec(command: str, timeout: int = 30) -> str:
    """Execute a shell command."""
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        output_parts = []
        if result.stdout.strip():
            output_parts.append(result.stdout.strip())
        if result.stderr.strip():
            output_parts.append(f"[stderr]\n{result.stderr.strip()}")
        if result.returncode != 0:
            output_parts.append(f"[exit code: {result.returncode}]")
        return "\n".join(output_parts) if output_parts else "(no output)"
    except subprocess.TimeoutExpired:
        return f"Error: Command timed out after {timeout}s"
    except Exception as e:
        return f"Error: {e}"


shell_exec_tool = StructuredTool.from_function(
    func=shell_exec,
    name="shell_exec",
    description=dedent(
        """\
        Execute a shell command and return its output.
        Use for running CLI tools, listing files, running tests, etc.
        For partial file edits, prefer: sed -i '' 's/old/new/' file
        WARNING: In plan mode, only use for read-only exploration commands."""
    ).strip(),
    args_schema=ShellExecInput,
)

shell_exec_meta = DaziTool(
    name="shell_exec",
    description="Execute a shell command.",
    safety=ToolSafety.DESTRUCTIVE,
)


# ─────────────────────────────────────────────────────────
# PLAN WRITER — plan file only
# ─────────────────────────────────────────────────────────

PLAN_DIR = DATA_DIR / "plans"
PLAN_FILE = PLAN_DIR / "plan.md"


class PlanWriterInput(BaseModel):
    content: str = Field(description="The plan content to write (markdown format)")


def plan_writer(content: str) -> str:
    """Write or update the plan file."""
    try:
        PLAN_DIR.mkdir(parents=True, exist_ok=True)
        PLAN_FILE.write_text(content, encoding="utf-8")
        return f"Plan written to {PLAN_FILE} ({len(content)} characters)"
    except Exception as e:
        return f"Error writing plan: {e}"


plan_writer_tool = StructuredTool.from_function(
    func=plan_writer,
    name="plan_writer",
    description="Write or update the plan file. Can ONLY write to the designated plan file — no other files.",
    args_schema=PlanWriterInput,
)

plan_writer_meta = DaziTool(
    name="plan_writer",
    description="Write to the plan file only.",
    safety=ToolSafety.SAFE,
)


# ─────────────────────────────────────────────────────────
# CALCULATOR — safe, always available
# ─────────────────────────────────────────────────────────


class CalculatorInput(BaseModel):
    expression: str = Field(
        description="A mathematical expression to evaluate. Examples: '2 + 3 * 4'",
    )


def calculator(expression: str) -> str:
    """Evaluate a mathematical expression safely."""
    allowed_names = {
        "abs": abs,
        "round": round,
        "min": min,
        "max": max,
        "sum": sum,
        "pow": pow,
        "len": len,
        "int": int,
        "float": float,
    }
    try:
        result = eval(expression, {"__builtins__": {}}, allowed_names)  # noqa: S307
        return str(result)
    except Exception as e:
        return f"Error: {e}"


calculator_tool = StructuredTool.from_function(
    func=calculator,
    name="calculator",
    description="Evaluate a mathematical expression. Safe arithmetic only.",
    args_schema=CalculatorInput,
)

calculator_meta = DaziTool(
    name="calculator",
    description="Evaluate a mathematical expression.",
    safety=ToolSafety.SAFE,
)


# ─────────────────────────────────────────────────────────
# FILE WRITER — write, execute mode only
# ─────────────────────────────────────────────────────────


class FileWriterInput(BaseModel):
    file_path: str = Field(description="Absolute path to the file to write")
    content: str = Field(description="The COMPLETE file content to write. This replaces the entire file — include ALL lines, not just changed ones.")

    @field_validator("file_path")
    @classmethod
    def validate_absolute_path(cls, v: str) -> str:
        if not v.startswith("/"):
            raise ValueError("file_path must be an absolute path")
        return v


def file_writer(file_path: str, content: str) -> str:
    """Write content to a file."""
    # WARNING: This OVERWRITES the entire file. Do NOT use for partial edits -
    # use shell_exec with sed/awk instead. Only use for creating new files or
    # completely replacing files when you have the FULL content.
    path = Path(file_path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        tmp_path.write_text(content, encoding="utf-8")
        tmp_path.rename(path)
        return f"Successfully wrote {len(content)} characters to {file_path}"
    except Exception as e:
        return f"Error writing file: {e}"


file_writer_tool = StructuredTool.from_function(
    func=file_writer,
    name="file_writer",
    description=dedent(
        """\
        CRITICAL: This tool OVERWRITES the entire file with the provided content.
        Any existing content not included in your `content` parameter will be permanently lost.

        WHEN TO USE:
        - Creating a brand new file
        - Completely replacing a file when you have the FULL updated content

        WHEN NOT TO USE:
        - Changing only a few lines → use shell_exec with sed instead
        - Fixing a single function → use shell_exec with sed instead
        - Appending to a file → use shell_exec with 'echo >> file' instead

        If you need to modify part of an existing file: read it first with file_reader,
        construct the full updated content in your response, then write it.
        Never provide only the changed portion."""
    ).strip(),
    args_schema=FileWriterInput,
)

file_writer_meta = DaziTool(
    name="file_writer",
    description="Create new files or fully replace existing files. OVERWRITES entirely — not for partial edits.",
    safety=ToolSafety.WRITE,
)
