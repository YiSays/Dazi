"""Low-level terminal manipulation helpers."""

from __future__ import annotations

import sys

from prompt_toolkit.formatted_text import FormattedText, to_plain_text
from rich.cells import cell_len


def display_width(text: str) -> int:
    """Return the terminal display width of *text*, handling wide Unicode."""
    return cell_len(text)


def count_prompt_lines(
    segments: list[tuple[str, str]],
    user_input: str,
    term_width: int,
) -> int:
    """Return how many terminal lines the prompt_toolkit prompt occupied.

    The prompt consists of:
      - Status bar: all segment text before the '\\n' separator
      - Input line(s): the prompt symbol ('❯ ') + user_input
    """  # noqa: RUF002
    ft = FormattedText(segments)
    plain = to_plain_text(ft)

    # Split at the newline that separates status bar from input line
    parts = plain.split("\n", 1)
    status_text = parts[0] if parts else ""

    # Status bar lines
    status_width = display_width(status_text)
    status_lines = max(1, -(-status_width // term_width)) if status_width else 1

    # Input lines: prompt symbol + user input (may contain newlines)
    prompt_symbol = "❯ "  # noqa: RUF001
    input_lines = 0
    for i, line in enumerate(user_input.split("\n")):
        line_text = (prompt_symbol + line) if i == 0 else line
        line_width = display_width(line_text)
        input_lines += max(1, -(-line_width // term_width)) if line_width else 1
    if input_lines == 0:
        input_lines = 1

    return status_lines + input_lines


def clear_lines(n: int) -> None:
    """Move cursor up *n* lines and clear each line using ANSI escapes.

    After this call the cursor is at column 0 of the topmost cleared line,
    ready for Rich to render a replacement.
    """
    if n <= 0 or not sys.stdout.isatty():
        return
    parts = ["\x1b[1G"]  # cursor to column 1
    for _ in range(n):
        parts.append("\x1b[1A")  # cursor up one line
        parts.append("\x1b[2K")  # erase entire line
    parts.append("\x1b[1G")  # ensure cursor at column 1
    sys.stdout.write("".join(parts))
    sys.stdout.flush()
