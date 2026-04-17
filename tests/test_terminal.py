"""Tests for dazi/terminal.py — display_width, count_prompt_lines, clear_lines."""

from __future__ import annotations

from unittest.mock import MagicMock, patch


class TestDisplayWidth:
    def test_ascii(self):
        from dazi.terminal import display_width

        assert display_width("hello") == 5

    def test_wide_unicode(self):
        from dazi.terminal import display_width

        # CJK characters are 2 cells wide
        assert display_width("你好") == 4


class TestCountPromptLines:
    def test_single_line_prompt(self):
        from dazi.terminal import count_prompt_lines

        segments = [("fg:green", "EXECUTE · 10%")]
        lines = count_prompt_lines(segments, "hello", term_width=80)
        assert lines == 2  # status bar + input line

    def test_wide_prompt_wraps(self):
        from dazi.terminal import count_prompt_lines

        # A 100-char status bar with 50-char terminal → 2 status lines + 1 input = 3
        long_status = "A" * 100
        segments = [("", long_status)]
        lines = count_prompt_lines(segments, "hi", term_width=50)
        assert lines == 3

    def test_multiline_input(self):
        from dazi.terminal import count_prompt_lines

        segments = [("", "EXEC")]
        # 2 lines of input → 1 status + 2 input
        lines = count_prompt_lines(segments, "line1\nline2", term_width=80)
        assert lines == 3

    def test_narrow_terminal_wraps_input(self):
        from dazi.terminal import count_prompt_lines

        segments = [("", "EX")]
        # 10-char input with 5-char terminal → wraps
        lines = count_prompt_lines(segments, "abcdefghij", term_width=5)
        assert lines >= 2

    def test_empty_input(self):
        from dazi.terminal import count_prompt_lines

        segments = [("", "EXEC")]
        lines = count_prompt_lines(segments, "", term_width=80)
        assert lines == 2  # status + empty input line

    def test_newline_separator(self):
        from dazi.terminal import count_prompt_lines

        # Status bar with explicit newline before input symbol
        segments = [("fg:green", "EXECUTE"), ("", "\n❯ ")]  # noqa: RUF001
        lines = count_prompt_lines(segments, "test", term_width=80)
        assert lines >= 2


class TestClearLines:
    def test_writes_ansi_escapes(self):
        from dazi.terminal import clear_lines

        mock_stdout = MagicMock()
        mock_stdout.isatty.return_value = True
        with patch("sys.stdout", mock_stdout):
            clear_lines(3)
            mock_stdout.write.assert_called_once()
            mock_stdout.flush.assert_called_once()
            written = mock_stdout.write.call_args[0][0]
            assert "\x1b[1A" in written  # cursor up
            assert "\x1b[2K" in written  # erase line

    def test_zero_lines_noop(self):
        from dazi.terminal import clear_lines

        mock_stdout = MagicMock()
        with patch("sys.stdout", mock_stdout):
            clear_lines(0)
            mock_stdout.write.assert_not_called()

    def test_negative_lines_noop(self):
        from dazi.terminal import clear_lines

        mock_stdout = MagicMock()
        with patch("sys.stdout", mock_stdout):
            clear_lines(-1)
            mock_stdout.write.assert_not_called()

    def test_non_tty_noop(self):
        from dazi.terminal import clear_lines

        mock_stdout = MagicMock()
        mock_stdout.isatty.return_value = False
        with patch("sys.stdout", mock_stdout):
            clear_lines(3)
            mock_stdout.write.assert_not_called()
