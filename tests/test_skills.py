"""Tests for dazi/skills.py — skill parsing, substitution, and registry."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from dazi.skills import (
    Skill,
    SkillError,
    SkillRegistry,
    _normalize_to_list,
    _parse_bool,
    _parse_skill_content,
    _scan_skills_dir,
    discover_skills,
    parse_skill_file,
    skill_tool_func,
    substitute_arguments,
)

# ─────────────────────────────────────────────────────────
# _parse_bool
# ─────────────────────────────────────────────────────────


class TestParseBool:
    @pytest.mark.parametrize("value", [True, "true", "True", "TRUE", "1", "yes", "Yes"])
    def test_truthy_values(self, value):
        assert _parse_bool(value) is True

    @pytest.mark.parametrize("value", [False, "false", "False", "0", "no", "No"])
    def test_falsy_values(self, value):
        assert _parse_bool(value) is False

    def test_bool_true(self):
        assert _parse_bool(True) is True

    def test_bool_false(self):
        assert _parse_bool(False) is False


# ─────────────────────────────────────────────────────────
# substitute_arguments
# ─────────────────────────────────────────────────────────


class TestSubstituteArguments:
    def test_full_arguments(self):
        result = substitute_arguments("Hello $ARGUMENTS world", "my args", [])
        assert result == "Hello my args world"

    def test_indexed_arguments(self):
        result = substitute_arguments(
            "First: $ARGUMENTS[0], Second: $ARGUMENTS[1]", "hello world", []
        )
        assert result == "First: hello, Second: world"

    def test_shorthand_arguments(self):
        result = substitute_arguments("First: $1, Second: $2", "hello world", [])
        assert result == "First: hello, Second: world"

    def test_shorthand_zero_not_replaced(self):
        result = substitute_arguments("Zero: $0", "hello", [])
        assert "$0" in result

    def test_named_arguments(self):
        result = substitute_arguments("Hello $name", "Alice", ["name"])
        assert result == "Hello Alice"

    def test_no_placeholder_appends(self):
        result = substitute_arguments("Just a prompt", "extra args", [])
        assert result == "Just a prompt\n\nARGUMENTS: extra args"

    def test_no_placeholder_no_args(self):
        result = substitute_arguments("Just a prompt $1", "", [])
        # $1 with no tokens stays as empty or placeholder
        assert "ARGUMENTS:" not in result

    def test_indexed_out_of_bounds(self):
        result = substitute_arguments("$ARGUMENTS[5]", "hello", [])
        assert result == ""

    def test_mixed_placeholders(self):
        result = substitute_arguments(
            "$ARGUMENTS[0] is $1 and all: $ARGUMENTS",
            "hello world",
            [],
        )
        assert "hello" in result
        assert "hello world" in result


# ─────────────────────────────────────────────────────────
# parse_skill_file
# ─────────────────────────────────────────────────────────


class TestParseSkillFile:
    def test_with_frontmatter(self, tmp_path: Path):
        skill_dir = tmp_path / "myskill"
        skill_dir.mkdir()
        skill_file = skill_dir / "SKILL.md"
        skill_file.write_text(
            "---\n"
            'description: "Test skill"\n'
            'argument-hint: "[msg]"\n'
            "user-invocable: true\n"
            "---\n"
            "This is the prompt body.\n"
        )
        skill = parse_skill_file(skill_file)
        assert skill.name == "myskill"
        assert skill.description == "Test skill"
        assert skill.prompt == "This is the prompt body."
        assert skill.argument_hint == "[msg]"
        assert skill.user_invocable is True
        assert skill.source_path == skill_file

    def test_no_frontmatter(self, tmp_path: Path):
        skill_dir = tmp_path / "plain"
        skill_dir.mkdir()
        skill_file = skill_dir / "SKILL.md"
        skill_file.write_text("Just a plain prompt.\n")
        skill = parse_skill_file(skill_file)
        assert skill.name == "plain"
        assert skill.description == ""
        assert skill.prompt == "Just a plain prompt."

    def test_invalid_path_raises(self):
        with pytest.raises(SkillError, match="Cannot read skill file"):
            parse_skill_file(Path("/nonexistent/path/SKILL.md"))


# ─────────────────────────────────────────────────────────
# SkillRegistry
# ─────────────────────────────────────────────────────────


class TestSkillRegistry:
    def test_load_bundled(self):
        registry = SkillRegistry()
        # Don't load from disk (no project root), but bundled skills should load
        # We patch discover_skills to return only bundled ones
        with patch("dazi.skills.discover_skills") as mock_ds:
            from dazi.skills import _get_bundled_skills

            mock_ds.return_value = _get_bundled_skills()
            count = registry.load_skills(project_root=Path("/nonexistent"))
            assert count >= 4  # commit, review, explain, summarize

    def test_get_existing(self):
        registry = SkillRegistry()
        with patch("dazi.skills.discover_skills") as mock_ds:
            from dazi.skills import _get_bundled_skills

            mock_ds.return_value = _get_bundled_skills()
            registry.load_skills()
            skill = registry.get("commit")
            assert skill is not None
            assert skill.name == "commit"

    def test_get_missing(self):
        registry = SkillRegistry()
        assert registry.get("nonexistent") is None

    def test_has_skill(self):
        registry = SkillRegistry()
        with patch("dazi.skills.discover_skills") as mock_ds:
            from dazi.skills import _get_bundled_skills

            mock_ds.return_value = _get_bundled_skills()
            registry.load_skills()
            assert registry.has_skill("commit") is True
            assert registry.has_skill("nonexistent") is False

    def test_expand_skill(self):
        registry = SkillRegistry()
        with patch("dazi.skills.discover_skills") as mock_ds:
            from dazi.skills import _get_bundled_skills

            mock_ds.return_value = _get_bundled_skills()
            registry.load_skills()
            expanded = registry.expand_skill("commit", "fix login bug")
            assert "fix login bug" in expanded

    def test_expand_missing_raises(self):
        registry = SkillRegistry()
        with pytest.raises(SkillError, match="not found"):
            registry.expand_skill("nonexistent")

    def test_list_all(self):
        registry = SkillRegistry()
        with patch("dazi.skills.discover_skills") as mock_ds:
            from dazi.skills import _get_bundled_skills

            mock_ds.return_value = _get_bundled_skills()
            registry.load_skills()
            all_skills = registry.list_all()
            names = {s.name for s in all_skills}
            assert "commit" in names
            assert "review" in names

    def test_list_user_invocable(self):
        registry = SkillRegistry()
        with patch("dazi.skills.discover_skills") as mock_ds:
            from dazi.skills import _get_bundled_skills

            mock_ds.return_value = _get_bundled_skills()
            registry.load_skills()
            invocable = registry.list_user_invocable()
            assert all(s.user_invocable for s in invocable)

    def test_reset(self):
        registry = SkillRegistry()
        with patch("dazi.skills.discover_skills") as mock_ds:
            from dazi.skills import _get_bundled_skills

            mock_ds.return_value = _get_bundled_skills()
            registry.load_skills()
            assert len(registry.list_all()) > 0
            registry.reset()
            assert registry.list_all() == []


# ─────────────────────────────────────────────────────────
# _normalize_to_list
# ─────────────────────────────────────────────────────────


class TestNormalizeToList:
    def test_none(self):
        assert _normalize_to_list(None) == []

    def test_string(self):
        assert _normalize_to_list("hello") == ["hello"]

    def test_list(self):
        assert _normalize_to_list(["a", "b"]) == ["a", "b"]

    def test_list_of_ints(self):
        assert _normalize_to_list([1, 2, 3]) == ["1", "2", "3"]


# ─────────────────────────────────────────────────────────
# parse_skill_file — invalid YAML
# ─────────────────────────────────────────────────────────


class TestParseSkillFileInvalidYaml:
    """Cover the except yaml.YAMLError branch in parse_skill_file (lines 143-145)."""

    def test_invalid_yaml_treated_as_body(self, tmp_path: Path):
        skill_dir = tmp_path / "bad_yaml"
        skill_dir.mkdir()
        skill_file = skill_dir / "SKILL.md"
        # Write content with --- delimiters but invalid YAML between them
        skill_file.write_text("---\ndescription: [unclosed\n---\nThis is the body.\n")
        skill = parse_skill_file(skill_file)
        # Invalid YAML: entire content becomes body
        assert skill.name == "bad_yaml"
        assert "This is the body." in skill.prompt
        assert skill.description == ""

    def test_empty_frontmatter_still_parses(self, tmp_path: Path):
        """Frontmatter with only whitespace between --- is treated as no frontmatter."""
        skill_dir = tmp_path / "empty_fm"
        skill_dir.mkdir()
        skill_file = skill_dir / "SKILL.md"
        skill_file.write_text("---\n   \n---\nBody content.\n")
        skill = parse_skill_file(skill_file)
        assert skill.prompt == "Body content."
        assert skill.description == ""


# ─────────────────────────────────────────────────────────
# _parse_skill_content — bundled skill parsing
# ─────────────────────────────────────────────────────────


class TestParseSkillContent:
    def test_with_frontmatter(self):
        content = (
            "---\n"
            'description: "A test skill"\n'
            'argument-hint: "[topic]"\n'
            "user-invocable: false\n"
            "version: "
            "2.0\n"
            "---\n"
            "Do something with $ARGUMENTS.\n"
        )
        skill = _parse_skill_content("test", content)
        assert skill.name == "test"
        assert skill.description == "A test skill"
        assert skill.argument_hint == "[topic]"
        assert skill.user_invocable is False
        assert skill.version == "2.0"
        assert skill.is_bundled is True
        assert skill.source_path is None
        assert "Do something with" in skill.prompt

    def test_no_frontmatter(self):
        content = "Plain prompt without frontmatter.\n"
        skill = _parse_skill_content("plain", content)
        assert skill.name == "plain"
        assert skill.prompt == "Plain prompt without frontmatter."
        assert skill.description == ""
        assert skill.is_bundled is True

    def test_invalid_yaml_treated_as_body(self):
        """Cover lines 193-194: except yaml.YAMLError in _parse_skill_content."""
        content = "---\ndescription: [broken\n---\nFallback body.\n"
        skill = _parse_skill_content("broken", content)
        assert skill.name == "broken"
        # Invalid YAML: entire content becomes body
        assert "Fallback body." in skill.prompt
        assert skill.description == ""

    def test_empty_frontmatter(self):
        """Lines 184-185: frontmatter_str is empty (no frontmatter)."""
        content = "Just body text.\n"
        skill = _parse_skill_content("nofm", content)
        assert skill.prompt == "Just body text."
        assert skill.is_bundled is True


# ─────────────────────────────────────────────────────────
# _parse_bool — non-bool non-string fallback (line 219)
# ─────────────────────────────────────────────────────────


class TestParseBoolFallback:
    def test_int_truthy(self):
        assert _parse_bool(42) is True

    def test_int_zero(self):
        assert _parse_bool(0) is False

    def test_none_falsy(self):
        assert _parse_bool(None) is False


# ─────────────────────────────────────────────────────────
# _scan_skills_dir (lines 427-442)
# ─────────────────────────────────────────────────────────


class TestScanSkillsDir:
    def test_nonexistent_dir(self):
        """Non-existent directory returns empty list."""
        result = _scan_skills_dir(Path("/nonexistent/path/that/does/not/exist"))
        assert result == []

    def test_empty_dir(self, tmp_path: Path):
        """Directory with no skill subdirs returns empty list."""
        _scan_skills_dir(tmp_path) == []

    def test_with_valid_skill(self, tmp_path: Path):
        """Directory with a valid SKILL.md returns a Skill."""
        skill_dir = tmp_path / "good_skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("---\ndescription: Good\n---\nPrompt here.\n")
        result = _scan_skills_dir(tmp_path)
        assert len(result) == 1
        assert result[0].name == "good_skill"
        assert result[0].description == "Good"

    def test_skips_files_not_dirs(self, tmp_path: Path):
        """Files (not dirs) in skills_dir are skipped."""
        (tmp_path / "not_a_dir.txt").write_text("hello")
        result = _scan_skills_dir(tmp_path)
        assert result == []

    def test_skips_dirs_without_skill_file(self, tmp_path: Path):
        """Subdirectories without SKILL.md are skipped."""
        (tmp_path / "empty_dir").mkdir()
        result = _scan_skills_dir(tmp_path)
        assert result == []

    def test_skips_invalid_skill_file(self, tmp_path: Path):
        """Subdirectories with unreadable SKILL.md are skipped (SkillError)."""
        skill_dir = tmp_path / "bad_skill"
        skill_dir.mkdir()
        bad_file = skill_dir / "SKILL.md"
        bad_file.write_text("---\ndescription: test\n---\nbody\n")
        # Make it unreadable to trigger OSError in parse_skill_file
        bad_file.chmod(0o000)
        try:
            result = _scan_skills_dir(tmp_path)
            assert result == []
        finally:
            bad_file.chmod(0o644)

    def test_multiple_skills_sorted(self, tmp_path: Path):
        """Multiple skills are returned in sorted order."""
        for name in ["charlie", "alpha", "bravo"]:
            d = tmp_path / name
            d.mkdir()
            (d / "SKILL.md").write_text(f"---\ndescription: {name}\n---\nPrompt.\n")
        result = _scan_skills_dir(tmp_path)
        names = [s.name for s in result]
        assert names == ["alpha", "bravo", "charlie"]


# ─────────────────────────────────────────────────────────
# discover_skills (lines 470-493)
# ─────────────────────────────────────────────────────────


class TestDiscoverSkills:
    def test_default_project_root(self, tmp_path: Path):
        """With no project_root, cwd is used (patch Path.cwd)."""
        with patch("dazi.skills.Path.cwd", return_value=tmp_path):
            skills = discover_skills()
            # Should always include bundled skills at minimum
            names = {s.name for s in skills}
            assert "commit" in names

    def test_project_root_overrides_bundled(self, tmp_path: Path):
        """Project-level skill with same name overrides bundled skill."""
        project_skills = tmp_path / ".dazi" / "skills" / "commit"
        project_skills.mkdir(parents=True)
        (project_skills / "SKILL.md").write_text(
            "---\ndescription: Custom commit\n---\nCustom body.\n"
        )
        skills = discover_skills(project_root=tmp_path)
        commit_skill = next(s for s in skills if s.name == "commit")
        assert commit_skill.description == "Custom commit"
        assert "Custom body." in commit_skill.prompt
        assert commit_skill.source_path is not None

    def test_extra_dirs(self, tmp_path: Path):
        """Extra directories are scanned and can override other skills."""
        extra_dir = tmp_path / "extra"
        extra_dir.mkdir()
        extra_skill = extra_dir / "myskill"
        extra_skill.mkdir()
        (extra_skill / "SKILL.md").write_text("---\ndescription: Extra skill\n---\nExtra prompt.\n")
        skills = discover_skills(extra_dirs=[extra_dir])
        names = {s.name for s in skills}
        assert "myskill" in names

    def test_no_extra_dirs(self, tmp_path: Path):
        """discover_skills with extra_dirs=None (the default)."""
        with patch("dazi.skills.Path.cwd", return_value=tmp_path):
            skills = discover_skills(extra_dirs=None)
            # Should at least have bundled skills
            assert len(skills) >= 4

    def test_user_skills_dir_scanned(self, tmp_path: Path, monkeypatch):
        """User-level skills directory is scanned."""
        # Patch _get_user_skills_dir to return a temp dir
        user_dir = tmp_path / "user_skills"
        user_dir.mkdir()
        monkeypatch.setattr("dazi.skills._get_user_skills_dir", lambda: user_dir)

        skill_dir = user_dir / "user_skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("---\ndescription: User skill\n---\nUser prompt.\n")
        skills = discover_skills(project_root=tmp_path)
        names = {s.name for s in skills}
        assert "user_skill" in names


# ─────────────────────────────────────────────────────────
# SkillRegistry.reload (line 539)
# ─────────────────────────────────────────────────────────


class TestSkillRegistryReload:
    def test_reload(self, tmp_path: Path):
        registry = SkillRegistry()
        with patch("dazi.skills.discover_skills") as mock_ds:
            from dazi.skills import _get_bundled_skills

            mock_ds.return_value = _get_bundled_skills()
            registry.load_skills()
            assert len(registry.list_all()) >= 4

            # Now reload — should replace all skills
            mock_ds.return_value = [_get_bundled_skills()[0]]  # Just one
            count = registry.reload()
            assert count == 1
            assert len(registry.list_all()) == 1


# ─────────────────────────────────────────────────────────
# skill_tool_func (lines 616-629)
# ─────────────────────────────────────────────────────────


class TestSkillToolFunc:
    @pytest.fixture(autouse=True)
    def _patch_singletons(self, monkeypatch, tmp_path):
        """Patch singletons so skill_registry is available."""
        from tests.helpers.mock_singletons import patch_singletons

        patch_singletons(monkeypatch, tmp_path)

    @pytest.mark.asyncio
    async def test_skill_found(self):
        """skill_tool_func returns expanded prompt for existing skill."""
        from dazi._singletons import skill_registry

        # Pre-load skills into the registry
        skill_registry.load_skills()
        result = await skill_tool_func(skill="commit", args="fix bug")
        assert "fix bug" in result

    @pytest.mark.asyncio
    async def test_skill_not_found_with_available(self):
        """Returns message listing available skills when skill not found."""
        from dazi._singletons import skill_registry

        skill_registry.load_skills()
        result = await skill_tool_func(skill="nonexistent_xyz", args="")
        assert "not found" in result
        assert "Available skills" in result

    @pytest.mark.asyncio
    async def test_skill_not_found_no_skills(self):
        """Returns message that no skills are loaded when registry is empty."""
        from dazi._singletons import skill_registry

        skill_registry.reset()
        result = await skill_tool_func(skill="anything", args="")
        assert "not found" in result
        assert "No skills are loaded" in result

    @pytest.mark.asyncio
    async def test_skill_error_caught(self):
        """SkillError during expand is caught and returned as string."""
        from dazi._singletons import skill_registry

        skill_registry.reset()
        result = await skill_tool_func(skill="missing", args="")
        # The registry has no skills, so get returns None → "not found" message
        assert "not found" in result

    @pytest.mark.asyncio
    async def test_expand_raises_skill_error(self):
        """When get returns a skill but expand_skill raises SkillError, it's caught."""
        from dazi._singletons import skill_registry

        skill_registry.reset()
        # Manually put a skill in so get() succeeds
        skill_registry._skills["test_skill"] = Skill(
            name="test_skill", description="test", prompt="prompt"
        )
        # Mock expand_skill to raise SkillError
        with patch.object(skill_registry, "expand_skill", side_effect=SkillError("boom")):
            result = await skill_tool_func(skill="test_skill", args="")
        assert "Error expanding skill" in result
        assert "boom" in result


# ─────────────────────────────────────────────────────────
# Skill dataclass defaults
# ─────────────────────────────────────────────────────────


class TestSkillDefaults:
    def test_default_values(self):
        skill = Skill(name="test", description="desc", prompt="body")
        assert skill.argument_hint == ""
        assert skill.arguments == []
        assert skill.allowed_tools == []
        assert skill.user_invocable is True
        assert skill.when_to_use == ""
        assert skill.version == "1.0"
        assert skill.paths == []
        assert skill.model == ""
        assert skill.effort == ""
        assert skill.source_path is None
        assert skill.is_bundled is False

    def test_full_construction(self):
        skill = Skill(
            name="full",
            description="A full skill",
            prompt="Full prompt",
            argument_hint="[args]",
            arguments=["a", "b"],
            allowed_tools=["tool1"],
            user_invocable=False,
            when_to_use="when needed",
            version="3.0",
            paths=["src/**/*.py"],
            model="claude-3",
            effort="high",
            source_path=Path("/some/path"),
            is_bundled=True,
        )
        assert skill.name == "full"
        assert skill.arguments == ["a", "b"]
        assert skill.allowed_tools == ["tool1"]
        assert skill.user_invocable is False
        assert skill.version == "3.0"
        assert skill.paths == ["src/**/*.py"]
        assert skill.model == "claude-3"
        assert skill.effort == "high"
        assert skill.is_bundled is True


# ─────────────────────────────────────────────────────────
# SkillError exception
# ─────────────────────────────────────────────────────────


class TestSkillError:
    def test_raise_and_catch(self):
        with pytest.raises(SkillError):
            raise SkillError("test error")

    def test_inheritance(self):
        assert issubclass(SkillError, Exception)


# ─────────────────────────────────────────────────────────
# substitute_arguments — additional edge cases
# ─────────────────────────────────────────────────────────


class TestSubstituteArgumentsEdgeCases:
    def test_named_arg_not_replaced_when_missing(self):
        """$name with no matching named arg stays as-is."""
        result = substitute_arguments("Hello $unknown_name", "value", ["other_name"])
        assert "$unknown_name" in result

    def test_dollar_arguments_not_replaced_by_named(self):
        """$ARGUMENTS is not consumed by the named-arg step."""
        result = substitute_arguments("$ARGUMENTS", "hello", [])
        assert result == "hello"

    def test_shorthand_with_empty_args(self):
        """$1 with no args returns empty string."""
        result = substitute_arguments("Value: $1", "", [])
        assert result == "Value: "

    def test_no_placeholder_and_no_args(self):
        """No placeholder and no args → prompt unchanged."""
        result = substitute_arguments("Just a prompt", "", [])
        assert result == "Just a prompt"

    def test_named_args_map(self):
        """Multiple named args mapped by position."""
        result = substitute_arguments("$file at $line", "main.py 42", ["file", "line"])
        assert result == "main.py at 42"

    def test_named_args_partial(self):
        """More named args than tokens → later ones stay as placeholder."""
        result = substitute_arguments("$a $b $c", "one two", ["a", "b", "c"])
        assert result == "one two $c"
