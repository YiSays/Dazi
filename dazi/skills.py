"""Shared Skills Module — Skill registry, frontmatter parsing, argument substitution.

KEY CONCEPTS:
- YAML frontmatter parsing from markdown files
- Skill discovery from directory scanning (user + project level)
- Argument substitution ($ARGUMENTS, $N, named args)
- Skill registry pattern (load, lookup, expand)
- Bundled skills (compiled-in skill definitions)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from dazi.base import DaziTool, ToolSafety

# ─────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────

SKILL_FILENAME = "SKILL.md"
PROJECT_SKILLS_DIR_NAME = ".dazi/skills"


def _get_user_skills_dir() -> Path:
    return Path.home() / ".dazi" / "skills"


# Skill creation paths:
#   - Project-level (preferred for new skills): <project_root>/.dazi/skills/<name>/SKILL.md
#   - User-level (personal skills, manual setup): ~/.dazi/skills/<name>/SKILL.md
# When creating a new skill via LLM, always write to the project-level path.
# User-level skills are personal config that users set up manually.

# Frontmatter delimiter pattern for YAML extraction from markdown
_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n?(.*)", re.DOTALL)


# ─────────────────────────────────────────────────────────
# EXCEPTIONS
# ─────────────────────────────────────────────────────────


class SkillError(Exception):
    """Error during skill loading, parsing, or invocation."""


# ─────────────────────────────────────────────────────────
# DATA MODELS
# ─────────────────────────────────────────────────────────


@dataclass
class Skill:
    """A loaded skill with parsed frontmatter and prompt body.

    Attributes:
        name: Skill identifier (from directory name or bundled name).
        description: Short description from frontmatter.
        prompt: The markdown body after frontmatter (the actual prompt).
        argument_hint: Display hint for arguments, e.g. "[message]".
        arguments: Named argument names from frontmatter.
        allowed_tools: Tool allowlist from frontmatter (parsed but not enforced).
        user_invocable: Whether users can invoke via /skill-name.
        when_to_use: Usage guidance string.
        version: Skill version string.
        paths: Conditional activation path patterns (parsed but not enforced).
        model: Model override (parsed but not applied).
        effort: Thinking effort level (parsed but not applied).
        source_path: File path where SKILL.md was loaded from.
        is_bundled: True for built-in skills.
    """

    name: str
    description: str
    prompt: str
    argument_hint: str = ""
    arguments: list[str] = field(default_factory=list)
    allowed_tools: list[str] = field(default_factory=list)
    user_invocable: bool = True
    when_to_use: str = ""
    version: str = "1.0"
    paths: list[str] = field(default_factory=list)
    model: str = ""
    effort: str = ""
    source_path: Path | None = None
    is_bundled: bool = False


# ─────────────────────────────────────────────────────────
# FRONTMATTER PARSING
# ─────────────────────────────────────────────────────────


def _normalize_to_list(value: str | list[str] | None) -> list[str]:
    """Normalize a frontmatter value to list[str]."""
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value]
    return [str(value)]


def parse_skill_file(path: Path) -> Skill:
    """Parse a SKILL.md file into a Skill object.

    Format: YAML frontmatter between --- delimiters, then markdown body.

    Args:
        path: Path to the SKILL.md file.

    Returns:
        A Skill object with parsed frontmatter fields and prompt body.

    Raises:
        SkillError: If file cannot be read.
    """
    try:
        content = path.read_text(encoding="utf-8")
    except OSError as e:
        raise SkillError(f"Cannot read skill file: {path}: {e}") from e

    # Extract frontmatter and body
    match = _FRONTMATTER_RE.match(content)

    if match:
        frontmatter_str = match.group(1)
        body = match.group(2).strip()
    else:
        # No frontmatter — entire content is the body
        frontmatter_str = ""
        body = content.strip()

    # Parse frontmatter YAML
    frontmatter: dict = {}
    if frontmatter_str.strip():
        try:
            parsed = yaml.safe_load(frontmatter_str)
            if isinstance(parsed, dict):
                frontmatter = parsed
        except yaml.YAMLError:
            # Invalid YAML — treat entire content as body
            body = content.strip()

    # Extract skill name from parent directory
    name = path.parent.name

    # Build Skill object from frontmatter fields
    return Skill(
        name=name,
        description=str(frontmatter.get("description", "")),
        prompt=body,
        argument_hint=str(frontmatter.get("argument-hint", "")),
        arguments=_normalize_to_list(frontmatter.get("arguments")),
        allowed_tools=_normalize_to_list(frontmatter.get("allowed-tools")),
        user_invocable=_parse_bool(frontmatter.get("user-invocable", True)),
        when_to_use=str(frontmatter.get("when_to_use", "")),
        version=str(frontmatter.get("version", "1.0")),
        paths=_normalize_to_list(frontmatter.get("paths")),
        model=str(frontmatter.get("model", "")),
        effort=str(frontmatter.get("effort", "")),
        source_path=path,
    )


def _parse_skill_content(name: str, content: str) -> Skill:
    """Parse skill content from a string (for bundled skills).

    Args:
        name: Skill name.
        content: Full SKILL.md content (frontmatter + body).

    Returns:
        A Skill object with is_bundled=True.
    """
    match = _FRONTMATTER_RE.match(content)

    if match:
        frontmatter_str = match.group(1)
        body = match.group(2).strip()
    else:
        frontmatter_str = ""
        body = content.strip()

    frontmatter: dict = {}
    if frontmatter_str.strip():
        try:
            parsed = yaml.safe_load(frontmatter_str)
            if isinstance(parsed, dict):
                frontmatter = parsed
        except yaml.YAMLError:
            body = content.strip()

    return Skill(
        name=name,
        description=str(frontmatter.get("description", "")),
        prompt=body,
        argument_hint=str(frontmatter.get("argument-hint", "")),
        arguments=_normalize_to_list(frontmatter.get("arguments")),
        allowed_tools=_normalize_to_list(frontmatter.get("allowed-tools")),
        user_invocable=_parse_bool(frontmatter.get("user-invocable", True)),
        when_to_use=str(frontmatter.get("when_to_use", "")),
        version=str(frontmatter.get("version", "1.0")),
        paths=_normalize_to_list(frontmatter.get("paths")),
        model=str(frontmatter.get("model", "")),
        effort=str(frontmatter.get("effort", "")),
        is_bundled=True,
    )


def _parse_bool(value: bool | str) -> bool:
    """Parse a boolean from frontmatter (handles string 'true'/'false')."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in ("true", "1", "yes")
    return bool(value)


# ─────────────────────────────────────────────────────────
# ARGUMENT SUBSTITUTION
# ─────────────────────────────────────────────────────────

_INDEXED_ARG_RE = re.compile(r"\$ARGUMENTS\[(\d+)\]")
_SHORTHAND_ARG_RE = re.compile(r"\$(\d+)")
_NAMED_ARG_RE = re.compile(r"\$([a-zA-Z_][a-zA-Z0-9_]*)")
_FULL_ARG_RE = re.compile(r"\$ARGUMENTS(?!\[)")


def substitute_arguments(prompt: str, args: str, named_args: list[str]) -> str:
    """Substitute argument placeholders in a skill prompt.

    Placeholders (applied in order):
        1. $ARGUMENTS[n] → n-th space-separated token (0-based)
        2. $N → (N-1)-th token (1-based, N >= 1; $0 is not a shorthand)
        3. $name → named arg from frontmatter ``arguments:`` field
        4. $ARGUMENTS → full args string

    If no $ placeholder is found in the original prompt, appends
    ``ARGUMENTS: {args}`` to the prompt.

    Args:
        prompt: The skill prompt template.
        args: The raw argument string from the user/tool call.
        named_args: Named argument names from frontmatter.

    Returns:
        The prompt with placeholders substituted.
    """
    # Split args into tokens for indexed access
    tokens = args.split() if args else []

    # Track whether any substitution happened
    has_placeholder = bool(
        _INDEXED_ARG_RE.search(prompt)
        or _SHORTHAND_ARG_RE.search(prompt)
        or _NAMED_ARG_RE.search(prompt)
        or _FULL_ARG_RE.search(prompt)
    )

    if not has_placeholder and args:
        # No placeholders found — append arguments at end
        return f"{prompt}\n\nARGUMENTS: {args}"

    result = prompt

    # Step 1: Replace $ARGUMENTS[n] → indexed token
    def _replace_indexed(m: re.Match) -> str:
        idx = int(m.group(1))
        return tokens[idx] if idx < len(tokens) else ""

    result = _INDEXED_ARG_RE.sub(_replace_indexed, result)

    # Step 2: Replace $N (1-based shorthand, N >= 1) → token at index N-1
    def _replace_shorthand(m: re.Match) -> str:
        n = int(m.group(1))
        if n < 1:
            return m.group(0)  # $0 is not a valid shorthand
        idx = n - 1
        return tokens[idx] if idx < len(tokens) else ""

    result = _SHORTHAND_ARG_RE.sub(_replace_shorthand, result)

    # Step 3: Replace $name → named argument by position
    named_map = {name: tokens[i] for i, name in enumerate(named_args) if i < len(tokens)}

    def _replace_named(m: re.Match) -> str:
        name = m.group(1)
        # Don't replace $ARGUMENTS (handled in step 4)
        if name == "ARGUMENTS":
            return m.group(0)
        return named_map.get(name, m.group(0))

    result = _NAMED_ARG_RE.sub(_replace_named, result)

    # Step 4: Replace $ARGUMENTS → full args string
    result = _FULL_ARG_RE.sub(args, result)

    return result


# ─────────────────────────────────────────────────────────
# BUNDLED SKILLS
# ─────────────────────────────────────────────────────────

_BUNDLED_COMMIT = """\
---
description: "Generate a conventional commit message for staged changes"
argument-hint: "[optional extra context]"
user-invocable: true
when_to_use: "When the user wants to commit their changes"
version: "1.0"
---

Review the current git diff (staged changes) and generate a conventional commit message.

## Instructions

1. Run `git diff --cached` to see staged changes
2. If no staged changes, run `git status` and suggest `git add`
3. Analyze the changes to determine the commit type (feat, fix, docs, refactor, test, chore)
4. Generate a concise subject line (50 chars max)
5. If there are multiple logical changes, suggest splitting into multiple commits
6. Present the commit message in this format:

```
<type>(<optional scope>): <subject>

<body explaining what and why>
```

$ARGUMENTS"""

_BUNDLED_REVIEW = """\
---
description: "Review code changes for quality, bugs, and improvements"
argument-hint: "[file or directory to review]"
user-invocable: true
when_to_use: "When the user wants a code review"
version: "1.0"
---

Perform a thorough code review. Look for:

1. **Bugs & Logic Errors**: Edge cases, null handling, off-by-one, race conditions
2. **Security Issues**: Injection, hardcoded secrets, missing validation
3. **Performance**: Unnecessary allocations, N+1 queries, missing indexes
4. **Readability**: Naming, complexity, missing comments for complex logic
5. **Best Practices**: DRY violations, proper error handling, resource cleanup

## Format

- Group findings by severity: Critical / Warning / Suggestion
- For each finding, cite the specific file and line
- Suggest concrete fixes

$ARGUMENTS"""

_BUNDLED_EXPLAIN = """\
---
description: "Explain code, architecture, or concepts in detail"
argument-hint: "[what to explain]"
user-invocable: true
when_to_use: "When the user wants to understand code or a concept"
version: "1.0"
---

Provide a clear, detailed explanation.

## Approach

1. Start with a high-level summary (1-2 sentences)
2. Break down the key components or steps
3. Explain the "why" behind design decisions
4. Use analogies if helpful for complex concepts
5. Note any gotchas, common misunderstandings, or important edge cases

$ARGUMENTS"""

_BUNDLED_SUMMARIZE = """\
---
description: "Summarize a file, conversation, or topic concisely"
argument-hint: "[what to summarize]"
user-invocable: true
when_to_use: "When the user wants a quick summary"
version: "1.0"
---

Provide a concise summary.

## Format

- Start with a 1-2 sentence TL;DR
- List the key points (3-7 bullet points)
- Note any action items or decisions needed
- Keep the total length under 200 words unless asked otherwise

$ARGUMENTS"""


def _get_bundled_skills() -> list[Skill]:
    """Create bundled (built-in) skill objects."""
    return [
        _parse_skill_content("commit", _BUNDLED_COMMIT),
        _parse_skill_content("review", _BUNDLED_REVIEW),
        _parse_skill_content("explain", _BUNDLED_EXPLAIN),
        _parse_skill_content("summarize", _BUNDLED_SUMMARIZE),
    ]


# ─────────────────────────────────────────────────────────
# SKILL DISCOVERY
# ─────────────────────────────────────────────────────────


def _scan_skills_dir(skills_dir: Path) -> list[Skill]:
    """Scan a skills directory for SKILL.md files and load them.

    Args:
        skills_dir: Directory containing skill subdirectories.

    Returns:
        List of loaded Skill objects.
    """
    if not skills_dir.is_dir():
        return []

    skills: list[Skill] = []
    for sub_dir in sorted(skills_dir.iterdir()):
        if not sub_dir.is_dir():
            continue
        skill_file = sub_dir / SKILL_FILENAME
        if skill_file.is_file():
            try:
                skill = parse_skill_file(skill_file)
                skills.append(skill)
            except SkillError:
                # Skip invalid skill files
                pass
    return skills


def discover_skills(
    project_root: Path | None = None,
    extra_dirs: list[Path] | None = None,
) -> list[Skill]:
    """Scan skill directories and load all SKILL.md files.

    Search order (later sources override earlier for same name):
        1. Bundled skills (hardcoded in code)
        2. User-level: ~/.dazi/skills/<name>/SKILL.md
        3. Project-level: <project_root>/.dazi/skills/<name>/SKILL.md
        4. Extra directories (if provided)

    Skill creation convention:
        - When creating new skills, write to project-level:
          <project_root>/.dazi/skills/<name>/SKILL.md
        - User-level (~/.dazi/skills/) is for personal skills that
          users set up manually, not for programmatic creation.

    Args:
        project_root: Project root directory. Defaults to cwd.
        extra_dirs: Additional skill directories to scan.

    Returns:
        Deduplicated list of skills (last-loaded wins for name conflicts).
    """
    if project_root is None:
        project_root = Path.cwd()

    # Start with bundled skills
    skill_map: dict[str, Skill] = {}
    for skill in _get_bundled_skills():
        skill_map[skill.name] = skill

    # Scan user-level skills
    for skill in _scan_skills_dir(_get_user_skills_dir()):
        skill_map[skill.name] = skill

    # Scan project-level skills
    project_skills_dir = project_root / PROJECT_SKILLS_DIR_NAME
    for skill in _scan_skills_dir(project_skills_dir):
        skill_map[skill.name] = skill

    # Scan extra directories
    if extra_dirs:
        for extra_dir in extra_dirs:
            for skill in _scan_skills_dir(extra_dir):
                skill_map[skill.name] = skill

    return list(skill_map.values())


# ─────────────────────────────────────────────────────────
# SKILL REGISTRY
# ─────────────────────────────────────────────────────────


class SkillRegistry:
    """Central registry for all loaded skills.

    Manages skill lifecycle: discovery → loading → lookup → expansion.

    Usage::

        registry = SkillRegistry()
        registry.load_skills(project_root=Path("."))
        skill = registry.get("commit")
        expanded = registry.expand_skill("commit", "fix login bug")
    """

    def __init__(self) -> None:
        self._skills: dict[str, Skill] = {}

    def load_skills(self, project_root: Path | None = None) -> int:
        """Discover and load all skills.

        Args:
            project_root: Project root directory for project-level skills.

        Returns:
            Number of skills loaded.
        """
        skills = discover_skills(project_root=project_root)
        self._skills = {s.name: s for s in skills}
        return len(self._skills)

    def reload(self, project_root: Path | None = None) -> int:
        """Reload all skills from disk.

        Args:
            project_root: Project root directory.

        Returns:
            Number of skills loaded.
        """
        return self.load_skills(project_root=project_root)

    def get(self, name: str) -> Skill | None:
        """Look up a skill by name.

        Args:
            name: Skill name (without leading /).

        Returns:
            The Skill object, or None if not found.
        """
        return self._skills.get(name)

    def list_all(self) -> list[Skill]:
        """List all registered skills.

        Returns:
            List of all Skill objects.
        """
        return list(self._skills.values())

    def list_user_invocable(self) -> list[Skill]:
        """List skills that users can invoke via /slash-command.

        Returns:
            List of user-invocable Skill objects.
        """
        return [s for s in self._skills.values() if s.user_invocable]

    def has_skill(self, name: str) -> bool:
        """Check if a skill exists.

        Args:
            name: Skill name.

        Returns:
            True if the skill is registered.
        """
        return name in self._skills

    def expand_skill(self, name: str, args: str = "") -> str:
        """Expand a skill into its full prompt with arguments substituted.

        Args:
            name: Skill name.
            args: Argument string from the user or tool call.

        Returns:
            The expanded prompt string.

        Raises:
            SkillError: If the skill is not found.
        """
        skill = self._skills.get(name)
        if skill is None:
            available = ", ".join(sorted(self._skills.keys()))
            raise SkillError(f"Skill '{name}' not found. Available skills: {available}")

        return substitute_arguments(skill.prompt, args, skill.arguments)

    def reset(self) -> None:
        """Clear all registered skills. For testing."""
        self._skills.clear()


# ─────────────────────────────────────────────────────────
# SKILL TOOL
# ─────────────────────────────────────────────────────────


class SkillToolInput(BaseModel):
    skill: str = Field(description="Name of the skill to invoke (e.g., 'commit', 'review')")
    args: str = Field(default="", description="Arguments to pass to the skill")


async def skill_tool_func(skill: str, args: str = "") -> str:
    """Invoke a skill by name with optional arguments."""
    from dazi._singletons import skill_registry

    s = skill_registry.get(skill)
    if s is None:
        available = ", ".join(sorted(s.name for s in skill_registry.list_all()))
        if available:
            return f"Skill '{skill}' not found. Available skills: {available}"
        return f"Skill '{skill}' not found. No skills are loaded."

    try:
        expanded = skill_registry.expand_skill(skill, args)
        return expanded
    except SkillError as e:
        return f"Error expanding skill '{skill}': {e}"


skill_tool = StructuredTool.from_function(
    func=lambda **kwargs: "",
    coroutine=skill_tool_func,
    name="skill",
    description=(
        "Invoke a named skill to get specialized instructions for a task. "
        "Skills expand into detailed prompts."
    ),
    args_schema=SkillToolInput,
)

skill_tool_meta = DaziTool(
    name="skill",
    description="Invoke a named skill to get specialized instructions.",
    safety=ToolSafety.SAFE,
)
