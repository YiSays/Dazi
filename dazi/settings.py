"""Multi-layer settings system.

KEY CONCEPTS:
  1. 3 setting layers: DEFAULT -> USER -> PROJECT
  2. Merge strategy: primitives override, lists concat+dedupe, dicts deep-merge
  3. Settings affect: permissions, model selection, env vars, features, hooks
  4. Security: project settings excluded from dangerous decisions (mode control)
  5. File format: JSON with $schema reference for IDE autocompletion
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, fields
from enum import StrEnum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────

SETTING_SOURCES = ["default", "user", "project"]

# Settings file names
USER_SETTINGS_FILENAME = "settings.json"
PROJECT_SETTINGS_FILENAME = "settings.json"


# ─────────────────────────────────────────────────────────
# SETTINGS SOURCE — from constants.ts SettingSource
# ─────────────────────────────────────────────────────────


class SettingsSource(StrEnum):
    """Priority order for settings layers (low to high).

    Higher-priority source overrides lower-priority for primitives.
    We simplify to: DEFAULT < USER < PROJECT.
    """

    DEFAULT = "default"
    USER = "user"
    PROJECT = "project"


# ─────────────────────────────────────────────────────────
# SETTINGS DATA MODEL
# ─────────────────────────────────────────────────────────
# 10 fields covering the most impactful settings:
#   - model, api_base_url: LLM configuration
#   - default_mode: initial permission mode
#   - allow_rules, deny_rules: permission rules
#   - env: environment variables
#   - auto_compact, auto_memory: feature toggles
#   - max_concurrent_tools: concurrency limit
#   - mcp_servers: MCP server configurations


@dataclass
class DaziSettings:
    """Merged settings from all layers.

    Fields:
        model: LLM model name. Settings override env var OPENAI_MODEL.
        api_base_url: OpenAI-compatible API base URL. Settings override env var.
        default_mode: Initial permission mode ("default" or "plan").
        allow_rules: Permission allow rules (raw strings, parsed by permissions.py).
        deny_rules: Permission deny rules (raw strings, parsed by permissions.py).
        env: Environment variables to set in subprocess/tool contexts.
        auto_compact: Whether to auto-compact when approaching context limit.
        auto_memory: Whether to auto-extract memories from conversations.
        max_concurrent_tools: Maximum number of tools that can run concurrently.
        mcp_servers: MCP server configurations (Stage 10). Dict of server name → config dict.
    """

    model: str | None = None
    api_base_url: str | None = None
    api_key: str | None = None
    default_mode: str = "default"
    allow_rules: list[str] = field(default_factory=list)
    deny_rules: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    auto_compact: bool = True
    auto_memory: bool = True
    max_concurrent_tools: int = 5
    mcp_servers: dict[str, dict] = field(default_factory=dict)
    context_window: int | None = None
    thinking_enabled: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict for JSON persistence.

        Outputs mcpServers (camelCase) to match the MCP standard.
        """
        result: dict[str, Any] = {}
        for f in fields(self):
            value = getattr(self, f.name)
            # Don't persist defaults that would bloat the file
            if isinstance(value, (list, dict)) and not value:
                continue
            if value is None:
                continue
            result[f.name] = value
        # Use standard MCP key name in JSON output
        if "mcp_servers" in result:
            result["mcpServers"] = result.pop("mcp_servers")
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DaziSettings:
        """Construct from dict, ignoring unknown fields.

        Accepts both mcpServers (standard) and mcp_servers (snake_case).
        """
        mapped = dict(data)
        if "mcpServers" in mapped:
            mapped["mcp_servers"] = mapped.pop("mcpServers")
        valid_fields = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in mapped.items() if k in valid_fields}
        return cls(**filtered)


# ─────────────────────────────────────────────────────────
# MERGE ALGORITHM
# ─────────────────────────────────────────────────────────
# Rules:
#   - Primitives (str, bool, int): later source wins outright
#   - Lists: concatenate and deduplicate (preserves order, first wins)
#   - Dicts: deep merge (later keys override earlier keys)


def _dedupe_preserve_order(items: list[str]) -> list[str]:
    """Remove duplicates preserving first-occurrence order."""
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def merge_settings(base: DaziSettings, override: DaziSettings) -> DaziSettings:
    """Merge two DaziSettings following the merge strategy.

    Called as: merge(DEFAULT, USER), then merge(result, PROJECT)

    Rules:
      - Primitives (str, bool, int): override wins
      - Lists (allow_rules, deny_rules): concat + dedupe (order preserved)
      - Dicts (env): shallow merge, override keys win
    """
    merged = DaziSettings()

    # Primitives: override wins (or base if override is default)
    merged.model = override.model
    # api_base_url: None means "not set", so fall back to base
    merged.api_base_url = (
        override.api_base_url if override.api_base_url is not None else base.api_base_url
    )
    # api_key: None means "not set", so fall back to base (or env var in create_llm)
    merged.api_key = override.api_key if override.api_key is not None else base.api_key
    merged.default_mode = override.default_mode
    merged.auto_compact = override.auto_compact
    merged.auto_memory = override.auto_memory
    merged.max_concurrent_tools = override.max_concurrent_tools
    merged.thinking_enabled = override.thinking_enabled

    # Lists: concatenate + dedupe
    merged.allow_rules = _dedupe_preserve_order(base.allow_rules + override.allow_rules)
    merged.deny_rules = _dedupe_preserve_order(base.deny_rules + override.deny_rules)

    # Dicts: shallow merge (override keys win for same key)
    merged.env = {**base.env, **override.env}
    merged.mcp_servers = {**base.mcp_servers, **override.mcp_servers}

    return merged


# ─────────────────────────────────────────────────────────
# SETTINGS MANAGER — from settings.ts loadAllSettings()
# ─────────────────────────────────────────────────────────


class SettingsManager:
    """Manages layered settings loading, merging, and file I/O.

    File paths:
        User:    ~/.dazi/settings.json     (global across all projects)
        Project: .dazi/settings.json       (per-project)
        Default: hardcoded in DaziSettings defaults

    The manager loads all layers at construction and merges them.
    Call reload() to re-read files (e.g., after /reload command).

    Settings files use JSON format. Unknown fields are ignored.

    Missing files are treated as empty defaults (not an error).
    Invalid JSON files log a warning and are skipped.
    """

    def __init__(self, project_root: Path | None = None, user_dir: Path | None = None) -> None:
        """Initialize settings manager.

        Args:
            project_root: Project root directory. Defaults to cwd.
            user_dir: User settings directory. Defaults to ~/.dazi.
        """
        self._project_root = project_root or Path.cwd()
        self._user_path = (user_dir or Path.home() / ".dazi") / USER_SETTINGS_FILENAME
        self._project_path = self._project_root / ".dazi" / PROJECT_SETTINGS_FILENAME
        self._settings: DaziSettings | None = None
        # Track which source set each field, for /settings display
        self._source_map: dict[str, str] = {}

    @property
    def settings(self) -> DaziSettings:
        """Get the merged effective settings (lazy-loaded)."""
        if self._settings is None:
            self.reload()
        return self._settings

    @property
    def source_map(self) -> dict[str, str]:
        """Get the source map showing which layer set each field."""
        if self._settings is None:
            self.reload()
        return self._source_map

    @property
    def user_path(self) -> Path:
        """Path to user settings file."""
        return self._user_path

    @property
    def project_path(self) -> Path:
        """Path to project settings file."""
        return self._project_path

    def reload(self) -> DaziSettings:
        """Reload settings from all layers and merge.

        Chain: DEFAULT -> merge(USER) -> merge(PROJECT)
        Missing files are skipped (not treated as "all defaults").
        """
        # Load each layer (None if file doesn't exist)
        default_layer = self._load_default()
        user_layer = self._load_layer(self._user_path, SettingsSource.USER)
        project_layer = self._load_layer(self._project_path, SettingsSource.PROJECT)

        # Merge in priority order: DEFAULT < USER < PROJECT
        merged = default_layer
        if user_layer is not None:
            merged = merge_settings(merged, user_layer)
        if project_layer is not None:
            merged = merge_settings(merged, project_layer)

        self._settings = merged

        # Build source map for display
        self._source_map = self._build_source_map(
            default_layer,
            user_layer or DaziSettings(),  # use defaults for source map
            project_layer or DaziSettings(),
            user_layer is not None,
            project_layer is not None,
        )

        return merged

    def _load_default(self) -> DaziSettings:
        """Return hardcoded defaults."""
        return DaziSettings()

    def _load_layer(self, path: Path, source: SettingsSource) -> DaziSettings | None:
        """Load settings from a JSON file.

        Missing files return None (not defaults — so they don't override).
        Invalid JSON logs warning and returns None.
        Unknown fields are silently ignored.
        """
        if not path.exists():
            return None

        try:
            text = path.read_text(encoding="utf-8")
            data = json.loads(text)
            if not isinstance(data, dict):
                logger.warning(f"Settings file {path} is not a JSON object, ignoring")
                return None
            return DaziSettings.from_dict(data)
        except json.JSONDecodeError as e:
            logger.warning(f"Invalid JSON in settings file {path}: {e}")
            return None

    def _build_source_map(
        self,
        default_layer: DaziSettings,
        user_layer: DaziSettings,
        project_layer: DaziSettings,
        user_exists: bool,
        project_exists: bool,
    ) -> dict[str, str]:
        """Build a map showing which source set each effective field.

        Used by /settings REPL command to annotate field values.
        """
        source_map: dict[str, str] = {}

        for f in fields(DaziSettings):
            field_name = f.name

            # Check if the field was explicitly set in a layer's file
            # (i.e., it's not just the DaziSettings default)
            if project_exists:
                raw_project = self._parse_settings_file(self._project_path)
                if raw_project and field_name in raw_project:
                    source_map[field_name] = SettingsSource.PROJECT.value
                    continue
            if user_exists:
                raw_user = self._parse_settings_file(self._user_path)
                if raw_user and field_name in raw_user:
                    source_map[field_name] = SettingsSource.USER.value
                    continue

            source_map[field_name] = SettingsSource.DEFAULT.value

        return source_map

    def get_permission_rules(self) -> list:
        """Parse allow_rules + deny_rules from settings into PermissionRule list.

        Rules use source="settings" which has priority 2 in SOURCE_PRIORITY
        (below "cli" at priority 3 in permissions.py). This means
        CLI rules (via /allow, /deny) always override settings rules.
        """
        from dazi.permissions import parse_rules

        rules = []
        s = self.settings
        # Deny rules first (they take precedence in the priority check)
        rules.extend(parse_rules(s.deny_rules, "settings"))
        # Then allow rules
        rules.extend(parse_rules(s.allow_rules, "settings"))
        return rules

    def get_model_name(self) -> str:
        """Get effective model name.

        Settings > env var OPENAI_MODEL > hardcoded default.
        """
        return self.settings.model

    def get_api_base_url(self) -> str | None:
        """Get effective API base URL.

        Settings > env var OPENAI_BASE_URL > None.
        """
        return self.settings.api_base_url

    def get_api_key(self) -> str | None:
        """Get effective API key.

        Settings > env var OPENAI_API_KEY > None.
        """
        if self.settings.api_key:
            return self.settings.api_key
        from dazi.config import OPENAI_API_KEY

        return OPENAI_API_KEY or None

    def is_thinking_enabled(self) -> bool:
        """Get whether extended thinking (reasoning_content) is enabled."""
        return self.settings.thinking_enabled

    def get_mcp_servers(self) -> dict[str, dict]:
        """Get MCP server configurations from merged settings.

        Returns dict of server_name -> config_dict.
        Each config has: command, args, env, description (type is always "stdio").
        """
        return self.settings.mcp_servers

    def save_user_settings(self, settings: DaziSettings) -> None:
        """Save settings to user-level file.

        Creates ~/.dazi/ directory if needed.
        """
        self._user_path.parent.mkdir(parents=True, exist_ok=True)
        self._user_path.write_text(
            json.dumps(settings.to_dict(), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    def save_project_settings(self, settings: DaziSettings) -> None:
        """Save settings to project-level file.

        Creates .dazi/ directory if needed.
        """
        self._project_path.parent.mkdir(parents=True, exist_ok=True)
        self._project_path.write_text(
            json.dumps(settings.to_dict(), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    @staticmethod
    def _parse_settings_file(path: Path) -> dict[str, Any] | None:
        """Parse a JSON settings file and return raw dict.

        Returns None if file doesn't exist or is invalid.
        """
        if not path.exists():
            return None
        try:
            text = path.read_text(encoding="utf-8")
            data = json.loads(text)
            if isinstance(data, dict):
                return data
        except (json.JSONDecodeError, OSError):
            pass
        return None
