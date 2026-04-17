"""Shared test fixtures for Dazi test suite."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from dazi.background import BackgroundTaskManager
from dazi.cost_tracker import CostTracker
from dazi.hooks import HookRegistry
from dazi.mailbox import Mailbox
from dazi.memory import MemoryCategory, MemoryEntry, MemoryStore
from dazi.permissions import PermissionBehavior, PermissionRule
from dazi.proactive import ProactiveManager
from dazi.settings import SettingsManager
from dazi.skills import SkillRegistry
from dazi.task_store import Task, TaskStatus, TaskStore
from dazi.team import TeamManager
from dazi.teammate import TeammateRunner

# ─────────────────────────────────────────────────────────
# ISOLATION GUARDS (autouse — run for every test)
# ─────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _isolate_home_dir(tmp_path: Path, monkeypatch):
    """Prevent ALL tests from accessing real Path.home().

    Patches Path.home() to return a fake directory under tmp_path.
    Per-test monkeypatch.setattr calls override this since monkeypatch
    is per-test scoped and later patches take precedence.
    """
    fake_home = tmp_path / "fake_home"
    fake_home.mkdir()
    monkeypatch.setattr("pathlib.Path.home", lambda *args, **kwargs: fake_home)


@pytest.fixture(autouse=True)
def _patch_singletons_guard(monkeypatch, tmp_path: Path):
    """Ensure all tests use isolated singletons as a safety net.

    Per-file autouse fixtures that call patch_singletons will override
    these patches since they run after conftest fixtures.
    """
    from tests.helpers.mock_singletons import patch_singletons

    patch_singletons(monkeypatch, tmp_path)


# ─────────────────────────────────────────────────────────
# DIRECTORY FIXTURES
# ─────────────────────────────────────────────────────────


@pytest.fixture
def tmp_data_dir(tmp_path: Path) -> Path:
    """Create a temporary .dazi-style data directory."""
    d = tmp_path / ".dazi"
    d.mkdir(exist_ok=True)
    return d


# ─────────────────────────────────────────────────────────
# ENVIRONMENT FIXTURES
# ─────────────────────────────────────────────────────────


@pytest.fixture
def mock_env(monkeypatch):
    """Set mock environment variables for LLM config."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-123")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4o")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)


# ─────────────────────────────────────────────────────────
# SINGLETON-STYLE FIXTURES (using tmp_path, not _singletons)
# ─────────────────────────────────────────────────────────


@pytest.fixture
def mock_cost_tracker(tmp_data_dir: Path) -> CostTracker:
    return CostTracker(tmp_data_dir)


@pytest.fixture
def mock_task_store(tmp_path: Path) -> TaskStore:
    return TaskStore(tmp_path / "tasks", list_id="test")


@pytest.fixture
def mock_memory_store(tmp_path: Path) -> MemoryStore:
    return MemoryStore(tmp_path / "memory")


@pytest.fixture
def mock_settings_manager(tmp_path: Path) -> SettingsManager:
    return SettingsManager(project_root=tmp_path, user_dir=tmp_path / "user_home" / ".dazi")


@pytest.fixture
def mock_team_manager(tmp_path: Path) -> TeamManager:
    return TeamManager(base_dir=tmp_path)


@pytest.fixture
def mock_mailbox(tmp_path: Path) -> Mailbox:
    return Mailbox(base_dir=tmp_path)


@pytest.fixture
def mock_hook_registry() -> HookRegistry:
    return HookRegistry()


@pytest.fixture
def mock_skill_registry() -> SkillRegistry:
    return SkillRegistry()


@pytest.fixture
def mock_proactive_manager() -> ProactiveManager:
    return ProactiveManager()


@pytest.fixture
def mock_teammate_runner() -> TeammateRunner:
    return TeammateRunner()


@pytest.fixture
def mock_background_manager(tmp_path: Path) -> BackgroundTaskManager:
    return BackgroundTaskManager(tmp_path / "background")


# ─────────────────────────────────────────────────────────
# SAMPLE DATA FIXTURES
# ─────────────────────────────────────────────────────────


@pytest.fixture
def sample_task() -> Task:
    return Task(
        id=1,
        subject="Test task",
        description="A test task description",
        status=TaskStatus.PENDING,
        created_at=datetime.now(UTC).isoformat(),
    )


@pytest.fixture
def sample_messages() -> list:
    """Basic message list for token counting tests."""
    return [
        HumanMessage(content="Hello"),
        AIMessage(content="Hi there!"),
    ]


@pytest.fixture
def sample_permission_rules() -> list[PermissionRule]:
    """Standard list of permission rules for testing."""
    return [
        PermissionRule(
            behavior=PermissionBehavior.ALLOW,
            tool_name="file_reader",
            source="cli",
        ),
        PermissionRule(
            behavior=PermissionBehavior.DENY,
            tool_name="shell_exec",
            pattern="rm *",
            source="settings",
        ),
        PermissionRule(
            behavior=PermissionBehavior.ASK,
            tool_name="file_writer",
            source="cli",
        ),
    ]


@pytest.fixture
def sample_memory_entry() -> MemoryEntry:
    return MemoryEntry(
        content="User prefers dark mode",
        category=MemoryCategory.USER,
        description="UI preference",
    )
