"""Tests for dazi.team — _sanitize_name, TeamConfig, TeamManager CRUD & member management."""

from __future__ import annotations

import pytest

from dazi.team import TeamConfig, TeamError, TeamManager, TeamMember

# ─────────────────────────────────────────────────────────
# _sanitize_name
# ─────────────────────────────────────────────────────────


class TestSanitizeName:
    def test_basic(self):
        assert TeamManager._sanitize_name("Web Dev") == "web-dev"

    def test_special_chars(self):
        assert TeamManager._sanitize_name("Team@#$%Name!") == "team-name"

    def test_empty_string(self):
        assert TeamManager._sanitize_name("") == "unnamed-team"

    def test_consecutive_hyphens(self):
        assert TeamManager._sanitize_name("A---B") == "a-b"

    def test_leading_trailing_hyphens(self):
        assert TeamManager._sanitize_name("--hello--") == "hello"

    def test_uppercase_to_lower(self):
        assert TeamManager._sanitize_name("MyTeam") == "myteam"

    def test_all_special_chars(self):
        assert TeamManager._sanitize_name("@#$") == "unnamed-team"


# ─────────────────────────────────────────────────────────
# TeamConfig serialization
# ─────────────────────────────────────────────────────────


class TestTeamConfig:
    def test_to_dict(self):
        cfg = TeamConfig(
            name="test-team",
            description="A team",
            members=[TeamMember(name="alice", agent_id="alice@test-team")],
            created_at="2025-01-01T00:00:00",
        )
        d = cfg.to_dict()
        assert d["name"] == "test-team"
        assert d["description"] == "A team"
        assert len(d["members"]) == 1
        assert d["members"][0]["name"] == "alice"

    def test_from_dict(self):
        d = {
            "name": "proj",
            "description": "desc",
            "members": [
                {
                    "name": "bob",
                    "agent_id": "bob@proj",
                    "agent_type": "general-purpose",
                    "status": "active",
                    "joined_at": "2025-01-01",
                }
            ],
            "created_at": "2025-01-01",
        }
        cfg = TeamConfig.from_dict(d)
        assert cfg.name == "proj"
        assert len(cfg.members) == 1
        assert cfg.members[0].name == "bob"

    def test_from_dict_defaults(self):
        d = {"name": "minimal"}
        cfg = TeamConfig.from_dict(d)
        assert cfg.description == ""
        assert cfg.members == []
        assert cfg.created_at == ""

    def test_roundtrip(self):
        cfg = TeamConfig(
            name="rt",
            description="roundtrip",
            members=[TeamMember(name="m1", agent_id="m1@rt")],
            created_at="2025-06-01",
        )
        restored = TeamConfig.from_dict(cfg.to_dict())
        assert restored.name == cfg.name
        assert restored.description == cfg.description
        assert len(restored.members) == len(cfg.members)
        assert restored.created_at == cfg.created_at


# ─────────────────────────────────────────────────────────
# TeamManager — CRUD
# ─────────────────────────────────────────────────────────


class TestTeamManagerCreate:
    def test_creates_config_and_task_dir(self, mock_team_manager):
        cfg = mock_team_manager.create_team("web-dev", "Web development team")
        assert cfg.name == "web-dev"
        assert cfg.description == "Web development team"
        assert mock_team_manager.team_exists("web-dev")

    def test_creates_task_directory(self, mock_team_manager):
        mock_team_manager.create_team("web-dev")
        task_dir = mock_team_manager._task_dir("web-dev")
        assert task_dir.exists()

    def test_duplicate_raises_error(self, mock_team_manager):
        mock_team_manager.create_team("web-dev")
        with pytest.raises(TeamError, match="already exists"):
            mock_team_manager.create_team("web-dev")


class TestTeamManagerGet:
    def test_get_existing(self, mock_team_manager):
        mock_team_manager.create_team("alpha", "Alpha team")
        cfg = mock_team_manager.get_team("alpha")
        assert cfg is not None
        assert cfg.name == "alpha"

    def test_get_nonexistent(self, mock_team_manager):
        assert mock_team_manager.get_team("nope") is None


class TestTeamManagerList:
    def test_list_empty(self, mock_team_manager):
        assert mock_team_manager.list_teams() == []

    def test_list_multiple(self, mock_team_manager):
        mock_team_manager.create_team("team-a")
        mock_team_manager.create_team("team-b")
        teams = mock_team_manager.list_teams()
        names = {t.name for t in teams}
        assert "team-a" in names
        assert "team-b" in names


class TestTeamManagerTeamExists:
    def test_exists(self, mock_team_manager):
        mock_team_manager.create_team("check-me")
        assert mock_team_manager.team_exists("check-me") is True

    def test_not_exists(self, mock_team_manager):
        assert mock_team_manager.team_exists("ghost") is False


# ─────────────────────────────────────────────────────────
# TeamManager — Member management
# ─────────────────────────────────────────────────────────


class TestTeamManagerAddMember:
    def test_add_member(self, mock_team_manager):
        mock_team_manager.create_team("dev")
        member = TeamMember(name="frontend", agent_id="frontend@dev")
        result = mock_team_manager.add_member("dev", member)
        assert result.name == "frontend"
        cfg = mock_team_manager.get_team("dev")
        assert len(cfg.members) == 1

    def test_add_duplicate_raises(self, mock_team_manager):
        mock_team_manager.create_team("dev")
        m = TeamMember(name="fe", agent_id="fe@dev")
        mock_team_manager.add_member("dev", m)
        with pytest.raises(TeamError, match="already in team"):
            mock_team_manager.add_member("dev", m)

    def test_add_member_to_nonexistent_team(self, mock_team_manager):
        m = TeamMember(name="x", agent_id="x@none")
        with pytest.raises(TeamError, match="not found"):
            mock_team_manager.add_member("ghost", m)


class TestTeamManagerRemoveMember:
    def test_remove_member(self, mock_team_manager):
        mock_team_manager.create_team("dev")
        m = TeamMember(name="fe", agent_id="fe@dev")
        mock_team_manager.add_member("dev", m)
        assert mock_team_manager.remove_member("dev", "fe@dev") is True
        cfg = mock_team_manager.get_team("dev")
        assert len(cfg.members) == 0

    def test_remove_nonexistent_member(self, mock_team_manager):
        mock_team_manager.create_team("dev")
        assert mock_team_manager.remove_member("dev", "nobody@dev") is False


class TestTeamManagerGetMember:
    def test_get_existing_member(self, mock_team_manager):
        mock_team_manager.create_team("dev")
        m = TeamMember(name="be", agent_id="be@dev")
        mock_team_manager.add_member("dev", m)
        found = mock_team_manager.get_member("dev", "be@dev")
        assert found is not None
        assert found.name == "be"

    def test_get_member_nonexistent(self, mock_team_manager):
        mock_team_manager.create_team("dev")
        assert mock_team_manager.get_member("dev", "ghost@dev") is None


class TestTeamManagerUpdateMemberStatus:
    def test_update_status(self, mock_team_manager):
        mock_team_manager.create_team("dev")
        m = TeamMember(name="fe", agent_id="fe@dev")
        mock_team_manager.add_member("dev", m)
        assert mock_team_manager.update_member_status("dev", "fe@dev", "completed") is True
        cfg = mock_team_manager.get_team("dev")
        assert cfg.members[0].status == "completed"

    def test_update_status_nonexistent_member(self, mock_team_manager):
        mock_team_manager.create_team("dev")
        assert mock_team_manager.update_member_status("dev", "ghost@dev", "idle") is False


# ─────────────────────────────────────────────────────────
# TeamManager — Delete
# ─────────────────────────────────────────────────────────


class TestTeamManagerDelete:
    def test_delete_with_no_members(self, mock_team_manager):
        mock_team_manager.create_team("temp")
        assert mock_team_manager.delete_team("temp") is True
        assert mock_team_manager.team_exists("temp") is False

    def test_delete_with_active_members_raises(self, mock_team_manager):
        mock_team_manager.create_team("active")
        m = TeamMember(name="worker", agent_id="worker@active", status="active")
        mock_team_manager.add_member("active", m)
        with pytest.raises(TeamError, match="active member"):
            mock_team_manager.delete_team("active")

    def test_delete_with_completed_members_ok(self, mock_team_manager):
        mock_team_manager.create_team("done")
        m = TeamMember(name="worker", agent_id="worker@done", status="completed")
        mock_team_manager.add_member("done", m)
        assert mock_team_manager.delete_team("done") is True

    def test_delete_nonexistent(self, mock_team_manager):
        assert mock_team_manager.delete_team("ghost") is False

    def test_delete_removes_task_dir(self, mock_team_manager):
        mock_team_manager.create_team("gone")
        task_dir = mock_team_manager._task_dir("gone")
        assert task_dir.exists()
        mock_team_manager.delete_team("gone")
        assert not task_dir.exists()


# ─────────────────────────────────────────────────────────
# Standalone tool functions
# ─────────────────────────────────────────────────────────


class TestCreateTeamFunc:
    @pytest.mark.asyncio
    async def test_success(self, monkeypatch, tmp_path):
        from unittest.mock import MagicMock

        import dazi.team as mod

        mock_tm = MagicMock()
        mock_tm.create_team.return_value = mod.TeamConfig(name="t1", description="desc")
        mock_tm._config_path.return_value = tmp_path / "c.json"
        mock_tm._task_dir.return_value = tmp_path / "tasks"
        monkeypatch.setattr("dazi._singletons.team_manager", mock_tm)
        result = await mod.create_team_func(name="t1", description="desc")
        assert "Team created: t1" in result

    @pytest.mark.asyncio
    async def test_error(self, monkeypatch):
        from unittest.mock import MagicMock

        import dazi.team as mod

        mock_tm = MagicMock(create_team=MagicMock(side_effect=mod.TeamError("exists")))
        monkeypatch.setattr("dazi._singletons.team_manager", mock_tm)
        result = await mod.create_team_func(name="t1")
        assert "Error" in result


class TestDeleteTeamFunc:
    @pytest.mark.asyncio
    async def test_not_found(self, monkeypatch):
        from unittest.mock import MagicMock

        import dazi.team as mod

        mock_tm = MagicMock(team_exists=MagicMock(return_value=False))
        monkeypatch.setattr("dazi._singletons.team_manager", mock_tm)
        result = await mod.delete_team_func(name="ghost")
        assert "not found" in result

    @pytest.mark.asyncio
    async def test_success(self, monkeypatch):
        from unittest.mock import MagicMock

        import dazi.team as mod

        mock_tm = MagicMock(
            team_exists=MagicMock(return_value=True), delete_team=MagicMock(return_value=True)
        )
        monkeypatch.setattr("dazi._singletons.team_manager", mock_tm)
        result = await mod.delete_team_func(name="t1")
        assert "deleted successfully" in result

    @pytest.mark.asyncio
    async def test_error(self, monkeypatch):
        from unittest.mock import MagicMock

        import dazi.team as mod

        mock_tm = MagicMock(
            team_exists=MagicMock(return_value=True),
            delete_team=MagicMock(side_effect=mod.TeamError("active members")),
        )
        monkeypatch.setattr("dazi._singletons.team_manager", mock_tm)
        result = await mod.delete_team_func(name="t1")
        assert "Error" in result


class TestListTeamsFunc:
    @pytest.mark.asyncio
    async def test_no_teams(self, monkeypatch):
        from unittest.mock import MagicMock

        import dazi.team as mod

        monkeypatch.setattr(
            "dazi._singletons.team_manager", MagicMock(list_teams=MagicMock(return_value=[]))
        )
        result = await mod.list_teams_func()
        assert "No teams" in result

    @pytest.mark.asyncio
    async def test_with_teams(self, monkeypatch):
        from unittest.mock import MagicMock

        import dazi.team as mod

        m = MagicMock(name="agent1", status="active")
        t = MagicMock(name="t1", description="desc", members=[m], created_at="2025-01-01")
        monkeypatch.setattr(
            "dazi._singletons.team_manager", MagicMock(list_teams=MagicMock(return_value=[t]))
        )
        result = await mod.list_teams_func()
        assert "t1" in result


class TestShowTeamFunc:
    @pytest.mark.asyncio
    async def test_not_found(self, monkeypatch):
        from unittest.mock import MagicMock

        import dazi.team as mod

        monkeypatch.setattr(
            "dazi._singletons.team_manager", MagicMock(get_team=MagicMock(return_value=None))
        )
        result = await mod.show_team_func(name="ghost")
        assert "not found" in result

    @pytest.mark.asyncio
    async def test_with_members(self, monkeypatch):
        from unittest.mock import MagicMock

        import dazi.team as mod

        member = MagicMock(name="a1", agent_id="a1@t", status="active")
        team = MagicMock(name="t1", description="d", members=[member], created_at="2025-01-01")
        monkeypatch.setattr(
            "dazi._singletons.team_manager", MagicMock(get_team=MagicMock(return_value=team))
        )
        result = await mod.show_team_func(name="t1")
        assert "a1" in result

    @pytest.mark.asyncio
    async def test_no_members(self, monkeypatch):
        from unittest.mock import MagicMock

        import dazi.team as mod

        team = MagicMock(name="t1", description="", members=[], created_at="")
        monkeypatch.setattr(
            "dazi._singletons.team_manager", MagicMock(get_team=MagicMock(return_value=team))
        )
        result = await mod.show_team_func(name="t1")
        assert "no members" in result
