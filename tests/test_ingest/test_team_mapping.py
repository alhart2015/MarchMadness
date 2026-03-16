"""Tests for team name fuzzy matching and overrides."""

import pandas as pd
import pytest

from src.ingest.team_mapping import build_team_mapping, apply_overrides


@pytest.fixture
def kaggle_teams():
    return pd.DataFrame({
        "TeamID": [1101, 1102, 1103],
        "TeamName": ["Connecticut", "Saint Mary's (CA)", "Miami (FL)"],
    })


@pytest.fixture
def external_names():
    return ["UConn", "Saint Mary's", "Miami FL"]


@pytest.fixture
def overrides_path(tmp_path):
    overrides = pd.DataFrame({
        "external_name": ["UConn"],
        "kaggle_team_id": [1101],
    })
    path = tmp_path / "overrides.csv"
    overrides.to_csv(path, index=False)
    return path


def test_apply_overrides(kaggle_teams, overrides_path):
    result = apply_overrides(["UConn", "Unknown Team"], overrides_path)
    assert result["UConn"] == 1101
    assert "Unknown Team" not in result


def test_build_team_mapping_with_overrides(kaggle_teams, external_names, overrides_path):
    mapping = build_team_mapping(
        kaggle_teams=kaggle_teams,
        external_names=external_names,
        overrides_path=str(overrides_path),
        auto_threshold=85,
        review_threshold=70,
    )
    # UConn should map via override
    assert mapping["UConn"] == 1101


def test_build_team_mapping_fuzzy(kaggle_teams):
    mapping = build_team_mapping(
        kaggle_teams=kaggle_teams,
        external_names=["Saint Mary's"],
        overrides_path=None,
        auto_threshold=80,
        review_threshold=60,
    )
    # Should fuzzy match to Saint Mary's (CA)
    assert mapping["Saint Mary's"] == 1102


def test_build_team_mapping_no_match(kaggle_teams):
    mapping = build_team_mapping(
        kaggle_teams=kaggle_teams,
        external_names=["Totally Unknown University"],
        overrides_path=None,
        auto_threshold=85,
        review_threshold=70,
    )
    assert "Totally Unknown University" not in mapping
