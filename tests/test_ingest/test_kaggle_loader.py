"""Tests for Kaggle data loading."""

import pandas as pd
import pytest

from src.ingest.kaggle_loader import load_kaggle_data


@pytest.fixture
def kaggle_dir(tmp_path, sample_detailed_results, sample_teams, sample_seeds):
    """Create a fake Kaggle data directory with CSVs."""
    d = tmp_path / "kaggle"
    d.mkdir()
    sample_detailed_results.to_csv(d / "MRegularSeasonDetailedResults.csv", index=False)
    sample_teams.to_csv(d / "MTeams.csv", index=False)
    sample_seeds.to_csv(d / "MNCAATourneySeeds.csv", index=False)
    # Create minimal compact results
    compact = sample_detailed_results[["Season", "DayNum", "WTeamID", "WScore", "LTeamID", "LScore", "WLoc", "NumOT"]]
    compact.to_csv(d / "MRegularSeasonCompactResults.csv", index=False)
    compact.to_csv(d / "MNCAATourneyCompactResults.csv", index=False)
    sample_detailed_results.to_csv(d / "MNCAATourneyDetailedResults.csv", index=False)
    # Minimal conferences
    pd.DataFrame({
        "ConfAbbrev": ["big12", "mwc", "mac", "sec"],
        "Description": ["Big 12", "Mountain West", "Mid-American", "SEC"],
    }).to_csv(d / "MConferences.csv", index=False)
    pd.DataFrame({
        "Season": [2023, 2023, 2023, 2023],
        "TeamID": [1101, 1102, 1103, 1104],
        "ConfAbbrev": ["big12", "mwc", "mac", "sec"],
    }).to_csv(d / "MTeamConferences.csv", index=False)
    # Minimal Massey ordinals
    pd.DataFrame({
        "Season": [2023, 2023, 2023, 2023],
        "RankingDayNum": [128, 128, 128, 128],
        "SystemName": ["POM", "POM", "POM", "POM"],
        "TeamID": [1101, 1102, 1103, 1104],
        "OrdinalRank": [1, 5, 50, 100],
    }).to_csv(d / "MMasseyOrdinals.csv", index=False)
    return d


def test_load_kaggle_data_returns_dict(kaggle_dir):
    data = load_kaggle_data(str(kaggle_dir))
    assert isinstance(data, dict)
    assert "teams" in data
    assert "regular_season" in data
    assert "tourney_results" in data
    assert "seeds" in data
    assert "massey" in data
    assert "conferences" in data
    assert "team_conferences" in data


def test_load_kaggle_data_teams_shape(kaggle_dir):
    data = load_kaggle_data(str(kaggle_dir))
    assert len(data["teams"]) == 4
    assert "TeamID" in data["teams"].columns
    assert "TeamName" in data["teams"].columns


def test_load_kaggle_data_massey_filtered(kaggle_dir):
    """Massey ordinals should be filtered to latest day per season."""
    data = load_kaggle_data(str(kaggle_dir))
    # Only one day in our test data, so all rows kept
    assert len(data["massey"]) == 4


def test_load_kaggle_data_missing_dir():
    with pytest.raises(FileNotFoundError):
        load_kaggle_data("/nonexistent/path")
