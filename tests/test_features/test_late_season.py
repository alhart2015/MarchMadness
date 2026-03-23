import pandas as pd
import numpy as np
import pytest
from src.features.late_season import compute_late_season_metrics


@pytest.fixture
def sample_detailed_results():
    """6-team schedule: teams 1-3 are strong (top-100), 4-6 are weak."""
    rows = []
    for day in [105, 110, 115, 120, 125, 130]:
        rows.append({"Season": 2025, "DayNum": day, "WTeamID": 1, "LTeamID": 4,
                      "WScore": 80, "LScore": 60, "WFGA": 60, "LFGA": 60,
                      "WOR": 10, "LOR": 10, "WTO": 12, "LTO": 12,
                      "WFTA": 20, "LFTA": 20, "NumOT": 0})
        rows.append({"Season": 2025, "DayNum": day, "WTeamID": 1, "LTeamID": 2,
                      "WScore": 72, "LScore": 70, "WFGA": 60, "LFGA": 60,
                      "WOR": 10, "LOR": 10, "WTO": 12, "LTO": 12,
                      "WFTA": 20, "LFTA": 20, "NumOT": 0})
    rows.append({"Season": 2025, "DayNum": 10, "WTeamID": 1, "LTeamID": 2,
                  "WScore": 90, "LScore": 50, "WFGA": 60, "LFGA": 60,
                  "WOR": 10, "LOR": 10, "WTO": 12, "LTO": 12,
                  "WFTA": 20, "LFTA": 20, "NumOT": 0})
    return pd.DataFrame(rows)


@pytest.fixture
def top_100_teams():
    return {1, 2, 3}


def test_returns_expected_columns(sample_detailed_results, top_100_teams):
    result = compute_late_season_metrics(sample_detailed_results, 2025, top_100_teams)
    assert set(result.columns) >= {"TeamID", "Season", "late_adj_oe", "late_adj_de",
                                    "late_adj_em", "late_sos"}


def test_filters_to_late_season(sample_detailed_results, top_100_teams):
    result = compute_late_season_metrics(sample_detailed_results, 2025, top_100_teams,
                                          late_window_days=30)
    team1 = result[result["TeamID"] == 1]
    assert len(team1) == 1


def test_filters_to_top100_opponents(sample_detailed_results, top_100_teams):
    result = compute_late_season_metrics(sample_detailed_results, 2025, top_100_teams)
    team1 = result[result["TeamID"] == 1].iloc[0]
    assert team1["late_adj_oe"] < 130


def test_late_sos_reflects_opponent_strength(sample_detailed_results, top_100_teams):
    result = compute_late_season_metrics(sample_detailed_results, 2025, top_100_teams)
    team1 = result[result["TeamID"] == 1]
    assert team1.iloc[0]["late_sos"] > 0


def test_team_with_no_top100_games_excluded(sample_detailed_results, top_100_teams):
    result = compute_late_season_metrics(sample_detailed_results, 2025, top_100_teams)
    team_ids = set(result["TeamID"].tolist())
    assert 6 not in team_ids


from src.features.late_season import compute_trajectory_features


@pytest.fixture
def trending_results():
    """Team 1 is improving, Team 2 is declining over 45 days."""
    rows = []
    for i, day in enumerate(range(90, 132, 3)):
        margin = 5 + i * 2
        rows.append({"Season": 2025, "DayNum": day,
                      "WTeamID": 1, "LTeamID": 99,
                      "WScore": 70 + margin, "LScore": 70,
                      "WFGA": 60, "LFGA": 60, "WOR": 10, "LOR": 10,
                      "WTO": 12, "LTO": 12, "WFTA": 20, "LFTA": 20,
                      "NumOT": 0})
    for i, day in enumerate(range(90, 132, 3)):
        margin = 20 - i * 2
        rows.append({"Season": 2025, "DayNum": day,
                      "WTeamID": 2, "LTeamID": 99,
                      "WScore": 70 + margin, "LScore": 70,
                      "WFGA": 60, "LFGA": 60, "WOR": 10, "LOR": 10,
                      "WTO": 12, "LTO": 12, "WFTA": 20, "LFTA": 20,
                      "NumOT": 0})
    return pd.DataFrame(rows)


def test_trajectory_returns_expected_columns(trending_results):
    result = compute_trajectory_features(trending_results, 2025)
    assert set(result.columns) >= {"TeamID", "Season", "efficiency_trend", "margin_trend"}


def test_improving_team_has_positive_trend(trending_results):
    result = compute_trajectory_features(trending_results, 2025)
    team1 = result[result["TeamID"] == 1].iloc[0]
    assert team1["margin_trend"] > 0


def test_declining_team_has_negative_trend(trending_results):
    result = compute_trajectory_features(trending_results, 2025)
    team2 = result[result["TeamID"] == 2].iloc[0]
    assert team2["margin_trend"] < 0
