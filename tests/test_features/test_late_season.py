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


from src.features.late_season import compute_conf_tourney_features


@pytest.fixture
def sample_conf_tourney():
    """Team 1 wins 3 games (champion), Team 2 wins 1, Team 3 wins 0."""
    return pd.DataFrame([
        {"Season": 2025, "ConfAbbrev": "B12", "DayNum": 120, "WTeamID": 1, "LTeamID": 3},
        {"Season": 2025, "ConfAbbrev": "B12", "DayNum": 121, "WTeamID": 1, "LTeamID": 2},
        {"Season": 2025, "ConfAbbrev": "B12", "DayNum": 122, "WTeamID": 1, "LTeamID": 5},
        {"Season": 2025, "ConfAbbrev": "SEC", "DayNum": 120, "WTeamID": 2, "LTeamID": 6},
        {"Season": 2025, "ConfAbbrev": "SEC", "DayNum": 121, "WTeamID": 7, "LTeamID": 2},
    ])


def test_conf_tourney_returns_expected_columns(sample_conf_tourney):
    result = compute_conf_tourney_features(sample_conf_tourney, 2025)
    assert set(result.columns) >= {"TeamID", "Season", "conf_tourney_wins", "conf_tourney_champ"}


def test_champion_has_most_wins(sample_conf_tourney):
    result = compute_conf_tourney_features(sample_conf_tourney, 2025)
    team1 = result[result["TeamID"] == 1].iloc[0]
    assert team1["conf_tourney_wins"] == 3
    assert team1["conf_tourney_champ"] == 1


def test_one_win_not_champion(sample_conf_tourney):
    result = compute_conf_tourney_features(sample_conf_tourney, 2025)
    team2 = result[result["TeamID"] == 2].iloc[0]
    assert team2["conf_tourney_wins"] == 1
    assert team2["conf_tourney_champ"] == 0


def test_zero_wins_team_included(sample_conf_tourney):
    result = compute_conf_tourney_features(sample_conf_tourney, 2025)
    team3 = result[result["TeamID"] == 3]
    assert len(team3) == 1
    assert team3.iloc[0]["conf_tourney_wins"] == 0


from src.features.late_season import compute_vegas_trend


@pytest.fixture
def sample_vegas_records():
    """Team with spreads getting better (more favored) late in season."""
    return pd.DataFrame([
        {"TeamID": 1, "Season": 2025, "date": "11/15/2025", "team_spread": 3.0},
        {"TeamID": 1, "Season": 2025, "date": "12/01/2025", "team_spread": 2.0},
        {"TeamID": 1, "Season": 2025, "date": "12/15/2025", "team_spread": 1.0},
        {"TeamID": 1, "Season": 2025, "date": "02/15/2026", "team_spread": -3.0},
        {"TeamID": 1, "Season": 2025, "date": "03/01/2026", "team_spread": -5.0},
        {"TeamID": 1, "Season": 2025, "date": "03/10/2026", "team_spread": -7.0},
    ])


def test_vegas_trend_returns_expected_columns(sample_vegas_records):
    result = compute_vegas_trend(sample_vegas_records, 2025)
    assert set(result.columns) >= {"TeamID", "Season", "vegas_late_spread_delta"}


def test_improving_team_has_negative_delta(sample_vegas_records):
    """More negative spread = more favored = team getting stronger."""
    result = compute_vegas_trend(sample_vegas_records, 2025)
    team1 = result[result["TeamID"] == 1].iloc[0]
    assert team1["vegas_late_spread_delta"] < 0
