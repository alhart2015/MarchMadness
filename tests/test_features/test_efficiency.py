"""Tests for adjusted efficiency rating computation."""

import numpy as np
import pandas as pd
import pytest

from src.features.efficiency import compute_adjusted_efficiency


@pytest.fixture
def game_data():
    """6-team round-robin with known strength ordering."""
    games = []
    teams = [1, 2, 3, 4, 5, 6]
    # Stronger teams score more, weaker teams score less
    # Team 1 is best, team 6 is worst
    np.random.seed(42)
    for i, t1 in enumerate(teams):
        for j, t2 in enumerate(teams):
            if t1 >= t2:
                continue
            # Higher-ranked (lower index) team wins more
            t1_score = 75 - i * 3 + np.random.randint(-3, 4)
            t2_score = 75 - j * 3 + np.random.randint(-3, 4)
            if t1_score == t2_score:
                t1_score += 1
            if t1_score > t2_score:
                games.append({"WTeamID": t1, "LTeamID": t2, "WScore": t1_score, "LScore": t2_score,
                              "WFGA": 60, "WOR": 10, "WTO": 12, "WFTA": 15, "WLoc": "N",
                              "LFGA": 60, "LOR": 10, "LTO": 12, "LFTA": 15,
                              "Season": 2023, "DayNum": 50})
            else:
                games.append({"WTeamID": t2, "LTeamID": t1, "WScore": t2_score, "LScore": t1_score,
                              "WFGA": 60, "WOR": 10, "WTO": 12, "WFTA": 15, "WLoc": "N",
                              "LFGA": 60, "LOR": 10, "LTO": 12, "LFTA": 15,
                              "Season": 2023, "DayNum": 50})
    return pd.DataFrame(games)


def test_compute_adjusted_efficiency_returns_all_teams(game_data):
    result = compute_adjusted_efficiency(game_data, season=2023, iterations=15, hca=3.5, half_life_days=30, ridge_alpha=1.0)
    assert isinstance(result, pd.DataFrame)
    assert "TeamID" in result.columns
    assert "adj_oe" in result.columns
    assert "adj_de" in result.columns
    assert "adj_em" in result.columns
    assert "adj_tempo" in result.columns
    assert len(result) == 6


def test_adj_em_is_oe_minus_de(game_data):
    result = compute_adjusted_efficiency(game_data, season=2023, iterations=15, hca=3.5, half_life_days=30, ridge_alpha=1.0)
    diff = abs(result["adj_em"] - (result["adj_oe"] - result["adj_de"]))
    assert (diff < 0.01).all()


def test_efficiency_ratings_reasonable_range(game_data):
    result = compute_adjusted_efficiency(game_data, season=2023, iterations=15, hca=3.5, half_life_days=30, ridge_alpha=1.0)
    # AdjEM should be in a reasonable range for college basketball
    assert (result["adj_em"] > -50).all()
    assert (result["adj_em"] < 50).all()


def test_strongest_team_ranked_first(game_data):
    result = compute_adjusted_efficiency(game_data, season=2023, iterations=15, hca=3.5, half_life_days=30, ridge_alpha=1.0)
    # Team 1 is best (highest scores), team 6 is worst
    # Best team should have highest AdjEM
    best_team = result.iloc[0]["TeamID"]  # sorted by adj_em desc
    assert best_team in [1, 2]  # allow small tolerance for randomness in fixture


def test_worst_team_ranked_last(game_data):
    result = compute_adjusted_efficiency(game_data, season=2023, iterations=15, hca=3.5, half_life_days=30, ridge_alpha=1.0)
    worst_team = result.iloc[-1]["TeamID"]
    assert worst_team in [5, 6]
