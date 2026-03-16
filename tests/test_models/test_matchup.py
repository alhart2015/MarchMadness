"""Tests for matchup training data construction."""

import pandas as pd
import pytest

from src.models.matchup import build_matchup_data

FEATURE_COLS = ["adj_em", "adj_oe", "adj_de", "seed"]


@pytest.fixture
def feature_matrix():
    return pd.DataFrame({
        "TeamID": [1, 2, 3, 4],
        "Season": [2023, 2023, 2023, 2023],
        "adj_em": [20.0, 12.0, 3.0, -8.0],
        "adj_oe": [115.0, 110.0, 105.0, 100.0],
        "adj_de": [95.0, 98.0, 102.0, 108.0],
        "seed": [1, 2, 3, 4],
    })


@pytest.fixture
def tourney_results():
    return pd.DataFrame({
        "Season": [2023, 2023],
        "WTeamID": [1, 2],
        "LTeamID": [4, 3],
    })


def test_build_matchup_data_symmetric(feature_matrix, tourney_results):
    X, y = build_matchup_data(feature_matrix, tourney_results, FEATURE_COLS)
    # 2 games * 2 (symmetric) = 4 rows
    assert len(X) == 4
    assert len(y) == 4


def test_build_matchup_data_labels(feature_matrix, tourney_results):
    X, y = build_matchup_data(feature_matrix, tourney_results, FEATURE_COLS)
    # Half should be wins (1), half losses (0)
    assert y.sum() == 2
    assert (y == 0).sum() == 2


def test_build_matchup_data_feature_differences(feature_matrix, tourney_results):
    X, y = build_matchup_data(feature_matrix, tourney_results, FEATURE_COLS)
    # First row: team 1 vs team 4 (winner perspective)
    # adj_em diff should be 20 - (-8) = 28
    first_win_row = X[y == 1].iloc[0]
    assert abs(first_win_row["adj_em"]) > 0  # non-zero difference
