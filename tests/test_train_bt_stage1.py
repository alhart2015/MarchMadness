"""Unit tests for src/train_bt_stage1.py.

Per-season Bradley-Terry trainer: design-matrix builder, home-court
extraction, and a synthetic-data recovery check. Full 22-season run
is exercised in Task 2, not here.
"""
import numpy as np
import pandas as pd
import pytest


def test_extract_home_court_value():
    """WLoc -> home-court column value: H -> +1 (winner home), A -> -1
    (winner away), N -> 0 (neutral)."""
    from src.train_bt_stage1 import extract_home_court_value

    assert extract_home_court_value("H") == 1
    assert extract_home_court_value("A") == -1
    assert extract_home_court_value("N") == 0


def test_build_design_matrix_shape():
    """3 teams, 4 games -> design matrix shape (4, 3+1)."""
    from src.train_bt_stage1 import build_design_matrix

    games = pd.DataFrame([
        {"WTeamID": 1101, "LTeamID": 1102, "WLoc": "H"},
        {"WTeamID": 1102, "LTeamID": 1103, "WLoc": "A"},
        {"WTeamID": 1101, "LTeamID": 1103, "WLoc": "N"},
        {"WTeamID": 1103, "LTeamID": 1101, "WLoc": "H"},
    ])
    team_ids = [1101, 1102, 1103]

    X, y = build_design_matrix(games, team_ids)

    # X: (4 games, 3 teams + 1 home-court column)
    assert X.shape == (4, 4)
    # y: all 1s (we encode one row per game with winner=+1)
    assert (y == 1).all()
    # Convert sparse to dense for inspection.
    X_dense = X.toarray()
    # Game 0: 1101 beats 1102 at home -> winner_col=0, loser_col=1, hc=+1
    assert X_dense[0, 0] == 1 and X_dense[0, 1] == -1 and X_dense[0, 3] == 1
    # Game 1: 1102 beats 1103 away -> winner_col=1, loser_col=2, hc=-1
    assert X_dense[1, 1] == 1 and X_dense[1, 2] == -1 and X_dense[1, 3] == -1
    # Game 2: neutral -> hc=0
    assert X_dense[2, 0] == 1 and X_dense[2, 2] == -1 and X_dense[2, 3] == 0


def test_recover_strengths_synthetic():
    """With teams of true strengths s = (0, 1, 2) and many simulated
    games, the fitted strengths should be in the right ORDER and within
    ~0.5 of the true differences after L2 regularization shrinks them."""
    from src.train_bt_stage1 import build_design_matrix, fit_bradley_terry

    rng = np.random.default_rng(42)
    true_s = np.array([0.0, 1.0, 2.0])
    n_games = 4000
    rows = []
    for _ in range(n_games):
        i, j = rng.choice(3, size=2, replace=False)
        # P(i beats j) = sigmoid(s_i - s_j); home-court irrelevant here.
        p = 1.0 / (1.0 + np.exp(-(true_s[i] - true_s[j])))
        winner = i if rng.random() < p else j
        loser = j if winner == i else i
        rows.append({
            "WTeamID": 1101 + winner,
            "LTeamID": 1101 + loser,
            "WLoc": "N",
        })
    games = pd.DataFrame(rows)
    team_ids = [1101, 1102, 1103]

    X, y = build_design_matrix(games, team_ids)
    coefs = fit_bradley_terry(X, y, C=10.0)
    s_fit = coefs[:3]
    # Order: team 0 weakest, team 2 strongest.
    assert s_fit[0] < s_fit[1] < s_fit[2]
    # Pairwise differences within ~0.5 of truth.
    assert abs((s_fit[2] - s_fit[0]) - 2.0) < 0.5
    assert abs((s_fit[1] - s_fit[0]) - 1.0) < 0.5


def test_predict_pairwise_for_field():
    """Given fitted strengths, predict_pairwise produces P(a beats b) =
    sigmoid(s_a - s_b) for every unordered pair (a < b) in the field."""
    from src.train_bt_stage1 import predict_pairwise_for_field

    team_ids = [1101, 1102, 1103]
    s = np.array([0.0, 1.0, 2.0])
    field = [1101, 1102, 1103]

    rows = predict_pairwise_for_field(season=2003, field=field,
                                       team_ids=team_ids, strengths=s)
    df = pd.DataFrame(rows)
    assert list(df.columns) == ["season", "team_a", "team_b", "p_a_wins"]
    assert len(df) == 3  # 3 unordered pairs from 3 teams
    assert (df["team_a"] < df["team_b"]).all()
    # P(1101 beats 1102) = sigmoid(0 - 1) = ~0.269
    p_1101_1102 = float(df[(df.team_a == 1101) & (df.team_b == 1102)].p_a_wins.iloc[0])
    assert abs(p_1101_1102 - 1.0 / (1.0 + np.exp(1.0))) < 1e-6
