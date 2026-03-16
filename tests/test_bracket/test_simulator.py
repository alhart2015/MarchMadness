"""Tests for Monte Carlo bracket simulation."""

import pandas as pd
import pytest

from src.bracket.simulator import (
    load_bracket,
    simulate_tournament,
    get_advancement_probabilities,
    FIRST_ROUND_MATCHUPS,
)


@pytest.fixture
def bracket_df():
    """64-team 4-region bracket."""
    regions = ["East", "West", "South", "Midwest"]
    rows = []
    team_id = 101
    for region in regions:
        for seed in range(1, 17):
            rows.append({"Region": region, "Seed": seed, "TeamID": team_id, "TeamName": f"Team{team_id}"})
            team_id += 1
    return pd.DataFrame(rows)


@pytest.fixture
def mock_predict():
    """Predict function: higher seed always wins with 70% prob."""
    def predict(team_a_features, team_b_features):
        seed_a = team_a_features["seed"]
        seed_b = team_b_features["seed"]
        if seed_a < seed_b:
            return 0.7
        elif seed_a > seed_b:
            return 0.3
        else:
            return 0.5
    return predict


def test_first_round_matchups():
    assert FIRST_ROUND_MATCHUPS == [(1, 16), (8, 9), (5, 12), (4, 13), (6, 11), (3, 14), (7, 10), (2, 15)]


def test_simulate_tournament_returns_results(bracket_df, mock_predict):
    feature_matrix = pd.DataFrame({
        "TeamID": list(range(101, 165)),
        "seed": list(range(1, 17)) * 4,
        "adj_em": [30 - (i % 16) * 2 for i in range(64)],
    })
    results = simulate_tournament(
        bracket=bracket_df,
        feature_matrix=feature_matrix,
        predict_fn=mock_predict,
        feature_cols=["seed", "adj_em"],
        n_simulations=100,
        random_seed=42,
    )
    assert "advancement_counts" in results
    assert "champions" in results
    # 1-seeds should collectively dominate championships
    one_seeds = [101, 117, 133, 149]  # 1-seeds from each region
    champ_counts = results["champions"]
    one_seed_wins = sum(champ_counts.get(t, 0) for t in one_seeds)
    assert one_seed_wins > 30  # 1-seeds dominate (~37% combined championship rate)


def test_get_advancement_probabilities():
    counts = {101: {1: 100, 2: 80, 3: 50, 4: 20}, 102: {1: 100, 2: 20, 3: 5, 4: 0}}
    probs = get_advancement_probabilities(counts, n_simulations=100)
    assert probs[101][4] == 0.2
    assert probs[102][3] == 0.05
