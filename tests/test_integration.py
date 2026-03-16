"""Integration test: full pipeline with synthetic data."""

import numpy as np
import pandas as pd
import pytest

from src.features.efficiency import compute_adjusted_efficiency
from src.features.four_factors import compute_four_factors
from src.features.feature_matrix import build_feature_matrix
from src.models.matchup import build_matchup_data
from src.models.train import train_model, predict_matchup
from src.bracket.simulator import simulate_tournament, get_advancement_probabilities
from src.bracket.strategies import chalk_bracket


def _generate_synthetic_season(season: int, n_teams: int = 16, games_per_team: int = 10) -> pd.DataFrame:
    """Generate synthetic box score data."""
    np.random.seed(season)
    teams = list(range(1, n_teams + 1))
    games = []
    for _ in range(n_teams * games_per_team // 2):
        t1, t2 = np.random.choice(teams, 2, replace=False)
        # Stronger teams (lower ID) score more on average
        s1 = int(75 - t1 + np.random.randint(-10, 11))
        s2 = int(75 - t2 + np.random.randint(-10, 11))
        if s1 == s2:
            s1 += 1
        w, l = (t1, t2) if s1 > s2 else (t2, t1)
        ws, ls = max(s1, s2), min(s1, s2)
        games.append({
            "Season": season, "DayNum": np.random.randint(1, 132),
            "WTeamID": w, "WScore": ws, "LTeamID": l, "LScore": ls, "WLoc": "N", "NumOT": 0,
            "WFGM": 28, "WFGA": 58, "WFGM3": 8, "WFGA3": 20, "WFTM": 10, "WFTA": 14,
            "WOR": 10, "WDR": 22, "WAst": 14, "WTO": 12, "WStl": 6, "WBlk": 3, "WPF": 17,
            "LFGM": 24, "LFGA": 58, "LFGM3": 6, "LFGA3": 18, "LFTM": 10, "LFTA": 14,
            "LOR": 8, "LDR": 20, "LAst": 12, "LTO": 14, "LStl": 5, "LBlk": 2, "LPF": 16,
        })
    return pd.DataFrame(games)


def test_full_pipeline_synthetic():
    """Test the complete pipeline from raw data to bracket output."""
    # Generate 3 seasons of data
    all_results = pd.concat([_generate_synthetic_season(s) for s in [2021, 2022, 2023]])
    seeds = pd.DataFrame({
        "Season": [s for s in [2021, 2022, 2023] for _ in range(16)],
        "Seed": [f"W{i:02d}" for _ in range(3) for i in range(1, 17)],
        "TeamID": list(range(1, 17)) * 3,
    })
    massey = pd.DataFrame({
        "Season": [s for s in [2021, 2022, 2023] for _ in range(16)],
        "RankingDayNum": [128] * 48,
        "SystemName": ["POM"] * 48,
        "TeamID": list(range(1, 17)) * 3,
        "OrdinalRank": list(range(1, 17)) * 3,
    })
    tourney = pd.DataFrame({
        "Season": [2021, 2021, 2021, 2021, 2021, 2022, 2022, 2022, 2022, 2022],
        "WTeamID": [1, 2, 3, 4, 5, 1, 2, 3, 4, 5],
        "LTeamID": [16, 15, 14, 13, 12, 16, 15, 14, 13, 12],
    })

    feature_cols = ["adj_em", "adj_oe", "adj_de", "seed"]
    all_features = []
    for season in [2021, 2022, 2023]:
        season_data = all_results[all_results["Season"] == season]
        eff = compute_adjusted_efficiency(season_data, season, iterations=5, hca=3.5, half_life_days=30, ridge_alpha=1.0)
        ff = compute_four_factors(season_data, season)
        season_massey = massey[massey["Season"] == season]
        matrix = build_feature_matrix(eff, ff, seeds, season_massey, all_results, season, ["POM"])
        matrix["Season"] = season
        all_features.append(matrix)
    full_matrix = pd.concat(all_features, ignore_index=True)

    # Train model
    X, y = build_matchup_data(full_matrix, tourney, feature_cols)
    assert len(X) > 0
    model = train_model(X, y, random_seed=42)

    # Simulate bracket for 2023 — need 4 regions, 64 teams
    # Reuse the 16 teams across 4 regions (same features, different bracket slots)
    regions = ["East", "West", "South", "Midwest"]
    bracket_rows = []
    for region in regions:
        for seed in range(1, 17):
            bracket_rows.append({
                "Region": region,
                "Seed": seed,
                "TeamID": seed,  # reuse same team IDs across regions for simplicity
                "TeamName": f"Team{seed}",
            })
    bracket = pd.DataFrame(bracket_rows)
    current = full_matrix[full_matrix["Season"] == 2023]

    def predict_fn(a_feats, b_feats):
        diff = {c: a_feats[c] - b_feats[c] for c in feature_cols}
        return predict_matchup(model, pd.DataFrame([diff]))

    results = simulate_tournament(bracket, current, predict_fn, feature_cols, n_simulations=100, random_seed=42)

    assert len(results["champions"]) > 0
    probs = get_advancement_probabilities(results["advancement_counts"], 100)
    picks = chalk_bracket(bracket, probs)
    assert len(picks) > 0
    # Champion should be a single team
    assert len(picks[6]) == 1
