"""Monte Carlo tournament bracket simulation."""

import logging
from collections import defaultdict
from typing import Callable

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Standard NCAA first-round matchups by seed within a region
FIRST_ROUND_MATCHUPS = [
    (1, 16), (8, 9), (5, 12), (4, 13),
    (6, 11), (3, 14), (7, 10), (2, 15),
]


def load_bracket(path: str) -> pd.DataFrame:
    """Load bracket CSV with columns: Region, Seed, TeamID, TeamName."""
    df = pd.read_csv(path)
    required = {"Region", "Seed", "TeamID", "TeamName"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Bracket CSV missing columns: {missing}")
    return df


def _get_team_features(team_id: int, feature_matrix: pd.DataFrame, feature_cols: list[str]) -> dict:
    """Look up features for a team."""
    row = feature_matrix[feature_matrix["TeamID"] == team_id]
    if row.empty:
        raise ValueError(f"Team {team_id} not found in feature matrix")
    return row.iloc[0][feature_cols].to_dict()


def _simulate_game(
    team_a: int,
    team_b: int,
    feature_matrix: pd.DataFrame,
    predict_fn: Callable,
    feature_cols: list[str],
    rng: np.random.Generator,
) -> int:
    """Simulate a single game, return winning team ID."""
    a_features = _get_team_features(team_a, feature_matrix, feature_cols)
    b_features = _get_team_features(team_b, feature_matrix, feature_cols)
    prob_a_wins = predict_fn(a_features, b_features)
    return team_a if rng.random() < prob_a_wins else team_b


def _simulate_region(
    region_teams: pd.DataFrame,
    feature_matrix: pd.DataFrame,
    predict_fn: Callable,
    feature_cols: list[str],
    rng: np.random.Generator,
    advancement: dict[int, dict[int, int]],
) -> int:
    """Simulate a single region through 4 rounds, return regional champion TeamID."""
    # Map seed -> TeamID
    seed_to_team = dict(zip(region_teams["Seed"], region_teams["TeamID"]))

    # Round 1: 8 games from FIRST_ROUND_MATCHUPS
    round_winners = []
    for seed_a, seed_b in FIRST_ROUND_MATCHUPS:
        team_a = seed_to_team[seed_a]
        team_b = seed_to_team[seed_b]
        winner = _simulate_game(team_a, team_b, feature_matrix, predict_fn, feature_cols, rng)
        round_winners.append(winner)
        advancement[winner][1] = advancement[winner].get(1, 0) + 1

    # Rounds 2-4
    current = round_winners
    for round_num in range(2, 5):
        next_round = []
        for i in range(0, len(current), 2):
            winner = _simulate_game(current[i], current[i + 1], feature_matrix, predict_fn, feature_cols, rng)
            next_round.append(winner)
            advancement[winner][round_num] = advancement[winner].get(round_num, 0) + 1
        current = next_round

    return current[0]  # regional champion


def simulate_tournament(
    bracket: pd.DataFrame,
    feature_matrix: pd.DataFrame,
    predict_fn: Callable,
    feature_cols: list[str],
    n_simulations: int = 10000,
    random_seed: int = 42,
) -> dict:
    """Run Monte Carlo simulation of the full tournament.

    Returns dict with:
        advancement_counts: {team_id: {round: count}}
        champions: {team_id: count}
        n_simulations: int
    """
    rng = np.random.default_rng(random_seed)
    regions = sorted(bracket["Region"].unique())
    advancement = defaultdict(lambda: defaultdict(int))
    champions = defaultdict(int)

    # NCAA semifinal pairings: East vs South, West vs Midwest
    _SEMI_PAIRINGS = [("East", "South"), ("West", "Midwest")]

    for sim in range(n_simulations):
        # Simulate each region (rounds 1-4: R64, R32, S16, E8)
        regional_champs = {}
        for region in regions:
            region_teams = bracket[bracket["Region"] == region].copy()
            champ = _simulate_region(
                region_teams, feature_matrix, predict_fn, feature_cols, rng, advancement
            )
            regional_champs[region] = champ
            # Round 5 = reached Final Four (won region)
            advancement[champ][5] = advancement[champ].get(5, 0) + 1

        # Semifinals: East vs South, West vs Midwest
        # Fall back to positional pairing if region names don't match
        if all(r in regional_champs for pair in _SEMI_PAIRINGS for r in pair):
            semi1 = _simulate_game(regional_champs["East"], regional_champs["South"], feature_matrix, predict_fn, feature_cols, rng)
            semi2 = _simulate_game(regional_champs["West"], regional_champs["Midwest"], feature_matrix, predict_fn, feature_cols, rng)
        else:
            champ_list = list(regional_champs.values())
            semi1 = _simulate_game(champ_list[0], champ_list[1], feature_matrix, predict_fn, feature_cols, rng)
            semi2 = _simulate_game(champ_list[2], champ_list[3], feature_matrix, predict_fn, feature_cols, rng)

        # Championship (round 6 = won championship)
        champion = _simulate_game(semi1, semi2, feature_matrix, predict_fn, feature_cols, rng)
        advancement[champion][6] = advancement[champion].get(6, 0) + 1
        champions[champion] += 1

        if (sim + 1) % 1000 == 0:
            logger.info("Completed %d / %d simulations", sim + 1, n_simulations)

    return {
        "advancement_counts": dict(advancement),
        "champions": dict(champions),
        "n_simulations": n_simulations,
    }


def get_advancement_probabilities(
    advancement_counts: dict[int, dict[int, int]],
    n_simulations: int,
) -> dict[int, dict[int, float]]:
    """Convert raw counts to probabilities."""
    probs = {}
    for team_id, rounds in advancement_counts.items():
        probs[team_id] = {r: count / n_simulations for r, count in rounds.items()}
    return probs
