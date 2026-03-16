"""Build symmetric matchup training data from tournament results."""

import pandas as pd
import numpy as np


def build_matchup_data(
    feature_matrix: pd.DataFrame,
    tourney_results: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[pd.DataFrame, pd.Series]:
    """Build training data for matchup prediction.

    Each tournament game produces two rows (A vs B and B vs A).
    Features are the difference: team_A_features - team_B_features.
    Target is 1 if team A won, 0 otherwise.

    Returns (X, y) where X is the feature difference DataFrame
    and y is the binary target Series.
    """
    rows = []
    labels = []

    for _, game in tourney_results.iterrows():
        season = game["Season"]
        w_id = game["WTeamID"]
        l_id = game["LTeamID"]

        w_features = feature_matrix[
            (feature_matrix["TeamID"] == w_id) & (feature_matrix["Season"] == season)
        ][feature_cols]
        l_features = feature_matrix[
            (feature_matrix["TeamID"] == l_id) & (feature_matrix["Season"] == season)
        ][feature_cols]

        if w_features.empty or l_features.empty:
            continue

        w_vals = w_features.iloc[0].values
        l_vals = l_features.iloc[0].values

        # Winner perspective: W - L, label = 1
        rows.append(w_vals - l_vals)
        labels.append(1)

        # Loser perspective: L - W, label = 0
        rows.append(l_vals - w_vals)
        labels.append(0)

    X = pd.DataFrame(rows, columns=feature_cols)
    y = pd.Series(labels, name="win")
    return X, y
