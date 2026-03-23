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


def build_weighted_matchup_data(
    feature_matrix: pd.DataFrame,
    tourney_results: pd.DataFrame,
    regular_results: pd.DataFrame,
    feature_cols: list[str],
    top_n_team_ids: set[int],
    supplemental_weight: float = 0.25,
    feb_cutoff_day: int = 90,
) -> tuple[pd.DataFrame, pd.Series, np.ndarray]:
    """Build matchup data with tournament games (weight 1.0) and
    supplemental late-season regular season games (weight 0.25).

    Regular season games are filtered to:
    - DayNum >= feb_cutoff_day (~Feb 1)
    - Both teams in top_n_team_ids

    Returns: (X, y, sample_weights)
    """
    # Tournament matchups (weight 1.0)
    X_t, y_t = build_matchup_data(feature_matrix, tourney_results, feature_cols)
    w_t = np.ones(len(y_t))

    # Supplemental matchups (weight 0.25)
    late_reg = regular_results[regular_results["DayNum"] >= feb_cutoff_day].copy()
    late_reg = late_reg[
        late_reg["WTeamID"].isin(top_n_team_ids)
        & late_reg["LTeamID"].isin(top_n_team_ids)
    ]

    if late_reg.empty:
        return X_t, y_t, w_t

    X_s, y_s = build_matchup_data(feature_matrix, late_reg, feature_cols)
    w_s = np.full(len(y_s), supplemental_weight)

    X = pd.concat([X_t, X_s], ignore_index=True)
    y = pd.concat([y_t, y_s], ignore_index=True)
    w = np.concatenate([w_t, w_s])

    return X, y, w
