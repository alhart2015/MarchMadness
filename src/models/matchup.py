"""Build symmetric matchup training data from tournament results.

Each game (team A vs team B) is represented by ONE feature row of diff
columns (A_feat - B_feat). The diff swaps sign in the loser-perspective
row of the symmetric pair so the model learns directional advantage.

History:
  v6 attempt: added (A+B)/2 avg columns (matchup-interaction features).
    22-season backtest delta: +7 pts vs v4 (within season-to-season
    noise). Reverted in favor of diff-only.
  v7 attempt: added a `round` column (1..6 for R64..Champ, 0 for
    supplemental regular-season). 22-season delta: -10 pts vs v4. CV log
    loss flat (0.4384 -> 0.4387). The model could not extract round-
    conditional signal from the existing features. Reverted.

The day_to_round helper is kept here for any future iteration that wants
to do per-round prediction infrastructure (rather than per-round training
features).
"""

import numpy as np
import pandas as pd


def day_to_round(day_num) -> int:
    """Map Kaggle DayNum to the tournament round number.

    1 = R64, 2 = R32, 3 = S16, 4 = E8, 5 = F4, 6 = Champ.
    Returns 0 for play-ins / regular-season / unknown days.
    """
    try:
        d = int(day_num)
    except (TypeError, ValueError):
        return 0
    if 136 <= d <= 137:
        return 1
    if 138 <= d <= 139:
        return 2
    if 143 <= d <= 144:
        return 3
    if 145 <= d <= 146:
        return 4
    if d == 152:
        return 5
    if d == 154:
        return 6
    return 0


def expand_feature_cols(feature_cols: list[str]) -> list[str]:
    """Matchup-row column names: <feat>_diff for each raw feature."""
    return [f"{c}_diff" for c in feature_cols]


def build_matchup_features(a_vals: np.ndarray, b_vals: np.ndarray) -> np.ndarray:
    """Build a single matchup feature row from team A's and B's raw features.

    Returns: flat array of length len(a_vals), the element-wise diff (A - B).
    """
    return a_vals - b_vals


def build_matchup_data(
    feature_matrix: pd.DataFrame,
    tourney_results: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[pd.DataFrame, pd.Series]:
    """Build training data for matchup prediction.

    Each game produces two rows (winner perspective, loser perspective);
    each row has len(feature_cols) columns (diff cols).
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

        rows.append(build_matchup_features(w_vals, l_vals))
        labels.append(1)
        rows.append(build_matchup_features(l_vals, w_vals))
        labels.append(0)

    X = pd.DataFrame(rows, columns=expand_feature_cols(feature_cols))
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
