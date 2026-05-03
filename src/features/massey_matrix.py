"""Massey-matrix MOV ratings: per-team-per-season ratings from a
least-squares solve over regular-season game results with a jointly
estimated home-court constant and MOV capping.

See docs/superpowers/specs/2026-05-03-massey-matrix-feature-design.md.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_PRODUCER_VERSION = "v1"


def _solve_one_season(games_df: pd.DataFrame, mov_cap: int) -> tuple[dict[int, float], float]:
    """Solve Massey-style least squares for one season.

    Parameters
    ----------
    games_df : DataFrame
        Subset of MRegularSeasonCompactResults for a single season.
        Required columns: WTeamID, LTeamID, WScore, LScore, WLoc.
    mov_cap : int
        Cap absolute score-margin contributions.

    Returns
    -------
    (ratings, h) : (dict, float)
        ratings maps TeamID -> rating; sum of ratings is 0.
        h is the home-court constant.
    """
    if mov_cap <= 0:
        raise ValueError(f"mov_cap must be positive, got {mov_cap}")

    team_ids = sorted(set(games_df["WTeamID"].tolist()) | set(games_df["LTeamID"].tolist()))
    n = len(team_ids)
    idx = {tid: i for i, tid in enumerate(team_ids)}

    # Bordered KKT system: (n+2) x (n+2)
    #   [ X^T X    e ] [ beta   ]   [ X^T y ]
    #   [   e^T    0 ] [ lambda ] = [   0   ]
    # where beta = [r_1, ..., r_n, h] and e = [1, ..., 1, 0]^T (ones in
    # the n team slots, zero in the home-constant slot).
    M = np.zeros((n + 2, n + 2), dtype=np.float64)
    rhs = np.zeros(n + 2, dtype=np.float64)

    # Constraint row/col: sum(r) = 0 (does not constrain h).
    for k in range(n):
        M[n + 1, k] = 1.0
        M[k, n + 1] = 1.0

    h_col = n  # column index of home-constant in beta

    for w, l, ws, ls, wloc in zip(
        games_df["WTeamID"].to_numpy(),
        games_df["LTeamID"].to_numpy(),
        games_df["WScore"].to_numpy(),
        games_df["LScore"].to_numpy(),
        games_df["WLoc"].to_numpy(),
    ):
        wi = idx[int(w)]
        li = idx[int(l)]
        z = 1 if wloc == "H" else (-1 if wloc == "A" else 0)
        s = int(ws) - int(ls)
        # Cap. Sign(s) is always +1 here since W beat L; keep abs(s) <= cap.
        y = min(s, mov_cap)

        # X row for this game has +1 in col wi, -1 in col li, +z in col h_col.
        # X^T X contributions:
        M[wi, wi] += 1.0
        M[li, li] += 1.0
        M[wi, li] -= 1.0
        M[li, wi] -= 1.0
        M[wi, h_col] += z
        M[h_col, wi] += z
        M[li, h_col] -= z
        M[h_col, li] -= z
        M[h_col, h_col] += z * z  # 1 if non-neutral, 0 if neutral

        # X^T y contributions:
        rhs[wi] += y
        rhs[li] -= y
        rhs[h_col] += z * y

    cond = np.linalg.cond(M)
    if cond > 1e10:
        logger.warning("Massey normal-equations matrix is ill-conditioned (cond=%.2e); "
                       "season may have a disconnected component", cond)

    sol = np.linalg.solve(M, rhs)
    ratings_arr = sol[:n]
    h_val = float(sol[n])

    return ({tid: float(ratings_arr[idx[tid]]) for tid in team_ids}, h_val)


def compute_massey_mov_ratings(
    reg_season: pd.DataFrame,
    seasons: list[int] | None = None,
    mov_cap: int = 21,
) -> pd.DataFrame:
    """Compute Massey-matrix MOV ratings per (Season, TeamID).

    Parameters
    ----------
    reg_season : DataFrame
        Kaggle MRegularSeasonCompactResults (or DetailedResults superset).
        Required columns: Season, WTeamID, LTeamID, WScore, LScore, WLoc.
    seasons : list of int or None
        Restrict to these seasons. None = all seasons present in reg_season.
    mov_cap : int
        Cap absolute score-margin (predictive Massey, default 21).

    Returns
    -------
    DataFrame with columns [Season, TeamID, massey_mov_rating],
    one row per (team, season) where the team appeared in the season's
    regular-season schedule.
    """
    required = {"Season", "WTeamID", "LTeamID", "WScore", "LScore", "WLoc"}
    missing = required - set(reg_season.columns)
    if missing:
        raise ValueError(f"reg_season missing required columns: {sorted(missing)}")

    season_iter = sorted(reg_season["Season"].unique()) if seasons is None else sorted(seasons)
    rows = []
    for season in season_iter:
        games = reg_season[reg_season["Season"] == season]
        if len(games) == 0:
            continue
        ratings, _h = _solve_one_season(games, mov_cap)
        for tid, r in ratings.items():
            rows.append({"Season": int(season), "TeamID": int(tid), "massey_mov_rating": r})

    return pd.DataFrame(rows)


def load_massey_mov_ratings(*args, **kwargs):
    """Placeholder; real implementation added in Task 5."""
    raise NotImplementedError("load_massey_mov_ratings is added in Task 5")
