"""Unit tests for src/features/colley_matrix.py.

Synthetic round-robin with closed-form Colley solution verifies solver
correctness; real-data smoke test verifies cached loader + sum-to-(n/2)
invariant on actual MRegularSeasonCompactResults.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.features.colley_matrix import (
    _PRODUCER_VERSION,
    compute_colley_ratings,
    load_colley_ratings,
)


def _make_round_robin_wins(team_ids, win_counts, season=2024):
    """Build a round-robin where each pair plays twice, with the winner of
    each pair determined by which team has higher win_counts. For
    win_counts = [6, 4, 2, 0] over 4 teams, this yields A: 6-0, B: 4-2,
    C: 2-4, D: 0-6.

    For pairs of equal win_counts, the games split 1-1.
    """
    rows = []
    daynum = 10
    n = len(team_ids)
    for i in range(n):
        for j in range(i + 1, n):
            wi, wj = win_counts[i], win_counts[j]
            if wi > wj:
                games = [(team_ids[i], team_ids[j]), (team_ids[i], team_ids[j])]
            elif wj > wi:
                games = [(team_ids[j], team_ids[i]), (team_ids[j], team_ids[i])]
            else:
                games = [(team_ids[i], team_ids[j]), (team_ids[j], team_ids[i])]
            for w, l in games:
                rows.append({
                    "Season": season,
                    "DayNum": daynum,
                    "WTeamID": w, "WScore": 75,
                    "LTeamID": l, "LScore": 70,
                    "WLoc": "N", "NumOT": 0,
                })
                daynum += 1
    return pd.DataFrame(rows)


def test_synthetic_round_robin_recovers_colley_ratings():
    """4-team round-robin (each pair plays twice) with W/L = 6-0, 4-2,
    2-4, 0-6 yields closed-form ratings (0.8, 0.6, 0.4, 0.2).

    Derivation: C = (T+2)*I - A where T=6, off-diagonal A_ij = 2
    (each pair plays twice). So C = 10*I - 2*J + 2*I = 10*I - 2*J in
    practice (the +2 prior is folded into the diagonal). Eigenvalue
    along the ones direction is 10 - 2*n = 2; along perp directions is
    10. b = [4, 2, 0, -2] = 1*ones + [3, 1, -1, -3]. Solution
    x = 0.5*ones + (1/10)*[3, 1, -1, -3] = [0.8, 0.6, 0.4, 0.2].
    Sum = 2 = n/2."""
    team_ids = [1101, 1102, 1103, 1104]
    win_counts = [6, 4, 2, 0]
    expected = {1101: 0.8, 1102: 0.6, 1103: 0.4, 1104: 0.2}
    games = _make_round_robin_wins(team_ids, win_counts)

    df = compute_colley_ratings(games)
    assert set(df.columns) == {"Season", "TeamID", "colley_rating"}
    assert len(df) == 4

    rating_by_team = dict(zip(df["TeamID"], df["colley_rating"]))
    for tid, expected_r in expected.items():
        assert rating_by_team[tid] == pytest.approx(expected_r, abs=1e-6), (
            f"Team {tid} expected {expected_r}, got {rating_by_team[tid]}"
        )
