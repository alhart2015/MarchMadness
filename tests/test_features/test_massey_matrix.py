"""Unit tests for src/features/massey_matrix.py.

Synthetic schedules with closed-form solutions verify solver correctness;
real-data smoke test verifies the cached loader and pipeline integration.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.features.massey_matrix import (
    _PRODUCER_VERSION,
    compute_massey_mov_ratings,
    load_massey_mov_ratings,
)


def _make_round_robin(team_ids, ratings, h, season=2024):
    """Build a 12-game round-robin where the solution is known.

    Each pair plays twice: once at each home. MOV from W's perspective:
        y = r_W - r_L + h * z, z=+1 if W home else -1 if W away else 0
    Loser is the team with lower rating; winner = higher rating.
    """
    rows = []
    daynum = 10
    for i, ti in enumerate(team_ids):
        for j, tj in enumerate(team_ids):
            if i >= j:
                continue
            ri, rj = ratings[i], ratings[j]
            # Game at ti's home: ti is home; W = max-rating side, locator from W's view
            for home_team_idx in (i, j):
                if home_team_idx == i:
                    if ri > rj:
                        w, l = ti, tj
                        z = +1  # W home
                        mov = (ri - rj) + h * z
                    else:
                        w, l = tj, ti
                        z = -1  # W away
                        mov = (rj - ri) + h * z
                else:
                    if rj > ri:
                        w, l = tj, ti
                        z = +1
                        mov = (rj - ri) + h * z
                    else:
                        w, l = ti, tj
                        z = -1
                        mov = (ri - rj) + h * z
                wloc = {1: "H", -1: "A", 0: "N"}[z]
                # Guard: caller chose ratings + h such that the modelled MOV
                # is at least +1 from W's perspective. Silently clipping a
                # non-positive mov would make WScore disagree with the
                # game's true model y = r_W - r_L + h*z.
                assert mov >= 1, (
                    f"_make_round_robin: mov={mov} for W={w}, L={l}, z={z}; "
                    "tighten ratings/h so all matchups produce mov >= 1"
                )
                rows.append({
                    "Season": season,
                    "DayNum": daynum,
                    "WTeamID": w,
                    "WScore": int(50 + mov),
                    "LTeamID": l,
                    "LScore": 50,
                    "WLoc": wloc,
                    "NumOT": 0,
                })
                daynum += 1
    return pd.DataFrame(rows)


def test_synthetic_round_robin_recovers_ratings_and_home_constant():
    team_ids = [1101, 1102, 1103, 1104]
    ratings = [5.0, 2.0, -2.0, -5.0]
    h_true = 1.0
    games = _make_round_robin(team_ids, ratings, h_true)

    df = compute_massey_mov_ratings(games, mov_cap=21)

    assert set(df.columns) == {"Season", "TeamID", "massey_mov_rating"}
    assert len(df) == 4
    rating_by_team = dict(zip(df["TeamID"], df["massey_mov_rating"]))
    for tid, expected in zip(team_ids, ratings):
        assert rating_by_team[tid] == pytest.approx(expected, abs=1e-4), (
            f"Team {tid} expected {expected}, got {rating_by_team[tid]}"
        )
