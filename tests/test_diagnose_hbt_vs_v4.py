"""Unit tests for src/diagnose_hbt_vs_v4.py.

Tests the per-cell scoring (residual correlation + ideal-weight search +
clause flags), the best-passing-cell selection logic, and the sigma-cell
filename discovery. Real-data sweep is exercised in Phase 3b.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.diagnose_hbt_vs_v4 import (
    GATE_HEADROOM_MIN,
    GATE_R_MAX,
    GATE_W_HIGH,
    GATE_W_LOW,
    discover_sigma_cells,
    pick_best_passing_cell,
    score_cell_from_files,
    score_one_cell,
)


def test_score_one_cell_perfect_correlation_fails_clause_r():
    # p_v4 and p_hbt are linearly related -> residuals perfectly correlated
    rng = np.random.default_rng(0)
    p_v4 = rng.uniform(0.5, 0.99, size=200)
    p_hbt = p_v4 - 0.05  # shifted but linearly tied
    cell = score_one_cell(p_v4, p_hbt)
    assert cell["r"] > 0.99
    assert cell["passes_r"] is False


def test_score_one_cell_independent_arrays():
    rng = np.random.default_rng(1)
    p_v4 = rng.uniform(0.5, 0.95, size=500)
    p_hbt = rng.uniform(0.5, 0.95, size=500)
    cell = score_one_cell(p_v4, p_hbt)
    # Independent arrays -> residual correlation near 0.
    assert abs(cell["r"]) < 0.20
    assert cell["passes_r"] is True


def test_score_one_cell_returns_required_fields():
    p_v4 = np.array([0.8, 0.6, 0.7, 0.9])
    p_hbt = np.array([0.7, 0.5, 0.6, 0.8])
    cell = score_one_cell(p_v4, p_hbt)
    required = {
        "n_games", "ll_v4", "ll_hbt", "acc_v4", "acc_hbt", "r",
        "disagree_n", "w_opt", "ll_blend", "headroom",
        "passes_r", "passes_w", "passes_headroom", "passes_all",
    }
    assert required.issubset(cell.keys())


def test_pick_best_passing_cell_chooses_max_headroom():
    cells = [
        {"sigma": 0.1, "r": 0.5, "w_opt": 0.5, "headroom": 0.003,
         "passes_r": True, "passes_w": True, "passes_headroom": False,
         "passes_all": False},
        {"sigma": 1.0, "r": 0.55, "w_opt": 0.6, "headroom": 0.010,
         "passes_r": True, "passes_w": True, "passes_headroom": True,
         "passes_all": True},
        {"sigma": 2.0, "r": 0.59, "w_opt": 0.7, "headroom": 0.015,
         "passes_r": True, "passes_w": True, "passes_headroom": True,
         "passes_all": True},
        {"sigma": 5.0, "r": 0.55, "w_opt": 0.95, "headroom": 0.0,
         "passes_r": True, "passes_w": False, "passes_headroom": False,
         "passes_all": False},
    ]
    best = pick_best_passing_cell(cells)
    assert best["sigma"] == 2.0  # max headroom among passing


def test_pick_best_passing_cell_returns_none_when_all_fail():
    cells = [
        {"sigma": 0.1, "passes_all": False},
        {"sigma": 1.0, "passes_all": False},
    ]
    assert pick_best_passing_cell(cells) is None


def test_discover_sigma_cells_extracts_sorted_floats(tmp_path):
    for s in ["1.00", "0.05", "5.00", "0.50"]:
        (tmp_path / f"pairwise_hbt_sigma_{s}.csv").write_text("")
    # Add a non-matching file
    (tmp_path / "pairwise_other.csv").write_text("")

    cells = discover_sigma_cells(str(tmp_path / "pairwise_hbt_sigma_*.csv"))
    sigmas = [s for s, _ in cells]
    assert sigmas == [0.05, 0.5, 1.0, 5.0]


def test_score_cell_from_files_end_to_end(tmp_path):
    # Build small pairwise CSVs + a tournament results frame.
    v4_csv = tmp_path / "pairwise_v4.csv"
    hbt_csv = tmp_path / "pairwise_hbt_sigma_1.00.csv"

    pd.DataFrame([
        {"season": 2024, "team_a": 1, "team_b": 2, "p_a_wins": 0.7},
        {"season": 2024, "team_a": 2, "team_b": 3, "p_a_wins": 0.4},
    ]).to_csv(v4_csv, index=False)

    pd.DataFrame([
        {"season": 2024, "team_a": 1, "team_b": 2, "p_a_wins": 0.6},
        {"season": 2024, "team_a": 2, "team_b": 3, "p_a_wins": 0.5},
    ]).to_csv(hbt_csv, index=False)

    results = pd.DataFrame([
        # Game 1: team 1 beats team 2. v4 says p(1 beats 2)=0.7 -> winner prob 0.7.
        # hbt says 0.6.
        {"Season": 2024, "WTeamID": 1, "LTeamID": 2, "DayNum": 1},
        # Game 2: team 3 beats team 2. v4 says p(2 beats 3)=0.4 -> p(3 beats 2)=0.6.
        # hbt says 0.5.
        {"Season": 2024, "WTeamID": 3, "LTeamID": 2, "DayNum": 2},
    ])

    cell = score_cell_from_files(1.0, str(v4_csv), str(hbt_csv), results)
    assert cell["n_games"] == 2
    assert cell["sigma"] == 1.0
    assert abs(cell["ll_v4"] - (-(np.log(0.7) + np.log(0.6)) / 2)) < 1e-9
    assert abs(cell["ll_hbt"] - (-(np.log(0.6) + np.log(0.5)) / 2)) < 1e-9


def test_thresholds_match_plain_bt():
    """Regression guard: gate thresholds must match plain-BT exactly so
    cells across experiments are directly comparable."""
    from src.diagnose_bt_vs_v4 import (
        GATE_HEADROOM_MIN as PLAIN_HM,
        GATE_R_MAX as PLAIN_R,
        GATE_W_HIGH as PLAIN_W_HIGH,
        GATE_W_LOW as PLAIN_W_LOW,
    )
    assert GATE_R_MAX == PLAIN_R
    assert GATE_W_LOW == PLAIN_W_LOW
    assert GATE_W_HIGH == PLAIN_W_HIGH
    assert GATE_HEADROOM_MIN == PLAIN_HM
