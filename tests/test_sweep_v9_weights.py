"""Tests for src/sweep_v9_weights.py (15-cell W_UPSET / W_MISS sweep)."""

import pytest

from src.sweep_v9_weights import GRID, validate_grid


def test_grid_contains_anchor_cell():
    """The anchor cell (W_UPSET=1.0, W_MISS=0.0) MUST be in the grid -- it
    is the v8 reproduction sanity check.
    """
    assert (1.0, 0.0) in GRID


def test_grid_has_15_unique_cells():
    """Spec calls for 5 * 3 = 15 cells."""
    assert len(GRID) == 15
    assert len(set(GRID)) == 15


def test_validate_grid_passes_with_anchor_cell():
    """validate_grid raises iff anchor is missing."""
    validate_grid([(1.0, 0.0), (1.5, 1.0)])  # contains anchor; should not raise


def test_validate_grid_raises_without_anchor_cell():
    with pytest.raises(ValueError, match="anchor cell"):
        validate_grid([(1.5, 1.0), (2.0, 0.0)])


import numpy as np
import pandas as pd

from src.sweep_v9_weights import run_single_cell


def _write_minimal_inputs(tmp_path):
    """Two seasons, two pairs each, with seeds and per-game results."""
    pw_v4 = pd.DataFrame({
        "season": [2022, 2022, 2023, 2023],
        "team_a": [1, 1, 1, 2],
        "team_b": [2, 3, 3, 3],
        "p_a_wins": [0.7, 0.6, 0.55, 0.45],
    })
    pw_path = tmp_path / "pairwise_v4.csv"
    pw_v4.to_csv(pw_path, index=False)

    seeds = pd.DataFrame({
        "Season": [2022, 2022, 2022, 2023, 2023, 2023],
        "Seed":   ["W01", "W08", "W16", "W01", "W08", "W16"],
        "TeamID": [1, 2, 3, 1, 2, 3],
    })
    seeds_path = tmp_path / "seeds.csv"
    seeds.to_csv(seeds_path, index=False)

    results = pd.DataFrame({
        "Season": [2022, 2022, 2023, 2023],
        "DayNum": [136, 138, 136, 138],
        "WTeamID": [1, 1, 1, 2],
        "WScore": [70, 75, 65, 70],
        "LTeamID": [2, 3, 3, 3],
        "LScore": [60, 65, 60, 65],
    })
    results_path = tmp_path / "results.csv"
    results.to_csv(results_path, index=False)

    return str(pw_path), str(seeds_path), str(results_path)


def test_run_single_cell_writes_pairwise_and_returns_metrics(tmp_path):
    """run_single_cell writes a v8-compatible pairwise CSV at the
    expected path and returns a dict with weight + scoring keys.
    """
    pw_path, seeds_path, results_path = _write_minimal_inputs(tmp_path)
    out_dir = tmp_path / "v9_sweep"

    metrics = run_single_cell(
        w_upset=1.0, w_miss=0.0,
        pairwise_v4_csv=pw_path,
        results_csv=results_path,
        seeds_csv=seeds_path,
        out_dir=str(out_dir),
    )

    # Pairwise CSV exists and has the expected schema.
    pw_path_out = out_dir / "pairwise_v9_WU1.00_WM0.00.csv"
    assert pw_path_out.exists()
    out = pd.read_csv(pw_path_out)
    assert list(out.columns) == ["season", "team_a", "team_b", "p_a_wins"]
    assert (out["team_a"] < out["team_b"]).all()

    # Returned metrics have all required fields.
    assert set(metrics.keys()) >= {
        "w_upset", "w_miss",
        "total_brkt_pts", "ll_loso_weighted_mean", "acc_loso_weighted_mean",
        "pairwise_csv",
    }
    assert metrics["w_upset"] == 1.0
    assert metrics["w_miss"] == 0.0
    assert metrics["pairwise_csv"] == str(pw_path_out)
    # 2-season synthetic data: total_brkt_pts may be zero or positive
    # (no full bracket, just R64 stubs); just assert it's a float.
    assert isinstance(metrics["total_brkt_pts"], float)
