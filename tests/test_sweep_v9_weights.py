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

    slots = pd.DataFrame({
        "Season": [2022, 2022, 2023, 2023],
        "Slot":   ["R1W1", "R2W1", "R1W1", "R2W1"],
        "StrongSeed": ["W01", "R1W1", "W01", "R1W1"],
        "WeakSeed":   ["W08", "W16",  "W08", "W16"],
    })
    slots_path = tmp_path / "slots.csv"
    slots.to_csv(slots_path, index=False)

    return str(pw_path), str(seeds_path), str(results_path), str(slots_path)


def test_run_single_cell_writes_pairwise_and_returns_metrics(tmp_path):
    """run_single_cell writes a v8-compatible pairwise CSV at the
    expected path and returns a dict with weight + scoring keys.
    """
    pw_path, seeds_path, results_path, slots_path = _write_minimal_inputs(tmp_path)
    out_dir = tmp_path / "v9_sweep"

    metrics = run_single_cell(
        w_upset=1.0, w_miss=0.0,
        pairwise_v4_csv=pw_path,
        results_csv=results_path,
        seeds_csv=seeds_path,
        out_dir=str(out_dir),
        slots_csv=slots_path,
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


def test_run_sweep_writes_results_csv(tmp_path):
    """run_sweep over a 2-cell mini-grid (anchor + one more) writes a
    results CSV with one row per cell and the expected columns.
    """
    pw_path, seeds_path, results_path, slots_path = _write_minimal_inputs(tmp_path)
    out_dir = tmp_path / "v9_sweep"
    results_csv_out = tmp_path / "v9_sweep_results.csv"

    from src.sweep_v9_weights import run_sweep

    mini_grid = [(1.0, 0.0), (1.5, 0.5)]

    run_sweep(
        grid=mini_grid,
        pairwise_v4_csv=pw_path,
        results_csv=results_path,
        seeds_csv=seeds_path,
        out_dir=str(out_dir),
        results_csv_path=str(results_csv_out),
        slots_csv=slots_path,
    )

    assert results_csv_out.exists()
    df = pd.read_csv(results_csv_out)
    assert len(df) == 2
    assert set(df.columns) >= {
        "w_upset", "w_miss",
        "total_brkt_pts",
        "ll_loso_weighted_mean", "acc_loso_weighted_mean",
        "pairwise_csv",
    }
    # Sorted by total_brkt_pts descending (non-increasing).
    assert df["total_brkt_pts"].is_monotonic_decreasing


def test_run_sweep_halts_without_anchor_cell(tmp_path):
    """run_sweep refuses to start if the anchor cell is missing."""
    pw_path, seeds_path, results_path, slots_path = _write_minimal_inputs(tmp_path)
    from src.sweep_v9_weights import run_sweep

    bad_grid = [(1.5, 0.5), (2.0, 1.0)]  # no (1.0, 0.0)

    with pytest.raises(ValueError, match="anchor cell"):
        run_sweep(
            grid=bad_grid,
            pairwise_v4_csv=pw_path,
            results_csv=results_path,
            seeds_csv=seeds_path,
            out_dir=str(tmp_path / "v9_sweep"),
            results_csv_path=str(tmp_path / "results.csv"),
            slots_csv=slots_path,
        )


def test_run_single_cell_v9c_writes_pairwise(tmp_path):
    """run_single_cell with feature_set='v9c' writes a pairwise CSV at the
    same path template and returns metrics dict with the same keys as v9-B.
    """
    pw_path, seeds_path, results_path, slots_path = _write_minimal_inputs(tmp_path)
    out_dir = tmp_path / "v9c_sweep"

    metrics = run_single_cell(
        w_upset=1.0, w_miss=0.0,
        pairwise_v4_csv=pw_path,
        results_csv=results_path,
        seeds_csv=seeds_path,
        out_dir=str(out_dir),
        slots_csv=slots_path,
        feature_set="v9c",
    )

    pw_path_out = out_dir / "pairwise_v9_WU1.00_WM0.00.csv"
    assert pw_path_out.exists()
    assert metrics["w_upset"] == 1.0
    assert metrics["w_miss"] == 0.0
    assert metrics["pairwise_csv"] == str(pw_path_out)


def _write_pairwise_bt(path, rows):
    """rows: list of (season, team_a, team_b, p_a_wins)."""
    df = pd.DataFrame(rows, columns=["season", "team_a", "team_b", "p_a_wins"])
    df.to_csv(path, index=False)


def test_run_single_cell_v9d_writes_pairwise(tmp_path):
    """run_single_cell with feature_set='v9d' + pairwise_bt_csv writes
    a pairwise CSV in the canonical schema and returns the metrics dict.
    """
    pw_path, seeds_path, results_path, slots_path = _write_minimal_inputs(tmp_path)
    pw_bt_path = tmp_path / "pw_bt.csv"
    # Match the (season, team_a, team_b) keys present in pw_v4 fixture.
    _write_pairwise_bt(pw_bt_path, [
        (2022, 1, 2, 0.6), (2022, 1, 3, 0.55),
        (2023, 1, 3, 0.45), (2023, 2, 3, 0.4),
    ])
    out_dir = tmp_path / "v9d_sweep"

    metrics = run_single_cell(
        w_upset=1.0, w_miss=0.0,
        pairwise_v4_csv=pw_path,
        results_csv=results_path,
        seeds_csv=seeds_path,
        out_dir=str(out_dir),
        slots_csv=slots_path,
        feature_set="v9d",
        pairwise_bt_csv=str(pw_bt_path),
    )

    pw_path_out = out_dir / "pairwise_v9_WU1.00_WM0.00.csv"
    assert pw_path_out.exists()
    out = pd.read_csv(pw_path_out)
    assert list(out.columns) == ["season", "team_a", "team_b", "p_a_wins"]
    assert (out["team_a"] < out["team_b"]).all()
    assert metrics["pairwise_csv"] == str(pw_path_out)


def test_run_single_cell_v9d_requires_pairwise_bt_csv(tmp_path):
    """feature_set='v9d' without pairwise_bt_csv raises before any work."""
    pw_path, seeds_path, results_path, slots_path = _write_minimal_inputs(tmp_path)
    out_dir = tmp_path / "v9d_sweep"

    with pytest.raises(ValueError, match="pairwise_bt_csv"):
        run_single_cell(
            w_upset=1.0, w_miss=0.0,
            pairwise_v4_csv=pw_path,
            results_csv=results_path,
            seeds_csv=seeds_path,
            out_dir=str(out_dir),
            slots_csv=slots_path,
            feature_set="v9d",
        )


def test_sweep_main_uses_v9_stage1_pairwise_env_var(tmp_path, monkeypatch):
    """When V9_STAGE1_PAIRWISE is set, main() uses that path as the
    stage-1 input and writes outputs to a basename-keyed dir.

    We don't run the full sweep here -- we monkey-patch run_sweep to
    capture its arguments and assert the env-var value flowed through.
    """
    from src import sweep_v9_weights as sw

    captured = {}

    def fake_run_sweep(**kwargs):
        captured.update(kwargs)
        # Return a minimal df so main() doesn't crash on the score-anchor block.
        return pd.DataFrame([{
            "w_upset": 1.0, "w_miss": 0.0, "total_brkt_pts": 100.0,
            "ll_loso_weighted_mean": 0.5, "acc_loso_weighted_mean": 0.5,
            "pairwise_csv": str(tmp_path / "fake.csv"),
        }])

    monkeypatch.setattr(sw, "run_sweep", fake_run_sweep)
    monkeypatch.setattr(
        sw, "score_pairwise_path",
        lambda *a, **k: {"total_pts": 100.0},
        raising=False,
    )
    monkeypatch.setenv("V9_FEATURE_SET", "v9c")

    pw_override = tmp_path / "pairwise_ensemble_e1.csv"
    pw_override.write_text("season,team_a,team_b,p_a_wins\n")
    monkeypatch.setenv("V9_STAGE1_PAIRWISE", str(pw_override))

    try:
        sw.main()
    except Exception:
        # The fake doesn't fully simulate downstream IO; we only care that
        # run_sweep was called with the override path.
        pass

    assert captured.get("pairwise_v4_csv") == str(pw_override)
    # Output dir basename incorporates the override filename.
    assert "ensemble_e1" in captured.get("out_dir", "")


def test_sweep_default_path_when_env_var_unset(monkeypatch):
    """When V9_STAGE1_PAIRWISE is unset, main() uses output/pairwise_v4.csv."""
    from src import sweep_v9_weights as sw

    captured = {}

    def fake_run_sweep(**kwargs):
        captured.update(kwargs)
        return pd.DataFrame([{
            "w_upset": 1.0, "w_miss": 0.0, "total_brkt_pts": 100.0,
            "ll_loso_weighted_mean": 0.5, "acc_loso_weighted_mean": 0.5,
            "pairwise_csv": "fake.csv",
        }])

    monkeypatch.setattr(sw, "run_sweep", fake_run_sweep)
    monkeypatch.setattr(
        sw, "score_pairwise_path",
        lambda *a, **k: {"total_pts": 100.0},
        raising=False,
    )
    monkeypatch.setenv("V9_FEATURE_SET", "v9c")
    monkeypatch.delenv("V9_STAGE1_PAIRWISE", raising=False)

    try:
        sw.main()
    except Exception:
        pass

    assert captured.get("pairwise_v4_csv") == "output/pairwise_v4.csv"
