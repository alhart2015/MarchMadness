"""Smoke tests for the Phase 2 LOSO sweep CLI driver.

Two tiers:

  - Fast monkeypatched tests stub out ``run_phase2_one_holdout`` and confirm
    the CLI driver wires arguments, emits all four output files, and writes
    pairwise rows in the v4 shape (``season,team_a,team_b,p_a_wins``;
    ``team_a < team_b``).

  - A slow real-data smoke test exercises the full stack on one holdout
    season. Marked ``@pytest.mark.slow`` so it skips by default and runs
    only when explicitly requested via ``pytest -m slow``.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest


def _stub_run_phase2_one_holdout_factory(seasons_seen: list[int]):
    """Build a monkeypatched stub that records the seasons it's called for
    and returns a synthetic dict shaped like the real function's output."""

    def _stub(data_dir, holdout_season, seasons, *, emit_pairwise=False, **_kwargs):
        seasons_seen.append(int(holdout_season))
        # Three teams -> 3 pairs, all team_a < team_b.
        if emit_pairwise:
            pdf = pd.DataFrame({
                "team_a": [1101, 1101, 1102],
                "team_b": [1102, 1103, 1103],
                "p_a_wins": [0.55, 0.40, 0.62],
            })
        else:
            pdf = None
        result = {
            "holdout_season": int(holdout_season),
            "gnn": {"ll": 0.55, "accuracy": 0.70, "n": 67},
            "predictions": [],
            "train_minutes": 0.001,
            "epochs_run": 3,
            "best_epoch": 2,
            "best_val_ll": 0.54,
        }
        if pdf is not None:
            result["pairwise_df"] = pdf
        return result

    return _stub


def test_cli_driver_help_runs_cleanly():
    """argparse --help exits 0 and lists --holdout-seasons."""
    from src.run_gnn_phase2 import main

    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0


def test_cli_driver_default_holdouts_are_22_seasons():
    """DEFAULT_HOLDOUTS = 2003-2025 minus 2020 -> 22 seasons."""
    from src.run_gnn_phase2 import DEFAULT_HOLDOUTS, DEFAULT_SEASONS

    assert len(DEFAULT_HOLDOUTS) == 22
    assert 2020 not in DEFAULT_HOLDOUTS
    assert min(DEFAULT_HOLDOUTS) == 2003
    assert max(DEFAULT_HOLDOUTS) == 2025
    # Default training pool matches the holdouts (every holdout sees the
    # other 21 seasons as training data).
    assert DEFAULT_SEASONS == DEFAULT_HOLDOUTS


def test_cli_driver_writes_all_outputs_with_stub(tmp_path, monkeypatch):
    """End-to-end: stub the inner driver, run main(), assert all 4 outputs exist."""
    import src.run_gnn_phase2 as mod

    seasons_seen: list[int] = []
    monkeypatch.setattr(
        mod,
        "run_phase2_one_holdout",
        _stub_run_phase2_one_holdout_factory(seasons_seen),
    )

    output_dir = tmp_path / "out"
    rc = mod.main([
        "--holdout-seasons", "2023,2024",
        "--seasons", "2022,2023,2024,2025",
        "--epochs", "3",
        "--output-dir", str(output_dir),
    ])
    assert rc == 0
    assert seasons_seen == [2023, 2024]

    pairwise_csv = output_dir / "pairwise_gnn_phase2.csv"
    per_holdout_json = output_dir / "gnn_phase2_loso_per_holdout.json"
    summary_json = output_dir / "gnn_phase2_loso_summary.json"
    log_path = output_dir / "gnn_phase2_loso_run.log"

    assert pairwise_csv.exists()
    assert per_holdout_json.exists()
    assert summary_json.exists()
    assert log_path.exists()

    # Pairwise CSV shape matches output/pairwise_v4.csv.
    pdf = pd.read_csv(pairwise_csv)
    assert list(pdf.columns) == ["season", "team_a", "team_b", "p_a_wins"]
    assert (pdf["team_a"] < pdf["team_b"]).all()
    # 2 holdouts * 3 stubbed pairs = 6 rows.
    assert len(pdf) == 6
    assert set(pdf["season"]) == {2023, 2024}

    # Per-holdout JSON has both holdouts; predictions field stripped.
    per_holdout = json.loads(per_holdout_json.read_text())
    assert len(per_holdout) == 2
    for r in per_holdout:
        assert "predictions" not in r
        assert "gnn" in r and {"ll", "accuracy", "n"} <= r["gnn"].keys()

    # Summary aggregates correctly.
    summary = json.loads(summary_json.read_text())
    assert summary["n_holdouts"] == 2
    assert summary["total_test_games"] == 67 * 2
    assert "weighted_mean_ll" in summary
    assert "wall_clock_minutes" in summary


def test_cli_driver_clears_stale_pairwise_csv(tmp_path, monkeypatch):
    """Stale pairwise_gnn_phase2.csv from a prior run is removed before append."""
    import src.run_gnn_phase2 as mod

    output_dir = tmp_path / "out"
    output_dir.mkdir()
    stale = output_dir / "pairwise_gnn_phase2.csv"
    stale.write_text("season,team_a,team_b,p_a_wins\n9999,1,2,0.5\n")

    monkeypatch.setattr(
        mod,
        "run_phase2_one_holdout",
        _stub_run_phase2_one_holdout_factory([]),
    )
    rc = mod.main([
        "--holdout-seasons", "2024",
        "--seasons", "2023,2024",
        "--output-dir", str(output_dir),
    ])
    assert rc == 0
    pdf = pd.read_csv(stale)
    # Stale row (season 9999) is gone.
    assert 9999 not in set(pdf["season"])


@pytest.mark.slow
def test_run_phase2_smoke_2024(tmp_path):
    """Real-data smoke: single holdout season; outputs exist + train_minutes < 60."""
    from src.run_gnn_phase2 import main

    output_dir = tmp_path / "output"
    rc = main([
        "--holdout-seasons", "2024",
        "--seasons", "2022,2023,2024,2025",
        "--epochs", "10",
        "--output-dir", str(output_dir),
    ])
    assert rc == 0
    assert (output_dir / "pairwise_gnn_phase2.csv").exists()
    assert (output_dir / "gnn_phase2_loso_summary.json").exists()
    assert (output_dir / "gnn_phase2_loso_per_holdout.json").exists()
    assert (output_dir / "gnn_phase2_loso_run.log").exists()

    summary = json.loads(
        (output_dir / "gnn_phase2_loso_summary.json").read_text()
    )
    assert summary["max_train_minutes"] < 60.0
    assert summary["n_holdouts"] == 1

    pdf = pd.read_csv(output_dir / "pairwise_gnn_phase2.csv")
    assert list(pdf.columns) == ["season", "team_a", "team_b", "p_a_wins"]
    assert (pdf["team_a"] < pdf["team_b"]).all()
    assert (pdf["season"] == 2024).all()
