"""15-cell W_UPSET / W_MISS tuning sweep over the v9-A trainer.

Grid: W_UPSET in {1.0, 1.25, 1.5, 1.75, 2.0} x W_MISS in {0.0, 0.5, 1.0}.

For each cell, run double-LOSO across 22 seasons (2003..2025), build
v9-adjusted pairwise probabilities, score with score_pairwise_path
against MNCAATourneyCompactResults.csv, and write one row to
output/v9_sweep_results.csv.

Anchor cell (1.0, 0.0) must be present in the grid -- it is the v8
reproduction sanity check.

Spec:  docs/superpowers/specs/2026-05-01-v9-weight-sweep.md
"""
import sys
from pathlib import Path
from typing import Iterable, List, Tuple

# Path setup: allow `python src/sweep_v9_weights.py` invocation.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

W_UPSET_VALUES = [1.0, 1.25, 1.5, 1.75, 2.0]
W_MISS_VALUES = [0.0, 0.5, 1.0]
GRID: List[Tuple[float, float]] = [
    (wu, wm) for wu in W_UPSET_VALUES for wm in W_MISS_VALUES
]
ANCHOR_CELL: Tuple[float, float] = (1.0, 0.0)


def validate_grid(grid: Iterable[Tuple[float, float]]) -> None:
    """Raise ValueError if the anchor cell (1.0, 0.0) is missing.

    The anchor is the v8 reproduction sanity check: at uniform weights
    the v9-A trainer should reproduce v8 within 1 bracket point. Without
    the anchor, the sweep cannot be sanity-checked.
    """
    cells = set((float(wu), float(wm)) for wu, wm in grid)
    if ANCHOR_CELL not in cells:
        raise ValueError(
            f"anchor cell {ANCHOR_CELL} missing from grid; sweep is invalid "
            "(no v8 reproduction sanity check possible)"
        )


import numpy as np
import pandas as pd

from src.train_upset_model import (
    build_v9_pairwise,
    double_loso_eval,
    load_per_game_data_with_upset,
)


def _cell_path(out_dir: str, w_upset: float, w_miss: float) -> str:
    """Per-cell pairwise CSV path: pairwise_v9_WU{u:.2f}_WM{m:.2f}.csv."""
    name = f"pairwise_v9_WU{w_upset:.2f}_WM{w_miss:.2f}.csv"
    return str(Path(out_dir) / name)


def run_single_cell(
    w_upset: float,
    w_miss: float,
    pairwise_v4_csv: str,
    results_csv: str,
    seeds_csv: str,
    out_dir: str,
) -> dict:
    """Run one (w_upset, w_miss) cell of the sweep.

    Steps:
      1. Load per-game training rows from pairwise_v4 + results + seeds.
      2. Build v9-adjusted pairwise CSV at out_dir/pairwise_v9_WU{u}_WM{m}.csv.
      3. Run per-season LOSO eval to capture log loss / accuracy.
      4. Score the pairwise CSV (best-effort: catches FileNotFoundError /
         missing slots in score_pairwise_path so unit tests with
         synthetic data work).
      5. Return dict with all metrics.
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    pairwise_csv_out = _cell_path(out_dir, w_upset, w_miss)

    per_game = load_per_game_data_with_upset(
        pairwise_v4_csv, results_csv, seeds_csv
    )

    build_v9_pairwise(
        per_game, pairwise_v4_csv, seeds_csv, pairwise_csv_out,
        w_upset=w_upset, w_miss=w_miss,
    )

    eval_df = double_loso_eval(
        per_game, w_upset=w_upset, w_miss=w_miss
    )
    if len(eval_df) > 0 and "n_games" in eval_df.columns:
        n_total = float(eval_df["n_games"].sum())
        if n_total > 0:
            ll_mean = float(
                (eval_df["ll_v9"] * eval_df["n_games"]).sum() / n_total
            )
            acc_mean = float(
                (eval_df["acc_v9"] * eval_df["n_games"]).sum() / n_total
            )
        else:
            ll_mean = float("nan")
            acc_mean = float("nan")
    else:
        ll_mean = float("nan")
        acc_mean = float("nan")

    # Bracket scoring: tolerate missing tournament slot data on synthetic
    # inputs (unit tests). Production runs with real Kaggle data will
    # produce meaningful totals.
    try:
        from src.score_chalk_brackets import score_pairwise_path
        scored = score_pairwise_path(pairwise_csv_out)
        total_pts = float(scored["total_pts"])
    except (FileNotFoundError, KeyError, ValueError):
        total_pts = 0.0

    return {
        "w_upset": float(w_upset),
        "w_miss": float(w_miss),
        "total_brkt_pts": total_pts,
        "ll_loso_weighted_mean": ll_mean,
        "acc_loso_weighted_mean": acc_mean,
        "pairwise_csv": pairwise_csv_out,
    }
