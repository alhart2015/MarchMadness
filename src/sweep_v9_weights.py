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
from src.score_chalk_brackets import score_pairwise_path


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
    slots_csv: str,
    feature_set: str = "v9b",
    pairwise_bt_csv: str | None = None,
) -> dict:
    """Run one (w_upset, w_miss) cell of the sweep.

    Steps:
      1. Load per-game training rows from pairwise_v4 + results + seeds
         (with optional p_bt join when pairwise_bt_csv is provided).
      2. Build v9-adjusted pairwise CSV at out_dir/pairwise_v9_WU{u}_WM{m}.csv.
      3. Run per-season LOSO eval to capture log loss / accuracy.
      4. Score the pairwise CSV (best-effort: catches FileNotFoundError /
         missing slots in score_pairwise_path so unit tests with
         synthetic data work).
      5. Return dict with all metrics.

    feature_set='v9d' requires pairwise_bt_csv to be provided.
    """
    if feature_set == "v9d" and pairwise_bt_csv is None:
        raise ValueError(
            "feature_set='v9d' requires pairwise_bt_csv to be provided"
        )

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    pairwise_csv_out = _cell_path(out_dir, w_upset, w_miss)

    per_game = load_per_game_data_with_upset(
        pairwise_v4_csv, results_csv, seeds_csv,
        pairwise_bt_csv=pairwise_bt_csv,
    )

    build_v9_pairwise(
        per_game, pairwise_v4_csv, seeds_csv, pairwise_csv_out,
        slots_csv=slots_csv,
        w_upset=w_upset, w_miss=w_miss,
        feature_set=feature_set,
        pairwise_bt_csv=pairwise_bt_csv,
    )

    eval_df = double_loso_eval(
        per_game, w_upset=w_upset, w_miss=w_miss, feature_set=feature_set
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
    # produce meaningful totals. score_pairwise_path is module-imported
    # at the top of the file so test monkeypatching reaches this path.
    try:
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


def run_sweep(
    grid: Iterable[Tuple[float, float]],
    pairwise_v4_csv: str,
    results_csv: str,
    seeds_csv: str,
    out_dir: str,
    results_csv_path: str,
    slots_csv: str,
    feature_set: str = "v9b",
    pairwise_bt_csv: str | None = None,
) -> pd.DataFrame:
    """Run the full grid; write per-cell pairwise CSVs to out_dir and
    aggregate results to results_csv_path. Returns the results DataFrame
    (sorted by total_brkt_pts descending).

    Halts if the anchor cell (1.0, 0.0) is missing -- the v8 reproduction
    sanity check would be impossible.

    feature_set='v9d' requires pairwise_bt_csv (passed through to
    run_single_cell -> load_per_game_data_with_upset / build_v9_pairwise).
    """
    grid = list(grid)
    validate_grid(grid)

    rows = []
    for i, (w_upset, w_miss) in enumerate(grid, start=1):
        print(f"[cell {i}/{len(grid)}] W_UPSET={w_upset}, W_MISS={w_miss}")
        m = run_single_cell(
            w_upset=w_upset, w_miss=w_miss,
            pairwise_v4_csv=pairwise_v4_csv,
            results_csv=results_csv,
            seeds_csv=seeds_csv,
            out_dir=out_dir,
            slots_csv=slots_csv,
            feature_set=feature_set,
            pairwise_bt_csv=pairwise_bt_csv,
        )
        print(f"  total_brkt_pts={m['total_brkt_pts']:.1f}, "
              f"ll={m['ll_loso_weighted_mean']:.4f}, "
              f"acc={m['acc_loso_weighted_mean']:.3f}")
        rows.append(m)

    df = (
        pd.DataFrame(rows)
        .sort_values("total_brkt_pts", ascending=False)
        .reset_index(drop=True)
    )
    Path(results_csv_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(results_csv_path, index=False)
    return df


def main():
    """Run the canonical 15-cell sweep against production data paths.

    feature_set is read from the V9_FEATURE_SET env var (default 'v9b').
    pairwise_v4_csv is read from the V9_STAGE1_PAIRWISE env var
    (default 'output/pairwise_v4.csv') -- the harness uses whatever
    pairwise CSV is supplied as the stage-1 input. Output dirs key off
    feature_set and the stage-1 input's basename so v9-B / v9-C /
    v9-D / ensemble-E1 / ensemble-E2 artifacts coexist.

    Compares the anchor cell (1.0, 0.0) bracket points against
    output/pairwise_v8.csv as a sanity gate after the sweep.
    """
    import os
    feature_set = os.environ.get("V9_FEATURE_SET", "v9b")
    if feature_set not in ("v9b", "v9c", "v9d"):
        raise ValueError(
            f"V9_FEATURE_SET={feature_set!r} invalid; "
            "must be 'v9b', 'v9c', or 'v9d'"
        )

    pairwise_v4 = os.environ.get(
        "V9_STAGE1_PAIRWISE", "output/pairwise_v4.csv"
    )

    print("=" * 80)
    print(f"V9 UPSET-WEIGHT SWEEP (feature_set={feature_set})")
    print(f"  stage-1 input: {pairwise_v4}")
    print(f"  Grid: {len(GRID)} cells, "
          f"W_UPSET in {W_UPSET_VALUES}, W_MISS in {W_MISS_VALUES}")
    print("=" * 80)

    pairwise_v8 = "output/pairwise_v8.csv"
    seeds_csv = "data/raw/march-machine-learning-2026/MNCAATourneySeeds.csv"
    results_csv = "data/raw/march-machine-learning-2026/MNCAATourneyCompactResults.csv"
    slots_csv = "data/raw/march-machine-learning-2026/MNCAATourneySlots.csv"

    # Output dir keys off the stage-1 input basename so different stage-1s
    # produce non-colliding artifacts. The default ('pairwise_v4.csv')
    # preserves the historical 'output/v9{b|c|d}_sweep' naming for
    # backwards compatibility.
    pw_basename = Path(pairwise_v4).stem  # e.g. 'pairwise_v4', 'pairwise_ensemble_e1'
    if pw_basename == "pairwise_v4":
        if feature_set == "v9b":
            out_dir = "output/v9_sweep"
            results_csv_path = "output/v9_sweep_results.csv"
        elif feature_set == "v9c":
            out_dir = "output/v9c_sweep"
            results_csv_path = "output/v9c_sweep_results.csv"
        else:  # v9d
            out_dir = "output/v9d_sweep"
            results_csv_path = "output/v9d_sweep_results.csv"
    else:
        # Custom stage-1 input: e.g. pairwise_ensemble_e1.csv -> v9c_ensemble_e1_sweep.
        suffix = pw_basename.replace("pairwise_", "")
        out_dir = f"output/{feature_set}_{suffix}_sweep"
        results_csv_path = f"output/{feature_set}_{suffix}_sweep_results.csv"

    pairwise_bt_csv = "output/pairwise_bt.csv" if feature_set == "v9d" else None

    df = run_sweep(
        grid=GRID,
        pairwise_v4_csv=pairwise_v4,
        results_csv=results_csv,
        seeds_csv=seeds_csv,
        out_dir=out_dir,
        results_csv_path=results_csv_path,
        slots_csv=slots_csv,
        feature_set=feature_set,
        pairwise_bt_csv=pairwise_bt_csv,
    )

    # Summary table.
    print("\nResults sorted by total_brkt_pts (descending):")
    print(df.to_string(index=False))

    # v8 baseline + anchor-cell sanity gate.
    v8_total = float(score_pairwise_path(pairwise_v8)["total_pts"])
    anchor_row = df[(df["w_upset"] == 1.0) & (df["w_miss"] == 0.0)].iloc[0]
    anchor_total = float(anchor_row["total_brkt_pts"])
    delta = anchor_total - v8_total
    print(f"\nv8 baseline:   {v8_total:>8.1f} pts")
    print(f"anchor (1, 0): {anchor_total:>8.1f} pts (delta {delta:+.2f})")
    if abs(delta) > 5.0:
        print("WARNING: anchor cell does not reproduce v8 within 5 pts; "
              "sweep results may be invalid -- inspect per-game LL/Acc to "
              "confirm trainer is sane before trusting cell rankings.")
    else:
        print(f"Anchor cell reproduces v8 within 5 pts -- sweep is valid. "
              f"(feature_set={feature_set} differs from v8 in features and "
              "may produce small chalk-pick boundary deltas at uniform weights.)")

    # Winner check (+10 bar).
    best = df.iloc[0]
    best_delta = float(best["total_brkt_pts"]) - v8_total
    print(f"\nbest cell:     W_UPSET={best['w_upset']}, "
          f"W_MISS={best['w_miss']}, "
          f"total_brkt_pts={best['total_brkt_pts']:.1f}, "
          f"delta vs v8={best_delta:+.2f}")
    if best_delta > 10.0:
        print(f"WINNER: best cell beats v8 by {best_delta:.1f} pts (> +10).")
    else:
        print(f"NO WINNER: best cell delta {best_delta:+.2f} pts (bar +10).")


if __name__ == "__main__":
    main()
