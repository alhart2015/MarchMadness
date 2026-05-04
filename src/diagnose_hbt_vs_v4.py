"""Per-sigma-cell gate diagnostic for hierarchical BT vs v4.

Spec: docs/superpowers/specs/2026-05-03-hierarchical-bt-feature-priors-design.md

Mirrors src/diagnose_bt_vs_v4.py but loops over a sigma sweep
(one CSV per cell named output/pairwise_hbt_sigma_<S>.csv), produces
a per-cell row, applies the same 3-clause gate per cell, and picks
the best-headroom passing cell.

Gate thresholds match plain BT exactly so cells are directly
comparable across experiments.
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.diagnose_bt_vs_v4 import (
    GATE_HEADROOM_MIN,
    GATE_R_MAX,
    GATE_W_HIGH,
    GATE_W_LOW,
    _load_pairwise_lookup,
)

DATA = Path("data/raw/march-machine-learning-2026")
DEFAULT_DIAGNOSTIC_OUT = "output/diag_hbt_sweep.json"
DEFAULT_PAIRWISE_GLOB = "output/pairwise_hbt_sigma_*.csv"
SIGMA_FILENAME_RE = re.compile(r"pairwise_hbt_sigma_(?P<sigma>[0-9]+(?:\.[0-9]+)?)\.csv$")


def score_one_cell(p_v4_winner: np.ndarray, p_hbt_winner: np.ndarray) -> dict:
    """Score one sigma cell from arrays of p(actual_winner) per game.

    p_v4_winner[k] = v4's predicted prob that the actual winner of game k won.
    p_hbt_winner[k] = HBT's same.

    Returns the standard diagnostic dict (r, w_opt, headroom, etc.) plus
    per-clause pass/fail flags.
    """
    eps = 1e-15
    n = len(p_v4_winner)

    v4_clip = np.clip(p_v4_winner, eps, 1 - eps)
    hbt_clip = np.clip(p_hbt_winner, eps, 1 - eps)

    ll_v4 = float(-np.mean(np.log(v4_clip)))
    ll_hbt = float(-np.mean(np.log(hbt_clip)))
    acc_v4 = float((p_v4_winner > 0.5).mean())
    acc_hbt = float((p_hbt_winner > 0.5).mean())

    v4_res = 1 - p_v4_winner
    hbt_res = 1 - p_hbt_winner
    if n > 1 and v4_res.std() > 0 and hbt_res.std() > 0:
        r = float(np.corrcoef(v4_res, hbt_res)[0, 1])
    else:
        r = float("nan")

    v4_pick = (p_v4_winner > 0.5).astype(int)
    hbt_pick = (p_hbt_winner > 0.5).astype(int)
    disagree_n = int((v4_pick != hbt_pick).sum())

    ws = np.linspace(0.0, 1.0, 101)
    ll_at_w = []
    for w in ws:
        p_blend = w * p_v4_winner + (1 - w) * p_hbt_winner
        p_blend = np.clip(p_blend, eps, 1 - eps)
        ll_at_w.append(float(-np.mean(np.log(p_blend))))
    ll_at_w_arr = np.array(ll_at_w)
    optimal_idx = int(np.argmin(ll_at_w_arr))
    optimal_w = float(ws[optimal_idx])
    optimal_ll = float(ll_at_w_arr[optimal_idx])
    headroom = ll_v4 - optimal_ll

    passes_r = bool(np.isfinite(r) and r < GATE_R_MAX)
    passes_w = bool(GATE_W_LOW <= optimal_w <= GATE_W_HIGH)
    passes_headroom = bool(headroom > GATE_HEADROOM_MIN)

    return {
        "n_games": n,
        "ll_v4": ll_v4,
        "ll_hbt": ll_hbt,
        "acc_v4": acc_v4,
        "acc_hbt": acc_hbt,
        "r": r,
        "disagree_n": disagree_n,
        "w_opt": optimal_w,
        "ll_blend": optimal_ll,
        "headroom": headroom,
        "passes_r": passes_r,
        "passes_w": passes_w,
        "passes_headroom": passes_headroom,
        "passes_all": passes_r and passes_w and passes_headroom,
    }


def _aligned_winner_probs(
    pairwise_v4_csv: str,
    pairwise_hbt_csv: str,
    results_df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """Join the two pairwise CSVs to tournament outcomes; return aligned
    p_v4_winner and p_hbt_winner arrays."""
    v4 = _load_pairwise_lookup(pairwise_v4_csv)
    hbt = _load_pairwise_lookup(pairwise_hbt_csv)

    v4_p_w, hbt_p_w = [], []
    for _, g in results_df.iterrows():
        s, w, l = int(g["Season"]), int(g["WTeamID"]), int(g["LTeamID"])
        if s < 2003 or s > 2025:
            continue
        a, b = (w, l) if w < l else (l, w)
        p_v4 = v4.get((s, a, b))
        p_hbt = hbt.get((s, a, b))
        if p_v4 is None or p_hbt is None:
            continue
        v4_p_w.append(p_v4 if a == w else 1.0 - p_v4)
        hbt_p_w.append(p_hbt if a == w else 1.0 - p_hbt)

    return np.array(v4_p_w), np.array(hbt_p_w)


def score_cell_from_files(
    sigma: float,
    pairwise_v4_csv: str,
    pairwise_hbt_csv: str,
    results_df: pd.DataFrame,
) -> dict:
    """File -> aligned arrays -> score_one_cell, with sigma + paths added."""
    p_v4, p_hbt = _aligned_winner_probs(
        pairwise_v4_csv, pairwise_hbt_csv, results_df
    )
    cell = score_one_cell(p_v4, p_hbt)
    cell["sigma"] = float(sigma)
    cell["pairwise_hbt"] = pairwise_hbt_csv
    return cell


def pick_best_passing_cell(cells: list[dict]) -> dict | None:
    """Return the cell with the highest headroom among cells that pass
    all three clauses, or None if no cell passes."""
    passing = [c for c in cells if c.get("passes_all")]
    if not passing:
        return None
    return max(passing, key=lambda c: c["headroom"])


def discover_sigma_cells(pairwise_glob: str) -> list[tuple[float, str]]:
    """Find sigma cells from filenames matching pairwise_hbt_sigma_<S>.csv."""
    out = []
    for path in sorted(glob.glob(pairwise_glob)):
        m = SIGMA_FILENAME_RE.search(path)
        if not m:
            continue
        out.append((float(m.group("sigma")), path))
    return sorted(out, key=lambda x: x[0])


def print_report(cells: list[dict], best: dict | None) -> None:
    print("=" * 78)
    print("HIERARCHICAL BT vs v4 -- SIGMA SWEEP GATE")
    print("=" * 78)
    if not cells:
        print("  no sigma cells found")
        return

    print(f"  v4 standalone LL: {cells[0]['ll_v4']:.4f}   "
          f"acc: {cells[0]['acc_v4']:.3f}   n_games: {cells[0]['n_games']}")
    print()

    header = (
        f"  {'sigma':>6}  {'r_resid':>8}  {'ll_hbt':>7}  {'w_opt':>6}  "
        f"{'headroom':>9}  c1  c2  c3  verdict"
    )
    print(header)
    print(f"  {'-' * (len(header) - 2)}")
    for c in cells:
        c1 = "Y" if c["passes_r"] else "N"
        c2 = "Y" if c["passes_w"] else "N"
        c3 = "Y" if c["passes_headroom"] else "N"
        v = "PASS" if c["passes_all"] else "FAIL"
        print(
            f"  {c['sigma']:>6.2f}  {c['r']:>8.3f}  {c['ll_hbt']:>7.4f}  "
            f"{c['w_opt']:>6.2f}  {c['headroom']:>+9.4f}  {c1}   {c2}   "
            f"{c3}   {v}"
        )
    print()

    print(f"  Gate thresholds: r < {GATE_R_MAX}, "
          f"w in [{GATE_W_LOW}, {GATE_W_HIGH}], "
          f"headroom > {GATE_HEADROOM_MIN}")
    print()

    print("=== VERDICT ===")
    if best is None:
        print("  ALL CELLS FAILED. v4 stays as stage-1.")
        print("  -> Stop. Write findings note. No v9-C compute.")
    else:
        print(f"  BEST PASSING CELL: sigma={best['sigma']:.2f}, "
              f"r={best['r']:.3f}, w_opt={best['w_opt']:.2f}, "
              f"headroom={best['headroom']:+.4f}")
        print("  -> Proceed to v9-C correction + bracket-points backtest")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--pairwise-v4", default="output/pairwise_v4.csv")
    parser.add_argument("--pairwise-hbt-glob", default=DEFAULT_PAIRWISE_GLOB)
    parser.add_argument("--out", default=DEFAULT_DIAGNOSTIC_OUT)
    parser.add_argument(
        "--results",
        default=str(DATA / "MNCAATourneyCompactResults.csv"),
    )
    args = parser.parse_args(argv)

    sigma_cells = discover_sigma_cells(args.pairwise_hbt_glob)
    if not sigma_cells:
        print(f"ERROR: no files matched {args.pairwise_hbt_glob}")
        return 2

    results_df = pd.read_csv(args.results)

    cells = []
    for sigma, hbt_path in sigma_cells:
        cell = score_cell_from_files(sigma, args.pairwise_v4, hbt_path, results_df)
        cells.append(cell)

    best = pick_best_passing_cell(cells)
    print_report(cells, best)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(
            {
                "cells": cells,
                "best_passing_cell": best,
                "thresholds": {
                    "r_max": GATE_R_MAX,
                    "w_low": GATE_W_LOW,
                    "w_high": GATE_W_HIGH,
                    "headroom_min": GATE_HEADROOM_MIN,
                },
            },
            f,
            indent=2,
        )
    print(f"\n  saved {args.out}")

    return 0 if best is not None else 1


if __name__ == "__main__":
    sys.exit(main())
