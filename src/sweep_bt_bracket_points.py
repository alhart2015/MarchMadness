"""Drive the plain-BT bracket-points re-test sweep.

Spec: docs/superpowers/specs/2026-05-04-bt-bracket-points-design.md

For each w in --weights:
    1. ensemble_stage1.average_pairwise_csvs(v4, bt, w_v4, w_bt) -> ensemble_csv
    2. run_v9c_on_stage1.run_v9c(ensemble_csv) -> v9c_csv
    3. score_chalk_brackets.score_pairwise_path(v9c_csv) -> per-season + total

Anchor: w_v4=1.0 must produce a v9c_csv that matches the existing
output/pairwise_v9c_v4_baseline.csv exactly. Halts before scoring
other cells if the anchor fails.

Verdict bands match every prior stage-1 experiment:
    delta >= +25 brkt pts -> CLEAR (separate swap-in commit)
    +10 to +25            -> MARGINAL (document, no swap)
    < +10                 -> NO-GO
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.ensemble_stage1 import average_pairwise_csvs
from src.run_v9c_on_stage1 import run_v9c
from src.score_chalk_brackets import score_pairwise_path

DEFAULT_WEIGHTS = [0.60, 0.70, 0.80, 0.90, 0.95, 1.00]
DEFAULT_OUT_JSON = "output/bt_bracket_sweep.json"


def _make_weight_pair(w_v4: float) -> tuple[float, float]:
    """Return (w_v4, w_bt) where w_bt = 1 - w_v4."""
    return (round(float(w_v4), 4), round(1.0 - float(w_v4), 4))


def _format_w(w: float) -> str:
    return f"{w:.2f}"


def _anchor_check(csv_actual: str, csv_expected: str) -> dict:
    """Verify csv_actual matches csv_expected on (season, team_a, team_b).
    Returns a dict with keys matches (bool), max_abs_diff (float), and
    optional n_only_actual / n_only_expected for coverage diffs."""
    a = pd.read_csv(csv_actual).drop_duplicates(
        ["season", "team_a", "team_b"], keep="last"
    )
    b = pd.read_csv(csv_expected).drop_duplicates(
        ["season", "team_a", "team_b"], keep="last"
    )
    merged = a.merge(b, on=["season", "team_a", "team_b"],
                     suffixes=("_actual", "_expected"))
    n_only_a = len(a) - len(merged)
    n_only_b = len(b) - len(merged)
    if n_only_a != 0 or n_only_b != 0:
        return {
            "matches": False,
            "max_abs_diff": float("nan"),
            "n_only_actual": int(n_only_a),
            "n_only_expected": int(n_only_b),
            "n_rows_overlap": int(len(merged)),
        }
    diff = (merged["p_a_wins_actual"] - merged["p_a_wins_expected"]).abs()
    return {
        "matches": bool(diff.max() < 1e-9),
        "max_abs_diff": float(diff.max()),
        "n_rows": int(len(merged)),
    }


def _score_pairwise(csv_path: str) -> dict:
    """Wrap score_chalk_brackets.score_pairwise_path -> {total_pts, per_season}.

    Normalizes the return shape: score_pairwise_path returns
    {total_pts, per_season_pts}; we flatten to per_season for the
    sweep summary so deltas are easier to read.
    """
    s = score_pairwise_path(csv_path)
    per_season = {int(k): float(v) for k, v in s["per_season_pts"].items()}
    return {
        "total_pts": float(s["total_pts"]),
        "per_season": per_season,
    }


def run_sweep(
    weights: list[float],
    v4_csv: str,
    bt_csv: str,
    baseline_v9c_csv: str,
    out_dir: str | Path,
    out_json: str = DEFAULT_OUT_JSON,
) -> dict:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"BASELINE: {baseline_v9c_csv}")
    print("=" * 70)
    baseline = _score_pairwise(baseline_v9c_csv)
    print(f"  total_pts: {baseline['total_pts']:.0f}")

    cells = []
    anchor_check = None

    for w in weights:
        w_v4, w_bt = _make_weight_pair(w)
        ens_csv = str(out_path / f"pairwise_v4bt_w{_format_w(w_v4)}.csv")
        v9c_csv = str(out_path / f"pairwise_v9c_v4bt_w{_format_w(w_v4)}.csv")

        print()
        print("=" * 70)
        print(f"CELL  w_v4={w_v4}  w_bt={w_bt}")
        print("=" * 70)

        t0 = time.time()
        average_pairwise_csvs(
            in_a=v4_csv, in_b=bt_csv, out=ens_csv,
            weights=(w_v4, w_bt),
        )
        run_v9c(pairwise_in=ens_csv, pairwise_out=v9c_csv)

        if abs(w_v4 - 1.0) < 1e-9:
            anchor_check = _anchor_check(v9c_csv, baseline_v9c_csv)
            print(
                f"  ANCHOR CHECK: matches={anchor_check['matches']}, "
                f"max_abs_diff={anchor_check.get('max_abs_diff', float('nan'))}"
            )
            if not anchor_check["matches"]:
                print(
                    "  *** ANCHOR FAILED ***  Halting sweep -- the w_v4=1.0 cell "
                    "must reproduce the baseline exactly. See bt_bracket_sweep.json."
                )

        cell_score = _score_pairwise(v9c_csv)
        deltas = {
            s: cell_score["per_season"][s] - baseline["per_season"].get(s, 0.0)
            for s in cell_score["per_season"]
        }
        wins = sum(1 for d in deltas.values() if d > 0)
        losses = sum(1 for d in deltas.values() if d < 0)
        ties = sum(1 for d in deltas.values() if d == 0)
        delta_total = cell_score["total_pts"] - baseline["total_pts"]

        max_delta_season, max_delta_val = max(
            deltas.items(), key=lambda kv: abs(kv[1]), default=(None, 0.0)
        )

        elapsed = time.time() - t0
        print(
            f"  total_pts: {cell_score['total_pts']:.0f}  "
            f"delta: {delta_total:+.0f}  "
            f"W/L/T: {wins}/{losses}/{ties}  "
            f"biggest_swing: {max_delta_season} ({max_delta_val:+.0f})  "
            f"({elapsed:.1f}s)"
        )

        cells.append({
            "w_v4": w_v4,
            "w_bt": w_bt,
            "ensemble_csv": ens_csv,
            "v9c_csv": v9c_csv,
            "total_pts": cell_score["total_pts"],
            "delta_vs_baseline": delta_total,
            "per_season": cell_score["per_season"],
            "per_season_delta": {int(k): v for k, v in deltas.items()},
            "wins": wins,
            "losses": losses,
            "ties": ties,
            "biggest_swing_season": int(max_delta_season) if max_delta_season is not None else None,
            "biggest_swing_value": float(max_delta_val),
            "fold_seconds": round(elapsed, 1),
        })

    best = max(cells, key=lambda c: c["delta_vs_baseline"])
    if best["delta_vs_baseline"] >= 25:
        verdict = "CLEAR"
    elif best["delta_vs_baseline"] >= 10:
        verdict = "MARGINAL"
    else:
        verdict = "NO-GO"

    summary = {
        "config": {
            "weights": [float(w) for w in weights],
            "v4_pairwise": v4_csv,
            "bt_pairwise": bt_csv,
            "v9c_baseline": baseline_v9c_csv,
        },
        "anchor_check": anchor_check,
        "v4_baseline": baseline,
        "cells": cells,
        "best_cell": {
            "w_v4": best["w_v4"],
            "w_bt": best["w_bt"],
            "delta_vs_baseline": best["delta_vs_baseline"],
            "wins": best["wins"],
            "losses": best["losses"],
            "ties": best["ties"],
        },
        "verdict": verdict,
    }

    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)

    print()
    print("=" * 70)
    print(
        f"VERDICT: {verdict}   best w_v4={best['w_v4']}   "
        f"delta={best['delta_vs_baseline']:+.0f}"
    )
    print(f"  saved {out_json}")
    return summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--weights",
        default=",".join(f"{w:.2f}" for w in DEFAULT_WEIGHTS),
    )
    parser.add_argument("--v4", default="output/pairwise_v4.csv")
    parser.add_argument("--bt", default="output/pairwise_bt.csv")
    parser.add_argument(
        "--baseline-v9c",
        default="output/pairwise_v9c_v4_baseline.csv",
    )
    parser.add_argument("--out-dir", default="output")
    parser.add_argument("--out-json", default=DEFAULT_OUT_JSON)
    args = parser.parse_args(argv)

    weights = [float(x) for x in args.weights.split(",") if x.strip()]
    summary = run_sweep(
        weights=weights,
        v4_csv=args.v4,
        bt_csv=args.bt,
        baseline_v9c_csv=args.baseline_v9c,
        out_dir=args.out_dir,
        out_json=args.out_json,
    )
    return 0 if summary["verdict"] != "NO-GO" else 1


if __name__ == "__main__":
    sys.exit(main())
