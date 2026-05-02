"""Pre-sweep falsification gate for v9-D (BT-as-feature).

Spec: docs/superpowers/specs/2026-05-02-bt-as-feature-design.md

Question: does v9-D@(1.0, 0.0) beat v9-C@(1.0, 0.0) on weighted-mean
per-game log loss across 22 LOSO seasons by at least
GATE_LL_HEADROOM_MIN?

If yes -> proceed to the 15-cell W_UPSET / W_MISS sweep with
V9_FEATURE_SET=v9d. If no -> stop, write findings note, NO-GO. Saves
the cost of running 14 additional sweep cells when the feature
isn't extracting meaningful signal even at uniform weights.

Mirrors src/diagnose_bt_vs_v4.py in shape (compute_gate /
check_gate / print_report / main / sys.exit nonzero on FAIL).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.train_upset_model import (
    double_loso_eval, load_per_game_data_with_upset,
)

DATA = Path("data/raw/march-machine-learning-2026")
DEFAULT_DIAGNOSTIC_OUT = "output/diag_v9d.json"

# Single-clause gate: v9-D's per-game LL must beat v9-C's by at least
# this much on the same per-game frame at uniform weights. A tighter
# threshold than the BT-vs-v4 ensemble gate (0.005) because this is
# a single-clause test, and a paired comparison (same per-game rows
# evaluated by both feature sets) cancels most variance.
GATE_LL_HEADROOM_MIN = 0.001


def _weighted_mean_ll(eval_df: pd.DataFrame) -> float:
    """Weighted-mean log loss across seasons, weighting by n_games."""
    n_total = float(eval_df["n_games"].sum())
    if n_total <= 0:
        return float("nan")
    return float((eval_df["ll_v9"] * eval_df["n_games"]).sum() / n_total)


def compute_gate(
    pairwise_v4_csv: str,
    pairwise_bt_csv: str,
    results_csv: str = str(DATA / "MNCAATourneyCompactResults.csv"),
    seeds_csv: str = str(DATA / "MNCAATourneySeeds.csv"),
) -> dict:
    """Compute the pre-sweep gate diagnostic.

    Loads the per-game data once with p_bt joined; runs double-LOSO eval
    twice (v9-c features = no p_bt; v9-d features = with p_bt) at
    uniform weights (W_UPSET=1.0, W_MISS=0.0); returns LL values,
    headroom, and the gate threshold.
    """
    per_game = load_per_game_data_with_upset(
        pairwise_v4_csv, results_csv, seeds_csv,
        pairwise_bt_csv=pairwise_bt_csv,
    )

    eval_v9c = double_loso_eval(
        per_game, w_upset=1.0, w_miss=0.0, feature_set="v9c"
    )
    eval_v9d = double_loso_eval(
        per_game, w_upset=1.0, w_miss=0.0, feature_set="v9d"
    )

    ll_v9c = _weighted_mean_ll(eval_v9c)
    ll_v9d = _weighted_mean_ll(eval_v9d)
    headroom = ll_v9c - ll_v9d  # positive = v9d beats v9c

    return {
        "n_games_v9c": int(eval_v9c["n_games"].sum()),
        "n_games_v9d": int(eval_v9d["n_games"].sum()),
        "ll_v9c": float(ll_v9c),
        "ll_v9d": float(ll_v9d),
        "headroom": float(headroom),
        "threshold": float(GATE_LL_HEADROOM_MIN),
    }


def check_gate(diag: dict) -> dict:
    """Single-clause: headroom >= threshold -> pass."""
    threshold = diag.get("threshold", GATE_LL_HEADROOM_MIN)
    if diag["headroom"] >= threshold:
        return {
            "pass": True,
            "reason": f"headroom {diag['headroom']:+.4f} >= {threshold}",
        }
    return {
        "pass": False,
        "reason": f"headroom {diag['headroom']:+.4f} < {threshold}",
    }


def print_report(diag: dict, gate: dict) -> None:
    print("=" * 70)
    print("v9-D PRE-SWEEP GATE (BT-as-feature)")
    print("=" * 70)
    print(f"  n games (v9c eval): {diag['n_games_v9c']}")
    print(f"  n games (v9d eval): {diag['n_games_v9d']}")
    print(f"\n  Per-game LL @ (W_UPSET=1.0, W_MISS=0.0), 22-season weighted mean:")
    print(f"    v9-C (5 features):       {diag['ll_v9c']:.4f}")
    print(f"    v9-D (6 features + BT):  {diag['ll_v9d']:.4f}")
    print(f"    headroom (v9c - v9d):    {diag['headroom']:+.4f}")
    print(f"    threshold:               {diag['threshold']:.4f}")
    print(f"\n=== VERDICT ===")
    if gate["pass"]:
        print(f"  GATE PASSED: {gate['reason']}")
        print(f"  -> Proceed to 15-cell V9_FEATURE_SET=v9d sweep")
    else:
        print(f"  GATE FAILED: {gate['reason']}")
        print(f"  -> Stop. Write findings note. No 15-cell sweep.")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--pairwise-v4", default="output/pairwise_v4.csv")
    parser.add_argument("--pairwise-bt", default="output/pairwise_bt.csv")
    parser.add_argument("--out-json", default=DEFAULT_DIAGNOSTIC_OUT)
    args = parser.parse_args(argv)

    diag = compute_gate(args.pairwise_v4, args.pairwise_bt)
    gate = check_gate(diag)
    print_report(diag, gate)

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump({"diagnostic": diag, "gate": gate}, f, indent=2)
    print(f"\n  saved {args.out_json}")
    return 0 if gate["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
