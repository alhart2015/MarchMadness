"""Standalone clause 2 (LL headroom) runner for colley_rating.

Mirrors src/clause2_decay_massey.py but for Colley. Constructs the v4
feature matrix via prepare_loso_inputs, computes colley_rating from
regular-season W/L data via compute_colley_ratings, merges into fm via
the (TeamID, Season) key, runs leave_one_season_out_cv_weighted twice
on the 3-season subset (with / without colley_rating in feature_cols),
compares mean test LL.

Pass: mean(LL_with) - mean(LL_without) <= +0.001 (LL_HEADROOM_MAX).

Used by the colley-massey clean re-eval (recovery step 5 marginal #4)
to avoid temporary wire-in of the reverted colley_rating column.

Usage: python src/clause2_colley.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.diagnose_colley import GATE_SUBSET_SEASONS, LL_HEADROOM_MAX  # noqa: E402
from src.enhanced_model_v3 import (  # noqa: E402
    leave_one_season_out_cv_weighted,
    prepare_loso_inputs,
)
from src.features.colley_matrix import compute_colley_ratings  # noqa: E402


def main():
    print("=== clause 2 (colley) ===")
    print("Loading v4 LOSO inputs...")
    inputs = prepare_loso_inputs()
    fm = inputs["feature_matrix"]
    tourney = inputs["tourney_filtered"]
    regular = inputs["regular_results"]
    feature_cols_full = inputs["feature_cols"]
    top_80 = inputs["top_80_by_season"]

    if "colley_rating" in fm.columns:
        # Wire-in is reverted, so we expect this NOT to be present. If it
        # somehow is (e.g., wire-in re-applied), drop it and use our own
        # locally-computed version instead.
        fm = fm.drop(columns=["colley_rating"])
        feature_cols_full = [c for c in feature_cols_full if c != "colley_rating"]

    print(f"  base feature matrix: {len(fm)} rows, {len(feature_cols_full)} feature cols")
    print("Computing colley_rating from regular-season W/L...")
    colley = compute_colley_ratings(regular)
    fm = fm.merge(colley, on=["TeamID", "Season"], how="left")
    n_pop = int(fm["colley_rating"].notna().sum())
    print(f"  merged: {n_pop} team-seasons populated with colley_rating")

    cols_with = list(feature_cols_full) + ["colley_rating"]
    cols_without = list(feature_cols_full)

    print(f"\nRunning LOSO with colley_rating in feature_cols, holdouts={GATE_SUBSET_SEASONS}...")
    res_with = leave_one_season_out_cv_weighted(
        fm, tourney, regular, cols_with, top_80, allowed_holdouts=GATE_SUBSET_SEASONS,
    )
    print(f"\nRunning LOSO WITHOUT colley_rating, holdouts={GATE_SUBSET_SEASONS}...")
    res_without = leave_one_season_out_cv_weighted(
        fm, tourney, regular, cols_without, top_80, allowed_holdouts=GATE_SUBSET_SEASONS,
    )

    df_with = res_with["per_season"]
    df_without = res_without["per_season"]
    per_with = {int(r["season"]): r for _, r in df_with.iterrows()}
    per_without = {int(r["season"]): r for _, r in df_without.iterrows()}

    per_season = []
    for season in GATE_SUBSET_SEASONS:
        rw = per_with.get(int(season))
        rwo = per_without.get(int(season))
        if rw is None or rwo is None:
            continue
        per_season.append({
            "season": int(season),
            "ll_with": float(rw["log_loss"]),
            "ll_without": float(rwo["log_loss"]),
            "ll_delta": float(rw["log_loss"] - rwo["log_loss"]),
        })

    mean_with = float(np.mean([r["ll_with"] for r in per_season]))
    mean_without = float(np.mean([r["ll_without"] for r in per_season]))
    delta = mean_with - mean_without
    passed = bool(delta <= LL_HEADROOM_MAX)

    result = {
        "subset_seasons": list(GATE_SUBSET_SEASONS),
        "per_season": per_season,
        "mean_ll_with_colley": mean_with,
        "mean_ll_without_colley": mean_without,
        "mean_ll_delta": delta,
        "ll_headroom_max": LL_HEADROOM_MAX,
        "pass": passed,
    }
    Path("output").mkdir(exist_ok=True)
    out_path = Path("output/diag_clause2_colley.json")
    out_path.write_text(json.dumps(result, indent=2))

    print("\n=== summary ===")
    for r in per_season:
        sign = "+" if r["ll_delta"] >= 0 else ""
        print(
            f"  season {r['season']}: ll_with={r['ll_with']:.4f} "
            f"ll_without={r['ll_without']:.4f} delta={sign}{r['ll_delta']:.4f}"
        )
    sign = "+" if delta >= 0 else ""
    print(f"  MEAN: ll_with={mean_with:.4f} ll_without={mean_without:.4f} "
          f"delta={sign}{delta:.4f}")
    print(f"  threshold: delta <= {LL_HEADROOM_MAX}")
    print(f"  CLAUSE 2: {'PASS' if passed else 'FAIL'}")
    print(f"\nWrote {out_path}")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
