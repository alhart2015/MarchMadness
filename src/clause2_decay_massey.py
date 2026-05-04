"""Clause 2 (LL headroom) for time-decayed Massey-MOV.

Mirrors src/diagnose_massey_mov.py:clause2_headroom but parameterized on
half_life_days. Constructs the v4 feature matrix via prepare_loso_inputs,
merges decay-aware massey_mov_rating in via the (TeamID, Season) key,
runs leave_one_season_out_cv_weighted twice on a 3-season subset (with /
without the new column in feature_cols), compares mean test LL.

Pass: mean(LL_with) - mean(LL_without) <= +0.001 (LL_HEADROOM_MAX).

Usage: python src/clause2_decay_massey.py 14
       (positional arg: half_life_days; required)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.diagnose_massey_mov import GATE_SUBSET_SEASONS, LL_HEADROOM_MAX  # noqa: E402
from src.enhanced_model_v3 import (  # noqa: E402
    leave_one_season_out_cv_weighted,
    prepare_loso_inputs,
)
from src.features.massey_matrix import compute_massey_mov_ratings  # noqa: E402


def main():
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {sys.argv[0]} <half_life_days>")
    hl = float(sys.argv[1])
    if hl <= 0:
        raise SystemExit(f"half_life_days must be positive, got {hl}")

    print(f"=== clause 2 (decay massey, hl={hl}d) ===")
    print("Loading v4 LOSO inputs...")
    inputs = prepare_loso_inputs()
    fm = inputs["feature_matrix"]
    tourney = inputs["tourney_filtered"]
    regular = inputs["regular_results"]
    feature_cols_full = inputs["feature_cols"]
    top_80 = inputs["top_80_by_season"]

    if "massey_mov_rating" in fm.columns:
        # Wire-in is reverted, so we expect this NOT to be present. If it
        # somehow is (e.g., wire-in re-applied), drop it and use our own
        # decay-aware version instead.
        fm = fm.drop(columns=["massey_mov_rating"])
        feature_cols_full = [c for c in feature_cols_full if c != "massey_mov_rating"]

    print(f"  base feature matrix: {len(fm)} rows, {len(feature_cols_full)} feature cols")
    print(f"Computing massey_mov_rating with half_life_days={hl}...")
    massey = compute_massey_mov_ratings(regular, mov_cap=21, half_life_days=hl)
    fm = fm.merge(massey, on=["TeamID", "Season"], how="left")
    n_pop = int(fm["massey_mov_rating"].notna().sum())
    print(f"  merged: {n_pop} team-seasons populated with massey_mov_rating")

    cols_with = list(feature_cols_full) + ["massey_mov_rating"]
    cols_without = list(feature_cols_full)

    print(f"\nRunning LOSO with massey_mov_rating in feature_cols, holdouts={GATE_SUBSET_SEASONS}...")
    res_with = leave_one_season_out_cv_weighted(
        fm, tourney, regular, cols_with, top_80, allowed_holdouts=GATE_SUBSET_SEASONS,
    )
    print(f"\nRunning LOSO WITHOUT massey_mov_rating, holdouts={GATE_SUBSET_SEASONS}...")
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
        "half_life_days": hl,
        "subset_seasons": list(GATE_SUBSET_SEASONS),
        "per_season": per_season,
        "mean_ll_with_massey": mean_with,
        "mean_ll_without_massey": mean_without,
        "mean_ll_delta": delta,
        "ll_headroom_max": LL_HEADROOM_MAX,
        "pass": passed,
    }
    Path("output").mkdir(exist_ok=True)
    out_path = Path(f"output/clause2_decay_massey_hl{int(hl)}.json")
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
