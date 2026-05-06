"""Standalone clause 2 (LL headroom) runner for uniform Massey-MOV.

Mirrors src/clause2_decay_massey.py but with half_life_days=None
(uniform weighting -- the variant PR 15 rejected at clause 1 with
mean |corr| 0.957 vs adj_em). Forces the clause-2 run anyway as a
diagnostic for the colley-massey-clean-rerun finding: does the
W/L-vs-margin distinction or the time-window distinction explain
why Colley clause-2 PASSED on clean v4 while Massey-decay-14d FAILed.

If uniform Massey ALSO passes (like Colley) -> the time window
(14-day half-life overlapping with v4's late-season features) was
the load-bearing distinction.

If uniform Massey FAILS -> the W/L-vs-margin distinction is what
makes Colley unique vs the v4 stack; full-season Massey margin info
is too redundant with adj_em / kenpom / clean vegas_avg_margin.

NOTE: clause 1 is INTENTIONALLY skipped here. Per TODO.md "What's NOT
contaminated", intra-season correlations are leak-invariant -- uniform
Massey would still fail clause 1 on the clean baseline (corr 0.957 vs
adj_em). This script is a diagnostic, not a gate.

Usage: python src/clause2_uniform_massey.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

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
    print("=== clause 2 (uniform massey, hl=None) -- DIAGNOSTIC ONLY ===")
    print("Loading v4 LOSO inputs...")
    inputs = prepare_loso_inputs()
    fm = inputs["feature_matrix"]
    tourney = inputs["tourney_filtered"]
    regular = inputs["regular_results"]
    feature_cols_full = inputs["feature_cols"]
    top_80 = inputs["top_80_by_season"]

    if "massey_mov_rating" in fm.columns:
        fm = fm.drop(columns=["massey_mov_rating"])
        feature_cols_full = [c for c in feature_cols_full if c != "massey_mov_rating"]

    print(f"  base feature matrix: {len(fm)} rows, {len(feature_cols_full)} feature cols")
    print("Computing massey_mov_rating with UNIFORM weighting (no decay)...")
    massey = compute_massey_mov_ratings(regular, mov_cap=21)
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
        "half_life_days": None,
        "subset_seasons": list(GATE_SUBSET_SEASONS),
        "per_season": per_season,
        "mean_ll_with_massey": mean_with,
        "mean_ll_without_massey": mean_without,
        "mean_ll_delta": delta,
        "ll_headroom_max": LL_HEADROOM_MAX,
        "pass_clause2": passed,
        "note": "Diagnostic only -- clause 1 still FAILs (corr ~0.957 vs adj_em, leak-invariant). Run to disambiguate the colley-massey-clean-rerun finding.",
    }
    Path("output").mkdir(exist_ok=True)
    out_path = Path("output/diag_clause2_uniform_massey.json")
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
    print(f"  CLAUSE 2: {'PASS' if passed else 'FAIL'}  (clause 1 STILL FAILs -- this is a diagnostic, not a gate)")
    print(f"\nWrote {out_path}")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
