"""Extract v4's feature_importances_ (XGB gain) on the pooled 22-season
training set, plus a per-(Season, TeamID) feature-matrix snapshot for v12
stage-2 joining.

Spec: docs/superpowers/specs/2026-05-14-v12-stage2-v4-feature-diffs-design.md
Plan: docs/superpowers/plans/2026-05-14-v12-stage2-v4-feature-diffs.md

Outputs:
  - output/v4_feature_importance.csv  (per raw feature, by _diff XGB gain)
  - output/v4_feature_matrix.parquet  (per (Season, TeamID), 67 raw features,
                                       NaN-filled with per-column median).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

OUTPUT = Path("output")


def _load_or_tune_params() -> dict:
    cached = OUTPUT / "v4_tuned_params.json"
    if cached.exists():
        with open(cached) as f:
            params = json.load(f)
        print(f"  Loaded cached tuned params from {cached}: {params}")
        return params
    print("  No cached tuned params; running Optuna (30 trials)...")
    from src.enhanced_model_v3 import prepare_loso_inputs
    inputs = prepare_loso_inputs()
    X_all = inputs["_X_all"]
    y_all = inputs["_y_all"]
    from src.models.tuning import tune_hyperparameters
    params = tune_hyperparameters(X_all, y_all, n_trials=30, random_seed=42)
    OUTPUT.mkdir(exist_ok=True)
    with open(cached, "w") as f:
        json.dump(params, f, indent=2)
    print(f"  Saved tuned params to {cached}")
    return params


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="output/v4_feature_importance.csv")
    p.add_argument("--matrix-out", default="output/v4_feature_matrix.parquet")
    args = p.parse_args(argv)

    print("=" * 70)
    print("v4 FEATURE IMPORTANCE EXTRACTION")
    print("=" * 70)

    from src.enhanced_model_v3 import prepare_loso_inputs
    inputs = prepare_loso_inputs()
    feature_matrix = inputs["feature_matrix"]
    feature_cols = inputs["feature_cols"]
    X_all = inputs["_X_all"]
    y_all = inputs["_y_all"]
    weights_all = inputs["_weights_all"]

    print(f"  feature_cols ({len(feature_cols)}): {feature_cols[:5]}...")
    print(f"  X_all shape: {X_all.shape}")
    expanded_cols = list(X_all.columns)
    assert all(c.endswith("_diff") for c in expanded_cols), \
        f"Unexpected non-diff columns in X_all: {[c for c in expanded_cols if not c.endswith('_diff')]}"
    raw_cols = [c.removesuffix("_diff") for c in expanded_cols]
    assert raw_cols == feature_cols, "X_all column order does not match feature_cols"

    # Tuned params -- load from cache or run Optuna
    params = _load_or_tune_params()

    # Fit one XGB on the pooled-22-season training set
    print("\nFitting XGB on pooled training set ...")
    model = xgb.XGBClassifier(
        random_state=42,
        eval_metric="logloss",
        **params,
    )
    model.fit(X_all.values, y_all, sample_weight=weights_all)

    # Feature importances are per expanded (_diff) column == per raw feature
    gains = model.feature_importances_  # default importance_type="gain"
    assert len(gains) == len(feature_cols), \
        f"importance len {len(gains)} != feature_cols len {len(feature_cols)}"

    ranking = pd.DataFrame({
        "feature_name": feature_cols,
        "gain": gains,
    }).sort_values("gain", ascending=False).reset_index(drop=True)
    ranking["gain_rank"] = ranking.index + 1
    ranking = ranking[["feature_name", "gain", "gain_rank"]]

    print(f"\nTop 15 features by XGB gain:")
    print(ranking.head(15).to_string(index=False))
    print(f"\nBottom 5:")
    print(ranking.tail(5).to_string(index=False))
    print(f"\n  Total gain (sanity): {gains.sum():.6f}  "
          f"(should be ~1.0 for normalized)")

    OUTPUT.mkdir(exist_ok=True)
    ranking.to_csv(args.out, index=False)
    print(f"\n  Wrote ranking to {args.out}")

    # Feature-matrix snapshot for v12 join. Filter to feature_cols only
    # (drops any non-feature columns the FM may carry).
    fm_snap = feature_matrix[["Season", "TeamID"] + feature_cols].copy()
    # Fill NaN per-column with median (over the pre-filtered FM, not over X_all).
    # v12 needs sensible diff values; matching v4's training-time fill exactly
    # is not required because stage-2 is a new model.
    for col in feature_cols:
        fm_snap[col] = fm_snap[col].fillna(fm_snap[col].median())
    fm_snap.to_parquet(args.matrix_out, index=False)
    print(f"  Wrote feature matrix snapshot to {args.matrix_out} "
          f"({len(fm_snap):,} rows x {len(feature_cols)} features)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
