"""Custom LOSO driver for the team-seed-residual experiment.

Built to avoid memory issues we hit when running enhanced_model_v3 with
MM_PAIRWISE_OUT set: the in-loop pairwise-write block accumulated memory
across seasons and Windows killed the process mid-loop.

This driver:
  1. Calls prepare_loso_inputs() once to build the feature matrix.
  2. For each season, trains XGB on other seasons + predicts pairwise.
  3. Writes per-season pairwise rows immediately and gc.collect()s.
  4. Records per-season log-loss/accuracy and writes cv_per_season at end.

Outputs:
  output/pairwise_v4_with_team_history.csv (full 22-season pairwise)
  output/cv_per_season_v3_team_history.csv (per-season CV summary)

Usage:
    MM_TUNED_PARAMS_V3="$(cat output/v4_tuned_params.json)" \\
      python -m src.loso_with_pairwise_for_team_history
"""
from __future__ import annotations

import gc
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss as sklearn_log_loss, roc_auc_score

from src.enhanced_model_v3 import (
    build_matchup_data_from_kaggle,
    build_weighted_matchup_data,
    prepare_loso_inputs,
)
from src.models.matchup import build_matchup_features, expand_feature_cols
from src.models.train import train_model

PAIRWISE_OUT = Path("output/pairwise_v4_with_team_history.csv")
CV_OUT = Path("output/cv_per_season_v3_team_history.csv")


def main():
    # Ensure clean output (we append per-season).
    PAIRWISE_OUT.unlink(missing_ok=True)

    # Tuned XGB params (from cached file via env var, mirrors v3 main).
    tuned_env = os.environ.get("MM_TUNED_PARAMS_V3")
    if not tuned_env:
        print("ERROR: set MM_TUNED_PARAMS_V3 to JSON-encoded XGB params dict",
              file=sys.stderr)
        return 1
    xgb_params = json.loads(tuned_env)
    print(f"Tuned XGB params: {xgb_params}", flush=True)

    # Build feature matrix once.
    print("Calling prepare_loso_inputs() ...", flush=True)
    inputs = prepare_loso_inputs()
    feature_matrix = inputs["feature_matrix"]
    tourney = inputs["tourney_filtered"]
    reg_season = inputs["regular_results"]
    feature_cols = inputs["feature_cols"]
    top_80 = inputs["top_80_by_season"]
    print(f"feature_matrix: {len(feature_matrix):,} rows, "
          f"{len(feature_cols)} feature cols", flush=True)
    print(f"feature_cols includes: {[c for c in feature_cols if 'team_seed_residual' in c]}",
          flush=True)

    # Per-season LOSO loop.
    seasons = sorted(s for s in tourney["Season"].unique() if s >= 2003)
    print(f"\nLOSO over {len(seasons)} seasons: {seasons[0]}-{seasons[-1]}",
          flush=True)
    cv_results = []

    for holdout in seasons:
        train_tourney = tourney[tourney["Season"] != holdout]
        test_tourney = tourney[tourney["Season"] == holdout]
        if len(test_tourney) == 0:
            continue

        train_top_ids = set()
        for s in train_tourney["Season"].unique():
            train_top_ids |= top_80.get(int(s), set())

        train_reg = reg_season[reg_season["Season"] != holdout]
        X_train, y_train, w_train = build_weighted_matchup_data(
            feature_matrix, train_tourney, train_reg, feature_cols,
            top_n_team_ids=train_top_ids,
            supplemental_weight=0.25,
        )
        X_test, y_test, _ = build_matchup_data_from_kaggle(
            feature_matrix, test_tourney, feature_cols
        )

        if len(X_train) == 0 or len(X_test) == 0:
            continue

        medians = X_train.median()
        X_train = X_train.fillna(medians)
        X_test = X_test.fillna(medians)

        model = train_model(
            X_train, y_train, random_seed=42,
            xgb_params=xgb_params, sample_weight=w_train,
        )

        # Test-set scoring.
        y_prob = model.predict_proba(X_test)[:, 1]
        season_loss = float(sklearn_log_loss(y_test, y_prob))
        season_acc = float((y_prob.round() == y_test).mean())
        try:
            season_auc = float(roc_auc_score(y_test, y_prob))
        except ValueError:
            season_auc = 0.5
        season_brier = float(np.mean((y_prob - y_test.values) ** 2))

        # Pairwise predictions for ALL team pairs in the holdout field.
        field = sorted(set(test_tourney["WTeamID"]) | set(test_tourney["LTeamID"]))
        fm_yr = feature_matrix[feature_matrix["Season"] == holdout].set_index("TeamID")
        have_feats = [t for t in field if t in fm_yr.index]
        pair_rows = []
        pair_ids = []
        for i in range(len(have_feats)):
            ai = have_feats[i]
            av = fm_yr.loc[ai, feature_cols].values.astype(float)
            for j in range(i + 1, len(have_feats)):
                bj = have_feats[j]
                bv = fm_yr.loc[bj, feature_cols].values.astype(float)
                pair_rows.append(build_matchup_features(av, bv))
                pair_ids.append((ai, bj))
        if pair_rows:
            pdf = pd.DataFrame(pair_rows,
                               columns=expand_feature_cols(feature_cols)).fillna(medians)
            pp = model.predict_proba(pdf)[:, 1]
            out = pd.DataFrame({
                "season": holdout,
                "team_a": [a for a, _ in pair_ids],
                "team_b": [b for _, b in pair_ids],
                "p_a_wins": pp,
            })
            out.to_csv(PAIRWISE_OUT, mode="a", index=False,
                       header=not PAIRWISE_OUT.exists())
            del pdf, pp, out, pair_rows, pair_ids

        cv_results.append({
            "season": holdout,
            "log_loss": season_loss,
            "brier_score": season_brier,
            "accuracy": season_acc,
            "auc": season_auc,
            "n_games": len(test_tourney),
        })
        print(f"  Season {holdout}: ll={season_loss:.4f}, "
              f"acc={season_acc:.3f}, n={len(test_tourney)}",
              flush=True)

        # Free per-season memory aggressively.
        del model, X_train, y_train, w_train, X_test, y_test, y_prob
        del medians, fm_yr, have_feats
        gc.collect()

    # Save CV summary.
    cv_df = pd.DataFrame(cv_results)
    cv_df.to_csv(CV_OUT, index=False)
    print(f"\nWrote {CV_OUT} ({len(cv_df)} seasons)", flush=True)
    print(f"Wrote {PAIRWISE_OUT}", flush=True)

    # Weighted mean LL/acc summary.
    n_total = cv_df["n_games"].sum()
    wt_ll = (cv_df["log_loss"] * cv_df["n_games"]).sum() / n_total
    wt_acc = (cv_df["accuracy"] * cv_df["n_games"]).sum() / n_total
    print(f"\nwt_mean LL = {wt_ll:.4f}", flush=True)
    print(f"wt_mean acc = {wt_acc:.4f}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
