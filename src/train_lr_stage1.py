"""LR stage-1 trainer over the same 22-season LOSO loop v4 uses.

Spec: docs/superpowers/specs/2026-05-01-ensemble-stage1-design.md

Mirrors src/enhanced_model_v3.py:leave_one_season_out_cv_weighted but
swaps the XGBoost classifier for a logistic regression with
StandardScaler + Platt calibration (CalibratedClassifierCV). Inputs --
feature_matrix, tourney_filtered, regular_results, feature_cols,
top_80_by_season -- come from prepare_loso_inputs(), which is the same
data-setup v4 calls. So XGB and LR see byte-identical training rows
in every fold.

L2 strength is chosen via inner CV on the training folds only (5
candidate Cs). Scaler is fit per fold; never on the test season.
Platt scaling: CalibratedClassifierCV(method='sigmoid', cv=5) wraps
the LR on the training folds. Every supervised fit -- the logistic
regression, the inner CV, and the Platt calibrator -- sees only train-
fold rows; the test season is held out end-to-end.

Output: appends to output/pairwise_lr.csv with rows
    (season, team_a, team_b, p_a_wins),  team_a < team_b
covering all unordered pairs of tournament-field teams in each
held-out season.
"""
import argparse
import sys
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss as sklearn_log_loss
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.enhanced_model_v3 import prepare_loso_inputs
from src.enhanced_model import build_matchup_data_from_kaggle
from src.models.matchup import (
    build_matchup_features,
    build_weighted_matchup_data,
    expand_feature_cols,
)

DEFAULT_PAIRWISE_OUT = "output/pairwise_lr.csv"
INNER_CV_FOLDS = 5
C_GRID = [0.01, 0.1, 1.0, 10.0, 100.0]


def fit_lr_with_calibration(
    X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray, seed: int = 42
):
    """Inner-CV-tune L2 strength then Platt-calibrate the LR.

    Returns a fitted CalibratedClassifierCV wrapping the best LR. Caller
    is responsible for any feature-scaling step before calling this --
    inputs here should already be standardized.
    """
    base = LogisticRegression(
        penalty="l2", solver="lbfgs", max_iter=2000, random_state=seed,
    )
    grid = GridSearchCV(
        base, {"C": C_GRID}, cv=INNER_CV_FOLDS, scoring="neg_log_loss",
        n_jobs=1, refit=True,
    )
    grid.fit(X, y, sample_weight=sample_weight)
    best_C = grid.best_params_["C"]

    base_calibrated = LogisticRegression(
        penalty="l2", solver="lbfgs", max_iter=2000, random_state=seed, C=best_C,
    )
    calibrated = CalibratedClassifierCV(
        base_calibrated, method="sigmoid", cv=INNER_CV_FOLDS,
    )
    calibrated.fit(X, y, sample_weight=sample_weight)
    return calibrated


def dump_pairwise_for_season(
    season: int,
    field_team_ids: Iterable[int],
    feature_lookup: dict,
    scaler,
    model,
    out_csv: str,
) -> int:
    """Append (season, team_a, team_b, p_a_wins) rows for the season to out_csv.

    field_team_ids: iterable of team IDs that appeared in this season's
        tournament. We materialize all unordered pairs (a, b) with a < b.
    feature_lookup: dict[team_id -> np.ndarray of raw features]; the diff
        between two teams is the matchup feature row. NaNs in raw features
        should be filled by the caller before calling this function.
    scaler: a fitted StandardScaler (or None for synthetic-test use); when
        not None, applied to the matchup-row matrix before predict_proba.
    model: a fitted classifier with predict_proba(X) -> [N, 2].

    Returns the number of pair rows written.
    """
    field = sorted(set(int(t) for t in field_team_ids if t in feature_lookup))
    if len(field) < 2:
        return 0

    pair_rows = []
    pair_ids = []
    for i in range(len(field)):
        for j in range(i + 1, len(field)):
            a, b = field[i], field[j]
            av = feature_lookup[a]
            bv = feature_lookup[b]
            pair_rows.append(build_matchup_features(av, bv))
            pair_ids.append((a, b))

    X = np.array(pair_rows, dtype=float)
    if scaler is not None:
        X = scaler.transform(X)
    p = model.predict_proba(X)[:, 1]

    out_df = pd.DataFrame({
        "season": season,
        "team_a": [a for a, _ in pair_ids],
        "team_b": [b for _, b in pair_ids],
        "p_a_wins": p,
    })
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    write_header = not Path(out_csv).exists()
    out_df.to_csv(out_csv, mode="a", index=False, header=write_header)
    return len(out_df)


def run_lr_loso(out_csv: str = DEFAULT_PAIRWISE_OUT) -> dict:
    """22-season LOSO loop using a logistic regression as the stage-1 model.

    For each held-out season, train LR (with StandardScaler + Platt) on
    every-other-season's weighted matchup data, then dump pairwise probs
    for the held-out season's full field to out_csv.
    """
    print("=" * 70)
    print("LR STAGE-1 LOSO TRAINER")
    print("=" * 70)
    inputs = prepare_loso_inputs()
    feature_matrix = inputs["feature_matrix"]
    tourney = inputs["tourney_filtered"]
    regular_results = inputs["regular_results"]
    feature_cols = inputs["feature_cols"]
    top_80_by_season = inputs["top_80_by_season"]

    if Path(out_csv).exists():
        Path(out_csv).unlink()
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)

    seasons = sorted(int(s) for s in tourney["Season"].unique() if int(s) >= 2003)
    diff_cols = expand_feature_cols(feature_cols)
    print(f"  feature_cols: {len(feature_cols)} ({len(diff_cols)} diff cols)")
    print(f"  seasons: {seasons[0]}..{seasons[-1]} ({len(seasons)} seasons)")

    per_season_metrics = []
    overall_start = time.time()

    for holdout in seasons:
        t0 = time.time()
        train_tourney = tourney[tourney["Season"] != holdout]
        test_tourney = tourney[tourney["Season"] == holdout]
        if len(test_tourney) == 0:
            continue

        train_top_ids = set()
        for s in train_tourney["Season"].unique():
            train_top_ids |= top_80_by_season.get(int(s), set())

        train_reg = regular_results[regular_results["Season"] != holdout]
        X_train, y_train, w_train = build_weighted_matchup_data(
            feature_matrix, train_tourney, train_reg, feature_cols,
            top_n_team_ids=train_top_ids,
            supplemental_weight=0.25,
        )
        X_test, y_test, _ = build_matchup_data_from_kaggle(
            feature_matrix, test_tourney, feature_cols
        )

        if X_train.empty or X_test.empty:
            print(f"  [{holdout}] empty train/test, skipping")
            continue

        # Match v3's NaN handling: per-fold medians from training data.
        diff_medians = X_train.median()
        X_train = X_train.fillna(diff_medians).to_numpy(dtype=float)
        X_test = X_test.fillna(diff_medians).to_numpy(dtype=float)

        scaler = StandardScaler().fit(X_train)
        X_train_s = scaler.transform(X_train)
        X_test_s = scaler.transform(X_test)

        # build_weighted_matchup_data and build_matchup_data_from_kaggle
        # may return Series or ndarray for y/w; np.asarray handles both.
        y_train_arr = np.asarray(y_train)
        w_train_arr = np.asarray(w_train, dtype=float)
        y_test_arr = np.asarray(y_test)

        model = fit_lr_with_calibration(
            X_train_s, y_train_arr, w_train_arr, seed=42,
        )
        p_test = model.predict_proba(X_test_s)[:, 1]

        ll = float(sklearn_log_loss(y_test_arr, p_test, labels=[0, 1]))
        acc = float(((p_test > 0.5).astype(int) == y_test_arr).mean())

        # Per-team raw-feature medians from the training-fold seasons,
        # used to NaN-fill any team's raw features before building diffs.
        train_seasons = set(int(s) for s in train_tourney["Season"].unique())
        fm_train = feature_matrix[feature_matrix["Season"].isin(train_seasons)]
        raw_medians_arr = (
            fm_train[feature_cols]
            .apply(pd.to_numeric, errors="coerce")
            .median()
            .to_numpy(dtype=float)
        )
        # Any feature whose training-fold median is itself NaN -> 0.0 fallback.
        raw_medians_arr = np.where(np.isnan(raw_medians_arr), 0.0, raw_medians_arr)

        fm_year = feature_matrix[feature_matrix["Season"] == holdout].set_index("TeamID")
        feature_lookup = {}
        for tid in fm_year.index:
            row = fm_year.loc[tid, feature_cols]
            vals = np.array(
                pd.to_numeric(row, errors="coerce").to_numpy(dtype=float),
                copy=True,
            )
            mask = np.isnan(vals)
            if mask.any():
                vals[mask] = raw_medians_arr[mask]
            feature_lookup[int(tid)] = vals

        field_ids = sorted(set(test_tourney["WTeamID"]) | set(test_tourney["LTeamID"]))

        n_pairs = dump_pairwise_for_season(
            season=holdout,
            field_team_ids=field_ids,
            feature_lookup=feature_lookup,
            scaler=scaler,
            model=model,
            out_csv=out_csv,
        )

        elapsed = time.time() - t0
        per_season_metrics.append({
            "season": holdout,
            "n_test_games": len(y_test),
            "log_loss": ll,
            "accuracy": acc,
            "n_pairs_written": n_pairs,
            "fold_seconds": round(elapsed, 1),
        })
        print(f"  [{holdout}] ll={ll:.4f} acc={acc:.3f} "
              f"pairs={n_pairs} ({elapsed:.1f}s)")

    df = pd.DataFrame(per_season_metrics)
    overall = time.time() - overall_start
    print(f"\nDONE in {overall:.1f}s")
    if len(df):
        wm_ll = (df["log_loss"] * df["n_test_games"]).sum() / df["n_test_games"].sum()
        wm_acc = (df["accuracy"] * df["n_test_games"]).sum() / df["n_test_games"].sum()
        print(f"  weighted-mean log_loss: {wm_ll:.4f}")
        print(f"  weighted-mean accuracy: {wm_acc:.4f}")
    print(f"  pairwise CSV: {out_csv}")
    return {"per_season": df, "out_csv": out_csv}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--out", default=DEFAULT_PAIRWISE_OUT,
        help=f"output pairwise CSV (default: {DEFAULT_PAIRWISE_OUT})"
    )
    args = parser.parse_args(argv)
    run_lr_loso(out_csv=args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
