"""Hierarchical Bradley-Terry stage-1 trainer with v4 feature priors.

Spec: docs/superpowers/specs/2026-05-03-hierarchical-bt-feature-priors-design.md

For each (sigma, holdout_season) cell:
  1. Compute per-LOSO-fold standardization stats (mean/std over
     Season != holdout) on the v4 feature columns.
  2. Fit hierarchical BT MAP for that season:
        s_team_i ~ Normal(beta . v4_features_team_i, sigma^2)
        P(a beats b) = sigmoid(s[a] - s[b] + home * h)
     via L-BFGS-B on the joint negative log posterior.
  3. Compute pairwise predictions for that season's tournament field
     and append to output/pairwise_hbt_sigma_<S>.csv (one CSV per sigma).

Reuses prepare_loso_inputs() from src/enhanced_model_v3 to get the
byte-identical v4 feature matrix.

Output schema (matches pairwise_v4.csv and pairwise_bt.csv):
    season, team_a, team_b, p_a_wins      with team_a < team_b.
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.features.hierarchical_bt import (
    fit_one_season,
    predict_pairs,
)

DEFAULT_SIGMAS = [0.05, 0.10, 0.20, 0.50, 1.00, 2.00, 5.00]
DEFAULT_OUT_DIR = "output"


def _format_sigma(sigma: float) -> str:
    """Filename-safe sigma label. 1.0 -> '1.00', 0.05 -> '0.05'."""
    return f"{sigma:.2f}"


def _filter_numeric_feature_cols(
    feature_matrix: pd.DataFrame, feature_cols: Sequence[str]
) -> list[str]:
    """Drop any column that is non-numeric or has all-NaN values."""
    keep = []
    for c in feature_cols:
        if c not in feature_matrix.columns:
            continue
        col = feature_matrix[c]
        if not pd.api.types.is_numeric_dtype(col):
            continue
        if col.isna().all():
            continue
        keep.append(c)
    return keep


def _standardize_stats(
    train_fm: pd.DataFrame, feature_cols: Sequence[str]
) -> tuple[pd.Series, pd.Series, list[str]]:
    """Compute (mean, std) over training-fold rows. Drops zero-variance
    columns (they would create NaN/inf after standardization)."""
    means = train_fm[list(feature_cols)].mean()
    stds = train_fm[list(feature_cols)].std(ddof=0)
    keep = [c for c in feature_cols if stds.get(c, 0) > 1e-12]
    return means[keep], stds[keep], keep


def _season_field(
    tourney_filtered: pd.DataFrame, season: int
) -> list[int]:
    """Tournament field for a season: the unique TeamIDs across both
    sides of MNCAATourneyCompactResults rows in that season."""
    sub = tourney_filtered[tourney_filtered["Season"] == season]
    return sorted(
        set(sub["WTeamID"].astype(int)) | set(sub["LTeamID"].astype(int))
    )


def _impute_features_for_season(
    fm_season: pd.DataFrame, feature_cols: Sequence[str], train_fm: pd.DataFrame
) -> pd.DataFrame:
    """Fill NaNs in the season's feature rows with train-fold medians.
    Matches v4's apply-time imputation behavior."""
    medians = train_fm[list(feature_cols)].median()
    return fm_season.copy().fillna(medians)


def run_hbt_loso(
    feature_matrix: pd.DataFrame,
    regular_results: pd.DataFrame,
    tourney_filtered: pd.DataFrame,
    feature_cols: Sequence[str],
    sigmas: Iterable[float],
    seasons: Iterable[int] | None = None,
    sigma_beta: float = 1.0,
    out_dir: str | Path = DEFAULT_OUT_DIR,
    verbose: bool = True,
) -> dict:
    """Per-(sigma, season) hierarchical BT fits. Writes one CSV per
    sigma into out_dir. Returns a per-(sigma, season) summary dict.

    Args:
        feature_matrix: From prepare_loso_inputs(). Has TeamID, Season,
                        and all v4 feature columns.
        regular_results: data["reg_season"] -- regular-season game rows.
        tourney_filtered: tournament results filtered to seasons present
                          in feature_matrix.
        feature_cols:   v4 feature columns (from prepare_loso_inputs).
        sigmas:         iterable of sigma values to sweep.
        seasons:        Optional list of seasons to fit. If None, uses
                        all seasons in feature_matrix.
        sigma_beta:     L2 prior std on beta. Default 1.0.
        out_dir:        Directory to write per-sigma CSVs.
        verbose:        Per-season log lines.
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    feature_cols = _filter_numeric_feature_cols(feature_matrix, feature_cols)
    if not feature_cols:
        raise ValueError("no numeric feature columns left after filtering")

    if seasons is None:
        seasons = sorted(feature_matrix["Season"].unique())
    seasons = sorted(int(s) for s in seasons)
    sigmas = list(sigmas)

    summary = []
    for sigma in sigmas:
        out_csv = out_path / f"pairwise_hbt_sigma_{_format_sigma(sigma)}.csv"
        if out_csv.exists():
            out_csv.unlink()

        if verbose:
            print("=" * 70)
            print(f"  SIGMA = {sigma:.4f}  ->  {out_csv}")
            print("=" * 70)

        sigma_t0 = time.time()
        all_rows = []
        for season in seasons:
            t0 = time.time()
            train_fm = feature_matrix[feature_matrix["Season"] != season]
            season_fm = feature_matrix[feature_matrix["Season"] == season]

            if len(season_fm) == 0:
                if verbose:
                    print(f"  [{season}] no feature rows, skipping")
                continue

            means, stds, feat_keep = _standardize_stats(train_fm, feature_cols)
            season_fm_imputed = _impute_features_for_season(
                season_fm, feat_keep, train_fm
            )

            season_games = regular_results[regular_results["Season"] == season]
            if len(season_games) == 0:
                if verbose:
                    print(f"  [{season}] no regular-season games, skipping")
                continue

            team_ids = sorted(
                set(season_games["WTeamID"].astype(int))
                | set(season_games["LTeamID"].astype(int))
            )
            team_ids_with_features = set(season_fm_imputed["TeamID"].astype(int))
            team_ids = [t for t in team_ids if t in team_ids_with_features]
            if not team_ids:
                if verbose:
                    print(f"  [{season}] no overlap with feature matrix, skipping")
                continue

            valid_games = season_games[
                season_games["WTeamID"].astype(int).isin(team_ids_with_features)
                & season_games["LTeamID"].astype(int).isin(team_ids_with_features)
            ]

            try:
                fit = fit_one_season(
                    valid_games,
                    team_ids,
                    season_fm_imputed,
                    feat_keep,
                    means,
                    stds,
                    sigma=sigma,
                    sigma_beta=sigma_beta,
                )
            except Exception as e:
                print(f"  [{season}] FIT ERROR: {e}")
                summary.append({
                    "sigma": sigma, "season": season, "success": False,
                    "error": str(e),
                })
                continue

            field = _season_field(tourney_filtered, season)
            field = [t for t in field if t in set(team_ids)]
            field_sorted = sorted(field)
            pairs = []
            for i in range(len(field_sorted)):
                for j in range(i + 1, len(field_sorted)):
                    pairs.append((field_sorted[i], field_sorted[j]))
            if not pairs:
                if verbose:
                    print(f"  [{season}] no tournament field, skipping pairs")
                continue

            probs = predict_pairs(fit, pairs)
            for (a, b), p in zip(pairs, probs):
                all_rows.append({
                    "season": season, "team_a": a, "team_b": b,
                    "p_a_wins": float(p),
                })

            elapsed = time.time() - t0
            summary.append({
                "sigma": sigma,
                "season": season,
                "n_teams": len(team_ids),
                "n_games": len(valid_games),
                "n_features": len(feat_keep),
                "n_pairs": len(pairs),
                "h": fit["h"],
                "n_iter": fit["n_iter"],
                "fun": fit["fun"],
                "success": fit["success"],
                "fold_seconds": round(elapsed, 2),
            })
            if verbose:
                print(
                    f"  [{season}] teams={len(team_ids):>3} "
                    f"games={len(valid_games):>5} feat={len(feat_keep):>2} "
                    f"h={fit['h']:>+5.3f} iter={fit['n_iter']:>3} "
                    f"fun={fit['fun']:.1f} pairs={len(pairs):>5} "
                    f"({elapsed:.1f}s)"
                )

        out_df = pd.DataFrame(all_rows, columns=["season", "team_a", "team_b", "p_a_wins"])
        out_df.to_csv(out_csv, index=False)
        if verbose:
            print(f"  -- wrote {len(out_df):,} pairs in {time.time() - sigma_t0:.1f}s")

    return {"per_cell": pd.DataFrame(summary)}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--sigmas",
        default=",".join(f"{s:.2f}" for s in DEFAULT_SIGMAS),
        help="Comma-separated list of sigma values.",
    )
    parser.add_argument(
        "--seasons",
        default="",
        help="Comma-separated season list. Empty = all seasons in the feature matrix.",
    )
    parser.add_argument(
        "--sigma-beta",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--out-dir",
        default=DEFAULT_OUT_DIR,
    )
    args = parser.parse_args(argv)

    sigmas = [float(x) for x in args.sigmas.split(",") if x.strip()]
    seasons: list[int] | None
    if args.seasons.strip():
        seasons = [int(x) for x in args.seasons.split(",") if x.strip()]
    else:
        seasons = None

    print("=" * 70)
    print("HIERARCHICAL BT STAGE-1 TRAINER (v4 feature priors)")
    print("=" * 70)
    print(f"  sigmas      : {sigmas}")
    print(f"  sigma_beta  : {args.sigma_beta}")
    print(f"  seasons     : {seasons or 'all'}")
    print(f"  out_dir     : {args.out_dir}")

    overall_t0 = time.time()

    from src.enhanced_model_v3 import prepare_loso_inputs

    inputs = prepare_loso_inputs()
    run_hbt_loso(
        feature_matrix=inputs["feature_matrix"],
        regular_results=inputs["regular_results"],
        tourney_filtered=inputs["tourney_filtered"],
        feature_cols=inputs["feature_cols"],
        sigmas=sigmas,
        seasons=seasons,
        sigma_beta=args.sigma_beta,
        out_dir=args.out_dir,
    )

    print(f"\nDONE in {time.time() - overall_t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
