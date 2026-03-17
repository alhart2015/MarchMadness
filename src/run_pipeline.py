"""End-to-end pipeline: load data → features → train → evaluate.

Uses the alternative KenPom/Barttorvik dataset (data/raw/kaggle/).

Usage
-----
    python -m src.run_pipeline
or
    python src/run_pipeline.py
"""

import logging
import sys
from pathlib import Path

import pandas as pd

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Resolve project root (works whether run as module or script) ───────────────
_HERE = Path(__file__).resolve().parent          # src/
_ROOT = _HERE.parent                             # project root
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

KAGGLE_DIR = _ROOT / "data" / "raw" / "kaggle"


def main() -> None:
    # ── Step 1: Load raw data ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 1 — Loading raw data")
    print("=" * 60)

    from src.ingest.kaggle2026_loader import load_kaggle2026_data
    data = load_kaggle2026_data(str(KAGGLE_DIR))

    kenpom      = data["kenpom"]
    matchups    = data["matchups"]
    resumes     = data["resumes"]
    ratings_538 = data["ratings_538"]

    print(f"  KenPom/Barttorvik : {len(kenpom):,} rows  (seasons {kenpom['YEAR'].min()}–{kenpom['YEAR'].max()})")
    print(f"  Tournament Matchups: {len(matchups):,} rows")
    print(f"  Resumes            : {len(resumes):,} rows")
    print(f"  538 Ratings        : {len(ratings_538):,} rows")

    # ── Step 2: Build tournament results (winner/loser pairs) ─────────────────
    print("\n" + "=" * 60)
    print("STEP 2 — Building tournament results")
    print("=" * 60)

    from src.ingest.build_tournament_results import build_tournament_results
    tourney_results = build_tournament_results(matchups)

    print(f"  Games parsed: {len(tourney_results):,}")
    print(f"  Seasons     : {sorted(tourney_results['Season'].unique())}")

    # ── Step 3: Build feature matrix ─────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 3 — Building feature matrix")
    print("=" * 60)

    from src.features.feature_matrix_v2 import build_feature_matrix_v2, get_feature_cols
    feature_matrix = build_feature_matrix_v2(kenpom, resumes, ratings_538)

    feature_cols = get_feature_cols(feature_matrix)
    print(f"  Feature matrix  : {len(feature_matrix):,} rows × {len(feature_matrix.columns)} cols")
    print(f"  Feature columns : {len(feature_cols)}")
    print(f"  Seasons covered : {int(feature_matrix['Season'].min())}–{int(feature_matrix['Season'].max())}")

    # Keep only seasons that appear in tourney results for training (exclude 2026)
    training_seasons = sorted(tourney_results["Season"].unique())
    tourney_fm = feature_matrix[feature_matrix["Season"].isin(training_seasons)]
    print(f"  Training seasons: {training_seasons}")

    # ── Step 4: Build matchup training data ───────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 4 — Building matchup training data")
    print("=" * 60)

    from src.models.matchup import build_matchup_data

    # Drop columns with too many NaNs in training data (> 20 % of rows)
    nan_thresh = 0.20
    pre_drop_cols = feature_cols[:]
    tmp_X, _ = build_matchup_data(tourney_fm, tourney_results, feature_cols)
    if not tmp_X.empty:
        null_fracs = tmp_X.isna().mean()
        drop_cols = null_fracs[null_fracs > nan_thresh].index.tolist()
        if drop_cols:
            print(f"  Dropping {len(drop_cols)} columns with >{nan_thresh*100:.0f}% NaN: {drop_cols}")
            feature_cols = [c for c in feature_cols if c not in drop_cols]

    X_all, y_all = build_matchup_data(tourney_fm, tourney_results, feature_cols)

    # Fill remaining NaNs with column median
    X_all = X_all.fillna(X_all.median())

    print(f"  Training samples: {len(X_all):,}  (balanced: {y_all.mean():.3f} win rate)")
    print(f"  Features used   : {len(feature_cols)}")

    # ── Step 5: Hyperparameter tuning (Optuna) ────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 5 — Optuna hyperparameter tuning (n_trials=20)")
    print("=" * 60)

    from src.models.tuning import tune_hyperparameters
    best_params = tune_hyperparameters(X_all, y_all, n_trials=20, random_seed=42)
    print(f"  Best params: {best_params}")

    # ── Step 6: Train final model ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 6 — Training final model on all seasons")
    print("=" * 60)

    from src.models.train import train_model
    model = train_model(X_all, y_all, random_seed=42, xgb_params=best_params)
    print("  Model trained successfully.")

    # ── Step 7: Leave-one-season-out CV evaluation ────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 7 — Leave-one-season-out cross-validation")
    print("=" * 60)

    from src.models.evaluate import leave_one_season_out_cv

    def _prep_matchup_df(fm: pd.DataFrame, tr: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Helper: fill NaN after building matchup data."""
        X, y = build_matchup_data(fm, tr, feature_cols)
        X = X.fillna(X.median() if not X.empty else 0)
        return X, y

    # We need to hook into leave_one_season_out_cv but it calls build_matchup_data
    # internally, so we pre-fill NaN in feature_matrix instead.
    # Strategy: fill NaN in feature_matrix with per-column median before CV.
    fm_filled = tourney_fm.copy()
    for col in feature_cols:
        if col in fm_filled.columns:
            fm_filled[col] = fm_filled[col].fillna(fm_filled[col].median())

    cv_results = leave_one_season_out_cv(
        feature_matrix=fm_filled,
        tourney_results=tourney_results,
        feature_cols=feature_cols,
        random_seed=42,
        xgb_params=best_params,
    )

    # ── Step 8: Print results ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS — Leave-One-Season-Out CV")
    print("=" * 60)

    per_season = cv_results["per_season"]
    print(f"\n{'Season':>8}  {'LogLoss':>9}  {'Brier':>7}  {'Accuracy':>9}  {'AUC':>7}  {'#Games':>7}")
    print("-" * 58)
    for _, row in per_season.sort_values("season").iterrows():
        print(
            f"  {int(row['season']):>6}  {row['log_loss']:>9.4f}  "
            f"{row['brier_score']:>7.4f}  {row['accuracy']:>9.3f}  "
            f"{row['auc']:>7.4f}  {int(row['n_games']):>7}"
        )

    print("-" * 58)
    print(
        f"  {'MEAN':>6}  {cv_results['mean_log_loss']:>9.4f}  "
        f"{cv_results['mean_brier_score']:>7.4f}  {cv_results['mean_accuracy']:>9.3f}  "
        f"{cv_results['mean_auc']:>7.4f}"
    )

    print("\nSummary:")
    print(f"  Mean Log Loss  : {cv_results['mean_log_loss']:.4f}")
    print(f"  Mean Brier     : {cv_results['mean_brier_score']:.4f}")
    print(f"  Mean Accuracy  : {cv_results['mean_accuracy']:.3f}")
    print(f"  Mean AUC       : {cv_results['mean_auc']:.4f}")
    print()

    # ── Optional: 2026 predictions ────────────────────────────────────────────
    fm_2026 = feature_matrix[feature_matrix["Season"] == 2026].copy()
    if not fm_2026.empty and fm_2026["SEED"].notna().any():
        tourney_2026 = fm_2026[fm_2026["SEED"].notna() & (fm_2026["SEED"] > 0)]
        if not tourney_2026.empty:
            print("=" * 60)
            print(f"2026 TOURNAMENT TEAMS IN FEATURE MATRIX: {len(tourney_2026)}")
            print("=" * 60)
            sample = tourney_2026[["TeamName", "SEED", "KADJ EM", "BARTHAG"]].sort_values("SEED")
            print(sample.head(20).to_string(index=False))
            print()


if __name__ == "__main__":
    main()
