"""CLI entry point: python -m src.models"""

import logging
from pathlib import Path

import pandas as pd

from src.config import load_config
from src.models.matchup import build_matchup_data
from src.models.train import train_model, save_model
from src.models.evaluate import leave_one_season_out_cv
from src.models.tuning import tune_hyperparameters

logging.basicConfig(level=logging.INFO, format="%(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

FEATURE_COLS = [
    "adj_oe", "adj_de", "adj_em", "adj_tempo",
    "off_efg", "off_to_rate", "off_or_rate", "off_ft_rate",
    "def_efg", "def_to_rate", "def_or_rate", "def_ft_rate",
    "seed", "massey_composite_rank",
    "win_pct_last_10", "road_win_pct",
]


def main() -> None:
    config = load_config()
    processed_dir = Path(config["data"]["processed_dir"])

    feature_matrix = pd.read_parquet(processed_dir / "feature_matrix.parquet")
    tourney_results = pd.read_parquet(processed_dir / "tourney_results.parquet")

    # Filter feature cols to those actually present
    available_cols = [c for c in FEATURE_COLS if c in feature_matrix.columns]
    logger.info("Using %d features: %s", len(available_cols), available_cols)

    # Build training data
    X, y = build_matchup_data(feature_matrix, tourney_results, available_cols)

    # Hyperparameter tuning
    logger.info("Tuning hyperparameters with Optuna (50 trials)")
    best_params = tune_hyperparameters(X, y, n_trials=50, random_seed=config["model"]["random_seed"])
    logger.info("Best params: %s", best_params)

    # Evaluate via LOSO CV with tuned params
    logger.info("Running leave-one-season-out cross-validation")
    cv_results = leave_one_season_out_cv(
        feature_matrix, tourney_results, available_cols,
        random_seed=config["model"]["random_seed"],
        xgb_params=best_params,
    )
    logger.info(
        "CV Results: log_loss=%.4f, accuracy=%.3f, auc=%.3f",
        cv_results["mean_log_loss"],
        cv_results["mean_accuracy"],
        cv_results["mean_auc"],
    )

    # Train final model on all data with tuned params
    logger.info("Training final model on all seasons")
    model = train_model(X, y, random_seed=config["model"]["random_seed"], xgb_params=best_params)

    seasons = sorted(feature_matrix["Season"].unique().tolist())
    save_model(model, "models", config, available_cols, seasons)

    logger.info("Model training complete")


if __name__ == "__main__":
    main()
