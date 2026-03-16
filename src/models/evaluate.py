"""Model evaluation with leave-one-season-out CV."""

import logging

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss as sklearn_log_loss, roc_auc_score

from src.models.matchup import build_matchup_data
from src.models.train import train_model

logger = logging.getLogger(__name__)


def compute_log_loss(y_true: pd.Series, y_prob: np.ndarray) -> float:
    """Compute log loss (cross-entropy)."""
    return float(sklearn_log_loss(y_true, y_prob))


def compute_brier_score(y_true: pd.Series, y_prob: np.ndarray) -> float:
    """Compute Brier score (mean squared error of probabilities)."""
    return float(np.mean((y_prob - y_true.values) ** 2))


def leave_one_season_out_cv(
    feature_matrix: pd.DataFrame,
    tourney_results: pd.DataFrame,
    feature_cols: list[str],
    random_seed: int = 42,
    xgb_params: dict | None = None,
) -> dict:
    """Run leave-one-season-out cross-validation.

    For each season, trains on all other seasons and evaluates
    on the held-out season's tournament games.

    Returns dict with per-season and aggregate metrics.
    """
    seasons = sorted(tourney_results["Season"].unique())
    results = []

    for holdout_season in seasons:
        # Split
        train_tourney = tourney_results[tourney_results["Season"] != holdout_season]
        test_tourney = tourney_results[tourney_results["Season"] == holdout_season]

        if len(test_tourney) == 0:
            continue

        X_train, y_train = build_matchup_data(feature_matrix, train_tourney, feature_cols)
        X_test, y_test = build_matchup_data(feature_matrix, test_tourney, feature_cols)

        if len(X_train) == 0 or len(X_test) == 0:
            continue

        model = train_model(X_train, y_train, random_seed=random_seed, xgb_params=xgb_params)
        y_prob = model.predict_proba(X_test)[:, 1]

        season_loss = compute_log_loss(y_test, y_prob)
        season_brier = compute_brier_score(y_test, y_prob)
        season_acc = float((y_prob.round() == y_test).mean())
        season_auc = float(roc_auc_score(y_test, y_prob))

        results.append({
            "season": holdout_season,
            "log_loss": season_loss,
            "brier_score": season_brier,
            "accuracy": season_acc,
            "auc": season_auc,
            "n_games": len(test_tourney),
        })
        logger.info("Season %d: log_loss=%.4f, brier=%.4f, acc=%.3f, auc=%.3f", holdout_season, season_loss, season_brier, season_acc, season_auc)

    results_df = pd.DataFrame(results)
    return {
        "per_season": results_df,
        "mean_log_loss": float(results_df["log_loss"].mean()),
        "mean_brier_score": float(results_df["brier_score"].mean()),
        "mean_accuracy": float(results_df["accuracy"].mean()),
        "mean_auc": float(results_df["auc"].mean()),
    }
