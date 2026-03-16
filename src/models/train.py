"""XGBoost model training with Platt calibration."""

import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold

logger = logging.getLogger(__name__)


def train_model(
    X: pd.DataFrame,
    y: pd.Series,
    random_seed: int = 42,
    xgb_params: dict | None = None,
) -> CalibratedClassifierCV:
    """Train XGBoost classifier with Platt scaling calibration.

    Returns a CalibratedClassifierCV wrapping the XGBoost model.
    """
    params = {
        "n_estimators": 300,
        "max_depth": 4,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": random_seed,
        "eval_metric": "logloss",
    }
    if xgb_params:
        params.update(xgb_params)

    base_model = xgb.XGBClassifier(**params)

    # Platt scaling via 5-fold CV
    calibrated = CalibratedClassifierCV(
        base_model, method="sigmoid", cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=random_seed)
    )
    calibrated.fit(X, y)

    logger.info("Model trained: %d samples, %d features", len(X), X.shape[1])
    return calibrated


def predict_matchup(model: CalibratedClassifierCV, X: pd.DataFrame) -> float:
    """Predict P(team A wins) for a single matchup feature vector."""
    proba = model.predict_proba(X)
    return float(proba[0, 1])


def save_model(
    model: CalibratedClassifierCV,
    output_dir: str,
    config: dict,
    feature_cols: list[str],
    seasons: list[int],
) -> Path:
    """Save model + metadata sidecar JSON."""
    import joblib

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    config_hash = hashlib.md5(json.dumps(config, sort_keys=True).encode()).hexdigest()[:8]
    year = max(seasons)
    model_name = f"xgb_{year}_{config_hash}"

    model_path = output_path / f"{model_name}.pkl"
    meta_path = output_path / f"{model_name}_meta.json"

    joblib.dump(model, model_path)

    meta = {
        "training_date": datetime.now().isoformat(),
        "config_hash": config_hash,
        "seasons": seasons,
        "feature_cols": feature_cols,
        "n_features": len(feature_cols),
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    logger.info("Model saved: %s", model_path)
    return model_path
