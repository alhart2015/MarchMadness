"""Tests for Optuna hyperparameter tuning."""

import numpy as np
import pandas as pd
import pytest

from src.models.tuning import tune_hyperparameters


@pytest.fixture
def training_data():
    np.random.seed(42)
    n = 200
    X = pd.DataFrame({
        "adj_em": np.random.randn(n) * 10,
        "seed": np.random.randn(n) * 3,
    })
    y = pd.Series((X["adj_em"] + np.random.randn(n) * 3 > 0).astype(int), name="win")
    return X, y


def test_tune_returns_params(training_data):
    X, y = training_data
    best_params = tune_hyperparameters(X, y, n_trials=5, random_seed=42)
    assert isinstance(best_params, dict)
    assert "max_depth" in best_params
    assert "learning_rate" in best_params


def test_tune_params_in_range(training_data):
    X, y = training_data
    best_params = tune_hyperparameters(X, y, n_trials=5, random_seed=42)
    assert 2 <= best_params["max_depth"] <= 8
    assert 0.01 <= best_params["learning_rate"] <= 0.3
