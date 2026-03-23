"""Tests for model training and calibration."""

import numpy as np
import pandas as pd
import pytest

from src.models.train import train_model, predict_matchup


@pytest.fixture
def training_data():
    """Synthetic matchup data with clear signal."""
    np.random.seed(42)
    n = 200
    # Feature: positive = team A is stronger
    X = pd.DataFrame({
        "adj_em": np.random.randn(n) * 10,
        "seed": np.random.randn(n) * 3,
    })
    # Label: team A wins when features are positive (with noise)
    y = pd.Series((X["adj_em"] + np.random.randn(n) * 3 > 0).astype(int), name="win")
    return X, y


def test_train_model_returns_pipeline(training_data):
    X, y = training_data
    model = train_model(X, y, random_seed=42)
    assert hasattr(model, "predict_proba")


def test_predict_matchup_returns_probability(training_data):
    X, y = training_data
    model = train_model(X, y, random_seed=42)
    prob = predict_matchup(model, X.iloc[[0]])
    assert 0.0 <= prob <= 1.0


def test_model_predicts_strong_team_wins(training_data):
    X, y = training_data
    model = train_model(X, y, random_seed=42)
    # Very strong team A: should have high win prob
    strong = pd.DataFrame({"adj_em": [25.0], "seed": [-5.0]})
    prob = predict_matchup(model, strong)
    assert prob > 0.7

    # Very weak team A: should have low win prob
    weak = pd.DataFrame({"adj_em": [-25.0], "seed": [5.0]})
    prob = predict_matchup(model, weak)
    assert prob < 0.3


def test_train_model_with_sample_weights(training_data):
    X, y = training_data
    weights = np.ones(len(y))
    weights[:len(y)//2] = 0.25  # first half weighted lower
    model = train_model(X, y, sample_weight=weights)
    assert hasattr(model, "predict_proba")
