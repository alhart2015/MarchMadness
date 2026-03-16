"""Tests for model evaluation."""

import numpy as np
import pandas as pd
import pytest

from src.models.evaluate import compute_log_loss, compute_brier_score, leave_one_season_out_cv
from src.models.baselines import seed_baseline_prob


def test_compute_log_loss_perfect():
    y_true = pd.Series([1, 0, 1])
    y_prob = np.array([0.99, 0.01, 0.99])
    loss = compute_log_loss(y_true, y_prob)
    assert loss < 0.05


def test_compute_log_loss_random():
    y_true = pd.Series([1, 0, 1, 0])
    y_prob = np.array([0.5, 0.5, 0.5, 0.5])
    loss = compute_log_loss(y_true, y_prob)
    assert abs(loss - 0.693) < 0.01  # ln(2) ≈ 0.693


def test_compute_brier_score_perfect():
    y_true = pd.Series([1, 0, 1])
    y_prob = np.array([1.0, 0.0, 1.0])
    score = compute_brier_score(y_true, y_prob)
    assert score == 0.0


def test_compute_brier_score_random():
    y_true = pd.Series([1, 0, 1, 0])
    y_prob = np.array([0.5, 0.5, 0.5, 0.5])
    score = compute_brier_score(y_true, y_prob)
    assert abs(score - 0.25) < 0.01


def test_seed_baseline_prob():
    # 1-seed vs 16-seed: should strongly favor 1-seed
    prob = seed_baseline_prob(1, 16)
    assert prob > 0.9

    # Equal seeds: should be ~0.5
    prob = seed_baseline_prob(8, 8)
    assert abs(prob - 0.5) < 0.01

    # 16-seed vs 1-seed: should strongly favor 1-seed (low prob for team A)
    prob = seed_baseline_prob(16, 1)
    assert prob < 0.1
