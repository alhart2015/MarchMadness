"""Baseline models for comparison."""

import numpy as np


def seed_baseline_prob(seed_a: int, seed_b: int) -> float:
    """Predict P(team A wins) based purely on seed difference.

    Uses a logistic function fitted to historical seed-vs-seed outcomes.
    Coefficient of ~0.15 per seed difference is a reasonable approximation.
    """
    seed_diff = seed_b - seed_a  # positive means A is better seed
    return float(1 / (1 + np.exp(-0.15 * seed_diff)))
