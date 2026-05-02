"""Smoke + unit tests for src/train_lr_stage1.py.

The full 22-season LOSO is a separate run (Task 4), not exercised here.
Tests below verify the per-fold training function on synthetic data and
the pairwise-dump function on a tiny field.
"""
import numpy as np
import pandas as pd
import pytest


def test_fit_lr_with_calibration_returns_probabilities():
    """Per-fold trainer produces a calibrated classifier whose predict_proba
    yields values in [0, 1] for each class."""
    from src.train_lr_stage1 import fit_lr_with_calibration

    rng = np.random.default_rng(42)
    X = rng.standard_normal((200, 5))
    # Linearly separable signal so a working LR can fit non-trivially.
    y = (X[:, 0] - X[:, 1] > 0).astype(int)
    w = np.ones(len(y))

    model = fit_lr_with_calibration(X, y, w, seed=42)
    p = model.predict_proba(X)
    assert p.shape == (200, 2)
    assert np.all((p >= 0.0) & (p <= 1.0))
    # Calibration shouldn't kill discrimination: roughly correct on training.
    acc = float(((p[:, 1] > 0.5) == y).mean())
    assert acc > 0.7, f"trainer should learn the linear separation: acc={acc}"


def test_fit_lr_handles_unbalanced_weights():
    """Sample weights must propagate -- weight-0 rows should be ignored
    even when their labels disagree with the weight-1 majority. Sized
    above the inner-CV fold count so GridSearchCV has enough samples."""
    from src.train_lr_stage1 import fit_lr_with_calibration

    # Weight-1 rows form a clean separation: X=1 -> 1, X=-1 -> 0.
    # Weight-0 rows have flipped labels (X=1 -> 0, X=-1 -> 1) and must be
    # ignored. 30 rows total -> plenty for the 5-fold inner CV.
    X = np.array([[1.0]] * 10 + [[-1.0]] * 10 + [[1.0]] * 5 + [[-1.0]] * 5)
    y = np.array([1] * 10 + [0] * 10 + [0] * 5 + [1] * 5)
    w = np.array([1.0] * 20 + [0.0] * 10)

    model = fit_lr_with_calibration(X, y, w, seed=42)
    p = model.predict_proba(np.array([[1.0], [-1.0]]))[:, 1]
    assert p[0] > 0.5
    assert p[1] < 0.5


def test_dump_pairwise_writes_expected_schema(tmp_path):
    """The pairwise-dump helper writes (season, team_a, team_b, p_a_wins)
    rows for the cartesian field, with team_a < team_b canonicalization."""
    from src.train_lr_stage1 import dump_pairwise_for_season

    feature_lookup = {
        1101: np.array([0.5, 0.1]),
        1102: np.array([-0.2, 0.3]),
        1103: np.array([0.0, -0.1]),
    }

    class FakeModel:
        def predict_proba(self, X):
            # Map first column to a probability via clip; deterministic.
            p = np.clip(0.5 + X[:, 0] / 2, 0.0, 1.0)
            return np.column_stack([1 - p, p])

    out = tmp_path / "pw.csv"
    n = dump_pairwise_for_season(
        season=2003,
        field_team_ids=[1101, 1102, 1103],
        feature_lookup=feature_lookup,
        scaler=None,  # bypass scaling for this synthetic test
        model=FakeModel(),
        out_csv=str(out),
    )

    assert n == 3  # 3 unordered pairs from 3 teams

    df = pd.read_csv(out)
    assert list(df.columns) == ["season", "team_a", "team_b", "p_a_wins"]
    assert len(df) == 3
    assert (df["team_a"] < df["team_b"]).all()
    assert (df["season"] == 2003).all()
    assert df["p_a_wins"].between(0, 1).all()
