"""Unit tests for src/audit_v4_gap_vegas.py."""
import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from src.audit_v4_gap_vegas import (
    SIGMA,
    _calibration_table,
    _compute_bucket_metrics,
    _ece,
    _extract_seed_number,
    _round_from_daynum,
    _seed_diff_bucket,
    _spread_to_prob,
    _v4_confidence_quintile,
)


def test_spread_to_prob_anchors():
    """SIGMA=11. Standard CBB convention.
    spread=0 -> 0.5; spread=11 -> N(0,1).cdf(1) ~= 0.8413;
    spread=-5.5 -> 1 - N(0,1).cdf(0.5) = 0.3085."""
    assert abs(_spread_to_prob(0.0) - 0.5) < 1e-9
    assert abs(_spread_to_prob(11.0) - norm.cdf(1.0)) < 1e-9
    assert abs(_spread_to_prob(-5.5) - norm.cdf(-0.5)) < 1e-9
    # Symmetry check
    assert abs(_spread_to_prob(7.0) + _spread_to_prob(-7.0) - 1.0) < 1e-9


def test_seed_diff_bucket_boundaries():
    assert _seed_diff_bucket(0) == "0-2"
    assert _seed_diff_bucket(2) == "0-2"
    assert _seed_diff_bucket(3) == "3-5"
    assert _seed_diff_bucket(5) == "3-5"
    assert _seed_diff_bucket(6) == "6-9"
    assert _seed_diff_bucket(9) == "6-9"
    assert _seed_diff_bucket(10) == "10-15"
    assert _seed_diff_bucket(15) == "10-15"


def test_v4_confidence_quintile_boundaries():
    """Mirror probabilities below 0.5 to the favored side."""
    assert _v4_confidence_quintile(0.55) == "0.50-0.60"
    assert _v4_confidence_quintile(0.65) == "0.60-0.70"
    assert _v4_confidence_quintile(0.95) == "0.90-1.00"
    # Mirroring
    assert _v4_confidence_quintile(0.40) == "0.50-0.60"  # 1-0.4=0.6
    assert _v4_confidence_quintile(0.05) == "0.90-1.00"  # 1-0.05=0.95


def test_round_from_daynum():
    assert _round_from_daynum(136) == "R64"
    assert _round_from_daynum(137) == "R64"
    assert _round_from_daynum(138) == "R32"
    assert _round_from_daynum(143) == "S16"
    assert _round_from_daynum(145) == "E8"
    assert _round_from_daynum(152) == "F4"
    assert _round_from_daynum(154) == "Champ"
    assert _round_from_daynum(134) == "FF"
    # Out of band -> OTHER
    assert _round_from_daynum(100) == "OTHER"


def test_extract_seed_number():
    """Seed string like 'W01', 'X16a' -> integer seed 1, 16."""
    assert _extract_seed_number("W01") == 1
    assert _extract_seed_number("X16") == 16
    assert _extract_seed_number("Y16a") == 16
    assert _extract_seed_number("Z16b") == 16
    assert _extract_seed_number("W08") == 8


def test_calibration_table_well_calibrated():
    """A perfectly calibrated synthetic dataset has small ECE."""
    rng = np.random.default_rng(0)
    n = 5000
    p = rng.uniform(0.5, 1.0, size=n)
    y = (rng.random(n) < p).astype(int)
    table = _calibration_table(p, y, n_bins=10)
    ece = _ece(table)
    assert ece < 0.03


def test_calibration_table_overconfident():
    """A model that always predicts 0.9 but only wins 0.7 of the time
    has high ECE."""
    n = 1000
    p = np.full(n, 0.9)
    y = np.zeros(n, dtype=int)
    y[:700] = 1
    table = _calibration_table(p, y, n_bins=10)
    ece = _ece(table)
    assert ece > 0.15


def test_compute_bucket_metrics_three_games():
    df = pd.DataFrame([
        {"bucket": "R64", "p_v4": 0.8, "p_vegas": 0.7, "winner_is_a": 1},
        {"bucket": "R64", "p_v4": 0.6, "p_vegas": 0.55, "winner_is_a": 1},
        {"bucket": "R64", "p_v4": 0.4, "p_vegas": 0.5, "winner_is_a": 0},
    ])
    by_bucket = _compute_bucket_metrics(df, "bucket")
    cell = by_bucket["R64"]
    assert cell["n_games"] == 3
    expected_ll = -np.mean([np.log(0.8), np.log(0.6), np.log(0.6)])
    assert abs(cell["ll_v4"] - expected_ll) < 1e-9
    # All three picks correct (chalk = winner): acc_v4 = 1.0.
    assert cell["acc_v4"] == pytest.approx(1.0)


def test_compute_bucket_metrics_two_buckets():
    df = pd.DataFrame([
        {"bucket": "A", "p_v4": 0.8, "p_vegas": 0.7, "winner_is_a": 1},
        {"bucket": "A", "p_v4": 0.7, "p_vegas": 0.6, "winner_is_a": 0},
        {"bucket": "B", "p_v4": 0.6, "p_vegas": 0.55, "winner_is_a": 1},
        {"bucket": "B", "p_v4": 0.4, "p_vegas": 0.5, "winner_is_a": 0},
    ])
    by_bucket = _compute_bucket_metrics(df, "bucket")
    assert by_bucket["A"]["n_games"] == 2
    assert by_bucket["B"]["n_games"] == 2
    # ll_delta computed = ll_v4 - ll_vegas
    for cell in by_bucket.values():
        assert abs(cell["ll_delta"] - (cell["ll_v4"] - cell["ll_vegas"])) < 1e-12


def test_sigma_constant_eq_eleven():
    """Regression guard: SIGMA=11 matches existing src/blend_sweep.py
    + src/alternate_bracket.py."""
    assert SIGMA == 11.0
