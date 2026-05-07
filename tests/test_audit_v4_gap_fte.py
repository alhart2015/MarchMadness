"""Unit tests for src/audit_v4_gap_fte.py."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.audit_v4_gap_fte import (
    FTE_RD_COL_FOR_ROUND,
    _bt_norm,
    _calibration_table,
    _compute_bucket_metrics,
    _ece,
    _round_from_daynum,
    _seed_diff_bucket,
    _v4_confidence_quintile,
)


def test_bt_norm_basic():
    assert _bt_norm(0.6, 0.4) == pytest.approx(0.6)
    assert _bt_norm(0.5, 0.5) == pytest.approx(0.5)
    assert _bt_norm(0.8, 0.2) == pytest.approx(0.8)


def test_bt_norm_with_538_rounding():
    """538 publishes rounded probs that don't sum to exactly 1; BT-norm
    handles by dividing by the actual sum."""
    out = _bt_norm(0.601, 0.401)
    assert 0.595 < out < 0.605
    expected = 0.601 / (0.601 + 0.401)
    assert out == pytest.approx(expected)


def test_bt_norm_zero_zero_safe():
    """Both teams have zero R-round survival prob (would never reach R in
    538's view). BT-norm returns 0.5 rather than NaN -- defensive only;
    unreachable for actual played matchups in our pipeline since both
    teams must be alive (rd1_win > 0) to be in the joined dataframe."""
    out = _bt_norm(0.0, 0.0)
    assert out == pytest.approx(0.5)


def test_fte_rd_col_for_round_offsets_by_one():
    """538's rdR_win = P(reach round R), so to audit round-of-X we need
    rd{X+1}_win where X is the spec's round_index 1..6."""
    assert FTE_RD_COL_FOR_ROUND["R64"] == "rd2_win"
    assert FTE_RD_COL_FOR_ROUND["R32"] == "rd3_win"
    assert FTE_RD_COL_FOR_ROUND["S16"] == "rd4_win"
    assert FTE_RD_COL_FOR_ROUND["E8"] == "rd5_win"
    assert FTE_RD_COL_FOR_ROUND["F4"] == "rd6_win"
    assert FTE_RD_COL_FOR_ROUND["Champ"] == "rd7_win"


def test_round_from_daynum_canonical():
    """Same convention as Vegas audit (regression guard if drift)."""
    assert _round_from_daynum(136) == "R64"
    assert _round_from_daynum(138) == "R32"
    assert _round_from_daynum(143) == "S16"
    assert _round_from_daynum(145) == "E8"
    assert _round_from_daynum(152) == "F4"
    assert _round_from_daynum(154) == "Champ"
    assert _round_from_daynum(134) == "FF"


def test_seed_diff_bucket_boundaries():
    assert _seed_diff_bucket(0) == "0-2"
    assert _seed_diff_bucket(2) == "0-2"
    assert _seed_diff_bucket(3) == "3-5"
    assert _seed_diff_bucket(6) == "6-9"
    assert _seed_diff_bucket(10) == "10-15"
    assert _seed_diff_bucket(15) == "10-15"


def test_v4_confidence_quintile_boundaries():
    assert _v4_confidence_quintile(0.55) == "0.50-0.60"
    assert _v4_confidence_quintile(0.61) == "0.60-0.70"
    assert _v4_confidence_quintile(0.95) == "0.90-1.00"
    # Below 0.5 mirrors to favored side
    assert _v4_confidence_quintile(0.40) == "0.50-0.60"
    assert _v4_confidence_quintile(0.05) == "0.90-1.00"


def test_calibration_table_perfect():
    rng = np.random.default_rng(0)
    n = 5000
    p = rng.uniform(0.5, 1.0, size=n)
    y = (rng.random(n) < p).astype(int)
    table = _calibration_table(p, y, n_bins=10)
    assert _ece(table) < 0.03


def test_compute_bucket_metrics_aggregates_correctly():
    """Three games, one bucket; LL + accuracy hand-computed."""
    df = pd.DataFrame([
        {"bucket": "R64", "p_v4": 0.8, "p_fte": 0.7, "winner_is_a": 1},
        {"bucket": "R64", "p_v4": 0.6, "p_fte": 0.55, "winner_is_a": 1},
        {"bucket": "R64", "p_v4": 0.4, "p_fte": 0.5, "winner_is_a": 0},
    ])
    by = _compute_bucket_metrics(df, "bucket")
    cell = by["R64"]
    assert cell["n_games"] == 3
    expected_ll = -np.mean([np.log(0.8), np.log(0.6), np.log(0.6)])
    assert cell["ll_v4"] == pytest.approx(expected_ll)
    # acc_v4: 0.8>=0.5 hit; 0.6>=0.5 hit; 0.4<0.5 hit (winner_is_a=0)
    assert cell["acc_v4"] == pytest.approx(1.0)
    # 538 ll: -mean(log(0.7), log(0.55), log(0.5))
    expected_ll_fte = -np.mean([np.log(0.7), np.log(0.55), np.log(0.5)])
    assert cell["ll_fte"] == pytest.approx(expected_ll_fte)
    assert cell["ll_delta"] == pytest.approx(cell["ll_v4"] - cell["ll_fte"])
