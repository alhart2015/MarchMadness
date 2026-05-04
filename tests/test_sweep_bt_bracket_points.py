"""Unit tests for src/sweep_bt_bracket_points.py."""
from pathlib import Path

import pandas as pd
import pytest

from src.sweep_bt_bracket_points import (
    _anchor_check,
    _format_w,
    _make_weight_pair,
    _score_pairwise,
)


def test_make_weight_pair_complements_to_one():
    assert _make_weight_pair(0.6) == (0.6, 0.4)
    assert _make_weight_pair(1.0) == (1.0, 0.0)
    assert _make_weight_pair(0.95) == (0.95, 0.05)
    assert _make_weight_pair(0.0) == (0.0, 1.0)


def test_format_w_two_decimals():
    assert _format_w(0.6) == "0.60"
    assert _format_w(1.0) == "1.00"
    assert _format_w(0.95) == "0.95"


def test_anchor_check_matches_when_csvs_equal(tmp_path):
    csv_a = tmp_path / "a.csv"
    csv_b = tmp_path / "b.csv"
    df = pd.DataFrame([
        {"season": 2024, "team_a": 1, "team_b": 2, "p_a_wins": 0.7},
        {"season": 2024, "team_a": 1, "team_b": 3, "p_a_wins": 0.6},
    ])
    df.to_csv(csv_a, index=False)
    df.to_csv(csv_b, index=False)
    result = _anchor_check(str(csv_a), str(csv_b))
    assert result["matches"] is True
    assert result["max_abs_diff"] < 1e-12


def test_anchor_check_detects_p_diff(tmp_path):
    csv_a = tmp_path / "a.csv"
    csv_b = tmp_path / "b.csv"
    pd.DataFrame([
        {"season": 2024, "team_a": 1, "team_b": 2, "p_a_wins": 0.7},
    ]).to_csv(csv_a, index=False)
    pd.DataFrame([
        {"season": 2024, "team_a": 1, "team_b": 2, "p_a_wins": 0.65},
    ]).to_csv(csv_b, index=False)
    result = _anchor_check(str(csv_a), str(csv_b))
    assert result["matches"] is False
    assert result["max_abs_diff"] > 0.04


def test_anchor_check_detects_coverage_diff(tmp_path):
    csv_a = tmp_path / "a.csv"
    csv_b = tmp_path / "b.csv"
    pd.DataFrame([
        {"season": 2024, "team_a": 1, "team_b": 2, "p_a_wins": 0.7},
        {"season": 2024, "team_a": 1, "team_b": 3, "p_a_wins": 0.6},
    ]).to_csv(csv_a, index=False)
    pd.DataFrame([
        {"season": 2024, "team_a": 1, "team_b": 2, "p_a_wins": 0.7},
    ]).to_csv(csv_b, index=False)
    result = _anchor_check(str(csv_a), str(csv_b))
    assert result["matches"] is False
    assert result["n_only_actual"] == 1
    assert result["n_only_expected"] == 0


def test_score_pairwise_smoke_on_baseline():
    """Smoke test: _score_pairwise wraps score_chalk_brackets without
    crashing on the existing pairwise_v9c_v4_baseline.csv."""
    csv = Path("output/pairwise_v9c_v4_baseline.csv")
    if not csv.exists():
        pytest.skip(f"{csv} not present in this checkout")
    summary = _score_pairwise(str(csv))
    assert "total_pts" in summary
    assert "per_season" in summary
    assert summary["total_pts"] > 0
    assert len(summary["per_season"]) >= 20
