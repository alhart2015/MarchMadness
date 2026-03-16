"""Tests for four factors computation."""

import pandas as pd
import pytest

from src.features.four_factors import compute_four_factors, estimate_possessions


def test_estimate_possessions():
    # FGA=60, OR=10, TO=12, FTA=15
    # possessions = 60 - 10 + 12 + 0.475 * 15 = 69.125
    result = estimate_possessions(fga=60, offensive_rebounds=10, turnovers=12, fta=15)
    assert abs(result - 69.125) < 0.01


def test_compute_four_factors(sample_detailed_results):
    result = compute_four_factors(sample_detailed_results, season=2023)
    assert isinstance(result, pd.DataFrame)
    assert "TeamID" in result.columns
    assert "off_efg" in result.columns
    assert "def_efg" in result.columns
    assert "off_to_rate" in result.columns
    assert "off_or_rate" in result.columns
    assert "off_ft_rate" in result.columns
    # Should have one row per team
    assert len(result) == 4


def test_four_factors_efg_range(sample_detailed_results):
    result = compute_four_factors(sample_detailed_results, season=2023)
    # eFG% should be between 0 and 1
    assert (result["off_efg"] >= 0).all()
    assert (result["off_efg"] <= 1).all()
