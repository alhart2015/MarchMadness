"""Tests for feature matrix assembly."""

import pandas as pd
import pytest

from src.features.feature_matrix import build_feature_matrix


@pytest.fixture
def efficiency_ratings():
    return pd.DataFrame({
        "TeamID": [1101, 1102, 1103, 1104],
        "adj_oe": [115.0, 110.0, 105.0, 100.0],
        "adj_de": [95.0, 98.0, 102.0, 108.0],
        "adj_em": [20.0, 12.0, 3.0, -8.0],
        "adj_tempo": [68.0, 65.0, 70.0, 72.0],
    })


@pytest.fixture
def four_factors():
    return pd.DataFrame({
        "TeamID": [1101, 1102, 1103, 1104],
        "off_efg": [0.55, 0.52, 0.48, 0.45],
        "off_to_rate": [0.15, 0.17, 0.19, 0.21],
        "off_or_rate": [0.32, 0.30, 0.28, 0.25],
        "off_ft_rate": [0.35, 0.33, 0.30, 0.28],
        "def_efg": [0.45, 0.48, 0.50, 0.53],
        "def_to_rate": [0.20, 0.18, 0.16, 0.14],
        "def_or_rate": [0.25, 0.28, 0.30, 0.33],
        "def_ft_rate": [0.28, 0.30, 0.33, 0.35],
    })


@pytest.fixture
def massey_ranks():
    return pd.DataFrame({
        "TeamID": [1101, 1102, 1103, 1104,
                    1101, 1102, 1103, 1104],
        "SystemName": ["POM", "POM", "POM", "POM",
                        "SAG", "SAG", "SAG", "SAG"],
        "OrdinalRank": [1, 5, 30, 80,
                        2, 6, 28, 75],
    })


def test_build_feature_matrix_shape(efficiency_ratings, four_factors, sample_seeds, massey_ranks, sample_detailed_results):
    result = build_feature_matrix(
        efficiency=efficiency_ratings,
        four_factors=four_factors,
        seeds=sample_seeds,
        massey=massey_ranks,
        results=sample_detailed_results,
        season=2023,
        massey_systems=["POM", "SAG"],
    )
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 4
    assert "TeamID" in result.columns
    assert "seed" in result.columns
    assert "adj_em" in result.columns
    assert "massey_POM" in result.columns


def test_build_feature_matrix_no_nulls(efficiency_ratings, four_factors, sample_seeds, massey_ranks, sample_detailed_results):
    result = build_feature_matrix(
        efficiency=efficiency_ratings,
        four_factors=four_factors,
        seeds=sample_seeds,
        massey=massey_ranks,
        results=sample_detailed_results,
        season=2023,
        massey_systems=["POM", "SAG"],
    )
    # Tournament teams should have no nulls in critical features
    assert not result[["adj_em", "seed"]].isna().any().any()
