"""Tests for Massey Ratings composite loader."""

from unittest.mock import patch

import pandas as pd
import pytest

from src.ingest.massey_loader import load_massey_composite, parse_massey_csv


@pytest.fixture
def sample_massey_csv():
    return "Team,1,2,3,Comp\nDuke,1,2,1,1\nUNC,3,4,5,4\n"


def test_parse_massey_csv(sample_massey_csv):
    result = parse_massey_csv(sample_massey_csv)
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 2
    assert "Team" in result.columns
    assert "Comp" in result.columns


def test_load_massey_composite_returns_dataframe(tmp_path, sample_massey_csv):
    with patch("src.ingest.massey_loader._download_massey_csv", return_value=sample_massey_csv):
        result = load_massey_composite(cache_dir=str(tmp_path))
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 2


def test_load_massey_composite_failure_returns_none(tmp_path):
    with patch("src.ingest.massey_loader._download_massey_csv", side_effect=Exception("Network error")):
        result = load_massey_composite(cache_dir=str(tmp_path))
    assert result is None
