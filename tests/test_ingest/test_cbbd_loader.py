"""Tests for cbbd API loader."""

import json
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

import pandas as pd
import pytest

from src.ingest.cbbd_loader import load_cbbd_data, _get_cache_path, _is_cache_valid


@pytest.fixture
def mock_cbbd_response():
    """Simulated cbbd API team ratings response."""
    return [
        {"team": "Duke", "year": 2026, "barthag": 0.95, "adj_o": 120.5, "adj_d": 90.2, "adj_t": 68.1,
         "efg_o": 0.55, "efg_d": 0.44, "tov_o": 0.15, "tov_d": 0.20, "orb_o": 0.33, "orb_d": 0.24, "ftr_o": 0.36, "ftr_d": 0.27},
        {"team": "UNC", "year": 2026, "barthag": 0.88, "adj_o": 115.0, "adj_d": 95.0, "adj_t": 70.0,
         "efg_o": 0.52, "efg_d": 0.47, "tov_o": 0.17, "tov_d": 0.18, "orb_o": 0.30, "orb_d": 0.27, "ftr_o": 0.33, "ftr_d": 0.30},
    ]


def test_cache_path(tmp_path):
    path = _get_cache_path(tmp_path, 2026)
    assert "cbbd_2026" in str(path)


def test_cache_validity(tmp_path):
    cache_file = tmp_path / "test_cache.json"
    cache_file.write_text("[]")
    assert _is_cache_valid(cache_file, ttl_hours=24)
    # File from far in the past would be invalid — tested via mocking


def test_load_cbbd_data_returns_dataframe(tmp_path, mock_cbbd_response):
    with patch("src.ingest.cbbd_loader._fetch_from_api", return_value=mock_cbbd_response):
        result = load_cbbd_data(season=2026, cache_dir=str(tmp_path))
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 2
    assert "team" in result.columns
    assert "adj_o" in result.columns


def test_load_cbbd_data_uses_cache(tmp_path, mock_cbbd_response):
    # First call: fetches from API
    with patch("src.ingest.cbbd_loader._fetch_from_api", return_value=mock_cbbd_response) as mock_fetch:
        load_cbbd_data(season=2026, cache_dir=str(tmp_path))
        assert mock_fetch.call_count == 1

    # Second call: should use cache
    with patch("src.ingest.cbbd_loader._fetch_from_api", return_value=mock_cbbd_response) as mock_fetch:
        load_cbbd_data(season=2026, cache_dir=str(tmp_path))
        assert mock_fetch.call_count == 0


def test_load_cbbd_data_api_failure_returns_none(tmp_path):
    with patch("src.ingest.cbbd_loader._fetch_from_api", side_effect=Exception("API down")):
        result = load_cbbd_data(season=2026, cache_dir=str(tmp_path))
    assert result is None
