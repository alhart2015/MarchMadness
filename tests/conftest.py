"""Shared test fixtures."""

import pandas as pd
import pytest


@pytest.fixture
def sample_detailed_results():
    """Minimal detailed results DataFrame matching Kaggle schema."""
    return pd.DataFrame({
        "Season": [2023, 2023, 2023, 2023],
        "DayNum": [11, 11, 15, 15],
        "WTeamID": [1101, 1102, 1101, 1103],
        "WScore": [75, 80, 68, 90],
        "LTeamID": [1103, 1104, 1102, 1104],
        "LScore": [60, 70, 65, 55],
        "WLoc": ["H", "A", "N", "H"],
        "NumOT": [0, 0, 0, 0],
        "WFGM": [28, 30, 25, 35],
        "WFGA": [55, 60, 52, 62],
        "WFGM3": [8, 10, 7, 12],
        "WFGA3": [20, 25, 18, 28],
        "WFTM": [11, 10, 11, 8],
        "WFTA": [15, 14, 14, 10],
        "WOR": [10, 12, 8, 14],
        "WDR": [22, 20, 24, 25],
        "WAst": [15, 18, 12, 20],
        "WTO": [12, 10, 14, 8],
        "WStl": [7, 5, 6, 9],
        "WBlk": [3, 4, 2, 5],
        "WPF": [18, 16, 20, 14],
        "LFGM": [22, 26, 24, 20],
        "LFGA": [58, 62, 55, 60],
        "LFGM3": [6, 8, 7, 5],
        "LFGA3": [18, 22, 20, 16],
        "LFTM": [10, 10, 10, 10],
        "LFTA": [14, 13, 12, 15],
        "LOR": [8, 10, 9, 7],
        "LDR": [20, 18, 22, 19],
        "LAst": [12, 14, 13, 10],
        "LTO": [15, 13, 12, 16],
        "LStl": [5, 6, 7, 4],
        "LBlk": [2, 3, 3, 2],
        "LPF": [16, 15, 17, 12],
    })


@pytest.fixture
def sample_teams():
    """Minimal teams DataFrame."""
    return pd.DataFrame({
        "TeamID": [1101, 1102, 1103, 1104],
        "TeamName": ["Abilene Chr", "Air Force", "Akron", "Alabama"],
    })


@pytest.fixture
def sample_seeds():
    """Minimal seeds DataFrame."""
    return pd.DataFrame({
        "Season": [2023, 2023, 2023, 2023],
        "Seed": ["W01", "W02", "W03", "W04"],
        "TeamID": [1101, 1102, 1103, 1104],
    })


@pytest.fixture
def sample_config(tmp_path):
    """Minimal config dict for testing."""
    return {
        "data": {
            "kaggle_dir": str(tmp_path / "raw"),
            "cache_dir": str(tmp_path / "cache"),
            "processed_dir": str(tmp_path / "processed"),
            "team_overrides": str(tmp_path / "overrides.csv"),
        },
        "seasons": {
            "train_start": 2003,
            "train_end": 2025,
            "predict_season": 2026,
        },
        "efficiency": {
            "iterations": 15,
            "recency_half_life_days": 30,
            "home_court_advantage": 3.5,
            "ridge_alpha": 1.0,
        },
        "matching": {
            "auto_accept_threshold": 85,
            "review_threshold": 70,
            "algorithm": "token_sort_ratio",
        },
        "model": {
            "random_seed": 42,
            "n_simulations": 10000,
        },
        "massey": {
            "systems": ["POM", "SAG", "BPI", "TRK", "RPI"],
        },
        "bracket": {
            "scoring": [1, 2, 4, 8, 16, 32],
            "strategies": ["chalk", "expected_value"],
        },
    }
