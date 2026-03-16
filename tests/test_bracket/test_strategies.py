"""Tests for bracket selection strategies."""

import pandas as pd
import pytest

from src.bracket.strategies import chalk_bracket, expected_value_bracket
from src.bracket.simulator import FIRST_ROUND_MATCHUPS


@pytest.fixture
def bracket_df():
    """Single-region 16-team bracket for strategy testing."""
    return pd.DataFrame({
        "Region": ["East"] * 16,
        "Seed": list(range(1, 17)),
        "TeamID": list(range(101, 117)),
        "TeamName": [f"Team{i}" for i in range(1, 17)],
    })


@pytest.fixture
def advancement_probs():
    """Advancement probabilities for 16 teams across 4 region rounds."""
    probs = {}
    for i in range(16):
        team_id = 101 + i
        seed = i + 1
        # Better seeds get higher advancement probs
        base = max(0.98 - seed * 0.05, 0.02)
        probs[team_id] = {
            1: min(base * 1.2, 0.99),
            2: base * 0.8,
            3: base * 0.5,
            4: base * 0.3,
        }
    return probs


def test_chalk_bracket_structure(bracket_df, advancement_probs):
    picks = chalk_bracket(bracket_df, advancement_probs)
    # Round 1: exactly 8 winners from 8 matchups
    assert len(picks[1]) == 8
    # Round 2: exactly 4 winners
    assert len(picks[2]) == 4
    # All round 2 picks must be in round 1
    assert all(t in picks[1] for t in picks[2])
    # Round 4: regional champion
    assert len(picks[4]) == 1
    assert picks[4][0] in picks[3]


def test_chalk_bracket_picks_favorites(bracket_df, advancement_probs):
    picks = chalk_bracket(bracket_df, advancement_probs)
    # 1-seed should beat 16-seed in round 1
    assert 101 in picks[1]
    # 1-seed should be regional champion
    assert 101 in picks[4]


def test_expected_value_bracket_structure(bracket_df, advancement_probs):
    scoring = [1, 2, 4, 8]
    picks = expected_value_bracket(bracket_df, advancement_probs, scoring=scoring)
    # Same structural constraints as chalk
    assert len(picks[1]) == 8
    assert len(picks[2]) == 4
    assert all(t in picks[1] for t in picks[2])
    assert len(picks[4]) == 1
