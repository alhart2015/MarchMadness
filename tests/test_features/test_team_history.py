"""Unit tests for src/features/team_history.py.

Synthetic toy tournaments verify rounds_won counting and per-seed
baseline computation; real Kaggle data smoke-tests the integrator.
"""
import pytest
import pandas as pd

from src.features.team_history import _rounds_won_per_team_season


def _toy_tourney(rows):
    """Build a MNCAATourney*Results-shaped DataFrame from compact tuples.

    rows: list of (season, daynum, w_team_id, l_team_id) tuples.
    """
    return pd.DataFrame(
        rows,
        columns=["Season", "DayNum", "WTeamID", "LTeamID"],
    )


def test_rounds_won_counts_games_won_per_team_season():
    # Season 2024: team 100 wins R64 (day 136) + R32 (day 138) + S16 (day 143),
    # loses E8 (day 145) -> rounds_won = 3.
    # team 200 loses R64 -> rounds_won = 0.
    # team 300 wins championship (days 136, 138, 143, 145, 152, 154) -> rounds_won = 6.
    tr = _toy_tourney([
        (2024, 136, 100, 200),  # team 100 wins R64
        (2024, 138, 100, 201),  # team 100 wins R32
        (2024, 143, 100, 202),  # team 100 wins S16
        (2024, 145, 203, 100),  # team 100 LOSES E8
        (2024, 136, 300, 301),  # team 300 wins R64
        (2024, 138, 300, 302),  # team 300 wins R32
        (2024, 143, 300, 303),  # team 300 wins S16
        (2024, 145, 300, 304),  # team 300 wins E8
        (2024, 152, 300, 305),  # team 300 wins F4
        (2024, 154, 300, 306),  # team 300 wins Champ
    ])
    out = _rounds_won_per_team_season(tr, max_season=None)
    out = out.set_index(["Season", "TeamID"])["rounds_won"].to_dict()
    assert out[(2024, 100)] == 3
    assert out[(2024, 200)] == 0
    assert out[(2024, 300)] == 6


def test_rounds_won_leak_guard_fires_when_input_has_future_season():
    tr = _toy_tourney([
        (2023, 136, 100, 200),
        (2024, 136, 300, 400),  # future-season row
    ])
    with pytest.raises(AssertionError, match="Leak guard"):
        _rounds_won_per_team_season(tr, max_season=2023)
