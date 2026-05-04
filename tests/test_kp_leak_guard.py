"""Guard: kp_cols cannot include columns populated post-NCAA tournament.

Findings: docs/notes/2026-05-04-massey-kenpom-leak-audit.md
Code: src/kaggle_submission.py: build_all_team_features.
"""
import pandas as pd
import pytest

from src.kaggle_submission import (
    _KP_POST_TOURNAMENT_COLUMNS,
    build_all_team_features,
)


def _empty_inputs():
    """Return minimal inputs that pass the early shape checks but produce 0 rows."""
    reg_season = pd.DataFrame(columns=[
        "Season", "DayNum", "WTeamID", "LTeamID", "WScore", "LScore",
        "WLoc", "WFGM", "WFGA", "WFGM3", "WFGA3", "WFTM", "WFTA",
        "WOR", "WDR", "WAst", "WTO", "WStl", "WBlk", "WPF",
        "LFGM", "LFGA", "LFGM3", "LFGA3", "LFTM", "LFTA",
        "LOR", "LDR", "LAst", "LTO", "LStl", "LBlk", "LPF",
    ])
    seeds = pd.DataFrame(columns=["Season", "Seed", "TeamID"])
    conferences = pd.DataFrame(columns=["Season", "TeamID", "ConfAbbrev"])
    return reg_season, seeds, conferences


def test_default_kp_cols_pass():
    """Default kp_cols (allowlist of 17 rating columns) must not raise."""
    reg_season, seeds, conferences = _empty_inputs()
    out = build_all_team_features(
        reg_season=reg_season, seeds=seeds, conferences=conferences,
        seasons=[],
    )
    assert isinstance(out, pd.DataFrame)


def test_round_in_kp_cols_raises():
    """Including ROUND in kp_cols must raise ValueError naming the column."""
    reg_season, seeds, conferences = _empty_inputs()
    with pytest.raises(ValueError, match="ROUND"):
        build_all_team_features(
            reg_season=reg_season, seeds=seeds, conferences=conferences,
            seasons=[], kp_cols=["KADJ EM", "ROUND"],
        )


def test_round_only_raises():
    """kp_cols=['ROUND'] alone must raise."""
    reg_season, seeds, conferences = _empty_inputs()
    with pytest.raises(ValueError, match="ROUND"):
        build_all_team_features(
            reg_season=reg_season, seeds=seeds, conferences=conferences,
            seasons=[], kp_cols=["ROUND"],
        )


def test_post_tournament_set_membership():
    """Sanity: ROUND is in the guard set; KADJ EM is not."""
    assert "ROUND" in _KP_POST_TOURNAMENT_COLUMNS
    assert "KADJ EM" not in _KP_POST_TOURNAMENT_COLUMNS
    assert "GAMES" not in _KP_POST_TOURNAMENT_COLUMNS  # pre-tournament per audit
    assert "W" not in _KP_POST_TOURNAMENT_COLUMNS
    assert "L" not in _KP_POST_TOURNAMENT_COLUMNS
