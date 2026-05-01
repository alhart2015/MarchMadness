"""Tests for src/train_upset_model.py (v9 upset-aware stage-2)."""

from pathlib import Path

import pandas as pd
import pytest

from src.train_upset_model import (
    load_per_game_data_with_upset,
    parse_seed,
)


# -----------------------------------------------------------------------------
# parse_seed
# -----------------------------------------------------------------------------

def test_parse_seed_numeric():
    assert parse_seed("W01") == 1
    assert parse_seed("X16b") == 16
    assert parse_seed("Y11a") == 11


def test_parse_seed_invalid():
    assert parse_seed(None) is None
    assert parse_seed("") is None


# -----------------------------------------------------------------------------
# load_per_game_data_with_upset: synthetic CSVs in tmp_path
# -----------------------------------------------------------------------------

def _write_csvs(tmp_path: Path, pairwise: pd.DataFrame, results: pd.DataFrame,
                seeds: pd.DataFrame):
    pw_path = tmp_path / "pairwise.csv"
    res_path = tmp_path / "results.csv"
    seeds_path = tmp_path / "seeds.csv"
    pairwise.to_csv(pw_path, index=False)
    results.to_csv(res_path, index=False)
    seeds.to_csv(seeds_path, index=False)
    return str(pw_path), str(res_path), str(seeds_path)


def test_loader_flags_5_over_12_as_upset(tmp_path):
    """A 12-seed beating a 5-seed is an upset."""
    # team_a < team_b convention in pairwise CSV.
    pairwise = pd.DataFrame({
        "season": [2023], "team_a": [1101], "team_b": [1112],
        "p_a_wins": [0.7],
    })
    results = pd.DataFrame({
        "Season": [2023], "DayNum": [136],
        "WTeamID": [1112], "WScore": [70], "LTeamID": [1101], "LScore": [65],
    })
    seeds = pd.DataFrame({
        "Season": [2023, 2023], "Seed": ["W05", "W12"],
        "TeamID": [1101, 1112],
    })
    pw_p, res_p, seeds_p = _write_csvs(tmp_path, pairwise, results, seeds)
    df = load_per_game_data_with_upset(pw_p, res_p, seeds_p)
    assert len(df) == 2  # symmetric pair
    assert df["upset"].all()


def test_loader_flags_1_beats_16_as_non_upset(tmp_path):
    """A 1-seed beating a 16-seed is NOT an upset."""
    pairwise = pd.DataFrame({
        "season": [2023], "team_a": [1101], "team_b": [1116],
        "p_a_wins": [0.95],
    })
    results = pd.DataFrame({
        "Season": [2023], "DayNum": [136],
        "WTeamID": [1101], "WScore": [85], "LTeamID": [1116], "LScore": [60],
    })
    seeds = pd.DataFrame({
        "Season": [2023, 2023], "Seed": ["W01", "W16"],
        "TeamID": [1101, 1116],
    })
    pw_p, res_p, seeds_p = _write_csvs(tmp_path, pairwise, results, seeds)
    df = load_per_game_data_with_upset(pw_p, res_p, seeds_p)
    assert len(df) == 2
    assert not df["upset"].any()


def test_loader_same_seed_is_non_upset(tmp_path):
    """Same-seed game (e.g., F4 1-vs-1): no higher seed, never an upset."""
    pairwise = pd.DataFrame({
        "season": [2023], "team_a": [1101], "team_b": [1102],
        "p_a_wins": [0.55],
    })
    results = pd.DataFrame({
        "Season": [2023], "DayNum": [152],
        "WTeamID": [1102], "WScore": [70], "LTeamID": [1101], "LScore": [65],
    })
    # Both teams seeded 1 (different regions).
    seeds = pd.DataFrame({
        "Season": [2023, 2023], "Seed": ["W01", "X01"],
        "TeamID": [1101, 1102],
    })
    pw_p, res_p, seeds_p = _write_csvs(tmp_path, pairwise, results, seeds)
    df = load_per_game_data_with_upset(pw_p, res_p, seeds_p)
    assert len(df) == 2
    assert not df["upset"].any()
    # abs_seed_diff is 0 in this case.
    assert (df["abs_seed_diff"] == 0).all()


def test_loader_produces_symmetric_rows(tmp_path):
    """Each game produces (a=W,label=1) and (a=L,label=0) with mirrored p_stage1."""
    pairwise = pd.DataFrame({
        "season": [2023], "team_a": [1101], "team_b": [1112],
        "p_a_wins": [0.7],
    })
    results = pd.DataFrame({
        "Season": [2023], "DayNum": [136],
        "WTeamID": [1101], "WScore": [80], "LTeamID": [1112], "LScore": [70],
    })
    seeds = pd.DataFrame({
        "Season": [2023, 2023], "Seed": ["W05", "W12"],
        "TeamID": [1101, 1112],
    })
    pw_p, res_p, seeds_p = _write_csvs(tmp_path, pairwise, results, seeds)
    df = load_per_game_data_with_upset(pw_p, res_p, seeds_p)
    win_row = df[df.label == 1].iloc[0]
    loss_row = df[df.label == 0].iloc[0]
    assert win_row["team_a"] == 1101 and win_row["team_b"] == 1112
    assert loss_row["team_a"] == 1112 and loss_row["team_b"] == 1101
    # Mirrored p_stage1: 0.7 (winner perspective) and 1 - 0.7 = 0.3 (loser).
    assert win_row["p_stage1"] == pytest.approx(0.7)
    assert loss_row["p_stage1"] == pytest.approx(0.3)
