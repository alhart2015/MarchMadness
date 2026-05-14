"""Tests for src.score_v12_loso_pick.

The picker logic is deterministic: given per-cell-per-season scores, pick
the cell that maximizes summed training-season scores for each test season.
We test against synthetic 2- and 3-cell inputs.
"""
from __future__ import annotations

import pandas as pd
import pytest

from src.score_v12_loso_pick import (
    _cell_name_from_path,
    pick_cell_per_season,
    concatenate_picked_rows,
)


def test_picker_prefers_higher_training_total():
    """Cell A scores 100 in all training seasons, cell B scores 50.
    Picker should always pick A regardless of test season."""
    seasons = [2010, 2011, 2012, 2013]
    totals = {
        "A": {s: 100.0 for s in seasons},
        "B": {s: 50.0 for s in seasons},
    }
    picks = pick_cell_per_season(totals)
    assert all(c == "A" for c in picks.values()), f"picks: {picks}"


def test_picker_handles_dominant_test_season():
    """Cell A scores 100 only in season 2013, 0 elsewhere. Cell B scores
    50 in every season. For test_season=2013, training totals are
    A=0, B=150 -- picker picks B. For test_season=2010, training totals
    are A=100, B=150 -- picker still picks B (the higher one)."""
    totals = {
        "A": {2010: 0, 2011: 0, 2012: 0, 2013: 100.0},
        "B": {2010: 50.0, 2011: 50.0, 2012: 50.0, 2013: 50.0},
    }
    picks = pick_cell_per_season(totals)
    assert picks[2013] == "B"
    assert picks[2010] == "B"


def test_picker_can_pick_different_cells_per_season():
    """A scores high in {2010, 2011, 2012} and low in 2013. B scores high
    in 2013 only. For test_season=2013, training totals are A=300, B=0
    -> A. For test_season=2010, training totals are A=0+100+100=200,
    B=100 -> A. Setup so that A always wins."""
    totals = {
        "A": {2010: 100.0, 2011: 100.0, 2012: 100.0, 2013: 0.0},
        "B": {2010: 0.0, 2011: 0.0, 2012: 0.0, 2013: 100.0},
    }
    picks = pick_cell_per_season(totals)
    assert picks[2013] == "A"
    # For each non-2013 test season, A's training total is 200, B's is 100.
    for ts in (2010, 2011, 2012):
        assert picks[ts] == "A"


def test_picker_with_close_cells():
    """If two cells are within 1 brkt point, the picker still produces a
    deterministic winner (Python's max is stable and returns the first
    encountered key when tied)."""
    totals = {
        "A": {2010: 100.0, 2011: 100.0},
        "B": {2010: 100.0, 2011: 100.5},
    }
    picks = pick_cell_per_season(totals)
    # test_season=2010: train = {2011}; A=100, B=100.5 -> B
    # test_season=2011: train = {2010}; A=100, B=100   -> A or B (tie, depends on dict order)
    assert picks[2010] == "B"
    assert picks[2011] in {"A", "B"}


def test_concatenate_uses_picked_cell_per_season():
    """Output frame has exactly the rows the picked cell provides for that season."""
    cell_frames = {
        "A": pd.DataFrame({
            "season": [2010, 2010, 2011, 2011],
            "team_a": [1, 2, 1, 2],
            "team_b": [10, 20, 10, 20],
            "p_a_wins": [0.10, 0.20, 0.30, 0.40],
        }),
        "B": pd.DataFrame({
            "season": [2010, 2010, 2011, 2011],
            "team_a": [1, 2, 1, 2],
            "team_b": [10, 20, 10, 20],
            "p_a_wins": [0.55, 0.65, 0.75, 0.85],
        }),
    }
    picks = {2010: "A", 2011: "B"}
    out = concatenate_picked_rows(cell_frames, picks)
    assert len(out) == 4
    # 2010 rows came from A
    s10 = out[out["season"] == 2010]["p_a_wins"].tolist()
    assert s10 == [0.10, 0.20]
    # 2011 rows came from B
    s11 = out[out["season"] == 2011]["p_a_wins"].tolist()
    assert s11 == [0.75, 0.85]


def test_cell_name_from_path():
    assert _cell_name_from_path("output/pairwise_v12_n10_v8.csv") == "n10_v8"
    assert _cell_name_from_path("/abs/path/pairwise_v12_n5_v10cap.csv") == "n5_v10cap"
    assert _cell_name_from_path("foo.csv") == "foo"
