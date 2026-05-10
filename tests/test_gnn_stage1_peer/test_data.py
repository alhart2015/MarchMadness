import pandas as pd
import pytest
from pathlib import Path


def _toy_rs_csv(tmp_path: Path) -> Path:
    df = pd.DataFrame([
        # Season 2024, mix of early- and late-season games
        {"Season": 2024, "DayNum": 50,  "WTeamID": 1101, "WScore": 80, "LTeamID": 1102, "LScore": 70, "WLoc": "H", "NumOT": 0},
        {"Season": 2024, "DayNum": 119, "WTeamID": 1102, "WScore": 75, "LTeamID": 1101, "LScore": 72, "WLoc": "A", "NumOT": 0},
        {"Season": 2024, "DayNum": 120, "WTeamID": 1101, "WScore": 78, "LTeamID": 1103, "LScore": 65, "WLoc": "N", "NumOT": 0},
        {"Season": 2024, "DayNum": 132, "WTeamID": 1103, "WScore": 70, "LTeamID": 1101, "LScore": 68, "WLoc": "H", "NumOT": 1},
        {"Season": 2024, "DayNum": 134, "WTeamID": 1101, "WScore": 80, "LTeamID": 1104, "LScore": 60, "WLoc": "N", "NumOT": 0},  # tournament, excluded
        # Season 2023, should not appear in season=2024 load
        {"Season": 2023, "DayNum": 50,  "WTeamID": 1101, "WScore": 90, "LTeamID": 1102, "LScore": 80, "WLoc": "H", "NumOT": 0},
    ])
    p = tmp_path / "MRegularSeasonCompactResults.csv"
    df.to_csv(p, index=False)
    return tmp_path


def test_load_rs_games_filters_to_season(tmp_path):
    from src.gnn_stage1_peer.data import load_rs_games
    data_dir = _toy_rs_csv(tmp_path)
    games = load_rs_games(data_dir, season=2024)
    assert len(games) == 5  # all 2024 rows including DayNum=134 (filtering happens in split_phase1)
    assert (games["Season"] == 2024).all()


def test_split_phase1_partitions_by_daynum(tmp_path):
    from src.gnn_stage1_peer.data import load_rs_games, split_phase1
    data_dir = _toy_rs_csv(tmp_path)
    games = load_rs_games(data_dir, season=2024)
    train, test = split_phase1(games)
    # Train: DayNum < 120 -> 2 games (DayNum=50, DayNum=119)
    assert len(train) == 2
    assert (train["DayNum"] < 120).all()
    # Test: 120 <= DayNum < 134 -> 2 games (DayNum=120, DayNum=132)
    assert len(test) == 2
    assert (test["DayNum"] >= 120).all() and (test["DayNum"] < 134).all()


def test_build_team_index_assigns_contiguous_indices(tmp_path):
    from src.gnn_stage1_peer.data import load_rs_games, build_team_index
    data_dir = _toy_rs_csv(tmp_path)
    games = load_rs_games(data_dir, season=2024)
    idx = build_team_index(games)
    # Three teams appear in 2024: 1101, 1102, 1103, 1104
    assert set(idx.keys()) == {1101, 1102, 1103, 1104}
    # Indices are contiguous 0..N-1
    assert sorted(idx.values()) == [0, 1, 2, 3]


def _multi_season_rs_csv(tmp_path: Path) -> Path:
    """Three seasons where some teams appear in only one season."""
    df = pd.DataFrame([
        # Season 2022: teams 1101, 1102, 1199 (1199 only here)
        {"Season": 2022, "DayNum": 30, "WTeamID": 1101, "WScore": 70, "LTeamID": 1102, "LScore": 60, "WLoc": "H", "NumOT": 0},
        {"Season": 2022, "DayNum": 40, "WTeamID": 1199, "WScore": 80, "LTeamID": 1101, "LScore": 75, "WLoc": "A", "NumOT": 0},
        # Season 2023: teams 1101, 1103, 1104 (1103, 1104 introduced)
        {"Season": 2023, "DayNum": 35, "WTeamID": 1103, "WScore": 65, "LTeamID": 1101, "LScore": 60, "WLoc": "H", "NumOT": 0},
        {"Season": 2023, "DayNum": 45, "WTeamID": 1104, "WScore": 90, "LTeamID": 1103, "LScore": 88, "WLoc": "N", "NumOT": 1},
        # Season 2024: teams 1102, 1104, 1250 (1250 only here)
        {"Season": 2024, "DayNum": 25, "WTeamID": 1250, "WScore": 77, "LTeamID": 1102, "LScore": 70, "WLoc": "A", "NumOT": 0},
        {"Season": 2024, "DayNum": 55, "WTeamID": 1104, "WScore": 81, "LTeamID": 1250, "LScore": 79, "WLoc": "H", "NumOT": 0},
        # Season 2025: not requested -- should be ignored
        {"Season": 2025, "DayNum": 33, "WTeamID": 1888, "WScore": 88, "LTeamID": 1101, "LScore": 80, "WLoc": "H", "NumOT": 0},
    ])
    p = tmp_path / "MRegularSeasonCompactResults.csv"
    df.to_csv(p, index=False)
    return tmp_path


def test_build_global_team_index_covers_union_of_seasons(tmp_path):
    from src.gnn_stage1_peer.data import build_global_team_index
    data_dir = _multi_season_rs_csv(tmp_path)
    idx = build_global_team_index(data_dir, seasons=[2022, 2023, 2024])
    # Union of teams across the three seasons.
    expected_teams = {1101, 1102, 1103, 1104, 1199, 1250}
    assert set(idx.keys()) == expected_teams
    # 2025-only team must not appear since 2025 was not requested.
    assert 1888 not in idx


def test_build_global_team_index_is_contiguous_and_sorted(tmp_path):
    from src.gnn_stage1_peer.data import build_global_team_index
    data_dir = _multi_season_rs_csv(tmp_path)
    idx = build_global_team_index(data_dir, seasons=[2022, 2023, 2024])
    n = len(idx)
    # Indices contiguous 0..N-1
    assert sorted(idx.values()) == list(range(n))
    # Mapping is in sorted-TeamID order (deterministic)
    sorted_team_ids = sorted(idx.keys())
    for expected_idx, team_id in enumerate(sorted_team_ids):
        assert idx[team_id] == expected_idx


def test_build_global_team_index_single_season_matches_local(tmp_path):
    """When given a single season, global index should equal per-season index."""
    from src.gnn_stage1_peer.data import (
        build_global_team_index,
        build_team_index,
        load_rs_games,
    )
    data_dir = _multi_season_rs_csv(tmp_path)
    games_2023 = load_rs_games(data_dir, season=2023)
    local_idx = build_team_index(games_2023)
    global_idx = build_global_team_index(data_dir, seasons=[2023])
    assert local_idx == global_idx


def test_build_team_index_unchanged_regression(tmp_path):
    """Regression: per-season build_team_index produces the exact mapping it always did."""
    from src.gnn_stage1_peer.data import build_team_index, load_rs_games
    data_dir = _toy_rs_csv(tmp_path)
    games = load_rs_games(data_dir, season=2024)
    idx = build_team_index(games)
    # Sorted ascending TeamIDs map to 0..N-1
    assert idx == {1101: 0, 1102: 1, 1103: 2, 1104: 3}
