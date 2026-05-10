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
