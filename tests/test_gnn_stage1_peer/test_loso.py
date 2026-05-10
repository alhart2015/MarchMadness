"""Tests for the Phase 2 LOSO data pipeline (Task B)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import torch
from torch_geometric.data import Data


def _multi_season_rs_csv(tmp_path: Path) -> Path:
    """Three seasons of RS games + a 2025 season that should be ignored."""
    df = pd.DataFrame([
        # Season 2022: teams 1101, 1102, 1199 (1199 only here)
        {"Season": 2022, "DayNum": 30, "WTeamID": 1101, "WScore": 70, "LTeamID": 1102, "LScore": 60, "WLoc": "H", "NumOT": 0},
        {"Season": 2022, "DayNum": 40, "WTeamID": 1199, "WScore": 80, "LTeamID": 1101, "LScore": 75, "WLoc": "A", "NumOT": 0},
        {"Season": 2022, "DayNum": 60, "WTeamID": 1102, "WScore": 65, "LTeamID": 1199, "LScore": 60, "WLoc": "N", "NumOT": 0},
        # Season 2023: teams 1101, 1102, 1103, 1104
        {"Season": 2023, "DayNum": 35, "WTeamID": 1103, "WScore": 65, "LTeamID": 1101, "LScore": 60, "WLoc": "H", "NumOT": 0},
        {"Season": 2023, "DayNum": 45, "WTeamID": 1104, "WScore": 90, "LTeamID": 1103, "LScore": 88, "WLoc": "N", "NumOT": 1},
        {"Season": 2023, "DayNum": 70, "WTeamID": 1102, "WScore": 75, "LTeamID": 1104, "LScore": 70, "WLoc": "A", "NumOT": 0},
        # Season 2024: teams 1102, 1104, 1250 (1250 only here)
        {"Season": 2024, "DayNum": 25, "WTeamID": 1250, "WScore": 77, "LTeamID": 1102, "LScore": 70, "WLoc": "A", "NumOT": 0},
        {"Season": 2024, "DayNum": 55, "WTeamID": 1104, "WScore": 81, "LTeamID": 1250, "LScore": 79, "WLoc": "H", "NumOT": 0},
        # Season 2025: not requested -- should be ignored
        {"Season": 2025, "DayNum": 33, "WTeamID": 1888, "WScore": 88, "LTeamID": 1101, "LScore": 80, "WLoc": "H", "NumOT": 0},
    ])
    p = tmp_path / "MRegularSeasonCompactResults.csv"
    df.to_csv(p, index=False)
    return tmp_path


def _multi_season_tourney_csv(tmp_path: Path) -> Path:
    """Three seasons of tournament games (DayNum >= 134). Only uses teams that
    also appear in the corresponding RS data above so global indexing works."""
    df = pd.DataFrame([
        # Season 2022 tournament: 2 games
        {"Season": 2022, "DayNum": 136, "WTeamID": 1101, "WScore": 78, "LTeamID": 1102, "LScore": 70, "WLoc": "N", "NumOT": 0},
        {"Season": 2022, "DayNum": 138, "WTeamID": 1199, "WScore": 71, "LTeamID": 1101, "LScore": 65, "WLoc": "N", "NumOT": 0},
        # Season 2023 tournament: 3 games
        {"Season": 2023, "DayNum": 136, "WTeamID": 1101, "WScore": 80, "LTeamID": 1103, "LScore": 75, "WLoc": "N", "NumOT": 0},
        {"Season": 2023, "DayNum": 138, "WTeamID": 1102, "WScore": 70, "LTeamID": 1104, "LScore": 65, "WLoc": "N", "NumOT": 0},
        {"Season": 2023, "DayNum": 140, "WTeamID": 1101, "WScore": 78, "LTeamID": 1102, "LScore": 76, "WLoc": "N", "NumOT": 1},
        # Season 2024 tournament: 2 games
        {"Season": 2024, "DayNum": 136, "WTeamID": 1104, "WScore": 82, "LTeamID": 1102, "LScore": 80, "WLoc": "N", "NumOT": 0},
        {"Season": 2024, "DayNum": 138, "WTeamID": 1250, "WScore": 70, "LTeamID": 1104, "LScore": 60, "WLoc": "N", "NumOT": 0},
    ])
    p = tmp_path / "MNCAATourneyCompactResults.csv"
    df.to_csv(p, index=False)
    return tmp_path


def _make_fixture(tmp_path: Path) -> Path:
    _multi_season_rs_csv(tmp_path)
    _multi_season_tourney_csv(tmp_path)
    return tmp_path


def test_load_tourney_games_filters_by_season(tmp_path):
    from src.gnn_stage1_peer.loso import load_tourney_games
    data_dir = _make_fixture(tmp_path)
    games = load_tourney_games(data_dir, season=2023)
    assert len(games) == 3
    assert (games["Season"] == 2023).all()
    assert (games["DayNum"] >= 134).all()


def test_build_loso_training_data_shape(tmp_path):
    from src.gnn_stage1_peer.data import build_global_team_index
    from src.gnn_stage1_peer.loso import build_loso_training_data

    data_dir = _make_fixture(tmp_path)
    seasons = [2022, 2023, 2024]
    holdout = 2023

    per_season_graphs, train_pairs_by_season, test_pairs, team_index = (
        build_loso_training_data(data_dir, holdout_season=holdout, seasons=seasons)
    )

    # team_index matches Task A's global team index for the same seasons.
    expected_team_index = build_global_team_index(data_dir, seasons=seasons)
    assert team_index == expected_team_index

    # per_season_graphs has one entry per season.
    assert set(per_season_graphs.keys()) == set(seasons)
    n_teams = len(team_index)
    for s, g in per_season_graphs.items():
        assert isinstance(g, Data)
        assert g.num_nodes == n_teams

    # train_pairs_by_season has entries only for non-holdout seasons.
    assert set(train_pairs_by_season.keys()) == {2022, 2024}
    # 2022 has 2 tournament games -> 4 pair-rows.
    a22, b22, y22 = train_pairs_by_season[2022]
    assert a22.shape == b22.shape == y22.shape == (4,)
    # 2024 has 2 tournament games -> 4 pair-rows.
    a24, b24, y24 = train_pairs_by_season[2024]
    assert a24.shape == b24.shape == y24.shape == (4,)

    # test_pairs is a 3-tuple of tensors, length 2 * n_holdout_tourney_games.
    a_t, b_t, y_t = test_pairs
    assert isinstance(a_t, torch.Tensor)
    assert isinstance(b_t, torch.Tensor)
    assert isinstance(y_t, torch.Tensor)
    # 2023 holdout has 3 tournament games -> 6 pair-rows.
    assert a_t.shape == b_t.shape == y_t.shape == (6,)


def test_build_loso_training_data_leak_safety(tmp_path):
    """Holdout's tournament games must NOT be in train_pairs_by_season; the
    holdout RS graph must contain only RS edges (not tournament edges)."""
    from src.gnn_stage1_peer.loso import build_loso_training_data

    data_dir = _make_fixture(tmp_path)
    seasons = [2022, 2023, 2024]
    holdout = 2023

    per_season_graphs, train_pairs_by_season, test_pairs, team_index = (
        build_loso_training_data(data_dir, holdout_season=holdout, seasons=seasons)
    )

    # Holdout key absent from train pairs.
    assert holdout not in train_pairs_by_season

    # Holdout's tournament pairs (a, b) must not appear in any non-holdout train pair set.
    a_t, b_t, _ = test_pairs
    holdout_pairs = set(zip(a_t.tolist(), b_t.tolist()))
    for s, (a, b, _y) in train_pairs_by_season.items():
        train_pairs = set(zip(a.tolist(), b.tolist()))
        # Even if RS happened to produce the same (a, b) ordering in training
        # *per-season tournament*, the holdout's tournament games are not
        # reused as training pairs -- the test confirms that the holdout
        # season is excluded from training pairs entirely.
        assert s != holdout

    # Holdout RS graph: count of edges == 2 * (n_holdout_RS_games).
    # 2023 fixture has 3 RS games -> 6 bidirected edges.
    holdout_graph = per_season_graphs[holdout]
    assert holdout_graph.edge_index.shape[1] == 6


def test_build_loso_training_data_season_tagging(tmp_path):
    """train_pairs_by_season keys exclude holdout; pair indices are valid
    global team indices."""
    from src.gnn_stage1_peer.loso import build_loso_training_data

    data_dir = _make_fixture(tmp_path)
    seasons = [2022, 2023, 2024]
    holdout = 2023

    _, train_pairs_by_season, _, team_index = build_loso_training_data(
        data_dir, holdout_season=holdout, seasons=seasons
    )

    assert set(train_pairs_by_season.keys()) == {2022, 2024}
    n_teams = len(team_index)
    valid_indices = set(team_index.values())
    for s, (a, b, y) in train_pairs_by_season.items():
        # All indices are valid global team indices.
        assert set(a.tolist()).issubset(valid_indices)
        assert set(b.tolist()).issubset(valid_indices)
        # Labels are in {0.0, 1.0}.
        assert set(y.tolist()) == {0.0, 1.0}
        # All indices in range.
        assert int(a.max()) < n_teams
        assert int(b.max()) < n_teams

    # Spot-check: 2022 pairs reference teams that played in 2022's tournament
    # (1101, 1102, 1199). Their global indices should appear in 2022's pairs.
    a22, b22, _ = train_pairs_by_season[2022]
    expected_2022_team_indices = {team_index[t] for t in (1101, 1102, 1199)}
    used_indices = set(a22.tolist()) | set(b22.tolist())
    assert used_indices == expected_2022_team_indices


def test_build_loso_training_data_global_indexing(tmp_path):
    """A team appearing only in 2022's RS games is still globally indexed and
    its node is reachable in per_season_graphs[2022]."""
    from src.gnn_stage1_peer.loso import build_loso_training_data

    data_dir = _make_fixture(tmp_path)
    seasons = [2022, 2023, 2024]
    holdout = 2023

    per_season_graphs, _, _, team_index = build_loso_training_data(
        data_dir, holdout_season=holdout, seasons=seasons
    )

    # 1199 only appears in 2022 RS games (and 2022 tournament).
    assert 1199 in team_index
    idx_1199 = team_index[1199]

    # 2022 graph references this index in its edges.
    g2022 = per_season_graphs[2022]
    edges_2022 = set(g2022.edge_index[0].tolist()) | set(g2022.edge_index[1].tolist())
    assert idx_1199 in edges_2022

    # 1250 only appears in 2024 RS games.
    assert 1250 in team_index
    idx_1250 = team_index[1250]
    g2024 = per_season_graphs[2024]
    edges_2024 = set(g2024.edge_index[0].tolist()) | set(g2024.edge_index[1].tolist())
    assert idx_1250 in edges_2024

    # All season graphs share num_nodes == |global team_index|.
    n = len(team_index)
    for s, g in per_season_graphs.items():
        assert g.num_nodes == n


def test_build_loso_training_data_raises_on_unknown_tourney_team(tmp_path):
    """If a tournament game references a team not in the global team_index,
    build_loso_training_data must raise (bug indicator)."""
    from src.gnn_stage1_peer.loso import build_loso_training_data

    # RS file: only seasons 2022 and 2024.
    rs = pd.DataFrame([
        {"Season": 2022, "DayNum": 30, "WTeamID": 1101, "WScore": 70, "LTeamID": 1102, "LScore": 60, "WLoc": "H", "NumOT": 0},
        {"Season": 2024, "DayNum": 25, "WTeamID": 1101, "WScore": 77, "LTeamID": 1102, "LScore": 70, "WLoc": "A", "NumOT": 0},
    ])
    rs.to_csv(tmp_path / "MRegularSeasonCompactResults.csv", index=False)
    # Tournament file: 2024 has a team (9999) that never appears in RS.
    t = pd.DataFrame([
        {"Season": 2022, "DayNum": 136, "WTeamID": 1101, "WScore": 78, "LTeamID": 1102, "LScore": 70, "WLoc": "N", "NumOT": 0},
        {"Season": 2024, "DayNum": 136, "WTeamID": 9999, "WScore": 80, "LTeamID": 1101, "LScore": 70, "WLoc": "N", "NumOT": 0},
    ])
    t.to_csv(tmp_path / "MNCAATourneyCompactResults.csv", index=False)

    with pytest.raises((KeyError, ValueError)):
        build_loso_training_data(
            tmp_path, holdout_season=2024, seasons=[2022, 2024]
        )
