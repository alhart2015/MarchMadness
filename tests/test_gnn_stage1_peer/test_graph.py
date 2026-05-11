import pandas as pd
import pytest
import torch


def _toy_train_games() -> pd.DataFrame:
    return pd.DataFrame([
        {"Season": 2024, "DayNum": 50,  "WTeamID": 1101, "WScore": 80, "LTeamID": 1102, "LScore": 70, "WLoc": "H", "NumOT": 0},
        {"Season": 2024, "DayNum": 100, "WTeamID": 1102, "WScore": 75, "LTeamID": 1103, "LScore": 60, "WLoc": "A", "NumOT": 0},
        {"Season": 2024, "DayNum": 119, "WTeamID": 1101, "WScore": 90, "LTeamID": 1103, "LScore": 85, "WLoc": "N", "NumOT": 1},
    ])


def test_build_pyg_graph_has_bidirected_edges():
    from src.gnn_stage1_peer.graph import build_pyg_graph
    from src.gnn_stage1_peer.data import build_team_index
    games = _toy_train_games()
    idx = build_team_index(games)  # 1101->0, 1102->1, 1103->2
    g = build_pyg_graph(games, idx)
    assert g.num_nodes == 3
    # 3 games -> 6 directed edges (bidirected)
    assert g.edge_index.shape == (2, 6)
    assert g.edge_attr.shape == (6, 4)


def test_build_pyg_graph_edge_attr_signed_score_diff():
    """Edge attribute score_diff is signed from the source-node perspective."""
    from src.gnn_stage1_peer.graph import build_pyg_graph
    from src.gnn_stage1_peer.data import build_team_index
    games = pd.DataFrame([
        {"Season": 2024, "DayNum": 50, "WTeamID": 1101, "WScore": 80, "LTeamID": 1102, "LScore": 70, "WLoc": "N", "NumOT": 0},
    ])
    idx = build_team_index(games)  # 1101->0, 1102->1
    g = build_pyg_graph(games, idx)
    # Two directed edges: 0->1 (score_diff = +10) and 1->0 (score_diff = -10).
    src = g.edge_index[0].tolist()
    dst = g.edge_index[1].tolist()
    diffs = g.edge_attr[:, 0].tolist()
    edges = list(zip(src, dst, diffs))
    assert (0, 1, 10.0) in edges
    assert (1, 0, -10.0) in edges


def test_build_matchup_pairs_symmetric():
    from src.gnn_stage1_peer.graph import build_matchup_pairs
    from src.gnn_stage1_peer.data import build_team_index
    games = pd.DataFrame([
        {"Season": 2024, "DayNum": 50, "WTeamID": 1101, "WScore": 80, "LTeamID": 1102, "LScore": 70, "WLoc": "H", "NumOT": 0},
    ])
    idx = build_team_index(games)
    a, b, y = build_matchup_pairs(games, idx)
    assert a.shape == (2,) and b.shape == (2,) and y.shape == (2,)
    # Both orientations: (1101, 1102, 1) and (1102, 1101, 0)
    pairs = sorted(zip(a.tolist(), b.tolist(), y.tolist()))
    assert pairs == sorted([(0, 1, 1.0), (1, 0, 0.0)])
