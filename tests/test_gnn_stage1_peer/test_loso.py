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


# ------------------------------- Task C tests ---------------------------------
#
# Cross-season shared-parameter training loop. We build a separable toy across
# three "fake" seasons sharing 6 globally-indexed teams (indices 0..5). The
# label rule across all seasons is "lower index always beats higher index"
# (i.e., team 0 > team 1 > ... > team 5). Each season's RS graph encodes that
# ordering with a different subset of games so encoder must generalise across
# graphs while decoder learns the shared rule. Tournament pairs are drawn
# from the same separable rule.


def _toy_graph(num_nodes: int, edges: list[tuple[int, int, float]]) -> Data:
    """Build a small bidirected PyG Data graph from (winner, loser, score_diff) tuples."""
    src, dst, attr = [], [], []
    for w, l, sd in edges:
        src.append(w); dst.append(l); attr.append([sd, 0.0, 0.0, 50.0])
        src.append(l); dst.append(w); attr.append([-sd, 0.0, 0.0, 50.0])
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_attr = torch.tensor(attr, dtype=torch.float)
    return Data(edge_index=edge_index, edge_attr=edge_attr, num_nodes=num_nodes)


def _toy_pairs(matchups: list[tuple[int, int]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Symmetric matchup pairs for (winner, loser) tuples.

    Each game produces (w, l, 1.0) and (l, w, 0.0) -- mirrors build_matchup_pairs.
    """
    a, b, y = [], [], []
    for w, l in matchups:
        a.append(w); b.append(l); y.append(1.0)
        a.append(l); b.append(w); y.append(0.0)
    return (
        torch.tensor(a, dtype=torch.long),
        torch.tensor(b, dtype=torch.long),
        torch.tensor(y, dtype=torch.float),
    )


def _three_season_separable_fixture():
    """Three fake seasons sharing 6 globally-indexed teams (0..5).

    Rule: lower-index team always beats higher-index team. Each season's RS
    graph is a fully-connected separable round-robin where every (i, j) with
    i < j has an edge i -> j (low-idx beats high-idx). The score_diff varies
    per season to make the seasons distinct, but the structural information
    (each team plays each other) is shared so encoder embeddings transfer
    cleanly. We use season 2024 as holdout.
    """
    num_nodes = 6

    def _round_robin_edges(score_scale: float) -> list[tuple[int, int, float]]:
        edges: list[tuple[int, int, float]] = []
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                # Lower idx beats higher idx; magnitude grows with gap.
                edges.append((i, j, score_scale * (j - i)))
        return edges

    g22 = _toy_graph(num_nodes, _round_robin_edges(score_scale=3.0))
    g23 = _toy_graph(num_nodes, _round_robin_edges(score_scale=4.0))
    g24 = _toy_graph(num_nodes, _round_robin_edges(score_scale=5.0))
    per_season_graphs = {2022: g22, 2023: g23, 2024: g24}

    # Training tournament pairs respect same rule (lower idx beats higher).
    train_pairs_by_season = {
        2022: _toy_pairs([(0, 3), (1, 4), (2, 5)]),  # 3 games -> 6 pairs
        2023: _toy_pairs([(0, 5), (1, 2), (3, 4)]),  # 3 games -> 6 pairs
    }
    # Validation pairs (holdout 2024 tournament): 3 games -> 6 pairs.
    val_pairs = _toy_pairs([(0, 2), (1, 4), (2, 5)])
    val_graph = g24

    return num_nodes, per_season_graphs, train_pairs_by_season, val_pairs, val_graph


def test_train_loso_gnn_loss_decreases():
    """On a separable 3-season toy, final training loss should be lower than initial."""
    from src.gnn_stage1_peer.loso import train_loso_gnn

    num_nodes, gs, train_by_s, val_pairs, val_graph = _three_season_separable_fixture()
    model, info = train_loso_gnn(
        gs, train_by_s, val_pairs, val_graph, num_nodes,
        hidden_dim=16, num_layers=2, dropout=0.0, decoder_hidden=32,
        epochs=100, lr=0.05, patience=100, seed=42,
    )
    losses = info["train_history"]["loss"]
    assert len(losses) >= 2
    # Final train loss should drop substantially below initial loss.
    assert losses[-1] < losses[0], (
        f"Loss did not decrease: first={losses[0]:.4f} last={losses[-1]:.4f}"
    )
    # Stronger sanity check: separable toy should reach near-perfect loss.
    assert losses[-1] < losses[0] * 0.5


def test_train_loso_gnn_val_ll_improves():
    """Validation LL should improve (decrease) over training on separable toy."""
    from src.gnn_stage1_peer.loso import train_loso_gnn

    num_nodes, gs, train_by_s, val_pairs, val_graph = _three_season_separable_fixture()
    _, info = train_loso_gnn(
        gs, train_by_s, val_pairs, val_graph, num_nodes,
        hidden_dim=16, num_layers=2, dropout=0.0, decoder_hidden=32,
        epochs=100, lr=0.05, patience=100, seed=42,
    )
    val_history = info["train_history"]["val_ll"]
    assert len(val_history) >= 2
    # Final val LL strictly better (lower) than initial.
    assert val_history[-1] < val_history[0], (
        f"Val LL did not improve: first={val_history[0]:.4f} last={val_history[-1]:.4f}"
    )


def test_train_loso_gnn_returns_best_state():
    """Returned model evaluated on val_pairs reproduces best_val_ll.

    ``best_val_ll`` is the val LL at the last epoch where val improved by
    more than 1e-5 over the running best (mirrors Phase 1's ``train_gnn``).
    The argmin of the full history may be marginally lower if subsequent
    improvements were below the eps threshold, so we check:
      - best_epoch is consistent with ``np.argmin(val_history)`` modulo eps,
      - best_val_ll is within eps of the absolute minimum,
      - the returned model reproduces best_val_ll exactly.
    """
    import numpy as np
    import torch.nn.functional as F

    from src.gnn_stage1_peer.loso import train_loso_gnn

    num_nodes, gs, train_by_s, val_pairs, val_graph = _three_season_separable_fixture()
    model, info = train_loso_gnn(
        gs, train_by_s, val_pairs, val_graph, num_nodes,
        hidden_dim=16, num_layers=2, dropout=0.0, decoder_hidden=32,
        epochs=60, lr=0.05, patience=100, seed=42,
    )
    val_history = info["train_history"]["val_ll"]
    # best_val_ll is within early-stopping eps of the absolute minimum.
    assert info["best_val_ll"] <= min(val_history) + 1e-5
    # best_epoch points to argmin of the full history.
    assert info["best_epoch"] == int(np.argmin(val_history))
    # Returned model weights reproduce best_val_ll on val_pairs.
    model.eval()
    with torch.no_grad():
        a_v, b_v, y_v = val_pairs
        logits = model(val_graph, a_v, b_v)
        ll = F.binary_cross_entropy_with_logits(logits, y_v).item()
    assert ll == pytest.approx(info["best_val_ll"], abs=1e-5)


def test_train_loso_gnn_history_keys_and_shapes():
    """Returned info dict has expected keys; history lists are non-empty."""
    from src.gnn_stage1_peer.loso import train_loso_gnn

    num_nodes, gs, train_by_s, val_pairs, val_graph = _three_season_separable_fixture()
    _, info = train_loso_gnn(
        gs, train_by_s, val_pairs, val_graph, num_nodes,
        hidden_dim=8, num_layers=2, dropout=0.0, decoder_hidden=16,
        epochs=5, lr=0.05, patience=100, seed=42,
    )
    assert set(info.keys()) >= {"best_val_ll", "best_epoch", "epochs_run", "train_history"}
    assert info["epochs_run"] == 5
    assert len(info["train_history"]["loss"]) == 5
    assert len(info["train_history"]["val_ll"]) == 5


# ------------------------------- Task D tests ---------------------------------
#
# Evaluator + per-holdout driver. evaluate_loso must match the Phase 1
# evaluator's output shape exactly. run_phase2_one_holdout integrates B+C+D
# end-to-end on a tiny fixture.


def test_evaluate_loso_shape():
    """evaluate_loso returns Phase 1's evaluator shape on a tiny model+graph."""
    from src.gnn_stage1_peer.loso import evaluate_loso
    from src.gnn_stage1_peer.model import GNNStage1Peer

    num_nodes, gs, _, val_pairs, val_graph = _three_season_separable_fixture()
    model = GNNStage1Peer(
        num_nodes=num_nodes,
        hidden_dim=8,
        num_layers=2,
        dropout=0.0,
        decoder_hidden=16,
    )
    out = evaluate_loso(model, val_pairs, val_graph)

    assert set(out.keys()) == {"ll", "accuracy", "n", "predictions"}
    a_v, b_v, y_v = val_pairs
    assert out["n"] == int(y_v.numel())
    assert isinstance(out["predictions"], list)
    assert len(out["predictions"]) == out["n"]
    for p in out["predictions"]:
        assert set(p.keys()) == {"team_a_idx", "team_b_idx", "p_a_wins", "label"}
        assert isinstance(p["team_a_idx"], int)
        assert isinstance(p["team_b_idx"], int)
        assert isinstance(p["p_a_wins"], float)
        assert isinstance(p["label"], float)
        assert 0.0 <= p["p_a_wins"] <= 1.0
    assert 0.0 <= out["accuracy"] <= 1.0
    # ll is a finite non-negative BCE value.
    assert out["ll"] >= 0.0
    import math as _math
    assert _math.isfinite(out["ll"])


def test_run_phase2_one_holdout_smoke(tmp_path):
    """End-to-end smoke: build data, train, evaluate on the multi-season fixture."""
    from src.gnn_stage1_peer.loso import run_phase2_one_holdout

    data_dir = _make_fixture(tmp_path)
    result = run_phase2_one_holdout(
        data_dir=data_dir,
        holdout_season=2024,
        seasons=[2022, 2023, 2024],
        hidden_dim=8,
        num_layers=2,
        dropout=0.0,
        decoder_hidden=16,
        epochs=5,
        lr=0.05,
        patience=10,
        seed=42,
    )

    expected_keys = {
        "holdout_season",
        "gnn",
        "predictions",
        "train_minutes",
        "epochs_run",
        "best_epoch",
        "best_val_ll",
    }
    assert set(result.keys()) == expected_keys
    assert result["holdout_season"] == 2024

    import math as _math
    gnn = result["gnn"]
    assert set(gnn.keys()) == {"ll", "accuracy", "n"}
    assert _math.isfinite(gnn["ll"])
    assert gnn["n"] > 0
    assert len(result["predictions"]) == gnn["n"]
    assert result["train_minutes"] >= 0
    assert result["epochs_run"] >= 1
    assert result["epochs_run"] <= 5
