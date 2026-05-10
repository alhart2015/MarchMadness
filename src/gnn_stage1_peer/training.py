"""Training loop for the Phase 1 GNN with early stopping + determinism."""
from __future__ import annotations

import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .baselines import evaluate_massey_baseline
from .data import build_team_index, load_rs_games, split_phase1
from .evaluation import compare_gnn_vs_massey, evaluate_gnn_phase1
from .graph import build_matchup_pairs, build_pyg_graph
from .model import GNNStage1Peer


def set_determinism(seed: int) -> None:
    """Set all relevant seeds for reproducibility (CPU)."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():  # no-op on CPU machines, hedged for completeness
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(False)  # SAGE/CUDA paths require non-strict mode


def _bce_logits_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(logits, targets)


def _eval_ll(model, graph, pairs) -> float:
    model.eval()
    with torch.no_grad():
        a, b, y = pairs
        logits = model(graph, a, b)
        return _bce_logits_loss(logits, y).item()


def train_gnn(
    model,
    graph,
    train_pairs,
    val_pairs,
    *,
    epochs: int = 50,
    lr: float = 1e-3,
    patience: int = 5,
    seed: int = 42,
) -> dict:
    """Train the GNN with Adam + early stopping on val LL."""
    set_determinism(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    a, b, y = train_pairs
    history = {"loss": [], "val_ll": []}
    best_val = math.inf
    best_state = None
    bad_epochs = 0
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(graph, a, b)
        loss = _bce_logits_loss(logits, y)
        loss.backward()
        optimizer.step()
        history["loss"].append(loss.item())
        val_ll = _eval_ll(model, graph, val_pairs)
        history["val_ll"].append(val_ll)
        if val_ll < best_val - 1e-5:
            best_val = val_ll
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return {
        "best_val_ll": best_val,
        "best_epoch": int(np.argmin(history["val_ll"])),
        "epochs_run": len(history["loss"]),
        "train_history": history,
    }


def run_phase1_one_season(
    data_dir: Path,
    season: int,
    *,
    hidden_dim: int = 64,
    num_layers: int = 2,
    dropout: float = 0.2,
    decoder_hidden: int = 128,
    epochs: int = 50,
    lr: float = 1e-3,
    patience: int = 5,
    seed: int = 42,
) -> dict:
    """Run one season's Phase 1: train GNN on early-RS, eval on late-RS, compare to Massey."""
    set_determinism(seed)
    games = load_rs_games(data_dir, season)
    train_games, test_games = split_phase1(games)
    if train_games.empty or test_games.empty:
        raise ValueError(f"Season {season}: train or test split empty.")

    team_index = build_team_index(games)
    graph = build_pyg_graph(train_games, team_index)
    train_pairs = build_matchup_pairs(train_games, team_index)
    test_pairs = build_matchup_pairs(test_games, team_index)

    model = GNNStage1Peer(
        num_nodes=len(team_index),
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        decoder_hidden=decoder_hidden,
    )
    t0 = time.time()
    train_result = train_gnn(
        model, graph, train_pairs, test_pairs,
        epochs=epochs, lr=lr, patience=patience, seed=seed,
    )
    train_minutes = (time.time() - t0) / 60.0

    gnn_eval = evaluate_gnn_phase1(model, graph, test_pairs)
    massey_eval = evaluate_massey_baseline(test_games, season=season, data_dir=data_dir)
    compare = compare_gnn_vs_massey(gnn_eval, massey_eval)

    return {
        "season": season,
        "gnn": {k: v for k, v in gnn_eval.items() if k != "predictions"},
        "massey": massey_eval,
        "compare": compare,
        "train_minutes": train_minutes,
        "epochs_run": train_result["epochs_run"],
        "best_epoch": train_result["best_epoch"],
    }
