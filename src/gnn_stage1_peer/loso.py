"""LOSO data pipeline + cross-season training for the GNN stage-1 peer (Phase 2).

Task B: build per-season RS graphs, train pairs (tournament games from
non-holdout seasons), and test pairs (holdout season's tournament games).
The global team_index is shared across seasons so a single
``nn.Embedding(num_teams, hidden_dim)`` covers every team that appears in any
requested season's regular-season data.

Task C: ``train_loso_gnn`` -- a single shared-parameter ``GNNStage1Peer`` is
trained across all training-season graphs. Each epoch forwards through every
training season's RS graph, decodes that season's tournament pairs, and
accumulates BCE loss; one ``backward`` + ``step`` is done per epoch.
Validation forwards through the caller-supplied ``val_graph`` and decodes
``val_pairs``; early stopping uses the best (lowest) validation LL with
patience.
"""
from __future__ import annotations

import math
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Data

from .data import build_global_team_index, load_rs_games
from .graph import build_matchup_pairs, build_pyg_graph
from .model import GNNStage1Peer
from .training import set_determinism


def load_tourney_games(data_dir: Path, season: int) -> pd.DataFrame:
    """Load tournament games for one season from MNCAATourneyCompactResults.csv."""
    path = Path(data_dir) / "MNCAATourneyCompactResults.csv"
    df = pd.read_csv(path)
    return df[df["Season"] == season].reset_index(drop=True)


def _validate_tourney_teams(
    games: pd.DataFrame, team_index: dict[int, int], season: int
) -> None:
    """Raise if any team in `games` is missing from `team_index`."""
    teams = set(games["WTeamID"].tolist()) | set(games["LTeamID"].tolist())
    missing = [t for t in teams if int(t) not in team_index]
    if missing:
        raise KeyError(
            f"Tournament season {season} references teams not in the global "
            f"team_index (no RS games for them in any requested season): "
            f"{sorted(missing)}"
        )


def build_loso_training_data(
    data_dir: Path,
    holdout_season: int,
    seasons: Iterable[int],
) -> tuple[
    dict[int, Data],
    dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    dict[int, int],
]:
    """Build LOSO training inputs for one holdout season.

    Parameters
    ----------
    data_dir
        Directory containing ``MRegularSeasonCompactResults.csv`` and
        ``MNCAATourneyCompactResults.csv``.
    holdout_season
        Season whose tournament games are held out for evaluation.
    seasons
        All seasons to include in the LOSO sweep. Must contain
        ``holdout_season``.

    Returns
    -------
    per_season_graphs
        ``{season: Data}`` -- one bidirected RS graph per season (including the
        holdout, since the holdout's RS data is used as INPUT at test time).
        Each graph uses the global ``team_index`` so ``num_nodes`` is identical
        across seasons.
    train_pairs_by_season
        ``{season: (a_idx, b_idx, y)}`` -- training tournament matchup pairs
        (both orientations) for every non-holdout season. The holdout season is
        absent from this dict.
    test_pairs
        ``(a_idx, b_idx, y)`` -- holdout season's tournament games, both
        orientations.
    team_index
        Global ``{TeamID: contiguous_idx}`` mapping covering the union of teams
        across all requested seasons' RS data.
    """
    seasons_list = list(seasons)
    if holdout_season not in seasons_list:
        raise ValueError(
            f"holdout_season={holdout_season} not in seasons={seasons_list}"
        )

    team_index = build_global_team_index(data_dir, seasons=seasons_list)

    per_season_graphs: dict[int, Data] = {}
    for season in seasons_list:
        rs_games = load_rs_games(data_dir, season=season)
        per_season_graphs[season] = build_pyg_graph(rs_games, team_index)

    train_pairs_by_season: dict[
        int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ] = {}
    test_pairs: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None
    for season in seasons_list:
        tourney_games = load_tourney_games(data_dir, season=season)
        _validate_tourney_teams(tourney_games, team_index, season)
        pairs = build_matchup_pairs(tourney_games, team_index)
        if season == holdout_season:
            test_pairs = pairs
        else:
            train_pairs_by_season[season] = pairs

    if test_pairs is None:
        # Defensive: only reachable if holdout has zero rows in tourney CSV.
        empty = torch.empty((0,), dtype=torch.long)
        empty_y = torch.empty((0,), dtype=torch.float)
        test_pairs = (empty, empty.clone(), empty_y)

    return per_season_graphs, train_pairs_by_season, test_pairs, team_index


def train_loso_gnn(
    per_season_graphs: dict[int, Data],
    train_pairs_by_season: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    val_pairs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    val_graph: Data,
    num_nodes: int,
    *,
    hidden_dim: int = 64,
    num_layers: int = 2,
    dropout: float = 0.2,
    decoder_hidden: int = 128,
    epochs: int = 50,
    lr: float = 1e-3,
    patience: int = 5,
    seed: int = 42,
) -> tuple[GNNStage1Peer, dict]:
    """Train ONE ``GNNStage1Peer`` across all training-season graphs (Phase 2).

    Per epoch, the model forwards through each training season's RS graph,
    decodes that season's tournament pairs, and concatenates per-season logits
    + labels into a single BCE loss. A single ``backward`` + ``step`` is done
    per epoch (gradients accumulate across seasons via the concatenated loss).

    Validation forwards through ``val_graph`` and decodes ``val_pairs`` to
    compute LL (BCE-with-logits). Early stopping tracks the best (lowest)
    val LL with the given ``patience``. The returned model has its weights
    restored to the best-epoch state.

    Parameters
    ----------
    per_season_graphs
        ``{season: Data}`` -- one PyG graph per training season. The keys
        of ``train_pairs_by_season`` must be a subset of these keys. The
        validation graph is supplied separately as ``val_graph`` and may or
        may not be included here (the caller decides whether to also pass
        the holdout's RS graph in this dict; this function only iterates
        keys present in ``train_pairs_by_season``).
    train_pairs_by_season
        ``{season: (a_idx, b_idx, y)}`` -- training tournament matchup pairs
        for each non-holdout season. Indices are global team indices; labels
        are 0/1 floats.
    val_pairs
        ``(a_idx, b_idx, y)`` -- validation/holdout tournament matchup pairs.
    val_graph
        PyG ``Data`` graph used as input when scoring ``val_pairs``. For
        Phase 2 LOSO this is typically the holdout season's RS graph.
    num_nodes
        Size of the global team-index embedding. Must equal
        ``len(team_index)`` from ``build_loso_training_data``.
    hidden_dim, num_layers, dropout, decoder_hidden
        Forwarded to ``GNNStage1Peer``.
    epochs, lr, patience, seed
        Training-loop hyperparameters. ``set_determinism(seed)`` is called
        before model construction so initialisation is reproducible.

    Returns
    -------
    model
        The trained ``GNNStage1Peer``. If at least one validation step
        produced a finite val LL, ``model.load_state_dict`` is called with
        the best-epoch weights before returning.
    info
        ``{"best_val_ll": float, "best_epoch": int, "epochs_run": int,
        "train_history": {"loss": [...], "val_ll": [...]}}``.

    Notes
    -----
    The function does NOT split off an internal validation set; the caller
    is responsible for providing ``val_pairs``/``val_graph`` (see Phase 2
    LOSO plan: the holdout season's tournament games and RS graph are used
    here, mirroring Phase 1's test-set early stopping).
    """
    set_determinism(seed)
    model = GNNStage1Peer(
        num_nodes=num_nodes,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        decoder_hidden=decoder_hidden,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_seasons = sorted(train_pairs_by_season.keys())
    if not train_seasons:
        raise ValueError("train_pairs_by_season is empty -- nothing to train on.")

    history: dict[str, list[float]] = {"loss": [], "val_ll": []}
    best_val = math.inf
    best_state: dict[str, torch.Tensor] | None = None
    bad_epochs = 0

    for _epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        all_logits: list[torch.Tensor] = []
        all_y: list[torch.Tensor] = []
        for season in train_seasons:
            graph_s = per_season_graphs[season]
            a, b, y = train_pairs_by_season[season]
            logits = model(graph_s, a, b)
            all_logits.append(logits)
            all_y.append(y)
        logits_cat = torch.cat(all_logits, dim=0)
        y_cat = torch.cat(all_y, dim=0)
        loss = F.binary_cross_entropy_with_logits(logits_cat, y_cat)
        loss.backward()
        optimizer.step()
        history["loss"].append(loss.item())

        # Validation: forward through val_graph, decode val_pairs, BCE LL.
        model.eval()
        with torch.no_grad():
            a_v, b_v, y_v = val_pairs
            v_logits = model(val_graph, a_v, b_v)
            val_ll = F.binary_cross_entropy_with_logits(v_logits, y_v).item()
        history["val_ll"].append(val_ll)

        if val_ll < best_val - 1e-5:
            best_val = val_ll
            best_state = {
                k: v.detach().clone() for k, v in model.state_dict().items()
            }
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, {
        "best_val_ll": best_val,
        "best_epoch": int(np.argmin(history["val_ll"])) if history["val_ll"] else 0,
        "epochs_run": len(history["loss"]),
        "train_history": history,
    }
