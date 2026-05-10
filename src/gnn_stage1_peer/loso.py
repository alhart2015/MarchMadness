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
import time
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Data

from .data import build_global_team_index, load_rs_games
from .evaluation import evaluate_gnn_phase1
from .graph import build_matchup_pairs, build_pyg_graph
from .model import GNNStage1Peer, GNNStage1PeerEdgeAttr
from .training import set_determinism

# Encoder choices for Phase 2 LOSO. "sage" is the original Phase 1/2
# GraphSAGE encoder (ignores edge_attr). "edge_attr" is the MARGINAL-row
# structural variant: a GINE encoder that consumes graph.edge_attr.
_ENCODER_CHOICES = ("sage", "edge_attr")


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
    encoder: str = "sage",
) -> tuple[GNNStage1Peer | GNNStage1PeerEdgeAttr, dict]:
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
    encoder
        Either ``"sage"`` (default; original ``GNNStage1Peer`` with
        ``GraphSAGEEncoder``) or ``"edge_attr"`` (``GNNStage1PeerEdgeAttr``
        with the ``EdgeAttrAwareEncoder`` GINE variant). Used by the Phase 2
        MARGINAL-row structural sweep.

    Returns
    -------
    model
        The trained model (``GNNStage1Peer`` or ``GNNStage1PeerEdgeAttr``
        depending on ``encoder``). If at least one validation step
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
    if encoder not in _ENCODER_CHOICES:
        raise ValueError(
            f"encoder={encoder!r} not in {_ENCODER_CHOICES}"
        )
    set_determinism(seed)
    model_cls = (
        GNNStage1PeerEdgeAttr if encoder == "edge_attr" else GNNStage1Peer
    )
    model = model_cls(
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


def predict_holdout_pairwise(
    model: GNNStage1Peer | GNNStage1PeerEdgeAttr,
    holdout_graph: Data,
    team_index: dict[int, int],
    holdout_field: list[int],
) -> pd.DataFrame:
    """Round-robin pairwise predictions over the holdout's tournament field.

    Builds every unordered pair ``(team_a, team_b)`` with ``team_a < team_b``
    drawn from ``holdout_field`` (filtered to teams present in ``team_index``),
    forwards the holdout graph through ``model``, and returns a DataFrame with
    columns ``["team_a", "team_b", "p_a_wins"]`` -- the same asymmetric shape as
    ``output/pairwise_v4.csv`` minus the ``season`` column (caller adds it).

    Parameters
    ----------
    model
        Trained ``GNNStage1Peer`` whose embedding table covers ``team_index``.
    holdout_graph
        PyG ``Data`` graph used as encoder input (typically the holdout
        season's RS graph).
    team_index
        Global ``{TeamID: contiguous_idx}`` mapping. Pairs are forwarded with
        the indexed positions, but the returned ``team_a``/``team_b`` columns
        carry the original Kaggle ``TeamID`` integers.
    holdout_field
        Iterable of ``TeamID`` integers in the holdout's tournament field.
        Teams missing from ``team_index`` are silently dropped (defensive --
        a well-formed LOSO input has all tournament teams indexed).

    Returns
    -------
    pd.DataFrame
        Asymmetric round-robin pairwise predictions with columns
        ``["team_a", "team_b", "p_a_wins"]``. Rows are ordered with
        ``team_a < team_b``. Caller adds the ``season`` column before
        appending to ``output/pairwise_gnn_phase2.csv``.
    """
    field = sorted(int(t) for t in holdout_field if int(t) in team_index)
    a_list: list[int] = []
    b_list: list[int] = []
    for i in range(len(field)):
        for j in range(i + 1, len(field)):
            a_list.append(field[i])
            b_list.append(field[j])
    if not a_list:
        return pd.DataFrame({"team_a": [], "team_b": [], "p_a_wins": []})
    a_idx = torch.tensor([team_index[t] for t in a_list], dtype=torch.long)
    b_idx = torch.tensor([team_index[t] for t in b_list], dtype=torch.long)
    model.eval()
    with torch.no_grad():
        logits = model(holdout_graph, a_idx, b_idx)
        probs = torch.sigmoid(logits)
    return pd.DataFrame({
        "team_a": a_list,
        "team_b": b_list,
        "p_a_wins": [float(p) for p in probs.tolist()],
    })


def evaluate_loso(
    model: GNNStage1Peer | GNNStage1PeerEdgeAttr,
    test_pairs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    holdout_graph: Data,
) -> dict:
    """Evaluate a trained LOSO GNN on the holdout season's tournament pairs.

    Mirrors Phase 1's ``evaluate_gnn_phase1`` shape exactly:
    ``{"ll", "accuracy", "n", "predictions"}`` where each prediction dict
    carries ``(team_a_idx, team_b_idx, p_a_wins, label)``. Implementation
    delegates to ``evaluate_gnn_phase1`` since the signatures match.

    Parameters
    ----------
    model
        Trained ``GNNStage1Peer`` (typically the output of ``train_loso_gnn``).
    test_pairs
        ``(a_idx, b_idx, y)`` -- holdout season's tournament matchup pairs,
        both orientations.
    holdout_graph
        PyG ``Data`` graph used as input when scoring ``test_pairs``. For
        Phase 2 LOSO this is the holdout season's RS graph.
    """
    return evaluate_gnn_phase1(model, holdout_graph, test_pairs)


def run_phase2_one_holdout(
    data_dir: Path,
    holdout_season: int,
    seasons: Iterable[int],
    *,
    hidden_dim: int = 64,
    num_layers: int = 2,
    dropout: float = 0.2,
    decoder_hidden: int = 128,
    epochs: int = 50,
    lr: float = 1e-3,
    patience: int = 5,
    seed: int = 42,
    emit_pairwise: bool = False,
    encoder: str = "sage",
) -> dict:
    """Run one LOSO holdout: build data, train cross-season GNN, evaluate.

    Composes Tasks B (``build_loso_training_data``), C (``train_loso_gnn``),
    and D (``evaluate_loso``). Mirrors ``training.run_phase1_one_season``'s
    style.

    Note: ``test_pairs`` is passed as ``val_pairs`` to ``train_loso_gnn`` --
    this is test-set early stopping, mirroring Phase 1's ``training.py``.
    Documented as a known compromise in the Phase 2 LOSO plan.

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
    hidden_dim, num_layers, dropout, decoder_hidden
        Forwarded to ``GNNStage1Peer`` via ``train_loso_gnn``.
    epochs, lr, patience, seed
        Training-loop hyperparameters.
    emit_pairwise
        If True, include a ``pairwise_df`` key in the returned dict with the
        round-robin pairwise predictions over the holdout's tournament field
        (columns ``team_a``, ``team_b``, ``p_a_wins``; ``team_a < team_b``).
        Default False so existing callers/tests are unaffected.
    encoder
        Either ``"sage"`` (default) or ``"edge_attr"``. Forwarded to
        ``train_loso_gnn``; selects between ``GNNStage1Peer`` (SAGE) and
        ``GNNStage1PeerEdgeAttr`` (GINE consuming ``edge_attr``).

    Returns
    -------
    dict
        ``{holdout_season, gnn, predictions, train_minutes, epochs_run,
        best_epoch, best_val_ll}`` where ``gnn`` carries the summary metrics
        (``ll``, ``accuracy``, ``n``) without the per-pair predictions, and
        ``predictions`` is the full list of per-pair dicts (split out so
        downstream pairwise-CSV emission can find them). When
        ``emit_pairwise=True`` the dict additionally carries ``pairwise_df``
        (a ``pd.DataFrame``) used by the Phase 2 LOSO sweep CLI driver.
    """
    per_season_graphs, train_pairs_by_season, test_pairs, team_index = (
        build_loso_training_data(data_dir, holdout_season, seasons)
    )

    t0 = time.time()
    model, train_info = train_loso_gnn(
        per_season_graphs,
        train_pairs_by_season,
        val_pairs=test_pairs,
        val_graph=per_season_graphs[holdout_season],
        num_nodes=len(team_index),
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        decoder_hidden=decoder_hidden,
        epochs=epochs,
        lr=lr,
        patience=patience,
        seed=seed,
        encoder=encoder,
    )
    train_minutes = (time.time() - t0) / 60.0

    gnn_eval = evaluate_loso(model, test_pairs, per_season_graphs[holdout_season])

    result: dict = {
        "holdout_season": holdout_season,
        "gnn": {k: v for k, v in gnn_eval.items() if k != "predictions"},
        "predictions": gnn_eval["predictions"],
        "train_minutes": train_minutes,
        "epochs_run": train_info["epochs_run"],
        "best_epoch": train_info["best_epoch"],
        "best_val_ll": train_info["best_val_ll"],
    }

    if emit_pairwise:
        tourney = load_tourney_games(data_dir, holdout_season)
        field = sorted(
            set(int(t) for t in tourney["WTeamID"].tolist())
            | set(int(t) for t in tourney["LTeamID"].tolist())
        )
        result["pairwise_df"] = predict_holdout_pairwise(
            model,
            per_season_graphs[holdout_season],
            team_index,
            field,
        )

    return result
