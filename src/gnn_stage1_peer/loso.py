"""LOSO data pipeline + cross-season training for the GNN stage-1 peer (Phase 2).

Task B: build per-season RS graphs, train pairs (tournament games from
non-holdout seasons), and test pairs (holdout season's tournament games).
The global team_index is shared across seasons so a single
``nn.Embedding(num_teams, hidden_dim)`` covers every team that appears in any
requested season's regular-season data.
"""
from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import pandas as pd
import torch
from torch_geometric.data import Data

from .data import build_global_team_index, load_rs_games
from .graph import build_matchup_pairs, build_pyg_graph


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
