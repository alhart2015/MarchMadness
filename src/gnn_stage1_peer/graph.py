"""PyG graph construction from RS games."""
from __future__ import annotations

import pandas as pd
import torch
from torch_geometric.data import Data


SITE_INDICATOR = {"H": 1.0, "A": -1.0, "N": 0.0}


def build_pyg_graph(train_games: pd.DataFrame, team_index: dict[int, int]) -> Data:
    """Build a bidirected PyG graph from training-set games.

    Edges are bidirected: each game (W, L) produces two directed edges:
    - (W -> L) with score_diff = +(WScore - LScore), site_indicator = +1 if WLoc=H else 0/-1
    - (L -> W) with score_diff = -(WScore - LScore), site_indicator = -1*above

    Edge attribute layout: [score_diff, site_indicator, days_rest, days_from_season_start].
    Node features are not added in this task -- the encoder uses learned embeddings.
    """
    src_list, dst_list, attr_list = [], [], []
    for _, g in train_games.iterrows():
        w_idx = team_index[int(g["WTeamID"])]
        l_idx = team_index[int(g["LTeamID"])]
        score_diff = float(g["WScore"] - g["LScore"])
        site = SITE_INDICATOR[g["WLoc"]]
        days_from_start = float(g["DayNum"])
        # Days rest is hard to compute without sorting; use 0 placeholder for now.
        # If signal is dependent on rest, refine in a follow-up task.
        days_rest = 0.0
        # W -> L edge: score_diff positive
        src_list.append(w_idx)
        dst_list.append(l_idx)
        attr_list.append([score_diff, site, days_rest, days_from_start])
        # L -> W edge: flip score_diff and site
        src_list.append(l_idx)
        dst_list.append(w_idx)
        attr_list.append([-score_diff, -site, days_rest, days_from_start])

    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    edge_attr = torch.tensor(attr_list, dtype=torch.float)
    num_nodes = len(team_index)
    return Data(edge_index=edge_index, edge_attr=edge_attr, num_nodes=num_nodes)


def build_matchup_pairs(
    games: pd.DataFrame, team_index: dict[int, int]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build symmetric matchup pairs for training/eval.

    Each game produces two pairs:
    - (W, L, label=1) -- W wins
    - (L, W, label=0) -- W still wins from L's perspective

    Mirrors v3's symmetric-matchup convention (`src/models/matchup.py`).
    """
    a_list, b_list, y_list = [], [], []
    for _, g in games.iterrows():
        w_idx = team_index[int(g["WTeamID"])]
        l_idx = team_index[int(g["LTeamID"])]
        a_list.append(w_idx); b_list.append(l_idx); y_list.append(1.0)
        a_list.append(l_idx); b_list.append(w_idx); y_list.append(0.0)
    return (
        torch.tensor(a_list, dtype=torch.long),
        torch.tensor(b_list, dtype=torch.long),
        torch.tensor(y_list, dtype=torch.float),
    )
