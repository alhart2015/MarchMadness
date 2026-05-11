"""GNN stage-1 peer model (encoder + matchup decoder).

Two encoder variants are provided:

- ``GraphSAGEEncoder`` -- 2-layer GraphSAGE with mean aggregation. Consumes
  only ``edge_index``; ignores ``edge_attr`` (Phase 1's deliberate
  simplification).
- ``EdgeAttrAwareEncoder`` -- 2-layer GINE with per-feature scaled
  ``edge_attr`` projected to ``hidden_dim``. The Phase 2 LOSO MARGINAL-row
  structural variant: tests whether feeding edge attributes (margin, site,
  rest, days-from-start) closes the +0.0011 LL-blend headroom gap left by
  the SAGE encoder.

The companion ``GNNStage1Peer`` (SAGE) and ``GNNStage1PeerEdgeAttr`` (GINE)
classes share the same ``MatchupDecoder`` so the comparison isolates the
encoder change.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GINEConv, SAGEConv


# Edge-attr feature scaling. Each feature is shifted+divided to roughly
# [-1, 1] before being projected. Order matches build_pyg_graph's edge_attr
# layout: [score_diff, site_indicator, days_rest, days_from_start].
EDGE_FEATURE_SCALES: tuple[float, float, float, float] = (30.0, 1.0, 7.0, 67.0)
EDGE_FEATURE_OFFSETS: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 67.0)


class GraphSAGEEncoder(nn.Module):
    """2-layer GraphSAGE with learned node embeddings as input features."""

    def __init__(
        self, num_nodes: int, hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.2
    ) -> None:
        super().__init__()
        self.node_emb = nn.Embedding(num_nodes, hidden_dim)
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim, aggr="mean"))
        self.dropout = dropout

    def forward(self, graph: Data) -> torch.Tensor:
        x = self.node_emb.weight  # (num_nodes, hidden_dim)
        for conv in self.convs[:-1]:
            x = conv(x, graph.edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, graph.edge_index)
        return x


class EdgeAttrAwareEncoder(nn.Module):
    """2-layer GINE encoder consuming normalized edge attributes.

    Edge attrs from ``build_pyg_graph``: ``[score_diff, site_indicator,
    days_rest, days_from_start]``. Each feature is normalized via the
    per-feature ``EDGE_FEATURE_OFFSETS`` / ``EDGE_FEATURE_SCALES`` constants
    before being projected to ``hidden_dim`` and consumed by ``GINEConv``.
    """

    def __init__(
        self,
        num_nodes: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        edge_dim: int = 4,
    ) -> None:
        super().__init__()
        self.node_emb = nn.Embedding(num_nodes, hidden_dim)
        self.edge_proj = nn.Linear(edge_dim, hidden_dim)
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.convs.append(GINEConv(nn=mlp, edge_dim=hidden_dim))
        self.dropout = dropout
        self.register_buffer(
            "edge_scales", torch.tensor(EDGE_FEATURE_SCALES, dtype=torch.float)
        )
        self.register_buffer(
            "edge_offsets", torch.tensor(EDGE_FEATURE_OFFSETS, dtype=torch.float)
        )

    def forward(self, graph: Data) -> torch.Tensor:
        x = self.node_emb.weight  # (num_nodes, hidden_dim)
        e_norm = (graph.edge_attr - self.edge_offsets) / self.edge_scales
        e = self.edge_proj(e_norm)
        for conv in self.convs[:-1]:
            x = conv(x, graph.edge_index, e)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, graph.edge_index, e)
        return x


class MatchupDecoder(nn.Module):
    """Concat(a, b, |a-b|) -> 2-layer MLP -> logit."""

    def __init__(self, embed_dim: int, hidden: int = 128) -> None:
        super().__init__()
        self.fc1 = nn.Linear(embed_dim * 3, hidden)
        self.fc2 = nn.Linear(hidden, 1)

    def forward(self, a_emb: torch.Tensor, b_emb: torch.Tensor) -> torch.Tensor:
        x = torch.cat([a_emb, b_emb, (a_emb - b_emb).abs()], dim=-1)
        x = F.relu(self.fc1(x))
        return self.fc2(x).squeeze(-1)


class GNNStage1Peer(nn.Module):
    """Encoder + decoder. Forward: (graph, a_idx, b_idx) -> logits."""

    def __init__(
        self,
        num_nodes: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        decoder_hidden: int = 128,
    ) -> None:
        super().__init__()
        self.encoder = GraphSAGEEncoder(num_nodes, hidden_dim, num_layers, dropout)
        self.decoder = MatchupDecoder(hidden_dim, decoder_hidden)

    def forward(
        self, graph: Data, a_idx: torch.Tensor, b_idx: torch.Tensor
    ) -> torch.Tensor:
        embeds = self.encoder(graph)
        return self.decoder(embeds[a_idx], embeds[b_idx])


class GNNStage1PeerEdgeAttr(nn.Module):
    """``EdgeAttrAwareEncoder`` + ``MatchupDecoder`` -- Phase 2 MARGINAL variant.

    Drop-in replacement for ``GNNStage1Peer`` that consumes ``graph.edge_attr``
    via a GINE encoder. Used only when the LOSO driver is invoked with
    ``encoder="edge_attr"``.
    """

    def __init__(
        self,
        num_nodes: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        decoder_hidden: int = 128,
        edge_dim: int = 4,
    ) -> None:
        super().__init__()
        self.encoder = EdgeAttrAwareEncoder(
            num_nodes, hidden_dim, num_layers, dropout, edge_dim
        )
        self.decoder = MatchupDecoder(hidden_dim, decoder_hidden)

    def forward(
        self, graph: Data, a_idx: torch.Tensor, b_idx: torch.Tensor
    ) -> torch.Tensor:
        embeds = self.encoder(graph)
        return self.decoder(embeds[a_idx], embeds[b_idx])
