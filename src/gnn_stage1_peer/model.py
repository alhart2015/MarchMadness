"""GNN stage-1 peer model (encoder + matchup decoder)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv


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
