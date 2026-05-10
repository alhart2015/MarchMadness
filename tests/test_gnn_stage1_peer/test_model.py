import torch
from torch_geometric.data import Data


def _toy_graph(num_nodes: int = 4) -> Data:
    # 4 nodes, 6 directed edges (3 bidirected pairs)
    edge_index = torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]], dtype=torch.long)
    edge_attr = torch.zeros((6, 4), dtype=torch.float)
    return Data(edge_index=edge_index, edge_attr=edge_attr, num_nodes=num_nodes)


def test_graphsage_encoder_output_shape():
    from src.gnn_stage1_peer.model import GraphSAGEEncoder
    enc = GraphSAGEEncoder(num_nodes=4, hidden_dim=64, num_layers=2, dropout=0.0)
    g = _toy_graph()
    out = enc(g)
    assert out.shape == (4, 64)


def test_matchup_decoder_output_shape():
    from src.gnn_stage1_peer.model import MatchupDecoder
    dec = MatchupDecoder(embed_dim=64, hidden=128)
    a_emb = torch.randn(8, 64)
    b_emb = torch.randn(8, 64)
    logits = dec(a_emb, b_emb)
    assert logits.shape == (8,)


def test_full_model_forward_pass():
    from src.gnn_stage1_peer.model import GNNStage1Peer
    m = GNNStage1Peer(num_nodes=4, hidden_dim=64, num_layers=2, dropout=0.0, decoder_hidden=128)
    g = _toy_graph()
    a = torch.tensor([0, 1, 2], dtype=torch.long)
    b = torch.tensor([1, 2, 3], dtype=torch.long)
    logits = m(g, a, b)
    assert logits.shape == (3,)
