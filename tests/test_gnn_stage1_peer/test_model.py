import torch
from torch_geometric.data import Data


def _toy_graph(num_nodes: int = 4) -> Data:
    # 4 nodes, 6 directed edges (3 bidirected pairs)
    edge_index = torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]], dtype=torch.long)
    edge_attr = torch.zeros((6, 4), dtype=torch.float)
    return Data(edge_index=edge_index, edge_attr=edge_attr, num_nodes=num_nodes)


def _toy_graph_with_edge_attr(num_nodes: int = 4) -> Data:
    """Same topology as _toy_graph but with non-zero edge attributes."""
    edge_index = torch.tensor(
        [[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]], dtype=torch.long
    )
    # Per-edge [score_diff, site, days_rest, days_from_start].
    edge_attr = torch.tensor(
        [
            [10.0, 1.0, 3.0, 30.0],
            [-10.0, -1.0, 3.0, 30.0],
            [5.0, 0.0, 2.0, 50.0],
            [-5.0, 0.0, 2.0, 50.0],
            [20.0, 1.0, 1.0, 80.0],
            [-20.0, -1.0, 1.0, 80.0],
        ],
        dtype=torch.float,
    )
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


def test_edge_attr_encoder_forward_shape():
    """EdgeAttrAwareEncoder forward returns (num_nodes, hidden_dim)."""
    from src.gnn_stage1_peer.model import EdgeAttrAwareEncoder

    enc = EdgeAttrAwareEncoder(
        num_nodes=4, hidden_dim=32, num_layers=2, dropout=0.0
    )
    g = _toy_graph_with_edge_attr()
    out = enc(g)
    assert out.shape == (4, 32)


def test_edge_attr_encoder_uses_edge_attr():
    """Gradient w.r.t. graph.edge_attr is nonzero -- encoder actually consumes it."""
    from src.gnn_stage1_peer.model import EdgeAttrAwareEncoder

    torch.manual_seed(0)
    enc = EdgeAttrAwareEncoder(
        num_nodes=4, hidden_dim=16, num_layers=2, dropout=0.0
    )
    g = _toy_graph_with_edge_attr()
    # Detach + require grad so we can read d(out)/d(edge_attr).
    g.edge_attr = g.edge_attr.detach().clone().requires_grad_(True)
    out = enc(g)
    out.sum().backward()
    assert g.edge_attr.grad is not None
    assert g.edge_attr.grad.abs().sum().item() > 0.0, (
        "edge_attr gradient is zero -- encoder is ignoring edge_attr."
    )


def test_gnn_stage1_peer_edge_attr_forward():
    """GNNStage1PeerEdgeAttr forward returns logits shape == (num_pairs,)."""
    from src.gnn_stage1_peer.model import GNNStage1PeerEdgeAttr

    m = GNNStage1PeerEdgeAttr(
        num_nodes=4,
        hidden_dim=16,
        num_layers=2,
        dropout=0.0,
        decoder_hidden=32,
    )
    g = _toy_graph_with_edge_attr()
    a = torch.tensor([0, 1, 2], dtype=torch.long)
    b = torch.tensor([1, 2, 3], dtype=torch.long)
    logits = m(g, a, b)
    assert logits.shape == (3,)
