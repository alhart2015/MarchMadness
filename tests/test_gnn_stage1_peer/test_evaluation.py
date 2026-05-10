def test_evaluate_gnn_phase1_returns_well_formed_dict():
    import torch
    from torch_geometric.data import Data
    from src.gnn_stage1_peer.model import GNNStage1Peer
    from src.gnn_stage1_peer.evaluation import evaluate_gnn_phase1
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    g = Data(edge_index=edge_index, edge_attr=torch.zeros((2, 4)), num_nodes=2)
    m = GNNStage1Peer(num_nodes=2, hidden_dim=4, dropout=0.0, decoder_hidden=4)
    a = torch.tensor([0, 1], dtype=torch.long)
    b = torch.tensor([1, 0], dtype=torch.long)
    y = torch.tensor([1, 0], dtype=torch.float)
    out = evaluate_gnn_phase1(m, g, (a, b, y))
    assert {"ll", "accuracy", "n", "predictions"} <= out.keys()
    assert out["n"] == 2


def test_compare_gnn_vs_massey_gate_logic():
    import pytest
    from src.gnn_stage1_peer.evaluation import compare_gnn_vs_massey
    # GNN clearly better
    out = compare_gnn_vs_massey({"ll": 0.50, "accuracy": 0.72}, {"ll": 0.51, "accuracy": 0.70})
    assert out["ll_delta"] == pytest.approx(0.01)
    assert out["gate_pass"] is True
    # GNN essentially flat
    out = compare_gnn_vs_massey({"ll": 0.508, "accuracy": 0.71}, {"ll": 0.510, "accuracy": 0.71})
    assert out["gate_pass"] is False
