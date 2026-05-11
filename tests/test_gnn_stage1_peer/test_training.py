import torch
from torch_geometric.data import Data


def _toy_setup(num_nodes: int = 6, seed: int = 0):
    """Build a separable toy: nodes 0-2 always beat nodes 3-5."""
    torch.manual_seed(seed)
    # Edges: each "good" team beat each "bad" team once
    src, dst, attr = [], [], []
    for w in (0, 1, 2):
        for l in (3, 4, 5):
            src += [w, l]; dst += [l, w]
            attr += [[10.0, 0.0, 0.0, 50.0], [-10.0, 0.0, 0.0, 50.0]]
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_attr = torch.tensor(attr, dtype=torch.float)
    g = Data(edge_index=edge_index, edge_attr=edge_attr, num_nodes=num_nodes)
    # Train pairs: same separable structure
    a = torch.tensor([0, 1, 2, 3, 4, 5, 0, 1, 2], dtype=torch.long)
    b = torch.tensor([3, 4, 5, 0, 1, 2, 4, 5, 3], dtype=torch.long)
    y = torch.tensor([1, 1, 1, 0, 0, 0, 1, 1, 1], dtype=torch.float)
    return g, (a, b, y)


def test_train_gnn_loss_decreases():
    from src.gnn_stage1_peer.model import GNNStage1Peer
    from src.gnn_stage1_peer.training import train_gnn, set_determinism
    set_determinism(42)
    g, (a, b, y) = _toy_setup()
    model = GNNStage1Peer(num_nodes=6, hidden_dim=16, dropout=0.0, decoder_hidden=32)
    history = train_gnn(model, g, (a, b, y), (a, b, y), epochs=30, lr=0.05, patience=10, seed=42)
    losses = history["train_history"]["loss"]
    # On a separable toy task, loss should drop substantially.
    assert losses[-1] < losses[0] * 0.5
    assert losses[-1] < 0.4  # near-perfect separation


def test_set_determinism_reproducible():
    from src.gnn_stage1_peer.training import set_determinism
    set_determinism(42)
    a1 = torch.randn(3)
    set_determinism(42)
    a2 = torch.randn(3)
    assert torch.equal(a1, a2)
