"""Training loop for the Phase 1 GNN with early stopping + determinism."""
from __future__ import annotations

import math
import os
import random

import numpy as np
import torch
import torch.nn.functional as F


def set_determinism(seed: int) -> None:
    """Set all relevant seeds for reproducibility (CPU)."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():  # no-op on CPU machines, hedged for completeness
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(False)  # SAGE/CUDA paths require non-strict mode


def _bce_logits_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(logits, targets)


def _eval_ll(model, graph, pairs) -> float:
    model.eval()
    with torch.no_grad():
        a, b, y = pairs
        logits = model(graph, a, b)
        return _bce_logits_loss(logits, y).item()


def train_gnn(
    model,
    graph,
    train_pairs,
    val_pairs,
    *,
    epochs: int = 50,
    lr: float = 1e-3,
    patience: int = 5,
    seed: int = 42,
) -> dict:
    """Train the GNN with Adam + early stopping on val LL."""
    set_determinism(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    a, b, y = train_pairs
    history = {"loss": [], "val_ll": []}
    best_val = math.inf
    best_state = None
    bad_epochs = 0
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(graph, a, b)
        loss = _bce_logits_loss(logits, y)
        loss.backward()
        optimizer.step()
        history["loss"].append(loss.item())
        val_ll = _eval_ll(model, graph, val_pairs)
        history["val_ll"].append(val_ll)
        if val_ll < best_val - 1e-5:
            best_val = val_ll
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return {
        "best_val_ll": best_val,
        "best_epoch": int(np.argmin(history["val_ll"])),
        "epochs_run": len(history["loss"]),
        "train_history": history,
    }
