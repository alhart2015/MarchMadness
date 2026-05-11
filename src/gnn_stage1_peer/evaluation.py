"""Phase 1 evaluation: GNN metrics + GNN-vs-Massey comparison."""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

PHASE1_GATE_LL_DELTA = 0.005


def evaluate_gnn_phase1(model, graph, eval_pairs) -> dict:
    """Run GNN on eval_pairs, return LL/accuracy/n/predictions."""
    model.eval()
    a, b, y = eval_pairs
    with torch.no_grad():
        logits = model(graph, a, b)
        probs = torch.sigmoid(logits)
        ll = F.binary_cross_entropy_with_logits(logits, y).item()
        preds = (probs >= 0.5).float()
        acc = (preds == y).float().mean().item()
        n = int(y.numel())
        predictions = [
            {
                "team_a_idx": int(a[i]),
                "team_b_idx": int(b[i]),
                "p_a_wins": float(probs[i]),
                "label": float(y[i]),
            }
            for i in range(n)
        ]
    return {"ll": ll, "accuracy": acc, "n": n, "predictions": predictions}


def compare_gnn_vs_massey(gnn_results: dict, massey_results: dict) -> dict:
    """Apply Phase 1 gate: GNN LL must be at least PHASE1_GATE_LL_DELTA below Massey's."""
    ll_delta = massey_results["ll"] - gnn_results["ll"]
    acc_delta = gnn_results["accuracy"] - massey_results["accuracy"]
    gate_pass = ll_delta >= PHASE1_GATE_LL_DELTA
    return {
        "ll_delta": ll_delta,
        "acc_delta": acc_delta,
        "gate_pass": gate_pass,
        "gnn_ll": gnn_results["ll"],
        "massey_ll": massey_results["ll"],
        "gnn_acc": gnn_results["accuracy"],
        "massey_acc": massey_results["accuracy"],
    }
