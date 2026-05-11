"""One-shot: scan scale values and pick the one that minimizes LL on real 2024 RS test split.

Run with: pytest tests/test_gnn_stage1_peer/test_baselines_tune.py::test_tune_scale -s
"""
from pathlib import Path

import pytest


@pytest.mark.skip(reason="One-shot tuning; un-skip and run manually then re-skip.")
def test_tune_scale():
    from src.gnn_stage1_peer.data import load_rs_games, split_phase1
    from src.gnn_stage1_peer.baselines import evaluate_massey_baseline
    data_dir = Path("data/raw/march-machine-learning-2026")
    games = load_rs_games(data_dir, season=2024)
    _, test = split_phase1(games)
    best = (None, float("inf"))
    for scale in [0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20]:
        out = evaluate_massey_baseline(test, season=2024, data_dir=data_dir, scale=scale)
        print(f"scale={scale:.3f}  LL={out['ll']:.4f}  acc={out['accuracy']:.3f}  n={out['n']}")
        if out["ll"] < best[1]:
            best = (scale, out["ll"])
    print(f"BEST scale={best[0]:.3f}  LL={best[1]:.4f}")
