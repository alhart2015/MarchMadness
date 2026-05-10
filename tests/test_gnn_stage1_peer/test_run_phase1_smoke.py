"""Real-data smoke test on season 2024.

This test EXPECTS the Kaggle data files to be present at
data/raw/march-machine-learning-2026/. It is slow (~30-90s on CPU);
mark @pytest.mark.slow if you want to skip it during fast runs.
"""
import pytest
from pathlib import Path


@pytest.mark.slow
def test_run_phase1_one_season_2024_returns_well_formed():
    from src.gnn_stage1_peer.training import run_phase1_one_season
    data_dir = Path("data/raw/march-machine-learning-2026")
    out = run_phase1_one_season(
        data_dir=data_dir,
        season=2024,
        hidden_dim=64,
        epochs=30,
        lr=1e-3,
        patience=5,
        seed=42,
    )
    assert {"season", "gnn", "massey", "compare", "train_minutes"} <= out.keys()
    assert out["season"] == 2024
    assert out["gnn"]["n"] > 100  # 2024 has hundreds of late-RS games
    assert 0.4 <= out["gnn"]["ll"] <= 1.0
    assert 0.4 <= out["massey"]["ll"] <= 1.0
    assert out["train_minutes"] < 30.0  # if this exceeds 30, escalate per spec
