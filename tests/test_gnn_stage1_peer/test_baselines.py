import pandas as pd
import pytest
from pathlib import Path


def _toy_massey_csv(tmp_path: Path) -> Path:
    df = pd.DataFrame([
        # Two systems, one season, three teams. Lower rank = better.
        {"Season": 2024, "RankingDayNum": 100, "SystemName": "POM", "TeamID": 1101, "OrdinalRank": 5},
        {"Season": 2024, "RankingDayNum": 100, "SystemName": "POM", "TeamID": 1102, "OrdinalRank": 50},
        {"Season": 2024, "RankingDayNum": 100, "SystemName": "POM", "TeamID": 1103, "OrdinalRank": 200},
        {"Season": 2024, "RankingDayNum": 100, "SystemName": "MAS", "TeamID": 1101, "OrdinalRank": 7},
        {"Season": 2024, "RankingDayNum": 100, "SystemName": "MAS", "TeamID": 1102, "OrdinalRank": 60},
        {"Season": 2024, "RankingDayNum": 100, "SystemName": "MAS", "TeamID": 1103, "OrdinalRank": 180},
        # Future-dated row should be ignored when ranking_day=100
        {"Season": 2024, "RankingDayNum": 133, "SystemName": "POM", "TeamID": 1101, "OrdinalRank": 1},
        # Other-season row should be ignored
        {"Season": 2023, "RankingDayNum": 100, "SystemName": "POM", "TeamID": 1101, "OrdinalRank": 99},
    ])
    p = tmp_path / "MMasseyOrdinals.csv"
    df.to_csv(p, index=False)
    return tmp_path


def test_load_massey_composite_filters_season_and_day(tmp_path):
    from src.gnn_stage1_peer.baselines import load_massey_composite
    data_dir = _toy_massey_csv(tmp_path)
    ranks = load_massey_composite(data_dir, season=2024, ranking_day=100)
    # Composite is mean across systems at the latest day <= ranking_day.
    # 1101: (5 + 7) / 2 = 6.0
    assert ranks[1101] == 6.0
    assert ranks[1102] == 55.0
    assert ranks[1103] == 190.0


def test_predict_massey_logit_signs():
    from src.gnn_stage1_peer.baselines import predict_massey_logit
    ranks = {1101: 5.0, 1102: 50.0}
    # 1101 better (lower rank); A=1101 favored -> positive logit.
    logit = predict_massey_logit(1101, 1102, ranks, scale=0.05)
    assert logit > 0
    # Reverse orientation -> negative.
    assert predict_massey_logit(1102, 1101, ranks, scale=0.05) == pytest.approx(-logit)


def test_evaluate_massey_baseline_returns_ll_and_acc(tmp_path):
    from src.gnn_stage1_peer.baselines import evaluate_massey_baseline
    data_dir = _toy_massey_csv(tmp_path)
    test_games = pd.DataFrame([
        {"Season": 2024, "DayNum": 125, "WTeamID": 1101, "WScore": 80, "LTeamID": 1103, "LScore": 60, "WLoc": "N"},
        {"Season": 2024, "DayNum": 130, "WTeamID": 1102, "WScore": 75, "LTeamID": 1103, "LScore": 65, "WLoc": "N"},
    ])
    out = evaluate_massey_baseline(test_games, season=2024, data_dir=data_dir, scale=0.05)
    # Both games: better-ranked team won. Massey logit positive both times -> p > 0.5 -> acc=1.0.
    assert out["accuracy"] == 1.0
    assert out["n"] == 4  # symmetric, 2 games -> 4 pairs
    assert out["ll"] > 0  # standard convention: positive LL = mean BCE loss
