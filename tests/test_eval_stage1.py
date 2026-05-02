"""Unit tests for src/eval_stage1.py."""
import math

import pandas as pd
import pytest


def _write_pairwise(path, rows):
    df = pd.DataFrame(rows, columns=["season", "team_a", "team_b", "p_a_wins"])
    df.to_csv(path, index=False)


def test_per_season_log_loss_known_values(tmp_path):
    """Two pre-computed games, known probabilities and outcomes."""
    from src.eval_stage1 import evaluate_pairwise

    pw = tmp_path / "pw.csv"
    _write_pairwise(pw, [
        (2003, 1101, 1102, 0.9),  # team_a (1101) wins -> p=0.9, label=1
        (2003, 1103, 1104, 0.4),  # team_b (1104) wins -> p_for_a=0.4, label=0
    ])
    results = pd.DataFrame([
        {"Season": 2003, "WTeamID": 1101, "LTeamID": 1102, "DayNum": 136},
        {"Season": 2003, "WTeamID": 1104, "LTeamID": 1103, "DayNum": 136},
    ])

    out = evaluate_pairwise(str(pw), results_df=results)
    assert 2003 in out["per_season"]
    season = out["per_season"][2003]
    # Per-game log loss: -log(0.9) for game 1, -log(0.6) for game 2
    # (game 2's actual winner is 1104; p(1103 beats 1104) = 0.4 ->
    #  p(1104 beats 1103) = 0.6).
    expected = (-math.log(0.9) - math.log(0.6)) / 2
    assert season["log_loss"] == pytest.approx(expected, abs=1e-6)
    assert season["n_games"] == 2
    assert season["accuracy"] == pytest.approx(1.0)


def test_weighted_mean_aggregation(tmp_path):
    """Weighted-mean log loss across seasons weights by n_games per season."""
    from src.eval_stage1 import evaluate_pairwise

    pw = tmp_path / "pw.csv"
    _write_pairwise(pw, [
        (2003, 1101, 1102, 0.9),
        (2004, 1103, 1104, 0.5),
        (2004, 1105, 1106, 0.5),
    ])
    results = pd.DataFrame([
        {"Season": 2003, "WTeamID": 1101, "LTeamID": 1102, "DayNum": 136},
        {"Season": 2004, "WTeamID": 1103, "LTeamID": 1104, "DayNum": 136},
        {"Season": 2004, "WTeamID": 1105, "LTeamID": 1106, "DayNum": 136},
    ])

    out = evaluate_pairwise(str(pw), results_df=results)
    assert out["weighted_mean_log_loss"] > 0
    assert out["weighted_mean_accuracy"] >= 0.0
    assert out["total_games"] == 3


def test_skips_games_without_pairwise_prob(tmp_path):
    """If a real game's pair isn't in the pairwise CSV, skip it without failing."""
    from src.eval_stage1 import evaluate_pairwise

    pw = tmp_path / "pw.csv"
    _write_pairwise(pw, [(2003, 1101, 1102, 0.9)])
    results = pd.DataFrame([
        {"Season": 2003, "WTeamID": 1101, "LTeamID": 1102, "DayNum": 136},
        {"Season": 2003, "WTeamID": 9999, "LTeamID": 8888, "DayNum": 136},  # not in pw
    ])

    out = evaluate_pairwise(str(pw), results_df=results)
    assert out["per_season"][2003]["n_games"] == 1
