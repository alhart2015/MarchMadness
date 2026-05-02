"""Unit tests for src/diagnose_bt_vs_v4.py."""
import pandas as pd
import pytest


def _write_pairwise(path, rows):
    df = pd.DataFrame(rows, columns=["season", "team_a", "team_b", "p_a_wins"])
    df.to_csv(path, index=False)


def test_compute_diagnostic_known_values(tmp_path):
    """Two games per season, known probabilities. Both models predict
    perfectly correlated residuals -> r close to +1; optimal weight
    indeterminate (any weighting gives same log loss when both are
    equally good)."""
    from src.diagnose_bt_vs_v4 import compute_diagnostic

    pw_a = tmp_path / "a.csv"
    pw_b = tmp_path / "b.csv"
    _write_pairwise(pw_a, [
        (2003, 1101, 1102, 0.9),
        (2003, 1103, 1104, 0.4),
    ])
    _write_pairwise(pw_b, [
        (2003, 1101, 1102, 0.9),
        (2003, 1103, 1104, 0.4),
    ])
    results = pd.DataFrame([
        {"Season": 2003, "WTeamID": 1101, "LTeamID": 1102, "DayNum": 136},
        {"Season": 2003, "WTeamID": 1104, "LTeamID": 1103, "DayNum": 136},
    ])

    out = compute_diagnostic(str(pw_a), str(pw_b), results_df=results)
    # Identical predictors -> residual correlation = 1.
    assert out["r_residual"] == pytest.approx(1.0)
    # Both correct: agreement on predicted winner.
    assert out["disagree_n"] == 0
    # Optimal weight: any value gives same loss; picker should still
    # pick something deterministic. Just assert it's in [0, 1].
    assert 0.0 <= out["optimal_w"] <= 1.0


def test_optimal_weight_picks_better_model(tmp_path):
    """If A is much better than B on the only games, optimal weight is
    biased toward A (close to 1.0)."""
    from src.diagnose_bt_vs_v4 import compute_diagnostic

    pw_a = tmp_path / "a.csv"
    pw_b = tmp_path / "b.csv"
    # 5 games where A is consistently right and B consistently wrong.
    _write_pairwise(pw_a, [
        (2003, 1100 + i, 1200 + i, 0.95) for i in range(5)
    ])
    _write_pairwise(pw_b, [
        (2003, 1100 + i, 1200 + i, 0.20) for i in range(5)
    ])
    results = pd.DataFrame([
        {"Season": 2003, "WTeamID": 1100 + i, "LTeamID": 1200 + i, "DayNum": 136}
        for i in range(5)
    ])

    out = compute_diagnostic(str(pw_a), str(pw_b), results_df=results)
    assert out["optimal_w"] >= 0.9, (
        f"optimal_w should heavily favor A, got {out['optimal_w']}"
    )


def test_gate_logic_each_clause():
    """check_gate flips on each clause individually."""
    from src.diagnose_bt_vs_v4 import check_gate

    # Pass case
    base = {"r_residual": 0.4, "optimal_w": 0.6, "headroom": 0.01}
    assert check_gate(base)["pass"] is True

    # Fail r
    diag = dict(base, r_residual=0.7)
    assert check_gate(diag)["pass"] is False
    assert "correlation" in check_gate(diag)["reason"].lower()

    # Fail optimal_w (degenerate v4-dominant)
    diag = dict(base, optimal_w=0.92)
    assert check_gate(diag)["pass"] is False
    assert "weight" in check_gate(diag)["reason"].lower()

    # Fail optimal_w (degenerate bt-dominant)
    diag = dict(base, optimal_w=0.05)
    assert check_gate(diag)["pass"] is False

    # Fail headroom
    diag = dict(base, headroom=0.001)
    assert check_gate(diag)["pass"] is False
    assert "headroom" in check_gate(diag)["reason"].lower()
