"""Unit tests for src/diagnose_gnn_vs_v4.py."""
import pandas as pd
import pytest


def _write_pairwise(path, rows):
    df = pd.DataFrame(rows, columns=["season", "team_a", "team_b", "p_a_wins"])
    df.to_csv(path, index=False)


def test_compute_diagnostic_shape(tmp_path):
    """Diagnostic dict has all expected keys with sensible types/ranges."""
    from src.diagnose_gnn_vs_v4 import compute_diagnostic

    pw_v4 = tmp_path / "v4.csv"
    pw_gnn = tmp_path / "gnn.csv"
    # 5 games, both models reasonable but different
    _write_pairwise(pw_v4, [
        (2003, 1100 + i, 1200 + i, 0.70) for i in range(5)
    ])
    _write_pairwise(pw_gnn, [
        (2003, 1100 + i, 1200 + i, 0.65) for i in range(5)
    ])
    results = pd.DataFrame([
        {"Season": 2003, "WTeamID": 1100 + i, "LTeamID": 1200 + i, "DayNum": 136}
        for i in range(5)
    ])

    out = compute_diagnostic(str(pw_v4), str(pw_gnn), results_df=results)

    expected_keys = {
        "n_games", "ll_v4", "ll_gnn", "acc_v4", "acc_gnn",
        "r_residual", "disagree_n", "both_correct", "v4_only_correct",
        "gnn_only_correct", "both_wrong",
        "optimal_w", "optimal_ll", "headroom", "ll_at_w",
    }
    assert expected_keys.issubset(set(out.keys()))

    assert out["n_games"] == 5
    assert isinstance(out["ll_v4"], float)
    assert isinstance(out["ll_gnn"], float)
    assert out["ll_v4"] > 0  # log loss is non-negative
    assert out["ll_gnn"] > 0
    assert 0.0 <= out["acc_v4"] <= 1.0
    assert 0.0 <= out["acc_gnn"] <= 1.0
    assert 0.0 <= out["optimal_w"] <= 1.0
    assert out["optimal_ll"] > 0
    assert len(out["ll_at_w"]) == 101


def test_check_gate_passes_when_all_three_clauses_clear():
    """check_gate returns pass=True when r<0.60, w in [0.40, 0.85], headroom>0.005."""
    from src.diagnose_gnn_vs_v4 import check_gate

    diag = {"r_residual": 0.5, "optimal_w": 0.6, "headroom": 0.01}
    result = check_gate(diag)
    assert result["pass"] is True
    assert "cleared" in result["reason"].lower()


def test_check_gate_fails_on_each_clause():
    """check_gate flips on each clause individually."""
    from src.diagnose_gnn_vs_v4 import check_gate

    base = {"r_residual": 0.5, "optimal_w": 0.6, "headroom": 0.01}

    # Fail clause 1: r too high
    diag = dict(base, r_residual=0.7)
    out = check_gate(diag)
    assert out["pass"] is False
    assert "correlation" in out["reason"].lower()

    # Fail clause 2: w outside [0.40, 0.85] -- too v4-heavy
    diag = dict(base, optimal_w=0.95)
    out = check_gate(diag)
    assert out["pass"] is False
    assert "weight" in out["reason"].lower()

    # Fail clause 2: w outside [0.40, 0.85] -- too GNN-heavy (lower bound is 0.40)
    diag = dict(base, optimal_w=0.30)
    out = check_gate(diag)
    assert out["pass"] is False
    assert "weight" in out["reason"].lower()

    # Fail clause 3: headroom too small
    diag = dict(base, headroom=0.001)
    out = check_gate(diag)
    assert out["pass"] is False
    assert "headroom" in out["reason"].lower()


def test_compute_per_season_diagnostic_returns_per_season_rows(tmp_path):
    """Synthetic 2-season dataset yields a 2-row DataFrame with right columns."""
    from src.diagnose_gnn_vs_v4 import compute_per_season_diagnostic

    pw_v4 = tmp_path / "v4.csv"
    pw_gnn = tmp_path / "gnn.csv"
    rows_v4 = []
    rows_gnn = []
    for season in (2003, 2004):
        for i in range(4):
            rows_v4.append((season, 1100 + i, 1200 + i, 0.70))
            rows_gnn.append((season, 1100 + i, 1200 + i, 0.60))
    _write_pairwise(pw_v4, rows_v4)
    _write_pairwise(pw_gnn, rows_gnn)

    results_rows = []
    for season in (2003, 2004):
        for i in range(4):
            results_rows.append({
                "Season": season,
                "WTeamID": 1100 + i,
                "LTeamID": 1200 + i,
                "DayNum": 136,
            })
    results = pd.DataFrame(results_rows)

    df = compute_per_season_diagnostic(
        str(pw_v4), str(pw_gnn), results_df=results,
    )
    expected_cols = [
        "season", "n_games", "ll_v4", "ll_gnn",
        "r_residual", "optimal_w", "optimal_ll", "headroom",
    ]
    assert list(df.columns) == expected_cols
    assert len(df) == 2
    assert sorted(df["season"].tolist()) == [2003, 2004]
    assert (df["n_games"] == 4).all()
    assert (df["ll_v4"] > 0).all()
    assert (df["ll_gnn"] > 0).all()


def test_optimal_w_finds_blend_strictly_between_endpoints(tmp_path):
    """Two synthetic frames where each model is right on a different half:
    optimal_w should be in (0, 1), not at an endpoint."""
    from src.diagnose_gnn_vs_v4 import compute_diagnostic

    pw_v4 = tmp_path / "v4.csv"
    pw_gnn = tmp_path / "gnn.csv"
    # 10 games. v4 confident-correct on first 5, confident-wrong on last 5.
    # GNN confident-wrong on first 5, confident-correct on last 5.
    rows_v4 = []
    rows_gnn = []
    for i in range(10):
        a = 1100 + i
        b = 1200 + i
        if i < 5:
            rows_v4.append((2003, a, b, 0.90))   # picks a
            rows_gnn.append((2003, a, b, 0.10))  # picks b
        else:
            rows_v4.append((2003, a, b, 0.10))   # picks b
            rows_gnn.append((2003, a, b, 0.90))  # picks a
    _write_pairwise(pw_v4, rows_v4)
    _write_pairwise(pw_gnn, rows_gnn)

    # In all 10 games, team a wins. So v4 is right on first 5, wrong on
    # last 5; GNN is wrong on first 5, right on last 5. A 50/50 blend
    # gives 0.50 every game which is much better than either alone.
    results = pd.DataFrame([
        {"Season": 2003, "WTeamID": 1100 + i, "LTeamID": 1200 + i, "DayNum": 136}
        for i in range(10)
    ])

    out = compute_diagnostic(str(pw_v4), str(pw_gnn), results_df=results)
    assert 0.0 < out["optimal_w"] < 1.0, (
        f"optimal_w should be strictly between 0 and 1, got {out['optimal_w']}"
    )
    # And the blend should beat either standalone.
    assert out["optimal_ll"] < out["ll_v4"]
    assert out["optimal_ll"] < out["ll_gnn"]
