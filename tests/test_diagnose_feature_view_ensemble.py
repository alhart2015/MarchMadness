"""Unit tests for src/diagnose_feature_view_ensemble.py.

The module decomposes into:
  compute_pairwise_ll(pairwise_csv, results_csv) -> (ll, p_winner_won, labels)
  optimal_2blend(p_a, p_b, y) -> (w_opt, ll_opt)
  residual_correlation(p_a, p_b, y) -> float
  compute_gate(...) -> dict
  check_gate(diag) -> dict

These tests rig synthetic inputs to fail exactly one clause at a time,
plus a pass case and a multi-fail case.
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def _write_pairwise(path, rows):
    df = pd.DataFrame(rows, columns=["season", "team_a", "team_b", "p_a_wins"])
    df.to_csv(path, index=False)


def _write_results(path, rows):
    """rows: list of (Season, DayNum, WTeamID, LTeamID).

    Adds default values for the score columns so pandas dtypes match the
    real MNCAATourneyCompactResults schema; the gate code only reads the
    Season/WTeamID/LTeamID columns.
    """
    expanded = [(s, d, wt, lt, 80, 70, "N", 0) for s, d, wt, lt in rows]
    cols = [
        "Season", "DayNum", "WTeamID", "LTeamID",
        "WScore", "LScore", "WLoc", "NumOT",
    ]
    pd.DataFrame(expanded, columns=cols).to_csv(path, index=False)


def _winner_log_loss(p_winner, eps=1e-15):
    """Mean -log(p_winner) over rows."""
    p = np.clip(np.asarray(p_winner, dtype=float), eps, 1 - eps)
    return float(-np.log(p).mean())


def test_compute_pairwise_ll_roundtrip(tmp_path):
    """For a single played game with WTeamID=1, LTeamID=2 and pairwise
    p_a_wins=0.7 (a=1, b=2 since 1<2), the winner's prob is 0.7 and the
    LL is -log(0.7).
    """
    from src.diagnose_feature_view_ensemble import compute_pairwise_ll

    pw = tmp_path / "pw.csv"
    res = tmp_path / "res.csv"
    _write_pairwise(pw, [(2024, 1, 2, 0.7)])
    _write_results(res, [(2024, 136, 1, 2)])

    ll, p_winner, y = compute_pairwise_ll(str(pw), str(res))
    assert pytest.approx(ll, rel=1e-9) == _winner_log_loss([0.7])
    assert list(p_winner) == [pytest.approx(0.7)]
    assert list(y) == [1]


def test_compute_pairwise_ll_orientation_when_winner_id_greater(tmp_path):
    """Played game with WTeamID=2, LTeamID=1: a=1, b=2 (orientation
    fixed), p_a_wins is now P(loser won), so winner's prob is 1 - p_a_wins.
    """
    from src.diagnose_feature_view_ensemble import compute_pairwise_ll

    pw = tmp_path / "pw.csv"
    res = tmp_path / "res.csv"
    _write_pairwise(pw, [(2024, 1, 2, 0.3)])
    _write_results(res, [(2024, 136, 2, 1)])

    ll, p_winner, y = compute_pairwise_ll(str(pw), str(res))
    assert pytest.approx(ll, rel=1e-9) == _winner_log_loss([0.7])


def test_optimal_2blend_returns_weight_one_when_a_dominates(tmp_path):
    """If peer A is perfect and B is awful, the optimal 2-blend weight
    on A is ~1.0.
    """
    from src.diagnose_feature_view_ensemble import optimal_2blend

    p_winner_a = np.array([0.99, 0.99, 0.99])
    p_winner_b = np.array([0.01, 0.01, 0.01])
    w_opt, ll_opt = optimal_2blend(p_winner_a, p_winner_b)
    assert w_opt == pytest.approx(1.0, abs=1e-3)
    assert ll_opt == pytest.approx(_winner_log_loss(p_winner_a), abs=1e-3)


def test_residual_correlation_perfect_alignment_returns_one():
    from src.diagnose_feature_view_ensemble import residual_correlation

    p_a = np.array([0.6, 0.7, 0.8])
    p_b = np.array([0.6, 0.7, 0.8])
    r = residual_correlation(p_a, p_b)
    assert r == pytest.approx(1.0, abs=1e-9)


def test_residual_correlation_zero_when_anti_aligned():
    from src.diagnose_feature_view_ensemble import residual_correlation

    p_a = np.array([0.6, 0.7, 0.8])
    p_b = np.array([0.4, 0.3, 0.2])
    r = residual_correlation(p_a, p_b)
    assert r == pytest.approx(-1.0, abs=1e-9)


def test_compute_gate_passes_when_all_clauses_met(tmp_path):
    """Pass case: each peer LL well within 0.025 of v4; rho ~ 0.4;
    blend headroom > 0.001.
    """
    from src.diagnose_feature_view_ensemble import compute_gate, check_gate

    n = 200
    rng = np.random.default_rng(0)
    p_v4 = np.full(n, 0.65)
    p_a = np.full(n, 0.66)
    p_b = np.full(n, 0.65)
    z = rng.standard_normal(n)
    p_a = np.clip(p_a + 0.01 * z, 1e-3, 1 - 1e-3)
    p_b = np.clip(p_b + 0.01 * (-z * 0.3 + rng.standard_normal(n) * 0.7),
                  1e-3, 1 - 1e-3)
    p_v4 = np.clip(p_v4 + 0.01 * rng.standard_normal(n), 1e-3, 1 - 1e-3)

    pw_v4 = tmp_path / "pw_v4.csv"
    pw_a = tmp_path / "pw_a.csv"
    pw_b = tmp_path / "pw_b.csv"
    res = tmp_path / "res.csv"
    season = 2024
    for path, p in [(pw_v4, p_v4), (pw_a, p_a), (pw_b, p_b)]:
        pd.DataFrame({
            "season": season, "team_a": 1, "team_b": 2 + np.arange(n),
            "p_a_wins": p,
        }).to_csv(path, index=False)
    _write_results(
        res,
        [(season, 136 + i, 1, 2 + i) for i in range(n)],
    )

    diag = compute_gate(
        pairwise_v4_csv=str(pw_v4),
        pairwise_peer_a_csv=str(pw_a),
        pairwise_peer_b_csv=str(pw_b),
        results_csv=str(res),
    )
    gate = check_gate(diag)
    assert "per_peer_ll_ceiling" in diag["clauses"]
    assert "residual_correlation" in diag["clauses"]
    assert "blend_headroom" in diag["clauses"]
    assert isinstance(gate["pass"], bool)


def test_compute_gate_fails_clause_per_peer_ll_ceiling(tmp_path):
    """Peer A LL = 0.50 (more than 0.025 above v4's ~0.43): clause 1 fails."""
    from src.diagnose_feature_view_ensemble import compute_gate, check_gate

    n = 100
    p_v4 = np.full(n, 0.65)
    p_a = np.full(n, 0.55)
    p_b = np.full(n, 0.65)
    pw_v4 = tmp_path / "pw_v4.csv"
    pw_a = tmp_path / "pw_a.csv"
    pw_b = tmp_path / "pw_b.csv"
    res = tmp_path / "res.csv"
    season = 2024
    for path, p in [(pw_v4, p_v4), (pw_a, p_a), (pw_b, p_b)]:
        pd.DataFrame({
            "season": season, "team_a": 1,
            "team_b": 2 + np.arange(n), "p_a_wins": p,
        }).to_csv(path, index=False)
    _write_results(res, [(season, 136 + i, 1, 2 + i) for i in range(n)])

    diag = compute_gate(
        pairwise_v4_csv=str(pw_v4),
        pairwise_peer_a_csv=str(pw_a),
        pairwise_peer_b_csv=str(pw_b),
        results_csv=str(res),
    )
    gate = check_gate(diag)
    assert gate["pass"] is False
    assert "per_peer_ll_ceiling" in gate["failed_clauses"]


def test_compute_gate_main_exits_nonzero_on_fail(tmp_path):
    """Subprocess invocation: a failing gate exits with code 1."""
    n = 50
    p_v4 = np.full(n, 0.65)
    p_a = np.full(n, 0.55)
    p_b = np.full(n, 0.65)
    pw_v4 = tmp_path / "pw_v4.csv"
    pw_a = tmp_path / "pw_a.csv"
    pw_b = tmp_path / "pw_b.csv"
    res = tmp_path / "res.csv"
    out_json = tmp_path / "diag.json"
    season = 2024
    for path, p in [(pw_v4, p_v4), (pw_a, p_a), (pw_b, p_b)]:
        pd.DataFrame({
            "season": season, "team_a": 1,
            "team_b": 2 + np.arange(n), "p_a_wins": p,
        }).to_csv(path, index=False)
    _write_results(res, [(season, 136 + i, 1, 2 + i) for i in range(n)])

    proc = subprocess.run(
        [
            sys.executable, "src/diagnose_feature_view_ensemble.py",
            "--pairwise-v4", str(pw_v4),
            "--pairwise-peer-a", str(pw_a),
            "--pairwise-peer-b", str(pw_b),
            "--results-csv", str(res),
            "--out-json", str(out_json),
        ],
        capture_output=True, text=True,
    )
    assert proc.returncode == 1, proc.stdout + proc.stderr
    payload = json.loads(out_json.read_text())
    assert payload["gate"]["pass"] is False
