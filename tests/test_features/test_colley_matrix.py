"""Unit tests for src/features/colley_matrix.py.

Synthetic round-robin with closed-form Colley solution verifies solver
correctness; real-data smoke test verifies cached loader + sum-to-(n/2)
invariant on actual MRegularSeasonCompactResults.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.features.colley_matrix import (
    _PRODUCER_VERSION,
    compute_colley_ratings,
    load_colley_ratings,
)


def _make_round_robin_wins(team_ids, win_counts, season=2024):
    """Build a round-robin where each pair plays twice, with the winner of
    each pair determined by which team has higher win_counts. For
    win_counts = [6, 4, 2, 0] over 4 teams, this yields A: 6-0, B: 4-2,
    C: 2-4, D: 0-6.

    For pairs of equal win_counts, the games split 1-1.
    """
    rows = []
    daynum = 10
    n = len(team_ids)
    for i in range(n):
        for j in range(i + 1, n):
            wi, wj = win_counts[i], win_counts[j]
            if wi > wj:
                games = [(team_ids[i], team_ids[j]), (team_ids[i], team_ids[j])]
            elif wj > wi:
                games = [(team_ids[j], team_ids[i]), (team_ids[j], team_ids[i])]
            else:
                games = [(team_ids[i], team_ids[j]), (team_ids[j], team_ids[i])]
            for w, l in games:
                rows.append({
                    "Season": season,
                    "DayNum": daynum,
                    "WTeamID": w, "WScore": 75,
                    "LTeamID": l, "LScore": 70,
                    "WLoc": "N", "NumOT": 0,
                })
                daynum += 1
    return pd.DataFrame(rows)


def test_synthetic_round_robin_recovers_colley_ratings():
    """4-team round-robin (each pair plays twice) with W/L = 6-0, 4-2,
    2-4, 0-6 yields closed-form ratings (0.8, 0.6, 0.4, 0.2).

    Derivation: C = (T+2)*I - A where T=6, off-diagonal A_ij = 2
    (each pair plays twice). So C = 10*I - 2*J + 2*I = 10*I - 2*J in
    practice (the +2 prior is folded into the diagonal). Eigenvalue
    along the ones direction is 10 - 2*n = 2; along perp directions is
    10. b = [4, 2, 0, -2] = 1*ones + [3, 1, -1, -3]. Solution
    x = 0.5*ones + (1/10)*[3, 1, -1, -3] = [0.8, 0.6, 0.4, 0.2].
    Sum = 2 = n/2."""
    team_ids = [1101, 1102, 1103, 1104]
    win_counts = [6, 4, 2, 0]
    expected = {1101: 0.8, 1102: 0.6, 1103: 0.4, 1104: 0.2}
    games = _make_round_robin_wins(team_ids, win_counts)

    df = compute_colley_ratings(games)
    assert set(df.columns) == {"Season", "TeamID", "colley_rating"}
    assert len(df) == 4

    rating_by_team = dict(zip(df["TeamID"], df["colley_rating"]))
    for tid, expected_r in expected.items():
        assert rating_by_team[tid] == pytest.approx(expected_r, abs=1e-6), (
            f"Team {tid} expected {expected_r}, got {rating_by_team[tid]}"
        )


def test_sum_to_n_over_two_invariant():
    """Solver enforces sum(ratings) = n/2 by construction."""
    team_ids = [1101, 1102, 1103, 1104]
    games = _make_round_robin_wins(team_ids, [6, 4, 2, 0])

    df = compute_colley_ratings(games)
    n = len(df)
    assert df["colley_rating"].sum() == pytest.approx(n / 2.0, abs=1e-8)


def test_cache_roundtrip(tmp_path: Path):
    """First call writes parquet + sidecar; second call returns cached frame."""
    team_ids = [1101, 1102, 1103, 1104]
    games = _make_round_robin_wins(team_ids, [6, 4, 2, 0])

    df1 = load_colley_ratings(games, cache_dir=tmp_path)
    parquet_path = tmp_path / "colley_ratings.parquet"
    meta_path = tmp_path / "colley_ratings.meta.json"
    assert parquet_path.exists()
    assert meta_path.exists()

    df2 = load_colley_ratings(games, cache_dir=tmp_path)
    pd.testing.assert_frame_equal(
        df1.sort_values(["Season", "TeamID"]).reset_index(drop=True),
        df2.sort_values(["Season", "TeamID"]).reset_index(drop=True),
    )

    meta = json.loads(meta_path.read_text())
    assert meta["producer_version"] == _PRODUCER_VERSION
    assert meta["n_input_rows"] == len(games)
    assert "sha_input" in meta


def test_cache_invalidates_on_meta_mismatch(tmp_path: Path):
    """Sidecar producer_version mismatch triggers rebuild."""
    team_ids = [1101, 1102, 1103, 1104]
    games = _make_round_robin_wins(team_ids, [6, 4, 2, 0])

    df1 = load_colley_ratings(games, cache_dir=tmp_path)
    parquet_path = tmp_path / "colley_ratings.parquet"
    initial_mtime = parquet_path.stat().st_mtime_ns

    meta_path = tmp_path / "colley_ratings.meta.json"
    meta = json.loads(meta_path.read_text())
    meta["producer_version"] = "v0-stale"
    meta_path.write_text(json.dumps(meta))

    df2 = load_colley_ratings(games, cache_dir=tmp_path)
    new_mtime = parquet_path.stat().st_mtime_ns
    assert new_mtime > initial_mtime
    refreshed = json.loads(meta_path.read_text())
    assert refreshed["producer_version"] == _PRODUCER_VERSION
    pd.testing.assert_frame_equal(
        df1.sort_values(["Season", "TeamID"]).reset_index(drop=True),
        df2.sort_values(["Season", "TeamID"]).reset_index(drop=True),
    )


_REG_SEASON_CSV = (
    Path(__file__).resolve().parents[2]
    / "data" / "raw" / "march-machine-learning-2026"
    / "MRegularSeasonCompactResults.csv"
)


@pytest.mark.skipif(not _REG_SEASON_CSV.exists(), reason="raw Kaggle data not available")
def test_real_data_shape_and_rating_range(tmp_path: Path):
    """Solver runs on real Kaggle data; sum-to-(n/2) per season.

    Note on rating range: the plan claimed ratings should be in [0, 1]
    based on Colley's "expected win-rate vs an average opponent"
    interpretation. That interpretation is approximate -- under
    severely unbalanced schedules (e.g., a low-major team that loses
    every game to high-major opponents, or vice versa), the linear
    Colley solve extrapolates beyond [0, 1]. Empirically across
    seasons 2003-2026 we see ratings in roughly [-0.12, 1.14] with
    ~2-3% of team-seasons outside [0, 1]. The test asserts a looser
    [-0.2, 1.2] bound which catches wild solver failures (NaN/inf,
    sign-flips) without falsely claiming a mathematical guarantee
    that does not hold."""
    reg = pd.read_csv(_REG_SEASON_CSV)
    reg = reg[reg["Season"] >= 2003]
    df = load_colley_ratings(reg, cache_dir=tmp_path)

    assert df["colley_rating"].notna().all(), "no NaN ratings"
    assert np.isfinite(df["colley_rating"]).all(), "no inf ratings"
    assert df["colley_rating"].min() >= -0.2
    assert df["colley_rating"].max() <= 1.2

    counts = df.groupby("Season").size()
    assert (counts >= 300).all(), f"min teams per season: {counts.min()}"
    assert (counts <= 380).all(), f"max teams per season: {counts.max()}"

    sums = df.groupby("Season")["colley_rating"].sum()
    expected_sums = counts / 2.0
    diffs = (sums - expected_sums).abs()
    assert diffs.max() < 1e-6, f"sum-to-(n/2) drift: {diffs.max()}"


def test_clause1_pass_when_uncorrelated():
    """Clause 1 passes when colley_rating is uncorrelated with all three
    baselines."""
    from src.diagnose_colley import clause1_correlations
    rng = np.random.default_rng(0)
    n = 100
    fm = pd.DataFrame({
        "Season": [2024] * n,
        "TeamID": list(range(1, n + 1)),
        "colley_rating": rng.standard_normal(n),
        "adj_em": rng.standard_normal(n),
        "massey_composite": rng.standard_normal(n),
        "season_win_pct": rng.standard_normal(n),
    })
    out = clause1_correlations(fm)
    assert out["pass"] is True
