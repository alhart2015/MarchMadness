"""Unit tests for src/eval_v4_calibration.py."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def test_drop_best_season_delta_subtracts_largest_positive():
    """drop_best_season_delta = sum(per_season_delta) - max(per_season_delta).
    The 'best' season is the largest single-season positive contribution."""
    from src.eval_v4_calibration import _drop_best_season_delta

    per_season = {2010: -1, 2011: 5, 2012: 2, 2013: -2}
    total = sum(per_season.values())  # 4
    drop = _drop_best_season_delta(per_season)  # 4 - 5 = -1
    assert drop == -1


def test_drop_best_season_delta_handles_all_negative():
    """If every season is a loss, drop_best subtracts 0 (no positive
    seasons to drop) -- equals the original total."""
    from src.eval_v4_calibration import _drop_best_season_delta

    per_season = {2010: -3, 2011: -1, 2012: -2}
    drop = _drop_best_season_delta(per_season)
    assert drop == sum(per_season.values())  # -6


def test_classify_verdict_pass_when_delta_above_25_and_robust():
    """delta >= +25, drop_best_delta >= 0, wins >= 6 -> PASS."""
    from src.eval_v4_calibration import _classify_verdict

    assert _classify_verdict(delta_total=30, drop_best_delta=15, wins=8) == "PASS"


def test_classify_verdict_marginal_when_above_10_below_25():
    """delta in [+10, +25) -> MARGINAL regardless of robustness."""
    from src.eval_v4_calibration import _classify_verdict

    assert _classify_verdict(delta_total=15, drop_best_delta=5, wins=5) == "MARGINAL"


def test_classify_verdict_marginal_when_pass_magnitude_but_concentrated():
    """delta >= +25 but drop_best_delta < 0 (concentrated in one season)
    -> MARGINAL (per spec: '>50% single-season concentration demotes PASS')."""
    from src.eval_v4_calibration import _classify_verdict

    # Total +30, but if you drop the best season the result is negative,
    # then that one season is doing more than 100% of the lift.
    assert _classify_verdict(delta_total=30, drop_best_delta=-5, wins=4) == "MARGINAL"


def test_classify_verdict_fail_when_below_10():
    """delta < +10 -> FAIL."""
    from src.eval_v4_calibration import _classify_verdict

    assert _classify_verdict(delta_total=8, drop_best_delta=8, wins=11) == "FAIL"
    assert _classify_verdict(delta_total=-5, drop_best_delta=-5, wins=6) == "FAIL"


def test_anchor_check_byte_equal():
    """Identical CSVs -> matches=True, max_abs_diff=0."""
    from src.eval_v4_calibration import _anchor_check

    df = pd.DataFrame({
        "season": [2024, 2024],
        "team_a": [1, 2], "team_b": [2, 3],
        "p_a_wins": [0.55, 0.62],
    })
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".csv") as f:
        df.to_csv(f.name, index=False)
        baseline = f.name
    res = _anchor_check(df, baseline)
    assert res["matches"] is True
    assert res["max_abs_diff"] == 0.0
    Path(baseline).unlink()


def test_anchor_check_flags_difference():
    """1e-3 difference -> matches=False."""
    from src.eval_v4_calibration import _anchor_check

    df = pd.DataFrame({
        "season": [2024], "team_a": [1], "team_b": [2], "p_a_wins": [0.55]})
    df_changed = df.copy(); df_changed["p_a_wins"] = 0.551
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".csv") as f:
        df.to_csv(f.name, index=False)
        baseline = f.name
    res = _anchor_check(df_changed, baseline)
    assert res["matches"] is False
    assert res["max_abs_diff"] == pytest.approx(0.001, abs=1e-9)
    Path(baseline).unlink()


def test_summarize_cell_computes_wlt_and_biggest_swing():
    """Per-season delta dict -> {total, wins, losses, ties,
    biggest_swing_value, biggest_swing_season, drop_best_season_delta}."""
    from src.eval_v4_calibration import _summarize_cell

    per_season = {2010: 5, 2011: -3, 2012: 0, 2013: 8, 2014: -1}
    out = _summarize_cell(per_season, baseline_total=2069)
    assert out["delta_total"] == sum(per_season.values())  # 9
    assert out["wins"] == 2  # 2010, 2013
    assert out["losses"] == 2  # 2011, 2014
    assert out["ties"] == 1  # 2012
    assert out["biggest_swing_value"] == 8
    assert out["biggest_swing_season"] == 2013
    # drop_best = 9 - 8 = 1
    assert out["drop_best_season_delta"] == 1
    # total = baseline + delta
    assert out["total"] == 2069 + 9


def _real_v8_present() -> bool:
    return Path("output/pairwise_v8.csv").exists() and Path(
        "data/raw/march-machine-learning-2026/MNCAATourneySeeds.csv"
    ).exists()


@pytest.mark.skipif(not _real_v8_present(), reason="canonical pairwise_v8.csv missing")
def test_global_T_anchor_cell_reproduces_2069():
    """T=1.0 must score 2069 to FP precision. Phase-1 anchor."""
    from src.eval_v4_calibration import run_global_T_sweep

    out = run_global_T_sweep(
        v8_csv="output/pairwise_v8.csv",
        T_grid=[1.0],
        baseline_total=2069.0,
    )
    assert "anchor" in out
    assert out["anchor"]["matches"] is True
    assert out["anchor"]["total"] == pytest.approx(2069.0, abs=1e-9)
    assert len(out["cells"]) == 1
    cell = out["cells"][0]
    assert cell["T"] == 1.0
    assert cell["total"] == pytest.approx(2069.0, abs=1e-9)
    assert cell["delta_total"] == pytest.approx(0.0, abs=1e-9)


@pytest.mark.skipif(not _real_v8_present(), reason="canonical pairwise_v8.csv missing")
def test_per_round_greedy_anchor_all_one_reproduces_2069():
    """All-1 per-round vector reproduces 2069 to FP precision."""
    from src.eval_v4_calibration import run_per_round_greedy

    out = run_per_round_greedy(
        v8_csv="output/pairwise_v8.csv",
        T_grid=[1.0],  # singleton grid -- only T=1 available
        round_order=["R64", "R32", "S16", "E8", "F4_NCG"],
        baseline_total=2069.0,
    )
    assert "anchor" in out
    assert out["anchor"]["total"] == pytest.approx(2069.0, abs=1e-9)
    assert out["winning_T"] == {b: 1.0 for b in ("R64", "R32", "S16", "E8", "F4_NCG")}
    assert out["winning_cell"]["total"] == pytest.approx(2069.0, abs=1e-9)


def test_per_round_greedy_monotonic_improvement_in_chain():
    """Each greedy step's chosen-cell total >= previous step's. Tested
    against a forced-pass synthetic scoring stub via monkeypatch.

    The greedy invariant is: at step k, holding rounds < k at their
    best-found T and rounds > k at 1.0, the picked T_round_k must
    yield total >= total at the previous step's pick."""
    import src.eval_v4_calibration as mod
    from src.eval_v4_calibration import run_per_round_greedy

    # Seed a synthetic v8.csv with one row per bucket.
    df = pd.DataFrame({
        "season": [2024] * 5,
        "team_a": [1, 2, 3, 4, 5],
        "team_b": [10, 11, 12, 13, 14],
        "p_a_wins": [0.6] * 5,
        "round_bucket": ["R64", "R32", "S16", "E8", "F4_NCG"],
    })
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".csv") as f:
        df.to_csv(f.name, index=False)
        v8_path = f.name

    # Monkeypatch _score_pairwise_df to return a synthetic score that
    # rewards lower T monotonically (via sum of p_a_wins).
    def fake_score(scaled_df, scratch_dir):
        s = scaled_df["p_a_wins"].sum()
        return {"total_pts": float(2069 + s), "per_season_pts": {2024: float(2069 + s)}}

    # Patch the baseline call too -- the synthetic frame isn't a real
    # bracket so the real score_pairwise_path would crash on it.
    def fake_baseline_score(path):
        df_local = pd.read_csv(path)
        s = df_local["p_a_wins"].sum()
        return {"total_pts": float(2069 + s), "per_season_pts": {2024: float(2069 + s)}}

    # Short-circuit assign_round_buckets -- the synthetic frame already
    # has round_bucket, but the driver re-derives via slots_df/seeds_df.
    real_assign = mod.assign_round_buckets
    real_score_df = mod._score_pairwise_df
    real_score_path = mod.score_pairwise_path

    def fake_assign(d, slots_df, seeds_df):
        return d["round_bucket"]
    mod.assign_round_buckets = fake_assign
    mod._score_pairwise_df = fake_score
    mod.score_pairwise_path = fake_baseline_score

    try:
        out = run_per_round_greedy(
            v8_csv=v8_path,
            T_grid=[0.5, 1.0, 2.0],
            round_order=["R64", "R32", "S16", "E8", "F4_NCG"],
            # baseline_total must equal what fake_baseline_score returns
            # for the v8_path (5 rows of p=0.6 each = 3.0 sum -> 2072.0).
            baseline_total=2072.0,
        )
    finally:
        mod.assign_round_buckets = real_assign
        mod._score_pairwise_df = real_score_df
        mod.score_pairwise_path = real_score_path
        Path(v8_path).unlink()

    chain = out["greedy_chain"]
    assert len(chain) == 5  # one entry per round
    # Monotonic invariant.
    for i in range(1, len(chain)):
        assert chain[i]["total_after_step"] >= chain[i - 1]["total_after_step"]
