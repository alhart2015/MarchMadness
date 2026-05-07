"""Unit tests for src/analyze_v4_per_season_variance.py."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analyze_v4_per_season_variance import (
    _flag_outliers,
    _per_season_metrics,
    _pick_verdict,
)


def _three_season_fixture() -> pd.DataFrame:
    """Synthetic per-game v4 vs Vegas frame, 3 seasons x 4 games each.
    Season 2010 v4 wins big, 2011 ties, 2012 v4 loses (constructed).
    """
    rows = [
        # 2010: v4 perfectly confident in winners (LL=0)
        {"season": 2010, "p_v4": 0.99, "p_vegas": 0.60, "winner_is_a": 1},
        {"season": 2010, "p_v4": 0.99, "p_vegas": 0.60, "winner_is_a": 1},
        {"season": 2010, "p_v4": 0.01, "p_vegas": 0.40, "winner_is_a": 0},
        {"season": 2010, "p_v4": 0.01, "p_vegas": 0.40, "winner_is_a": 0},
        # 2011: tied
        {"season": 2011, "p_v4": 0.70, "p_vegas": 0.70, "winner_is_a": 1},
        {"season": 2011, "p_v4": 0.50, "p_vegas": 0.50, "winner_is_a": 1},
        {"season": 2011, "p_v4": 0.30, "p_vegas": 0.30, "winner_is_a": 0},
        {"season": 2011, "p_v4": 0.50, "p_vegas": 0.50, "winner_is_a": 0},
        # 2012: v4 wrongly confident (LL much worse than Vegas)
        {"season": 2012, "p_v4": 0.99, "p_vegas": 0.55, "winner_is_a": 0},
        {"season": 2012, "p_v4": 0.99, "p_vegas": 0.55, "winner_is_a": 0},
        {"season": 2012, "p_v4": 0.01, "p_vegas": 0.45, "winner_is_a": 1},
        {"season": 2012, "p_v4": 0.01, "p_vegas": 0.45, "winner_is_a": 1},
    ]
    return pd.DataFrame(rows)


def test_per_season_metrics_aggregates_correctly():
    """Per-season LL/acc/ECE on the 3-season fixture."""
    df = _three_season_fixture()
    out = _per_season_metrics(df, ref_label="vegas")
    assert list(out["season"]) == [2010, 2011, 2012]
    assert (out["n_games"] == 4).all()
    # 2010: v4 LL ~ -log(0.99) ~ 0.01
    assert out.loc[out["season"] == 2010, "ll_v4"].iloc[0] == pytest.approx(
        -np.log(0.99), abs=1e-3
    )
    # 2012: v4 LL ~ -log(0.01) ~ 4.6 (catastrophically wrong)
    assert out.loc[out["season"] == 2012, "ll_v4"].iloc[0] > 4.0
    # ll_v4_minus_vegas: 2010 negative (v4 better), 2012 positive (v4 worse)
    assert out.loc[out["season"] == 2010, "ll_v4_minus_vegas"].iloc[0] < 0
    assert out.loc[out["season"] == 2012, "ll_v4_minus_vegas"].iloc[0] > 0


def test_per_season_metrics_weighted_aggregate_matches_overall():
    """Invariant: weighted average of per-season LL (by n_games) equals
    the overall LL on the same frame."""
    df = _three_season_fixture()
    per_season = _per_season_metrics(df, ref_label="vegas")
    weighted_ll = float(np.average(per_season["ll_v4"],
                                    weights=per_season["n_games"]))
    eps = 1e-15
    winner = df["winner_is_a"].to_numpy()
    p_v4 = df["p_v4"].to_numpy()
    p_v4_w = np.where(winner == 1, p_v4, 1 - p_v4)
    overall_ll = float(-np.mean(np.log(np.clip(p_v4_w, eps, 1 - eps))))
    assert weighted_ll == pytest.approx(overall_ll, abs=1e-6)


def test_flag_outliers_flags_high_sigma_value():
    """One value 2.5 sigma above the mean is flagged at threshold 1.5."""
    df = pd.DataFrame({
        "season": list(range(2000, 2010)),
        "n_games": [60] * 10,
        # 9 values with mean 0.5, std small; one outlier at 1.5
        "ll_v4_minus_vegas": [0.05, -0.05, 0.05, -0.05, 0.05, -0.05, 0.05, -0.05, 0.05, 1.5],
    })
    out = _flag_outliers(df, columns=["ll_v4_minus_vegas"], sigma=1.5)
    assert "ll_v4_minus_vegas" in out
    assert len(out["ll_v4_minus_vegas"]) == 1
    assert out["ll_v4_minus_vegas"][0]["season"] == 2009
    assert out["ll_v4_minus_vegas"][0]["sigma_delta"] >= 1.5


def test_flag_outliers_skips_missing_column_and_short_series():
    """Missing columns are skipped; series shorter than 2 returns empty."""
    df = pd.DataFrame({
        "season": [2000, 2001],
        "n_games": [60, 60],
        "ll_v4": [0.5, 0.6],
    })
    # Column not present -> skipped
    out = _flag_outliers(df, columns=["ll_v4", "nonexistent"], sigma=1.5)
    assert "nonexistent" not in out
    # Series too short / std too small -> no outliers flagged
    assert out["ll_v4"] == []


def test_pick_verdict_flat_when_no_outliers():
    """No outliers on any tracked metric -> verdict='flat'."""
    df = pd.DataFrame({
        "season": [2000, 2001, 2002],
        "n_games": [60, 60, 60],
        "ll_v4": [0.55, 0.56, 0.54],
        "ll_v4_minus_vegas": [0.01, -0.01, 0.0],
        "ll_v4_minus_fte": [None, None, None],
        "ece_v4": [0.04, 0.05, 0.04],
    })
    outliers = {
        "ll_v4_minus_vegas": [],
        "ll_v4_minus_fte": [],
        "ll_v4": [],
        "ece_v4": [],
    }
    verdict = _pick_verdict(df, outliers, sigma=1.5)
    assert verdict["label"] == "flat"


def test_pick_verdict_outlier_when_one_or_two_seasons_flagged():
    """One outlier season on the v4-vs-Vegas delta -> verdict='outlier'."""
    df = pd.DataFrame({
        "season": list(range(2000, 2010)),
        "n_games": [60] * 10,
        "ll_v4": [0.55] * 10,
        "ll_v4_minus_vegas": [0.0] * 9 + [0.2],
        "ll_v4_minus_fte": [None] * 10,
        "ece_v4": [0.04] * 10,
    })
    outliers = {
        "ll_v4_minus_vegas": [{"season": 2009, "value": 0.2,
                                "sigma_delta": 3.0, "n_games": 60}],
        "ll_v4_minus_fte": [],
        "ll_v4": [],
        "ece_v4": [],
    }
    verdict = _pick_verdict(df, outliers, sigma=1.5)
    assert verdict["label"] == "outlier"
    assert 2009 in verdict["outlier_seasons"]
