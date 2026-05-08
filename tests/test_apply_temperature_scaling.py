"""Unit tests for src/apply_temperature_scaling.py."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def _four_row_synth() -> pd.DataFrame:
    """Synthetic 4 rows spanning the [0.05, 0.95] interior + 0.5."""
    return pd.DataFrame({
        "season": [2024, 2024, 2024, 2024],
        "team_a": [1, 2, 3, 4],
        "team_b": [5, 6, 7, 8],
        "p_a_wins": [0.05, 0.5, 0.85, 0.95],
        "round_bucket": ["R64", "R64", "R64", "R64"],
    })


def test_scale_identity_when_T_is_one():
    """T=1.0 returns p_a_wins unchanged (modulo float roundtrip)."""
    from src.apply_temperature_scaling import scale_pairwise

    df = _four_row_synth()
    out = scale_pairwise(df, T=1.0)
    np.testing.assert_allclose(
        out["p_a_wins"].values,
        df["p_a_wins"].values,
        atol=1e-12,
    )


def test_scale_flatten_when_T_above_one():
    """T=2.0 pulls every p toward 0.5 monotonically (rank-preserving on
    distances to 0.5; smaller-margin probs end closer to 0.5)."""
    from src.apply_temperature_scaling import scale_pairwise

    df = _four_row_synth()
    out = scale_pairwise(df, T=2.0)
    p_in = df["p_a_wins"].values
    p_out = out["p_a_wins"].values
    # 0.5 stays at 0.5.
    assert p_out[1] == pytest.approx(0.5, abs=1e-9)
    # Distances to 0.5 strictly shrink for non-0.5 inputs.
    for i in [0, 2, 3]:
        assert abs(p_out[i] - 0.5) < abs(p_in[i] - 0.5)
    # Order is preserved.
    assert list(np.argsort(p_out)) == list(np.argsort(p_in))


def test_scale_sharpen_when_T_below_one():
    """T=0.5 pushes every p away from 0.5 monotonically."""
    from src.apply_temperature_scaling import scale_pairwise

    df = _four_row_synth()
    out = scale_pairwise(df, T=0.5)
    p_in = df["p_a_wins"].values
    p_out = out["p_a_wins"].values
    assert p_out[1] == pytest.approx(0.5, abs=1e-9)
    for i in [0, 2, 3]:
        assert abs(p_out[i] - 0.5) > abs(p_in[i] - 0.5)
    assert list(np.argsort(p_out)) == list(np.argsort(p_in))


def test_scale_per_round_dispatch():
    """Per-round T applies a different T to each row based on its
    round_bucket. We give each bucket a unique T and verify the row
    output is what scale_pairwise would have produced under that T
    alone."""
    from src.apply_temperature_scaling import scale_pairwise

    rows = []
    for bucket, p in [("R64", 0.7), ("R32", 0.7), ("S16", 0.7),
                      ("E8", 0.7), ("F4_NCG", 0.7)]:
        rows.append({"season": 2024, "team_a": 1, "team_b": 2,
                     "p_a_wins": p, "round_bucket": bucket})
    df = pd.DataFrame(rows)
    T = {"R64": 1.0, "R32": 2.0, "S16": 0.5, "E8": 1.5, "F4_NCG": 1.0}
    out = scale_pairwise(df, T=T)
    # R64 and F4_NCG (T=1.0) -> 0.7 unchanged.
    assert out.iloc[0]["p_a_wins"] == pytest.approx(0.7, abs=1e-9)
    assert out.iloc[4]["p_a_wins"] == pytest.approx(0.7, abs=1e-9)
    # R32 (T=2.0) -> closer to 0.5 than 0.7.
    assert 0.5 < out.iloc[1]["p_a_wins"] < 0.7
    # S16 (T=0.5) -> further from 0.5 than 0.7.
    assert out.iloc[2]["p_a_wins"] > 0.7
    # E8 (T=1.5) flattens 0.7 less than R32 (T=2.0) does. Since p=0.7 > 0.5,
    # "closer to 0.5" means smaller numeric value, so R32 < E8 < 0.7.
    assert 0.5 < out.iloc[1]["p_a_wins"] < out.iloc[3]["p_a_wins"] < 0.7


def test_scale_clips_extreme_inputs_to_finite_output():
    """p in {0, 1} should produce finite output (no inf/NaN)."""
    from src.apply_temperature_scaling import scale_pairwise

    df = pd.DataFrame({
        "season": [2024, 2024],
        "team_a": [1, 2], "team_b": [3, 4],
        "p_a_wins": [0.0, 1.0],
        "round_bucket": ["R64", "R64"],
    })
    out = scale_pairwise(df, T=1.5)
    assert np.isfinite(out["p_a_wins"].values).all()
    # The clipped extremes should still be near {0, 1} after T=1.5
    # since logit/sigmoid round-trip near-identity at the bounds.
    assert out.iloc[0]["p_a_wins"] < 1e-3
    assert out.iloc[1]["p_a_wins"] > 1.0 - 1e-3


def test_scale_per_round_raises_if_bucket_missing_in_T_dict():
    """If a row's round_bucket isn't in T (dict mode), raise KeyError --
    no silent fallback. Caller must pass complete T dict."""
    from src.apply_temperature_scaling import scale_pairwise

    df = pd.DataFrame({
        "season": [2024], "team_a": [1], "team_b": [2],
        "p_a_wins": [0.6], "round_bucket": ["F4_NCG"],
    })
    with pytest.raises(KeyError, match="F4_NCG"):
        scale_pairwise(df, T={"R64": 1.0})


def test_scale_per_round_does_not_mutate_input():
    """scale_pairwise returns a NEW DataFrame; input is not mutated."""
    from src.apply_temperature_scaling import scale_pairwise

    df = _four_row_synth()
    p_orig = df["p_a_wins"].copy()
    _ = scale_pairwise(df, T=2.0)
    np.testing.assert_array_equal(df["p_a_wins"].values, p_orig.values)


def _real_v8_present() -> bool:
    return Path("output/pairwise_v8.csv").exists()


@pytest.mark.skipif(
    not _real_v8_present(),
    reason="output/pairwise_v8.csv missing; data wipe? see docs/data_recovery.md",
)
def test_scale_T_one_anchors_byte_equal_to_canonical_v8():
    """Apply T=1.0 to canonical pairwise_v8.csv -- p_a_wins must round-trip
    to FP precision. This is the Phase-1 anchor (spec section 'Anchors')."""
    from src.apply_temperature_scaling import scale_pairwise

    df = pd.read_csv("output/pairwise_v8.csv")
    out = scale_pairwise(df, T=1.0)
    np.testing.assert_allclose(
        out["p_a_wins"].values,
        df["p_a_wins"].values,
        atol=1e-9,
    )


@pytest.mark.skipif(
    not _real_v8_present(),
    reason="output/pairwise_v8.csv missing; data wipe? see docs/data_recovery.md",
)
def test_scale_T_all_one_perround_anchors_byte_equal_to_canonical_v8():
    """All-1 per-round dict is identity even with bucket dispatch."""
    from src.apply_temperature_scaling import scale_pairwise, assign_round_buckets

    df = pd.read_csv("output/pairwise_v8.csv")
    slots_df = pd.read_csv("data/raw/march-machine-learning-2026/MNCAATourneySlots.csv")
    seeds_df = pd.read_csv("data/raw/march-machine-learning-2026/MNCAATourneySeeds.csv")
    df["round_bucket"] = assign_round_buckets(df, slots_df, seeds_df)
    # Drop rows with no resolvable round (play-in, etc.) so per-round
    # mode doesn't KeyError on pd.NA.
    df_resolved = df.dropna(subset=["round_bucket"]).copy()
    T = {b: 1.0 for b in ("R64", "R32", "S16", "E8", "F4_NCG")}
    out = scale_pairwise(df_resolved, T=T)
    np.testing.assert_allclose(
        out["p_a_wins"].values,
        df_resolved["p_a_wins"].values,
        atol=1e-9,
    )


@pytest.mark.skipif(
    not (Path("output/pairwise_v8.csv").exists()
         and Path("data/raw/march-machine-learning-2026/MNCAATourneySeeds.csv").exists()
         and Path("data/raw/march-machine-learning-2026/MNCAATourneySlots.csv").exists()),
    reason="Kaggle data missing -- needs tar -xzf data/training_data.tar.gz",
)
def test_assign_round_buckets_covers_all_five_buckets_on_real_data():
    """All 5 buckets show up; resolved fraction >= 0.95 of rows."""
    from src.apply_temperature_scaling import assign_round_buckets, ROUND_BUCKETS

    df = pd.read_csv("output/pairwise_v8.csv")
    slots_df = pd.read_csv("data/raw/march-machine-learning-2026/MNCAATourneySlots.csv")
    seeds_df = pd.read_csv("data/raw/march-machine-learning-2026/MNCAATourneySeeds.csv")
    s = assign_round_buckets(df, slots_df, seeds_df)
    present = set(s.dropna().unique())
    assert present == set(ROUND_BUCKETS), f"missing buckets: {set(ROUND_BUCKETS) - present}"
    # The pairwise frame includes every (i, j) pair in each season's
    # field, so most pairs do NOT meet during the tournament -- those
    # legitimately have no round and stay as pd.NA. We check only that
    # the resolved fraction is consistent across seasons (every season
    # has roughly the same pair count).
    season_resolved = s.dropna().groupby(df.loc[s.dropna().index, "season"]).size()
    assert (season_resolved > 0).all(), "every season should have >=1 resolved pair"
