"""Tests for v12 plumbing on top of train_stage2_v10.

Spec: docs/superpowers/specs/2026-05-14-v12-stage2-v4-feature-diffs-design.md
Plan: docs/superpowers/plans/2026-05-14-v12-stage2-v4-feature-diffs.md

Guards:
  - v8 anchor invariance: --features v8 --seeds 42 still reproduces
    canonical pairwise_v8.csv byte-equal even after v12 plumbing is added.
  - Diff sign flip: label=0 row's diff_<feat> is the exact negation of
    the label=1 row's diff_<feat> for the same game.
  - Join key: v4 feature lookup is on (Season, TeamID), never on names.
  - No leak: v4 feature columns used in diffs cannot include any
    in-tournament-game-derived feature.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def _real_data_present() -> bool:
    return (Path("output/pairwise_v8.csv").exists()
            and Path("output/pairwise_v4.csv").exists()
            and Path("data/raw/march-machine-learning-2026/MNCAATourneyCompactResults.csv").exists()
            and Path("data/raw/march-machine-learning-2026/MNCAATourneySeeds.csv").exists()
            and Path("data/raw/march-machine-learning-2026/MNCAATourneySlots.csv").exists())


@pytest.mark.skipif(not _real_data_present(), reason="Kaggle data + v4/v8 frames missing")
def test_v8_anchor_unchanged_after_v12_plumbing(tmp_path):
    """v8 feature set + single seed reproduces canonical pairwise_v8.csv
    byte-equal. This test must continue to pass after v12 plumbing is added
    -- the v8 code path must not see any v4-diff loading."""
    from src.train_stage2_v10 import main as v10_main
    out = tmp_path / "v8_anchor.csv"
    v10_main([
        "--features", "v8",
        "--seeds", "42",
        "--pairwise-out", str(out),
    ])
    canonical = pd.read_csv("output/pairwise_v8.csv")
    rerun = pd.read_csv(out)
    pd.testing.assert_frame_equal(
        canonical.reset_index(drop=True),
        rerun.reset_index(drop=True),
    )


# v12-specific tests follow once the plumbing is in place. They depend on
# output/v4_feature_matrix.parquet existing (Phase 0 artifact).


def _v12_artifacts_present() -> bool:
    return (Path("output/v4_feature_importance.csv").exists()
            and Path("output/v4_feature_matrix.parquet").exists())


@pytest.mark.skipif(not _v12_artifacts_present(), reason="v4 feature artifacts not produced yet")
def test_v4_feature_ranking_shape():
    """v4_feature_importance.csv has the expected columns and ~67 rows."""
    ranking = pd.read_csv("output/v4_feature_importance.csv")
    assert set(ranking.columns) >= {"feature_name", "gain", "gain_rank"}
    assert 50 <= len(ranking) <= 80, f"unexpected feature count: {len(ranking)}"
    # Top feature should have non-trivial gain
    assert ranking.iloc[0]["gain"] > 0.005
    # Ranks are 1..N contiguous
    assert (ranking["gain_rank"].values == np.arange(1, len(ranking) + 1)).all()


@pytest.mark.skipif(not _v12_artifacts_present(), reason="v4 feature artifacts not produced yet")
def test_v4_feature_matrix_snapshot_shape():
    """v4_feature_matrix.parquet has (Season, TeamID) keys + 67 feature cols,
    no NaN values (post fill)."""
    fm = pd.read_parquet("output/v4_feature_matrix.parquet")
    assert "Season" in fm.columns and "TeamID" in fm.columns
    feat_cols = [c for c in fm.columns if c not in ("Season", "TeamID")]
    assert 50 <= len(feat_cols) <= 80, f"unexpected feature count: {len(feat_cols)}"
    # No NaN after fill
    assert fm[feat_cols].isna().sum().sum() == 0, "feature matrix has NaN after fill"
    # 22 seasons (2003..2025 minus 2020)
    assert fm["Season"].nunique() >= 20, f"only {fm['Season'].nunique()} seasons"


@pytest.mark.skipif(not _v12_artifacts_present(), reason="v4 feature artifacts not produced yet")
def test_v4_feature_ranking_excludes_in_tournament_features():
    """No leak: top-ranked features cannot include columns derived from
    in-tournament-game outcomes (the PR 19/20 leak fix audit pattern)."""
    ranking = pd.read_csv("output/v4_feature_importance.csv")
    feat_names = ranking["feature_name"].tolist()
    # Patterns that would indicate in-tournament leakage. These are the
    # specific column names PR 19/20 audited against.
    forbidden_substrings = ["tourney_round", "tourney_result", "ROUND_label",
                            "tournament_outcome"]
    for forbidden in forbidden_substrings:
        leaks = [f for f in feat_names if forbidden.lower() in f.lower()]
        assert not leaks, f"forbidden substring '{forbidden}' found in: {leaks}"


@pytest.mark.skipif(not _v12_artifacts_present(), reason="v4 feature artifacts not produced yet")
def test_diff_sign_flip_on_symmetric_pair():
    """For a single tournament game, the label=1 row (winner's perspective)
    and label=0 row (loser's perspective) must have diff_<feat> values that
    are exact negations of each other."""
    from src.train_stage2_v10 import load_per_game_data, FEATURE_SETS

    assert "v12_n5" in FEATURE_SETS
    assert "v12_n10" in FEATURE_SETS
    assert "v12_n15" in FEATURE_SETS

    fm = pd.read_parquet("output/v4_feature_matrix.parquet")
    per_game = load_per_game_data(
        "output/pairwise_v4.csv",
        "data/raw/march-machine-learning-2026/MNCAATourneyCompactResults.csv",
        "data/raw/march-machine-learning-2026/MNCAATourneySeeds.csv",
        "data/raw/march-machine-learning-2026/MNCAATourneySlots.csv",
        v4_feature_matrix_df=fm,
    )
    # Identify diff columns
    diff_cols = [c for c in per_game.columns if c.startswith("diff_")]
    assert len(diff_cols) >= 5, f"expected diff_* columns, got {diff_cols}"
    # Pick one game (the first label=1 row and the corresponding label=0 row).
    # Pairs are stored as consecutive rows in load_per_game_data.
    label_1_rows = per_game[per_game["label"] == 1]
    label_0_rows = per_game[per_game["label"] == 0]
    # Pair by (season, winner_id) -- label=1 row's team_a IS the winner.
    paired = label_1_rows.merge(
        label_0_rows,
        left_on=["season", "team_a", "team_b"],
        right_on=["season", "team_b", "team_a"],
        suffixes=("_w", "_l"),
    )
    assert len(paired) > 0, "no W/L symmetric pairs found"
    for col in diff_cols:
        w = paired[f"{col}_w"].values
        l = paired[f"{col}_l"].values
        np.testing.assert_allclose(w, -l, atol=1e-9,
                                    err_msg=f"sign flip violated for {col}")


@pytest.mark.skipif(not _v12_artifacts_present(), reason="v4 feature artifacts not produced yet")
def test_v12_pairwise_row_count_matches_v4(tmp_path):
    """build_pairwise for v12_n5 produces a frame with the same row count as
    pairwise_v4.csv, with p_a_wins in [0, 1]."""
    from src.train_stage2_v10 import main as v10_main
    out = tmp_path / "v12_n5.csv"
    v10_main([
        "--features", "v12_n5",
        "--seeds", "42",
        "--v4-feature-matrix", "output/v4_feature_matrix.parquet",
        "--pairwise-out", str(out),
    ])
    v12 = pd.read_csv(out)
    v4 = pd.read_csv("output/pairwise_v4.csv").drop_duplicates(
        ["season", "team_a", "team_b"], keep="last"
    )
    assert len(v12) == len(v4)
    assert (v12["p_a_wins"] >= 0).all() and (v12["p_a_wins"] <= 1).all()
