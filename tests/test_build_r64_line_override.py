"""Unit tests for src/build_r64_line_override.py."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def _three_season_v4_fixture() -> pd.DataFrame:
    """3 seasons * 4 pairwise rows each = 12 rows. Probabilities all 0.5
    so we can detect override clearly."""
    rows = []
    for season in [2010, 2011, 2012]:
        for (a, b) in [(1101, 1102), (1101, 1103), (1102, 1103), (1101, 1104)]:
            rows.append({
                "season": season, "team_a": a, "team_b": b, "p_a_wins": 0.5,
            })
    return pd.DataFrame(rows)


def test_apply_r64_override_hard_mode_replaces_p_for_r64_pairs(tmp_path):
    """Hard mode: for the 3 R64 pairs we synthesize, replace 0.5 with the
    Vegas-implied probability. Other 9 rows pass through unchanged.

    The real `vegas_lookup` shape is {(season, daynum, a, b): p_a_wins (float)},
    populated by `_build_vegas_lookup` which precomputes the Vegas-implied
    probability at lookup-build time. We use that shape directly here.
    """
    from src.build_r64_line_override import _apply_overrides

    v4_df = _three_season_v4_fixture()
    # Synthesize a precomputed Vegas lookup: a is the underdog by 5.5
    # at sigma=11 -> p_a_wins = norm.cdf(-5.5/11) = norm.cdf(-0.5) ~= 0.3085.
    p_vegas = 0.3085
    vegas_lookup = {
        (2010, 137, 1101, 1102): p_vegas,
        (2011, 137, 1101, 1102): p_vegas,
        (2012, 137, 1101, 1102): p_vegas,
    }
    r64_pairs = {  # (season, team_a, team_b) -> daynum
        (2010, 1101, 1102): 137,
        (2011, 1101, 1102): 137,
        (2012, 1101, 1102): 137,
    }
    out_df, stats = _apply_overrides(
        v4_df, vegas_lookup, r64_pairs, mode="hard", sigma=11.0,
    )

    assert len(out_df) == 12
    # The 3 R64 pairs should have p == p_vegas (hard replaces).
    overridden = out_df[(out_df["team_a"] == 1101) & (out_df["team_b"] == 1102)]
    assert len(overridden) == 3
    assert (overridden["p_a_wins"] != 0.5).all()
    for v in overridden["p_a_wins"]:
        assert v == pytest.approx(p_vegas)
    # All other 9 rows untouched.
    untouched = out_df[~((out_df["team_a"] == 1101) & (out_df["team_b"] == 1102))]
    assert (untouched["p_a_wins"] == 0.5).all()
    # Stats: 3 R64 pairs targeted; 3 overridden; 0 missing.
    assert stats["n_r64_target"] == 3
    assert stats["n_overridden"] == 3
    assert stats["n_missing_line"] == 0


def test_apply_r64_override_mean_mode_50_50_blend():
    """Mean mode: replace 0.5 with mean(0.5, p_vegas) -- which equals
    (p_v4 + p_vegas) / 2. With v4 = 0.5, the result is (0.5 + p_vegas) / 2.

    Synthesize a Vegas lookup where a is the underdog by 11 points at
    sigma=11; norm.cdf(-1) ~= 0.1587. With v4 = 0.5,
    mean = (0.5 + 0.1587) / 2 = 0.3293.
    """
    from src.build_r64_line_override import _apply_overrides

    v4_df = _three_season_v4_fixture()
    p_vegas = 0.15866  # ~= norm.cdf(-1)
    vegas_lookup = {
        (2010, 137, 1101, 1102): p_vegas,
    }
    r64_pairs = {(2010, 1101, 1102): 137}
    out_df, stats = _apply_overrides(
        v4_df, vegas_lookup, r64_pairs, mode="mean", sigma=11.0,
    )
    overridden = out_df[
        (out_df["season"] == 2010) & (out_df["team_a"] == 1101) & (out_df["team_b"] == 1102)
    ]
    # Expected mean = (0.5 + 0.15866) / 2 = 0.32933.
    assert overridden["p_a_wins"].iloc[0] == pytest.approx(0.3293, abs=1e-3)


def test_apply_r64_override_missing_line_passes_through_v4():
    """If a R64 pair has no Vegas line, that pair's v4 prob passes through
    unchanged. Stats records the miss."""
    from src.build_r64_line_override import _apply_overrides

    v4_df = _three_season_v4_fixture()
    vegas_lookup: dict = {}  # No Vegas lines at all.
    r64_pairs = {(2010, 1101, 1102): 137}  # 1 R64 pair targeted.
    out_df, stats = _apply_overrides(
        v4_df, vegas_lookup, r64_pairs, mode="hard", sigma=11.0,
    )
    assert (out_df["p_a_wins"] == 0.5).all()  # All passed through.
    assert stats["n_r64_target"] == 1
    assert stats["n_overridden"] == 0
    assert stats["n_missing_line"] == 1


def test_apply_r64_override_unknown_mode_raises():
    """Defensive: only 'hard' and 'mean' are supported."""
    from src.build_r64_line_override import _apply_overrides

    with pytest.raises(ValueError, match="mode"):
        _apply_overrides(
            _three_season_v4_fixture(), {}, {}, mode="learned", sigma=11.0,
        )


# --- real-data tests (skip on fresh clone) ---


def _real_data_present() -> bool:
    return (
        Path("output/pairwise_v4.csv").exists()
        and Path("data/raw/march-machine-learning-2026/MTeams.csv").exists()
        and Path("data/raw/march-machine-learning-2026/MNCAATourneyCompactResults.csv").exists()
        and Path("data/raw/vegas_lines").exists()
    )


def test_r64_pair_index_real_data_has_32_per_season():
    """For each season 2003-2025 with tournament data, the R64 round
    has exactly 32 games (the 64-team field has 32 R64 matchups)."""
    if not _real_data_present():
        pytest.skip("Kaggle / Vegas / pairwise_v4 data not present")
    from src.build_r64_line_override import _build_r64_pair_index

    results = pd.read_csv(
        "data/raw/march-machine-learning-2026/MNCAATourneyCompactResults.csv"
    )
    # Pick a recent season with full tournament; 2019 had 64 teams + 4 play-in.
    pairs_2019 = _build_r64_pair_index(2019, results)
    assert len(pairs_2019) == 32, f"expected 32 R64 games in 2019, got {len(pairs_2019)}"


def test_apply_r64_override_real_data_hard_mode_coverage(tmp_path):
    """Smoke: apply hard override over all 22 seasons of pairwise_v4.csv;
    expect coverage >= 0.85 of the 22 * 32 = 704 R64 pairs."""
    if not _real_data_present():
        pytest.skip("Kaggle / Vegas / pairwise_v4 data not present")
    from src.build_r64_line_override import apply_r64_override

    out_csv = tmp_path / "pairwise_v4_r64lineblend_hard_sigma11.csv"
    stats = apply_r64_override(
        v4_csv="output/pairwise_v4.csv",
        mode="hard", sigma=11.0,
        out_csv=str(out_csv),
    )
    # 22 seasons * 32 = 704 in theory; in practice 2021 (COVID bubble)
    # only has 20 R64 entries in the Kaggle data so we land near 692.
    # Tolerate up to 20 missing pairs.
    assert stats["n_r64_target"] >= 22 * 32 - 20
    coverage = stats["n_overridden"] / max(stats["n_r64_target"], 1)
    assert coverage >= 0.85, f"R64 line coverage {coverage:.2%} below 85%"
    assert out_csv.exists()
    out_df = pd.read_csv(out_csv)
    assert set(out_df.columns) == {"season", "team_a", "team_b", "p_a_wins"}


def test_apply_r64_override_mean_real_data_differs_from_hard(tmp_path):
    """Mean mode should produce a different frame than hard mode (since
    v4 prob is not exactly equal to vegas prob for most R64 pairs)."""
    if not _real_data_present():
        pytest.skip("Kaggle / Vegas / pairwise_v4 data not present")
    from src.build_r64_line_override import apply_r64_override

    hard_csv = tmp_path / "hard.csv"
    mean_csv = tmp_path / "mean.csv"
    apply_r64_override("output/pairwise_v4.csv", "hard", 11.0, str(hard_csv))
    apply_r64_override("output/pairwise_v4.csv", "mean", 11.0, str(mean_csv))
    a = pd.read_csv(hard_csv).sort_values(["season", "team_a", "team_b"]).reset_index(drop=True)
    b = pd.read_csv(mean_csv).sort_values(["season", "team_a", "team_b"]).reset_index(drop=True)
    assert (a["p_a_wins"] != b["p_a_wins"]).any(), "hard and mean produced identical frames"
