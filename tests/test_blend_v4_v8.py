"""Tests for src.blend_v4_v8.BlendEvaluator and src.score_v13_blend.make_blend."""
import numpy as np
import pandas as pd
import pytest

from src.blend_v4_v8 import BlendEvaluator
from src.score_v13_blend import make_blend, bucket_for_p


@pytest.fixture(scope="module")
def v8_rerun():
    return pd.read_csv("output/pairwise_v8_rerun.csv")


@pytest.fixture(scope="module")
def v4():
    return pd.read_csv("output/pairwise_v4.csv").drop_duplicates(
        ["season", "team_a", "team_b"], keep="last"
    )


@pytest.fixture(scope="module")
def evaluator():
    return BlendEvaluator()


def test_evaluator_total_matches_score_pairwise_path(v8_rerun, evaluator):
    """BlendEvaluator.score_probs_df should equal score_chalk_brackets.score_pairwise_path
    on the canonical v8 frame, to machine precision."""
    from src.score_chalk_brackets import score_pairwise_path
    pts = evaluator.score_probs_df(v8_rerun)
    ref = score_pairwise_path("output/pairwise_v8_rerun.csv")["per_season_pts"]
    for s in pts:
        assert abs(pts[s] - ref[s]) < 1e-9, f"season {s} differs: {pts[s]} vs {ref[s]}"


def test_blend_alpha_one_is_v8(v8_rerun, v4, evaluator):
    """alpha=1.0 should be byte-equal to v8."""
    blended = evaluator.score_blend(v8_rerun, v4, alpha=1.0)
    direct = evaluator.score_probs_df(v8_rerun)
    for s in blended:
        assert abs(blended[s] - direct[s]) < 1e-9


def test_blend_alpha_zero_is_v4(v8_rerun, v4, evaluator):
    """alpha=0.0 should be byte-equal to v4."""
    blended = evaluator.score_blend(v8_rerun, v4, alpha=0.0)
    direct = evaluator.score_probs_df(v4)
    for s in blended:
        assert abs(blended[s] - direct[s]) < 1e-9


def test_bucket_for_p_symmetric():
    """The bucket should be symmetric around 0.5."""
    for p in [0.50, 0.51, 0.53, 0.55, 0.6, 0.7, 0.9]:
        assert bucket_for_p(p) == bucket_for_p(1 - p), f"asymmetric at p={p}"


def test_bucket_boundary():
    assert bucket_for_p(0.49, upper_edge=0.55) == 0  # symmetric: 0.49 -> conf 0.51 < 0.55
    assert bucket_for_p(0.50, upper_edge=0.55) == 0
    assert bucket_for_p(0.54, upper_edge=0.55) == 0
    assert bucket_for_p(0.55, upper_edge=0.55) == 1
    assert bucket_for_p(0.80, upper_edge=0.55) == 1


def test_make_blend_toss_up_only():
    """make_blend should only modify p in the toss-up bucket; others stay at v4."""
    v8 = pd.DataFrame({
        "season": [2024, 2024, 2024],
        "team_a": [1, 2, 3],
        "team_b": [101, 102, 103],
        "p_a_wins": [0.50, 0.80, 0.20],  # one toss-up, one strong A, one strong B
    })
    v4 = pd.DataFrame({
        "season": [2024, 2024, 2024],
        "team_a": [1, 2, 3],
        "team_b": [101, 102, 103],
        "p_a_wins": [0.52, 0.85, 0.10],  # v4 confidence: 0.52 (toss-up), 0.85, 0.90
    })
    out = make_blend(v8, v4, toss_up_alpha=0.6, toss_up_upper_edge=0.55)
    out = out.sort_values("team_a").reset_index(drop=True)
    # Row 1: toss-up -> 0.6 * 0.50 + 0.4 * 0.52 = 0.508
    assert abs(out.iloc[0]["p_a_wins"] - 0.508) < 1e-9
    # Row 2: non-toss-up -> v4 prob 0.85
    assert abs(out.iloc[1]["p_a_wins"] - 0.85) < 1e-9
    # Row 3: non-toss-up -> v4 prob 0.10
    assert abs(out.iloc[2]["p_a_wins"] - 0.10) < 1e-9


# Locked-in v13 production score on the committed pairwise_v8_ens30.csv +
# pairwise_v4.csv inputs. Update only if you regenerate either input.
EXPECTED_V13_TOTAL = 2106


def test_v13_total_reproduces(tmp_path):
    """Running v13 with the canonical v8-ens30 input should produce EXPECTED_V13_TOTAL."""
    from src.score_v13_blend import main
    total = main([
        "--v8", "output/pairwise_v8_ens30.csv",
        "--v4", "output/pairwise_v4.csv",
        "--alpha", "0.6",
        "--upper-edge", "0.55",
        "--out", str(tmp_path / "v13.csv"),
    ])
    assert int(total) == EXPECTED_V13_TOTAL, f"v13 should score {EXPECTED_V13_TOTAL}, got {total}"
