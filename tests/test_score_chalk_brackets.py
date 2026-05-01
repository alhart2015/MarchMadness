"""Tests that score_pairwise_path runs cleanly on the canonical v4 file
and returns the structure the ablation driver expects."""
from pathlib import Path
import pytest
from src.score_chalk_brackets import score_pairwise_path


@pytest.mark.skipif(
    not Path("output/pairwise_v4.csv").exists(),
    reason="pairwise_v4.csv missing -- run v4 LOSO first",
)
def test_score_v4_returns_known_shape():
    result = score_pairwise_path("output/pairwise_v4.csv")
    assert "total_pts" in result
    assert "per_season_pts" in result
    assert isinstance(result["total_pts"], (int, float))
    assert isinstance(result["per_season_pts"], dict)
    # 22 seasons (2003-2024 typical), give or take 2.
    assert 18 <= len(result["per_season_pts"]) <= 25
    # v4's known mean is ~121 -- total across 22 seasons should be in range.
    assert 2000 <= result["total_pts"] <= 3500


def test_missing_pairwise_path_raises():
    with pytest.raises(FileNotFoundError):
        score_pairwise_path("output/this_does_not_exist.csv")
