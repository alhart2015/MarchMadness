"""Unit tests for src/train_hbt_stage1.py.

Synthetic-data tests to validate the LOSO loop, per-fold standardization,
CSV writing, and sigma loop. The full real-data sweep is exercised in
the Phase 3b sweep run, not here.
"""
import numpy as np
import pandas as pd
import pytest

from src.train_hbt_stage1 import (
    _filter_numeric_feature_cols,
    _format_sigma,
    _season_field,
    _standardize_stats,
    run_hbt_loso,
)


def _synthetic_inputs(n_teams=8, n_seasons=3, games_per_pair=4, seed=0):
    """Build a (feature_matrix, regular_results, tourney_filtered,
    feature_cols) tuple over multiple synthetic seasons. Each season has
    a different set of team_ids so we exercise the per-season filtering
    too."""
    rng = np.random.default_rng(seed)
    feature_cols = ["feat_a", "feat_b", "feat_const"]

    fm_rows, reg_rows, tour_rows = [], [], []
    daynum = 10
    for season_offset, season in enumerate(range(2003, 2003 + n_seasons)):
        team_ids = list(range(1000 + season_offset * 100,
                              1000 + season_offset * 100 + n_teams))
        true_s = rng.normal(0, 1, size=n_teams)

        for k, t in enumerate(team_ids):
            fm_rows.append({
                "TeamID": t,
                "Season": season,
                "feat_a": true_s[k] + rng.normal(0, 0.3),
                "feat_b": rng.normal(0, 1),
                "feat_const": 5.0,  # zero-variance: should be dropped
            })

        for i in range(n_teams):
            for j in range(i + 1, n_teams):
                for _ in range(games_per_pair):
                    p_i_wins = 1.0 / (1.0 + np.exp(-(true_s[i] - true_s[j])))
                    if rng.random() < p_i_wins:
                        w, l = team_ids[i], team_ids[j]
                    else:
                        w, l = team_ids[j], team_ids[i]
                    reg_rows.append({
                        "Season": season, "DayNum": daynum,
                        "WTeamID": w, "LTeamID": l,
                        "WLoc": rng.choice(["H", "A", "N"]),
                    })
                    daynum += 1

        # Pretend the top half of teams made the "tournament"
        field = team_ids[: n_teams // 2]
        for i in range(len(field)):
            for j in range(i + 1, len(field)):
                tour_rows.append({
                    "Season": season,
                    "DayNum": daynum + 1,
                    "WTeamID": field[i], "LTeamID": field[j], "WLoc": "N",
                })

    return (
        pd.DataFrame(fm_rows),
        pd.DataFrame(reg_rows),
        pd.DataFrame(tour_rows),
        feature_cols,
    )


def test_format_sigma():
    assert _format_sigma(1.0) == "1.00"
    assert _format_sigma(0.05) == "0.05"
    assert _format_sigma(5.0) == "5.00"


def test_filter_numeric_drops_string_and_all_nan():
    fm = pd.DataFrame({
        "TeamID": [1, 2],
        "Season": [2024, 2024],
        "feat_a": [1.0, 2.0],
        "feat_str": ["x", "y"],
        "feat_allnan": [np.nan, np.nan],
    })
    keep = _filter_numeric_feature_cols(
        fm, ["feat_a", "feat_str", "feat_allnan", "feat_missing"]
    )
    assert keep == ["feat_a"]


def test_standardize_drops_zero_variance():
    train = pd.DataFrame({
        "feat_a": [1.0, 2.0, 3.0],
        "feat_const": [5.0, 5.0, 5.0],
    })
    means, stds, keep = _standardize_stats(train, ["feat_a", "feat_const"])
    assert keep == ["feat_a"]
    assert "feat_const" not in means.index


def test_season_field_extracts_unique_team_ids():
    tour = pd.DataFrame([
        {"Season": 2024, "WTeamID": 1, "LTeamID": 2},
        {"Season": 2024, "WTeamID": 3, "LTeamID": 1},
        {"Season": 2025, "WTeamID": 9, "LTeamID": 8},
    ])
    assert _season_field(tour, 2024) == [1, 2, 3]
    assert _season_field(tour, 2025) == [8, 9]


def test_run_hbt_loso_writes_per_sigma_csvs(tmp_path):
    fm, reg, tour, feature_cols = _synthetic_inputs(seed=42)

    sigmas = [0.5, 5.0]
    out = run_hbt_loso(
        feature_matrix=fm,
        regular_results=reg,
        tourney_filtered=tour,
        feature_cols=feature_cols,
        sigmas=sigmas,
        out_dir=tmp_path,
        verbose=False,
    )

    summary = out["per_cell"]
    assert (summary["success"]).all(), \
        f"some fits failed: {summary[~summary['success']]}"

    for sigma in sigmas:
        p = tmp_path / f"pairwise_hbt_sigma_{sigma:.2f}.csv"
        assert p.exists(), f"missing {p}"
        df = pd.read_csv(p)
        assert list(df.columns) == ["season", "team_a", "team_b", "p_a_wins"]
        assert (df["team_a"] < df["team_b"]).all()
        assert ((df["p_a_wins"] > 0) & (df["p_a_wins"] < 1)).all()
        # 3 seasons x C(4, 2) = 3 * 6 = 18 pairs total
        assert len(df) == 18

    # Sigma matters: predictions should differ between cells.
    df_loose = pd.read_csv(tmp_path / "pairwise_hbt_sigma_5.00.csv")
    df_tight = pd.read_csv(tmp_path / "pairwise_hbt_sigma_0.50.csv")
    assert not np.allclose(df_loose["p_a_wins"], df_tight["p_a_wins"]), \
        "loose vs tight prior produced identical probabilities"


def test_run_hbt_loso_subset_seasons(tmp_path):
    """Restricting --seasons to one season writes only that season's pairs."""
    fm, reg, tour, feature_cols = _synthetic_inputs(seed=43)

    out = run_hbt_loso(
        feature_matrix=fm,
        regular_results=reg,
        tourney_filtered=tour,
        feature_cols=feature_cols,
        sigmas=[1.0],
        seasons=[2004],
        out_dir=tmp_path,
        verbose=False,
    )

    df = pd.read_csv(tmp_path / "pairwise_hbt_sigma_1.00.csv")
    assert (df["season"] == 2004).all()
    assert len(df) == 6  # one season, C(4, 2) = 6 pairs


def test_run_hbt_loso_fails_when_no_numeric_features():
    fm = pd.DataFrame({"TeamID": [1, 2], "Season": [2024, 2024]})
    reg = pd.DataFrame([{
        "Season": 2024, "DayNum": 10, "WTeamID": 1, "LTeamID": 2, "WLoc": "N",
    }])
    tour = pd.DataFrame([{
        "Season": 2024, "DayNum": 11, "WTeamID": 1, "LTeamID": 2, "WLoc": "N",
    }])
    with pytest.raises(ValueError, match="no numeric"):
        run_hbt_loso(
            feature_matrix=fm,
            regular_results=reg,
            tourney_filtered=tour,
            feature_cols=["nonexistent"],
            sigmas=[1.0],
            verbose=False,
        )
