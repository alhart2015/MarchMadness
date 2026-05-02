"""prepare_loso_inputs() is a refactor extraction; the contract is:
Returns the same feature_matrix, tourney_filtered, regular_results,
feature_cols, top_80_by_season that v3's main() previously built inline.

This test pins the contract by checking shape, dtypes, and a few sentinel
values. It does NOT re-run the full backtest -- that's Task 2's smoke."""
import pandas as pd
import pytest


def test_prepare_loso_inputs_returns_expected_shape():
    from src.enhanced_model_v3 import prepare_loso_inputs

    out = prepare_loso_inputs()

    assert isinstance(out, dict)
    required_keys = {"feature_matrix", "tourney_filtered",
                     "regular_results", "feature_cols", "top_80_by_season"}
    assert set(out.keys()) >= required_keys, (
        f"missing keys: {required_keys - set(out.keys())}"
    )

    fm = out["feature_matrix"]
    assert isinstance(fm, pd.DataFrame)
    assert {"TeamID", "Season"} <= set(fm.columns)
    # Should span 2003+ seasons; >= 20 distinct seasons in the build.
    assert fm["Season"].nunique() >= 20

    fc = out["feature_cols"]
    assert isinstance(fc, list)
    assert len(fc) >= 30  # v4 has many feature cols; safeguard against accidental truncation
    # Sanity: a few well-known v4 feature names should be present.
    expected_subset = {"adj_oe", "adj_de", "coach_career_winpct"}
    assert expected_subset <= set(fc), (
        f"missing canonical v4 features: {expected_subset - set(fc)}"
    )

    tf = out["tourney_filtered"]
    assert isinstance(tf, pd.DataFrame)
    assert {"Season", "WTeamID", "LTeamID"} <= set(tf.columns)

    tn = out["top_80_by_season"]
    assert isinstance(tn, dict)
    # Every season in feature_matrix should have an entry, possibly empty.
    fm_seasons = set(int(s) for s in fm["Season"].unique())
    assert fm_seasons <= set(tn.keys())
