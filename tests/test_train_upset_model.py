"""Unit tests for src/train_upset_model.py.

Coverage focuses on the v9-D extension: pairwise_bt_csv join in
load_per_game_data_with_upset, the 'v9d' branch of upset_features, and
the apply-time pairwise builder's BT threading. v9-A/B/C behavior is
exercised end-to-end in tests/test_sweep_v9_weights.py and the existing
real-data sweep artifacts; this file pins the new code paths.
"""
import numpy as np
import pandas as pd
import pytest


def _write_pairwise(path, rows):
    df = pd.DataFrame(rows, columns=["season", "team_a", "team_b", "p_a_wins"])
    df.to_csv(path, index=False)


def _write_seeds(path, rows):
    """rows: list of (Season, Seed, TeamID)."""
    df = pd.DataFrame(rows, columns=["Season", "Seed", "TeamID"])
    df.to_csv(path, index=False)


def _write_results(path, rows):
    """rows: list of (Season, DayNum, WTeamID, LTeamID)."""
    df = pd.DataFrame(rows, columns=["Season", "DayNum", "WTeamID", "LTeamID"])
    df.to_csv(path, index=False)


def test_load_per_game_data_no_pbt_backwards_compat(tmp_path):
    """When pairwise_bt_csv is omitted, the returned frame has no p_bt
    column -- v9-A/B/C consumers continue to work unchanged.
    """
    from src.train_upset_model import load_per_game_data_with_upset

    pw = tmp_path / "pw_v4.csv"
    seeds = tmp_path / "seeds.csv"
    results = tmp_path / "results.csv"
    _write_pairwise(pw, [(2022, 1, 2, 0.7)])
    _write_seeds(seeds, [(2022, "W01", 1), (2022, "W08", 2)])
    _write_results(results, [(2022, 136, 1, 2)])

    df = load_per_game_data_with_upset(str(pw), str(results), str(seeds))

    assert "p_bt" not in df.columns
    # Sanity: the two symmetric rows are present.
    assert len(df) == 2


def test_load_per_game_data_joins_pbt(tmp_path):
    """When pairwise_bt_csv is provided, p_bt is joined per row with
    correct A/B orientation. The (W=1, L=2) row gets the (1, 2, p) BT
    lookup directly; the (W=2, L=1) symmetric row gets 1 - p.
    """
    from src.train_upset_model import load_per_game_data_with_upset

    pw_v4 = tmp_path / "pw_v4.csv"
    pw_bt = tmp_path / "pw_bt.csv"
    seeds = tmp_path / "seeds.csv"
    results = tmp_path / "results.csv"
    _write_pairwise(pw_v4, [(2022, 1, 2, 0.7)])
    _write_pairwise(pw_bt, [(2022, 1, 2, 0.6)])
    _write_seeds(seeds, [(2022, "W01", 1), (2022, "W08", 2)])
    _write_results(results, [(2022, 136, 1, 2)])

    df = load_per_game_data_with_upset(
        str(pw_v4), str(results), str(seeds),
        pairwise_bt_csv=str(pw_bt),
    )

    assert "p_bt" in df.columns
    # Two rows: (W=1, L=2) with label=1, and (W=2, L=1) with label=0.
    # The pairwise CSV stores (1, 2, 0.6); for (W=1, L=2) row, p_bt is
    # the WIN-perspective probability for team 1 = 0.6.
    win_row = df[(df.team_a == 1) & (df.team_b == 2)].iloc[0]
    assert win_row["label"] == 1
    assert win_row["p_bt"] == pytest.approx(0.6)
    # For (W=2, L=1) symmetric row -- team_a=1 (loser perspective with
    # label=0). The function records this row with team_a=L=2,
    # team_b=W=1, label=0; p_bt is the LOSER perspective = 1 - 0.6.
    los_row = df[(df.team_a == 2) & (df.team_b == 1)].iloc[0]
    assert los_row["label"] == 0
    assert los_row["p_bt"] == pytest.approx(0.4)


def test_load_per_game_data_pbt_drops_missing_lookups(tmp_path):
    """If a (season, a, b) pair appears in pairwise_v4 but not in
    pairwise_bt, the row is dropped (consistent with how missing v4
    lookups already drop rows). This avoids silent NaN propagation
    into the feature matrix.
    """
    from src.train_upset_model import load_per_game_data_with_upset

    pw_v4 = tmp_path / "pw_v4.csv"
    pw_bt = tmp_path / "pw_bt.csv"
    seeds = tmp_path / "seeds.csv"
    results = tmp_path / "results.csv"
    # v4 has (2022, 1, 2) AND (2022, 1, 3); BT has only (2022, 1, 2).
    _write_pairwise(pw_v4, [(2022, 1, 2, 0.7), (2022, 1, 3, 0.6)])
    _write_pairwise(pw_bt, [(2022, 1, 2, 0.6)])
    _write_seeds(seeds, [
        (2022, "W01", 1), (2022, "W08", 2), (2022, "W16", 3),
    ])
    _write_results(results, [(2022, 136, 1, 2), (2022, 138, 1, 3)])

    df = load_per_game_data_with_upset(
        str(pw_v4), str(results), str(seeds),
        pairwise_bt_csv=str(pw_bt),
    )

    # Only the (1, 2) game survives -- 2 symmetric rows, both with p_bt set.
    assert len(df) == 2
    assert df["p_bt"].notna().all()
    assert set(zip(df.team_a, df.team_b)) == {(1, 2), (2, 1)}


def test_upset_features_v9d_shape_and_columns(tmp_path):
    """feature_set='v9d' returns (n, 6) matrix with columns
    [p_stage1, seed_a, seed_b, abs_seed_diff, round, p_bt].
    """
    from src.train_upset_model import upset_features

    df = pd.DataFrame([
        {"p_stage1": 0.7, "seed_a": 1, "seed_b": 8, "abs_seed_diff": 7,
         "round": 1, "p_bt": 0.6},
        {"p_stage1": 0.3, "seed_a": 16, "seed_b": 1, "abs_seed_diff": 15,
         "round": 1, "p_bt": 0.2},
    ])
    X = upset_features(df, feature_set="v9d")

    assert X.shape == (2, 6)
    # Column order: p_stage1, seed_a, seed_b, abs_seed_diff, round, p_bt.
    assert X[0, 0] == 0.7
    assert X[0, 1] == 1
    assert X[0, 2] == 8
    assert X[0, 3] == 7
    assert X[0, 4] == 1
    assert X[0, 5] == 0.6


def test_upset_features_v9d_missing_pbt_raises():
    """If feature_set='v9d' is requested but the frame lacks 'p_bt',
    raise ValueError with a helpful message rather than silently
    producing a bad column.
    """
    from src.train_upset_model import upset_features

    df = pd.DataFrame([
        {"p_stage1": 0.7, "seed_a": 1, "seed_b": 8, "abs_seed_diff": 7,
         "round": 1},
    ])
    with pytest.raises(ValueError, match="p_bt"):
        upset_features(df, feature_set="v9d")


def test_upset_features_unknown_feature_set_raises():
    """Defensive: 'v9z' is not a known set."""
    from src.train_upset_model import upset_features

    df = pd.DataFrame([
        {"p_stage1": 0.7, "seed_a": 1, "seed_b": 8, "abs_seed_diff": 7,
         "round": 1},
    ])
    with pytest.raises(ValueError, match="v9z"):
        upset_features(df, feature_set="v9z")
