"""Tests for src/train_upset_model.py (v9 upset-aware stage-2)."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.train_upset_model import (
    load_per_game_data_with_upset,
    parse_seed,
)


# -----------------------------------------------------------------------------
# parse_seed
# -----------------------------------------------------------------------------

def test_parse_seed_numeric():
    assert parse_seed("W01") == 1
    assert parse_seed("X16b") == 16
    assert parse_seed("Y11a") == 11


def test_parse_seed_invalid():
    assert parse_seed(None) is None
    assert parse_seed("") is None


# -----------------------------------------------------------------------------
# load_per_game_data_with_upset: synthetic CSVs in tmp_path
# -----------------------------------------------------------------------------

def _write_csvs(tmp_path: Path, pairwise: pd.DataFrame, results: pd.DataFrame,
                seeds: pd.DataFrame):
    pw_path = tmp_path / "pairwise.csv"
    res_path = tmp_path / "results.csv"
    seeds_path = tmp_path / "seeds.csv"
    pairwise.to_csv(pw_path, index=False)
    results.to_csv(res_path, index=False)
    seeds.to_csv(seeds_path, index=False)
    return str(pw_path), str(res_path), str(seeds_path)


def test_loader_flags_5_over_12_as_upset(tmp_path):
    """A 12-seed beating a 5-seed is an upset."""
    # team_a < team_b convention in pairwise CSV.
    pairwise = pd.DataFrame({
        "season": [2023], "team_a": [1101], "team_b": [1112],
        "p_a_wins": [0.7],
    })
    results = pd.DataFrame({
        "Season": [2023], "DayNum": [136],
        "WTeamID": [1112], "WScore": [70], "LTeamID": [1101], "LScore": [65],
    })
    seeds = pd.DataFrame({
        "Season": [2023, 2023], "Seed": ["W05", "W12"],
        "TeamID": [1101, 1112],
    })
    pw_p, res_p, seeds_p = _write_csvs(tmp_path, pairwise, results, seeds)
    df = load_per_game_data_with_upset(pw_p, res_p, seeds_p)
    assert len(df) == 2  # symmetric pair
    assert df["upset"].all()


def test_loader_flags_1_beats_16_as_non_upset(tmp_path):
    """A 1-seed beating a 16-seed is NOT an upset."""
    pairwise = pd.DataFrame({
        "season": [2023], "team_a": [1101], "team_b": [1116],
        "p_a_wins": [0.95],
    })
    results = pd.DataFrame({
        "Season": [2023], "DayNum": [136],
        "WTeamID": [1101], "WScore": [85], "LTeamID": [1116], "LScore": [60],
    })
    seeds = pd.DataFrame({
        "Season": [2023, 2023], "Seed": ["W01", "W16"],
        "TeamID": [1101, 1116],
    })
    pw_p, res_p, seeds_p = _write_csvs(tmp_path, pairwise, results, seeds)
    df = load_per_game_data_with_upset(pw_p, res_p, seeds_p)
    assert len(df) == 2
    assert not df["upset"].any()


def test_loader_same_seed_is_non_upset(tmp_path):
    """Same-seed game (e.g., F4 1-vs-1): no higher seed, never an upset."""
    pairwise = pd.DataFrame({
        "season": [2023], "team_a": [1101], "team_b": [1102],
        "p_a_wins": [0.55],
    })
    results = pd.DataFrame({
        "Season": [2023], "DayNum": [152],
        "WTeamID": [1102], "WScore": [70], "LTeamID": [1101], "LScore": [65],
    })
    # Both teams seeded 1 (different regions).
    seeds = pd.DataFrame({
        "Season": [2023, 2023], "Seed": ["W01", "X01"],
        "TeamID": [1101, 1102],
    })
    pw_p, res_p, seeds_p = _write_csvs(tmp_path, pairwise, results, seeds)
    df = load_per_game_data_with_upset(pw_p, res_p, seeds_p)
    assert len(df) == 2
    assert not df["upset"].any()
    # abs_seed_diff is 0 in this case.
    assert (df["abs_seed_diff"] == 0).all()


def test_loader_produces_symmetric_rows(tmp_path):
    """Each game produces (a=W,label=1) and (a=L,label=0) with mirrored p_stage1."""
    pairwise = pd.DataFrame({
        "season": [2023], "team_a": [1101], "team_b": [1112],
        "p_a_wins": [0.7],
    })
    results = pd.DataFrame({
        "Season": [2023], "DayNum": [136],
        "WTeamID": [1101], "WScore": [80], "LTeamID": [1112], "LScore": [70],
    })
    seeds = pd.DataFrame({
        "Season": [2023, 2023], "Seed": ["W05", "W12"],
        "TeamID": [1101, 1112],
    })
    pw_p, res_p, seeds_p = _write_csvs(tmp_path, pairwise, results, seeds)
    df = load_per_game_data_with_upset(pw_p, res_p, seeds_p)
    win_row = df[df.label == 1].iloc[0]
    loss_row = df[df.label == 0].iloc[0]
    assert win_row["team_a"] == 1101 and win_row["team_b"] == 1112
    assert loss_row["team_a"] == 1112 and loss_row["team_b"] == 1101
    # Mirrored p_stage1: 0.7 (winner perspective) and 1 - 0.7 = 0.3 (loser).
    assert win_row["p_stage1"] == pytest.approx(0.7)
    assert loss_row["p_stage1"] == pytest.approx(0.3)


# -----------------------------------------------------------------------------
# compute_sample_weights
# -----------------------------------------------------------------------------

from src.train_upset_model import compute_sample_weights


def _make_row(p_stage1: float, label: int, upset: bool) -> dict:
    return {
        "p_stage1": p_stage1, "label": label, "upset": upset,
        # The other columns aren't read by compute_sample_weights, but
        # included so the DataFrame mirrors the loader output.
        "season": 2023, "team_a": 1, "team_b": 2,
        "seed_a": 1, "seed_b": 2, "abs_seed_diff": 1,
    }


def test_weights_non_upset_well_predicted_is_one():
    """Non-upset row, v4 confidently right: weight ~ 1."""
    df = pd.DataFrame([_make_row(p_stage1=0.95, label=1, upset=False)])
    w = compute_sample_weights(df, w_upset=3.0, w_miss=4.0)
    # residual^2 = (1 - 0.95)^2 = 0.0025; w = 1 * (1 + 4 * 0.0025) = 1.01
    assert w.shape == (1,)
    assert w[0] == pytest.approx(1.0 + 4.0 * (1 - 0.95) ** 2)
    assert 1.0 < w[0] < 1.05


def test_weights_non_upset_missed_amplifies():
    """Non-upset row, v4 confidently wrong: weight ~ 5."""
    df = pd.DataFrame([_make_row(p_stage1=0.05, label=1, upset=False)])
    w = compute_sample_weights(df, w_upset=3.0, w_miss=4.0)
    # residual = 1 - 0.05 = 0.95; (1 + 4 * 0.9025) = 4.61
    expected = 1.0 + 4.0 * (1 - 0.05) ** 2
    assert w[0] == pytest.approx(expected)
    assert w[0] > 4.0


def test_weights_upset_predicted_uses_upset_factor():
    """Upset row, v4 nearly hit: weight ~ 3."""
    df = pd.DataFrame([_make_row(p_stage1=0.6, label=1, upset=True)])
    w = compute_sample_weights(df, w_upset=3.0, w_miss=4.0)
    # base 1 * 3 (upset) * (1 + 4 * 0.4^2) = 3 * 1.64 = 4.92
    expected = 3.0 * (1.0 + 4.0 * (1 - 0.6) ** 2)
    assert w[0] == pytest.approx(expected)
    # Sanity: well above non-upset baseline.
    assert w[0] > 3.0


def test_weights_upset_confidently_missed_is_largest():
    """Upset row, v4 confidently wrong: weight ~ 15."""
    df = pd.DataFrame([_make_row(p_stage1=0.05, label=1, upset=True)])
    w = compute_sample_weights(df, w_upset=3.0, w_miss=4.0)
    # 3 * (1 + 4 * 0.95^2) = 3 * 4.61 = 13.83
    expected = 3.0 * (1.0 + 4.0 * (1 - 0.05) ** 2)
    assert w[0] == pytest.approx(expected)
    assert w[0] > 13.0


def test_weights_disabled_when_factors_are_unit():
    """w_upset=1, w_miss=0 -> all weights == 1."""
    df = pd.DataFrame([
        _make_row(0.5, 1, True),
        _make_row(0.9, 0, False),
        _make_row(0.05, 1, True),
    ])
    w = compute_sample_weights(df, w_upset=1.0, w_miss=0.0)
    assert np.allclose(w, 1.0)


def test_weights_uses_correct_residual_for_loser_perspective():
    """Loser-perspective row (label=0): residual is computed against label=0,
    not label=1. Otherwise the symmetric pair would carry asymmetric weights
    even when v4 was perfectly calibrated.
    """
    # v4 says p(A wins) = 0.7. Symmetric pair: winner row p_stage1=0.7,label=1;
    # loser row p_stage1=0.3,label=0. Both should have residual^2 = 0.09.
    df = pd.DataFrame([
        _make_row(p_stage1=0.7, label=1, upset=False),
        _make_row(p_stage1=0.3, label=0, upset=False),
    ])
    w = compute_sample_weights(df, w_upset=3.0, w_miss=4.0)
    assert w[0] == pytest.approx(w[1])


# -----------------------------------------------------------------------------
# upset_features + fit_upset_model
# -----------------------------------------------------------------------------

from src.train_upset_model import fit_upset_model, upset_features


def test_upset_features_extracts_four_columns():
    df = pd.DataFrame([
        _make_row(p_stage1=0.7, label=1, upset=True),
        _make_row(p_stage1=0.3, label=0, upset=True),
    ])
    X = upset_features(df)
    assert X.shape == (2, 4)
    # Expected column order: p_stage1, seed_a, seed_b, abs_seed_diff
    assert X[0, 0] == pytest.approx(0.7)


def test_fit_upset_model_returns_classifier_with_predict_proba():
    """Smoke test: 100-row synthetic dataset, model trains and predicts."""
    np.random.seed(42)
    n = 100
    p_stage1 = np.random.uniform(0.05, 0.95, n)
    label = (p_stage1 > 0.5).astype(int)
    seed_a = np.random.randint(1, 17, n)
    seed_b = np.random.randint(1, 17, n)
    df = pd.DataFrame({
        "p_stage1": p_stage1, "label": label,
        "seed_a": seed_a, "seed_b": seed_b,
        "abs_seed_diff": np.abs(seed_a - seed_b),
        "upset": np.random.choice([True, False], n),
    })
    X = upset_features(df)
    y = df["label"].values
    w = compute_sample_weights(df)
    model = fit_upset_model(X, y, w, seed=42)
    assert hasattr(model, "predict_proba")
    p = model.predict_proba(X)[:, 1]
    assert p.shape == (n,)
    assert np.all((p >= 0.0) & (p <= 1.0))
