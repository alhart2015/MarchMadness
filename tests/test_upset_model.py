"""Tests for src/train_upset_model.py (v9 upset-aware stage-2)."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.train_upset_model import (
    build_v9_pairwise,
    compute_sample_weights,
    double_loso_eval,
    fit_upset_model,
    load_per_game_data_with_upset,
    parse_seed,
    upset_features,
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


def test_upset_features_extracts_seven_columns():
    df = pd.DataFrame([
        {**_make_row(p_stage1=0.7, label=1, upset=True), "round": 1},
        {**_make_row(p_stage1=0.3, label=0, upset=True), "round": 1},
    ])
    X = upset_features(df)
    assert X.shape == (2, 7)
    # Column 0 is p_stage1.
    assert X[0, 0] == pytest.approx(0.7)
    # Column 5 is v4 confidence = |p - 0.5|.
    assert X[0, 5] == pytest.approx(0.2)
    # Column 6 is is_a_higher_seed (seed_a=1 < seed_b=2 -> 1.0).
    assert X[0, 6] == pytest.approx(1.0)


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
        "round": np.random.randint(1, 7, n),
    })
    X = upset_features(df)
    y = df["label"].values
    w = compute_sample_weights(df)
    model = fit_upset_model(X, y, w, seed=42)
    assert hasattr(model, "predict_proba")
    p = model.predict_proba(X)[:, 1]
    assert p.shape == (n,)
    assert np.all((p >= 0.0) & (p <= 1.0))


# -----------------------------------------------------------------------------
# double_loso_eval: leakage guard
# -----------------------------------------------------------------------------


def test_double_loso_eval_never_trains_on_test_season(monkeypatch):
    """For each test season Y, the training fold passed to fit_upset_model
    must contain zero rows from season Y. Patch fit_upset_model to capture
    the training X / y / w it sees and assert the season filter held.
    """
    # Three seasons, one game each.
    rows = []
    for season in [2021, 2022, 2023]:
        rows.append({
            "season": season, "team_a": 1, "team_b": 2,
            "p_stage1": 0.7, "seed_a": 5, "seed_b": 12,
            "abs_seed_diff": 7, "upset": True, "round": 1, "label": 1,
        })
        rows.append({
            "season": season, "team_a": 2, "team_b": 1,
            "p_stage1": 0.3, "seed_a": 12, "seed_b": 5,
            "abs_seed_diff": 7, "upset": True, "round": 1, "label": 0,
        })
    per_game = pd.DataFrame(rows)

    captured = []

    class _StubModel:
        def predict_proba(self, X):
            # Probability that mirrors p_stage1 input (column 0).
            p = X[:, 0]
            return np.column_stack([1 - p, p])

    def _stub_fit(X, y, w, seed=42):
        captured.append({"n_rows": len(X)})
        return _StubModel()

    monkeypatch.setattr("src.train_upset_model.fit_upset_model", _stub_fit)

    # Run eval -- should call _stub_fit once per test season (3 times).
    eval_df = double_loso_eval(per_game)

    # 3 fits, each trained on the 4 rows from the OTHER 2 seasons (2 rows/game * 2 games).
    assert len(captured) == 3
    for c in captured:
        assert c["n_rows"] == 4

    assert set(eval_df["season"].tolist()) == {2021, 2022, 2023}


# -----------------------------------------------------------------------------
# build_v9_pairwise: writes output/pairwise_v9.csv with v9-adjusted probs
# -----------------------------------------------------------------------------


def test_build_v9_pairwise_writes_expected_schema(tmp_path):
    """build_v9_pairwise emits a CSV with columns season, team_a, team_b,
    p_a_wins, with team_a < team_b (v8-compatible schema)."""
    # Two seasons, two pairs each.
    pw_v4 = pd.DataFrame({
        "season": [2022, 2022, 2023, 2023],
        "team_a": [1, 1, 1, 2],
        "team_b": [2, 3, 3, 3],
        "p_a_wins": [0.7, 0.6, 0.55, 0.45],
    })
    pw_path = tmp_path / "pairwise_v4.csv"
    pw_v4.to_csv(pw_path, index=False)

    seeds = pd.DataFrame({
        "Season": [2022, 2022, 2022, 2023, 2023, 2023],
        "Seed":   ["W01", "W08", "W16", "W01", "W08", "W16"],
        "TeamID": [1, 2, 3, 1, 2, 3],
    })
    seeds_path = tmp_path / "seeds.csv"
    seeds.to_csv(seeds_path, index=False)

    # Per-game training rows mirroring two seasons (drives both LOSO folds).
    per_game = pd.DataFrame([
        {"season": 2022, "team_a": 1, "team_b": 2, "p_stage1": 0.7,
         "seed_a": 1, "seed_b": 8, "abs_seed_diff": 7,
         "upset": False, "round": 1, "label": 1},
        {"season": 2022, "team_a": 2, "team_b": 1, "p_stage1": 0.3,
         "seed_a": 8, "seed_b": 1, "abs_seed_diff": 7,
         "upset": False, "round": 1, "label": 0},
        {"season": 2023, "team_a": 2, "team_b": 3, "p_stage1": 0.45,
         "seed_a": 8, "seed_b": 16, "abs_seed_diff": 8,
         "upset": True, "round": 1, "label": 1},
        {"season": 2023, "team_a": 3, "team_b": 2, "p_stage1": 0.55,
         "seed_a": 16, "seed_b": 8, "abs_seed_diff": 8,
         "upset": True, "round": 1, "label": 0},
    ])

    out_path = tmp_path / "pairwise_v9.csv"
    build_v9_pairwise(per_game, str(pw_path), str(seeds_path), str(out_path))

    out = pd.read_csv(out_path)
    assert list(out.columns) == ["season", "team_a", "team_b", "p_a_wins"]
    # team_a < team_b on every row.
    assert (out["team_a"] < out["team_b"]).all()
    # All seasons from input represented.
    assert set(out["season"].tolist()) == {2022, 2023}
    # Probabilities in [0, 1].
    assert ((out["p_a_wins"] >= 0.0) & (out["p_a_wins"] <= 1.0)).all()


# -----------------------------------------------------------------------------
# double_loso_eval / build_v9_pairwise: weight threading
# -----------------------------------------------------------------------------


def test_double_loso_eval_threads_weights_to_sample_weight(monkeypatch):
    """When called with w_upset=1.0, w_miss=0.0, the sample_weight array
    passed to fit_upset_model is uniform (all ones). This is the
    sanity-check anchor for the v9 weight sweep.
    """
    rows = []
    for season in [2021, 2022, 2023]:
        # One upset + one non-upset per season.
        rows.extend([
            {"season": season, "team_a": 1, "team_b": 2,
             "p_stage1": 0.7, "seed_a": 5, "seed_b": 12,
             "abs_seed_diff": 7, "upset": True, "round": 1, "label": 1},
            {"season": season, "team_a": 2, "team_b": 1,
             "p_stage1": 0.3, "seed_a": 12, "seed_b": 5,
             "abs_seed_diff": 7, "upset": True, "round": 1, "label": 0},
            {"season": season, "team_a": 3, "team_b": 4,
             "p_stage1": 0.9, "seed_a": 1, "seed_b": 16,
             "abs_seed_diff": 15, "upset": False, "round": 1, "label": 1},
            {"season": season, "team_a": 4, "team_b": 3,
             "p_stage1": 0.1, "seed_a": 16, "seed_b": 1,
             "abs_seed_diff": 15, "upset": False, "round": 1, "label": 0},
        ])
    per_game = pd.DataFrame(rows)

    captured_weights = []

    class _StubModel:
        def predict_proba(self, X):
            p = X[:, 0]
            return np.column_stack([1 - p, p])

    def _stub_fit(X, y, w, seed=42):
        captured_weights.append(np.array(w, copy=True))
        return _StubModel()

    monkeypatch.setattr("src.train_upset_model.fit_upset_model", _stub_fit)

    double_loso_eval(per_game, w_upset=1.0, w_miss=0.0)

    assert len(captured_weights) == 3
    for w in captured_weights:
        assert np.allclose(w, 1.0), f"expected uniform weights, got {w}"


def test_double_loso_eval_default_weights_match_module_globals(monkeypatch):
    """Default call (no w_upset/w_miss) preserves the existing 3.0/4.0
    behavior: the captured sample weights match
    compute_sample_weights(train, w_upset=3.0, w_miss=4.0).
    """
    rows = []
    for season in [2021, 2022]:
        rows.extend([
            {"season": season, "team_a": 1, "team_b": 2,
             "p_stage1": 0.7, "seed_a": 5, "seed_b": 12,
             "abs_seed_diff": 7, "upset": True, "round": 1, "label": 1},
            {"season": season, "team_a": 2, "team_b": 1,
             "p_stage1": 0.3, "seed_a": 12, "seed_b": 5,
             "abs_seed_diff": 7, "upset": True, "round": 1, "label": 0},
        ])
    per_game = pd.DataFrame(rows)

    captured_weights = []

    class _StubModel:
        def predict_proba(self, X):
            return np.column_stack([1 - X[:, 0], X[:, 0]])

    def _stub_fit(X, y, w, seed=42):
        captured_weights.append(np.array(w, copy=True))
        return _StubModel()

    monkeypatch.setattr("src.train_upset_model.fit_upset_model", _stub_fit)
    double_loso_eval(per_game)  # default args

    # Re-derive expected weights from compute_sample_weights with the
    # canonical defaults.
    for capt, season in zip(captured_weights, sorted(per_game.season.unique())):
        train = per_game[per_game.season != season]
        expected = compute_sample_weights(train, w_upset=3.0, w_miss=4.0)
        assert np.allclose(capt, expected)
