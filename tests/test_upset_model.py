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

    # Synthetic slots for the bracket-walk helper to exercise both
    # round=1 (R1 game) and round=2 (R2 game) without needing a full
    # 64-team bracket.
    slots = pd.DataFrame({
        "Season": [2022, 2022, 2023, 2023],
        "Slot":   ["R1W1", "R2W1", "R1W1", "R2W1"],
        "StrongSeed": ["W01", "R1W1", "W01", "R1W1"],
        "WeakSeed":   ["W08", "W16",  "W08", "W16"],
    })
    slots_path = tmp_path / "slots.csv"
    slots.to_csv(slots_path, index=False)

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
    build_v9_pairwise(
        per_game, str(pw_path), str(seeds_path), str(out_path),
        slots_csv=str(slots_path),
    )

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


def test_build_v9_pairwise_threads_weights_to_sample_weight(
    tmp_path, monkeypatch
):
    """build_v9_pairwise(..., w_upset=1.0, w_miss=0.0) -> uniform weights."""
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

    slots = pd.DataFrame({
        "Season": [2022, 2022, 2023, 2023],
        "Slot":   ["R1W1", "R2W1", "R1W1", "R2W1"],
        "StrongSeed": ["W01", "R1W1", "W01", "R1W1"],
        "WeakSeed":   ["W08", "W16",  "W08", "W16"],
    })
    slots_path = tmp_path / "slots.csv"
    slots.to_csv(slots_path, index=False)

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

    captured_weights = []

    class _StubModel:
        def predict_proba(self, X):
            return np.column_stack([1 - X[:, 0], X[:, 0]])

    def _stub_fit(X, y, w, seed=42):
        captured_weights.append(np.array(w, copy=True))
        return _StubModel()

    monkeypatch.setattr("src.train_upset_model.fit_upset_model", _stub_fit)

    out_path = tmp_path / "pairwise_v9.csv"
    build_v9_pairwise(
        per_game, str(pw_path), str(seeds_path), str(out_path),
        slots_csv=str(slots_path),
        w_upset=1.0, w_miss=0.0,
    )

    # 2 LOSO fits (one per season in pw_v4), both must be uniform.
    assert len(captured_weights) == 2
    for w in captured_weights:
        assert np.allclose(w, 1.0), f"expected uniform weights, got {w}"


# -----------------------------------------------------------------------------
# build_pair_round_lookup: bracket-walk for apply-time round
# -----------------------------------------------------------------------------


def _real_2024_slots_seeds():
    """Load real 2024 MNCAATourneySlots and MNCAATourneySeeds for the
    bracket-walk helper tests. Test data lives under
    data/raw/march-machine-learning-2026/.
    """
    from pathlib import Path
    data_dir = Path("data/raw/march-machine-learning-2026")
    slots = pd.read_csv(data_dir / "MNCAATourneySlots.csv")
    seeds = pd.read_csv(data_dir / "MNCAATourneySeeds.csv")
    return slots, seeds


def test_build_pair_round_lookup_r1_same_region():
    """1-seed vs 16-seed in the same region meet in R1."""
    from src.train_upset_model import build_pair_round_lookup
    slots, seeds = _real_2024_slots_seeds()
    season = 2024
    season_seeds = seeds[seeds.Season == season]
    w01_team = int(season_seeds[season_seeds.Seed == "W01"]["TeamID"].iloc[0])
    w16_team = int(season_seeds[season_seeds.Seed == "W16"]["TeamID"].iloc[0])
    lookup = build_pair_round_lookup(season, slots, seeds)
    a, b = sorted([w01_team, w16_team])
    assert lookup[(a, b)] == 1


def test_build_pair_round_lookup_r2_same_region():
    """1-seed vs 8-seed (or 9-seed) in the same region meet in R2."""
    from src.train_upset_model import build_pair_round_lookup
    slots, seeds = _real_2024_slots_seeds()
    season = 2024
    season_seeds = seeds[seeds.Season == season]
    w01_team = int(season_seeds[season_seeds.Seed == "W01"]["TeamID"].iloc[0])
    w08_team = int(season_seeds[season_seeds.Seed == "W08"]["TeamID"].iloc[0])
    lookup = build_pair_round_lookup(season, slots, seeds)
    a, b = sorted([w01_team, w08_team])
    assert lookup[(a, b)] == 2


def test_build_pair_round_lookup_r4_same_region():
    """1-seed vs 2-seed in the same region meet in R4 (regional final / E8)."""
    from src.train_upset_model import build_pair_round_lookup
    slots, seeds = _real_2024_slots_seeds()
    season = 2024
    season_seeds = seeds[seeds.Season == season]
    w01_team = int(season_seeds[season_seeds.Seed == "W01"]["TeamID"].iloc[0])
    w02_team = int(season_seeds[season_seeds.Seed == "W02"]["TeamID"].iloc[0])
    lookup = build_pair_round_lookup(season, slots, seeds)
    a, b = sorted([w01_team, w02_team])
    assert lookup[(a, b)] == 4


def test_build_pair_round_lookup_cross_region_r5_or_r6():
    """1-seed in W vs 1-seed in X meet in R5 or R6 (F4 or Champ),
    depending on F4 pairings for that season.
    """
    from src.train_upset_model import build_pair_round_lookup
    slots, seeds = _real_2024_slots_seeds()
    season = 2024
    season_seeds = seeds[seeds.Season == season]
    w01_team = int(season_seeds[season_seeds.Seed == "W01"]["TeamID"].iloc[0])
    x01_team = int(season_seeds[season_seeds.Seed == "X01"]["TeamID"].iloc[0])
    lookup = build_pair_round_lookup(season, slots, seeds)
    a, b = sorted([w01_team, x01_team])
    assert lookup[(a, b)] in (5, 6)


def test_build_pair_round_lookup_covers_all_seed_pairs():
    """For 64 main-bracket teams (after play-in resolution), expect
    C(64, 2) = 2016 pairs in the lookup. Allow some slack for play-in
    seeds that do not have a unique team.
    """
    from src.train_upset_model import build_pair_round_lookup
    slots, seeds = _real_2024_slots_seeds()
    season = 2024
    lookup = build_pair_round_lookup(season, slots, seeds)
    # Lower bound: at least 1900 pairs covered. (Some 2024 play-in slots
    # may resolve to multiple seed strings but only one team_id, so
    # exact-2016 is unrealistic; 1900 is a safe floor.)
    assert len(lookup) >= 1900
    # All round values in 1..6.
    rounds = set(lookup.values())
    assert rounds.issubset({1, 2, 3, 4, 5, 6})


def test_build_pair_round_lookup_canonical_pair_ordering():
    """All keys are (a, b) with a < b -- canonical ordering matches
    the pairwise CSV's team_a < team_b convention.
    """
    from src.train_upset_model import build_pair_round_lookup
    slots, seeds = _real_2024_slots_seeds()
    season = 2024
    lookup = build_pair_round_lookup(season, slots, seeds)
    for (a, b) in lookup.keys():
        assert a < b, f"non-canonical key ({a}, {b})"


def test_build_v9_pairwise_uses_real_round_at_apply(tmp_path, monkeypatch):
    """build_v9_pairwise must populate apply_df['round'] from the
    season's bracket structure, not hardcode 0. Capture the apply-time
    feature matrix via a stub fit and assert round-column values are
    in 1..6 for at least the canonical 1-vs-16 R1 pair.
    """
    real_slots, real_seeds = _real_2024_slots_seeds()
    season = 2024
    season_seeds = real_seeds[real_seeds.Season == season]
    w01_team = int(season_seeds[season_seeds.Seed == "W01"]["TeamID"].iloc[0])
    w16_team = int(season_seeds[season_seeds.Seed == "W16"]["TeamID"].iloc[0])
    a, b = sorted([w01_team, w16_team])

    pw_v4 = pd.DataFrame({
        "season": [season], "team_a": [a], "team_b": [b],
        "p_a_wins": [0.95],
    })
    pw_path = tmp_path / "pairwise_v4.csv"
    pw_v4.to_csv(pw_path, index=False)

    seeds_path = tmp_path / "seeds.csv"
    real_seeds[real_seeds.Season == season].to_csv(seeds_path, index=False)
    slots_path = tmp_path / "slots.csv"
    real_slots[real_slots.Season == season].to_csv(slots_path, index=False)

    # Per-game training rows -- need at least one OTHER season so the
    # LOSO loop has training data when 2024 is the test season.
    other_season = 2023
    per_game = pd.DataFrame([
        {"season": other_season, "team_a": a, "team_b": b, "p_stage1": 0.95,
         "seed_a": 1, "seed_b": 16, "abs_seed_diff": 15,
         "upset": False, "round": 1, "label": 1},
        {"season": other_season, "team_a": b, "team_b": a, "p_stage1": 0.05,
         "seed_a": 16, "seed_b": 1, "abs_seed_diff": 15,
         "upset": False, "round": 1, "label": 0},
    ])

    captured_X = []

    class _StubModel:
        def predict_proba(self, X):
            captured_X.append(np.array(X, copy=True))
            return np.column_stack([1 - X[:, 0], X[:, 0]])

    def _stub_fit(X, y, w, seed=42):
        return _StubModel()

    monkeypatch.setattr("src.train_upset_model.fit_upset_model", _stub_fit)

    out_path = tmp_path / "pairwise_v9.csv"
    build_v9_pairwise(
        per_game,
        str(pw_path),
        str(seeds_path),
        str(out_path),
        slots_csv=str(slots_path),
    )

    # The apply path called predict_proba with a feature matrix whose
    # round column (index 4) is non-zero for the W01-vs-W16 pair.
    assert len(captured_X) == 1
    X = captured_X[0]
    # 7 features: p_stage1, seed_a, seed_b, abs_seed_diff, round,
    # v4_confidence, is_a_higher_seed. Round is column index 4.
    assert X.shape[1] == 7
    rounds = X[:, 4]
    # Both rows of the symmetric pair should be R1 = 1.0.
    assert np.allclose(rounds, 1.0), \
        f"expected round=1 at apply time, got {rounds}"


# ---------------------------------------------------------------------------
# upset_features feature_set parameterization
# ---------------------------------------------------------------------------

def _per_game_fixture():
    """Minimal per-game DataFrame with the columns upset_features reads."""
    return pd.DataFrame({
        "p_stage1": [0.7, 0.3, 0.55, 0.45],
        "seed_a":   [1.0, 16.0, 5.0, 12.0],
        "seed_b":   [16.0, 1.0, 12.0, 5.0],
        "abs_seed_diff": [15.0, 15.0, 7.0, 7.0],
        "round": [1.0, 1.0, 2.0, 2.0],
    })


def test_upset_features_default_is_v9b():
    """Default (no feature_set kwarg) returns the 7-feature v9-B matrix."""
    X = upset_features(_per_game_fixture())
    assert X.shape == (4, 7)


def test_upset_features_v9b_explicit_matches_default():
    """Passing feature_set='v9b' is bit-identical to the default."""
    df = _per_game_fixture()
    X_default = upset_features(df)
    X_v9b = upset_features(df, feature_set="v9b")
    assert np.array_equal(X_default, X_v9b)


def test_upset_features_v9c_shape_5():
    """feature_set='v9c' returns shape (n, 5): drops v4_confidence and
    is_a_higher_seed."""
    X = upset_features(_per_game_fixture(), feature_set="v9c")
    assert X.shape == (4, 5)


def test_upset_features_v9c_columns_match_v9b_subset():
    """v9-C columns 0..4 must equal v9-B columns 0..4 elementwise.
    The first 5 columns (p_stage1, seed_a, seed_b, abs_seed_diff, round)
    are identical between variants; v9-B just appends 2 more.
    """
    df = _per_game_fixture()
    X_v9b = upset_features(df, feature_set="v9b")
    X_v9c = upset_features(df, feature_set="v9c")
    assert np.array_equal(X_v9c, X_v9b[:, :5])


def test_upset_features_invalid_feature_set_raises():
    """Unknown feature_set values raise ValueError -- typos must fail fast."""
    with pytest.raises(ValueError, match="feature_set"):
        upset_features(_per_game_fixture(), feature_set="v9a")


# -----------------------------------------------------------------------------
# double_loso_eval / build_v9_pairwise feature_set threading
# -----------------------------------------------------------------------------

def _two_season_per_game_fixture(tmp_path: Path):
    """Build a per-game DataFrame across 2 seasons via load_per_game_data_with_upset."""
    pairwise = pd.DataFrame({
        "season": [2022, 2022, 2023, 2023],
        "team_a": [1, 1, 1, 2],
        "team_b": [2, 3, 3, 3],
        "p_a_wins": [0.7, 0.6, 0.55, 0.45],
    })
    results = pd.DataFrame({
        "Season": [2022, 2022, 2023, 2023],
        "DayNum": [136, 138, 136, 138],
        "WTeamID": [1, 1, 1, 2],
        "WScore": [70, 75, 65, 70],
        "LTeamID": [2, 3, 3, 3],
        "LScore": [60, 65, 60, 65],
    })
    seeds = pd.DataFrame({
        "Season": [2022, 2022, 2022, 2023, 2023, 2023],
        "Seed":   ["W01", "W08", "W16", "W01", "W08", "W16"],
        "TeamID": [1, 2, 3, 1, 2, 3],
    })
    pw_p, res_p, seeds_p = _write_csvs(tmp_path, pairwise, results, seeds)
    return load_per_game_data_with_upset(pw_p, res_p, seeds_p), pw_p, seeds_p


def test_double_loso_eval_v9c_runs(tmp_path):
    """double_loso_eval accepts feature_set='v9c' and returns valid metrics."""
    per_game, _, _ = _two_season_per_game_fixture(tmp_path)
    eval_df = double_loso_eval(per_game, feature_set="v9c")
    assert len(eval_df) > 0
    assert "ll_v9" in eval_df.columns
    assert "acc_v9" in eval_df.columns


def test_build_v9_pairwise_v9c_writes_csv(tmp_path):
    """build_v9_pairwise accepts feature_set='v9c' and writes a v8-compatible
    pairwise CSV with the expected schema and row count.
    """
    per_game, pw_path, seeds_path = _two_season_per_game_fixture(tmp_path)
    slots = pd.DataFrame({
        "Season": [2022, 2022, 2023, 2023],
        "Slot":   ["R1W1", "R2W1", "R1W1", "R2W1"],
        "StrongSeed": ["W01", "R1W1", "W01", "R1W1"],
        "WeakSeed":   ["W08", "W16",  "W08", "W16"],
    })
    slots_path = tmp_path / "slots.csv"
    slots.to_csv(slots_path, index=False)

    out_path = tmp_path / "pairwise_v9c.csv"
    build_v9_pairwise(
        per_game, pw_path, seeds_path, str(out_path),
        slots_csv=str(slots_path),
        feature_set="v9c",
    )
    out = pd.read_csv(out_path)
    assert list(out.columns) == ["season", "team_a", "team_b", "p_a_wins"]
    # Same row count as input pairwise (4 rows).
    assert len(out) == 4
    assert (out["team_a"] < out["team_b"]).all()
