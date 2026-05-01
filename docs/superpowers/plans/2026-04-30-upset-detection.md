# Upset-Detection Sub-Model (v9) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace v8 stage-2 corrector with v9, an upset-aware variant trained on the same v4 out-of-fold pairwise predictions and the same 4 features, but with sample weights that emphasize upsets and high-confidence-miss rows. Run a 22-season LOSO head-to-head against v8 on bracket points. Ship v9-B (feature-extended fallback) only if v9-A ties or loses.

**Architecture:** New file `src/train_upset_model.py` mirrors `src/train_stage2.py` (untouched). Shared inputs (`output/pairwise_v4.csv`), shared output schema (per-pair CSV with `season,team_a,team_b,p_a_wins`). Differentiator from v8 lives entirely in sample weights computed in `compute_sample_weights`. Eval / scoring reuses `src/score_chalk_brackets.py` after adding "v9" to its `versions` list.

**Tech Stack:** Python 3.11, XGBoost, pandas, numpy, scikit-learn (for `log_loss`), pytest.

**Spec:** `docs/superpowers/specs/2026-04-30-upset-detection-design.md`

---

## File Plan

**Files to create:**

- `src/train_upset_model.py` -- v9 trainer. Single file. Sibling to `src/train_stage2.py`.
- `tests/test_upset_model.py` -- unit tests for the weight scheme, per-game loader, leakage guard, and a small synthetic-data smoke test.
- `docs/notes/2026-04-30-upset-detection-v9.md` -- writeup with verdict (created in Task 10).

**Files to modify:**

- `src/score_chalk_brackets.py` -- add "v9" to the `versions` list at line 222 (single-line change in Task 8).
- `TODO.md` -- update active queue / Done section based on the verdict (Task 10).

**Files to read (do not modify):**

- `src/train_stage2.py` -- reference pattern for the trainer. Copy helpers (`parse_seed`, the `load_per_game_data` shape, `double_loso_eval` skeleton, `build_v8_pairwise` skeleton) and adapt; do not import.
- `output/pairwise_v4.csv` -- v4's out-of-fold predictions across 22 backtest seasons. Input.
- `data/raw/march-machine-learning-2026/MNCAATourneySeeds.csv`, `MNCAATourneyCompactResults.csv` -- input.

---

## Phase 1: Per-game loader with upset flag

### Task 1: Skeleton + constants + per-game loader

**Files:**
- Create: `src/train_upset_model.py`

- [ ] **Step 1: Create `src/train_upset_model.py` with module docstring, imports, constants, and the `parse_seed` helper.**

```python
"""Upset-aware stage-2 model (v9): replacement for v8 with upset-weighted training.

Like v8 (src/train_stage2.py), v9 is a small XGBoost trained on v4's
out-of-fold pairwise predictions plus seed-pair context, predicting the
actual game outcome under double-LOSO. The differentiator is the loss:
training rows are weighted to emphasize upsets (higher seed lost) and
high-confidence-miss rows (where v4 was wrong with high confidence).

Inputs (4 features, identical to v8):
    p_v4_stage1, seed_a, seed_b, abs_seed_diff
Target:
    label = 1 if A beat B else 0 (symmetric: each game contributes 2 rows).

Sample weights:
    w = 1.0
    if higher_seed_team_in_this_game_lost: w *= W_UPSET   (default 3.0)
    w *= 1 + W_MISS * residual ** 2                       (default W_MISS = 4.0)
    where residual = label - p_v4_for_this_perspective.

Same-seed games (rare; F4 / Champ): no upset flag; W_UPSET multiplier
skipped. W_MISS multiplier still applies.

Outputs:
    output/pairwise_v9.csv -- v9-adjusted pairwise probs across 22 LOSO
        seasons. Same schema as pairwise_v8.csv.
    output/v9_eval.csv     -- per-season comparison row (v4, v8, v9).

Spec: docs/superpowers/specs/2026-04-30-upset-detection-design.md
"""
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import log_loss as sklearn_log_loss

DATA = Path("data/raw/march-machine-learning-2026")
OUTPUT = Path("output")

SEASONS_TO_BACKTEST = list(range(2003, 2026))  # 2003..2025 (excluding 2020 implicitly via data)

# Sample weighting hyperparameters. Tunable via narrow sweep.
W_UPSET = 3.0
W_MISS = 4.0


def parse_seed(seed_str):
    """Pull the integer seed out of strings like 'W01', 'W11a', 'X16b'.

    Copied from src/train_stage2.py: same Kaggle seed-string format.
    """
    if not isinstance(seed_str, str):
        return None
    digits = "".join(c for c in seed_str if c.isdigit())
    return int(digits) if digits else None
```

- [ ] **Step 2: Add `load_per_game_data_with_upset` to the same file.**

Append to `src/train_upset_model.py`:

```python
def load_per_game_data_with_upset(
    pairwise_csv: str, results_csv: str, seeds_csv: str
) -> pd.DataFrame:
    """Build per-played-game training rows for v9.

    Each row: (season, team_a, team_b, p_stage1, seed_a, seed_b,
              abs_seed_diff, upset, label). Symmetric: each game produces
              two rows (a=W,b=L; a=L,b=W). The upset flag is per-game
              (independent of A/B perspective) -- True iff the higher-
              seeded team lost. Same-seed games are flagged upset=False.

    Adapted from src/train_stage2.py:load_per_game_data: identical except
    for the added upset column.
    """
    pw = pd.read_csv(pairwise_csv)
    pw["pair_key"] = list(zip(pw["season"], pw["team_a"], pw["team_b"]))
    # Last write wins (default + tuned LOSO each appended); take final row per pair.
    pw = pw.drop_duplicates("pair_key", keep="last")
    pw_lookup = {(s, a, b): float(p)
                 for s, a, b, p in zip(pw.season, pw.team_a, pw.team_b, pw.p_a_wins)}

    results = pd.read_csv(results_csv)
    seeds = pd.read_csv(seeds_csv)
    seeds["seed_int"] = seeds["Seed"].apply(parse_seed)
    seed_lookup = {(int(r["Season"]), int(r["TeamID"])): r["seed_int"]
                   for _, r in seeds.iterrows() if r["seed_int"] is not None}

    rows = []
    for _, g in results.iterrows():
        season = int(g["Season"])
        if season not in SEASONS_TO_BACKTEST:
            continue
        w, l = int(g["WTeamID"]), int(g["LTeamID"])
        # pairwise CSV stores pairs as (min_id, max_id, p_min_wins).
        a, b = (w, l) if w < l else (l, w)
        p_a_wins = pw_lookup.get((season, a, b))
        if p_a_wins is None:
            continue
        # Map back to (W, L) perspective.
        p_w = p_a_wins if a == w else (1.0 - p_a_wins)
        seed_w = seed_lookup.get((season, w))
        seed_l = seed_lookup.get((season, l))
        if seed_w is None or seed_l is None:
            continue

        # Upset flag (per-game; same value for both symmetric rows): True
        # iff the higher-seeded team lost. Same-seed games: False.
        # Lower seed_int = better seed (1 is the top seed).
        if seed_w == seed_l:
            is_upset = False
        else:
            is_upset = seed_w > seed_l  # winner had a worse seed than loser

        # Symmetric pair: A=W (label=1), then A=L (label=0).
        rows.append({
            "season": season, "team_a": w, "team_b": l,
            "p_stage1": p_w,
            "seed_a": seed_w, "seed_b": seed_l,
            "abs_seed_diff": abs(seed_w - seed_l),
            "upset": is_upset,
            "label": 1,
        })
        rows.append({
            "season": season, "team_a": l, "team_b": w,
            "p_stage1": 1.0 - p_w,
            "seed_a": seed_l, "seed_b": seed_w,
            "abs_seed_diff": abs(seed_w - seed_l),
            "upset": is_upset,
            "label": 0,
        })
    return pd.DataFrame(rows)
```

- [ ] **Step 3: Smoke-test the file imports clean.**

Run: `python -c "import src.train_upset_model"`
Expected: no output, exit code 0.

- [ ] **Step 4: Commit.**

```bash
git add src/train_upset_model.py
git commit -m "feat(v9): skeleton + per-game loader with upset flag"
```

---

### Task 2: Unit tests for `load_per_game_data_with_upset`

**Files:**
- Create: `tests/test_upset_model.py`

- [ ] **Step 1: Create `tests/test_upset_model.py` with imports and the upset-flag tests.**

Write the full file:

```python
"""Tests for src/train_upset_model.py (v9 upset-aware stage-2)."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.train_upset_model import (
    W_MISS,
    W_UPSET,
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
```

- [ ] **Step 2: Run the new tests, expect PASS.**

Run: `python -m pytest tests/test_upset_model.py -v`
Expected: 6 passed (`test_parse_seed_numeric`, `test_parse_seed_invalid`, `test_loader_flags_5_over_12_as_upset`, `test_loader_flags_1_beats_16_as_non_upset`, `test_loader_same_seed_is_non_upset`, `test_loader_produces_symmetric_rows`).

- [ ] **Step 3: Commit.**

```bash
git add tests/test_upset_model.py
git commit -m "test(v9): per-game loader upset-flag and symmetry tests"
```

---

## Phase 2: Sample weights, model fit, and double-LOSO eval

### Task 3: `compute_sample_weights` (TDD)

**Files:**
- Modify: `tests/test_upset_model.py` (append new test class)
- Modify: `src/train_upset_model.py` (append function)

- [ ] **Step 1: Append failing tests for `compute_sample_weights` to `tests/test_upset_model.py`.**

Append to the file (after the existing tests):

```python
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
```

- [ ] **Step 2: Run the new tests, expect FAIL with `ImportError` on `compute_sample_weights`.**

Run: `python -m pytest tests/test_upset_model.py -v`
Expected: 6 of the new tests fail at import time. Existing 6 tests still pass.

- [ ] **Step 3: Implement `compute_sample_weights` in `src/train_upset_model.py`.**

Append to `src/train_upset_model.py`:

```python
def compute_sample_weights(
    df: pd.DataFrame, w_upset: float = W_UPSET, w_miss: float = W_MISS
) -> np.ndarray:
    """Per-row training weight for v9.

    For each row:
        w = 1.0
        if upset: w *= w_upset                        # upset multiplier
        w *= 1 + w_miss * (label - p_stage1) ** 2     # miss multiplier
                                                      # residual is per-perspective

    Returns: np.ndarray of length len(df), aligned with df rows.
    """
    base = np.ones(len(df), dtype=float)
    upset_factor = np.where(df["upset"].values, w_upset, 1.0)
    residual = df["label"].values.astype(float) - df["p_stage1"].values.astype(float)
    miss_factor = 1.0 + w_miss * (residual ** 2)
    return base * upset_factor * miss_factor
```

- [ ] **Step 4: Re-run the tests, expect all PASS.**

Run: `python -m pytest tests/test_upset_model.py -v`
Expected: 12 passed (6 existing + 6 new).

- [ ] **Step 5: Commit.**

```bash
git add src/train_upset_model.py tests/test_upset_model.py
git commit -m "feat(v9): compute_sample_weights for upset-aware training"
```

---

### Task 4: `upset_features` + `fit_upset_model`

**Files:**
- Modify: `src/train_upset_model.py` (append two functions)
- Modify: `tests/test_upset_model.py` (append small fit smoke test)

- [ ] **Step 1: Append a failing fit-smoke test to `tests/test_upset_model.py`.**

```python
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
```

- [ ] **Step 2: Run, expect FAIL on import of `fit_upset_model` / `upset_features`.**

Run: `python -m pytest tests/test_upset_model.py -v`
Expected: 2 new tests fail at import time. 12 existing pass.

- [ ] **Step 3: Implement both functions in `src/train_upset_model.py`.**

Append to `src/train_upset_model.py`:

```python
def upset_features(df: pd.DataFrame) -> np.ndarray:
    """Pull the v9 input matrix from a per-game DataFrame.

    Same 4 features as v8 (src/train_stage2.py:stage2_features). The
    differentiator from v8 is the sample weight, not the feature set.
    """
    return df[["p_stage1", "seed_a", "seed_b", "abs_seed_diff"]].values


def fit_upset_model(
    X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray, seed: int = 42
) -> xgb.XGBClassifier:
    """Small XGBoost trained with upset-aware sample weights.

    Same shape as v8's fit_stage2 (src/train_stage2.py): n_estimators=100,
    max_depth=3, lr=0.05. Adds sample_weight at fit time. Capacity stays
    low (~3000 training rows in the full backtest).
    """
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=1.0,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=seed,
        eval_metric="logloss",
    )
    model.fit(X, y, sample_weight=sample_weight)
    return model
```

- [ ] **Step 4: Re-run tests, expect all PASS.**

Run: `python -m pytest tests/test_upset_model.py -v`
Expected: 14 passed.

- [ ] **Step 5: Commit.**

```bash
git add src/train_upset_model.py tests/test_upset_model.py
git commit -m "feat(v9): upset_features extractor and fit_upset_model"
```

---

### Task 5: `double_loso_eval` + leakage guard test

**Files:**
- Modify: `src/train_upset_model.py` (append function)
- Modify: `tests/test_upset_model.py` (append leakage test)

- [ ] **Step 1: Append a failing leakage-guard test.**

Append to `tests/test_upset_model.py`:

```python
# -----------------------------------------------------------------------------
# double_loso_eval: leakage guard
# -----------------------------------------------------------------------------

from src.train_upset_model import double_loso_eval


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
            "abs_seed_diff": 7, "upset": True, "label": 1,
        })
        rows.append({
            "season": season, "team_a": 2, "team_b": 1,
            "p_stage1": 0.3, "seed_a": 12, "seed_b": 5,
            "abs_seed_diff": 7, "upset": True, "label": 0,
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
```

- [ ] **Step 2: Run, expect FAIL on import of `double_loso_eval`.**

Run: `python -m pytest tests/test_upset_model.py::test_double_loso_eval_never_trains_on_test_season -v`
Expected: ImportError.

- [ ] **Step 3: Implement `double_loso_eval` in `src/train_upset_model.py`.**

Append to `src/train_upset_model.py`:

```python
def double_loso_eval(per_game: pd.DataFrame) -> pd.DataFrame:
    """For each test season, train v9 on all-other-seasons and evaluate on it.

    Mirrors src/train_stage2.py:double_loso_eval. Differences:
    - Uses upset_features / fit_upset_model / compute_sample_weights.
    - Reports v8-style metrics (log loss, accuracy) but for v9.

    Returns DataFrame with per-season metrics:
        season, n_games, ll_v9, acc_v9.

    Per-row stage-1 (v4) predictions remain available in per_game; the
    caller is responsible for joining v4 / v8 numbers if it wants the
    full head-to-head table.
    """
    seasons = sorted(per_game["season"].unique())
    results = []

    for test_season in seasons:
        train = per_game[per_game.season != test_season]
        test = per_game[per_game.season == test_season]
        if len(train) == 0 or len(test) == 0:
            continue

        X_train = upset_features(train)
        y_train = train["label"].values
        w_train = compute_sample_weights(train)
        X_test = upset_features(test)
        y_test = test["label"].values

        model = fit_upset_model(X_train, y_train, w_train)
        p_v9 = model.predict_proba(X_test)[:, 1]

        # Keep only one row per game (winner perspective) for clean reporting,
        # matching v8's convention.
        is_winner = test["label"].values == 1
        if is_winner.sum() == 0:
            continue
        ll_v9 = sklearn_log_loss(y_test[is_winner], p_v9[is_winner], labels=[0, 1])
        acc_v9 = float((p_v9[is_winner] > 0.5).mean())

        results.append({
            "season": test_season,
            "n_games": int(is_winner.sum()),
            "ll_v9": ll_v9, "acc_v9": acc_v9,
        })

    return pd.DataFrame(results).sort_values("season").reset_index(drop=True)
```

- [ ] **Step 4: Re-run all upset-model tests, expect all PASS.**

Run: `python -m pytest tests/test_upset_model.py -v`
Expected: 15 passed.

- [ ] **Step 5: Commit.**

```bash
git add src/train_upset_model.py tests/test_upset_model.py
git commit -m "feat(v9): double_loso_eval with leakage guard test"
```

---

## Phase 3: Pairwise output writer + main

### Task 6: `build_v9_pairwise` (writes the per-pair CSV downstream tools consume)

**Files:**
- Modify: `src/train_upset_model.py` (append function)
- Modify: `tests/test_upset_model.py` (append output-shape test)

- [ ] **Step 1: Append a failing output-shape test.**

Append to `tests/test_upset_model.py`:

```python
# -----------------------------------------------------------------------------
# build_v9_pairwise: writes output/pairwise_v9.csv with v9-adjusted probs
# -----------------------------------------------------------------------------

from src.train_upset_model import build_v9_pairwise


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
         "upset": False, "label": 1},
        {"season": 2022, "team_a": 2, "team_b": 1, "p_stage1": 0.3,
         "seed_a": 8, "seed_b": 1, "abs_seed_diff": 7,
         "upset": False, "label": 0},
        {"season": 2023, "team_a": 2, "team_b": 3, "p_stage1": 0.45,
         "seed_a": 8, "seed_b": 16, "abs_seed_diff": 8,
         "upset": True, "label": 1},
        {"season": 2023, "team_a": 3, "team_b": 2, "p_stage1": 0.55,
         "seed_a": 16, "seed_b": 8, "abs_seed_diff": 8,
         "upset": True, "label": 0},
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
```

- [ ] **Step 2: Run, expect FAIL on `build_v9_pairwise` import.**

Run: `python -m pytest tests/test_upset_model.py::test_build_v9_pairwise_writes_expected_schema -v`
Expected: ImportError.

- [ ] **Step 3: Implement `build_v9_pairwise`.**

Append to `src/train_upset_model.py`:

```python
def build_v9_pairwise(
    per_game: pd.DataFrame,
    pairwise_v4_csv: str,
    seeds_csv: str,
    out_path: str,
) -> None:
    """For each LOSO season, train v9 on other-seasons' per-game rows and
    apply to every pair in that season's pairwise_v4.csv. Writes a CSV in
    v8-compatible schema (season, team_a, team_b, p_a_wins) with team_a <
    team_b on every row.

    Mirrors src/train_stage2.py:build_v8_pairwise. Differences: feeds
    sample weights to fit_upset_model, no other functional change.
    """
    pw = pd.read_csv(pairwise_v4_csv).drop_duplicates(
        ["season", "team_a", "team_b"], keep="last"
    )
    seeds = pd.read_csv(seeds_csv)
    seeds["seed_int"] = seeds["Seed"].apply(parse_seed)
    seed_lookup = {(int(r["Season"]), int(r["TeamID"])): r["seed_int"]
                   for _, r in seeds.iterrows() if r["seed_int"] is not None}

    out_rows = []
    for season in sorted(pw.season.unique()):
        train = per_game[per_game.season != season]
        if len(train) == 0:
            # Pass-through stage-1 if we have no other-season training data.
            for _, r in pw[pw.season == season].iterrows():
                out_rows.append({
                    "season": int(season), "team_a": int(r.team_a),
                    "team_b": int(r.team_b), "p_a_wins": float(r.p_a_wins),
                })
            continue

        X_train = upset_features(train)
        y_train = train["label"].values
        w_train = compute_sample_weights(train)
        model = fit_upset_model(X_train, y_train, w_train)

        season_pw = pw[pw.season == season].copy()
        feat_rows = []
        keep = []
        for _, r in season_pw.iterrows():
            seed_a = seed_lookup.get((int(r["season"]), int(r["team_a"])))
            seed_b = seed_lookup.get((int(r["season"]), int(r["team_b"])))
            if seed_a is None or seed_b is None:
                keep.append(False)
                continue
            feat_rows.append([float(r["p_a_wins"]), seed_a, seed_b,
                              abs(seed_a - seed_b)])
            keep.append(True)

        if not feat_rows:
            for _, r in season_pw.iterrows():
                out_rows.append({
                    "season": int(season), "team_a": int(r.team_a),
                    "team_b": int(r.team_b), "p_a_wins": float(r.p_a_wins),
                })
            continue

        X = np.array(feat_rows)
        p_v9 = model.predict_proba(X)[:, 1]

        i = 0
        for (_, r), keep_row in zip(season_pw.iterrows(), keep):
            if keep_row:
                p = float(p_v9[i])
                i += 1
            else:
                p = float(r["p_a_wins"])
            out_rows.append({
                "season": int(season), "team_a": int(r.team_a),
                "team_b": int(r.team_b), "p_a_wins": p,
            })

    pd.DataFrame(out_rows).to_csv(out_path, index=False)
```

- [ ] **Step 4: Re-run, expect all PASS.**

Run: `python -m pytest tests/test_upset_model.py -v`
Expected: 16 passed.

- [ ] **Step 5: Commit.**

```bash
git add src/train_upset_model.py tests/test_upset_model.py
git commit -m "feat(v9): build_v9_pairwise writes v8-compatible CSV"
```

---

### Task 7: `main()` -- orchestrate train + eval + write outputs

**Files:**
- Modify: `src/train_upset_model.py` (append `main` and `__main__` guard)

- [ ] **Step 1: Append `main()` and the script entrypoint to `src/train_upset_model.py`.**

```python
def main():
    print("=" * 80)
    print("V9 TRAINING (upset-aware stage-2 on v4 OOF predictions)")
    print(f"  W_UPSET={W_UPSET}, W_MISS={W_MISS}")
    print("=" * 80)

    pairwise_v4 = "output/pairwise_v4.csv"
    pairwise_v8 = "output/pairwise_v8.csv"
    seeds_csv = str(DATA / "MNCAATourneySeeds.csv")
    results_csv = str(DATA / "MNCAATourneyCompactResults.csv")

    per_game = load_per_game_data_with_upset(pairwise_v4, results_csv, seeds_csv)
    print(f"  Per-game training rows: {len(per_game):,} "
          f"(across {per_game.season.nunique()} seasons; "
          f"{int(per_game['upset'].sum() / 2)} upset games)")

    # Per-season log loss / accuracy -- v9 alone, v4 and v8 joined for context.
    eval_v9 = double_loso_eval(per_game)

    # v4 / v8 per-season stats from the same per_game frame for v4 (p_stage1)
    # plus pairwise_v8.csv joined back for v8.
    pw_v4 = per_game[per_game.label == 1].copy()
    v4_stats = (
        pw_v4.groupby("season")
        .apply(lambda g: pd.Series({
            "n": len(g),
            "ll_v4": sklearn_log_loss(g["label"].values, g["p_stage1"].values,
                                      labels=[0, 1]),
            "acc_v4": float((g["p_stage1"].values > 0.5).mean()),
        }))
        .reset_index()
    )

    if Path(pairwise_v8).exists():
        v8_pw = pd.read_csv(pairwise_v8).drop_duplicates(
            ["season", "team_a", "team_b"], keep="last"
        )
        v8_lookup = {(int(s), int(a), int(b)): float(p)
                     for s, a, b, p in zip(v8_pw.season, v8_pw.team_a,
                                           v8_pw.team_b, v8_pw.p_a_wins)}
        v8_per_season = []
        for season, g in pw_v4.groupby("season"):
            ps = []
            for _, row in g.iterrows():
                a, b = int(row.team_a), int(row.team_b)
                if a < b:
                    p = v8_lookup.get((int(season), a, b))
                else:
                    raw = v8_lookup.get((int(season), b, a))
                    p = (1.0 - raw) if raw is not None else None
                if p is None:
                    p = float(row.p_stage1)  # pass-through
                ps.append(p)
            ps_arr = np.array(ps)
            v8_per_season.append({
                "season": int(season),
                "ll_v8": sklearn_log_loss(g["label"].values, ps_arr, labels=[0, 1]),
                "acc_v8": float((ps_arr > 0.5).mean()),
            })
        v8_stats = pd.DataFrame(v8_per_season)
    else:
        print("  (output/pairwise_v8.csv not found; skipping v8 column.)")
        v8_stats = pd.DataFrame(columns=["season", "ll_v8", "acc_v8"])

    merged = (
        v4_stats.merge(v8_stats, on="season", how="left")
        .merge(eval_v9, on="season", how="left")
    )

    print(f"\n{'Season':>6}  {'N':>3}  {'LL_v4':>6}  {'LL_v8':>6}  {'LL_v9':>6}  "
          f"{'Acc_v4':>6}  {'Acc_v8':>6}  {'Acc_v9':>6}")
    print("-" * 72)
    for _, r in merged.iterrows():
        ll_v8 = f"{r.ll_v8:>6.3f}" if not pd.isna(r.get('ll_v8', np.nan)) else "   --"
        acc_v8 = f"{r.acc_v8 * 100:>5.1f}%" if not pd.isna(r.get('acc_v8', np.nan)) else "   --"
        print(f"  {int(r.season):>4}  {int(r.n):>3}  "
              f"{r.ll_v4:>6.3f}  {ll_v8}  {r.ll_v9:>6.3f}  "
              f"{r.acc_v4 * 100:>5.1f}%  {acc_v8}  {r.acc_v9 * 100:>5.1f}%")
    print("-" * 72)
    n_total = merged["n"].sum()
    mean_ll_v4 = (merged["ll_v4"] * merged["n"]).sum() / n_total
    mean_ll_v9 = (merged["ll_v9"] * merged["n"]).sum() / n_total
    mean_acc_v4 = (merged["acc_v4"] * merged["n"]).sum() / n_total
    mean_acc_v9 = (merged["acc_v9"] * merged["n"]).sum() / n_total
    if "ll_v8" in merged.columns and merged["ll_v8"].notna().any():
        mean_ll_v8 = (merged["ll_v8"] * merged["n"]).sum() / n_total
        mean_acc_v8 = (merged["acc_v8"] * merged["n"]).sum() / n_total
        ll_v8_str = f"{mean_ll_v8:>6.3f}"
        acc_v8_str = f"{mean_acc_v8 * 100:>5.1f}%"
    else:
        ll_v8_str = "   --"
        acc_v8_str = "   --"
    print(f"  {'WT MEAN':>6}        "
          f"{mean_ll_v4:>6.3f}  {ll_v8_str}  {mean_ll_v9:>6.3f}  "
          f"{mean_acc_v4 * 100:>5.1f}%  {acc_v8_str}  {mean_acc_v9 * 100:>5.1f}%")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    eval_path = OUTPUT / "v9_eval.csv"
    merged.to_csv(eval_path, index=False)
    print(f"\nWrote per-season eval to {eval_path}")

    pairwise_v9 = OUTPUT / "pairwise_v9.csv"
    print(f"Writing v9-adjusted pairwise to {pairwise_v9} ...")
    build_v9_pairwise(per_game, pairwise_v4, seeds_csv, str(pairwise_v9))
    print("  Done.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-test that the file is still importable and `main` is callable.**

Run: `python -c "from src.train_upset_model import main; print(main)"`
Expected: prints the function repr; no errors.

- [ ] **Step 3: Commit.**

```bash
git add src/train_upset_model.py
git commit -m "feat(v9): main() ties together training, eval, and pairwise output"
```

---

## Phase 4: Score the head-to-head against v8

### Task 8: Run training + add v9 to the bracket scorer + capture results

**Files:**
- Modify: `src/score_chalk_brackets.py:222` (add "v9" to `versions`)

- [ ] **Step 1: Run the v9 trainer end-to-end.**

Run: `python src/train_upset_model.py`
Expected: prints the per-season table for ~22 seasons with v4/v8/v9 columns; writes `output/v9_eval.csv` and `output/pairwise_v9.csv`.

If the trainer crashes on any season, stop and diagnose -- do not paper over by skipping. Report the failure to the user.

- [ ] **Step 2: Verify the output CSV has the expected shape.**

Run:
```
python -c "import pandas as pd; df = pd.read_csv('output/pairwise_v9.csv'); print(df.shape, sorted(df.season.unique())); print(df.head())"
```
Expected: shape roughly `(N, 4)` with N matching `pairwise_v8.csv`'s row count, columns `season, team_a, team_b, p_a_wins`, seasons matching `SEASONS_TO_BACKTEST` minus 2020.

- [ ] **Step 3: Add "v9" to the `versions` list in `src/score_chalk_brackets.py`.**

In `src/score_chalk_brackets.py`, change:

```python
    versions = ["v1", "v2", "v4", "v8"]
```

to:

```python
    versions = ["v1", "v2", "v4", "v8", "v9"]
```

(Single-line edit on what is currently line 222 of that file.)

- [ ] **Step 4: Run the scorer.**

Run: `python src/score_chalk_brackets.py`
Expected output ends with TOTAL and MEAN lines that include a `v9` column. Capture:
- 22-season TOTAL bracket points: v4 / v8 / v9.
- 22-season MEAN per season: v4 / v8 / v9.
- Wins per version (which model picked the best chalk bracket in how many seasons).
- Mean accuracy per round (for the per-round upset-recall sanity check the spec calls for).

Save the printed table to a temp file: `python src/score_chalk_brackets.py > /tmp/v9_scorecard.txt 2>&1`

- [ ] **Step 5: Commit the scorer change and the generated outputs.**

```bash
git add src/score_chalk_brackets.py output/pairwise_v9.csv output/v9_eval.csv
git commit -m "feat(v9): score v9 head-to-head against v8 in chalk bracket"
```

(`output/` is in `.gitignore` but several CSV artifacts under it are already tracked from earlier commits -- e.g., `pairwise_v4.csv`, `pairwise_v8.csv`. New files in `output/` are subject to the gitignore rule, so use `git add -f output/pairwise_v9.csv output/v9_eval.csv` if `git add` silently skips them. Verify with `git status -- output/pairwise_v9.csv` before committing.)

- [ ] **Step 6: Decide the verdict.**

Apply the success-criteria table from the spec:

| v9 vs v8 (TOTAL bracket pts, summed over 22 LOSO seasons)  | LOSO log loss check       | Decision                                                |
|------------------------------------------------------------|---------------------------|---------------------------------------------------------|
| `total_v9 - total_v8 >= +3`                                | `mean_ll_v9 <= mean_ll_v8`| **WIN.** Skip Task 9 (no fallback). Proceed to Task 10. |
| `-3 < total_v9 - total_v8 < +3`                            | --                        | **TIE.** Run Task 9 (v9-B fallback).                    |
| `total_v9 - total_v8 <= -3`                                | --                        | **LOSE.** Run Task 9 (v9-B fallback).                   |

Record the verdict (WIN / TIE / LOSE) in conversation; the writeup in Task 10 documents it.

---

## Phase 5: Conditional fallback + writeup

### Task 9 [conditional -- run only if Task 8 verdict is TIE or LOSE]: v9-B feature-extended fallback

**Files:**
- Modify: `src/train_upset_model.py` (add 3 features to `upset_features` and `build_v9_pairwise`)
- Modify: `tests/test_upset_model.py` (update the 4-column shape assertion in `test_upset_features_extracts_four_columns`; rename to reflect 7 cols)

- [ ] **Step 1: If Task 8's verdict was WIN, skip to Task 10. Otherwise, continue.**

- [ ] **Step 2: Extend `upset_features` and `build_v9_pairwise` to take 7 features.**

In `src/train_upset_model.py`, change `upset_features` to:

```python
def upset_features(df: pd.DataFrame) -> np.ndarray:
    """Pull the v9-B input matrix from a per-game DataFrame.

    7 features:
      p_stage1, seed_a, seed_b, abs_seed_diff,
      round (1..6 for R64..Champ; 0 if unknown DayNum),
      v4_confidence (|p_stage1 - 0.5|),
      is_a_higher_seed (1 if seed_a < seed_b else 0).
    """
    p = df["p_stage1"].values.astype(float)
    sa = df["seed_a"].values.astype(float)
    sb = df["seed_b"].values.astype(float)
    diff = df["abs_seed_diff"].values.astype(float)
    rnd = df["round"].values.astype(float)
    conf = np.abs(p - 0.5)
    higher = (sa < sb).astype(float)
    return np.column_stack([p, sa, sb, diff, rnd, conf, higher])
```

The per-game DataFrame already needs a `round` column, so update
`load_per_game_data_with_upset` to add it. Locate the row builder
inside that function and add `"round": day_to_round(int(g["DayNum"]))`
(import `day_to_round` from `src.models.matchup`) to both the
`a=W,label=1` and `a=L,label=0` row dicts.

Add at the top of `src/train_upset_model.py`:

```python
from src.models.matchup import day_to_round
```

In `build_v9_pairwise`, the per-pair feature build also needs the
extra columns. The pairwise CSV has no DayNum, so the round must be
inferred from the pair *and the bracket structure*. For now, set
`round = 0` for the pairwise application (the trainer learned a
round-conditioned correction; at apply time, we apply the
"round-unknown" branch). This is a known limitation of v9-B and is
called out explicitly in the writeup. **Mention this trade-off to
the user when reporting the v9-B verdict.**

In `build_v9_pairwise`, change the feature row construction inside
the `season_pw.iterrows()` loop:

```python
            feat_rows.append([
                float(r["p_a_wins"]),
                seed_a, seed_b,
                abs(seed_a - seed_b),
                0.0,                                    # round unknown at apply time
                abs(float(r["p_a_wins"]) - 0.5),        # v4 confidence
                1.0 if seed_a < seed_b else 0.0,        # is_a_higher_seed
            ])
```

- [ ] **Step 3: Update tests for the schema change.**

Two specific edits in `tests/test_upset_model.py`:

(a) Change `test_upset_features_extracts_four_columns` (Task 4) to:

```python
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
```

(b) `test_fit_upset_model_returns_classifier_with_predict_proba` (Task 4) builds a synthetic df via inline column construction and then calls `upset_features(df)`. Add a `"round": np.random.randint(1, 7, n)` column to that DataFrame. No other tests need updating: `compute_sample_weights` does not read `round`, and the leakage test in Task 5 passes per-game rows directly without going through `upset_features`.

Add `"round": 1` to the per-game rows in the leakage test (Task 5) only if you also rebuilt the loader's emitted schema in this task -- the leakage test rows do not feed `upset_features` so are otherwise unaffected. Strict consistency is cleaner; pick a single approach and apply it.

- [ ] **Step 4: Run all upset-model tests, expect all PASS.**

Run: `python -m pytest tests/test_upset_model.py -v`
Expected: all green.

- [ ] **Step 5: Re-run the trainer + scorer.**

```bash
python src/train_upset_model.py
python src/score_chalk_brackets.py > /tmp/v9b_scorecard.txt 2>&1
```

Compare v9-B's totals to v8 using the same success-criteria table.

- [ ] **Step 6: Commit.**

```bash
git add src/train_upset_model.py tests/test_upset_model.py output/pairwise_v9.csv output/v9_eval.csv
git commit -m "feat(v9-B): feature-extended fallback (round, conf, is_higher_seed)"
```

---

### Task 10: Writeup + TODO update

**Files:**
- Create: `docs/notes/2026-04-30-upset-detection-v9.md`
- Modify: `TODO.md`

- [ ] **Step 1: Run the full FORCED VERIFICATION pytest gate from CLAUDE.md.**

```bash
python -m pytest -v
```

All tests must pass. Capture output for the writeup.

- [ ] **Step 2: Create `docs/notes/2026-04-30-upset-detection-v9.md`.**

Use the verdict from Task 8 (and Task 9 if it ran) to fill in the writeup. Required sections:

```markdown
# Upset-Detection Sub-Model (v9) -- Findings

## Summary

[One sentence: WIN / TIE / LOSE, with the bracket-points delta.]

## Numbers

| Model | 22-season TOTAL bracket pts | 22-season MEAN per season | Weighted-mean LOSO log loss |
|-------|-----------------------------|---------------------------|-----------------------------|
| v4    | <fill in>                   | <fill in>                 | <fill in>                   |
| v8    | <fill in>                   | <fill in>                 | <fill in>                   |
| v9-A  | <fill in>                   | <fill in>                 | <fill in>                   |
| v9-B  | <fill in if ran>            | <fill in if ran>          | <fill in if ran>            |

## Per-round upset accuracy (sanity check)

[paste the "MEAN ACCURACY PER ROUND" table from the scorer, restricted to v8 and v9 columns]

## Verdict

[Replace v8 / Tie / Lose. If TIE or LOSE on v9-A, state whether v9-B was run and what its outcome was.]

## What this means for next steps

[If WIN: promote per-round upset specialists (Q1 option B from brainstorm) to active queue position #1.
 If TIE or LOSE after fallback: keep v8 in production, shelve upset-direction. Document any signal worth chasing later.]

## Files / commits

- Spec: docs/superpowers/specs/2026-04-30-upset-detection-design.md
- Plan: docs/superpowers/plans/2026-04-30-upset-detection.md
- Code: src/train_upset_model.py
- Outputs: output/pairwise_v9.csv, output/v9_eval.csv

## Implementation notes / caveats

[If v9-B ran: note the round=0 limitation in build_v9_pairwise -- the
training rows are round-conditioned (round in 1..6) but apply-time uses
round=0 for all pairs. Future work to fix this would require resolving
each (team_a, team_b) pair to its round in the bracket structure.]
```

ASCII-only. Verify with: `python -c "open('docs/notes/2026-04-30-upset-detection-v9.md', encoding='utf-8').read().encode('ascii')"`

- [ ] **Step 3: Update `TODO.md`.**

Read the current `TODO.md`. Apply edits based on verdict.

If WIN:
- Move "Upset-detection sub-model" item from Active queue position #1 to Done.
- Promote "Per-round upset specialists" to new Active queue position #1 with a one-paragraph description (as the v9 result motivates it).
- Add "v9 (upset-aware stage-2)" to the model-evolution log in `README.md` if there is one.

If TIE or LOSE:
- Move "Upset-detection sub-model" item to Done with the TIE/LOSE outcome and the v9-B caveat (round=0 at apply time).
- Promote item #2 (Ensemble of model classes) to position #1.

In both cases:
- Add a one-line entry under Done with date 2026-04-30 and a pointer to `docs/notes/2026-04-30-upset-detection-v9.md`.

- [ ] **Step 4: Verify ASCII for both writeup and TODO.**

```bash
python -c "open('docs/notes/2026-04-30-upset-detection-v9.md', encoding='utf-8').read().encode('ascii')"
python -c "open('TODO.md', encoding='utf-8').read().encode('ascii')"
```

Both should exit 0 with no output.

- [ ] **Step 5: Commit.**

```bash
git add docs/notes/2026-04-30-upset-detection-v9.md TODO.md
git commit -m "docs(v9): findings writeup + TODO update with verdict"
```

- [ ] **Step 6: Push and open PR (only if requested by the user).**

The user may want this on its own PR, or bundled with the test fixes (commit `c10c43a`) or the gitignore housekeeping (commit `a82448d`). Ask before pushing.

---

## Final verification (CLAUDE.md FORCED VERIFICATION)

Per `CLAUDE.md` Section 4, before reporting this work complete:

- [ ] `python -m pytest -v` -- all tests pass.
- [ ] `python src/train_upset_model.py` runs end-to-end without error and prints a per-season table.
- [ ] `output/pairwise_v9.csv` and `output/v9_eval.csv` exist and have the expected schema.
- [ ] `python src/score_chalk_brackets.py` includes a v9 column in its output.
- [ ] State the actual numbers in the final summary message:
  - 22-season TOTAL bracket pts: v4 = ___, v8 = ___, v9 = ___
  - 22-season MEAN log loss: v4 = ___, v8 = ___, v9 = ___
  - Verdict: WIN / TIE / LOSE
  - If v9-B ran: same numbers for v9-B and its verdict.

A silent regression from v8's +9 pts to v9's -5 pts is exactly the failure mode CLAUDE.md is warning about. State the numbers explicitly. Do not say "checks pass."
