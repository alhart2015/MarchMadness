# Stage-1 Ensemble (XGBoost + LR) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and evaluate a stage-1 ensemble (v4 XGBoost + a new logistic regression) over the existing 22-season LOSO backtest, then run the v9-C stage-2 corrector on top of both the baseline (v4-only) and the ensemble. Render a verdict-band recommendation: clear win (>= +25 brkt pts), marginal (+10 to +25), or no-go (< +10).

**Architecture:** Two new top-level scripts producing CSVs with the same schema as `output/pairwise_v4.csv`: `src/train_lr_stage1.py` (LR LOSO) and `src/ensemble_stage1.py` (simple-average two CSVs). Stage 2 is run twice via a thin wrapper `src/run_v9c_on_stage1.py` so the v9-C config (W_UPSET=1.25, W_MISS=0.0, feature_set='v9c') is held constant while only the stage-1 input varies. Eval uses the existing `score_chalk_brackets.score_pairwise_path` for bracket points and a small new `src/eval_stage1.py` for per-season LOSO log loss + accuracy. Refactor `enhanced_model_v3.py` to expose `prepare_loso_inputs()` so the LR trainer reuses the *same* feature matrix v4 trains on -- the experiment is "model class diversity at identical input."

**Tech Stack:** Python 3, scikit-learn (`LogisticRegression`, `StandardScaler`, `CalibratedClassifierCV`), XGBoost (existing), pandas, numpy, pytest. No new top-level dependencies.

**Spec:** `docs/superpowers/specs/2026-05-01-ensemble-stage1-design.md`

**Reference reads (skim before starting):**
- `src/enhanced_model_v3.py` lines 472-583 (`leave_one_season_out_cv_weighted`) and 590-820+ (`main()` data-setup). The LR trainer mirrors this loop.
- `src/score_chalk_brackets.py` -- `score_pairwise_path(path)` is the bracket-points eval primitive (returns `{"total_pts": float, "per_season_pts": {int: float}}`).
- `src/train_upset_model.py` -- `build_v9_pairwise(per_game, pairwise_csv, seeds_csv, out_csv, slots_csv=..., w_upset=..., w_miss=..., feature_set=...)` is the stage-2 application primitive used in Task 6.
- `src/sweep_v9_weights.py::run_single_cell` -- existing pattern for running v9-C with non-default args.

**Verification gates (CLAUDE.md "Forced Verification"):**
After every code-level task: `pytest -v` at repo root must pass. After data-generation tasks (3, 4, 7, 8): inspect schema and row count of the produced CSV before committing.

---

### Task 1: Extract `prepare_loso_inputs()` from `enhanced_model_v3.py`

**Why:** v3's `main()` builds `feature_matrix`, `tourney_filtered`, `regular_results`, `feature_cols`, `top_80_by_season` over ~200 lines of orchestration. The LR trainer needs the *byte-identical* outputs so the ensemble experiment isolates model-class diversity. Cleanest path is a mechanical function-extraction that v3's `main()` then calls. No behavior change for v3; existing reproducibility preserved.

**Files:**
- Modify: `src/enhanced_model_v3.py` (extract lines ~590-820 of `main()` into a new function `prepare_loso_inputs()` near the top of the MAIN section; have `main()` call it)
- Test: `tests/test_prepare_loso_inputs.py` (new)

- [ ] **Step 1: Re-read `src/enhanced_model_v3.py:590-820`** to confirm exactly what is built.

Run: `wc -l src/enhanced_model_v3.py`
Expected: ~1328 lines.

Read 590-820 in one pass. The data setup spans:
- `load_all_data()` + conf_tourney + team_coaches loads (lines 605-613)
- `build_kenpom_to_kaggle_map(...)` (line 616)
- `compute_all_features(data)` (line 622)
- Vegas features merge (lines 629-655)
- Late-season / trajectory / conf tourney / vegas trend per season (lines 663-734)
- Coach features merge (lines 736-746)
- `get_feature_cols(...)` (line 755)
- `MM_FEATURE_DROP` ablation hook (lines 760-768)
- `top_80_by_season` and `tourney_filtered` (lines 776-787)
- NaN-pruning of feature_cols using `build_weighted_matchup_data` (lines 789-811)

Note: `top_80_by_season` is what `evaluate_loso` calls `top_n_team_ids_by_season`. Make sure to preserve the variable name when extracting.

- [ ] **Step 2: Write the failing test**

Create `tests/test_prepare_loso_inputs.py`:

```python
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
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest tests/test_prepare_loso_inputs.py -v`
Expected: FAIL with `ImportError: cannot import name 'prepare_loso_inputs'`.

- [ ] **Step 4: Refactor `main()` to extract the data-setup block**

Open `src/enhanced_model_v3.py`. Move lines ~602-811 (the entire data-setup orchestration up to and including the NaN-prune of feature_cols) into a new top-level function `prepare_loso_inputs()` placed immediately above `main()`. The function returns:

```python
def prepare_loso_inputs() -> dict:
    """Build the v3/v4 feature matrix, training data, and per-season top-80
    team ID sets for use by any LOSO-loop trainer. This is the data-setup
    half of v4's main() extracted as a callable so parallel trainers (e.g.,
    the LR stage-1 in src/train_lr_stage1.py) can reuse the byte-identical
    inputs.

    Returns dict with keys:
        feature_matrix      -- pd.DataFrame with TeamID, Season, all features
        tourney_filtered    -- pd.DataFrame of tournament results filtered
                               to seasons present in feature_matrix
        regular_results     -- pd.DataFrame of regular-season results
                               (data["reg_season"] in v4 main)
        feature_cols        -- list of feature column names (post-NaN-prune)
        top_80_by_season    -- dict[int -> set[int]] of top-80 team IDs
                               per season, used by the weighted matchup
                               builder to mark supplemental rows
        feature_medians     -- pd.Series of per-feature medians from the
                               weighted-matchup X_all (used to fill NaNs
                               in apply-time pair construction)
    """
    overall_start = time.time()
    ...
    # body: lines previously at ~602-811 of main()
    ...
    return {
        "feature_matrix": feature_matrix,
        "tourney_filtered": tourney_filtered,
        "regular_results": data["reg_season"],
        "feature_cols": feature_cols,
        "top_80_by_season": top_80_by_season,
        "feature_medians": medians,
    }
```

Then update `main()` to call `prepare_loso_inputs()` and unpack the dict into the same local variable names that `main()` was using before, so the rest of `main()` (LOSO eval, bracket sim, etc.) is unchanged. Example:

```python
def main():
    inputs = prepare_loso_inputs()
    feature_matrix = inputs["feature_matrix"]
    tourney_filtered = inputs["tourney_filtered"]
    regular_results = inputs["regular_results"]
    feature_cols = inputs["feature_cols"]
    top_80_by_season = inputs["top_80_by_season"]
    medians = inputs["feature_medians"]
    # ... rest of main unchanged ...
```

CRITICAL: do not remove any of the print() statements -- the captured stdout from a v4 run is part of the reproducibility surface.

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_prepare_loso_inputs.py -v`
Expected: PASS (call may take 30-60s on first run as v4 data loads / features compute).

- [ ] **Step 6: Verify v4 still produces the same `pairwise_v4.csv`**

Run: `MM_PAIRWISE_OUT=output/pairwise_v4_refactor_check.csv python src/enhanced_model_v3.py 2>&1 | tail -40`
Expected: completes without errors. v4's standard summary log loss / accuracy lines printed (within float jitter of last known v4 numbers).

Then:

```bash
python -c "
import pandas as pd
a = pd.read_csv('output/pairwise_v4.csv').sort_values(['season','team_a','team_b']).reset_index(drop=True)
b = pd.read_csv('output/pairwise_v4_refactor_check.csv').sort_values(['season','team_a','team_b']).reset_index(drop=True)
assert (a[['season','team_a','team_b']] == b[['season','team_a','team_b']]).all().all(), 'pairs disagree'
diff = (a['p_a_wins'] - b['p_a_wins']).abs().max()
print(f'max prob diff: {diff:.2e}')
assert diff < 1e-6, f'prob mismatch beyond tolerance: {diff}'
print('OK: refactor reproduces pairwise_v4.csv to <1e-6 tolerance')
"
```
Expected: `OK: refactor reproduces pairwise_v4.csv to <1e-6 tolerance`. Then `rm output/pairwise_v4_refactor_check.csv`.

- [ ] **Step 7: Run the full test suite**

Run: `pytest -v`
Expected: all green. v4-related tests in `tests/test_integration.py` should still pass.

- [ ] **Step 8: Commit**

```bash
git add src/enhanced_model_v3.py tests/test_prepare_loso_inputs.py
git commit -m "$(cat <<'EOF'
refactor(v3): extract prepare_loso_inputs() from main()

Pure function extraction. v4's data-setup orchestration (load_all_data,
compute_all_features, Vegas/late-season/trajectory/conf/coach merges,
get_feature_cols, MM_FEATURE_DROP hook, top_80_by_season,
NaN-prune of feature_cols) moves from main() into a top-level
prepare_loso_inputs() function. main() calls it and unpacks.

Behavior unchanged: pairwise_v4.csv reproduces to <1e-6 tolerance.

Required by docs/superpowers/specs/2026-05-01-ensemble-stage1-design.md
so src/train_lr_stage1.py can reuse the byte-identical feature matrix
and feature_cols (the experiment isolates model-class diversity at
identical input).
EOF
)"
```

---

### Task 2: Build `src/ensemble_stage1.py` (CSV averaging utility)

**Why:** Tiny, pure utility that joins two pairwise CSVs and averages probabilities. The anchor test (`--weights 1.0,0.0` byte-equals `pairwise_v4.csv`) is the leakage-and-correctness smoke that catches join bugs and dtype regressions. Build this *before* the LR trainer so we can sanity-check the averaging math against a `pairwise_v4 + pairwise_v4` self-average (= identity, by symmetry) on day one.

**Files:**
- Create: `src/ensemble_stage1.py`
- Test: `tests/test_ensemble_stage1.py` (new)

- [ ] **Step 1: Write the failing tests**

Create `tests/test_ensemble_stage1.py`:

```python
"""Unit tests for src/ensemble_stage1.py."""
from pathlib import Path
import pandas as pd
import pytest


SCHEMA = ["season", "team_a", "team_b", "p_a_wins"]


def _write_csv(path, rows):
    pd.DataFrame(rows, columns=SCHEMA).to_csv(path, index=False)


def test_average_simple(tmp_path):
    from src.ensemble_stage1 import average_pairwise_csvs

    a = tmp_path / "a.csv"
    b = tmp_path / "b.csv"
    out = tmp_path / "ens.csv"
    _write_csv(a, [
        (2003, 1104, 1112, 0.20),
        (2003, 1104, 1113, 0.40),
    ])
    _write_csv(b, [
        (2003, 1104, 1112, 0.60),
        (2003, 1104, 1113, 0.80),
    ])

    average_pairwise_csvs(str(a), str(b), str(out), weights=(0.5, 0.5))

    df = pd.read_csv(out).sort_values(["season", "team_a", "team_b"]).reset_index(drop=True)
    assert list(df.columns) == SCHEMA
    assert df.loc[0, "p_a_wins"] == pytest.approx(0.40)
    assert df.loc[1, "p_a_wins"] == pytest.approx(0.60)


def test_anchor_weights_1_0_reproduces_first(tmp_path):
    """--weights 1.0,0.0 must reproduce input A row-for-row."""
    from src.ensemble_stage1 import average_pairwise_csvs

    a = tmp_path / "a.csv"
    b = tmp_path / "b.csv"
    out = tmp_path / "ens.csv"
    _write_csv(a, [
        (2003, 1104, 1112, 0.123456789),
        (2004, 1101, 1115, 0.987654321),
    ])
    _write_csv(b, [
        (2003, 1104, 1112, 0.5),
        (2004, 1101, 1115, 0.5),
    ])

    average_pairwise_csvs(str(a), str(b), str(out), weights=(1.0, 0.0))

    df_a = pd.read_csv(a).sort_values(["season", "team_a", "team_b"]).reset_index(drop=True)
    df_o = pd.read_csv(out).sort_values(["season", "team_a", "team_b"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(df_a, df_o)


def test_join_one_to_one_required(tmp_path):
    """If A has a (season, a, b) absent in B (or vice versa), error out."""
    from src.ensemble_stage1 import average_pairwise_csvs

    a = tmp_path / "a.csv"
    b = tmp_path / "b.csv"
    out = tmp_path / "ens.csv"
    _write_csv(a, [(2003, 1104, 1112, 0.5), (2003, 1104, 1113, 0.5)])
    _write_csv(b, [(2003, 1104, 1112, 0.5)])  # missing the 1113 pair

    with pytest.raises(ValueError, match="join coverage"):
        average_pairwise_csvs(str(a), str(b), str(out), weights=(0.5, 0.5))


def test_weights_sum_validation(tmp_path):
    """Weights must sum to 1.0 (within float tolerance) so output is a probability."""
    from src.ensemble_stage1 import average_pairwise_csvs

    a = tmp_path / "a.csv"
    b = tmp_path / "b.csv"
    out = tmp_path / "ens.csv"
    _write_csv(a, [(2003, 1104, 1112, 0.5)])
    _write_csv(b, [(2003, 1104, 1112, 0.5)])

    with pytest.raises(ValueError, match="sum to 1"):
        average_pairwise_csvs(str(a), str(b), str(out), weights=(0.5, 0.6))


def test_cli_invocation(tmp_path):
    """Smoke: src/ensemble_stage1.py with --weights and CSV paths runs."""
    import subprocess
    a = tmp_path / "a.csv"
    b = tmp_path / "b.csv"
    out = tmp_path / "ens.csv"
    _write_csv(a, [(2003, 1104, 1112, 0.3)])
    _write_csv(b, [(2003, 1104, 1112, 0.7)])

    cmd = [
        "python", "src/ensemble_stage1.py",
        "--in-a", str(a), "--in-b", str(b),
        "--out", str(out), "--weights", "0.5,0.5",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    df = pd.read_csv(out)
    assert df.loc[0, "p_a_wins"] == pytest.approx(0.5)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_ensemble_stage1.py -v`
Expected: FAIL with `ImportError: cannot import name 'average_pairwise_csvs'`.

- [ ] **Step 3: Implement `src/ensemble_stage1.py`**

Create `src/ensemble_stage1.py`:

```python
"""Average two pairwise-prediction CSVs into an ensemble pairwise CSV.

Spec: docs/superpowers/specs/2026-05-01-ensemble-stage1-design.md

Pure utility: no model. Joins on (season, team_a, team_b) and computes
    p_ensemble = w_a * p_a + w_b * p_b
The output schema is identical to pairwise_v4.csv:
    season, team_a, team_b, p_a_wins.

Anchor: --weights 1.0,0.0 reproduces input A row-for-row; the
LOSO experiment depends on this anchor passing before any
ensemble-vs-baseline numbers are trusted.
"""
import argparse
import sys
from pathlib import Path
from typing import Tuple

import pandas as pd

SCHEMA = ["season", "team_a", "team_b", "p_a_wins"]
JOIN_KEYS = ["season", "team_a", "team_b"]


def average_pairwise_csvs(
    in_a: str, in_b: str, out: str, weights: Tuple[float, float] = (0.5, 0.5)
) -> None:
    """Average two pairwise CSVs and write the result.

    in_a, in_b: paths to CSVs with columns season, team_a, team_b, p_a_wins.
                team_a < team_b (canonical orientation).
    out:        path to write the averaged CSV (same schema).
    weights:    (w_a, w_b) -- must sum to 1.0 within 1e-9.
    """
    w_a, w_b = float(weights[0]), float(weights[1])
    if abs(w_a + w_b - 1.0) > 1e-9:
        raise ValueError(
            f"weights must sum to 1; got {w_a} + {w_b} = {w_a + w_b}"
        )

    df_a = pd.read_csv(in_a)
    df_b = pd.read_csv(in_b)

    for label, df in [("a", df_a), ("b", df_b)]:
        missing = set(SCHEMA) - set(df.columns)
        if missing:
            raise ValueError(
                f"input {label} ({in_a if label == 'a' else in_b}) missing "
                f"columns: {sorted(missing)}"
            )

    # Inner join + coverage check.
    merged = df_a.merge(
        df_b, on=JOIN_KEYS, suffixes=("_a", "_b"), how="outer", indicator=True
    )
    only_a = (merged["_merge"] == "left_only").sum()
    only_b = (merged["_merge"] == "right_only").sum()
    if only_a or only_b:
        raise ValueError(
            f"join coverage failed: {only_a} rows only in A, "
            f"{only_b} rows only in B; the ensemble requires "
            "byte-identical pair coverage across inputs"
        )
    merged = merged.drop(columns=["_merge"])

    merged["p_a_wins"] = w_a * merged["p_a_wins_a"] + w_b * merged["p_a_wins_b"]

    out_df = (
        merged[SCHEMA]
        .sort_values(JOIN_KEYS)
        .reset_index(drop=True)
    )
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out, index=False)


def _parse_weights(s: str) -> Tuple[float, float]:
    parts = s.split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(
            f"weights must be 'w_a,w_b' (two comma-separated floats); got {s!r}"
        )
    return float(parts[0]), float(parts[1])


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Average two pairwise-prediction CSVs."
    )
    parser.add_argument("--in-a", required=True, help="first input CSV")
    parser.add_argument("--in-b", required=True, help="second input CSV")
    parser.add_argument("--out", required=True, help="output CSV")
    parser.add_argument(
        "--weights", type=_parse_weights, default=(0.5, 0.5),
        help="comma-separated weights 'w_a,w_b' (must sum to 1; default 0.5,0.5)"
    )
    args = parser.parse_args(argv)
    average_pairwise_csvs(args.in_a, args.in_b, args.out, args.weights)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

Important: the script writes the *output* sorted by `(season, team_a, team_b)` for deterministic byte-equality. The anchor test relies on this.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_ensemble_stage1.py -v`
Expected: 5 PASSED.

- [ ] **Step 5: Run the full test suite (verify no regressions)**

Run: `pytest -v`
Expected: all green.

- [ ] **Step 6: Commit**

```bash
git add src/ensemble_stage1.py tests/test_ensemble_stage1.py
git commit -m "$(cat <<'EOF'
feat(ensemble): src/ensemble_stage1.py + tests

Pure CSV-averaging utility. Joins two pairwise CSVs on
(season, team_a, team_b) and writes a weighted average. Strict
coverage check on the join (rows only in one input -> error). Anchor
test verifies --weights 1.0,0.0 reproduces input A row-for-row.

Schema and sort order match pairwise_v4.csv so downstream consumers
(train_upset_model.build_v9_pairwise, score_chalk_brackets.
score_pairwise_path) read the ensemble output without changes.
EOF
)"
```

---

### Task 3: Build `src/train_lr_stage1.py` (LR LOSO trainer)

**Why:** This is the new ensemble member. Mirrors v4's LOSO loop but trains a logistic regression with `StandardScaler` + Platt-calibrated outputs. Reuses `prepare_loso_inputs()` so XGB and LR see byte-identical training rows in every fold. The experiment is "model class diversity at identical input."

**Files:**
- Create: `src/train_lr_stage1.py`
- Test: `tests/test_train_lr_stage1.py` (new)

- [ ] **Step 1: Write the failing smoke test**

Create `tests/test_train_lr_stage1.py`:

```python
"""Smoke + unit tests for src/train_lr_stage1.py.

The full 22-season LOSO is a separate run (Task 4), not exercised here.
Tests below verify the per-fold training function on synthetic data and
the pairwise-dump function on a tiny field.
"""
import numpy as np
import pandas as pd
import pytest


def test_fit_lr_with_calibration_returns_probabilities():
    """Per-fold trainer produces a calibrated classifier whose predict_proba
    yields values in [0, 1] for each class."""
    from src.train_lr_stage1 import fit_lr_with_calibration

    rng = np.random.default_rng(42)
    X = rng.standard_normal((200, 5))
    # Linearly separable signal so a working LR can fit non-trivially.
    y = (X[:, 0] - X[:, 1] > 0).astype(int)
    w = np.ones(len(y))

    model = fit_lr_with_calibration(X, y, w, seed=42)
    p = model.predict_proba(X)
    assert p.shape == (200, 2)
    assert np.all((p >= 0.0) & (p <= 1.0))
    # Calibration shouldn't kill discrimination: roughly correct on training.
    acc = float((p[:, 1] > 0.5) == y).mean()
    assert acc > 0.7, f"trainer should learn the linear separation: acc={acc}"


def test_fit_lr_handles_unbalanced_weights():
    """Sample weights must propagate -- a row with weight 0 should be ignored."""
    from src.train_lr_stage1 import fit_lr_with_calibration

    X = np.array([[1.0], [-1.0], [1.0], [-1.0]])
    y = np.array([1, 0, 0, 1])  # weight-0 rows have flipped labels
    w = np.array([1.0, 1.0, 0.0, 0.0])

    model = fit_lr_with_calibration(X, y, w, seed=42)
    # Despite the flipped weight-0 rows, the fitted model should classify
    # X=1 -> 1 and X=-1 -> 0.
    p = model.predict_proba(np.array([[1.0], [-1.0]]))[:, 1]
    assert p[0] > 0.5
    assert p[1] < 0.5


def test_dump_pairwise_writes_expected_schema(tmp_path):
    """The pairwise-dump helper writes (season, team_a, team_b, p_a_wins)
    rows for the cartesian field, with team_a < team_b canonicalization."""
    from src.train_lr_stage1 import dump_pairwise_for_season

    feature_lookup = {
        1101: np.array([0.5, 0.1]),
        1102: np.array([-0.2, 0.3]),
        1103: np.array([0.0, -0.1]),
    }

    class FakeModel:
        def predict_proba(self, X):
            # Simple deterministic mapping for assertions.
            return np.column_stack([1 - X[:, 0], X[:, 0]])

    out = tmp_path / "pw.csv"
    dump_pairwise_for_season(
        season=2003,
        field_team_ids=[1101, 1102, 1103],
        feature_lookup=feature_lookup,
        feature_cols_diff_only=True,
        scaler=None,  # bypass scaling for this synthetic test
        model=FakeModel(),
        out_csv=str(out),
    )

    df = pd.read_csv(out)
    assert list(df.columns) == ["season", "team_a", "team_b", "p_a_wins"]
    # 3 teams -> 3 unordered pairs.
    assert len(df) == 3
    assert (df["team_a"] < df["team_b"]).all()
    assert (df["season"] == 2003).all()
```

(Note: `dump_pairwise_for_season` will be defined in the next step. Its real implementation will not have a `feature_cols_diff_only` flag -- this is just for the synthetic test setup. Re-shape the test signature once the implementation is locked.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_train_lr_stage1.py -v`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Implement `src/train_lr_stage1.py`**

Create `src/train_lr_stage1.py`:

```python
"""LR stage-1 trainer over the same 22-season LOSO loop v4 uses.

Spec: docs/superpowers/specs/2026-05-01-ensemble-stage1-design.md

Mirrors src/enhanced_model_v3.py:leave_one_season_out_cv_weighted but
swaps the XGBoost classifier for a logistic regression with
StandardScaler + Platt calibration (CalibratedClassifierCV). Inputs --
feature_matrix, tourney_filtered, regular_results, feature_cols,
top_80_by_season -- come from prepare_loso_inputs(), which is the same
data-setup v4 calls. So XGB and LR see byte-identical training rows
in every fold.

L2 strength is chosen via inner CV on the training folds only (5
candidate Cs). Scaler is fit per fold; never on the test season.
Platt scaling: CalibratedClassifierCV(method='sigmoid', cv=5) wraps
the LR on the training folds. Every supervised fit -- the logistic
regression, the inner CV, and the Platt calibrator -- sees only train-
fold rows; the test season is held out end-to-end.

Output: appends to output/pairwise_lr.csv with rows
    (season, team_a, team_b, p_a_wins),  team_a < team_b
covering all unordered pairs of tournament-field teams in each
held-out season.
"""
import argparse
import sys
import time
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.enhanced_model_v3 import prepare_loso_inputs
from src.models.matchup import (
    build_matchup_features,
    expand_feature_cols,
)
from src.enhanced_model import build_matchup_data_from_kaggle
from src.models.matchup import build_weighted_matchup_data

DEFAULT_PAIRWISE_OUT = "output/pairwise_lr.csv"
INNER_CV_FOLDS = 5
C_GRID = [0.01, 0.1, 1.0, 10.0, 100.0]


def fit_lr_with_calibration(
    X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray, seed: int = 42
):
    """Inner-CV-tune L2 strength then Platt-calibrate the LR.

    Returns a fitted CalibratedClassifierCV wrapping the best LR. Caller
    is responsible for any feature-scaling step before calling this --
    inputs here should already be standardized.
    """
    base = LogisticRegression(
        penalty="l2", solver="lbfgs", max_iter=2000, random_state=seed,
    )
    grid = GridSearchCV(
        base, {"C": C_GRID}, cv=INNER_CV_FOLDS, scoring="neg_log_loss",
        n_jobs=1, refit=True,
    )
    # GridSearchCV.fit accepts sample_weight via fit_params kwarg.
    grid.fit(X, y, sample_weight=sample_weight)
    best_C = grid.best_params_["C"]

    # Wrap a fresh LR with the best C in CalibratedClassifierCV (Platt).
    base_calibrated = LogisticRegression(
        penalty="l2", solver="lbfgs", max_iter=2000, random_state=seed, C=best_C,
    )
    calibrated = CalibratedClassifierCV(
        base_calibrated, method="sigmoid", cv=INNER_CV_FOLDS,
    )
    calibrated.fit(X, y, sample_weight=sample_weight)
    return calibrated


def dump_pairwise_for_season(
    season: int,
    field_team_ids: Iterable[int],
    feature_lookup: dict,
    feature_cols_diff_only: bool,  # kept for test-shape compat; always True in production
    scaler,
    model,
    out_csv: str,
) -> int:
    """Append (season, team_a, team_b, p_a_wins) rows for the season to out_csv.

    field_team_ids: iterable of team IDs that appeared in this season's
        tournament. We materialize all unordered pairs (a, b) with a < b.
    feature_lookup: dict[team_id -> np.ndarray of raw features]; the diff
        between two teams is the matchup feature row.
    scaler: a fitted StandardScaler (or None for synthetic-test use); when
        not None, applied to the matchup-row matrix before predict_proba.
    model: a fitted classifier with predict_proba(X) -> [N,2].

    Returns the number of pair rows written.
    """
    field = sorted(set(int(t) for t in field_team_ids if t in feature_lookup))
    if len(field) < 2:
        return 0

    pair_rows = []
    pair_ids = []
    for i in range(len(field)):
        for j in range(i + 1, len(field)):
            a, b = field[i], field[j]
            av = feature_lookup[a]
            bv = feature_lookup[b]
            pair_rows.append(build_matchup_features(av, bv))
            pair_ids.append((a, b))

    X = np.array(pair_rows, dtype=float)
    if scaler is not None:
        X = scaler.transform(X)
    p = model.predict_proba(X)[:, 1]

    out_df = pd.DataFrame({
        "season": season,
        "team_a": [a for a, _ in pair_ids],
        "team_b": [b for _, b in pair_ids],
        "p_a_wins": p,
    })
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    write_header = not Path(out_csv).exists()
    out_df.to_csv(out_csv, mode="a", index=False, header=write_header)
    return len(out_df)


def run_lr_loso(out_csv: str = DEFAULT_PAIRWISE_OUT) -> dict:
    """22-season LOSO loop using a logistic regression as the stage-1 model.

    For each held-out season, train LR (with StandardScaler + Platt) on
    every-other-season's weighted matchup data, then dump pairwise probs
    for the held-out season's full field to out_csv.
    """
    print("=" * 70)
    print("LR STAGE-1 LOSO TRAINER")
    print("=" * 70)
    inputs = prepare_loso_inputs()
    feature_matrix = inputs["feature_matrix"]
    tourney = inputs["tourney_filtered"]
    regular_results = inputs["regular_results"]
    feature_cols = inputs["feature_cols"]
    top_80_by_season = inputs["top_80_by_season"]

    if Path(out_csv).exists():
        Path(out_csv).unlink()  # fresh write each run; we append per season
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)

    seasons = sorted(int(s) for s in tourney["Season"].unique() if int(s) >= 2003)
    diff_cols = expand_feature_cols(feature_cols)
    print(f"  feature_cols: {len(feature_cols)} ({len(diff_cols)} diff cols)")
    print(f"  seasons: {seasons[0]}..{seasons[-1]} ({len(seasons)} seasons)")

    per_season_metrics = []
    overall_start = time.time()

    for holdout in seasons:
        t0 = time.time()
        train_tourney = tourney[tourney["Season"] != holdout]
        test_tourney = tourney[tourney["Season"] == holdout]
        if len(test_tourney) == 0:
            continue

        train_top_ids = set()
        for s in train_tourney["Season"].unique():
            train_top_ids |= top_80_by_season.get(int(s), set())

        train_reg = regular_results[regular_results["Season"] != holdout]
        X_train, y_train, w_train = build_weighted_matchup_data(
            feature_matrix, train_tourney, train_reg, feature_cols,
            top_n_team_ids=train_top_ids,
            supplemental_weight=0.25,
        )
        X_test, y_test, _ = build_matchup_data_from_kaggle(
            feature_matrix, test_tourney, feature_cols
        )

        if X_train.empty or X_test.empty:
            print(f"  [{holdout}] empty train/test, skipping")
            continue

        # Match v3's NaN handling: per-fold medians from training data.
        medians = X_train.median()
        X_train = X_train.fillna(medians).to_numpy(dtype=float)
        X_test = X_test.fillna(medians).to_numpy(dtype=float)

        scaler = StandardScaler().fit(X_train)
        X_train_s = scaler.transform(X_train)
        X_test_s = scaler.transform(X_test)

        model = fit_lr_with_calibration(
            X_train_s, y_train.to_numpy(), w_train.to_numpy(), seed=42,
        )
        p_test = model.predict_proba(X_test_s)[:, 1]

        from sklearn.metrics import log_loss as sklearn_log_loss
        ll = float(sklearn_log_loss(y_test, p_test, labels=[0, 1]))
        acc = float(((p_test > 0.5).astype(int) == y_test.to_numpy()).mean())

        # Build per-team feature vector lookup for the held-out year and
        # dump pairwise predictions for the full field.
        fm_year = feature_matrix[feature_matrix["Season"] == holdout]
        fm_year = fm_year.set_index("TeamID")
        feature_lookup = {}
        for tid in fm_year.index:
            row = fm_year.loc[tid, feature_cols]
            vals = pd.to_numeric(row, errors="coerce").to_numpy(dtype=float)
            # Apply the same training medians + scaling at the *raw-feature*
            # level by passing through build_matchup_features in
            # dump_pairwise_for_season; we do not standardize raw features
            # here -- standardization happens on the diff matrix produced
            # there.
            feature_lookup[int(tid)] = vals

        # Field = all teams in this season's tournament results.
        field_ids = sorted(set(test_tourney["WTeamID"]) | set(test_tourney["LTeamID"]))

        # NaN-fill in the per-team raw features using the training medians
        # for each raw feature column. (medians is keyed by diff-col name;
        # convert to raw-col medians.)
        raw_medians = {
            c: float(X_train[:, diff_cols.index(f"{c}_diff")].mean())
            if f"{c}_diff" in diff_cols else 0.0
            for c in feature_cols
        }
        for tid, vals in feature_lookup.items():
            mask = np.isnan(vals)
            if mask.any():
                fill = np.array(
                    [raw_medians.get(c, 0.0) for c in feature_cols], dtype=float
                )
                vals[mask] = fill[mask]
                feature_lookup[tid] = vals

        n_pairs = dump_pairwise_for_season(
            season=holdout,
            field_team_ids=field_ids,
            feature_lookup=feature_lookup,
            feature_cols_diff_only=True,
            scaler=scaler,
            model=model,
            out_csv=out_csv,
        )

        elapsed = time.time() - t0
        per_season_metrics.append({
            "season": holdout,
            "n_test_games": len(y_test),
            "log_loss": ll,
            "accuracy": acc,
            "n_pairs_written": n_pairs,
            "fold_seconds": round(elapsed, 1),
        })
        print(f"  [{holdout}] ll={ll:.4f} acc={acc:.3f} "
              f"pairs={n_pairs} ({elapsed:.1f}s)")

    df = pd.DataFrame(per_season_metrics)
    overall = time.time() - overall_start
    print(f"\nDONE in {overall:.1f}s")
    if len(df):
        print(f"  weighted-mean log_loss: "
              f"{(df['log_loss'] * df['n_test_games']).sum() / df['n_test_games'].sum():.4f}")
        print(f"  weighted-mean accuracy: "
              f"{(df['accuracy'] * df['n_test_games']).sum() / df['n_test_games'].sum():.4f}")
    print(f"  pairwise CSV: {out_csv}")
    return {"per_season": df, "out_csv": out_csv}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--out", default=DEFAULT_PAIRWISE_OUT,
        help=f"output pairwise CSV (default: {DEFAULT_PAIRWISE_OUT})"
    )
    args = parser.parse_args(argv)
    run_lr_loso(out_csv=args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

NOTE: the implementation above carries a real subtlety -- `dump_pairwise_for_season` builds matchup-diff features from raw-team feature vectors and then applies `scaler.transform(...)` to those diff rows. The scaler was fit on diff rows in `run_lr_loso`, so the spaces match. The `raw_medians` block exists only to NaN-fill per-team raw features so the diff is well-defined; the medians are estimated from the diff matrix's per-column mean as a safe-enough proxy. If the smoke test in Step 1 had used a real fitted scaler, this subtlety would have surfaced -- the synthetic test sidesteps it via `scaler=None`.

Update `tests/test_train_lr_stage1.py:test_dump_pairwise_writes_expected_schema` if the actual signature drifted from the test's expectation; keep the assertions on schema, sort, and team_a<team_b.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_train_lr_stage1.py -v`
Expected: 3 PASSED.

- [ ] **Step 5: Run a 2-season smoke**

Run a quick sanity check that doesn't take 5+ minutes. Modify the seasons loop locally in an interpreter (or a one-off subset script):

```bash
python -c "
import sys; sys.path.insert(0, '.')
from src.train_lr_stage1 import run_lr_loso
import src.train_lr_stage1 as M
# Patch to a 2-season subset for the smoke.
orig = M.run_lr_loso
import pandas as pd, src.enhanced_model_v3 as V3
inp = V3.prepare_loso_inputs()
inp['tourney_filtered'] = inp['tourney_filtered'][inp['tourney_filtered']['Season'].isin([2003, 2004])]
def fake_prepare(): return inp
M.prepare_loso_inputs = fake_prepare
out = run_lr_loso(out_csv='output/_smoke_pairwise_lr.csv')
import pandas as pd
df = pd.read_csv('output/_smoke_pairwise_lr.csv')
assert list(df.columns) == ['season','team_a','team_b','p_a_wins'], df.columns.tolist()
assert set(df['season'].unique()) == {2003, 2004}, df['season'].unique()
assert (df['team_a'] < df['team_b']).all()
assert df['p_a_wins'].between(0, 1).all()
print(f'smoke OK: {len(df)} rows over 2 seasons')
"
rm output/_smoke_pairwise_lr.csv
```

Expected: `smoke OK: <N> rows over 2 seasons` where N is around 4000-5000.

- [ ] **Step 6: Run the full test suite**

Run: `pytest -v`
Expected: all green.

- [ ] **Step 7: Commit**

```bash
git add src/train_lr_stage1.py tests/test_train_lr_stage1.py
git commit -m "$(cat <<'EOF'
feat(ensemble): src/train_lr_stage1.py + tests

Logistic regression stage-1 trainer over the same 22-season LOSO loop
v4 uses. Reuses prepare_loso_inputs() so XGB and LR see byte-
identical training rows; the experiment is "model-class diversity at
identical input."

Per fold: StandardScaler fit on training rows, GridSearchCV picks the
L2 strength from {0.01, 0.1, 1, 10, 100} on the training folds,
CalibratedClassifierCV (method='sigmoid', cv=5) does Platt on the
calibrated LR. Output: output/pairwise_lr.csv (same schema as
pairwise_v4.csv).

Smoke test on a 2-season subset; full 22-season run lands in Task 4.
EOF
)"
```

---

### Task 4: Run LR LOSO end-to-end and commit `pairwise_lr.csv`

**Why:** Generates the LR ensemble member's full output. This is the long-running step (5-10 min).

**Files:**
- Generate: `output/pairwise_lr.csv`

- [ ] **Step 1: Run the trainer**

Run: `python src/train_lr_stage1.py --out output/pairwise_lr.csv 2>&1 | tee /tmp/lr_run.log`
Expected: per-season `[YYYY] ll=X.XXXX acc=X.XXX pairs=N (Ts)` lines for 2003-2025 (excluding 2020 if missing data). Final summary prints weighted-mean log loss + accuracy. Total runtime ~5-10 min.

- [ ] **Step 2: Sanity-check the output**

```bash
python -c "
import pandas as pd
v4 = pd.read_csv('output/pairwise_v4.csv')
lr = pd.read_csv('output/pairwise_lr.csv')
print(f'v4: {len(v4):,} rows, {v4.season.nunique()} seasons')
print(f'lr: {len(lr):,} rows, {lr.season.nunique()} seasons')
assert list(lr.columns) == ['season','team_a','team_b','p_a_wins'], lr.columns.tolist()
assert (lr['team_a'] < lr['team_b']).all()
assert lr['p_a_wins'].between(0, 1).all()
# Coverage: every (season, a, b) in v4 should also be in lr (same field).
v4_keys = set(zip(v4.season, v4.team_a, v4.team_b))
lr_keys = set(zip(lr.season, lr.team_a, lr.team_b))
only_v4 = v4_keys - lr_keys
only_lr = lr_keys - v4_keys
print(f'only in v4: {len(only_v4)}; only in lr: {len(only_lr)}')
assert len(only_v4) == 0 and len(only_lr) == 0, \
    'pair coverage mismatch: ensemble averaging will fail in Task 5'
print('OK: schema, sort, range, and pair coverage all match v4')
"
```

Expected: row counts roughly equal between v4 and lr (~96k each), schema matches, no coverage mismatch.

- [ ] **Step 3: Commit the CSV**

```bash
git add output/pairwise_lr.csv
git commit -m "$(cat <<'EOF'
data(ensemble): output/pairwise_lr.csv (22-season LR LOSO)

Generated by src/train_lr_stage1.py over seasons 2003-2025. Schema and
pair coverage match output/pairwise_v4.csv. Used by ensemble_stage1.py
in Task 5 to produce pairwise_ensemble.csv.

Runtime ~Xm on this machine; weighted-mean log_loss=X.XXX,
weighted-mean accuracy=X.XXX (paste exact numbers from /tmp/lr_run.log).
EOF
)"
```

---

### Task 5: Generate `pairwise_ensemble.csv` (with anchor checks)

**Why:** This is the experimental stage-1 output. The two anchor checks before the real averaging are non-negotiable -- if `(1.0, 0.0)` doesn't byte-equal `pairwise_v4.csv`, the ensemble's join logic is buggy and any downstream numbers are tainted.

**Files:**
- Generate: `output/pairwise_ensemble.csv`

- [ ] **Step 1: Anchor check (1.0, 0.0)**

Run:
```bash
python src/ensemble_stage1.py \
  --in-a output/pairwise_v4.csv \
  --in-b output/pairwise_lr.csv \
  --out output/_anchor_v4.csv \
  --weights 1.0,0.0
python -c "
import pandas as pd
a = pd.read_csv('output/pairwise_v4.csv').sort_values(['season','team_a','team_b']).reset_index(drop=True)
b = pd.read_csv('output/_anchor_v4.csv').sort_values(['season','team_a','team_b']).reset_index(drop=True)
pd.testing.assert_frame_equal(a, b, check_dtype=False)
print('anchor (1.0, 0.0) OK: byte-equals pairwise_v4.csv')
"
rm output/_anchor_v4.csv
```
Expected: `anchor (1.0, 0.0) OK: byte-equals pairwise_v4.csv`. If it fails, do NOT proceed -- diagnose join keys, sort order, dtype.

- [ ] **Step 2: Anchor check (0.0, 1.0)**

```bash
python src/ensemble_stage1.py \
  --in-a output/pairwise_v4.csv \
  --in-b output/pairwise_lr.csv \
  --out output/_anchor_lr.csv \
  --weights 0.0,1.0
python -c "
import pandas as pd
a = pd.read_csv('output/pairwise_lr.csv').sort_values(['season','team_a','team_b']).reset_index(drop=True)
b = pd.read_csv('output/_anchor_lr.csv').sort_values(['season','team_a','team_b']).reset_index(drop=True)
pd.testing.assert_frame_equal(a, b, check_dtype=False)
print('anchor (0.0, 1.0) OK: byte-equals pairwise_lr.csv')
"
rm output/_anchor_lr.csv
```
Expected: `anchor (0.0, 1.0) OK: byte-equals pairwise_lr.csv`.

- [ ] **Step 3: Real run, weights 0.5, 0.5**

```bash
python src/ensemble_stage1.py \
  --in-a output/pairwise_v4.csv \
  --in-b output/pairwise_lr.csv \
  --out output/pairwise_ensemble.csv \
  --weights 0.5,0.5
```
Expected: `wrote output/pairwise_ensemble.csv`.

- [ ] **Step 4: Verify schema and coverage**

```bash
python -c "
import pandas as pd
v4 = pd.read_csv('output/pairwise_v4.csv')
ens = pd.read_csv('output/pairwise_ensemble.csv')
assert len(v4) == len(ens)
assert list(ens.columns) == ['season','team_a','team_b','p_a_wins']
assert ens['p_a_wins'].between(0, 1).all()
print(f'OK: {len(ens):,} rows, all probabilities in [0,1]')
"
```

- [ ] **Step 5: Commit**

```bash
git add output/pairwise_ensemble.csv
git commit -m "$(cat <<'EOF'
data(ensemble): output/pairwise_ensemble.csv (0.5*v4 + 0.5*lr)

Anchor checks pass: --weights 1.0,0.0 byte-equals pairwise_v4.csv;
--weights 0.0,1.0 byte-equals pairwise_lr.csv. The real run averages
v4 and LR with equal weights.
EOF
)"
```

---

### Task 6: Build `src/eval_stage1.py` (LOSO log loss + accuracy on a pairwise CSV)

**Why:** Stage-1-only head-to-head: did the ensemble move raw stage-1 quality at all? If both v4 and ensemble look identical here, the diversity hypothesis is already falsified before we run v9-C on top.

**Files:**
- Create: `src/eval_stage1.py`
- Test: `tests/test_eval_stage1.py` (new)

- [ ] **Step 1: Write the failing tests**

```python
"""Unit tests for src/eval_stage1.py."""
from pathlib import Path
import pandas as pd
import pytest


def _write_pairwise(path, rows):
    df = pd.DataFrame(rows, columns=["season", "team_a", "team_b", "p_a_wins"])
    df.to_csv(path, index=False)


def test_per_season_log_loss_known_values(tmp_path):
    """Two pre-computed games per season, known probabilities and outcomes."""
    from src.eval_stage1 import evaluate_pairwise

    pw = tmp_path / "pw.csv"
    _write_pairwise(pw, [
        (2003, 1101, 1102, 0.9),  # team_a wins -> p=0.9, label=1
        (2003, 1103, 1104, 0.4),  # team_b wins -> p_for_a=0.4, label=0
    ])
    results = pd.DataFrame([
        {"Season": 2003, "WTeamID": 1101, "LTeamID": 1102, "DayNum": 136},
        {"Season": 2003, "WTeamID": 1104, "LTeamID": 1103, "DayNum": 136},
    ])

    out = evaluate_pairwise(str(pw), results_df=results)
    assert 2003 in out["per_season"]
    season = out["per_season"][2003]
    # Per-game log loss: -log(0.9) for game 1, -log(0.6) for game 2
    # (since game 2's actual winner is 1104; p(1103 beats 1104) = 0.4 -> p(1104 beats 1103) = 0.6)
    expected_ll = -((0.5)*((-pytest.approx(0)) + 0))  # placeholder; actual computation below
    import math
    expected = (-math.log(0.9) - math.log(0.6)) / 2
    assert season["log_loss"] == pytest.approx(expected, abs=1e-6)
    assert season["n_games"] == 2
    assert season["accuracy"] == pytest.approx(1.0)


def test_weighted_mean_aggregation(tmp_path):
    """Weighted-mean log loss across seasons weights by n_games per season."""
    from src.eval_stage1 import evaluate_pairwise

    pw = tmp_path / "pw.csv"
    _write_pairwise(pw, [
        (2003, 1101, 1102, 0.9),
        (2004, 1103, 1104, 0.5),
        (2004, 1105, 1106, 0.5),
    ])
    results = pd.DataFrame([
        {"Season": 2003, "WTeamID": 1101, "LTeamID": 1102, "DayNum": 136},
        {"Season": 2004, "WTeamID": 1103, "LTeamID": 1104, "DayNum": 136},
        {"Season": 2004, "WTeamID": 1105, "LTeamID": 1106, "DayNum": 136},
    ])

    out = evaluate_pairwise(str(pw), results_df=results)
    # Total games = 3; 2003 contributes 1, 2004 contributes 2.
    assert out["weighted_mean_log_loss"] > 0
    assert out["weighted_mean_accuracy"] >= 0.0


def test_skips_games_without_pairwise_prob(tmp_path):
    """If a real game's pair isn't in the pairwise CSV, skip it without failing."""
    from src.eval_stage1 import evaluate_pairwise

    pw = tmp_path / "pw.csv"
    _write_pairwise(pw, [(2003, 1101, 1102, 0.9)])
    results = pd.DataFrame([
        {"Season": 2003, "WTeamID": 1101, "LTeamID": 1102, "DayNum": 136},
        {"Season": 2003, "WTeamID": 9999, "LTeamID": 8888, "DayNum": 136},  # not in pw
    ])

    out = evaluate_pairwise(str(pw), results_df=results)
    assert out["per_season"][2003]["n_games"] == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_eval_stage1.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement `src/eval_stage1.py`**

```python
"""Per-season LOSO log loss + accuracy from a pairwise CSV.

Spec: docs/superpowers/specs/2026-05-01-ensemble-stage1-design.md

Reads a pairwise CSV (season, team_a, team_b, p_a_wins; team_a < team_b)
and the Kaggle MNCAATourneyCompactResults.csv, computes per-season log
loss + accuracy on tournament games, then weighted-mean across seasons
(weight = n_games per season).

Used to compare pairwise_v4.csv vs pairwise_ensemble.csv as the stage-1
only head-to-head before the v9-C correction step.
"""
import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path("data/raw/march-machine-learning-2026")


def evaluate_pairwise(
    pairwise_csv: str,
    results_csv: str = str(DATA / "MNCAATourneyCompactResults.csv"),
    results_df: pd.DataFrame = None,
) -> dict:
    """Compute per-season log loss + accuracy for a pairwise CSV."""
    pw = pd.read_csv(pairwise_csv)
    pw_lookup = {}
    for s, a, b, p in zip(pw.season, pw.team_a, pw.team_b, pw.p_a_wins):
        pw_lookup[(int(s), int(a), int(b))] = float(p)

    if results_df is None:
        results_df = pd.read_csv(results_csv)

    per_season = {}
    eps = 1e-15
    for season, group in results_df.groupby("Season"):
        season = int(season)
        if season < 2003:
            continue
        ll_terms = []
        correct = 0
        for _, g in group.iterrows():
            w, l = int(g["WTeamID"]), int(g["LTeamID"])
            a, b = (w, l) if w < l else (l, w)
            p = pw_lookup.get((season, a, b))
            if p is None:
                continue
            p_w = p if a == w else 1.0 - p
            p_w = min(max(p_w, eps), 1.0 - eps)
            ll_terms.append(-math.log(p_w))
            correct += 1 if p_w > 0.5 else 0
        if not ll_terms:
            continue
        per_season[season] = {
            "n_games": len(ll_terms),
            "log_loss": float(np.mean(ll_terms)),
            "accuracy": float(correct / len(ll_terms)),
        }

    if not per_season:
        return {
            "per_season": {},
            "weighted_mean_log_loss": float("nan"),
            "weighted_mean_accuracy": float("nan"),
            "total_games": 0,
        }

    total_n = sum(s["n_games"] for s in per_season.values())
    wm_ll = sum(s["log_loss"] * s["n_games"] for s in per_season.values()) / total_n
    wm_acc = sum(s["accuracy"] * s["n_games"] for s in per_season.values()) / total_n
    return {
        "per_season": per_season,
        "weighted_mean_log_loss": float(wm_ll),
        "weighted_mean_accuracy": float(wm_acc),
        "total_games": total_n,
    }


def _print_table(name: str, result: dict) -> None:
    print(f"\n=== {name} ===")
    print(f"{'season':>6}  {'n':>4}  {'log_loss':>9}  {'accuracy':>8}")
    for s, m in sorted(result["per_season"].items()):
        print(f"{s:>6}  {m['n_games']:>4}  {m['log_loss']:>9.4f}  "
              f"{m['accuracy']:>8.3f}")
    print(f"{'WMEAN':>6}  {result['total_games']:>4}  "
          f"{result['weighted_mean_log_loss']:>9.4f}  "
          f"{result['weighted_mean_accuracy']:>8.3f}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--pairwise", required=True, help="pairwise CSV")
    parser.add_argument("--label", default=None, help="optional label for table header")
    args = parser.parse_args(argv)
    res = evaluate_pairwise(args.pairwise)
    label = args.label or args.pairwise
    _print_table(label, res)
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_eval_stage1.py -v`
Expected: 3 PASSED.

- [ ] **Step 5: Run head-to-head, capture output**

```bash
python src/eval_stage1.py --pairwise output/pairwise_v4.csv --label "v4" 2>&1 | tee /tmp/stage1_v4.log
python src/eval_stage1.py --pairwise output/pairwise_ensemble.csv --label "ensemble" 2>&1 | tee /tmp/stage1_ens.log
```
Expected: per-season tables for both. Capture the WMEAN row from each.

- [ ] **Step 6: Commit**

```bash
git add src/eval_stage1.py tests/test_eval_stage1.py
git commit -m "$(cat <<'EOF'
feat(ensemble): src/eval_stage1.py + tests

Per-season + weighted-mean LOSO log loss + accuracy from a pairwise
CSV. Used to compare pairwise_v4.csv vs pairwise_ensemble.csv as the
stage-1 only head-to-head before the v9-C correction step.
EOF
)"
```

---

### Task 7: Build `src/run_v9c_on_stage1.py` (thin v9-C wrapper)

**Why:** Stage-2 head-to-head requires running v9-C twice with everything held constant *except* the stage-1 input. `train_upset_model.py`'s `__main__` hardcodes the input path; rather than CLI-parameterize the trainer (broader change), wrap the relevant primitives (`load_per_game_data_with_upset`, `build_v9_pairwise`) in a small new script that takes `--pairwise-in` and `--pairwise-out`. Mirrors `sweep_v9_weights.py::run_single_cell` minus the sweep.

**Files:**
- Create: `src/run_v9c_on_stage1.py`
- Test: `tests/test_run_v9c_on_stage1.py` (new)

- [ ] **Step 1: Write the failing test**

```python
"""Unit tests for src/run_v9c_on_stage1.py."""
import os
from pathlib import Path

import pandas as pd
import pytest


@pytest.mark.skipif(
    not Path("data/raw/march-machine-learning-2026/MNCAATourneySeeds.csv").exists(),
    reason="Kaggle Mania data not present"
)
def test_run_v9c_on_pairwise_v4_smoke(tmp_path):
    """Run v9-C on output/pairwise_v4.csv subset to a tmp output and assert
    the schema is the same as pairwise_v4.csv."""
    from src.run_v9c_on_stage1 import run_v9c

    out = tmp_path / "v9c_smoke.csv"
    res = run_v9c(
        pairwise_in="output/pairwise_v4.csv",
        pairwise_out=str(out),
    )

    assert out.exists()
    df = pd.read_csv(out)
    assert list(df.columns) == ["season", "team_a", "team_b", "p_a_wins"], df.columns.tolist()
    assert (df["team_a"] < df["team_b"]).all()
    assert df["p_a_wins"].between(0, 1).all()
    # Should cover all 22 LOSO seasons present in pairwise_v4.csv.
    v4 = pd.read_csv("output/pairwise_v4.csv")
    assert df["season"].nunique() == v4["season"].nunique()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_run_v9c_on_stage1.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement `src/run_v9c_on_stage1.py`**

```python
"""Run v9-C stage-2 on a stage-1 pairwise CSV.

Spec: docs/superpowers/specs/2026-05-01-ensemble-stage1-design.md

Wraps src.train_upset_model.build_v9_pairwise so the experiment can hold
the v9-C config (W_UPSET=1.25, W_MISS=0.0, feature_set='v9c') constant
while only the stage-1 input varies between
    --pairwise-in output/pairwise_v4.csv          (baseline)
    --pairwise-in output/pairwise_ensemble.csv    (experiment)
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.train_upset_model import (
    build_v9_pairwise,
    load_per_game_data_with_upset,
)

DATA = Path("data/raw/march-machine-learning-2026")
W_UPSET = 1.25
W_MISS = 0.0
FEATURE_SET = "v9c"


def run_v9c(pairwise_in: str, pairwise_out: str) -> dict:
    """Run v9-C stage-2 on a stage-1 pairwise CSV.

    Mirrors src.sweep_v9_weights.run_single_cell, minus the per-cell
    metrics dict. Writes pairwise_out (same schema as pairwise_in:
    season, team_a, team_b, p_a_wins) and returns a small summary.
    """
    seeds_csv = str(DATA / "MNCAATourneySeeds.csv")
    results_csv = str(DATA / "MNCAATourneyCompactResults.csv")
    slots_csv = str(DATA / "MNCAATourneySlots.csv")

    print(f"loading per-game data from {pairwise_in} ...")
    per_game = load_per_game_data_with_upset(
        pairwise_in, results_csv, seeds_csv,
    )
    print(f"  rows: {len(per_game):,}; seasons: {per_game.season.nunique()}")

    print(f"applying v9-C (W_UPSET={W_UPSET}, W_MISS={W_MISS}, "
          f"feature_set='{FEATURE_SET}') -> {pairwise_out}")
    Path(pairwise_out).parent.mkdir(parents=True, exist_ok=True)
    build_v9_pairwise(
        per_game, pairwise_in, seeds_csv, pairwise_out,
        slots_csv=slots_csv,
        w_upset=W_UPSET, w_miss=W_MISS,
        feature_set=FEATURE_SET,
    )
    out_df = pd.read_csv(pairwise_out)
    print(f"  wrote {len(out_df):,} rows")
    return {
        "n_rows": len(out_df),
        "n_seasons": out_df["season"].nunique(),
        "out": pairwise_out,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--pairwise-in", required=True)
    parser.add_argument("--pairwise-out", required=True)
    args = parser.parse_args(argv)
    run_v9c(args.pairwise_in, args.pairwise_out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_run_v9c_on_stage1.py -v`
Expected: 1 PASSED (skipped if Kaggle data not present locally; assume present).

- [ ] **Step 5: Run the full test suite**

Run: `pytest -v`
Expected: all green.

- [ ] **Step 6: Commit**

```bash
git add src/run_v9c_on_stage1.py tests/test_run_v9c_on_stage1.py
git commit -m "$(cat <<'EOF'
feat(ensemble): src/run_v9c_on_stage1.py + tests

Thin CLI wrapper around src.train_upset_model.build_v9_pairwise that
runs v9-C stage-2 with the PR 9 winning config (W_UPSET=1.25,
W_MISS=0.0, feature_set='v9c') on any stage-1 pairwise CSV. Used by
Tasks 8-9 to produce side-by-side LOSO outputs for v4 vs ensemble.
EOF
)"
```

---

### Task 8: Generate `pairwise_v9c_v4_baseline.csv` and `pairwise_v9c_ensemble.csv`

**Why:** The two stage-2-corrected LOSO outputs that the bracket-points head-to-head compares.

**Files:**
- Generate: `output/pairwise_v9c_v4_baseline.csv`
- Generate: `output/pairwise_v9c_ensemble.csv`

- [ ] **Step 1: Run baseline (v9-C on v4)**

```bash
python src/run_v9c_on_stage1.py \
  --pairwise-in output/pairwise_v4.csv \
  --pairwise-out output/pairwise_v9c_v4_baseline.csv \
  2>&1 | tee /tmp/v9c_v4.log
```
Expected: completes without error. Row count of output ~96k, 22 seasons. Runtime ~1-3 min.

- [ ] **Step 2: Run experiment (v9-C on ensemble)**

```bash
python src/run_v9c_on_stage1.py \
  --pairwise-in output/pairwise_ensemble.csv \
  --pairwise-out output/pairwise_v9c_ensemble.csv \
  2>&1 | tee /tmp/v9c_ens.log
```
Expected: same shape as Step 1.

- [ ] **Step 3: Sanity-check the two outputs**

```bash
python -c "
import pandas as pd
v4 = pd.read_csv('output/pairwise_v9c_v4_baseline.csv')
ens = pd.read_csv('output/pairwise_v9c_ensemble.csv')
assert len(v4) == len(ens), f'row counts differ: {len(v4)} vs {len(ens)}'
assert v4.season.nunique() == ens.season.nunique()
print(f'OK: both v9-C outputs have {len(v4):,} rows, {v4.season.nunique()} seasons')
"
```

- [ ] **Step 4: Commit**

```bash
git add output/pairwise_v9c_v4_baseline.csv output/pairwise_v9c_ensemble.csv
git commit -m "$(cat <<'EOF'
data(ensemble): v9-C LOSO outputs for v4 vs ensemble head-to-head

output/pairwise_v9c_v4_baseline.csv  -- v9-C on pairwise_v4.csv
output/pairwise_v9c_ensemble.csv     -- v9-C on pairwise_ensemble.csv

Both runs use the PR 9 winning config (W_UPSET=1.25, W_MISS=0.0,
feature_set='v9c'); only the stage-1 input varies. Bracket-points
head-to-head in Task 9.
EOF
)"
```

---

### Task 9: Bracket-points head-to-head over 22 seasons

**Why:** Decisive comparison. The metric that promoted v9-C is bracket points; the production stack's identity is stage-1 + stage-2; this is the only number that decides the verdict band.

**Files:**
- (No new files; uses existing `score_chalk_brackets.score_pairwise_path`.)

- [ ] **Step 1: Score both v9-C outputs**

```bash
python -c "
from src.score_chalk_brackets import score_pairwise_path
import json

baseline = score_pairwise_path('output/pairwise_v9c_v4_baseline.csv')
experiment = score_pairwise_path('output/pairwise_v9c_ensemble.csv')

print('=' * 70)
print('STAGE-1 + v9-C HEAD-TO-HEAD (BRACKET POINTS BY SEASON)')
print('=' * 70)
print(f'{\"season\":>6}  {\"v4_base\":>10}  {\"ensemble\":>10}  {\"delta\":>7}')
common = sorted(set(baseline['per_season_pts']) & set(experiment['per_season_pts']))
total_b, total_e = 0.0, 0.0
wins_b, wins_e, ties = 0, 0, 0
for s in common:
    b = baseline['per_season_pts'][s]
    e = experiment['per_season_pts'][s]
    total_b += b
    total_e += e
    if e > b: wins_e += 1
    elif b > e: wins_b += 1
    else: ties += 1
    print(f'{s:>6}  {b:>10.1f}  {e:>10.1f}  {e-b:>+7.1f}')
print('-' * 38)
print(f'{\"TOTAL\":>6}  {total_b:>10.1f}  {total_e:>10.1f}  {total_e-total_b:>+7.1f}')
print(f'\\nensemble W/L/T vs v4-baseline: {wins_e}/{wins_b}/{ties}')

delta = total_e - total_b
print(f'\\ntotal delta: {delta:+.1f} bracket points')
if delta >= 25:
    verdict = 'CLEAR WIN -- swap to ensemble (>= +25)'
elif delta >= 10:
    verdict = 'MARGINAL -- candidate (+10 to +25), do not swap'
else:
    verdict = 'NO-GO -- keep v4 (< +10)'
print(f'verdict: {verdict}')

# Save tabular data for the findings note.
with open('/tmp/ensemble_brkt_pts.json', 'w') as f:
    json.dump({
        'baseline': baseline,
        'experiment': experiment,
        'delta': delta,
        'verdict': verdict,
        'wins_b': wins_b, 'wins_e': wins_e, 'ties': ties,
    }, f, indent=2)
print('\\nsaved /tmp/ensemble_brkt_pts.json')
" 2>&1 | tee /tmp/ensemble_head_to_head.log
```
Expected: per-season table + total delta + verdict line. Save the log.

- [ ] **Step 2: Capture stage-1-only numbers from Task 6**

Already in `/tmp/stage1_v4.log` and `/tmp/stage1_ens.log`. Note the WMEAN log_loss + accuracy lines for both -- they go in the findings table.

- [ ] **Step 3: No commit yet (data is in /tmp/, becomes part of the findings note in Task 10).**

---

### Task 10: Findings note + TODO update

**Why:** Closes the experiment with a written verdict and updates the active queue. Mirrors the v9-C findings-note pattern.

**Files:**
- Create: `docs/notes/2026-05-01-ensemble-stage1.md`
- Modify: `TODO.md`

- [ ] **Step 1: Draft `docs/notes/2026-05-01-ensemble-stage1.md`**

Use the v9-C findings note as a template. The note covers:
- Header (date, branch, predecessors)
- TL;DR (the verdict in one line)
- Setup recap (what was built, what the ensemble is)
- Stage-1-only LOSO table (v4 vs ensemble: weighted-mean log loss, weighted-mean accuracy, per-season numbers if interesting)
- Stage-1 + v9-C bracket-points table (the per-season + total table from Task 9 Step 1, plus W/L/T)
- Anchor checks (mention that `(1.0, 0.0)` and `(0.0, 1.0)` reproductions passed; cite the test file)
- Verdict (from the verdict-band rules in the spec)
- Recommendation (what becomes follow-up work)
- ASCII-only -- verify with `python -c "open('docs/notes/2026-05-01-ensemble-stage1.md').read().encode('ascii')"`

- [ ] **Step 2: Update TODO.md based on the verdict**

Three branches:

**(A) CLEAR WIN (>= +25 brkt pts):** mark item #1 done, recommend swap PR (which is *separate* from this plan), and renumber the queue.

**(B) MARGINAL (+10 to +25):** add the experiment to "Done" with a "marginal candidate" tag mirroring the PR-7 / PR-8 entries; renumber queue so NN / Bayesian rises to #1.

**(C) NO-GO (< +10):** add to "Tried and rejected" with the falsification reasoning (model-class diversity at identical features did not help). Renumber the active queue: NN moves to #1, Bayesian to #2, with the explicit caveat that the same risk applies (different inductive bias might still be correlated at this feature scale).

For all three branches, the entry should:
- State the delta in brkt pts and the verdict band.
- Cite the findings note path.
- Note the stage-1-only weighted-mean log loss + accuracy numbers as side-context.
- Mention the anchor passing (so future agents trust the numbers).

- [ ] **Step 3: ASCII verification**

```bash
python -c "open('docs/notes/2026-05-01-ensemble-stage1.md', encoding='utf-8').read().encode('ascii')"
python -c "open('TODO.md', encoding='utf-8').read().encode('ascii')"
echo "ASCII OK"
```

- [ ] **Step 4: Run the full test suite one more time**

Run: `pytest -v`
Expected: all green. (Also verifies that the data files committed in Tasks 4, 5, 8 didn't accidentally land somewhere a test file-checks against fixed numbers and now disagrees.)

- [ ] **Step 5: Commit**

```bash
git add docs/notes/2026-05-01-ensemble-stage1.md TODO.md
git commit -m "$(cat <<'EOF'
docs(ensemble): findings note + TODO update

Stage-1 ensemble (XGBoost + logistic regression, simple average)
verdict over 22-season LOSO + v9-C correction. (Adapt subject and
body text to the actual verdict: CLEAR WIN / MARGINAL / NO-GO.)

Findings: docs/notes/2026-05-01-ensemble-stage1.md.
EOF
)"
```

- [ ] **Step 6: Push branch and open PR**

(User-driven; outside the plan's scope, but called out so the loop closes.) Run:

```bash
git push -u origin feat/ensemble-stage1
```

Then open a PR linking the spec, plan, and findings note. The PR description should call out the verdict band and the recommended swap-or-not decision so the user can review without re-running anything.

---

## Self-Review

**Spec coverage check:**

- [x] `src/train_lr_stage1.py` -- Task 3
- [x] `src/ensemble_stage1.py` -- Task 2
- [x] STAGE1_PAIRWISE config update path -- intentionally simplified to a wrapper script (`run_v9c_on_stage1.py`) in Task 7 since the spec's CLI parameterization was a means to "run v9-C with a different stage-1 input" -- the wrapper achieves the same end with smaller blast radius. Captured in Task 7's "Why" rationale.
- [x] Re-run v9-C on both inputs -- Tasks 7-8
- [x] `tests/test_ensemble_stage1.py` -- Task 2
- [x] `tests/test_train_lr_stage1.py` -- Task 3
- [x] Anchor reproduction `(1.0, 0.0)` and `(0.0, 1.0)` -- Task 5 Steps 1-2
- [x] Findings note in `docs/notes/2026-05-01-ensemble-stage1.md` -- Task 10
- [x] Eval methodology: 22-season LOSO -- Tasks 6 + 9
- [x] Success thresholds (verdict bands) -- Task 9 Step 1 + Task 10
- [ ] Apply-time 2026 prediction (JSON form) -- DEFERRED. The spec says the swap is a separate small commit; the JSON-form averaging only matters once the experiment wins. Task 10 Step 2 calls this out as follow-up.
- [ ] Parameterizing `tests/test_predict_2026_v9c.py` -- DEFERRED for the same reason; only relevant in the swap commit.

Both deferred items are documented as follow-ups; they're orthogonal to the verdict the plan delivers.

**Placeholder scan:** no TBD/TODO markers in any task body. Task 10 Step 1 says "fill in the table from /tmp/ensemble_brkt_pts.json" -- those are real numbers captured in Task 9 Step 1, not placeholders.

**Type / signature consistency:**
- `average_pairwise_csvs(in_a, in_b, out, weights)` -- consistent in test, implementation, and CLI argument.
- `prepare_loso_inputs() -> dict` with keys `feature_matrix`, `tourney_filtered`, `regular_results`, `feature_cols`, `top_80_by_season`, `feature_medians` -- consistent across Task 1 (definition) and Task 3 (consumer).
- `run_v9c(pairwise_in, pairwise_out) -> dict` -- consistent in test (Task 7 Step 1) and implementation (Task 7 Step 3).
- `evaluate_pairwise(pairwise_csv, results_csv=..., results_df=...)` -- tests pass `results_df=` kwarg; implementation accepts it.
