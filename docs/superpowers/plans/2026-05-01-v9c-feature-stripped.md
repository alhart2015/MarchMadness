# v9-C Feature-Stripped Variant Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a 5-feature v9-C variant of the upset-aware stage-2 trainer (drops `v4_confidence` and `is_a_higher_seed`, keeps `round`) and re-run PR 8's identical 15-cell W_UPSET / W_MISS sweep on it to test whether the dropped features were noise or signal.

**Architecture:** Parameterize `src/train_upset_model.py` and `src/sweep_v9_weights.py` with a `feature_set: str` arg accepting `"v9b"` (current 7-feature default) or `"v9c"` (new 5-feature). Default preserves v9-B behavior so PR 8 reproduces bit-identical. Sweep driver reads `V9_FEATURE_SET` env var in `main()` and routes outputs to `output/v9c_sweep/` for the new variant, leaving PR 8's `output/v9_sweep/` artifacts untouched.

**Tech Stack:** Python 3, pandas, numpy, xgboost (seed=42, fixed), pytest. Same dependencies as PR 8 -- no new packages.

**Spec:** `docs/superpowers/specs/2026-05-01-v9c-feature-stripped-design.md`

**Branch:** feat/v9-b-followup (already created, spec committed at 03e4c75)

---

## File Structure

| File | Change | Responsibility |
|---|---|---|
| `src/train_upset_model.py` | Modify | Add `feature_set` param to `upset_features`, `double_loso_eval`, `build_v9_pairwise`. Default `"v9b"`. |
| `src/sweep_v9_weights.py` | Modify | Add `feature_set` param to `run_single_cell`, `run_sweep`. `main()` reads `V9_FEATURE_SET` env var and switches output paths. |
| `tests/test_upset_model.py` | Modify | Add 7 new tests for v9-C feature set behavior. |
| `tests/test_sweep_v9_weights.py` | Modify | Add 1 new test for v9-C sweep cell + 1 for env-var output path switching. |
| `output/v9c_sweep/pairwise_v9_WU{u}_WM{m}.csv` | Create (run output) | Per-cell pairwise probs with v9-C trainer. |
| `output/v9c_sweep_results.csv` | Create (run output) | 15-cell sorted results table. |
| `output/v9c_sweep_run.log` | Create (run output) | Driver log including anchor warning + winner declaration. |
| `docs/notes/2026-05-01-v9c-feature-stripped.md` | Create | Findings note in PR 8's template. |
| `TODO.md` | Modify | Update Done section with v9-C outcome; close v9 thread or promote to swap candidate per decision matrix. |

---

## Task 1: Parameterize `upset_features` with `feature_set`

**Files:**
- Modify: `src/train_upset_model.py:257-274` (`upset_features` function)
- Modify: `tests/test_upset_model.py` (add tests)

- [ ] **Step 1: Write 5 failing tests**

Append to `tests/test_upset_model.py`:

```python
# -----------------------------------------------------------------------------
# upset_features feature_set parameterization
# -----------------------------------------------------------------------------

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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_upset_model.py -v -k "upset_features and (v9b or v9c or default or invalid)"`
Expected: 5 tests FAIL. The default-shape test passes if `upset_features(df)` already works on the fixture; the explicit-v9b, v9c-shape, columns-match, and invalid-raises tests fail with `TypeError: upset_features() got an unexpected keyword argument 'feature_set'`.

- [ ] **Step 3: Implement parameterization**

Replace `upset_features` in `src/train_upset_model.py` (currently lines 257-274) with:

```python
def upset_features(df: pd.DataFrame, feature_set: str = "v9b") -> np.ndarray:
    """Pull the v9 input matrix from a per-game DataFrame.

    feature_set:
      "v9b" (default, 7 features): p_stage1, seed_a, seed_b, abs_seed_diff,
        round (1..6 for R64..Champ; 0 if unknown DayNum / not in lookup),
        v4_confidence (|p_stage1 - 0.5|),
        is_a_higher_seed (1.0 if seed_a < seed_b else 0.0).
      "v9c" (5 features): drops v4_confidence and is_a_higher_seed; keeps
        the other five.
    """
    if feature_set not in ("v9b", "v9c"):
        raise ValueError(
            f"unknown feature_set {feature_set!r}; must be 'v9b' or 'v9c'"
        )
    p = df["p_stage1"].values.astype(float)
    sa = df["seed_a"].values.astype(float)
    sb = df["seed_b"].values.astype(float)
    diff = df["abs_seed_diff"].values.astype(float)
    rnd = df["round"].values.astype(float)
    if feature_set == "v9b":
        conf = np.abs(p - 0.5)
        higher = (sa < sb).astype(float)
        return np.column_stack([p, sa, sb, diff, rnd, conf, higher])
    return np.column_stack([p, sa, sb, diff, rnd])
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_upset_model.py -v -k "upset_features and (v9b or v9c or default or invalid)"`
Expected: 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/train_upset_model.py tests/test_upset_model.py
git commit -m "feat(v9): parameterize upset_features with feature_set arg

Adds feature_set in {'v9b', 'v9c'} kwarg defaulting to v9b. v9c is the
5-feature variant (drops v4_confidence + is_a_higher_seed). Unknown
values raise ValueError to fail fast on typos.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: Thread `feature_set` through `double_loso_eval` and `build_v9_pairwise`

**Files:**
- Modify: `src/train_upset_model.py:301-354` (`double_loso_eval`) and `src/train_upset_model.py:357-427` (`build_v9_pairwise`)
- Modify: `tests/test_upset_model.py` (add tests)

- [ ] **Step 1: Write 2 failing tests**

Append to `tests/test_upset_model.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_upset_model.py -v -k "v9c and (double_loso_eval or build_v9_pairwise)"`
Expected: Both tests FAIL with `TypeError: ... got an unexpected keyword argument 'feature_set'`.

- [ ] **Step 3: Add `feature_set` kwarg to `double_loso_eval`**

In `src/train_upset_model.py`, modify `double_loso_eval` signature (line 301) and the two `upset_features` calls (lines 331, 334):

```python
def double_loso_eval(
    per_game: pd.DataFrame,
    w_upset: float = W_UPSET,
    w_miss: float = W_MISS,
    feature_set: str = "v9b",
) -> pd.DataFrame:
```

Then change both `upset_features(train)` -> `upset_features(train, feature_set=feature_set)` and `upset_features(test)` -> `upset_features(test, feature_set=feature_set)`. Update the docstring's "Weights are forwarded" sentence to also mention `feature_set` is forwarded.

- [ ] **Step 4: Add `feature_set` kwarg to `build_v9_pairwise`**

In `src/train_upset_model.py`, modify `build_v9_pairwise` signature (line 357) and the two `upset_features` calls (lines 398, 415):

```python
def build_v9_pairwise(
    per_game: pd.DataFrame,
    pairwise_v4_csv: str,
    seeds_csv: str,
    out_path: str,
    slots_csv: str,
    w_upset: float = W_UPSET,
    w_miss: float = W_MISS,
    feature_set: str = "v9b",
) -> None:
```

Then change `X_train = upset_features(train)` -> `X_train = upset_features(train, feature_set=feature_set)` and `p_v9 = model.predict_proba(upset_features(apply_df))[:, 1]` -> `p_v9 = model.predict_proba(upset_features(apply_df, feature_set=feature_set))[:, 1]`.

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_upset_model.py -v -k "v9c and (double_loso_eval or build_v9_pairwise)"`
Expected: Both tests PASS.

- [ ] **Step 6: Run the full upset-model test file to confirm no v9-B regression**

Run: `pytest tests/test_upset_model.py -v`
Expected: All tests PASS (existing + new).

- [ ] **Step 7: Commit**

```bash
git add src/train_upset_model.py tests/test_upset_model.py
git commit -m "feat(v9): thread feature_set through double_loso_eval + build_v9_pairwise

Both functions take feature_set kwarg defaulting to 'v9b' and forward
it to every upset_features call. v9-B behavior is unchanged when the
default is used.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: Thread `feature_set` through sweep driver helpers

**Files:**
- Modify: `src/sweep_v9_weights.py:63-132` (`run_single_cell`) and `src/sweep_v9_weights.py:135-177` (`run_sweep`)
- Modify: `tests/test_sweep_v9_weights.py` (add test)

- [ ] **Step 1: Write a failing test**

Append to `tests/test_sweep_v9_weights.py` (after the existing `test_run_single_cell_writes_pairwise_and_returns_metrics`):

```python
def test_run_single_cell_v9c_writes_pairwise(tmp_path):
    """run_single_cell with feature_set='v9c' writes a pairwise CSV at the
    same path template and returns metrics dict with the same keys as v9-B.
    """
    pw_path, seeds_path, results_path, slots_path = _write_minimal_inputs(tmp_path)
    out_dir = tmp_path / "v9c_sweep"

    metrics = run_single_cell(
        w_upset=1.0, w_miss=0.0,
        pairwise_v4_csv=pw_path,
        results_csv=results_path,
        seeds_csv=seeds_path,
        out_dir=str(out_dir),
        slots_csv=slots_path,
        feature_set="v9c",
    )

    pw_path_out = out_dir / "pairwise_v9_WU1.00_WM0.00.csv"
    assert pw_path_out.exists()
    assert metrics["w_upset"] == 1.0
    assert metrics["w_miss"] == 0.0
    assert metrics["pairwise_csv"] == str(pw_path_out)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_sweep_v9_weights.py::test_run_single_cell_v9c_writes_pairwise -v`
Expected: FAIL with `TypeError: run_single_cell() got an unexpected keyword argument 'feature_set'`.

- [ ] **Step 3: Add `feature_set` to `run_single_cell`**

In `src/sweep_v9_weights.py`, modify `run_single_cell` signature (line 63) and the two trainer calls inside it:

```python
def run_single_cell(
    w_upset: float,
    w_miss: float,
    pairwise_v4_csv: str,
    results_csv: str,
    seeds_csv: str,
    out_dir: str,
    slots_csv: str,
    feature_set: str = "v9b",
) -> dict:
```

Then update the calls (currently lines 90 and 96):

```python
build_v9_pairwise(
    per_game, pairwise_v4_csv, seeds_csv, pairwise_csv_out,
    slots_csv=slots_csv,
    w_upset=w_upset, w_miss=w_miss,
    feature_set=feature_set,
)

eval_df = double_loso_eval(
    per_game, w_upset=w_upset, w_miss=w_miss, feature_set=feature_set,
)
```

- [ ] **Step 4: Add `feature_set` to `run_sweep`**

In `src/sweep_v9_weights.py`, modify `run_sweep` signature (line 135) and the `run_single_cell` call inside it (line 157):

```python
def run_sweep(
    grid: Iterable[Tuple[float, float]],
    pairwise_v4_csv: str,
    results_csv: str,
    seeds_csv: str,
    out_dir: str,
    results_csv_path: str,
    slots_csv: str,
    feature_set: str = "v9b",
) -> pd.DataFrame:
```

```python
m = run_single_cell(
    w_upset=w_upset, w_miss=w_miss,
    pairwise_v4_csv=pairwise_v4_csv,
    results_csv=results_csv,
    seeds_csv=seeds_csv,
    out_dir=out_dir,
    slots_csv=slots_csv,
    feature_set=feature_set,
)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_sweep_v9_weights.py -v`
Expected: All tests PASS (existing + new).

- [ ] **Step 6: Commit**

```bash
git add src/sweep_v9_weights.py tests/test_sweep_v9_weights.py
git commit -m "feat(v9-sweep): thread feature_set through run_single_cell + run_sweep

Both helpers take feature_set kwarg defaulting to 'v9b' and forward
to build_v9_pairwise + double_loso_eval. v9-B behavior unchanged when
the default is used.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: Wire `V9_FEATURE_SET` env var into sweep `main()`

**Files:**
- Modify: `src/sweep_v9_weights.py:180-241` (`main` function)

- [ ] **Step 1: Add the env-var read and output-path switching**

Replace the `main()` function in `src/sweep_v9_weights.py` (currently lines 180-245) with:

```python
def main():
    """Run the canonical 15-cell sweep against production data paths.

    feature_set is read from the V9_FEATURE_SET env var (default 'v9b').
    Output paths key off the choice so v9-B and v9-C artifacts coexist.

    Compares the anchor cell (1.0, 0.0) bracket points against
    output/pairwise_v8.csv as a sanity gate after the sweep.
    """
    import os
    feature_set = os.environ.get("V9_FEATURE_SET", "v9b")
    if feature_set not in ("v9b", "v9c"):
        raise ValueError(
            f"V9_FEATURE_SET={feature_set!r} invalid; must be 'v9b' or 'v9c'"
        )

    print("=" * 80)
    print(f"V9 UPSET-WEIGHT SWEEP (feature_set={feature_set})")
    print(f"  Grid: {len(GRID)} cells, "
          f"W_UPSET in {W_UPSET_VALUES}, W_MISS in {W_MISS_VALUES}")
    print("=" * 80)

    pairwise_v4 = "output/pairwise_v4.csv"
    pairwise_v8 = "output/pairwise_v8.csv"
    seeds_csv = "data/raw/march-machine-learning-2026/MNCAATourneySeeds.csv"
    results_csv = "data/raw/march-machine-learning-2026/MNCAATourneyCompactResults.csv"
    slots_csv = "data/raw/march-machine-learning-2026/MNCAATourneySlots.csv"

    if feature_set == "v9b":
        out_dir = "output/v9_sweep"
        results_csv_path = "output/v9_sweep_results.csv"
    else:  # v9c
        out_dir = "output/v9c_sweep"
        results_csv_path = "output/v9c_sweep_results.csv"

    df = run_sweep(
        grid=GRID,
        pairwise_v4_csv=pairwise_v4,
        results_csv=results_csv,
        seeds_csv=seeds_csv,
        out_dir=out_dir,
        results_csv_path=results_csv_path,
        slots_csv=slots_csv,
        feature_set=feature_set,
    )

    # Summary table.
    print("\nResults sorted by total_brkt_pts (descending):")
    print(df.to_string(index=False))

    # v8 baseline + anchor-cell sanity gate.
    from src.score_chalk_brackets import score_pairwise_path
    v8_total = float(score_pairwise_path(pairwise_v8)["total_pts"])
    anchor_row = df[(df["w_upset"] == 1.0) & (df["w_miss"] == 0.0)].iloc[0]
    anchor_total = float(anchor_row["total_brkt_pts"])
    delta = anchor_total - v8_total
    print(f"\nv8 baseline:   {v8_total:>8.1f} pts")
    print(f"anchor (1, 0): {anchor_total:>8.1f} pts (delta {delta:+.2f})")
    if abs(delta) > 5.0:
        print("WARNING: anchor cell does not reproduce v8 within 5 pts; "
              "sweep results may be invalid -- inspect per-game LL/Acc to "
              "confirm trainer is sane before trusting cell rankings.")
    else:
        print(f"Anchor cell reproduces v8 within 5 pts -- sweep is valid. "
              f"(feature_set={feature_set} differs from v8 in features and "
              "may produce small chalk-pick boundary deltas at uniform weights.)")

    # Winner check (+10 bar).
    best = df.iloc[0]
    best_delta = float(best["total_brkt_pts"]) - v8_total
    print(f"\nbest cell:     W_UPSET={best['w_upset']}, "
          f"W_MISS={best['w_miss']}, "
          f"total_brkt_pts={best['total_brkt_pts']:.1f}, "
          f"delta vs v8={best_delta:+.2f}")
    if best_delta > 10.0:
        print(f"WINNER: best cell beats v8 by {best_delta:.1f} pts (> +10).")
    else:
        print(f"NO WINNER: best cell delta {best_delta:+.2f} pts (bar +10).")
```

Note: `import os` moved inline at the top of `main()` to keep it adjacent to its only caller. The existing top-of-file imports are unchanged.

- [ ] **Step 2: Smoke test the env-var read with a dry validation**

Run: `V9_FEATURE_SET=invalid python -c "from src.sweep_v9_weights import main; main()" 2>&1 | head -3`
Expected: Stack trace with `ValueError: V9_FEATURE_SET='invalid' invalid; must be 'v9b' or 'v9c'`.

If on Windows bash and env-var-prefix syntax is finicky, use:
```sh
export V9_FEATURE_SET=invalid && python -c "from src.sweep_v9_weights import main; main()" 2>&1 | head -3
unset V9_FEATURE_SET
```

- [ ] **Step 3: Commit**

```bash
git add src/sweep_v9_weights.py
git commit -m "feat(v9-sweep): V9_FEATURE_SET env var routes outputs to v9c_sweep/

main() reads V9_FEATURE_SET (default 'v9b'), validates it in {'v9b',
'v9c'}, and switches out_dir + results_csv_path so v9-C artifacts go
to output/v9c_sweep/ and output/v9c_sweep_results.csv. v9-B paths
are unchanged when the env var is unset or 'v9b'.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: Full pytest gate

**Files:**
- None modified. Verification step.

- [ ] **Step 1: Run the full test suite**

Run: `pytest -v`
Expected: All tests PASS. State the count in the output. Per CLAUDE.md, this is the gate before any experimental run.

- [ ] **Step 2: If any failure, halt and debug**

Do not proceed to Task 6 until pytest is green. Failures here likely indicate the parameterization broke a v9-B code path.

---

## Task 6: v9-B regression check (bit-identical to PR 8)

**Files:**
- None modified. Verification step.
- Reads: `output/v9_sweep_results.csv` (committed on main from PR 8) and the new run output.

- [ ] **Step 1: Re-run the v9-B sweep with the parameterized code**

Run: `V9_FEATURE_SET=v9b python src/sweep_v9_weights.py 2>&1 | tee output/v9b_repro_run.log`
Expected runtime: ~5 minutes. The script should print "feature_set=v9b" in the banner.

This overwrites `output/v9_sweep_results.csv` and the per-cell CSVs in `output/v9_sweep/` -- both are checked into main from PR 8, so they can be inspected via git after to confirm bit-identity.

- [ ] **Step 2: Diff the new results CSV against PR 8's (read PR 8 via git, no temp file)**

Run:
```bash
python -c "
import io, subprocess, pandas as pd
pr8_bytes = subprocess.check_output(['git', 'show', 'main:output/v9_sweep_results.csv'])
a = pd.read_csv(io.BytesIO(pr8_bytes)).sort_values(['w_upset','w_miss']).reset_index(drop=True)
b = pd.read_csv('output/v9_sweep_results.csv').sort_values(['w_upset','w_miss']).reset_index(drop=True)
cols = ['w_upset','w_miss','total_brkt_pts','ll_loso_weighted_mean','acc_loso_weighted_mean']
diff = (a[cols] - b[cols]).abs()
print('max abs diff per column:')
print(diff.max())
"
```

Expected: All max abs diffs are 0.0 (or below 1e-9 for float jitter). xgboost has fixed seed=42 and PR 7 already verified bit-reproducibility on a re-run.

- [ ] **Step 3: If any column differs materially, halt and debug**

Materially = `total_brkt_pts` differs by > 0.1 pts in any cell, or LL/Acc differs by > 1e-6. The parameterization is supposed to be a no-op for v9-B; a difference indicates an unintended behavior change.

If diffs are zero, do NOT commit `output/v9_sweep_results.csv` (it's bit-identical to main; no point churning git). Run `git checkout -- output/v9_sweep_results.csv output/v9_sweep/` to revert the regenerated artifacts.

- [ ] **Step 4: Note the result inline in the eventual findings note**

No commit at this step. The repro evidence will be referenced in the findings note (Task 8).

---

## Task 7: Run the v9-C sweep

**Files:**
- Create: `output/v9c_sweep/pairwise_v9_WU{u}_WM{m}.csv` (15 files)
- Create: `output/v9c_sweep_results.csv`
- Create: `output/v9c_sweep_run.log`

- [ ] **Step 1: Run the v9-C sweep end-to-end**

Run: `V9_FEATURE_SET=v9c python src/sweep_v9_weights.py 2>&1 | tee output/v9c_sweep_run.log`
Expected runtime: ~5 minutes. The banner should print `feature_set=v9c` and outputs should land in `output/v9c_sweep/`.

- [ ] **Step 2: Verify the sweep completed**

```bash
ls output/v9c_sweep/ | wc -l    # expect 15 per-cell CSVs
test -f output/v9c_sweep_results.csv && echo "results CSV exists"
grep -E "(WINNER|NO WINNER|WARNING)" output/v9c_sweep_run.log
```

Expected: 15 CSV files, results CSV exists, log shows the anchor verdict and winner verdict lines.

- [ ] **Step 3: Inspect the anchor cell delta**

Run: `python -c "import pandas as pd; df = pd.read_csv('output/v9c_sweep_results.csv'); anchor = df[(df.w_upset == 1.0) & (df.w_miss == 0.0)].iloc[0]; print(f'anchor total_brkt_pts: {anchor.total_brkt_pts:.1f}, LL: {anchor.ll_loso_weighted_mean:.4f}, Acc: {anchor.acc_loso_weighted_mean:.3f}')"`

Expected: LL ~0.4323, Acc ~0.807 (matching v8 to 3 decimals). brkt pts may differ from v8's 2670 by more than 5 -- if so, document the drift in the findings; do not halt unless LL/Acc are out of band.

- [ ] **Step 4: Identify the v9-C winner cell**

Run: `python -c "import pandas as pd; df = pd.read_csv('output/v9c_sweep_results.csv').sort_values('total_brkt_pts', ascending=False); print(df.head(5).to_string(index=False))"`

Note the top cell's `(w_upset, w_miss)` and `total_brkt_pts`. Compute `delta_v8 = total_brkt_pts - 2670` and `delta_v9b = total_brkt_pts - 2690` (PR 8's v9-B winner total).

- [ ] **Step 5: Per-season decomposition for the v9-C winner**

Identify the v9-C winner pairwise CSV (e.g., `output/v9c_sweep/pairwise_v9_WU1.25_WM0.00.csv` if that wins). Compare per-season bracket totals to v8:

```python
import pandas as pd
from src.score_chalk_brackets import score_pairwise_path

winner_csv = "output/v9c_sweep/pairwise_v9_WU<wu>_WM<wm>.csv"  # FILL IN actual winner
v8_csv = "output/pairwise_v8.csv"

w = score_pairwise_path(winner_csv)
v8 = score_pairwise_path(v8_csv)

w_per_season = w["per_season"]  # list of dicts; verify in score_chalk_brackets
v8_per_season = v8["per_season"]
# Print per-season comparison
import json
print(json.dumps({"v9c_winner_per_season": w_per_season,
                  "v8_per_season": v8_per_season}, indent=2))
```

If `score_pairwise_path` does not expose `per_season`, read it directly: `score_chalk_brackets.py` (PR 8 added per-season tracking; check the function signature first via `python -c "from src.score_chalk_brackets import score_pairwise_path; help(score_pairwise_path)"`). If unavailable, write a one-off loop calling `score_pairwise_path` per season slice -- the helper exists from PR 8 and findings notes use it.

Capture the per-season list (season, v8 pts, v9c pts, delta) as the table for Task 8. Note total wins / losses / ties.

- [ ] **Step 6: F4/E8 accuracy lens (per the spec's decision matrix)**

Compute v9-C winner's F4 and E8 chalk accuracy and compare to v8's. The PR 6 v9 findings note (`docs/notes/2026-04-30-upset-detection-v9.md`) shows the per-round accuracy table format -- reproduce that for v9-C winner vs v8. If `score_pairwise_path` exposes per-round accuracy, use it; otherwise reuse the same approach the PR 6 note used.

This number determines whether a +10 to +25 winner qualifies for the "F4/E8 lens" swap-in path (decision matrix row 2) or just the "marginal candidate" path (row 3).

- [ ] **Step 7: Commit the run artifacts**

```bash
git add output/v9c_sweep/ output/v9c_sweep_results.csv output/v9c_sweep_run.log
git commit -m "feat(v9c): 15-cell sweep results

Captures all 15 per-cell pairwise CSVs, sorted results table, and
driver log for the v9-C 5-feature variant. Findings note next.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 8: Write findings note

**Files:**
- Create: `docs/notes/2026-05-01-v9c-feature-stripped.md`

- [ ] **Step 1: Draft the findings note**

Use PR 8's `docs/notes/2026-05-01-v9-round-fix.md` as the template (read it first to mirror structure). Sections to include:

1. **Header.** Spec / Plan / PR 8 findings link.
2. **Verdict.** One paragraph: WINNER / MARGINAL WINNER / NEGATIVE per the decision matrix in the spec.
3. **Numbers.** v8 baseline (2670), v9-B winner (2690 at WU=1.25 WM=0.0), v9-C winner cell + total + delta vs v8 + delta vs v9-B-at-same-cell.
4. **Sweep results table.** All 15 cells with `w_upset, w_miss, brkt, LL, Acc, dv8, dv9b` columns. Sorted by brkt desc.
5. **Per-season decomposition** of v9-C winner vs v8 (table from Task 7 step 5).
6. **F4/E8 accuracy comparison** (from Task 7 step 6).
7. **Active ingredient analysis.** Was the winning weight pattern dominated by W_UPSET or W_MISS? (Compare to PR 7's W_MISS-only and PR 8's W_UPSET-only winners.)
8. **Recommendation.** Per decision matrix: swap candidate, marginal candidate, or negative result. State explicitly.
9. **Caveats.** Anchor delta, multiple-comparisons (now best of 30+ cells across PR 7 / PR 8 / this), v9-A vs v9-C feature comparison nuance.
10. **Artifacts.** List of files written.

Use straight ASCII only (no em-dashes, smart quotes, etc. -- CLAUDE.md rule).

- [ ] **Step 2: Verify ASCII compliance**

Run: `python -c "open('docs/notes/2026-05-01-v9c-feature-stripped.md', encoding='utf-8').read().encode('ascii')" && echo "ASCII OK"`
Expected: prints `ASCII OK`. If it errors, find and replace the offending character.

- [ ] **Step 3: Commit**

```bash
git add docs/notes/2026-05-01-v9c-feature-stripped.md
git commit -m "docs: v9-C feature-stripped findings

[Verdict one-liner here -- e.g. 'MARGINAL: v9-C winner +X vs v8',
'NEGATIVE: v9-C does not beat v8', or 'WINNER: v9-C beats v8 by Y;
swap candidate'.]

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 9: Update TODO.md

**Files:**
- Modify: `TODO.md`

- [ ] **Step 1: Add v9-C to the Done section**

Append a new entry to `TODO.md`'s Done section (between the existing v9-B round-fix entry and the Architecture Rethink section), in the same prose style as existing entries. Include:
- Date.
- One-sentence verdict.
- Active ingredient (W_UPSET vs W_MISS dominance).
- Decision (swap / marginal / negative) and reason from the decision matrix.
- Link to the findings note: `docs/notes/2026-05-01-v9c-feature-stripped.md`.

- [ ] **Step 2: Update the Active queue ordering**

Per the spec's decision matrix:

- If v9-C is a **swap candidate** (>+25 vs v8 OR clear F4/E8 win): leave queue untouched (#1 = ensemble); add a follow-up note "v9-C swap commit" as a separate Done entry once the production swap commit lands. Note the swap was not auto-applied here.
- If v9-C is **marginal or negative**: ensure the v9 thread is explicitly closed in the queue framing. Existing queue already lists "Ensemble of model classes" as #1 and notes "Promoted to position #1 after upset-detection v9 LOSE" -- update that line to "Promoted to position #1 after upset-detection v9 / v9-B / v9-C all marginal-or-worse" or similar.

- [ ] **Step 3: Verify ASCII compliance**

Run: `python -c "open('TODO.md', encoding='utf-8').read().encode('ascii')" && echo "ASCII OK"`
Expected: prints `ASCII OK`.

- [ ] **Step 4: Commit**

```bash
git add TODO.md
git commit -m "docs: TODO update for v9-C outcome

[Verdict one-liner.]

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 10: Open PR

**Files:**
- None modified.

- [ ] **Step 1: Push the branch and open a PR**

```bash
git push -u origin feat/v9-b-followup
gh pr create --title "v9-C feature-stripped variant + 15-cell sweep" --body "$(cat <<'EOF'
## Summary
- Parameterizes train_upset_model.py and sweep_v9_weights.py with feature_set kwarg
- Adds v9-C 5-feature variant (drops v4_confidence + is_a_higher_seed; keeps round)
- Re-runs PR 8's identical 15-cell sweep on v9-C
- Documents findings + updates TODO

## Test plan
- [ ] pytest -v passes
- [ ] V9_FEATURE_SET=v9b sweep produces results CSV bit-identical to PR 8's
- [ ] V9_FEATURE_SET=v9c sweep completes; anchor cell within 5 pts of v8 (or LL/Acc match v8 to 3 decimals)
- [ ] Findings note documents winner, per-season decomposition, F4/E8 lens, recommendation

Spec: docs/superpowers/specs/2026-05-01-v9c-feature-stripped-design.md
Plan: docs/superpowers/plans/2026-05-01-v9c-feature-stripped.md
Findings: docs/notes/2026-05-01-v9c-feature-stripped.md

Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

Return the PR URL when done.
