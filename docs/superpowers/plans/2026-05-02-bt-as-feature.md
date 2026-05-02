# BT-as-Feature for v9-C Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `p_bt_stage1` (from `output/pairwise_bt.csv`) as a 6th input feature to v9-C's upset-aware stage-2 model under `feature_set='v9d'`. Gate the 15-cell sweep on a cheap pre-sweep falsification check (does v9-D@(1.0, 0.0) beat v9-C@(1.0, 0.0) on per-game LL by >= 0.001?). If gate clears, run the sweep and compare per-cell bracket points to v9-C's production cell (2713 brkt pts at WU=1.25, WM=0.0).

**Architecture:** Additive extension. `load_per_game_data_with_upset` accepts an optional `pairwise_bt_csv`; `upset_features` learns a new `'v9d'` selector that returns 6 columns; `build_v9_pairwise` threads the BT csv through the apply-time per-pair lookup; `sweep_v9_weights.py` accepts `V9_FEATURE_SET=v9d`. Existing v9-A/B/C code paths are untouched. New module `src/diagnose_v9d.py` is the pre-sweep gate, mirroring `src/diagnose_bt_vs_v4.py` in shape.

**Tech Stack:** Python 3.11+, scikit-learn (`log_loss`), xgboost (existing trainer), numpy, pandas, pytest. No new top-level dependencies.

**Spec:** `docs/superpowers/specs/2026-05-02-bt-as-feature-design.md`

**Predecessor (rejected, frozen artifact reused here):** `docs/notes/2026-05-01-bayesian-stage1.md`. The committed `output/pairwise_bt.csv` (22 LOSO seasons, 48,465 pairs, schema `season,team_a,team_b,p_a_wins`) is reused as-is.

**Reference reads (skim before starting):**
- `src/train_upset_model.py` -- the file being extended; pay attention to:
  - `load_per_game_data_with_upset` (signature `(pairwise_csv, results_csv, seeds_csv)` returning per-game DataFrame; rows are A/B-symmetric pairs of `(W, L)` and `(L, W)`).
  - `upset_features(df, feature_set='v9b')` -- column-order convention for v9-B (7 cols) and v9-C (5 cols); v9-D is a third branch.
  - `build_v9_pairwise` -- the apply-time builder; the per-pair `apply_df` block is where BT lookup must inject the `p_bt` column.
- `src/sweep_v9_weights.py` -- the harness; `run_single_cell` and `main`'s `V9_FEATURE_SET` switch.
- `src/diagnose_bt_vs_v4.py` -- shape reference for the new `diagnose_v9d.py`. Note the `compute_diagnostic` / `check_gate` / `print_report` / `main` decomposition; mirror it.
- `tests/test_sweep_v9_weights.py` -- existing fixture pattern (`_write_minimal_inputs`) is reusable for v9-D tests.
- `tests/test_train_bt_stage1.py` -- TDD pattern (failing test, ImportError, implement, pass) that this plan follows.
- `output/pairwise_bt.csv` (head): `season,team_a,team_b,p_a_wins` -- 48,466 lines (header + 48,465 pairs).
- `output/v9c_sweep/pairwise_v9_WU1.25_WM0.00.csv` -- the v9-C production-cell artifact; baseline for the bracket-points comparison in Task 8.
- `output/v9c_sweep/pairwise_v9_WU1.00_WM0.00.csv` -- the v9-C anchor cell; baseline for the trainer-harness anchor check in Task 8 (PASS branch).

**Verification gates (CLAUDE.md "Forced Verification"):**
After every code-level task: `pytest -v` for the touched test file(s) must pass. Task 6 runs the full ingest/feature/integration suite once before any real-data run. After data-generation tasks (7, 8): inspect schema and row count of the produced artifact before committing.

**ASCII discipline (CLAUDE.md):** All files written must be ASCII-only. After every Write/Edit, run:
```bash
python -c "open('PATH', encoding='utf-8').read().encode('ascii')" && echo "ASCII OK"
```

---

### Task 1: Extend `load_per_game_data_with_upset` to optionally join `p_bt`

**Why:** The per-game DataFrame returned by this loader is the upstream input to `upset_features`, `compute_sample_weights`, `fit_upset_model`, `double_loso_eval`. Adding `p_bt` here means every consumer downstream gets it for free as long as the column is present.

**Files:**
- Modify: `src/train_upset_model.py` (function `load_per_game_data_with_upset`)
- Create: `tests/test_train_upset_model.py` (new test file)

- [ ] **Step 1: Write failing tests for the BT-join extension**

Create `tests/test_train_upset_model.py` with these tests:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_train_upset_model.py -v`
Expected: 3 failures -- the first test passes (`p_bt` not in columns is trivially true today), the other two fail with `TypeError: load_per_game_data_with_upset() got an unexpected keyword argument 'pairwise_bt_csv'`.

Note: if Step 2 shows the first test failing too (e.g., due to dtype or row-count mismatch), debug before proceeding -- it pins existing v9-A/B/C behavior.

- [ ] **Step 3: Extend `load_per_game_data_with_upset`**

Open `src/train_upset_model.py`. Modify the function signature and body. The current signature is:

```python
def load_per_game_data_with_upset(
    pairwise_csv: str, results_csv: str, seeds_csv: str
) -> pd.DataFrame:
```

Change it to:

```python
def load_per_game_data_with_upset(
    pairwise_csv: str, results_csv: str, seeds_csv: str,
    pairwise_bt_csv: str | None = None,
) -> pd.DataFrame:
```

Update the docstring's "Each row" line to mention the optional `p_bt` column:

```python
    """Build per-played-game training rows for v9.

    Each row: (season, team_a, team_b, p_stage1, seed_a, seed_b,
              abs_seed_diff, upset, round, label) plus, when
              pairwise_bt_csv is provided, p_bt -- the Bradley-Terry
              winner-perspective probability for the (team_a, team_b)
              pair, oriented to match p_stage1 (i.e., for the (W, L)
              row p_bt is the BT prob of W winning; for the (L, W)
              symmetric row p_bt is the BT prob of L winning =
              1 - the (W, L) value).
    ... (rest of docstring unchanged) ...
    """
```

After the existing `pw_lookup = ...` line, add the BT lookup if requested:

```python
    bt_lookup: dict | None = None
    if pairwise_bt_csv is not None:
        bt = pd.read_csv(pairwise_bt_csv)
        bt = bt.drop_duplicates(["season", "team_a", "team_b"], keep="last")
        bt_lookup = {(int(s), int(a), int(b)): float(p)
                     for s, a, b, p in zip(bt.season, bt.team_a, bt.team_b, bt.p_a_wins)}
```

Inside the row-building loop, after the `if p_a_wins is None: continue` line, add a parallel BT lookup with the same skip-on-miss behavior:

```python
        if bt_lookup is not None:
            p_bt_a_wins = bt_lookup.get((season, a, b))
            if p_bt_a_wins is None:
                continue
            p_bt_w = p_bt_a_wins if a == w else (1.0 - p_bt_a_wins)
        else:
            p_bt_w = None
```

Then inside both `rows.append(...)` blocks, add the `p_bt` key only when `bt_lookup` is set. The cleanest way:

```python
        win_row = {
            "season": season, "team_a": w, "team_b": l,
            "p_stage1": p_w,
            "seed_a": seed_w, "seed_b": seed_l,
            "abs_seed_diff": abs(seed_w - seed_l),
            "upset": is_upset,
            "round": day_to_round(int(g["DayNum"])),
            "label": 1,
        }
        los_row = {
            "season": season, "team_a": l, "team_b": w,
            "p_stage1": 1.0 - p_w,
            "seed_a": seed_l, "seed_b": seed_w,
            "abs_seed_diff": abs(seed_w - seed_l),
            "upset": is_upset,
            "round": day_to_round(int(g["DayNum"])),
            "label": 0,
        }
        if bt_lookup is not None:
            win_row["p_bt"] = p_bt_w
            los_row["p_bt"] = 1.0 - p_bt_w
        rows.append(win_row)
        rows.append(los_row)
```

(Replace the existing two inline `rows.append({...})` calls with this construction.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_train_upset_model.py -v`
Expected: 3 PASS.

- [ ] **Step 5: Verify ASCII compliance and existing v9-A/B/C tests still pass**

```bash
python -c "open('src/train_upset_model.py', encoding='utf-8').read().encode('ascii')" && echo "ASCII OK"
pytest tests/test_sweep_v9_weights.py -v
```

Expected: ASCII OK, all existing sweep tests pass (the optional `pairwise_bt_csv` arg is a no-op when omitted, so the v9-A/B/C path is unchanged).

- [ ] **Step 6: Commit**

```bash
git add src/train_upset_model.py tests/test_train_upset_model.py
git commit -m "$(cat <<'EOF'
feat(v9d): load_per_game_data_with_upset accepts pairwise_bt_csv

When provided, joins p_bt from output/pairwise_bt.csv on
(season, team_a, team_b). Per-game (W, L) row gets the BT
winner-perspective probability; the symmetric (L, W) row gets
1 - that value. Rows missing from the BT lookup are dropped
(same skip-on-miss behavior as v4 lookups). When the arg is
omitted, the returned frame has no p_bt column -- v9-A/B/C
consumers are unchanged.

First step toward v9-D: BT as input feature to v9-C's
upset-aware stage-2 model. Spec:
docs/superpowers/specs/2026-05-02-bt-as-feature-design.md.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: Extend `upset_features` to support `feature_set='v9d'`

**Why:** v9-D adds `p_bt` as the 6th column of the input feature matrix. The function dispatches on `feature_set`; this task adds the third branch.

**Files:**
- Modify: `src/train_upset_model.py` (function `upset_features`)
- Modify: `tests/test_train_upset_model.py` (additional tests)

- [ ] **Step 1: Write failing tests**

Append to `tests/test_train_upset_model.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_train_upset_model.py -v -k "v9d or unknown"`
Expected: `test_upset_features_v9d_shape_and_columns` and `test_upset_features_v9d_missing_pbt_raises` fail with `ValueError: unknown feature_set 'v9d'`. The unknown-set test passes (already covered by existing logic).

- [ ] **Step 3: Implement the v9d branch**

In `src/train_upset_model.py`, modify `upset_features`. Current body:

```python
def upset_features(df: pd.DataFrame, feature_set: str = "v9b") -> np.ndarray:
    """..."""
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

Update the docstring to mention v9d, then update the validation and add the v9d branch:

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
      "v9d" (6 features): v9c plus p_bt (Bradley-Terry winner-perspective
        probability from output/pairwise_bt.csv, joined upstream by
        load_per_game_data_with_upset). Raises ValueError if the input
        frame lacks the 'p_bt' column.
    """
    if feature_set not in ("v9b", "v9c", "v9d"):
        raise ValueError(
            f"unknown feature_set {feature_set!r}; "
            "must be 'v9b', 'v9c', or 'v9d'"
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
    if feature_set == "v9d":
        if "p_bt" not in df.columns:
            raise ValueError(
                "feature_set='v9d' requires a 'p_bt' column on the input "
                "frame; pass pairwise_bt_csv to load_per_game_data_with_upset"
            )
        p_bt = df["p_bt"].values.astype(float)
        return np.column_stack([p, sa, sb, diff, rnd, p_bt])
    return np.column_stack([p, sa, sb, diff, rnd])
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_train_upset_model.py -v`
Expected: all tests pass (3 from Task 1 + 3 from Task 2 = 6 total).

- [ ] **Step 5: Verify ASCII + existing tests still pass**

```bash
python -c "open('src/train_upset_model.py', encoding='utf-8').read().encode('ascii')" && echo "ASCII OK"
pytest tests/test_sweep_v9_weights.py -v
```

Expected: ASCII OK, all sweep tests pass (the v9-A/B/C branches are unchanged).

- [ ] **Step 6: Commit**

```bash
git add src/train_upset_model.py tests/test_train_upset_model.py
git commit -m "$(cat <<'EOF'
feat(v9d): upset_features accepts feature_set='v9d'

Returns 6-column matrix [p_stage1, seed_a, seed_b, abs_seed_diff,
round, p_bt]. Raises ValueError if 'p_bt' is absent on the input
frame (load_per_game_data_with_upset must have been called with
pairwise_bt_csv set). v9-A/B/C branches unchanged.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: Extend `build_v9_pairwise` to thread `pairwise_bt_csv` at apply time

**Why:** `build_v9_pairwise` is the apply-time builder: trained model -> per-LOSO-season -> 48,465 pairwise predictions. v9-D needs `p_bt` looked up for every pair in the apply grid (not just played games), or the trained model can't be invoked. The lookup happens once per season inside the per-season loop.

**Files:**
- Modify: `src/train_upset_model.py` (function `build_v9_pairwise`)
- Modify: `tests/test_train_upset_model.py` (additional test)

- [ ] **Step 1: Write failing test**

Append to `tests/test_train_upset_model.py`:

```python
def test_build_v9_pairwise_v9d_threads_pbt(tmp_path):
    """When called with feature_set='v9d' and pairwise_bt_csv, the
    apply-time builder writes a pairwise CSV in the canonical schema
    (season, team_a, team_b, p_a_wins) -- proving that the per-pair
    BT lookup at apply time works without raising on missing columns.
    """
    from src.train_upset_model import (
        build_v9_pairwise, load_per_game_data_with_upset,
    )

    # Two seasons, two teams each, identical seeding pattern.
    pw_v4 = tmp_path / "pw_v4.csv"
    pw_bt = tmp_path / "pw_bt.csv"
    seeds = tmp_path / "seeds.csv"
    results = tmp_path / "results.csv"
    slots = tmp_path / "slots.csv"
    out_csv = tmp_path / "pw_v9d.csv"

    _write_pairwise(pw_v4, [
        (2022, 1, 2, 0.7), (2023, 1, 2, 0.55),
    ])
    _write_pairwise(pw_bt, [
        (2022, 1, 2, 0.6), (2023, 1, 2, 0.5),
    ])
    _write_seeds(seeds, [
        (2022, "W01", 1), (2022, "W08", 2),
        (2023, "W01", 1), (2023, "W08", 2),
    ])
    _write_results(results, [
        (2022, 136, 1, 2), (2023, 136, 2, 1),  # alternating winners
    ])
    pd.DataFrame({
        "Season": [2022, 2023],
        "Slot":   ["R1W1", "R1W1"],
        "StrongSeed": ["W01", "W01"],
        "WeakSeed":   ["W08", "W08"],
    }).to_csv(slots, index=False)

    per_game = load_per_game_data_with_upset(
        str(pw_v4), str(results), str(seeds),
        pairwise_bt_csv=str(pw_bt),
    )

    build_v9_pairwise(
        per_game, str(pw_v4), str(seeds), str(out_csv),
        slots_csv=str(slots),
        feature_set="v9d",
        pairwise_bt_csv=str(pw_bt),
    )

    out = pd.read_csv(out_csv)
    assert list(out.columns) == ["season", "team_a", "team_b", "p_a_wins"]
    assert (out["team_a"] < out["team_b"]).all()
    # Two seasons, one pair each -> 2 output rows.
    assert len(out) == 2


def test_build_v9_pairwise_v9d_requires_pairwise_bt_csv(tmp_path):
    """feature_set='v9d' without pairwise_bt_csv raises -- the apply-time
    grid has no p_bt column otherwise.
    """
    from src.train_upset_model import (
        build_v9_pairwise, load_per_game_data_with_upset,
    )

    pw_v4 = tmp_path / "pw_v4.csv"
    pw_bt = tmp_path / "pw_bt.csv"
    seeds = tmp_path / "seeds.csv"
    results = tmp_path / "results.csv"
    slots = tmp_path / "slots.csv"
    out_csv = tmp_path / "pw_v9d.csv"

    _write_pairwise(pw_v4, [(2022, 1, 2, 0.7)])
    _write_pairwise(pw_bt, [(2022, 1, 2, 0.6)])
    _write_seeds(seeds, [(2022, "W01", 1), (2022, "W08", 2)])
    _write_results(results, [(2022, 136, 1, 2)])
    pd.DataFrame({
        "Season": [2022], "Slot": ["R1W1"],
        "StrongSeed": ["W01"], "WeakSeed": ["W08"],
    }).to_csv(slots, index=False)

    per_game = load_per_game_data_with_upset(
        str(pw_v4), str(results), str(seeds),
        pairwise_bt_csv=str(pw_bt),
    )

    with pytest.raises(ValueError, match="pairwise_bt_csv"):
        build_v9_pairwise(
            per_game, str(pw_v4), str(seeds), str(out_csv),
            slots_csv=str(slots),
            feature_set="v9d",  # no pairwise_bt_csv
        )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_train_upset_model.py -v -k "v9d_threads_pbt or v9d_requires"`
Expected: both tests fail. The first fails with `TypeError: build_v9_pairwise() got an unexpected keyword argument 'pairwise_bt_csv'`. The second fails with the same TypeError (no `pairwise_bt_csv` arg yet).

- [ ] **Step 3: Extend `build_v9_pairwise`**

In `src/train_upset_model.py`, modify `build_v9_pairwise`. Current signature:

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

Change to:

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
    pairwise_bt_csv: str | None = None,
) -> None:
```

Add a guard near the top of the function body (before the existing CSV reads):

```python
    if feature_set == "v9d" and pairwise_bt_csv is None:
        raise ValueError(
            "feature_set='v9d' requires pairwise_bt_csv to be provided"
        )
```

After the existing `seeds = pd.read_csv(seeds_csv)` block, build the BT lookup once if needed:

```python
    bt_lookup: dict | None = None
    if pairwise_bt_csv is not None:
        bt = pd.read_csv(pairwise_bt_csv)
        bt = bt.drop_duplicates(["season", "team_a", "team_b"], keep="last")
        bt_lookup = {(int(s), int(a), int(b)): float(p)
                     for s, a, b, p in zip(bt.season, bt.team_a, bt.team_b, bt.p_a_wins)}
```

Inside the per-season loop, after `apply_df["round"] = apply_df.apply(_round_for_pair, axis=1)` and before `apply_df["p_stage1"] = apply_df["p_a_wins"]`, add the per-pair BT lookup when needed:

```python
            if bt_lookup is not None:
                apply_df["p_bt"] = [
                    bt_lookup.get((int(s), int(min(a, b)), int(max(a, b))))
                    for s, a, b in zip(apply_df["season"], apply_df["team_a"], apply_df["team_b"])
                ]
                # Pairs missing from BT lookup are dropped from the apply
                # frame -- the trained model has no defined behavior on
                # NaN inputs and silent fallback would leak v4-pass-through
                # under a v9-D label.
                apply_df = apply_df[apply_df["p_bt"].notna()].copy()
                # apply_df already has team_a as min(a, b) by construction
                # of season_pw upstream, so p_bt is already in (a, b)
                # orientation -- no flip needed.
```

The existing `p_v9 = model.predict_proba(upset_features(apply_df, feature_set=feature_set))[:, 1]` line will then see the `p_bt` column and the `'v9d'` branch will pick it up.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_train_upset_model.py -v`
Expected: all 8 tests pass (3 from Task 1, 3 from Task 2, 2 from this task).

- [ ] **Step 5: Verify ASCII + existing tests still pass**

```bash
python -c "open('src/train_upset_model.py', encoding='utf-8').read().encode('ascii')" && echo "ASCII OK"
pytest tests/test_sweep_v9_weights.py -v
```

Expected: ASCII OK, all sweep tests pass (the new kwarg has default `None`, so the v9-A/B/C path is unchanged).

- [ ] **Step 6: Commit**

```bash
git add src/train_upset_model.py tests/test_train_upset_model.py
git commit -m "$(cat <<'EOF'
feat(v9d): build_v9_pairwise threads pairwise_bt_csv at apply time

When feature_set='v9d', requires pairwise_bt_csv. Builds the BT
lookup once outside the per-season loop, then injects p_bt into
the apply DataFrame before XGB scoring. Pairs missing from the BT
lookup are dropped (consistent with the per-game loader's
skip-on-miss semantics). v9-A/B/C apply paths are unchanged.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: Extend `sweep_v9_weights.py` to support `V9_FEATURE_SET=v9d`

**Why:** The sweep harness is the entry point for the 15-cell W_UPSET / W_MISS grid. Adding `'v9d'` here wires v9-D into the existing CI / CLI flow so `V9_FEATURE_SET=v9d python src/sweep_v9_weights.py` "just works" once the trainer extension above is in place.

**Files:**
- Modify: `src/sweep_v9_weights.py`
- Modify: `tests/test_sweep_v9_weights.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_sweep_v9_weights.py`:

```python
def _write_pairwise_bt(path, rows):
    """rows: list of (season, team_a, team_b, p_a_wins)."""
    df = pd.DataFrame(rows, columns=["season", "team_a", "team_b", "p_a_wins"])
    df.to_csv(path, index=False)


def test_run_single_cell_v9d_writes_pairwise(tmp_path):
    """run_single_cell with feature_set='v9d' + pairwise_bt_csv writes
    a pairwise CSV in the canonical schema and returns the metrics dict.
    """
    pw_path, seeds_path, results_path, slots_path = _write_minimal_inputs(tmp_path)
    pw_bt_path = tmp_path / "pw_bt.csv"
    # Match the (season, team_a, team_b) keys present in pw_v4 fixture.
    _write_pairwise_bt(pw_bt_path, [
        (2022, 1, 2, 0.6), (2022, 1, 3, 0.55),
        (2023, 1, 3, 0.45), (2023, 2, 3, 0.4),
    ])
    out_dir = tmp_path / "v9d_sweep"

    metrics = run_single_cell(
        w_upset=1.0, w_miss=0.0,
        pairwise_v4_csv=pw_path,
        results_csv=results_path,
        seeds_csv=seeds_path,
        out_dir=str(out_dir),
        slots_csv=slots_path,
        feature_set="v9d",
        pairwise_bt_csv=str(pw_bt_path),
    )

    pw_path_out = out_dir / "pairwise_v9_WU1.00_WM0.00.csv"
    assert pw_path_out.exists()
    out = pd.read_csv(pw_path_out)
    assert list(out.columns) == ["season", "team_a", "team_b", "p_a_wins"]
    assert (out["team_a"] < out["team_b"]).all()
    assert metrics["pairwise_csv"] == str(pw_path_out)


def test_run_single_cell_v9d_requires_pairwise_bt_csv(tmp_path):
    """feature_set='v9d' without pairwise_bt_csv raises before any work."""
    pw_path, seeds_path, results_path, slots_path = _write_minimal_inputs(tmp_path)
    out_dir = tmp_path / "v9d_sweep"

    with pytest.raises(ValueError, match="pairwise_bt_csv"):
        run_single_cell(
            w_upset=1.0, w_miss=0.0,
            pairwise_v4_csv=pw_path,
            results_csv=results_path,
            seeds_csv=seeds_path,
            out_dir=str(out_dir),
            slots_csv=slots_path,
            feature_set="v9d",
        )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_sweep_v9_weights.py -v -k "v9d"`
Expected: both tests fail with `TypeError: run_single_cell() got an unexpected keyword argument 'pairwise_bt_csv'`.

- [ ] **Step 3: Extend `run_single_cell` and `run_sweep`**

In `src/sweep_v9_weights.py`, modify `run_single_cell`. Current signature:

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

Change to:

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
    pairwise_bt_csv: str | None = None,
) -> dict:
```

Add a guard at the top of the function body:

```python
    if feature_set == "v9d" and pairwise_bt_csv is None:
        raise ValueError(
            "feature_set='v9d' requires pairwise_bt_csv to be provided"
        )
```

In the existing `load_per_game_data_with_upset(pairwise_v4_csv, results_csv, seeds_csv)` call, add the kwarg:

```python
    per_game = load_per_game_data_with_upset(
        pairwise_v4_csv, results_csv, seeds_csv,
        pairwise_bt_csv=pairwise_bt_csv,
    )
```

In the existing `build_v9_pairwise(...)` call, add the kwarg:

```python
    build_v9_pairwise(
        per_game, pairwise_v4_csv, seeds_csv, pairwise_csv_out,
        slots_csv=slots_csv,
        w_upset=w_upset, w_miss=w_miss,
        feature_set=feature_set,
        pairwise_bt_csv=pairwise_bt_csv,
    )
```

Then modify `run_sweep` similarly. Add `pairwise_bt_csv: str | None = None` to the signature, and pass it through to `run_single_cell` inside the loop:

```python
        m = run_single_cell(
            w_upset=w_upset, w_miss=w_miss,
            pairwise_v4_csv=pairwise_v4_csv,
            results_csv=results_csv,
            seeds_csv=seeds_csv,
            out_dir=out_dir,
            slots_csv=slots_csv,
            feature_set=feature_set,
            pairwise_bt_csv=pairwise_bt_csv,
        )
```

Finally, modify `main`. The current `main` reads `V9_FEATURE_SET` and gates on `('v9b', 'v9c')`. Update to allow `'v9d'`:

```python
    feature_set = os.environ.get("V9_FEATURE_SET", "v9b")
    if feature_set not in ("v9b", "v9c", "v9d"):
        raise ValueError(
            f"V9_FEATURE_SET={feature_set!r} invalid; "
            "must be 'v9b', 'v9c', or 'v9d'"
        )
```

Set output paths and `pairwise_bt_csv` for the v9d case:

```python
    if feature_set == "v9b":
        out_dir = "output/v9_sweep"
        results_csv_path = "output/v9_sweep_results.csv"
    elif feature_set == "v9c":
        out_dir = "output/v9c_sweep"
        results_csv_path = "output/v9c_sweep_results.csv"
    else:  # v9d
        out_dir = "output/v9d_sweep"
        results_csv_path = "output/v9d_sweep_results.csv"

    pairwise_bt_csv = "output/pairwise_bt.csv" if feature_set == "v9d" else None
```

Pass `pairwise_bt_csv` to `run_sweep`:

```python
    df = run_sweep(
        grid=GRID,
        pairwise_v4_csv=pairwise_v4,
        results_csv=results_csv,
        seeds_csv=seeds_csv,
        out_dir=out_dir,
        results_csv_path=results_csv_path,
        slots_csv=slots_csv,
        feature_set=feature_set,
        pairwise_bt_csv=pairwise_bt_csv,
    )
```

Update the print banner to mention v9-D when applicable -- the existing banner string `f"V9 UPSET-WEIGHT SWEEP (feature_set={feature_set})"` already covers it.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_sweep_v9_weights.py -v`
Expected: all sweep tests pass (existing ones plus 2 new v9-D tests).

- [ ] **Step 5: Verify ASCII**

```bash
python -c "open('src/sweep_v9_weights.py', encoding='utf-8').read().encode('ascii')" && echo "ASCII OK"
```

Expected: ASCII OK.

- [ ] **Step 6: Commit**

```bash
git add src/sweep_v9_weights.py tests/test_sweep_v9_weights.py
git commit -m "$(cat <<'EOF'
feat(v9d): sweep_v9_weights accepts V9_FEATURE_SET=v9d

Adds 'v9d' to the V9_FEATURE_SET env var dispatcher. Output paths
key off the choice: output/v9d_sweep/ and output/v9d_sweep_results.csv.
The sweep injects pairwise_bt_csv='output/pairwise_bt.csv' into
run_single_cell -> load_per_game_data_with_upset / build_v9_pairwise.
v9-B and v9-C paths unchanged.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 5: Create `src/diagnose_v9d.py` pre-sweep gate

**Why:** Single-clause falsification gate. If v9-D@(1.0, 0.0) doesn't beat v9-C@(1.0, 0.0) on weighted-mean per-game LL by >= 0.001, NO-GO without running the remaining 14 sweep cells. Mirrors `src/diagnose_bt_vs_v4.py` in shape so the operational pattern is consistent across experiments.

**Files:**
- Create: `src/diagnose_v9d.py`
- Create: `tests/test_diagnose_v9d.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_diagnose_v9d.py`:

```python
"""Unit tests for src/diagnose_v9d.py.

Gate function tests use plain-dict inputs (no real-data dependency).
A small integration test exercises compute_gate end-to-end on
synthetic fixtures to pin the call chain.
"""
import pandas as pd
import pytest


def _write_pairwise(path, rows):
    df = pd.DataFrame(rows, columns=["season", "team_a", "team_b", "p_a_wins"])
    df.to_csv(path, index=False)


def _write_seeds(path, rows):
    df = pd.DataFrame(rows, columns=["Season", "Seed", "TeamID"])
    df.to_csv(path, index=False)


def _write_results(path, rows):
    df = pd.DataFrame(rows, columns=["Season", "DayNum", "WTeamID", "LTeamID"])
    df.to_csv(path, index=False)


def test_check_gate_passes_when_headroom_above_threshold():
    """LL_v9c - LL_v9d >= GATE_LL_HEADROOM_MIN -> pass."""
    from src.diagnose_v9d import check_gate, GATE_LL_HEADROOM_MIN

    diag = {
        "ll_v9c": 0.45, "ll_v9d": 0.44,
        "headroom": 0.01, "threshold": GATE_LL_HEADROOM_MIN,
    }
    out = check_gate(diag)
    assert out["pass"] is True
    assert "headroom" in out["reason"].lower()


def test_check_gate_fails_when_headroom_at_threshold_minus_epsilon():
    """Headroom of exactly threshold - 0.0001 -> fail (strict >=)."""
    from src.diagnose_v9d import check_gate, GATE_LL_HEADROOM_MIN

    diag = {
        "ll_v9c": 0.45, "ll_v9d": 0.45 - (GATE_LL_HEADROOM_MIN - 0.0001),
        "headroom": GATE_LL_HEADROOM_MIN - 0.0001,
        "threshold": GATE_LL_HEADROOM_MIN,
    }
    out = check_gate(diag)
    assert out["pass"] is False


def test_check_gate_fails_when_v9d_is_worse():
    """Negative headroom -> fail."""
    from src.diagnose_v9d import check_gate, GATE_LL_HEADROOM_MIN

    diag = {
        "ll_v9c": 0.44, "ll_v9d": 0.45,
        "headroom": -0.01, "threshold": GATE_LL_HEADROOM_MIN,
    }
    out = check_gate(diag)
    assert out["pass"] is False


def test_check_gate_passes_at_threshold_exactly():
    """Headroom == threshold -> pass (strict >=)."""
    from src.diagnose_v9d import check_gate, GATE_LL_HEADROOM_MIN

    diag = {
        "ll_v9c": 0.45, "ll_v9d": 0.45 - GATE_LL_HEADROOM_MIN,
        "headroom": GATE_LL_HEADROOM_MIN,
        "threshold": GATE_LL_HEADROOM_MIN,
    }
    out = check_gate(diag)
    assert out["pass"] is True


def test_compute_gate_returns_expected_keys(tmp_path):
    """compute_gate runs end-to-end on synthetic inputs and returns
    a dict with all expected fields. Assertions on the LL values
    themselves are intentionally loose -- the unit test pins shape,
    not a magic number.
    """
    from src.diagnose_v9d import compute_gate

    pw_v4 = tmp_path / "pw_v4.csv"
    pw_bt = tmp_path / "pw_bt.csv"
    seeds = tmp_path / "seeds.csv"
    results = tmp_path / "results.csv"
    # 4 teams, 2 seasons, 2 games per season -- enough rows for
    # double_loso_eval to fit and predict.
    _write_pairwise(pw_v4, [
        (2022, 1, 2, 0.7), (2022, 3, 4, 0.6),
        (2023, 1, 2, 0.55), (2023, 3, 4, 0.5),
    ])
    _write_pairwise(pw_bt, [
        (2022, 1, 2, 0.6), (2022, 3, 4, 0.55),
        (2023, 1, 2, 0.5), (2023, 3, 4, 0.45),
    ])
    _write_seeds(seeds, [
        (2022, "W01", 1), (2022, "W08", 2), (2022, "X01", 3), (2022, "X08", 4),
        (2023, "W01", 1), (2023, "W08", 2), (2023, "X01", 3), (2023, "X08", 4),
    ])
    _write_results(results, [
        (2022, 136, 1, 2), (2022, 138, 3, 4),
        (2023, 136, 1, 2), (2023, 138, 3, 4),
    ])

    diag = compute_gate(
        pairwise_v4_csv=str(pw_v4),
        pairwise_bt_csv=str(pw_bt),
        results_csv=str(results),
        seeds_csv=str(seeds),
    )
    assert set(diag.keys()) >= {
        "n_games_v9c", "n_games_v9d",
        "ll_v9c", "ll_v9d", "headroom", "threshold",
    }
    # Same per-game frame underlies both evals -> n_games equal.
    assert diag["n_games_v9c"] == diag["n_games_v9d"]
    # LLs are both finite, non-negative.
    assert diag["ll_v9c"] >= 0 and diag["ll_v9d"] >= 0
    assert diag["threshold"] == 0.001
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_diagnose_v9d.py -v`
Expected: 5 ImportError failures (`No module named 'src.diagnose_v9d'`).

- [ ] **Step 3: Implement `src/diagnose_v9d.py`**

Create `src/diagnose_v9d.py`:

```python
"""Pre-sweep falsification gate for v9-D (BT-as-feature).

Spec: docs/superpowers/specs/2026-05-02-bt-as-feature-design.md

Question: does v9-D@(1.0, 0.0) beat v9-C@(1.0, 0.0) on weighted-mean
per-game log loss across 22 LOSO seasons by at least
GATE_LL_HEADROOM_MIN?

If yes -> proceed to the 15-cell W_UPSET / W_MISS sweep with
V9_FEATURE_SET=v9d. If no -> stop, write findings note, NO-GO. Saves
the cost of running 14 additional sweep cells when the feature
isn't extracting meaningful signal even at uniform weights.

Mirrors src/diagnose_bt_vs_v4.py in shape (compute_gate /
check_gate / print_report / main / sys.exit nonzero on FAIL).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.train_upset_model import (
    double_loso_eval, load_per_game_data_with_upset,
)

DATA = Path("data/raw/march-machine-learning-2026")
DEFAULT_DIAGNOSTIC_OUT = "output/diag_v9d.json"

# Single-clause gate: v9-D's per-game LL must beat v9-C's by at least
# this much on the same per-game frame at uniform weights. A tighter
# threshold than the BT-vs-v4 ensemble gate (0.005) because this is
# a single-clause test, and a paired comparison (same per-game rows
# evaluated by both feature sets) cancels most variance.
GATE_LL_HEADROOM_MIN = 0.001


def _weighted_mean_ll(eval_df: pd.DataFrame) -> float:
    """Weighted-mean log loss across seasons, weighting by n_games."""
    n_total = float(eval_df["n_games"].sum())
    if n_total <= 0:
        return float("nan")
    return float((eval_df["ll_v9"] * eval_df["n_games"]).sum() / n_total)


def compute_gate(
    pairwise_v4_csv: str,
    pairwise_bt_csv: str,
    results_csv: str = str(DATA / "MNCAATourneyCompactResults.csv"),
    seeds_csv: str = str(DATA / "MNCAATourneySeeds.csv"),
) -> dict:
    """Compute the pre-sweep gate diagnostic.

    Loads the per-game data once with p_bt joined; runs double-LOSO eval
    twice (v9-c features = no p_bt; v9-d features = with p_bt) at
    uniform weights (W_UPSET=1.0, W_MISS=0.0); returns LL values,
    headroom, and the gate threshold.
    """
    per_game = load_per_game_data_with_upset(
        pairwise_v4_csv, results_csv, seeds_csv,
        pairwise_bt_csv=pairwise_bt_csv,
    )

    eval_v9c = double_loso_eval(
        per_game, w_upset=1.0, w_miss=0.0, feature_set="v9c"
    )
    eval_v9d = double_loso_eval(
        per_game, w_upset=1.0, w_miss=0.0, feature_set="v9d"
    )

    ll_v9c = _weighted_mean_ll(eval_v9c)
    ll_v9d = _weighted_mean_ll(eval_v9d)
    headroom = ll_v9c - ll_v9d  # positive = v9d beats v9c

    return {
        "n_games_v9c": int(eval_v9c["n_games"].sum()),
        "n_games_v9d": int(eval_v9d["n_games"].sum()),
        "ll_v9c": float(ll_v9c),
        "ll_v9d": float(ll_v9d),
        "headroom": float(headroom),
        "threshold": float(GATE_LL_HEADROOM_MIN),
    }


def check_gate(diag: dict) -> dict:
    """Single-clause: headroom >= threshold -> pass."""
    threshold = diag.get("threshold", GATE_LL_HEADROOM_MIN)
    if diag["headroom"] >= threshold:
        return {
            "pass": True,
            "reason": f"headroom {diag['headroom']:+.4f} >= {threshold}",
        }
    return {
        "pass": False,
        "reason": f"headroom {diag['headroom']:+.4f} < {threshold}",
    }


def print_report(diag: dict, gate: dict) -> None:
    print("=" * 70)
    print("v9-D PRE-SWEEP GATE (BT-as-feature)")
    print("=" * 70)
    print(f"  n games (v9c eval): {diag['n_games_v9c']}")
    print(f"  n games (v9d eval): {diag['n_games_v9d']}")
    print(f"\n  Per-game LL @ (W_UPSET=1.0, W_MISS=0.0), 22-season weighted mean:")
    print(f"    v9-C (5 features):       {diag['ll_v9c']:.4f}")
    print(f"    v9-D (6 features + BT):  {diag['ll_v9d']:.4f}")
    print(f"    headroom (v9c - v9d):    {diag['headroom']:+.4f}")
    print(f"    threshold:               {diag['threshold']:.4f}")
    print(f"\n=== VERDICT ===")
    if gate["pass"]:
        print(f"  GATE PASSED: {gate['reason']}")
        print(f"  -> Proceed to 15-cell V9_FEATURE_SET=v9d sweep")
    else:
        print(f"  GATE FAILED: {gate['reason']}")
        print(f"  -> Stop. Write findings note. No 15-cell sweep.")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--pairwise-v4", default="output/pairwise_v4.csv")
    parser.add_argument("--pairwise-bt", default="output/pairwise_bt.csv")
    parser.add_argument("--out-json", default=DEFAULT_DIAGNOSTIC_OUT)
    args = parser.parse_args(argv)

    diag = compute_gate(args.pairwise_v4, args.pairwise_bt)
    gate = check_gate(diag)
    print_report(diag, gate)

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump({"diagnostic": diag, "gate": gate}, f, indent=2)
    print(f"\n  saved {args.out_json}")
    return 0 if gate["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_diagnose_v9d.py -v`
Expected: all 5 tests pass.

- [ ] **Step 5: Verify ASCII**

```bash
python -c "open('src/diagnose_v9d.py', encoding='utf-8').read().encode('ascii')" && echo "ASCII OK"
python -c "open('tests/test_diagnose_v9d.py', encoding='utf-8').read().encode('ascii')" && echo "ASCII OK"
```

Expected: both ASCII OK.

- [ ] **Step 6: Commit**

```bash
git add src/diagnose_v9d.py tests/test_diagnose_v9d.py
git commit -m "$(cat <<'EOF'
feat(v9d): src/diagnose_v9d.py pre-sweep gate

Single-clause falsification: v9-D@(1.0, 0.0) must beat v9-C@(1.0, 0.0)
on weighted-mean per-game LL by >= 0.001. PASS -> proceed to the
15-cell sweep. FAIL -> stop, no remaining cells.

Mirrors src/diagnose_bt_vs_v4.py in shape (compute_gate / check_gate /
print_report / main, with sys.exit nonzero on FAIL so a wrapper can
short-circuit the sweep).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 6: Forced-verification suite (CLAUDE.md gate)

**Why:** Per CLAUDE.md: "For tasks that touch ingest, team mapping, or feature assembly: run `pytest -v tests/test_ingest tests/test_features tests/test_integration.py` even if your change is elsewhere -- these are the seams that catch dtype regressions and cross-source join breakage." The trainer extension touches the per-game data loader, which is a feature-assembly seam.

**Files:** None modified. Verification only.

- [ ] **Step 1: Run the broad test suite**

```bash
pytest -v
```

Expected: all tests pass. If any fail that look unrelated to BT-as-feature changes (e.g., a pre-existing flake), STOP and triage before continuing -- a passing-test baseline is the precondition for trusting Task 7's real-data run.

- [ ] **Step 2: Spot-check the v9-C path is unchanged on real data**

The v9-C trainer must produce identical artifacts when run with `feature_set='v9c'` after the trainer extension as before. This is exercised manually here as a quick sanity check (the unit-test fixtures don't catch real-data dtype edge cases).

```bash
V9_FEATURE_SET=v9c python -c "
from src.train_upset_model import load_per_game_data_with_upset, double_loso_eval
per_game = load_per_game_data_with_upset(
    'output/pairwise_v4.csv',
    'data/raw/march-machine-learning-2026/MNCAATourneyCompactResults.csv',
    'data/raw/march-machine-learning-2026/MNCAATourneySeeds.csv',
)
print('per_game shape:', per_game.shape)
print('p_bt absent:', 'p_bt' not in per_game.columns)
ev = double_loso_eval(per_game, w_upset=1.0, w_miss=0.0, feature_set='v9c')
n = ev['n_games'].sum()
ll = (ev['ll_v9'] * ev['n_games']).sum() / n
print(f'v9c@(1,0) wt-mean LL: {ll:.4f}')
"
```

Expected: `p_bt absent: True`, and the wt-mean LL at uniform weights is the same value v9-C produced before this PR (you'll know what that is from `output/v9c_sweep_results.csv` at the (1.0, 0.0) row -- check the `ll_loso_weighted_mean` column there). State the value in the commit message of the next data-generation task as the trainer-harness sanity number.

If the LL doesn't match the prior committed v9-C anchor cell to 4 decimal places, the trainer extension regressed the v9-C path -- abort and debug before any v9-D run.

- [ ] **Step 3: No commit (verification only)**

This is a gate task -- no files modified, no commit.

---

### Task 7: Run the pre-sweep gate on real data

**Why:** The diagnostic gate is the experiment's first decision point. Run it once, commit the JSON, and branch on the verdict in Task 8.

**Files:**
- Create: `output/diag_v9d.json` (committed)

- [ ] **Step 1: Run the pre-sweep gate on real data**

```bash
python src/diagnose_v9d.py
echo "EXIT CODE: $?"
```

Expected: a printed report with v9-C and v9-D wt-mean LL, the headroom, and a PASS or FAIL verdict. Exit code 0 on PASS, 1 on FAIL. The script writes `output/diag_v9d.json` either way.

- [ ] **Step 2: Inspect the JSON before committing**

```bash
cat output/diag_v9d.json
```

Confirm:
- `diagnostic.n_games_v9c == diagnostic.n_games_v9d` (same per-game frame underlies both evals)
- `diagnostic.threshold == 0.001`
- `gate.pass` is `true` or `false`
- `gate.reason` reads naturally

- [ ] **Step 3: Commit the gate result**

```bash
git add output/diag_v9d.json
git commit -m "$(cat <<'EOF'
data(v9d): output/diag_v9d.json -- gate <PASS|FAIL>

v9-C @ (W_UPSET=1.0, W_MISS=0.0) wt-mean LL: <fill in from JSON>
v9-D @ (W_UPSET=1.0, W_MISS=0.0) wt-mean LL: <fill in from JSON>
headroom: <fill in> (threshold: 0.001)

<PASS: proceed to 15-cell sweep | FAIL: stop, no sweep>

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

(Replace the `<fill in>` placeholders with the actual numbers from `output/diag_v9d.json`. Replace `<PASS|FAIL>` and the parenthetical conclusion with the actual verdict.)

---

### Task 8: BRANCH on gate verdict

**This task forks based on the JSON written in Task 7.** Read `output/diag_v9d.json` and proceed to either Task 8a (FAIL branch) or Task 8b (PASS branch). Do NOT do both.

#### Task 8a: Gate FAILED -- write NO-GO findings, update TODO, PR

**When:** `gate.pass` is `false` in `output/diag_v9d.json`.

**Files:**
- Create: `docs/notes/2026-05-02-bt-as-feature.md`
- Modify: `TODO.md` (move queue #2 to "Tried and rejected")

- [ ] **Step 1: Write the NO-GO findings note**

Create `docs/notes/2026-05-02-bt-as-feature.md` (use the structure of `docs/notes/2026-05-01-bayesian-stage1.md` -- TL;DR, setup, gate result table, falsification reasoning, recommendation, files of record). Keep it short -- 4-6 short sections, ~200 lines, ASCII-only.

Skeleton (replace with real numbers from `output/diag_v9d.json`):

```markdown
# BT-as-Feature for v9-C -- Findings

**Date:** 2026-05-02
**Branch:** feat/bt-as-feature
**Verdict:** **NO-GO** -- pre-sweep gate failed. v9-C stays in production.
**Spec:** `docs/superpowers/specs/2026-05-02-bt-as-feature-design.md`
**Plan:** `docs/superpowers/plans/2026-05-02-bt-as-feature.md`

## TL;DR

Adding `p_bt` from `output/pairwise_bt.csv` as a 6th input feature to
v9-C's upset-aware stage-2 model (under `feature_set='v9d'`) was tested
via the pre-sweep falsification gate. Gate FAILED: at uniform weights
(W_UPSET=1.0, W_MISS=0.0), v9-D's weighted-mean per-game LL was
`<LL_v9d>` vs v9-C's `<LL_v9c>` -- headroom `<headroom>` < 0.001
threshold. The gate's question is "does p_bt supply marginal
information v9-C can extract"; the answer is no, by paired comparison
on the same per-game frame.

## Setup recap

(short version of the spec's "Approach" section)

## Pre-sweep gate result

| measure | value | clause |
|---|---|---|
| v9-C wt-mean LL @ (1.0, 0.0) | `<LL_v9c>` | baseline |
| v9-D wt-mean LL @ (1.0, 0.0) | `<LL_v9d>` | candidate |
| headroom | `<headroom>` | **FAIL** (< 0.001) |
| **gate verdict** | - | **FAIL** |

## Falsification reasoning

(why the gate failed; what it tells us about whether v9-C's
representation can extract value from p_bt at all)

## Verdict

NO-GO. The 15-cell W_UPSET / W_MISS sweep was not run -- saved
~45-75 minutes of compute. v9-C stays in production at
`(W_UPSET=1.25, W_MISS=0.0)` = 2713 brkt pts.

## Recommendation

(what queue item advances; what this tells us about queue items #1
and #3)

## Files of record

```
src/diagnose_v9d.py             -- pre-sweep gate
src/train_upset_model.py        -- extended (v9d feature_set, p_bt join)
src/sweep_v9_weights.py         -- extended (V9_FEATURE_SET=v9d)

output/diag_v9d.json            -- gate diagnostic with verdict
                                   (15-cell sweep NOT run)

tests/test_train_upset_model.py
tests/test_diagnose_v9d.py
tests/test_sweep_v9_weights.py
```

The `'v9d'` feature_set / pre-sweep gate machinery remains on the
branch as the experiment record. Re-running the gate against a new
BT variant (margin-aware, hierarchical-with-feature-priors) is just
`python src/diagnose_v9d.py --pairwise-bt output/pairwise_NEW.csv`.
```

- [ ] **Step 2: Update `TODO.md`**

Move active queue #2 to the "Tried and rejected" section. Renumber the remaining active items (former #1 stays at #1; former #3 becomes #2; etc.).

The new "Tried and rejected" entry (concise, matching the existing format):

```markdown
- **BT-as-feature for v9-C (2026-05-02).** Added p_bt from
  output/pairwise_bt.csv as a 6th input feature to v9-C's
  upset-aware stage-2 model (feature_set='v9d'). Pre-sweep
  falsification gate FAILED: at uniform weights (1.0, 0.0),
  v9-D wt-mean LL `<LL_v9d>` vs v9-C `<LL_v9c>`, headroom
  `<headroom>` < 0.001 threshold. Saved ~45-75 min of compute by
  not running the 15-cell sweep. v9-C's 5-feature representation
  is already extracting essentially everything p_bt could
  contribute on top of v4 + seed/round context. Code retained
  on feat/bt-as-feature as the experiment record. Findings:
  docs/notes/2026-05-02-bt-as-feature.md.
```

- [ ] **Step 3: Verify ASCII for both files**

```bash
python -c "open('docs/notes/2026-05-02-bt-as-feature.md', encoding='utf-8').read().encode('ascii')" && echo "ASCII OK (notes)"
python -c "open('TODO.md', encoding='utf-8').read().encode('ascii')" && echo "ASCII OK (TODO)"
```

- [ ] **Step 4: Final pytest before PR**

```bash
pytest -v
```

Expected: all tests pass.

- [ ] **Step 5: Commit + PR**

```bash
git add docs/notes/2026-05-02-bt-as-feature.md docs/superpowers/plans/2026-05-02-bt-as-feature.md TODO.md
git commit -m "$(cat <<'EOF'
docs(v9d): findings note + plan + TODO update -- gate FAILED

BT-as-feature for v9-C tested via the single-clause pre-sweep gate
in src/diagnose_v9d.py. v9-D@(1,0) wt-mean LL <LL_v9d> vs
v9-C@(1,0) <LL_v9c>; headroom <headroom> below the 0.001
threshold. 15-cell sweep not run -- saved ~45-75 min of compute.
v9-C stays in production. Queue item #2 moves to Tried-and-rejected;
queue #1 (feature-view diversity) advances.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"

git push -u origin feat/bt-as-feature
```

Then open the PR from CLI or the GitHub UI. Standard PR description: TL;DR (gate failed), what was tested, what it falsifies, what advances next.

#### Task 8b: Gate PASSED -- run sweep, anchor check, write findings, PR

**When:** `gate.pass` is `true` in `output/diag_v9d.json`.

**Files:**
- Create: `output/v9d_sweep_results.csv` and 15 files in `output/v9d_sweep/`
- Create: `docs/notes/2026-05-02-bt-as-feature.md`
- Modify: `TODO.md` (move queue #2 to Tried-and-rejected, marginal, or done depending on band)

- [ ] **Step 1: Run the 15-cell sweep with V9_FEATURE_SET=v9d**

```bash
V9_FEATURE_SET=v9d python src/sweep_v9_weights.py 2>&1 | tee output/v9d_sweep_log.txt
```

Expected runtime: 45-75 minutes. The script writes:
- `output/v9d_sweep/pairwise_v9_WU{u:.2f}_WM{m:.2f}.csv` (15 files)
- `output/v9d_sweep_results.csv` (one row per cell, sorted by `total_brkt_pts` descending)

The script's tail prints the v8 baseline, the v9-D anchor cell delta vs v8 (informational, not gating), and the best-cell delta vs v8.

- [ ] **Step 2: Trainer-harness anchor check**

Per the spec's anchor discipline: re-run the v9-C anchor cell with the post-extension trainer and confirm it reproduces the committed v9-C anchor within 1e-9.

```bash
python -c "
import pandas as pd
import numpy as np
from src.train_upset_model import load_per_game_data_with_upset, build_v9_pairwise

per_game = load_per_game_data_with_upset(
    'output/pairwise_v4.csv',
    'data/raw/march-machine-learning-2026/MNCAATourneyCompactResults.csv',
    'data/raw/march-machine-learning-2026/MNCAATourneySeeds.csv',
)
build_v9_pairwise(
    per_game, 'output/pairwise_v4.csv',
    'data/raw/march-machine-learning-2026/MNCAATourneySeeds.csv',
    'output/pairwise_v9c_anchor_recheck.csv',
    slots_csv='data/raw/march-machine-learning-2026/MNCAATourneySlots.csv',
    w_upset=1.0, w_miss=0.0, feature_set='v9c',
)

ref = pd.read_csv('output/v9c_sweep/pairwise_v9_WU1.00_WM0.00.csv')
new = pd.read_csv('output/pairwise_v9c_anchor_recheck.csv')

ref_lookup = {(s, a, b): p for s, a, b, p
              in zip(ref.season, ref.team_a, ref.team_b, ref.p_a_wins)}
deltas = []
for s, a, b, p_new in zip(new.season, new.team_a, new.team_b, new.p_a_wins):
    p_ref = ref_lookup.get((int(s), int(a), int(b)))
    if p_ref is not None:
        deltas.append(abs(p_new - p_ref))

max_delta = max(deltas) if deltas else float('nan')
print(f'n_pairs compared: {len(deltas)}')
print(f'max abs delta: {max_delta:.2e}')
assert max_delta < 1e-9, f'TRAINER REGRESSION: max delta {max_delta:.2e} > 1e-9'
print('TRAINER ANCHOR PASSED')
"
```

Expected: `TRAINER ANCHOR PASSED`. If max delta exceeds `1e-9`, the v9-C trainer path was perturbed by the extension -- abort and debug. Do NOT delete `output/pairwise_v9c_anchor_recheck.csv` until the findings note has been written referencing the max-delta number.

- [ ] **Step 3: Compute per-cell delta vs v9-C production cell**

The v9-C production cell is `(W_UPSET=1.25, W_MISS=0.0)` at 2713 brkt pts. Re-score it fresh from the committed artifact to make the comparison apples-to-apples on the same scoring code:

```bash
python -c "
import pandas as pd
from src.score_chalk_brackets import score_pairwise_path

# Re-score v9-C production cell from its committed artifact.
v9c_prod = score_pairwise_path('output/v9c_sweep/pairwise_v9_WU1.25_WM0.00.csv')
v9c_prod_pts = float(v9c_prod['total_pts'])
print(f'v9-C production cell (1.25, 0.0): {v9c_prod_pts:.1f} brkt pts')

# Read v9-D sweep results.
v9d = pd.read_csv('output/v9d_sweep_results.csv').sort_values(
    'total_brkt_pts', ascending=False
)
v9d['delta_vs_v9c_prod'] = v9d['total_brkt_pts'] - v9c_prod_pts

print()
print('v9-D sweep, sorted by total_brkt_pts (descending):')
print(v9d.to_string(index=False))

best = v9d.iloc[0]
print()
print(f'Best v9-D cell: W_UPSET={best.w_upset}, W_MISS={best.w_miss}, '
      f'total_brkt_pts={best.total_brkt_pts:.1f}, '
      f'delta vs v9-C prod={best.delta_vs_v9c_prod:+.1f}')

# Verdict band.
d = float(best.delta_vs_v9c_prod)
if d >= 25.0:
    print(f'VERDICT: CLEAR WINNER ({d:+.1f} brkt pts >= +25)')
elif d >= 10.0:
    print(f'VERDICT: MARGINAL CANDIDATE (+10 <= {d:+.1f} < +25)')
else:
    print(f'VERDICT: NO-GO ({d:+.1f} < +10)')
" 2>&1 | tee output/v9d_delta_vs_v9c.txt
```

Note the verdict; it determines what the findings note recommends.

- [ ] **Step 4: Clean up the anchor recheck file**

```bash
rm output/pairwise_v9c_anchor_recheck.csv
```

(Only after the max-delta number from Step 2 is captured for the findings note.)

- [ ] **Step 5: Write the findings note**

Create `docs/notes/2026-05-02-bt-as-feature.md`. Structure mirrors `docs/notes/2026-05-01-v9c-feature-stripped.md` (closest analog: a feature-set variant tested via the same sweep harness). Include:

1. Verdict line in the header (CLEAR WINNER / MARGINAL / NO-GO).
2. TL;DR: gate result + best-cell result + verdict band.
3. Setup recap (short -- spec is the canonical source).
4. Pre-sweep gate result (table).
5. 22-season head-to-head table (per-cell, with delta_vs_v9c_prod).
6. Trainer-harness anchor result (the max-delta number from Step 2).
7. Verdict + recommendation (what queue item advances; whether to follow up with a production-swap PR).
8. Files of record.

Replace all `<fill in>` placeholders with the actual numbers. ASCII-only.

- [ ] **Step 6: Update `TODO.md`**

Three cases by verdict:

- **NO-GO (`delta < +10`):** move queue #2 to "Tried and rejected" with the verdict line. Renumber remaining active items.
- **Marginal (`+10 <= delta < +25`):** add a new "Done" entry capturing the result; mark queue #2 as "tested, candidate, do not swap." Renumber remaining active items.
- **Clear winner (`delta >= +25`):** add a new "Done" entry; add a new "Active queue" item: "Production swap to v9-D (predict_2026_v9d.py mirror + extend train_bt_stage1.py to a 2026-final-snapshot + repoint output/pairwise_probs.json consumer)." Renumber remaining active items.

- [ ] **Step 7: Verify ASCII for findings + TODO**

```bash
python -c "open('docs/notes/2026-05-02-bt-as-feature.md', encoding='utf-8').read().encode('ascii')" && echo "ASCII OK (notes)"
python -c "open('TODO.md', encoding='utf-8').read().encode('ascii')" && echo "ASCII OK (TODO)"
```

- [ ] **Step 8: Final pytest before PR**

```bash
pytest -v
```

Expected: all tests pass.

- [ ] **Step 9: Commit data + findings + TODO**

```bash
# Data first.
git add output/v9d_sweep_results.csv output/v9d_sweep/ output/v9d_sweep_log.txt output/v9d_delta_vs_v9c.txt
git commit -m "$(cat <<'EOF'
data(v9d): output/v9d_sweep/ -- 15-cell sweep results

22-season LOSO bracket-points sweep over W_UPSET in
{1.0, 1.25, 1.5, 1.75, 2.0} x W_MISS in {0, 0.5, 1.0} for
feature_set='v9d' (v9-C 5 features + p_bt from output/pairwise_bt.csv).

Best cell: W_UPSET=<wu>, W_MISS=<wm>, total_brkt_pts=<pts>
v9-C production cell baseline: 2713 brkt pts at (1.25, 0.0)
delta vs v9-C: <delta>

Trainer-harness anchor: re-running v9-C @ (1.0, 0.0) after the
trainer extension reproduces the committed v9-C anchor cell within
<max_delta:.2e> (< 1e-9).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"

# Findings + plan + TODO.
git add docs/notes/2026-05-02-bt-as-feature.md docs/superpowers/plans/2026-05-02-bt-as-feature.md TODO.md
git commit -m "$(cat <<'EOF'
docs(v9d): findings note + plan + TODO update -- gate <PASSED|FAILED>, sweep <CLEAR WINNER|MARGINAL|NO-GO>

(Replace bracketed placeholders with the actual verdicts.)

Best v9-D cell: W_UPSET=<wu>, W_MISS=<wm>, total_brkt_pts=<pts>
delta vs v9-C production cell: <delta:+.1f>
verdict band: <CLEAR WINNER (>= +25) | MARGINAL (+10..+24) | NO-GO (< +10)>

(One short paragraph: what the result means; what queue advances.)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"

git push -u origin feat/bt-as-feature
```

Then open the PR from CLI or the GitHub UI.

---

## Self-review

After writing the plan, run through this list:

1. **Spec coverage:** Every spec section has at least one task.
   - "Goals" Goal 1 (add v9d feature_set) -> Tasks 1-3. Goal 2 (pre-sweep gate) -> Task 5. Goal 3 (sweep + bracket-points compare) -> Task 8b. Goal 4 (verdict bands -> action) -> Task 8a/8b disposition.
   - "Module changes" -> Tasks 1, 2, 3, 4, 5.
   - "Anchor + join sanity discipline" -> Trainer-harness anchor in Task 8b Step 2; join-orientation anchor in Task 1 Step 1 (and the v9d-shape tests in Task 2).
   - "Disposition matrix" -> Task 8a (FAIL) / 8b (PASS, all three result bands).
   - "File deliverables" -> all 9 files covered across the plan.
   - "Tests" -> Tasks 1-5 each include the `tests/...` test file.
   - "Implementation order" -> matches the 10-step ordering: spec steps 1-2 = Tasks 1-3, spec step 3 = Task 4, spec step 5 = Task 5, spec step 4 = Task 6, spec step 6 = Task 7, spec step 7 = Task 8 (gate decision), spec steps 8-10 = Task 8b.

2. **Placeholder scan:** placeholders only appear in commit-message templates inside Task 7 / 8a / 8b, with explicit "replace with actual numbers from JSON" instructions. No `TODO`/`TBD` outside those.

3. **Type consistency:** function names match (`load_per_game_data_with_upset`, `upset_features`, `build_v9_pairwise`, `run_single_cell`, `run_sweep`, `compute_gate`, `check_gate`); kwarg name `pairwise_bt_csv` is consistent across Tasks 1, 3, 4, 5; column name `p_bt` is consistent throughout. `feature_set='v9d'` literal is consistent.

4. **Critical-path verification:** the trainer-harness anchor in Task 8b Step 2 is the only "real-data" check that catches a v9-C-path regression introduced by the trainer extension. If the unit tests pass but this fails, the extension is broken in a way the synthetic fixtures don't catch -- abort.
