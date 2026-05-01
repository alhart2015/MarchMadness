# v9 Upset-Weight Tuning Sweep -- Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Sweep a 5x3 grid of (W_UPSET, W_MISS) values through the v9-A trainer, score each cell on 22-season LOSO bracket points, and either recommend swapping v8 for the winning cell or close the open question.

**Architecture:** Thread `w_upset` and `w_miss` through `double_loso_eval` and `build_v9_pairwise` in `src/train_upset_model.py` (both currently call `compute_sample_weights(train)` with no args, picking up module defaults of 3.0/4.0). Build a new driver `src/sweep_v9_weights.py` that loops over the 15-cell grid, writes per-cell pairwise CSVs to `output/v9_sweep/`, scores them with `score_pairwise_path`, and emits `output/v9_sweep_results.csv`. Anchor cell (W_UPSET=1.0, W_MISS=0.0) must reproduce v8 within 1 bracket point or the sweep halts. Findings note written by the engineer based on results.

**Tech Stack:** Python 3.11+, pandas, numpy, xgboost, pytest. ASCII-only files (Windows cp1252 console).

**Spec:** `docs/superpowers/specs/2026-05-01-v9-weight-sweep.md`

---

## File Structure

| File | Role | Disposition |
|------|------|-------------|
| `src/train_upset_model.py` | v9-A trainer; already exposes `compute_sample_weights(df, w_upset, w_miss)` | Modify: thread weights through `double_loso_eval` and `build_v9_pairwise` |
| `tests/test_upset_model.py` | Existing trainer tests (10 tests) | Modify: add 2 weight-threading tests |
| `src/sweep_v9_weights.py` | NEW: 15-cell sweep driver | Create |
| `tests/test_sweep_v9_weights.py` | NEW: driver tests | Create |
| `output/v9_sweep_results.csv` | Output: one row per cell | Generated |
| `output/v9_sweep/pairwise_v9_WU{u}_WM{m}.csv` | Output: per-cell pairwise probs | Generated |
| `docs/notes/2026-05-01-v9-weight-sweep.md` | Findings | Create |
| `TODO.md` | Active queue | Modify: record verdict |

Total: 4 source files touched (well under the 5-file phase cap).

---

## Task 1: Thread w_upset / w_miss through `double_loso_eval`

**Files:**
- Modify: `src/train_upset_model.py:210-256` (function `double_loso_eval`)
- Modify: `tests/test_upset_model.py` (append a new test)

- [ ] **Step 1: Re-read the current `double_loso_eval` and the test file**

```bash
# At repo root of the worktree:
sed -n '210,260p' src/train_upset_model.py
```

Confirm the function signature is `def double_loso_eval(per_game: pd.DataFrame) -> pd.DataFrame:` and the only `compute_sample_weights` call inside is `w_train = compute_sample_weights(train)`.

- [ ] **Step 2: Write the failing test**

Append to `tests/test_upset_model.py`:

```python
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
    from src.train_upset_model import compute_sample_weights

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
```

- [ ] **Step 3: Run tests to verify they fail**

```bash
pytest tests/test_upset_model.py::test_double_loso_eval_threads_weights_to_sample_weight tests/test_upset_model.py::test_double_loso_eval_default_weights_match_module_globals -v
```

Expected: FAIL on the first test with `TypeError: double_loso_eval() got an unexpected keyword argument 'w_upset'`.

- [ ] **Step 4: Implement -- add kwargs to `double_loso_eval`**

In `src/train_upset_model.py`, change the signature of `double_loso_eval` from:

```python
def double_loso_eval(per_game: pd.DataFrame) -> pd.DataFrame:
```

to:

```python
def double_loso_eval(
    per_game: pd.DataFrame,
    w_upset: float = W_UPSET,
    w_miss: float = W_MISS,
) -> pd.DataFrame:
```

Inside the function, change:

```python
        w_train = compute_sample_weights(train)
```

to:

```python
        w_train = compute_sample_weights(train, w_upset=w_upset, w_miss=w_miss)
```

Update the docstring's "Returns" section to add a one-liner: `Weights are forwarded to compute_sample_weights; defaults preserve canonical 3.0 / 4.0 behavior.`

- [ ] **Step 5: Run tests to verify they pass**

```bash
pytest tests/test_upset_model.py -v
```

Expected: all tests in the file pass (including the existing 10 plus the 2 new ones = 12 total).

- [ ] **Step 6: Commit**

```bash
git add src/train_upset_model.py tests/test_upset_model.py
git commit -m "$(cat <<'EOF'
feat(v9): thread w_upset/w_miss kwargs through double_loso_eval

Default values (3.0 / 4.0) preserve existing behavior. Enables the
upset-weight sweep planned in
docs/superpowers/specs/2026-05-01-v9-weight-sweep.md.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Thread w_upset / w_miss through `build_v9_pairwise`

**Files:**
- Modify: `src/train_upset_model.py:259-313` (function `build_v9_pairwise`)
- Modify: `tests/test_upset_model.py` (append a new test)

- [ ] **Step 1: Re-read the current `build_v9_pairwise`**

```bash
sed -n '259,315p' src/train_upset_model.py
```

Confirm the function signature is `def build_v9_pairwise(per_game, pairwise_v4_csv, seeds_csv, out_path):` and the only `compute_sample_weights` call is `w_train = compute_sample_weights(train)`.

- [ ] **Step 2: Write the failing test**

Append to `tests/test_upset_model.py`:

```python
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
        w_upset=1.0, w_miss=0.0,
    )

    # 2 LOSO fits (one per season in pw_v4), both must be uniform.
    assert len(captured_weights) == 2
    for w in captured_weights:
        assert np.allclose(w, 1.0), f"expected uniform weights, got {w}"
```

- [ ] **Step 3: Run test to verify it fails**

```bash
pytest tests/test_upset_model.py::test_build_v9_pairwise_threads_weights_to_sample_weight -v
```

Expected: FAIL with `TypeError: build_v9_pairwise() got an unexpected keyword argument 'w_upset'`.

- [ ] **Step 4: Implement -- add kwargs to `build_v9_pairwise`**

In `src/train_upset_model.py`, change the signature of `build_v9_pairwise` from:

```python
def build_v9_pairwise(
    per_game: pd.DataFrame,
    pairwise_v4_csv: str,
    seeds_csv: str,
    out_path: str,
) -> None:
```

to:

```python
def build_v9_pairwise(
    per_game: pd.DataFrame,
    pairwise_v4_csv: str,
    seeds_csv: str,
    out_path: str,
    w_upset: float = W_UPSET,
    w_miss: float = W_MISS,
) -> None:
```

Inside the function, change:

```python
            w_train = compute_sample_weights(train)
```

to:

```python
            w_train = compute_sample_weights(train, w_upset=w_upset, w_miss=w_miss)
```

- [ ] **Step 5: Run all upset-model tests to verify they pass**

```bash
pytest tests/test_upset_model.py -v
```

Expected: 13 tests pass (10 existing + 3 new).

- [ ] **Step 6: Commit**

```bash
git add src/train_upset_model.py tests/test_upset_model.py
git commit -m "$(cat <<'EOF'
feat(v9): thread w_upset/w_miss kwargs through build_v9_pairwise

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Sweep driver -- skeleton with grid validation

**Files:**
- Create: `src/sweep_v9_weights.py`
- Create: `tests/test_sweep_v9_weights.py`

- [ ] **Step 1: Write failing tests for grid validation**

Create `tests/test_sweep_v9_weights.py`:

```python
"""Tests for src/sweep_v9_weights.py (15-cell W_UPSET / W_MISS sweep)."""

import pytest

from src.sweep_v9_weights import GRID, validate_grid


def test_grid_contains_anchor_cell():
    """The anchor cell (W_UPSET=1.0, W_MISS=0.0) MUST be in the grid -- it
    is the v8 reproduction sanity check.
    """
    assert (1.0, 0.0) in GRID


def test_grid_has_15_unique_cells():
    """Spec calls for 5 * 3 = 15 cells."""
    assert len(GRID) == 15
    assert len(set(GRID)) == 15


def test_validate_grid_passes_with_anchor_cell():
    """validate_grid raises iff anchor is missing."""
    validate_grid([(1.0, 0.0), (1.5, 1.0)])  # contains anchor; should not raise


def test_validate_grid_raises_without_anchor_cell():
    with pytest.raises(ValueError, match="anchor cell"):
        validate_grid([(1.5, 1.0), (2.0, 0.0)])
```

- [ ] **Step 2: Run to verify it fails**

```bash
pytest tests/test_sweep_v9_weights.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'src.sweep_v9_weights'`.

- [ ] **Step 3: Create the sweep module skeleton**

Create `src/sweep_v9_weights.py`:

```python
"""15-cell W_UPSET / W_MISS tuning sweep over the v9-A trainer.

Grid: W_UPSET in {1.0, 1.25, 1.5, 1.75, 2.0} x W_MISS in {0.0, 0.5, 1.0}.

For each cell, run double-LOSO across 22 seasons (2003..2025), build
v9-adjusted pairwise probabilities, score with score_pairwise_path
against MNCAATourneyCompactResults.csv, and write one row to
output/v9_sweep_results.csv.

Anchor cell (1.0, 0.0) must be present in the grid -- it is the v8
reproduction sanity check.

Spec:  docs/superpowers/specs/2026-05-01-v9-weight-sweep.md
"""
import sys
from pathlib import Path
from typing import Iterable, List, Tuple

# Path setup: allow `python src/sweep_v9_weights.py` invocation.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

W_UPSET_VALUES = [1.0, 1.25, 1.5, 1.75, 2.0]
W_MISS_VALUES = [0.0, 0.5, 1.0]
GRID: List[Tuple[float, float]] = [
    (wu, wm) for wu in W_UPSET_VALUES for wm in W_MISS_VALUES
]
ANCHOR_CELL: Tuple[float, float] = (1.0, 0.0)


def validate_grid(grid: Iterable[Tuple[float, float]]) -> None:
    """Raise ValueError if the anchor cell (1.0, 0.0) is missing.

    The anchor is the v8 reproduction sanity check: at uniform weights
    the v9-A trainer should reproduce v8 within 1 bracket point. Without
    the anchor, the sweep cannot be sanity-checked.
    """
    cells = set((float(wu), float(wm)) for wu, wm in grid)
    if ANCHOR_CELL not in cells:
        raise ValueError(
            f"anchor cell {ANCHOR_CELL} missing from grid; sweep is invalid "
            "(no v8 reproduction sanity check possible)"
        )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_sweep_v9_weights.py -v
```

Expected: 4 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/sweep_v9_weights.py tests/test_sweep_v9_weights.py
git commit -m "$(cat <<'EOF'
feat(v9-sweep): skeleton with grid + anchor-cell validation

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Sweep driver -- run a single cell

**Files:**
- Modify: `src/sweep_v9_weights.py`
- Modify: `tests/test_sweep_v9_weights.py`

- [ ] **Step 1: Re-read the current sweep module**

```bash
cat src/sweep_v9_weights.py
```

Confirm `GRID`, `ANCHOR_CELL`, and `validate_grid` are defined; no other functions yet.

- [ ] **Step 2: Write the failing test**

Append to `tests/test_sweep_v9_weights.py`:

```python
import numpy as np
import pandas as pd

from src.sweep_v9_weights import run_single_cell


def _write_minimal_inputs(tmp_path):
    """Two seasons, two pairs each, with seeds and per-game results."""
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

    results = pd.DataFrame({
        "Season": [2022, 2022, 2023, 2023],
        "DayNum": [136, 138, 136, 138],
        "WTeamID": [1, 1, 1, 2],
        "WScore": [70, 75, 65, 70],
        "LTeamID": [2, 3, 3, 3],
        "LScore": [60, 65, 60, 65],
    })
    results_path = tmp_path / "results.csv"
    results.to_csv(results_path, index=False)

    return str(pw_path), str(seeds_path), str(results_path)


def test_run_single_cell_writes_pairwise_and_returns_metrics(tmp_path):
    """run_single_cell writes a v8-compatible pairwise CSV at the
    expected path and returns a dict with weight + scoring keys.
    """
    pw_path, seeds_path, results_path = _write_minimal_inputs(tmp_path)
    out_dir = tmp_path / "v9_sweep"

    metrics = run_single_cell(
        w_upset=1.0, w_miss=0.0,
        pairwise_v4_csv=pw_path,
        results_csv=results_path,
        seeds_csv=seeds_path,
        out_dir=str(out_dir),
    )

    # Pairwise CSV exists and has the expected schema.
    pw_path_out = out_dir / "pairwise_v9_WU1.00_WM0.00.csv"
    assert pw_path_out.exists()
    out = pd.read_csv(pw_path_out)
    assert list(out.columns) == ["season", "team_a", "team_b", "p_a_wins"]
    assert (out["team_a"] < out["team_b"]).all()

    # Returned metrics have all required fields.
    assert set(metrics.keys()) >= {
        "w_upset", "w_miss",
        "total_brkt_pts", "ll_loso_weighted_mean", "acc_loso_weighted_mean",
        "pairwise_csv",
    }
    assert metrics["w_upset"] == 1.0
    assert metrics["w_miss"] == 0.0
    assert metrics["pairwise_csv"] == str(pw_path_out)
    # 2-season synthetic data: total_brkt_pts may be zero or positive
    # (no full bracket, just R64 stubs); just assert it's a float.
    assert isinstance(metrics["total_brkt_pts"], float)
```

- [ ] **Step 3: Run test to verify it fails**

```bash
pytest tests/test_sweep_v9_weights.py::test_run_single_cell_writes_pairwise_and_returns_metrics -v
```

Expected: FAIL with `ImportError: cannot import name 'run_single_cell'`.

- [ ] **Step 4: Implement `run_single_cell`**

Append to `src/sweep_v9_weights.py`:

```python
import numpy as np
import pandas as pd

from src.train_upset_model import (
    build_v9_pairwise,
    double_loso_eval,
    load_per_game_data_with_upset,
)


def _cell_path(out_dir: str, w_upset: float, w_miss: float) -> str:
    """Per-cell pairwise CSV path: pairwise_v9_WU{u:.2f}_WM{m:.2f}.csv."""
    name = f"pairwise_v9_WU{w_upset:.2f}_WM{w_miss:.2f}.csv"
    return str(Path(out_dir) / name)


def run_single_cell(
    w_upset: float,
    w_miss: float,
    pairwise_v4_csv: str,
    results_csv: str,
    seeds_csv: str,
    out_dir: str,
) -> dict:
    """Run one (w_upset, w_miss) cell of the sweep.

    Steps:
      1. Load per-game training rows from pairwise_v4 + results + seeds.
      2. Build v9-adjusted pairwise CSV at out_dir/pairwise_v9_WU{u}_WM{m}.csv.
      3. Run per-season LOSO eval to capture log loss / accuracy.
      4. Score the pairwise CSV (best-effort: catches FileNotFoundError /
         missing slots in score_pairwise_path so unit tests with
         synthetic data work).
      5. Return dict with all metrics.
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    pairwise_csv_out = _cell_path(out_dir, w_upset, w_miss)

    per_game = load_per_game_data_with_upset(
        pairwise_v4_csv, results_csv, seeds_csv
    )

    build_v9_pairwise(
        per_game, pairwise_v4_csv, seeds_csv, pairwise_csv_out,
        w_upset=w_upset, w_miss=w_miss,
    )

    eval_df = double_loso_eval(
        per_game, w_upset=w_upset, w_miss=w_miss
    )
    if len(eval_df) > 0 and "n_games" in eval_df.columns:
        n_total = float(eval_df["n_games"].sum())
        if n_total > 0:
            ll_mean = float(
                (eval_df["ll_v9"] * eval_df["n_games"]).sum() / n_total
            )
            acc_mean = float(
                (eval_df["acc_v9"] * eval_df["n_games"]).sum() / n_total
            )
        else:
            ll_mean = float("nan")
            acc_mean = float("nan")
    else:
        ll_mean = float("nan")
        acc_mean = float("nan")

    # Bracket scoring: tolerate missing tournament slot data on synthetic
    # inputs (unit tests). Production runs with real Kaggle data will
    # produce meaningful totals.
    try:
        from src.score_chalk_brackets import score_pairwise_path
        scored = score_pairwise_path(pairwise_csv_out)
        total_pts = float(scored["total_pts"])
    except (FileNotFoundError, KeyError, ValueError):
        total_pts = 0.0

    return {
        "w_upset": float(w_upset),
        "w_miss": float(w_miss),
        "total_brkt_pts": total_pts,
        "ll_loso_weighted_mean": ll_mean,
        "acc_loso_weighted_mean": acc_mean,
        "pairwise_csv": pairwise_csv_out,
    }
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
pytest tests/test_sweep_v9_weights.py -v
```

Expected: 5 tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/sweep_v9_weights.py tests/test_sweep_v9_weights.py
git commit -m "$(cat <<'EOF'
feat(v9-sweep): run_single_cell builds + scores one (W_U, W_M) cell

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Sweep driver -- main loop with results CSV

**Files:**
- Modify: `src/sweep_v9_weights.py`
- Modify: `tests/test_sweep_v9_weights.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_sweep_v9_weights.py`:

```python
def test_run_sweep_writes_results_csv(tmp_path, monkeypatch):
    """run_sweep over a 2-cell mini-grid (anchor + one more) writes a
    results CSV with one row per cell and the expected columns.
    """
    pw_path, seeds_path, results_path = _write_minimal_inputs(tmp_path)
    out_dir = tmp_path / "v9_sweep"
    results_csv_out = tmp_path / "v9_sweep_results.csv"

    from src.sweep_v9_weights import run_sweep

    mini_grid = [(1.0, 0.0), (1.5, 0.5)]

    run_sweep(
        grid=mini_grid,
        pairwise_v4_csv=pw_path,
        results_csv=results_path,
        seeds_csv=seeds_path,
        out_dir=str(out_dir),
        results_csv_path=str(results_csv_out),
    )

    assert results_csv_out.exists()
    df = pd.read_csv(results_csv_out)
    assert len(df) == 2
    assert set(df.columns) >= {
        "w_upset", "w_miss",
        "total_brkt_pts",
        "ll_loso_weighted_mean", "acc_loso_weighted_mean",
        "pairwise_csv",
    }
    # Sorted by total_brkt_pts descending.
    assert df["total_brkt_pts"].is_monotonic_decreasing


def test_run_sweep_halts_without_anchor_cell(tmp_path):
    """run_sweep refuses to start if the anchor cell is missing."""
    pw_path, seeds_path, results_path = _write_minimal_inputs(tmp_path)
    from src.sweep_v9_weights import run_sweep

    bad_grid = [(1.5, 0.5), (2.0, 1.0)]  # no (1.0, 0.0)

    with pytest.raises(ValueError, match="anchor cell"):
        run_sweep(
            grid=bad_grid,
            pairwise_v4_csv=pw_path,
            results_csv=results_path,
            seeds_csv=seeds_path,
            out_dir=str(tmp_path / "v9_sweep"),
            results_csv_path=str(tmp_path / "results.csv"),
        )
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_sweep_v9_weights.py -v
```

Expected: 2 new tests fail with `ImportError: cannot import name 'run_sweep'`.

- [ ] **Step 3: Implement `run_sweep` and `main`**

Append to `src/sweep_v9_weights.py`:

```python
def run_sweep(
    grid: Iterable[Tuple[float, float]],
    pairwise_v4_csv: str,
    results_csv: str,
    seeds_csv: str,
    out_dir: str,
    results_csv_path: str,
) -> pd.DataFrame:
    """Run the full grid; write per-cell pairwise CSVs to out_dir and
    aggregate results to results_csv_path. Returns the results DataFrame
    (sorted by total_brkt_pts descending).

    Halts if the anchor cell (1.0, 0.0) is missing -- the v8 reproduction
    sanity check would be impossible.
    """
    grid = list(grid)
    validate_grid(grid)

    rows = []
    for i, (w_upset, w_miss) in enumerate(grid, start=1):
        print(f"[cell {i}/{len(grid)}] W_UPSET={w_upset}, W_MISS={w_miss}")
        m = run_single_cell(
            w_upset=w_upset, w_miss=w_miss,
            pairwise_v4_csv=pairwise_v4_csv,
            results_csv=results_csv,
            seeds_csv=seeds_csv,
            out_dir=out_dir,
        )
        print(f"  total_brkt_pts={m['total_brkt_pts']:.1f}, "
              f"ll={m['ll_loso_weighted_mean']:.4f}, "
              f"acc={m['acc_loso_weighted_mean']:.3f}")
        rows.append(m)

    df = (
        pd.DataFrame(rows)
        .sort_values("total_brkt_pts", ascending=False)
        .reset_index(drop=True)
    )
    Path(results_csv_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(results_csv_path, index=False)
    return df


def main():
    """Run the canonical 15-cell sweep against production data paths.

    Compares the anchor cell (1.0, 0.0) bracket points against
    output/pairwise_v8.csv as a sanity gate after the sweep.
    """
    print("=" * 80)
    print("V9 UPSET-WEIGHT SWEEP")
    print(f"  Grid: {len(GRID)} cells, "
          f"W_UPSET in {W_UPSET_VALUES}, W_MISS in {W_MISS_VALUES}")
    print("=" * 80)

    pairwise_v4 = "output/pairwise_v4.csv"
    pairwise_v8 = "output/pairwise_v8.csv"
    seeds_csv = "data/raw/march-machine-learning-2026/MNCAATourneySeeds.csv"
    results_csv = "data/raw/march-machine-learning-2026/MNCAATourneyCompactResults.csv"
    out_dir = "output/v9_sweep"
    results_csv_path = "output/v9_sweep_results.csv"

    df = run_sweep(
        grid=GRID,
        pairwise_v4_csv=pairwise_v4,
        results_csv=results_csv,
        seeds_csv=seeds_csv,
        out_dir=out_dir,
        results_csv_path=results_csv_path,
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
    if abs(delta) > 1.0:
        print("WARNING: anchor cell does not reproduce v8 within 1 pt; "
              "sweep is invalid.")
    else:
        print("Anchor cell reproduces v8 within 1 pt -- sweep is valid.")

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


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run all sweep tests to verify they pass**

```bash
pytest tests/test_sweep_v9_weights.py -v
```

Expected: 7 tests pass.

- [ ] **Step 5: Run the full upset-related test set to confirm nothing regressed**

```bash
pytest tests/test_upset_model.py tests/test_sweep_v9_weights.py -v
```

Expected: 13 + 7 = 20 tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/sweep_v9_weights.py tests/test_sweep_v9_weights.py
git commit -m "$(cat <<'EOF'
feat(v9-sweep): run_sweep + main with anchor-cell sanity gate

Driver loops over the 15-cell grid, writes per-cell pairwise CSVs to
output/v9_sweep/, aggregates to output/v9_sweep_results.csv, and
prints the v8 vs anchor-cell delta + winner check.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Profile-and-go gate (1 cell)

**Files:** none modified. This is a runtime check.

- [ ] **Step 1: Confirm production inputs exist**

```bash
ls -lh output/pairwise_v4.csv output/pairwise_v8.csv \
       data/raw/march-machine-learning-2026/MNCAATourneySeeds.csv \
       data/raw/march-machine-learning-2026/MNCAATourneyCompactResults.csv
```

Expected: all four files present and non-empty.

- [ ] **Step 2: Time a single anchor cell**

```bash
python -c "
import time
t0 = time.time()
from src.sweep_v9_weights import run_single_cell
m = run_single_cell(
    w_upset=1.0, w_miss=0.0,
    pairwise_v4_csv='output/pairwise_v4.csv',
    results_csv='data/raw/march-machine-learning-2026/MNCAATourneyCompactResults.csv',
    seeds_csv='data/raw/march-machine-learning-2026/MNCAATourneySeeds.csv',
    out_dir='output/v9_sweep_profile',
)
print(f'elapsed: {time.time() - t0:.1f}s')
print(m)
"
```

Expected: completes in under ~3 minutes per cell. Total 15-cell runtime: under ~45 min.

- [ ] **Step 3: Decision gate**

If the single cell took more than 3 minutes:
- Reduce the grid to W_MISS in {0.0, 1.0} (10 cells) by editing
  `W_MISS_VALUES` in `src/sweep_v9_weights.py` to `[0.0, 1.0]`. Re-run
  the test suite (`pytest tests/test_sweep_v9_weights.py -v`); the
  cell-count test (`test_grid_has_15_unique_cells`) will fail and
  needs to be updated to match. State the change in your final note.
- Otherwise: proceed to Task 7.

If `output/v9_sweep_profile/pairwise_v9_WU1.00_WM0.00.csv` differs
from v8 by more than 1 bracket point, halt: the threading is wrong.
Re-read `compute_sample_weights` and `build_v9_pairwise` and confirm
the kwargs reach `compute_sample_weights`.

- [ ] **Step 4: Clean up profile output**

```bash
rm -rf output/v9_sweep_profile
```

(no commit needed -- this is a runtime check, no code changes)

---

## Task 7: Run the full sweep

**Files:** generates `output/v9_sweep/*.csv` and `output/v9_sweep_results.csv`. No source files modified. (output/ is gitignored.)

- [ ] **Step 1: Run**

```bash
python src/sweep_v9_weights.py 2>&1 | tee output/v9_sweep_run.log
```

Expected output (roughly):
- 15 lines of `[cell N/15] W_UPSET=..., W_MISS=...` followed by
  `total_brkt_pts=..., ll=..., acc=...`.
- A sorted-by-brkt-pts table.
- v8 baseline + anchor delta lines.
- WINNER / NO WINNER line.

- [ ] **Step 2: Verify the anchor-cell sanity gate**

Look at the printed `Anchor cell reproduces v8 within 1 pt -- sweep
is valid.` line. If instead it says `WARNING: anchor cell does not
reproduce v8 within 1 pt; sweep is invalid.`, halt and debug; do
not write the findings note. Likely causes: feature-list drift,
seed-lookup edge case, or bracket-scoring tournament mismatch.

- [ ] **Step 3: Verify the sweep results CSV is well-formed**

```bash
python -c "
import pandas as pd
df = pd.read_csv('output/v9_sweep_results.csv')
print(df)
assert len(df) == 15  # or 10 if Task 6 reduced the grid
assert df['total_brkt_pts'].is_monotonic_decreasing  # sorted desc
"
```

Expected: 15 rows, sorted descending. (10 if grid was reduced in
Task 6.)

(no commit -- output files are gitignored)

---

## Task 8: Findings note + TODO update

**Files:**
- Create: `docs/notes/2026-05-01-v9-weight-sweep.md`
- Modify: `TODO.md`

- [ ] **Step 1: Write the findings note**

Capture the actual numbers from `output/v9_sweep_results.csv`. The
note must include:

- Decision bar (+10) and rationale
- v8 baseline total bracket pts
- Anchor cell delta (must be within 1 pt; state the actual)
- Sorted results table (15 rows, formatted plain-text)
- Winner: yes/no, and which cell if yes
- Recommendation:
  - If WINNER: recommend swapping that (W_U, W_M) into production
    (update `W_UPSET` and `W_MISS` defaults in
    `src/train_upset_model.py`, regenerate `pairwise_v9.csv`,
    update bracket pipeline), and recommend rerunning the same
    sweep with v9-B (after fixing v9-B's round-asymmetry bug).
  - If NO WINNER: recommend closing the open question and moving
    to the next active-queue item (ensemble of model classes).
- Caveats / known limitations.

Suggested template (fill in actual numbers):

```markdown
# v9 Upset-Weight Tuning Sweep -- Findings (2026-05-01)

**Spec:** docs/superpowers/specs/2026-05-01-v9-weight-sweep.md
**Plan:** docs/superpowers/plans/2026-05-01-v9-weight-sweep.md
**Predecessor finding:** docs/notes/2026-04-30-upset-detection-v9.md

## Verdict

**[WINNER (W_U=..., W_M=...) / NO WINNER -- close open question]**

[1-2 sentence summary]

## Method

15-cell sweep over W_UPSET in {1.0, 1.25, 1.5, 1.75, 2.0} x W_MISS in
{0.0, 0.5, 1.0}, using v9-A (4 features). Each cell: 22-season double
LOSO, scored with score_pairwise_path against
MNCAATourneyCompactResults.csv. Decision bar: best cell total bracket
pts > v8 + 10. Anchor cell (1.0, 0.0) sanity gate: must reproduce v8
within 1 pt.

v8 baseline: ___ pts.
Anchor cell (1.0, 0.0): ___ pts (delta ___). [Sanity gate PASSED / FAILED.]

## Results

| W_UPSET | W_MISS | brkt_pts | LL    | Acc   | delta vs v8 |
|---------|--------|----------|-------|-------|-------------|
| ...     | ...    | ...      | ...   | ...   | ...         |
| ... (15 rows)                                              |

## Recommendation

[Per Verdict.]

## Follow-ups

[v9-B sweep (if winner) / ensemble work (if no winner)]

## Caveats

[Note grid reduction if Task 6 trimmed it. Note any sanity-gate
near-misses.]
```

- [ ] **Step 2: Update TODO.md**

If WINNER: add a Done entry for the sweep with the recommendation,
and add a new active-queue item to repeat the sweep with v9-B (after
fixing the round-asymmetry bug).

If NO WINNER: add a Done entry recording the verdict, and confirm the
existing top of the active queue ("Ensemble of model classes") stays
at #1.

Either way, ensure the TODO.md reads coherently. Do not duplicate
content -- summarize and link to the findings note.

- [ ] **Step 3: Verify ASCII-only**

```bash
python -c "open('docs/notes/2026-05-01-v9-weight-sweep.md').read().encode('ascii')"
python -c "open('TODO.md').read().encode('ascii')"
```

Expected: no errors.

- [ ] **Step 4: Run the full test suite**

```bash
pytest -q
```

Expected: all tests pass (113 tests, 1 skipped: 111 baseline + 3 new
upset-model + 7 new sweep). State the actual count in your final
message as evidence per CLAUDE.md "FORCED VERIFICATION".

- [ ] **Step 5: Commit**

```bash
git add docs/notes/2026-05-01-v9-weight-sweep.md TODO.md
git commit -m "$(cat <<'EOF'
docs: v9 upset-weight sweep findings + TODO update

[ONE-SENTENCE VERDICT: WINNER (W_U=..., W_M=..., +__ pts) / NO WINNER]

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 9: Final verification + PR

**Files:** none modified.

- [ ] **Step 1: Re-run the full pytest suite at the repo root**

```bash
pytest -v 2>&1 | tail -20
```

Expected: 121 passed, 1 skipped (111 + 3 new upset-model + 7 new
sweep). State exact pass/fail counts in the final message.

- [ ] **Step 2: Verify clean git status**

```bash
git status --short
```

Expected: empty (or only output/* files, which are gitignored).

- [ ] **Step 3: Push branch and open PR**

```bash
git push -u origin feat/upset-weight-sweep
```

Then ask the user to open the PR (do not auto-open). PR title:
`v9 upset-weight tuning sweep: [WINNER / NO WINNER]`. Body:
1-2 sentence summary, link to spec/plan/findings.

---

## Summary of expected deliverables

After completing all tasks:

1. `src/train_upset_model.py` -- `double_loso_eval` and `build_v9_pairwise` accept `w_upset` / `w_miss` kwargs (defaults preserve existing behavior).
2. `src/sweep_v9_weights.py` -- new sweep driver with grid + anchor validation + main().
3. `tests/test_upset_model.py` -- 13 tests (10 existing + 3 new).
4. `tests/test_sweep_v9_weights.py` -- 7 new tests.
5. `output/v9_sweep_results.csv` + `output/v9_sweep/pairwise_v9_*.csv` -- 15-cell sweep artifacts (gitignored).
6. `docs/notes/2026-05-01-v9-weight-sweep.md` -- findings + recommendation.
7. `TODO.md` -- updated with verdict.
8. PR opened with branch `feat/upset-weight-sweep`.
