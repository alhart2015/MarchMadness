# Hierarchical Bradley-Terry with v4 Feature Priors -- Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Per-season hierarchical BT MAP solver with `s_team ~ Normal(beta . v4_features_team, sigma^2)` prior, swept over a sigma grid and gated against v4 with the same 3-clause diagnostic from the plain-BT experiment. If any cell passes, run a v9-C head-to-head and apply the standard ladder.

**Architecture:** Joint `(s, beta, h)` MAP via L-BFGS in a new module `src/features/hierarchical_bt.py`; CLI trainer `src/train_hbt_stage1.py` that loops sigmas x seasons and writes one pairwise CSV per cell; gate runner `src/diagnose_hbt_vs_v4.py` mirroring the plain-BT diagnostic.

**Tech Stack:** SciPy (optimize.minimize L-BFGS-B), NumPy (sparse + dense), pandas, scikit-learn (already a dep), pytest.

**Spec:** `docs/superpowers/specs/2026-05-03-hierarchical-bt-feature-priors-design.md`

---

## File Structure

**Created (committed):**

- `src/features/hierarchical_bt.py` -- solver + predictor (~250 LOC).
  - Public: `fit_one_season(games_df, feature_matrix, sigma, sigma_beta, feature_cols, feature_means, feature_stds, max_iter, tol) -> dict`
  - Public: `predict_pairs(fit_result, pair_team_ids) -> np.ndarray`
  - Private: `_neg_log_posterior(theta, ...) -> (float, np.ndarray)` -- returns (loss, grad) for L-BFGS
  - Private: `_pack(s, beta, h) -> theta`, `_unpack(theta, n_teams, n_features) -> (s, beta, h)`
  - Module constant: `_PRODUCER_VERSION = "v1"`
- `src/train_hbt_stage1.py` -- CLI trainer (~200 LOC). Loops sigma x season, writes one CSV per sigma cell.
- `src/diagnose_hbt_vs_v4.py` -- gate runner CLI (~250 LOC). Mirror of `src/diagnose_bt_vs_v4.py` with per-cell aggregation.
- `tests/test_features/test_hierarchical_bt.py` -- 5 unit tests (~150 LOC).
- `tests/test_train_hbt_stage1.py` -- 1 smoke test (~60 LOC).
- `tests/test_diagnose_hbt_vs_v4.py` -- 2 unit tests (~80 LOC).

**Modified:**

- None to v4 / production code. This branch adds parallel modules only.

**Generated (committed via `git add -f`):**

- `output/pairwise_hbt_sigma_<S>.csv` x 7 cells (one per sigma).
- `output/diag_hbt_sweep.json`.
- `docs/notes/2026-05-03-hierarchical-bt.md` (findings).
- If any cell passes: `output/pairwise_hbt_ensemble.csv`, `output/pairwise_v9c_hbt_ensemble.csv`, additional backtest section in findings.

**Generated (NOT committed):**

- None. No parquet caches needed -- the per-cell CSVs are the artifacts.

---

## Phase 1: Solver core + unit tests

Goal: a correct, well-tested `fit_one_season` whose anchor cells reproduce expected behavior on synthetic data. Phase 1 must pass before any LOSO loop runs against real data.

### Task 1: Implement `_pack` / `_unpack` and the joint negative log-posterior with analytic gradient

**Files:**
- Create: `src/features/hierarchical_bt.py`
- Test: `tests/test_features/test_hierarchical_bt.py` (gradient-check test only at this stage)

**Math reference:**

```
theta = concat([s (n_teams,), beta (n_features,), h (1,)])

L(theta) = sum_{games g} -log sigmoid(s[w_g] - s[l_g] + home_g * h)
         + (1 / (2 sigma^2))      * ||s - X @ beta||^2
         + (1 / (2 sigma_beta^2)) * ||beta||^2

dL/ds[i]   = sum_{g where w_g==i} -(1 - p_g)
           + sum_{g where l_g==i} +(1 - p_g)
           + (1 / sigma^2) * (s[i] - X[i] @ beta)
dL/dbeta   = -(1 / sigma^2) * X.T @ (s - X @ beta) + (1 / sigma_beta^2) * beta
dL/dh      = sum_g -home_g * (1 - p_g)

where p_g = sigmoid(s[w_g] - s[l_g] + home_g * h)
```

- [ ] **Step 1: Write the failing gradient-check test**

Create `tests/test_features/test_hierarchical_bt.py` with a finite-difference check on a 4-team, 12-game synthetic season. The test calls `_neg_log_posterior` with random theta, compares the returned analytic gradient to a central-difference numerical gradient (`eps=1e-6`), asserts max abs error < `1e-5`. Use a fixed RNG seed.

```python
"""Unit tests for src/features/hierarchical_bt.py."""

import numpy as np
import pandas as pd
import pytest

from src.features.hierarchical_bt import (
    _PRODUCER_VERSION,
    _neg_log_posterior,
    _pack,
    _unpack,
    fit_one_season,
    predict_pairs,
)


def _synthetic_season(n_teams=4, n_games_per_pair=3, seed=0):
    """Round-robin-ish schedule with random outcomes weighted by an
    underlying true strength vector. Returns (games_df, team_ids,
    feature_matrix_df, true_s)."""
    rng = np.random.default_rng(seed)
    team_ids = list(range(100, 100 + n_teams))
    true_s = rng.normal(0, 1, size=n_teams)

    rows = []
    daynum = 10
    for i in range(n_teams):
        for j in range(i + 1, n_teams):
            for _ in range(n_games_per_pair):
                p_i_wins = 1.0 / (1.0 + np.exp(-(true_s[i] - true_s[j])))
                if rng.random() < p_i_wins:
                    w, l = team_ids[i], team_ids[j]
                else:
                    w, l = team_ids[j], team_ids[i]
                wloc = rng.choice(["H", "A", "N"])
                rows.append({
                    "Season": 2024, "DayNum": daynum,
                    "WTeamID": w, "LTeamID": l, "WLoc": wloc,
                })
                daynum += 1
    games_df = pd.DataFrame(rows)

    # 2-feature synthetic feature matrix, weakly correlated with true_s
    fm = pd.DataFrame({
        "TeamID": team_ids,
        "Season": [2024] * n_teams,
        "feat_a": true_s + rng.normal(0, 0.5, n_teams),
        "feat_b": rng.normal(0, 1, n_teams),
    })
    return games_df, team_ids, fm, true_s


def test_gradient_matches_finite_difference():
    games, team_ids, fm, _ = _synthetic_season(seed=42)
    feature_cols = ["feat_a", "feat_b"]
    n_teams, n_feat = len(team_ids), len(feature_cols)

    # Build standardization stats from the (single) season so the
    # call signature mirrors production usage.
    means = fm[feature_cols].mean()
    stds = fm[feature_cols].std(ddof=0).replace(0, 1)

    # Random theta
    rng = np.random.default_rng(0)
    theta = rng.normal(0, 0.5, n_teams + n_feat + 1)

    # Pack args
    args = _build_neg_log_posterior_args(
        games, team_ids, fm, feature_cols, means, stds,
        sigma=0.5, sigma_beta=1.0,
    )

    loss, grad = _neg_log_posterior(theta, *args)
    eps = 1e-6
    grad_fd = np.empty_like(theta)
    for k in range(len(theta)):
        tp = theta.copy(); tp[k] += eps
        tm = theta.copy(); tm[k] -= eps
        lp, _ = _neg_log_posterior(tp, *args)
        lm, _ = _neg_log_posterior(tm, *args)
        grad_fd[k] = (lp - lm) / (2 * eps)

    assert np.max(np.abs(grad - grad_fd)) < 1e-5, \
        f"max grad error {np.max(np.abs(grad - grad_fd)):.2e}"
```

The test imports `_build_neg_log_posterior_args` -- a small helper inside `hierarchical_bt.py` that prepares the constant-across-iterations matrices (sparse design rows, standardized X, etc.). Including the helper in the public-test interface keeps `_neg_log_posterior` itself clean of pandas concerns (it should take only NumPy arrays).

Run: `pytest tests/test_features/test_hierarchical_bt.py::test_gradient_matches_finite_difference` -- expect ImportError.

- [ ] **Step 2: Implement the solver module skeleton**

Create `src/features/hierarchical_bt.py` with imports, `_PRODUCER_VERSION = "v1"`, `_pack`, `_unpack`, `_build_neg_log_posterior_args` (returns a tuple of pre-computed constants), and `_neg_log_posterior(theta, *args) -> (loss, grad)`. Use the math in the section above. Use NumPy throughout; the design is small enough that dense beats sparse.

Notes:
- For numerical stability, compute `log_sigmoid(z) = -log1p(exp(-z))` and a clipped variant when `z` is large negative; or use `scipy.special.expit` and compute `1 - p` carefully.
- `home_g` value: `+1` if `WLoc=='H'`, `-1` if `WLoc=='A'`, `0` if `WLoc=='N'`. Same convention as `src/train_bt_stage1.py:extract_home_court_value`.
- Pre-compute team index arrays `w_idx` and `l_idx` (np.int64 arrays of length n_games) and the home-court vector once -- per-iteration cost should be O(n_games + n_teams^2 / sigma^2).

Run the gradient-check test. Expect PASS.

- [ ] **Step 3: Add identifiability + anchor unit tests**

Add to `tests/test_features/test_hierarchical_bt.py`:

```python
def test_loose_prior_recovers_plain_bt_ranking():
    """sigma very large -> recovers a strength ordering matching the
    underlying true_s used to simulate games. Loose prior shouldn't
    distort the BT MLE direction."""
    games, team_ids, fm, true_s = _synthetic_season(
        n_teams=6, n_games_per_pair=20, seed=1,
    )
    feature_cols = ["feat_a", "feat_b"]
    means = fm[feature_cols].mean()
    stds = fm[feature_cols].std(ddof=0).replace(0, 1)

    fit = fit_one_season(
        games, team_ids, fm, feature_cols, means, stds,
        sigma=1e3, sigma_beta=1e3,  # both priors near-uninformative
    )
    assert fit["success"], f"failed to converge: {fit}"

    # Spearman-rank correlation between fitted s and true_s should be high.
    from scipy.stats import spearmanr
    rho, _ = spearmanr(fit["s"], true_s)
    assert rho > 0.7, f"rank correlation {rho:.3f} too low"


def test_tight_prior_pulls_s_toward_X_beta():
    """sigma very small -> ||s - X @ beta|| approaches zero."""
    games, team_ids, fm, _ = _synthetic_season(
        n_teams=8, n_games_per_pair=30, seed=2,
    )
    feature_cols = ["feat_a", "feat_b"]
    means = fm[feature_cols].mean()
    stds = fm[feature_cols].std(ddof=0).replace(0, 1)

    fit = fit_one_season(
        games, team_ids, fm, feature_cols, means, stds,
        sigma=1e-3, sigma_beta=1.0,
    )
    assert fit["success"]

    s = fit["s"]
    Xz = ((fm.set_index("TeamID").loc[team_ids][feature_cols] - means) / stds).values
    pred = Xz @ fit["beta"]
    # In the sigma -> 0 limit, s should match X @ beta within ~1e-2.
    max_dev = np.max(np.abs(s - pred))
    assert max_dev < 5e-2, f"s deviates from X@beta by {max_dev:.3f}"


def test_predict_pairs_uses_only_strengths():
    games, team_ids, fm, _ = _synthetic_season(seed=3)
    feature_cols = ["feat_a", "feat_b"]
    means = fm[feature_cols].mean()
    stds = fm[feature_cols].std(ddof=0).replace(0, 1)

    fit = fit_one_season(
        games, team_ids, fm, feature_cols, means, stds,
        sigma=0.5, sigma_beta=1.0,
    )
    pairs = [(team_ids[0], team_ids[1]), (team_ids[1], team_ids[0])]
    probs = predict_pairs(fit, pairs)
    assert np.isclose(probs[0] + probs[1], 1.0, atol=1e-9), \
        "p(a beats b) + p(b beats a) must be 1 under symmetric BT"
    # No home-court term applied at predict time
    assert 0.0 < probs[0] < 1.0


def test_producer_version_constant():
    assert _PRODUCER_VERSION == "v1"
```

- [ ] **Step 4: Implement `fit_one_season` and `predict_pairs`**

Add to `src/features/hierarchical_bt.py`:

- `fit_one_season(games_df, team_ids, feature_matrix, feature_cols, feature_means, feature_stds, sigma, sigma_beta=1.0, max_iter=500, tol=1e-8) -> dict`

Steps inside:
1. Filter `feature_matrix` to the season's `team_ids` (in `team_ids` order). Standardize: `Xz = (X - feature_means) / feature_stds`.
2. Build the per-game arrays: `w_idx`, `l_idx`, `home`.
3. Pack constants via `_build_neg_log_posterior_args`.
4. `theta0 = np.zeros(n_teams + n_features + 1)`.
5. `result = scipy.optimize.minimize(_neg_log_posterior, theta0, args=args, method='L-BFGS-B', jac=True, options={'maxiter': max_iter, 'ftol': tol, 'gtol': tol})`.
6. Unpack to `(s, beta, h)`. Return dict with keys `s, beta, h, team_ids, success, n_iter, fun`.

`predict_pairs(fit, pairs) -> np.ndarray`: for each `(a, b)` pair, returns `sigmoid(s[a_idx] - s[b_idx])`. No home-court.

Run all tests in `tests/test_features/test_hierarchical_bt.py`. All four should pass.

- [ ] **Step 5: Verify with `pytest -v tests/test_features/test_hierarchical_bt.py`**

All five tests should pass. State the runtime in the PR description -- Phase 1 expected total < 30 sec.

**Phase 1 exit criterion:** all 5 unit tests green. Solver behavior at sigma extremes is anchored.

---

## Phase 2: CLI trainer + 2-season smoke

Goal: an end-to-end CLI that produces `output/pairwise_hbt_sigma_<S>.csv` for a given sigma list, validated on a small subset before running the full sweep.

### Task 2: Build `src/train_hbt_stage1.py`

**Files:**
- Create: `src/train_hbt_stage1.py`
- Test: `tests/test_train_hbt_stage1.py`

- [ ] **Step 1: Write the failing smoke test**

```python
"""Smoke test for src/train_hbt_stage1.py."""
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest


def test_smoke_two_seasons(tmp_path):
    """Train one sigma cell on 2024-2025 only and validate the output CSV."""
    out_dir = tmp_path / "output"
    out_dir.mkdir()
    cmd = [
        sys.executable, "src/train_hbt_stage1.py",
        "--sigmas", "1.0",
        "--seasons", "2024,2025",
        "--out-dir", str(out_dir),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, f"stderr: {proc.stderr}"

    csv_path = out_dir / "pairwise_hbt_sigma_1.0.csv"
    assert csv_path.exists()
    df = pd.read_csv(csv_path)
    assert set(df.columns) == {"season", "team_a", "team_b", "p_a_wins"}
    assert df["season"].isin([2024, 2025]).all()
    assert ((df["team_a"] < df["team_b"])).all()
    assert ((df["p_a_wins"] > 0) & (df["p_a_wins"] < 1)).all()
    # Rough pair-count sanity: ~2200 pairs/season for a 68-team field
    assert 1000 < len(df) < 5000
```

- [ ] **Step 2: Implement the trainer**

Pattern from `src/train_bt_stage1.py`. Key additions:

1. Import `prepare_loso_inputs` from `src.enhanced_model_v3`. Call once; reuse the returned `feature_matrix`, `feature_cols`, `regular_results`, `tourney_filtered`.
2. CLI flags: `--sigmas` (comma-separated floats), `--seasons` (comma-separated ints, default = all from feature_matrix), `--out-dir` (default `output/`), `--sigma-beta` (default `1.0`).
3. Filter `feature_cols` to numeric cols only (drop TeamID, Season, any string columns).
4. For each sigma in the requested list:
   a. Open `out_dir / f"pairwise_hbt_sigma_{sigma}.csv"` for writing; write header.
   b. For each held-out season `Y` in the requested list:
      - Compute standardization stats from `feature_matrix[feature_matrix.Season != Y][feature_cols]`. Drop columns with zero variance in the train fold.
      - Get this season's `team_ids` from `regular_results[regular_results.Season == Y]`.
      - Get this season's `games_df` filtered to `Season == Y`.
      - Call `fit_one_season(...)`. If `not fit["success"]`, log a warning with the season + sigma + final fun + n_iter; continue.
      - Get the tournament field (set of unique TeamIDs from `tourney_filtered[tourney_filtered.Season == Y]` across both teams). Enumerate unordered pairs `(a, b)` with `a < b`.
      - Compute `predict_pairs(fit, pairs)`. Append rows.
      - Print per-season log: `Season Y, sigma S: n_teams=..., n_games=..., h=..., success=..., n_iter=..., fit_time=...`.
   c. Close the CSV.
5. Print sweep summary: per-sigma fraction of seasons that converged, total wall time.

- [ ] **Step 3: Run the smoke test**

`pytest -v tests/test_train_hbt_stage1.py` -- expect PASS in < 5 minutes.

**Phase 2 exit criterion:** smoke test passes; one sigma cell on 2 seasons writes a valid CSV with the expected schema and pair count.

---

## Phase 3: Gate runner + sigma sweep on real data

Goal: run the full 7-cell x 22-season sweep, score the gate per cell, write the diagnostic JSON.

### Task 3: Build `src/diagnose_hbt_vs_v4.py`

**Files:**
- Create: `src/diagnose_hbt_vs_v4.py`
- Test: `tests/test_diagnose_hbt_vs_v4.py`

- [ ] **Step 1: Write failing unit tests**

```python
"""Unit tests for src/diagnose_hbt_vs_v4.py."""
import json
import numpy as np
import pandas as pd
import pytest

from src.diagnose_hbt_vs_v4 import (
    GATE_HEADROOM_MIN,
    GATE_R_MAX,
    GATE_W_HIGH,
    GATE_W_LOW,
    score_one_cell,
    pick_best_passing_cell,
)


def _hand_residual_inputs():
    """Two models with hand-computed residuals on 4 games."""
    # winner residual = 1 - p_winner
    # game outcomes: all "A" wins
    p_v4   = np.array([0.8, 0.6, 0.7, 0.9])
    p_hbt  = np.array([0.7, 0.5, 0.6, 0.8])
    # residuals: v4 = [.2, .4, .3, .1], hbt = [.3, .5, .4, .2]
    # corr(v4, hbt) = 1.0 (perfectly correlated)
    return p_v4, p_hbt


def test_score_one_cell_perfect_correlation_fails_clause_1():
    p_v4, p_hbt = _hand_residual_inputs()
    cell = score_one_cell(p_v4, p_hbt)
    assert cell["r"] > 0.99
    assert cell["passes_r"] is False


def test_pick_best_passing_cell_chooses_max_headroom():
    cells = [
        {"sigma": 0.1, "r": 0.5, "w_opt": 0.5, "headroom": 0.003,
         "passes_r": True, "passes_w": True, "passes_headroom": False},
        {"sigma": 1.0, "r": 0.55, "w_opt": 0.6, "headroom": 0.010,
         "passes_r": True, "passes_w": True, "passes_headroom": True},
        {"sigma": 5.0, "r": 0.55, "w_opt": 0.95, "headroom": 0.0,
         "passes_r": True, "passes_w": False, "passes_headroom": False},
    ]
    best = pick_best_passing_cell(cells)
    assert best["sigma"] == 1.0
```

- [ ] **Step 2: Implement the gate runner**

Pattern from `src/diagnose_bt_vs_v4.py`. Adapt:

1. Constants: `GATE_R_MAX = 0.60`, `GATE_W_LOW = 0.30`, `GATE_W_HIGH = 0.85`, `GATE_HEADROOM_MIN = 0.005`. Match plain-BT exactly.
2. CLI flags: `--pairwise-v4` (default `output/pairwise_v4.csv`), `--pairwise-hbt-glob` (default `output/pairwise_hbt_sigma_*.csv`), `--out` (default `output/diag_hbt_sweep.json`).
3. Auto-discover sigma cells from filenames (regex `pairwise_hbt_sigma_(?P<sigma>[0-9.]+)\.csv`).
4. For each cell: load, dedup'd-join on `(season, team_a, team_b)` against v4 + tournament outcomes; compute the same metrics as plain-BT diagnostic via `score_one_cell(p_v4_arr, p_hbt_arr) -> dict`. Returns `{r, w_opt, headroom, ll_v4, ll_hbt, ll_blend, disagreement_rate, passes_r, passes_w, passes_headroom, passes_all}`.
5. `pick_best_passing_cell(cells) -> dict | None`: filters to `passes_all=True`, picks `max(headroom)`, returns the dict (or None).
6. Write `diag_hbt_sweep.json` with `{cells: [...], best_passing_cell: ...}`.
7. Print a markdown table to stdout: `| sigma | r | w_opt | headroom | ll_hbt | verdict |`.
8. Exit code: 0 if best_passing_cell is not None, 1 otherwise.

- [ ] **Step 3: Run unit tests**

`pytest -v tests/test_diagnose_hbt_vs_v4.py` -- expect PASS.

### Task 4: Run the full sigma sweep on real data

- [ ] **Step 1: Verify v4 pairwise is up to date**

```bash
ls -la output/pairwise_v4.csv
head -3 output/pairwise_v4.csv
wc -l output/pairwise_v4.csv
```

The file should exist on `main` (force-added precedent) and cover 2003-2025. If missing, regenerate via `python src/enhanced_model_v3.py` first -- but this should be unnecessary on a fresh checkout.

- [ ] **Step 2: Run the trainer over the full sigma grid**

```bash
python src/train_hbt_stage1.py \
    --sigmas 0.05,0.10,0.20,0.50,1.00,2.00,5.00 \
    --out-dir output/ \
    2>&1 | tee output/train_hbt_log.txt
```

Expected wall time: ~5 minutes. Verify each `output/pairwise_hbt_sigma_*.csv` exists and has ~48,465 rows total across 22 seasons (matching plain-BT's pair coverage).

- [ ] **Step 3: Verify anchor cells**

Quickly score the extreme cells against expectations:

```bash
python -c "
import pandas as pd, numpy as np
for s in [0.05, 5.0]:
    df = pd.read_csv(f'output/pairwise_hbt_sigma_{s}.csv')
    print(f'sigma={s}: rows={len(df)}, mean_p={df.p_a_wins.mean():.3f}, std={df.p_a_wins.std():.3f}')
"
```

A sane sweep: mean `p_a_wins` near 0.5, std should grow as sigma increases (pure BT has the most extreme strength gaps).

- [ ] **Step 4: Run the gate**

```bash
python src/diagnose_hbt_vs_v4.py \
    --pairwise-v4 output/pairwise_v4.csv \
    --pairwise-hbt-glob 'output/pairwise_hbt_sigma_*.csv' \
    --out output/diag_hbt_sweep.json \
    2>&1 | tee output/diag_hbt_log.txt
```

Read the markdown table. Identify which cells pass all 3 clauses. Three outcomes:

| outcome                            | next                                   |
|------------------------------------|----------------------------------------|
| no cell passes (likely)            | proceed to Phase 5 (NO-GO findings)    |
| 1+ cells pass                      | proceed to Phase 4 (v9-C head-to-head) |
| anchor cells violate expectations  | debug -- do NOT proceed past this step |

**Phase 3 exit criterion:** sweep complete, gate JSON written, anchor cells (`sigma=5.0` close to plain BT, `sigma=0.05` close to v4 standalone) sane.

---

## Phase 4: Conditional v9-C head-to-head (skip if no cell passes)

Only run if `pick_best_passing_cell` returns a non-null cell.

### Task 5: Run v9-C ensemble + bracket scoring

- [ ] **Step 1: Build the ensemble at the best cell's optimal weight**

```bash
# Use w_opt from the diagnostic for the best cell.
SIGMA=$(jq -r '.best_passing_cell.sigma' output/diag_hbt_sweep.json)
W_V4=$(jq -r '.best_passing_cell.w_opt' output/diag_hbt_sweep.json)
W_HBT=$(python -c "print(1 - $W_V4)")
python src/ensemble_stage1.py \
    --in-a output/pairwise_v4.csv \
    --in-b output/pairwise_hbt_sigma_$SIGMA.csv \
    --weights $W_V4,$W_HBT \
    --out output/pairwise_hbt_ensemble.csv
```

- [ ] **Step 2: Run v9-C on both v4 baseline and the ensemble**

```bash
python src/run_v9c_on_stage1.py \
    --pairwise-in output/pairwise_v4.csv \
    --pairwise-out output/pairwise_v9c_v4_baseline.csv

python src/run_v9c_on_stage1.py \
    --pairwise-in output/pairwise_hbt_ensemble.csv \
    --pairwise-out output/pairwise_v9c_hbt_ensemble.csv
```

- [ ] **Step 3: Score both and compute the bracket-points delta**

Reuse the LR experiment's `score_chalk_brackets.score_pairwise_path` invocation pattern.

```bash
python -c "
from src.score_chalk_brackets import score_pairwise_path
v4 = score_pairwise_path('output/pairwise_v9c_v4_baseline.csv')
hbt = score_pairwise_path('output/pairwise_v9c_hbt_ensemble.csv')
print('v4+v9c   :', v4['total_pts'])
print('hbt+v9c  :', hbt['total_pts'])
print('delta    :', hbt['total_pts'] - v4['total_pts'])
print()
for s in sorted(v4['per_season']):
    print(f'  {s}: v4={v4[\"per_season\"][s]:.0f}  hbt={hbt[\"per_season\"][s]:.0f}  d={hbt[\"per_season\"][s]-v4[\"per_season\"][s]:+.0f}')
" | tee output/hbt_bracket_score.txt
```

- [ ] **Step 4: Apply the verdict ladder**

| delta             | verdict                                         |
|-------------------|-------------------------------------------------|
| `delta >= +25`    | CLEAR. Document; separate swap-in commit.       |
| `+10 <= d < +25`  | MARGINAL. Document but do NOT swap.             |
| `delta < +10`     | NO-GO. Document; v4 stays as stage-1.           |

Production swap is **not** part of this branch. If CLEAR, the followup commit lives in a separate PR mirroring the v9-C production-swap pattern (`docs/superpowers/specs/2026-05-01-v9c-production-swap-design.md`).

**Phase 4 exit criterion:** verdict assigned + numeric backing.

---

## Phase 5: Findings note + TODO update

Run regardless of gate / backtest verdict. Even a clean NO-GO is a valuable falsification record.

### Task 6: Write `docs/notes/2026-05-03-hierarchical-bt.md`

- [ ] **Step 1: Capture the verdict, gate numbers, and lesson**

Mirror the structure of `docs/notes/2026-05-01-bayesian-stage1.md` (the plain-BT findings):

```markdown
# Hierarchical Bradley-Terry with v4 Feature Priors -- Findings

**Date:** 2026-05-03
**Branch:** feat/hierarchical-bt-priors
**Verdict:** [NO-GO | MARGINAL | CLEAR]
**Spec:** docs/superpowers/specs/2026-05-03-hierarchical-bt-feature-priors-design.md
**Plan:** docs/superpowers/plans/2026-05-03-hierarchical-bt-feature-priors.md

## TL;DR
[1-paragraph summary including the diversity-strength frontier framing.]

## Setup recap
[Implementation summary, total wall time.]

## Sigma sweep result table
| sigma | r        | w_opt | headroom | ll_hbt | clause 1 | clause 2 | clause 3 | verdict |
|-------|----------|-------|----------|--------|----------|----------|----------|---------|
[per-cell rows from diag_hbt_sweep.json]

## Diversity-strength frontier
| candidate          | residual r | standalone LL | optimal w | gate verdict |
|--------------------|------------|---------------|-----------|--------------|
| LR (PR 11)         | 0.77       | 0.498         | 0.93      | NO-GO        |
| plain BT (PR 12)   | 0.58       | 0.565         | 0.98      | NO-GO        |
| HBT @ best sigma   | ?          | ?             | ?         | ?            |
[Concrete reading: did the HBT cells thread the needle, or did they
 trace a curve from LR-like to plain-BT-like as sigma swept?]

## Falsification reasoning / lesson
[How this constrains the next attempt. If NO-GO across all cells, frame
 as: "the diversity-strength frontier at v4 data scale appears to be
 a Pareto curve, not a tradeoff with a useful interior optimum.
 Attempts to lift BT's standalone strength via v4 priors regress
 residual correlation by approximately the same amount." This makes
 item #4 (full Bayesian) the next swing -- or item #2 (external
 rankings) which doesn't try to engineer diversity from internal
 data.]

## Files of record
[List of created files + the 3 thresholds + how to reuse them.]
```

- [ ] **Step 2: Update `TODO.md`**

Move active queue item #1 to "Tried and rejected" with a one-paragraph summary including the verdict + key numbers + branch name. If verdict is CLEAR, the move is to "Done" instead with a note that the production swap is a separate followup commit.

Renumber the active queue (item #2 -> #1, etc).

- [ ] **Step 3: Force-add output artifacts**

```bash
git add -f output/pairwise_hbt_sigma_*.csv output/diag_hbt_sweep.json
# only if Phase 4 ran:
git add -f output/pairwise_hbt_ensemble.csv output/pairwise_v9c_hbt_ensemble.csv output/pairwise_v9c_v4_baseline.csv output/hbt_bracket_score.txt
```

- [ ] **Step 4: Verify ASCII on all written .md files**

```bash
for f in docs/superpowers/specs/2026-05-03-hierarchical-bt-feature-priors-design.md \
         docs/superpowers/plans/2026-05-03-hierarchical-bt-feature-priors.md \
         docs/notes/2026-05-03-hierarchical-bt.md \
         TODO.md; do
    python -c "open('$f').read().encode('ascii')" && echo "$f OK" || echo "$f FAIL"
done
```

- [ ] **Step 5: Run full pytest sweep**

```bash
pytest -v tests/test_features/test_hierarchical_bt.py \
          tests/test_train_hbt_stage1.py \
          tests/test_diagnose_hbt_vs_v4.py
pytest -v   # full suite -- nothing else should regress
```

State the runtime + pass count in the final commit message.

**Phase 5 exit criterion:** findings note complete, TODO updated, all tests green, ASCII-clean.

---

## Risks (carried from spec)

1. **No passing cell.** Most likely outcome; the falsification record is the deliverable.
2. **Optimizer non-convergence at extreme sigma.** Logged per season + cell; cells with > 5% non-convergence get a warning annotation in the findings.
3. **Standardization leakage.** Per-LOSO-fold standardization is enforced by Task 2 Step 2; verify by inspection in code review.
4. **`prepare_loso_inputs` is slow on cold cache** (~few minutes). One-time cost per trainer invocation.

## Out-of-scope (carried from spec)

- Full Bayesian (PyMC / NumPyro / Stan).
- Margin-aware hierarchical BT.
- Cross-season pooling for `beta`.
- Sigma_beta sweep (only triggered if anchor `sigma=0.05` fails to recover v4-like LL).
- Production swap / live-bracket wiring.
