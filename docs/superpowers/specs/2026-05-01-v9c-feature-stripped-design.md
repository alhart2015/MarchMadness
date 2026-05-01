# v9-C Feature-Stripped Variant -- Design

**Date:** 2026-05-01
**Branch:** feat/v9-b-followup
**Predecessors:**
- v9 spec: `docs/superpowers/specs/2026-04-30-upset-detection-design.md`
- v9 weight-sweep spec: `docs/superpowers/specs/2026-05-01-v9-weight-sweep.md`
- v9 weight-sweep findings: `docs/notes/2026-05-01-v9-weight-sweep.md`
- v9-B round-fix spec: `docs/superpowers/specs/2026-05-01-v9-round-fix.md`
- v9-B round-fix findings: `docs/notes/2026-05-01-v9-round-fix.md`

## Motivation

The v9-B round-fix findings recommended candidate-only status for the
post-fix winner (W_UPSET=1.25, W_MISS=0.0 at +20 brkt pts vs v8) and
explicitly flagged a structural concern as the next experiment to run:

> v9-B's structural quirks unaddressed: 7 features still include
> `v4_confidence` and `is_a_higher_seed` which the model may or may
> not be using productively. A v9-A (4-feature) variant -- not in
> the merged code -- could be a cleaner test.

v9-A as originally tested (PR 6) had 4 features (`p_v4_stage1`,
`seed_a`, `seed_b`, `abs_seed_diff` -- no `round`) and was only run
at the catastrophic high weights (W_UPSET=3.0, W_MISS=4.0), which
established that high weights destroy the model regardless of feature
set. v9-A has never been tested at the productive low-weight cells
identified by PR 7 / PR 8, and removing only `v4_confidence` and
`is_a_higher_seed` while keeping the round signal we just paid for
in PR 8 has never been tested at all.

This spec defines that variant -- **v9-C** = v9-A's original 4
features + the apply-time-correct `round` feature (5 features
total) -- and re-runs PR 8's identical 15-cell sweep on it. Outcome
either (a) confirms `v4_confidence` + `is_a_higher_seed` were noise
and produces a cleaner / stronger candidate, or (b) shows v9-B's
extra features were carrying signal, in which case PR 8's marginal
+20 stands as the v9 ceiling and the thread closes.

## Scope

**In scope.**

- A `feature_set` parameter on the v9 trainer that selects between
  v9-B (7 features, current behavior) and v9-C (5 features). Default
  preserves v9-B for backward compatibility.
- Threading the parameter through `upset_features`, `double_loso_eval`,
  `build_v9_pairwise`, and the sweep driver (`run_single_cell`,
  `run_sweep`, `main`).
- Re-running PR 8's identical 15-cell weight grid against v9-C, with
  the same anchor sanity gate (W_U=1.0, W_M=0.0 within 5 brkt pts of
  v8) and the same decision bars (+10 = marginal candidate;
  +25 or better F4/E8 accuracy = swap-in consideration).
- Per-cell pairwise CSVs and a results CSV named to coexist with
  PR 8's v9-B artifacts (output suffix `v9c_*` instead of `v9_*`).
- Findings note documenting before/after deltas, per-season
  decomposition for the v9-C winner, and a swap-or-don't-swap
  recommendation. Same template as PR 8's findings note for direct
  comparability.

**Out of scope.**

- Hyperparameter tuning of XGBoost (n_estimators, max_depth, lr,
  reg_alpha, reg_lambda). Same model architecture as v9-B; only the
  feature set changes.
- A finer-resolution grid (e.g., {1.0, 1.1, 1.15, 1.2, 1.25, 1.3} x
  {0, 0.25, 0.5}). PR 8's grid is reused so v9-C's per-cell numbers
  are directly comparable to v9-B's. If v9-C produces a clear winner
  in the productive region, a finer follow-up grid is a separate
  cheap experiment.
- Other feature-set variants beyond v9-C (e.g., dropping only
  `v4_confidence`, dropping only `is_a_higher_seed`, adding new
  features). The hypothesis under test is "the two added features
  are noise"; the cleanest test removes both at once.
- Round-conditional weighting (per-round W_UPSET). Adds knobs
  without addressing the structural-quirk question this spec asks.
- Forking a parallel `train_upset_model_v9c.py` module. Two
  near-identical training drivers in the repo is the kind of
  divergence CLAUDE.md's reuse-before-writing rule warns against.
- Production swap. If v9-C wins by +25 or shows distinctly better
  F4/E8 accuracy, a follow-up commit can swap defaults; that is
  separate from this spec's scope.

## Approach

### Parameterization

Add `feature_set: str` to `upset_features` in
`src/train_upset_model.py` with accepted values `"v9b"` (current
7-feature default) and `"v9c"` (new 5-feature variant). Raise
`ValueError` on any other value -- typos must fail fast rather than
silently degrade.

```python
def upset_features(df: pd.DataFrame, feature_set: str = "v9b") -> np.ndarray:
    """Pull the v9 input matrix from a per-game DataFrame.

    feature_set:
      "v9b" (default, 7 features): p_stage1, seed_a, seed_b,
        abs_seed_diff, round, v4_confidence (|p_stage1 - 0.5|),
        is_a_higher_seed (1.0 if seed_a < seed_b else 0.0).
      "v9c" (5 features): drops v4_confidence and is_a_higher_seed,
        keeps the other five.
    """
    if feature_set not in ("v9b", "v9c"):
        raise ValueError(f"unknown feature_set {feature_set!r}; "
                         "must be 'v9b' or 'v9c'")
    p = df["p_stage1"].values.astype(float)
    sa = df["seed_a"].values.astype(float)
    sb = df["seed_b"].values.astype(float)
    diff = df["abs_seed_diff"].values.astype(float)
    rnd = df["round"].values.astype(float)
    if feature_set == "v9b":
        conf = np.abs(p - 0.5)
        higher = (sa < sb).astype(float)
        return np.column_stack([p, sa, sb, diff, rnd, conf, higher])
    return np.column_stack([p, sa, sb, diff, rnd])  # v9c
```

Thread `feature_set` through `double_loso_eval` and `build_v9_pairwise`
as a kwarg, default `"v9b"`. Both functions pass it on every call
to `upset_features`. No other changes to the trainer file.

### Sweep driver

Add `feature_set: str = "v9b"` to `run_single_cell` and `run_sweep`
in `src/sweep_v9_weights.py`. Pass it through to `build_v9_pairwise`
and `double_loso_eval`.

In `main()`, pick the feature set via env var
`V9_FEATURE_SET` (default `"v9b"`). Output paths and printed banner
key off the choice:

- `feature_set="v9b"`: existing paths
  (`output/v9_sweep/`, `output/v9_sweep_results.csv`).
- `feature_set="v9c"`: new paths
  (`output/v9c_sweep/`, `output/v9c_sweep_results.csv`).

Per-cell CSV name keeps the `pairwise_v9_WU{u}_WM{m}.csv` template
inside the variant-specific output dir, so file names within each
variant's directory are unchanged. Both variants' outputs coexist
without collision.

The sweep is invoked end-to-end by setting the env var:

```sh
V9_FEATURE_SET=v9c python src/sweep_v9_weights.py
```

Anchor cell `(W_U=1.0, W_M=0.0)` for v9-C is checked against v8 by
the same gate the existing sweep driver uses (`sweep_v9_weights.py`
line 222: prints WARNING if `abs(delta) > 5.0` brkt pts; sweep
continues either way). The gate is informational, not a halt.
Removing two features may move v9-C's apply-time predictions by
more than v9-B's +7 anchor delta -- if the gate fires, the operator
must inspect per-game LL/Acc (expected to match v8's 0.4323 / 80.7%
to 3 decimals; if they do, the trainer is sane and the brkt-pt
delta is from chalk-pick boundary effects, not calibration drift).
Per-game LL/Acc materially out of band would indicate a bug in the
parameterization and require debugging before trusting any cell.

### Tests (TDD)

Add to `tests/test_upset_model.py`:

- **`test_upset_features_v9b_default_shape_unchanged`** -- existing
  test renamed if needed; assert `upset_features(df)` returns
  shape `(n_rows, 7)`.
- **`test_upset_features_v9b_explicit_matches_default`** -- assert
  `upset_features(df, feature_set="v9b")` is bit-identical to the
  default.
- **`test_upset_features_v9c_shape_5`** -- assert
  `upset_features(df, feature_set="v9c")` returns shape `(n_rows, 5)`.
- **`test_upset_features_v9c_columns_match_v9b_subset`** -- assert
  v9-C column 0..4 equals v9-B column 0..4 elementwise (same five
  base features, same order).
- **`test_upset_features_invalid_feature_set_raises`** -- assert
  `upset_features(df, feature_set="v9a")` raises ValueError.
- **`test_double_loso_eval_v9c_runs`** -- on the existing 2-season
  fixture, run `double_loso_eval(per_game, feature_set="v9c")` and
  assert it returns a non-empty DataFrame with `ll_v9` and `acc_v9`
  columns. Numerical values are not asserted (XGBoost on a tiny
  fixture).
- **`test_build_v9_pairwise_v9c_writes_csv`** -- on the existing
  fixture, run `build_v9_pairwise(..., feature_set="v9c")` and
  assert the output CSV has the v8-compatible schema and row count
  equal to the input pairwise.

Add to `tests/test_sweep_v9_weights.py`:

- **`test_run_single_cell_feature_set_v9c`** -- run the existing
  synthetic-data fixture with `feature_set="v9c"` and assert the
  returned dict has the same keys as the v9-B path and that
  `pairwise_csv` exists on disk.
- **`test_run_sweep_anchor_validation_v9c`** -- assert the same
  anchor-validation error raises if anchor is missing, regardless
  of feature_set.

All existing v9-B tests must continue to pass unchanged.

### Output naming

| Artifact | v9-B (existing) | v9-C (new) |
|---|---|---|
| Per-cell pairwise CSV dir | `output/v9_sweep/` | `output/v9c_sweep/` |
| Sweep results CSV | `output/v9_sweep_results.csv` | `output/v9c_sweep_results.csv` |
| Driver log (manual capture) | `output/v9_sweep_run.log` | `output/v9c_sweep_run.log` |
| Findings note | `docs/notes/2026-05-01-v9-round-fix.md` | `docs/notes/2026-05-01-v9c-feature-stripped.md` |

## Success criteria

- `pytest -v` passes (existing tests + new v9-C tests).
- v9-C anchor cell `(W_U=1.0, W_M=0.0)` reproduces v8 within 5
  brkt pts (same gate as PR 8). Per-game LL/Acc match v8's
  `0.4323 / 80.7%` to 3 decimals.
- 15-cell v9-C sweep completes (PR 8 ran in ~5 minutes; v9-C should
  be similar -- fewer features, slightly faster fit per cell).
- Findings note exists with: full 15-cell v9-C results table sorted
  by `total_brkt_pts`, per-cell delta vs v8 AND vs PR 8's v9-B at
  the same cell, per-season decomposition for v9-C's winner, and
  a swap-or-don't-swap recommendation.

## Decision matrix (recommendation taken in the findings note)

| v9-C winner delta vs v8 | F4/E8 accuracy vs v8 | Recommendation |
|---|---|---|
| > +25 | any | Swap candidate. Document as production-swap path; separate follow-up commit handles default flips and bracket-pipeline pointers. |
| +10 to +25 | distinctly better than v8 *and* v9-B | Swap candidate (F4/E8 lens). Same handling as above. |
| +10 to +25 | comparable to v8 / v9-B | Document as candidate, do not swap. Close v9 thread; advance to active-queue #1 (ensemble). |
| <= +10 (or worse than v9-B) | any | Negative result. Document and close v9 thread for real. |

The +25 bar matches the swap-in language from the v9-B round-fix
findings; the F4/E8 lens matches the "or" clause from that note.

## Risks and mitigations

- **Regression on v9-B baseline.** Parameterization touches several
  call sites and could subtly change v9-B behavior. Mitigation:
  v9-B remains the default for `feature_set`; existing v9-B tests
  cover default behavior; the v9-B anchor cell can be re-run as a
  spot check (`V9_FEATURE_SET=v9b python src/sweep_v9_weights.py`).
  Since the parameterization is a code-only refactor with v9-B as
  the default branch, v9-B's anchor pairwise CSV is expected to be
  bit-identical to PR 8's (xgboost seed=42; PR 7 already verified
  bit-reproducibility on a re-run). If it differs, the
  parameterization changed v9-B behavior unintentionally and must
  be debugged before running v9-C.
- **v9-C anchor drift exceeds 5 pts.** Possible if removing two
  features moves the model's apply-time predictions enough to flip
  several chalk picks. Mitigation: the gate prints a WARNING (does
  not halt); operator inspects per-game LL/Acc to confirm the
  trainer is still calibrated. If LL/Acc match v8 to 3 decimals,
  proceed and document the brkt-pt drift in the findings as a
  feature-set effect, not a bug. If LL/Acc are out of band, halt
  manually and debug the parameterization before continuing.
- **Multiple-comparisons inflation.** Same 15 cells x v9-C is the
  16th-31st model in the v9 line of inquiry. Mitigation: same +10
  bar for "candidate," +25 for "swap." Per-season decomposition in
  the findings note is the primary check on whether the win is
  concentrated (fragile) or spread (durable). PR 8's winner had a
  4W-2L spread with max single-season delta +8 -- v9-C's winner
  needs a similar or better profile to be considered.
- **v9-C wins similar magnitude to v9-B (~+20).** Possible if
  XGBoost was effectively ignoring the two extra features anyway,
  or if a different cell wins by a similar margin. Outcome
  interpretation: the dropped features were not load-bearing; v9-C
  is the cleaner test but does not point to a stronger production
  candidate. Falls in the "+10 to +25, comparable to v8 / v9-B"
  bucket of the decision matrix; document as candidate, do not
  swap, close v9 thread.

## Follow-ups (not in this spec)

- If v9-C wins by +25+ vs v8: production-swap commit (defaults in
  `train_upset_model.py`, bracket pipeline pointers from
  `pairwise_v8.csv` to `pairwise_v9c.csv`, regenerate 2026 chalk
  bracket). Same branch, separate commit.
- If v9-C is comparable to v9-B (~+20): finer-resolution grid
  around the productive region as a separate cheap follow-up,
  documented as the last v9 experiment before closing the thread.
- If v9-C closes the thread: TODO.md update marking the v9 line
  closed for real, and explicit promotion of active-queue item #1
  (ensemble of model classes) to next-up.
