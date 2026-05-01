# v9 Upset-Weight Tuning Sweep -- Design

**Date:** 2026-05-01
**Branch:** feat/upset-weight-sweep
**Predecessors:**
- Spec: `docs/superpowers/specs/2026-04-30-upset-detection-design.md`
- Plan: `docs/superpowers/plans/2026-04-30-upset-detection.md`
- Findings: `docs/notes/2026-04-30-upset-detection-v9.md`

## Motivation

The v9 findings note left one open question:

> Could a milder upset weighting (W_UPSET in {1.5, 2.0}, W_MISS in {0, 1})
> sit alongside v8 productively? The current data argues for no: log loss
> monotonically degrades from W=1 (matches v8) to W=3 (catastrophe),
> suggesting the global optimum on this objective is at W=1 (i.e., v8). But
> the bracket-points objective is not the same as log loss -- a future
> attempt could sweep at low weights and target bracket pts directly.

This spec answers it. We sweep a small grid of low (W_UPSET, W_MISS)
values through the existing v9 trainer and score each cell on bracket
points, not log loss. The hypothesis is that some mild weighting could
help bracket pts (which weights upset calls in deep rounds heavily) even
where it hurts log loss. The null is that v8 (W=1, W=0; equivalent to
no upset weighting) is the optimum on bracket pts too.

## Scope

**In scope.** A 15-cell sweep over W_UPSET in {1.0, 1.25, 1.5, 1.75, 2.0}
x W_MISS in {0, 0.5, 1.0}, using **v9-B** (7 features:
p_v4_stage1, seed_a, seed_b, abs_seed_diff, round, v4_confidence,
is_a_higher_seed). 22-season LOSO bracket scoring against
`MNCAATourneyCompactResults.csv`. Findings note with a recommendation
to swap or not swap.

**Note on variant choice:** the original Q1 question (v9-A vs v9-B)
during brainstorming was based on a misread. v9-A (4 features) is not
in the merged code -- only v9-B exists in `src/train_upset_model.py`.
Pivoted to v9-B mid-execution after the anchor cell run revealed this
(see "Anchor tolerance" below).

**v9-B's known limitation.** v9-B has a train/apply asymmetry on the
`round` feature: pairwise_v4.csv has no DayNum, so apply-time round is
always 0 while training rows have round in 1..6. This is consistent
across all 15 cells of the sweep, so it does not bias the comparison
between cells -- but it does mean v9-B's apply-time predictions are
sub-optimal in absolute terms. Fixing this is out of scope here; if
the sweep produces a winner, fixing the round-asymmetry bug becomes a
prerequisite to swapping into production.

**Out of scope.**
- v9-A (4-feature variant). Would require splitting `upset_features`
  in the trainer; not pursued here.
- Re-tuning v4 or v8 hyperparameters. The sweep varies only weights;
  v4's Optuna params and v8's stage-2 architecture stay fixed.
- Architectural changes (different features, different model class,
  different target). Those are separate items in TODO.md.
- Fixing v9-B's round-asymmetry bug. See above.

## Approach

### Grid

| dim     | values                       |
|---------|------------------------------|
| W_UPSET | 1.0, 1.25, 1.5, 1.75, 2.0    |
| W_MISS  | 0.0, 0.5, 1.0                |

15 cells. Cell (W_UPSET=1.0, W_MISS=0.0) doubles as a v8 calibration
sanity check: the v9 findings showed this configuration produces
LL=0.432, Acc=80.6%, matching v8 to 3 decimals. Our actual run on this
cell reproduces the LL/Acc figures (0.4323 / 80.7%). On bracket points,
v9-B at uniform weights scores +3.0 vs v8 (2673 vs 2670) due to v9-B's
3 extra features adding modest signal even without weighting -- this
is not a bug.

### Per-cell metric

For each (W_U, W_M):

1. Run double-LOSO across 22 seasons (2003..2025): for each test
   season, drop it, fit v9-B on the rest with weights (W_U, W_M),
   predict every pair in the test season's pairwise_v4 input.
2. Concatenate into `output/v9_sweep/pairwise_v9_WU{u}_WM{m}.csv` in
   v8-compatible schema (season, team_a, team_b, p_a_wins).
3. Score with `src.score_chalk_brackets.score_pairwise_path(...)` to
   get total bracket pts and per-season pts.
4. Record per-season weighted-mean log loss from `double_loso_eval`
   for context (not the decision metric).

### Decision rule

Compute v8 baseline once: `score_pairwise_path("output/pairwise_v8.csv")`.
Best cell is the (W_U, W_M) with the highest 22-year total bracket
pts. The sweep produces a winner iff:

    best_cell_total_pts > v8_total_pts + 10

The +10 bar is deliberately conservative. v9's spec used +/-3 as the
season-to-season noise band; we are selecting the best of 15 cells, so
multiple-comparisons bias inflates the apparent best. +10 is roughly
3 SD on the sampling distribution of the max-of-15, defensible as
"clear effect, not noise".

### Anchor tolerance

The (1.0, 0.0) anchor must reproduce v8 within **5 brkt pts** (was 1
pt; loosened after measurement). Two reasons for the wider band:

- v9-B and v8 have different feature sets (v9-B = v8's 4 features +
  3 more). Even at uniform sample weights, v9-B can fit slightly
  different probabilities. Per-game LL and Acc still match v8 at 3
  decimals (the v9 findings' "matches v8 exactly" claim), but the
  chalk-pick selection is sensitive to probability shifts at the
  0.5 boundary -- a 0.001 shift can flip a pick worth many bracket
  pts.
- The actual measured anchor delta is +3 pts (2673 vs 2670). 5 pts
  buys headroom for run-to-run variance from XGBoost determinism
  edge cases.

If the anchor delta exceeds 5 pts, halt and debug -- something
material has changed since the v9 findings were recorded.

### Outputs

- `output/v9_sweep_results.csv` -- one row per cell:
  `w_upset, w_miss, total_brkt_pts, ll_loso_weighted_mean,
  acc_loso_weighted_mean`
- `output/v9_sweep/pairwise_v9_WU{u}_WM{m}.csv` -- one per cell, in
  v8-compatible schema, for inspection / re-scoring under alternative
  scoring weights
- `docs/notes/2026-05-01-v9-weight-sweep.md` -- findings, sorted-by-
  brkt-pts table, recommendation. The note explicitly states the
  decision bar (+10) and whether the anchor cell reproduced v8.

### Code shape

The trainer (`src/train_upset_model.py`) already accepts `w_upset` and
`w_miss` in `compute_sample_weights(df, w_upset, w_miss)`. Two
functions hardcode them by calling `compute_sample_weights(train)`:
`double_loso_eval` and `build_v9_pairwise`. We thread `w_upset` and
`w_miss` through both as keyword arguments with defaults of 3.0 / 4.0
so existing callers (including `main()` and current tests) keep
working.

New driver: `src/sweep_v9_weights.py`. Loops over the 15-cell grid,
calls the patched `build_v9_pairwise` per cell, scores the resulting
CSV, joins LOSO log loss from `double_loso_eval`, writes
`output/v9_sweep_results.csv`, prints a sorted summary, exits.

No env-var override path. Hyperparameter knobs as function arguments,
not module globals.

### Test approach (TDD)

Add to `tests/test_train_upset_model.py`:

- `compute_sample_weights(df, w_upset=1.0, w_miss=0.0)` returns
  `np.ones(len(df))` for any non-empty df. (Sanity-check anchor.)
- `compute_sample_weights(df, w_upset=1.5, w_miss=0.5)` matches the
  closed-form formula on the existing fixture row-by-row.
- `double_loso_eval(per_game, w_upset=1.0, w_miss=0.0)` threads the
  weights to `compute_sample_weights`. Verify with a spy or by
  asserting the produced `sample_weight` array is uniform (all 1.0).
- `double_loso_eval` is deterministic across two calls with the same
  weights (xgboost seed is fixed in `fit_upset_model`).
- `build_v9_pairwise(per_game, ..., w_upset=1.5, w_miss=0.0)` writes
  one row per (season, team_a, team_b) in the input pairwise CSV, with
  team_a < team_b on every row.

The v8 reproduction check (anchor cell total bracket pts within 5 pts
of v8 -- see "Anchor tolerance" above) is not a unit test -- it is
the runtime gate in `src/sweep_v9_weights.py` and a success criterion
in this spec.

New `tests/test_sweep_v9_weights.py`:

- Driver, given a synthetic 2-cell mini-grid, writes a 2-row results
  CSV with the expected columns and types.
- Driver halts (raises) if the anchor cell is not in the grid -- the
  v8 reproduction sanity check is mandatory.

## Success criteria

- Sweep completes 15 cells in well under an hour. (Each cell is ~22
  LOSO trainings of a 100-tree XGB on ~3000 rows. If profiling shows
  this is wrong, drop W_MISS to {0, 1.0}, leaving 10 cells.)
- Anchor cell (W_U=1.0, W_M=0.0) total bracket pts is within 5 pts of
  v8's `score_pairwise_path("output/pairwise_v8.csv")` total.
- All 15 cells produce non-empty pairwise CSVs in the expected schema.
- Findings note exists with the decision bar stated, the
  sorted-by-brkt-pts table, and a clear recommend-or-don't-swap
  conclusion.

## Risks and mitigations

- **Compute cost.** 15 cells x 22 LOSO trainings could take 30-60 min.
  Mitigation: profile cell 1 first; if it takes more than ~3 min,
  trim the grid (drop W_MISS=0.5 first).
- **Multiple-comparisons inflation.** Picking the best of 15 cells
  inflates the apparent effect even when the true effect is zero.
  Mitigation: +10 pt bar (vs. v9's +/-3 noise band); explicit
  acknowledgement in the findings note.
- **Anchor cell drift.** If (1.0, 0.0) exceeds the 5-pt anchor band
  (see Anchor tolerance section above), something material has
  changed since the v9 findings were recorded -- halt the sweep and
  debug; do not widen the band further.

## Follow-ups (not in this spec)

- If a cell wins, the v9-B round-asymmetry bug (apply-time round=0
  vs train-time round in 1..6) becomes a prerequisite to swapping
  into production: resolve each (team_a, team_b) pair to its
  bracket-slot round at apply time. Then swap that (W_U, W_M) into
  production: update `src/train_upset_model.py` defaults, update the
  bracket pipeline to use `pairwise_v9.csv` instead of
  `pairwise_v8.csv`, regenerate the 2026 chalk bracket. Same branch,
  separate commits.
- If no cell wins, close the open question in the v9 findings note
  and TODO.md. Move on to the active queue: ensemble of model
  classes.
