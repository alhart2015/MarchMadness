# BT-as-Feature for v9-C -- Design

**Date:** 2026-05-02
**Branch:** feat/bt-as-feature
**Predecessors:**
- BT stage-1 swap experiment (rejected, PR 12):
  `docs/superpowers/specs/2026-05-01-bayesian-stage1-design.md`
  `docs/notes/2026-05-01-bayesian-stage1.md`
- LR ensemble experiment (rejected, PR 11):
  `docs/superpowers/specs/2026-05-01-ensemble-stage1-design.md`
  `docs/notes/2026-05-01-ensemble-stage1.md`
- v9-C production swap (current production stage-2):
  `docs/superpowers/specs/2026-05-01-v9c-production-swap-design.md`

## Motivation

PR 12 tested Bradley-Terry as a stage-1 *replacement* (and as an
ensemble peer with v4). Verdict: NO-GO at the diagnostic gate. BT is
too weak standalone (LL 0.565 vs v4's 0.437) for any non-trivial
blend weight to help -- the cheating-best ideal weight was `w_v4 = 0.98`
with `+0.0000` log-loss headroom. **But** the residual-correlation
clause PASSED at `r = 0.577 < 0.60` -- BT's errors are meaningfully
less correlated with v4's than the LR experiment's were (r=0.77).
The diversity hypothesis is partially validated; the blocker is
standalone strength, not correlation.

This design tests a different way to consume that uncorrelated signal:
add `p_bt_stage1` (from `output/pairwise_bt.csv`) as a 6th input
feature to v9-C's upset-aware stage-2 model. The hypothesis is that
v9-C is a *learnable trust-weight function* -- not a fixed-weight
average -- and giving it BT as input lets it learn round/seed-conditional
gating that simple ensembling cannot express.

Concretely:
- v9-C currently sees `(p_v4_stage1, seed_a, seed_b, abs_seed_diff,
  round)` and learns when to nudge v4's prediction in upset-prone
  contexts (deep-round wide-seed-mismatch games being the design
  intent).
- Adding `p_bt` as a 6th feature lets v9-C synthesize `p_bt - p_v4`
  internally (the "two views disagree here" signal) and gate on it
  against (round, seeds, confidence). BT alone is too weak; v4 alone
  is the production baseline; the *disagreement* between them carries
  potentially uncorrelated extra signal v9-C can learn to consult.
- This sidesteps PR 12's standalone-strength bottleneck entirely
  because BT never has to stand alone -- v4 is in the same input
  vector.

The "as feature" framing is doing real work: a learned per-context
trust weight is strictly more expressive than a global blend weight,
which is the exact representational gap PR 12's failure exposed.

## Goals

- Add `p_bt_stage1` as a 6th feature in `upset_features` under a new
  `feature_set='v9d'` selector. Keep v9-A/B/C feature paths untouched.
- Run a cheap *pre-sweep falsification gate* before the 15-cell
  W_UPSET / W_MISS sweep. If v9-D@(1.0, 0.0) doesn't beat v9-C@(1.0,
  0.0) on weighted-mean per-game LL by >= 0.001, NO-GO without
  running the remaining 14 cells.
- If the gate clears, run the existing 15-cell sweep harness with
  `V9_FEATURE_SET=v9d` and compare per-cell bracket points to v9-C's
  production cell (`(W_UPSET=1.25, W_MISS=0.0)` = 2713 brkt pts).
- Verdict bands match the v9-C-vs-v8 swap spec:
  - `delta_vs_v9c >= +25` -> clear winner, follow-up swap-in PR.
  - `+10 <= delta < +25` -> marginal candidate, document, do not swap.
  - `delta < +10` -> NO-GO.
- Either way, ship a clean experiment record so queue items #1
  (feature-view diversity) and #3 (hierarchical BT with feature
  priors) start from real evidence about whether v9-C's "learnable
  trust weight" representation can extract value from a structurally
  diverse but standalone-weak signal.

## Non-Goals

- Production swap (`predict_2026_v9d.py` mirror of `predict_2026_v9c.py`).
  Out of scope. If v9-D wins, that's a follow-up commit. The
  canonical `output/pairwise_probs.json` continues to be v9-C's
  output until/unless that follow-up ships.
- Extending `train_bt_stage1.py` to a 2026-final-snapshot fit. The
  committed `output/pairwise_bt.csv` covers 22 LOSO seasons (2003-2025)
  only, which is exactly what this experiment needs. A 2026 BT fit is
  a prerequisite for production swap, not for the experiment.
- Derived feature variants (`p_bt - p_v4` as an explicit column;
  `bt_strength_a` / `bt_strength_b`). XGBoost can synthesize the
  disagreement signal from `(p_v4, p_bt)` via two-tree splits; the
  raw form is the canonical first test per the BT findings note.
  If raw v9-D wins, derived variants become a clean follow-up. If
  raw v9-D loses, the spec falsifies the broader "v9-C as a learnable
  trust weight on BT" hypothesis cleanly.
- Sample-weight changes. W_UPSET / W_MISS still key off v4's residual
  and the seed-derived upset flag. BT enters as input feature only,
  not as a re-weighting axis.
- Re-tuning v9-C's W_UPSET / W_MISS grid against the BT-feature
  variant. Reuse the same 15-cell grid as PR 9 / v9-C sweep so
  results are directly comparable.
- Live bracket integration (`generate_bracket_real.py`). Same
  out-of-scope rationale as the BT stage-1 experiment -- live
  bracket pipeline is currently pure-v4-MC anyway.
- Pairwise BT regeneration. The committed `pairwise_bt.csv` (frozen
  artifact from PR 12) is reused as-is. If the BT trainer changes,
  this experiment's results would change; we hold BT fixed and
  test only the v9-C-side hypothesis.

## Approach

### Architecture

```
output/pairwise_v4.csv  +  output/pairwise_bt.csv  +  results + seeds
                              |
                              v
        load_per_game_data_with_upset(pairwise_bt_csv=...)
                              |
                              v
                  per-game DataFrame with new p_bt column
                              |
                              v
        src/diagnose_v9d.py  -> output/diag_v9d.json
                              |
                  GATE check: LL_v9c@(1,0) - LL_v9d@(1,0) >= 0.001
                              |
            FAIL --------------|--------------- PASS
              |                                  |
              v                                  v
        stop, write findings    V9_FEATURE_SET=v9d sweep_v9_weights.py
                                                  |
                                                  v
                          15 cells -> output/v9d_sweep_results.csv
                                                  |
                                                  v
                          per-cell delta vs v9-C production cell
                                                  |
                                                  v
                          verdict bands -> findings note
```

`output/pairwise_bt.csv` is read-only. All new logic is additive --
existing v9-A/B/C code paths are unchanged.

### Module changes

| Module | Change |
|---|---|
| `src/train_upset_model.py` | (1) `load_per_game_data_with_upset` accepts `pairwise_bt_csv: str \| None = None`. When provided, joins `p_bt` from `pairwise_bt.csv` on `(season, team_a, team_b)` with the same A/B-orientation convention as `p_v4` (lookup is by `(season, min_id, max_id)` -> `p_min_wins`; reverse to `1 - p_min_wins` for the symmetric `(L, W)` row). When `None`, falls back to existing behavior with no `p_bt` column. (2) `upset_features(feature_set='v9d')` returns the 6-column matrix `[p_stage1, seed_a, seed_b, abs_seed_diff, round, p_bt]`. Raises `ValueError` if `feature_set='v9d'` is requested but `p_bt` is absent from the input frame. (3) `build_v9_pairwise` accepts `pairwise_bt_csv` and threads it through the apply-time per-pair lookup so the trained v9-D model can be applied to the full 48,465-pair grid. (4) `compute_sample_weights` and `fit_upset_model` are unchanged. |
| `src/sweep_v9_weights.py` | Accept `'v9d'` in the `V9_FEATURE_SET` env var. Output dirs key off the choice: `output/v9d_sweep/` and `output/v9d_sweep_results.csv`. Pass `pairwise_bt_csv='output/pairwise_bt.csv'` through `run_single_cell` -> `load_per_game_data_with_upset` and `build_v9_pairwise`. Existing `'v9b'` and `'v9c'` paths unchanged. |
| `src/diagnose_v9d.py` (new) | Pre-sweep falsification gate. Loads `pairwise_v4.csv` + `pairwise_bt.csv` + tournament results + seeds; builds per-game data once with `p_bt` joined. Calls `double_loso_eval` twice (`feature_set='v9c'` and `feature_set='v9d'`) at uniform weights `(W_UPSET=1.0, W_MISS=0.0)`. Computes weighted-mean LL across 22 seasons for both. Headroom = `LL_v9c - LL_v9d`. Writes `output/diag_v9d.json` with `{ll_v9c, ll_v9d, headroom, threshold, gate_verdict}`. Prints verdict; exits nonzero on FAIL so a wrapper can short-circuit the sweep. Mirrors `src/diagnose_bt_vs_v4.py` in shape. Gate threshold lives at the top of the file as a module constant: `GATE_LL_HEADROOM_MIN = 0.001`. |

No other modules touched. `enhanced_model_v3.py`, `predict_2026_v9c.py`,
`generate_bracket_real.py`, `score_chalk_brackets.py`, etc. all unchanged.

### Falsification reasoning

The gate isolates one question -- *does p_bt supply marginal
information v9-C can extract* -- from the W_UPSET/W_MISS-tuning
question downstream. Two distinct ways the experiment can fail:

1. **Pre-gate FAIL.** v9-D@(1.0, 0.0) doesn't beat v9-C@(1.0, 0.0)
   on per-game LL. Headroom < 0.001. Means: even with all six
   features visible to a deterministically-trained XGB at uniform
   weights, BT contributes no extractable signal on top of v4 +
   seed/round context. Upset-weight schedules are downstream
   re-weightings of the same per-game scores and cannot rescue this.
   NO-GO without the 14 other cells.

2. **Pre-gate PASS, sweep NO-GO.** v9-D@(1.0, 0.0) does extract
   marginal LL, but no W_UPSET/W_MISS cell converts that into
   `+10` bracket points vs v9-C. Means: BT supplies signal but it
   doesn't survive the bracket-points scoring (which weights
   round-by-round chalk picks heavily). v9-C's existing 5-feature
   gating is already extracting most of the relevant signal at the
   bracket level. NO-GO with full sweep evidence.

The first failure mode kills the hypothesis at its root. The second
falsifies the bracket-points claim while leaving open the LL claim
(which would be documented for queue #1 and #3 to consider).

The `0.001` threshold is one-fifth of PR 12's `0.005`. PR 12 was a
three-clause gate testing ensemble suitability; this is a
single-clause gate testing whether feature-level signal exists. A
tighter threshold matches the tighter scope. Calibration: at 1449
played games over 22 seasons, weighted-mean LL of `0.4369` (v4
baseline) has a sample-noise standard error around `0.005-0.008`;
a `0.001` headroom requirement is well below noise on a single
season but well *above* noise on the full 22-season weighted mean
when the same model evaluates both feature sets on the same per-game
rows (paired comparison cancels most variance).

### Verdict bands

Match the v9-C-vs-v8 swap spec exactly:

| `delta_vs_v9c` | label | action |
|---|---|---|
| `>= +25` | clear winner | follow-up PR for production swap (`predict_2026_v9d.py` + 2026 BT extension) |
| `+10` to `+24` | marginal candidate | document, do not swap; queue advances |
| `0` to `+9` | no-go | findings note, queue advances |
| `< 0` | regression | findings note, queue advances |

Comparison baseline is v9-C re-scored fresh from
`output/v9c_sweep/pairwise_v9_WU1.25_WM0.00.csv` using
`score_pairwise_path` -- *not* lifted from a prior log -- so
v9-D's per-cell scores and v9-C's baseline come from the same
scoring code on the same call.

### Anchor + join sanity discipline

Two anchors must pass before trusting cell rankings.

**1. Trainer-harness anchor.** Re-run the same sweep harness with
`V9_FEATURE_SET=v9c` (no code path changes for v9-C should have
happened) and confirm the resulting `pairwise_v9_WU1.00_WM0.00.csv`
reproduces the existing committed v9-C anchor cell within float
tolerance (max prob delta < `1e-9`). This catches harness
regressions from the trainer extension. If this fails, the
extension to `load_per_game_data_with_upset` or `build_v9_pairwise`
broke the v9-C path -- abort and debug before any v9-D numbers are
trusted.

**2. Join-orientation anchor (unit test).** In
`tests/test_train_upset_model.py`, fixture-based test asserting
that for 5 sampled `(season, a, b)` tuples loaded by
`load_per_game_data_with_upset(..., pairwise_bt_csv=...)`:
- The `(W, L)` row's `p_bt` equals the `pairwise_bt.csv` lookup at
  `(season, min(W,L), max(W,L))` if `W < L`, else `1 - lookup`.
- The symmetric `(L, W)` row's `p_bt` equals `1 - (W,L row's p_bt)`.

This catches BT-side join orientation bugs and the symmetric-pair
mirror bug, both of which would silently produce "winning" sweep
results from garbage features.

If either anchor fails, the sweep result is invalid -- abort and
debug before reading any cell numbers.

### Disposition matrix

| outcome | branch deliverables | TODO.md update |
|---|---|---|
| Pre-gate FAIL | gate JSON + findings note (~3 paragraphs) + spec + plan + tests | move queue #2 to Tried-and-rejected; queue #1 (feature-view diversity) advances |
| Pre-gate PASS, sweep delta < +10 | full sweep CSV + 15 per-cell pairwise CSVs + findings note + spec + plan + tests | move queue #2 to Tried-and-rejected; queue #1 advances |
| Pre-gate PASS, sweep delta in `[+10, +24]` | full sweep CSV + 15 per-cell pairwise CSVs + findings note recommending "candidate, do not swap" + spec + plan + tests | mark queue #2 done as marginal; queue #1 advances |
| Pre-gate PASS, sweep delta `>= +25` | full sweep CSV + 15 per-cell pairwise CSVs + findings note + spec + plan + tests | mark queue #2 done as clear winner; **add new queue item: production swap PR** (`predict_2026_v9d.py`, extend `train_bt_stage1.py` to 2026-final-snapshot, repoint `output/pairwise_probs.json` consumer) |

In all cases, `docs/notes/2026-05-02-bt-as-feature.md` is committed
with the verdict, and `TODO.md` is updated as part of the same PR
that ships the experiment record.

## File deliverables

```
src/diagnose_v9d.py                                          (new)
src/train_upset_model.py                                     (extended)
src/sweep_v9_weights.py                                      (extended)

output/diag_v9d.json                                         (gate result; always)
output/v9d_sweep/pairwise_v9_WU{u:.2f}_WM{m:.2f}.csv         (15 cells; only on PASS)
output/v9d_sweep_results.csv                                 (only on PASS)

docs/notes/2026-05-02-bt-as-feature.md                       (findings; always)
docs/superpowers/specs/2026-05-02-bt-as-feature-design.md    (this file)
docs/superpowers/plans/2026-05-02-bt-as-feature.md           (next step)

tests/test_diagnose_v9d.py                                   (new)
tests/test_train_upset_model.py                              (extended)
tests/test_sweep_v9_weights.py                               (extended)
```

## Tests

Three test files touched (all CI-fast; no real-data-grade fixtures
in unit tests):

1. `tests/test_train_upset_model.py` -- additions:
   - `test_load_per_game_data_with_upset_joins_pbt`: synthetic
     per-game and pairwise_bt fixtures; assert `p_bt` column appears
     and matches the lookup with correct A/B orientation for both
     `(W, L)` and `(L, W)` symmetric rows.
   - `test_upset_features_v9d_shape`: `feature_set='v9d'` returns
     a `(n_rows, 6)` matrix in the documented column order.
   - `test_upset_features_v9d_missing_pbt_raises`: requesting
     `feature_set='v9d'` on a frame missing the `p_bt` column raises
     `ValueError` with a clear message.
   - `test_load_per_game_data_with_upset_pbt_csv_omitted_backwards_compat`:
     when `pairwise_bt_csv=None`, the returned frame has the same
     columns as the existing v9-A/B/C path (no `p_bt` column).

2. `tests/test_sweep_v9_weights.py` -- additions:
   - `test_run_single_cell_v9d_feature_set`: synthetic per-game,
     pairwise_v4, and pairwise_bt fixtures; assert `run_single_cell(..., feature_set='v9d', pairwise_bt_csv=...)`
     completes without error and returns a metrics dict with all
     expected keys.

3. `tests/test_diagnose_v9d.py` (new):
   - `test_gate_passes_when_v9d_beats_v9c_by_threshold`: synthetic
     fixtures rigged so v9-D's LL is `0.4` and v9-C's is `0.42`;
     assert `gate_verdict == 'PASS'`.
   - `test_gate_fails_when_v9d_does_not_beat_v9c`: fixtures where
     v9-D's LL is `0.42` and v9-C's is `0.42` (no improvement);
     assert `gate_verdict == 'FAIL'` and the script exits nonzero.
   - `test_gate_fails_when_v9d_is_worse`: fixtures where v9-D's
     LL is `0.43` and v9-C's is `0.42` (regression); assert
     `gate_verdict == 'FAIL'`.

The trainer-harness anchor (re-running the v9-C sweep cell to within
`1e-9`) is verified manually as part of running the sweep -- a full
sweep on real data is too slow for CI. The README of the findings
note will paste both the v9-C anchor reproduction max-delta and
the join-orientation unit test status as evidence.

## Implementation order

1. Extend `src/train_upset_model.py`: `load_per_game_data_with_upset`,
   `upset_features`, `build_v9_pairwise`. Add tests in
   `tests/test_train_upset_model.py`. Run `pytest tests/test_train_upset_model.py`.
2. Extend `src/sweep_v9_weights.py`: `'v9d'` env var support, output
   path keying, BT csv threading. Add test in
   `tests/test_sweep_v9_weights.py`. Run `pytest tests/test_sweep_v9_weights.py`.
3. Add `src/diagnose_v9d.py` and `tests/test_diagnose_v9d.py`. Run
   `pytest tests/test_diagnose_v9d.py`.
4. Run the full ingest/feature/integration test suite per CLAUDE.md
   forced-verification rule:
   `pytest -v tests/test_ingest tests/test_features tests/test_integration.py`
   (the trainer extension touches data-loading code that flows into
   the v9-C/D feature pipeline).
5. Run `python src/diagnose_v9d.py` against real data. Commit
   `output/diag_v9d.json`.
6. Branch on the gate verdict:
   - **FAIL:** write findings note (NO-GO), update TODO.md, commit, PR.
   - **PASS:** continue to step 7.
7. Run `V9_FEATURE_SET=v9d python src/sweep_v9_weights.py`. Commit
   `output/v9d_sweep_results.csv` and the 15 per-cell CSVs.
8. Re-run `V9_FEATURE_SET=v9c python src/sweep_v9_weights.py` on
   just the anchor cell; assert max prob delta vs the committed
   v9-C anchor < `1e-9`. Document in findings note.
9. Compute per-cell `delta_vs_v9c` against re-scored v9-C baseline.
   Identify best cell. Apply verdict bands.
10. Write `docs/notes/2026-05-02-bt-as-feature.md` (findings).
    Update `TODO.md`. Commit, PR.

Steps 1-4 are pure code + tests with no real-data dependency and
are the bulk of the implementation work. Steps 5-10 are
running-the-experiment work whose total runtime is dominated by the
sweep (~45-75 min on real data) and is gated by the pre-sweep gate.
