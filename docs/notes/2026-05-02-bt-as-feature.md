# BT-as-Feature for v9-C -- Findings

**Date:** 2026-05-02
**Branch:** feat/bt-as-feature
**Verdict:** **NO-GO** -- pre-sweep gate failed. v9-C stays in production.
**Spec:** `docs/superpowers/specs/2026-05-02-bt-as-feature-design.md`
**Plan:** `docs/superpowers/plans/2026-05-02-bt-as-feature.md`

## TL;DR

Adding `p_bt` from `output/pairwise_bt.csv` (the frozen 22-LOSO
artifact from PR 12) as a 6th input feature to v9-C's upset-aware
stage-2 model -- under a new `feature_set='v9d'` selector -- was
tested via the pre-sweep falsification gate. Gate FAILED: at uniform
weights `(W_UPSET=1.0, W_MISS=0.0)`, v9-D's weighted-mean per-game
log loss across 22 LOSO seasons was `0.4339` vs v9-C's `0.4324` --
**headroom of -0.0015 < 0.001 threshold**, and crucially, *negative*.
Adding BT as an input feature didn't merely fail to help; it actively
hurt the model. v9-C stays in production. The 15-cell W_UPSET /
W_MISS sweep was not run -- saved ~45-75 minutes of compute.

## Setup recap

Production today: `v4 (XGBoost stage 1) -> v9-C (XGBoost upset-aware
stage 2)` at `(W_UPSET=1.25, W_MISS=0.0)` = 2713 brkt pts on 22
LOSO seasons.

Experiment: extend `upset_features` with a new `'v9d'` selector
returning a 6-column matrix `[p_stage1, seed_a, seed_b, abs_seed_diff,
round, p_bt]`. `p_bt` is joined upstream by an extended
`load_per_game_data_with_upset(pairwise_bt_csv=...)` -- per-game
`(W, L)` row gets the BT winner-perspective probability; symmetric
`(L, W)` row gets `1 - that value`. `build_v9_pairwise` threads
`pairwise_bt_csv` to the apply-time per-pair lookup. `compute_sample_weights`
unchanged -- W_UPSET / W_MISS still key off `p_v4` and the seed-derived
upset flag. BT enters as input feature only.

Pre-sweep gate (`src/diagnose_v9d.py`): single-clause falsification.
Builds the per-game frame once with `p_bt` joined; runs `double_loso_eval`
twice (`feature_set='v9c'` vs `'v9d'`) at uniform weights `(1.0, 0.0)`;
requires `LL_v9c - LL_v9d >= 0.001` to PASS. Threshold 1/5 of the
PR 12 ensemble gate's 0.005 because this is a single-clause test on
a paired comparison (same per-game rows evaluated by both feature
sets, which cancels most variance).

## Pre-sweep gate result

The verdict comes straight from `output/diag_v9d.json`:

| measure | value | clause |
|---|---|---|
| n played games (both evals) | 1449 | sanity |
| v9-C wt-mean LL @ (1.0, 0.0) | **0.4324** | baseline |
| v9-D wt-mean LL @ (1.0, 0.0) | **0.4339** | candidate |
| headroom (`v9c - v9d`) | **-0.0015** | **FAIL** (< 0.001) |
| **gate verdict** | - | **FAIL** |

Note: the v9-C LL of `0.4324404529787647` matches the committed
`output/v9c_sweep_results.csv` row for `(1.0, 0.0)` to 16 decimal
places. This implicitly anchors the trainer-extension: when called
without `pairwise_bt_csv`, `load_per_game_data_with_upset` produces
a per-game frame and v9-C eval byte-identical to the pre-extension
behavior. The trainer-harness anchor passes implicitly; no
regression to v9-A/B/C.

## Falsification reasoning

The single-clause gate isolates one question: *does p_bt supply
marginal information v9-C can extract on top of v4 + seed/round
context?* The answer is no, at two distinct severities:

1. **It doesn't help.** Headroom 0.0015 short of the 0.001
   threshold means even the most generous reading -- "BT contributes
   sub-noise signal" -- isn't supported.
2. **It actively hurts.** The headroom is *negative*. Adding `p_bt`
   to the feature matrix made the model worse by 0.0015 LL on the
   same paired games. The mechanism is straightforward: XGBoost has
   `n_estimators=100` trees and each tree has bounded depth; some
   fraction of those trees end up splitting on `p_bt` instead of the
   already-strong `p_v4` / seed / round features. Splits on the
   weaker noisier feature are net loss compared to additional splits
   on the stronger features.

The implication for the broader hypothesis: v9-C is a learnable
trust-weight function in principle, but the actual learning signal
(2898 per-game training rows under double-LOSO) is too thin to
discover round/seed-conditional gating that beats just-using-v4. BT
*as a global ensemble peer* failed in PR 12 because BT is too weak
standalone; BT *as a v9-C feature* fails here because v9-C's training
data is too small to extract useful per-context gating from a noisy
feature. Two distinct failure modes at the two natural granularities.

The upset-weight schedule (W_UPSET / W_MISS) operates by re-weighting
training rows to emphasize upsets and high-confidence misses. It does
not introduce any new information and cannot rescue a feature that
adds net noise at uniform weights. Therefore the 15-cell sweep was
correctly not run.

## Comparison to predecessors

| experiment | mechanism | failure mode | LL signal |
|---|---|---|---|
| LR ensemble (PR 11) | global avg, fixed w | residual correlation r=0.77 too high | n/a (sweep ran, lost on brkt pts) |
| BT ensemble (PR 12) | global avg, fixed w | BT too weak standalone (LL 0.565 vs 0.437) | optimal w=0.98, headroom +0.0000 |
| **BT-as-feature (PR 13, this experiment)** | **learned per-context weight** | **v9-C can't learn useful gating from p_bt at this data scale** | **headroom -0.0015** |

All three close one cell of the (model-class, ensemble-form) grid.
LR-ensemble closed "diversity at identical features doesn't beat
correlation." BT-ensemble closed "structurally diverse but standalone-
weak doesn't blend." BT-as-feature closes "v9-C can't substitute
for global-blend by learning per-context gating, at least not from
the BT signal at this data scale."

## Verdict

NO-GO. v9-C stays in production. The trainer extension (`pairwise_bt_csv`
threading through `load_per_game_data_with_upset` /
`build_v9_pairwise` / `sweep_v9_weights`) and the `'v9d'` feature_set
remain on the branch as the experiment record. The pre-sweep gate
machinery in `src/diagnose_v9d.py` is reusable for any future
"add-a-feature-to-v9c" experiment -- just point it at a different
upstream pairwise CSV.

Saved compute: ~45-75 minutes by gating before the 15-cell sweep.
The gate's threshold `0.001` was tight enough to catch a `-0.0015`
headroom cleanly; the result wasn't even close.

## Recommendation

The diversity-correlation axis (PR 11) and the standalone-strength
axis (PR 12) are both well-mapped now, plus the per-context-gating
escape hatch (this PR) is closed. The remaining swings on the
diversity-stage-1 program need a *different* axis.

Active queue advances:

1. **Feature-view diversity ensemble** (formerly queue #1, now still
   #1). XGBoost trained on disjoint feature subsets (KenPom-only,
   Vegas-only, raw efficiency). Same model class -> same standalone
   strength as v4 (sidesteps PR 12's bottleneck). Disjoint feature
   views -> different errors by construction (sidesteps PR 11's
   bottleneck). The "v9-C learns context-dependent gating" mechanism
   still applies but with model peers that are individually as strong
   as v4, the per-context training signal is also stronger. Most
   likely of the remaining items to clear all relevant clauses.

2. **Hierarchical Bradley-Terry with feature priors** (formerly #3).
   Couples BT back to the v4 feature view to gain standalone
   strength. Risk: residual correlation may regress from the 0.58
   PR 12 measured back toward 0.77 as the models re-converge. With
   the BT-as-feature route now closed, this is the next angle on
   getting a stronger BT signal.

3-5. Small NN, full Bayesian BT, external rankings, returning
   experience -- unchanged.

Item #2 of the previous queue (this experiment) moves to "Tried and
rejected."

## Files of record

```
src/train_upset_model.py        -- extended (load_per_game_data_with_upset
                                   accepts pairwise_bt_csv; upset_features
                                   has 'v9d' branch; build_v9_pairwise
                                   threads pairwise_bt_csv at apply time)
src/sweep_v9_weights.py         -- extended (V9_FEATURE_SET=v9d allowed;
                                   output paths key off 'v9d_sweep')
src/diagnose_v9d.py             -- pre-sweep single-clause gate (new)

output/diag_v9d.json            -- gate diagnostic with verdict
                                   (15-cell sweep NOT run)

tests/test_upset_model.py       -- 7 v9-D tests added under "v9-D extension"
                                   section: load_per_game_data BT join,
                                   v9d feature_set, build_v9_pairwise BT
                                   threading, ValueError guards
tests/test_diagnose_v9d.py      -- 5 gate tests (check_gate edge cases +
                                   compute_gate end-to-end on synthetic)
tests/test_sweep_v9_weights.py  -- 2 v9d sweep tests added
```

The `src/diagnose_v9d.py` gate threshold (`GATE_LL_HEADROOM_MIN = 0.001`)
is a module constant. Re-running the gate against a new BT variant
(e.g., margin-aware BT, hierarchical BT with feature priors) is
just `python src/diagnose_v9d.py --pairwise-bt output/pairwise_NEW.csv`.
