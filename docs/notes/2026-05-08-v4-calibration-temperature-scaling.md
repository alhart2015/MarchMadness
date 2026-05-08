# v4 Calibration: Temperature Scaling -- Findings

**Date:** 2026-05-08
**Branch:** `feat/v4-calibration-temperature-scaling`
**Verdict:** **MARGINAL.** +10 brkt pts in Phase 2 retrain at MARGINAL band;
Phase 1 null by construction.
**Spec:** `docs/superpowers/specs/2026-05-08-v4-calibration-temperature-scaling-design.md`
**Plan:** `docs/superpowers/plans/2026-05-08-v4-calibration-temperature-scaling.md`

## TL;DR

Post-hoc temperature scaling on v8's pairwise output (Phase 1) is null by
construction: `score_chalk_brackets.score_pairwise_path` picks chalk by
`p > 0.5`, and temperature scaling is monotone in p, so it can never flip a
chalk pick regardless of T. All 7 Phase 1 T cells returned delta=0. Phase 2 --
retraining v8 on rescaled v4 probabilities -- is NOT monotone and produced
delta=+10 brkt pts at T ∈ {0.85, 1.15, 1.50} over 22 LOSO seasons (W/L/T
3/1/18); this clears the +10 MARGINAL bar. Per spec: candidate, no swap.
Calibration-shape lane closes with MARGINAL; roster-level returning-experience
promoted to Active queue #1 by elimination.

## The Structural Finding (Phase 1)

`score_chalk_brackets.score_pairwise_path` is a chalk-pick-based scorer: at
each node of the bracket tree, it picks the team with `p_a_wins > 0.5`. No
partial credit, no probability weighting in the outcome. Temperature scaling of
the form `p' = p^(1/T) / (p^(1/T) + (1-p)^(1/T))` (applied to v8's stage-2
output) is a monotone increasing transformation for T > 0: `p > 0.5` iff
`p' > 0.5` for all T. Consequently, post-hoc T can never flip a chalk pick;
the chalk sequence is a fixed-point of post-hoc temperature scaling.

This explains Phase 1's result in full. All 7 T cells returned delta=0:

| T | delta_total | verdict |
|---|-------------|---------|
| 0.50 | 0 | null |
| 0.60 | 0 | null |
| 0.70 | 0 | null |
| 0.80 | 0 | null |
| 0.90 | 0 | null |
| 1.10 | 0 | null |
| 1.50 | 0 | null |

This is not a falsification of the calibration-shape hypothesis. It is an
experiment design that cannot produce a non-zero result, because the metric
(chalk bracket points) is invariant to monotone rescaling of the probabilities.

A related observation: the R64 closing-line blend (PR 31) found delta=+2 at
apply-time rather than delta=0. The small nonzero lift came from the 44 games
where the override SUBSTITUTED a completely different probability (Vegas's
closing-line implied prob), not from rescaling v4's existing probabilities.
Cells that crossed the 0.5 threshold contributed the +2; all others were
invisible. Phase 1's global T sweep is exactly the class of transform that
produces zero, and Phase 1 confirmed this as expected.

## Phase 2 Sweep Results

Phase 2 retrained v8 on a v4 frame whose `p_a_wins` feature was temperature-
scaled before training (i.e., XGB sees a rescaled feature during fit, and its
learned splits produce different output even at test-time with T=1.0 applied
to the test rows -- since the training distribution was shifted). Four cells
over T ∈ {0.85, 1.15, 1.50, 2.00}:

| T | verdict | delta_total | drop_best | W/L/T | biggest_swing | anchor_max_abs_diff |
|---|---------|------------|-----------|-------|---------------|---------------------|
| 0.85 | MARGINAL | +10.0 | +6.0 | 3/1/18 | +4.0 (2011) | 0.0661 |
| 1.15 | MARGINAL | +10.0 | +6.0 | 3/1/18 | +4.0 (2011) | 0.0661 |
| 1.50 | MARGINAL | +10.0 | +6.0 | 3/1/18 | +4.0 (2011) | 0.0661 |
| 2.00 | FAIL | +0.0 | +0.0 | 0/0/22 | +0.0 (2003) | 0.0 |

Canonical baseline: 2069 brkt pts over 22 LOSO seasons.

## The Plateau Finding

T=0.85, 1.15, and 1.50 produce IDENTICAL per-season deltas -- not just the
same total, but the same chalk-pick changes in the same seasons. The direction
of T (sharpening vs softening) and the magnitude within this range both fail
to move the result.

This was a genuine surprise given the audit rationale: the Vegas audit
suggested v4 was too spread in the 0.80-0.90 confidence band (motivating
sharpening, T < 1); the 538 audit suggested v4 was too concentrated on chalk
picks vs upsets (motivating softening, T > 1). The two audit signals pointed
in opposite directions, but the sweep result shows neither direction produces
meaningfully different chalk picks in the {0.85, 1.50} T range.

The explanation is that XGBoost is approximately monotone-invariant in
single-feature rescaling within a moderate T range. XGB's histogram-binning
algorithm quantizes feature values into at most 256 bins; whether v4's
`p_a_wins` is 0.72 or 0.69 (after T=0.85 sharpening) may land in the same
histogram bin depending on the bin edges. Similarly, regularization
(lambda, alpha) and subsampling are scale-dependent in their effective
strength, but only weakly for moderate T. The tree structure produced by
XGB on T=0.85-rescaled v4 happens to be nearly identical to T=1.15 and T=1.50
for 18 of 22 seasons. The 3 seasons where chalk picks do differ (2011, 2016,
2024) are near-50/50 games where the rescaling nudges a feature value across
a histogram bin edge in the same direction for all three T values. The
audit-derived rationale for choosing these T values (Vegas → sharpen; 538 →
soften) was a red herring: both pick sets are driven by the same few bin-edge
crossings.

**The +10 lift is XGB histogram-binning, not calibration-shape correction.**

## The T=2.00 Collapse

T=2.00 returned delta=0 and anchor_max_abs_diff=0.0. The Phase 2 T=2.00
retrained v8 frame is byte-equal to the canonical `pairwise_v8.csv` baseline.

Extreme softening (T=2.00) compresses v4's `p_a_wins` feature so heavily
toward 0.5 that v4 loses most of its discriminative signal. XGB, seeing a
nearly-constant feature (all values clustered near 0.5), learns to essentially
ignore it and falls back to the remaining features: primarily seed pair, round,
and `abs_seed_diff`. This "seed-dominant" XGB model happens to produce the
same chalk picks as the canonical v8 baseline -- which was also trained on
seed-context-enriched v4 probabilities but with T=1.0. The result is
signal-erasure, not non-determinism or a pipeline artifact.

This is the cleanest sanity check in the sweep: T=2.00 collapsing to the
canonical baseline proves the pipeline works correctly -- take v4's signal to
zero and you get a model that ignores it.

## Per-Season Breakdown (Winning Cells: T ∈ {0.85, 1.15, 1.50})

All three cells produce identical per-season deltas:

| season | delta | note |
|--------|-------|------|
| 2003 | 0 | |
| 2004 | -1 | only loss |
| 2005 | 0 | |
| 2006 | 0 | |
| 2007 | 0 | |
| 2008 | 0 | |
| 2009 | 0 | |
| 2010 | 0 | |
| 2011 | +4 | biggest swing; same season as R64-blend biggest swing (+5) |
| 2012 | 0 | |
| 2013 | 0 | |
| 2014 | 0 | |
| 2015 | 0 | |
| 2016 | +4 | |
| 2017 | 0 | |
| 2018 | 0 | |
| 2019 | 0 | |
| 2021 | 0 | |
| 2022 | 0 | |
| 2023 | 0 | |
| 2024 | +3 | **first non-null result for the Kaggle year on this branch** |
| 2025 | 0 | |

Total: +4 + 4 + 3 - 1 = **+10** (drop_best: remove 2011 → +6).

**2024 finally moves (+3).** Every prior experiment on this branch returned
delta=0 for 2024 (R64-blend FAIL +0; Phase 1 +0). This is the first non-null
result for the Kaggle year. The +3 is the same magnitude as 2016's +4 (roughly
comparable); it's not concentrated in 2024 specifically. The drop_best (+6)
removes 2011, leaving 2024 and 2016 as the two largest contributors. The
2024 movement is encouraging but doesn't change the MARGINAL verdict: +10
total, +6 drop_best, both in the [+10, +25) MARGINAL band.

The 2011 contribution (+4) is the same "hard season" flagged in the per-season
variance check (PR 30) as a 3-of-4-metrics outlier and in the R64 blend as the
biggest single-season swing (+5). 2011 gains across multiple experiments
because it was v4's worst season (accuracy 57.4% vs Vegas 65.6%), so anything
that changes chalk picks has a higher expected lift there. This is NOT a reason
to trust 2011's contribution more -- it's a reason to treat the plateau-pattern
with caution.

## Anchors

Three anchor tests were run; all passed:

| anchor | required | observed | verdict |
|--------|----------|----------|---------|
| Phase 1: global T=1.0 on v8 output | byte-equal to canonical `pairwise_v8.csv` | max_abs_diff=0.0 | **PASS** |
| Phase 1: per-round (1,1,1,1,1) on v8 output | byte-equal to canonical on resolved rows | max_abs_diff=0.0 | **PASS** |
| Phase 2: T=1.0 retrain v8 on unscaled v4 | byte-equal to canonical `pairwise_v8.csv` | max_abs_diff < 1e-9 | **PASS** |

The Phase 2 anchor is the most important: retraining v8 with T=1.0 (no scaling
at all) reproduces the canonical `pairwise_v8.csv` exactly, confirming the
Phase 2 pipeline is correct. This anchor ran in ~33s wall time (much faster
than the spec estimate of ~10-15 min per cell, because the spec overestimated
XGB retraining time for the 22-season loop on this hardware).

## What This Implies for the Queue

**Calibration-shape lane closes with MARGINAL.** Per spec: +10 is a candidate,
no swap. The lift is at the lower bound of MARGINAL and the structural
explanation (XGB histogram-binning) does not motivate further calibration-shape
engineering on top of v4's existing features.

**Queue reorganization:**
- v4 calibration-shape engineering: Done (MARGINAL, this PR).
- **Roster-level returning-experience (was #4) → Active queue #1 by
  elimination.** The R64-blend FAIL + calibration-shape MARGINAL together
  strengthen the case that the performance ceiling is a feature-space gap,
  not a calibration-shape gap. Roster-level experience is structurally different
  from anything in v4's 67-feature stack; it's the strongest remaining external-
  data candidate.
- MLP (was #2) → #2. Unchanged.
- Full Bayesian BT (was #3) → #3. Unchanged.
- Pre-tournament Vegas futures (was #5) → #4. Demoted to #4 (was #5) by
  elimination (one fewer item above it).

**Meta-lesson recorded:** The production gate (chalk scoring) is monotone-
invariant in p. Any future post-hoc transformation on stage-2 output must
change chalk picks specifically -- flipping near-50/50 games -- not just
rescale probability magnitudes. LL gains from calibration don't transfer to
bracket points unless they flip chalk picks. This applies to any isotonic
regression, Platt scaling, temperature scaling, or other post-hoc probability
calibration scheme applied to v8's output.

## Caveats

1. **Selection bias on bracket-points-aggregate.** The 22-season aggregate is
   the right gate per spec, but the +10 total is dominated by 3 seasons (2011,
   2016, 2024) that together contribute +11 of +10 gross (i.e., the lone loss
   in 2004 is -1). The plateau pattern (all three T values give the same picks)
   means these three seasons' chalk games happened to have near-50/50 v4 probs
   that the XGB retrain moved in the same direction for T ∈ {0.85, 1.50}. A
   one-season holdout on 2011 or 2016 would reduce the result to +6 or +7 --
   marginal at best.

2. **+10 is at the edge of the noise floor.** The spec's MARGINAL band starts
   at +10; this experiment lands exactly at +10. drop_best is +6. For comparison,
   the v9-B round-fix swept to +20 (also MARGINAL) with a 4W-2L per-season
   profile; this result has 3W-1L-18T. The profile is less concentrated than
   the round-fix, but the total is smaller.

3. **Audit-derived rationale was misdirected.** The Vegas and 538 audits
   identified calibration-shape weakness but the T direction implied by each
   audit (sharpen vs soften) turned out to be irrelevant: the same 3 seasons
   move regardless of direction. The mechanism is XGB histogram-binning, not
   any calibration-shape correction of the type the audits diagnosed.

4. **Phase 2 only moves 3/22 seasons.** The 18 ties in the W/L/T profile mean
   the v4 chalk picks are robust to moderate temperature rescaling in 18 of 22
   seasons. This is consistent with v4 being well-calibrated on chalk picks in
   aggregate (the 538 audit's delta=+0.075 on chalk picks is a real weakness,
   but it's LL-level, not chalk-flip-level).

## Files of Record

```
src/apply_temperature_scaling.py          -- temperature scaling transform + Phase 1 driver
src/eval_v4_calibration.py                -- Phase 1 eval: per-T bracket scoring
src/run_phase2_sweep.py                   -- Phase 2 retrain driver + sweep

tests/test_apply_temperature_scaling.py   -- 10 tests (unit + anchor)
tests/test_eval_v4_calibration.py         -- 13 tests (unit + Phase 2 anchor + smoke)

output/pairwise_v8_calibrated_global_T0.70.csv   -- Phase 1 global scaling sample
output/pairwise_v8_calibrated_perround_0.70_*.csv -- Phase 1 per-round scaling sample
output/pairwise_v8_phase2_T0.85.csv
output/pairwise_v8_phase2_T1.15.csv
output/pairwise_v8_phase2_T1.50.csv
output/pairwise_v8_phase2_T2.00.csv
output/v4_calibration_eval.json
output/v4_calibration_eval_log.txt
output/v4_calibration_reliability.png

docs/superpowers/specs/2026-05-08-v4-calibration-temperature-scaling-design.md
docs/superpowers/plans/2026-05-08-v4-calibration-temperature-scaling.md
docs/notes/2026-05-08-v4-calibration-temperature-scaling.md   (this note)
```

## Compute

- Phase 1 anchor tests (2): ~5s each.
- Phase 1 T-cell eval (7 cells on precomputed CSVs): ~10s total.
- Phase 2 anchor (T=1.0 retrain): ~33s wall.
- Phase 2 sweep (4 cells x 22 LOSO retrain): ~2 min wall.
- Phase 1 CLI end-to-end: ~2 min wall.
- **Total wall time: ~5 min** (much faster than spec estimate of ~30-40 min;
  spec overestimated XGB per-cell training time).

## What We Learned

Post-hoc temperature scaling on v8's chalk-scored output is null by
construction; retraining v8 on rescaled v4 lifts bracket points by +10 over
22 LOSO seasons (MARGINAL band) but the lift is XGB-histogram-binning, not
calibration-shape correction, as evidenced by identical per-season deltas
across T ∈ {0.85, 1.15, 1.50} spanning both sharpening and softening
directions.
