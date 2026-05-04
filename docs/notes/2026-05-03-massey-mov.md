# Massey-Matrix MOV Feature -- Gate FAILED

**Date:** 2026-05-03
**Branch:** feat/todo-massey-colley
**Spec:** docs/superpowers/specs/2026-05-03-massey-matrix-feature-design.md
**Plan:** docs/superpowers/plans/2026-05-03-massey-matrix-feature.md
**Verdict:** REJECTED at clause 1 (non-redundancy). Clause 2 not run.

## Numbers

Aggregate (24 seasons, 65 tournament teams per season):

```json
{
  "mean_abs_corr_vs_adj_em":              0.9574,
  "max_abs_corr_vs_adj_em":               0.9734,
  "mean_abs_corr_vs_massey_composite":    0.9463,
  "max_abs_corr_vs_massey_composite":     0.9654,
  "thresholds": {"mean_max": 0.95, "per_season_max": 0.97},
  "pass": false
}
```

Per-season range: `corr_vs_adj_em` in [+0.94, +0.97] across all 24
seasons (consistently strong positive). `corr_vs_massey_composite`
in [-0.97, -0.93] (negative because composite is a *rank* -- lower
number = better -- while `massey_mov_rating` is a *rating* -- higher
number = better; absolute value is what the threshold compares).

Both thresholds against `adj_em` failed: mean 0.957 > 0.95, max 0.973
> 0.97. Both thresholds against `massey_composite` passed: mean 0.946
< 0.95, max 0.965 < 0.97.

## Diagnosis

The redundancy is specifically with our **own** iterative
opponent-adjusted efficiency loop (`adj_em`), not with the external
Massey-composite rankings. This is the failure mode the spec's "Risks
#1" section anticipated explicitly:

> Massey duplicates `adj_em`. Most likely failure mode. Our own
> iterative opponent-adjusted efficiency loop already extracts the
> "score margin adjusted for opponent strength" signal. If r > 0.95,
> clause 1 fails and we know why.

Both algorithms operate on the same regular-season game scores and
both produce a per-team continuous "strength" rating that is
opponent-adjusted on margin-of-victory. They differ in mechanism
(iterative fixed-point vs. closed-form least squares with home-court
joint estimation), but the signal they extract is essentially the
same. At a per-season correlation of 0.96-0.97, XGBoost cannot
materially exploit the residual signal; adding `massey_mov_rating`
as a 68th feature would mostly contribute noise plus tree-split
overhead.

The fact that `massey_composite` (external composite rankings: POM,
SAG, MOR, BPI, RPI) cleared both thresholds is consistent with this
diagnosis -- those external systems mix in non-margin signals (RPI is
W/L only, BPI weights game site differently, etc.), so they preserve
distinct information v4 doesn't already extract.

The MOV cap at 21 is meant to give Massey a *predictive* (not
retrodictive) flavor; that's a 2nd-order effect that doesn't change
the underlying redundancy with `adj_em` (which has no such cap and
shows a slightly *higher* correlation in absolute magnitude in 22 of
24 seasons -- the cap is not what's making them similar).

## Followup attempted: time-decay weighting (also REJECTED)

The first failure suggested time-decay might rescue the experiment by
giving Massey "different signal" than `adj_em`. Sweep over half-lives
{None, 7, 14, 30, 60, 120}d revealed the redundancy gradient is
non-monotonic:

| half_life | mean \|corr\| vs adj_em | clause 1 |
|-----------|-------------------------|----------|
| uniform   | 0.957                   | FAIL     |
| 7d        | 0.725                   | PASS     |
| 14d       | 0.931                   | PASS     |
| **30d**   | **0.979 (worst)**       | FAIL     |
| 60d       | 0.976                   | FAIL     |
| 120d      | 0.968                   | FAIL     |

Why 30d is worst: `adj_em` itself uses 30-day half-life decay
(`src/features/efficiency.py:19`), so Massey at 30d sees identical
effective game weighting + identical input data, producing maximum
correlation. Shorter half-lives diverge from this resonance.

So weighting CAN reduce clause-1 correlation below 0.95 -- the spec's
prediction "weighting won't help" was wrong on that narrow point.

But hl=14d (the strongest passing candidate) **failed clause 2**:

| season | ll_with | ll_without | delta |
|--------|---------|------------|-------|
| 2019   | 0.3866  | 0.3746     | +0.012 (hurt) |
| 2022   | 0.5111  | 0.5042     | +0.007 (hurt) |
| 2024   | 0.4357  | 0.4376     | -0.002 (marginal help) |
| **mean** | **0.4445** | **0.4388** | **+0.0057 vs threshold +0.001 -- FAIL** |

The "different signal" hl=14d captures (last ~2 weeks margin) is
already extracted by v4's late-season feature stack: `late_season`
(last 30 days vs top-100 opponents), `trajectory` (efficiency/margin
trend slopes), `vegas_trend` (late-season Vegas spread delta). Net
contribution to v4 is noise plus tree-split overhead, not new
information.

Code retained as experiment record:
- `half_life_days` kwarg on `compute_massey_mov_ratings`
- `src/sweep_massey_decay.py` -- correlation sweep across half-lives
- `src/clause2_decay_massey.py` -- clause-2 runner parameterized on
  half_life_days
- `output/sweep_massey_decay.json`, `output/clause2_decay_massey_hl14.json`

Did not try hl=7d on clause 2: lower-correlation than hl=14d but
3-4 games per team in the window would be noise-dominated. Did not
try hl=21d: between 14 and 30 means it would inherit hl=14's clause-2
failure mode without gaining margin on clause 1.

## Lessons for Colley (TODO #1, separate work item)

The Massey failure does NOT directly indict Colley. Colley solves
`(C - A) x = b` on **win/loss only** -- it discards margin entirely.
That's structurally distinct from `adj_em` (which is margin-based) in
exactly the way Massey-MOV was not. Concrete prediction: Colley's
correlation with `adj_em` should be lower -- a guess of 0.80-0.92
based on the W/L-vs-margin distinction -- which is below the 0.95 mean
threshold and would clear clause 1. The redundancy worry shifts to
`massey_composite` (where RPI, also W/L-based, is one of the
sub-systems) but the composite mixes in 4 other non-W/L systems and
the per-team composite *rank* may carry more residual signal vs. a
cleanly-derived Colley *rating*.

Recommendation for Colley: keep the same two-clause gate, same
thresholds. If clause 1 fails on Colley too, the next experiment
should be hierarchical Bradley-Terry with v4 feature priors (TODO #2),
which couples a structurally distinct ratings model to v4 features
through priors rather than as a parallel feature.

## Time-spent

Cheap gate worked exactly as designed. Total wall-clock to falsify:

- Tasks 1-7 (solver + tests + cache build): ~30 min cumulative agent
  time across 7 small commits.
- Task 8 (v4 wire-in): ~5 min code + 5 min `pytest -v test_integration`.
- Tasks 9-10 (gate clauses 9 + 10): ~25 min cumulative.
- Task 11 (gate run): ~30 sec for clause 1 against the cached real
  data. Clause 2 (the expensive ~10-15 min part) was correctly
  short-circuited by clause 1's STOP.

Net: gate caught the redundancy in ~30 seconds of compute against
the v4 feature matrix. Saved the ~10-15 min clause-2 cost AND the
~30-90 min full LOSO backtest cost. Spec's "Cheap gate first" pattern
continues to pay.

## Code retained as experiment record

All Tasks 1-10 commits remain on the branch as the experiment record.
Notable artifacts:

- `src/features/massey_matrix.py` -- standalone solver + cached
  loader, fully tested (9 unit tests). The all-neutral-games edge
  case fix discovered in Task 3 is genuinely useful and would carry
  over if any future variant of Massey is revisited.
- `src/diagnose_massey_mov.py` -- two-clause gate runner. Reusable
  pattern for future feature-addition experiments.
- `src/enhanced_model_v3.py` -- the `allowed_holdouts` kwarg added to
  `leave_one_season_out_cv_weighted` is generally useful for any
  cheap-subset diagnostic going forward (Colley experiment can reuse
  it directly).
- `tests/test_features/test_massey_matrix.py` -- 9 tests covering
  the solver math.
- `data/cache/massey_mov_ratings.parquet` -- the production cache
  artifact (gitignored, regenerable). Will be invalidated automatically
  if the producer is changed (sidecar version stamp).

## Next steps

1. Task 8's wire-in to `compute_all_features` is reverted in a
   follow-up commit (this commit). The new feature column does NOT
   ship in v4.
2. The branch `feat/todo-massey-colley` remains open for the Colley
   work item, which is the natural follow-up. Colley reuses the
   solver-module pattern, the diagnostic-runner pattern, and the
   `allowed_holdouts` kwarg, but is a fresh experiment with its own
   gate run.
3. TODO.md "Tried and rejected" updated. TODO #1's Colley portion
   remains active.
