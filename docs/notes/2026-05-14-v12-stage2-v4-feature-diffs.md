# v12 -- Stage-2 enrichment with v4 top-N feature diffs -- FAIL

**Date:** 2026-05-14
**Spec:** `docs/superpowers/specs/2026-05-14-v12-stage2-v4-feature-diffs-design.md`
**Plan:** `docs/superpowers/plans/2026-05-14-v12-stage2-v4-feature-diffs.md`
**Branch:** `feat/v12-stage2-v4-feature-diffs`

## Headline

22-season LOSO-picked bracket-points total: **2078 brkt pts** vs v13
baseline 2106 = **-28 (FAIL active regression band)**.

Best single cell (n10_v10cap @ alpha=0.6) scored 2126 (+20, MARGINAL band)
but with **>71% of its gross lift concentrated in a single season (2015)**,
which triggers the spec's >50%-single-season-concentration demotion rule
that would knock MARGINAL back to FAIL anyway.

**Both readings collapse to FAIL.** v12 lane closes. Eighth same-data-peer
addition to v4 that has failed (now: BT-feature, feature-view, HBT, Colley,
Massey-MOV, Massey-decay-14d, team-seed-residual, v12 v4-feature-diffs).

## Per-cell results (with v13 toss-up blend, alpha=0.6, upper-edge=0.55)

| Cell | Pure stage-2 | +v13 blend | Delta vs v13 (2106) |
|------|--------------|------------|---------------------|
| n5_v8 | 2057 | 2084 | -22 |
| n5_v10cap | 1924 | 2022 | -84 |
| n10_v8 | 1995 | 1999 | -107 |
| **n10_v10cap** | **1834** | **2126** | **+20** |
| n15_v8 | 1953 | 1999 | -107 |
| n15_v10cap | 1734 | 2017 | -89 |

Reference baselines (current XGB 3.2.0 env):
- v4 alone (pure stage-1): 1955
- v8 single-seed: 2034
- v8 30-seed ensemble (pure): 1965
- v13 (v8-ens30 @ toss-up blend): 2106

## LOSO pick distribution

The picker chose n10_v10cap in 21/22 seasons and n5_v8 in 1/22 (season 2015).
For test_season=2015:
- n10_v10cap 21-training-season sum: 2126 - 156 = **1970**
- n5_v8 21-training-season sum: 2084 - 108 = **1976**

A 6-point edge on training-season totals (1976 > 1970) caused the picker
to choose n5_v8 -- correctly hiding the held-out test season per LOSO
discipline. But the held-out season was n10_v10cap's biggest season
(156 brkt pts vs n5_v8's 108 -- a 48-point gap), so the picker traded
+6 in apparent training-season fit for -48 on the held-out score.

A 2-cell sensitivity test (n5_v8 + n10_v10cap only) produced the same
2078 total -- the 2015 pathology is not a wide-grid issue.

## Mechanism

The architecture has signal, but the signal is concentrated in one season
and the picker can't see it without peeking at the holdout.

**Concentration analysis (n10_v10cap vs v13):**
- 22-season total delta: +20
- Single-best season delta (2015): ~+50 brkt pts (n10_v10cap 156 vs
  v13's typical 100-110 for that season)
- Gross concentration: >50% of the lift comes from one season

By the spec's pre-registered rule (>50% single-season concentration
demotes PASS to MARGINAL), even cherry-picking the best single cell
collapses to MARGINAL with a fragility flag. The strict LOSO-disciplined
picker correctly rejects this.

## Pure stage-2 collapse outside the toss-up bucket

A striking finding: every v10cap cell scored DRAMATICALLY worse than v4
alone on pure stage-2 (n15_v10cap pure = 1734 vs v4 alone = 1955).
The v13-style toss-up blend rescues them by using pure v4 outside the
narrow [0.5, 0.55) confidence bucket.

This is internally consistent: with 15 v4-feature-diff inputs + 200 trees
of depth 4 on only ~3000 tournament training rows, stage-2 overfits chalk-
pick decisions outside the toss-up bucket. v4's `p_stage1` already
condenses the same information, so re-exposing the diffs to a wider
model lets stage-2 flip chalk picks the wrong way.

The toss-up bucket is where v4 is least confident, so chalk picks are
already 50/50; stage-2 only needs to nudge them, and the v4 diffs provide
useful conditioning at that narrow margin. Outside the toss-up bucket
the model has nothing to add and only adds noise.

## Capacity-N interaction

The grid surfaced an inverse interaction between feature count and
hparams capacity:
- Small N (5): v8 hparams (depth=3, n=100) wins (n5_v8 +blend = 2084 vs n5_v10cap = 2022)
- Medium N (10): v10cap hparams (depth=4, n=200) wins (n10_v10cap = 2126 vs n10_v8 = 1999)
- Large N (15): both hparams lose (n15_v8 = 1999, n15_v10cap = 2017)

So "more features need more capacity" holds at the N=5 to N=10 transition,
but breaks down at N=15 -- the extra features are noise-class and no
amount of capacity helps.

## Comparison to prior FAILs

This is the eighth same-data-peer addition to v4 that has failed
the 22-season bracket-points backtest:

| # | Experiment | Verdict |
|---|------------|---------|
| 1 | BT-as-feature (PR 14) | FAIL |
| 2 | Feature-view ensemble | FAIL |
| 3 | HBT (Hierarchical BT) | FAIL |
| 4 | Colley | FAIL |
| 5 | Massey-MOV | FAIL |
| 6 | Massey-decay-14d | FAIL |
| 7 | team-seed-residual (PR 34) | FAIL |
| 8 | v12 (v4 top-N feature diffs, this PR) | **FAIL** |

Important distinction from prior FAILs: 1-7 added the new signal as
**stage-1 features in v4 itself**. v12 added the signal as **stage-2
diffs on top of the existing v4 stage-1**. The mechanism is the same --
v4's 67-feature stack at the data scale we have is so well-fit that
re-exposing its inputs (in any form) doesn't compound -- but the
architectural surface is different. The v12 FAIL reinforces the
"v4 is near-saturated on tabular team-aggregate features at this data
scale" prior more strongly than 1-7 alone could (because it now also
applies at the stage-2 layer).

The v13 PASS (PR #37) remains the exception: v13 worked because it
added VARIANCE REDUCTION (30-seed averaging) and SELECTIVITY (toss-up
bucket only), not new signal. v12 tried to add new signal and failed
in the same pattern as 1-7.

## Decisions

- **Close the v12 lane.** Architecture-level "enrich stage-2 with
  v4 derivatives" is closed.
- **Do NOT change production frame.** `pairwise_v13.csv` remains the
  production output. `pairwise_v8.csv` remains the stage-2 same-env
  baseline.
- **Keep the v4 feature-importance artifact** (`output/v4_feature_importance.csv`)
  as a general-purpose reference -- it's useful context for the next
  active-queue lane (pool-aware bracket construction or roster data
  sourcing), even though v12 itself didn't pan out.

## Followups (not pursued)

- **Permutation importance ranking.** Would have changed which features
  v12 saw, but unlikely to escape the saturation + concentration
  problems we surfaced. Not worth the compute.
- **N > 15 grid.** The n=15 cells are already worse than n=5/10. Going
  higher would only make it worse.
- **alpha != 0.6 sweep on v12.** Quick sweep showed alpha=0.6 is
  near-optimal for n10_v10cap (alphas 0.6/0.8/1.0 all within 3 brkt
  pts). Not the bottleneck.
- **Wider hparams grid.** v10cap was already the deeper config. Going
  even deeper (n=400, depth=5) is unlikely to break the saturation.
- **2-stage stage-2 (per-bucket models).** Could fit different stage-2
  models for the toss-up bucket vs the rest. Risky given the n15_v10cap
  result -- more flexibility likely overfits at this data scale.

## Pointers

- Spec: `docs/superpowers/specs/2026-05-14-v12-stage2-v4-feature-diffs-design.md`
- Plan: `docs/superpowers/plans/2026-05-14-v12-stage2-v4-feature-diffs.md`
- Per-cell outputs: `output/pairwise_v12_n{5,10,15}_{v8,v10cap}.csv`
- LOSO-picked output: `output/pairwise_v12.csv`, `output/pairwise_v12_blend.csv`
- Summary: `output/v12_loso_pick_summary.json`
- v4 feature ranking: `output/v4_feature_importance.csv`
- v4 feature matrix snapshot: `output/v4_feature_matrix.parquet`
- v13 production frame (unchanged): `output/pairwise_v13.csv`
