# Plain BT Bracket-Points Re-Test -- Findings

**Date:** 2026-05-04
**Branch:** feat/bt-bracket-points
**Verdict:** **NO-GO**. v4 + v9-C stays as production.
**Spec:** `docs/superpowers/specs/2026-05-04-bt-bracket-points-design.md`
**Plan:** `docs/superpowers/plans/2026-05-04-bt-bracket-points.md`

## TL;DR

Tested whether plain BT helps v4+v9-C on the production metric
(bracket points) despite failing the log-loss-blend gate. It does
not. Every non-anchor weight cell LOST bracket points vs v4+v9-C,
ranging from -29 (`w_v4=0.90`) to -67 (`w_v4=0.80`). The LL gate's
NO-GO verdict for plain BT (PR 12) was correct on the production
metric too. The HBT findings note's framing-correction concern --
that the LL gate might be filtering on the wrong metric -- is
falsified for plain BT specifically.

## Setup recap

- **Inputs (force-added in prior PRs):**
  - `output/pairwise_v4.csv` -- v4 stage-1, 96,930 rows.
  - `output/pairwise_bt.csv` -- plain BT stage-1, 48,465 rows.
  - `output/pairwise_v9c_v4_baseline.csv` -- v4 + v9-C baseline,
    canonical reference for delta computation.
- **Pipeline:** ensemble (`average_pairwise_csvs`) -> v9-C
  (`run_v9c_on_stage1`, `W_UPSET=1.25, W_MISS=0.0,
  feature_set='v9c'`) -> bracket scoring
  (`score_chalk_brackets.score_pairwise_path`).
- **Anchor:** `w_v4=1.00` cell reproduced the baseline exactly
  (`max_abs_diff=0.0` over 48,465 rows; per-season W/L/T = 0/0/22).
  Pipeline is sound.
- **Wall time:** ~90 seconds for the full 6-cell sweep.

## Per-cell results

v4 + v9-C baseline: 2713 brkt pts over 22 LOSO seasons.

| w_v4 | w_bt | total_pts | delta | W | L | T | biggest single-season swing |
|------|------|-----------|-------|---|---|---|------------------------------|
| 0.60 | 0.40 | 2657      | -56   | 5 | 10| 7 | 2006 (-33)                   |
| 0.70 | 0.30 | 2654      | -59   | 7 | 10| 5 | 2025 (+32)                   |
| 0.80 | 0.20 | 2646      | -67   | 6 | 8 | 8 | 2006 (-29)                   |
| 0.90 | 0.10 | 2684      | -29   | 5 | 6 | 11| 2003 (-16)                   |
| 0.95 | 0.05 | 2665      | -48   | 5 | 7 | 10| 2003 (-16)                   |
| 1.00 | 0.00 | **2713**  | +0    | 0 | 0 | 22| (anchor)                     |

Best non-anchor cell: `w_v4=0.90` at -29 brkt pts. Every cell with
non-zero BT weight loses. Verdict: **NO-GO**.

## What the LL gate said vs what bracket points said

| metric                     | plain-BT verdict | reasoning                                  |
|----------------------------|------------------|--------------------------------------------|
| LL-blend gate (PR 12)      | NO-GO            | `w_opt=0.98, headroom=+0.0000`             |
| Bracket points (this)      | NO-GO            | best non-anchor delta -29 at `w_v4=0.90`   |

**Both metrics agree.** For plain BT specifically, the LL gate was
right. The HBT findings note's hypothesis -- that the LL gate
might be screening unsoundly -- is falsified at this point in the
design space. The cheap diagnostic and the production metric
agree on plain BT.

This does NOT generalize to "the LL gate is right in all cases."
What it shows is: **plain BT's standalone weakness sinks the
ensemble on every metric we have**, not just LL. The
`r=0.577` residual diversity is real, but BT's individual
upset predictions are wrong more often than they are right when
they disagree with v4 -- the biggest single-season swings are
mostly negative (`2006 -33`, `2003 -16` repeated, `2008 -16`),
with one large positive outlier at `2025 +32`.

## Why doesn't BT's diversity translate?

The plain-BT findings note (PR 12) already showed:

> When v4 and BT disagree on the predicted winner, BT is right
> only 97 / (97 + 251) = 27.9% of the time.

So 72% of disagreements go BT's way wrong. v9-C cannot extract
useful upset signal from a stage-1 source whose disagreement
direction is more often wrong than right. The W_MISS sweep on PR 9
that promoted residual weighting was acting on v4's *own* residuals
under v9-C's training -- NOT on BT residuals. We mistakenly thought
the W_MISS finding generalized to "BT-style residuals carry useful
signal." It doesn't.

## Lesson

For a stage-1 candidate to help v4+v9-C, it likely needs either:
- **Higher per-disagreement accuracy than v4's complement.** Plain
  BT is at 28% on disagreements; that's too low. A stage-1 with say
  >= 45% would actually help on the upsets it flags.
- **Or accuracy on a region v4 misses systematically.** External
  data (Vegas, 538) might do this -- their disagreements with v4
  may concentrate where v4 is over- or under-confident.

The "structurally diverse" framing was insufficient by itself. We
also need "the diverse predictions, in the regions where they
disagree, are more often right than wrong." Plain BT failed that
second test.

## Implications for active queue

Active queue item #2 (this experiment) is closed as NO-GO. The
"LL gate may be the wrong metric" hypothesis is falsified for
plain BT specifically. New ordering:

1. **Localize v4's gap vs an external benchmark** (was #1 -- stays).
   This is now the strongest lever -- both architectural ensemble
   experiments (LL-gate and bracket-points-gate) have failed for
   plain BT. The bottleneck is more likely inside v4 itself or in
   adding genuinely new signal, not in re-engineering BT-class
   stage-1s. Vegas closing-line implied probabilities is the
   cheapest start (data already in `data/raw/vegas_lines/`).
2. **External rankings / external data as features** (was #3 ->
   bumped up). Genuinely outside the Kaggle + KenPom + Bart Torvik
   archive. 538, Vegas prop-bets, roster injuries.
3. **Small NN (MLP) as a stage-1.** Lower priority -- the
   per-disagreement-accuracy lesson means *any* same-data BT-class
   peer faces the same risk. NN on the same 67-feature space is a
   different model class but trained on the same target as v4, so
   the LR-experiment correlation problem still applies.
4. **Full Bayesian BT.** Lower priority -- HBT showed prior
   structure doesn't lift BT-class standalone strength on either
   metric.
5. **Roster-level returning-experience.** Closely related to #2.

## Files of record

```
src/sweep_bt_bracket_points.py            -- 6-cell driver
tests/test_sweep_bt_bracket_points.py     -- 6 unit tests

output/pairwise_v4bt_w<W>.csv x 6         -- ensemble outputs (force-added)
output/pairwise_v9c_v4bt_w<W>.csv x 6     -- post-v9-C (force-added)
output/bt_bracket_sweep.json              -- per-(w, season) numbers + verdict
output/bt_bracket_log.txt                 -- run stdout
output/bt_bracket_anchor_only.json        -- pre-flight anchor verification
```

## Compute

~90 seconds for the full 6-cell sweep + ~12 seconds for the
pre-flight anchor-only run. Anchor reproduced baseline to 0.0
max-abs-diff over 48,465 rows. The pipeline is correct; the
verdict is the data.
