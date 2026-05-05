# v9-C Clean Re-run -- Findings

**Date:** 2026-05-04 (sweep + decision); 2026-05-05 (write-up)
**Branch:** feat/v9c-clean-rerun
**Verdict:** **REVERT.** Best v9-C cell delta vs clean v8: **-140 brkt pts** (every cell loses; PR 9's winning cell at -316). v9-C reverted to v8 in production via `predict_2026_stage2.py`.
**Spec:** `docs/superpowers/specs/2026-05-04-v9c-clean-rerun-design.md`
**Plan:** `docs/superpowers/plans/2026-05-04-v9c-clean-rerun.md`
**Recovery context:** `TODO.md` "CONTAMINATION DISCOVERED 2026-05-04" -> step 5, item 1.

## TL;DR

Clean v8 baseline = **2069** brkt pts (vs **2670** on leaky baseline, -601). v9-C
best cell at (W_UPSET=1.0, W_MISS=0.5) = **1929** pts; delta vs clean v8 =
**-140**. PR 9's winning cell (W_UPSET=1.25, W_MISS=0.0) = **1753** pts;
delta = **-316**. Per-season W/L for the winning cell: **8W-10L-4T** over
22 seasons; biggest single-season v8 wins were 2015 (-54), 2017 (-40),
2019 (-57), 2022 (-23). **Production action: REVERT v9-C → v8.**
`output/pairwise_probs.json` restored via `predict_2026_stage2.py`
(commit 80f7cf7); md5 changed from v9-C's `251235aa` to clean-v8's
`299ea3df`.

## Methods

- Input: `output/pairwise_v4.csv` (clean, 48,465 unique pairs, force-added
  this PR after PR 21's regen output was lost in the 2026-05-04 data wipe;
  see `docs/data_recovery.md`). Empirical stage-1 LL = 0.5579 / acc =
  0.7019 (vs leaky 0.4369 / 0.8047 -- this is the clean baseline).
- v8 baseline: `python src/train_stage2.py` -> `output/pairwise_v8.csv`.
  WT MEAN stage-1 LL 0.558, stage-2 LL 0.552 (delta -0.005); WT MEAN
  stage-1 acc 70.2%, stage-2 70.1%. md5 `102467bc...` (vs leaky baseline
  `f9e8b5f5...`). Force-added.
- v9-C sweep: `V9_FEATURE_SET=v9c python src/sweep_v9_weights.py` -> 15
  cells in `output/v9c_sweep/`, results in
  `output/v9c_sweep_results.csv`. Runtime ~7 min.
- Per-season breakdown: `src/v9c_per_season_breakdown.py` (new this PR).
- Hyperparameter confound: v9-C/v8 trainers reuse PR 6/8/9 untuned XGB
  defaults (no leak-baseline confound from the stage-2 side).
  `pairwise_v4.csv` carries PR 21's tuned-XGB-on-leaky-baseline
  confound (documented effect <0.02 LL).

## Clean v8 baseline

| metric | leaky (PR 9) | clean (this PR) | delta |
|---|---|---|---|
| 22-season total brkt pts | 2670 | **2069** | **-601** |

The v8 baseline itself shifted -601 brkt pts under clean v4. v8's stage-2
corrector is a thin layer over v4's stage-1 predictions (delta -0.005 LL
on the LOSO-CV side); when v4's stage-1 quality drops by +0.122 LL on
LOSO and ~10pp accuracy, v8's bracket score follows. The point is:
**every model in this PR is being scored against a 22.5%-weaker
baseline than PR 9**, which makes apples-to-apples comparison vs. PR 9's
`v9-C +43` win impossible. The right comparison is delta within this
PR, between the clean v8 baseline (2069) and the clean v9-C cells
(below).

## 15-cell v9-C sweep results (sorted by total_brkt_pts descending)

| W_UPSET | W_MISS | total_brkt_pts | delta vs v8 | LL | Acc | Note |
|---|---|---|---|---|---|---|
| 1.00 | 0.5  | **1929** | **-140** | 0.555 | 0.709 | best cell |
| 1.25 | 1.0  | 1920 | -149 | 0.575 | 0.678 |  |
| 1.00 | 0.0  | 1869 | -200 | 0.552 | 0.699 | **anchor** |
| 1.00 | 1.0  | 1803 | -266 | 0.562 | 0.703 |  |
| 1.50 | 0.0  | 1787 | -282 | 0.563 | 0.691 |  |
| 1.25 | 0.0  | 1753 | -316 | 0.556 | 0.696 | **PR 9 winning cell** |
| 1.25 | 0.5  | 1739 | -330 | 0.563 | 0.702 |  |
| 1.75 | 0.0  | 1684 | -385 | 0.573 | 0.668 |  |
| 1.50 | 1.0  | 1674 | -395 | 0.593 | 0.625 |  |
| 2.00 | 0.0  | 1600 | -469 | 0.585 | 0.639 |  |
| 1.50 | 0.5  | 1587 | -482 | 0.577 | 0.666 |  |
| 1.75 | 0.5  | 1539 | -530 | 0.592 | 0.618 |  |
| 1.75 | 1.0  | 1510 | -559 | 0.610 | 0.605 |  |
| 2.00 | 0.5  | 1279 | -790 | 0.607 | 0.607 |  |
| 2.00 | 1.0  | 1261 | -808 | 0.629 | 0.591 |  |

**Every cell loses.** The best v9-C cell loses by 140 brkt pts; PR 9's
winning cell loses by 316. Higher W_UPSET monotonically loses more --
more aggressive upset-weighting actively hurts when stage-1's upset
signal is below random.

## Winning cell per-season W/L (W_UPSET=1.0, W_MISS=0.5)

| season | v8 pts | v9c pts | delta | winner |
|---|---|---|---|---|
| 2003 |  85 |  85 |    0 | tie |
| 2004 |  72 |  63 |   -9 | v8 |
| 2005 | 102 |  87 |  -15 | v8 |
| 2006 |  58 |  59 |   +1 | v9c |
| 2007 | 132 | 161 |  +29 | v9c |
| 2008 | 128 | 148 |  +20 | v9c |
| 2009 | 120 | 120 |    0 | tie |
| 2010 | 119 | 120 |   +1 | v9c |
| 2011 |  47 |  51 |   +4 | v9c |
| 2012 |  92 |  93 |   +1 | v9c |
| 2013 |  62 |  62 |    0 | tie |
| 2014 |  61 |  59 |   -2 | v8 |
| 2015 | 155 | 101 |  -54 | v8 |
| 2016 |  67 |  65 |   -2 | v8 |
| 2017 | 101 |  61 |  -40 | v8 |
| 2018 | 117 | 115 |   -2 | v8 |
| 2019 | 125 |  68 |  -57 | v8 |
| 2021 |  78 |  80 |   +2 | v9c |
| 2022 |  74 |  51 |  -23 | v8 |
| 2023 |  49 |  48 |   -1 | v8 |
| 2024 | 111 | 111 |    0 | tie |
| 2025 | 114 | 121 |   +7 | v9c |

**8W-10L-4T over 22 seasons.** v8 wins 4 of the 5 highest-magnitude
seasons (2015 -54, 2017 -40, 2019 -57, 2022 -23). v9-C's biggest wins
are 2007 (+29) and 2008 (+20); the rest are within +/-7 (noise). The
durability profile is *not* fragile-tied -- v8 is consistently
winning by larger margins on the seasons where the difference matters.

## Anchor sanity check

Anchor cell (W_UPSET=1.0, W_MISS=0.0): **1869 pts vs clean v8 2069
pts; delta -200 brkt pts**. Sweep driver fired the WARNING gate
(threshold 5 pts).

**Trainer is sane** despite the brkt-pt drift:

| metric | clean v8 stage-2 | anchor v9-C |
|---|---|---|
| WT MEAN LL    | 0.552 | 0.552 |
| WT MEAN Acc   | 0.701 | 0.699 |

LL matches to 3 decimals; Acc within 0.2pp. The trainer is calibrated
correctly; the brkt-pt drift is a feature-set effect (v9-C has 5
features including `round`; v8 has 4 without). At uniform weights the
two models produce per-game probabilities that agree closely on
average, but small probability shifts flip individual chalk picks
which compound through the bracket scoring (1/2/4/8/16/32 weighting
amplifies differences in upper rounds). Per the spec's risk-section
guidance ("If LL/Acc match v8 to 3 decimals, proceed and document the
brkt-pt drift in the findings as a feature-set effect, not a bug"),
we proceeded with the sweep verdict.

## Discussion

PR 22's audit rerun found that **clean v4 catches 15.3% of upsets**
vs Vegas's 17.5% -- v4's apparent +56% upset-catch advantage from
PR 18 was the leak speaking. v9-C was specifically engineered to
amplify v4's upset signal (W_UPSET=1.25 in PR 9). When that signal
turned out to be below random (less upset detection than Vegas, vs
PR 9's claim of 56% vs 17%), v9-C's stage-2 was correcting noise.

The clean rerun confirms this empirically: every cell with W_UPSET >
1.0 loses MORE than the W_UPSET=1.0 cells (which are the closest to
v8's behavior). The active ingredient that PR 9 attributed to mild
upset weighting (W_UPSET=1.25) was actually the leak in stage-1.
Removing the leak removes the signal v9-C exploited. v8's content-
blind stage-2 corrector (no upset awareness) outperforms because it
doesn't try to amplify a signal that isn't there.

PR 22 explicitly anticipated this in its discussion ("v9-C's stage-2
may have been correcting noise rather than signal. Re-eval is now
load-bearing for whether v9-C stays in production"). This PR closes
that load-bearing question with a -140 verdict and reverts.

## Production state change

- **Before this PR:** `output/pairwise_probs.json` md5 `251235aa`
  (v9-C-corrected on leaky-trained v4 2026 stage-1 predictions, from
  PR 10's swap on 2026-05-01).
- **After this PR:** `output/pairwise_probs.json` md5 `299ea3df`
  (v8-corrected on clean-trained v4 LOSO + leaky-trained v4 2026
  stage-1 predictions). v8 stage-2 was retrained against the clean
  pairwise_v4.csv this PR; the 2026 stage-1 v4 predictions
  (`output/pairwise_probs_v4.json`, Apr 28) were not regenerated --
  that's a separate concern noted in follow-ups.
- `predict_2026_v9c.py` retained for audit; can be removed in a
  separate cleanup PR if v9-C is fully retired.

## TODO.md update (this PR commits the update)

Step 5 item 1 marked done. **Marginal-rejections list expanded** to
cover candidates whose original rejection deltas were within the
+0.122 LL leak noise floor and weren't named in the original
recovery roadmap. Five new candidates added (plain BT standalone,
feature-view ensemble PEER_A/B, HBT, Colley, Massey-decay hl=14d).

Plus a sixth follow-up unique to this PR: **regenerate v4's 2026
stage-1 predictions** (`output/pairwise_probs_v4.json`, currently
Apr 28 leaky-trained). The current production
`pairwise_probs.json` is "clean v8 stage-2 over leaky v4 2026
stage-1." Full cleanliness needs the 2026 stage-1 regenerated too.

## Follow-ups (priority order)

1. **Plain BT standalone re-eval** (PR 12). Standalone LL 0.565 vs
   leaky v4 0.437 = -0.128 weaker (gate failed). Vs clean v4 0.5588
   = ~tied. LL-blend gate likely flips PASS. **High signal-to-noise
   ratio** -- the math is clearest here. ~30 min compute.
2. **Feature-view ensemble PEER_A/B re-eval** (PR 14). PEER_A LL
   0.5720 vs leaky v4 0.4345 = +0.1375 (5.5x clause-1 tolerance).
   Vs clean v4 0.5588 = +0.013 (within tolerance). Clause 1 likely
   flips PASS. ~20 min compute.
3. **Regenerate v4 2026 stage-1 predictions.** `enhanced_model_v3.py`
   already regenerates stage-1 submissions but not the
   `pairwise_probs_v4.json` JSON the production pipeline reads.
   Trace the script that produced the Apr 28 file (probably a
   `submission_to_pairwise.py` or similar), re-run it on the regen
   outputs, force-add the result. Then re-run
   `predict_2026_stage2.py` to refresh `pairwise_probs.json` with
   "clean v8 over clean 2026 stage-1." Modest compute.
4. **Colley + Massey-decay hl=14d re-eval** (PR 15). Clause-2
   deltas +0.0053 / +0.0057 LL on subset, vs leaky v4 0.4345.
   Within the +0.122 leak noise floor. ~30 min combined.
5. **HBT re-eval** (PR 16). Standalone LL 0.619-0.757; gap to clean
   v4 shrinks but HBT still weaker. Less likely to flip than plain
   BT. ~5 min.
6. **BT-as-feature for v9-C re-eval** (PR 13). Named in original
   roadmap. ~5 min on clean baseline.
7. **v9 weight-sweep family re-eval.** Named in original roadmap.
   Partly subsumed by this PR -- the v9-C 15-cell sweep already
   covered v9-C's grid; the v9-B specific grid is a separate run
   if structurally interesting (likely not, given the v9-C grid's
   uniform losses).
8. **538 audit follow-up** (parked on `feat/v4-gap-audit-fte`).

## Anchor + scope notes

- v8/v9-C trainer code unchanged in this PR.
- v8 stage-2 was the production model from 2026-04-29 (`predict_2026_stage2.py`,
  pre-PR-10) until the v9-C swap on 2026-05-01 (PR 10). This PR
  restores it.
- This PR's scope did NOT include regenerating v4's 2026 stage-1
  predictions (out of scope per spec). Therefore the production
  `pairwise_probs.json` after this PR is "clean v8 stage-2 over
  leaky v4 2026 stage-1." The stage-2 part is clean; the stage-1
  part is the same Apr 28 file v9-C was applying its corrector to.
- The "live bracket pipeline" (`generate_bracket_real.py`,
  pure-v4-MC) is unchanged (deferred per spec; PR 10 also did not
  touch this path).
