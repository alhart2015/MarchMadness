# Plain BT Re-Test on Bracket Points (Skip the LL Gate) -- Design

**Date:** 2026-05-04
**Branch:** feat/bt-bracket-points
**Predecessors:**
- Plain BT (PR 12, NO-GO on LL gate): `docs/notes/2026-05-01-bayesian-stage1.md`
- Hierarchical BT (PR 16, NO-GO on LL gate, framing corrected):
  `docs/notes/2026-05-03-hierarchical-bt.md`
- v9-C production swap (current production stage-2):
  `docs/superpowers/specs/2026-05-01-v9c-production-swap-design.md`

## Motivation

The HBT findings note's framing-correction postscript flagged a
specific concern that was masked by the original "training-data
ceiling" overclaim:

> The 3-clause gate measures LL-blend headroom. The production
> metric is bracket points, where correctly-predicted upsets are
> scored at heavy multipliers (1, 2, 4, 8, 16, 32 per round). A
> weak-but-diverse stage-1 that flips a few v4 picks toward true
> upsets could lift bracket points without lifting log-loss-blend
> headroom. Plain BT's `r=0.577` is genuine residual diversity that
> v9-C's `W_MISS` weight sweep (PR 9) already showed contains
> useful signal.

In other words: plain BT was rejected by the LL gate, but never
tested against the metric we actually care about. The user's
2159 / 3462 Kaggle finish underlines the point -- v4 has headroom we
have not localized, and bracket points is one place that headroom
might surface.

This experiment skips the LL gate entirely. It blends v4 + plain BT
at a weight grid, runs v9-C on each, scores 22-season bracket points
head-to-head against the canonical `v4 + v9-C` baseline, and applies
the standard verdict ladder.

## Goals

- Score `(w * pairwise_v4 + (1 - w) * pairwise_bt) -> v9-C` against
  bracket points for `w in {0.60, 0.70, 0.80, 0.90, 0.95, 1.00}`.
- Verify the `w=1.00` cell reproduces the existing `v4 + v9-C`
  baseline (`pairwise_v9c_v4_baseline.csv`) exactly. Anchor.
- Apply the standard ladder:
  - `delta >= +25` brkt pts -> CLEAR (separate swap-in commit).
  - `+10 <= delta < +25` -> MARGINAL (document but do not swap).
  - `delta < +10` -> NO-GO.
- Either way, ship a clean record of what the bracket-points re-test
  showed about plain BT specifically, and update the LL-gate's
  status as a screening tool going forward.

## Non-Goals

- Replacing the LL gate. The LL gate is a cheap filter; whether it
  is well-calibrated for the bracket-points task is the *finding*
  of this experiment, not its premise.
- Re-tuning v9-C's `W_UPSET` / `W_MISS` / `feature_set`. Reuse the
  PR 9 winning cell verbatim (`W_UPSET=1.25, W_MISS=0.0,
  feature_set='v9c'`). v9-C is the production stage-2; we are
  testing whether plain BT helps that stage-2 see different stage-1
  signal, NOT swapping out v9-C.
- Re-running plain-BT training. Reuse `output/pairwise_bt.csv`
  (force-added on PR 12, ASCII-clean, 48,465 rows, exact pair
  coverage match with `pairwise_v4.csv`). The trainer code lives
  on PR 12's branch as the experiment record; running it again is
  unnecessary.
- HBT cells. The HBT result already showed standalone weakness
  *worse* than plain BT across all sigmas; bracket-points re-test
  for HBT cells would just reconfirm. If plain BT clears the ladder,
  HBT @ best-sigma is a follow-up.
- BT-as-feature for v9-C. Already tested on PR 13 with NO-GO.
- Live bracket integration (`generate_bracket_real.py`). Same
  out-of-scope rationale as every prior stage-1 experiment.
- Cross-season weight tuning. Single weight value applied uniformly
  across all 22 seasons (LOSO discipline). Per-season weight
  tuning is a follow-up if the uniform sweep is borderline.

## Approach

### Architecture

```
For each w in {0.60, 0.70, 0.80, 0.90, 0.95, 1.00}:
    src/ensemble_stage1.py --in-a pairwise_v4.csv --in-b pairwise_bt.csv \
        --weights w,(1-w) --out output/pairwise_v4bt_w<W>.csv
    src/run_v9c_on_stage1.py --pairwise-in output/pairwise_v4bt_w<W>.csv \
        --pairwise-out output/pairwise_v9c_v4bt_w<W>.csv

Then score each output via score_chalk_brackets.score_pairwise_path
and compute:
    delta_w = total_brkt_pts(v4bt_w) - total_brkt_pts(v4_baseline)

Where v4_baseline = output/pairwise_v9c_v4_baseline.csv (already exists,
force-added on prior PR).
```

The whole experiment is a single driver script
`src/sweep_bt_bracket_points.py` that loops the grid, calls existing
modules unchanged, scores, and writes `output/bt_bracket_sweep.json`
with per-(w, season) bracket points + per-w totals + verdict.

### Weight grid

`w in {0.60, 0.70, 0.80, 0.90, 0.95, 1.00}` -- six cells.

Rationale:
- `1.00` is the anchor: must reproduce v4 + v9-C exactly. If it does
  not, the pipeline is broken; halt before scoring other cells.
- `0.95, 0.90, 0.80, 0.70, 0.60` covers the realistic range. Plain
  BT's LL-optimal blend was 0.98, well outside the band where any
  cell would help LL. But the bracket-points objective is different:
  the W_UPSET=1.25 / W_MISS=0.0 v9-C is sensitive to upset
  predictions, and even 5-15% BT weighting could meaningfully nudge
  v9-C inputs toward more-upset-predicting probabilities for
  pairs where v4 is overconfident on the favorite.
- No cell below 0.60. Plain BT's standalone LL is 0.565 -- pulling
  v4 below 60% weight makes the input substantially worse on log
  loss, and v9-C's input quality matters at some level (residuals
  computed from the input feed v9-C's training labels).

### Anchor / sanity checks

- **`w=1.00` cell.** `pairwise_v4bt_w1.00.csv` should equal
  `pairwise_v4.csv` row-for-row (modulo dedup). After v9-C,
  `pairwise_v9c_v4bt_w1.00.csv` should equal
  `output/pairwise_v9c_v4_baseline.csv` exactly. Verify before
  scoring.
- **Pair coverage.** v4 + BT join on `(season, team_a, team_b)` is
  perfect (HBT findings note confirmed via direct check: zero
  pairs unique to either side, 48,465 overlap rows). The ensemble
  output should match plain BT's pair count per season (~2200).
- **Per-season scoring sanity.** `score_chalk_brackets.score_pairwise_path`
  on the existing `pairwise_v9c_v4_baseline.csv` should produce v4 +
  v9-C's known per-season bracket points. Use that as the baseline
  comparison for every grid cell.

### Eval methodology

22-season LOSO (2003-2025, 2020 implicit). Bracket points per season
+ total. Same scoring weights as the rest of the codebase: `[1, 2,
4, 8, 16, 32]` per round (R64 -> Champ).

### Compute budget

Per cell: ~1-2 minutes for v9-C run, seconds for ensemble + scoring.
Six cells = ~10-15 min total. Plus the anchor verification.

### Output schema

`output/bt_bracket_sweep.json`:

```json
{
  "config": {
    "weights": [0.60, 0.70, 0.80, 0.90, 0.95, 1.00],
    "v4_pairwise": "output/pairwise_v4.csv",
    "bt_pairwise": "output/pairwise_bt.csv",
    "v9c_baseline": "output/pairwise_v9c_v4_baseline.csv"
  },
  "anchor_check": {
    "w_1.00_matches_v4_baseline": true,
    "max_abs_diff": 1e-15
  },
  "v4_baseline": {
    "total_pts": NNNN,
    "per_season": { "2003": ... }
  },
  "cells": [
    {
      "w": 0.60,
      "ensemble_csv": "...",
      "v9c_csv": "...",
      "total_pts": NNNN,
      "delta_vs_baseline": +NN,
      "per_season": {...},
      "wins": N, "losses": N, "ties": N
    },
    ...
  ],
  "best_cell": { "w": ..., "delta": ... },
  "verdict": "CLEAR | MARGINAL | NO-GO"
}
```

## Falsification ladder (per cell, applied to the best cell)

Same bands as every prior stage-1 experiment:

- `delta >= +25` -> CLEAR. Separate swap-in commit.
- `+10 <= delta < +25` -> MARGINAL. Document; do not swap.
- `delta < +10` -> NO-GO.

If multiple cells exceed +25, pick the cell with the largest delta
that has a sane W/L/T profile (not driven by one season).

## Test plan

- `tests/test_sweep_bt_bracket_points.py`:
  - Unit: anchor verification (`w=1.00` produces v4 baseline exactly).
  - Unit: weight grid order + (1-w) computation correct.
  - Unit: per-cell summary dict shape correct.
- Existing test suite (`pytest -v`) must remain green.

## Risks

1. **No cell beats v4+v9-C by >= +10 brkt pts.** Likely outcome.
   Tightens the conclusion: plain BT does not help v4 even on the
   bracket-points metric. Falsifies the LL-gate-was-wrong hypothesis
   for plain BT specifically. Note: this would NOT mean the LL gate
   was right in general; it would mean plain BT specifically is
   bracket-points-equivalent at uniform blending. Other future
   stage-1 candidates may still benefit from a bracket-points eval
   alongside the LL gate.
2. **Anchor at `w=1.00` doesn't reproduce baseline.** Bug in
   `ensemble_stage1.py` weight=1.0 anchor (already tested on PR 11
   to pass) or a dedup difference between the v4 baseline and the
   freshly-rerun v9-C output. Halt and investigate.
3. **One-season skew.** Bracket points is high-variance per season;
   a +30 delta could be one fluky bracket. Track W/L/T and the
   biggest single-season swing to flag fragile wins.
4. **v9-C re-runs are non-deterministic.** v9-C uses XGBoost, which
   may have a small stochastic component. Set seeds or re-run v4
   baseline and confirm match. The existing `pairwise_v9c_v4_baseline.csv`
   was force-added precisely to be a stable reference; we
   compare against that, not against a re-run.

## File-touch summary

```
new   docs/superpowers/specs/2026-05-04-bt-bracket-points-design.md
new   docs/superpowers/plans/2026-05-04-bt-bracket-points.md
new   src/sweep_bt_bracket_points.py
new   tests/test_sweep_bt_bracket_points.py
new   output/pairwise_v4bt_w<W>.csv x 6      (force-added)
new   output/pairwise_v9c_v4bt_w<W>.csv x 6  (force-added)
new   output/bt_bracket_sweep.json           (force-added)
new   docs/notes/2026-05-04-bt-bracket-points.md   (findings)
edit  TODO.md                                (move active queue
                                              item #2 to done /
                                              tried-and-rejected
                                              with verdict)
```

## Promotion path

```
sweep runs (~15 min)
   |
   +-- best delta < +10  -> NO-GO. Findings: plain BT specifically
   |                        does not help bracket points. Tightens
   |                        the HBT framing correction.
   |
   +-- best delta in     -> MARGINAL. Document; do not swap. Note
   |   [+10, +25)           the LL gate is screening this case
   |                        unsoundly (passed on bracket points
   |                        despite failing LL).
   |
   +-- best delta >= +25 -> CLEAR. Separate swap-in commit.
                            v4 -> v4 + plain BT @ best w as new
                            stage-1 input to v9-C.
```
