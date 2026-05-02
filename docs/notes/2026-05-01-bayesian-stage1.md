# Bayesian / Bradley-Terry Stage-1 -- Findings

**Date:** 2026-05-01
**Branch:** feat/bayesian-stage1
**Verdict:** **NO-GO** -- gate failed. v4 stays as stage-1.
**Spec:** `docs/superpowers/specs/2026-05-01-bayesian-stage1-design.md`
**Plan:** `docs/superpowers/plans/2026-05-01-bayesian-stage1.md`

## TL;DR

A per-season plain Bradley-Terry stage-1 (binary outcomes, regular-
season-only, MAP via L2 logistic regression with team-indicator +
home-court design matrix) was tested against v4 for ensemble
suitability via the diagnostic gate. Gate FAILED at two of three
clauses: optimal blend weight `w_v4 = 0.98` (degenerate, outside
the spec's [0.30, 0.85] band) and headroom `+0.0000` log loss
(below the 0.005 threshold). Crucially, the residual correlation
clause PASSED at `r = 0.577 < 0.60` -- BT errors *are* meaningfully
less correlated with v4 than the LR experiment's r=0.77 were, so
the diversity hypothesis is partially validated. The blocker is
not correlation; it is BT's standalone weakness (LL 0.565 vs v4's
0.437). Saved ~3 hours of v9-C compute by gating before the full
backtest.

## Setup recap

Production today: `v4 (XGBoost stage 1) -> v9-C (XGBoost upset-aware
stage 2)`.

Experiment: per-season plain Bradley-Terry stage-1 fit on regular-
season games only, predicting tournament games. Implementation:
`sklearn.linear_model.LogisticRegression` with `penalty='l2'`,
`fit_intercept=False`, `C=10.0`, sparse design matrix of size
(n_games, n_teams + 1), where the extra column is the home-court
signal (`+1` winner-home, `-1` winner-away, `0` neutral).

Per-season fit times: ~0.3-0.4s each. Total runtime: 8 seconds (vs
~20 minutes for the LR experiment).

Per-season home-court coefficients: 0.41-0.67. Consistent with
public BT estimates (~0.3-0.5); on the high side here likely due
to conference-heavy regular-season schedules in college basketball.

Per-season tournament metrics:
- Log loss range: 0.45-0.67 (weakest model class on this task; the
  experiment's whole point is diversity, not standalone strength)
- Accuracy range: 0.62-0.78

Pair coverage exact: 48,465 unique pairs across 22 seasons (2003-
2025, excluding 2020), zero pairs unique to either side vs
`pairwise_v4.csv`.

## Diagnostic gate result

Computed on the existing pairwise outputs in 0.1s. The verdict comes
straight from `output/diag_bt_vs_v4.json`.

| measure                                       | value    | clause              |
|-----------------------------------------------|----------|---------------------|
| Pearson r(residual_v4, residual_bt)           | **0.577**| **PASS** (< 0.60)   |
| optimal blend weight w_v4 (cheating, no LOSO) | **0.98** | **FAIL** ([0.30, 0.85]) |
| log-loss headroom = LL_v4 - LL_optimal        | **0.0000**| **FAIL** (> 0.005) |
| **gate verdict**                              | -        | **FAIL**            |

Standalone metrics on the 1449 played 2003-2025 tournament games:

| metric                 | v4     | BT     |
|------------------------|--------|--------|
| weighted-mean log loss | 0.4369 | 0.5650 |
| weighted-mean accuracy | 0.805  | 0.698  |

Disagreement breakdown:

| outcome                  | count | %      |
|--------------------------|-------|--------|
| both correct             | 915   | 63.1%  |
| v4 only correct          | 251   | 17.3%  |
| BT only correct          | 97    | 6.7%   |
| both wrong               | 186   | 12.8%  |
| total disagreements      | 348   | 24.0%  |

When v4 and BT disagree on the predicted winner, BT is right only
97/(97+251) = 27.9% of the time. So BT *is* providing genuinely
different signal (24% disagreement rate, vs LR's 14%), but the
signal is mostly noise relative to v4 on this task.

## Falsification reasoning

The gate's three clauses isolate three distinct ways an ensemble
candidate can fail:

1. **Residual correlation > 0.60.** Errors too similar; averaging
   doesn't reduce variance. (LR experiment failed here at r=0.77;
   the LR's standalone quality was nearly v4's, but the diversity
   wasn't there.)
2. **Optimal weight outside [0.30, 0.85].** Optimal blend is
   degenerate -- effectively pure v4 or pure BT alone. (BT failed
   here at w=0.98.)
3. **Headroom <= 0.005 log loss.** Even the optimal blend doesn't
   meaningfully beat v4 alone. (BT failed here at +0.0000.)

The LR experiment hit clause 1: high correlation. The BT experiment
hits clauses 2 and 3: BT is too weak standalone. Two distinct failure
modes, both caught by the same gate -- which is what falsification
gates are for.

The interesting positive signal is clause 1 PASSING. r=0.577 is
substantially below LR's 0.77, confirming that **a structurally
different inductive bias does produce uncorrelated errors when we
hold feature inputs separate from v4's**. The hypothesis "model-class
diversity matters when training data is also disjoint" is supported.
The blocker is just standalone strength -- BT alone is too weak.

## Comparison to LR experiment

| measure                                       | LR (r=0.77) | BT (r=0.58) |
|-----------------------------------------------|-------------|-------------|
| residual correlation                          | 0.767       | 0.577       |
| disagreement rate                             | 14.0%       | 24.0%       |
| weaker model's standalone log loss            | 0.498       | 0.565       |
| optimal blend weight (cheating)               | w=0.93      | w=0.98      |
| headroom vs v4 alone                          | +0.0006     | +0.0000     |

LR was a "moderately strong but error-correlated" weak member; BT is
a "genuinely diverse but too weak" weak member. Neither makes a
useful ensemble partner with v4 at simple average weighting.

## Verdict

NO-GO. v4 stays as stage-1. The BT trainer code remains on this
branch as the experiment record. The diagnostic gate paid for itself
in compute saved (no v9-C run, no 22-season bracket-points
backtest -- ~3 hours saved over running the full pipeline).

## Recommendation

The diversity-correlation axis (clause 1) is now well-mapped after
two experiments at opposite ends. The next swing should target the
remaining axis: standalone strength of the diverse member.

Three concrete next steps, ordered by ratio of expected payoff to
implementation cost:

1. **Feature-view diversity ensemble: XGBoost trained on disjoint
   feature subsets** (e.g., one model on KenPom-only, one on
   Vegas-only, one on raw efficiency). Same model class -> same
   standalone strength as v4. Disjoint feature views -> different
   errors by construction. Promotes to active queue #1.

2. **Use BT predictions as a *feature* for v9-C, not as an ensemble
   peer.** v9-C currently sees `(p_v4_stage1, seed_a, seed_b,
   abs_seed_diff, round)`. Add a sixth feature: `p_bt_stage1`. This
   sidesteps the standalone-strength problem entirely -- v9-C gets
   to learn when to weight BT vs v4 conditional on context (seeds,
   round, confidence). Cheaper than adding a fourth model class:
   reuse the existing `pairwise_bt.csv`, modify `upset_features` in
   `train_upset_model.py`, re-tune. Promotes to active queue #2.

3. **Hierarchical Bradley-Terry with feature priors:** `s_team ~
   Normal(beta . v4_features_team, sigma)`. Couples BT back to the
   v4 feature view so it gains standalone strength. The risk is
   that this re-couples BT's errors to v4's (clause 1 may regress
   from 0.58 back toward 0.77). Stays on the queue but not as
   immediately actionable as items 1 and 2.

The "full Bayesian with strength + variance" item (PyMC, MCMC) is
deferred further -- standalone strength is the bottleneck right
now, and switching from MAP to full posterior won't change that.

## Files of record

```
src/train_bt_stage1.py             -- per-season BT trainer
src/diagnose_bt_vs_v4.py           -- gate diagnostic

output/pairwise_bt.csv             -- BT LOSO output (48,465 pairs)
output/diag_bt_vs_v4.json          -- gate diagnostic with verdict

tests/test_train_bt_stage1.py
tests/test_diagnose_bt_vs_v4.py
```

The gate's three thresholds (`GATE_R_MAX=0.60`, `GATE_W_LOW=0.30`,
`GATE_W_HIGH=0.85`, `GATE_HEADROOM_MIN=0.005`) are constants in
`src/diagnose_bt_vs_v4.py` and reusable for future stage-1
candidates. Just point the diagnostic at a new pairwise CSV
(`--pairwise-bt output/pairwise_NEW.csv`) and read off the verdict.
