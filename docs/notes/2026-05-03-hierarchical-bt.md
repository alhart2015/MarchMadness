# Hierarchical Bradley-Terry with v4 Feature Priors -- Findings

**Date:** 2026-05-03
**Branch:** feat/hierarchical-bt-priors
**Verdict:** **NO-GO on the LL-blend gate.** v4 stays as stage-1.
**Spec:** `docs/superpowers/specs/2026-05-03-hierarchical-bt-feature-priors-design.md`
**Plan:** `docs/superpowers/plans/2026-05-03-hierarchical-bt-feature-priors.md`

> **Framing correction (2026-05-04):** the original write-up of this
> experiment (Sections "Why the anchors miss," "The deeper lesson,"
> "Updated diversity-strength frontier," and "Implications for active
> queue" below) extrapolated to a "training-data ceiling" -- the claim
> that no BT-class model trained on regular-season W/L can ever close
> the gap with v4. That claim is too strong: the user finished
> 2159 / 3462 in a recent Kaggle Mania, so 2/3 of entries beat v4 on
> the same data. The data clearly has more in it than v4 extracts.
> See "What this experiment actually shows" and "What it does NOT
> show" at the bottom for the corrected interpretation. The numerical
> results and "Why the anchors miss" diagnoses are correct; only the
> generalization to "no model can help" was wrong.

## TL;DR

Per-season hierarchical BT MAP with `s_team_i ~ Normal(beta . v4_features_team_i, sigma^2)`
was tested across 7 sigma cells {0.05, 0.10, 0.20, 0.50, 1.00, 2.00, 5.00}.
Every cell FAILED the LL-blend gate at clauses 2 and 3 (`w_opt =
0.99-1.00`, `headroom = 0.0000`). The hypothesis was that priors-from
-v4-features would lift HBT's standalone strength toward v4's; in
practice they made standalone log loss **worse** than plain BT's
0.565 across the full sigma range (HBT range: 0.619-0.757). Clause 1
(residual correlation) passed at every cell with `r in [0.448,
0.507]` -- even *lower* than plain BT's 0.577.

Narrow conclusion: **adding v4 feature priors to BT does not improve
its standalone log loss enough to clear the LL-optimal-blend gate,
across any tested sigma**. Broader extrapolations from this single
experiment to "no BT-class model can help v4" or "the training data
is exhausted" are NOT supported -- see the framing-correction note
above and the corrected interpretation at the bottom.

## Setup recap

- **Model:** per-season joint MAP over `(s, beta, h)`:
  - `L = sum_g -log sigmoid(s[w_g] - s[l_g] + home_g h)`
  - `+ (1/(2 sigma^2))      ||s - Xz @ beta||^2`
  - `+ (1/(2 sigma_beta^2)) ||beta||^2`
  with `sigma_beta = 1.0` fixed.
- **Inputs:** v4's 67-feature matrix from `prepare_loso_inputs()`,
  per-LOSO-fold standardized (mean/std over `Season != Y`, applied to
  held-out season `Y`). Regular-season games from
  `MRegularSeasonDetailedResults`. Tournament-field pairs only in the
  output (matching plain BT's pair coverage).
- **Solver:** `scipy.optimize.minimize(method='L-BFGS-B', jac=analytic)`,
  `maxiter=500, tol=1e-8`. ~400 parameters per (season, sigma) cell.
- **Total wall time:** 249 seconds for 7 cells x 22 LOSO seasons +
  ~4 minutes of `prepare_loso_inputs()` cold-start, plus ~5 seconds
  for the gate diagnostic.

## Sigma sweep result

`output/diag_hbt_sweep.json`. Same 3-clause gate as plain BT
(`r < 0.60`, `w in [0.30, 0.85]`, `headroom > 0.005`). v4 standalone
LL = 0.4369, acc = 0.805, n_games = 1449.

| sigma | r_resid | ll_hbt | acc_hbt | w_opt | headroom | clause 1 | clause 2 | clause 3 | verdict |
|-------|---------|--------|---------|-------|----------|----------|----------|----------|---------|
| 0.05  | 0.448   | 0.6194 | 0.650   | 1.00  | +0.0000  | PASS     | FAIL     | FAIL     | FAIL    |
| 0.10  | 0.485   | 0.6305 | 0.660   | 1.00  | +0.0000  | PASS     | FAIL     | FAIL     | FAIL    |
| 0.20  | 0.505   | 0.6220 | 0.669   | 0.99  | +0.0000  | PASS     | FAIL     | FAIL     | FAIL    |
| 0.50  | 0.507   | 0.6306 | 0.667   | 0.99  | +0.0000  | PASS     | FAIL     | FAIL     | FAIL    |
| 1.00  | 0.492   | 0.6507 | 0.663   | 1.00  | +0.0000  | PASS     | FAIL     | FAIL     | FAIL    |
| 2.00  | 0.472   | 0.6880 | 0.656   | 1.00  | +0.0000  | PASS     | FAIL     | FAIL     | FAIL    |
| 5.00  | 0.448   | 0.7569 | 0.644   | 1.00  | +0.0000  | PASS     | FAIL     | FAIL     | FAIL    |

Best passing cell: **None**.

## Anchor expectations vs reality

The spec laid out two anchor predictions:

| anchor                       | predicted               | observed                |
|------------------------------|-------------------------|-------------------------|
| sigma -> 0 (`s = Xz @ beta`) | LL approaching v4 ~0.44 | LL ~0.62 (worse)        |
| sigma -> infinity (plain BT) | LL ~0.565 (plain BT)    | LL ~0.76 (much worse)   |

**Both anchors miss.** The standalone-strength curve over sigma is
non-monotonic and bottoms out around sigma=0.05-0.20 at LL ~0.62,
worse than plain BT at every cell tested.

## Why the anchors miss -- the actual lesson

### Anchor 1 (sigma -> 0): different training task

When `sigma -> 0`, `s` is forced to equal `Xz @ beta`, and the
likelihood reduces to a logistic regression with team-level features
on **regular-season game outcomes**. v4, in contrast, is trained on
**tournament-pair feature differences with tournament-game labels**.

These are different learning problems:
- HBT@small_sigma: "predict regular-season W/L from the difference of
  two teams' v4 feature vectors."
- v4: "predict tournament-game outcomes from the difference of two
  teams' v4 feature vectors, trained directly on tournament games."

Even with identical features, the *training distribution* differs:
regular-season games have ~6000/season per league, almost all
in-conference, with massive home-court asymmetry. Tournament games
are ~63/season, all neutral, all between top-68 teams. v4 learns
parameters tuned to that target distribution; HBT learns parameters
tuned to regular-season W/L, which doesn't transfer cleanly to
tournament prediction.

So `sigma=0.05` is **not** a recovery of v4 -- it's a parallel
feature-LR trained on a different target. LR-on-tournament-pairs (the
PR 11 LR experiment) achieved LL 0.498. HBT@sigma=0.05 achieves LL
0.619, much worse, because it's optimizing the regular-season
likelihood instead.

### Anchor 2 (sigma -> infinity): tighter combined regularization

Plain BT used `sklearn.LogisticRegression(C=10, fit_intercept=False)`
with no feature prior, equivalent to `sigma_team^2 = C = 10`,
`sigma_beta = infinity`. HBT@sigma=5.0 has `sigma=5.0` (looser per-team
prior) BUT `sigma_beta=1.0` (tight beta prior). The combined effect
is roughly: `s` gets pulled toward `Xz @ beta`, and `Xz @ beta` is
pinned near zero by the tight beta prior. Net: `s` shrinks toward
zero MORE than in plain BT's purely-on-strengths L2.

Effective per-team regularization at sigma=5, sigma_beta=1.0 on a
67-feature standardized X is somewhere between plain BT's effective
sigma~3.16 and a much tighter shrinkage, depending on how aggressively
beta absorbs the team-strength signal. In practice the optimizer
finds a beta that explains some of the per-team strength variance,
which means `Xz @ beta` is not zero, but the combined prior still
over-shrinks compared to plain BT. Result: HBT@sigma=5 makes more
homogenized predictions than plain BT and loses calibration on
high-confidence games -- standalone LL goes UP, not down.

If we cared, we could sweep `sigma_beta` to recover plain-BT-like
behavior at the loose end. The HBT result shows priors don't lift
standalone strength on this scoring rule, so further sigma_beta
tuning is unlikely to flip the verdict.

## What this experiment actually shows

- Adding `s ~ Normal(beta . v4_features, sigma^2)` priors to a
  per-season BT model does NOT improve standalone tournament log
  loss across sigma in {0.05, ..., 5.00}. HBT is uniformly worse
  standalone than plain BT.
- Tested on the LL-optimal-blend gate (cheating ideal-weight search
  on tournament games), every cell collapses to `w_opt = 0.99-1.00`,
  meaning the LL-optimal blend is essentially v4 alone.
- The residual-correlation finding is positive: HBT errors are LESS
  correlated with v4's than plain BT's were (`r ~0.45-0.51` vs plain
  BT's 0.577). The diversity is real; the standalone weakness
  prevents log-loss-blending from exploiting it.

## What this experiment does NOT show

- **It does not show that no BT-class model can help v4.** The gate
  measures log-loss-blend headroom on tournament games. The
  production metric is bracket points, where correctly-predicted
  upsets are scored at heavy multipliers. A weak-but-diverse
  stage-1 that flips a few v4 picks toward true upsets could lift
  bracket points without lifting log-loss-blend headroom. Plain
  BT's `r=0.577` represents real diversity that we never tested
  against the right metric. v9-C's weight sweep (PR 9) already
  showed the active ingredient was `W_MISS` (residual weighting),
  consistent with "the BT residual carries some signal v4 doesn't."
  Re-testing plain BT against bracket points -- skipping the LL
  gate entirely -- is now active queue item #2.
- **It does not show that the training data is exhausted.** The
  user finished 2159 / 3462 in a recent Kaggle Mania, so 2/3 of
  entries extract more value from this same data than v4 does.
  Whatever the bottleneck is, it is not "nothing more can be
  squeezed out of regular-season + tournament outcomes." The
  bottleneck is far more likely to be inside v4 itself: feature
  engineering, calibration, hyperparameter tuning, or model-class
  choices that haven't been audited against the leaderboard.
- **It does not falsify external data.** External signals (538
  tournament forecasts, Vegas prop-bet implied probs, roster
  injury data) can still help -- not as the *only* lever, just as
  one of several plausible levers we haven't tried.

## Updated diversity-strength snapshot (within the LL-blend gate)

| candidate              | trained on        | residual r vs v4 | standalone LL | optimal w | gate verdict |
|------------------------|-------------------|------------------|---------------|-----------|--------------|
| LR (PR 11)             | tournament pairs  | 0.77             | 0.498         | 0.93      | NO-GO        |
| plain BT (PR 12)       | regular-season W/L| 0.577            | 0.565         | 0.98      | NO-GO (LL-only) |
| HBT (this branch)      | regular-season W/L + v4-prior | 0.448 (best) | 0.619 (best) | 1.00      | NO-GO        |
| BT-as-feature (PR 13)  | -- (feature)      | n/a              | n/a           | n/a       | NO-GO        |

Within the LL-optimal-blend gate, four ensemble corners have failed.
That is the genuine finding. Whether bracket points tells the same
story for plain BT specifically is an open question scheduled as
the next experiment.

## Implications for active queue

Active queue item #1 (this experiment) is **falsified on the
LL-blend gate, narrowly**. New ordering, prioritizing audits over
more architecture exploration:

1. **Localize v4's gap.** Diff v4 game-by-game against an external
   benchmark (top-quartile public Kaggle entry, 538 tournament
   forecast, or Vegas-line implied probabilities). Bucket by round,
   seed pair, higher-vs-lower-seed, and confidence bin to find
   where v4 specifically loses. Without this, we keep testing
   ensemble add-ons without knowing what gap they would need to
   close.
2. **Re-test plain BT against bracket points.** Skip the LL gate.
   Blend `pairwise_v4` + `pairwise_bt` at uniform weight (or a
   small grid), run v9-C on the result, score 22-season
   bracket-points head-to-head. Cheap (~1 hour). Reuses existing
   `pairwise_bt.csv`. The LL gate may be filtering on the wrong
   metric.
3. **External rankings / external data as features** (was #2 in the
   prior queue). 538 forecast, Vegas prop-bet implied probability,
   roster injury data. Stays on the queue but no longer the sole
   non-architectural lever.
4. **Small NN (MLP) as stage-1.** Lower priority -- same
   ensemble-with-v4 question we've now stress-tested on four
   corners.
5. **Full Bayesian BT with strength + variance per team.** Lower
   priority -- the HBT result suggests prior structure doesn't
   move standalone strength on this metric.
6. **Roster-level returning-experience.** External data; promote
   alongside #3 if sourcing is easy.

## Files of record

```
src/features/hierarchical_bt.py       -- MAP solver (L-BFGS, analytic gradient)
src/train_hbt_stage1.py               -- per-(sigma, season) trainer
src/diagnose_hbt_vs_v4.py             -- per-cell gate runner

output/pairwise_hbt_sigma_<S>.csv x 7 -- per-cell pairwise outputs (force-added)
output/diag_hbt_sweep.json            -- per-cell metrics + verdicts
output/train_hbt_log.txt              -- per-season fit log
output/diag_hbt_log.txt               -- gate stdout

tests/test_features/test_hierarchical_bt.py  --  5 unit tests
tests/test_train_hbt_stage1.py               --  7 unit tests
tests/test_diagnose_hbt_vs_v4.py             --  8 unit tests
```

The 3-clause gate's thresholds (`GATE_R_MAX=0.60`, `GATE_W_LOW=0.30`,
`GATE_W_HIGH=0.85`, `GATE_HEADROOM_MIN=0.005`) are reused verbatim
from `src/diagnose_bt_vs_v4.py` (a regression-guard test enforces
this), so any future stage-1 candidate with a pairwise CSV in the
canonical schema can be scored cell-by-cell against the same
falsification bar.

## Time budget

Diagnostic gate paid for itself: total compute = ~4 min trainer +
~5 sec gate, vs ~3 hours for a v9-C + 22-season backtest. Saved
~3 hours of compute by not running the full backtest, AND mapped a
new corner of the diversity-strength frontier.
