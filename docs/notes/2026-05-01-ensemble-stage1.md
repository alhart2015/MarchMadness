# Stage-1 Ensemble (XGBoost + Logistic Regression) -- Findings

**Date:** 2026-05-01
**Branch:** feat/ensemble-stage1
**Verdict:** **NO-GO** -- v4 alone stays as stage-1.
**Spec:** `docs/superpowers/specs/2026-05-01-ensemble-stage1-design.md`
**Plan:** `docs/superpowers/plans/2026-05-01-ensemble-stage1.md`

## TL;DR

A simple-average ensemble of v4 (XGBoost) + new logistic regression
(both trained on the same 67-feature matrix) loses to v4 alone by
**-105 bracket points** over 22 LOSO seasons after the v9-C stage-2
correction. Ensemble wins only 7/22 seasons (vs v4-baseline 13/22,
2 ties). The TODO active queue's noted risk -- "if both see the same
features and converge to the same ceiling, errors are highly
correlated and ensembling won't help" -- is exactly what happened.
Model-class diversity at identical inputs does not buy bracket
points at this scale.

## Setup recap

Production today: `v4 (XGBoost stage 1) -> v9-C (XGBoost upset-aware
stage 2)`.

Experiment: replace v4 with `0.5 * v4 + 0.5 * LR` at stage 1, hold
v9-C config constant (PR 9 winning weights: W_UPSET=1.25, W_MISS=0.0,
feature_set='v9c').

LR setup (`src/train_lr_stage1.py`): scikit-learn `LogisticRegression`
with L2 penalty, `lbfgs` solver, max_iter=2000. Per fold:
`StandardScaler` fit on train rows, `GridSearchCV` over
C in {0.01, 0.1, 1, 10, 100} (5-fold inner CV, neg log loss),
`CalibratedClassifierCV(method='sigmoid', cv=5)` Platt-calibrates
the chosen LR. All supervised fits see only train-fold rows; LOSO
discipline preserved end-to-end.

Pair coverage exact: 48,465 unique pairs in both `pairwise_v4.csv`
(after dedup of default + tuned passes) and `pairwise_lr.csv`.

## Anchor checks (mandatory; both passed)

- `python src/ensemble_stage1.py --weights 1.0,0.0` byte-equals
  dedup'd `pairwise_v4.csv` -- max prob diff `0.0e+00` over all
  48,465 pairs.
- `python src/ensemble_stage1.py --weights 0.0,1.0` byte-equals
  `pairwise_lr.csv` -- max prob diff `0.0e+00`.

These confirm join logic, schema, sort order, and weighting math
are all correct before any head-to-head numbers are trusted.

## Stage-1-only LOSO head-to-head

Raw stage-1 quality, before v9-C correction:

| metric                 | v4     | ensemble | delta   |
|------------------------|--------|----------|---------|
| weighted-mean log loss | 0.4369 | 0.4513   | +0.0144 |
| weighted-mean accuracy | 0.805  | 0.801    | -0.004  |

The ensemble is **slightly worse** on log loss and accuracy. This
is consistent with LR being a weaker base learner on this 67-feature
space (LR's standalone weighted-mean log loss was 0.5004 vs v4's
0.4369), and the average inheriting some of LR's worse calibration.
For ensembling to *improve* metrics here, LR's errors would need
to be uncorrelated enough with v4's that averaging cancels variance.
That is not what we see -- the patterns are correlated, and the
average just imports LR's noise into v4's already-good predictions.

## Stage-1 + v9-C head-to-head (decisive)

Same 22 LOSO seasons, but with v9-C correcting the stage-1 output:

```
season     v4_base    ensemble    delta
  2003        90.0        82.0     -8.0
  2004       106.0       105.0     -1.0
  2005       112.0       111.0     -1.0
  2006       108.0        87.0    -21.0
  2007       104.0       106.0     +2.0
  2008       147.0       147.0     +0.0
  2009       115.0       112.0     -3.0
  2010       123.0       110.0    -13.0
  2011        70.0        59.0    -11.0
  2012       138.0       138.0     +0.0
  2013       167.0       161.0     -6.0
  2014       159.0       144.0    -15.0
  2015       174.0       176.0     +2.0
  2016       123.0       127.0     +4.0
  2017       135.0       129.0     -6.0
  2018       155.0       158.0     +3.0
  2019       116.0       119.0     +3.0
  2021        71.0        78.0     +7.0
  2022       101.0        85.0    -16.0
  2023       156.0       135.0    -21.0
  2024       102.0        93.0     -9.0
  2025       141.0       146.0     +5.0
--------------------------------------
 TOTAL      2713.0      2608.0   -105.0
```

ensemble W/L/T vs v4-baseline: **7 / 13 / 2**.

Total delta: **-105.0 bracket points** (NO-GO band per spec:
clear win >= +25, marginal +10 to +25, no-go < +10).

The losing seasons are not isolated noise: 2006 (-21), 2010 (-13),
2014 (-15), 2022 (-16), 2023 (-21) all swing by 10-21 points
against the ensemble. The ensemble's wins are smaller in magnitude
(max +7 in 2021).

## Verdict

NO-GO. The ensemble loses by 105 bracket points -- well below the
+10 marginal floor and far below the +25 clear-win bar.

This falsifies the diversity-at-identical-features hypothesis at
this scale: a logistic regression's inductive bias does not produce
errors uncorrelated enough with v4's to make averaging worthwhile,
and LR is a weaker base learner besides. v4 stays as stage-1.

## Recommendation

- v4 + v9-C remains the production stack. No code changes outside
  this branch are needed; the new modules (`train_lr_stage1.py`,
  `ensemble_stage1.py`, `eval_stage1.py`, `run_v9c_on_stage1.py`)
  stay on the branch as the experiment record. The data files
  (`pairwise_lr.csv`, `pairwise_ensemble.csv`,
  `pairwise_v9c_v4_baseline.csv`, `pairwise_v9c_ensemble.csv`)
  document the head-to-head reproducibly.
- Promote NN and Bayesian (deferred follow-ups in TODO.md) to the
  active queue with the explicit caveat that the same correlated-
  error risk applies. A NN on the same 67-feature input is unlikely
  to disagree with XGB much more than LR did. A Bayesian /
  hierarchical Bradley-Terry model has the most structurally
  different inductive bias on the queue and is the better next
  swing if we keep pulling on the model-class lever.
- A fundamentally different angle worth flagging: feature-view
  diversity rather than model-class diversity (e.g., one model on
  KenPom-only features, one on Vegas-only, average). The current
  experiment intentionally held inputs identical to isolate the
  model-class question; pivoting to deliberate input diversity is a
  separate experiment with its own spec.

## Files of record

```
src/train_lr_stage1.py        -- LR stage-1 trainer
src/ensemble_stage1.py        -- CSV averaging utility
src/eval_stage1.py            -- per-season LOSO log loss/accuracy
src/run_v9c_on_stage1.py      -- v9-C wrapper for any stage-1 CSV

output/pairwise_lr.csv               -- LR LOSO output (22 seasons)
output/pairwise_ensemble.csv         -- 0.5*v4 + 0.5*lr
output/pairwise_v9c_v4_baseline.csv  -- v9-C on v4 (head-to-head A)
output/pairwise_v9c_ensemble.csv     -- v9-C on ensemble (head-to-head B)

tests/test_prepare_loso_inputs.py
tests/test_ensemble_stage1.py
tests/test_train_lr_stage1.py
tests/test_eval_stage1.py
tests/test_run_v9c_on_stage1.py
```

The refactor of `enhanced_model_v3.py` to expose `prepare_loso_inputs()`
is reusable -- any future model-class swap that wants the
byte-identical v4 feature matrix should call into it the way
`train_lr_stage1.py` does.
