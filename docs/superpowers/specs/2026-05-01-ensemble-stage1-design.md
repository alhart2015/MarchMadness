# Stage-1 Ensemble (XGBoost + Logistic Regression) -- Design

**Date:** 2026-05-01
**Branch:** feat/ensemble-stage1
**Predecessors:**
- v9-C production swap: `docs/superpowers/specs/2026-05-01-v9c-production-swap-design.md`
- v9-C feature-stripped: `docs/superpowers/specs/2026-05-01-v9c-feature-stripped-design.md`
- TODO active queue item #1 (this work).

## Motivation

Production today is `v4 (XGBoost stage 1) -> v9-C (XGBoost upset-aware
stage 2)`. v9-C lifted the LOSO bracket-points score by +43 vs v8 by
working on stage-2's residual side; the leverage on stage-1 has not
been touched. The TODO active queue's #1 is to test whether ensembling
stage-1 across model classes captures partially-uncorrelated error
patterns and lifts bracket points further.

The 2026 high-confidence busts (Vanderbilt, Iowa St., Texas Tech, Duke)
were a recurring failure mode. The v4 ablation
(`docs/notes/2026-04-30-ablation-v4-findings.md`) ruled out feature-side
fixes; v9-C absorbed some of the residual via meta-learning. The
remaining hypothesis is that XGBoost's inductive bias (recursive
partitioning, axis-aligned splits) systematically biases certain
matchups in ways a different model class would not. A logistic
regression sees the same features but assumes additive linearity and
will get different things wrong. If those error patterns are even
partially uncorrelated, averaging the two predictions reduces variance
in the corrected probability and should lift bracket points.

The risk, called out in the TODO entry: if both models see the same
features and converge to the same ~80% R64 / ~50-60% deep-round
ceiling, the errors are highly correlated and the average is no
better than v4 alone. This spec is built so that outcome is read
cleanly off a single LOSO backtest.

## Scope

**In scope.**

- New file `src/train_lr_stage1.py`. Mirrors v4's 22-season LOSO
  shape: for each test season Y, train a logistic regression on every
  other season's matchup pairs, predict Y's tournament games, and
  append `(season, team_a, team_b, p_a_wins)` rows to
  `output/pairwise_lr.csv`. Same schema as `output/pairwise_v4.csv`.
- New file `src/ensemble_stage1.py`. Joins `pairwise_v4.csv` and
  `pairwise_lr.csv` on `(season, team_a, team_b)`, computes
  `p_ensemble = 0.5 * p_v4 + 0.5 * p_lr`, writes
  `output/pairwise_ensemble.csv` (same schema). No model -- pure
  averaging.
- One-line config update to `src/train_upset_model.py` and
  `src/predict_2026_v9c.py`: a `STAGE1_PAIRWISE` constant that
  defaults to `"output/pairwise_v4.csv"` (legacy) but can be flipped
  to `"output/pairwise_ensemble.csv"`. The default stays v4 so the
  legacy v9-C reproducibility is preserved; the swap is a deliberate
  flip in a follow-up commit if the experiment wins.
- Re-run v9-C twice -- once on `pairwise_v4.csv` (baseline) and once
  on `pairwise_ensemble.csv` -- to produce two named LOSO outputs
  for the head-to-head bracket-points comparison. v9-C reuses PR 9's
  winning weights (W_UPSET=1.25, W_MISS=0.0, feature_set='v9c') --
  no re-tuning here. Add an `--out` arg (or equivalent) to
  `src/train_upset_model.py` so each run writes to a distinct file
  (e.g., `output/pairwise_v9c_v4_baseline.csv` and
  `output/pairwise_v9c_ensemble.csv`) instead of always
  overwriting `output/pairwise_v9.csv`. Today
  `train_upset_model.py` hardcodes the output path
  (`output/pairwise_v9.csv`) -- the param is the smallest change
  needed to keep both runs side-by-side.
- Unit tests:
  - `ensemble_stage1.py` averaging math (mock CSVs, assert output).
  - Anchor: averaging weight `(1.0, 0.0)` reproduces `pairwise_v4.csv`
    row-for-row across all 22 seasons. This is the leakage-and-correctness
    smoke that catches join bugs and dtype regressions.
  - `train_lr_stage1.py` end-to-end on a 2-season subset writes a
    valid CSV with the expected schema.
- Findings note in `docs/notes/2026-05-01-ensemble-stage1.md`
  with the full LOSO comparison table (per-season log loss, accuracy,
  bracket points), the verdict band, and the recommendation.

**Out of scope.**

- Adding a neural net or Bayesian model as a third ensemble member.
  Both are deliberately deferred per the TODO update -- revisit if
  XGB+LR shows real signal.
- Weighted averaging or stacking. Locked simple average. If A wins
  and a follow-up wants to tune the weight, that becomes its own
  experiment with proper held-out meta-fold discipline.
- Feature-set diversity (LR sees a different feature view than XGB).
  The experiment isolates *model-class* diversity given identical
  inputs. Feature-view diversity is a separate hypothesis.
- Re-tuning v9-C against the ensemble. Reuse PR 9's winning cell. If
  the ensemble lifts bracket points, a small v9-C re-sweep on top of
  it is a follow-up.
- Live bracket integration (`src/generate_bracket_real.py`). The live
  generator is pure-v4 today and would consume the ensemble through
  a separate wiring change, called out in the v9-C swap spec as a
  follow-up.
- Modifying `enhanced_model_v3.py` (v4). v4 stays untouched and
  reproducible side-by-side. The new LR trainer is a parallel module
  that reuses v4's matchup-pair builder
  (`src.models.matchup.build_weighted_matchup_data`) and feature
  matrix; that's the only shared infrastructure.

## Approach

### Architecture

```
v4 (enhanced_model_v3.py)  -> output/pairwise_v4.csv         (existing)
LR (train_lr_stage1.py)    -> output/pairwise_lr.csv         (new)
                                |
                                v
                  ensemble_stage1.py  -> output/pairwise_ensemble.csv (new)
                                |
                                v
                  v9-C (train_upset_model.py with STAGE1_PAIRWISE flipped)
                                |
                                v
                  output/pairwise_v9c_ensemble.csv  (new -- the comparison output)
                                |
                                v
              22-season bracket-points head-to-head: ensemble + v9-C
                              vs current production (v4 + v9-C)
```

The ensemble's identity at every stage is a CSV file with the same
schema as `pairwise_v4.csv`. That choice keeps `train_upset_model.py`
and `predict_2026_v9c.py` completely agnostic to which stage-1 they're
correcting -- they just read a CSV. It also means the existing v9-C
LOSO smoke test (`tests/test_predict_2026_v9c.py`) gets parameterized
to also run against the ensemble file.

### `src/train_lr_stage1.py`

Shape mirrors v4's training loop. The matchup-pair construction is
shared (`build_weighted_matchup_data` from `src/models/matchup.py`),
so XGB and LR see byte-identical training rows in every fold. The
only differences from v4:

- **Model:** `sklearn.linear_model.LogisticRegression` with `penalty='l2'`,
  `solver='lbfgs'`, `max_iter=2000`. L2 strength `C` chosen via inner
  CV on the training folds (e.g., 5 values: 0.01, 0.1, 1, 10, 100;
  pick the best by training-fold log loss).
- **Standardization:** `sklearn.preprocessing.StandardScaler` fit on
  the training folds and applied to the test fold. LR needs feature
  scales comparable; XGB doesn't, which is part of why the two see
  different optima. The scaler is fit *per LOSO fold* -- never on the
  test season.
- **Training-data weighting:** identical to v4 (tournament rows full
  weight + supplemental late-season regular-season rows reduced
  weight, per `build_weighted_matchup_data`'s existing behavior). No
  deviation. The point is to test "model class given the same data,"
  not "model class given a different data setup."
- **Calibration:** Platt scaling on a per-fold holdout from the
  training data, mirroring how v4 calibrates. Implemented with
  `sklearn.calibration.CalibratedClassifierCV(method='sigmoid', cv=5)`
  wrapped around the inner-CV-best LR. Saves us hand-rolling the Platt
  fit.
- **Output:** `(season, team_a, team_b, p_a_wins)` rows appended to
  `output/pairwise_lr.csv`. Pair canonicalization same as v4
  (`team_a < team_b`).

LOSO discipline is enforced the same way v4 does: the test season is
held out of training rows, scaler-fitting, inner-CV `C` selection, and
Platt calibration. Only the trained-and-calibrated model touches the
test season's matchups.

### `src/ensemble_stage1.py`

Tiny module. Reads both pairwise CSVs, joins on
`(season, team_a, team_b)`, asserts the join is one-to-one and covers
all rows, computes the simple average, writes the output. Includes a
`--weights` CLI arg defaulting to `0.5,0.5` so the anchor test
`(1.0, 0.0)` is a one-line invocation that reproduces `pairwise_v4.csv`.

### Stage-2 update

Two narrow parameterizations to `src/train_upset_model.py` and
`src/predict_2026_v9c.py`:

- `train_upset_model.py`: `load_per_game_data_with_upset` already
  takes `pairwise_csv` as a positional arg (no signature change), but
  the script's `__main__` hardcodes both the input path
  (`output/pairwise_v4.csv`) and the output path
  (`output/pairwise_v9.csv`). Add CLI args for both
  (`--pairwise-in`, `--pairwise-out`) defaulting to the existing
  values so legacy invocations are unchanged.
- `predict_2026_v9c.py`: extract the v4-pairwise path
  (`output/pairwise_probs_v4.json`) into a `STAGE1_PAIRWISE_JSON`
  constant. For the experiment, the apply-time output stays on a
  versioned snapshot (e.g., `output/pairwise_probs_ensemble_2026.json`)
  and does NOT overwrite the canonical
  `output/pairwise_probs.json` until the experiment wins and a
  separate swap commit lands. This protects the production analysis
  scripts from picking up an unverified change mid-experiment.

Note: the apply-time predict for the ensemble's 2026 numbers requires
a 2026 ensemble pairwise JSON. That's produced by the same averaging
logic in `ensemble_stage1.py` extended to also handle the JSON form,
or by a small sibling script. This spec keeps the LOSO comparison
(over CSVs) and the apply-time 2026 prediction (over JSONs) on the
same averaging logic so the math is shared, not duplicated.

### Eval methodology

22-season LOSO (2003-2025, excluding 2020 implicitly via missing data),
matching every prior backtest in this codebase. Two head-to-heads,
both reported per-season and aggregated:

1. **Stage-1 only:** `pairwise_ensemble.csv` vs `pairwise_v4.csv`.
   Metrics: weighted-mean LOSO log loss across tournament games,
   accuracy. This isolates "did the ensemble improve raw stage-1
   probability quality?" If both numbers are flat the diversity
   hypothesis is already falsified at this stage.
2. **Stage-1 + v9-C:** `pairwise_v9c_ensemble.csv` vs
   `pairwise_v9c_v4_baseline.csv` (both produced fresh in this
   experiment via the parameterized `--out`-aware
   `train_upset_model.py`) on **bracket points** across 22 seasons
   (the same metric that promoted v9-C). This is the decisive
   head-to-head -- the production stack's identity is the
   stage-1+stage-2 chain, and bracket points is the metric the swap
   would be evaluated on. Both runs use identical v9-C config so the
   only varying input is which stage-1 they correct.

### Success thresholds (verdict bands)

Mirroring the v9 spec's bands so cross-experiment comparison is
apples-to-apples:

- **>= +25 bracket points** over the 22-season backtest: clear win.
  Recommend swapping the production stage-1 from `pairwise_v4.csv` to
  `pairwise_ensemble.csv`. A separate small commit flips
  `STAGE1_PAIRWISE` in the production script, regenerates
  `pairwise_probs.json`, and updates TODO. This is the v9-C-style
  swap path.
- **+10 to +25 bracket points:** marginal. Document as a candidate
  in `docs/notes/`, but don't swap. Neutral outcome that deserves a
  note for future iterations.
- **< +10 bracket points** (or worse): no-go. v4 alone stays as
  stage-1. Falsifies the "model-class diversity at the same feature
  view" hypothesis at this scale. The NN and Bayesian follow-ups
  become the next swing.

### Anchor sanity check (mandatory)

Before reporting any numbers: run
`python src/ensemble_stage1.py --weights 1.0,0.0` and confirm the
output equals `pairwise_v4.csv` row-for-row across all 22 seasons.
Then run with `--weights 0.0,1.0` and confirm the output equals
`pairwise_lr.csv` row-for-row. Both must pass before the head-to-head
numbers are trusted. This catches join bugs, schema drift, missing
rows, and dtype regressions in one pass.

## Testing

- `tests/test_ensemble_stage1.py`:
  - Unit: averaging math on a small synthetic two-row pair.
  - Unit: anchor reproduction on a small synthetic input.
  - Unit: schema check on output CSV (correct columns, dtypes,
    no NaNs).
- `tests/test_train_lr_stage1.py`:
  - End-to-end smoke on a 2-season subset. Asserts the output CSV
    has the expected schema and row count.
- Existing `tests/test_predict_2026_v9c.py` parameterized to also
  run against the ensemble pairwise file.
- `pytest -v` must pass at the repo root before merging.

## Risks and mitigations

- **Risk: LR underperforms standalone (expected) and the average is
  worse than v4.** Mitigation: this is the falsification path the
  spec is built around. If the answer is "model-class diversity at
  identical feature view doesn't help," that's a clean negative
  result and the no-go band catches it.
- **Risk: LR's calibration is poor enough that averaging hurts even
  if the underlying signal is uncorrelated.** Mitigation: Platt
  calibration per fold is the standard remedy. Worth checking the
  per-fold reliability diagrams in the findings note.
- **Risk: feature-scaling leakage from fitting `StandardScaler` on
  the wrong rows.** Mitigation: scaler is fit on train-fold rows
  only and applied to test-fold rows. Anchor + LOSO smoke test
  catches gross violations; reviewer should still scrutinize the
  loop.
- **Risk: v9-C's tuned weights (W_UPSET=1.25, W_MISS=0.0,
  feature_set='v9c') are calibrated against v4's residual pattern;
  the ensemble's residual pattern may be different.** Mitigation:
  not directly mitigated in this spec -- we explicitly reuse the
  same v9-C config to keep this experiment scoped. If the ensemble
  beats v4 alone but loses the head-to-head against
  `v4 + v9-C`, that's signal for a v9-C re-sweep on top of the
  ensemble in a follow-up, not a bug in this experiment.

## Follow-ups (deliberately deferred)

- Add a neural net (MLP) as a third ensemble member.
- Add a Bayesian / hierarchical Bradley-Terry model with explicit
  team-strength variance.
- Tune the average's weight (or stack) -- requires a held-out
  meta-fold so the LOSO measurement isn't reused for tuning.
- Re-sweep v9-C's W_UPSET/W_MISS on top of the ensemble's residuals.
- Wire the ensemble (or the corrected output) into
  `src/generate_bracket_real.py` so the live bracket pipeline picks
  it up.

## File-touch summary

```
new   docs/superpowers/specs/2026-05-01-ensemble-stage1-design.md  (this file)
new   docs/superpowers/plans/2026-05-01-ensemble-stage1.md         (next step)
new   src/train_lr_stage1.py
new   src/ensemble_stage1.py
new   tests/test_ensemble_stage1.py
new   tests/test_train_lr_stage1.py
edit  src/train_upset_model.py        (add STAGE1_PAIRWISE constant)
edit  src/predict_2026_v9c.py         (add STAGE1_PAIRWISE constant)
edit  tests/test_predict_2026_v9c.py  (parameterize on stage-1 path)
new   docs/notes/2026-05-01-ensemble-stage1.md  (findings, written after run)
edit  TODO.md                         (mark Active queue #1 complete or
                                       update verdict)
```
