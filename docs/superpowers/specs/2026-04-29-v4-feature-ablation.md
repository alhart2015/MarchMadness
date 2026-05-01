# v4 Feature Ablation on 2026 High-Confidence Misses

## Problem Statement

v4 (v3 features + coach career features) is our best model on the 22-season
LOSO backtest, but on 2026 specifically it shipped four costly high-confidence
busts:

| Team        | Round eliminated | v4 P(advance past) | Outcome     |
|-------------|------------------|--------------------|-------------|
| Vanderbilt  | R32              | 0.79               | Lost in R32 |
| Iowa St.    | S16              | 0.82               | Lost in S16 |
| Texas Tech  | R32              | 0.86               | Lost in R32 |
| Duke        | E8               | 0.88               | Lost in E8  |

These four picks alone account for the bulk of v4's bracket-point gap to a
better-calibrated bracket on 2026. v4 still beats v1 on 20 of 22 LOSO seasons,
so we are not aiming to replace v4 -- we want to know whether a single feature
(or feature group) added in v3/v4 is structurally over-confident on deep-round
favorites and thus a candidate for tuning, replacement, or removal.

The TODO entry that motivated this work was written when v3 was the latest
model. The correct ablation target is v4 (current production) and the
candidate set must include the coach features added in v4.

## Goals

- Identify the feature(s) that drive v4's 2026 over-confidence on the four
  busts above.
- Quantify the effect per ablation: change in 2026 pairwise probability for
  each bust team and change in 22-season LOSO log loss / bracket-points.
- Produce a written diagnostic (`docs/superpowers/specs/` follow-up or
  `docs/notes/`) with the verdict.
- Stay diagnostic. We are not shipping a v9 in this work; any new model
  variant is a separate spec informed by these results.

## Non-Goals

- Building an ensemble or upset-detection sub-model (separate TODO items).
- Re-tuning hyperparameters per ablation. Use v4's current Optuna-tuned params
  for every retrain to keep the comparison clean. (Hyperparameter sensitivity
  is a separate question.)
- Adding new features. This is subtractive only.
- Replacing v4 if a feature looks suspect on 2026 -- one season of evidence
  is not enough; the LOSO-wide effect must agree.

## Approach

### Ablation method

**Drop-and-retrain**, not zero-out-at-predict. Drop the column from the
feature matrix and re-train v4 fresh under the existing LOSO + Platt pipeline.
Mean/zero imputation at predict time tells you the marginal contribution of
the feature to a single trained model's score; it does not tell you whether
the model would have learned different patterns in the rest of the feature
set without it. We want the latter.

The same Optuna-tuned XGBoost params, line-blending settings, and supplemental
training weights are reused across all ablations -- only the feature set
changes. This isolates the feature contribution.

### Two-pass design: groups first, then individuals

A flat per-feature sweep is 14 retrains (7 v3 + 6 coach + 1 group-level).
Cheaper and more interpretable to do it in two passes:

**Pass 1: feature groups (4 retrains).**

| Group              | Features dropped                                                                |
|--------------------|---------------------------------------------------------------------------------|
| `late_season`      | `late_adj_oe`, `late_adj_de`, `late_adj_em`, `late_sos`                         |
| `trajectory`       | `efficiency_trend`, `margin_trend`                                              |
| `conf_tourney`     | `conf_tourney_wins`, `conf_tourney_champ` (and `conf_tourney_margin` if present)|
| `vegas_trend`      | `vegas_late_spread_delta`                                                       |
| `coach`            | `coach_career_games`, `coach_career_wins`, `coach_career_winpct`, `coach_career_f4_apps`, `coach_career_champs`, `coach_career_seasons` |

That is 5 groups, 5 retrains. For each, log the new 2026 P(advance) for the
four bust teams and the change in 22-season LOSO mean log loss.

**Pass 2: drill-down on the suspicious group(s).**

If group G's removal moves any bust's 2026 P(advance) by >= 5 percentage points
*and* does not degrade 22-season LOSO log loss by more than +0.005 (~1%), drill
down inside G with one retrain per individual feature in that group.

If multiple groups look suspicious, drill down into each. If no single group
moves the four busts meaningfully, the answer is "no single feature
dominates" -- the over-confidence is emergent, and the right next step is one
of the architectural items in TODO.md (ensemble, upset model), not a feature
fix. State that clearly in the writeup.

### Threshold rationale

- **5 percentage points on 2026 P(advance)**: a single chalk-pick flip needs
  the model to drop a team below 0.5 against its actual loser. Vanderbilt at
  0.79 has plenty of headroom; Texas Tech at 0.86 needs ~36pp. We're not
  expecting any single feature to flip a chalk pick on its own, but a 5pp
  shift is large enough to indicate the feature is doing real work on that
  team's projection.
- **+0.005 LOSO log loss tolerance** (~1% relative degradation): below that,
  the feature is providing real signal across 22 seasons and removing it
  costs more than it gains. A feature whose removal improves 2026 by 5pp but
  costs 0.01 log loss across 22 seasons is not a fix -- it is overfitting
  to 2026.

## Deliverables

1. **`src/ablate_v4.py`**: driver script that runs the v4 pipeline N times
   with the configured feature drop list, captures per-run pairwise
   predictions for the four bust teams (and the full 2026 grid for
   reference), and writes a results table.

2. **`output/ablation_v4_results.csv`**: one row per (ablation, bust_team)
   with columns `ablation`, `team`, `round_eliminated`, `p_advance_v4`,
   `p_advance_ablated`, `delta_pp`, `loso_logloss_v4`,
   `loso_logloss_ablated`, `loso_logloss_delta`, `bracket_pts_22yr_v4`,
   `bracket_pts_22yr_ablated`.

3. **Writeup**: short markdown note (`docs/notes/2026-04-29-ablation-v4.md`
   or appended to this spec) with the verdict, the drill-down results, and a
   recommendation for the next active-queue item.

## Implementation Notes

- v4 lives in `enhanced_model_v3.py` (not a separate file). The script
  imports `compute_coach_features` and merges them onto the per-team feature
  rows before building matchup pairs (`enhanced_model_v3.py:699-708`).
  Ablation must drop columns *after* merging but *before* matchup-pair
  construction, so the difference-feature set in `src/models/matchup.py`
  reflects the drop.
- `src/models/matchup.py` builds symmetric A-vs-B / B-vs-A pairs as feature
  *differences*. A column dropped from the per-team frame is automatically
  absent from the pair frame.
- LOSO retrain must reuse the existing v4 Platt-calibration setup. Do not
  silently disable calibration to save time -- a uncalibrated comparison
  changes the bracket-points metric.
- Optuna is *not* re-run per ablation. Reuse v4's existing best params.
- Vegas line blending (R64 post-processing) is applied identically across
  ablations. It is not a feature, so dropping a feature does not change it,
  and we want each comparison to reflect the full pipeline as shipped.
- Run the 22-season LOSO backtest for each ablation -- the 2026-only number
  is suggestive but the cross-season number is the gate.
- Watch runtime: a single LOSO backtest is the slow step. Estimate one full
  pipeline run, multiply by 5 (Pass 1) + however many features the
  drill-down touches. If Pass 1 alone is >4 hours, parallelize across
  ablations on independent processes (each ablation is fully independent;
  they share no state).

## Evaluation / Acceptance

This spec is satisfied when the writeup answers, for each of the four bust
teams: "removing feature/group X changes v4's 2026 P(advance) by Y pp at a
22-season LOSO log-loss cost of Z." A clean answer is one of:

- **One feature/group is the culprit** (large 2026 effect, small LOSO cost):
  recommend tuning or removal in a follow-up spec.
- **Multiple features share blame**: recommend a sensitivity or interaction
  analysis in a follow-up.
- **No single feature is the culprit**: state explicitly that the next step
  is architectural (ensemble, upset model) rather than feature-side.

Each of those is a useful answer. The failure mode to avoid is shipping a
result that says "feature X looks bad on 2026" without checking the
22-season cost -- that is how you delete a feature that is helping you.
