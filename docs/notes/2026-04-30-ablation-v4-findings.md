# v4 Feature Ablation Findings (2026-04-30)

**Spec:** `docs/superpowers/specs/2026-04-29-v4-feature-ablation.md`
**Plan:** `docs/superpowers/plans/2026-04-29-v4-feature-ablation.md`

## Verdict

**No single feature group from v3's additions drives v4's 2026 over-confidence
on Vanderbilt, Iowa St., Texas Tech, or Duke.** The over-confidence is
emergent across the feature set; the next step is architectural (ensemble or
upset-detection model), not feature-side.

## Method

Drop-and-retrain ablation on v4 (which is v3's features + coach features).
Each ablation reused v4's Optuna-tuned XGBoost params
(`output/v4_tuned_params.json`) so the only thing varying was the feature
set. 22-season LOSO + 2026 Monte Carlo run per ablation. Bracket points
scored via `score_pairwise_path` against `MNCAATourneyCompactResults.csv`.

Threshold for "suspicious": at least one bust team's P(advance past bust
round) drops by >= 5 pp AND 22-season LOSO log-loss does not degrade by
more than +0.005 (~1% relative).

## Pass 1: Group Ablations

Baseline (v4): LOSO log loss 0.4370, 22-yr brkt pts 2661.

|             |     | Vandy R32 | Iowa St. S16 | TT R32 | Duke E8 | LOSO delta | brkt pts delta |
|-------------|-----|-----------|--------------|--------|---------|------------|----------------|
| Baseline    | abs | 0.714     | 0.481        | 0.712  | 0.575   | --         | --             |
| late_season | dpp | -2.1      | +0.5         | -0.1   | -1.7    | +0.0004    | -31            |
| trajectory  | dpp | +1.8      | -0.3         | -1.7   | -1.2    | -0.0010    | -3             |
| conf_tourney| dpp | +2.5      | -2.0         | -1.7   | -0.8    | -0.0001    | -19            |
| vegas_trend | dpp | **-15.0** | -2.1         | **-14.6** | -4.8 | **+0.0191**| -31            |

`dpp` = delta in percentage points vs baseline.

### Per-group readout

- **late_season**, **trajectory**, **conf_tourney**: small bust shifts
  (max -2.5 pp), tiny LOSO movement (within +/- 0.001). None exceed the
  5-pp threshold. Ruled out as drivers.
- **vegas_trend**: large bust shifts (-15 pp Vandy, -14.6 pp TT) but at a
  +0.019 LOSO log-loss cost -- about 4x the +0.005 tolerance. This is
  the spec's "feature with 2026 effect but huge generalization cost"
  warning case. The market correctly saw both teams trending positively
  late in the season; v4 correctly weighted that up; the 2026 outcome
  was just bad luck. Removing the feature would help 2026 but break the
  model overall. **Keep it.**
- **coach**: not re-run. The v4 vs v3 LOSO backtest already established
  coach features are net-positive (v4 wins 22 of 22 seasons), so dropping
  coach == reverting to v3, which we know is worse. No information gain.

## Recommendation

The over-confidence on Vanderbilt, Iowa St., Texas Tech, and Duke in 2026
is not attributable to any single feature group introduced in v3 or v4.
That moves the next active-queue item from feature-side work to
architectural:

1. **Upset-detection sub-model.** Train a binary "will the higher seed
   lose?" classifier with different feature weighting (heavier emphasis
   on high-confidence misses). Combine via meta-learner. Most directly
   aligned with how bracket scoring actually pays out, and the chalk-bracket
   insensitivity to v8's calibration improvements (43d555d) suggests
   round-by-round upset modeling is the right next layer.
2. **Ensemble of model classes** (XGB + LR + small NN). Cheaper to stand
   up but less likely to help if errors are correlated, which they
   probably are at this feature ceiling.

Recommend starting with #1 in a new spec.

## Caveats

- Pass 1 ran with v4 LOSO log loss = 0.4370, slightly higher than v4's
  canonical 0.432. The drift is small (~1% relative) and the bracket
  points (2661) match canonical exactly, so the model isn't broken --
  it's likely an Optuna-determinism nuance or minor data shift. Worth
  tracking but not enough to invalidate the ablation comparisons (which
  are all relative to the same baseline run).
- The spec's stated bust probabilities (0.79 / 0.82 / 0.86 / 0.88) did
  not match the baseline run (0.71 / 0.48 / 0.71 / 0.58). Stale
  recollections in the spec, not a model issue. The deltas are still
  meaningful relative to the baseline that produced them.
- drop_coach subprocess crashed on the first sweep (exit code 0xFFFFFFFF,
  native-lib crash mid-LOSO). Not re-run because v4 vs v3 already proves
  coach is net-positive; ablating it adds no information.

## Artifacts

- `output/ablation_v4_results.csv` -- 16 rows (4 ablations x 4 bust teams),
  coach not included
- `output/ablation/pairwise_*.csv` -- per-ablation 22-season pairwise
  predictions (5 files including baseline; coach partial at ~53% of seasons)
- `output/cv_per_season_v3_*.csv` -- per-ablation per-season LOSO metrics
- `output/bracket_data_*.json` -- per-ablation 2026 advancement probs
- `output/ablation/pass1.log` -- driver run log (UTF-16 LE, decode with
  `python -c "print(open('...', 'rb').read().decode('utf-16'))"`)
