# Future Work

## Tried and rejected

- **Quality-wins-vs-tournament-field (v5):** -93 pts vs v4 over 22 LOSO
  seasons. F4 accuracy fell 9pp. Already captured by KenPom/SOS.
- **Matchup-interaction features (v6):** avg = `(A+B)/2` columns
  alongside diffs. +7 pts vs v4 (within noise). Reverted.
- **Round-as-a-feature (v7):** added round (1..6) as a column. -10 pts
  vs v4. CV log loss flat (0.4384 -> 0.4387). Reverted. The model could
  not extract round-conditional signal from the existing features.

## Active queue

1. **Upset-detection sub-model.** Train a binary "will the higher seed
   lose?" classifier with different feature weighting (heavier emphasis
   on high-confidence misses). Combine v4 via meta-learner. Promoted
   to position #1 by the v4 feature-ablation findings (see Done):
   the 2026 over-confidence is not feature-side, so the next leverage
   is round-by-round upset modeling -- which also aligns directly
   with how bracket scoring pays out.
2. **Ensemble of model classes.** XGBoost + logistic regression +
   small neural net averaged (or stacked). The TODO already had this
   under Tier C. The hypothesis: different model classes capture
   partially-uncorrelated error patterns. Risk: if all three see the
   same features and reach the same ~80% R64 / ~50-60% deep-round
   ceiling, the errors are highly correlated and ensembling won't help
   much. Worth trying after #1.
3. **External rankings (538, KenPom-public, BPI as features).** Note:
   we already have BPI, Sagarin, KenPom (POM), Bart Torvik (TRK), RPI
   via Massey ordinals (config.yaml lines 30-36). Truly external would
   be 538's tournament forecast or Vegas prop-bet predictions, which
   need data sourcing outside the Kaggle archive.
4. **Roster-level returning-experience.** Player-level data is not in
   the Kaggle Mania archive; would need an external roster CSV per
   season. Different signal from coach experience.

## Done

- **v4 feature ablation (2026-04-30).** Drop-and-retrain on v4 across
  4 v3 feature groups (late_season, trajectory, conf_tourney,
  vegas_trend); coach skipped because v4 vs v3 LOSO already proves
  it's net-positive. Verdict: no single group drives 2026
  over-confidence on Vanderbilt / Iowa St. / Texas Tech / Duke.
  vegas_trend would help 2026 (-15 pp Vandy, -14.6 pp TT) but at a
  +0.019 LOSO cost (4x tolerance) -- keep it. Recommendation:
  architectural next step (item #1 above). Findings:
  `docs/notes/2026-04-30-ablation-v4-findings.md`. Code in
  `src/ablate_v4.py` and `src/enhanced_model_v3.py` env-var hooks.
- **Two-stage model (v8).** Stage-2 corrector trained on v4's
  out-of-fold pairwise predictions with seed-pair / abs-seed-diff
  context features. +9 bracket pts over 22-season backtest, CV log
  loss 0.437 -> 0.432. Lands in commit 43d555d. 2026 chalk bracket
  is unchanged (corrections too small to flip picks).



## Architecture Rethink (Tier C)
Consider an ensemble approach or second-stage model that adjusts first-stage
probabilities based on matchup context. Ideas:
- Ensemble of XGBoost + logistic regression + neural net for diversity
- Two-stage model: stage 1 produces raw probabilities, stage 2 adjusts them
  using matchup-specific context (seed pairing history, round number,
  conference matchup dynamics)
- Model teams as distributions (mean + variance) rather than point estimates,
  so "consistent but mediocre" and "volatile but talented" teams get
  differentiated even when their averages are similar
