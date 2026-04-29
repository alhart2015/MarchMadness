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

1. **Ensemble of model classes.** XGBoost + logistic regression +
   small neural net averaged (or stacked). The TODO already had this
   under Tier C. The hypothesis: different model classes capture
   partially-uncorrelated error patterns. Risk: if all three see the
   same features and reach the same ~80% R64 / ~50-60% deep-round
   ceiling, the errors are highly correlated and ensembling won't help
   much. Worth trying.
2. **Two-stage model.** Stage 1 produces a per-game probability (v4 as
   the base). Stage 2 trains on stage-1 errors using context features
   (seed pairing, round, conference-vs-conference, model-vs-Vegas
   disagreement). Different from a single-stage model with the same
   features because stage 2 is a residual learner.
3. **Upset-detection sub-model.** Train a separate classifier on
   "will the higher-seeded team lose?" using different feature
   weighting (heavier emphasis on high-confidence misses). Combine via
   meta-learner.
4. **External rankings (538, KenPom-public, BPI as features).** Note:
   we already have BPI, Sagarin, KenPom (POM), Bart Torvik (TRK), RPI
   via Massey ordinals (config.yaml lines 30-36). Truly external would
   be 538's tournament forecast or Vegas prop-bet predictions, which
   need data sourcing outside the Kaggle archive.
5. **Roster-level returning-experience.** Player-level data is not in
   the Kaggle Mania archive; would need an external roster CSV per
   season. Different signal from coach experience.

## Feature ablation on 2026 high-confidence misses
After the bracket-points backtest, run targeted feature ablation on v3 to
identify which v3-specific feature pushed Vanderbilt (R32 bust at 79%
confidence), Iowa St. (S16 bust at 82%), Texas Tech (R32 bust at 86%), and
Duke (E8 bust at 88%) too high in 2026. Re-run v3 with each new v3 feature
ablated in turn (late_adj_oe/de/em, late_sos, efficiency_trend,
margin_trend, conf_tourney_wins/champ, vegas_late_spread_delta) and compare
the 2026 pairwise probs for those four teams. Goal: identify a single
feature (or feature group) responsible for the over-confidence in deep
rounds, since v3 generally beats v1 on CV (20 of 22 LOSO seasons) but the
specific 2026 misses were costly.



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
