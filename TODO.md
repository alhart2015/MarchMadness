# Future Work

## Active queue (in order)

1. **Matchup-interaction features (v5).** The symmetric-pair architecture
   feeds the model only `feat_A - feat_B` differences, so it loses the
   *sum* / average information ("are both teams fast?", "are both elite?").
   Add `(feat_A + feat_B) / 2` average features for each existing feature
   in the matchup builder. Symmetry note: avg features are the same in both
   rows of a symmetric pair (winner-perspective and loser-perspective), so
   they don't break symmetric training.
2. **Round-as-a-feature (v6).** Add the round of the game as a feature in
   training. Allows the model to learn round-conditional interactions
   (e.g., "coach career F4 apps matters more in S16+ than R64"). Cheap
   change to the matchup builder + 2026 prediction code (which currently
   doesn't know rounds).
3. **External rankings as features (v7).** 538 / ESPN BPI / Bart Torvik
   tournament-day rankings as features. We have Massey ordinals already;
   these would add other public-model methodologies as direct signal.
   Needs data sourcing.

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
