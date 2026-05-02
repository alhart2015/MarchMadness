# Future Work

## Tried and rejected

- **Quality-wins-vs-tournament-field (v5):** -93 pts vs v4 over 22 LOSO
  seasons. F4 accuracy fell 9pp. Already captured by KenPom/SOS.
- **Matchup-interaction features (v6):** avg = `(A+B)/2` columns
  alongside diffs. +7 pts vs v4 (within noise). Reverted.
- **Round-as-a-feature (v7):** added round (1..6) as a column. -10 pts
  vs v4. CV log loss flat (0.4384 -> 0.4387). Reverted. The model could
  not extract round-conditional signal from the existing features.
- **Stage-1 ensemble: XGBoost + LR (2026-05-01).** Simple-average
  ensemble of v4 + new logistic regression on the same 67-feature
  matrix. -105 brkt pts vs v4 alone over 22 LOSO seasons (v4 2713,
  ensemble 2608) after the v9-C correction. W/L/T = 7/13/2. Stage-1-
  only quality also worse (LL 0.4513 vs 0.4369; acc 0.801 vs 0.805).
  Diagnostic: residual correlation 0.77, optimal blend w=0.93,
  headroom +0.0006. Falsifies the diversity-at-identical-features
  hypothesis at this scale: LR's inductive bias does not produce
  errors uncorrelated enough with XGB's to make averaging worthwhile,
  and LR is a weaker base learner besides. v4 stays as stage-1.
  Findings: `docs/notes/2026-05-01-ensemble-stage1.md`. Code retained
  on `feat/ensemble-stage1` branch as the experiment record
  (`src/train_lr_stage1.py`, `src/ensemble_stage1.py`,
  `src/eval_stage1.py`, `src/run_v9c_on_stage1.py`); the refactor
  of `enhanced_model_v3.py` exposing `prepare_loso_inputs()` is
  reusable for any future model-class swap that wants the byte-
  identical v4 feature matrix.
- **Stage-1 ensemble: XGBoost + Bradley-Terry (2026-05-01).** Per-
  season plain BT (binary outcomes, regular-season-only, MAP via L2
  LR with team-indicator + home-court design). Gate FAILED at
  optimal w=0.98 and headroom +0.0000, but PASSED at residual
  correlation 0.577 < 0.60. Lesson: BT is structurally different
  enough to lower correlation from LR's 0.77 to 0.58, but too weak
  standalone (LL 0.565 vs v4's 0.437) for any meaningful blend
  weight to help. Two distinct failure modes from the LR experiment
  (correlation vs standalone weakness), both caught by the same
  gate. Findings: `docs/notes/2026-05-01-bayesian-stage1.md`. Saved
  ~3 hours of compute by gating before the v9-C backtest.
- **BT-as-feature for v9-C (2026-05-02).** Added p_bt from
  output/pairwise_bt.csv as a 6th input feature to v9-C's
  upset-aware stage-2 model under feature_set='v9d'. Pre-sweep
  falsification gate FAILED: at uniform weights (1.0, 0.0),
  v9-D wt-mean LL 0.4339 vs v9-C 0.4324 -- headroom -0.0015 < 0.001
  threshold, and *negative*. Adding BT as input feature didn't
  merely fail to help; it actively hurt the model (XGB spends some
  splits on the noisy p_bt feature, net loss vs v9-C's 5-feature
  baseline). Saved ~45-75 min of compute by not running the 15-cell
  sweep. v9-C's representation already extracts essentially
  everything p_bt could contribute on top of v4 + seed/round
  context. Closes the "v9-C as a learnable trust-weight function"
  escape hatch from the BT-ensemble failure -- v9-C's training data
  (~2898 per-game rows under double-LOSO) is too thin to learn
  useful per-context gating from a noisy feature. Code retained on
  feat/bt-as-feature as the experiment record. Findings:
  docs/notes/2026-05-02-bt-as-feature.md.

## Active queue

1. **Feature-view diversity ensemble: XGBoost on disjoint feature
   subsets** (e.g., one model on KenPom-only, one on Vegas-only,
   one on raw efficiency). Same model class -> same standalone
   strength as v4 (sidesteps the BT experiment's bottleneck).
   Disjoint feature views -> different errors by construction
   (sidesteps the LR experiment's bottleneck). With BT-as-feature
   now also closed, the per-context gating mechanism still applies
   but with model peers individually as strong as v4, the per-
   context training signal v9-C sees is also stronger. Most likely
   of the remaining items to clear all relevant diagnostic clauses.
2. **Hierarchical Bradley-Terry with feature priors** (`s_team ~
   Normal(beta . v4_features_team, sigma)`). Couples BT back to
   v4 features to gain standalone strength. Risk: residual
   correlation may regress from 0.58 back toward 0.77 as the
   models re-converge. With BT-as-feature now closed, this is the
   next angle on getting a stronger BT signal. Worth trying if
   item 1 hits a ceiling.
3. **Small neural net (MLP) as a stage-1.** Adds PyTorch tooling
   cost; diversity vs XGBoost on the 67-feature tabular space is
   the open question. Same correlated-error caveat as LR if it
   reuses the same feature matrix.
4. **Full Bayesian Bradley-Terry with strength + variance per team**
   (PyMC / NumPyro / Stan). The TODO's "Architecture Rethink (Tier
   C)" entry. Lets "consistent vs volatile" teams differentiate.
   Standalone-strength bottleneck likely persists (full posterior
   over weak strengths is still a weak point estimate); deferred
   until item 1 or 2 is settled.
5. **External rankings (538, KenPom-public, BPI as features).**
   Note: we already have BPI, Sagarin, KenPom (POM), Bart Torvik
   (TRK), RPI via Massey ordinals (config.yaml lines 30-36).
   Truly external would be 538's tournament forecast or Vegas
   prop-bet predictions, which need data sourcing outside the
   Kaggle archive.
6. **Roster-level returning-experience.** Player-level data is not
   in the Kaggle Mania archive; would need an external roster CSV
   per season. Different signal from coach experience.

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
- **Upset-detection sub-model (v9) -- LOSE (2026-04-30).** Tested two
  variants of an upset-aware stage-2 corrector replacing v8: v9-A
  (4 features, W_UPSET=3.0, W_MISS=4.0) and v9-B (7 features adding
  round + v4 confidence + is_higher_seed, same weights). Both lost
  catastrophically: v9-A scored 1552 bracket pts vs v8's 2670 (-1118)
  over 22 LOSO seasons; v9-B scored 1588 (-1082). The aggressive
  upset weighting pulls the model toward predicting upsets that
  aren't there. Sanity-check at W_UPSET=1.0, W_MISS=0.0 reproduced v8
  exactly, confirming the trainer is correct and the failure is
  weighting-magnitude. Code retained in src/train_upset_model.py
  for future low-weight sweeps; v8 stays in production. Findings:
  docs/notes/2026-04-30-upset-detection-v9.md.
- **v9 weight-sweep -- MARGINAL / closes the open question (2026-05-01).**
  15-cell sweep of W_UPSET in {1.0, 1.25, 1.5, 1.75, 2.0} x W_MISS in
  {0, 0.5, 1.0} on v9-B against the bracket-points objective. Best
  cell: (W_UPSET=1.0, W_MISS=0.5) at 2688 pts (+18 vs v8). Clears the
  spec's +10 bar but is fragile -- 17/22 seasons identical to v8,
  +12 of +18 from 2024 alone. Active ingredient is W_MISS (residual
  weighting), NOT W_UPSET (every cell with W_UPSET > 1 lost). The
  literal "milder upset weighting" hypothesis is unsupported. v8
  stays in production. Documented as a candidate, not a swap-in;
  v9-B's round-asymmetry bug is a prerequisite to any future
  production swap. Findings: docs/notes/2026-05-01-v9-weight-sweep.md.
- **v9-B round-asymmetry fix + sweep re-run (2026-05-01).** Replaced
  apply-time `round=0` hardcoding with a season-aware
  `MNCAATourneySlots` walk via `build_pair_round_lookup`. Re-ran the
  same 15-cell weight sweep. Result: MARGINAL WINNER on a different
  cell -- (W_UPSET=1.25, W_MISS=0.0) at 2690 pts (+20 vs v8, +2 vs
  PR 7's pre-fix winner). More robust profile than PR 7's: 4W-2L
  spread, max single-season delta +8 (vs PR 7's +12). Active
  ingredient flipped to W_UPSET (mild upset weighting), not W_MISS.
  Recommendation: candidate-only, not swap-in -- still within +20 of
  v8 (spec's "marginal" band: 10 < d <= 25). v8 stays in production.
  Findings: docs/notes/2026-05-01-v9-round-fix.md.
- **v9-C feature-stripped variant -- SWAP CANDIDATE (2026-05-01).**
  Parameterized `train_upset_model.py` with a `feature_set` arg;
  v9-C drops `v4_confidence` and `is_a_higher_seed` (5 features:
  v9-A's 4 + apply-time-correct round). Re-ran PR 8's identical
  15-cell sweep on v9-C. WINNER: (W_UPSET=1.25, W_MISS=0.0) at
  2713 brkt pts -- **+43 vs v8, +23 vs v9-B at the same cell**.
  Anchor reproduces v8 exactly (delta 0). F4/E8 chalk accuracy
  distinctly better than both v8 and v9-B (E8 58.6% vs 55.2%; F4
  62.8% vs 60.5%). Profile durable: 6W-3L-13T over 22 seasons;
  +27 even with the largest single-season win removed. Both
  clauses of the spec's swap-in path are satisfied. Production
  swap is a separate follow-up commit (Active queue #1).
  Findings: docs/notes/2026-05-01-v9c-feature-stripped.md.
- **v9-C production swap (2026-05-01).** Added
  `src/predict_2026_v9c.py` (mirror of `src/predict_2026_stage2.py`
  for v9-C). Trains on all 22 LOSO seasons with W_UPSET=1.25,
  W_MISS=0.0, feature_set='v9c'; applies to v4's 2026 JSON via
  apply-time round lookup; writes versioned snapshot
  `output/pairwise_probs_v9c_2026.json` and overwrites the
  canonical `output/pairwise_probs.json` (the file analysis scripts
  consume). Live bracket pipeline (`generate_bracket_real.py`)
  unchanged -- it's pure-v4-MC today and v8 was never wired in
  there either; live-bracket stage-2 integration is a separate
  follow-up. Spec:
  `docs/superpowers/specs/2026-05-01-v9c-production-swap-design.md`.



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
