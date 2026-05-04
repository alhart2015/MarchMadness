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
- **Feature-view diversity ensemble: K=2 semantic split (2026-05-02).**
  Trained two same-class XGBoost peers on disjoint v4 feature
  subsets: PEER_A (40 team-strength features: efficiency, four
  factors, KenPom, Massey, conf strength, season summary) and
  PEER_B (27 form/market/meta features: late-season, trajectory,
  rolling form, conf tourney, coach, Vegas). Pre-sweep 3-clause
  gate FAILED on 2 of 3 clauses: peer_A LL 0.5720 was +0.1375
  above v4's 0.4345 (clause 1 fails by 5.5x tolerance), and best
  2-blend headroom was -0.0206 (clause 3 fails). Clause 2
  (residual correlation r=0.45 < 0.60) PASSED -- the disjoint-view
  *mechanism* works, but PEER_A's standalone weakness sinks the
  ensemble. The 3-blend optimum (v4 + peer_A + peer_B) was
  (0.757, 0.000, 0.243) at LL 0.4316 -- v4 dominates and PEER_A
  contributes nothing; PEER_B's 24% is a side observation but
  BT-as-feature already falsified "added feature on top of v4" at
  this data scale. Saved ~90-150 min of compute by gating before
  E1+E2 sweeps. Code retained on feat/feature-view-ensemble as
  the experiment record (src/feature_views.py, src/train_peer_stage1.py,
  src/diagnose_feature_view_ensemble.py, blend_pairwise_csvs in
  ensemble_stage1.py, V9_STAGE1_PAIRWISE env var in
  sweep_v9_weights.py). Findings:
  docs/notes/2026-05-02-feature-view-ensemble.md.
- **Massey-matrix MOV as v4 feature (2026-05-03).** Closed-form
  least-squares solve over regular-season MOV with home-court
  estimated jointly and MOV cap=21. Two-clause cheap gate FAILED
  at clause 1 (non-redundancy): mean |corr| vs `adj_em` = 0.957
  (threshold 0.95), max 0.973 (threshold 0.97), across 24 seasons
  of tournament teams. Vs `massey_composite` was 0.946 / 0.965 --
  passed both thresholds. The redundancy is specifically with our
  own iterative `adj_em` efficiency loop, which is also opponent-
  adjusted on margin -- different mechanism (iterative fixed-point
  vs closed-form least squares), same signal.
  **Followup: time-decay weighting (REJECTED).** Sweep over
  half-lives {None, 7, 14, 30, 60, 120}d showed the redundancy
  gradient is non-monotonic: hl=30d resonates with adj_em (also
  30d) at mean |corr|=0.979 (worst), shorter half-lives diverge.
  hl=14d cleared clause 1 (mean 0.931 < 0.95) but FAILED clause 2:
  delta=+0.0057 LL vs threshold +0.001 on subset {2019, 2022, 2024}.
  The "different signal" hl=14d captures (last ~2 weeks margin) is
  already extracted by v4's `late_season`/`trajectory`/`vegas_trend`
  feature stack -- net contribution is noise + tree-split overhead.
  Code retained on branch feat/todo-massey-colley:
  src/features/massey_matrix.py (solver + cached loader + half_life_days
  kwarg, 9 unit tests including all-neutral edge-case fix),
  src/diagnose_massey_mov.py (two-clause gate runner),
  src/sweep_massey_decay.py (half-life sweep),
  src/clause2_decay_massey.py (parameterized clause-2 runner),
  allowed_holdouts kwarg added to leave_one_season_out_cv_weighted
  in src/enhanced_model_v3.py (reusable for any future cheap-subset
  diagnostic). Wire-in to compute_all_features reverted (Massey
  does NOT ship). Findings: docs/notes/2026-05-03-massey-mov.md.
- **Colley-matrix rating as v4 feature (2026-05-03).** Standard
  Colley `(2I + diag(T) - A) x = b` with +2 Bayesian prior. 3-baseline
  clause 1 (added `season_win_pct` per the Massey-decay lesson) PASSED
  on all three: mean |corr| vs adj_em = 0.907, vs massey_composite =
  0.948 (tight), vs season_win_pct = 0.687 (wide margin). Colley IS
  structurally distinct from existing features. But clause 2 FAILED:
  +0.0053 LL on subset {2019, 2022, 2024} vs threshold +0.001
  (2019 +0.007, 2022 +0.014, 2024 -0.005). Colley's clause-2 delta is
  near-identical to Massey-decay-14d's +0.0057. **Generalizable
  lesson:** at v4's data scale (~2898 tourney games for training),
  individual structural distinctness is necessary but not sufficient.
  v4's joint 67-feature stack already extracts opponent-adjusted
  team-strength via different decompositions; adding any single new
  rating feature provides no marginal value. Code retained on
  feat/todo-massey-colley: src/features/colley_matrix.py (solver +
  cache, 6 unit tests), src/diagnose_colley.py (3-baseline gate
  runner -- the 3-baseline pattern is the reusable artifact),
  data/cache/colley_ratings.parquet (gitignored). Wire-in reverted
  in 3b4c374. Findings: docs/notes/2026-05-03-colley.md.
- **Hierarchical Bradley-Terry with v4 feature priors (2026-05-03,
  framing corrected 2026-05-04).** Per-season MAP over `(s, beta, h)`
  of `s_team_i ~ Normal(beta . v4_features_team_i, sigma^2)` swept
  over sigma in {0.05, 0.10, 0.20, 0.50, 1.00, 2.00, 5.00},
  sigma_beta=1.0 fixed. **All 7 cells FAILED the LL-blend gate at
  clauses 2/3** (`w_opt = 0.99-1.00`, `headroom = +0.0000`). Clause 1
  PASSED at every cell with `r in [0.448, 0.507]` -- even *lower*
  than plain BT's 0.577 -- but standalone HBT LL was uniformly
  *worse* than plain BT's 0.565 across the full sigma range (HBT
  range: 0.619-0.757). **Narrow conclusion:** adding v4 feature
  priors to BT does not improve its standalone log loss enough to
  clear the LL-optimal-blend gate at any tested sigma. **The
  original write-up overclaimed** a "training-data ceiling"
  generalization; that claim is incompatible with the user's
  2159 / 3462 Kaggle finish (2/3 of entries beat v4 on the same
  data), so it has been retracted. The LL gate may also be
  filtering on the wrong metric -- production scoring is bracket
  points, where correctly-predicted upsets are weighted heavily,
  and a weak-but-diverse stage-1 could lift bracket points without
  lifting LL-blend headroom. Plain-BT's bracket-points re-test is
  active queue #2. Code retained on feat/hierarchical-bt-priors:
  src/features/hierarchical_bt.py (L-BFGS solver with analytic
  gradient, 5 unit tests), src/train_hbt_stage1.py (per-(sigma,
  season) trainer, 7 unit tests), src/diagnose_hbt_vs_v4.py
  (per-cell sigma-sweep gate runner, 8 unit tests; thresholds
  shared verbatim with plain-BT diagnostic via cross-module
  regression test). Findings + framing-correction postscript:
  docs/notes/2026-05-03-hierarchical-bt.md.

## Active queue

> **Re-prioritization 2026-05-04.** v4 finished 2159 / 3462 in a
> recent Kaggle Mania -- 2/3 of entries beat it on the same data.
> That falsifies any framing where "Kaggle data is exhausted" or
> "v4 is near the achievable ceiling." The bottleneck is much more
> likely to be inside v4 itself (features, calibration, hyperparams,
> model class) or in how stage-1 errors are scored against the
> production metric (bracket points, not log loss). Audits and
> metric corrections are now ahead of more ensemble/architecture
> exploration.

1. **Localize v4's gap vs an external benchmark.** Pull v4's per-
   game predictions, diff against an external strong baseline
   (Vegas closing-line implied probabilities from
   `data/raw/vegas_lines/`, the Vegas-trend module's source; or
   public 538 forecasts; or top-quartile public Kaggle predictions
   if any are recoverable). Bucket by round (R64 .. NCG), seed
   pair, higher-vs-lower-seed status, and v4-confidence bin. Find
   where v4 specifically loses -- which rounds, which seed bands,
   over- vs under-confidence. Without this we keep proposing
   ensemble add-ons without knowing what they need to fix.
   Vegas-implied-prob comparison is the cheapest start (Vegas
   data already ingested for the regular season; tournament Vegas
   data exists in The Prediction Tracker per `src/ingest`).
2. **Re-test plain BT against bracket points (skip the LL gate).**
   Reuse `output/pairwise_bt.csv` (force-added on PR 12). Blend
   `pairwise_v4 * w_v4 + pairwise_bt * (1 - w_v4)` at a small grid
   of weights {0.6, 0.7, 0.8, 0.9, 1.0}, run v9-C on each, score
   22-season bracket-points head-to-head vs `v4 + v9-C`. Cheap
   (~1 hour). The HBT findings note explains why the LL gate may
   have been filtering on the wrong metric -- plain BT's
   `r=0.577` is genuine residual diversity that v9-C's `W_MISS`
   sweep already showed contains useful signal.
3. **External rankings / external data as features (538 / Vegas
   prop-bet / roster injury, etc.).** Genuinely outside the
   Kaggle + KenPom + Bart Torvik archive (we already have BPI,
   Sagarin, KenPom, Bart Torvik, RPI via Massey ordinals).
   Sourcing question -- which sources are programmatically
   accessible across 22 seasons.
4. **Small neural net (MLP) as a stage-1.** Adds PyTorch tooling
   cost; diversity vs XGBoost on the 67-feature tabular space is
   the open question. Lower priority after Massey + Colley + HBT
   plus the framing correction: more ensemble exploration without
   first localizing v4's gap is wasted compute.
5. **Full Bayesian Bradley-Terry with strength + variance per team**
   (PyMC / NumPyro / Stan). HBT confirmed prior structure doesn't
   lift BT-class standalone strength on the LL-blend gate;
   switching to full posterior won't change that. Deferred until
   item 1 is settled.
6. **Roster-level returning-experience.** Player-level data is not
   in the Kaggle Mania archive; would need an external roster CSV
   per season. Different signal from coach experience. Closely
   related to #3.

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
