# Future Work

## CONTAMINATION DISCOVERED 2026-05-04 (active recovery)

**TL;DR.** v4's Vegas-derived per-team-per-season features
(`vegas_avg_*`, `vegas_ats_pct`, `vegas_power_rating`,
`vegas_consistency`, `vegas_late_spread_delta`) were computed over
the full Vegas dataset INCLUDING NCAA tournament games. In LOSO CV,
this leaks the holdout season's tournament outcomes into the test
feature row for season S. v4's reported LOSO accuracy of ~80.4% per-
season (pre-fix) and the PR 18 finding "v4 beats Vegas everywhere"
cannot be trusted at face value. Falsified by the user's actual
Kaggle finish of 2159 / 3462 -- a model that genuinely beats Vegas
in every bucket does not finish in the bottom half of a real
prediction contest. Discovery thread: 2026-05-04 chat investigation
following the PR 18 merge. Quantified leak: 2024 UConn
vegas_avg_margin +1.98 above regular-season-only; 2024 Purdue
+1.98; 2018 Virginia -0.83. Leak correlates with tournament
success. **Clean-baseline measurement (PR <pending>, recovery step
3): 22-season mean LL 0.4370 (pre-fix) -> 0.5588 (clean), delta
+0.122. Mean accuracy 80.4% (pre-fix) -> 70.7% (clean), delta
-9.7pp. Verdict pass-and-flag: leak is much bigger than the spec's
0.45-0.47 LL anchor band.**

### Recovery plan (5 PRs, in order)

1. **[DONE -- PR 19]** Filter the Vegas leak. Merged 2026-05-04. Added
   `filter_vegas_to_pre_tournament()` and wired it before
   `compute_vegas_features` and `_build_vegas_team_records_with_dates`.
   Spec: `docs/superpowers/specs/2026-05-04-v4-vegas-leak-fix-design.md`.

2. **[DONE -- PR 20]** Audit Massey + KenPom inputs for the same
   class of leak. **Verdict: no leak found.** Massey is clean by file
   construction (max RankingDayNum = 133 = Selection Sunday). KenPom
   Barttorvik mixes pre-tournament rating columns with one post-tournament
   label (`ROUND`); the v3 feature pipeline uses an explicit 17-column
   allowlist that excludes `ROUND`, so no leak today. Defensive guard +
   unit tests added in `build_all_team_features` so a future change to
   the allowlist cannot silently regress this property. Findings:
   `docs/notes/2026-05-04-massey-kenpom-leak-audit.md`.

3. **[DONE -- PR 21]** Regenerate `output/pairwise_v4.csv` via
   clean LOSO. Mean LL 0.4370 -> 0.5588 (+0.122); mean acc 80.4% ->
   70.7% (-9.7pp). 21/22 seasons worse on LL; 20/22 worse on acc.
   Largest per-season shifts: 2017 (+0.190 LL, -14.2pp), 2010 (+0.179),
   2024 (+0.177). Verdict pass-and-flag (clean LL > 0.50 spec
   threshold). New canonical `pairwise_v4.csv` is the single
   tuned-pass output (48,465 rows; downstream consumers all dedup
   with `keep="last"` so the row count change is invisible to them).
   Procedure-side: added `MM_SKIP_DEFAULT_LOSO` env-var gate to
   `enhanced_model_v3.py` to halve regen runtime; reused leaky-run
   tuned XGB hyperparameters via `MM_TUNED_PARAMS_V3` (documented
   confound; expected effect <0.02 LL); fixed pre-existing
   `NameError` in v3 final-summary block (`new_feature_names` /
   `n_tourney` / `n_supplemental` orphaned by the
   `prepare_loso_inputs()` extraction). Findings:
   `docs/notes/2026-05-04-v4-clean-loso-regen.md`.

4. **[DONE -- PR <pending>]** Re-run the v4-vs-Vegas audit. Ran
   `python src/audit_v4_gap_vegas.py` against the regenerated
   `pairwise_v4.csv`. **Verdict: "no weak spots" RETRACTED.**
   Overall: clean v4 LL 0.5595 vs Vegas 0.5447 (delta +0.0148, Vegas
   wins); acc v4 69.9% vs Vegas 70.6% (-0.7pp); ECE v4 0.037 vs
   Vegas 0.029. **Six bucket-level weak spots** at the n>=50,
   ll_delta>=+0.02 threshold: round=E8 (+0.055), chalk_won=upset
   (+0.054), round=S16 (+0.027), seed_diff=6-9 (+0.026),
   v4_confidence=0.80-0.90 (+0.025), seed_diff=10-15 (+0.021).
   Two big collapses vs the contaminated baseline: (1) v4's
   upset-detection edge over Vegas (originally 56% vs 17%) is
   fully gone -- clean v4 catches 15.3% of upsets vs Vegas's 17.5%;
   (2) v4's 0.90-1.00 confidence bucket on tournament games is now
   EMPTY (was n=413 pre-fix), confirming the leak was inflating
   v4's confidence by peeking at within-season tournament outcomes.
   Recovery framing changes: 538 audit is still useful but no
   longer the only path to localize headroom -- Vegas has surfaced
   six concrete weak spots to engineer against. Findings:
   `docs/notes/2026-05-04-v4-gap-audit-vegas.md` (rewritten with
   clean numbers; original contaminated tables preserved as an
   appendix). The PR 18 audit's framework code is unchanged
   (`src/audit_v4_gap_vegas.py`, `tests/test_audit_v4_gap_vegas.py`).

5. **Re-run the swap-decided / swap-candidate evaluations against
   the clean baseline.** **Now the immediate next PR.** Priority order:
   - **v9-C production swap** (currently deployed -- top priority).
   - **v8 vs v9-C** bracket-points head-to-head.
   - **Plain BT bracket-points** (PR 17 finding).
   - The "marginal" rejections in `Tried and rejected` whose deltas
     were within ~0.05 LL or ~30 brkt pts of v4 (BT-as-feature at
     -0.0015 LL; v9 weight-sweep family at +18 to +20 pts).
   Big-magnitude rejections (-93 quality wins, -105 LR ensemble,
   +0.0057 Massey-decay clause-2 fail, etc.) do not need re-eval --
   a baseline shift of 0.02-0.05 LL won't flip them. With the leak
   shift now measured at +0.122 LL (vs the 0.02-0.05 estimate when
   this section was first written), the redo-or-skip cutoff for
   "marginal" experiments should be re-checked: at +0.122, even
   the v9 weight-sweep +18 to +20 brkt pts wins fall well within
   the leak's noise floor. **Additional motivation from the audit
   rerun (step 4):** v4's upset-detection edge (the headline
   evidence used to justify v9-C's upset-aware stage-2) was the
   leak speaking. The clean v4 catches 15.3% of upsets vs Vegas's
   17.5%, so v9-C's stage-2 may have been correcting noise rather
   than signal. Re-eval is now load-bearing for whether v9-C
   stays in production.

### What's NOT contaminated

- Diagnostics computed within a season across teams (e.g. Massey-
  vs-adj_em correlation = 0.957) -- the leak shifts both sides
  similarly within a season; redundancy verdicts stand.
- Anchor-equality checks (e.g. "weights (1.0, 0.0) reproduces
  pairwise_v4 byte-equal") -- these test plumbing, not signal.
- Selection of non-v4 models against absolute thresholds (e.g.
  Plain BT standalone LL=0.565 vs v4's 0.437 -- the gap survives
  any plausible shift in v4).
- The PR 18 audit's *framework* (per-bucket LL/acc/ECE), only the
  numerical verdict.

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
- **Plain BT bracket-points re-test (2026-05-04).** Tested whether
  the LL gate's NO-GO on plain BT (PR 12) reflected a metric
  mismatch: blended `pairwise_v4 + pairwise_bt` at `w_v4` in
  {0.60, 0.70, 0.80, 0.90, 0.95, 1.00}, ran v9-C on each, scored
  22-season bracket points head-to-head against the canonical
  `v4 + v9-C` baseline (2713 brkt pts). **NO-GO.** Every non-anchor
  cell LOST: best non-anchor delta was `w_v4=0.90` at -29 brkt pts;
  worst was `w_v4=0.80` at -67. The anchor (`w_v4=1.00`) reproduced
  the baseline exactly to floating-point precision (max_abs_diff
  = 0.0 over 48,465 rows). **The LL gate was correct for plain BT
  specifically**: both the cheap LL diagnostic and the production
  metric agree. The HBT findings note's framing-correction concern
  ("LL gate may be filtering on the wrong metric") is falsified
  for this case. **Generalized lesson:** structural diversity
  (`r=0.577`) is necessary but not sufficient -- we also need
  per-disagreement accuracy in the regions where the candidate
  disagrees with v4. Plain BT was right only 27.9% of the time on
  v4-disagreements (PR 12 finding), so v9-C cannot extract upset
  signal from a stage-1 whose disagreements are 72% wrong. Code
  retained on feat/bt-bracket-points: src/sweep_bt_bracket_points.py
  (driver + ladder, 6 unit tests), output/bt_bracket_sweep.json
  (per-(w, season) numbers + verdict), 6 force-added per-cell
  pairwise CSVs. Findings: docs/notes/2026-05-04-bt-bracket-points.md.
- **v4 gap audit vs Vegas closing-line implied probs (2026-05-04).**
  Audited v4 stage-1 vs Vegas-implied probs (SIGMA=11) across 1326
  played 2003-2025 tournament games (91.5% join coverage), bucketed
  by round, chalk-vs-upset, v4-confidence quintile, seed-diff
  magnitude. **No weak spots at the n>=50, ll_delta>=+0.02
  threshold.** v4 BEATS Vegas on every bucket: overall
  `ll_v4=0.4305` vs `ll_vegas=0.5447` (delta -0.114),
  `acc_v4=80.9%` vs `acc_vegas=70.6%`. v4 catches 56% of upsets
  vs Vegas's 17%. ECE comparable (v4 0.025, Vegas 0.030). The audit
  did its job (produced a per-bucket map) but the verdict is "Vegas
  is the wrong benchmark to localize v4's headroom" -- 538 audit
  is now strongly motivated as the immediate next experiment.
  **Caveat:** SIGMA=11 may be too peaky for tournament games (10pt
  spread implies 0.82 win-prob but empirical rate is ~0.75-0.78); a
  larger SIGMA would shrink Vegas's LL toward v4's. Even so, v4's
  10.3pp accuracy advantage is robust to SIGMA. Code retained on
  feat/v4-gap-audit-vegas: src/audit_v4_gap_vegas.py (one-shot
  driver, 10 unit tests), output/v4_gap_audit_vegas.json (per-
  bucket metrics + weak_spots), 3 calibration PNGs. pyproject.toml
  adds matplotlib>=3.7. Findings: docs/notes/2026-05-04-v4-gap-
  audit-vegas.md.

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

1. **538 v4 gap audit (next benchmark in the audit framework).**
   Reuse the audit framework from PR 18 (`src/audit_v4_gap_vegas.py`
   pattern -- bucket-and-compute-metrics, per-cell calibration,
   weak-spot threshold). Same buckets (round, chalk-vs-upset,
   v4-confidence quintile, seed-diff magnitude). Difference: 538
   publishes calibrated tournament-forecast probabilities directly
   (no SIGMA conversion). 538 is widely regarded as a strong public
   benchmark; if v4 also beats 538 across the board we have a real
   "v4 is competitive" finding; if 538 beats v4 in specific buckets,
   we have weak-spot signatures to engineer against. **Sourcing
   investigation is the first task** -- 538's tournament-forecast
   archive: API access? scraping? historical-archive coverage 2014+?
   Promoted from "next" status in the Vegas-audit findings note.
2. **Single-season v4 variance check.** The Vegas audit shows v4
   beats Vegas on the 22-season aggregate. The user's Kaggle finish
   (2159 / 3462) is a single-season result. Plot per-season v4 LL +
   ECE; identify any season where v4's calibration is materially
   worse than the 22-season average. Cheap (~30 min) follow-up
   audit. Surfaces whether v4's 22-season-average story hides
   high-variance per-season behavior that hurts on single-season
   Kaggle scoring.
3. **External rankings / external data as features (538 / Vegas
   prop-bet / roster injury, etc.).** Distinct from item #1: that
   item AUDITS v4 against 538; this item adds 538-derived signals
   as input features to v4 itself. Sequence: do the audit first,
   then engineer against the weak spots it surfaces (if any).
   Sourcing question shared with item #1.
4. **Small neural net (MLP) as a stage-1.** Adds PyTorch tooling
   cost; diversity vs XGBoost on the 67-feature tabular space is
   the open question. Lower priority after Massey + Colley + HBT
   + plain-BT-on-bracket-points. The lesson from the bracket-points
   re-test (PR 17): same-data peers aren't sufficient even on the
   production metric -- a stage-1 needs per-disagreement accuracy
   >= ~45% in the regions where it differs from v4. MLP on the
   same 67-feature target faces the same risk profile as LR did.
5. **Full Bayesian Bradley-Terry with strength + variance per team**
   (PyMC / NumPyro / Stan). HBT confirmed prior structure doesn't
   lift BT-class standalone strength on the LL-blend gate; the
   bracket-points re-test (PR 17) confirmed plain BT also doesn't
   help on the production metric. Switching to full posterior
   won't change either. Deferred until items 1-3 are settled.
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
