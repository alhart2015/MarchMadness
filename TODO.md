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
   the clean baseline.** Priority order:
   - **[DONE -- PR <pending>]** v9-C production swap re-eval. **REVERTED
     to v8.** Best v9-C cell (W_UPSET=1.0, W_MISS=0.5) at 1929 brkt pts
     vs clean v8 baseline 2069 -- delta -140 over 22 LOSO seasons.
     PR 9's winning cell (W_UPSET=1.25, W_MISS=0.0) at 1753 -- delta
     -316. Every cell in the 15-cell sweep loses; higher W_UPSET loses
     more (the upset signal v9-C amplified was leak speech).
     `output/pairwise_probs.json` restored via `predict_2026_stage2.py`
     (clean-v8 stage-2 over leaky 2026 v4 stage-1; full cleanliness
     pending v4 2026 stage-1 regen -- see new follow-up below).
     Per-season W/L for the winning cell: 8W-10L-4T; biggest single-
     season v8 wins 2015 (-54), 2017 (-40), 2019 (-57), 2022 (-23).
     Findings: `docs/notes/2026-05-04-v9c-clean-rerun.md`.
     **Compounding work:** PR 21's clean `pairwise_v4.csv` was lost
     in the 2026-05-04 data wipe (gitignored, lived only in the
     wiped worktree); this PR re-ran PR 21's regen procedure and
     force-added the result. Same with `pairwise_v8.csv`. New runbook:
     `docs/data_recovery.md`.
   - **[DONE -- PR <pending>]** Plain BT standalone re-eval. **GATE
     FAILED** under clean baseline -- with a *flipped* failure mode.
     PR 12 failed clauses 2 and 3 (degenerate w_v4=0.98, headroom +0.0000)
     while passing clause 1 (r=0.577). This PR PASSES clauses 2 and 3
     (w_v4=0.58, headroom +0.0058) but FAILS clause 1 (r=0.868, well
     above the 0.60 threshold). When v4 lost its tournament-leak signal,
     its errors became much more similar to BT's errors -- both models
     now miss the same "hard regular-season-information" games.
     Robust NO-GO across both leaky and clean baselines. Plain BT
     closed as a stage-1 ensemble peer; bracket-points re-test
     (PR 17 redo) skipped per spec decision matrix (LL-gate failure
     across both baselines is sufficient). Standalone metrics: clean
     v4 LL 0.5579 / acc 70.2%; BT LL 0.5650 / acc 69.8% (delta -0.0071
     LL, +0.4pp acc). Disagreement rate 13.6% (was 24.0% under leaky);
     when they disagree, BT is right 48.7% (was 27.9%). Findings:
     `docs/notes/2026-05-05-plain-bt-clean-rerun.md`. Procedure-side
     change: added `--curve-out` flag to `diagnose_bt_vs_v4.py` that
     persists the full LL(w) blend curve as a 2-column CSV.
   - **Plain BT bracket-points** (PR 17 finding) -- DROPPED. LL-gate
     failure across both leaky and clean baselines closes plain BT
     as a stage-1 ensemble peer; bracket-points re-test no longer
     load-bearing.
   - The "marginal" rejections in `Tried and rejected` whose deltas
     were within the +0.122 LL leak noise floor of v4. Two named in
     the original roadmap (BT-as-feature at -0.0015 LL; v9 weight-
     sweep family at +18 to +20 pts). **All five named items addressed
     across PR 24, PR 25, PR 26, and this PR -- four closed; Colley
     advanced to full LOSO backtest queue (see new sub-priority below):**
       - Plain BT standalone (PR 12): **CLOSED in PR 24.** Standalone
         LL 0.565 was ~tied with clean v4 0.5588 -- gate clauses 2/3
         flipped PASS as predicted, but clause 1 (residual correlation)
         flipped FAIL (r=0.577 -> 0.868). Robust NO-GO.
       - Feature-view ensemble PEER_A/B (PR 14): **CLOSED in PR 25.**
         All 3 clauses FAIL on clean baseline. PEER_A's clause 1
         individually flipped PASS (delta_a +0.0140) as predicted, but
         PEER_B individually flipped FAIL (delta_b +0.0277); clause 2
         flipped FAIL with rho=0.45 -> 0.726 (matches PR 24's BT-vs-v4
         residual-correlation jump direction); clause 3 stayed FAIL
         (headroom -0.0084). Robust NO-GO.
       - HBT (PR 16): **CLOSED in PR 26.** All 7 sigma cells FAIL on
         clean baseline -- with a *flipped* failure mode vs PR 16.
         PR 16 had every cell PASS clause 1 (r in [0.448, 0.507]) and
         FAIL clauses 2/3 (w_opt 0.99-1.00, headroom +0.0000); this PR
         has every cell FAIL clause 1 (r in [0.678, 0.767], jump of
         +0.21 to +0.27 on every cell), two cells (sigma=0.20, 0.50)
         PASS clause 2 (w_opt 0.83, 0.84), but every cell still FAIL
         clause 3 (headroom +0.0009 to +0.0021, all below the +0.005
         threshold). Standalone HBT LL barely shifted (per-cell delta
         in [-0.0065, +0.0022]) -- regular-season-W/L data dominates
         the prior at every swept sigma -- so the residual-correlation
         jump comes entirely from clean v4 getting weaker. Third
         independent confirmation of PR 24's "same-data-peer residual
         ceiling" hypothesis (BT model class, XGB feature-view model
         class, hierarchical BT model class -- all show the same r
         jump). Robust NO-GO across both leaky and clean baselines.
         v9-C correction + bracket-points backtest skipped per spec
         decision matrix (PR 17's re-test on plain BT already showed
         LL-gate failure transfers to the production metric for
         r > 0.60 candidates). Findings:
         `docs/notes/2026-05-05-hbt-clean-rerun.md`.
       - Colley (PR 15): **CLAUSE 2 PASSED in this PR.** Clean delta
         -0.0100 vs leaky +0.0053 -- a -0.0153 LL swing. All three
         subset seasons help under clean v4 (2019 -0.0166, 2022 -0.0074,
         2024 -0.0059); 2019 and 2022 fully flipped from hurt to help.
         Pre-registered "redundancy is structural, not threshold-tight"
         prediction REFUTED -- the redundancy was specifically against
         vegas-inflated adj_em/efficiency loops, not against the clean
         stack. Promoted to "Colley full LOSO backtest" sub-priority
         below. Findings: `docs/notes/2026-05-05-colley-massey-clean-rerun.md`.
       - Massey-decay hl=14d (PR 15): **CLOSED in this PR.** Clean
         clause-2 delta +0.0018 vs threshold +0.001 -- shrunk from leaky
         +0.0057 but still FAIL. Mixed per-season pattern: 2019 helped
         (-0.0121, similar magnitude to Colley's 2019 swing), 2022 still
         hurt slightly (+0.0046), 2024 hurt the most (+0.0131 --
         inverted from leaky -0.002). Robust NO-GO across both leaky
         and clean baselines. Different failure mechanism per baseline:
         leaky failure was redundancy with vegas-inflated adj_em; clean
         failure is redundancy with clean non-vegas late-season features
         (`late_adj_em`, `late_sos`, `efficiency_trend`, `margin_trend`).
         Half-life sweep NOT re-opened (rest of curve was already
         filtered out at clause 1 in PR 15). Findings:
         `docs/notes/2026-05-05-colley-massey-clean-rerun.md`.
   - **[DONE -- PR <pending>] Colley full LOSO backtest REJECTed.**
     LL_delta=-0.0003 LL on 22-season aggregate (Marginal band:
     in (-0.005, +0.001), did NOT trigger Reject on LL); brkt_delta=
     -24 brkt pts vs clean v8 baseline (2069), which DOES trigger
     Reject (<= +10 bar). Acc_delta=+0.47pp (10/22 seasons help on
     acc); seasons help/hurt on bracket points 10/10 (worst hurts
     2009 -33, 2015 -30, 2019 -28). Wire-in REVERTED on this branch.
     Closes Colley as v4-stack feature. **Generalized lesson:** the
     3-season clause-2 PASS in PR 27 over-represented Colley-helpful
     seasons on LL; W/L-only opponent-adjusted strength produces a
     small LL improvement on the 22-season aggregate but the
     probability shifts flip enough chalk picks in the upper rounds
     that the bracket-points score regresses materially. The clause-2
     gate's LL-on-3-seasons signal does not translate to bracket-points
     headroom under v4's existing feature stack. Findings:
     `docs/notes/2026-05-05-colley-full-loso-backtest.md`.
   - **NEW:** Regenerate v4's 2026 stage-1 predictions. The current
     production `pairwise_probs.json` is clean v8 stage-2 over LEAKY
     2026 stage-1 (`output/pairwise_probs_v4.json` is Apr 28). Trace
     the script that produced it, re-run on clean-trained v4, force-add
     the JSON (yet another gitignored canonical artifact gap).
   Big-magnitude rejections (-93 quality wins, -105 LR ensemble) do
   not need re-eval -- a leak shift of +0.122 LL won't flip them.
   With the leak shift now measured at +0.122 LL (vs the 0.02-0.05
   estimate when this section was first written), the redo-or-skip
   cutoff for "marginal" experiments was re-checked in the v9-C
   re-eval and the marginal-rejections list expanded above.

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

> **Re-prioritization 2026-05-07.** Audit lane closed by the 538 audit
> (now in Done). Two public benchmarks (Vegas at SIGMA=11,
> 538 round-survival forecasts) have produced contradictory weak-spot
> signatures: Vegas surfaces "v4 worse on upsets, late rounds, mid-
> seed-gap, 0.80-0.90 confidence band"; 538 surfaces "v4 worse on
> chalk picks." The two together imply the bottleneck is calibration
> *shape* rather than any single bucket. Engineering against any one
> bucket is now under-motivated; the next levers are the cheaper
> single-season variance check and the external-data-as-features
> experiment, with the calibration-shape insight available as a
> backup engineering target.
>
> **Update 2026-05-07 (per-season variance check came back MIXED).** 4
> seasons flagged at 1.5 sigma (2011, 2013, 2015, 2023), no clean
> trend. 2011 was the standout (3 of 4 metrics) but predates the
> user's Kaggle period by ~13 years. 2024 (the Kaggle year) was
> unremarkable. **No single season explains the user's 2159/3462
> finish**; the bottleneck is calibration shape, not single-season
> variance. Per the plan's MIXED rule, queue ordering is retained.
> External Data (now item #1) leads by elimination; calibration-
> shape engineering (now item #2) remains the backup engineer-to
> target; no items promoted.
>
> **Update 2026-05-07 (R64 closing-line blend came back FAIL).** Cheap
> apply-time test of the data hypothesis: override v4's 32 R64
> pairwise probs with Vegas closing-line implied probs (best sigma
> 12), train v8 on UNMODIFIED v4, apply to OVERRIDDEN frame, score
> 22-season bracket points. Hard mode delta **+2 brkt pts** (W/L/T
> 7/11/4); mean mode delta **0**. Both well below the +10 MARGINAL
> bar despite anchor invariance passing exactly (max_abs_diff=0).
> The R64 LL improvement (+0.012 from the Vegas audit) **does NOT
> transfer to bracket points** at the cheap apply-time level. **The
> data hypothesis at apply-time is falsified.** Consequence: External
> Data (was #1) is dropped; calibration-shape engineering is promoted
> to **Active queue #1**. Phase 2 (re-train v8 on overridden frame)
> NOT triggered per spec. Futures-as-feature de-prioritized -- the
> null result is meaningful evidence against external-data-as-feature
> in general for v4's existing 67-feature stack.

1. **v4 calibration-shape engineering (audit-derived).** **Promoted
   to #1 by the R64 line blend FAIL (2026-05-07).** 538 has v4 LL
   +0.075 worse on chalk picks (where it counts); Vegas has v4 LL
   +0.025 worse in the 0.80-0.90 confidence band. Both audits flag
   confidence-shape weakness, just on different sides of the
   distribution. The variance check (PR 30) showed ~25% CV on ECE
   across 21 seasons -- v4's calibration shape varies year to year
   in a way the model doesn't currently correct for. Candidate
   fixes: temperature scaling, isotonic regression on a held-out
   tournament-only validation set, late-stage "confidence sharpening"
   feature. Gate: 22-season bracket points (not just LL). The
   R64-blend null result confirms LL gains don't auto-translate to
   bracket points -- whatever fix is tried here MUST be evaluated
   on bracket points directly.
2. **Small neural net (MLP) as a stage-1.** Adds PyTorch tooling
   cost; diversity vs XGBoost on the 67-feature tabular space is
   the open question. Lower priority after Massey + Colley + HBT
   + plain-BT-on-bracket-points. The lesson from the bracket-points
   re-test (PR 17): same-data peers aren't sufficient even on the
   production metric -- a stage-1 needs per-disagreement accuracy
   >= ~45% in the regions where it differs from v4. MLP on the
   same 67-feature target faces the same risk profile as LR did.
3. **Full Bayesian Bradley-Terry with strength + variance per team**
   (PyMC / NumPyro / Stan). HBT confirmed prior structure doesn't
   lift BT-class standalone strength on the LL-blend gate; the
   bracket-points re-test (PR 17) confirmed plain BT also doesn't
   help on the production metric. Switching to full posterior
   won't change either. Deferred until items 1-2 are settled.
4. **Roster-level returning-experience.** Player-level data is not
   in the Kaggle Mania archive; would need an external roster CSV
   per season. Different signal from coach experience. **Now the
   leading external-data candidate** (since R64-line external-data
   was falsified) -- but data sourcing cost is high and the broader
   "v4 has a structural ceiling on bracket-points lift from any
   apply-time feature" pattern from the R64-blend FAIL applies here
   too. Reconsider only after calibration-shape (#1) settles.
5. **Pre-tournament Vegas futures (championship odds, F4-reach
   odds) as a per-team strength feature.** Was #1b in the
   `2026-05-07-v4-kaggle-gap-strategy.md` strategy note. **Now
   demoted** by the R64-blend FAIL: if R64 line apply-time override
   doesn't move bracket points, the related "futures-derived team
   strength as v4 stage-1 feature" experiment is now lower-prob.
   Re-evaluate only if calibration-shape engineering (#1) also
   fails. Sourcing the historical futures data is its own cost;
   not worth paying yet.

## Done

- **R64 closing-line blend -- FAIL (2026-05-07).** Cheap apply-time
  test of the data-hypothesis (`docs/notes/2026-05-07-v4-kaggle-gap-strategy.md`):
  override v4's 32 R64 pairwise probabilities per tournament with
  Vegas closing-line implied probs, train v8 stage-2 on UNMODIFIED
  v4 + apply to OVERRIDDEN frame, score 22-season bracket points
  vs canonical clean v4+v8 baseline 2069. **Verdict: FAIL.** Hard
  mode total **2071 (delta +2 brkt pts; W/L/T 7/11/4; biggest swing
  2011 +5)**; mean mode total **2069 (delta 0; W/L/T 8/8/6)**. Both
  well below the +10 MARGINAL bar. SIGMA sweep picked sigma=12.0
  (LL=0.4996, true interior min on the {9, 10, 11, 12, 13} grid).
  Anchor invariance PASSED exactly (v8 trained on UNMODIFIED v4
  + applied to UNMODIFIED reproduces canonical `pairwise_v8.csv`
  byte-equal: matches=True, max_abs_diff=0.0, total=2069). The R64
  LL improvement (+0.012 from the Vegas audit on 648 R64 games)
  **does NOT transfer to bracket points** at the cheap apply-time
  level. R64 line coverage 648/692 = 93.6% (consistent with the
  Vegas audit's 91.5% on full frame). 2024 (Kaggle year) showed
  delta=0 on both modes -- the override doesn't move anything for
  the year that motivated this work. Phase 2 (re-train v8 on
  overridden frame) NOT triggered per spec. **Generalized lesson:**
  external apply-time market consensus on R64 -- the cheapest, most
  public external-data signal v4 doesn't currently use -- doesn't
  lift bracket points despite measurably improving R64 LL.
  Re-prioritization: calibration-shape engineering promoted to
  Active queue #1; futures-as-feature deprioritized; the "data
  hypothesis is the leading explanation for the Kaggle gap" claim
  from the strategy note is now WEAK evidence rather than strong.
  Code retained on `feat/v4-r64-line-blend`:
  `src/build_r64_line_override.py` (override + sigma sweep),
  `src/eval_r64_line_blend.py` (eval driver), 15 unit + smoke tests
  (8 in `test_build_r64_line_override.py`, 7 in
  `test_eval_r64_line_blend.py`). Outputs (force-added):
  `output/pairwise_{v4,v8}_r64lineblend_{hard,mean}_sigma12.csv`,
  `output/pairwise_v8_r64lineblend_v4only.csv` (anchor),
  `output/r64_line_blend_eval.json`,
  `output/r64_line_blend_eval_log.txt`,
  `output/r64_line_blend_calibration.png`.
  Findings: `docs/notes/2026-05-07-v4-r64-line-blend.md`.
  Strategy frame: `docs/notes/2026-05-07-v4-kaggle-gap-strategy.md`.

- **Per-season v4 variance check -- MIXED (2026-05-07).** Cheap
  diagnostic gate over 22 LOSO seasons (21 with Vegas, 7 with 538) to
  test whether single-season variance dominates the user's 2159/3462
  Kaggle finish. **Verdict: MIXED.** 4 seasons flagged at 1.5 sigma
  across the 4 tracked metrics (`ll_v4_minus_vegas`,
  `ll_v4_minus_fte`, `ll_v4`, `ece_v4`): **2011, 2013, 2015, 2023.**
  No 3-consecutive-season trend. **Standout: 2011** -- flagged in 3
  of 4 metrics. Worst single-season `ll_v4` in the frame (0.699 vs
  21-season mean 0.557, 2.22 sigma above), worst Vegas delta (+0.074,
  1.77 sigma), high ECE (0.186, 1.54 sigma). v4 accuracy 57.4% in
  2011 vs Vegas's 65.6% on the same 61 games. 2011 was the
  Butler/UConn-Kemba/VCU-FF11 tournament with heavy upset traffic --
  predates the user's Kaggle interest by ~13 years and is not
  actionable for production. **2024 (the Kaggle year) was unremarkable**
  in the per-season frame: `ll_v4`=0.591, `ll_v4_minus_vegas`=+0.037,
  not flagged. Per the plan's MIXED rule, queue ordering is retained;
  ambiguity noted in the Active queue preamble. Anchors: weighted
  per-season ll_v4 (538 subset, 428 games) reproduces 538 audit's
  0.5799 to FP precision; weighted ll_fte reproduces 0.6011; weighted
  ll_v4 (Vegas R64-Champ subset, 1261 games) is 0.5565, slightly
  below the Vegas audit's full-frame 0.5595 because the variance
  check excludes FF/OTHER buckets (intentional, direction consistent
  with R64-dominated LL arithmetic). ECE variance is non-trivial
  (~25% CV across 21 seasons) -- modest but real evidence v4's
  calibration shape varies year to year. Code retained on
  feat/v4-per-season-variance: `src/analyze_v4_per_season_variance.py`
  (driver), `tests/test_analyze_v4_per_season_variance.py` (7 unit
  tests including 1 smoke). Outputs:
  `output/v4_per_season_variance.json` + 2 PNGs + log (force-added).
  Findings: `docs/notes/2026-05-07-v4-per-season-variance.md`.
- **538 v4 gap audit -- PASS-AND-FLAG (2026-05-07).** 7-season audit
  (2016-2019, 2021-2023) on 428 R64-Champ games. Sourcing pivoted to
  Wayback Machine -- 538's live endpoints went dark in the March 2025
  shutdown; spec's GitHub raw URL pattern (`raw.githubusercontent.com/.../master/march-madness-predictions/<year>/`)
  was wrong (that dir only holds 2014's 62 bracket-challenge CSVs),
  the actual canonical pattern was `projects.fivethirtyeight.com/march-madness-api/<year>/...`
  and is now dead (302 redirect to abcnews.go.com). Internet Archive
  has 200-status text/csv captures of the original CSVs for 2016-2023;
  2014/2015 predate the API, 2024/2025 not archived. Snapshots from
  2025-03-06 pinned in `_FTE_URL_BY_YEAR`.
  Schema correction during impl: 538's `rdR_win` is P(reach round R),
  not P(win round R's game) -- audit reads `rd{X+1}_win` for round-of-X.
  Verdict: **v4 marginally beats 538 on overall LL** (0.5799 vs 0.6011,
  delta -0.0212) but trails on accuracy (-1.8 pp). **One weak spot at
  threshold:** chalk_won=chalk (n=298, ll_v4=0.322, ll_fte=0.247,
  delta=+0.0754) -- 538 is materially more confident in winning
  chalk picks than v4 is. Pattern across rounds: 538 leads R64 (+0.011),
  v4 dominates S16 (-0.123) and E8 (-0.155); 538's BT-norm
  approximation gets shaky in late rounds where rdR_win averages over
  multiple expected opponents. Cross-audit comparison: Vegas surfaced
  6 weak spots (upsets, late rounds, mid-seed-gap, 0.80-0.90 conf
  band); 538 surfaces 1 (chalk picks). The two benchmarks find
  *different* weak spots, implying calibration shape (not any single
  bucket) is the bottleneck. Code retained on feat/v4-gap-audit-fte:
  `src/ingest/fte_forecasts.py` (loader, 10 unit tests),
  `src/audit_v4_gap_fte.py` (driver, 9 unit tests). Outputs:
  `output/v4_gap_audit_fte.json` + 3 calibration PNGs (force-added).
  Findings: `docs/notes/2026-05-04-v4-gap-audit-fte.md`. Anchors:
  coverage 99.1% (428/432), R64 rd2_win sum-to-1 1.0/1.0/1.0 across
  50 sampled matchups, overall ll_v4 within +/-0.05 of clean v4's
  22-season LL 0.5588.
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



## Engineering follow-ups (deferred)

- **Test-suite hygiene: 10 tests fail on a fresh clone / fresh worktree
  until `tar -xzf data/training_data.tar.gz -C data/raw/` is run.** Has
  bitten us on every data wipe (2026-05-02, 2026-05-04, again at the
  PR 25 worktree setup on 2026-05-05). The unzipped Kaggle CSVs are
  gitignored; only the tarball is tracked. Three clusters of failures
  with different fix paths:
    - **Cluster 1: 6 bracket-walk tests in `tests/test_upset_model.py`**
      (`test_build_pair_round_lookup_*`, `test_build_v9_pairwise_uses_real_round_at_apply`).
      Use real 2024 `MNCAATourneySlots.csv` / `MNCAATourneySeeds.csv`
      purely as a fixture; the assertions ("1v16 in same region -> R1",
      etc.) are about `build_pair_round_lookup`'s logic, not 2024
      specifically. **Replace with a hardcoded toy bracket fixture
      (~40 lines)**; closes 6 of 10 with no value loss.
    - **Cluster 2: scoring/range smoke (2 tests in
      `test_score_chalk_brackets.py`, `test_sweep_bt_bracket_points.py`).**
      `test_score_v4_returns_known_shape` asserts brkt pts in [1800, 3500]
      -- this is a real regression sentinel (clean ~1955, leaky ~2713).
      Both tests already have skip guards on missing pairwise CSVs but
      bypass them when the CSV exists and only the underlying tournament
      data is missing. **Add `skipif` guard on
      `data/raw/march-machine-learning-2026/MNCAATourneySeeds.csv`
      presence with a "needs Kaggle data" message**; keeps value when
      data is present.
    - **Cluster 3: `test_prepare_loso_inputs.py` (1 test).** Builds the
      full v3/v4 feature matrix end-to-end as a contract pin. Hard to
      dummy without mocking every loader. **Same `skipif` guard fix.**
  Total effort: ~30 min. Net: 10 fragile tests -> 6 robust + 4 cleanly-
  skipped. Surfaced during the PR 25 (feature-view ensemble clean
  re-eval) data-wipe recovery on 2026-05-05; not blocking but recurs
  on every fresh worktree setup.

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
