# Feature-View Ensemble PEER_A/B Re-eval (Clean Baseline) -- Findings

**Date:** 2026-05-05
**Branch:** feat/feature-view-clean-rerun
**Verdict:** **FAIL.** All three clauses fail. delta_a=+0.0140 PASS but delta_b=+0.0277 FAIL clause 1; rho=+0.726 FAIL clause 2; 2-blend headroom=-0.0084 FAIL clause 3. Robust NO-GO across both leaky and clean baselines.
**Spec:** `docs/superpowers/specs/2026-05-05-feature-view-clean-rerun-design.md`
**Plan:** `docs/superpowers/plans/2026-05-05-feature-view-clean-rerun.md`
**Recovery context:** TODO.md "CONTAMINATION DISCOVERED 2026-05-04" -> step 5,
sub-priority "Feature-view ensemble PEER_A/B re-eval" (named highest signal/noise
of the marginal-rejections list remaining after plain BT was closed in PR 24).

## TL;DR

Re-running PR 14's 3-clause pre-sweep gate against clean v4 (PR 23 force-add)
plus a freshly-trained clean PEER_B keeps the NO-GO verdict and **deepens the
failure**: PR 14 failed clauses 1 and 3 (PEER_A standalone weakness +
negative-headroom 2-blend) while passing clause 2 (rho=0.45). This PR fails
*all three* clauses. The two pre-registered prediction targets split:
**PEER_A's clause 1 individually flipped PASS** as the spec predicted
(`delta_a=+0.0140 < 0.025`), but PEER_B's clause 1 individually flipped
**FAIL** (`delta_b=+0.0277 > 0.025`) -- PEER_B got *worse* in standalone LL
on the clean baseline by more than v4 did. The material new finding is
**clause 2 (residual correlation) flipping from rho=0.45 to rho=+0.726** --
the same direction and similar magnitude as PR 24's plain-BT finding
(rho 0.577 -> 0.868 between v4 and BT residuals). Two distinct same-class
peer pairs (PEER_A vs PEER_B; v4 vs BT) both show the residual-correlation
jump; the generalizing hypothesis ("residual correlation between any two
peers is bounded below by what they all miss for hard
regular-season-information reasons") gets a second clean-baseline data
point. Feature-view ensemble at K=2 semantic split is closed; the
marginal-rejections list shrinks by one.

## Methods

- Inputs (read-only):
  - `output/pairwise_v4.csv` (clean baseline, force-added in PR 23,
    md5 `795d8ddfcd7a0a09a50c3732825c6316`).
  - `output/pairwise_peer_a.csv` (PR 14 force-add, md5 `950624e528d506964994f35380ceffc1`).
    **Reproducibility check passed (Task 1):** a fresh `train_peer_stage1.py
    --peer a` rerun produced byte-identical output. PEER_A is unchanged
    across the leaky->clean transition by construction (its 40 features
    contain no Vegas columns) and the byte-compare confirms the trainer
    + feature pipeline produce the same artifact.
  - `output/pairwise_peer_b.csv` (re-emitted this PR, md5 `dd672bc28b0d9f35b232e3f99224af06`;
    was `422d9960a6a83e60e9935f0bd55217c4` under leaky baseline).
- Diagnostic: `python src/diagnose_feature_view_ensemble.py`. Same
  thresholds as PR 14 (`PEER_LL_CEILING_DELTA=0.025`, `RESID_CORR_MAX=0.60`,
  `HEADROOM_MIN=0.001`). No source-code changes this PR.
- Matched-game count: `n_played_games = 1449` (identical to PR 14 + PR 24).
- Setup-side note: this PR's worktree was the first work after the
  2026-05-04 data wipe required a top-level rerun. `data/raw/march-machine-learning-2026/`
  was empty when the worktree was created (the May 4 wipe truncated the
  subdir; PR 23/24 had restored it for those branches but the main-repo
  copy was empty by the time this PR started). Recovered by `tar -xzf
  data/training_data.tar.gz` per `docs/data_recovery.md` step 3, then
  set up subdir-level junctions in the worktree (step 5). 35 files in
  `march-machine-learning-2026/`, 38 in `kaggle/`, 23 in `vegas_lines/`
  match the runbook's expected counts.

## Gate result

| measure                                  | value          | clause           |
|------------------------------------------|----------------|------------------|
| ll_v4                                    | 0.5579         | baseline         |
| ll_peer_a (delta_a vs v4)                | 0.5720 (+0.0140) | PASS individually (< 0.025) |
| ll_peer_b (delta_b vs v4)                | 0.5857 (+0.0277) | **FAIL** individually (> 0.025) |
| **per_peer_ll_ceiling clause**           | -              | **FAIL** (any peer over ceiling fails) |
| Pearson rho(resid_A, resid_B)            | **+0.726**     | **FAIL** (>= 0.60) |
| 2-blend optimal weights (w_A, w_B)       | (0.658, 0.342) | -                |
| 2-blend optimal LL                       | 0.5664         | -                |
| 2-blend headroom = ll_v4 - ll_2blend     | **-0.0084**    | **FAIL** (< 0.001) |
| **gate verdict**                         | -              | **FAIL** (all 3 clauses) |

3-blend side observation:
- weights (w_v4, w_A, w_B) = (0.924, 0.000, 0.076)
- ll_3blend_optimal = 0.5577
- 3-blend headroom vs v4 = +0.0002 (within `HEADROOM_MIN=0.001`; not a
  clause this gate scores)

## Standalone metrics (1449 played 2003-2025 tournament games)

| metric        | clean v4 | PEER_A   | PEER_B   |
|---------------|----------|----------|----------|
| weighted LL   | 0.5579   | 0.5720   | 0.5857   |
| delta vs v4   | -        | +0.0140  | +0.0277  |

Under the clean baseline the ranking is **v4 > PEER_A > PEER_B** (best LL
to worst). Under the leaky baseline (PR 14) the ranking was
**v4 > PEER_B > PEER_A**. The crossover is the most surprising structural
shift this PR: PEER_A is now the closer-to-v4 peer despite being
the much weaker peer pre-fix.

## Comparison to PR 14 (leaky baseline)

| measure                    | PR 14 (leaky)        | this PR (clean)        | delta              |
|----------------------------|----------------------|------------------------|--------------------|
| ll_v4                      | 0.4345               | 0.5579                 | +0.1234 (worse)    |
| ll_peer_a                  | 0.5720               | 0.5720                 |  0.0000 (byte-equal)|
| ll_peer_b                  | 0.4566               | 0.5857                 | +0.1291 (worse)    |
| delta_a (peer_a - v4)      | +0.1375              | +0.0140                | -0.1235 (clause 1 individual flip PASS) |
| delta_b (peer_b - v4)      | +0.0221              | +0.0277                | +0.0056 (small but flips clause 1 individual) |
| rho(resid_A, resid_B)      | +0.447               | **+0.726**             | +0.279 (clause 2 flip FAIL) |
| 2-blend optimal weights    | (0.083, 0.917)       | (0.658, 0.342)         | center-of-simplex shift |
| 2-blend ll                 | 0.4551               | 0.5664                 | +0.1113            |
| 2-blend headroom           | -0.0206              | -0.0084                | +0.0122 (closer to passing but still negative) |
| 3-blend weights (v4,A,B)   | (0.757, 0.000, 0.243)| (0.924, 0.000, 0.076)  | v4 weight up; B weight down |
| 3-blend ll                 | 0.4316               | 0.5577                 | +0.1261            |
| gate verdict               | FAIL (clauses 1, 3)  | FAIL (clauses 1, 2, 3) | failure deepened   |

PEER_A's row is invariant in LL (byte-equal csv -> identical join -> identical
LL). v4 and PEER_B both shifted; v4 by +0.1234 (matches PR 21's clean-LOSO
finding), PEER_B by +0.1291 (~0.006 *more* than v4 -- PEER_B lost slightly
more standalone strength than v4 did when Vegas features became
pre-tournament-only).

## Discussion

**The pre-registered split came back half-correct.** The spec's
prediction was "PEER_A clause 1 likely flips PASS"; that held
(`delta_a` went from +0.1375 to +0.0140). What the spec did not predict
was that PEER_B would flip clause 1 in the *opposite* direction: PEER_B's
LL got worse by ~0.006 more than v4's did, pushing it just over the
0.025 ceiling. PEER_B's relative-to-v4 weakening is small in absolute
terms but enough to flip the binary clause.

The mechanistic story for PEER_B's marginal extra weakening: PEER_B's 27
features include 7 Vegas features whose values changed under PR 19's
filter (`vegas_avg_spread`, `vegas_avg_margin`, `vegas_ats_pct`,
`vegas_power_rating`, `vegas_consistency`, `vegas_game_count`,
`vegas_late_spread_delta`). Under leaky training, these features carried
within-season tournament-game signal that PEER_B's XGBoost classifier
extracted on top of its 20 non-Vegas features. Under clean training,
the Vegas features still encode pre-tournament market priors but lost
the tournament-game leak. The features are now noisier signals of
the same underlying market view, and PEER_B's classifier can no longer
extract the leak signal that previously made it competitive with v4.
v4 itself lost the same Vegas leak (and by the same mechanism) but v4
also has the 40 non-Vegas features PEER_A trained on, which carry
some opponent-adjusted signal that helps v4 maintain a slightly
larger LL margin over PEER_B than the leaky baseline showed.

**The residual-correlation flip is the material finding.** PR 14
measured `rho(resid_A, resid_B) = +0.447` -- well below the 0.60
threshold -- and the original write-up framed it as "the disjoint-view
mechanism works on the decorrelation axis, just not on the strength
axis." This PR measures `rho = +0.726`, well above the threshold.
The two PEER models -- same XGBoost class, disjoint feature subsets,
same target -- now produce strongly correlated residuals on the same
matched-game set.

The mechanistic story is the same as PR 24's plain-BT finding: when
v4-class features lose the within-season tournament-game leak, the
residual landscape is dominated by games that *every* same-data peer
fails on -- "hard regular-season-information" games. PR 24 measured
`r(resid_v4, resid_bt) jump 0.577 -> 0.868` for v4 versus a
structurally-different peer (BT vs XGB). This PR measures
`rho(resid_A, resid_B) jump 0.45 -> 0.73` for two same-class peers
trained on disjoint feature subsets. Different model pairs, similar
shift direction and magnitude.

**Generalizing hypothesis (now with two clean-baseline data points).**
PR 24's findings note named a hypothesis worth pre-registering:
*residual correlation between v4 and a peer is bounded below by what
both models miss for "hard regular-season-information" reasons*. The
PEER_A vs PEER_B pair generalizes that statement to "any two same-data
peers of clean v4." If both feature-view peers and a structurally-
different BT peer all hit r >= 0.7 against either v4 or each other on
the clean baseline, that is concrete evidence the falsification gate's
clause 2 is sharp -- in the K=2 ensemble program, candidates fail
clause 2 by default unless they bring genuinely-out-of-distribution
information (something neither the v4 feature stack nor a regular-
season-binary-outcome BT model carries). The remaining marginal-
rejection candidates (HBT, Colley, Massey-decay-hl=14d) are all
v4-feature-based or v4-class-feature-based and inherit the same
expected-r profile. **The clause 2 prediction for those candidates
is now a strong NO-GO prior**, not just a baseline-shift question.

**The 3-blend optimum tells the same story from the other side.**
Under leaky baseline the 3-blend was `(w_v4=0.757, w_A=0.000, w_B=0.243)`
at LL 0.4316 -- v4 dominated, PEER_A contributed nothing,
PEER_B got 24% as a residual feature. Under clean baseline the
3-blend is `(w_v4=0.924, w_A=0.000, w_B=0.076)` at LL 0.5577 -- v4
dominates more, PEER_A still contributes nothing, PEER_B's
contribution shrunk from 24% to 7.6% with a measured headroom of
+0.0002 (within `HEADROOM_MIN`). PR 14 floated PEER_B-as-feature for
v4 as a future experiment; this re-eval falsifies that escape hatch
on the clean baseline. PEER_B's marginal value as a residual on top
of v4 was already small under leaky (+0.0029 LL) and is now within
gate-precision of zero.

**Why the 2-blend headroom moved closer to passing without crossing.**
The 2-blend (PEER_A, PEER_B) headroom went from -0.0206 to -0.0084.
That is closer to zero than the leaky baseline by +0.0122. Mechanistic
read: under leaky, the 2-blend was dominated by PEER_B (`w_B=0.917`)
because PEER_B was the much stronger peer; under clean, the optimum
weights are nearly balanced (`w_A=0.658, w_B=0.342`) because the two
peers have similar standalone strength. The 2-blend gets to take the
average, which is closer to v4 than either peer alone -- but v4
remains the strict winner because PEER_A and PEER_B's residuals are
correlated enough (rho=0.726) that averaging them recovers most but
not all of the gap. If rho had stayed at PR 14's 0.447, the headroom
might have crossed zero; the residual-correlation flip is what keeps
clause 3 in the FAIL band.

## Verdict + recommendation

**NO-GO.** The feature-view ensemble at K=2 semantic split fails the
gate on the clean baseline, with all three clauses failing. The robust
NO-GO across both leaky and clean baselines (with the failure pattern
*deepening* under clean rather than flipping or relaxing) closes the
K=2 semantic-split feature-view as a stage-1 ensemble peer for v4.

The marginal-rejections list shrinks by one. Per the priority order in
`docs/notes/2026-05-04-v9c-clean-rerun.md` § Follow-ups, after PR 24
closed plain BT and this PR closes feature-view, the next sub-priority
is **Hierarchical BT (PR 16) re-eval** (~5 min compute). The HBT spec's
rho-prediction was already weak (rho 0.45-0.51 leaky; PR 24 + this PR
suggest clean rho will likely jump similar to BT's pattern). HBT's
expected verdict on the clean baseline: clause 1 FAIL (HBT standalone
LL 0.619-0.757, far weaker than clean v4's 0.5579), clause 2 likely
FAIL (rho jump). HBT is a much weaker prior than feature-view was,
so the HBT re-eval is mostly closing the marginal-rejections list
cleanly rather than expecting a flip.

After HBT, the remaining candidates are Colley (clause-2 delta +0.0053
LL leaky) and Massey-decay-hl=14d (clause-2 delta +0.0057 LL leaky).
Both were borderline on a *different* gate (the per-feature-add cheap
diagnostic) -- they may or may not flip. Whether to re-eval them
depends on whether Active queue items 1-3 (538 audit, single-season
v4 variance, external rankings as features) move ahead of finishing
the marginal-rejections re-eval pass. Per TODO.md's re-prioritization
(Kaggle 2159/3462 finish), the audits/metric corrections are arguably
higher-priority signal than closing the last two marginal rejections.

## TODO.md update (this PR commits the update)

- Sub-priority "Feature-view ensemble PEER_A/B re-eval" marked DONE
  with FAIL verdict + headline numbers.
- Marginal-rejections list: feature-view PEER_A/B entry replaced with
  the closure note. Plain BT, feature-view, and HBT (when re-eval
  closes) shrink the list to two: Colley + Massey-decay-hl=14d.
- The "next sub-priority" advancement points at HBT (~5 min compute).

## Files of record

- `output/diag_feature_view_ensemble.json` (overwritten with clean numbers;
  force-added in commit 46ae3a1).
- `output/pairwise_peer_b.csv` (overwritten with clean run; same commit).
- `output/pairwise_peer_a.csv` -- unchanged (PR 14 byte-equal under clean
  baseline; tracked file from PR 14 stays).
- `docs/superpowers/specs/2026-05-05-feature-view-clean-rerun-design.md`
  (commit 2058a37).
- `docs/superpowers/plans/2026-05-05-feature-view-clean-rerun.md`
  (commit 9f079c5).
- (no source-code changes this PR -- the diagnostic, trainer, and partition
  scripts on main from PR 14 were used as-is)

## Open follow-up (not for this PR)

The K=2 semantic-split partition is closed. PR 14's findings note
left two related angles partially open which this PR's verdict does
NOT change:

- **K=3 partition by information source.** The PR 14 note suggested
  e.g. KenPom-only / Massey-only / Vegas-only / efficiency-only as a
  different cut. Each peer would be smaller still, almost certainly
  weaker individually, and would inherit the clause-2 prior from this
  PR's residual-correlation finding. Bayesian update: under the
  generalizing hypothesis, K=3 same-data peers will have pairwise rho
  >= 0.7 unless one peer brings out-of-distribution information.
  K=3 information-source partition does not bring out-of-distribution
  information (all the views are read off v4's existing inputs).
  Effective prior: K=3 will fail clause 2 by structural reasons that
  this PR strengthened. Lower priority than the audits / metric
  corrections in the active queue.
- **PEER_B-as-feature for v4.** This PR's 3-blend `w_3blend_b=0.076`
  with headroom +0.0002 falsifies this on the clean baseline, but
  the falsification is gate-precision-tight rather than emphatic.
  If a future external-features experiment (538 / external rankings)
  surfaces a feature with similar headroom signature on top of v4,
  the PEER_B-as-feature pattern is structurally similar -- worth
  noting as a calibration data point even though this PR closes
  PEER_B specifically.

The two clean-baseline `rho` jumps (PR 24's 0.577 -> 0.868; this PR's
0.45 -> 0.73) are the more important data points. They suggest the
ensemble-peer search at this data scale is bounded above by a
structural ceiling we now have two measurements of, not by clever
choice of peer. Further peer-search experiments without a way to
beat the structural ceiling are predictable failures; the active
queue's audit-and-improve-v4 direction is the higher-EV path.
