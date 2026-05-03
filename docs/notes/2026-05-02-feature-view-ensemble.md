# Feature-View Diversity Ensemble -- Findings

**Date:** 2026-05-02
**Branch:** feat/feature-view-ensemble
**Verdict:** **NO-GO** -- pre-sweep gate failed on 2 of 3 clauses. v9-C / v4 stay in production.
**Spec:** `docs/superpowers/specs/2026-05-02-feature-view-ensemble-design.md`
**Plan:** `docs/superpowers/plans/2026-05-02-feature-view-ensemble.md`

## TL;DR

Tested same-class XGBoost peers on disjoint feature subsets (PEER_A:
40 team-strength features; PEER_B: 27 form/market/meta features) as
a stage-1 ensemble alternative to v4. Pre-sweep gate FAILED at 2 of
3 clauses: PEER_A standalone LL `0.5720` was `+0.1375` above v4's
`0.4345` (clause 1 fails by `5.5x` the tolerance), and the optimal
2-blend headroom was *negative* at `-0.0206` (clause 3 fails). The
disjoint-view hypothesis was *confirmed* on the decorrelation axis
-- residual correlation `r = 0.45 < 0.60` -- but PEER_A's standalone
weakness sinks the ensemble. The 15-cell sweeps for E1 and E2 were
not run; saved ~90-150 minutes of compute.

## Setup recap

The experiment tested whether two same-class XGBoost peers, trained
on disjoint subsets of v4's 67-feature matrix, produce a stage-1
ensemble that beats v4 standalone. Spec partition (locked at design
time):

- **PEER_A (team strength, 40 features)**: adjusted efficiency,
  four factors, KenPom full-season metrics, Massey orderings,
  conference strength, season summary stats.
- **PEER_B (form + market + meta, 27 features)**: late-season
  efficiency, trajectory, rolling form, conference tournament,
  coach meta, Vegas market.

Two ensemble variants were planned for the post-gate sweep:

- **E1 (clean test)**: ensemble = blend(peer_A, peer_B), no v4.
  The cleanest falsification of the disjoint-view hypothesis.
- **E2 (production-shape)**: ensemble = blend(v4, peer_A, peer_B),
  v4 in the blend.

The 3-clause pre-sweep gate (`src/diagnose_feature_view_ensemble.py`)
isolated the three known stage-1-diversity failure modes:
- **Clause 1 (per-peer LL ceiling)**: each peer LL within `0.025` of
  v4. Catches "peer too thin to contribute" (PR 12 BT-ensemble lesson).
- **Clause 2 (residual correlation)**: `rho(resid_A, resid_B) < 0.60`.
  Catches "errors too coupled" (PR 11 LR-ensemble lesson).
- **Clause 3 (best-blend headroom)**: optimal 2-blend beats v4 by
  `>= 0.001` LL. Catches "no signal lift" (PR 13 BT-as-feature lesson).

All three clauses must PASS for the sweep to run.

## Pre-sweep gate result

The verdict comes straight from `output/diag_feature_view_ensemble.json`:

| measure | value | clause |
|---|---|---|
| n played games | 1449 | sanity |
| v4 LL | **0.4345** | baseline |
| peer_A LL | **0.5720** (delta_a `+0.1375`) | **FAIL** clause 1 |
| peer_B LL | **0.4566** (delta_b `+0.0221`) | PASS clause 1 |
| 2-blend optimal LL | **0.4551** | -- |
| 2-blend optimal weights `(w_A, w_B)` | `(0.083, 0.917)` | -- |
| 3-blend optimal LL | 0.4316 | (info only) |
| 3-blend optimal weights `(w_v4, w_A, w_B)` | `(0.757, 0.000, 0.243)` | (info only) |
| residual correlation `rho(A, B)` | **+0.447** | **PASS** clause 2 |
| 2-blend headroom vs v4 (`ll_v4 - ll_2blend`) | **-0.0206** | **FAIL** clause 3 |
| **gate verdict** | -- | **FAIL on clauses 1 + 3** |

Clause 1 fails at PEER_A: peer_A is `5.5x` over the `0.025`
ceiling. Clause 3 fails by a wide margin (`-0.0206` is `20x` past the
`+0.001` threshold and on the wrong side). Clause 2 passes
comfortably (`r = 0.45` is well below the `0.60` ceiling).

## Falsification reasoning

Three observations from the diagnostic:

1. **PEER_A is dramatically weaker than v4 standalone.** v4 has
   `0.4345` LL across 1449 played games; PEER_A on 40 of v4's 67
   features lands at `0.5720`. That's a `+0.1375` per-game LL gap on
   a paired comparison -- well past noise. Counterintuitively, PEER_A
   contains the features I'd expect to drive raw predictive
   power (KenPom efficiency, Massey orderings, four factors,
   adjusted O/D efficiency). Yet the trained model on these alone
   is worse than v4 by an enormous margin. Hypothesis: the
   information density across PEER_A's 40 features is highly
   redundant (many KenPom + Massey columns rank-encode the same
   underlying team-strength signal), so XGBoost on this subset
   ends up under-regularizing on a low-effective-dimension input.
   PEER_B's "form + market + meta" features carry additional
   tournament-conditional signal (Vegas spreads encode market
   priors; late-season efficiency captures recency that full-season
   averages wash out) that PEER_A lacks entirely.

2. **PEER_B is much closer to v4 than expected.** PEER_B at `0.4566`
   with only 27 features is within `0.025` of v4's `0.4345`. This
   passes the per-peer ceiling clause comfortably. The "form + market"
   view alone captures most of v4's predictive power -- which is a
   strong domain finding in itself, even though it's a side effect
   here.

3. **Residual correlation `r = 0.45` is below the `0.60` clause-2
   threshold.** Disjoint feature views *do* produce decorrelated
   errors, exactly as the design predicted. The structural mechanism
   works. But mechanism without strength is insufficient -- a
   uniformly-decorrelated weak peer doesn't help against a strong
   single-model baseline.

The optimal 2-blend places `0.917` weight on PEER_B and only `0.083`
on PEER_A -- i.e., the blend mostly *is* PEER_B. Even at this
"throw away PEER_A" weighting, the blend's `0.4551` LL still loses
to v4's `0.4345` because PEER_B is by itself weaker than v4. So
clause 3 fails not because the blend is poorly weighted, but because
neither peer (and no convex combination of them) reaches v4's level.

The 3-blend optimum is more interesting as a side observation. With
v4 in the mix, the optimum is `(w_v4=0.757, w_a=0.000, w_b=0.243)` at
LL `0.4316`. PEER_A's optimal weight is exactly `0.000` -- the
optimizer assigns it no signal at all. PEER_B contributes `24%` of
the blend, producing a `+0.0029` LL improvement over v4 alone (within
gate precision but not a clause that was tested for E1). This is
*architecturally suggestive* for follow-up work but is not the
experiment that was gated; E2's headroom would need its own
end-to-end bracket-points evaluation to be a candidate.

The headroom clause fails for E1 (which was the gated variant), so
the spec's disposition matrix says: stop, write findings, NO-GO.

## Comparison to predecessors

| experiment | mechanism | gate or sweep failure mode | LL signal |
|---|---|---|---|
| LR ensemble (PR 11) | global avg, fixed w on identical features | residual correlation `r=0.77` too high | (no LL gate; sweep ran, lost on brkt pts) |
| BT ensemble (PR 12) | global avg, fixed w on different model class | BT too weak standalone (LL 0.565 vs 0.437) | optimal `w=0.98`, headroom `+0.0000` |
| BT-as-feature (PR 13) | learned per-context weight (v9-C as gate) | v9-C can't extract gating from a noisy added feature | LL gate FAIL, headroom `-0.0015` |
| **Feature-view ensemble (PR 14, this experiment)** | **same model class, disjoint feature views** | **PEER_A standalone weak (LL 0.572) AND best 2-blend headroom negative (`-0.0206`)** | **clause 1 FAIL, clause 3 FAIL** |

Position in the falsification grid: PR 14 is the third closure of
the diversity-stage-1 program, and it closes a different cell than
its predecessors. PR 11 ruled out "diversity at identical features."
PR 12 ruled out "structural diversity at low standalone strength."
PR 13 ruled out "v9-C as a learnable trust weight on a noisy
feature." PR 14 rules out "**disjoint feature views split by
semantic role,** when the partition produces a peer with
fewer-than-v4 features that is much weaker than v4 standalone."

The hypothesis was not entirely falsified -- the residual-correlation
clause *passed*, validating that disjoint inputs decorrelate errors
by construction. What failed is the per-peer strength assumption.
Specifically, the spec's per-peer-LL-ceiling threshold of `0.025`
was calibrated on the assumption that XGBoost's information curve
flattens out well before 40 features in this domain. The actual
result for PEER_A (`+0.1375`) shows the curve is much steeper than
that. Either the partition's PEER_A is missing something load-bearing
that lives in PEER_B, or there is no 2-way semantic split of v4's
features that yields two peers each within `0.025` of v4.

## Verdict

NO-GO. v9-C / v4 stay in production. The trainer extension
(`src/train_peer_stage1.py`), the gate diagnostic
(`src/diagnose_feature_view_ensemble.py`), the K-way blend helper
(`src/ensemble_stage1.py:blend_pairwise_csvs`), and the
`V9_STAGE1_PAIRWISE` env-var override in `src/sweep_v9_weights.py`
remain on the branch as the experiment record and as reusable
infrastructure for any future feature-view experiment. The committed
peer pairwise CSVs (`output/pairwise_peer_a.csv`,
`output/pairwise_peer_b.csv`, `output/diag_feature_view_ensemble.json`)
are frozen artifacts of this run.

Saved compute: ~90-150 minutes by gating before the E1 + E2 sweeps.

## Recommendation

Active queue advances. The disjoint-feature-view route is closed for
the K=2 semantic-split variant. Two related questions are *partially*
open:

- **K-way feature partitions or different splits.** The 27-feature
  PEER_B was within ceiling; a different 2-way cut, or a K=3
  partition, *might* land two peers each within ceiling. But the
  domain-knowledge signal here is that "team strength" features
  alone (PEER_A's 40 cols) carry far less predictive power than
  expected. A clean follow-up would be a 67/3 partition along
  different axes (e.g., information-source rather than role:
  KenPom-only / Massey-only / Vegas-only / efficiency-only). Each
  peer would be smaller still and likely weaker, so this is not
  obviously a recovery.
- **3-blend with v4 (E2)**: the side observation that
  `w_3blend = (0.757, 0.0, 0.243)` produces LL `0.4316` (vs v4's
  `0.4345`, headroom `+0.0029`) suggests PEER_B might be a useful
  *additive* feature on top of v4 -- but that's the BT-as-feature
  pattern PR 13 already falsified at this data scale (v9-C can't
  extract per-context gating from one extra feature). Worth flagging
  as a future "PEER_B-as-feature" experiment but the prior is weak.

Active queue items advance:

1. **Hierarchical Bradley-Terry with feature priors** (formerly #2,
   now top of queue). `s_team ~ Normal(beta . v4_features_team, sigma)`.
   Couples BT back to the v4 feature view to gain standalone
   strength. Risk: residual correlation may regress from the 0.58
   PR 12 measured back toward 0.77 as the models re-converge. With
   feature-view ensemble now closed at K=2 semantic split, this is
   the next angle on getting a stronger BT signal that survives
   ensemble criteria.
2. **Small NN (MLP) as stage-1.** Adds PyTorch tooling cost; same
   correlated-error caveat as LR if it reuses the same feature
   matrix. After hierarchical BT.
3. **Full Bayesian Bradley-Terry with strength + variance per team.**
   Standalone-strength bottleneck likely persists.
4. **External rankings** (538, KenPom-public, etc.) -- defer.
5. **Roster-level returning-experience** -- needs external data.

Item #1 of the previous queue (this experiment) moves to "Tried
and rejected."

## Files of record

```
src/feature_views.py                        -- partition lists + validate_partition (new)
src/train_peer_stage1.py                    -- per-peer XGBoost LOSO trainer (new)
src/diagnose_feature_view_ensemble.py       -- 3-clause pre-sweep gate (new)
src/ensemble_stage1.py                      -- blend_pairwise_csvs K-way generalization (extended)
src/sweep_v9_weights.py                     -- V9_STAGE1_PAIRWISE env var override (extended)

output/pairwise_peer_a.csv                  -- 22-LOSO-season OOF predictions (PEER_A, 48,465 rows)
output/pairwise_peer_b.csv                  -- 22-LOSO-season OOF predictions (PEER_B, 48,465 rows)
output/diag_feature_view_ensemble.json      -- gate diagnostic with verdict (E1 + E2 sweeps NOT run)

tests/test_feature_views.py                 -- 5 partition tests (new)
tests/test_train_peer_stage1.py             -- 4 trainer tests (new)
tests/test_diagnose_feature_view_ensemble.py -- 10 gate tests, all 3 clauses covered (new)
tests/test_ensemble_stage1.py               -- 4 K-way blend tests added
tests/test_sweep_v9_weights.py              -- 2 env var tests added
```

The gate diagnostic's threshold constants live at the top of
`src/diagnose_feature_view_ensemble.py`:
`PEER_LL_CEILING_DELTA = 0.025`, `RESID_CORR_MAX = 0.60`,
`HEADROOM_MIN = 0.001`. Each is annotated with the prior-experiment
failure mode it maps to. Re-running the gate against any future
peer pairwise CSVs is just
`python src/diagnose_feature_view_ensemble.py --pairwise-peer-a NEW_A.csv --pairwise-peer-b NEW_B.csv`.
