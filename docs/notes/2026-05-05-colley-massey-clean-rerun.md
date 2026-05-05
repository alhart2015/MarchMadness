# Colley + Massey-Decay-14d Clean Re-eval -- Colley FLIPPED PASS

**Date:** 2026-05-05
**Branch:** feat/colley-massey-clean-rerun
**Spec:** `docs/superpowers/specs/2026-05-05-colley-massey-clean-rerun-design.md`
**Plan:** `docs/superpowers/plans/2026-05-05-colley-massey-clean-rerun.md`
**Predecessors:**
- Original Colley findings (clause 2 FAIL): `docs/notes/2026-05-03-colley.md`
- Original Massey-decay findings (clause 2 FAIL at hl=14d): `docs/notes/2026-05-03-massey-mov.md`
- Clean LOSO regen: `docs/notes/2026-05-04-v4-clean-loso-regen.md`
- Pattern predecessors: PR 24 (`docs/notes/2026-05-04-v9c-clean-rerun.md`),
  PR 25 (`docs/notes/2026-05-05-feature-view-clean-rerun.md`),
  PR 26 (`docs/notes/2026-05-05-hbt-clean-rerun.md`)
**Verdict:**
- **Colley clause 2: PASS** under clean v4 baseline (mean delta -0.0100 vs threshold +0.001). Flipped from PR 15 leaky FAIL (+0.0053). All three subset seasons help.
- **Massey-decay-14d clause 2: FAIL** (mean delta +0.0018 vs threshold +0.001). Marginal -- shrunk from PR 15 leaky +0.0057 but still above the bar. Mixed per-season pattern.

## TL;DR

The pre-registered prediction in the spec was FAIL/FAIL on the basis
that "redundancy is structural, not threshold-tight" -- both candidates
had passed clause 1 against `adj_em` and `massey_composite` by clear
margins under PR 15 and the leak fix doesn't change the rating columns.
**That prediction was wrong for Colley.** When clean v4 lost its Vegas-
leak signal, its 67-feature stack stopped extracting enough opponent-
adjusted strength signal that Colley's W/L-only rating started carrying
genuine marginal information; the per-season delta flipped from +0.0073
/ +0.0135 / -0.0051 (leaky) to -0.0166 / -0.0074 / -0.0059 (clean), a
mean shift of -0.0153 LL. Colley is promoted to recovery step 5's
"next-immediate full LOSO backtest" sub-priority per the spec's
decision matrix row 3. Massey-decay-14d also shifted in the same
direction (mean -0.0039 LL) but the per-season pattern is much messier
(2019 helped a lot, 2024 hurt the most -- inverted vs leaky), and the
mean delta is still +0.0018 above the +0.001 threshold; closes as
robust NO-GO across the marginal-rejections list.

## Methods

**Pipeline.** `src/clause2_colley.py` (NEW; mirrors
`src/clause2_decay_massey.py`) and `src/clause2_decay_massey.py 14`
both invoked `prepare_loso_inputs()` from `src/enhanced_model_v3.py`,
which now flows through PR 19's `filter_vegas_to_pre_tournament()`.
Each computes its rating column inline (Colley via
`compute_colley_ratings(regular)`; Massey via
`compute_massey_mov_ratings(regular, mov_cap=21, half_life_days=14.0)`),
merges into `fm` on (TeamID, Season), and runs
`leave_one_season_out_cv_weighted` twice on subset {2019, 2022, 2024}
with `allowed_holdouts=GATE_SUBSET_SEASONS` -- once with the column in
`feature_cols`, once without.

**Threshold.** `LL_HEADROOM_MAX = +0.001` LL, inherited verbatim from
PR 15 (defined in `src/diagnose_colley.py` and `src/diagnose_massey_mov.py`,
imported into both runners).

**Worktree setup.** `feat/colley-massey-clean-rerun` off `main`
(d5846d9, post-PR-26) with subdir-level junctions for
`data/raw/march-machine-learning-2026/`, `data/raw/kaggle/`,
`data/raw/vegas_lines/`. Required `tar -xzf data/training_data.tar.gz
-C data/raw/` to repopulate the wiped subdirs in main repo before
junctioning -- same wipe pattern recurring on every fresh worktree
setup (engineering follow-up at TODO.md "Test-suite hygiene").

**Cold-start.** First runner (Colley) paid the full
`prepare_loso_inputs()` cold-start (~4 min) including rebuilding
`data/cache/colley_ratings.parquet`,
`data/cache/massey_mov_ratings.parquet`, and the efficiency caches.
Second runner (Massey-decay-14d) hit warm parquet caches; cold-start
on `prepare_loso_inputs()` was still paid because it's a separate
Python process, but per-rating-table caches were warm.

**Sanity gate -- `mean_ll_without` agreement.** Both runners ran the
same v4 LOSO with the same 67-feature stack on the same 3-season
holdout subset. The expected agreement was within ~1e-6 LL (XGB tuned-
params noise floor across processes). **Actual agreement: 0.00e+00 --
byte-identical.**

```text
Colley:  mean_ll_without = 0.5907361657002678
Massey:  mean_ll_without = 0.5907361657002678
diff     = 0.00e+00
```

The without-arm is fully deterministic between separate Python
processes given the same fm shape, the same `MM_TUNED_PARAMS_V3`-
equivalent params (here: the v3 default tuned params, since neither
runner sets `MM_TUNED_PARAMS_V3`), and the same seed handling in the
trainer. Strengthens the side-by-side delta comparison: any
with-vs-without LL difference is fully attributable to the added
column, not to baseline drift.

## Colley clause 2 -- PASS

| season | leaky `ll_with` | leaky `ll_without` | leaky delta | clean `ll_with` | clean `ll_without` | clean delta |
|---|---|---|---|---|---|---|
| 2019 | 0.3819 | 0.3746 | +0.0073 | 0.5227 | 0.5393 | **-0.0166** |
| 2022 | 0.5177 | 0.5042 | +0.0135 | 0.6305 | 0.6378 | **-0.0074** |
| 2024 | 0.4325 | 0.4376 | -0.0051 | 0.5891 | 0.5951 | **-0.0059** |
| **mean** | **0.4440** | **0.4388** | **+0.0053 FAIL** | **0.5808** | **0.5907** | **-0.0100 PASS** |

All three subset seasons now help. The biggest individual flip is 2022
(+0.0135 -> -0.0074, a -0.0209 swing) -- the season that hurt the most
under leaky baseline now helps. 2019 also flipped from hurt to help
(-0.0239 swing). 2024 was already marginally helping under leaky and
helps slightly more on clean.

The clean-baseline `ll_without` shifted from 0.4388 to 0.5907 -- v4
lost ~0.152 LL on this 3-season subset (vs 0.122 on the 22-season
aggregate per PR 21). The subset-vs-aggregate gap (+0.030) is
attributable to Vegas's leak being unevenly distributed across seasons,
and 2019/2022/2024 are above-average leak sensitivity (specifically:
2024 was +0.177 LL in PR 21's per-season-shift table, the second-
biggest leak shift in 22 seasons).

## Massey-decay-14d clause 2 -- FAIL (marginal)

| season | leaky `ll_with` | leaky `ll_without` | leaky delta | clean `ll_with` | clean `ll_without` | clean delta |
|---|---|---|---|---|---|---|
| 2019 | 0.3866 | 0.3746 | +0.012 | 0.5272 | 0.5393 | **-0.0121** |
| 2022 | 0.5111 | 0.5042 | +0.007 | 0.6424 | 0.6378 | **+0.0046** |
| 2024 | 0.4357 | 0.4376 | -0.002 | 0.6081 | 0.5951 | **+0.0131** |
| **mean** | **0.4445** | **0.4388** | **+0.0057 FAIL** | **0.5926** | **0.5907** | **+0.0018 FAIL** |

Per-season pattern is **inverted on 2019 and 2024 vs leaky**:

- 2019: leaky +0.012 hurt -> clean -0.0121 helps (a -0.024 swing,
  similar magnitude to Colley's 2019 swing).
- 2022: leaky +0.007 hurt -> clean +0.0046 still hurts (a -0.002
  swing -- mostly unchanged).
- 2024: leaky -0.002 marginal help -> clean +0.0131 hurts the most
  (a +0.015 swing -- the *opposite* direction).

This makes Massey-decay-14d's clean-baseline mean delta a small
positive (+0.0018), well below the +0.005 ballpark of the leaky
baseline but above the +0.001 threshold. Closer to the threshold
than Colley's pre-fix +0.0053 was, but on the wrong side.

The 2024 inversion is the load-bearing detail. 2024 was the leakiest
season per PR 21 and the season that helped Massey-decay-14d under
leaky baseline (where 2024's UConn / Purdue win-margin signal was
already in v4's vegas_avg_margin); on the clean baseline, 2024 is
also the season where v4 weakens the most, so margin-decayed Massey
*could* have been expected to help most. It hurts most instead. The
likely reading: Massey-decay-14d's last-2-weeks margin signal is
informationally redundant with v4's clean *non-vegas* late-season
features (`late_adj_em`, `late_sos`, `efficiency_trend`,
`margin_trend`) in a year where those features alone are insufficient
to recover the predictive accuracy v4 had under the leak -- so the
extra column adds tree-split overhead without adding distinct signal.

## Comparison to PR 15 / generalized lesson

**The redundancy story was leak-tight, not structural.** The original
Colley findings note generalized PR 15's verdict as "v4's joint 67-
feature stack already extracts opponent-adjusted team-strength via
different decompositions; adding any single new rating feature provides
no marginal value." The clean-baseline result for Colley refutes this
generalization: when v4 loses its Vegas-leak signal, the marginal
information content of a W/L-only rating becomes large enough to clear
the gate. The redundancy was specifically against vegas-inflated
adj_em/efficiency loops, not against the clean stack.

**Massey-decay-14d does NOT flip with Colley.** Both candidates had
similar leaky-baseline deltas (+0.0053 and +0.0057), and the spec
treated them as a paired test of the redundancy hypothesis -- if both
flipped together, the "redundancy is leak-tight" lesson would
generalize cleanly. Instead, only Colley flipped. The asymmetry is
informative: Colley uses W/L only (zero margin information); Massey-
decay-14d uses margin with 14-day half-life. The 14-day window's
"different signal" claim from PR 15 is now competing with v4's clean
non-vegas late-season features (which carry margin information but
were never blocked by the leak filter). The clean stack still extracts
late-season margin redundantly with Massey-decay-14d -- but does NOT
extract W/L-only opponent-adjusted strength redundantly with Colley.

**Contrast with the residual-correlation jumps in PR 24/25/26.** Those
prior reruns showed v4's *errors* became more correlated with weaker
peers' errors when v4 lost its leak (BT r 0.577 -> 0.868; PEER_B rho
0.45 -> 0.726; HBT r 0.45-0.51 -> 0.68-0.77). That mechanism does NOT
apply here -- this gate compares two same-model arms with one column
toggled, not v4-vs-peer residuals. The clean-baseline shift here is in
the *opposite* direction and traces to a different mechanism: clean v4
has more headroom for any single feature to add information, simply
because clean v4 is weaker. PR 24/25/26 were about residual-correlation
ceilings on independent stage-1 candidates; this PR is about feature-
vs-feature redundancy at the v4 layer.

**For future feature-addition work post-leak-fix.** Two candidates
that previously failed clause 2 by similar margins now have divergent
verdicts on the clean baseline. The leak-shifted clean baseline
materially changes which redundancy thresholds bite. Re-running
*every* leaky-baseline feature-addition rejection at clean baseline
would over-extend this PR's scope, but the result here suggests the
"big-magnitude rejections do not need re-eval" line in TODO.md (line
168) holds for ones with deltas >> +0.005 LL, while ones in the
+0.005-0.010 ballpark are now load-bearing-uncertain. Specifically,
revisiting the BT-as-feature -0.0015 LL result is unmotivated (the
clean shift is most likely to make it MORE negative, not less); the
v9 weight-sweep family is also unaffected (different metric --
bracket points -- which was already validated under PR 17).

## Verdict + recommendation

**Colley:** PASS clause 2 cleanly. Promote to recovery step 5's
"next-immediate full LOSO backtest" sub-priority per the spec's
decision matrix row 3. The original Colley spec
(`docs/superpowers/specs/2026-05-03-colley-matrix-feature-design.md`)
already has the if-PASS branch laid out: full 22-season LOSO with
`colley_rating` wired into `compute_all_features` + a v4-vs-v4+colley
comparison on bracket points (or LL on aggregate, before bracket
points if compute is a concern). The wire-in revert in commit 3b4c374
is the inverse of the change needed -- the if-pass branch unrevert
should be a near-trivial commit.

**Massey-decay-14d:** FAIL clause 2 (marginal). Robust NO-GO across
both leaky and clean baselines closes Massey-decay-14d as a feature
candidate; the 14-day window does not survive the clean baseline shift
either. Note this is a *different* failure mode from PR 15: leaky
failure was "redundancy with vegas-inflated adj_em"; clean failure is
"redundancy with clean non-vegas late-season features" -- separate
mechanisms, same outcome. No re-opening of the half-life sweep.

**Marginal-rejections list status.** Of the 5 named items in recovery
step 5's sub-priority list, 4 are now closed (plain BT in PR 24,
feature-view ensemble PEER_A/B in PR 25, HBT in PR 26, Massey-decay-14d
in this PR) and 1 is promoted (Colley in this PR -> full LOSO backtest
queue). The list is fully unwound at the cheap-gate layer.

## Files of record

**Created on this branch:**
- `src/clause2_colley.py` -- standalone clause-2 runner (mirrors
  `src/clause2_decay_massey.py`).
- `output/diag_clause2_colley.json` -- new canonical artifact
  (force-added).
- `docs/notes/2026-05-05-colley-massey-clean-rerun.md` (this file).
- `docs/superpowers/specs/2026-05-05-colley-massey-clean-rerun-design.md`.
- `docs/superpowers/plans/2026-05-05-colley-massey-clean-rerun.md`.

**Modified on this branch:**
- `output/clause2_decay_massey_hl14.json` -- regenerated under clean v4
  pipeline (force-added; replaces PR 15 leaky values).
- `TODO.md` -- recovery step 5 sub-priority list updated per decision
  matrix.

**Untouched on this branch (no source-side change required):**
- `src/diagnose_colley.py` (still requires wire-in to run; this PR
  bypasses it via the standalone clause2_colley.py).
- `src/diagnose_massey_mov.py` (same).
- `src/clause2_decay_massey.py` (used as-is).
- `src/features/colley_matrix.py`, `src/features/massey_matrix.py`
  (the solvers; no math changes).

## Follow-ups

1. **Colley full LOSO backtest** (next-immediate step-5 sub-priority).
   Wire `colley_rating` back into `compute_all_features` (inverse of
   commit 3b4c374), regenerate `output/pairwise_v4.csv` with the new
   feature, score 22-season LOSO and bracket-points head-to-head
   against the canonical clean v4 + v9-C baseline (1929 brkt pts on
   v9-C / 2069 on clean v8 per PR 24). ~3 hours compute. If the
   bracket-points delta clears the spec's swap-in threshold (+10
   bpts), Colley becomes a v4 feature-add candidate.
2. **Possible follow-up: Colley half-life sweep.** Original Colley
   solver is W/L-only, no half-life parameter. Adding decay would mean
   re-deriving the solver math. Not required for the recovery
   roadmap; deferred.
3. **Massey-decay sweep** (NOT re-opened). hl=14d closing as robust
   NO-GO across both baselines; the rest of the half-life curve was
   already filtered out at clause 1 in PR 15 (hl in {30, 60, 120}d
   FAILed clause 1) and lower (hl=7d) was not run for clause 2 in
   PR 15 with rationale ("3-4 games per team in the window would be
   noise-dominated") that holds against any v4 baseline. Closed.
