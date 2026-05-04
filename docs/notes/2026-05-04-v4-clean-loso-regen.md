# v4 Clean LOSO Regeneration -- Findings

**Date:** 2026-05-04
**Branch:** feat/v4-clean-loso-regen
**Verdict:** **PASS-AND-FLAG.** The Vegas leak was substantially larger than
the spec's anchor band. Clean v4 LOSO is **+0.122 LL worse** and
**-9.7pp accuracy worse** than the leaky baseline -- a much bigger shift
than the pre-registered "perhaps 0.45-0.47 LL, 73-77% acc" range.
**Major retractions required for downstream verdicts.**

## TL;DR

| metric                       | leaky (pre-fix) | clean (post-fix) | delta    | spec anchor       |
|------------------------------|-----------------|------------------|----------|-------------------|
| 22-season mean LL            | 0.4370          | **0.5588**       | **+0.1219** | 0.45-0.47       |
| 22-season mean accuracy      | 80.35%          | **70.66%**       | **-9.7pp**  | 73-77%          |
| seasons with delta_LL > 0    | --              | **21 / 22**      | --       | most (qualitative) |
| seasons with delta_acc <= 0  | --              | **20 / 22**      | --       | most (qualitative) |

Clean LL of 0.5588 lands well outside even the spec's pass-and-flag
threshold (LL > 0.50). The leak was correlated with tournament
success exactly as PR 19's spec hypothesized; removing it shifted v4
from "beats Vegas everywhere" (PR 18 audit verdict, leaky) to a model
whose 22-season LL is closer to plain-BT (0.565 standalone) than to
the Vegas-line implied probability (~0.5447 in the PR 18 audit).

## Per-season shift

| season | LL_leaky | LL_clean | dLL    | acc_leaky | acc_clean | dAcc    | n  |
|--------|----------|----------|--------|-----------|-----------|---------|----|
| 2003   | 0.5841   | 0.5770   | -0.007 | 0.6953    | 0.7031    | +0.008  | 64 |
| 2004   | 0.4282   | 0.5416   | +0.113 | 0.7812    | 0.6875    | -0.094  | 64 |
| 2005   | 0.4115   | 0.5164   | +0.105 | 0.8125    | 0.7500    | -0.062  | 64 |
| 2006   | 0.4577   | 0.5778   | +0.120 | 0.8359    | 0.7109    | -0.125  | 64 |
| 2007   | 0.4536   | 0.4785   | +0.025 | 0.7734    | 0.7578    | -0.016  | 64 |
| 2008   | 0.4032   | 0.4812   | +0.078 | 0.8125    | 0.7734    | -0.039  | 64 |
| 2009   | 0.3994   | 0.5079   | +0.108 | 0.8516    | 0.7188    | -0.133  | 64 |
| 2010   | 0.3833   | 0.5620   | +0.179 | 0.8047    | 0.7031    | -0.102  | 64 |
| 2011   | 0.5171   | 0.6673   | +0.150 | 0.7687    | 0.6045    | -0.164  | 67 |
| 2012   | 0.4030   | 0.5451   | +0.142 | 0.8209    | 0.7090    | -0.112  | 67 |
| 2013   | 0.4848   | 0.6160   | +0.131 | 0.7687    | 0.6343    | -0.134  | 67 |
| 2014   | 0.4363   | 0.5814   | +0.145 | 0.7910    | 0.6716    | -0.119  | 67 |
| 2015   | 0.3386   | 0.4859   | +0.147 | 0.8582    | 0.7761    | -0.082  | 67 |
| 2016   | 0.5555   | 0.5841   | +0.029 | 0.7090    | 0.7164    | +0.007  | 67 |
| 2017   | 0.3597   | 0.5495   | +0.190 | 0.8657    | 0.7239    | -0.142  | 67 |
| 2018   | 0.4469   | 0.5842   | +0.137 | 0.8284    | 0.7015    | -0.127  | 67 |
| 2019   | 0.3643   | 0.5101   | +0.146 | 0.8881    | 0.7537    | -0.134  | 67 |
| 2021   | 0.4377   | 0.5849   | +0.147 | 0.8182    | 0.6818    | -0.136  | 66 |
| 2022   | 0.5185   | 0.6469   | +0.128 | 0.7761    | 0.6493    | -0.127  | 67 |
| 2023   | 0.4778   | 0.6225   | +0.145 | 0.7687    | 0.6418    | -0.127  | 67 |
| 2024   | 0.4262   | 0.6030   | +0.177 | 0.7836    | 0.6940    | -0.090  | 67 |
| 2025   | 0.3254   | 0.4708   | +0.145 | 0.8657    | 0.7836    | -0.082  | 67 |

21 of 22 seasons got worse on LL after the fix. Only 2003 improved
(-0.007 LL), and only 2003 + 2016 improved on accuracy (+0.7pp each).

## Largest shifts (top 3 by |dLL|)

| season | dLL    | dAcc    | tournament narrative                                         |
|--------|--------|---------|--------------------------------------------------------------|
| 2017   | +0.190 | -14.2pp | UNC champion; leaky model "knew" via inflated UNC Vegas avg. |
| 2010   | +0.179 | -10.2pp | Duke champion as #1; Butler #5 to title game.                |
| 2024   | +0.177 | -9.0pp  | UConn champion (per PR 19 spec: vegas_avg_margin shifted +1.98). |

The 2024 case is the canary anchor: PR 19 measured the leak directly
on UConn's per-team aggregate (+1.98 margin under leak vs reg-only)
and the 2024 LL shift here (+0.177) is the second-largest of the 22
seasons. The mechanism predicted by the leak hypothesis -- chalk-y
tournaments shift more than chaos-y ones because the leak inflates
champions' features -- shows up in the 2017/2010/2024 trio (clear
favorites won) vs 2003/2016 (chaotic tournaments, near-flat shift).

## Anchor verdict

Per the spec's pass criteria:

- pass-as-expected: clean LL in [0.43, 0.50] -- **NO** (0.5588).
- pass-and-flag: clean LL > 0.50 -- **YES**.
- surprising-pass: clean LL <= 0.4369 -- **NO**.

**Verdict: PASS-AND-FLAG.** v4 was much weaker than the leaky
metrics indicated. The 0.122 LL inflation is qualitatively
consistent with the leak structure but quantitatively about 3x the
spec's pre-registered band.

## Downstream impact -- retractions required

Every existing finding that compared v4 against another model's
absolute LL needs re-eval. Most retractions are flips:

1. **PR 18 audit ("v4 beats Vegas everywhere", `2026-05-04-v4-gap-
   audit-vegas.md`).** Leaky verdict was `ll_v4=0.4305` vs
   `ll_vegas=0.5447` (delta -0.114). Clean mean LL is 0.5588, which
   is **0.014 WORSE than Vegas** on this dataset. The "no weak
   spots" verdict is **retracted**; whether v4 still beats Vegas at
   all on this benchmark is unclear and depends on the per-bucket
   re-run (recovery step 4, immediate next PR).

2. **Plain BT bracket-points re-test (PR 17, 2026-05-04-bt-bracket-
   points.md).** Leaky baseline: v4 + v9-C scored 2713 brkt pts;
   plain BT was a NO-GO at every blend weight. **Need to re-run
   the sweep against the clean `pairwise_v4.csv`** -- v4's
   per-disagreement accuracy may have changed materially. The
   "structural diversity is necessary but not sufficient" lesson
   stands; the specific NO-GO verdict on plain BT does not.

3. **All "marginal" rejections** in TODO.md "Tried and rejected"
   whose deltas were within ~0.05 LL of v4 (e.g. BT-as-feature at
   -0.0015, v9 weight-sweep family at +18 to +20 brkt pts vs v4).
   With v4's LL shifting +0.122, anything within 0.05 of the old
   leaky 0.4369 needs re-eval against the new 0.5588 baseline.
   The "ceiling against leaky baseline" framing in those notes is
   now incorrect.

4. **Big-magnitude rejections** (HBT, Colley, Massey-MOV, plain BT
   standalone, LR ensemble, feature-view ensemble, quality-wins,
   matchup-interaction, round-as-feature) **do NOT need re-eval**.
   Their failure margins (-93 brkt pts, -105, +0.0057 LL, etc.)
   exceed the leak shift by a large factor. A ~0.1 LL baseline
   move does not flip a -0.10 LL relative loser.

5. **v9-C production swap** is the highest-priority re-eval. v9-C
   is currently deployed and trained against the leaky pairwise.
   Step 5 of the recovery roadmap.

## Notes on procedure

- Skipped Step 6 (default-params LOSO) via new `MM_SKIP_DEFAULT_LOSO`
  env-var gate -- its pairwise rows are dedup'd away by every
  downstream consumer (`drop_duplicates(..., keep="last")`). Cut
  ~half the runtime.
- Reused the leaky run's tuned XGBoost hyperparameters via
  `MM_TUNED_PARAMS_V3` (`{n_estimators: 424, max_depth: 4, lr:
  0.0139, subsample: 0.874, colsample: 0.776}`). Confound: optimal
  hyperparameters under the clean feature distribution may differ.
  Followup retune is a candidate experiment but unlikely to recover
  the 0.122 LL gap -- that gap is feature-driven, not
  hyperparameter-driven.
- Output shapes: `pairwise_v4.csv` 48,465 data rows (one tuned-pass
  only, per skip gate); `cv_per_season_v3.csv` 22 rows (LOSO test
  seasons). Both gitignored; numbers above are the headline
  artifact for downstream use.
- Fixed a pre-existing `NameError` crash in `enhanced_model_v3.py`'s
  final summary block (`new_feature_names`, `n_tourney`,
  `n_supplemental` were locals in the pre-refactor monolithic
  `main()` and got orphaned when `prepare_loso_inputs()` was
  extracted). Reconstructed in `main()` from the inputs dict. The
  crash was post-write so no data loss, but every recent v3 run
  exited with code 1 on this print.

## What this does NOT establish

- v4's actual position vs Vegas / 538 -- the audit framework reruns
  in step 4. The bucketed picture may differ from the aggregate.
- Whether v9-C still adds points on the clean baseline -- step 5.
- Whether re-tuning hyperparameters recovers any of the lost LL --
  potential followup; expected effect <0.02 LL on prior evidence.
- A new ground truth for any rejected experiment beyond marking
  which need re-eval. Each candidate has its own cost-benefit.
