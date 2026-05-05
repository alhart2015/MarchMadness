# Feature-View Ensemble PEER_A/B Clean Re-eval (Step 5 marginal #2) -- Design

**Date:** 2026-05-05
**Branch:** feat/feature-view-clean-rerun
**Recovery context:** TODO.md "CONTAMINATION DISCOVERED 2026-05-04" -> step 5,
sub-priority "Feature-view ensemble PEER_A/B re-eval" (named highest signal/noise
of the marginal-rejections list remaining after plain BT was closed in PR 24).
**Predecessors:**
- Original feature-view findings (NO-GO under leaky baseline): `docs/notes/2026-05-02-feature-view-ensemble.md`
- Original spec: `docs/superpowers/specs/2026-05-02-feature-view-ensemble-design.md`
- Clean LOSO regen findings: `docs/notes/2026-05-04-v4-clean-loso-regen.md`
- Clean v9-C re-eval findings: `docs/notes/2026-05-04-v9c-clean-rerun.md`
- Clean plain BT re-eval findings: `docs/notes/2026-05-05-plain-bt-clean-rerun.md`

## Motivation

PR 14 ran a 3-clause pre-sweep gate on a same-class XGBoost feature-view
ensemble (PEER_A: 40 team-strength features; PEER_B: 27 form/market/meta
features) and ruled NO-GO: residual correlation `r=0.45` PASSED, but
PEER_A standalone LL `0.5720` was `+0.1375` above v4's leaky LL `0.4345`
(clause 1 fails by `5.5x` the `0.025` tolerance), and the optimal 2-blend
headroom was `-0.0206` (clause 3 fails). The dominant blocker was the
strength gap between PEER_A and v4.

PR 21's clean-baseline regen shifted v4's LL from `0.4345` to `0.5588`
(+0.122 under clean LOSO). PEER_A's 40 features include adjusted
efficiency, four factors, KenPom, Massey, conf strength, and season
summary -- **none of which are Vegas features and none of which were
contaminated by the PR 19 leak**. PEER_A's standalone LL is therefore
predicted to be unchanged across the leaky->clean transition, making
the clean-baseline gap `0.5720 - 0.5588 = +0.013` (within the 0.025
clause-1 tolerance). PEER_B's 27 features include the seven Vegas
features filtered by PR 19 (`vegas_avg_spread`, `vegas_avg_margin`,
`vegas_ats_pct`, `vegas_power_rating`, `vegas_consistency`,
`vegas_game_count`, `vegas_late_spread_delta`); PEER_B's LL is expected
to shift, with direction unknown a priori.

This PR tests the prediction directly: same trainer, same diagnostic,
same partition, but `prepare_loso_inputs()` now flows through PR 19's
`filter_vegas_to_pre_tournament()`. The verdict gates whether the
feature-view ensemble approach (E1 / E2 sweeps) is worth reopening,
or whether the marginal-rejections list closes one more candidate.

## Scope

**In scope.**

- Reproducibility-via-rerun: run `python src/train_peer_stage1.py --peer a`
  and `python src/train_peer_stage1.py --peer b` against the clean
  feature-pipeline (post-PR-19). Compare md5 of new outputs vs the
  tracked `output/pairwise_peer_a.csv` and `output/pairwise_peer_b.csv`
  (PR 14 force-added) on `main` (0bde1f1).
  - Expected: PEER_A byte-identical (no Vegas features in PEER_A's 40
    columns). PEER_B differs (Vegas features present). If PEER_A
    diverges, halt and investigate -- the assumption "PEER_A is
    invariant across leaky->clean" is load-bearing.
- Run `python src/diagnose_feature_view_ensemble.py
  --pairwise-v4 output/pairwise_v4.csv
  --pairwise-peer-a output/pairwise_peer_a.csv
  --pairwise-peer-b output/pairwise_peer_b.csv` against the regenerated
  artifacts. Capture the 3-clause gate verdict.
- Force-add only the artifacts that actually changed:
  - `output/pairwise_peer_a.csv` -- only if md5 differs from tracked.
  - `output/pairwise_peer_b.csv` -- expected to differ; force-add.
  - `output/diag_feature_view_ensemble.json` -- always force-add
    (overwritten with new clean numbers).
- Findings doc `docs/notes/2026-05-05-feature-view-clean-rerun.md`
  mirroring PR 24's structure: TL;DR, methods, gate result table
  (3 clauses + 3-blend side observation), standalone metrics,
  comparison-to-PR-14 side-by-side, discussion (especially: residual
  correlation comparison vs PR 24's r=0.868 finding), verdict,
  recommendation.
- TODO.md update: mark "Feature-view ensemble PEER_A/B re-eval"
  sub-priority done with verdict + headline numbers (peer LLs,
  residual r, headroom). Advance the priority list per decision matrix.

**Out of scope.**

- E1 sweep (blend(peer_A, peer_B), no v4) -- only run if gate PASSES.
  Compute is one v9-C 22-season backtest per cell; full sweep is
  ~hours. This PR is gate-or-no-gate only.
- E2 sweep (blend(v4, peer_A, peer_B)) -- same condition.
- Production-side changes to `output/pairwise_probs.json` or any
  `predict_*.py`. This PR is gate-only.
- Re-tuning gate thresholds (`PEER_LL_CEILING_DELTA = 0.025`,
  `RESID_CORR_MAX = 0.60`, `HEADROOM_MIN = 0.001`). They are absolute
  thresholds inherited from PR 14 and are not sensitive to v4's
  baseline shift.
- Re-evaluating the other marginal-rejections candidates (HBT, Colley,
  Massey-decay hl=14d, BT-as-feature). Each gets its own PR per the
  TODO.md priority list.
- The 2026-05-04 data wipe recovery: data restoration (extracting
  `training_data.tar.gz`, setting up junctions) was completed at the
  start of this PR's worktree setup; recorded in the findings note's
  Methods section but not the focus of this PR.

## Decision matrix

| Gate verdict | Action |
|---|---|
| PASS (all 3 clauses clear) | Findings note records new numbers + diff vs PR 14. TODO.md promotes "Feature-view ensemble E1+E2 sweep" to next slot in step 5 sub-priorities, with explicit ~hours compute warning and pre-registered hypothesis (E2 better than E1 since v4 dominates). Production unchanged this PR. |
| FAIL (any clause fails) | Findings note records new numbers + diff vs PR 14. TODO.md drops feature-view ensemble entirely from the marginal-rejections list (gate failed under both leaky and clean baselines = robust NO-GO). Next sub-priority becomes "HBT re-eval" (~5 min). Production unchanged. |

The pre-registered prediction is **clause 1 PASS, clause 2 unclear,
clause 3 unclear**. Clause 1 is the falsifier the user named in TODO.md
line 134. Clauses 2 and 3 depend on PEER_B's clean-baseline behavior
which has not been observed.

PR 24's plain-BT re-eval surfaced a hypothesis worth pre-registering:
**residual correlation between v4 and a peer is bounded below by what
both models miss for "hard regular-season-information" reasons**. PR 24
saw r jump from 0.577 (leaky) to 0.868 (clean) for plain BT vs v4. The
analogous quantity here is `r(resid_A, resid_B)` -- not `r(resid_v4, resid_A)`
-- so PR 24's finding does not directly predict PEER_A vs PEER_B
correlation. But if PEER_B's predictions become more v4-like once Vegas
features are filtered (Vegas was a strong differentiator under leaky
training), `r(resid_A, resid_B)` could rise.

## Procedure

1. Worktree `feat/feature-view-clean-rerun` off `main` (0bde1f1)
   with `data/raw/march-machine-learning-2026/`,
   `data/raw/kaggle/`, and `data/raw/vegas_lines/` junctioned to
   `C:\Users\alden\MarchMadness\data\raw\<subdir>\`. Status: complete
   (this PR's worktree setup; required `tar -xzf training_data.tar.gz`
   to repopulate the wiped `march-machine-learning-2026/` subdir).
2. Reproducibility step PEER_A: `python src/train_peer_stage1.py --peer a
   --output output/pairwise_peer_a_repro.csv`. Then md5 compare
   `pairwise_peer_a.csv` (tracked) vs `pairwise_peer_a_repro.csv` (new).
   If equal, delete repro file and keep tracked. If different, halt and
   investigate (the partition listing or trainer code may have drifted).
3. Reproducibility step PEER_B: `python src/train_peer_stage1.py --peer b
   --output output/pairwise_peer_b_repro.csv`. md5 compare; expect
   difference. Document md5 of both old and new in findings doc.
   Replace tracked `pairwise_peer_b.csv` with new run output, force-add.
4. Run `python -m pytest tests/test_diagnose_feature_view_ensemble.py
   tests/test_train_peer_stage1.py tests/test_feature_views.py
   tests/test_ensemble_stage1.py -q`. All tests pass before proceeding.
5. Run `python src/diagnose_feature_view_ensemble.py`. Capture stdout
   and `output/diag_feature_view_ensemble.json`.
6. Sanity gates on the run output:
   - `n_played_games == 1449` (matches PR 14 + PR 24 matched-game set).
   - Stdout `ll_v4` ~ 0.5588 (clean v4 LL from PR 21/23, within 0.005).
   - Stdout `ll_peer_a` ~ 0.5720 (PR 14 leaky number; expected
     byte-equal trip → identical LL on identical data → identical CSV).
   - PEER_B coverage: `n_played_games` identical between v4, PEER_A,
     PEER_B (the diagnostic raises ValueError on mismatch).
7. Read the gate verdict from JSON. Apply decision matrix.
8. Write findings doc + TODO.md update. Force-add JSON + (only-if-changed)
   pairwise CSVs. Commit per the plan in
   `docs/superpowers/plans/2026-05-05-feature-view-clean-rerun.md`.

## Risks

1. **PEER_A byte-equality check fails.** Possible causes: sklearn /
   xgboost version drift (no env pin); Massey or KenPom column drift
   (their inputs were audited clean in PR 20, but the underlying CSV
   files may have grown across the last two months); `prepare_loso_inputs()`
   itself changing slightly between branches. Halt and investigate
   before drawing conclusions; the assumption "PEER_A is invariant
   across leaky->clean" is load-bearing for this PR's framing.
2. **`n_played_games` mismatch with PR 14 (1449).** Either pairwise
   CSVs have different pair coverage, or the join logic in
   `compute_pairwise_ll` drifted. The diagnostic itself raises
   `ValueError` on mismatch (line 194-198). Halt and reconcile.
3. **Gate boundary case on clause 1.** If `delta_a` lands within
   `[0.020, 0.030]`, the PASS/FAIL split is sensitive to dedup ordering
   and clip epsilon. Mitigation: report the full delta to 4 decimal
   places in the findings doc and characterize the result honestly
   instead of forcing a binary verdict.
4. **PEER_B regen takes longer than the user's TODO budget (~20 min).**
   PR 24's plain-BT regen was ~10 sec because BT trains directly on
   regular-season binary outcomes with no v4 feature dependency.
   `train_peer_stage1.py` calls `prepare_loso_inputs()`, which builds
   the entire v3/v4 feature matrix (Massey, Colley, KenPom, Vegas merge).
   With `data/cache/` wiped (per the recovery runbook), the first call
   will rebuild caches from raw data; subsequent calls (e.g. PEER_B
   after PEER_A) reuse the cache. Mitigation: run PEER_A first to
   warm cache; PEER_B should be roughly ` cache_warm_loso_only` time.
5. **Junction wipe risk during cleanup.** Per
   `feedback_windows_junction_delete.md` memory and
   `docs/data_recovery.md` § Prevention. Use `git worktree remove` for
   cleanup; do not use PowerShell `(Get-Item).Delete()` or recursive
   PowerShell removes on the worktree dir.

## Files of record

**Modified:**
- (none expected on the source side -- this PR is rerun + diagnostic
  only; no code changes are required by the spec)

**Created (force-added):**
- `output/pairwise_peer_b.csv` (overwritten; already tracked from PR 14)
- `output/diag_feature_view_ensemble.json` (overwritten; already tracked
  from PR 14)
- `docs/notes/2026-05-05-feature-view-clean-rerun.md` (findings)
- `docs/superpowers/specs/2026-05-05-feature-view-clean-rerun-design.md`
  (this file)
- `docs/superpowers/plans/2026-05-05-feature-view-clean-rerun.md`
  (next step)

**Conditional:**
- `output/pairwise_peer_a.csv` -- force-add ONLY if md5 differs from
  tracked. Expected: no change. If it changes, the spec's
  "PEER_A invariant" framing is wrong and the findings doc must
  reconcile.

**Updated:**
- `TODO.md` (mark sub-priority done, advance priority list per decision matrix)

## Test plan

- Existing tests: `tests/test_diagnose_feature_view_ensemble.py`
  (10 tests), `tests/test_train_peer_stage1.py` (4 tests),
  `tests/test_feature_views.py` (5 tests),
  `tests/test_ensemble_stage1.py` (4+ tests). All must continue to pass.
- No new test code planned (the procedure adds no new code paths).
- Procedural sanity gates (Procedure step 6): `n_played_games == 1449`,
  `ll_v4` matches clean baseline within 0.005 LL, PEER_A LL matches
  PR 14 (0.5720) to 4 decimals.

## Acceptance criteria

- All existing pytest pass.
- PEER_A byte-equality check succeeds OR mismatch is investigated and
  documented in the findings doc.
- `n_played_games == 1449` and `ll_v4` matches clean baseline within
  0.005 LL.
- Findings doc cites the diagnostic JSON with its post-run numbers
  and shows the comparison-to-PR-14 side-by-side table.
- TODO.md updated per decision matrix.
- All artifacts force-added per `docs/data_recovery.md` policy.
