# Colley + Massey-Decay Clean Re-eval (Step 5 marginals #4 + #5) -- Design

**Date:** 2026-05-05
**Branch:** feat/colley-massey-clean-rerun
**Recovery context:** TODO.md "CONTAMINATION DISCOVERED 2026-05-04" -> step 5,
the last two named candidates remaining on the marginal-rejections list
after PR 24 closed plain BT, PR 25 closed feature-view ensemble, and PR 26
closed HBT.
**Predecessors:**
- Original Colley findings (clause 2 FAIL): `docs/notes/2026-05-03-colley.md`
- Original Massey-decay findings (clause 2 FAIL at hl=14d): `docs/notes/2026-05-03-massey-mov.md`
- Original Colley spec: `docs/superpowers/specs/2026-05-03-colley-matrix-feature-design.md`
- Original Massey spec: `docs/superpowers/specs/2026-05-03-massey-matrix-feature-design.md`
- Clean LOSO regen findings: `docs/notes/2026-05-04-v4-clean-loso-regen.md`
- Pattern predecessors: `docs/notes/2026-05-04-v9c-clean-rerun.md` (PR 24),
  `docs/notes/2026-05-05-feature-view-clean-rerun.md` (PR 25),
  `docs/notes/2026-05-05-hbt-clean-rerun.md` (PR 26)

## Motivation

Two feature-addition candidates were rejected at clause 2 (LL headroom on
3-season subset {2019, 2022, 2024}) under the leaky-v4 baseline, both
within the PR 21 leak-shift noise floor of +0.122 LL:

| candidate | clause-2 mean delta | threshold | leaky verdict |
|---|---|---|---|
| Colley (PR 15)               | +0.0053 | +0.001 | FAIL |
| Massey-MOV decay hl=14d (PR 15) | +0.0057 | +0.001 | FAIL |

Both ratings are computed from regular-season game results only (Colley
on W/L; Massey-MOV on capped margin). Neither uses Vegas data, so the
leak fix (PR 19) does NOT change the rating columns themselves. What
changes on the clean baseline is v4's *other* features (vegas_*) and
therefore the without-feature LL on the 3-season subset (and the
with-feature LL via the same v4 stack).

Two competing intuitions on the expected outcome:

1. **Redundancy holds at any v4 baseline (clean v4 still extracts
   opponent-adjusted strength via adj_em + massey_composite + season_win_pct).**
   The deltas should stay positive in the same +0.005 ballpark. Robust
   NO-GO across both baselines, mirroring PR 24/25/26's pattern that
   diversity claims survive the leak shift but standalone-strength gaps
   do not.
2. **Clean v4 is materially weaker (+0.122 LL on 22-season aggregate),
   so any extra feature has more room to add marginal info.** The
   deltas could shrink toward zero or flip negative (features now help).
   That would re-open the original "proceed to full LOSO backtest"
   branch from each candidate's spec.

This PR runs both candidates' clause 2 against the post-PR-19 clean v4
feature pipeline and reports the verdict for each independently.

## Scope

**In scope.**

- Massey-decay-14d: re-run `python src/clause2_decay_massey.py 14`
  against the clean v4 stack. The script already exists (PR 15); no
  code changes. Writes `output/clause2_decay_massey_hl14.json`.
- Colley: write a new standalone runner `src/clause2_colley.py` that
  mirrors `src/clause2_decay_massey.py` (computes Colley ratings inline,
  merges into `fm` via (TeamID, Season), runs LOSO with/without on the
  3-season subset). Avoids the temporary-wire-in that
  `src/diagnose_colley.py` would otherwise require. Writes
  `output/diag_clause2_colley.json`.
- Force-add both diagnostic JSONs as canonical artifacts.
- Findings doc `docs/notes/2026-05-05-colley-massey-clean-rerun.md`
  with side-by-side leaky-vs-clean tables for both candidates and a
  generalized lesson section (does the redundancy story survive the
  baseline shift, or do features-as-features pattern differently from
  models-as-stage-1-peers?).
- TODO.md update: replace the "Colley (PR 15): clause-2 delta +0.0053
  LL" and "Massey-decay hl=14d (PR 15): clause-2 delta +0.0057 LL"
  bullets with their verdict bullets per the decision matrix below.
  Net effect: marginal-rejections list closes (PR 24/25/26/this PR
  cover all named items).

**Out of scope.**

- Re-running clause 1. The rating columns themselves do NOT change under
  the Vegas filter; clause 1 baselines (`adj_em`, `massey_composite`,
  `season_win_pct`) are themselves computed from regular-season data
  and shift only via the small downstream effect of efficiency-loop
  changes. Both candidates' original clause 1 PASSED with wide margin
  (Colley 0.907 / 0.948 / 0.687; Massey-decay-14d 0.931); a sub-0.001
  baseline shift cannot flip these.
- Re-tuning the gate threshold (`LL_HEADROOM_MAX = +0.001`). Shared
  verbatim with clause-1 thresholds across the two diagnostics; not
  sensitive to v4's baseline shift.
- Massey-decay sweep over other half-lives. PR 15's hl=14d was the only
  passing-clause-1 candidate that proceeded to clause 2; if hl=14d
  stays FAIL on clean baseline, the rest of the curve (hl in {7, 30,
  60, 120}) is uninteresting. If hl=14d flips PASS on clean, we'd
  re-open the sweep in a follow-up PR; this PR is not that follow-up.
- Wire-in of either feature to `compute_all_features`. Both wire-ins
  remain reverted; this PR is gate-only.
- Production-side changes to `output/pairwise_probs.json`.

## Decision matrix

Verdict per candidate; both are reported independently in the findings
doc. The combined-verdict cases below describe how the TODO.md edit
fans out.

| Colley c2 | Massey-14d c2 | Action |
|---|---|---|
| FAIL | FAIL | Most likely. Findings note records both new clause-2 deltas + diff vs PR 15. TODO.md replaces both bullets with "[DONE -- PR <pending>]" lines marking robust NO-GO across both baselines. Marginal-rejections list closes; recovery step 5 is fully unwound. Production unchanged. |
| FAIL | PASS | Surprising mid-case. Findings note records both. TODO.md marks Colley closed, promotes Massey-decay-14d to "next-immediate full LOSO backtest" sub-priority (per the original Massey spec's if-pass branch). Half-life sweep ALSO re-opens as a now-relevant question (does the curve shift?). Production unchanged. |
| PASS | FAIL | Symmetric to row 2. Promotes Colley to "next-immediate full LOSO backtest." Production unchanged. |
| PASS | PASS | Most surprising. Findings note records both; TODO.md promotes BOTH to full LOSO backtest with explicit ordering (Colley first; Massey-decay second since the half-life sweep would compound). Generalized lesson section in findings emphasizes that the leak shift was large enough to flip features-as-features verdicts even when it didn't flip models-as-peers (PR 24/25/26 all failed). Production unchanged. |

The pre-registered prediction is **both FAIL** (row 1) at deltas in the
same +0.005 ballpark as PR 15. Two reasons:

1. **The redundancy is structural, not threshold-tight.** Both
   candidates' clause 1 against `adj_em` (mean 0.907 / 0.931) and
   `massey_composite` (mean 0.948 / 0.946) is high enough that the
   marginal info each rating contributes is dominated by the noise
   floor of XGB on 3 seasons * ~129 holdout games. The leak fix
   doesn't change adj_em or massey_composite themselves, so the
   redundancy story is unchanged.
2. **PR 24/25/26 confirmed a related shift in the *opposite* direction
   for models-as-peers** -- residual correlations rose (BT r 0.577
   -> 0.868; PEER_B rho 0.45 -> 0.726; HBT r 0.45-0.51 -> 0.68-0.77)
   when v4 lost its leak. That mechanism (clean v4's errors became
   more correlated with weaker peers' errors) does NOT apply to
   features-as-features: the gate compares two LOSO runs of the same
   v4 model, with and without one extra column, on the same data.

But the overall framework predicts these two clauses should be the
*least sensitive* to the baseline shift of any clause we've re-run --
which is exactly why they're being closed last.

## Procedure

1. Worktree `feat/colley-massey-clean-rerun` off `main` (d5846d9) with
   `data/raw/march-machine-learning-2026/`, `data/raw/kaggle/`, and
   `data/raw/vegas_lines/` junctioned to
   `C:\Users\alden\MarchMadness\data\raw\<subdir>\`. Status: complete
   (this PR's worktree setup; required `tar -xzf training_data.tar.gz`
   to repopulate the wiped subdirs in main repo before junctioning).
2. Run `python -m pytest tests/test_features/test_colley_matrix.py
   tests/test_features/test_massey_matrix.py -q`. All 15 tests pass
   before proceeding. (Status: complete -- 15/15 in 11.83s.)
3. Write `src/clause2_colley.py` mirroring `src/clause2_decay_massey.py`:
   - Load LOSO inputs (`prepare_loso_inputs()`).
   - Compute Colley ratings via `compute_colley_ratings(regular)` from
     `src.features.colley_matrix`.
   - Drop any pre-existing `colley_rating` from fm (defensive; wire-in
     is reverted in main but harmless if it ever comes back).
   - Merge ratings into fm via (TeamID, Season).
   - Build `cols_with` and `cols_without` lists.
   - Run `leave_one_season_out_cv_weighted` twice with
     `allowed_holdouts=GATE_SUBSET_SEASONS`.
   - Emit `output/diag_clause2_colley.json` mirroring the schema of
     `output/clause2_decay_massey_hl14.json`.
4. Run both runners (order does not matter; Colley is faster because no
   decay weights needed):
   - `python src/clause2_colley.py 2>&1 | tee output/_clause2_colley_log.txt`
   - `python src/clause2_decay_massey.py 14 2>&1 | tee output/_clause2_massey_log.txt`
   First call rebuilds Massey/Colley/efficiency parquet caches from
   raw data (cache wiped along with raw subdirs); expect ~4 min cold-start
   + ~30 sec each runner = ~5 min total.
5. Sanity gates on the JSONs:
   - Each has `subset_seasons == [2019, 2022, 2024]`.
   - `mean_ll_without_*` is materially shifted from PR 15's 0.4388 (since
     v4's vegas_* features changed under PR 19); accept any value > 0.4
     and report the new value in findings.
   - `mean_ll_with_*` materially shifted from PR 15's 0.4445 / 0.4440
     for the same reason.
   - `mean_ll_delta` is the load-bearing number; threshold +0.001.
6. Apply the decision matrix per candidate; write findings + TODO update.
7. Force-add both JSONs.
8. Commit per the plan's task structure.

## Risks

1. **`prepare_loso_inputs()` cold-start time.** First call rebuilds
   Massey + Colley + efficiency caches from raw data (caches wiped or
   never populated in this worktree). Expect ~4 min one-time; subsequent
   calls within the same Python process are sub-second. The two clause-2
   runners spawn separate Python processes, but only the first one pays
   the cold-start. Not blocking; flag in findings methods section.
2. **`fm` already has `colley_rating` populated.** The Colley wire-in
   was reverted in commit 3b4c374, so this should not happen. Defensive
   `if "colley_rating" in fm.columns: drop` in `clause2_colley.py`
   matches the same defensive check in `clause2_decay_massey.py`.
3. **`pairwise_v4.csv` is byte-identical between the two arms by
   construction.** Both arms run their own LOSO; we are not blending
   against `pairwise_v4.csv` here (this is a feature-addition gate,
   not an LL-blend gate). Sanity: the without-arm's `ll_without` should
   match across the two scripts (same code path, same fm minus the
   added column) modulo random-seed effects in the XGB tuned-params
   re-run. Expect both `mean_ll_without` values to be within 1e-6.
4. **Junction wipe risk during cleanup.** Per
   `feedback_windows_junction_delete.md` memory and
   `docs/data_recovery.md` § Prevention. Use `git worktree remove` for
   cleanup; do not use PowerShell `(Get-Item).Delete()` or recursive
   PowerShell removes on the worktree dir.

## Files of record

**Modified:**
- (none expected on the source side beyond the new clause2_colley.py)

**Created:**
- `src/clause2_colley.py` (new standalone clause-2 runner; ~70 lines)
- `output/clause2_decay_massey_hl14.json` (overwritten; tracked from PR 15)
- `output/diag_clause2_colley.json` (NEW; net-new tracked artifact)
- `docs/notes/2026-05-05-colley-massey-clean-rerun.md` (findings)
- `docs/superpowers/specs/2026-05-05-colley-massey-clean-rerun-design.md` (this file)
- `docs/superpowers/plans/2026-05-05-colley-massey-clean-rerun.md` (next step)

**Updated:**
- `TODO.md` (replace 2 marginal-rejections bullets per decision matrix;
  step 5 closing note if both FAIL)

## Test plan

- Existing tests (run before code changes):
  - `tests/test_features/test_colley_matrix.py` (6 tests)
  - `tests/test_features/test_massey_matrix.py` (9 tests)
  All pass before proceeding. Status: 15/15 verified.
- No new tests for `clause2_colley.py`. The script is a thin driver
  composing `compute_colley_ratings` (tested in
  `test_colley_matrix.py`), `prepare_loso_inputs` (tested in
  `test_prepare_loso_inputs.py`), and `leave_one_season_out_cv_weighted`
  (tested across the v4 LOSO suite). This mirrors the
  `src/clause2_decay_massey.py` precedent (no dedicated test file).
- Procedural sanity gates (Procedure step 5): subset_seasons match,
  ll_without and ll_with both shift from PR 15's 0.4388 / 0.444, the
  two scripts agree on `ll_without` to ~1e-6.

## Acceptance criteria

- All existing pytest pass before regen.
- `output/clause2_decay_massey_hl14.json` regenerated with clean
  baseline numbers (delta could be PASS or FAIL; report actual).
- `output/diag_clause2_colley.json` produced by new runner.
- Both candidates' verdicts independently determined; findings doc
  has zero `<<...>>` placeholders.
- TODO.md updated per decision matrix.
- Both diag JSONs force-added per `docs/data_recovery.md` policy.
- `src/clause2_colley.py` mirrors the structure of
  `src/clause2_decay_massey.py` so that a future colley half-life
  sweep (if relevant) could be parameterized identically.
