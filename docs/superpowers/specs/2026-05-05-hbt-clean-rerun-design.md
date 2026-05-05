# Hierarchical BT Clean Re-eval (Step 5 marginal #3) -- Design

**Date:** 2026-05-05
**Branch:** feat/hbt-clean-rerun
**Recovery context:** TODO.md "CONTAMINATION DISCOVERED 2026-05-04" -> step 5,
sub-priority "HBT (PR 16) re-eval" (named the next-immediate marginal-rejection
remaining after PR 24 closed plain BT and PR 25 closed feature-view ensemble).
**Predecessors:**
- Original HBT findings (NO-GO under leaky baseline): `docs/notes/2026-05-03-hierarchical-bt.md`
- Original spec: `docs/superpowers/specs/2026-05-03-hierarchical-bt-feature-priors-design.md`
- Clean LOSO regen findings: `docs/notes/2026-05-04-v4-clean-loso-regen.md`
- Clean plain BT re-eval findings: `docs/notes/2026-05-05-plain-bt-clean-rerun.md`
- Clean feature-view re-eval findings: `docs/notes/2026-05-05-feature-view-clean-rerun.md`

## Motivation

PR 16 ran a per-cell 3-clause LL-blend gate over a 7-cell sigma sweep
{0.05, 0.10, 0.20, 0.50, 1.00, 2.00, 5.00} of hierarchical BT with v4
feature priors. Every cell PASSED clause 1 (residual correlation
`r in [0.448, 0.507]`) but FAILED clauses 2 and 3 (`w_opt = 0.99-1.00`,
`headroom = +0.0000`) -- HBT's standalone LL was uniformly worse than
plain BT (HBT range `0.619-0.757` vs plain BT `0.565`). Verdict: NO-GO
under leaky baseline.

PR 21's clean-baseline regen shifted v4's standalone LL from `0.4369`
to `0.5588` (+0.122). HBT's training inputs (v4's 67-feature matrix
via `prepare_loso_inputs()`) include the seven Vegas features filtered
by PR 19, so HBT's standalone LL is expected to shift on the clean
baseline (direction unknown a priori; v4-feature priors lose their
tournament-leak signal). The dominant question is whether residual
correlation tracks the pattern PR 24 + PR 25 surfaced -- when v4 loses
its leak signal, its errors become much more correlated with structurally
weaker peers (BT vs v4: `r=0.577 -> 0.868`; PEER_B vs v4: `rho=0.45 -> 0.726`).
If that pattern holds for HBT, clause 1 flips FAIL on most or all sigma
cells, while clauses 2/3 STAY FAIL because HBT remains weaker than
clean v4 standalone.

This PR tests the prediction directly: same trainer, same diagnostic,
same sigma grid, but `prepare_loso_inputs()` now flows through PR 19's
`filter_vegas_to_pre_tournament()`. The verdict closes (or, less
likely, reopens) the marginal-rejections list at HBT.

## Scope

**In scope.**

- Re-run `python src/train_hbt_stage1.py` against the clean v4 feature
  pipeline (post-PR-19). Regenerates 7 per-sigma pairwise CSVs at
  `output/pairwise_hbt_sigma_<S>.csv`, overwriting the tracked PR 16
  versions.
- Run `python src/diagnose_hbt_vs_v4.py` against `output/pairwise_v4.csv`
  (clean from PR 23) and the regenerated HBT cells. Capture the per-cell
  3-clause gate verdict + sweep-level "best passing cell" (or None).
- Force-add the artifacts:
  - 7 x `output/pairwise_hbt_sigma_<S>.csv` -- always force-add
    (HBT priors depend on Vegas-affected v4 features; CSVs WILL differ).
  - `output/diag_hbt_sweep.json` -- always force-add.
- Findings doc `docs/notes/2026-05-05-hbt-clean-rerun.md` mirroring
  PR 24/PR 25 structure: TL;DR, methods, per-cell gate result table
  (7 rows + "best passing cell" line), standalone metrics, comparison
  to PR 16 side-by-side, discussion (residual-correlation comparison
  vs PR 24's r=0.868 and PR 25's rho=0.726 findings), verdict,
  recommendation.
- TODO.md update: mark "HBT (PR 16) re-eval" sub-priority done with
  verdict + headline numbers (best-cell r, w_opt, headroom). Advance
  the priority list per decision matrix.

**Out of scope.**

- Re-tuning gate thresholds (`GATE_R_MAX = 0.60`, `GATE_W_LOW = 0.30`,
  `GATE_W_HIGH = 0.85`, `GATE_HEADROOM_MIN = 0.005`). They are absolute
  thresholds shared verbatim with `src/diagnose_bt_vs_v4.py` via
  cross-module regression test and are not sensitive to v4's baseline shift.
- Re-tuning `sigma_beta` (fixed at `1.0` per PR 16). The original
  findings note already speculated that sweeping `sigma_beta` would not
  flip the verdict; this PR doesn't reopen that question.
- Extending the sigma grid beyond the 7-cell {0.05, ..., 5.00} sweep.
- v9-C correction or production-side changes to `output/pairwise_probs.json`.
  This PR is gate-only.
- Re-evaluating the remaining marginal-rejections candidates
  (Colley clause-2 delta +0.0053, Massey-decay-hl=14d clause-2 delta
  +0.0057). Each gets its own PR per the TODO.md priority list.

## Decision matrix

| Gate verdict | Action |
|---|---|
| ALL CELLS FAIL (matches prior + closes list) | Findings note records new numbers + diff vs PR 16 (especially the residual-correlation pattern). TODO.md marks HBT closed; "recovery step 5" marginal-rejections list contains only Colley + Massey-decay-14d remaining. Production unchanged. |
| ANY CELL PASSES (best passing cell exists) | Findings note records new numbers + diff vs PR 16. Surprising result -- pre-registered prediction was that all cells would FAIL with c1 flipping FAIL across the board. TODO.md promotes "HBT-as-stage-1 candidate" to the next slot in step 5 sub-priorities, with v9-C correction + bracket-points backtest as the follow-up gate (per the original HBT spec's "if-pass" branch). Production unchanged this PR. |

The pre-registered prediction is **all 7 cells FAIL with clause 1 flipping
FAIL on most cells**, mirroring PR 24's BT residual-correlation jump
(r 0.577 -> 0.868) and PR 25's PEER_B residual jump (rho 0.45 -> 0.726).
HBT had the lowest residual correlation of any LL-gate candidate under
leaky baseline (0.448-0.507, vs plain BT 0.577 and PEER_A/B 0.45);
the question is whether that diversity survives the clean v4 baseline
or collapses the same way the others did. **Two scenarios are consistent
with prior pattern:**

1. **Strong-pattern repeat (most likely).** All 7 cells flip clause 1
   FAIL with `r > 0.60`. HBT remains standalone-weaker than clean v4,
   so clauses 2/3 STAY FAIL. Gate FAILs uniformly; HBT closes.
2. **Soft-pattern repeat (possible).** Residual correlation rises but
   stays below threshold for some cells (e.g. `r in [0.55, 0.59]`).
   Cells PASS clause 1 but FAIL clauses 2/3 as before. Gate still
   FAILs uniformly; HBT closes; the finding becomes "HBT diversity
   survives the clean baseline shift but is still not enough."

A clean-baseline cell flipping clauses 2/3 PASS (HBT becoming
standalone-comparable to clean v4) is unlikely a priori -- HBT's worst
sigma cell standalone LL was 0.619 leaky, clean v4 is 0.5588, so the
gap is ~0.06 LL even before HBT itself shifts on the clean baseline.

## Procedure

1. Worktree `feat/hbt-clean-rerun` off `main` (88447e5) with
   `data/raw/march-machine-learning-2026/`, `data/raw/kaggle/`, and
   `data/raw/vegas_lines/` junctioned to
   `C:\Users\alden\MarchMadness\data\raw\<subdir>\`. Status: complete
   (this PR's worktree setup; required `tar -xzf training_data.tar.gz`
   to repopulate the wiped subdirs in main repo before junctioning).
2. Run `python -m pytest tests/test_features/test_hierarchical_bt.py
   tests/test_train_hbt_stage1.py tests/test_diagnose_hbt_vs_v4.py -q`.
   All tests pass before proceeding.
3. **Append-mode caveat does NOT apply to HBT.** `train_hbt_stage1.py:142-144`
   calls `out_csv.unlink()` before each sigma's writes -- so the 7
   tracked CSVs get unlinked at run start and replaced with fresh
   clean-baseline content. (Plain BT and v4 use append-mode writers
   and require an `rm -f` first; HBT does not.)
4. Run `python -u src/train_hbt_stage1.py 2>&1 | tee output/_train_hbt_log.txt`.
   First call rebuilds Massey/Colley/efficiency caches from raw data
   (cache wiped along with raw subdirs); expect ~4 min cold-start +
   ~4 min trainer = ~8 min total.
5. Sanity gates on the trainer log:
   - 7 sigma blocks completed (one per `{0.05, 0.10, 0.20, 0.50, 1.00, 2.00, 5.00}`).
   - Each sigma block writes ~48,465 pairs (per-season tournament-field
     pair-counts identical to PR 16; field is data-driven and unchanged).
   - No `FIT ERROR` lines in the per-season summary.
6. Run `python src/diagnose_hbt_vs_v4.py 2>&1 | tee output/_diag_hbt_log.txt`.
   Captures per-cell + best-passing-cell into
   `output/diag_hbt_sweep.json` and stdout report.
7. Sanity gates on the diagnostic output:
   - `n_games == 1449` for every cell (matches PR 16 + PR 24 + PR 25
     matched-game set).
   - `ll_v4` matches clean baseline within 0.005 LL (target: `~0.5588`,
     accept `0.554-0.564`). If outside, halt -- the diagnostic is not
     seeing the clean v4 we think.
   - `ll_hbt` for each sigma cell shifts vs PR 16 (PR 16 leaky range
     `0.619-0.757`); direction unknown but a non-shift is suspicious
     (would imply v4 features are not in fact different on the clean
     baseline, contradicting PR 21).
8. Read the verdict from `output/diag_hbt_sweep.json`. Apply decision matrix.
9. Write findings doc + TODO.md update. Force-add 7 CSVs + diag JSON.
   Commit per the plan in `docs/superpowers/plans/2026-05-05-hbt-clean-rerun.md`.

## Risks

1. **Trainer compute time exceeds the user's TODO budget (~5 min).**
   PR 16's findings note reported 249 seconds for 7 cells x 22 LOSO seasons
   plus ~4 minutes of `prepare_loso_inputs()` cold-start. The data
   wipe today emptied `data/cache/` along with raw subdirs, so the
   first `prepare_loso_inputs()` call will rebuild Massey, Colley, and
   efficiency parquet caches. Total expected: ~8 min, comparable to
   PR 16's wall time. Not blocking; flag for findings doc methods section.
2. **Sanity gate `ll_v4 ~ 0.5588` failure.** The diagnostic reads
   `output/pairwise_v4.csv` (PR 23 clean force-add); if anything has
   touched that file since, gate fails. Mitigation: check md5 of
   `output/pairwise_v4.csv` against PR 23 record before running
   diagnostic. The expected md5 is in PR 21's findings note (or
   reproducible from `git log -- output/pairwise_v4.csv` in this
   branch's history).
3. **Trainer non-determinism between leaky and clean runs.** L-BFGS-B
   is deterministic given same inputs + same starting point. Initial
   `s_team_i = 0` per PR 16 setup. So the difference in HBT outputs
   should be entirely attributable to v4 feature shift (Vegas filter).
   If a cell's `ll_hbt` is byte-identical to PR 16, that's evidence
   the priors are not actually using clean v4 features -- halt and
   investigate (check `prepare_loso_inputs()` is calling
   `filter_vegas_to_pre_tournament`).
4. **Junction wipe risk during cleanup.** Per
   `feedback_windows_junction_delete.md` memory and
   `docs/data_recovery.md` § Prevention. Use `git worktree remove` for
   cleanup; do not use PowerShell `(Get-Item).Delete()` or recursive
   PowerShell removes on the worktree dir.

## Files of record

**Modified:**
- (none expected on the source side -- this PR is rerun + diagnostic
  only; no code changes are required by the spec)

**Created (force-added):**
- `output/pairwise_hbt_sigma_0.05.csv` (overwritten; tracked from PR 16)
- `output/pairwise_hbt_sigma_0.10.csv` (overwritten; tracked from PR 16)
- `output/pairwise_hbt_sigma_0.20.csv` (overwritten; tracked from PR 16)
- `output/pairwise_hbt_sigma_0.50.csv` (overwritten; tracked from PR 16)
- `output/pairwise_hbt_sigma_1.00.csv` (overwritten; tracked from PR 16)
- `output/pairwise_hbt_sigma_2.00.csv` (overwritten; tracked from PR 16)
- `output/pairwise_hbt_sigma_5.00.csv` (overwritten; tracked from PR 16)
- `output/diag_hbt_sweep.json` (overwritten; tracked from PR 16)
- `docs/notes/2026-05-05-hbt-clean-rerun.md` (findings)
- `docs/superpowers/specs/2026-05-05-hbt-clean-rerun-design.md` (this file)
- `docs/superpowers/plans/2026-05-05-hbt-clean-rerun.md` (next step)

**Updated:**
- `TODO.md` (mark sub-priority done, advance priority list per decision matrix)

## Test plan

- Existing tests:
  - `tests/test_features/test_hierarchical_bt.py` (5 tests)
  - `tests/test_train_hbt_stage1.py` (7 tests)
  - `tests/test_diagnose_hbt_vs_v4.py` (8 tests)
  All must continue to pass before regen + diagnostic. No new test code
  planned (the procedure adds no new code paths).
- Procedural sanity gates (Procedure step 5 + 7): 7 sigma blocks complete,
  ~48,465 pairs per cell, `n_games == 1449` per cell, `ll_v4 ~ 0.5588`
  within 0.005 LL.

## Acceptance criteria

- All existing pytest pass.
- 7 HBT pairwise CSVs regenerated, byte-different from PR 16 versions
  (HBT priors include Vegas-affected v4 features; non-shift would be
  evidence of a leak-fix bug).
- `n_games == 1449` for every diagnostic cell and `ll_v4` matches clean
  baseline within 0.005 LL.
- Findings doc cites the diagnostic JSON with its post-run numbers and
  shows the comparison-to-PR-16 side-by-side table.
- TODO.md updated per decision matrix.
- All artifacts force-added per `docs/data_recovery.md` policy.
