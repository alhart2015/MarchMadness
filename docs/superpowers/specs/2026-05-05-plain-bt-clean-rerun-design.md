# Plain BT Standalone Clean Re-eval (Step 5 marginal #1) -- Design

**Date:** 2026-05-05
**Branch:** feat/plain-bt-clean-rerun
**Recovery context:** TODO.md "CONTAMINATION DISCOVERED 2026-05-04" -> step 5,
sub-priority "Plain BT standalone re-eval" (named highest signal/noise of the
marginal-rejections list in `docs/notes/2026-05-04-v9c-clean-rerun.md` § Follow-ups).
**Predecessors:**
- Original BT findings (NO-GO under leaky baseline): `docs/notes/2026-05-01-bayesian-stage1.md`
- Original BT spec: `docs/superpowers/specs/2026-05-01-bayesian-stage1-design.md`
- Clean LOSO regen findings: `docs/notes/2026-05-04-v4-clean-loso-regen.md`
- Clean v9-C re-eval findings: `docs/notes/2026-05-04-v9c-clean-rerun.md`

## Motivation

PR 12 ran a 3-clause LL-blend gate on plain Bradley-Terry vs v4 and ruled
NO-GO: residual correlation `r=0.577` PASSED, but optimal blend weight
`w_v4=0.98` and headroom `+0.0000` LL FAILED. The dominant blocker was
that BT's standalone log loss `0.565` was `-0.128` below v4's leaky LL
`0.437` -- BT was simply too weak to add value on top of a strong baseline.

PR 21's clean-baseline regen shifted v4's LL from `0.437` to `0.5588`
(+0.122 under clean LOSO). BT, which trains on regular-season binary
outcomes only and has no Vegas-feature dependency, is unchanged. Under
the clean baseline, BT and v4 are now within `~0.006` LL of each other.
The PR 12 gate's two failing clauses were both consequences of the
strength gap; with the gap collapsed, the gate plausibly flips PASS.

This PR tests that hypothesis directly: same diagnostic, same thresholds,
same BT pairwise predictions, only `pairwise_v4.csv` swapped from leaky
to clean. The verdict gates whether plain BT is worth re-testing on the
production metric (bracket points, PR 17) -- which is a separate ~3 hr
follow-up PR if and only if this gate flips PASS.

## Scope

**In scope.**

- Reproducibility check: re-run `python src/train_bt_stage1.py` (~8 sec)
  and byte-compare new `output/pairwise_bt.csv` against the tracked one
  (PR 12 force-added). If md5 matches → discard re-run, keep tracked.
  If md5 differs → halt and investigate before drawing conclusions.
- Modify `src/diagnose_bt_vs_v4.py`: add `--curve-out` flag (default
  `output/diag_bt_vs_v4_curve.csv`). Always emit a 2-column CSV
  `w,ll_blend` with 101 rows (`w` from 0.00 to 1.00 step 0.01). Keep the
  JSON shape unchanged (do NOT re-add `ll_at_w` to the JSON; PR 12's
  consumers depend on the slim shape).
- Add one unit test in `tests/test_diagnose_bt_vs_v4.py` covering the
  new curve writer: header line `w,ll_blend`, 101 rows, value at the
  optimum row matches `optimal_ll` to 6 decimals.
- Run `python src/diagnose_bt_vs_v4.py --pairwise-bt output/pairwise_bt.csv
  --pairwise-v4 output/pairwise_v4.csv` against the clean v4 csv pulled
  from `main` (8a311be). Read 3-clause gate verdict.
- Force-add `output/diag_bt_vs_v4.json` (overwritten with new clean
  numbers) and `output/diag_bt_vs_v4_curve.csv` (new file) per the
  canonical-artifact pattern in `docs/data_recovery.md`.
- Findings doc `docs/notes/2026-05-05-plain-bt-clean-rerun.md` mirroring
  PR 12's structure: TL;DR, methods, gate result table (3 clauses),
  selected-w curve table, standalone metrics, disagreement breakdown,
  comparison-to-PR-12 side-by-side, verdict, recommendation.
- TODO.md update: mark "Plain BT standalone re-eval" sub-priority done
  with verdict + headline numbers (LL-blend headroom, optimal_w, residual r).
  Advance the priority list per decision matrix below.

**Out of scope.**

- Plain BT bracket-points re-test (PR 17). Separate PR if and only if
  this gate PASSES. Compute is ~3 hr (v9-C 22-season backtest); skipping
  it on a FAIL verdict saves the same amount.
- Production-side changes to `output/pairwise_probs.json` or any `predict_*.py`.
  This PR is gate-or-no-gate only; production changes (if any) belong to
  the bracket-points re-test PR.
- Re-tuning gate thresholds (`GATE_R_MAX=0.60`, `GATE_W_LOW=0.30`,
  `GATE_W_HIGH=0.85`, `GATE_HEADROOM_MIN=0.005`). They are absolute
  log-loss thresholds and are not sensitive to v4's baseline shift.
- Re-evaluating any of the other marginal-rejections candidates
  (Feature-view ensemble PEER_A/B, HBT, Colley, Massey-decay hl=14d,
  BT-as-feature, v9 weight-sweep family). Each gets its own PR per the
  TODO.md priority list.

## Decision matrix

| Gate verdict | Action |
|---|---|
| PASS (all 3 clauses clear) | Findings note records new numbers + diff vs PR 12. TODO.md promotes "Plain BT bracket-points re-test (PR 17)" to next slot in step 5 sub-priorities, with explicit ~3 hr compute warning. Production unchanged this PR. |
| FAIL (any clause fails) | Findings note records new numbers + diff vs PR 12. TODO.md drops plain BT entirely from the marginal-rejections list (LL-gate failed under both leaky and clean baselines = robust NO-GO). Next sub-priority becomes "Feature-view ensemble PEER_A/B re-eval" (~20 min). Production unchanged. |

The pre-registered prediction is PASS, per the findings doc that named
this re-eval. If the prediction holds, this PR is informative but
unsurprising; if it falsifies (gate FAILS even on clean baseline),
that's a stronger signal that BT-class peers don't help v4 regardless
of v4's own quality, and the marginal-rejections list shrinks faster
than the priority list expects.

## Procedure

1. Worktree `feat/plain-bt-clean-rerun` off `main` (8a311be) with
   `data/raw/march-machine-learning-2026/` junctioned to
   `C:\Users\alden\MarchMadness\data\raw\march-machine-learning-2026`.
   Status: complete (this PR).
2. Reproducibility step: `python src/train_bt_stage1.py`, then
   `md5sum output/pairwise_bt.csv` and compare to the tracked file's
   md5 in main. Halt on mismatch.
3. Code change: add `--curve-out` to `diagnose_bt_vs_v4.py`; add unit test.
4. Run `python -m pytest tests/test_diagnose_bt_vs_v4.py
   tests/test_train_bt_stage1.py -q`. All tests pass before proceeding.
5. Run `python src/diagnose_bt_vs_v4.py --pairwise-bt output/pairwise_bt.csv
   --pairwise-v4 output/pairwise_v4.csv`. Capture stdout and the two
   output artifacts.
6. Sanity gates on the run output:
   - `n_games == 1449` (matches PR 12's matched-game set).
   - Stdout `ll_v4` ≈ 0.5588 (clean v4 LL from PR 21/23, within 0.005).
   - Curve CSV has 101 rows, header `w,ll_blend`.
7. Read the gate verdict from `output/diag_bt_vs_v4.json`. Apply
   decision matrix.
8. Write findings doc + TODO.md update. Force-add JSON + curve CSV.
   Commit per the plan in
   `docs/superpowers/plans/2026-05-05-plain-bt-clean-rerun.md`.

## Risks

1. **BT byte-equality check fails.** Could be sklearn version drift
   (no env pin), CSV schema drift, RNG (BT trainer is deterministic but
   sklearn `LogisticRegression` may not be across versions), or silent
   data corruption. Halt and investigate before drawing conclusions
   from the diagnostic; the assumption "BT is unchanged across the
   leaky→clean transition" is load-bearing for this PR.
2. **`n_games` mismatch with PR 12 (1449).** Either `pairwise_v4.csv`
   has different pair coverage than PR 12 expected, or the join logic
   in `compute_diagnostic` drifted. Halt and reconcile; the matched-game
   count anchors the comparison-to-PR-12 table.
3. **Gate boundary case.** If `headroom` lands within `[0.005, 0.010]`
   or `optimal_w` lands within `[0.27, 0.33]` or `[0.82, 0.88]`, the
   PASS/FAIL split is sensitive to dedup ordering and clip epsilon.
   Mitigation: the curve CSV gives full visibility into LL(w) shape, so
   the findings doc can characterize a borderline result honestly
   instead of forcing a binary verdict.
4. **Append-mode artifact accumulation.** `train_bt_stage1.py` writes
   `pairwise_bt.csv` in append mode (per PR 12). The reproducibility
   check writes a *new* file -- but if the trainer's default output
   path already exists in the worktree (it does, from the tracked
   force-add), append mode would corrupt the byte-compare. Mitigation:
   write to a temp path (`output/pairwise_bt_repro.csv`), then compare
   against tracked, then delete temp.
5. **Junction wipe risk during cleanup.** Per `feedback_windows_junction_delete.md`
   memory and `docs/data_recovery.md` § Prevention. Use `git worktree
   remove` for cleanup; do not use PowerShell `(Get-Item).Delete()` or
   recursive PowerShell removes on the worktree dir.

## Files of record

**Modified:**
- `src/diagnose_bt_vs_v4.py` (+ ~15 lines: arg, helper, call site)
- `tests/test_diagnose_bt_vs_v4.py` (+ ~25 lines: one test)

**Created (force-added):**
- `output/diag_bt_vs_v4.json` (overwritten; already tracked from PR 12)
- `output/diag_bt_vs_v4_curve.csv` (new tracked artifact, ~5 KB)
- `docs/notes/2026-05-05-plain-bt-clean-rerun.md` (findings)
- `docs/superpowers/specs/2026-05-05-plain-bt-clean-rerun-design.md` (this file)
- `docs/superpowers/plans/2026-05-05-plain-bt-clean-rerun.md` (next step)

**Updated:**
- `TODO.md` (mark sub-priority done, advance priority list per decision matrix)

## Test plan

- Existing tests: `tests/test_diagnose_bt_vs_v4.py` (3 tests),
  `tests/test_train_bt_stage1.py` (4 tests). All must continue to pass.
- New test: 1 added test for curve CSV writer.
- Procedural sanity gates (Procedure step 6): `n_games == 1449`,
  `ll_v4` matches clean baseline within 0.005 LL, curve CSV shape.

## Acceptance criteria

- All existing + new pytest pass.
- BT byte-equality check succeeds OR mismatch is investigated and
  documented in the findings doc.
- `n_games == 1449` and `ll_v4` matches clean baseline within 0.005 LL.
- Findings doc cites the diagnostic JSON + curve CSV with their
  post-run numbers.
- TODO.md updated per decision matrix.
- All artifacts force-added per `docs/data_recovery.md` policy.
