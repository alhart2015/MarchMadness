# Hierarchical BT Re-eval (Clean Baseline) -- Findings

**Date:** 2026-05-05
**Branch:** feat/hbt-clean-rerun
**Verdict:** **FAIL on all 7 sigma cells.** Robust NO-GO across both leaky and clean baselines. **Failure mode flipped vs PR 16.**
**Spec:** `docs/superpowers/specs/2026-05-05-hbt-clean-rerun-design.md`
**Plan:** `docs/superpowers/plans/2026-05-05-hbt-clean-rerun.md`
**Recovery context:** TODO.md "CONTAMINATION DISCOVERED 2026-05-04" -> step 5,
sub-priority "HBT (PR 16) re-eval" -- third and final marginal-rejection re-run
after PR 24 (plain BT) and PR 25 (feature-view ensemble PEER_A/B). All three
clean-baseline re-runs converged on the same closing pattern.

## TL;DR

Re-running PR 16's per-cell 3-clause LL-blend gate over the same
7-cell sigma sweep against clean `pairwise_v4.csv` (PR 23 force-added)
**keeps the NO-GO verdict but flips the failure mode on every cell**.
PR 16 had all 7 cells PASS clause 1 (`r in [0.448, 0.507]`) and FAIL
clauses 2/3 (`w_opt 0.99-1.00`, `headroom +0.0000`). This PR has all
7 cells FAIL clause 1 (`r in [0.678, 0.767]`); two cells (sigma=0.20,
0.50) PASS clause 2 but every cell still FAILs clause 3 (`headroom
+0.0009 to +0.0021`, all below the 0.005 threshold). Standalone HBT
LL barely moved (per-cell shifts in `[-0.0065, +0.0022]`); the
residual-correlation jump is entirely driven by clean v4 getting
worse, not by HBT changing. This is the third clean-baseline re-run
(after PR 24 plain-BT and PR 25 PEER_B) showing the same mechanistic
pattern: when v4 loses its tournament-leak signal, its errors align
with structurally weaker models' errors, and the LL-blend gate's
clause 1 falsifies the candidate. Marginal-rejections list shrinks by
one and the named items in PR 16/24/25 are now closed.

## Methods

- Inputs (read-only):
  - `output/pairwise_v4.csv` (clean baseline, PR 23 force-added).
    Diagnostic-measured `ll_v4 = 0.5579` on the 1449-game matched set
    (matches PR 24 / PR 25 to 4 decimals).
  - `data/raw/march-machine-learning-2026/MNCAATourneyCompactResults.csv`
    (matched-game source).
- HBT regen: `python -u src/train_hbt_stage1.py` against the clean v4
  feature pipeline (post-PR-19 `filter_vegas_to_pre_tournament`).
  269.6s wall time = 22 LOSO seasons x 7 sigma cells L-BFGS-B fits +
  one cold `prepare_loso_inputs()` call rebuilding Massey/Colley/
  efficiency caches (the data wipe earlier today emptied
  `data/cache/`). 7 fresh `output/pairwise_hbt_sigma_<S>.csv` files
  written, all md5-different from PR 16 versions:

| sigma | PR 16 md5             | clean md5             |
|-------|-----------------------|-----------------------|
| 0.05  | 1cb847e7059fd...      | eb64422415374...      |
| 0.10  | b2043166eee99...      | ec8bc0ed4fc1d...      |
| 0.20  | 3dc01f489465f...      | 4f3ba361ea037...      |
| 0.50  | 32ce8a71c71fe...      | 8dbb422cf7338...      |
| 1.00  | 93e81e2497145...      | 497cce1c6a8bc...      |
| 2.00  | 65503d1b497c9...      | 8a2caf7aab31c...      |
| 5.00  | ad24daa3f79b9...      | 2acdc88be1e8d...      |

- Diagnostic: `python src/diagnose_hbt_vs_v4.py`. Same gate thresholds
  as PR 16 / PR 24 (`GATE_R_MAX=0.60`, `GATE_W_LOW=0.30`,
  `GATE_W_HIGH=0.85`, `GATE_HEADROOM_MIN=0.005`; shared verbatim with
  `src/diagnose_bt_vs_v4.py` via cross-module regression test). 7-cell
  sigma sweep `{0.05, 0.10, 0.20, 0.50, 1.00, 2.00, 5.00}`, one row
  per cell; verdict picks best-headroom passing cell or returns None.
- Matched-game count: `n_games = 1449` per cell (identical to PR 16,
  PR 24, PR 25).
- Pre-run pytest: 20 tests pass
  (`tests/test_features/test_hierarchical_bt.py`,
  `tests/test_train_hbt_stage1.py`,
  `tests/test_diagnose_hbt_vs_v4.py`).

## Per-cell sweep result (clean baseline)

| sigma | r_resid | ll_hbt | acc_hbt | w_opt | headroom | c1 | c2 | c3 | verdict |
|-------|---------|--------|---------|-------|----------|----|----|----|---------|
| 0.05  | 0.697   | 0.6167 | 0.637   | 0.89  | +0.0009  | N  | N  | N  | FAIL    |
| 0.10  | 0.737   | 0.6240 | 0.658   | 0.88  | +0.0011  | N  | N  | N  | FAIL    |
| 0.20  | 0.767   | 0.6197 | 0.667   | 0.83  | +0.0021  | N  | Y  | N  | FAIL    |
| 0.50  | 0.764   | 0.6328 | 0.668   | 0.84  | +0.0020  | N  | Y  | N  | FAIL    |
| 1.00  | 0.747   | 0.6518 | 0.661   | 0.87  | +0.0016  | N  | N  | N  | FAIL    |
| 2.00  | 0.716   | 0.6886 | 0.658   | 0.90  | +0.0012  | N  | N  | N  | FAIL    |
| 5.00  | 0.678   | 0.7574 | 0.644   | 0.92  | +0.0010  | N  | N  | N  | FAIL    |

`v4 standalone LL = 0.5579, acc = 0.702`. Best passing cell: **None**.

## Standalone metrics (1449 played 2003-2025 tournament games)

| metric                 | clean v4 | HBT range across sigma | best HBT cell        |
|------------------------|----------|------------------------|----------------------|
| weighted-mean log loss | 0.5579   | 0.6167 - 0.7574        | 0.6167 (sigma=0.05)  |
| weighted-mean accuracy | 0.7019   | 0.637 - 0.668          | 0.668 (sigma=0.50)   |

The non-monotonic curve over sigma persists from PR 16: best LL at
sigma=0.05 (`0.6167`), best accuracy at sigma=0.50 (`0.668`), worst
on both at sigma=5.00. Even the strongest cell remains ~0.06 LL worse
than clean v4 -- a substantial gap that no blend-weight choice can
close on log loss alone.

## Comparison to PR 16 (leaky baseline)

| sigma | leaky ll_hbt | clean ll_hbt | leaky r | clean r | leaky w_opt | clean w_opt | leaky headroom | clean headroom | leaky verdict | clean verdict |
|-------|--------------|--------------|---------|---------|-------------|-------------|----------------|----------------|---------------|---------------|
| 0.05  | 0.6194       | 0.6167       | 0.448   | 0.697   | 1.00        | 0.89        | +0.0000        | +0.0009        | FAIL (c2,c3)  | FAIL (c1,c3)  |
| 0.10  | 0.6305       | 0.6240       | 0.485   | 0.737   | 1.00        | 0.88        | +0.0000        | +0.0011        | FAIL (c2,c3)  | FAIL (c1,c3)  |
| 0.20  | 0.6220       | 0.6197       | 0.505   | 0.767   | 0.99        | 0.83        | +0.0000        | +0.0021        | FAIL (c2,c3)  | FAIL (c1,c3)  |
| 0.50  | 0.6306       | 0.6328       | 0.507   | 0.764   | 0.99        | 0.84        | +0.0000        | +0.0020        | FAIL (c2,c3)  | FAIL (c1,c3)  |
| 1.00  | 0.6507       | 0.6518       | 0.492   | 0.747   | 1.00        | 0.87        | +0.0000        | +0.0016        | FAIL (c2,c3)  | FAIL (c1,c3)  |
| 2.00  | 0.6880       | 0.6886       | 0.472   | 0.716   | 1.00        | 0.90        | +0.0000        | +0.0012        | FAIL (c2,c3)  | FAIL (c1,c3)  |
| 5.00  | 0.7569       | 0.7574       | 0.448   | 0.678   | 1.00        | 0.92        | +0.0000        | +0.0010        | FAIL (c2,c3)  | FAIL (c1,c3)  |

Two non-trivial shifts and one striking non-shift across the sweep:

1. **Residual correlation jumps +0.21 to +0.27 on every cell**, all
   crossing the `0.60` threshold. Largest jump at sigma=0.20 (`0.505 -> 0.767`,
   delta `+0.262`); smallest at sigma=5.00 (`0.448 -> 0.678`, delta
   `+0.230`). Every cell flips clause 1 from PASS to FAIL.
2. **Optimal blend weight collapses from the degenerate 0.99-1.00 to
   a real interior optimum in [0.83, 0.92]**. Two cells (sigma=0.20,
   0.50) land inside the `[0.30, 0.85]` clause-2 PASS band; the other
   five remain just above it. Headroom inches up from `+0.0000` to
   `+0.0009 to +0.0021` -- materially nonzero but still well below
   the `+0.005` clause-3 threshold.
3. **Standalone HBT LL barely moves** (delta in `[-0.0065, +0.0022]`
   across the 7 cells, all an order of magnitude smaller than v4's
   `+0.121` LL shift). The leak fix in PR 19 changed v4's predictions
   on tournament games but did not meaningfully change HBT's
   tournament predictions. HBT's per-team strength `s_team` is fit
   jointly against regular-season W/L data (the dominant log-likelihood
   term) with v4 features acting as a soft Gaussian prior; the
   regular-season data is informative enough to dominate the prior at
   every sigma in the swept range, so reshaping v4's features barely
   shifts where HBT lands.

## Discussion

**Three independent confirmations of the same residual-correlation
pattern.** PR 24 found `r(resid_v4, resid_bt)` jumped from `0.577 -> 0.868`
when v4 went leaky->clean. PR 25 found `rho(resid_v4, resid_peer_b)`
jumped from `0.45 -> 0.726`. This PR finds the same: across all 7
HBT sigma cells, residual correlation jumps by `+0.21 to +0.27`, with
every cell crossing the 0.60 threshold from PASS to FAIL. Three
independent stage-1 candidates with different model classes (BT
logistic, XGB on a feature subset, hierarchical BT with v4 priors)
all show the same mechanistic effect: clean v4's residuals are
dominated by "hard regular-season-information" failures that any
same-data peer also fails, and that shared failure surface inflates
residual correlation back into the gate's reject zone. The hypothesis
floated as an open follow-up at the end of PR 24's findings -- "residual
correlation between v4 and a peer is bounded below by what both
models miss for hard regular-season-information reasons" -- now has
three confirmations.

**HBT specifically has the cleanest version of this story.** Plain BT
(PR 24) shifted under leaky->clean only via v4's residual change;
PEER_B (PR 25) shifted both because v4 changed AND because PEER_B's
own training inputs included Vegas features. HBT shifted **only**
through the residual-on-v4 channel: HBT's own predictions barely
moved (LL delta ~0.005 LL) despite its priors being computed from
v4's full 67-feature stack including the seven Vegas features
filtered by PR 19. This is direct mechanistic evidence that HBT's
sigma values across `{0.05, ..., 5.00}` are loose enough relative to
the regular-season W/L likelihood that the prior carries little
weight in the posterior. (For tighter priors at sigma << 0.05, this
would change; the original PR 16 spec sweep deliberately stopped at
0.05 because the spec-anchor `sigma -> 0` reduces HBT to a logistic
regression on regular-season W/L with v4 features, which the original
PR 11 LR experiment already characterized.)

**Why two cells passed clause 2 and what it means.** sigma=0.20 and
sigma=0.50 produced `w_opt = 0.83, 0.84` -- inside the clause-2
window. These are the two sigma values that produced the strongest
HBT standalone LL (0.6197 and 0.6328 respectively, the lower end of
the sweep range). Tighter HBT priors yielded slightly stronger
standalone HBT, which in turn shifted the optimum blend toward HBT
slightly. But these cells still failed clause 1 (`r > 0.7`) and
clause 3 (`headroom < 0.005`). The clause-2 PASS does not represent
a "near-miss" candidate; it confirms the gate's three clauses are
testing different things, and even when one clause flips PASS the
other two block the candidate.

**The headroom is real but small, and the LL gate is doing its job.**
Every cell shows `headroom > 0` -- a real LL improvement at the
optimal blend weight is available. Two cells ( sigma=0.20, 0.50)
beat clean v4 alone by `+0.002` LL each. PR 17's bracket-points re-test
on plain BT confirmed that the LL gate's verdict for plain BT was
correct on the production metric (every non-anchor cell lost vs the
canonical `v4 + v9-C` baseline). For HBT, the equivalent test would
require a v9-C correction + 22-season bracket-points backtest (~3
hours of compute). Per the spec's decision matrix and the precedent
of PR 17, we do **not** run that test when the LL gate fails. The
`+0.002` LL room available at `w_opt ~ 0.84` is the kind of small
edge that PR 17 already showed is not extractable on the production
metric -- and it sits inside a known-failed gate clause anyway.

**Marginal-rejections list closes.** PR 16/24/25 closed three of the
five named candidates from `docs/notes/2026-05-04-v9c-clean-rerun.md`
§ Follow-ups item 2. Two remain (Colley clause-2 delta `+0.0053`,
Massey-decay-hl=14d clause-2 delta `+0.0057`); both are smaller
clause-2 LL deltas on a different (Massey-MOV) gate than the
LL-blend gate the three closed candidates share, so they are not
load-bearing on the same shared mechanism. The three closures
strongly suggest those two will close the same way -- but the
TODO.md priority list properly defers them as their own PRs and they
are not gated on this finding.

**Implications for active queue.** Per the TODO.md re-prioritization
note (2026-05-04), the active queue items 1-3 (538 audit, single-
season v4 variance, external rankings as features) are all v4-internal
or v4-vs-external work, not "find another stage-1 ensemble peer."
This PR's strong confirmation of the same-data-peer ceiling makes
that prioritization tighter: the audit results from PR 22 (Vegas
buckets) plus the upcoming 538 audit are the most likely path to
lift v4's calibration in the regions where it actually loses, and
ensemble work at this data scale has run out of independent corners
to test.

## Verdict + recommendation

**NO-GO.** Hierarchical BT with v4 feature priors does not clear the
LL-blend gate at any sigma cell on the clean baseline. Robust NO-GO
across both leaky and clean baselines (with **failure mode flipped on
every cell**: PR 16 failed clauses 2/3 with c1 PASS, this PR fails
clauses 1/3 with c1 FAIL). The mechanism that closes HBT here is the
same one that closed plain BT in PR 24 and feature-view PEER_A/B in
PR 25: clean v4's residuals correlate with same-data peers' residuals
strongly enough to push Pearson r above the 0.60 falsification
threshold, regardless of model class.

HBT v9-C correction + bracket-points backtest (the if-pass branch
from the original HBT spec) is **SKIPPED** per spec decision matrix.
PR 17 already showed that for a candidate with `r > 0.60` clean,
v9-C does not extract bracket-points improvement.

## TODO.md update (this PR commits the update)

- Marginal-rejections list: HBT marked done with FAIL verdict + the
  failure-mode flip noted (clauses 1/3 fail clean, not clauses 2/3
  as in PR 16). Robust NO-GO across both baselines.
- Active queue: no change (538 audit, single-season variance, and
  external rankings remain in priority order).
- Step 5 sub-priorities: the three named LL-blend marginal candidates
  (plain BT, feature-view PEER_A/B, HBT) are now all closed. Two
  remaining (Colley, Massey-decay-hl=14d) deferred as own PRs.

## Files of record

- `output/pairwise_hbt_sigma_<S>.csv` x 7 (force-added with clean
  numbers; commit `61c3299`).
- `output/diag_hbt_sweep.json` (force-added with clean numbers; same
  commit).
- `docs/superpowers/specs/2026-05-05-hbt-clean-rerun-design.md` (commit `84d42dd`).
- `docs/superpowers/plans/2026-05-05-hbt-clean-rerun.md` (commit `49960fd`).
- No source-code changes -- this PR is rerun + diagnostic only.
- The PR 16-tracked `output/train_hbt_log.txt` and
  `output/diag_hbt_log.txt` are NOT updated by this PR (precedent: PR
  24 / PR 25 also did not update their leaky-baseline log files).
  They remain as the historical PR 16 record; the clean-rerun
  numerical record is in this findings doc + `output/diag_hbt_sweep.json`.

## Open follow-ups (not for this PR)

- **Colley + Massey-decay-hl=14d re-eval (TODO recovery step 5).**
  Both reported clause-2 LL deltas of `+0.0053` and `+0.0057`
  respectively in their original PRs (PR 15). On the clean baseline,
  the v4 LL anchor has shifted by `+0.121`, so those deltas could
  flip sign or shrink into noise. **Predicted outcome:** both close
  cleanly, similar to HBT here. They share the same mechanism (v4 +
  same-data peer hits the residual-correlation ceiling). But neither
  uses the LL-blend gate this PR tests; they use the
  `clause2_decay_massey.py`-style cheap gate over a 3-season subset.
  The mechanism transfer is plausible but not airtight.
- **538 audit (active queue #1).** Stays the immediate-next non-marginal
  experiment. The pre-registered framework from PR 18 (`audit_v4_gap_*.py`
  pattern) extends cleanly to a 538 baseline; sourcing investigation
  is the first sub-step.
- **Single-season v4 variance audit (active queue #2).** ~30 min
  cheap follow-up; would surface whether the user's Kaggle 2159/3462
  finish reflects a single-season tail of v4 calibration that the
  22-season-aggregate metrics smooth over.
