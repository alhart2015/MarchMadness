# Plain BT Standalone Re-eval (Clean Baseline) -- Findings

**Date:** 2026-05-05
**Branch:** feat/plain-bt-clean-rerun
**Verdict:** **FAIL.** Gate clauses: r=0.868 FAIL, optimal_w=0.58 PASS, headroom=+0.0058 PASS. **Failure mode flipped vs PR 12.** Robust NO-GO across both leaky and clean baselines.
**Spec:** `docs/superpowers/specs/2026-05-05-plain-bt-clean-rerun-design.md`
**Plan:** `docs/superpowers/plans/2026-05-05-plain-bt-clean-rerun.md`
**Recovery context:** TODO.md "CONTAMINATION DISCOVERED 2026-05-04" -> step 5,
sub-priority "Plain BT standalone re-eval" (named highest signal/noise of the
marginal-rejections list in `docs/notes/2026-05-04-v9c-clean-rerun.md` § Follow-ups).

## TL;DR

Re-running PR 12's 3-clause LL-blend gate against the clean
`pairwise_v4.csv` (PR 23 force-added) **keeps the NO-GO verdict but
flips the failure mode**. PR 12 failed clauses 2 and 3 (degenerate
`w_v4=0.98`, headroom `+0.0000`) while passing clause 1 (`r=0.577`).
This PR PASSES clauses 2 and 3 (`w_v4=0.58`, headroom `+0.0058`) but
FAILS clause 1 (`r=0.868`, well above the 0.60 threshold). The
strength-gap collapse predicted by the spec did flip the two clauses
it was supposed to flip; what the spec did NOT predict (this is the
material finding) is that residual correlation jumped from 0.577 to
0.868 -- when v4 lost its tournament-leak signal, its errors became
much more similar to BT's errors. Ensemble averaging cannot help when
both models miss the same games. Plain BT is closed as a stage-1
ensemble peer; the marginal-rejections list shrinks by one.

## Methods

- Inputs (read-only):
  - `output/pairwise_v4.csv` (clean baseline, force-added in PR 23,
    md5 `<computed at audit>`).
  - `output/pairwise_bt.csv` (PR 12 force-add, md5 `3da859453...`).
    **Reproducibility check passed:** a fresh `train_bt_stage1.py` rerun
    (~10s) produced byte-identical output to the tracked file. BT is
    unchanged across the leaky→clean transition by construction (it
    trains on regular-season binary outcomes only) and the byte-compare
    confirms no environment drift either.
- Diagnostic: `python src/diagnose_bt_vs_v4.py --pairwise-v4 output/pairwise_v4.csv
  --pairwise-bt output/pairwise_bt.csv`. Same gate thresholds as PR 12
  (`GATE_R_MAX=0.60`, `GATE_W_LOW=0.30`, `GATE_W_HIGH=0.85`,
  `GATE_HEADROOM_MIN=0.005`). Procedure-side change this PR: added
  `--curve-out` flag (default `output/diag_bt_vs_v4_curve.csv`) so the
  full LL(w) curve is persisted alongside the slim JSON. Curve has 101
  cells (`w` in [0.00, 1.00] step 0.01).
- Matched-game count: `n_games = 1449` (identical to PR 12).

## Gate result

| measure                               | value      | clause                  |
|---------------------------------------|------------|-------------------------|
| Pearson r(residual_v4, residual_bt)   | **0.868**  | **FAIL** (< 0.60)       |
| optimal blend weight w_v4 (cheating)  | **0.58**   | **PASS** ([0.30, 0.85]) |
| headroom = LL_v4 - LL_optimal         | **+0.0058**| **PASS** (> 0.005)      |
| **gate verdict**                      | -          | **FAIL**                |

The two non-correlation clauses both flipped to PASS as the spec
predicted. Clause 1 went the other way -- the spec's risk #2 named
"residual correlation likely to stay near 0.58" as a low-probability
risk; that risk hit.

## Standalone metrics (1449 played 2003-2025 tournament games)

| metric                 | clean v4 | BT     | delta (v4 - BT) |
|------------------------|----------|--------|-----------------|
| weighted-mean log loss | 0.5579   | 0.5650 | -0.0071         |
| weighted-mean accuracy | 0.7019   | 0.6984 | +0.0035         |

v4 is now ~tied with BT in standalone strength (delta within ±0.01 LL
and ±0.5pp accuracy). PR 12's strength gap of -0.128 LL collapsed into
noise.

## Disagreement breakdown

| outcome                  | count | %     |
|--------------------------|-------|-------|
| both correct             |  916  | 63.2% |
| v4 only correct          |  101  |  7.0% |
| BT only correct          |   96  |  6.6% |
| both wrong               |  336  | 23.2% |
| total disagreements      |  197  | 13.6% |

When v4 and BT disagree on the predicted winner, BT is right 96/(96+101)
= 48.7% of the time. Compared to PR 12's 27.9%, BT's disagreement-side
accuracy nearly doubled -- but disagreements themselves dropped from
24.0% to 13.6% of games. Both effects are mechanistic consequences of
v4 losing its leak-derived edge: clean v4 makes fewer "confidently
different from BT" calls, and when it does, the disagreement is closer
to a coin flip rather than a v4 win.

## Selected w values from `diag_bt_vs_v4_curve.csv`

| w    | ll_blend |
|------|----------|
| 0.00 | 0.5650 (= ll_bt) |
| 0.25 | 0.5559   |
| 0.50 | 0.5523   |
| **0.58** | **0.5521 (= optimal_ll)** |
| 0.75 | 0.5530   |
| 1.00 | 0.5579 (= ll_v4) |

Shape: smooth U with strict monotone descent on [0.00, 0.58] and strict
monotone ascent on [0.58, 1.00]. The minimum is shallow -- the LL drop
from `ll_v4=0.5579` at the v4-anchor to the optimum is `+0.0058`, and
it's gone again by `w=0.30` (`ll=0.5547`, +0.0032 below the anchor).
A marginally-helpful blend exists in principle, but the gate's clause-1
failure says we shouldn't trust it -- the headroom comes from the
small residual-correlation slack that does exist (r=0.87 < 1.00), not
from genuine error-pattern diversity.

## Comparison to PR 12 (leaky baseline)

| measure                    | PR 12 (leaky)       | this PR (clean) | delta            |
|----------------------------|---------------------|-----------------|------------------|
| ll_v4                      | 0.4369              | 0.5579          | +0.1210 (worse)  |
| ll_bt                      | 0.5650              | 0.5650          |  0.0000 (unchanged)|
| ll_v4 - ll_bt              | -0.1281             | -0.0071         | +0.1210 (gap collapsed) |
| acc_v4                     | 0.805               | 0.702           | -10.3pp          |
| acc_bt                     | 0.698               | 0.698           |  0.0pp (unchanged)|
| residual r                 | 0.577               | 0.868           | +0.291           |
| optimal_w                  | 0.98                | 0.58            | -0.40            |
| headroom                   | +0.0000             | +0.0058         | +0.0058          |
| disagreement rate          | 24.0%               | 13.6%           | -10.4pp          |
| BT-when-disagree correct   | 27.9%               | 48.7%           | +20.8pp          |
| gate verdict               | FAIL (clauses 2, 3) | FAIL (clause 1) | failure flipped  |

BT's row is invariant (LL, accuracy unchanged) -- BT trains on
regular-season binary outcomes only and does not see the Vegas-feature
leak. Every shifted number traces to v4 alone, which lost ~0.121 LL
and ~10.3pp accuracy when its tournament-aware Vegas features were
filtered to pre-tournament games (PR 19 + PR 21).

## Discussion

**Why the gate's failure mode flipped.** The original PR 12 gate was
designed to filter ensemble candidates against three independent
failure modes:

1. **Residual correlation > 0.60** → errors too similar; averaging
   doesn't reduce variance.
2. **Optimal blend weight outside [0.30, 0.85]** → optimum is degenerate
   (effectively pure v4 or pure BT); averaging is a no-op.
3. **Headroom ≤ 0.005 LL** → optimal blend doesn't meaningfully beat
   v4 alone.

PR 12 hit (2) and (3) because BT was much weaker standalone. This PR
flips (2) and (3) because v4 and BT are now within 0.007 LL of each
other -- "much weaker standalone" is no longer the issue. But (1) hit
hard: residual correlation jumped from 0.577 to 0.868.

The mechanistic story is consistent with PR 22's finding that v4's
upset-detection edge over Vegas (56% vs 17% in the leaky baseline)
collapsed to 15.3% vs 17.5% in the clean baseline. Leaky v4's residuals
were dominated by genuine surprises (the leak couldn't make v4 right
on every tournament game; some upsets the leak couldn't predict),
which were also the games that *no* regular-season-only model could
predict. Those errors had a specific pattern uncorrelated with BT's.
Clean v4, having lost the tournament-leak signal, now misses the same
games BT misses -- both models' errors are dominated by "hard
regular-season-information games", which is a shared error pattern.
That shared pattern shows up directly as residual correlation.

**Generalized lesson** (third-time evidence). PR 12 found that LR's
residual correlation with v4 was 0.77; PR 14 found that PEER_A
(KenPom-only XGB) and PEER_B (form/market-only XGB) had non-trivial
correlation with v4 too. This PR adds a fourth data point: even a
**structurally different model class** (BT vs XGB) on **structurally
different features** (regular-season binary outcomes vs the 67-feature
v4 stack) produces residuals that correlate strongly with v4 once v4's
inflated edge is removed. The pattern is becoming hard to ignore: at
this data scale (~1449 played tournament games for the LL evaluation),
ensemble averaging on top of v4 is fighting a structural ceiling that
isn't model-class-specific. The 538/Vegas weak-spots from PR 22
(round=E8, chalk_won=upset, S16, etc.) are concrete bucket-level signal
that v4 itself is missing; "make v4 better at those buckets" is a more
promising direction than "average v4 with a peer that has the same
errors".

**The headroom is real but small.** The +0.0058 LL improvement at
w=0.58 is real (curve is smooth, optimum is well-defined, the gate's
own clauses 2 and 3 confirm it). PR 12's leaky-baseline headroom was
+0.0000; this PR's is +0.0058. We could pursue this on the production
metric anyway -- but the LL-gate clause-1 failure is exactly the
falsifier we asked for. Pushing past it would be selective reasoning,
and the spec's decision matrix (FAIL → drop BT) is clean: take the
NO-GO and move on.

**What this means for the bracket-points re-test (PR 17 redo).** The
spec said "Plain BT bracket-points re-test (PR 17 redo) -- separate
~3 hr follow-up if and only if this gate PASSES." Gate FAILED.
Bracket-points re-test SKIPPED. Two reasons:

1. The clean LL-gate has captured what the bracket-points re-test
   would have measured. Residual correlation 0.868 means a v9-C
   stage-2 over a v4+BT blend would not get materially different
   per-game probabilities than a v9-C stage-2 over v4 alone -- the
   blend is the v4 prediction nudged ~6% toward BT's prediction on
   each game, but on the games where it matters (disagreements), BT
   is at 48.7% accuracy.
2. PR 23 already showed that the bracket-points objective on the
   clean baseline penalizes ANY stage-2 manipulation that amplifies a
   weak signal (every v9-C cell lost ≥-140 brkt pts vs clean v8). A
   v4+BT blend that "moves" v4's predictions slightly is structurally
   similar to v9-C's role of moving v4's predictions slightly --
   both sit in the same "tweak v4's per-game probabilities" surface
   that just got falsified for v9-C.

## Verdict + recommendation

**NO-GO.** Plain BT does not clear the LL-blend gate even on the clean
baseline. The robust NO-GO across both leaky and clean baselines (with
flipped failure modes -- two distinct mechanisms isolated by the same
gate) closes plain BT as a stage-1 ensemble peer for v4. Plain BT
bracket-points re-test (PR 17 redo) skipped per spec decision matrix.

Marginal-rejections list shrinks by one. Next sub-priority becomes
**Feature-view ensemble PEER_A/B re-eval** (~20 min compute), per the
priority order in `docs/notes/2026-05-04-v9c-clean-rerun.md` § Follow-ups
item 2. The PEER_A/B re-eval is structurally similar to this re-eval
(rerun PR 14's clause-1 gate against clean v4) and is now the highest
signal/noise candidate left.

## TODO.md update (this PR commits the update)

- Marginal-rejections list: plain BT marked done with FAIL verdict;
  bracket-points re-test removed entirely (LL-gate failure across both
  baselines is sufficient to close BT).
- Active queue (step 5 sub-priorities): Feature-view ensemble PEER_A/B
  re-eval promoted to immediate next.

## Files of record

- `src/diagnose_bt_vs_v4.py` (modified: added `--curve-out` flag +
  `_write_curve` helper, ~25 lines net; commit 8070dc6).
- `tests/test_diagnose_bt_vs_v4.py` (modified: added 1 test for curve
  CSV writer, ~31 lines; same commit).
- `output/diag_bt_vs_v4.json` (overwritten with clean numbers; force-added
  in commit 974066e).
- `output/diag_bt_vs_v4_curve.csv` (new tracked artifact, 102 lines
  including header; same commit).
- `docs/superpowers/specs/2026-05-05-plain-bt-clean-rerun-design.md`
  (commit 9e046f8).
- `docs/superpowers/plans/2026-05-05-plain-bt-clean-rerun.md`
  (commit 84d3790).

## Open follow-up (not for this PR)

The residual-correlation jump from 0.577 to 0.868 generalizes a
hypothesis worth pre-registering for future stage-1 ensemble candidates:
**residual correlation between v4 and a peer is bounded below by what
both models miss for "hard regular-season-information" reasons**. If
this PR's r=0.868 is roughly the floor for any same-data peer of
clean v4, that's a strong falsification gate for future candidates.
The Feature-view ensemble PEER_A/B re-eval (next PR) will give a
second clean-baseline data point against this hypothesis -- if both
peers also land near r=0.85+, the gate has a sharper interpretation
("ensemble peers do not exist for clean v4 at this data scale; pursue
v4-internal improvements instead").
