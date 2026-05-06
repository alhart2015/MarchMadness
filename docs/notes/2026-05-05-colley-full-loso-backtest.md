# Colley Full LOSO Backtest -- REJECT

**Date:** 2026-05-05
**Branch:** feat/colley-full-loso-backtest
**Spec:** `docs/superpowers/specs/2026-05-05-colley-full-loso-backtest-design.md`
**Plan:** `docs/superpowers/plans/2026-05-05-colley-full-loso-backtest.md`
**Predecessors:**
- Original Colley spec (clauses 1+2): `docs/superpowers/specs/2026-05-03-colley-matrix-feature-design.md`
- Original Colley findings (clause 2 FAIL on leaky): `docs/notes/2026-05-03-colley.md`
- Wire-in revert: commit `3b4c374`
- Clean clause-2 PASS that triggered this PR: `docs/notes/2026-05-05-colley-massey-clean-rerun.md` (PR 27)
- Clean v4 baseline: `docs/notes/2026-05-04-v4-clean-loso-regen.md` (PR 21)
- Clean v8 bracket baseline: `docs/notes/2026-05-04-v9c-clean-rerun.md` (PR 24)
**Verdict:** **REJECT** per spec ladder. `LL_delta=-0.0003` LL is in
the Marginal band (in `(-0.005, +0.001)`); `brkt_delta=-24` brkt pts
triggers reject (`<= +10`). Wire-in reverted on this branch.

## TL;DR

PR 27's clean-baseline clause-2 result for Colley flipped from leaky
FAIL (+0.0053 LL on 3-season subset) to clean PASS (-0.0100 LL),
triggering the original Colley spec's full-LOSO-backtest if-pass
branch. This PR executed that branch: re-wired `colley_rating` into
`compute_all_features` (un-revert of `3b4c374`), regenerated
`pairwise_v4_with_colley.csv` via clean LOSO, retrained v8 stage-2 on
top, scored 22-season LL + bracket points head-to-head against
canonical clean baselines (LL 0.5588 / brkt 2069).

**Result: REJECT.** The LL improvement was real but tiny
(`mean_ll_delta = -0.0003` LL on 22-season aggregate, with 10/22
seasons helping vs 12/22 hurting -- a coin-flip pattern). Acc improved
modestly (`mean_acc_delta = +0.47pp`, 10/22 seasons help). But bracket
points regressed materially: `brkt_delta = -24` over 22 seasons,
worse on 10/22 seasons with bigger magnitude than the 10/22 seasons
where Colley helped. **Three big single-season hurts** drove the
verdict: 2009 (-33), 2015 (-30), 2019 (-28).

The pre-registered prediction in the spec leaned Marginal (~50%) /
Clear (~30%) / Reject (~20%). The actual outcome is Reject via the
bracket-points clause -- the 20% probability case. The pre-registered
"3-season subset over-represented Colley-helpful seasons" risk note
fired exactly: 2/3 of the subset seasons (2019, 2024) helped on
22-season LL but ALL THREE hurt on 22-season bracket points.

## Methods

**Worktree setup.** `feat/colley-full-loso-backtest` off `main`
(de763ad, post-PR-27) with subdir-level junctions for
`data/raw/march-machine-learning-2026/`, `data/raw/kaggle/`,
`data/raw/vegas_lines/`. Required `tar -xzf data/training_data.tar.gz
-C data/raw/` to repopulate the wiped subdirs in main repo before
junctioning -- same recurring data wipe pattern as PRs 24/25/26/27
(engineering follow-up at TODO.md "Test-suite hygiene").

**Wire-in.** 12 lines across 3 hunks of `src/enhanced_model.py`
restored, exactly inverting commit `3b4c374`:

- Top of `compute_all_features`: `from src.features.colley_matrix
  import load_colley_ratings; colley_full = load_colley_ratings(reg)`.
- Per-season block 2j: `season_colley_df = colley_full[...]; colley_map
  = dict(...)`.
- Per-team row assembly: `if tid in colley_map: row_data["colley_rating"]
  = colley_map[tid]`.

Diff verified with `git show 3b4c374` (12 lines added match the 12 lines
deleted by the revert). Pre-flight: 6/6 colley + 9/9 massey solver tests
pass on clean worktree.

**Pairwise regen.** `MM_PAIRWISE_OUT=output/pairwise_v4_with_colley.csv
MM_SKIP_DEFAULT_LOSO=1 MM_TUNED_PARAMS_V3=<PR 21 tuned params>` --
identical procedure to PR 21's clean LOSO regen, output redirected to
the non-canonical path so `output/pairwise_v4.csv` (the no-colley
canonical) was preserved untouched (md5 `795d8ddfcd7a0a09a50c3732825c6316`
verified before and after). Total runtime **57.8 minutes** (vs 3-hour
estimate; `MM_SKIP_DEFAULT_LOSO=1` halved it). Trainer summary at
68 features (+1 vs canonical 67): WT MEAN LL 0.5564, Acc 0.707, AUC
0.7877, Brier 0.1885.

**v8 retrain.** `train_stage2.py` reads `output/pairwise_v4.csv`
directly (hardcoded path). Used the documented swap-and-restore
pattern: copy `pairwise_v4_with_colley.csv` -> `pairwise_v4.csv`, run
trainer, rename trainer output to `pairwise_v8_with_colley.csv`,
restore canonical `pairwise_v4.csv` from backup, restore canonical
`pairwise_v8.csv` from git (force-added per PR 24). Both canonical
md5s verified unchanged after the swap.

Trainer summary: WT MEAN stage-1 LL 0.558, stage-2 LL 0.552 (delta
-0.006), stage-1 acc 70.7%, stage-2 acc 71.3% (delta +0.6pp). Matches
PR 24's clean v8 stage-2 contribution (-0.005 / +0.6pp) to within the
XGB tuned-params noise floor.

**Bracket-points scoring.** `src.score_chalk_brackets.score_pairwise_path`
on both `pairwise_v8_with_colley.csv` and `pairwise_v8.csv`. Canonical
sum reproduces PR 24's 2069 brkt pts exactly (drift +0). 1/2/4/8/16/32
weighting per round; chalk-bracket walk against
`MNCAATourneyCompactResults.csv`.

**LL/acc scoring.** Per-game scoring against actuals (DayNum >= 134
to filter to tournament games), unweighted. Canonical re-score: 0.5575
LL (vs PR 21's reported 0.5588; -0.0013 drift, well within the 0.01
sanity threshold and consistent with XGB process-noise plus the v3
weighted-LL vs unweighted-LL metric difference).

## Aggregate verdict table

| metric | without colley | with colley | delta | spec threshold | clause |
|---|---|---|---|---|---|
| 22-season mean LL    | 0.5575 | 0.5572 | **-0.0003** | reject `>= +0.001`; clear `<= -0.005` | LL: Marginal |
| 22-season mean acc   | 0.7022 | 0.7069 | **+0.0047** | (informational) | acc: helps +0.47pp |
| 22-season brkt total | 2069   | 2045   | **-24**     | reject `<= +10`; clear `>= +25` | **brkt: Reject** |
| seasons help on LL   | -- | 10 / 22 | -- | -- | -- |
| seasons help on acc  | -- | 10 / 22 | -- | -- | -- |
| seasons help on brkt | -- | 10 / 22 (10 hurt, 2 tie) | -- | -- | -- |

## Per-season detail

| season | n | LL_without | LL_with | LL_delta | Acc_without | Acc_with | Acc_delta | Brkt_without | Brkt_with | Brkt_delta |
|--------|----|---|---|---|---|---|---|----|----|---|
| 2003 | 64 | 0.578 | 0.564 | -0.0131 | 70.3% | 70.3% | +0.0pp | 85  | 73  | **-12** |
| 2004 | 64 | 0.543 | 0.538 | -0.0052 | 68.8% | 70.3% | +1.6pp | 72  | 67  | -5      |
| 2005 | 64 | 0.521 | 0.524 | +0.0035 | 73.4% | 71.9% | -1.6pp | 102 | 133 | **+31** |
| 2006 | 64 | 0.578 | 0.572 | -0.0055 | 71.9% | 73.4% | +1.6pp | 58  | 60  | +2      |
| 2007 | 64 | 0.481 | 0.479 | -0.0015 | 73.4% | 73.4% | +0.0pp | 132 | 156 | **+24** |
| 2008 | 64 | 0.479 | 0.492 | +0.0132 | 76.6% | 78.1% | +1.6pp | 128 | 140 | +12     |
| 2009 | 64 | 0.509 | 0.504 | -0.0054 | 73.4% | 71.9% | -1.6pp | 120 | 87  | **-33** |
| 2010 | 64 | 0.556 | 0.558 | +0.0020 | 70.3% | 67.2% | -3.1pp | 119 | 119 | 0       |
| 2011 | 67 | 0.668 | 0.658 | -0.0097 | 59.7% | 64.2% | +4.5pp | 47  | 53  | +6      |
| 2012 | 67 | 0.537 | 0.532 | -0.0045 | 71.6% | 71.6% | +0.0pp | 92  | 123 | **+31** |
| 2013 | 67 | 0.614 | 0.616 | +0.0017 | 64.2% | 64.2% | +0.0pp | 62  | 62  | 0       |
| 2014 | 67 | 0.578 | 0.581 | +0.0033 | 67.2% | 68.7% | +1.5pp | 61  | 64  | +3      |
| 2015 | 67 | 0.478 | 0.493 | +0.0148 | 74.6% | 77.6% | +3.0pp | 155 | 125 | **-30** |
| 2016 | 67 | 0.583 | 0.589 | +0.0059 | 71.6% | 70.1% | -1.5pp | 67  | 81  | +14     |
| 2017 | 67 | 0.547 | 0.545 | -0.0017 | 71.6% | 68.7% | -3.0pp | 101 | 93  | -8      |
| 2018 | 67 | 0.583 | 0.584 | +0.0016 | 70.1% | 70.1% | +0.0pp | 117 | 109 | -8      |
| 2019 | 67 | 0.509 | 0.494 | -0.0147 | 74.6% | 74.6% | +0.0pp | 125 | 97  | **-28** |
| 2021 | 66 | 0.582 | 0.593 | +0.0107 | 68.2% | 68.2% | +0.0pp | 78  | 84  | +6      |
| 2022 | 67 | 0.643 | 0.651 | +0.0078 | 64.2% | 65.7% | +1.5pp | 74  | 60  | -14     |
| 2023 | 67 | 0.624 | 0.626 | +0.0018 | 62.7% | 65.7% | +3.0pp | 49  | 45  | -4      |
| 2024 | 67 | 0.607 | 0.596 | -0.0107 | 68.7% | 70.1% | +1.5pp | 111 | 96  | **-15** |
| 2025 | 67 | 0.469 | 0.469 | +0.0003 | 77.6% | 79.1% | +1.5pp | 114 | 118 | +4      |
| **mean / total** | -- | **0.5575** | **0.5572** | **-0.0003** | **70.22%** | **70.69%** | **+0.47pp** | **2069** | **2045** | **-24** |

Bold-flagged: |brkt_delta| >= 12 (the threshold below which a single
season cannot, on its own, flip the 22-season verdict given the +/-10
spec band). Eight seasons cross that bar; five hurt (2009 -33, 2015
-30, 2019 -28, 2003 -12, 2024 -15) vs three help (2005 +31, 2007 +24,
2012 +31). Net of the bold-flagged seasons alone: -33 -30 -28 -12 -15
+ 31 + 24 + 31 = -32. The remaining 14 seasons sum to +8.

## Spec ladder application

Evaluated in order, per spec:

1. **Reject** if `LL_delta >= +0.001` LL OR `brkt_delta <= +10` brkt pts.
   - LL_delta = `-0.0003` -- does NOT trigger reject (in `(-0.005, +0.001)`).
   - brkt_delta = `-24` -- **DOES trigger reject** (`-24 <= +10`).
   - **VERDICT: Reject.** Stop ladder evaluation.

If the ladder had reached step 2 (Clear, `LL_delta <= -0.005` OR
`brkt_delta >= +25`), neither would have fired:
- LL_delta = `-0.0003` is far from `-0.005`.
- brkt_delta = `-24` is far from `+25`.

Step 3 (Marginal) would have applied if step 1 had not fired -- but
brkt-pts knocked it out at step 1.

The verdict is unambiguous: bracket points, not LL, drove the Reject.
The two metrics gave directionally consistent signals on bracket points
(reject) but disagreed on LL (small help). Per spec construction
(OR-of-fail at step 1), bracket-points failure dominates LL improvement
when the LL improvement is small.

## Comparison to PR 27 / generalized lesson

**The 3-season clause-2 PASS over-represented Colley-helpful seasons
on LL.** PR 27's clause-2 test on subset {2019, 2022, 2024} reported
mean delta `-0.0100` LL with all three subset seasons helping. On the
full 22-season test:

- 2019: subset clean `-0.0166`, full LOSO `-0.0147` (consistent direction; magnitude similar).
- 2022: subset clean `-0.0074`, full LOSO `+0.0078` (**inverted**).
- 2024: subset clean `-0.0059`, full LOSO `-0.0107` (consistent direction; magnitude grew).

The 2022 inversion is informative: the clause-2 subset run uses
`allowed_holdouts=[2019, 2022, 2024]`, meaning ALL three subset
seasons are simultaneously held out. The training set is thus 19
seasons (2003-2018, 2020-2021, 2023, 2025 minus 2020 which is excluded).
The full LOSO uses each season as the sole holdout in turn, so for
season 2022 the trainer sees 2019 and 2024 as well as the rest. The
training-distribution shift between subset and full LOSO can flip a
season's verdict for marginal-magnitude features like Colley. This is
exactly the pre-registered risk in the spec (Section "Pre-registered
prediction"): "If LL_delta lands at +0 to -0.001 LL ... the most
likely explanation is that PR 27's 3-season subset over-represented
Colley-helpful seasons." Outcome: LL_delta landed at `-0.0003`, just
inside that range.

**The 3-season subset says nothing about bracket points.** Clause 2
is an LL-only gate. Bracket points are a different, more discontinuous
metric -- a small probability shift can flip a chalk pick in any round,
and the 1/2/4/8/16/32 weighting amplifies upper-round flips. PR 27's
results gave no information about bracket-point sensitivity to Colley.

**Three seasons drove most of the bracket-points damage.** 2009
(-33), 2015 (-30), 2019 (-28). All three have LL deltas with mixed
signal:

- 2009 LL `-0.0054` (Colley helps slightly), brkt `-33`. Colley pulls
  some chalk picks toward upset directions where the LL improvement
  comes from better-calibrated upsets in lower rounds, but the upper-
  round picks shifted into wrong upsets.
- 2015 LL `+0.0148` (Colley hurts on LL), brkt `-30`. Direction-
  consistent with Reject. 2015 was Duke's championship -- a 1-seed
  champion tournament where chalk was the right pick and Colley's
  W/L-only signal pushed away from chalk in upper rounds.
- 2019 LL `-0.0147` (Colley helps significantly on LL), brkt `-28`.
  Most extreme directional disagreement. 2019 was Virginia's
  championship -- another 1-seed champion. Colley's LL help came from
  better calibrated lower-round picks (more "right amount of confident"
  on individual games), but the chalk picks in upper rounds got
  perturbed enough to flip wrong.

**Direction of bracket-point hurts is not random.** Pattern: chalk-
champion tournaments (2009 NC, 2015 Duke, 2019 Virginia, 2024 UConn)
are over-represented among the bracket-points hurts. Colley's W/L-
only opponent-adjusted strength penalizes top seeds slightly relative
to canonical v4 (which has efficiency-margin info), and the
penalization is enough to flip upper-round chalk picks in years where
chalk was correct.

**Contrast with Massey-decay-14d's clause-2 FAIL.** PR 27 found
Massey-decay-14d FAILed clause 2 at `+0.0018` LL on the same 3-season
subset. It was rejected at the cheap-gate layer; we never got to the
full LOSO. Had we run it, the full LOSO LL delta would likely have
been similar in magnitude to Colley's (both are W/L-or-margin variants
of the same opponent-adjusted-rating idea); the Massey-decay-14d
bracket-points behavior is unknown.

**Generalized lesson: a small clause-2 LL pass on a 3-season subset
is necessary but not sufficient -- bracket points can independently
veto.** This is the *third* layered failure mode the recovery work has
surfaced:

1. PR 24 (plain BT) and PR 25 (feature-view ensemble): residual
   correlation on the full 22-season backtest can flip from PASS to
   FAIL when v4 loses its leak (clean v4's errors become more
   correlated with weaker peers' errors).
2. PR 26 (HBT): the LL-blend gate filters on a metric (LL-blend
   headroom) that does not directly measure bracket-points
   improvement; PR 17's plain-BT bracket-points re-test confirmed
   the LL gate was right for plain BT but the framing carries
   forward.
3. **This PR (Colley): clause-2 LL on a 3-season subset can clear
   while clause-2 LL on 22-season aggregate barely clears AND
   bracket points actively hurt.** New mode -- the cheap gate's
   choice of test set (3 leak-sensitive seasons) over-represents
   the favorable case, and the production metric (bracket points)
   doesn't share the cheap gate's metric.

For future feature-addition gates, this argues for adding a small
"3-season clause 2.5" sanity check using a different, less leak-
sensitive subset (e.g. {2003, 2014, 2025}) before committing to the
3-hour full LOSO. The 3-hour cost was justified by PR 27's clean
PASS being the trigger -- the spec called for the full LOSO -- but
adding a cheap variance-of-subset diagnostic at clause 2 would have
flagged this PR's outcome ahead of compute.

## Verdict + recommendation

**Reject.** Wire-in reverted on this branch. Audit artifacts
(`pairwise_v4_with_colley.csv`, `pairwise_v8_with_colley.csv`,
`colley_full_loso_summary.json`) retained as the verdict's evidence
chain.

Closes Colley as a v4-stack feature candidate. The original Colley
spec's Out-of-scope items (time-decay weighting, prior tuning,
Colley-Massey blends, alternative clause-2 subsets) remain
out-of-scope -- the failure mechanism here (bracket-points
regression on chalk-champion years) is structural, not parameter-
tight; tuning won't fix it.

**Recovery step 5 marginal-rejections list status.** All 5 named
items are now closed:

- Plain BT standalone (PR 24): NO-GO, robust across both baselines.
- Feature-view ensemble PEER_A/B (PR 25): NO-GO.
- HBT (PR 26): NO-GO.
- Massey-decay-14d (PR 27): NO-GO.
- **Colley (this PR): NO-GO** (clause 2 PASSed but full LOSO REJECTed).

The marginal-rejections list is fully unwound. The next-up item on
recovery step 5 ("Regenerate v4's 2026 stage-1 predictions") and the
Active queue's #1 ("538 v4 gap audit") are now the highest-priority
next experiments.

## Files of record

**Created on this branch:**
- `docs/superpowers/specs/2026-05-05-colley-full-loso-backtest-design.md`
- `docs/superpowers/plans/2026-05-05-colley-full-loso-backtest.md`
- `output/pairwise_v4_with_colley.csv` (force-added, 48,465 rows;
  audit artifact)
- `output/pairwise_v8_with_colley.csv` (force-added, 48,465 rows;
  audit artifact)
- `output/colley_full_loso_summary.json` (force-added, per-season +
  aggregate metrics + verdict)
- `docs/notes/2026-05-05-colley-full-loso-backtest.md` (this file)

**Modified on this branch:**
- `src/enhanced_model.py` -- 12-line wire-in, then **reverted** in a
  separate commit on this branch (audit artifact only).
- `TODO.md` -- recovery step 5 sub-priority list marked DONE for
  Colley + marginal-rejections list closing summary updated.

**Untouched (canonical artifacts preserved):**
- `output/pairwise_v4.csv` (md5 `795d8ddfcd7a0a09a50c3732825c6316`)
- `output/pairwise_v8.csv` (md5 `102467bc485c20ffecc7e6644b46c85a`)
- `src/features/colley_matrix.py` (the solver)
- `src/diagnose_colley.py`, `src/clause2_colley.py` (cheap-gate runners)

## Follow-ups

None for Colley. Closed. Recovery roadmap moves to:

1. **Regenerate v4's 2026 stage-1 predictions** (recovery step 5
   sub-priority "NEW"). The current production
   `output/pairwise_probs.json` is clean v8 stage-2 over LEAKY 2026
   stage-1 (`output/pairwise_probs_v4.json` is Apr 28). Trace the
   producer, re-run on clean-trained v4, force-add. Modest compute.
2. **538 v4 gap audit** (Active queue #1). Reuse PR 18 framework
   against 538's published tournament-forecast probabilities. Sourcing
   investigation is the first task.
3. **Single-season v4 variance check** (Active queue #2). Plot per-
   season v4 LL + ECE; identify high-variance seasons. ~30 min audit.
