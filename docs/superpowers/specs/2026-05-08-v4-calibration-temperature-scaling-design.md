# v4 Calibration: Temperature Scaling -- Design

**Date:** 2026-05-08
**Branch:** `feat/v4-calibration-temperature-scaling`
**Predecessors:**
- R64 closing-line blend (FAIL): `docs/notes/2026-05-07-v4-r64-line-blend.md`
- Strategy frame: `docs/notes/2026-05-07-v4-kaggle-gap-strategy.md`
- Vegas audit (clean baseline): `docs/notes/2026-05-04-v4-gap-audit-vegas.md`
- 538 audit: `docs/notes/2026-05-04-v4-gap-audit-fte.md`
- Per-season variance: `docs/notes/2026-05-07-v4-per-season-variance.md`
- Vegas leak fix: `docs/superpowers/specs/2026-05-04-v4-vegas-leak-fix-design.md`

## Motivation

The R64 closing-line blend FAILed (delta +2 brkt pts hard, +0 mean
vs canonical 2069). The data hypothesis at the apply-time level is
falsified, and the TODO promoted **calibration-shape engineering** to
Active queue #1 by elimination.

The two audits gave a coherent picture even before the R64-blend null
result:

| audit | weak spot | direction | n |
|-------|-----------|-----------|---|
| Vegas | round=E8, S16 | v4 LL +0.055, +0.027 worse | 88, 176 |
| Vegas | v4 confidence 0.80-0.90 | v4 LL +0.025 worse | ~ |
| Vegas | upset, mid-seed-gap | v4 LL +0.054, +0.026 worse | ~ |
| 538   | chalk-won bucket   | v4 LL +0.075 worse | 298 |

Read together, these aren't *level* miscalibration (a single
multiplicative T can fix that). They're *shape* miscalibration:

- 538's chalk-LL gap (+0.075) means on chalk-won rows, v4's predicted
  prob for the chalk team is too far from 1.0 -- v4 hedges chalk
  picks. Sharpening (T<1) on those rows would push chalk probs higher
  and lower the LL.
- Vegas's 0.80-0.90 confidence-band gap (+0.025) means on rows where
  v4 already says ~0.85 for the favorite, v4 is over-confident on
  ones that lose and under-confident on ones that win -- but the net
  direction in this band on aggregate is that v4 is too low on the
  winners. Same direction as 538's chalk finding.
- Vegas's late-round gap (S16 +0.027, E8 +0.055) widens with round.
  This is what motivates per-round T (a round-level shape lever)
  rather than only a global T.

Temperature scaling can't be applied row-conditionally on
chalk-vs-upset (that's a row-level distinction, not a round-level
one), but R64 and R32 are where chalk concentrates by seed-gap
distribution, so per-round T_R64 and T_R32 indirectly address the
538-chalk finding. T_S16 and T_E8 directly address the Vegas
late-round finding. Phase 1 tests both levers together.

The per-season variance check (PR 30) showed ~25% CV on ECE across
21 seasons -- v4's calibration shape is variable year to year. A
season-invariant calibration won't fix a per-season-varying problem,
but it can lift the *expected* shape if the cross-season
miscalibration is monotone in some shared direction.

**The hypothesis under test:** v4's stage-1 (and v8's stage-2) outputs
are misshapen for the bracket-points objective in a way that a small
post-hoc rescaling of the final probabilities can correct. If true,
a Phase 1 post-hoc temperature sweep should clear the +10 MARGINAL
bar; if false, the calibration-shape lane is closed and the
data-hypothesis null + the calibration-shape null together strongly
elevate roster-level data (TODO #4) as the next investment.

## Goals

- **Implement post-hoc temperature scaling** as a pure function over
  v8's pairwise probabilities, parameterized by either a single
  global T or a 5-element per-round T vector (R64, R32, S16, E8,
  F4_NCG with F4 and NCG collapsed for n>=66 per knob).
- **Run two Phase-1 sub-experiments**, both gated on 22-season
  bracket points (the production metric):
  1. Single global T over a 7-cell grid `{0.7, 0.85, 1.0, 1.15, 1.3,
     1.5, 2.0}`. Anchor T=1.0 reproduces canonical `pairwise_v8.csv`
     byte-equal and 2069 brkt pts.
  2. Per-round T sequential greedy: hold all rounds at 1.0, sweep
     T_R64, fix at best, sweep T_R32, fix, ..., sweep T_F4_NCG. 5
     rounds * 7 cells = 35 evals. Anchor (1,1,1,1,1) reproduces 2069.
- **Phase 2 (conditional, only on PASS or MARGINAL from Phase 1):**
  scale v4 stage-1 with the winning T configuration, retrain v8 LOSO
  on scaled v4, re-score 22-season bracket points. Anchor: T=1.0
  retrain reproduces canonical `pairwise_v8.csv` byte-equal.
- **Decide.** Three outcomes per the spec decision matrix below:
  PASS (swap candidate), MARGINAL (candidate only / probe further),
  FAIL (close lane).
- **Ship a single experiment-record commit set + findings note.** No
  feature engineering on top; this is a stage-2 (and conditionally
  stage-1) post-hoc rescaling experiment.

## Non-Goals (deliberately deferred)

- **Isotonic regression.** Higher capacity than temperature scaling
  but not directly grid-searchable on bracket points (would be
  LL-fit + bracket-points-gate, with the known translation risk
  the R64 blend exposed). Deferred until Phase 1 lands. If Phase 1
  PASSes, isotonic is no longer needed; if Phase 1 FAILs, isotonic
  can be a separate spec but with elevated null-result risk.
- **Per-season T (T_S as a function of season features).** Adds a
  feature space that requires its own train/eval split and can leak
  test-set bracket-points info into T selection. Out of scope.
- **Late-stage "confidence sharpening" feature** (third option from
  the TODO note). Requires retraining v4; orthogonal to post-hoc
  rescaling and substantially more expensive.
- **Re-running v9-C with rescaled v8.** Canonical production stage-2
  is v8 (`pairwise_v8.csv` + 2069 baseline); v9-C exists in the repo
  as an experiment record but is not the production path per
  TODO's "Compounding work" note. v9-C eval over rescaled v8 can be
  added later if Phase 1/2 PASSes.
- **Live 2026 bracket regeneration.** This spec is backtest-only; a
  positive verdict triggers a separate "wire into the 2026 pipeline"
  follow-up.
- **Comparing against the leaky-baseline v4.** Leaky baseline is
  retracted; comparison is against clean v4 + clean v8 (PR 21
  numbers) only.
- **Women's tournament.** v4 is mens-only.

## Architecture

### Phase 1 data flow (post-hoc on v8)

```
output/pairwise_v8.csv              (canonical clean v8 stage-2 output, 2069 brkt pts)
       |
       v
[apply temperature scaling]   <----  T or T_per_round (sweep parameter)
       |                       <----  build_pair_round_lookup(season, slots, seeds)
       v                              for per-round mode
output/pairwise_v8_calibrated_<config>.csv  (per cell)
       |
       v
[score_chalk_brackets.score_pairwise_path]
       |
       v
total brkt pts + per-season + verdict
```

The scaling itself is pure logit-space math:

```python
def scale(p, T):
    # T = scalar or per-round dict; returns scaled probability
    return sigmoid(logit(p) / T)
```

Numerical guards: clip `p` to `[1e-9, 1 - 1e-9]` before logit to
avoid `inf`. Match `score_chalk_brackets`'s existing precision
conventions.

### Phase 2 data flow (conditional)

```
output/pairwise_v4.csv             (clean v4 stage-1)
       |
       v
[apply temperature scaling]   <----  winning T or T_per_round from Phase 1
       |
       v
pairwise_v4_calibrated.csv        (intermediate)
       |
       v
[train_stage2.fit_stage2 + build_v8_pairwise]   <-- LOSO retrain
       |
       v
output/pairwise_v8_phase2_<config>.csv
       |
       v
[score_chalk_brackets.score_pairwise_path]
       |
       v
verdict
```

Phase 2 anchor: feeding T=1.0-scaled v4 (== unmodified v4 to FP
precision) into the same retrain pipeline must reproduce canonical
`pairwise_v8.csv` byte-equal. This proves the retrain pipeline is
deterministic and the only difference between Phase 1 and Phase 2
is whether v8 trains on rescaled v4 or untouched v4.

### Per-round scaling: round labels

R64-blend uses `build_pair_round_lookup` from
`src/train_upset_model.py` to assign apply-time rounds. Reuse it
directly.

Round bucket definitions:

| bucket | rounds covered | n games (22 seasons) |
|--------|----------------|----------------------|
| R64 | 1st round (32 games/season) | 692 (incl. 2021's COVID-shortened 20) |
| R32 | 2nd round | 352 |
| S16 | Sweet 16 | 176 |
| E8 | Elite 8 | 88 |
| **F4_NCG** | **Final Four + championship combined** | **66** |

The F4-NCG collapse is non-trivial -- with NCG at n=22 alone, picking
T_NCG on a 7-cell grid would pick a 1- or 2-game-driven cell on
average. Collapsing forces n>=66 per knob (still small, but the
robustness reporting -- drop-best-season delta, W/L/T -- exposes
fragility). The collapse is documented and called out in the
findings doc.

### Sequential greedy for per-round T

Order: R64 -> R32 -> S16 -> E8 -> F4_NCG. Rationale:

- **R64 has the largest n (692)** -- the cell picked there is the
  most data-supported, so anchoring downstream rounds against it is
  robust.
- **Bracket-points scoring is round-stratified.** Each round
  contributes a fixed share of total points (R64 contributes 1pt *
  32 = 32 max/season; R32 = 64; S16 = 96; E8 = 128; F4 = 128; NCG
  = 64 if not double-counted -- actual point structure in
  `score_chalk_brackets`). Sequential greedy from earliest round
  matches the order of point accumulation.

Selection bias caveat: sequential greedy is in-sample selection on a
35-cell space. Joint optimization is exponential (7^5 = 16,807 cells)
and not necessary for a Phase-1 falsification test. Sequential
greedy is the codebase pattern (mirrors v9-C's W_UPSET-then-W_MISS
sweep and the SIGMA-then-mode pattern in R64-blend).

## Components (what gets built)

### New files (committed)

- `src/apply_temperature_scaling.py` (~120 LOC).
  - Public: `scale_pairwise(df, T) -> pd.DataFrame` -- T is `float`
    or `dict[str, float]` keyed on round bucket. Returns a new
    DataFrame with `p_a_wins` rescaled.
  - Public: `main(argv)` -- CLI for one-shot frame generation.
  - Private: `_logit(p)`, `_sigmoid(x)` with clipping guards.
  - Private: `_assign_round_bucket(season, team_a, team_b, lookup)`
    using `build_pair_round_lookup`. Maps (F4, NCG) -> "F4_NCG".

- `src/eval_v4_calibration.py` (~300 LOC).
  - Public: `run_global_T_sweep(v8_csv, T_grid, baseline_total) -> dict`.
  - Public: `run_per_round_greedy(v8_csv, T_grid, round_order,
    starting_T, baseline_total) -> dict`.
  - Public: `run_phase2(winning_config, v4_csv, baseline_total) -> dict`
    -- LOSO retrain + score.
  - Public: `main(argv)` -- CLI for full eval.
  - Reuses: `src.score_chalk_brackets.score_pairwise_path`
    (defined at `src/score_chalk_brackets.py:194`),
    `src.train_stage2.fit_stage2` (`src/train_stage2.py:104`),
    `src.train_stage2.build_v8_pairwise` (`src/train_stage2.py:167`),
    `src.train_upset_model.build_pair_round_lookup`
    (`src/train_upset_model.py:111`).

- `tests/test_apply_temperature_scaling.py` (~120 LOC, 6 unit + 1 smoke).
  - T=1.0 produces output equal to input (max_abs_diff < 1e-9).
  - T=2.0 monotonically flattens probabilities toward 0.5 (rank-preserving).
  - T=0.5 monotonically sharpens probabilities away from 0.5
    (rank-preserving).
  - Per-round T dispatches correctly: a 4-game synthetic frame with
    one game per round + an artificial "F4_NCG" row uses the
    correct T for each row's round bucket.
  - Numerical guard: input p in {0.0, 1.0} produces finite output
    after clipping (no NaN/inf).
  - Anchor regression: applying T=1.0 to canonical `pairwise_v8.csv`
    produces a frame byte-equal in the (season, team_a, team_b,
    p_a_wins) columns.
  - Smoke: end-to-end `scale_pairwise` over real `pairwise_v8.csv`
    completes without error and returns same row count.

- `tests/test_eval_v4_calibration.py` (~150 LOC, 5 unit + 1 smoke).
  - Anchor cell (T=1.0) in global sweep produces total brkt pts
    == 2069 to FP precision.
  - Anchor cell (1,1,1,1,1) in per-round greedy produces 2069.
  - Sequential greedy correctness: monotone improvement (or no
    decrease) in chosen-cell total across the 5 rounds (greedy
    invariant).
  - Verdict band assignment: a synthetic "+30 brkt pts" cell maps
    to PASS, "+15" to MARGINAL, "+5" to NO-GO.
  - Phase-2 anchor: re-feeding T=1.0 v4 through `fit_stage2 +
    build_v8_pairwise` reproduces canonical `pairwise_v8.csv`
    byte-equal (gated test; skips if `tests/data` Kaggle CSVs are
    absent, per Engineering follow-ups item).
  - Smoke: full Phase-1 sweep (global + per-round) runs in <5 min
    wall and writes the expected JSON keys.

### Generated artifacts (force-added per the existing pattern)

- `output/pairwise_v8_calibrated_global_T<best>.csv` -- best global
  T cell (only the winner, not all 7).
- `output/pairwise_v8_calibrated_perround_<T_R64>_<T_R32>_<T_S16>_<T_E8>_<T_F4NCG>.csv`
  -- best per-round cell.
- `output/pairwise_v8_phase2_<config>.csv` -- Phase 2 retrain output
  for the winning config (only if Phase 2 is triggered).
- `output/v4_calibration_eval.json` -- per-cell metrics + verdict +
  drop-best-season delta + per-season table for global and
  per-round sweeps.
- `output/v4_calibration_eval_log.txt` -- captured stdout.
- `output/v4_calibration_reliability.png` -- 3-line diagram (v8
  baseline, v8 with best global T, v8 with best per-round T) on the
  full 22-season frame, 10 bins.

### Modified files

- `TODO.md` -- on completion, move "v4 calibration-shape engineering
  (audit-derived)" entry from Active queue #1 to Done with the
  verdict + brkt-pts numbers. Update the queue's preamble to reflect
  the new lead candidate (depending on outcome).

## Anchors / verification

Pre-registered anchors. If any fails by more than the tolerance,
halt and debug before declaring a verdict.

| anchor | expected | tolerance | role |
|--------|----------|-----------|------|
| `scale_pairwise(df, T=1.0)` round-trip vs canonical `pairwise_v8.csv` | byte-equal in (season, team_a, team_b, p_a_wins) | max_abs_diff < 1e-9 | proves the scaling math is identity at T=1 |
| Global T anchor cell total brkt pts | **2069** (clean v4 + v8 baseline from PR 21) | exact (FP precision) | proves the eval harness reproduces canonical scoring |
| Per-round T anchor cell (1,1,1,1,1) total brkt pts | **2069** | exact (FP precision) | proves the per-round dispatch is identity at all-T=1 |
| Phase 2 anchor: retrain v8 on T=1.0-scaled v4 | byte-equal vs canonical `pairwise_v8.csv` | max_abs_diff < 1e-9 | proves Phase 2 retrain pipeline is deterministic and identity at T=1 |

The Phase 2 anchor is critical: an evaluation that doesn't reproduce
2069 with no rescaling can't be trusted to evaluate a rescaled
training input either. This mirrors the R64-blend Phase-1 anchor
exactly (which passed at max_abs_diff=0.0).

## Decision matrix

22-season aggregate brkt pts vs the **2069** baseline (canonical
clean v4 + v8). Per-cell verdict bands match the codebase's
established convention (R64-blend, BT-bracket-points, v9-C):

| outcome | brkt-pts delta | follow-up action |
|---------|----------------|------------------|
| **CLEAR PASS** (swap candidate) | `>= +25` on at least one cell, with W/L/T profile having >=6 wins, max-single-season-win <= 50% of total delta | trigger Phase 2 retrain; if positive there too, separate production-swap commit (analogous to v9-C swap) |
| **MARGINAL** (candidate only) | `+10 .. +24` aggregate **OR** PASS-magnitude with >50% single-season concentration | trigger Phase 2 retrain; do NOT swap into production based on Phase 1 alone; document as candidate |
| **FAIL** | `< +10` aggregate on every cell across both global and per-round sub-experiments | close the lane; calibration-shape hypothesis falsified at the post-hoc level; roster-level data (TODO #4) becomes lead candidate by elimination |

**Robustness columns reported per cell (in addition to aggregate
delta):**

- W/L/T over 22 seasons (cell vs 2069).
- Biggest single-season swing (signed, season-tagged).
- **Drop-best-season delta:** total minus the highest-positive-season
  contribution. Concretely, if the cell is "+18 of which +12 from
  2024 alone", drop-best-season delta is +6. If a cell's
  drop-best-season delta is negative, that flags single-season
  concentration even at PASS-magnitude. The R64-blend findings
  called out exactly this concentration ("250% of the +2 came from
  2011 alone"); making it a first-class column means it shows up
  consistently.
- Per-season delta table (22 rows) included in the JSON output.

The `>= +25` PASS bar is calibrated against the v9-C swap (which
won by +43 pts). Any swap-in needs to clear a similar bar to justify
pipeline complexity.

**Phase 2 trigger:** PASS or MARGINAL on at least one Phase 1 cell.
NO-GO across the board closes the lane and skips Phase 2 entirely
(matches the R64-blend FAIL-no-Phase-2 rule).

**Phase 2 verdict:** uses the same +25 / +10 / <10 bands but with
its own anchor (Phase-2 T=1.0 retrain reproduces 2069). A Phase 2
PASS is the swap-in trigger; Phase 2 MARGINAL stays as a candidate;
Phase 2 NO-GO documents that the apply-time signal didn't survive
retraining (the inverse of the R64-blend Phase-2-deferred concern).

## Risks

1. **Phase 1 selection bias.** Sweeping T on the same 22-season
   aggregate we score on is in-sample selection. For 7-cell global
   the bias is bounded; for 35-cell per-round greedy it's larger.
   **Mitigation:** drop-best-season delta as a first-class column;
   any cell whose delta is concentrated in a single season is
   flagged and not promoted to Phase 2 alone (must be backed by an
   independent cell to trigger Phase 2). Nested-LOSO selection is
   deferred to a Phase 2 robustness check only if a cell PASSes.
2. **Same-data-peer ceiling.** Every same-v4-data experiment since
   the leak fix (PR 12 BT, PR 14 PEER, PR 16 HBT, PR 24/25/26
   re-runs, PR 31 R64-blend) has either failed the gate or failed
   to translate LL gains to bracket points. Temperature scaling
   on v8's own output is also in this category. **Mitigation:** the
   experiment is cheap; a NO-GO is itself decisive evidence and the
   Phase 2 retrain (~30 min) gives the hypothesis a second chance
   before the lane is closed.
3. **F4_NCG collapse may hide signal.** Combining F4 (n=44) and NCG
   (n=22) into one bucket means a single T applies to two
   structurally different rounds. F4 is "best 4 vs best 4"; NCG is
   "best 2 vs best 2". **Mitigation:** if Phase 1 PASSes with a
   non-trivial T_F4_NCG, run a sensitivity probe in the findings
   note splitting the bucket and reporting per-round contribution.
4. **Sequential greedy may miss the joint optimum.** The 7^5 joint
   space is intractable to grid-search but the 35-cell sequential
   greedy is a known approximation. **Mitigation:** report the cell
   sequence (which T was picked at each step + the marginal delta);
   if any later step's marginal is larger than R64's, that signals
   sequential greedy may be suboptimal and a small joint refinement
   probe is warranted (e.g., re-sweep R64 holding the others at
   their best-found values).
5. **PR 19 leak-fix invariance.** Temperature scaling acts on
   pairwise probabilities only. It does not enter feature
   construction, season aggregates, or LOSO splits. The leak-fix
   path is orthogonal. **Mitigation:** verify by running
   `tests/test_filter_vegas_to_pre_tournament.py` (or whatever PR
   19's leak-fix regression test is named) as part of CI.
6. **Phase 2 retrain may diverge.** v8's XGBoost training is
   `random_state=42`-deterministic, but the input pairwise frame
   changes when v4 is rescaled. Stage-2 may converge to a different
   tree structure that interacts unpredictably with the rescaling.
   **Mitigation:** Phase 2 anchor (T=1.0 reproduces 2069 byte-equal)
   gates this; report the change in tree structure (n_estimators,
   actual feature importances) in the findings doc.

## Test plan summary

- **Unit:** scaling math (T=1 identity, T=2 flatten, T=0.5 sharpen,
  per-round dispatch, numerical guards), anchor regression on real
  CSV, verdict band assignment. ~6-8 tests on synthetic + real
  fixtures.
- **Smoke:** Phase 1 sweep produces well-formed JSON with anchors
  landing.
- **Phase 2 anchor:** retrain on T=1.0 v4 reproduces canonical
  `pairwise_v8.csv` byte-equal. Gated on Kaggle CSVs being unzipped
  (skip with informative message if absent).
- **Existing-suite regression:** full `pytest -q` green; no
  regression in leak-fix, v8/v9-C, or upset-model tests.

## Compute estimate

- Apply temperature scaling per cell: <100 ms (pure pandas + numpy
  on 48,465 rows).
- Score per cell via `score_chalk_brackets.score_pairwise_path`:
  ~10s wall (no XGB; just chalk simulation).
- Phase 1 global: 7 cells * ~10s = ~70s.
- Phase 1 per-round: 35 cells * ~10s = ~6 min.
- Phase 2 (conditional): 22-season LOSO XGB retrain * 1-2 configs
  = ~50s/season * 22 = ~20 min/config.
- **Total Phase 1 wall: ~7 min.** Total Phase 2 wall (if triggered):
  ~20-40 min.

## Forward links (post-merge)

If Phase 1 + Phase 2 PASS: write follow-up spec for the production
swap (analogous to v9-C production swap). The swap injects
temperature scaling between v8's output and the live bracket
pipeline (`generate_bracket_real.py` and `predict_2026_stage2.py`).

If Phase 1 PASSes but Phase 2 FAILs: the apply-time signal didn't
survive retraining. Document the discrepancy in the findings note;
this is a stronger signal than R64-blend's reverse pattern (apply-
time NO-GO doesn't even trigger Phase 2). Suggests T should be
treated as part of v8 training (T as a learned hyperparameter)
rather than post-hoc, which is a separate spec.

If Phase 1 FAILs entirely: write retrospective in the findings note
explaining what the null result means. A confirmed FAIL on
calibration-shape (after the data-hypothesis FAIL on R64-blend)
strongly elevates roster-level data (TODO #4) as the leading next
investment despite its sourcing cost. The pattern would be: same-data
fixes (BT, PEER, HBT, plain BT bracket-points, R64 line, calibration
shape) all fail; only structurally different signal moves the
bracket-points needle.

## Open design questions for plan-writing time

- Should the per-round bucket mapping (with F4_NCG collapse) live in
  `apply_temperature_scaling.py` or in
  `train_upset_model.build_pair_round_lookup`? Decide during
  plan-writing -- prefer the former if the F4_NCG collapse is the
  only deviation, since `build_pair_round_lookup` is also consumed
  by the upset-model code path that should NOT see the collapse.
- Should the eval driver write 35 intermediate CSVs (one per
  per-round-greedy cell) or pass DataFrames in-memory and only
  persist the winning cell's frame? Prefer in-memory for the sweep,
  persist only the winners (global + per-round) to disk. Decide at
  plan-writing time after checking `score_chalk_brackets`'s actual
  API surface (function takes a path; may need a sibling
  `score_pairwise_df` or a tempfile dance).
- Where do per-cell-per-season numbers go in the JSON shape?
  Mirror `output/r64_line_blend_eval.json` structure unless that
  layout proves awkward for the per-round greedy sequence reporting.
- Does Phase 2 need to also run the v9-C overlay, or is v8 alone
  sufficient given v8 is the canonical baseline? Default: v8 alone
  (matches R64-blend); if any Phase-2 cell PASSes, add v9-C overlay
  as a sensitivity probe (do not gate the verdict on it).
