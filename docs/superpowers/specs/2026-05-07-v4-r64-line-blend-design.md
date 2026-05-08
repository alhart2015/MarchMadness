# v4 R64 Closing-Line Blend -- Design

**Date:** 2026-05-07
**Branch:** `feat/v4-r64-line-blend`
**Predecessors:**
- Strategy frame: `docs/notes/2026-05-07-v4-kaggle-gap-strategy.md`
- Vegas audit (clean baseline): `docs/notes/2026-05-04-v4-gap-audit-vegas.md`
- 538 audit: `docs/notes/2026-05-04-v4-gap-audit-fte.md`
- Per-season variance: `docs/notes/2026-05-07-v4-per-season-variance.md`
- v9-C production (stage-2 over which we apply the override):
  `docs/superpowers/specs/2026-05-01-v9c-production-swap-design.md`
- Vegas leak fix (defines what's leak-free):
  `docs/superpowers/specs/2026-05-04-v4-vegas-leak-fix-design.md`

## Motivation

The strategy note (above) lays out the case: v4's clean LOSO log loss
(0.5588) is competitive on the 22-season aggregate but the user
finished 2159/3462 on Kaggle. Same-data peers all fail; Vegas
beats v4 head-to-head on the aggregate (LL +0.0148). The most
likely explanation is **information starvation, not modeling
deficiency** -- v4 doesn't have access to a market-consensus signal
that median Kaggle competitors plausibly do.

The cleanest leak-free Vegas signal we can use at submission time is
the **R64 closing line** for each of the 32 first-round games. R32+
lines are not available at submission time (matchups don't exist
until prior rounds resolve), so this experiment is scoped to R64
games only. It addresses ~51% of the bracket-points denominator in
one shot.

**The Vegas audit's R64-specific numbers** (from
`output/v4_gap_audit_vegas.json::by_round.R64`, 648 games over 22
seasons):

| metric | v4 | Vegas | delta (v4 - vegas) |
|--------|------|------|---------------------|
| log loss | 0.5164 | 0.5045 | **+0.0118** (Vegas better) |
| accuracy | 0.7253 | 0.7315 | -0.6 pp |
| ECE | 0.0605 | 0.0395 | +0.021 (Vegas better-calibrated) |

The +0.0118 LL gap on R64 is the upper bound on what a perfect
R64-line override could buy us in stage-1 LL on those 32 games per
tournament. The translation through v9-C to bracket points is the
unknown the experiment exists to measure.

## Goals

- **Build the leak-free R64-line override.** A function that, given
  v4's pairwise probabilities and the loaded Vegas line table,
  produces a *modified* pairwise frame where the 32 R64 games per
  season have their probabilities replaced (or blended) with the
  closing-line implied probability.
- **Score it through v9-C on 22-season LOSO.** Use the existing
  bracket-points scoring infrastructure (`src/score_chalk_brackets.py`
  / `src/sweep_bt_bracket_points.py` style) to score the modified
  pairwise frame against actual tournament outcomes for each LOSO
  season. Compare to the canonical `v4 + v9-C` baseline of **2069
  brkt pts over 22 seasons** (PR 21 clean re-run number).
- **Decide.** Three outcomes per the spec decision matrix below:
  PASS (swap candidate), MARGINAL (candidate only / probe further),
  FAIL (close lane).
- **Ship a single experiment-record commit set + findings note.** No
  feature engineering on top; this is a stage-1-substitution
  experiment, not a feature-add. (Feature-add is a separate spec --
  see Non-Goals.)

## Non-Goals (deliberately deferred)

- **Adding R64 line as a v4 input feature and retraining v4.** That's
  a different experiment (an actual training change, full LOSO regen,
  multi-day compute). This spec is the *cheap apply-time
  override* -- if the override doesn't move bracket points, no point
  paying the retraining cost.
- **Pre-tournament futures-derived team strength as a v4 feature
  (#1b in the strategy note).** Needs new data sourcing (Vegas
  Insider / Action Network historical futures); separate spec.
- **Round-aware (R32+) lines.** Impossible at submission time (see
  strategy note). Out of scope by construction.
- **Re-training v9-C on the modified stage-1 distribution.** v9-C is
  trained on v4's R64 distribution; an R64 override changes stage-2's
  input distribution. Re-training v9-C on the modified distribution
  is a follow-up if the apply-time override is MARGINAL or PASS-and-
  fragile. The cheap apply-only test runs first.
- **Live 2026 bracket regeneration.** This spec is backtest-only; a
  positive verdict would trigger a separate "wire into the 2026
  pipeline" follow-up (analogous to PR's v9-C production swap).
- **Comparing against the leaky-baseline v4.** Leaky baseline is
  retracted; comparison is against clean v4 + clean v9-C only.
- **Women's tournament.** v4 is mens-only.

## Architecture

### Data flow at apply time

```
output/pairwise_v4.csv           (clean v4 stage-1, 22 seasons * ~63 games)
       |
       v
[apply R64 override]  <----  data/raw/vegas/ncaabb*.csv  (closing lines)
       |                <----  data/raw/march-machine-learning-2026/
       v                       MNCAATourneyCompactResults.csv (round labels)
output/pairwise_v4_r64lineblend.csv
       |
       v
[v9-C stage-2]  <----  trained on UNMODIFIED v4 in LOSO (per Non-Goals)
       |
       v
[bracket points scorer]
       |
       v
verdict
```

### Override semantics: hard vs blended

Two override variants. Spec implements both behind a `--blend MODE`
flag; backtest sweeps both and reports.

- **`hard` (replace).** For each of the 32 R64 (season, team_a, team_b)
  triples per tournament, replace v4's `p_a_wins` with the
  Vegas-line-implied `p_a_wins`. All other rows pass through
  unchanged. Maximum signal injection but most disruptive to
  v9-C's input distribution.
- **`mean` (50/50 blend).** Replace with `0.5 * p_v4 + 0.5 * p_vegas`.
  Compromise between v4's representation and Vegas's market signal.
  Conservative.

A learned blend (logistic of `(p_v4, p_vegas, line, seed_diff)`) is a
deferred Phase-2 if Phase-1 is MARGINAL.

### Spread-to-probability conversion

Vegas closing lines are point spreads; we need a probability. Use
the Vegas audit's existing `_spread_to_prob(spread, sigma=SIGMA=11)`
from `src/audit_v4_gap_vegas.py`. Re-import directly; do not
re-implement.

The SIGMA=11 caveat carries over: the value was chosen for the audit
and may be slightly too peaky for tournament games. Sensitivity check
included (sweep `sigma in {9, 10, 11, 12, 13}`); pick the one
that minimizes 22-season aggregate LL on the R64 subset, then run
the bracket-points backtest only at the picked sigma.

### Join logic for R64 games

Reuse the same join pipeline that `_build_per_game_audit_df` in
`src/audit_v4_gap_vegas.py` uses:

1. Tournament round labels via `_round_from_daynum`.
2. Filter to `round == "R64"`.
3. Vegas-line lookup keyed on `(season, daynum, team_a, team_b)`,
   resolved via the same name-resolution path
   (`_build_vegas_name_to_kaggle_map` + `_resolve_vegas_name` +
   fuzzy-cache).

Coverage at the R64 level was ~92% in the Vegas audit; same here.
For uncovered R64 games (no Vegas line resolved), pass through v4's
probability unchanged. Log per-season coverage in the JSON output.

### Backtest harness

We need a 22-season LOSO scoring run that produces total bracket
points + per-season bracket points. The existing
`src/sweep_bt_bracket_points.py` is the closest analog (it sweeps a
weighted blend of v4 + BT and scores each cell). Patterns to reuse:

- `score_chalk_brackets` style: read pairwise CSV, simulate v9-C
  stage-2, score against `MNCAATourneyCompactResults.csv` per
  season.
- LOSO discipline: each season's score is computed using v9-C
  trained on the other 21 seasons. **v9-C's LOSO training does NOT
  see the R64-overridden frame** -- v9-C is trained as it normally
  is, and the override is applied at apply time only. (The exact
  disciplined version is "train v9-C on the unmodified `pairwise_v4`,
  then apply v9-C to the R64-overridden frame for that holdout
  season.")

If `sweep_bt_bracket_points.py` cannot be cleanly reused, write a
sibling driver `src/eval_r64_line_blend.py` modeled on it. Avoid
forking core scoring logic -- use the same pre-existing chalk
scorer.

## Components (what gets built)

### New files (committed)

- `src/build_r64_line_override.py` (~150 LOC).
  - Public: `apply_r64_override(pairwise_v4_csv, vegas_lines_df, mode, sigma) -> pd.DataFrame`
  - Public: `main(argv)` -- CLI for one-shot frame generation.
  - Private: `_build_r64_pair_index(season, results, seeds)` --
    enumerate the 32 R64 (season, team_a, team_b) triples.
  - Private: `_get_vegas_p_for_pair(season, daynum, a, b, vegas_lookup, sigma)`.
  - Reuses (does not re-implement): `load_vegas_lines`,
    `_build_vegas_name_to_kaggle_map`, `_resolve_vegas_name`,
    `_vegas_to_seasonday`, `_build_vegas_lookup`,
    `_build_day_zero_map`, `_load_seeds_lookup`, `_load_v4_lookup`,
    `_round_from_daynum`, `_spread_to_prob`. Cross-module imports
    same pattern as the variance check (audit drivers as helper
    libraries; one-off diagnostic, not long-lived dependency).

- `src/eval_r64_line_blend.py` (~200 LOC).
  - Public: `run_eval(v4_csv, mode, sigma, out_json) -> dict` -- the
    22-season LOSO bracket-points runner.
  - Public: `main(argv)` -- CLI for full eval.
  - Reuses v8/v9-C scoring infrastructure (whichever
    `sweep_bt_bracket_points.py` uses).

- `tests/test_build_r64_line_override.py` (~100 LOC, ~6-8 tests).
  - Override correctness on a 3-game synthetic fixture.
  - Coverage path: 32 R64 games per season expected; flag short
    seasons (e.g., 2021 had 4 cancellations).
  - SIGMA sensitivity smoke (probability falls in [0.05, 0.95] for
    `|spread| <= 25`).
  - Pass-through correctness for non-R64 games (round != "R64").
  - Hard-vs-mean mode produces different probabilities only on
    R64 rows.

- `tests/test_eval_r64_line_blend.py` (~80 LOC, 1 smoke + 2 unit).
  - LOSO discipline: a unit test that v9-C training does NOT see
    the override (the override-training-isolation invariant).
  - Smoke: full 22-season run, verify JSON shape + total bracket
    points reasonable.

### Generated artifacts (force-added per the existing pattern)

- `output/pairwise_v4_r64lineblend_hard.csv` -- 22-season pairwise
  frame with hard override applied.
- `output/pairwise_v4_r64lineblend_mean.csv` -- same with mean blend.
- `output/r64_line_blend_eval.json` -- per-season + aggregate
  bracket-points + LL + accuracy under each (sigma, mode) cell.
- `output/r64_line_blend_eval_log.txt` -- captured stdout.
- `output/r64_line_blend_calibration.png` -- two-panel R64 calibration
  diagonal: v4 vs (hard, mean) blend, side-by-side.

### Modified files

- `TODO.md` -- on completion, move "External Data #1: Vegas
  closing-line blend" entry from Active queue to Done with the
  verdict + brkt-pts numbers.

## Anchors / verification

Three pre-registered numerical anchors. If any fails by more than the
tolerance, halt and debug before declaring a verdict.

| anchor | expected | tolerance | role |
|--------|----------|-----------|------|
| Hard-mode R64-only LL on 22-season aggregate | 0.5045 (Vegas audit's R64 ll_vegas, by construction) | <= 0.001 | proves the override is doing what we think it's doing |
| Anchor cell `(mode=v4-only)` total brkt pts | **2069** (clean v4 + v9-C baseline from PR 21) | exact (FP precision; v9-C unchanged) | proves the eval harness reproduces canonical scoring |
| R64 coverage | >= 90% (matches Vegas audit's 91.5% join coverage; expect ~92% on R64 specifically) | informational | flags any new join regression |

The `mode=v4-only` anchor is critical: an evaluation that doesn't
reproduce 2069 with no override applied can't be trusted to evaluate
the override either. Implement this as a pinned regression test.

## Decision matrix

22-season aggregate brkt pts vs the **2069** baseline (canonical
v4 + v9-C):

| outcome | brkt-pts delta | action |
|---------|----------------|--------|
| **PASS** (swap candidate) | `>= +25` on at least one (mode, sigma) cell, robust 6+ winning seasons in a 6W-Xl-Yt profile (Y >= 8 ties), max-single-season-win <= 50% of total delta | proceed to follow-up: re-train v9-C on overridden frame; if positive there too, swap to production |
| **MARGINAL** (candidate-only) | `+10 .. +25` aggregate, OR PASS-magnitude but with single-season >50% concentration | retain code on branch as experiment record; do NOT swap; calibration-shape engineering takes lead by elimination |
| **FAIL** | `< +10` aggregate on every (mode, sigma) cell | close the lane; calibration-shape engineering becomes #1; futures-as-feature spec (#1b) reconsidered with this null result as evidence |

Direction matters: if `hard` and `mean` both lose to v4-only, the
override is *worse* than v4 -- that would be a strong falsification
of the data hypothesis and a meaningful update toward
calibration-shape engineering being the real lever.

The `>= +25` PASS bar is calibrated against the v9-C swap (which
won by +43 pts). Any external-data swap-in needs to clear a similar
bar to justify the additional pipeline complexity (an extra data
source on the apply-time path).

## Risks

1. **v9-C distribution mismatch.** v9-C is trained on v4's R64
   distribution. Overriding R64 at apply time changes stage-2's
   input distribution. Stage-2 may misbehave on out-of-distribution
   inputs; this is the largest single risk. **Mitigation:**
   pre-registered anchor (`mode=v4-only` reproduces 2069); if the
   override loses bracket points despite winning R64 LL, this is
   the first hypothesis to test by retraining v9-C on the overridden
   frame (a follow-up, not in scope here).

2. **Coverage holes.** ~8% of R64 games may not have a Vegas line
   resolved (name-resolution misses, missing CSVs). For those, we
   pass through v4 unchanged -- which means the override is partial
   per season and could understate the lift. **Mitigation:** report
   per-season coverage; if any season has < 80% R64 coverage,
   flag in the findings note.

3. **SIGMA sensitivity.** Vegas audit used SIGMA=11; tournament
   games may want a flatter sigma. **Mitigation:** sweep sigma
   in {9, 10, 11, 12, 13}; pick min-LL sigma on the R64 subset;
   only run bracket-points eval at the picked sigma to keep the
   experiment cheap.

4. **PR 19 leak-fix invariance.** R64 line is a per-game signal
   attached only to the 32 R64 rows of each tournament's pairwise
   frame. It does NOT enter team-aggregate or season-aggregate
   features. **Mitigation:** verify by running existing leak-fix
   regression test (`tests/test_filter_vegas_to_pre_tournament.py`
   or whatever PR 19's test was named); the override path is
   orthogonal to that fix.

5. **2026 application is out of scope.** Even on PASS, this PR is
   backtest-only. Production wiring is a follow-up with its own
   spec analogous to v9-C production swap. Risks live there:
   need 2026 R64 lines fetched at apply time (not a problem -- they
   exist publicly); need pipeline change to inject override before
   v9-C; documented as a follow-up checklist.

6. **Single-tournament noise.** 22-season aggregate is the right
   gate (per the v9-C / v9-B / BT bracket-points patterns). A PASS
   on aggregate doesn't guarantee a PASS in any specific year. Per
   the per-season variance check's findings, 2024-style "median
   year" outcomes are within v4's variance band -- the experiment
   is testing expected lift, not single-season certainty.

## Test plan summary

- **Unit:** override-correctness, pass-through, mode-symmetry,
  coverage, SIGMA range. ~6-8 tests on synthetic + real fixtures.
- **Smoke:** full 22-season eval produces well-formed JSON with the
  three anchors landing.
- **Anchor:** `mode=v4-only` reproduces 2069 exactly (FP precision).
- **Existing-suite regression:** full `pytest -q` green; no test in
  the leak-fix or v9-C suites breaks.

## Compute estimate

- Build override: 30-60s wall (Vegas join is the heaviest step;
  reuse-from-audit so cache benefits apply).
- 22-season LOSO bracket-points eval per (mode, sigma) cell: ~5-10
  minutes (v9-C training dominates; we have prior runtime numbers
  from `sweep_bt_bracket_points.py`).
- Sigma sweep at the cheap LL gate (no v9-C): seconds per cell.
- Total budget: **2-5 hours wall** for unit tests + override + sigma
  sweep + 2 cells (hard/mean) of the bracket-points eval.

## Forward links (post-merge)

If PASS: write follow-up spec for "re-train v9-C on R64-overridden
frame" before any production swap. If MARGINAL: retain code on
branch, no production change. If FAIL: write retrospective in the
findings note explaining what the null result means for the
calibration-shape engineering target -- a confirmed FAIL on the
data hypothesis is meaningful evidence for the structural-flaw
hypothesis.

## Open design questions for plan-writing time

- Should `eval_r64_line_blend.py` reuse `sweep_bt_bracket_points.py`
  by import or by copy-and-modify? Decide during plan-writing after
  reading the sweeper's structure.
- Is there a simpler harness already in the repo
  (`predict_2026_v9c.py`, `score_chalk_brackets.py`) that gets us
  closer to the target with less new code? Same -- decide at
  plan-writing time.
- Where do per-season per-(mode,sigma) cell numbers go in the JSON
  shape? Mirror the structure of `output/v9_weight_sweep.json` if
  it exists.
