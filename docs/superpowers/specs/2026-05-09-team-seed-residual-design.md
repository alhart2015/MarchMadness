# Team-Program Tournament History Features -- Design

**Date:** 2026-05-09
**Branch:** `feat/team-seed-residual` (to be created via worktree)
**Predecessors:**
- TODO retire-Kaggle-framing (PR 33): production objective is now the
  22-season bracket-points backtest (clean baseline 2069), with
  log-loss as a secondary signal for the user's secondary pool.
- v4 calibration temperature scaling (MARGINAL): closes
  calibration-shape lane, leaves engine-improvement lanes #3/#4/#5 plus
  this new program-history lane. `docs/notes/2026-05-08-v4-calibration-temperature-scaling.md`.
- 538 audit: 1 weak spot vs 538 (chalk picks). `docs/notes/2026-05-04-v4-gap-audit-fte.md`.
- Vegas audit (clean baseline): 6 weak spots vs Vegas, no upset-detection
  edge. `docs/notes/2026-05-04-v4-gap-audit-vegas.md`.

## Motivation

v4 has coach-keyed history features (`coach_career_{games, wins, winpct,
f4_apps, champs, seasons}`) but no team-program-keyed history features.
When a team hires a new coach, the team starts at 0 on every coach
feature; the program's tournament DNA is invisible to v4.

The UConn 2023 case: Hurley joined UConn in 2018 with `coach_career_f4_apps`
of 0 (Rhode Island and BU produced no F4s). UConn-the-program had four
prior championships (1999, 2004, 2011, 2014) but those didn't transfer
onto Hurley's coach card. v4 had no team-keyed slot to encode "this is
a program that converts elite regular seasons into deep tournament runs."

A 9-champion empirical scan over 2015-2024 confirmed the signal exists
but is not dominant. 6/9 champions had clear strong prior-10-year
program track records (Duke 2015, UNC 2017, Villanova 2018, Kansas 2022,
UConn 2023, UConn 2024); 1/9 medium (Villanova 2016); 2/9 had **zero
F4 appearances in the prior 10 years** (Virginia 2019, Baylor 2021).
The 2 weak counter-examples were emergence stories: Bennett at Virginia
and Drew at Baylor had been quietly upgrading their programs (E8 in
2016 and 2017 respectively) for years.

The single program-DNA signal cannot capture both continuity (Duke /
UNC / Kansas / UConn) and emergence (Bennett / Drew). The design ships
**two features** -- one optimized for each pattern -- and lets XGB
learn when to weight which.

**The hypothesis under test:** team-program tournament history,
encoded as a seed-residual signal that is keyed on team rather than
coach, adds bracket-points headroom on top of v4's 67-feature stack.
If the 22-season aggregate clears the +20 PASS bar without single-
season fragility, the feature ships. If MARGINAL (+10..+20), it
becomes a candidate for one HL or k sensitivity sweep before swap-in.
If FAIL (< +10), the team-program lane closes and we update the
program-DNA prior in the TODO.

## Goals

- Add **two new XGB features** to v4's per-team-per-season feature
  matrix:
  - `team_seed_residual_mean_10yr` (continuity): unweighted mean of
    seed-residuals over the prior 10 tournament appearances within
    the prior 10 calendar years, with Bayesian shrinkage k=3 toward 0.
  - `team_seed_residual_ewma_hl2` (momentum): exponentially-weighted
    mean of the same residuals with half-life 2 years, with the same
    shrinkage k=3.
- Ship a **Phase 1 sanity-check** before the LOSO run: per-seed baseline
  table, 9-champion residual values for hand-verification, correlation
  matrix vs incumbent features, distribution percentiles, top-10/bottom-10
  team-seasons by each feature.
- Run **Phase 2 full 22-season LOSO** with pre-registered verdict bands
  (PASS / MARGINAL / FAIL) and a per-season fragility check.
- Maintain leak safety: per-seed baseline and per-team residuals computed
  using only `Season < S` data (mirrors PR 19/20 leak-fix audit pattern).

## Non-Goals

- **Hyperparameter sweeps in v1.** Single cell only: window=10 years,
  HL=2 years, k=3 pseudo-counts. If MARGINAL, *then* sweep one knob
  (likely HL); do not pre-emptively burn compute.
- **Bracket-structure correction in baseline.** Empirical per-seed
  average rounds_won is the v1 baseline. A model-based baseline that
  accounts for "a 9-seed has to face a 1-seed in R32" is a v2 upgrade
  if v1 proves out.
- **Exposure feature.** A separate `team_tourney_apps_10yr` feature was
  considered (Question 4 option B) and deferred. Bayesian shrinkage
  with k=3 handles thin-history teams within the residual feature; if
  Phase 1 diagnostic surfaces residual issues for thin-history teams,
  exposure is a one-line add.
- **Program-DNA features keyed on something other than `TeamID`.**
  Conference-keyed program history (e.g., "Big East tournament
  over-performance over prior 10 years") is conceptually adjacent but
  out of scope; the existing `conf_strength` feature partially covers
  it.
- **Production swap.** Like Massey/Colley specs, the production swap
  to a new feature stack is a separate follow-up PR if v1 passes.

## Feature definitions

### Per-seed empirical baseline

For a target season S and seed s in {1, ..., 16}:

`E_baseline[S][s] = mean(rounds_won)` across all (Season, TeamID) rows
in `MNCAATourneyCompactResults` with `Season < S` and seed = s.

Where `rounds_won` is the number of tournament games a team won in
that tournament (0 for first-round losers, 6 for champions).
First-Four games (DayNum 134-135) count as round 0 wins, mirroring
`src/features/coach.py` convention.

A defensive assertion in `compute_per_seed_baseline` raises if any
input row violates `Season < max_season` (mirrors PR 19/20 leak-fix
pattern). For seeds with 0 historical observations (shouldn't happen
for seeds 1-16 with 1985+ data), fall back to overall mean
`rounds_won` across all observed seeds.

### Per-team-season residual

For each (Season S, TeamID T) in S's tournament field, look up the
team's prior tournament appearances:

`prior_appearances[S][T] = [(prior_season, prior_seed, prior_rounds_won)
                             for prior_season in (S-10, ..., S-1)
                             where T appeared in prior_season's tournament]`

Compute residuals against the leak-free baseline:

`residuals[S][T] = [(prior_season, prior_rounds_won
                       - E_baseline[S][prior_seed])
                      for (prior_season, prior_seed, prior_rounds_won)
                      in prior_appearances[S][T]]`

Note that the baseline `E_baseline[S][prior_seed]` uses data from
**all seasons before S**, not just the prior_season specifically.
This makes the baseline more stable across thin-data eras and makes
it explicitly leak-free.

### Continuity feature: shrunk mean

```python
def shrunk_mean(residuals, k=3):
    n = len(residuals)
    if n == 0:
        return 0.0
    return (sum(residuals) + k * 0.0) / (n + k)

team_seed_residual_mean_10yr = shrunk_mean(
    [r for (_, r) in residuals[S][T]], k=3
)
```

For a team with 0 prior appearances in window: returns 0 (no evidence).
For a team with 1 prior appearance with residual +5 (e.g., a 7-seed
champion like UConn 2014, where champion `rounds_won = 6` and the
7-seed baseline is approximately 1.0): returns
`(5 + 0) / (1 + 3) = 1.25`. For a team with 10 appearances summing
to +6 in residuals: returns `(6 + 0) / (10 + 3) ≈ 0.46`.

Note on `rounds_won` convention: a champion has `rounds_won = 6` (six
games won), an E8 loser has `rounds_won = 3`, an R64 loser has
`rounds_won = 0`. The 9-champion data table reviewed in the design
session used a +1 display offset (champion shown as `r7`); the spec
and implementation use the conventional games-won definition.

### Momentum feature: shrunk EWMA

For each prior appearance, compute years-ago `a = S - prior_season`
(in {1, ..., 10}). The EWMA weight at half-life HL=2 is
`w(a) = 0.5 ** ((a - 1) / 2)`. Note: `w(1) = 1`, `w(3) = 0.5`,
`w(5) = 0.25`.

```python
def shrunk_ewma(residuals_with_age, half_life=2, k=3):
    n = len(residuals_with_age)
    if n == 0:
        return 0.0
    weights = [0.5 ** ((a - 1) / half_life)
               for (a, _) in residuals_with_age]
    weight_sum = sum(weights)
    weighted_mean = sum(w * r for (w, (_, r))
                        in zip(weights, residuals_with_age)) / weight_sum
    # k pseudo-obs at value 0, counted as raw observations (not weighted).
    # Decouples "which residuals matter" (EWMA weights) from "how confident
    # we are in our estimate" (raw n).
    return (n * weighted_mean + k * 0.0) / (n + k)

team_seed_residual_ewma_hl2 = shrunk_ewma(
    [(S - ps, r) for (ps, r) in residuals[S][T]], half_life=2, k=3
)
```

Design choice on the EWMA shrinkage: shrinkage strength is set by raw
count of prior appearances (`n`), not by the sum of EWMA weights.
Reason: the EWMA's effective sample size is bounded (~3.4 for
HL=2 even with infinite history), so weight-based shrinkage with k=3
would over-shrink even well-established programs. Raw-`n` shrinkage
makes the formula match `shrunk_mean`'s behavior for thin-history
teams (1-app team gets 25% weight on data, 75% on prior 0; 10-app
team gets 77% on data, 23% on prior).

For a team with 0 prior appearances: returns 0.

UConn 2023 walkthrough (illustrative; actual baseline values come from
the implemented `compute_per_seed_baseline`). Prior appearances at
years-ago = (9, 7, 2, 1) with approximate residuals (+5, +0.3, -1, -1.5)
where the +5 is the 2014 7-seed championship (`rounds_won = 6` minus
~1.0 baseline for 7-seeds) and the small negatives are 2021/2022 R64
exits as a 7- and 5-seed.

- **Continuity (mean):** sum=+2.8, n=4, k=3 → `(2.8 + 0) / 7 ≈ +0.40`.
- **Momentum (EWMA, HL=2):** weights = (0.063, 0.125, 0.71, 1.0),
  weighted mean ≈ -0.98 (the recent two -1.0 / -1.5 dominate the
  far-back +5), shrunk with n=4, k=3 → `(4 × -0.98 + 0) / 7 ≈ -0.56`.

The two features disagree by ~1 unit on this team-season. Continuity
reads weakly positive (the 2014 championship pulls the long average
up); momentum reads weakly negative (recent R64 exits dominate the
exponential weighting). **This disagreement is the additional signal
XGB gets to model** -- a single-feature design could not represent
"strong long-term program with weak recent form" vs "strong long-term
program with strong recent form" as different inputs.

## Architecture

```
src/features/team_history.py          (new module, ~150 LOC)
  - day_to_round                       (or import from coach.py)
  - rounds_won_for_team_season         (helper)
  - compute_per_seed_baseline          (Season-1 leak-safe)
  - compute_team_residuals_in_window   (returns list of (year_ago, residual))
  - shrunk_mean
  - shrunk_ewma
  - compute_team_history_features      (DataFrame returner)

src/diagnose_team_seed_residual.py    (new Phase 1 driver, ~80 LOC)
  - prints/dumps the 5 sanity-check artifacts to
    output/team_seed_residual_diagnostic.{json,log}

tests/test_team_history.py            (new, ~10 unit tests)

src/enhanced_model_v3.py              (Phase 2 modification: 1-2 lines)
  - call compute_team_history_features in the per-team-feature build
  - join the two new columns onto the team feature frame
```

The Phase 1 / Phase 2 split: Phase 1 PR ships the module + diagnostic
+ unit tests but does NOT wire into v4's feature build. The user
inspects the diagnostic artifacts before approving Phase 2. Phase 2
PR adds the wire-in + runs the full LOSO + writes findings.

If Phase 1 surfaces a computational bug (e.g., off-by-one in the
window definition, sign error in residual, baseline leak), the fix
is contained to one PR with no LOSO compute wasted.

## Data flow

```
MNCAATourneyCompactResults.csv ─┐
MNCAATourneySeeds.csv ──────────┼─→ compute_team_history_features()
tournament_field (per-season)  ─┘         │
                                          ↓
                          DataFrame[(Season, TeamID), 2 cols]
                                          │
                          ┌───────────────┴───────────────┐
                          ↓ (Phase 1)                     ↓ (Phase 2)
            diagnose_team_seed_residual.py    enhanced_model_v3 feature build
                          │                              │
            sanity-check artifacts (JSON+log)   joined into v4's per-team frame
                                                          │
                                                  full 22-season LOSO
                                                          │
                                                pre-registered verdict bands
```

## Error handling

- **Zero prior appearances in window:** `shrunk_mean` and `shrunk_ewma`
  of empty input return 0.0. By design -- "no program-history evidence"
  maps to neutral.
- **Per-seed baseline for rare/missing seeds:** fall back to overall
  mean `rounds_won` across all observed seeds. Unlikely to trigger for
  seeds 1-16 with 1985+ data; defensive only.
- **First Four games (DayNum 134-135):** count toward apps and
  contribute round 0 to rounds-won, matching `coach.py` convention.
- **Leak prevention:** `max_season = Season - 1` is passed explicitly
  to `compute_per_seed_baseline`. A defensive `assert` in that function
  raises if any input row has `Season > max_season`. Mirrors the
  PR 19/20 leak-fix audit pattern.
- **Coach-pre-2003 history:** the dataset starts at 1985. For target
  seasons S in {2003, ..., 2025}, the prior-10-year window covers
  Season S-10 onward, all within the dataset. No edge-of-dataset
  handling needed for the LOSO seasons.

## Phase 1 sanity-check artifacts

Written to `output/team_seed_residual_diagnostic.json` (machine-
readable) and `output/team_seed_residual_diagnostic.log` (human-
readable):

1. **Per-seed baseline table.** 16 rows of `[seed, n_observations,
   E_baseline]`. Sanity check: 1-seeds should be ~3.0-3.3, 2-seeds
   ~2.4-2.7, 8/9-seeds ~1.0-1.3, 16-seeds ~0.0-0.05 (16-over-1 upsets
   are exceedingly rare).
2. **9-champion residual values.** For each of the 9 champions
   2015-2024, show: prior_window apps, residuals per appearance with
   year_ago + seed + rounds_won, the computed `mean_10yr` and
   `ewma_hl2` values, and a hand-computable cross-check.
3. **Correlation matrix.** Pearson correlation of both new features
   vs `[adj_em, kp_TALENT, kp_BARTHAG, coach_career_f4_apps,
   coach_career_winpct, coach_career_seasons, season_win_pct,
   conf_strength]`. Flag anything > 0.85 as a redundancy concern (not
   a hard gate; a correlation of 0.7-0.85 is acceptable if the feature
   adds non-overlapping signal in some regions).
4. **Distribution percentiles.** 5/25/50/75/95th percentile of each
   feature across all team-seasons. Sanity check: median should be
   close to 0 (since residuals net to 0 by construction); the
   distribution should be slightly right-skewed (a few elite programs
   have large positive residuals).
5. **Top-10 / bottom-10.** Highest and lowest 10 (Season, TeamID,
   feature_value) pairs for each feature. Face-validity check: the
   top-10 should include obvious historical-powerhouse seasons; the
   bottom-10 should include programs with notable seed-vs-result
   collapses.

## Phase 2: pre-registered verdict bands

Run the full 22-season LOSO with the two new features wired into v4's
feature build. Compare 22-season bracket points vs the canonical clean
v4 + v8 baseline (2069 brkt pts).

- **PASS:** delta >= +20 brkt pts on 22-season aggregate, AND
  `aggregate_delta - max_single_season_delta >= +5` (fragility check).
  The fragility check rules out v9-B-style "+18 driven by +12 from
  one season" wins.
- **MARGINAL:** delta in [+10, +20) brkt pts, OR PASS-magnitude but
  failing the fragility check. Triggers one HL or k sensitivity sweep
  before swap-in decision.
- **FAIL:** delta < +10 brkt pts. Team-program lane closes; update
  the TODO program-DNA prior with the empirical evidence.

Also reported (informative, not gating):
- 22-season LL delta (relevant for the user's secondary log-loss pool).
- Per-season W/L/T spread on bracket points.
- The season(s) driving the largest single-season swings.
- SHAP feature importance for the two new features in the trained XGB.
- Anchor invariance check: a build with both features set to 0
  reproduces the canonical clean v4 LL/brkt-pts baseline (sanity check
  on the wire-in code).

## Test plan

**Unit tests (`tests/test_team_history.py`, ~10 tests):**

1. `compute_per_seed_baseline` on a hand-built 3-tournament toy:
   verifies seed-1 expected_rounds_won matches mean of the 3
   tournament-1-seed observations.
2. `compute_per_seed_baseline` raises when input violates `max_season`
   (leak-prevention guard).
3. `compute_per_seed_baseline` falls back to overall mean for a seed
   with 0 historical observations.
4. `shrunk_mean(empty, k=3)` returns 0.
5. `shrunk_mean([6.0], k=3)` returns 1.5.
6. `shrunk_mean([1.0]*100, k=3)` returns ~0.97 (n=∞ → raw_mean).
7. `shrunk_ewma(empty, hl=2, k=3)` returns 0.
8. `shrunk_ewma([(1, 2.0)], hl=2, k=3)` weights single observation
   correctly.
9. `compute_team_residuals_in_window` window edge: year-10 in,
   year-11 out (verified via two-tournament-history toy).
10. `compute_team_history_features` integration: hand-computed UConn
    2024 spot check. UConn's 5 prior appearances in 2024's window are
    2014 (7-seed, won 6 games as champion), 2016 (9-seed, won 1 game,
    lost in R32), 2021 (7-seed, won 0 games, lost in R64), 2022
    (5-seed, won 0 games, lost in R64), 2023 (4-seed, won 6 games as
    champion). From the empirical baseline (computed against Season
    < 2024 tournaments), the test computes residuals per appearance,
    aggregates with `shrunk_mean(k=3)` and `shrunk_ewma(hl=2, k=3)`,
    and asserts the result matches `compute_team_history_features`
    output for `(2024, UConn)` within 1e-6.

**Smoke test:**

11. `compute_team_history_features` returns a DataFrame with 22 *
    ~64 rows (one row per (Season, TeamID) in the 2003-2024 LOSO
    fields), 2 numeric columns, no NaNs.

## Risks

- **Redundancy with coach features.** `coach_career_f4_apps` and
  `coach_career_seasons` already encode some "this team has tournament
  history" signal -- an old coach who has stayed at a strong program
  has accumulated tournament appearances and F4s. Phase 1 correlation
  check is the cheap detector; mitigation if it fires is to limit the
  scope to programs with high coach turnover (more design work,
  defer).
- **Variance dominance.** The 9-champion data showed UConn 2014 and
  UConn 2023 as ~+5-residual events (champion at 7-seed and 4-seed
  respectively, against ~1.0 and ~1.7 baselines) that drive UConn
  2023/2024 feature values. If Phase 2 returns +12 brkt pts driven by
  2023 and 2024 alone, that's not a model improvement -- it's the
  in-sample cases that motivated the feature. The per-season fragility
  check is the explicit guard.
- **Emergence-team penalty.** A naive program-DNA feature would have
  penalized Virginia 2019 and Baylor 2021 (zero F4s in the prior 10
  years). The momentum feature is the design response, but if HL=2 is
  too aggressive (over-fits recent variance) or too gentle (still
  penalizes emergence), MARGINAL Phase 2 triggers an HL sweep.
- **Bayesian shrinkage masking signal.** k=3 shrinks a one-shot
  championship residual (e.g., +5 for a 7-seed champion) to +1.25
  (vs raw +5.0). The signal is preserved but weaker. If MARGINAL
  Phase 2, k=1 vs k=3 vs k=5 is the second candidate sensitivity sweep.

## Open questions to defer

- Should the residual feature use signed `rounds_won - baseline` (the
  v1 design) or `(rounds_won + 1) / (baseline + 1)` (a ratio)? Signed
  ties to the bracket-points objective more directly (each round
  doubles the points); ratio handles low-baseline seeds (16-seeds at
  baseline ~0) more gracefully but isn't motivated by the scoring.
  Defer to v2 if v1 returns FAIL with low-seed-team noise as the
  diagnosed issue.
- Should we add `team_seed_residual_consistency` (std-of-residuals in
  window) as a third feature? Captures "this program is a steady
  performer" vs "this program is volatile." Leave for v2; would
  conflict with the 2-feature design intent for v1.

## File-of-record locations

- This spec: `docs/superpowers/specs/2026-05-09-team-seed-residual-design.md`
- Phase 1 implementation: `src/features/team_history.py`,
  `src/diagnose_team_seed_residual.py`,
  `tests/test_team_history.py`
- Phase 1 artifacts: `output/team_seed_residual_diagnostic.{json,log}`
- Phase 2 implementation: 1-2 line modification to
  `src/enhanced_model_v3.py` (or wherever `build_all_team_features`
  joins coach features)
- Phase 2 findings: `docs/notes/2026-05-09-team-seed-residual.md`
