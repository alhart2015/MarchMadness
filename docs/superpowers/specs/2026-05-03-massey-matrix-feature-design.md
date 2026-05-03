# Massey-Matrix MOV Rating as v4 Feature -- Design

**Date:** 2026-05-03
**Branch:** feat/todo-massey-colley
**Scope:** Massey only. Colley is the next work item under TODO #1.

## Predecessors

This experiment lives in the same falsification-gated pattern as the
recent feature/ensemble work:

- BT-as-feature (rejected at gate, PR 13):
  `docs/superpowers/specs/2026-05-02-bt-as-feature-design.md`
  `docs/notes/2026-05-02-bt-as-feature.md`
- Feature-view diversity ensemble (rejected at gate, PR 14):
  `docs/superpowers/specs/2026-05-02-feature-view-ensemble-design.md`
  `docs/notes/2026-05-02-feature-view-ensemble.md`
- BT stage-1 (rejected at gate, PR 12):
  `docs/notes/2026-05-01-bayesian-stage1.md`
- LR ensemble (rejected at gate, PR 11):
  `docs/notes/2026-05-01-ensemble-stage1.md`
- v9-C production swap (current production stage-2):
  `docs/superpowers/specs/2026-05-01-v9c-production-swap-design.md`

## Motivation

The four prior closed experiments converged on the same diagnosis: at
v4's data scale, adding stage-1 *peers* on the same feature matrix
cannot beat v4 because (a) any peer trained on the same features
produces highly correlated errors (LR: r=0.77) and (b) any structurally
distinct peer (BT, single-view XGB) is too weak standalone to earn
non-trivial blend weight (BT standalone LL 0.565 vs v4's 0.437).

The remaining productive entry point is the one TODO #1 names first:
**add new per-team-per-season features to v4's input matrix** rather
than try to ensemble with peer models. That sidesteps the
standalone-strength bottleneck entirely, because v4 itself is the
consumer.

User has prior real-world success with classical linear-algebraic
ranking systems (Massey matrix, Colley matrix) on March Madness
prediction. These ratings are computed from raw Kaggle game-result
data and are *distinct* from v4's existing `massey_*` features (which
are external composite *orderings* from massey-ratings.com -- POM,
SAG, MOR, BPI, RPI -- not our own Massey-matrix solve).

This design tests **plain Massey on margin-of-victory** as a single
new feature column. Colley is a separate work item filed in the
same TODO entry.

## Goals

- Add one new column `massey_mov_rating` to v4's per-team-per-season
  feature matrix, computed by least-squares solve of a Massey-style
  linear system over regular-season-only games.
- Recipe: home-court constant `h` estimated jointly with team
  ratings; MOV capped at +/- 21 (predictive Massey rather than
  retrodictive); per-season independent solve; sum-to-zero
  identifiability constraint.
- Run a cheap two-clause falsification gate before the full 22-season
  LOSO backtest, matching the gate-pattern of PRs 11-14 that saved
  ~3-5 hours of compute by short-circuiting bad ideas.
- If the gate clears, run the standard v4 22-season LOSO backtest and
  decide against the same ladder as the v9-C-vs-v8 swap spec.

## Non-Goals

- Colley matrix. Separate work item, separate spec, separate gate.
- Hyperparameter re-tune. Use v4's existing tuned XGBoost params; a
  feature that doesn't help at v4's hypers is not interesting enough
  to spend ~1 hour re-running Optuna for. (See Q3 in brainstorming
  log: explicitly chose option A over option D.)
- Stage-1 peer / ensemble. We are not producing pairwise probabilities
  from the rating; the rating is consumed only as a column in v4's
  feature matrix. The "rating-diff -> sigmoid as ensemble peer" path
  (TODO #1 entry (b)) is rejected up front because it would face the
  same standalone-strength bottleneck that closed PRs 11-13.
- Stage-2 input. The "feed Massey rating to v9-C as 6th feature" path
  (TODO #1 entry (c)) is rejected up front because BT-as-feature
  already falsified "added feature on top of v4 stage-1 output passed
  to v9-C" at this data scale.
- Production swap. If the feature passes, the column stays in
  `compute_all_features()` and v4 effectively becomes v4-with-massey;
  no separate v5/predict_2026_v5.py mirror is in scope. If a follow-up
  changes the live-bracket pipeline, that's a separate commit.
- Time-decayed weighting (e.g., recent games count more) or Bayesian
  prior toward zero. Plain Massey only -- those are tunable knobs we
  can revisit if and only if plain Massey clears the gate.
- Women's tournament data. Not used by v4; not in scope.

## Background: what existing `massey_*` features are not

v4 already includes columns like `massey_POM`, `massey_SAG`,
`massey_MOR`, `massey_BPI`, `massey_RPI`, `massey_composite` (mean of
those). Those come from `MMasseyOrdinals.csv` (Kaggle-supplied
external composite *rankings*; integer ranks, not numeric ratings).
They aggregate other people's Massey-style and non-Massey-style
systems.

The *Massey matrix* in this spec is something different: a
linear-algebraic least-squares solve we run ourselves, on raw game
scores, producing a continuous rating in points. It can correlate with
the existing composites (which is what clause 1 of the gate checks)
but it is not the same input.

## Solver: math

### Per-game model

For each regular-season game with winner `i`, loser `j`, score-diff
`s_g = WScore - LScore`, location `z_g`:

- `z_g = +1` if `WLoc='H'`, `-1` if `WLoc='A'`, `0` if `WLoc='N'`
- Capped MOV: `y_g = sign(s_g) * min(|s_g|, mov_cap)`, mov_cap=21

Linear model: `y_g = r_i - r_j + h * z_g + eps_g`

### Linear system (normal equations + sum-to-zero constraint)

Let `X` be the `n_games x (n_teams + 1)` design matrix:

- `X[g, i] = +1`, `X[g, j] = -1`, `X[g, n_teams] = z_g`

OLS minimizer satisfies `X^T X * beta = X^T y` for
`beta = [r_1, ..., r_{n_teams}, h]`. The team-ratings block of the
normal-equations matrix has a one-dimensional null space (adding a
constant to all `r_k` is invariant). We resolve it with the
identifiability constraint `sum(r) = 0` via the bordered KKT system:

```
[  X^T X    e ] [ beta   ]   [ X^T y ]
[   e^T     0 ] [ lambda ] = [   0   ]
```

where `e` is the `(n_teams + 1)`-vector with `1`s in team slots and
`0` in the home-constant slot. This is a `(n_teams + 2) x (n_teams + 2)`
dense linear solve.

### Building `X^T X` and `X^T y` directly (no game-level X)

Per season, aggregating across games:

- `(X^T X)[k, k] = games_k`
- `(X^T X)[k, j] = -h2h_count(k, j)` for `k != j`
- `(X^T X)[k, h] = home_games_k - away_games_k` (net home appearances
  for team k; computed by signing each game's `z` from k's perspective
  rather than from W's perspective)
- `(X^T X)[h, h] = non_neutral_games_total`
- `(X^T y)[k] = sum_{games of k} signed_capped_MOV_for_k`
- `(X^T y)[h] = sum_{games} z_g * y_g`

n_teams per season is ~360 D-I teams; matrix is ~362 x 362; solve
runs in milliseconds via `numpy.linalg.solve`.

### Edge cases

- **Disconnected components.** Detect via `numpy.linalg.cond(M) > 1e10`
  and warn. In practice D-I has been fully connected back to the early
  1990s; for v4's training range (Season >= 2003), this should never
  trigger.
- **Zero non-neutral games in a season.** Theoretically degenerate;
  not observed in D-I. If detected, set `h = 0` and solve the
  team-only sub-system.
- **mov_cap <= 0.** Assert at function entry; this is a programming
  error, not a runtime configuration.
- **Pre-2003 seasons.** v4 training already filters to Season >= 2003
  (detailed-results availability). The Massey solver itself works
  fine on compact-results-only data going back to 1985 if ever needed.

## Module shape

### New module: `src/features/massey_matrix.py`

Single public function (with one private solver helper and one private
per-season aggregator):

```python
_PRODUCER_VERSION = "v1"

def compute_massey_mov_ratings(
    reg_season: pd.DataFrame,
    seasons: list[int] | None = None,
    mov_cap: int = 21,
) -> pd.DataFrame:
    """Compute Massey-matrix MOV ratings per (Season, TeamID).

    See spec docs/superpowers/specs/2026-05-03-massey-matrix-feature-design.md.

    Returns DataFrame with columns [Season, TeamID, massey_mov_rating],
    one row per (team, season) where the team played at least one game
    in the season.
    """


def load_massey_mov_ratings(
    reg_season: pd.DataFrame,
    mov_cap: int = 21,
    cache_dir: str | Path = "data/cache",
) -> pd.DataFrame:
    """Cached wrapper. Reads/writes parquet at <cache_dir>/massey_mov_ratings.parquet
    with sidecar metadata at <cache_dir>/massey_mov_ratings.meta.json.

    Cache invalidation: rebuilds when sidecar metadata mismatches the
    current (_PRODUCER_VERSION, mov_cap, n_input_rows, sha_input).
    """
```

### Cache: `data/cache/massey_mov_ratings.parquet`

- One file across all seasons; ~360 teams x ~23 seasons ~ 8k rows.
- Sidecar `data/cache/massey_mov_ratings.meta.json` records
  `{producer_version, mov_cap, n_input_rows, sha_input, written_at}`.
  Mismatch with current state -> rebuild.
- Per CLAUDE.md: cache is reproducible-artifact territory; if the
  producer changes, delete the cache (or change `_PRODUCER_VERSION`,
  same effect).

### Wire-in to v4: `src/enhanced_model.py:compute_all_features`

Two surgical edits totaling ~8 lines:

1. **Top of function** (around line 200, before the per-season loop):

   ```python
   massey_mov_full = load_massey_mov_ratings(reg)
   ```

2. **Inside the per-season loop**, in a new section "2i: Massey-matrix
   MOV rating" (between section 2h "Seed features" and the per-team
   assembly loop):

   ```python
   # -- 2i: Massey-matrix MOV rating ----------------------------------
   season_mov = massey_mov_full[massey_mov_full["Season"] == season]
   massey_mov = dict(zip(season_mov["TeamID"], season_mov["massey_mov_rating"]))
   ```

3. **In the per-team assembly loop** (around line 373):

   ```python
   if tid in massey_mov:
       row_data["massey_mov_rating"] = massey_mov[tid]
   ```

That's the entire v4 wire-in. `get_feature_cols()` in
`feature_matrix_v2.py` already auto-includes any numeric column not
in its exclude set, so no list edits are needed there. Reverting the
experiment is `git revert` of one commit.

### Diagnostic / gate runner: `src/diagnose_massey_mov.py`

Mirrors `src/diagnose_feature_view_ensemble.py`. One CLI entry point:
runs both gate clauses, writes `output/diag_massey_mov.json`, prints
PASS/FAIL summary, exits non-zero on FAIL. See "Falsification gate"
section below for the actual computations.

## Falsification gate

Run before the full backtest. Both clauses must pass.

### Clause 1 -- non-redundancy with existing v4 features

**Computation.** For each season in v4's LOSO range
(2003-2024 excluding 2020), compute Pearson correlation between
`massey_mov_rating` and:

- `adj_em` -- our iterative opponent-adjusted efficiency
- `massey_composite` -- mean rank of the existing
  `MMasseyOrdinals`-derived composite columns

Report per-season correlations; aggregate to (mean, max-abs) per baseline.

**Pass criterion.** For BOTH baselines:
`mean_abs_corr < 0.95` AND `max_abs_corr < 0.97`.

The 0.95/0.97 split allows for one-season outliers without nuking the
experiment. A feature with mean |corr| >= 0.95 against an existing
column is essentially a duplicate; XGBoost cannot extract residual
signal worth the +1 column of complexity.

**Rationale for 0.95.** v4's own `adj_em` is itself an iterative
opponent-adjusted efficiency loop; Massey-matrix MOV is a closed-form
opponent-adjusted MOV solve. They will correlate. The question is by
how much. 0.95 is the threshold below which we expect XGBoost to find
nontrivial residual splits in our experience with the existing v4
feature set; above 0.95, the feature is doing the same job as `adj_em`.

### Clause 2 -- no-harm headroom on a 3-season subset

**Subset.** Holdouts {2019, 2022, 2024}. Recent, diverse, none affected
by the 2020 COVID cancellation; one recent chalk year (2019), one
chaos year (2022), one mixed year (2024). Each requires one v4-style
fit with the new column included (~3-5 min/season).

**Computation.** For each of the 3 holdouts:

1. Train v4 (XGBoost with v4's existing best_params) on all other
   v4 seasons except the holdout, with the feature matrix augmented
   by `massey_mov_rating`.
2. Predict the holdout's tournament games; compute LL and accuracy on
   tournament-only games.
3. Compare against v4's recorded per-season LL/acc on the same
   holdouts. Source of baseline: re-run v4-without-massey on the same
   3 seasons in the same gate runner, to ensure identical training-set
   composition. (No comparing against a possibly-stale cached number.)

**Pass criterion.**
`mean(LL_with_massey_3seasons) - mean(LL_v4_baseline_3seasons) <= +0.001`

i.e., at worst, the new feature is neutral within noise on the cheap
subset. If it's already actively hurting at -0.001 on 3 seasons, the
22-season LOSO is not going to recover that.

**Rationale.** The BT-as-feature gate used `+0.001` as the headroom
threshold on a single-cell pre-sweep check. That gate caught an
actively-harmful feature (-0.0015 vs threshold +0.001) in ~5 minutes
vs ~45-75 minutes for the doomed full sweep. We use the same threshold
here for consistency.

### Aggregate gate decision

Both clauses must pass to proceed to the full backtest. Total wall-clock
estimate: ~5-15 min (clause 1 is seconds; clause 2 is the bulk of it).

### Failure path

Either clause fails:

- Write `output/diag_massey_mov.json` with all measured numbers.
- Write `docs/notes/2026-05-03-massey-mov.md` with:
  - which clause failed and the measured-vs-threshold delta
  - which existing feature subsumes Massey, if clause 1 (e.g., `adj_em`
    at r=0.97 -- our own efficiency loop already extracts the
    Massey-style signal)
  - if clause 2: the per-season LL deltas
  - lessons for the Colley work item that follows
- Branch retained as experiment record (commit history is the
  documentation).
- Update TODO.md "Tried and rejected" section.
- Stop. Do not run the 22-season backtest.

## Full LOSO backtest (only if gate passes)

### Run

`python src/enhanced_model_v3.py` with the new feature column included
by default in `compute_all_features()`. 22-season LOSO backtest
(Seasons 2003-2024 excluding 2020). v4's existing tuned hyperparameters
(no Optuna re-tune; see Q3 / Non-Goals).

### Outputs

- `output/v4_with_massey_loso.json` -- per-season LL, accuracy, brkt
  pts, deltas vs v4 baseline.
- `docs/notes/2026-05-03-massey-mov-backtest.md` -- summary table,
  per-season W/L/T, F4/E8 chalk accuracy, decision against the ladder.

### Decision ladder

Sign convention: lower LL is better; positive `brkt_delta` means
v4-with-massey scores more bracket points than v4 over 22 seasons.

The ladder is evaluated **in order**, so mixed signals resolve cleanly
(e.g., a result with LL win + brkt loss hits Reject, not Clear --
shipping a feature that loses bracket points is the failure mode we
care about preventing):

1. **Reject** if `LL_delta >= +0.001` OR `brkt_delta <= +10`.
   One bad dimension kills it. Action: revert wire-in commit, write
   findings note, retain branch as experiment record, update TODO.md
   "Tried and rejected".
2. Otherwise **Clear** if `LL_delta <= -0.005` OR `brkt_delta >= +25`.
   Action: feature stays in `compute_all_features()`. PR merge. v4
   effectively becomes v4-with-massey.
3. Otherwise **Marginal** (i.e., `LL_delta in (-0.005, +0.001)` AND
   `brkt_delta in (+10, +25)`). Action: document candidate; keep code
   on branch but do NOT merge. Findings note records the per-season
   detail and what would close the gap.

These bands match the v9-C-vs-v8 swap spec for consistency with how
prior PRs were judged. The ordering rule (Reject before Clear) is the
new clarification specific to this spec -- prior swap specs evaluated
a single metric (brkt pts) so the ambiguity didn't arise.

## Tests

### New: `tests/test_features/test_massey_matrix.py`

1. **Synthetic round-robin -- known answer.** 4 teams, each plays each
   other twice (once home, once away). MOVs designed so the
   closed-form solution is `r = [+10, +5, -5, -10]` and `h = +3`.
   Solver must recover both within 1e-6 (relax to 1e-4 if numerical
   drift surfaces).
2. **Sum-to-zero constraint.** `r.sum()` < 1e-8 for any synthetic
   input.
3. **MOV cap actually clips.** Schedule where one team has a single
   100-point blowout vs ten close losses: rating with `mov_cap=21` is
   bounded by what a 21-point cap would predict, not 100. (Concretely:
   compute the rating with `mov_cap=21` and `mov_cap=100`; assert the
   capped-21 rating is materially smaller in absolute value.)
4. **Home-court constant sign.** Schedule where the home team always
   wins by 5 (and teams are otherwise balanced) produces `h ~= 5`
   and team ratings near zero.
5. **Cache round-trip.** Write parquet, read it back, columns and rows
   match exactly (including dtype).
6. **Cache invalidation.** Bump `_PRODUCER_VERSION`, confirm rebuild
   triggers without manual cache deletion.

### Existing tests

`tests/test_integration.py` may need a tolerance widening if the new
feature changes v4's predicted probabilities by a hair on the small
fixture games it checks. Per CLAUDE.md, this is acceptable only with
explicit user confirmation. If we hit it, we flag it in the findings
note rather than silently widening tolerances.

## File inventory

**New (committed):**

- `src/features/massey_matrix.py` -- solver + cache loader
- `src/diagnose_massey_mov.py` -- gate runner
- `tests/test_features/test_massey_matrix.py` -- unit tests
- `docs/superpowers/specs/2026-05-03-massey-matrix-feature-design.md` -- this spec
- `docs/superpowers/plans/2026-05-03-massey-matrix-feature.md` -- implementation plan (next step, separate file)

**Generated artifacts (committed for reproducibility audit):**

- `output/diag_massey_mov.json` -- gate output (PASS or FAIL)
- `docs/notes/2026-05-03-massey-mov.md` -- findings note
- If gate passes:
  - `output/v4_with_massey_loso.json` -- backtest detail
  - `docs/notes/2026-05-03-massey-mov-backtest.md` -- backtest findings

**Generated artifacts (NOT committed, regenerable):**

- `data/cache/massey_mov_ratings.parquet` (+ sidecar `.meta.json`)

**Edited:**

- `src/enhanced_model.py` -- ~8 LOC added in `compute_all_features()`,
  one new section "2i" plus three small additions (top-of-function
  cache load, per-season dict build, per-team-row assembly).

## Risks

1. **Massey duplicates `adj_em`.** Most likely failure mode. Our own
   iterative opponent-adjusted efficiency loop already extracts the
   "score margin adjusted for opponent strength" signal. If r > 0.95,
   clause 1 fails and we know why. Outcome: kill the experiment cheap;
   the lesson generalizes to Colley-on-margins (Colley's distinctness
   is win/loss-only, so this risk is materially smaller for Colley).
2. **Massey duplicates the existing `massey_composite` external rank.**
   The external composite is a *rank*, not a rating, but ranks of
   ratings often collapse to similar information. Same gate clause
   catches it.
3. **Gate clause 2 catches a fluke.** A 3-season subset is small;
   one anomalous holdout could push us to abort a real winner. Mitigation:
   the threshold is `+0.001` (not `0.0`), giving a one-decision-place
   noise band. If the experiment is otherwise compelling but fails
   clause 2 by a hair, the findings note records the per-season
   numbers for manual review.
4. **Cache staleness from Kaggle data refresh.** If a new Kaggle data
   pull updates `MRegularSeasonCompactResults.csv`, the sidecar's
   `n_input_rows` and `sha_input` will mismatch and force a rebuild.
   This is the intended behavior; no manual cache nuke required.
5. **mov_cap=21 is wrong.** Massey's own published predictive recipe
   uses caps in the 21-28 range. We pick 21; if the gate fails, a
   followup with `mov_cap in {28, 35, no-cap}` is cheap to run, but
   that would be a separate spec/branch. We do NOT sweep mov_cap in
   this experiment -- the recipe is fixed up front per Q1's "default
   to most defensible" answer.
6. **Massey overweights early-season games equally with late-season
   games.** Possibly. Time-decay is a known knob for predictive Massey.
   Out of scope; if plain Massey passes the gate, time-decay is a
   followup; if plain Massey fails the gate, time-decay is unlikely
   to rescue it (the redundancy issue is structural, not weighting).

## Out-of-scope followups (only if this passes)

These are NOT part of this spec. Listed only for context:

- Colley matrix (TODO #1, separate work item).
- mov_cap sweep ({21, 28, 35, no-cap}).
- Time-decayed Massey (recent games weighted heavier).
- Bayesian-prior Massey (regularize toward zero rating).
- Wiring `massey_mov_rating` into the live-bracket prediction
  pipeline (`generate_bracket_real.py`), which today is pure-v4-MC
  and does not currently consume v4 directly through
  `compute_all_features` at apply-time. (See v9-C production swap
  spec for the parallel of this gap.)
