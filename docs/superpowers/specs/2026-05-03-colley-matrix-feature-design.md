# Colley-Matrix Rating as v4 Feature -- Design

**Date:** 2026-05-03
**Branch:** feat/todo-massey-colley (continued -- Massey rejected, Colley is the sibling experiment)
**Spec sibling:** docs/superpowers/specs/2026-05-03-massey-matrix-feature-design.md

## Predecessors and lesson carried forward

The Massey-MOV experiment (uniform + decay variants) on this branch
was REJECTED at clause 1 (uniform: mean |corr| vs adj_em = 0.957) and
at clause 2 (hl=14d: +0.0057 LL on subset, threshold +0.001).
Findings: docs/notes/2026-05-03-massey-mov.md.

Lesson carried forward: clause 1 (low correlation with existing v4
features) is necessary but not sufficient. v4's feature stack covers
more angles than the spec's original Risks #1 anticipated -- in
particular, late-season margin signal is captured via `late_season`,
`trajectory`, and `vegas_trend`. So this Colley spec broadens clause
1 to a 3rd baseline (`season_win_pct`) -- the W/L-based v4 feature
most likely to duplicate Colley's W/L-only signal.

## Motivation

Colley solves `C x = b` on win/loss only:

- `C_{ii} = 2 + T_i` (T_i = total games played by team i)
- `C_{ij} = -n_{ij}` (negative head-to-head count between i and j)
- `b_i = 1 + (W_i - L_i) / 2`

The matrix `C = (2 I + diag(T) - A)` where `A` is the head-to-head
adjacency matrix. The +2 in the diagonal is a Bayesian prior of 2
fictitious "split" games per team that ensures `C` is positive
definite, so the solve is always well-posed (no rank-deficiency
escape hatch needed, unlike Massey's all-neutral case). Sum-to-(n/2)
identifiability falls out of the formulation; no constraint row
required.

Critically, **Colley discards margin entirely**. This is the structural
distinction that motivates trying it after Massey-MOV failed: where
Massey duplicates `adj_em` (both opponent-adjusted on margin),
Colley produces an opponent-adjusted W/L rating that is structurally
different from `adj_em`. The relevant duplication risk shifts from
`adj_em` to W/L-based features.

User has prior real-world success with classical linear-algebraic
ranking systems (Massey + Colley) on March Madness; Colley remains
worth trying despite Massey's failure because of this structural
distinction.

## Goals

- Add one new column `colley_rating` to v4's per-team-per-season
  feature matrix.
- Run a 3-baseline cheap clause 1 (vs `adj_em`, `massey_composite`,
  `season_win_pct`), then clause 2 (LL headroom on 3-season subset).
- If both clauses pass, run the full 22-season LOSO backtest with
  the standard Reject -> Clear -> Marginal ladder.

## Non-Goals

- Time-decay weighting on Colley. Massey-decay was already rejected;
  the Colley analog faces the same v4-feature-stack-already-covers-it
  failure mode and is not in scope. If plain Colley clears clause 1
  but fails clause 2, decay would be a separate followup.
- Stage-1 / ensemble use. Same reasoning as the Massey spec: closed
  prior PRs.
- Bayesian-prior tuning (the +2 prior is the standard published
  recipe; not a knob we tune in this experiment).
- Production swap / live-bracket wiring. Same scope as the Massey
  spec.

## Solver: math

### Linear system

For each season's regular-season games:

- `T_i` = total games involving team i
- `W_i` = wins, `L_i` = losses (so `T_i = W_i + L_i`)
- `n_{ij}` = head-to-head count between teams i and j

Build the `n x n` matrix `C`:

- `C_{ii} = 2 + T_i`
- `C_{ij} = -n_{ij}` for `i != j`

Build the `n` vector `b`:

- `b_i = 1 + (W_i - L_i) / 2`

Solve `C x = b` via `numpy.linalg.solve(C, b)`. Output `x_i` is
team i's Colley rating.

### Properties

- `C` is symmetric positive-definite (always invertible due to the
  +2 prior), so no constraint or pseudo-inverse needed.
- Sum-to-(n/2) by construction: sum of `b` = n (since the +1 per
  team contributes n; the (W-L)/2 terms sum to 0 in any league
  where total wins = total losses), and `1^T C 1 = 2n` (each
  diagonal `2 + T_i` summed gives `2n + sum(T_i)`; subtracting
  `sum(T_i)` for the off-diagonal -h2h sums leaves `2n`), so
  `1^T x = n/2`.
- `n_teams` per season ~360 for D-I, system is ~360x360 dense,
  solve in milliseconds.

### Edge cases

- **Disconnected components.** Theoretically possible if a season
  had two cliques of teams that never play each other. The +2 prior
  makes `C` invertible regardless, so the solver always returns a
  valid rating (though disconnected components would just float
  toward 0.5). Detect via `cond(C) > 1e10` and warn -- as with
  Massey, this should never trigger in D-I.
- **No edge cases require special handling.** Colley is materially
  cleaner than Massey because of the prior.

## Module shape

### New module: `src/features/colley_matrix.py`

Mirrors `src/features/massey_matrix.py` with one less knob (no
`mov_cap`, no venue parameter):

```python
_PRODUCER_VERSION = "v1"

def _solve_one_season(games_df: pd.DataFrame) -> dict[int, float]:
    """Solve Colley (C - A) x = b for one season. Returns {TeamID: rating}."""

def compute_colley_ratings(
    reg_season: pd.DataFrame,
    seasons: list[int] | None = None,
) -> pd.DataFrame:
    """One row per (Season, TeamID). Column: colley_rating."""

def _hash_input(reg_season) -> str: ...

def load_colley_ratings(
    reg_season: pd.DataFrame,
    cache_dir: str | Path = "data/cache",
) -> pd.DataFrame:
    """Cached wrapper. Reads/writes <cache_dir>/colley_ratings.parquet
    with sidecar metadata. Cache invalidates on (_PRODUCER_VERSION,
    n_input_rows, sha_input) mismatch."""
```

### Cache

`data/cache/colley_ratings.parquet` + sidecar `.meta.json`. Same
pattern as Massey. ~8k rows; gitignored.

### Wire-in

3 small additions to `src/enhanced_model.py:compute_all_features` --
identical pattern to the (since-reverted) Massey wire-in:

1. Top: `from src.features.colley_matrix import load_colley_ratings;
   colley_full = load_colley_ratings(reg)`
2. Per-season block "2j" (after Massey ordinals, before the assembly
   loop): build `colley_map = {tid: rating}` for the season.
3. Per-team row assembly: `if tid in colley_map: row_data["colley_rating"] = colley_map[tid]`

### Diagnostic runner

`src/diagnose_colley.py` mirrors `src/diagnose_massey_mov.py`:

- `clause1_correlations(feature_matrix)` -- 3 baselines now
  (`adj_em`, `massey_composite`, `season_win_pct`).
- `clause2_headroom()` -- reuses `leave_one_season_out_cv_weighted`
  with `allowed_holdouts=[2019, 2022, 2024]`, toggles
  `colley_rating` in `feature_cols`.
- `main()` CLI ties them together; writes `output/diag_colley.json`;
  exit code encodes pass/fail.

## Falsification gate

### Clause 1 -- non-redundancy

Pass: for ALL THREE of (`adj_em`, `massey_composite`,
`season_win_pct`):

- `mean_abs_corr < 0.95`
- `max_abs_corr < 0.97`

The 3rd baseline (`season_win_pct`) is the lesson-from-Massey-decay
addition. It's the W/L-based feature most likely to duplicate
Colley.

Concrete prediction (calibrated against Massey's 0.957 vs adj_em):

- `corr(colley, adj_em)` in [0.80, 0.92]. Margin-vs-W/L distinction
  is the dominant axis. Likely PASS.
- `corr(colley, massey_composite)` in [0.85, 0.93]. Composite
  includes RPI (W/L-only), so some redundancy. Likely PASS but
  closer.
- `corr(colley, season_win_pct)` in [0.92, 0.97]. The most direct
  comparable -- Colley IS opponent-adjusted W/L. THIS IS THE
  LIKELY-FAIL CLAUSE. The 0.95 threshold is on a knife edge.

### Clause 2 -- no-harm headroom

Same as Massey: 3-season subset {2019, 2022, 2024}, train v4 with
and without `colley_rating` in `feature_cols`, compare mean test LL.
Pass: delta <= +0.001.

## Full LOSO backtest

If gate passes, run `python src/enhanced_model_v3.py` and apply the
ladder evaluated in order:

1. **Reject** if `LL_delta >= +0.001` OR `brkt_delta <= +10`.
2. Otherwise **Clear** if `LL_delta <= -0.005` OR `brkt_delta >= +25`.
3. Otherwise **Marginal**.

Same baselines as Massey: v4 = LL 0.4369, brkt = 2670 over 22 LOSO.

## Tests

Mirror the Massey tests but smaller (no mov_cap, no all-neutral, no
home-court, no decay):

1. **Synthetic schedule -- known answer.** 4 teams in a round-robin
   where wins are arranged so the closed-form Colley solution is
   computable by hand. Verify rating recovery to `1e-6`.
2. **Sum-to-(n/2) invariant.** Regression guard.
3. **Cache round-trip.** Write parquet + sidecar; read back; equal.
4. **Cache invalidation.** Bump producer version, confirm rebuild.
5. **Real-data smoke.** Solver runs on actual MRegularSeasonCompactResults
   (24 seasons, ~360 teams/season). Asserts no NaN/inf, ratings
   bounded (in [0, 1] per Colley's interpretation as expected
   win-rate-against-equal-opponent), sum-to-(n/2) per season within
   `1e-6`.

5 tests vs 9 for Massey -- the prior makes the system always
well-conditioned, so several Massey edge-case tests don't apply.

## File inventory

**New (committed):**

- `src/features/colley_matrix.py`
- `src/diagnose_colley.py`
- `tests/test_features/test_colley_matrix.py`
- `docs/superpowers/specs/2026-05-03-colley-matrix-feature-design.md`
  (this spec)
- `docs/superpowers/plans/2026-05-03-colley-matrix-feature.md`

**Generated artifacts (committed):**

- `output/diag_colley.json`
- `docs/notes/2026-05-03-colley.md`
- If gate passes: `output/v4_with_colley_loso.json`,
  `docs/notes/2026-05-03-colley-backtest.md`

**Generated artifacts (NOT committed):**

- `data/cache/colley_ratings.parquet` (+ sidecar)

**Edited:**

- `src/enhanced_model.py` -- 3-line wire-in.
- `TODO.md` -- update on verdict.

## Risks

1. **Colley duplicates `season_win_pct` (the new clause-1 baseline).**
   Most likely failure mode given the structural argument: Colley IS
   "opponent-adjusted W/L." If `season_win_pct` already provides
   enough W/L signal, and other features (KenPom WAB, late_season
   wins-vs-top-100) cover the opponent-adjustment angle, Colley adds
   nothing.
2. **Colley duplicates `massey_composite` via RPI.** RPI is one of
   the systems averaged into the composite, and it's W/L-based with
   opponent adjustment. Concrete duplication test on the data.
3. **Clause 1 passes but clause 2 fails (decay-Massey replay).**
   Even if Colley is structurally distinct from any single existing
   feature, v4 may already extract the joint signal via its 67-feature
   stack. The clause 2 gate is the real test.
4. **Disconnected components in early years.** Defensive `cond` check
   handles it; in practice D-I has been connected for decades.

## Out-of-scope followups

- Time-decay weighting on Colley.
- Bayesian-prior sweep (vary the +2 prior).
- Colley-Massey blend ratings (linear combination of the two).
- Production swap / live-bracket wiring.

If clause 2 fails, the followup priority is TODO #2 (hierarchical
Bradley-Terry with feature priors), not Colley variants -- the
shared "v4 feature stack already covers it" failure mode at this
data scale suggests the productive direction is to couple a
structurally distinct ratings model TO v4 features through priors,
rather than as a parallel feature.
