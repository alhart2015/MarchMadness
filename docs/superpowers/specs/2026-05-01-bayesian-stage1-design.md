# Bayesian / Bradley-Terry Stage-1 -- Design

**Date:** 2026-05-01
**Branch:** feat/bayesian-stage1
**Predecessors:**
- LR ensemble experiment (rejected, PR 11):
  `docs/superpowers/specs/2026-05-01-ensemble-stage1-design.md`
  `docs/notes/2026-05-01-ensemble-stage1.md`
- v9-C production swap (current production stage-2):
  `docs/superpowers/specs/2026-05-01-v9c-production-swap-design.md`

## Motivation

The LR ensemble (PR 11) tested whether averaging v4's XGBoost stage-1
with a logistic regression on the same 67-feature matrix produces
uncorrelated errors. Verdict: NO-GO at -105 brkt pts, with the
post-mortem diagnostic showing residual correlation `r=0.77` and a
cheating ideal-weight search picking `w=0.93` (use v4 93%, LR 7%) for
just `+0.0006` log-loss headroom. The result is structural, not a
tuning artifact: at identical feature inputs, two general-purpose
classifiers learn similar things and their errors stay correlated.

Bradley-Terry sidesteps that ceiling by construction. Each season's
team strengths are fit on per-season regular-season game outcomes
only -- nothing from the v4 feature matrix, no cross-season learning,
no margin info. The inductive bias is "teams have a single latent
strength inferred from win/loss outcomes against direct opponents,"
which is structurally disjoint from "teams have many features
recursively partitioned by trees on tournament + supplemental
late-season data." If errors from this signal class are still 0.77+
correlated with v4's, then no model-class diversity at this scale
will help; if they're meaningfully lower, ensembling has a real shot.

## Goals

- Build a per-season Bradley-Terry stage-1 model that produces
  pairwise tournament-game predictions (`p_a_wins` per `(season,
  team_a, team_b)` pair, same schema as `output/pairwise_v4.csv`).
- Run a cheap *diagnostic gate* (residual correlation + ideal-weight
  search) before any expensive backtest. If the gate fails, stop.
- If the gate clears, run the v9-C correction step and the 22-season
  bracket-points head-to-head, with the same verdict bands as the
  LR experiment (>= +25 brkt pts = clear win, +10 to +25 = marginal,
  < +10 = no-go).
- Either way, ship a clean falsification record so the next
  diversity attempt (feature-view diversity, or a richer Bayesian
  model with feature priors) starts from real evidence.

## Non-Goals

- Replacing v4 standalone. BT alone almost certainly loses to v4 on
  bracket points (binary outcomes ignore the 67 v4 features); the
  experiment is whether averaging v4 + BT helps.
- Margin-aware BT (Massey-style). Binary outcomes are the maximum
  departure from v4's already-margin-aware features; pulling margin
  back in re-introduces v4-flavored signal and partially defeats the
  diversity hypothesis. Held back as a follow-up if plain BT looks
  promising but underperforms standalone.
- Hierarchical Bayesian with feature priors (`s_team ~ Normal(beta .
  features_team, sigma)`). That couples the BT signal back to v4's
  feature view -- exactly the correlation source we are testing
  against. Held back as a follow-up.
- MCMC / variational inference (PyMC, NumPyro, Stan). Plain
  L2-regularized logistic regression with team indicators is
  equivalent to MAP Bradley-Terry under a Gaussian prior; no new
  dependency needed. The full posterior over strengths would be
  nice for "consistent vs volatile" team differentiation, but that
  is a separate experiment with its own dependency cost.
- Cross-season parameter sharing. BT strengths are per-season
  parameters by design.
- Re-tuning v9-C's W_UPSET / W_MISS / feature_set against the BT
  ensemble. Reuse PR 9's winning cell. If the ensemble wins on
  bracket points, a small re-sweep is a follow-up.
- Live bracket integration (`generate_bracket_real.py`). Same
  out-of-scope rationale as the LR experiment.

## Approach

### Architecture

```
src/train_bt_stage1.py    -> output/pairwise_bt.csv         (new)
src/diagnose_bt_vs_v4.py  -> stdout + JSON summary          (new)
                                |
                                v
                  GATE check: r(resid_v4, resid_bt) < 0.6
                              optimal blend w in [0.3, 0.85]
                              log-loss headroom > 0.005
                                |
            FAIL ----------------|--------------- PASS
              |                                     |
              v                                     v
          stop, write findings        ensemble_stage1.py + run_v9c_on_stage1.py
                                          (existing infrastructure, no code changes)
                                                    |
                                                    v
                              22-season bracket-points head-to-head
```

The two new scripts produce CSVs with the same schema as
`pairwise_v4.csv` so the existing `ensemble_stage1.py` /
`run_v9c_on_stage1.py` / `score_chalk_brackets.score_pairwise_path`
toolchain works unchanged. Both of those modules land on `main`
through PR 11; this branch will rebase off `main` once that PR
merges, or use the existing branch's modules verbatim if PR 11 is
still open at run time. (See `Risks and mitigations` -- the modules
are pure utility code with no v4 / LR specific assumptions, so they
work identically on `pairwise_bt.csv`.)

### Per-season Bradley-Terry by MAP / L2 logistic regression

Mathematical equivalence: maximizing the L2-regularized logistic
likelihood for game outcomes, with one indicator per team and one
home-court column, is exactly Bradley-Terry MAP estimation under a
zero-mean Gaussian prior on team strengths. So the implementation
is just `sklearn.linear_model.LogisticRegression` with a hand-built
sparse design matrix.

For each season Y in 2003-2025 (excluding 2020 implicitly via
missing data):

1. **Load Y's regular-season games** from
   `data/raw/march-machine-learning-2026/MRegularSeasonCompactResults.csv`,
   filtered to `Season == Y`. Each row has columns
   `WTeamID, LTeamID, WLoc` (`H`/`A`/`N`).
2. **Enumerate teams.** Collect `team_ids = sorted(unique union of
   WTeamID and LTeamID))`. ~330-365 teams per season.
3. **Build the design matrix.** ~5000-6000 games per season. For
   each game: row vector of length `n_teams + 1`, with `+1` at
   the winner's index, `-1` at the loser's index, and a final
   home-court column with value `+1` if the winner was home, `-1`
   if the winner was away, `0` if neutral. Label is `1` (since
   the winner's column is `+1`, the model is being trained to
   predict the implied probability of a `+1` outcome). Use a
   `scipy.sparse.csr_matrix` -- the matrix is mostly zeros.
   Symmetric pair construction is NOT needed here -- one row per
   game suffices, because each game's "perspective" (winner-as-A
   vs loser-as-A) is already encoded by the sign pattern.
4. **Fit `sklearn.linear_model.LogisticRegression`** with
   `penalty='l2'`, `solver='lbfgs'`, `fit_intercept=False`,
   `C=10.0` (mild L2 regularization, defensible default;
   tunable in a follow-up if standalone BT looks pathologically
   weak). `max_iter=2000`. The fitted coefficient vector of
   length `n_teams + 1` is `(s_team_0, s_team_1, ..., s_team_n,
   h_advantage)`.
5. **Identifiability note.** Logistic regression with one indicator
   per team has a one-dimensional null space (adding a constant to
   every strength leaves predictions unchanged). L2 regularization
   pins the null direction (it pulls all strengths toward zero on
   average), so no team needs to be dropped as a reference and
   `fit_intercept=False` is the right choice.
6. **Pairwise output.** For each unordered pair `(a, b)` of teams
   in season Y's tournament field (`a < b`, taken from
   `MNCAATourneyCompactResults` or `MNCAATourneySeeds`), compute
   `p_a_wins = sigmoid(s_a - s_b)`. NO home-court term -- the
   tournament is neutral. Append `(season, team_a, team_b,
   p_a_wins)` rows to `output/pairwise_bt.csv`.
7. **Per-season logging.** For each fold, print n_teams, n_games,
   fitted home-court coefficient `h`, weighted-mean log loss on
   the season's tournament games (using `evaluate_pairwise` from
   the existing `src/eval_stage1.py`), and accuracy. The fitted
   `h` should be roughly `0.3-0.5` from public BT fits; a value
   far outside that range is a smoke signal.

### Diagnostic-first gate

`src/diagnose_bt_vs_v4.py` is a standalone module that mirrors the
post-mortem diagnostic from the LR experiment, runs *before* any
v9-C correction or bracket-points scoring. Concretely:

1. Load `output/pairwise_v4.csv` (dedup last-write-wins on
   `(season, team_a, team_b)`) and `output/pairwise_bt.csv`.
2. Load `data/raw/march-machine-learning-2026/MNCAATourneyCompactResults.csv`.
3. For each played tournament game in 2003-2025: extract `p_v4` and
   `p_bt` for the actual winner. Should yield ~1449 games.
4. Compute and print:
   - Standalone weighted-mean log loss + accuracy for v4 and BT.
   - `r(residual_v4, residual_bt)` -- Pearson on `(1 - p_winner)`
     residuals.
   - Disagreement rate on predicted winner.
   - Confusion-style breakdown: both correct, v4 only, BT only,
     both wrong.
   - Cheating ideal-weight search over `w in linspace(0, 1, 101)`:
     log loss at `w=1.0`, `w=0.5`, `w=0.0`, and `w_optimal`. Print
     headroom = `LL_v4 - LL_optimal`.
5. **Print verdict.**

   ```
   GATE PASSED: residual correlation X.XX < 0.6,
                optimal w = X.XX in [0.3, 0.85],
                headroom +X.XXXX > 0.005
   GATE FAILED: <which clause(s) failed and by how much>
   ```

6. **Save the diagnostic JSON** to a versioned path
   (`output/diag_bt_vs_v4.json`) for the findings note.

### Promotion bar

All three must clear:

- `r(residual_v4, residual_bt) < 0.6`. Set higher than the
  observed LR value (0.77) but not loose -- below 0.6 is the band
  where averaging meaningfully reduces variance.
- `optimal w in [0.3, 0.85]`. Non-degenerate weighting -- a model
  whose ideal weight is 0.92 is essentially v4 alone.
- `LL_v4 - LL_optimal > 0.005`. About an order of magnitude above
  the LR experiment's 0.0006 noise-level headroom; loosely
  consistent with the smallest standalone log-loss gap that has
  historically translated into +25 bracket points after v9-C
  correction.

If any clause fails, stop and write findings. No bracket-points
backtest.

### Conditional v9-C head-to-head (only if gate passes)

Reuse existing infrastructure with no code changes:

1. Average v4 + BT at the diagnostic's optimal weight (or 0.5/0.5
   for a more conservative LOSO-disciplined first read; the
   spec's bar is high enough that the choice of weight in this
   range is unlikely to change the verdict band):

   ```bash
   python src/ensemble_stage1.py \
     --in-a output/pairwise_v4.csv \
     --in-b output/pairwise_bt.csv \
     --out output/pairwise_bt_ensemble.csv \
     --weights <w_a>,<w_b>
   ```

2. Run v9-C on both `pairwise_v4.csv` and `pairwise_bt_ensemble.csv`:

   ```bash
   python src/run_v9c_on_stage1.py \
     --pairwise-in output/pairwise_v4.csv \
     --pairwise-out output/pairwise_v9c_v4_baseline.csv

   python src/run_v9c_on_stage1.py \
     --pairwise-in output/pairwise_bt_ensemble.csv \
     --pairwise-out output/pairwise_v9c_bt_ensemble.csv
   ```

3. Score both via `score_chalk_brackets.score_pairwise_path` and
   compute the per-season + total bracket-points delta. Same code
   pattern as the LR experiment's Task 9.

### Eval methodology and success bands

22-season LOSO (2003-2025, excluding 2020 implicitly via missing
data). Mirrors every prior backtest in this codebase.

- **>= +25 brkt pts** over `v4 + v9-C` baseline: clear win,
  recommend swap path (separate follow-up commit).
- **+10 to +25**: marginal candidate, document but do not swap.
- **< +10**: no-go. v4 stays as stage-1.

If the diagnostic gate fails: same `< +10` no-go classification,
but with the diagnostic numbers as the falsification evidence
instead of bracket points.

## Anchor / sanity checks

- **Standalone BT log loss should be in a believable range.** Public
  Bradley-Terry fits on college basketball regular-season data
  typically achieve ~0.55-0.65 log loss on tournament games. If the
  per-season LL prints look much worse than that (e.g., > 0.7), the
  L2 regularization or design-matrix construction is suspect.
- **Home-court coefficient sanity.** Reported `h` per season should
  be roughly `0.3-0.5` (mapping to a ~7-12% home-win-rate uplift).
  Far outside that range = bug in the home-court column.
- **Pair coverage match.** `pairwise_bt.csv` must have exactly the
  same `(season, team_a, team_b)` keys as the dedup'd
  `pairwise_v4.csv`. Verified the same way as the LR experiment
  (Pearson check on join coverage). Failure indicates a bug in the
  field enumeration -- likely a play-in / First Four edge case.

## Testing

- `tests/test_train_bt_stage1.py`:
  - Unit: design-matrix builder produces the expected sparse
    matrix on a 3-team, 4-game synthetic season.
  - Unit: fitted BT on synthetic data with strengths
    `(0.0, 1.0, 2.0)` and many simulated games recovers
    strengths in the right *order* and within ~0.3 of the
    true values (rough recovery; LR sigmoid coupling makes
    exact recovery noisy on small samples).
  - Unit: home-court column extraction from `WLoc` field is
    correct (`H -> +1`, `A -> -1`, `N -> 0`).
  - Smoke: end-to-end on a 2-season subset writes a valid CSV
    with the expected schema and pair count for that season's
    field.
- `tests/test_diagnose_bt_vs_v4.py`:
  - Unit: residual-correlation + ideal-weight search on a small
    synthetic input matches a hand-calculation.
  - Unit: gate logic: passing / failing each clause individually
    flips the verdict.
- Existing test suite (`pytest -v`) must remain green.

## Risks and mitigations

- **Risk: LR / ensemble modules from PR 11 not yet on `main`.** The
  Bayesian work depends on `src/ensemble_stage1.py`,
  `src/run_v9c_on_stage1.py`, and `src/eval_stage1.py`. Mitigation:
  rebase this branch on `main` after PR 11 merges. If PR 11 is
  rejected without merge, copy the relevant modules into this
  branch (they are clean utilities with no v4 / LR specific code).
  Either way, this branch should not run the bracket-points
  backtest until those modules are present.
- **Risk: the gate's thresholds are too tight or too loose.**
  Mitigation: the gate prints the actual numbers regardless of
  pass/fail. If the result lands in a borderline zone (e.g.,
  `r=0.62, headroom=0.004`), the findings note can argue for a
  case-by-case relaxation. The thresholds are a *default policy*,
  not a hard contract.
- **Risk: BT's standalone log loss is so weak that no blend
  weight makes the ensemble competitive.** This is the central
  hypothesis being tested. If it fails, the diagnostic captures
  it cheaply -- this is the failure mode the gate is designed to
  catch.
- **Risk: identifiability / convergence issues.** L2 regularization
  + `lbfgs` with `max_iter=2000` is well-behaved on this scale
  (~5k rows x ~350 cols, sparse). If `LogisticRegression.fit`
  warns about non-convergence on any season, the per-season log
  output will show it; investigate before trusting that season's
  numbers.

## Lessons from the LR experiment carried forward

1. **Diagnostic-first.** The LR experiment ran the full v9-C +
   bracket-points pipeline (~3 hours of compute) before learning
   that the residual correlation was 0.77 and the optimal blend
   was 93/7 -- a gate that takes 30 seconds. This spec puts the
   gate first.
2. **No backwards-compatibility shims.** This branch produces a
   new pairwise CSV alongside the existing ones. No env-var
   toggles in `train_upset_model.py` or `predict_2026_v9c.py`.
3. **Force-add output CSVs** (output/ is gitignored) following
   the existing `pairwise_v9.csv` precedent so the experiment
   record is reproducible from `git checkout`.
4. **Reuse `prepare_loso_inputs()` is NOT needed here.** BT does
   not consume v4's feature matrix. This is intentional --
   maximally disjoint inputs from v4 is the experiment's whole
   point. (`prepare_loso_inputs` lives on PR 11 / `feat/ensemble-
   stage1` and is reusable for future hierarchical-BT-with-feature-
   priors experiments.)

## Follow-ups (deliberately deferred)

- **Margin-aware BT (Massey-style regression).** If plain BT looks
  promising standalone but underperforms slightly, extending to
  margin-aware adds information without changing the model class.
- **Hierarchical BT with feature priors.** `s_team ~ Normal(beta .
  features_team, sigma)`, fit by MAP. Couples BT to v4 features --
  best done as a *separate* experiment after seeing whether the
  pure-BT version moves the correlation needle.
- **Full Bayesian Bradley-Terry with strength + variance per team.**
  Adds PyMC / NumPyro dependency. Lets "consistent but mediocre"
  vs "volatile but talented" teams differentiate. The TODO's
  Architecture Rethink (Tier C) entry is exactly this. Held back
  unless the simpler experiments justify the dependency cost.
- **Re-tuning v9-C's weights** against the BT ensemble. Reuse
  PR 9's winning cell for the first head-to-head; sweep is a
  follow-up if the ensemble lands in the marginal band.
- **Feature-view diversity (XGBoost on different feature subsets).**
  Tracked separately as TODO active queue item #3.
- **Wiring the BT ensemble into `generate_bracket_real.py`.** Same
  out-of-scope as the LR experiment.

## File-touch summary

```
new   docs/superpowers/specs/2026-05-01-bayesian-stage1-design.md  (this file)
new   docs/superpowers/plans/2026-05-01-bayesian-stage1.md         (next step)
new   src/train_bt_stage1.py
new   src/diagnose_bt_vs_v4.py
new   tests/test_train_bt_stage1.py
new   tests/test_diagnose_bt_vs_v4.py
new   output/pairwise_bt.csv                          (force-added)
new   docs/notes/2026-05-01-bayesian-stage1.md        (findings, after run)

conditional (only if gate passes):
new   output/pairwise_bt_ensemble.csv                 (force-added)
new   output/pairwise_v9c_bt_ensemble.csv             (force-added)
maybe output/pairwise_v9c_v4_baseline.csv if not yet on main from PR 11

edit  TODO.md                                         (mark active queue
                                                       item #1 done; renumber)
```
