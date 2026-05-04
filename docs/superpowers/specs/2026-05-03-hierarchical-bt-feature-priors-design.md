# Hierarchical Bradley-Terry with v4 Feature Priors -- Design

**Date:** 2026-05-03
**Branch:** feat/hierarchical-bt-priors
**Predecessors:**
- Plain BT stage-1 (NO-GO, gate FAILED at clauses 2/3 -- weak standalone):
  `docs/superpowers/specs/2026-05-01-bayesian-stage1-design.md`
  `docs/notes/2026-05-01-bayesian-stage1.md`
- BT-as-feature for v9-C (NO-GO, pre-sweep gate FAILED, headroom -0.0015):
  `docs/superpowers/specs/2026-05-02-bt-as-feature-design.md`
  `docs/notes/2026-05-02-bt-as-feature.md`
- LR ensemble (NO-GO, residual correlation 0.77):
  `docs/notes/2026-05-01-ensemble-stage1.md`
- Massey + Colley parallel-feature failures (clause-2 LL regression at v4
  data scale): `docs/notes/2026-05-03-massey-mov.md`,
  `docs/notes/2026-05-03-colley.md`

## Motivation

The diversity-vs-strength axis is now well-mapped. Two stage-1 candidates
sit at opposite corners:

| candidate | residual corr vs v4 | standalone LL | gate verdict |
|-----------|---------------------|---------------|--------------|
| LR        | 0.77 (FAIL)         | 0.498         | NO-GO        |
| plain BT  | 0.58 (PASS)         | 0.565         | NO-GO (weak standalone) |

Plain BT showed that a **structurally disjoint inductive bias really does
produce uncorrelated errors** when the feature inputs are also disjoint
(BT sees only regular-season W/L). The blocker was standalone strength:
LL 0.565 is too far below v4's 0.437 for any non-degenerate blend weight
to help.

Hierarchical BT with feature priors targets exactly that blocker. Each
team's strength is shrunk toward a feature-derived mean:

```
s_team_i ~ Normal(beta . v4_features_team_i, sigma^2)
```

`sigma` is the central knob:
- `sigma -> 0`: strengths collapse to `beta . features` exactly. The
  model degenerates to a logistic regression on game outcomes with
  `beta` as the only learnable parameters. Standalone strength should
  approach v4's; residual correlation should also approach v4's
  (clause-1 risk).
- `sigma -> infinity`: strengths float free of the prior. Reduces to
  plain BT (already tested -- clauses 2/3 fail).
- **Intermediate sigma** is the only place hierarchical BT can land
  somewhere new on the diversity-strength plane. The whole experiment
  is a sweep over sigma.

The hypothesis: there exists a sigma at which standalone LL improves
materially over plain BT (closer to v4's 0.437) WITHOUT residual
correlation regressing past 0.60. Two-axis movement is what the prior
parallel-feature failures (Massey, Colley) and prior ensemble failures
(LR, plain BT) could not achieve.

## Goals

- Implement a per-season hierarchical BT MAP solver with a tunable
  `sigma` hyperparameter.
- Run a **sigma sweep** under the same 3-clause cheap gate from the
  plain-BT experiment. Each sigma cell = one diagnostic row.
- If the sigma sweep produces at least one cell that passes all three
  clauses, run the v9-C correction step + 22-season bracket-points
  head-to-head at that cell. Same verdict bands as the LR experiment
  (>= +25 = clear win, +10 to +25 = marginal, < +10 = no-go).
- Either way, ship a clean falsification record so the
  diversity-vs-strength frontier is fully mapped: plain BT (high
  diversity / weak), LR (low diversity / moderate), hierarchical BT at
  optimal sigma (the explicit interpolation point).

## Non-Goals

- Replacing v4 standalone. As with plain BT, the experiment is whether
  the hierarchical model **as an ensemble peer** of v4 wins on bracket
  points after v9-C correction.
- Full Bayesian posterior (PyMC / NumPyro / Stan with MCMC). MAP +
  L-BFGS keeps the experiment cheap and matches the plain-BT spec's
  "MAP-only" stance. Per-team posterior variance for "consistent vs
  volatile" team differentiation is item #4 in the active queue;
  deferred until standalone strength is settled.
- Cross-season parameter pooling for `beta`. Each season fits its own
  `(beta_y, sigma)` pair. Pooling adds tooling cost without obviously
  helping the diversity question.
- Margin-aware extension (Massey-style hierarchical regression). Adding
  margin re-introduces v4-flavored signal and partially defeats the
  diversity hypothesis. Held back as a follow-up if the binary version
  clears the gate.
- Adding `p_hbt` as a v9-C input feature (BT-as-feature replay). That
  experiment was already falsified at v4 data scale (NO-GO,
  docs/notes/2026-05-02-bt-as-feature.md). The hierarchical BT here is
  evaluated as an ensemble peer only.
- Re-tuning v9-C's W_UPSET / W_MISS / feature_set against the new
  ensemble. Reuse PR 9's winning cell. If the ensemble wins on bracket
  points, a small re-sweep is a follow-up.
- Live bracket integration (`generate_bracket_real.py`). Same
  out-of-scope rationale as every prior stage-1 experiment.
- Tuning `sigma_beta` (the L2 prior on beta). Use a single defensible
  default (`sigma_beta = 1.0` in standardized units); revisit only if
  the sigma sweep is clearly distorted by an over-regularized beta.
- Pre-existing unrelated v4 wire-in changes. This branch makes one
  surgical addition (`output/pairwise_hbt.csv` + the gate runner). v4
  itself is unchanged.

## Approach

### Architecture

```
src/train_hbt_stage1.py   -> output/pairwise_hbt_sigma_<S>.csv  (one per cell)
src/diagnose_hbt_vs_v4.py -> output/diag_hbt_sweep.json + stdout summary
                                |
                                v
                  GATE check per sigma cell:
                              r(resid_v4, resid_hbt) < 0.60
                              optimal blend w in [0.30, 0.85]
                              log-loss headroom > 0.005
                                |
            FAIL all cells ----|--------- PASS at >= 1 cell
              |                              |
              v                              v
       stop, write findings   pick best-headroom passing cell
                              ensemble_stage1.py + run_v9c_on_stage1.py
                              (existing infrastructure, no code changes)
                                              |
                                              v
                              22-season bracket-points head-to-head
```

The pairwise output schema matches `pairwise_v4.csv` and `pairwise_bt.csv`,
so all downstream tooling (`ensemble_stage1.py`, `run_v9c_on_stage1.py`,
`score_chalk_brackets.score_pairwise_path`) works unchanged.

### Per-season hierarchical BT by joint MAP

For each season `Y` in 2003-2025 (excluding 2020 implicitly):

**Inputs:**
- Y's regular-season games (`MRegularSeasonCompactResults` filtered to
  `Season == Y`), columns `WTeamID, LTeamID, WLoc`. ~5000-6000 games.
- Y's per-team v4 feature row from the matrix returned by
  `prepare_loso_inputs()`. ~330-365 teams x ~67 features. **Critical:**
  features must be available at the time of the regular-season fit,
  meaning v4's `prepare_loso_inputs` already constructs them as a
  function of regular-season + supplemental data and is evaluated at
  the appropriate horizon. We use the same per-team-per-season feature
  vector v4 trains on -- this is intentional, since the experiment is
  whether priors-derived-from-v4 lift BT's standalone strength.

**Feature standardization:**
- Compute per-season z-scores using train-season stats. For LOSO with
  held-out season `Y`, fit standardization (mean, std) over feature
  rows from all seasons `!= Y`, then apply to season `Y`. This avoids
  leakage and matches v4's training pattern. Drop columns with zero
  variance in the training fold (rare; defensive).

**Parameters per season:**
- `s` = vector of length `n_teams` (team strengths for that season).
- `beta` = vector of length `n_features` (feature -> strength
  coefficients, season-specific).
- `h` = scalar home-court advantage.

**Joint MAP objective (negative log posterior, minimize):**

```
L(s, beta, h) =
    sum_{games g} -log sigmoid(sign_g * (s[w_g] - s[l_g] + home_g * h))
  + (1 / (2 * sigma^2))    * sum_i (s[i] - X[i] @ beta)^2
  + (1 / (2 * sigma_beta^2)) * ||beta||^2
```

where `home_g` is `+1` (winner home), `-1` (winner away), `0`
(neutral); `sign_g = +1` (every row is winner-perspective). `X[i]` is
the standardized v4 feature row for team `i`.

**Properties:**
- The negative log-likelihood is convex in `(s, beta, h)` jointly (sum
  of logistic terms + quadratic prior, all convex).
- ~330 + 67 + 1 ~= 400 parameters; sparse design. L-BFGS converges in
  seconds.
- Identifiability: as in plain BT, the likelihood is invariant to a
  global shift of `s`. The Gaussian prior on `s - X @ beta` pins the
  null direction (its quadratic minimum is unique given the
  standardized feature centering). No reference team needed.

**Solver:**
`scipy.optimize.minimize(method='L-BFGS-B', jac=analytic_gradient)`.
Initialize `s` at zero, `beta` at zero, `h` at zero. `max_iter=500`,
`tol=1e-8`. Per-season fit budget ~1-3 seconds per cell.

**Pairwise output:**
For each unordered tournament-field pair `(a, b)` in season `Y`,
`p_a_wins = sigmoid(s[a] - s[b])`. NO home-court term (tournament is
neutral). Write to `output/pairwise_hbt_sigma_<S>.csv` per cell.

### Sigma sweep grid

Coarse-then-fine. Initial pass:

```
sigma in {0.05, 0.10, 0.20, 0.50, 1.00, 2.00, 5.00}    (7 cells)
sigma_beta = 1.0  (fixed)
```

Anchor cells:
- `sigma = 0.05`: tight prior; `s` close to `X @ beta`. Expect
  standalone LL approaching v4's, residual correlation high (clause-1
  risk).
- `sigma = 5.00`: loose prior; near plain-BT. Expect r ~ 0.58, LL ~
  0.56 (matching plain-BT findings).
- Intermediate cells: the actual frontier.

If a passing cell is found, optionally do a fine sweep around it
(`+/- 0.5` log-decade) to maximize headroom. The coarse sweep is
sufficient for the gate verdict.

7 cells x 22 seasons x ~2 sec = ~5 minutes wall time. Cheap.

### Diagnostic-first gate (per cell)

`src/diagnose_hbt_vs_v4.py` is structurally identical to
`src/diagnose_bt_vs_v4.py`, parameterized over the sigma cell:

1. Load `output/pairwise_v4.csv` (dedup'd) and the cell's
   `output/pairwise_hbt_sigma_<S>.csv`.
2. Load `MNCAATourneyCompactResults`. Extract `(p_v4, p_hbt)` for the
   actual winner of each played game in 2003-2025.
3. Compute per cell:
   - Standalone weighted-mean log loss + accuracy (v4 baseline shared
     across cells).
   - `r(residual_v4, residual_hbt)`.
   - Disagreement rate.
   - Cheating ideal-weight search `w in linspace(0, 1, 101)`. Print
     headroom = `LL_v4 - LL_optimal_blend`.
4. Apply the same three thresholds as the plain-BT spec:
   - `r < 0.60`
   - `optimal w in [0.30, 0.85]`
   - `headroom > 0.005`
5. Aggregate to `output/diag_hbt_sweep.json`: list of cells, per-cell
   metrics, per-cell pass/fail per clause, and **the best-headroom
   passing cell** (or `null` if none pass).

### Conditional v9-C head-to-head (only if >=1 cell passes)

Same flow as the plain-BT spec:

1. `python src/ensemble_stage1.py --in-a output/pairwise_v4.csv \
     --in-b output/pairwise_hbt_sigma_<S>.csv \
     --weights <w_v4>,<w_hbt> --out output/pairwise_hbt_ensemble.csv`
2. `python src/run_v9c_on_stage1.py --pairwise-in <ensemble> --pairwise-out <ensemble_v9c>`
3. Score via `score_chalk_brackets.score_pairwise_path` and compute
   per-season + total bracket-points delta vs v4+v9-C baseline.

Verdict bands: `>= +25` = clear win (separate swap-in commit), `+10 to
+25` = marginal candidate (document, do not swap), `< +10` = no-go.

### Anchor / sanity checks

- **Cell `sigma = 5.0` should approximately reproduce plain-BT
  numbers.** Standalone LL within ~0.02 of plain-BT's 0.565,
  r(residual) within ~0.05 of 0.58. Large divergence = bug in either
  the feature standardization, the optimizer, or the prior coupling.
- **Cell `sigma = 0.05` should have standalone LL approaching v4's.**
  In the limit it's a single-feature-coefficient logistic regression
  on game outcomes (mediated by per-team strengths that are nearly
  pinned to `beta . features`). Standalone LL above 0.50 at this cell
  = bug.
- **Home-court coefficient sanity.** Per-season `h` should land in
  `[0.3, 0.7]` (matching plain-BT). Out-of-band = bug in the home-court
  column.
- **Pair coverage match.** All cells produce CSVs with exactly the
  same `(season, team_a, team_b)` keys as `pairwise_v4.csv`. Mismatch
  = bug in field enumeration.
- **Convergence.** `result.success == True` for every (season, cell)
  fit. Any failures get logged with the season + sigma + final
  gradient norm; investigate before trusting the cell.

### Eval methodology and success bands

Same 22-season LOSO (2003-2025, 2020 implicit). Mirrors every prior
stage-1 backtest.

- **>= +25 brkt pts** over `v4 + v9-C`: clear win, separate swap-in.
- **+10 to +25**: marginal candidate, document but do not swap.
- **< +10**: no-go.

If all sigma cells fail the gate: same `< +10` no-go classification,
falsification recorded with the sigma sweep numbers.

## Falsification gate detail

The 3 clauses are reused verbatim from
`src/diagnose_bt_vs_v4.py`:

```
GATE_R_MAX        = 0.60
GATE_W_LOW        = 0.30
GATE_W_HIGH       = 0.85
GATE_HEADROOM_MIN = 0.005
```

These thresholds are deliberately unchanged from plain BT so the two
experiments are directly comparable cell-by-cell. If hierarchical BT
fails by a wider or narrower margin than plain BT, that delta is the
finding.

The expected sigma-frontier shape is the experiment's open question:

| regime          | expected r       | expected LL  | expected blend  |
|-----------------|------------------|--------------|-----------------|
| sigma = 0.05    | high (-> 0.7+)   | low (-> 0.45)| degenerate (~0.95) |
| sigma = 0.50    | ?                | ?            | **the open question** |
| sigma = 5.00    | ~0.58            | ~0.565       | degenerate (~0.98) |

If the open-question middle has no cell with `r < 0.60` AND
`headroom > 0.005`, the experiment cleanly falsifies the hypothesis
that this axis exists at v4 data scale.

## Module shape

### New module: `src/features/hierarchical_bt.py`

```python
_PRODUCER_VERSION = "v1"

def fit_one_season(
    games_df: pd.DataFrame,
    feature_matrix: pd.DataFrame,    # rows for this season's teams
    sigma: float,
    sigma_beta: float = 1.0,
    feature_cols: Sequence[str] = ...,
    feature_means: pd.Series = ...,  # train-fold standardization stats
    feature_stds: pd.Series = ...,
    max_iter: int = 500,
    tol: float = 1e-8,
) -> dict:
    """Fit per-season hierarchical BT MAP. Returns dict with keys
    s (np.ndarray, length n_teams), beta (np.ndarray, length n_features),
    h (float), team_ids (list[int]), success (bool), n_iter (int)."""

def predict_pairs(
    fit_result: dict,
    pair_team_ids: list[tuple[int, int]],
) -> np.ndarray:
    """Return p_a_wins for each (a, b) pair using s[a] - s[b]."""
```

Joint optimization via `scipy.optimize.minimize(method='L-BFGS-B')`
with analytic gradient.

### New module: `src/train_hbt_stage1.py`

CLI mirror of `src/train_bt_stage1.py`, but:
- Imports `prepare_loso_inputs()` from `enhanced_model_v3.py` to get
  the v4 feature matrix.
- Loops over a CLI-provided sigma list.
- Per sigma value: per season, fit `feature_matrix.Season != Y`
  standardization stats -> fit hierarchical BT for season `Y` ->
  emit pairs.
- Writes `output/pairwise_hbt_sigma_<S>.csv` per cell.

CLI:
```
python src/train_hbt_stage1.py \
    --sigmas 0.05,0.10,0.20,0.50,1.00,2.00,5.00 \
    --out-dir output/
```

### New module: `src/diagnose_hbt_vs_v4.py`

Mirror of `src/diagnose_bt_vs_v4.py` but:
- Loops over CLI-provided sigma cells (or auto-discovers
  `output/pairwise_hbt_sigma_*.csv`).
- Writes one summary JSON `output/diag_hbt_sweep.json` with per-cell
  rows AND a top-level `best_passing_cell` field.
- Prints a markdown table of cell -> (r, w*, headroom, verdict) to
  stdout.

### New tests

```
tests/test_features/test_hierarchical_bt.py:
  - synthetic 4-team season, sigma very small: recovers s ~ X @ beta
  - synthetic 4-team season, sigma very large: recovers plain BT
  - per-team prior interpolation: intermediate sigma yields s between
    BT MLE and X @ beta (qualitative)
  - identifiability: shifting all s by +c yields equivalent loss
    (within numerical noise) under loose prior, NOT under tight prior
  - home-court coefficient sanity on a synthetic schedule

tests/test_train_hbt_stage1.py:
  - smoke: train one sigma cell on a 2-season subset, validate CSV
    schema and pair count

tests/test_diagnose_hbt_vs_v4.py:
  - synthetic gate logic on a 2-cell sweep with known (r, w*, headroom)
  - best-cell selection picks max-headroom passing cell
```

Existing test suite (`pytest -v`) must remain green.

## Risks and mitigations

1. **Sigma sweep finds no passing cell.** The hypothesized middle
   ground may not exist at v4 data scale -- coupling BT to v4 features
   may pull the residual correlation up to LR-like 0.77 *before* the
   standalone strength gain materializes. Mitigation: the sweep is
   cheap (~5 min compute). Even a NO-GO produces the
   diversity-strength frontier map, which is a publishable falsification
   record and informs item #4 (full Bayesian) prioritization.
2. **Joint optimization non-convergence.** The 400-parameter joint
   problem is convex but can converge slowly at extreme sigma. Mitigation:
   `max_iter=500` is generous; per-season convergence is logged; cells
   with > 5% non-convergence rate get a warning and a "treat with
   suspicion" annotation in the findings.
3. **v4 feature matrix at fit-time is conditioning on tournament-aware
   information.** v4's features are computed end-of-season; the
   regular-season-only purity of plain BT is partly broken when we
   feed them in as a prior. This is *expected* -- the whole point is
   to lift standalone strength using v4-derived information -- but it
   means the diversity numbers from this experiment are NOT directly
   comparable to plain BT's "fully disjoint inputs" diversity.
   Mitigation: document this explicitly in the findings note. Compare
   against LR (which also uses v4's features) AND plain BT (which does
   not), and frame the finding as "where on the diversity-strength
   frontier does this land relative to those two anchors."
4. **Standardization leakage.** Computing standardization stats on all
   23 seasons (instead of train-fold only) would leak. Mitigation:
   per-LOSO-fold standardization is explicit in the spec and tested.
5. **Per-cell pairwise CSVs accumulate.** 7 cells x ~2200 pairs/season
   x 22 seasons = ~340K rows total spread across 7 files. Disk-cheap
   but should be force-added (`output/` is gitignored) to make the
   experiment reproducible. Mitigation: same `git add -f` precedent as
   the BT and feature-view-ensemble experiments.
6. **Sigma_beta default chosen wrong.** A too-tight `sigma_beta` over-
   shrinks the feature coefficients and effectively uncouples the
   prior from the features. Mitigation: anchor check at `sigma = 0.05`
   should approach v4 standalone LL. If it does NOT, sigma_beta is
   the suspect knob; do a small sweep over `sigma_beta in {0.5, 1.0,
   2.0}` at the best sigma.

## Lessons from prior experiments carried forward

1. **Diagnostic-first gate.** Per-cell gate runs in seconds before
   any v9-C / bracket-points work. Same paid-for-itself logic as
   plain BT and BT-as-feature.
2. **Reuse existing infrastructure.** No new pairwise-CSV schema, no
   new ensemble blender, no new v9-C runner. The hierarchical BT
   trainer drops a CSV; downstream tooling consumes it unchanged.
3. **Per-LOSO-fold standardization.** The LR experiment's
   `prepare_loso_inputs()` is the canonical entry point; we reuse it.
4. **Force-add CSVs** under `output/` per the existing precedent.
5. **No backwards-compatibility shims.** No env-var toggles in
   `train_upset_model.py` or `predict_2026_v9c.py`. The hierarchical
   BT lives in its own modules end-to-end.

## Out-of-scope follow-ups

- **Full Bayesian posterior with PyMC / NumPyro.** Exposes per-team
  variance for "consistent vs volatile" differentiation. Item #4 in
  the active queue. Held back unless this MAP version moves the
  needle, since the bottleneck (standalone strength vs diversity) is
  an inductive-bias question, not a posterior-vs-MAP question.
- **Margin-aware hierarchical BT.** Replace binary outcomes with a
  Gaussian-error margin model (Massey-style, but with feature priors).
  Adds a margin signal that v4 partially has via efficiency features;
  the diversity benefit is uncertain.
- **Cross-season pooling for `beta`.** Single global `beta` with
  per-season `s` deviations. Adds tooling cost (joint optimization
  across all seasons). Defer until single-season version's verdict is
  in.
- **Sigma_beta sweep.** Done only if the sigma sweep is clearly
  distorted (anchor check at `sigma = 0.05` fails to recover v4-like
  standalone LL).
- **Production swap / live-bracket wiring.** Same out-of-scope as every
  prior stage-1 experiment.

## File-touch summary

```
new   docs/superpowers/specs/2026-05-03-hierarchical-bt-feature-priors-design.md
new   docs/superpowers/plans/2026-05-03-hierarchical-bt-feature-priors.md
new   src/features/hierarchical_bt.py
new   src/train_hbt_stage1.py
new   src/diagnose_hbt_vs_v4.py
new   tests/test_features/test_hierarchical_bt.py
new   tests/test_train_hbt_stage1.py
new   tests/test_diagnose_hbt_vs_v4.py
new   output/pairwise_hbt_sigma_<S>.csv  x 7 cells   (force-added)
new   output/diag_hbt_sweep.json                     (force-added)
new   docs/notes/2026-05-03-hierarchical-bt.md       (findings)

conditional (only if any cell passes):
new   output/pairwise_hbt_ensemble.csv               (force-added)
new   output/pairwise_v9c_hbt_ensemble.csv           (force-added)

edit  TODO.md                                        (mark active queue
                                                      item #1 done; renumber)
```

## Promotion path summary

```
gate runs (~5 min)
   |
   +-- ALL cells fail        -> NO-GO. Findings note. v4 stays as stage-1.
   |                            Item #1 falsified at v4 scale.
   |
   +-- >=1 cell passes        -> v9-C run on best cell (~minutes)
                                |
                                +-- delta < +10        -> NO-GO
                                +-- delta in [10, 25)  -> MARGINAL (document)
                                +-- delta >= +25       -> CLEAR. Separate
                                                          swap-in commit.
```
