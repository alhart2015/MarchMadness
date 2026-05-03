# Feature-View Diversity Ensemble -- Design

**Date:** 2026-05-02
**Branch:** feat/feature-view-ensemble
**Predecessors:**
- LR ensemble experiment (rejected, PR 11):
  `docs/superpowers/specs/2026-05-01-ensemble-stage1-design.md`
  `docs/notes/2026-05-01-ensemble-stage1.md`
- BT ensemble experiment (rejected, PR 12):
  `docs/superpowers/specs/2026-05-01-bayesian-stage1-design.md`
  `docs/notes/2026-05-01-bayesian-stage1.md`
- BT-as-feature for v9-C (rejected, PR 13):
  `docs/superpowers/specs/2026-05-02-bt-as-feature-design.md`
  `docs/notes/2026-05-02-bt-as-feature.md`
- v9-C production swap (current production stage-2):
  `docs/superpowers/specs/2026-05-01-v9c-production-swap-design.md`

## Motivation

Three prior experiments closed three cells of the (model-class,
ensemble-form) grid for stage-1 diversity:

| experiment | mechanism | failure mode |
|---|---|---|
| LR ensemble (PR 11) | global avg of v4 + LR on identical features | residual correlation r=0.77 too high |
| BT ensemble (PR 12) | global avg of v4 + BT (different model class) | BT too weak standalone (LL 0.565 vs 0.437) |
| BT-as-feature (PR 13) | learned per-context weight via v9-C | v9-C training data (2898 rows) too thin to gate on a noisy feature |

The remaining axis is **same model class on disjoint feature views**:
same XGBoost recipe as v4 means same standalone-strength profile
(sidesteps PR 12's bottleneck), and disjoint feature inputs mean errors
that cannot be correlated through shared inputs (sidesteps PR 11's
bottleneck). v9-C's role is unchanged from current production; the
diversity work happens upstream of v9-C in stage 1, so PR 13's
data-thinness diagnosis does not apply.

The hypothesis: a uniform or optimal-weight blend of two same-class
XGB peers on disjoint feature subsets beats v4 standalone on per-game
LL, and the win survives v9-C's stage-2 transform to bracket points.

This experiment runs two variants in the same PR:
- **E1 (clean test):** ensemble = blend(peer_A, peer_B). v4 not in
  the blend. The cleanest falsification of the disjoint-view hypothesis
  -- a win is unambiguously attributable to the disjoint mechanism.
- **E2 (production-shape):** ensemble = blend(v4, peer_A, peer_B). v4
  stays as the third peer. More likely to land a win in absolute
  terms but conflates the disjoint mechanism with regularization /
  bagging effects.

Both variants run end-to-end through the existing 15-cell W_UPSET /
W_MISS sweep harness against v9-C's production cell as baseline.

## Goals

- Define a feature partition `(PEER_A, PEER_B)` of v4's 67 features
  with `PEER_A & PEER_B == empty_set` and `PEER_A | PEER_B ==
  v4_feature_cols`. Single source of truth in `src/feature_views.py`.
- Train two same-class (XGBoost, v4 hyperparameters) stage-1 peers
  on the partitioned views, producing 22-LOSO-season per-game OOF
  predictions: `output/pairwise_peer_a.csv`, `output/pairwise_peer_b.csv`.
- Run a cheap **3-clause pre-sweep falsification gate** (`src/diagnose_feature_view_ensemble.py`):
  1. Per-peer LL ceiling: each peer's weighted-mean LL within `0.025`
     of v4's (catches BT-ensemble-style standalone weakness).
  2. Inter-peer residual correlation: `rho(resid_A, resid_B) < 0.60`
     (catches LR-ensemble-style correlated errors).
  3. Best-blend headroom: cheating-best 2-way blend of A and B beats
     v4 standalone by `>= 0.001` LL (catches BT-as-feature-style
     "decorrelated but no signal lift").

  All three clauses must PASS for the gate to clear. If any fails,
  write findings + TODO update + PR; do not run the sweep.
- If the gate clears, materialize E1 and E2 as ensembled stage-1
  pairwise CSVs and run the existing 15-cell sweep harness on each
  against v9-C's production cell at `(W_UPSET=1.25, W_MISS=0.0)` =
  2713 brkt pts. Report `delta_vs_v9c` per cell for both variants.
- Verdict bands per variant match the v9-C-vs-v8 swap spec:
  - `delta >= +25` -> clear winner, follow-up swap-in PR.
  - `+10 <= delta < +25` -> marginal candidate, document, do not swap.
  - `0 <= delta < +10` -> no-go, findings note.
  - `delta < 0` -> regression, findings note.

## Non-Goals

- Production swap. If E1 or E2 wins (`delta >= +25`), the swap-in
  PR is a follow-up: `predict_2026_feature_view_ensemble.py` mirror
  of `predict_2026_v9c.py`, plus repointing `output/pairwise_probs.json`
  consumers. The canonical pairwise output continues to be v9-C's
  until/unless that follow-up ships.
- 2026-final-snapshot peer fits. The committed `pairwise_peer_a.csv`
  and `pairwise_peer_b.csv` cover 22 LOSO seasons (2003-2025) only,
  which is what the experiment needs. A 2026 ensemble fit is a
  prerequisite for production swap, not for the experiment.
- Algorithmic / correlation-driven partitioning of features. The
  partition is hand-defined by feature semantics (team-strength view
  vs form+market view) and asserted disjoint at import time. A
  correlation-driven partition is an obvious follow-up if the
  hand partition's gate fails on clause 2.
- More than 2 peers. K=3+ partitions (per-source: KenPom-only,
  Vegas-only, raw-efficiency-only) make each peer thinner and risk
  failing clause 1. K=2 is the safer first cut; K=3+ is a follow-up
  if K=2 wins.
- v9-C feature-set changes. v9-C consumes whatever `p_stage1` it's
  pointed at; the diversity work is upstream. `train_upset_model.py`,
  `predict_2026_v9c.py`, and the `'v9a'/'v9b'/'v9c'/'v9d'` feature_set
  branches are unchanged.
- Live bracket integration (`generate_bracket_real.py`). Same
  out-of-scope rationale as predecessor experiments.
- Re-tuning v9-C's W_UPSET / W_MISS grid. Reuse the same 15-cell grid
  as PR 9 / v9-C sweep so results are directly comparable.
- Per-fold blend-weight tuning. The blend weights for E1 and E2 come
  from the gate diagnostic's full-22-season optimal weights. Per-fold
  optimization is reasonable but is a separate experiment because
  it would need its own LOSO discipline.

## Approach

### Feature partition

v4's 67 features partition by semantic role:

**PEER_A: team strength (40 features)** -- the "what's their level"
view. Full-season measures of how good a team is.

```
adj_oe, adj_de, adj_em, adj_tempo
off_efg, off_ft_rate, off_or_rate, off_to_rate
def_efg, def_ft_rate, def_or_rate, def_to_rate
kp_BARTHAG, kp_DREB%, kp_EFG%, kp_EFG%D, kp_ELITE SOS, kp_EXP,
kp_FTR, kp_FTRD, kp_K TEMPO, kp_KADJ D, kp_KADJ EM, kp_KADJ O,
kp_OREB%, kp_TALENT, kp_TOV%, kp_TOV%D, kp_WAB
massey_COL, massey_DOL, massey_MOR, massey_POM, massey_RPI,
massey_SAG, massey_WOL, massey_composite
conf_strength, season_avg_mov, season_win_pct
```

**PEER_B: form + market + meta (27 features)** -- the
"trajectory and outside-info" view. Recent-form measures, market
information, and team-meta features.

```
late_adj_oe, late_adj_de, late_adj_em, late_sos
efficiency_trend, margin_trend, scoring_trend
rolling_oe, rolling_de
win_pct_last10, win_pct_30d, avg_mov_last10
conf_tourney_wins, conf_tourney_champ
coach_career_games, coach_career_wins, coach_career_winpct,
coach_career_f4_apps, coach_career_champs, coach_career_seasons
vegas_avg_spread, vegas_avg_margin, vegas_ats_pct,
vegas_power_rating, vegas_consistency, vegas_game_count,
vegas_late_spread_delta
```

Rationale: the partition cuts at the cleanest semantic seam in v4's
feature set. Full-season strength stats (PEER_A) and
trajectory/market/meta signals (PEER_B) naturally inform different
aspects of game outcomes; the LR-ensemble's r=0.77 came from feeding
the same XGB feature set into a model with the same training-row
view, so the residuals were forced to track each other. Two XGBs on
strictly disjoint inputs cannot couple their predictions through
shared features.

The partition lives in `src/feature_views.py` as two frozen lists.
A `validate_partition(all_cols)` helper asserts disjointness and
union-completeness against `enhanced_model_v3.get_feature_cols`'s
output at import time -- if v4 gains a feature that's not assigned
to a peer, the next pytest run breaks.

### Architecture

```
data/raw/...  ->  prepare_loso_inputs()  ->  feature_matrix (67 cols)
                                                     |
                              +----------------------+----------------------+
                              v                                             v
                  PEER_A: team-strength view                  PEER_B: form+market view
                  (40 features)                                (27 features)
                              |                                             |
                              v                                             v
                  XGBoost (v4 hyperparams)                     XGBoost (v4 hyperparams)
                  trained per LOSO fold                        trained per LOSO fold
                              |                                             |
                              v                                             v
                  output/pairwise_peer_a.csv                   output/pairwise_peer_b.csv
                              |                                             |
                              +----------------------+----------------------+
                                                     v
                                  src/diagnose_feature_view_ensemble.py
                                  3-clause pre-sweep gate
                                                     |
                                FAIL --------+-------|--------------- PASS
                                  |                                     |
                                  v                                     v
                              findings,            blend at gate-derived optimal weights:
                              TODO,                E1: blend(A, B)         -> output/pairwise_ensemble_e1.csv
                              PR                   E2: blend(v4, A, B)     -> output/pairwise_ensemble_e2.csv
                                                                       |
                                                                       v
                                                  V9_STAGE1_PAIRWISE=<csv> python src/sweep_v9_weights.py
                                                  (existing 15-cell W_UPSET/W_MISS sweep, twice)
                                                                       |
                                                                       v
                                                  per-cell delta_vs_v9c
                                                                       |
                                                                       v
                                                  verdict bands -> findings note
```

`output/pairwise_v4.csv` is read-only (existing artifact). All new
logic is additive -- existing v4, v8, v9-A/B/C/D code paths
unchanged.

### Module changes

| Module | Change |
|---|---|
| `src/feature_views.py` *(new)* | Defines `PEER_A_FEATURES` and `PEER_B_FEATURES` as frozen lists. `validate_partition(all_cols: list[str]) -> None` asserts (a) every col in `all_cols` is in exactly one of PEER_A or PEER_B, (b) every PEER_A and PEER_B element is in `all_cols`. Raises `ValueError` with a clear message on failure. Module constants are the single source of truth for the partition; downstream modules import these. |
| `src/train_peer_stage1.py` *(new)* | Mirrors `src/train_lr_stage1.py`'s shape (PR 11 retained as reusable scaffolding). Args: `--peer {a\|b}` selects the feature list; `--output PATH` for the destination pairwise CSV. For each LOSO fold (using `enhanced_model_v3.prepare_loso_inputs` -> `feature_matrix`, `tourney_results`, `feature_cols`), calls the same XGBoost trainer and hyperparameters as v4 but restricts `feature_cols` to PEER_A or PEER_B. Emits OOF predictions in the same schema as `output/pairwise_v4.csv`: `(season, team_a, team_b, p_a_wins, p_min_wins)`. |
| `src/diagnose_feature_view_ensemble.py` *(new)* | Pre-sweep 3-clause falsification gate. Loads `pairwise_v4.csv`, `pairwise_peer_a.csv`, `pairwise_peer_b.csv`, `MNCAATourneyDetailedResults.csv`. For each LOSO season, joins played-game outcomes; computes per-game LL contribution. Computes weighted-mean LL across 22 seasons for v4, peer_A, peer_B. Computes residuals `(p - y)` on played-game rows for peer_A and peer_B; computes Pearson `r(resid_A, resid_B)`. Computes optimal 2-blend weight `w*` minimizing `LL(w * p_a + (1-w) * p_b)` via `scipy.optimize.minimize_scalar` on `[0, 1]`. Writes `output/diag_feature_view_ensemble.json` with all clause values, thresholds, and an overall `gate_verdict`. Threshold constants at top of file: `PEER_LL_CEILING_DELTA = 0.025`, `RESID_CORR_MAX = 0.60`, `HEADROOM_MIN = 0.001`. Exits nonzero on FAIL so a wrapper short-circuits. Mirrors `src/diagnose_v9d.py` and `src/diagnose_bt_vs_v4.py` in shape. |
| `src/ensemble_stage1.py` *(extend, retained from PR 11)* | If existing helpers don't cover K-way blending, add `blend_weighted(predictions: list[Path], weights: list[float], output: Path) -> None`. The blend takes the per-row average of `p_min_wins` across the input pairwise CSVs at the supplied weights, weighted to sum to 1. Used to materialize `pairwise_ensemble_e1.csv` and `pairwise_ensemble_e2.csv`. |
| `src/sweep_v9_weights.py` *(extend)* | Add `V9_STAGE1_PAIRWISE` env var that overrides the default `output/pairwise_v4.csv` input path. Output dir keys off the input filename basename: `output/v9c_<basename>_sweep/` (e.g., `output/v9c_ensemble_e1_sweep/`, `output/v9c_ensemble_e2_sweep/`). Existing `V9_FEATURE_SET=v9c\|v9d` paths and default behavior (when env var unset) unchanged. |

No changes to `src/train_upset_model.py`, `src/predict_2026_v9c.py`,
`enhanced_model_v3.py`, `generate_bracket_real.py`, `score_chalk_brackets.py`.

### Falsification reasoning

The 3-clause gate isolates the three known stage-1-diversity failure
modes. Each clause is attributable to a specific prior result:

1. **Per-peer LL ceiling (`peer_LL <= v4_LL + 0.025`).** Maps to
   PR 12's BT-ensemble: a peer too weak standalone cannot produce a
   useful blend weight, regardless of error decorrelation. v4's
   weighted-mean LL is approximately `0.4369`; the ceiling sits at
   approximately `0.4619`. Either subset XGB is expected to land
   within this -- v4 has 67 features and the most-feature-rich peer
   (PEER_A) has 40, so the LL gap from feature-count alone is bounded
   by the shape of the validation curve, which historically flattens
   well before 40 features in this domain.
2. **Inter-peer residual correlation (`rho < 0.60`).** Maps to PR 11's
   LR-ensemble (r=0.77 on identical features). The threshold matches
   PR 12's identical clause; PR 12 measured r=0.58 and that *passed*
   the clause, so 0.60 is calibrated to the boundary between
   "structurally diverse enough to blend" and "too coupled to
   contribute." Disjoint XGB feature inputs cannot share variance
   through a common input column, so the prior on this clause is
   strongly favorable -- the only way it fails is if the underlying
   tournament-game signal is so dominantly driven by a single latent
   factor that two views of orthogonal subsets still see the same
   rank ordering of teams.
3. **Best-blend LL headroom (`>= 0.001`).** Maps to PR 13's
   BT-as-feature gate. The threshold is the same as PR 13's. The
   reasoning: at 1449 played games over 22 seasons the paired-comparison
   weighted-mean LL has standard error well below 0.001, so the
   threshold is empirically detectable but tight enough that a
   marginal pass is not noise.

If all three pass, the disjoint-view hypothesis has cleared every
prior failure mode by construction, and the question reduces to "does
the LL win convert to bracket points." If any one fails, the
hypothesis is falsified at that mode and the sweep is not run.

The gate uses 22-season weighted-mean LL with the same per-season
weighting as `score_pairwise_path` and `train_upset_model.py`'s
`double_loso_eval`. Using the same scoring code on the same per-game
rows for v4, peer_A, peer_B, and any blend cancels the bulk of
sample variance via paired comparison.

### Verdict bands (E1 and E2 independently)

Match the v9-C-vs-v8 swap spec exactly. Each variant gets a
verdict against v9-C's production cell at `(W_UPSET=1.25, W_MISS=0.0)`
= 2713 brkt pts.

| `delta_vs_v9c` | label | action |
|---|---|---|
| `>= +25` | clear winner | follow-up swap-in PR (production stage-1 -> ensemble) |
| `+10` to `+24` | marginal candidate | document, do not swap |
| `0` to `+9` | no-go | findings note |
| `< 0` | regression | findings note |

E1 is the clean hypothesis test; E2 is the practical "what would
actually go to production" variant. Both run in the same PR if the
gate clears. If E1 fails verdict but E2 wins, the win is documented
as "the ensemble works, but only when v4 is in the blend"
(architectural finding, not a clean validation of the disjoint
hypothesis).

Comparison baseline is v9-C re-scored fresh from
`output/v9c_sweep/pairwise_v9_WU1.25_WM0.00.csv` using
`score_pairwise_path` -- not lifted from a prior log -- so v9-C's
baseline and the ensemble variants come from the same scoring code
on the same call.

### Anchor + join sanity discipline

Three anchors must pass before trusting cell rankings.

**1. Partition completeness anchor (unit test).**
`validate_partition(get_feature_cols(feature_matrix))` asserts
PEER_A and PEER_B are disjoint and exhaustively cover v4's feature
list. Catches drift if v4 gains a feature that's not assigned.

**2. v9-C harness anchor.** Run
`V9_STAGE1_PAIRWISE=output/pairwise_v4.csv V9_FEATURE_SET=v9c python
src/sweep_v9_weights.py` for the anchor cell `(W_UPSET=1.0, W_MISS=0.0)`;
confirm max prob delta vs the committed v9-C anchor cell `1e-9`. This
catches harness regressions from the env-var threading. If it fails,
the env-var extension to `sweep_v9_weights.py` broke the v9-C path
-- abort and debug before any ensemble numbers are trusted.

**3. Peer pairwise CSV symmetric-pair anchor (unit test).** For 5
sampled `(season, a, b)` tuples loaded from `pairwise_peer_a.csv`
and `pairwise_peer_b.csv`, assert
`p_a_wins(season, a, b) + p_a_wins(season, b, a) == 1.0` within float
tolerance. Catches symmetric-pair generation bugs in the peer
trainer.

If any anchor fails, the result is invalid -- abort and debug before
reading any cell numbers.

### Disposition matrix

| outcome | E1 verdict | E2 verdict | branch deliverables | TODO.md update |
|---|---|---|---|---|
| Pre-gate FAIL | n/a | n/a | gate JSON + peer CSVs + findings + spec + plan + tests | move queue #1 to Tried-and-rejected; queue #2 (hierarchical BT) advances |
| Pre-gate PASS, both `< +10` | no-go | no-go | full deliverables (below) | move queue #1 to Tried-and-rejected; queue #2 advances |
| Pre-gate PASS, E1 `< +10`, E2 `>= +10` and `< +25` | no-go | marginal | full deliverables; findings emphasize "ensemble needs v4 in blend" | mark queue #1 partially closed; queue #2 advances |
| Pre-gate PASS, E1 `< +10`, E2 `>= +25` | no-go | clear winner | full deliverables; **add new queue item: production swap PR for E2** | mark queue #1 partially closed; queue #2 advances |
| Pre-gate PASS, E1 `>= +10` and `< +25`, E2 any | marginal | any | full deliverables; document E1 as marginal candidate | mark queue #1 done at the better verdict |
| Pre-gate PASS, E1 `>= +25` | clear winner | any | full deliverables; **add new queue item: production swap PR for E1 (or E2 if better)** | mark queue #1 done as clear winner |

In all cases, `docs/notes/2026-05-02-feature-view-ensemble.md` is
committed with the verdict and `TODO.md` is updated as part of the
same PR.

## File deliverables

```
src/feature_views.py                                                  (new)
src/train_peer_stage1.py                                              (new)
src/diagnose_feature_view_ensemble.py                                 (new)
src/ensemble_stage1.py                                                (extended)
src/sweep_v9_weights.py                                               (extended)

output/pairwise_peer_a.csv                                            (always after gate diag)
output/pairwise_peer_b.csv                                            (always after gate diag)
output/diag_feature_view_ensemble.json                                (always)

output/pairwise_ensemble_e1.csv                                       (only on PASS)
output/pairwise_ensemble_e2.csv                                       (only on PASS)
output/v9c_ensemble_e1_sweep/pairwise_v9_WU{u:.2f}_WM{m:.2f}.csv      (15 cells, only on PASS)
output/v9c_ensemble_e2_sweep/pairwise_v9_WU{u:.2f}_WM{m:.2f}.csv      (15 cells, only on PASS)
output/v9c_ensemble_e1_sweep_results.csv                              (only on PASS)
output/v9c_ensemble_e2_sweep_results.csv                              (only on PASS)

docs/notes/2026-05-02-feature-view-ensemble.md                        (findings; always)
docs/superpowers/specs/2026-05-02-feature-view-ensemble-design.md     (this file)
docs/superpowers/plans/2026-05-02-feature-view-ensemble.md            (next step)

tests/test_feature_views.py                                           (new)
tests/test_train_peer_stage1.py                                       (new)
tests/test_diagnose_feature_view_ensemble.py                          (new)
tests/test_sweep_v9_weights.py                                        (extended)
```

## Tests

All CI-fast; no real-data-grade fixtures in unit tests:

1. **`tests/test_feature_views.py` (new):**
   - `test_partition_disjoint`: `set(PEER_A_FEATURES) & set(PEER_B_FEATURES) == set()`.
   - `test_partition_complete_against_v4`: With a synthetic feature-matrix fixture mirroring v4's column shape, `validate_partition(get_feature_cols(fm))` returns without raising.
   - `test_validate_partition_raises_on_missing_feature`: a feature in `all_cols` but in neither peer list raises `ValueError` with a message naming the feature.
   - `test_validate_partition_raises_on_extra_feature`: a feature in PEER_A but not in `all_cols` raises `ValueError`.

2. **`tests/test_train_peer_stage1.py` (new):**
   - `test_train_peer_a_uses_only_peer_a_features`: synthetic fixture with both peers' features; assert the trained model's `feature_names_in_` equals `PEER_A_FEATURES` (XGBoost exposes this attribute).
   - `test_train_peer_writes_documented_schema`: assert the output CSV has columns `[season, team_a, team_b, p_a_wins, p_min_wins]` and that the symmetric-pair invariant holds on at least one sampled tuple.
   - `test_train_peer_unknown_peer_raises`: `--peer c` raises `ValueError`.

3. **`tests/test_diagnose_feature_view_ensemble.py` (new):**
   - `test_gate_passes_all_clauses`: synthetic peers rigged so peer LLs are 0.45 and 0.46, residual correlation 0.40, best-blend LL 0.435 (vs v4 0.437). All three clauses pass; `gate_verdict == 'PASS'`.
   - `test_gate_fails_clause_1_per_peer_ll_ceiling`: peer A LL 0.50 (ceiling 0.462); other clauses pass; `gate_verdict == 'FAIL'` with `failed_clauses == ['per_peer_ll_ceiling']`.
   - `test_gate_fails_clause_2_residual_correlation`: residuals correlated 0.75; other clauses pass; `gate_verdict == 'FAIL'` with `failed_clauses == ['residual_correlation']`.
   - `test_gate_fails_clause_3_headroom`: best-blend LL 0.4369 (no headroom); other clauses pass; `gate_verdict == 'FAIL'` with `failed_clauses == ['blend_headroom']`.
   - `test_gate_fails_multiple_clauses`: rigged to fail clauses 1 and 3 simultaneously; `failed_clauses` contains both.
   - `test_gate_exits_nonzero_on_fail`: subprocess invocation exits with nonzero code on a FAIL fixture.

4. **`tests/test_sweep_v9_weights.py` (extended):**
   - `test_sweep_uses_v9_stage1_pairwise_env_var`: monkeypatch the env var to a fixture path; assert the sweep loads from that path and writes outputs to the basename-keyed dir.
   - `test_sweep_default_path_unchanged_when_env_var_unset`: regression test; the env-var-unset path matches the pre-extension default.

The v9-C anchor (re-running the v9-C sweep cell to within `1e-9`) is
verified manually as part of running the full sweep -- a full sweep
on real data is too slow for CI. The findings note will paste both
the v9-C anchor reproduction max-delta and the partition / join unit
test status as evidence.

## Implementation order

1. `src/feature_views.py` + `tests/test_feature_views.py`. Read v4's
   feature list, define PEER_A/PEER_B lists, `validate_partition`,
   tests. Run `pytest tests/test_feature_views.py`.
2. `src/train_peer_stage1.py` + `tests/test_train_peer_stage1.py`.
   Mirror `train_lr_stage1.py`'s shape; reuse v4 hyperparameters and
   `prepare_loso_inputs`. Restrict to peer's feature list. Run
   `pytest tests/test_train_peer_stage1.py`.
3. Run `python src/train_peer_stage1.py --peer a --output
   output/pairwise_peer_a.csv` and `--peer b --output
   output/pairwise_peer_b.csv` against real data. Sanity-check schema
   and symmetric-pair invariant.
4. `src/diagnose_feature_view_ensemble.py` +
   `tests/test_diagnose_feature_view_ensemble.py`. Implement the
   3-clause gate; tests for each clause failure mode plus pass.
   Run `pytest tests/test_diagnose_feature_view_ensemble.py`.
5. Run `python src/diagnose_feature_view_ensemble.py` against real
   data. Commit `output/diag_feature_view_ensemble.json`.
6. **Branch on the gate verdict.**
   - **FAIL:** write findings note (NO-GO), update TODO.md, commit, PR.
   - **PASS:** continue to step 7.
7. Run the full ingest/feature/integration test suite per CLAUDE.md
   forced-verification rule:
   `pytest -v tests/test_ingest tests/test_features tests/test_integration.py`.
8. Extend `src/ensemble_stage1.py` (`blend_weighted` if needed); use
   gate-derived optimal weights to materialize
   `output/pairwise_ensemble_e1.csv` (peer_A and peer_B at optimal
   2-way weights) and `output/pairwise_ensemble_e2.csv` (v4, peer_A,
   peer_B at optimal 3-way weights computed alongside the 2-way
   weight in the gate diagnostic).
9. Extend `src/sweep_v9_weights.py` for `V9_STAGE1_PAIRWISE` env var.
   Add tests in `tests/test_sweep_v9_weights.py`. Run that file.
10. Run `V9_STAGE1_PAIRWISE=output/pairwise_ensemble_e1.csv
    V9_FEATURE_SET=v9c python src/sweep_v9_weights.py`. Commit
    `output/v9c_ensemble_e1_sweep_results.csv` and the 15 per-cell
    CSVs.
11. Run `V9_STAGE1_PAIRWISE=output/pairwise_ensemble_e2.csv
    V9_FEATURE_SET=v9c python src/sweep_v9_weights.py`. Commit
    `output/v9c_ensemble_e2_sweep_results.csv` and the 15 per-cell
    CSVs.
12. Run `V9_STAGE1_PAIRWISE=output/pairwise_v4.csv V9_FEATURE_SET=v9c
    python src/sweep_v9_weights.py` on the anchor cell only; assert
    max prob delta vs the committed v9-C anchor `< 1e-9`. Document
    in findings note.
13. Compute per-cell `delta_vs_v9c` for E1 and E2 against re-scored
    v9-C baseline. Identify best cell for each. Apply verdict bands.
14. Write `docs/notes/2026-05-02-feature-view-ensemble.md` (findings).
    Update `TODO.md`. Commit, PR.

Steps 1-4, 8-9 are pure code + tests with no real-data dependency
and are the bulk of the implementation work. Steps 3, 5, 7, 10-13
are running-the-experiment work. Total real-data runtime is
dominated by the two sweeps (~45-75 min each on real data, run
sequentially) and the per-fold peer training (~5-10 min per peer x
22 folds, can be parallelized).
