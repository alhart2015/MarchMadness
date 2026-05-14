# v12 -- Stage-2 enrichment with v4 top-N feature diffs (on top of v13 ensemble + blend) -- Design

**Date:** 2026-05-14
**Branch:** `feat/v12-stage2-v4-feature-diffs` (already created off main 1ad48d3)
**Predecessors:**
- v13 PASS (PR #37, 2026-05-14): toss-up-bucket v4 x v8-ensemble blend scored
  2106 brkt pts vs same-env v8 single-seed rerun 2034 (+72 apples-to-apples
  under LOSO discipline; alpha grid `{0, 0.6}` picks 0.6 in 22/22 seasons).
- XGB env drift cleanup (PR #38, 2026-05-14): canonical pairwise_v8.csv
  regenerated under XGB 3.2.0 (new baseline 2034 brkt pts); v13's
  pairwise_v13.csv is the production bracket-selection frame.
- The architectural finding from v13: v4's 67-feature stack is NOT
  saturated -- a different stage-2 architecture extracts residual signal.
  This falsifies the "near-saturated" framing that built up across the
  seven same-data-peer FAILs (BT, feature-view, HBT, Colley, Massey-MOV,
  Massey-decay-14d, team-seed-residual). Implication: enriching stage-2's
  feature view (not v4's) may have additional headroom.

## Motivation

v13's stage-2 sees only 4 features: `[p_stage1, seed_a, seed_b, abs_seed_diff]`.
Everything v4 learned from its 67-feature stack is collapsed into the scalar
`p_stage1`. The hypothesis under test: stage-2 has access to under-expressed
signal that v4 already condensed into `p_stage1` but that XGB at the stage-2
level can re-use as conditioning context.

Concretely: passing the top-N v4 feature *differences* (signed
`feat_a - feat_b`) into the stage-2 input gives the corrector two pieces of
information it doesn't get from `p_stage1` alone:

1. **Which features drove this pair's stage-1 prediction.** Two pairs with
   the same `p_stage1=0.55` can have very different underlying feature
   profiles (e.g. one pair driven by a huge KenPom-AdjEM gap, another by
   a Vegas-power-rating gap).
2. **The marginal stage-1 model's structure.** v4 maps a high-dimensional
   feature delta to a scalar via XGB; stage-2 currently has to invert that
   mapping with only seed context. Passing the deltas directly lets stage-2
   learn corrections conditional on the input structure, not just the
   output magnitude.

The v13 finding that the 4-feature stage-2 ensemble lifts the 22-season
backtest by +72 is direct evidence that more-expressive stage-2 architecture
can extract signal v4 leaves on the table. v12 tests whether widening the
*feature view* of that stage-2 architecture compounds the v13 lift.

**The hypothesis under test:** v13's stage-2 architecture (30-seed XGB
ensemble + toss-up bucket blend at alpha=0.6) enriched with the top-N
most predictive v4 feature differences lifts the 22-season LOSO bracket-
points number above v13's 2106 baseline.

## Goals

- Add a **`v12_n5` / `v12_n10` / `v12_n15` feature set** to
  `train_stage2_v10.py::FEATURE_SETS`. Each extends v13's stage-2 view
  with the top-N v4 features (ranked by XGB gain on the fitted-once v4
  model) as signed differences `feat_team_a - feat_team_b`.
- Ship a **deterministic v4 feature-ranking artifact**
  (`output/v4_feature_importance.csv`) so the choice of "top 10" is
  auditable and reproducible. Source: fit v4 once on the full LOSO-pooled
  training set with the canonical tuned params; sort features by XGB
  gain. Commit as a force-added artifact.
- Run the **6-cell joint LOSO grid** `(N, hparams)` where
  `N in {5, 10, 15}` and `hparams in {v8, v10cap}`. Pick the
  (N, hparams) cell per test season Y from training-season scores
  (LOSO discipline). Hold alpha=0.6 fixed (already LOSO-confirmed for
  v13, no need to re-search).
- Produce **`output/pairwise_v12.csv`** as the picked-cell stage-2 frame
  and **`output/pairwise_v12_blend.csv`** as the v13-style toss-up
  blend with v4. Score both against the 22-season bracket-points
  backtest and compare to v13's 2106.

## Non-Goals

- **No new stage-2 hyperparameter exploration beyond `{v8, v10cap}`.**
  v10cap already exists from v10 work; widening the search before any
  v12 signal is premature.
- **No re-tuning of the v13 blend alpha.** v13 already searched
  `{0, 0.6}` under LOSO and picked 0.6 in 22/22 seasons. If v12 shifts
  the optimal alpha by a meaningful amount, that's a follow-up.
- **No expansion to N > 15.** The top-15 features cover ~85-95% of total
  XGB gain on typical v4-shaped models; beyond that the marginal
  features are noise-class. If LOSO picks N=15 in many seasons we'll
  consider a follow-up with `{15, 20, 25}` -- but not in v1.
- **No change to v4 itself.** v4's pairwise_v4.csv is the input; stage-2
  changes only.
- **No new pool-aware bracket-construction work.** v12 stays on the
  22-season chalk-walk objective. Pool-aware (Active Queue #2) is a
  separate spec.

## Architecture

### Feature plumbing

```
v4 feature matrix (data/cache/feature_matrix_v3.parquet, 67 features per
                   (Season, TeamID))
        |
        v Phase 0 prep
v4 feature ranking (output/v4_feature_importance.csv,
                    columns: feature_name, gain, gain_rank)
        |
        +--> top-N feature names (N in {5, 10, 15})
        |
        v Phase 1 plumbing
train_stage2_v10.py::load_per_game_data extended to join
v4_feature_matrix on (season, w, l) and emit signed diffs
        diff_<i>_<feat> = feat_W - feat_L  (label=1 row)
        diff_<i>_<feat> = feat_L - feat_W  (label=0 row)
        |
        v FEATURE_SETS extended
{"v12_n5":  v8_base + ["expected_round"] + [diff_1..diff_5],
 "v12_n10": v8_base + ["expected_round"] + [diff_1..diff_10],
 "v12_n15": v8_base + ["expected_round"] + [diff_1..diff_15]}
```

Notes:
- v13's stage-2 is `v8` feature set (4 features). v12 includes
  `expected_round` (already added in v10a; same code path) AS WELL AS
  the v4 diffs -- the expected_round signal is cheap and v10a-alt-ens
  scored 2109 at the v13 blend, so its inclusion is a strict superset
  of v13's stage-2 view.
- Diffs are signed so the symmetric-pair label=0 row sees the negation.
  This matches the matchup pair convention in `src/models/matchup.py`
  and v4's training itself.
- Joining is on `(Season, TeamID)` -- the canonical join key. Per
  CLAUDE.md, never on names.
- v4 features for season Y are *already LOSO out-of-fold by construction*
  (pairwise_v4.csv is the per-season tuned-LOSO output). The diff inputs
  to v12 stage-2 thus introduce no leak.

### LOSO grid

| Cell | feature_set | hparams | seeds |
|------|-------------|---------|-------|
| 1    | v12_n5      | v8      | 30 (same as v13: 42, 142, 242, ..., 2942) |
| 2    | v12_n5      | v10cap  | 30 |
| 3    | v12_n10     | v8      | 30 |
| 4    | v12_n10     | v10cap  | 30 |
| 5    | v12_n15     | v8      | 30 |
| 6    | v12_n15     | v10cap  | 30 |

For each cell, run train_stage2_v10 with 30 seeds and produce
`output/pairwise_v12_<feature_set>_<hparams>.csv` (6 files).

**Per-season LOSO pick** (the discipline that keeps v12 honest): for
test season Y, score each of the 6 cells on the 21 training seasons
{T : T != Y} (after the v13-style toss-up blend at alpha=0.6). Pick
the cell maximizing summed brkt pts across those 21 training seasons.
Apply that picked cell to season Y. Repeat for all 22 test seasons.
Concatenate to produce the picked-cell pairwise frame.

This mirrors v13's per-season alpha pick exactly; just over a 6-cell
grid instead of a 2-cell grid.

### Anchor invariance

- **v12_n0 == v13 anchor.** `train_stage2_v10 --features v8 --seeds <30>`
  reproduces `pairwise_v8_ens30.csv` byte-equal. Adding `expected_round`
  alone (= v10a) shifts but stays close. We do NOT add a v12_n0 cell;
  v13 itself IS the n=0 baseline. The +25 PASS bar is against 2106.
- **Single-seed reproducibility.** `train_stage2_v10 --features v12_n5
  --seeds 42` produces an identical output across reruns under the same
  XGB env. Cell-level pairwise CSVs are force-added so the LOSO pick is
  reproducible.

## Decision matrix

| 22-season blended total | Verdict | Action |
|-------------------------|---------|--------|
| >= 2131 (+25 vs v13)    | **PASS** | Ship v12 as the new production frame. Update README/TODO. Adopt as input to score_v13_blend.py (or rename to score_v12.py). |
| 2117..2130 (+11..+24)   | **MARGINAL** | Document; defer to next lane (pool-aware #2 or roster #1). Optionally sweep one more knob (extended N grid or +5 hparams) before closing. |
| 2089..2116 (-17..+10)   | **FAIL (null)** | Document the null. Strengthens the "v4-derived feature diffs don't compound v13" finding. Close v12 lane; move to next active-queue item. |
| < 2089 (worse by >17)   | **FAIL (active regression)** | Investigate -- likely a leak or implementation bug, not a model finding. Block close-out until root cause is identified. |

PASS/MARGINAL/FAIL bands mirror the project convention (+25 / +10) used
across the seven prior same-data-peer experiments and the v13 spec.

## Risks and mitigations

- **R1: Top-15 feature diffs include the Vegas leak features.** PR 21
  regenerated pairwise_v4.csv with the leak-filtered Vegas features,
  but `vegas_avg_*` etc. are still in the v4 feature matrix as
  *non-leaky-version* features. Confirm the v4 feature ranking artifact
  is generated against the leak-filtered pipeline (post-PR 21). The
  prep task (Phase 0) re-runs v4 on the current main; no risk if the
  Phase 0 output goes through clean LOSO.
- **R2: 30-seed x 6-cell x 22-LOSO = ~3960 XGB fits.** ~1-2 hours wall
  time on this hardware. Each fit is small (~3000 rows, 100-200 trees).
  Acceptable; comparable to a single Optuna run for v4.
- **R3: Capacity may not need a bump even with 15 features.** v8 hparams
  (depth=3, n=100) trained successfully on v10/v10a (8 features) without
  issues; the LOSO pick between v8 and v10cap is the principled answer
  to whether 15 features change the optimum.
- **R4: Feature gain is gradient-boosting-class biased toward
  high-cardinality continuous features.** This biases toward
  KenPom/Vegas/Massey rating-scale numbers and against the
  binary/coach-count features. Acceptable for a v1 ranking; if v12
  PASSES, we re-test the cell with a permutation-importance ranking
  in a follow-up.
- **R5: Adding v4 feature diffs may make stage-2 a "near-duplicate of
  v4" -- the diff IS what v4 trained on.** Stage-2 already has
  `p_stage1`, which is the boosted nonlinear of the same diffs. The
  question this experiment answers is whether re-exposing the linear
  diffs alongside the boosted scalar adds anything. If FAIL, that's a
  legitimate finding -- v4 already extracted the diff signal.

## Test plan

### Phase 0 -- v4 feature ranking artifact (~10 min)

- Run `python -m src.enhanced_model_v3` end-to-end (or a thin wrapper
  that skips the Optuna step and pulls feature_importances from the
  final tuned XGB).
- Save `output/v4_feature_importance.csv` with columns
  `feature_name, gain, gain_rank` sorted by descending gain.
- Force-add.

### Phase 1 -- plumbing + anchor invariance (no compute)

- Extend `train_stage2_v10.py::FEATURE_SETS` with v12_n5/n10/n15.
- Extend `load_per_game_data` and `build_pairwise` to join v4 feature
  matrix and emit signed diffs.
- Unit test: `tests/test_train_stage2_v10/test_v12_plumbing.py`
  - top-N diff columns appear in the per-game frame
  - signed flip: label=0 row's diff column == -label=1 row's diff column
  - join key is `(Season, TeamID)`
  - no leak: v4 features for season Y come from the row of
    pairwise_v4.csv that was trained without season Y
- Anchor: `train_stage2_v10 --features v8 --seeds 42` still reproduces
  canonical pairwise_v8.csv byte-equal.

### Phase 2 -- run the 6-cell grid (~1-2 hours)

- One CLI invocation per cell, writing to
  `output/pairwise_v12_<feature_set>_<hparams>.csv`.
- Verify each cell's anchor (single-seed reproducibility on a rerun).
- Force-add all 6 outputs.

### Phase 3 -- LOSO pick + scoring

- Write `src/score_v12_loso_pick.py`:
  1. Load 6 cell pairwise frames.
  2. For each test season Y, score each cell on the 21 training seasons
     under the v13 toss-up blend (alpha=0.6, upper_edge=0.55) using the
     BlendEvaluator from `src/blend_v4_v8.py`. Pick the max cell.
  3. Concatenate per-season picks into `output/pairwise_v12.csv`.
  4. Run the v13-style blend on the picked frame; emit
     `output/pairwise_v12_blend.csv`; print 22-season total.

### Phase 4 -- verdict + ship-or-document

- Compare 22-season total to the decision matrix.
- PASS: update README, TODO.md, swap production frame to
  pairwise_v12_blend.csv.
- MARGINAL/FAIL: document, close the lane, leave v13 as production.

## Out-of-scope follow-ups

- Permutation-importance v4 ranking (R4 mitigation if v12 PASS).
- N > 15 sweep (only if LOSO picks N=15 in >50% of seasons).
- v12 + extended alpha grid -- re-search v13's blend alpha conditional
  on a wider stage-2 feature view (only if v12 MARGINAL with consistent
  alpha=0.6 selection).
- Replacing diffs with both diff and sum/min/max -- a richer per-pair
  feature transformation (only if v12 PASS).

## Pointers

- `src/train_stage2_v10.py` -- stage-2 ensemble + feature-set toggles
- `src/blend_v4_v8.py::BlendEvaluator` -- in-memory chalk-bracket scorer
- `src/score_v13_blend.py` -- v13 production blend reference
- `src/features/feature_matrix_v2.py::get_feature_cols` -- canonical
  v4 feature-column list
- `output/pairwise_v4.csv` -- v4 stage-1 (clean LOSO, post-PR 21)
- `output/pairwise_v13.csv` -- v13 baseline (2106 brkt pts)
- `output/pairwise_v8.csv` -- regenerated v8 baseline (2034 brkt pts,
  XGB 3.2.0, post-PR 38)
