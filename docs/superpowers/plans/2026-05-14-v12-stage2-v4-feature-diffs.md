# v12 Stage-2 Enrichment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add top-N v4 feature *differences* to the v13 stage-2 architecture (30-seed XGB ensemble + toss-up bucket blend at alpha=0.6). LOSO-pick the (N, hparams) cell per season over a 6-cell grid (`N in {5, 10, 15}` x `hparams in {v8, v10cap}`). Score the picked-cell frame under the v13 toss-up blend and emit a verdict per the spec's decision matrix.

**Architecture:** Extend `src/train_stage2_v10.py::FEATURE_SETS` with `v12_n5 / v12_n10 / v12_n15`. Extend `load_per_game_data` and `build_pairwise` to join v4's per-team-per-season feature matrix and emit signed diffs `feat_team_a - feat_team_b`. Reuse `src/blend_v4_v8.py::BlendEvaluator` for all scoring. New `src/score_v12_loso_pick.py` does the per-season cell pick. Anchor: `train_stage2_v10 --features v8 --seeds 42` must still byte-equal canonical `pairwise_v8.csv`.

**Tech Stack:** Python, pandas, numpy, xgboost (already used throughout), pytest. Inputs: `output/pairwise_v4.csv` (clean LOSO post-PR 21), `output/pairwise_v13.csv` (v13 baseline 2106), v4 feature matrix (from `enhanced_model_v3.py`'s `compute_all_features`). Existing pieces reused: `BlendEvaluator`, `score_chalk_brackets`, `train_stage2_v10` feature plumbing.

**Spec:** `docs/superpowers/specs/2026-05-14-v12-stage2-v4-feature-diffs-design.md`
**Predecessors:**
- v13 PASS (PR #37): 2106 brkt pts baseline.
- XGB env drift cleanup (PR #38): canonical pairwise_v8 regenerated; production frame is pairwise_v13.csv.

> **PASS bar:** 22-season blended total >= 2131 (+25 vs v13).
> **FAIL null band:** 2089..2116 (-17..+10). Close the lane.
> **FAIL active regression:** < 2089. Investigate before close.

---

## File Structure

**Created (committed):**

- `src/extract_v4_feature_importance.py` (~80 LOC, Phase 0 prep)
  - CLI: `python -m src.extract_v4_feature_importance --out output/v4_feature_importance.csv`
  - Builds the v4 feature matrix and trains a single XGB on the pooled 22-season tournament data using cached tuned params (`output/v4_tuned_params.json`). Pulls `feature_importances_` (gain), writes a sorted CSV.
- `src/score_v12_loso_pick.py` (~150 LOC, Phase 3 driver)
  - CLI: `python -m src.score_v12_loso_pick --cells <comma-list> --out output/pairwise_v12.csv`
  - Loads each cell's pairwise CSV, runs the per-season LOSO pick (training-season scores under v13 toss-up blend at alpha=0.6), concatenates picked rows into `output/pairwise_v12.csv`, then writes `output/pairwise_v12_blend.csv` (v13-style blended frame) and prints the 22-season total + verdict.
- `tests/test_train_stage2_v10/test_v12_plumbing.py` (~120 LOC, ~6 tests)
  - Unit tests for the new v12 feature plumbing (diff sign flip, join key, leak safety, anchor invariance to v13 single-seed).
- `tests/test_score_v12_loso_pick.py` (~80 LOC, ~3 tests)
  - Tests for the LOSO cell-picker logic on synthetic 2-cell inputs.

**Modified:**

- `src/train_stage2_v10.py`
  - `FEATURE_SETS`: add `v12_n5`, `v12_n10`, `v12_n15`.
  - `load_per_game_data`: accept a `v4_feature_diff_df` arg and emit `diff_<feat>` columns.
  - `build_pairwise`: same -- emit `diff_<feat>` per row of the season pairwise frame.
  - `main`: add `--v4-feature-diffs` arg pointing at the FM parquet or CSV; if set, load and pass through.

**Generated (force-added per `.gitignore: output/`):**

- `output/v4_feature_importance.csv` (Phase 0 output)
- `output/v4_feature_matrix.parquet` (Phase 0 output, the feature matrix snapshot v12 stage-2 joins against)
- `output/pairwise_v12_n5_v8.csv`, `..._n5_v10cap.csv`, `..._n10_v8.csv`, `..._n10_v10cap.csv`, `..._n15_v8.csv`, `..._n15_v10cap.csv` (Phase 2 outputs, 6 files)
- `output/pairwise_v12.csv` (Phase 3 picked-cell frame)
- `output/pairwise_v12_blend.csv` (Phase 3 v13-style blended frame -- the candidate production output)
- `output/v12_loso_pick_summary.json` (per-season pick + cell scores)

---

## Phase 0: v4 feature ranking artifact

### Task 1: Build `src/extract_v4_feature_importance.py` and run it

**Files:**
- Create: `src/extract_v4_feature_importance.py`

- [ ] **Step 1: Spike the v4 fit + importance extraction.** First, run `python -m src.enhanced_model_v3` with `MM_SKIP_DEFAULT_LOSO=1` to produce `output/v4_tuned_params.json` and warm the feature-matrix caches. Estimated runtime ~5 min. Verify `output/v4_tuned_params.json` exists.

- [ ] **Step 2: Write the extractor.** Implement `src/extract_v4_feature_importance.py`:
  1. Load tuned params from `output/v4_tuned_params.json`.
  2. Re-run `prepare_loso_inputs()` to get `feature_matrix`, `feature_cols`, `X_all`, `y_all`, `weights_all` (the same pooled-22-season training set used by the LOSO inner loop).
  3. Fit one XGB with the tuned params on `(X_all, y_all, sample_weight=weights_all)`.
  4. Pull `model.feature_importances_` (sklearn API; importance_type defaults to "gain"). Pair with `feature_cols`, sort descending.
  5. Write `output/v4_feature_importance.csv` with columns `feature_name, gain, gain_rank` (rank starts at 1).
  6. Also snapshot the feature matrix to `output/v4_feature_matrix.parquet` -- the per-(Season, TeamID, feature) frame that v12 stage-2 will join against. This must use the same filled-NaN frame the v4 model trained on (fill via column median; see line 985 of `enhanced_model_v3.py`).

- [ ] **Step 3: Run the extractor.** `python -m src.extract_v4_feature_importance --out output/v4_feature_importance.csv --matrix-out output/v4_feature_matrix.parquet`. Verify:
  - CSV has 67 rows (the v4 feature count).
  - Top 5 should include at least one rating-scale efficiency feature (kp_AdjEM, kp_AdjOE, kp_AdjDE, or a Vegas power feature). If the top 5 are all binary/coach-count features, something is wrong -- inspect.
  - Parquet has one row per (Season, TeamID) with all 67 feature columns; row count ~5000 across 22 seasons.

- [ ] **Step 4: Force-add both artifacts.** `git add -f output/v4_feature_importance.csv output/v4_feature_matrix.parquet`. Commit with message `feat(v12): extract v4 feature ranking + per-team feature matrix snapshot`.

**Decision gate after Phase 0:** Print the top-15 ranked features. If the ranking looks pathological (e.g., dominated by features the v4 model shouldn't lean on), stop and revisit before Phase 1.

---

## Phase 1: Plumbing + anchor invariance

### Task 2: Extend `train_stage2_v10.py` with v12 feature sets and v4-diff joining

**Files:**
- Modify: `src/train_stage2_v10.py`
- Create: `tests/test_train_stage2_v10/__init__.py`
- Create: `tests/test_train_stage2_v10/test_v12_plumbing.py`

- [ ] **Step 1: Write the failing test for v8 anchor invariance under the new plumbing.** Before any code change to `train_stage2_v10`, write the anchor test that v8 single-seed still byte-equals canonical pairwise_v8.csv. This guards against the new code path leaking into the v8 case.

```python
# tests/test_train_stage2_v10/test_v12_plumbing.py
def test_v8_anchor_unchanged_after_v12_plumbing(tmp_path):
    """v8 feature set + single seed reproduces canonical pairwise_v8.csv byte-equal,
    even after v12 plumbing is added (no v4-diff loading on the v8 code path)."""
    from src.train_stage2_v10 import main
    out = tmp_path / "v8_anchor.csv"
    main([
        "--features", "v8", "--seeds", "42",
        "--pairwise-out", str(out),
    ])
    canonical = pd.read_csv("output/pairwise_v8.csv")
    rerun = pd.read_csv(out)
    pd.testing.assert_frame_equal(canonical, rerun)
```

- [ ] **Step 2: Add `v12_n5`, `v12_n10`, `v12_n15` to `FEATURE_SETS`.** These are dynamic -- they depend on the top-N from `v4_feature_importance.csv`. Add a helper `_v12_feature_cols(n: int) -> list[str]` that reads the ranking CSV once at import time (or first call) and returns `_V8_BASE + ["expected_round"] + [f"diff_{name}" for name in top_n_names]`. Cache.

- [ ] **Step 3: Extend `load_per_game_data` to accept an optional `v4_feature_matrix_df`.** When provided, join on `(season, w)` to get team_w's features, on `(season, l)` for team_l, and emit `diff_<feat>` columns. Critical: the label=1 row gets `feat_w - feat_l`; the label=0 row gets `feat_l - feat_w`. This is the symmetric signed-diff pattern -- mirrors `matchup.py`.

- [ ] **Step 4: Extend `build_pairwise` similarly.** For each `(season, team_a, team_b)` row in the pairwise frame, look up both teams' features and emit signed diffs `feat_a - feat_b`. team_a < team_b is the canonical orientation.

- [ ] **Step 5: Extend `main` with `--v4-feature-matrix output/v4_feature_matrix.parquet` arg.** When `--features` starts with `v12_`, this arg is required. When `--features v8` (the v8 anchor), the v4-diff loading code path is skipped entirely -- no `feat_a - feat_b` join, no behavior change vs current v8.

- [ ] **Step 6: Run the anchor test (Step 1).** Must pass before adding v12-specific tests.

- [ ] **Step 7: Add v12-specific tests:**

```python
def test_diff_sign_flip_on_symmetric_pair():
    """Label=0 row's diff_<feat> == -1 * label=1 row's diff_<feat>."""
    # Build a 1-game synthetic per-game frame, verify the W-perspective and
    # L-perspective rows are exact negations on every diff column.

def test_join_key_is_season_teamid():
    """v4 feature lookup uses (Season, TeamID), not team names. Smoke test
    on real feature matrix; assert no NaN diffs after lookup."""

def test_no_leak_v4_feat_for_season_y_is_loso_out_of_fold():
    """v4 features for season Y come from the FM row that was trained
    on data excluding Y. Verify by inspecting a snapshot of feature_matrix_v3:
    the row's `Season` field matches Y, and there's no in-season tournament
    info baked in (cf. PR 19/20 leak fix)."""
    # Sanity check on column names: assert "tournament_*" / "ROUND" not in
    # the diff columns. This catches a regression where someone re-adds a
    # leaky column to the v4 feature set.

def test_v12_pairwise_anchor_n0_reduces_to_v13():
    """With no v4 diffs (N=0), the v12 code path with v10a feature set
    (v8_base + expected_round) reproduces v10a single-seed. v13 itself
    is the n=0 ensemble equivalent; we anchor on the simpler v10a single
    seed for byte-equal here."""

def test_v12_n5_diff_columns_count():
    """build_pairwise for v12_n5 produces a frame with the same row count
    as pairwise_v4.csv, with p_a_wins in [0, 1]."""
```

- [ ] **Step 8: Run all the new tests + the existing test_train_stage2_v10 suite.** All must pass.

**Decision gate after Phase 1:** Anchor invariance to v8/v10a verified, v12 plumbing produces valid output, no leak in diff columns. Commit with message `feat(v12): plumbing for top-N v4 feature diffs in stage-2`.

---

## Phase 2: Run the 6-cell grid

### Task 3: Produce the 6 per-cell pairwise outputs

**Files:**
- (No new files; run existing `src/train_stage2_v10.py` CLI 6 times.)

- [ ] **Step 1: Build the 30-seed list.** The same seeds v13 used: `42, 142, 242, ..., 2942` (step 100). Stash in a shell variable.

- [ ] **Step 2: Run cells 1-3 (N x v8 hparams).** Three commands, each ~10-20 min wall:

```
python -m src.train_stage2_v10 --features v12_n5  --hparams v8     --seeds 42,142,...,2942 --v4-feature-matrix output/v4_feature_matrix.parquet --pairwise-out output/pairwise_v12_n5_v8.csv
python -m src.train_stage2_v10 --features v12_n10 --hparams v8     --seeds 42,142,...,2942 --v4-feature-matrix output/v4_feature_matrix.parquet --pairwise-out output/pairwise_v12_n10_v8.csv
python -m src.train_stage2_v10 --features v12_n15 --hparams v8     --seeds 42,142,...,2942 --v4-feature-matrix output/v4_feature_matrix.parquet --pairwise-out output/pairwise_v12_n15_v8.csv
```

- [ ] **Step 3: Run cells 4-6 (N x v10cap hparams).** Three commands, each ~20-40 min wall (v10cap is 2x deeper, 2x trees):

```
python -m src.train_stage2_v10 --features v12_n5  --hparams v10cap --seeds 42,142,...,2942 --v4-feature-matrix output/v4_feature_matrix.parquet --pairwise-out output/pairwise_v12_n5_v10cap.csv
python -m src.train_stage2_v10 --features v12_n10 --hparams v10cap --seeds 42,142,...,2942 --v4-feature-matrix output/v4_feature_matrix.parquet --pairwise-out output/pairwise_v12_n10_v10cap.csv
python -m src.train_stage2_v10 --features v12_n15 --hparams v10cap --seeds 42,142,...,2942 --v4-feature-matrix output/v4_feature_matrix.parquet --pairwise-out output/pairwise_v12_n15_v10cap.csv
```

- [ ] **Step 4: Quick sanity-score each cell.** For each of the 6 frames, run a single-blend score (alpha=0.6) and print 22-season total. Document the per-cell totals in a row of the running log. Expectation: at least one cell should land near 2106 (v13's baseline) -- if all 6 are far below, suspect a plumbing bug.

- [ ] **Step 5: Force-add all 6 outputs.** `git add -f output/pairwise_v12_n*.csv`. Commit `feat(v12): 6-cell stage-2 grid outputs (N in {5,10,15} x hparams in {v8, v10cap})`.

**Decision gate after Phase 2:** All 6 cells produced valid pairwise CSVs (48,465 rows each, p_a_wins in [0, 1]). Smoke totals roughly span v8-baseline-to-v13-baseline range.

---

## Phase 3: LOSO pick + scoring

### Task 4: Build `src/score_v12_loso_pick.py`

**Files:**
- Create: `src/score_v12_loso_pick.py`
- Create: `tests/test_score_v12_loso_pick.py`

- [ ] **Step 1: Write failing tests for the cell-picker.**

```python
# tests/test_score_v12_loso_pick.py
def test_picker_prefers_higher_training_season_total(tmp_path):
    """Synthetic 2-cell input: cell A scores 100 in training seasons,
    cell B scores 50. Picker should select A for the held-out test season."""

def test_picker_returns_per_season_winners(tmp_path):
    """Output frame contains exactly one cell-source per (season, team_a, team_b)."""

def test_picker_falls_back_when_cell_missing_season(tmp_path):
    """If a cell has no rows for a given season (shouldn't happen with our
    pipeline but defensive), picker uses the next-best cell."""
```

- [ ] **Step 2: Implement the picker.**

```python
# src/score_v12_loso_pick.py outline
def main(argv=None):
    args = parse(argv)
    cell_paths = args.cells  # 6 paths
    cell_frames = {name: pd.read_csv(p) for name, p in zip(cell_names, cell_paths)}
    v4 = pd.read_csv("output/pairwise_v4.csv").drop_duplicates(...)
    ev = BlendEvaluator()
    seasons = sorted(cell_frames[cell_names[0]]["season"].unique())
    picks = {}
    cell_totals_per_season = {name: {} for name in cell_names}
    for cell_name, frame in cell_frames.items():
        blend = make_blend(frame, v4, toss_up_alpha=0.6, toss_up_upper_edge=0.55)
        per_season_pts = ev.score_probs_df(blend)
        for s, pts in per_season_pts.items():
            cell_totals_per_season[cell_name][s] = pts
    for test_season in seasons:
        train_seasons = [s for s in seasons if s != test_season]
        cell_train_totals = {
            name: sum(cell_totals_per_season[name][s] for s in train_seasons)
            for name in cell_names
        }
        picks[test_season] = max(cell_train_totals, key=cell_train_totals.get)
    # Concatenate picked-cell rows per season
    picked_rows = pd.concat([
        cell_frames[picks[s]][cell_frames[picks[s]]["season"] == s]
        for s in seasons
    ])
    picked_rows.to_csv(args.out, index=False)
    # Re-blend the picked frame and score
    blend = make_blend(picked_rows, v4, ...)
    blend.to_csv(args.blend_out, index=False)
    total = sum(ev.score_probs_df(blend).values())
    summary = {"picks": picks, "cell_totals_per_season": cell_totals_per_season,
               "v12_total": total, "v13_baseline": 2106, "delta_vs_v13": total - 2106}
    Path(args.summary_out).write_text(json.dumps(summary, indent=2))
    print(f"v12 picked-cell total: {total:.0f} brkt pts (delta vs v13 = {total - 2106:+.0f})")
    return total
```

- [ ] **Step 3: Run all tests.** Picker tests must pass.

- [ ] **Step 4: Run the picker on the 6 real cells.**

```
python -m src.score_v12_loso_pick \
  --cells output/pairwise_v12_n5_v8.csv,output/pairwise_v12_n5_v10cap.csv,output/pairwise_v12_n10_v8.csv,output/pairwise_v12_n10_v10cap.csv,output/pairwise_v12_n15_v8.csv,output/pairwise_v12_n15_v10cap.csv \
  --out output/pairwise_v12.csv \
  --blend-out output/pairwise_v12_blend.csv \
  --summary-out output/v12_loso_pick_summary.json
```

- [ ] **Step 5: Inspect `output/v12_loso_pick_summary.json`.** Two things to check:
  - **Cell pick distribution.** If 22/22 seasons pick the same cell, the LOSO grid was over-specified for the signal -- the +25 PASS bar is still the right gate, but the LOSO discipline didn't earn its keep. Document.
  - **Per-cell training-season totals.** Are the 6 cells visibly differentiated, or are their training-season totals within ~5 brkt pts of each other? If the latter, signal is noise-class -- LOSO picks may be unstable.

- [ ] **Step 6: Force-add the outputs.** `git add -f output/pairwise_v12.csv output/pairwise_v12_blend.csv output/v12_loso_pick_summary.json`. Commit `feat(v12): LOSO-picked cell frame + v13-blended output + per-season pick summary`.

---

## Phase 4: Verdict + ship-or-document

### Task 5: Apply the decision matrix and update docs

**Files:**
- Modify: `TODO.md` (add a top-of-file Done or Active entry with the verdict)
- Modify (if PASS): `README.md` (swap production frame reference to pairwise_v12_blend.csv)
- Create (if FAIL or MARGINAL): `docs/notes/2026-05-14-v12-stage2-v4-feature-diffs.md` (findings)

- [ ] **Step 1: Read the total from `output/v12_loso_pick_summary.json`.** Apply the decision matrix:

| 22-season blended total | Verdict |
|-------------------------|---------|
| >= 2131 (+25 vs v13)    | **PASS** |
| 2117..2130 (+11..+24)   | **MARGINAL** |
| 2089..2116 (-17..+10)   | **FAIL (null)** |
| < 2089 (worse by >17)   | **FAIL (active regression)** |

- [ ] **Step 2: PASS path.**
  - Update `README.md` Model Evolution table to add a v12 row.
  - Add Done entry to top of `TODO.md` (similar to v13's 2026-05-14 entry).
  - Note in TODO that `pairwise_v12_blend.csv` is now the production frame (`pairwise_v13.csv` becomes legacy).
  - Consider whether `score_v13_blend.py` defaults should point at v12. Discuss with user before changing.

- [ ] **Step 3: MARGINAL path.**
  - Write `docs/notes/2026-05-14-v12-stage2-v4-feature-diffs.md` capturing per-cell totals, picked-cell distribution, the final total, and verdict.
  - Update TODO.md with a Done entry. Note any candidate next-steps (extended N grid, permutation importance ranking).
  - Do NOT swap production frame; v13 stays.

- [ ] **Step 4: FAIL path.**
  - Write the findings note (same structure as MARGINAL).
  - Update TODO.md with a Done entry. Add to the "Tried and rejected" section.
  - Close the v12 lane. Active queue moves to #1 (roster) or #2 (pool-aware) next.

- [ ] **Step 5: PR open + final pytest gate.**
  - Run `pytest -v --ignore=tests/test_gnn_stage1_peer` -- all tests must pass (modulo the known colley-cache flake).
  - ASCII-clean check on all edited/created files.
  - Open the PR with `gh pr create`. Title: `feat(v12): stage-2 enrichment with top-N v4 feature diffs ({PASS|MARGINAL|FAIL})`.

---

## Risks during execution

- **R1: enhanced_model_v3 runtime in Phase 0.** ~5-10 min for the v4 fit. Acceptable but if Optuna re-runs (no cached `output/v4_tuned_params.json`), it's 20-30 min. The plan's Step 1 builds the cached params first.
- **R2: 30-seed x 6-cell run blows out runtime.** Estimated 1-2 hours. If it exceeds 4 hours, suspect XGB hyperparameter pathology -- bail out to a single-seed sweep of the same grid (~12 min total) to spot-check the architecture before paying the full 30-seed cost.
- **R3: Anchor test fails after Phase 1 Step 2-5.** Means the new plumbing leaked into the v8 code path. Revert the per-step diff, isolate the culprit. Do NOT proceed to Phase 2 until v8 anchor is byte-equal.
- **R4: All 6 cells score within 3 brkt pts of each other.** LOSO discipline picks one but the verdict is dominated by noise. Treat as MARGINAL even if the headline total clears +25 -- compute the cell-total spread and document.
- **R5: One cell is far below the others (e.g. n=15 v8 with 200 trees on 15 features).** Could indicate over- or under-fit. Inspect the per-season LL deltas in the train_stage2_v10 console output for that cell before drawing conclusions.

## Pointers

- Spec: `docs/superpowers/specs/2026-05-14-v12-stage2-v4-feature-diffs-design.md`
- v13 reference plumbing: `src/score_v13_blend.py`, `src/blend_v4_v8.py`
- Stage-2 trainer: `src/train_stage2_v10.py`
- v4 LOSO entry point: `src/enhanced_model_v3.py::prepare_loso_inputs`
- Canonical feature column list: `src/features/feature_matrix_v2.py::get_feature_cols`
- v4 stage-1 frame (post-leak-fix): `output/pairwise_v4.csv`
- v13 baseline: `output/pairwise_v13.csv` (2106 brkt pts)
- v8 same-env baseline (post XGB drift cleanup): `output/pairwise_v8.csv` (2034 brkt pts)
