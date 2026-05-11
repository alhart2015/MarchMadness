# GNN Phase 2 LOSO -- After Phase 1 Retraction (Continuation Instructions)

> **For agentic workers:** Use superpowers:subagent-driven-development to
> execute this plan task-by-task. STAY ON branch
> `feat/non-tabular-model-class-scoping`. Do NOT merge PR 35; do NOT create
> new branches or worktrees. The user will merge PR 35 once this Phase 2
> work supersedes the (invalid) Phase 1 verdict.

## Status when this file was written (2026-05-10)

- PR 35 (https://github.com/alhart2015/MarchMadness/pull/35) is OPEN with a
  FAIL verdict from Phase 1 that has been **invalidated** -- Massey leak +
  wrong-proxy critique. The user is keeping the PR open and will not merge
  until Phase 2 supersedes it.
- Branch HEAD as of 2026-05-09 was `4a2ce96` (Phase 1 FAIL findings + TODO).
  Phase 2 commits go on top of this on the same branch.
- Phase 1 implementation is preserved in-place. The retraction happens in
  the FINDINGS NOTE and the PR description, not by reverting commits.

## Why Phase 1 was invalidated

**1. Massey leak.** `src/gnn_stage1_peer/baselines.py::evaluate_massey_baseline`
defaults to `ranking_day=133` (Selection Sunday). Massey ratings at DayNum
133 already incorporate every regular-season game played that season --
including the March-1-to-Selection-Sunday games that were Phase 1's TEST
SET (DayNum 120-133). So when Phase 1 asked "predict game X using Massey,"
Massey had already updated on game X's outcome. The 0.10 LL "Massey wins"
gap was largely Massey looking up the answer.

**2. Wrong proxy.** Even leak-free, Massey is nearly tautological at RS-game
prediction (it's a sum of RS results predicting RS results), and a model
that can't beat Massey at being-Massey on regular-season games could still
plausibly win on tournament games (different distribution: neutral sites,
win-or-go-home stakes, much smaller per-season sample).

The user's call: skip the Phase 1 RS-prediction proxy entirely. Run the
full 22-season tournament LOSO that the spec originally defined as Phase 2.
Be thorough.

## Are the other 7 same-data-peer FAIL verdicts also tainted? NO

The 7 prior failures (BT-feature, feature-view, HBT, Colley, Massey-MOV,
Massey-decay-14d, team-seed-residual) are leak-safe and remain valid. The
Phase 1 leak does NOT apply to those experiments because:

1. **They were evaluated against v4, not against Massey directly.** v4's
   pairwise frame (`output/pairwise_v4.csv`, post PR 19/20/21 leak fix) is
   the canonical baseline for stage-1 peers. Each of the 7 candidates
   produced its own tournament pairwise frame and was compared to v4 via
   the BT-class LL-blend gate.
2. **They predicted TOURNAMENT games, not RS games.** Massey ratings at
   `ranking_day=133` (or any RankingDayNum, since Massey's data source is
   RS-only) are leak-safe with respect to tournament outcomes. The leak
   only manifests when predicting RS games using Massey at a ranking_day
   AFTER those RS games were played.
3. **Massey-MOV and Massey-decay-14d candidates used Massey as INPUT
   features** for their own stage-1 model, not as a comparison baseline.
   Their gate was vs v4 on tournament games. Leak-safe.

So the only result that needs retracting is GNN Phase 1. The seven-failure
pattern stands at seven; this Phase 2 work either makes it eight (FAIL) or
breaks the pattern (PASS / MARGINAL).

## Goal

Run the 22-season LOSO tournament-prediction experiment per the spec's
original Phase 2 design. Produce a leak-free verdict via the BT-class
LL-blend gate against v4, followed by v8 retrain + bracket-points re-test
if the LL-blend gate passes. Update PR 35 with the Phase 2 verdict.

## Architecture (from the spec, lines 235-267)

For each LOSO season S in {2003-2025} \ {2020}:

- **Training labels (supervised signal):** Tournament-game outcomes from
  all 21 seasons except S. ~67 tournament games per season x 21 seasons x
  2 orientations = ~2,800 symmetric training pairs.
- **Training input graphs:** Every season's RS-derived graph (~22 graphs
  total, each built from games with `DayNum < 134`). The graphs are model
  INPUT, not labels -- consumed at every training step and inference.
- **Test (held out):** S's tournament games (~67 x 2 = ~134 symmetric
  pairs).
- **Cross-season parameter sharing:** ONE GNN trained on all 21
  train-seasons' tournament outcomes; encoder + decoder weights are
  shared. Only the inputs (per-season graphs + global team-index lookup)
  differ per game. **This is different from Phase 1's per-season
  independent training.** Phase 2 has dramatically more supervision.
- **Output:** 22-season pairwise frame in the same shape as
  `output/pairwise_v4.csv`.

## Files to create

**Code (~600 LOC total):**
- `src/gnn_stage1_peer/loso.py` -- LOSO data pipeline + cross-season
  training loop + evaluator. Public functions:
  - `build_loso_training_data(data_dir, holdout_season) -> tuple` returns
    `(per_season_graphs, train_pairs_by_season, test_pairs, team_index)`.
  - `train_loso_gnn(per_season_graphs, train_pairs_by_season, val_pairs,
    *, epochs=50, lr=1e-3, patience=5, seed=42) -> nn.Module`.
  - `evaluate_loso(model, test_pairs, holdout_graph) -> dict` matching
    Phase 1 evaluator shape.
  - `run_phase2_one_holdout(data_dir, holdout_season, ...) -> dict`
    composes the three above.
- `src/run_gnn_phase2.py` -- CLI driver. Loops over 22 LOSO seasons,
  emits `output/pairwise_gnn_phase2.csv` + per-season + summary JSONs.

**Tests (~250 LOC):**
- `tests/test_gnn_stage1_peer/test_loso.py` -- unit tests for
  `build_loso_training_data` (global team index correctness, per-season
  graph leak-safety, train/test split per holdout); separable toy across
  multiple seasons for `train_loso_gnn`; smoke test for `evaluate_loso`.
- `tests/test_gnn_stage1_peer/test_phase2_smoke.py` (slow, real data,
  one-holdout) -- end-to-end on a single LOSO season.

**Outputs (force-added per CLAUDE.md):**
- `output/gnn_phase2_loso_run.log`
- `output/gnn_phase2_loso_per_holdout.json`
- `output/gnn_phase2_loso_summary.json`
- `output/pairwise_gnn_phase2.csv` (22-season tournament pairwise frame)
- `output/gnn_phase2_anchor_invariance.log`
- `output/cv_per_season_gnn_phase2_blend.csv` (LL-blend gate diagnostics)
- (If LL-blend passes:) `output/pairwise_v8_with_gnn_blend.csv`,
  `output/v8_retrain_gnn_phase2_run.log`,
  `output/gnn_phase2_bracket_points.json`

**Documentation:**
- `docs/notes/2026-05-10-gnn-phase2-loso.md` -- NEW findings note. Mirror
  the structure of `docs/notes/2026-05-09-team-seed-residual.md`.
- `docs/notes/2026-05-09-gnn-phase1.md` -- ADD a "RETRACTED 2026-05-10"
  header at the top pointing to the Phase 2 findings. Do NOT delete the
  Phase 1 note; preserve it as a record of the methodological mistake.

**Updates:**
- `TODO.md` -- replace the Phase 1 FAIL Done entry with the Phase 2
  verdict; if Phase 2 PASSes, update Active queue item #3 from
  "Candidate 4 promoted" back to "Phase 2 GNN promoted"; if FAIL, leave
  Candidate 4 in slot #3 and add a note that Phase 2 also failed.

## Tasks (suggested decomposition; subagent-driven)

### Task A: Refactor `data.py` for the global team-index needed by LOSO

The Phase 1 `build_team_index(games)` builds a per-season index. LOSO
needs a global index spanning all seasons (because the GNN's
`nn.Embedding(num_nodes, hidden_dim)` allocates one row per team across
the entire training set). Add `build_global_team_index(data_dir,
seasons) -> dict[int, int]` to `src/gnn_stage1_peer/data.py`. Keep the
per-season `build_team_index` for Phase 1 backward-compatibility (Phase 1
code still works as-is).

Tests: confirm global index covers every team appearing in any season's
RS games; contiguous; per-season indexing is unchanged.

### Task B: Build LOSO data pipeline

`src/gnn_stage1_peer/loso.py::build_loso_training_data(data_dir,
holdout_season)`:

- Load all-season RS games (`MRegularSeasonCompactResults.csv`, no
  filter), all-season tournament games
  (`MNCAATourneyCompactResults.csv`).
- Build the global team_index (Task A).
- For each season S, build a PyG graph from S's RS games (DayNum < 134;
  uses the global team_index). Cap memory: 22 graphs of ~5K bidirected
  edges each is fine; total ~110K edges across seasons.
- Build training pairs: for each season != holdout, all that season's
  tournament games with their season-tag, both orientations.
- Build test pairs: holdout season's tournament games.
- Return `(per_season_graphs: dict[int, Data], train_pairs_by_season:
  dict[int, tuple[Tensor, Tensor, Tensor]], test_pairs: tuple[Tensor,
  Tensor, Tensor], team_index: dict[int, int])`.

Tests: 4-5 unit tests covering shape + leak-safety + season tagging.

### Task C: Cross-season shared-parameter training loop

`src/gnn_stage1_peer/loso.py::train_loso_gnn(per_season_graphs,
train_pairs_by_season, val_pairs, val_graph, *, epochs=50, lr=1e-3,
patience=5, seed=42) -> tuple[nn.Module, dict]`:

- Instantiate ONE `GNNStage1Peer(num_nodes=len(team_index), ...)`.
- Per epoch: iterate over training seasons; for each season, forward
  through that season's graph (encoder produces per-team embeddings),
  then decode the season's tournament-game pairs, accumulate BCE loss.
  Single backward + step per epoch (or per-season-step if memory is a
  concern -- decide based on profiling).
- Validation: forward through `val_graph`, decode `val_pairs`, compute LL.
- Early stopping with patience on val LL.

Tests: separable toy across 3 fake seasons; verify loss decreases and the
model's val LL improves.

### Task D: LOSO evaluator + per-holdout driver

`src/gnn_stage1_peer/loso.py::evaluate_loso(model, test_pairs,
holdout_graph) -> dict`: returns `{ll, accuracy, n, predictions}` where
`predictions` carries `(team_a_idx, team_b_idx, p_a_wins, label)` per
pair (matches Phase 1's evaluator shape).

`src/gnn_stage1_peer/loso.py::run_phase2_one_holdout(data_dir,
holdout_season, ...) -> dict`: composes Tasks B/C/D. Returns
`{holdout_season, gnn_eval, train_minutes, epochs_run, best_epoch}`.

Tests: shape test for `evaluate_loso`; integration smoke for
`run_phase2_one_holdout` on a single fake season.

### Task E: 22-season LOSO sweep CLI driver

`src/run_gnn_phase2.py`:

- `--holdout-seasons` argument (default: `2003-2025` minus 2020 -> 22
  seasons).
- For each holdout: call `run_phase2_one_holdout`. Aggregate per-season
  results.
- **Crucially: emit `output/pairwise_gnn_phase2.csv` in the same shape as
  `output/pairwise_v4.csv`.** This is the artifact the LL-blend gate
  consumes. Look at `output/pairwise_v4.csv` columns first to match
  exactly. Each row is one pairwise prediction `(season, team_a_id,
  team_b_id, p_a_wins, ...)`. Symmetric (both A-vs-B and B-vs-A
  orientations).
- Write per-holdout JSON, summary JSON, run log.
- Force-add the 4 outputs.
- **Memory hygiene:** explicit `gc.collect()` between holdouts. Per the
  team-seed-residual findings, `MM_PAIRWISE_OUT` in
  `enhanced_model_v3.py` died on Windows for runs longer than ~6-20
  seasons. The pattern from
  `src/loso_with_pairwise_for_team_history.py` is the working template.
  If the 22-season loop crashes mid-run, fall back to running 2024 and
  2025 (or whichever seasons crashed) as separate one-off invocations
  from the command line.

Smoke test: real-data run on a single holdout season; assert outputs
exist and `train_minutes < 60.0`.

### Task F: Run the 22-season sweep

```bash
cd C:/Users/alden/MarchMadness && python -m src.run_gnn_phase2 \
    --holdout-seasons 2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2016,2017,2018,2019,2021,2022,2023,2024,2025 \
    2>&1 | tee output/gnn_phase2_console.log
```

Expected wall-clock: highly uncertain on CPU. Phase 1's per-season
training was <2 seconds (single-season independent training, no
cross-season supervision). Phase 2's cross-season loop is structurally
heavier (one model trained on ~2,800 pairs across 21 graphs per epoch).
Realistic estimate: 1-10 minutes per holdout, total 22 min - 4 hours. If
exceeds 8 hours, escalate (consider sample subsampling or GPU access).

### Task G: BT-class LL-blend gate

Once `output/pairwise_gnn_phase2.csv` exists, run the standard BT-class
gate against `output/pairwise_v4.csv`. Use the existing diagnostic
tooling pattern:

- Find the closest precedent (`src/diagnose_colley.py`,
  `src/diagnose_bt_vs_v4.py`, `src/diagnose_massey_mov.py`) and adapt.
- Compute: per-game disagreement correlation r (when GNN and v4 disagree
  on the picked side, GNN is right with probability p; r = 2p - 1 or the
  equivalent metric used in prior experiments -- match the BT-class
  convention exactly).
- Compute optimal blend weight `w_v4` via grid search or 1D optimization.
- Compute LL headroom: blended LL minus v4-standalone LL.
- **Gate clauses (all must pass):**
  1. `r >= 0.60`
  2. `w_v4 in [0.40, 0.85]` (non-degenerate)
  3. `headroom >= +0.005` LL
- Save diagnostics to
  `output/cv_per_season_gnn_phase2_blend.csv`.

### Task H: v8 retrain + bracket-points re-test (only if LL-blend passes)

If all three LL-blend clauses pass, retrain v8 stage-2 on the
v4+GNN-blended pairwise frame:

- Mirror PR 34's procedure exactly. The relevant modules are
  `src/enhanced_model_v3.py` (`MM_PAIRWISE_OUT` is the v8-retrain hook;
  use the team-seed-residual custom-driver pattern if it crashes).
- Output: `output/pairwise_v8_with_gnn_blend.csv`.
- Score: `python -c "from src.score_chalk_brackets import
  score_pairwise_path; print(score_pairwise_path('output/
  pairwise_v8_with_gnn_blend.csv')['total_pts'])"`.
- Compare to canonical 2069. **Bracket-points gate: delta >= +25 pts**
  over 22 LOSO seasons.

### Task I: Anchor invariance check

Drop-features control: replace the GNN's pairwise output with v4's, run
the v8 retrain on that frame, verify the resulting LL is within
`< 1e-3` max-abs-diff of canonical v4. Confirms the wire-in is
non-invasive and any drift in the with-GNN run reflects the GNN's
signal, not a wire-in defect. Force-add log to
`output/gnn_phase2_anchor_invariance.log`.

(Same pattern as `output/anchor_invariance_run.log` from PR 34.)

### Task J: Findings note + retraction header

1. **Add retraction header to `docs/notes/2026-05-09-gnn-phase1.md`.**
   Insert at line 1 (above the existing `# GNN Stage-1 Peer Phase 1...`):

   ```markdown
   > **RETRACTED 2026-05-10:** The Phase 1 FAIL verdict is invalid due
   > to a data leak in the Massey baseline (`evaluate_massey_baseline`
   > used `ranking_day=133` -- Selection Sunday Massey ratings already
   > incorporate the test-set RS games). Additionally, RS-prediction is
   > a structurally biased proxy for tournament prediction. The
   > superseding Phase 2 verdict is at
   > `docs/notes/2026-05-10-gnn-phase2-loso.md`.
   ```

2. **Write `docs/notes/2026-05-10-gnn-phase2-loso.md`.** Mirror
   `docs/notes/2026-05-09-team-seed-residual.md` structure: TL;DR,
   per-season detail, LL-blend gate result, bracket-points result (if
   applicable), what-this-means, methodological notes (including
   acknowledgment of the Phase 1 leak retraction), open questions, files
   of record. Be honest about what was tested vs assumed.

### Task K: TODO update

Open `TODO.md`. The current Done section starts with the GNN Phase 1
FAIL entry (`## Done` near line 575). Two cases:

- **Phase 2 PASS:** REMOVE the Phase 1 Done entry (it's retracted, not
  done). Insert a new Done entry: "GNN stage-1 peer Phase 2 LOSO -- PASS
  (2026-05-10)..." with verdict numbers. Update Active queue: the GNN is
  now a confirmed v4 stage-1 peer; demote/replace item #3 accordingly.
- **Phase 2 FAIL:** REMOVE the Phase 1 Done entry. Insert a new Done
  entry: "GNN stage-1 peer Phase 2 LOSO -- FAIL (2026-05-10), and Phase
  1 verdict retracted (had Massey leak)..." Update Active queue item #3
  to keep Candidate 4 promoted (the spec's sequel-ordering matrix still
  applies on FAIL).

In either case, the Phase 1 Done entry must NOT remain -- it captures
an invalid verdict.

### Task L: Update PR 35 description

After the findings note + TODO are committed:

```bash
gh pr edit 35 --title "GNN stage-1 peer Phase 2 LOSO -- <PASS|FAIL> (<delta details>)"
gh pr edit 35 --body "$(cat <<'EOF'
... <updated PR description with Phase 2 verdict, retraction note, full per-season detail> ...
EOF
)"
```

The new PR description should:
- Lead with the Phase 2 verdict.
- Have a clearly-marked "Phase 1 retraction" section explaining the
  Massey leak.
- Reference the new findings note `docs/notes/2026-05-10-...`.
- Carry the full per-season detail (mirror PR 34's PR description style).

## Procedural anchors

1. **Stay on `feat/non-tabular-model-class-scoping`.** PR 35 tracks this
   branch. Do NOT create new branches or worktrees.
2. **Do NOT merge PR 35.** The user will merge once Phase 2 is
   complete.
3. **Force-add ALL output data** (per CLAUDE.md). Mirror PR 34's
   `git add -f` pattern.
4. **ASCII only** (Windows cp1252 console safety per CLAUDE.md).
5. **Determinism.** Set torch / numpy / python seeds at the start of
   every LOSO holdout. Document seed in summary JSON (default 42).
6. **MM_PAIRWISE_OUT instability.** Per PR 34's findings, the v3 LOSO
   loop crashes on Windows for runs longer than ~6-20 seasons. Carry
   forward the explicit-`gc.collect()`-between-holdouts pattern from
   `src/loso_with_pairwise_for_team_history.py`. If the run crashes
   mid-sweep, fall back to running the missing seasons as separate
   one-off Python invocations -- structurally identical, just different
   process boundaries.
7. **No shortcuts on the gate.** If LL-blend or bracket-points fails,
   document and FAIL. Do NOT tune to make it pass post-hoc. The
   integrity of the seven-failure-pattern (and any continuation thereof)
   depends on this discipline.
8. **Honor the spec's sequel-ordering matrix on FAIL** (spec lines
   367-381). If Phase 2 FAILs, the GNN candidate is closed; promote
   Candidate 4 in TODO Active queue item #3.

## Cost estimate

| Task | Estimate |
|---|---|
| A: Global team index helper | 30 min |
| B: LOSO data pipeline | 1 day |
| C: Cross-season training loop | 1-2 days |
| D: Evaluator + per-holdout driver | 4 hours |
| E: CLI driver + pairwise emission | 1 day |
| F: 22-season sweep wall-clock | 22 min - 4 hours (uncertain) |
| G: LL-blend gate (reuse tooling) | half day |
| H: v8 retrain + bracket-points (only if LL-blend passes) | half day |
| I: Anchor invariance | 2 hours |
| J: Findings note + retraction header | 2 hours |
| K: TODO update | 30 min |
| L: PR description update | 30 min |
| **Total** | **~5-7 person-days** + 22 min - 4 hours wall-clock |

## Decision matrix on Phase 2 verdict

| Outcome | LL-blend r | LL-blend headroom | Bracket pts | Action |
|---|---|---|---|---|
| **PASS** | >= 0.60 | >= +0.005 | >= +25 | Update PR title to PASS; promote GNN to confirmed v4 stage-1 peer; spec next-step (multi-peer ensemble?). Phase 1 retracted. |
| **LL-blend PASS, BP FAIL** | >= 0.60 | >= +0.005 | < +25 | Eighth same-data-equivalent failure, but tournament-tested (no proxy critique). Update PR to FAIL. Phase 1 retracted; sequel re-rank applies (Candidate 4 promoted). |
| **LL-blend FAIL clause 1 (r < 0.60)** | < 0.60 | -- | (skip) | GNN is same-data peer in disguise. FAIL. Phase 1 retracted; Candidate 4 promoted. |
| **LL-blend FAIL clause 3 (headroom < +0.005)** | -- | < +0.005 | (skip) | GNN doesn't add information at tournament level. FAIL. Phase 1 retracted; Candidate 4 promoted. |
| **MARGINAL (headroom +0.000 to +0.005)** | >= 0.60 | 0..+0.005 | -- | Document; consider one hyperparameter sweep (hidden_dim 32 vs 128, edge-attr-aware encoder, lr 1e-3 vs 5e-4) before final verdict. If still MARGINAL, treat as FAIL. |

In every case, the Phase 1 retraction header goes on
`docs/notes/2026-05-09-gnn-phase1.md` regardless of Phase 2 outcome.

## Definition of done

- [ ] All tests pass: `python -m pytest tests/test_gnn_stage1_peer/ -v`.
- [ ] 22-season pairwise frame `output/pairwise_gnn_phase2.csv`
      force-added.
- [ ] Per-season + summary JSONs and run log force-added.
- [ ] LL-blend gate diagnostics force-added.
- [ ] (If LL-blend passes:) v8 retrain log + bracket-points JSON
      force-added.
- [ ] Anchor invariance log force-added.
- [ ] Findings note at `docs/notes/2026-05-10-gnn-phase2-loso.md` with
      verdict.
- [ ] `docs/notes/2026-05-09-gnn-phase1.md` carries a RETRACTED header.
- [ ] `TODO.md` updated (Phase 1 entry replaced; Active queue corrected
      per verdict).
- [ ] PR 35 title + description updated via `gh pr edit 35`.
- [ ] All commits on `feat/non-tabular-model-class-scoping` branch.
- [ ] ASCII verified for all touched files.
- [ ] User notified that Phase 2 is complete and PR 35 is ready for
      review.

## Files of record (continuation context)

- This instructions file:
  `docs/superpowers/plans/2026-05-10-gnn-phase2-loso-after-phase1-retraction.md`
- Original spec (defines Phase 2 architecture):
  `docs/superpowers/specs/2026-05-09-non-tabular-model-class-scoping-design.md`
- Phase 1 plan (preserved as-is; Phase 1 verdict retracted):
  `docs/superpowers/plans/2026-05-09-non-tabular-model-class-scoping-phase1.md`
- Phase 1 findings note (will get RETRACTED header in Task J):
  `docs/notes/2026-05-09-gnn-phase1.md`
- Predecessor template for v8-retrain + force-add patterns:
  `docs/notes/2026-05-09-team-seed-residual.md`
- Open PR: https://github.com/alhart2015/MarchMadness/pull/35
