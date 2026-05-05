# v9-C Clean Re-run (Step 5 PR 1) -- Design

**Date:** 2026-05-04
**Branch:** feat/v9c-clean-rerun
**Recovery context:** TODO.md "CONTAMINATION DISCOVERED 2026-05-04" -> step 5, item 1
**Predecessors:**
- v9-C feature-stripped findings: `docs/notes/2026-05-01-v9c-feature-stripped.md`
- v9-C production swap design: `docs/superpowers/specs/2026-05-01-v9c-production-swap-design.md`
- Vegas leak fix: `docs/superpowers/specs/2026-05-04-v4-vegas-leak-fix-design.md`
- Clean LOSO regen findings: `docs/notes/2026-05-04-v4-clean-loso-regen.md`
- Vegas audit rerun findings: `docs/notes/2026-05-04-v4-gap-audit-vegas.md`

## Motivation

v9-C is the live-production stage-2 corrector (`output/pairwise_probs.json`
was overwritten by `predict_2026_v9c.py` on 2026-05-01). Its swap-in
evidence chain rested on +43 brkt pts vs v8 over the 22-season LOSO
backtest, with v4 used as stage-1. After PR 19 fixed the Vegas leak in
v4's feature pipeline and PR 21 regenerated `pairwise_v4.csv` under the
clean pipeline, the v9-C swap-in numbers no longer reflect the current
stage-1 input.

PR 22 (audit rerun) sharpened the concern: v4's apparent upset-detection
edge over Vegas (56% vs 17% in the original audit) was the leak speaking.
Clean v4 catches 15.3% of upsets vs Vegas's 17.5%. v9-C's stage-2 is
upset-aware -- training weight `W_UPSET=1.25` directly amplifies the
upset signal v4 was claimed to have. If that signal was a leak artifact,
v9-C may have been correcting noise rather than signal, and the +43 vs v8
verdict could collapse or invert under the clean baseline. This PR
measures the actual delta and acts on it.

This is the first PR of recovery step 5 ("Re-run the swap-decided /
swap-candidate evaluations against the clean baseline"), and the
named top priority within step 5: **v9-C production swap (currently
deployed)**.

## Scope

**In scope.**

- Re-run `python src/train_stage2.py` on the clean `output/pairwise_v4.csv`
  to regenerate `output/pairwise_v8.csv` (clean v8 OOF stage-2
  corrector). Force-add per the existing canonical-artifact pattern.
- Re-run `V9_FEATURE_SET=v9c python src/sweep_v9_weights.py` to do the
  full 15-cell weight sweep (W_UPSET in {1.0, 1.25, 1.5, 1.75, 2.0} x
  W_MISS in {0, 0.5, 1.0}) against the clean v8 baseline. Anchor cell
  (W_U=1.0, W_M=0.0) is the v8-reproduction sanity gate; sweep driver
  already prints WARNING if `abs(delta) > 5.0` brkt pts and continues.
- Re-run the v9-C winning cell's bracket-points number using the
  per-cell pairwise CSV the sweep emits (`output/v9c_sweep/`).
- Add `src/v9c_per_season_breakdown.py` -- small post-processing script
  that reads `output/pairwise_v8.csv` + the v9-C winning cell's per-cell
  pairwise CSV, scores each season individually via
  `score_pairwise_path` from `src.score_chalk_brackets`, emits
  `output/v9c_clean_per_season.csv` with per-season `season`, `v8_pts`,
  `v9c_pts`, `delta`. The existing sweep driver only emits 22-season
  totals; the findings note needs the per-season W/L spread (matches
  PR 9's profile reporting).
- Apply the decision matrix below: if the v9-C best cell's delta vs
  clean v8 is `<= 0`, revert by re-running
  `python src/predict_2026_stage2.py` to restore
  `output/pairwise_probs.json` to v8-corrected output, and commit the
  restored canonical artifact.
- Force-add `output/pairwise_v9.csv` (overwrite with v9-C winning cell's
  per-cell CSV under clean v4) per PR 9's canonical-artifact pattern.
- Findings note `docs/notes/2026-05-04-v9c-clean-rerun.md` mirroring
  PR 22 / PR 21 structure: verdict TL;DR, methods + paths, clean v8
  baseline number, 15-cell results table sorted by delta, per-season
  W/L for the winning cell, anchor sanity check, discussion linking
  result to PR 22's clean upset-detection numbers, TODO.md update
  justification, follow-ups.
- TODO.md update: mark step 5 item 1 done; **expand the
  "marginal rejections" follow-up list** to include experiments not
  named in the original recovery roadmap whose rejections were within
  the leak's noise floor (+0.122 LL): plain BT standalone (PR 12),
  feature-view ensemble PEER_A/B (PR 14), HBT (PR 16), Colley (PR 15),
  Massey-decay hl=14d (PR 15). The roadmap already named BT-as-feature
  (PR 13) and v9 weight-sweep family (PRs 7-9) as marginal; this PR
  audits the rest.

**Out of scope.**

- The plain BT standalone re-eval (recovery step 5 item 3 in the
  roadmap; would deserve its own PR even if rejected on the clean
  baseline). Plain BT is a **stage-1** alternative; v9-C is a stage-2
  corrector. They are orthogonal.
- The expanded marginal-rejections re-evals listed above (PEER_A/B,
  HBT, Colley, Massey-decay). Each gets its own PR per the prior
  recovery pattern.
- Any change to v9-C's or v8's training code (`train_upset_model.py`,
  `train_stage2.py`). Both stage-2 trainers' XGB hyperparameters are
  PR 6/8/9 defaults -- never tuned, so they don't carry the leaky-
  baseline confound that affected `pairwise_v4.csv`'s tuned XGB params
  (per PR 21's `MM_TUNED_PARAMS_V3` reuse note).
- Wiring v9-C or v8 corrections into the live bracket pipeline
  (`generate_bracket_real.py`). Today's HTML is pure-v4-MC; bracket
  integration is a separate behavior change deferred until v9-C's
  production status is settled.
- Re-running 538 audit follow-up (`feat/v4-gap-audit-fte` parked
  branch). The audit is queued behind step 5 in the recovery
  roadmap.
- A finer-resolution weight grid (e.g., `W_UPSET in {1.10, 1.15,
  1.20, 1.25, 1.30}`). PR 9's grid is reused so the v9-C clean numbers
  are directly comparable to PR 9's leaky numbers. A finer follow-up
  grid is a separate cheap experiment if the clean numbers warrant it.

## Approach

### Inputs

The clean `output/pairwise_v4.csv` from PR 21 (48,465 rows; downstream
consumers dedup with `keep="last"`). No regeneration of v4 -- step 3
of the recovery already did that.

### Step 1: Clean v8 baseline

```sh
python src/train_stage2.py 2>&1 | tee output/v8_clean_rerun.log
```

`train_stage2.py:main()` does the full double-LOSO build via
`build_v8_pairwise()` and writes `output/pairwise_v8.csv`. Reads
`output/pairwise_v4.csv`, `data/raw/march-machine-learning-2026/MNCAATourneyCompactResults.csv`,
`data/raw/march-machine-learning-2026/MNCAATourneySeeds.csv`. No
`MNCAATourneySlots.csv` dependency for v8 (round is not a v8 feature).

Expected runtime: ~2-3 minutes (22 LOSO seasons of XGB stage-2 fits on
~2898 per-game rows each).

Snapshot the leaky-baseline v8 numbers BEFORE overwriting, so the
findings note can quote both:

```sh
cp output/pairwise_v8.csv output/pairwise_v8_pre_clean_rerun.csv
```

(Local-only; not committed. The pre-clean file is the recovery audit
trail: if the clean run produces a surprise, we can compare against
the leaky baseline.)

### Step 2: v9-C 15-cell sweep

```sh
V9_FEATURE_SET=v9c python src/sweep_v9_weights.py 2>&1 | tee output/v9c_clean_sweep_run.log
```

`sweep_v9_weights.py:main()` does:
1. Loads per-game data from `output/pairwise_v4.csv` (clean).
2. For each (W_U, W_M) in the 15-cell grid:
   - Trains v9-C via `build_v9_pairwise(..., feature_set='v9c')`.
   - Writes `output/v9c_sweep/pairwise_v9_WU{u}_WM{m}.csv`.
   - Runs `double_loso_eval` for per-game LL/Acc.
3. Loads `output/pairwise_v8.csv` (clean, from step 1) for the
   comparison.
4. Scores each cell's pairwise CSV via `score_pairwise_path` for
   total bracket points across 22 seasons.
5. Anchor cell (W_U=1.0, W_M=0.0) checked against v8 within 5 brkt
   pts -- prints WARNING if exceeded; sweep continues either way.
6. Writes `output/v9c_sweep_results.csv` with one row per cell:
   `w_upset`, `w_miss`, `total_brkt_pts`, `delta_vs_v8`,
   `ll_loso_weighted_mean`, `acc_loso_weighted_mean`, `pairwise_csv`.

Expected runtime: ~5-7 minutes (15 cells x 22 LOSO seasons).

### Step 3: Per-season W/L breakdown for the winning cell

New script `src/v9c_per_season_breakdown.py`:

```python
"""Per-season bracket-points breakdown for v9-C clean re-run.

The 15-cell sweep driver only emits 22-season totals. This script
reads the v9-C winning cell's per-cell pairwise CSV and the clean v8
pairwise CSV, scores each season individually via
score_pairwise_path, and writes a per-season comparison CSV. Used
in the findings note to show v9-C's W/L spread (matches PR 9's
'profile' reporting).

Inputs (CLI args, all required):
  --v9c-pairwise   Path to v9-C winning cell's pairwise CSV (e.g.
                   output/v9c_sweep/pairwise_v9_WU1.25_WM0.csv).
  --v8-pairwise    Path to clean v8 pairwise CSV
                   (output/pairwise_v8.csv).
  --output         Output CSV path (default
                   output/v9c_clean_per_season.csv).

Output schema:
  season,v8_pts,v9c_pts,delta,winner

Where `winner` is one of {'v8', 'v9c', 'tie'} per
`abs(delta) < 0.5` tie threshold.
"""
```

Implementation: filter both pairwise CSVs by season, call
`score_pairwise_path` per season, assemble the rows. ~50 LOC.

### Step 4: Findings + decision

After the sweep completes, the v9-C best cell's `delta_vs_v8` is the
production-decision input.

| Best v9-C cell delta vs clean v8 | Action |
|---|---|
| `> 0` | v9-C stays in production. Document new winning cell if different from PR 9's (W_U=1.25, W_M=0.0). |
| `<= 0` | Revert: re-run `predict_2026_stage2.py` to restore `output/pairwise_probs.json`; commit the restored canonical artifact alongside the findings note. |
| Anchor cell (W_U=1.0, W_M=0.0) `abs(delta) > 5` brkt pts | Halt; investigate trainer drift before trusting any cell. (Same gate the existing sweep driver enforces.) |

The "stays" branch updates `output/pairwise_v9.csv` to the new winning
cell's per-cell CSV (force-add per PR 9 pattern).

The "revert" branch:
1. `python src/predict_2026_stage2.py` -- regenerates v8-corrected
   2026 pairwise via `train_stage2.py`'s LOSO and applies to
   `output/pairwise_probs_v4.json`.
2. Commits the restored `output/pairwise_probs.json` with a message
   noting v9-C's clean-baseline failure.
3. Documents in the findings note the production state change
   (v9-C deployed -> v8 restored).

### Step 5: Findings note

`docs/notes/2026-05-04-v9c-clean-rerun.md` structure (mirrors PR 22's):

1. **Verdict TL;DR + revert decision** -- one-paragraph summary,
   numerical verdict, production state change (or no-op).
2. **Methods** -- input file paths, hyperparameter confound footnote
   (v9-C/v8 trainers reuse PR 6/8/9 untuned defaults; `pairwise_v4.csv`
   carries the PR 21 tuned-XGB confound documented in step 3).
3. **Clean v8 baseline** -- new total + per-season; comparison vs the
   leaky-baseline v8's 2670 brkt pts.
4. **15-cell v9-C sweep results** -- table sorted by `delta_vs_v8`,
   showing all 15 cells with `total_brkt_pts`, `delta`, `ll`, `acc`.
5. **Winning cell per-season W/L** -- output of step 3's script;
   matches PR 9's "6W-3L-13T" profile reporting.
6. **Anchor sanity check** -- (W_U=1.0, W_M=0.0) cell's delta vs
   clean v8 (expected near 0 -- the trainer is sane).
7. **Discussion** -- link result to PR 22's clean upset-detection
   number (clean v4 catches 15.3%; Vegas 17.5%). If v9-C lost,
   propose interpretation: "v9-C was correcting noise, not signal";
   v8's content-blind stage-2 corrector outperforms an upset-aware
   stage-2 when the upset signal in v4 is below random.
8. **TODO.md update** -- mark step 5 item 1 done; expand
   marginal-rejections list to plain BT standalone (PR 12),
   feature-view ensemble PEER_A/B (PR 14), HBT (PR 16), Colley
   (PR 15), Massey-decay hl=14d (PR 15).
9. **Follow-ups** -- explicit list:
   - Plain BT standalone re-eval (PR 12; standalone LL 0.565 was
     rejected vs leaky v4 0.437 = -0.128 weaker; clean v4 ~0.5588 =
     **tied**; LL-blend gate may now pass).
   - Feature-view ensemble re-eval (PR 14; PEER_A standalone 0.5720
     vs leaky v4 0.4345 = +0.1375 = 5.5x clause-1 tolerance; vs
     clean v4 0.5588 = **+0.013 within tolerance**; clause 1 may
     now pass).
   - HBT re-eval (PR 16; weakest cell HBT LL 0.619; vs clean v4
     gap shrinks from -0.182 to -0.060; less likely to flip than
     plain BT but worth checking).
   - Colley + Massey-decay hl=14d re-eval (PR 15; clause-2 deltas
     +0.0053 / +0.0057 LL; within the +0.122 leak noise floor).
   - BT-as-feature for v9-C re-eval (PR 13; -0.0015 LL; named in
     original roadmap as marginal).
   - v9 weight-sweep family re-eval (PRs 7-9; +18 to +20 brkt pts;
     named in original roadmap as marginal -- this PR provides a
     cheap re-eval since the 15-cell sweep already covers v9-B's
     grid; re-eval is essentially "did v9-B win at any cell on
     clean v4?").

## Tests

Add `tests/test_v9c_per_season_breakdown.py` with one test:

- **`test_v9c_per_season_breakdown_smoke`** -- write tiny synthetic
  v9-C and v8 pairwise CSVs covering 2 seasons with non-trivial
  outcomes; run the script's main entry; assert the output CSV
  has columns `season, v8_pts, v9c_pts, delta, winner`, has 2 rows,
  `winner in {'v8', 'v9c', 'tie'}`, and `delta == v9c_pts - v8_pts`
  per row.

Existing v9-B / v9-C / v8 trainer tests stay green (no trainer code
changes). The existing 137+ test suite stays green; this PR adds 1.

## Success criteria

- `pytest -v` passes (existing suite + 1 new test).
- `python src/train_stage2.py` runs end-to-end and writes
  `output/pairwise_v8.csv`.
- `V9_FEATURE_SET=v9c python src/sweep_v9_weights.py` runs end-to-end,
  writes 15 per-cell CSVs in `output/v9c_sweep/` and
  `output/v9c_sweep_results.csv`. Anchor sanity gate's WARNING (if
  any) does not halt the sweep.
- `python src/v9c_per_season_breakdown.py` runs against the v9-C
  winning cell + clean v8 outputs and writes
  `output/v9c_clean_per_season.csv`.
- The findings note exists and contains all 9 sections above.
- TODO.md step 5 item 1 marked done; marginal-rejections list
  expanded.
- If decision = revert: `output/pairwise_probs.json` restored via
  `python src/predict_2026_stage2.py`; commit notes the production
  state change.
- If decision = stay: `output/pairwise_v9.csv` overwritten with the
  new winning cell's per-cell CSV.

## Risks and mitigations

- **Anchor cell drifts > 5 brkt pts from clean v8.** Possible if v9-C
  trainer's per-game upset-weighting interacts with clean v4's
  different upset distribution in a way that PR 9's leaky-baseline
  anchor (which reproduced v8 exactly) didn't surface. Mitigation:
  the existing gate prints WARNING; operator inspects per-game LL/Acc
  before trusting any cell. If LL/Acc match clean v8 to 3 decimals
  (expected, since the stage-2 trainer code is unchanged), proceed
  and document the brkt-pt drift in the findings as a feature-set x
  baseline interaction.
- **Hyperparameter confound across baselines.** `pairwise_v4.csv`
  carries PR 21's tuned-XGB-on-leaky-baseline confound (documented
  effect <0.02 LL). v8 and v9-C trainers are not tuned, so no
  additional confound from the stage-2 side. Mitigation: footnote
  this in the findings note's Methods section; do not re-tune in
  this PR.
- **Marginal verdict (delta within +/- 5 brkt pts).** If v9-C's best
  cell delta vs clean v8 is, say, +3 brkt pts (within noise), the
  decision matrix's strict `> 0 -> stay` rule keeps an effectively-
  tied model in production. The strict rule is authoritative for
  this PR's production action so the revert/stay decision is
  unambiguous. The per-season profile (matches PR 9's "6W-3L-13T")
  is recorded in the findings note as an *observation* about the
  result's durability, not as an override on the revert decision.
  A fragile profile (e.g., 1W-21T concentrated in a single season)
  becomes a flagged item in the follow-ups section -- "revisit
  v9-C's robustness in a fresh PR" -- but does not flip this PR's
  stay-or-revert action.
- **Revert touches a tracked production artifact.** `output/pairwise_probs.json`
  is force-added per the existing canonical-artifact pattern. The
  revert step writes a different file content (v8-corrected vs v9-C-
  corrected). Mitigation: the revert is `python src/predict_2026_stage2.py`
  -- the same script that produced the v8 version pre-PR 10. The
  diff is auditable in git history.
- **Per-cell pairwise CSV size.** 15 cells x ~3.4 MB each = ~50 MB
  written to `output/v9c_sweep/`. Already gitignored
  (`output/v9c_sweep/` is not in main); only the winning cell's
  CSV is force-added as `output/pairwise_v9.csv`. No diff bloat.
- **Findings note follow-ups list grows large.** Five new
  marginal-rejection re-eval items (plus the two named in the
  original roadmap) could overwhelm the recovery's active queue.
  Mitigation: the follow-ups are listed under the existing
  step-5-marginal-rejections roadmap entry, not promoted to top-
  level queue items. They share priority order: re-eval cheapness
  first (Colley + Massey-decay are 30 min compute; HBT is ~5 min;
  PEER_A/B is ~20 min; plain BT is ~30 min). The findings note
  recommends the order in the follow-ups section.

## Follow-ups (not in this spec)

- Plain BT standalone LL-blend re-eval (recovery step 5 item 3).
- Feature-view ensemble PEER_A/B re-eval (newly added marginal).
- HBT re-eval (newly added marginal).
- Colley + Massey-decay hl=14d re-eval (newly added marginal).
- BT-as-feature for v9-C re-eval (named in original roadmap).
- v9 weight-sweep family re-eval (named in original roadmap;
  partly subsumed by this PR's v9-C 15-cell sweep -- v9-B specific
  grid is a separate cheap follow-up).
- 538 audit follow-up (parked on `feat/v4-gap-audit-fte`).
- Live bracket pipeline integration of stage-2 (deferred since
  PR 10).
