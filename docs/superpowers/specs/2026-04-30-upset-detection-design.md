# Upset-Detection Sub-Model (v9) Design

## Problem Statement

v4 is the production stage-1 model. v8 (the stage-2 corrector in
`src/train_stage2.py`) is a generic meta-learner trained on v4's
out-of-fold pairwise predictions and adds +9 bracket points over v4 in
the 22-season LOSO backtest. v8's win is structural -- it learns the
average residual pattern of v4's mistakes -- but it is target-agnostic:
it optimizes plain log loss on `did A beat B?` and is not built to flag
upsets specifically.

The 2026 high-confidence busts (Vanderbilt, Iowa St., Texas Tech, Duke)
showed up as a recurring failure mode that v8 only partially absorbs.
The v4 feature ablation (`docs/notes/2026-04-30-ablation-v4-findings.md`)
established that the over-confidence is not feature-side -- no single
v3/v4 feature group drives the misses -- so the next leverage is in the
*meta-layer*: train a corrector whose loss explicitly emphasizes the
upset cases and the high-confidence misses. This spec is the design for
that corrector. Working name: v9.

## Goals

- Replace v8 with v9, an upset-aware stage-2 corrector trained on v4's
  out-of-fold pairwise predictions, using an upset-weighted loss.
- Direct head-to-head against v8 over the same 22-season LOSO backtest
  on bracket points and weighted-mean LOSO log loss.
- Ship a clean, documented experiment whose verdict is decisive: v9 wins
  (replace v8, promote per-round specialists), v9 ties (keep v8,
  abandon expansion), or v9 loses (try a feature-extension fallback
  before abandoning).

## Non-Goals

- Per-round specialists (one model per round). Held back as a follow-up
  contingent on v9 winning.
- Replacing v4 stage-1. Out of scope.
- Refactoring `train_stage2.py`. v8 stays intact and reproducible
  side-by-side with v9 so the head-to-head is clean.
- Re-tuning v4's Optuna params or Platt calibration. Reuse the v4
  pairwise CSV that v8 already consumes.

## Approach

### Architecture: replace v8, reuse v8's training scaffold

Pipeline becomes `v4 -> v9`. v9 fully takes v8's slot in the chain. The
new file `src/train_upset_model.py` mirrors `src/train_stage2.py` shape:

- Same XGBoost shape (n_estimators=100, max_depth=3, lr=0.05, seed=42).
- Same per-game row construction (each game contributes two symmetric
  rows: A=W with label=1 and A=L with label=0).
- Same target: `label = 1 if A beat B else 0`. Symmetric in A/B.
- Same input features:
  `(p_v4_stage1, seed_a, seed_b, abs_seed_diff)`. Four columns.
- Same double-LOSO leakage discipline: for test season Y, v4's
  predictions for Y come from the already-out-of-fold
  `output/pairwise_v4.csv`, and v9 is trained on every other season's
  per-game tuples and applied to Y.

What changes from v8: **sample weights**. The "upset specialization"
lives entirely in the training-row weight scheme; the model never sees
an upset flag as an input feature, only as something the loss cares
about more.

### Sample weighting scheme

For each training row (one symmetric pair contributes two rows):

```
w = 1.0
if higher_seeded_team_in_this_game_lost:    w *= W_UPSET     # default 3.0
w *= (1 + W_MISS * residual ** 2)           # residual = label - p_v4_for_this_perspective
                                            # default W_MISS = 4.0
```

Effect:

- Non-upset, v4 was right (residual ~ 0):                         `w ~ 1`
- Non-upset, v4 was wrong (residual ~ 1):                         `w ~ 5`
- Upset, v4 nearly got it (residual ~ 0):                         `w ~ 3`
- Upset, v4 was confidently wrong (residual ~ 1):                 `w ~ 15`

`W_UPSET` and `W_MISS` are constants near the top of the script. A
narrow sweep (e.g. `W_UPSET in {1, 2, 3, 5}`, `W_MISS in {0, 2, 4, 8}`)
is cheap because each LOSO backtest is the same size as v8's.

### Same-seed tiebreaker

Same-seed games (rare; happen at F4 / Champ) have no "higher seed".
Treat them as **non-upset** -- skip the `W_UPSET` multiplier; the
`W_MISS` multiplier still applies. Documented in code with a one-line
comment near the upset-flag computation.

### Why same features as v8

Whole thesis here is "upset specialization comes from training, not
features." If we change weights *and* features simultaneously, we cannot
attribute the effect. v8's 4-feature baseline is the right control. A
pure A/B against v8 isolates the weighting hypothesis cleanly.

The 4-feature minimalism is a real concern -- 4 features may not carry
enough signal for the model to do upset-aware corrections that go
beyond what v8 already learns. That concern is addressed by the
explicit feature-extension fallback below: if v9 ties or loses to v8,
we run a feature-extended variant before declaring the upset-detection
direction dead.

### Feature-extension fallback

If the 4-feature v9 (call it v9-A) does not beat v8, do **not** abandon
the upset-direction yet. Run a follow-up variant -- v9-B -- before the
verdict is final:

- v9-B inputs:
  `(p_v4_stage1, seed_a, seed_b, abs_seed_diff, round, |p_v4 - 0.5|, is_a_higher_seed)`.
  7 features. Same loss, same weighting scheme, same training scaffold.
- Reasoning: round number and v4 confidence let the model condition its
  upset adjustment on round-specific upset frequencies and on v4's
  confidence band. v7's negative result on round-as-feature was for
  stage-1 (raw outcomes); stage-2 already has v4's prediction as input,
  so round can now modulate v4's confidence per round rather than being
  asked to predict outcomes from raw features.
- v9-B ships only if v9-A produced ambiguous or negative head-to-head
  results. If v9-A wins outright, v9-B is deferred along with per-round
  specialists.

The success-criteria table in this spec applies to **whichever variant
ends up shipping**: v9-A first, v9-B as fallback if needed.

### Code structure

- **New file:** `src/train_upset_model.py`. Sibling to
  `src/train_stage2.py`, not a refactor of it. Keeping v8's file
  untouched preserves a clean reproducible baseline alongside v9.
- **Reuse from `train_stage2.py`:** `parse_seed`, the seed-lookup
  pattern in `load_per_game_data`, the `double_loso_eval` skeleton.
  Reuse via copy with attribution comments rather than a shared helper
  module -- the helpers are small, the dependency is unstable, and the
  per-game row builder for v9 needs to also tag the upset flag (a v8
  refactor would break v8's reproducibility).
- **New helpers in `train_upset_model.py`:**
  - `load_per_game_data_with_upset(...)` -- per-game DataFrame with an
    extra `upset` bool column (`higher_seed_loses_in_this_game`).
  - `compute_sample_weights(df, w_upset, w_miss)` -> 1-D numpy array
    of weights aligned with rows.
- **Eval + output mirror v8:** double-LOSO eval table printed to
  console + `output/pairwise_v9.csv` written for the 22-season backtest
  scorer (`bracket_scorecard.py` and friends consume this CSV
  unchanged).

## Deliverables

1. **`src/train_upset_model.py`** -- the v9 trainer, runnable as
   `python src/train_upset_model.py` from the repo root, mirroring
   v8's invocation.

2. **`output/pairwise_v9.csv`** -- per-season, per-pair v9 probabilities
   across all 22 LOSO seasons. Same schema as `pairwise_v8.csv` so the
   downstream bracket scorer is a drop-in.

3. **`output/v9_eval.csv`** -- per-season comparison row:
   `season, n_games, ll_v4, ll_v8, ll_v9, acc_v4, acc_v8, acc_v9,
   bracket_pts_v4, bracket_pts_v8, bracket_pts_v9`.

4. **Tests under `tests/`:**
   - `tests/test_upset_model.py` -- unit tests for
     `compute_sample_weights` (all four weight regimes),
     `load_per_game_data_with_upset` (upset flag for known cases:
     1-vs-16 win, 5-vs-12 loss, same-seed F4), and double-LOSO leakage
     guard.
   - Synthetic-data integration test mirroring the existing
     `tests/test_integration.py` shape: 3 seasons in, expected output
     CSV columns and shape out.

5. **Writeup** -- short markdown note
   `docs/notes/2026-04-30-upset-detection-v9.md` with the verdict, the
   per-round upset recall/precision sanity numbers, and the
   recommendation: replace v8 (and promote per-round), tie (keep v8 +
   shelve expansion), or lose (run v9-B feature-extended fallback or
   abandon).

## Success Criteria

Bracket points are the bottom-line metric (matches v8's +9-pt yardstick
already in TODO.md):

| v9 vs v8 (22-season LOSO bracket pts)   | LOSO log loss check    | Decision                                                                                       |
|-----------------------------------------|------------------------|------------------------------------------------------------------------------------------------|
| +3 or better                            | not worse than v8      | **Replace v8.** Promote per-round specialists (Q1 option B from brainstorm) to active-queue #1.|
| within +/- 3                            | not worse than v8      | **Tie.** Run v9-B feature-extended fallback. If still tied or worse, keep v8 and shelve expansion.|
| < -3                                    | any                    | **Lose.** Run v9-B feature-extended fallback. If still losing, keep v8, document, abandon direction.|

The +/- 3 magnitude is roughly the season-to-season noise band on this
dataset; v6 came in at +7 vs v4 and was correctly classified as noise,
so 3 is the smallest delta we should trust as signal.

Sanity check (descriptive, not a gate): per-round upset recall (fraction
of actual upsets where v9 puts > 0.5 on the underdog) and precision.
Used to decide which rounds the per-round specialists should target if
v9 wins.

## Implementation Notes

- `output/pairwise_v4.csv` already contains v4's out-of-fold predictions
  across all backtest seasons; v9 reads it the same way v8 does. Do
  not re-run v4 LOSO -- that is the slow step and we want a clean
  comparison anyway.
- The seed lookup must use the integer `seed_int` parsed from the
  Kaggle seed string (`W01`, `W11a`, etc.) -- `train_stage2.py`'s
  `parse_seed` is the reference implementation.
- Same-seed games: when `seed_a == seed_b`, no upset flag triggers. The
  `is_higher_seed` concept does not apply. Skip the `W_UPSET`
  multiplier; `W_MISS` multiplier still applies. Document inline.
- Sample weight computation must be applied **only on the training
  side** of each LOSO fold. Test rows are evaluated unweighted via
  log loss / accuracy on a 1-row-per-game basis (winner perspective)
  to mirror v8's reporting.
- Double-LOSO leakage guard: the test added in `tests/test_upset_model.py`
  must construct a synthetic per-game DataFrame with a known season
  marker, fit the trainer's per-season loop, and assert that no test
  fold's training data contains rows from its own test season.
- Reuse v8's per-pair output schema exactly:
  `season, team_a, team_b, p_a_wins` with `team_a < team_b`. The
  bracket scorer expects this normalization.
- ASCII-only in all written files (CLAUDE.md). No em-dashes, smart
  quotes, arrows, etc. Especially in `print()` statements -- a
  non-ASCII char crashes the script on Windows console (cp1252).

## Evaluation / Acceptance

This spec is satisfied when the writeup answers:

- "v9 vs v8 over 22 LOSO seasons: bracket pts delta = X, weighted-mean
  log loss delta = Y."
- "Per-round upset recall and precision for v9: R64 = ..., R32 = ...,
  ..., Champ = ...."
- "Decision: replace v8 / tie / lose, with the feature-extension
  fallback (v9-B) status if applicable."

A clean answer in any of the three cells in the success-criteria table
is a satisfactory result. The failure mode to avoid is shipping a v9
that beats v8 on log loss but loses on bracket points without flagging
the trade-off explicitly -- bracket points are the bottom-line
optimization target for this project, and a calibration-only win is
not a ship-it.
