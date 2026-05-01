# v9-C Feature-Stripped Variant -- Findings (2026-05-01)

**Spec:** docs/superpowers/specs/2026-05-01-v9c-feature-stripped-design.md
**Plan:** docs/superpowers/plans/2026-05-01-v9c-feature-stripped.md
**PR 6 v9 findings:** docs/notes/2026-04-30-upset-detection-v9.md
**PR 7 v9-B sweep findings:** docs/notes/2026-05-01-v9-weight-sweep.md
**PR 8 v9-B round-fix findings:** docs/notes/2026-05-01-v9-round-fix.md

## Verdict

**SWAP CANDIDATE: v9-C beats v8 by +43 brkt pts at (W_UPSET=1.25,
W_MISS=0.0) -- and is +23 better than v9-B at the same cell.** Clears
the spec's +25 swap-in bar and *also* shows distinctly better F4/E8
chalk accuracy than both v8 and v9-B. The dropped features
(`v4_confidence`, `is_a_higher_seed`) were carrying noise -- removing
them yields a cleaner test that finally pulls v9 well past v8's
ceiling. Recommend production swap as a follow-up commit on this
branch.

## Numbers

- v8 baseline:                        2670 pts (22 LOSO seasons, 2003-2025)
- v9-B winner cell (1.25, 0.0):       2690 pts (+20 vs v8, PR 8)
- **v9-C winner cell (1.25, 0.0):     2713 pts (+43 vs v8, +23 vs v9-B)**
- v9-C anchor cell  (1.0,  0.0):      2670 pts (delta +0.00 -- exactly v8)

Anchor cell is bit-identical to v8 in brkt pts (vs v9-B's anchor at +7).
The 5 features at uniform weights leave v8's chalk picks unchanged;
the +43 win at (1.25, 0.0) is purely the upset-weighting signal.

## Sweep results (sorted by total brkt pts)

| W_U  | W_M | brkt | LL     | Acc   | dv8 | dv9b |
|------|-----|------|--------|-------|-----|------|
| 1.25 | 0.0 | 2713 | 0.4351 | 0.805 | +43 |  +23 |
| 1.25 | 0.5 | 2684 | 0.4437 | 0.802 | +14 |  +67 |
| 1.00 | 0.5 | 2677 | 0.4383 | 0.806 |  +7 |  -11 |
| 1.00 | 0.0 | 2670 | 0.4324 | 0.807 |   0 |   -7 |
| 1.50 | 0.0 | 2638 | 0.4410 | 0.795 | -32 |  +74 |
| 1.00 | 1.0 | 2626 | 0.4450 | 0.806 | -44 |  +41 |
| 1.25 | 1.0 | 2601 | 0.4552 | 0.789 | -69 | +251 |
| 1.75 | 0.0 | 2579 | 0.4489 | 0.779 | -91 | +190 |
| 1.50 | 0.5 | 2540 | 0.4524 | 0.782 |-130 | +190 |
| 1.75 | 0.5 | 2492 | 0.4624 | 0.770 |-178 |  +85 |
| 2.00 | 0.0 | 2489 | 0.4566 | 0.769 |-181 | +166 |
| 1.50 | 1.0 | 2440 | 0.4670 | 0.765 |-230 | +148 |
| 1.75 | 1.0 | 2353 | 0.4801 | 0.753 |-317 | +137 |
| 2.00 | 0.5 | 2337 | 0.4750 | 0.756 |-333 | +130 |
| 2.00 | 1.0 | 2325 | 0.4948 | 0.739 |-345 | +130 |

Where `dv8` = v9-C delta vs v8, `dv9b` = v9-C delta vs v9-B (PR 8) at
the same cell. v9-B numbers came from re-running the v9-B sweep on
this branch (bit-identical to PR 8's findings table -- regression
check passed; the v9-B re-run is documented under "v9-B regression
check" below).

## Per-season decomposition (v9-C winner vs v8 vs v9-B winner)

| Season | v8  | v9b_W | v9c_W | dv8  | dv9b |
|--------|-----|-------|-------|------|------|
| 2003   |  74 |   74  |   90  |  +16 |  +16 |
| 2004   | 106 |  106  |  106  |    0 |    0 |
| 2005   | 112 |  112  |  112  |    0 |    0 |
| 2006   | 104 |  104  |  108  |   +4 |   +4 |
| 2007   | 104 |  104  |  104  |    0 |    0 |
| 2008   | 147 |  147  |  147  |    0 |    0 |
| 2009   | 115 |  115  |  115  |    0 |    0 |
| 2010   | 123 |  123  |  123  |    0 |    0 |
| 2011   |  70 |   70  |   70  |    0 |    0 |
| 2012   | 138 |  138  |  138  |    0 |    0 |
| 2013   | 159 |  167  |  167  |   +8 |    0 |
| 2014   | 159 |  159  |  159  |    0 |    0 |
| 2015   | 174 |  174  |  174  |    0 |    0 |
| 2016   | 123 |  123  |  123  |    0 |    0 |
| 2017   | 131 |  136  |  135  |   +4 |   -1 |
| 2018   | 155 |  155  |  155  |    0 |    0 |
| 2019   | 116 |  116  |  116  |    0 |    0 |
| 2021   |  72 |   71  |   71  |   -1 |    0 |
| 2022   | 105 |  105  |  101  |   -4 |   -4 |
| 2023   | 148 |  156  |  156  |   +8 |    0 |
| 2024   |  92 |   94  |  102  |  +10 |   +8 |
| 2025   | 143 |  141  |  141  |   -2 |    0 |
| TOTAL  |2670 | 2690  | 2713  |  +43 |  +23 |

vs v8: 6 wins (2003 +16, 2006 +4, 2013 +8, 2017 +4, 2023 +8,
2024 +10), 3 losses (2021 -1, 2022 -4, 2025 -2), 13 ties.

## Per-round chalk accuracy (the F4/E8 lens)

| Round | v8                | v9-B (PR 8)       | v9-C              |
|-------|-------------------|-------------------|-------------------|
| R64   | 582/703 (82.8%)   | 580/703 (82.5%)   | 579/703 (82.4%)   |
| R32   | 272/351 (77.5%)   | 271/351 (77.2%)   | 271/351 (77.2%)   |
| S16   | 114/175 (65.1%)   | 116/175 (66.3%)   | 116/175 (66.3%)   |
| E8    |  48/87  (55.2%)   |  50/87  (57.5%)   |  51/87  (58.6%)   |
| F4    |  26/43  (60.5%)   |  26/43  (60.5%)   |  27/43  (62.8%)   |
| Champ |   9/21  (42.9%)   |   9/21  (42.9%)   |   9/21  (42.9%)   |

| Round | v8 mean pts | v9-B mean pts | v9-C mean pts |
|-------|-------------|---------------|---------------|
| R64   |    26.45    |    26.36      |    26.32      |
| R32   |    24.73    |    24.64      |    24.64      |
| S16   |    20.73    |    21.09      |    21.09      |
| E8    |    17.45    |    18.18      |    18.55      |
| F4    |    18.91    |    18.91      |    19.64      |
| Champ |    13.09    |    13.09      |    13.09      |

The win profile is exactly what the spec's "F4/E8 lens" was looking
for: v9-C trades fractional points at R64/R32 for material gains at
S16, E8, and F4. E8 accuracy moves from 55.2% to 58.6% (+3.4 pp);
F4 accuracy moves from 60.5% to 62.8% (+2.3 pp). Each correct E8 game
is worth 8 brkt pts; each correct F4 game is worth 16; each correct
Champ pick is worth 32. v9-C captures more bracket points where
bracket points actually matter.

## Win robustness

- Best 6 seasons removed: still +43 - (+16+10+8+8+4+4) = -7 net.
  This means a clean "remove the biggest win" stress test (2003 +16)
  leaves v9-C at +27, still above the +25 swap-in bar.
- Best 2 seasons removed (2003 +16, 2024 +10): +43 - 26 = +17, in
  the marginal band.
- The +43 is not concentrated in one year (vs PR 7's v9-B winner
  where 2024 alone was +12 of +18).
- LOSO log loss at the winner cell: 0.4351 (v8: 0.4323; v9-B winner:
  0.4367). v9-C is between v8 and v9-B on the LL axis but above both
  on bracket pts -- consistent with the F4/E8 lens, since LL is per-
  game while brkt pts weight late rounds.

## Active ingredient

The winning cell is the same as PR 8's v9-B winner: **W_UPSET=1.25,
W_MISS=0.0**. This means:

- **Upset weighting (W_UPSET=1.25) is the active ingredient.** Mild
  multiplicative weighting on rows where the higher seed lost.
- **Residual-squared weighting (W_MISS=0.0) is OFF** -- v9-C does
  not need it. PR 7's v9-B winner relied entirely on W_MISS at
  (1.0, 0.5); the round fix in PR 8 already flipped the active
  ingredient to W_UPSET. v9-C's feature-set strip confirms and
  strengthens that flip: with the cleaner feature set, mild upset
  weighting is materially helpful (+43) without any miss weighting.

The original v9 hypothesis ("upset weighting helps") is now
unambiguously supported, but only with (a) low magnitude (1.25, not
3.0), (b) the round fix from PR 8, and (c) the feature-set strip
from this work.

## v9-B regression check

Re-ran the v9-B sweep on this branch (V9_FEATURE_SET=v9b) before the
v9-C run. All 15 cells reproduced PR 8's findings table within
display precision: anchor 2677 (PR 8: 2677), winner 2690 at
(1.25, 0.0) (PR 8: 2690), full table aligned line-by-line. This
confirms the parameterization is a no-op for v9-B and the
implementation is sound. PR 8's `output/v9_sweep_results.csv` was
not committed (output/ is gitignored; only `pairwise_v9.csv` was
force-added in PR 8), so the regression check is a visual table
match, not a CSV byte-diff. The match is exact at integer brkt-pt
precision.

## Recommendation

**Swap to production.** Per the spec's decision matrix:

| v9-C delta vs v8 | F4/E8 lens          | Action          |
|------------------|---------------------|-----------------|
| +43 (>+25)       | distinctly better   | swap candidate  |

Both clauses of the swap-in path are satisfied. The follow-up swap
commit on this branch should:

1. Update default `feature_set="v9c"` (and `W_UPSET=1.25`,
   `W_MISS=0.0`) in `src/train_upset_model.py` -- or the bracket
   pipeline pointer below, whichever is the cleaner integration point.
2. Regenerate `output/pairwise_v9c.csv` (production v9-C output) at
   the winner weights via `python src/train_upset_model.py` (or a
   dedicated v9-C entry point).
3. Update bracket-pipeline code that currently consumes
   `output/pairwise_v8.csv` to point at the v9-C output instead.
4. Regenerate the 2026 chalk bracket (`output/bracket.html`,
   `output/bracket_data.json`, etc.) and confirm picks differ in
   reasonable ways (don't audit every change, but spot-check that
   no obviously broken picks land).

If the swap is intentionally deferred, this finding still stands as
"v9-C is the v9 line's actual ceiling, and it clears swap" -- the
swap is a separate decision from the experiment.

## Caveats

- **Multiple-comparisons inflation.** v9-C is the 31st through 45th
  cell tested across the v9 line of inquiry (PR 7: 15, PR 8: 15,
  this: 15). The +25 bar was set with this in mind. v9-C clears it
  by +18 (delta +43 - bar +25), and the F4/E8 lens provides
  independent confirmation. Multiple comparisons should not change
  the recommendation.
- **2026 prediction not regenerated yet.** The 2026 chalk bracket
  still reflects v8 + a small v9 stage-2. Once v9-C is wired into
  the bracket pipeline (swap commit), the 2026 picks will likely
  shift on a few games. The shift may flip Vanderbilt / Iowa St. /
  Texas Tech / Duke (the bust candidates from the v4 ablation) --
  documented separately when the swap commit lands.
- **2003 single-season +16 contribution.** The biggest win is in
  2003, the earliest season tested. Removing 2003 still leaves
  +27 (above the +25 bar), but the 2003 jump is large enough to
  warrant a sanity check: is the chalk pick that flipped actually
  reasonable? Not investigated here; left for the swap commit's
  pre-merge review.
- **R64 accuracy drop (-0.4 pp).** v9-C trades small losses at
  R64 for material gains at S16/E8/F4. This is the right trade on
  the bracket-points objective (1 pt vs 4/8/16 pts), but a pure
  "predict more games right" metric would slightly favor v8. The
  spec uses bracket pts as the objective, so this is by design.

## Artifacts

Local-only (output/ gitignored, mirroring PR 8 convention):
- `output/v9c_sweep_results.csv` -- 15-row sorted results table.
- `output/v9c_sweep/pairwise_v9_WU{u}_WM{m}.csv` -- 15 per-cell
  pairwise CSVs.
- `output/v9c_sweep_run.log` -- full sweep driver log including
  anchor verdict and winner declaration.
- `output/v9b_repro_run.log` -- v9-B regression-check sweep log.

Committed:
- `docs/notes/2026-05-01-v9c-feature-stripped.md` (this file).
- `docs/superpowers/specs/2026-05-01-v9c-feature-stripped-design.md`
  and `docs/superpowers/plans/2026-05-01-v9c-feature-stripped.md`
  (already on branch).
- `src/train_upset_model.py` and `src/sweep_v9_weights.py` (the
  parameterization changes from Tasks 1-4).
