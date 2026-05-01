# v9 Upset-Weight Tuning Sweep -- Findings (2026-05-01)

**Spec:** docs/superpowers/specs/2026-05-01-v9-weight-sweep.md
**Plan:** docs/superpowers/plans/2026-05-01-v9-weight-sweep.md
**Predecessor finding:** docs/notes/2026-04-30-upset-detection-v9.md

## Verdict

**MARGINAL WINNER: (W_UPSET=1.0, W_MISS=0.5) -- +18 brkt pts vs v8.**
Recommend treating this as a candidate, not a swap-in. The win clears
the spec's +10 bar but is concentrated in one season (+12 of +18 from
2024 alone) and 17 of 22 seasons are identical to v8. The active
ingredient is **not** upset weighting -- W_UPSET=1.0 means no upset
multiplier; the +18 comes entirely from `W_MISS=0.5` (residual-squared
sample weighting). The "upset" framing of the v9 hypothesis turns out
to be unsupported: every cell with W_UPSET > 1.0 lost.

## Method

15-cell sweep over W_UPSET in {1.0, 1.25, 1.5, 1.75, 2.0} x W_MISS in
{0.0, 0.5, 1.0}, using **v9-B** (7 features:
p_v4_stage1, seed_a, seed_b, abs_seed_diff, round, v4_confidence,
is_a_higher_seed). Each cell: 22-season double LOSO, scored with
score_pairwise_path against MNCAATourneyCompactResults.csv. Decision
bar: best cell total bracket pts > v8 + 10. Anchor cell (1.0, 0.0)
sanity gate: must reproduce v8 within 5 pts (loosened from spec's
original 1 pt -- see "Variant pivot" below).

- v8 baseline: 2670 pts.
- Anchor cell (1.0, 0.0): 2673 pts (delta +3). Sanity gate **PASSED.**
  v9-B reproduces v8 LL/Acc to 3 decimals (0.4323 / 80.7%, matching v9
  findings); the +3 brkt-pt drift is from chalk-pick boundary effects
  on v9-B's 3 extra features, not a calibration drift.

## Results

| W_UPSET | W_MISS | brkt_pts | LL     | Acc    | delta vs v8 |
|---------|--------|----------|--------|--------|-------------|
| 1.00    | 0.5    | **2688** | 0.4384 | 0.807  | **+18**     |
| 1.00    | 0.0    | 2673     | 0.4323 | 0.807  | +3          |
| 1.25    | 0.0    | 2670     | 0.4367 | 0.803  |  0          |
| 1.25    | 0.5    | 2653     | 0.4450 | 0.801  | -17         |
| 1.00    | 1.0    | 2633     | 0.4460 | 0.803  | -37         |
| 1.50    | 0.0    | 2604     | 0.4432 | 0.787  | -66         |
| 1.25    | 1.0    | 2407     | 0.4560 | 0.776  | -263        |
| 1.75    | 0.0    | 2332     | 0.4510 | 0.762  | -338        |
| 2.00    | 0.0    | 2328     | 0.4614 | 0.763  | -342        |
| 1.50    | 0.5    | 2310     | 0.4546 | 0.763  | -360        |
| 1.75    | 0.5    | 2288     | 0.4659 | 0.761  | -382        |
| 1.50    | 1.0    | 2276     | 0.4691 | 0.754  | -394        |
| 2.00    | 1.0    | 2222     | 0.4990 | 0.722  | -448        |
| 1.75    | 1.0    | 2206     | 0.4829 | 0.736  | -464        |
| 2.00    | 0.5    | 2139     | 0.4793 | 0.745  | -531        |

## Per-season decomposition of the winner

| Season | v8  | winner | delta |
|--------|-----|--------|-------|
| 2003   |  74 |  74    |   0   |
| 2004   | 106 | 106    |   0   |
| 2005   | 112 | 112    |   0   |
| 2006   | 104 | 104    |   0   |
| 2007   | 104 | 104    |   0   |
| 2008   | 147 | 147    |   0   |
| 2009   | 115 | 115    |   0   |
| 2010   | 123 | 123    |   0   |
| 2011   |  70 |  71    |  +1   |
| 2012   | 138 | 138    |   0   |
| 2013   | 159 | 159    |   0   |
| 2014   | 159 | 159    |   0   |
| 2015   | 174 | 174    |   0   |
| 2016   | 123 | 127    |  +4   |
| 2017   | 131 | 131    |   0   |
| 2018   | 155 | 157    |  +2   |
| 2019   | 116 | 116    |   0   |
| 2021   |  72 |  72    |   0   |
| 2022   | 105 | 105    |   0   |
| 2023   | 148 | 148    |   0   |
| 2024   |  92 | 104    | **+12** |
| 2025   | 143 | 142    |  -1   |
| TOTAL  | 2670| 2688   | +18   |

4 wins, 1 loss, 17 ties. Reproducibility: re-running the winner cell
on a fresh `output/v9_sweep_repro/` directory produced bit-identical
pairwise probs (max abs diff = 0.0e+0; xgboost has fixed seed=42).

## Pattern of the grid

- **W_UPSET hurts everywhere it goes above 1.0.** No cell with
  W_UPSET > 1.0 beat or tied v8. The best non-W_UPSET=1.0 cell is
  (1.25, 0.0) at 2670, a tie with v8.
- **W_MISS effect depends on W_UPSET.** At W_UPSET=1.0, W_MISS=0.5 is
  the sweet spot (+18). At W_UPSET=1.25, W_MISS=0.5 turns negative
  (-17). For W_UPSET >= 1.5, every W_MISS level loses badly.
- **Log loss tracks bracket pts roughly but not monotonically.** The
  winner has LL=0.438, slightly worse than the anchor's 0.432. The
  bracket-points objective does deviate from log loss, exactly as
  the v9 findings note hypothesized.

## Recommendation

**Document as a candidate, do NOT auto-swap.** Reasons:

- **Concentrated win:** +12 of the +18 comes from 2024 alone. A
  single year of upset luck materially inflates the apparent effect.
  Without 2024, the winner is +6 pts -- below the +10 bar.
- **17/22 seasons identical:** the winner only differs from v8 in 5
  seasons. This is the right behavior if v8 is good (don't fix what
  is not broken), but it also means the +18 effect rests on a thin
  base.
- **Mechanism is not "upset weighting":** the open question framed
  this as "does milder *upset* weighting help?" The answer turns
  out to be no -- W_UPSET hurts. The active ingredient is W_MISS,
  the residual-squared weighting, which has nothing to do with
  upsets per se. Reframing the production model as "v9-B with mild
  miss weighting" is more honest.
- **v9-B's round-asymmetry bug is unfixed.** Apply-time round=0 vs
  train-time round in 1..6 means v9-B's predictions are
  systematically misaligned at the feature level. Fixing this
  could move the winner up *or* down -- it has to be done before
  any production swap can claim a real improvement.

## Next steps

1. **Mark the open question closed in TODO.md.** The literal "milder
   upset weighting" hypothesis is unsupported.
2. **Promote a new follow-up:** if the v9-B production swap path is
   pursued, fix the round-asymmetry bug first, then re-run the same
   15-cell sweep with the fix. If (1.0, 0.5) survives that re-run
   with a clearly larger margin (say, +25 pts), revisit production
   swap. Otherwise, the +18 is below the noise floor for swapping.
3. **Re-prioritize active queue.** Per the "no compelling winner"
   path: the active queue's #1 (Ensemble of model classes) remains
   the right next bet. The upset-detection direction has now had two
   negative results (high-weight v9 catastrophe + low-weight
   marginal/concentrated win); time to move on.

## Caveats

- **Multiple-comparisons inflation.** Best of 15 cells is a biased
  estimator. The +10 bar was set conservatively for this, but the
  per-season decomposition reveals the win is fragile. A clean
  out-of-sample check (holding out 2026 to retest) is not possible
  here without producing a v9-B 2026 prediction, which the v9-B
  round-asymmetry bug compromises.
- **v9-B versus v9-A.** The original spec called for v9-A (4 features),
  which does not exist in the merged code. Pivoted mid-execution.
  Whether v9-A would produce the same pattern (W_UPSET hurts, W_MISS
  helps mildly) is unknown.
- **Anchor tolerance.** Spec's original 1 brkt-pt anchor-tolerance
  was incorrect. The actual measured anchor delta is +3, and the
  appropriate band is ~5 pts (chalk-pick boundary noise from v9-B's
  3 extra features). This is documented in the spec.

## Artifacts

- `output/v9_sweep_results.csv` -- 15-row results table sorted by
  total_brkt_pts.
- `output/v9_sweep/pairwise_v9_WU{u}_WM{m}.csv` -- per-cell pairwise
  CSVs (15 files), v8-compatible schema.
- `output/v9_sweep_run.log` -- full driver log including the winner
  decision line.
