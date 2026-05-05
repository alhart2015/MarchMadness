# v4 Gap Audit vs Vegas -- Findings

**Date:** 2026-05-04 (audit rerun under clean baseline)
**Branch:** feat/v4-vegas-audit-rerun (rerun); original audit on feat/v4-gap-audit-vegas
**Verdict (REVISED):** **v4 loses to Vegas on overall log loss (+0.0148 worse) and matches it on accuracy.** Six bucket-level weak spots surface at the n>=50, ll_delta>=+0.02 threshold: rounds E8 / S16, upset games, seed-diff 6-9 / 10-15, and the 0.80-0.90 v4-confidence quintile. The original audit's "no weak spots" verdict was an artifact of the v4 Vegas-feature contamination (recovery step 4 of the 2026-05-04 leak fix).
**Spec:** `docs/superpowers/specs/2026-05-04-v4-gap-audit-vegas-design.md`
**Plan:** `docs/superpowers/plans/2026-05-04-v4-gap-audit-vegas.md`
**Recovery context:** TODO.md "CONTAMINATION DISCOVERED 2026-05-04" -> step 4.

## Retraction notice

The original version of this note (committed under `feat/v4-gap-audit-vegas`,
PR 18) reported that v4 beat Vegas on every bucket -- overall LL 0.4305 vs
0.5447, no weak spots. That conclusion is **WITHDRAWN.** The `pairwise_v4.csv`
the original audit consumed was produced by a feature pipeline that aggregated
Vegas closing-line statistics over the FULL Vegas dataset including NCAA
tournament games, leaking the holdout season's tournament outcomes into the
test feature row under LOSO. PR 19 fixed the leak; PR 21 regenerated
`pairwise_v4.csv` under the clean pipeline (mean LL 0.4370 -> 0.5588 across
22 LOSO seasons, mean acc 80.4% -> 70.7%). This note rewrites the audit
against the clean `pairwise_v4.csv`.

The original (contaminated) per-bucket tables are preserved at the bottom of
this note in `## Appendix: original (contaminated) numbers` for reference.

## TL;DR (clean baseline)

Compared the regenerated v4 stage-1 tournament predictions against Vegas
closing-line implied probabilities (SIGMA=11) across 1326 played 2003-2025
tournament games, bucketed by round, chalk-vs-upset, v4-confidence
quintile, and seed-diff magnitude.

**v4 loses to Vegas on log loss and roughly ties on accuracy.** Overall:

| metric    | v4     | Vegas  | delta (v4 - Vegas) |
|-----------|--------|--------|--------------------|
| log loss  | 0.5595 | 0.5447 | **+0.0148** (Vegas better) |
| accuracy  | 69.9%  | 70.6%  | -0.7 pp            |
| ECE       | 0.037  | 0.029  | +0.008             |

This is a complete reversal of the original audit. The leak was inflating
v4's apparent edge by ~0.13 LL and ~10pp accuracy on this 1326-game
subset; under the clean pipeline that edge is gone, and Vegas wins or
ties on every aggregate metric.

**Six weak spots** at the n>=50, ll_delta>=+0.02 threshold:

| axis                   | bucket    | n   | ll_v4  | ll_vegas | delta   | acc_v4 | acc_vegas |
|------------------------|-----------|-----|--------|----------|---------|--------|-----------|
| round                  | E8        |  80 | 0.7105 | 0.6555   | +0.0550 | 0.525  | 0.575     |
| chalk_won              | upset     | 360 | 1.1135 | 1.0592   | +0.0543 | 0.153  | 0.175     |
| round                  | S16       | 152 | 0.6293 | 0.6027   | +0.0266 | 0.658  | 0.658     |
| seed_diff_bucket       | 6-9       | 385 | 0.5648 | 0.5387   | +0.0261 | 0.727  | 0.743     |
| v4_confidence_quintile | 0.80-0.90 | 400 | 0.3768 | 0.3518   | +0.0250 | 0.870  | 0.868     |
| seed_diff_bucket       | 10-15     | 264 | 0.3077 | 0.2869   | +0.0208 | 0.913  | 0.913     |

The weak spots cluster around two themes: (1) v4 underperforms Vegas
specifically on upset detection (the upset bucket at -0.054 is the
single largest n>=50 gap, and S16/E8 are the rounds where upsets become
load-bearing), and (2) v4 is mis-calibrated in the 0.80-0.90 confidence
band (it claims more certainty than its empirical hit rate supports
relative to Vegas, even though both models are right ~87% of the time).

## What changed under the clean baseline

1. **v4 loses LL by +0.0148** instead of beating Vegas by -0.114 (a
   net swing of +0.129 on aggregate LL).
2. **v4's upset edge disappears.** Original audit: v4 caught 56% of
   upsets vs Vegas's 17%. Clean: v4 catches 15.3% of upsets vs Vegas's
   17.5%. The "v4 has a real upset-detection edge" finding from PR 18
   was driven entirely by the leak -- the tournament-game Vegas
   averages were peeking at upset outcomes within the holdout season.
3. **v4's confidence distribution collapsed.** Original audit had
   413 games in the 0.90-1.00 v4-confidence quintile. Clean: ZERO
   games land in 0.90-1.00. The mass shifted down: 0.50-0.60 went
   from n=130 to n=288, 0.60-0.70 from 127 to 290, 0.70-0.80 from
   216 to 348. The leak was making v4 feel certain about tournament
   outcomes because it had peeked at them.
4. **Per-round LL deltas flipped sign in every round except F4 / FF /
   Champ** (where n is small and the delta is near zero). E8 and S16
   are the worst.

## Setup recap

- **Inputs:** `output/pairwise_v4.csv` (48,465 unique pair keys, the
  PR 21 clean regen; md5 `795d8ddfcd7a0a09a50c3732825c6316`),
  Vegas closing-line CSVs in `data/raw/vegas_lines/` (95,355 line
  rows, 22 season files), `MNCAATourneyCompactResults` for outcomes,
  `MNCAATourneySeeds` + `MSeasons.csv` for seed + DayZero lookups.
- **Spread-to-prob conversion:** `norm.cdf(line / SIGMA)` with
  `SIGMA=11.0`, matching existing `src/blend_sweep.py` and
  `src/alternate_bracket.py`.
- **Join:** by `(Season, DayNum, sorted_team_pair)` with +/- 1 day
  slack. Vegas team names resolved via the existing
  `_build_vegas_name_to_kaggle_map` + `_resolve_vegas_name`
  fuzzy-matcher (400 / 405 unique names resolved).
- **Coverage:** 91.5% (1326 of 1449 played 2003-2025 games joined).
  Misses concentrated in 2003 (64 games -- the entire 2003 tournament
  Vegas data is unmatchable, likely a date-format or team-name issue
  specific to that year) and 2023 (16 missing). Other seasons have
  near-complete coverage. **Same coverage profile as the original
  audit -- the leak fix did not change the join.**
- **Wall time:** ~10 seconds (unchanged).

## Anchors (clean baseline)

| anchor                              | expected band     | observed | verdict |
|-------------------------------------|-------------------|----------|---------|
| `overall.ll_v4`                     | ~0.5588 (regen mean) | 0.5595 | PASS (subset within 0.001 of full-population mean) |
| `overall.acc_vegas`                 | [0.70, 0.72]      | 0.706    | PASS    |
| `coverage`                          | >= 60%            | 91.5%    | PASS    |
| `ll_vegas`                          | [0.42, 0.46]      | 0.5447   | OFF (same SIGMA=11 caveat as before; see Caveats) |

The clean `ll_v4` of 0.5595 sits 0.0007 above the 22-season clean-LOSO
mean of 0.5588 reported in PR 21's findings note -- consistent with
the audit running on a 1326-game subset (the 123 missing games skew
slightly easier for v4 in this subset).

## Per-bucket results (clean baseline)

### By round

| round | n   | ll_v4  | ll_vegas | delta   | acc_v4 | acc_vegas | mean(p_v4 - p_vegas) |
|-------|-----|--------|----------|---------|--------|-----------|----------------------|
| FF    |  53 | 0.6457 | 0.6475   | -0.0019 | 0.623  | 0.660     | -0.023               |
| R64   | 648 | 0.5164 | 0.5045   | +0.0118 | 0.725  | 0.731     | +0.001               |
| R32   | 319 | 0.5630 | 0.5510   | +0.0120 | 0.727  | 0.712     | -0.002               |
| S16   | 152 | 0.6293 | 0.6027   | +0.0266 | 0.658  | 0.658     | -0.002               |
| E8    |  80 | 0.7105 | 0.6555   | +0.0550 | 0.525  | 0.575     | -0.011               |
| F4    |  42 | 0.5703 | 0.5704   | -0.0001 | 0.667  | 0.714     | -0.030               |
| Champ |  20 | 0.5534 | 0.5603   | -0.0069 | 0.700  | 0.750     | -0.015               |

R64 / R32 / S16 / E8 are all v4-loss zones. FF / F4 / Champ are
near-ties but with small n. The S16 -> E8 gap widens: in the
original audit, v4 led E8 by -0.114 LL; clean, v4 lags E8 by
+0.055 LL. Elite Eight is the round with the worst clean-baseline
gap.

### By chalk vs upset

| outcome | n   | ll_v4  | ll_vegas | delta   | acc_v4 | acc_vegas |
|---------|-----|--------|----------|---------|--------|-----------|
| chalk   | 966 | 0.3530 | 0.3530   | +0.0000 | 0.903  | 0.904     |
| upset   | 360 | 1.1135 | 1.0592   | +0.0543 | 0.153  | 0.175     |

**Upset detection is the single biggest weak spot.** Both models are
bad at upsets (v4 15.3% vs Vegas 17.5% accuracy, both ~0.5 LL above
their chalk LL), but v4 is meaningfully worse than Vegas in the
upset bucket. The clean v4's upset accuracy of 15% is markedly
LOWER than Vegas's 17.5%. The original audit's "v4 catches 56% of
upsets" claim was the largest single piece of evidence the leak was
fabricating signal -- it has now collapsed below even Vegas's
upset rate.

### By v4-confidence quintile

| quintile     | n   | ll_v4  | ll_vegas | delta   | acc_v4 | acc_vegas |
|--------------|-----|--------|----------|---------|--------|-----------|
| 0.50-0.60    | 288 | 0.7006 | 0.6967   | +0.0039 | 0.465  | 0.528     |
| 0.60-0.70    | 290 | 0.6371 | 0.6256   | +0.0115 | 0.666  | 0.659     |
| 0.70-0.80    | 348 | 0.5880 | 0.5733   | +0.0147 | 0.724  | 0.707     |
| 0.80-0.90    | 400 | 0.3768 | 0.3518   | +0.0250 | 0.870  | 0.868     |
| 0.90-1.00    |   0 | --     | --       | --      | --     | --        |

**The 0.90-1.00 row is empty.** No v4 prediction on the joined
1326-game dataset reaches 0.90 confidence under the clean pipeline.
That's the visible signature of the leak collapse: pre-fix v4 had
413 games in this bucket; clean v4 has zero. Mass redistributed
into the 0.50-0.80 range. **The 0.80-0.90 bucket (n=400) is the
single bucket where v4's calibration looks usable but Vegas still
edges it out by +0.025 LL** -- this is where post-clean v4 is
"pretty confident and right" most of the time, but Vegas is still
slightly tighter.

In the lowest-confidence bucket (0.50-0.60, n=288), v4 is barely
above coin-flip accuracy (46.5%) -- *worse than Vegas* (52.8%).
This bucket isn't a weak spot under the +0.02 threshold (delta
is only +0.004) because both models are near 0.69 LL, but the
6.3pp accuracy gap is a notable signal: when v4 is uncertain,
Vegas is more often right about which side to lean.

### By seed-diff magnitude

| seed_diff | n   | ll_v4  | ll_vegas | delta   | acc_v4 | acc_vegas |
|-----------|-----|--------|----------|---------|--------|-----------|
| 0-2       | 310 | 0.6688 | 0.6680   | +0.0007 | 0.558  | 0.565     |
| 3-5       | 367 | 0.6427 | 0.6324   | +0.0103 | 0.635  | 0.638     |
| 6-9       | 385 | 0.5648 | 0.5387   | +0.0261 | 0.727  | 0.743     |
| 10-15     | 264 | 0.3077 | 0.2869   | +0.0208 | 0.913  | 0.913     |

Tightest games (0-2 seed diff) and chalkiest (10-15) are both
near-ties on accuracy, but v4 trails Vegas by +0.021 LL in the
10-15 bucket -- meaning v4 isn't as sharp as Vegas even on
1-vs-16 / 2-vs-15 type matchups. The 6-9 bucket (mid-magnitude
upsets) is the worst by LL of the four.

## Weak spots (clean baseline)

Six buckets cleared the n>=50, ll_delta>=+0.02 weak-spot threshold,
ranked by ll_delta:

1. **round=E8** (n=80, +0.0550) -- Elite Eight v4 LL 0.7105 vs Vegas 0.6555.
2. **chalk_won=upset** (n=360, +0.0543) -- v4 upset accuracy 15.3% vs Vegas 17.5%.
3. **round=S16** (n=152, +0.0266) -- Sweet 16 v4 LL 0.6293 vs Vegas 0.6027.
4. **seed_diff_bucket=6-9** (n=385, +0.0261) -- mid-magnitude upset zone.
5. **v4_confidence_quintile=0.80-0.90** (n=400, +0.0250) -- v4's main "confident" band.
6. **seed_diff_bucket=10-15** (n=264, +0.0208) -- chalkiest matchups.

Cross-bucket pattern: **upset signal is the biggest single weakness.**
The S16/E8 rounds are where regular-season form correlates least
with tournament outcomes (these are the rounds that matter for
bracket-points scoring), and they're where v4 trails Vegas the
most. The seed-diff buckets that overlap with "competitive but not
even" matchups (6-9 spread) are also worse. The 10-15 bucket is a
small absolute LL gap but it's chalk territory where v4 should
mechanically match a sharp implied-prob source.

## Caveats / what this audit does NOT say

1. **SIGMA=11 may be too peaky for tournament games.** A 10-point
   spread under SIGMA=11 implies p=0.82 of the favorite winning, but
   the empirical rate at 10-point spreads in tournament games is
   typically ~0.75-0.78. A larger SIGMA (12-15) would shrink Vegas's
   peakiness and bring its LL up (closer to v4's). Under the
   contaminated baseline this caveat went the wrong direction --
   SIGMA tweaks would have closed the v4-vs-Vegas gap. Under the
   clean baseline, SIGMA tweaks could plausibly flip the sign of
   the overall LL delta (Vegas could end up worse than v4 once it's
   not over-confident). **A SIGMA sensitivity sweep is now
   load-bearing for any "Vegas beats v4" claim** -- the +0.0148 LL
   gap is small enough that it could be SIGMA-dependent. The acc
   gap (-0.7pp) is roughly SIGMA-invariant since accuracy uses
   only argmax; that's the more robust signal.

2. **The original audit's "v4 catches 56% of upsets" was the leak
   talking.** Under the clean pipeline v4's upset accuracy is
   15.3%, which is *worse* than Vegas's 17.5%. The leak was
   peeking at within-season tournament outcomes via the Vegas-
   feature aggregates, and that peek was concentrated on upset
   games (the leak quantified +1.98 vegas_avg_margin for 2024
   UConn -- a tournament champion -- in TODO.md). Recovery step 5
   (re-run swap-decided / swap-candidate evaluations) needs to
   account for this: every "v4 vs candidate" comparison on the
   pre-fix baseline was inflating v4's apparent strength in the
   buckets the leak fed.

3. **2003 missing entirely.** Same as the original audit -- the
   Vegas file `ncaabb03.csv` exists but no rows joined. Likely a
   date-format or team-name resolution issue specific to that year.
   Worth a separate fix-up if a future audit needs 2003 coverage;
   not load-bearing for this conclusion.

4. **Clean v4's 0.90-1.00 confidence bucket is empty.** This means
   the audit cannot evaluate v4's ultra-high-confidence calibration
   on tournament games at all under the clean baseline. The original
   note's claim that "at very high confidence (0.90-1.00) both
   models do well" is no longer testable from this audit.

## What this implies for the queue

1. **538 audit -- still motivated, but framing has changed.** Under
   the original verdict, 538 was the next benchmark because Vegas
   had been "exhausted" (v4 beat it everywhere) and we needed a
   stronger benchmark to localize headroom. Under the clean
   verdict, **Vegas itself is now the benchmark v4 fails to beat,
   and we have 6 localized weak spots to engineer against right
   here** -- before reaching for 538. The 538 audit remains valuable
   as an independent corroboration / benchmark broadening, but the
   immediate engineering signal is "fix v4's S16/E8/upset
   underperformance vs Vegas." Active queue prioritization should
   reflect this: localize-and-fix beats find-a-better-benchmark
   when the cheap benchmark already shows weak spots.

2. **Recovery step 5 is critical.** Any swap-decided model
   (v8 -> v9-C, +43 brkt pts) or swap-candidate (LR ensemble at
   -105 pts, BT-as-feature at -0.0015 LL, weight-sweep at +18 to
   +20 brkt pts) was evaluated on the pre-fix baseline. With the
   leak shift now measured at +0.122 LL (compared to the 0.02-0.05
   estimate when the recovery roadmap was first written), every
   "marginal" rejection in TODO.md's "Tried and rejected" section
   is now within the leak's noise floor -- the rejections may have
   been correct but cannot be assumed correct. Re-eval queue:
   v9-C production swap (top), v8-vs-v9-C bracket-points
   head-to-head, plain BT bracket-points (PR 17), then the
   marginal-band rejections.

3. **SIGMA sensitivity sweep on this audit.** Now that Vegas's
   LL gap is the load-bearing signal (rather than v4's dominance
   over Vegas), a SIGMA sweep over {11, 12, 13, 14, 15} would
   bound how robust the +0.0148 LL gap is. Cheap (~10 sec per
   SIGMA value -- the join is the expensive step and is
   SIGMA-independent).

4. **Single-season variance check** stays in the queue (TODO.md
   active queue #2). Now even more motivated since 22-season
   aggregate v4 mean LL of 0.5588 is on the edge of the
   "production threshold" band; per-season variance behavior
   matters more when the absolute level is mediocre.

## Charts (PNG outputs)

The audit regenerated three PNGs in `output/`. **Note: the original
audit's PNGs in the repo are now stale (they reflect the
contaminated baseline).** The charts here are from the rerun:

- `output/v4_gap_calibration_overall.png` -- overall calibration
  curve. Under the clean baseline, v4's calibration mass shifts
  toward the diagonal in the 0.4-0.8 range but the high-confidence
  bins are mostly empty.
- `output/v4_gap_calibration_by_round.png` -- 6-panel calibration
  by round. The R64 / R32 panels show the cleanest calibration;
  S16 / E8 show v4 systematically over-predicting favorites
  relative to actuals.
- `output/v4_gap_per_bucket_ll_delta.png` -- horizontal bars of
  LL delta per bucket. The mass is now to the RIGHT of zero (Vegas
  better) for every round except F4 / FF / Champ (small n) and a
  handful of seed-diff / confidence buckets that are near-ties.

## Files of record

```
src/audit_v4_gap_vegas.py            -- one-shot driver (~600 LOC, unchanged from PR 18)
tests/test_audit_v4_gap_vegas.py     -- 10 unit tests, all green

output/v4_gap_audit_vegas.json       -- per-bucket metrics + weak_spots (regenerated)
output/v4_gap_calibration_overall.png      (regenerated)
output/v4_gap_calibration_by_round.png     (regenerated)
output/v4_gap_per_bucket_ll_delta.png      (regenerated)
```

The audit script itself was not modified -- only its inputs
(`output/pairwise_v4.csv`) changed. `pyproject.toml` adds
`matplotlib>=3.7` to dependencies for the calibration plots
(unchanged from PR 18).

## Compute

~10 seconds for the audit (unchanged from PR 18). Vegas data load
+ name resolution is the slow part (~5s); per-bucket aggregation
is essentially instant.

## Appendix: original (contaminated) numbers

For the record, the pre-fix audit numbers (PR 18) are preserved
below. **None of these numbers should be used to inform decisions.**

### Original headline numbers (CONTAMINATED)

| metric    | v4     | Vegas  | delta (v4 - Vegas) |
|-----------|--------|--------|--------------------|
| log loss  | 0.4305 | 0.5447 | -0.1142 (v4 better, FALSE) |
| accuracy  | 80.9%  | 70.6%  | +10.3 pp (FALSE)   |
| ECE       | 0.025  | 0.030  | -0.005 (FALSE)     |

### Original by-round (CONTAMINATED)

| round | n   | ll_v4  | ll_vegas | delta   | acc_v4 | acc_vegas |
|-------|-----|--------|----------|---------|--------|-----------|
| R64   | 648 | 0.4040 | 0.5045   | -0.1005 | 0.824  | 0.731     |
| R32   | 319 | 0.4061 | 0.5510   | -0.1449 | 0.828  | 0.712     |
| S16   | 152 | 0.5093 | 0.6027   | -0.0934 | 0.770  | 0.658     |
| E8    |  80 | 0.5419 | 0.6555   | -0.1136 | 0.725  | 0.575     |
| F4    |  42 | 0.4944 | 0.5704   | -0.0760 | 0.762  | 0.714     |
| Champ |  20 | 0.5608 | 0.5603   | +0.0005 | 0.700  | 0.750     |
| FF    |  53 | 0.3485 | 0.6475   | -0.2991 | 0.849  | 0.660     |

### Original by chalk vs upset (CONTAMINATED)

| outcome | n   | ll_v4  | ll_vegas | delta   | acc_v4 | acc_vegas |
|---------|-----|--------|----------|---------|--------|-----------|
| chalk   | 966 | 0.2756 | 0.3530   | -0.0774 | 0.901  | 0.904     |
| upset   | 360 | 0.8461 | 1.0592   | -0.2131 | 0.564  | 0.175     |

The pre-fix "v4 catches 56% of upsets" was the single largest piece
of leak-fabricated signal. Clean v4 catches 15.3% of upsets.

### Original verdict (RETRACTED)

"No weak spots vs Vegas at the n>=50, ll_delta>=+0.02 threshold. v4
outperforms Vegas across every bucket measured. This audit does NOT
localize v4's gap -- Vegas is the wrong benchmark for that. The 538
audit (queued as the immediate next experiment) is now strongly
motivated."

This is incorrect under the clean baseline. v4 has 6 weak spots
versus Vegas, the largest at +0.055 LL on Elite Eight games.
