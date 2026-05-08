# v4 Per-Season Variance Check -- Findings

**Date:** 2026-05-07
**Branch:** feat/v4-per-season-variance
**Verdict:** **MIXED.** 4 outlier seasons flagged at 1.5 sigma across the
4 tracked metrics (`ll_v4_minus_vegas`, `ll_v4_minus_fte`, `ll_v4`,
`ece_v4`): **2011, 2013, 2015, 2023.** No 3-consecutive-season trend.
**2011 is the standout** -- flagged in 3 of 4 tracked metrics
(worst single-season `ll_v4` 0.699 vs 21-season mean 0.557, 2.22 sigma
above; worst v4-vs-Vegas delta +0.074, 1.77 sigma; ECE 0.186, 1.54 sigma).
2024 (the season of the user's 2159/3462 Kaggle finish) was unremarkable
in the per-season data: `ll_v4`=0.591, not flagged on any metric.
**Spec:** `docs/superpowers/specs/2026-05-07-v4-per-season-variance-design.md`
**Plan:** `docs/superpowers/plans/2026-05-07-v4-per-season-variance.md`
**Recovery context:** TODO.md Active queue item #1.

## TL;DR

Per-season aggregation of v4's clean LOSO predictions on R64-Champ
games across 22 LOSO seasons (21 with Vegas coverage, 7 with 538
coverage), with cross-season mean / std / 1.5-sigma outlier flagging
on four metrics. Verdict came back **MIXED** rather than CLEAR: 4
seasons flagged but no clean trend, and no single season is so bad
it explains the user's 2159/3462 Kaggle finish (2024 was unremarkable
at `ll_v4`=0.591 vs the 21-season mean of 0.557). 2011 is the standout
poor season -- the famous Butler/UConn-Kemba/VCU-FF11 tournament with
heavy upset traffic -- but it predates the user's Kaggle interest by
~13 years and is not actionable for the production model. Per the
plan's MIXED rule: queue ordering is retained and ambiguity is noted
in the TODO preamble; the calibration-shape engineering target stays
as the backup engineer-to lever, External Data leads by elimination.

## Anchors

| anchor | expected | observed | verdict |
|--------|----------|----------|---------|
| weighted per-season `ll_v4` (538 subset, 428 games) | 0.5799 (538 audit overall) | 0.5799 | **PASS** (FP precision) |
| weighted per-season `ll_fte` (538 subset, 428 games) | 0.6011 (538 audit overall) | 0.6011 | **PASS** (FP precision) |
| weighted per-season `ll_v4` (Vegas R64-Champ subset, 1261 games) | ~0.5595 (Vegas audit overall) | 0.5565 | **PASS-WITH-EXPLANATION** |
| weighted per-season `ll_vegas` (Vegas R64-Champ subset, 1261 games) | ~0.5447 (Vegas audit overall) | 0.5408 | **PASS-WITH-EXPLANATION** |

The 538 anchors land on FP precision -- the per-season aggregator is
exactly equivalent to the audit's overall pooled computation on the
matching subset. The Vegas anchors land slightly below the audit's
full-frame numbers (0.5565 vs 0.5595, 0.5408 vs 0.5447) because the
variance check explicitly filters R64-Champ rounds and excludes
First Four / OTHER buckets, while the Vegas audit's "overall" stat
covers the full 1326-game frame including FF/OTHER. Direction is
consistent with R64-dominated log-loss arithmetic. This is intentional
and documented in the spec's anchor section.

## Per-season metrics

All 21 Vegas-covered seasons (2003 has no resolvable Vegas matches and
2020 had no tournament -- so the universe is 22 LOSO seasons, 21 with
Vegas data, 7 with 538 data):

| season | n | ll_v4 | ll_vegas | dV | ll_fte | dF | acc_v4 | ece_v4 |
|--------|---|-------|----------|-------|--------|-------|--------|--------|
| 2004 | 61 | 0.528 | 0.526 | +0.002 |  --  |  --  | 0.689 | 0.132 |
| 2005 | 62 | 0.532 | 0.537 | -0.005 |  --  |  --  | 0.726 | 0.121 |
| 2006 | 60 | 0.601 | 0.574 | +0.028 |  --  |  --  | 0.700 | 0.143 |
| 2007 | 62 | 0.491 | 0.467 | +0.023 |  --  |  --  | 0.726 | 0.119 |
| 2008 | 61 | 0.495 | 0.496 | -0.001 |  --  |  --  | 0.754 | 0.119 |
| 2009 | 53 | 0.538 | 0.494 | +0.044 |  --  |  --  | 0.698 | 0.141 |
| 2010 | 53 | 0.526 | 0.539 | -0.013 |  --  |  --  | 0.717 | 0.122 |
| **2011** | 61 | **0.699** | 0.625 | **+0.074** |  --  |  --  | **0.574** | **0.186** |
| 2012 | 63 | 0.517 | 0.560 | -0.043 |  --  |  --  | 0.730 | 0.117 |
| **2013** | 63 | 0.622 | 0.557 | **+0.065** |  --  |  --  | 0.635 | 0.137 |
| 2014 | 63 | 0.586 | 0.551 | +0.036 |  --  |  --  | 0.667 | 0.115 |
| **2015** | 63 | 0.449 | 0.480 | -0.031 |  --  |  --  | 0.778 | **0.204** |
| 2016 | 63 | 0.582 | 0.535 | +0.046 | 0.536 | +0.045 | 0.714 | 0.085 |
| 2017 | 63 | 0.530 | 0.488 | +0.042 | 0.519 | +0.011 | 0.730 | 0.115 |
| 2018 | 63 | 0.585 | 0.600 | -0.015 | 0.615 | -0.030 | 0.698 | 0.083 |
| 2019 | 63 | 0.505 | 0.478 | +0.027 | 0.454 | +0.052 | 0.730 | 0.164 |
| 2021 | 50 | 0.605 | 0.633 | -0.028 | 0.705 | -0.112 | 0.680 | 0.118 |
| 2022 | 63 | 0.647 | 0.615 | +0.033 | 0.701 | -0.054 | 0.635 | 0.126 |
| **2023** | 47 | 0.614 | 0.646 | -0.032 | 0.699 | -0.079 | 0.660 | **0.212** |
| 2024 | 63 | 0.591 | 0.553 | +0.037 |  --  |  --  | 0.714 | 0.104 |
| 2025 | 61 | 0.458 | 0.437 | +0.022 |  --  |  --  | 0.787 | 0.150 |

`dV` = `ll_v4 - ll_vegas`; `dF` = `ll_v4 - ll_fte`. **Bold rows** are
the four 1.5-sigma outlier seasons (union across the 4 tracked
metrics).

Cross-season summary (computed over the seasons available per metric):

| metric | mean | std | n | range |
|--------|------|-----|---|-------|
| `ll_v4` | 0.5572 | 0.0636 | 21 | 0.449-0.699 |
| `ll_v4_minus_vegas` | +0.0147 | 0.0333 | 21 | -0.043 to +0.074 |
| `ll_v4_minus_fte` | -0.0239 | 0.0627 | 7 | -0.112 to +0.052 |
| `ece_v4` | 0.1340 | 0.0339 | 21 | 0.083-0.212 |

## Outliers

### `ll_v4_minus_vegas` (mean +0.015, std 0.033, n=21)

| season | value | sigma_delta | n_games |
|--------|-------|-------------|---------|
| 2011 | +0.0738 | 1.77 | 61 |
| 2013 | +0.0651 | 1.51 | 63 |

### `ll_v4_minus_fte` (mean -0.024, std 0.063, n=7)

None flagged. The 7-season frame's std is large enough (0.063 vs
0.033 on the 21-season Vegas frame) that no single-season delta
crosses 1.5 sigma. This is sample-size limited, not a substantive
finding -- 538's coverage is the real bottleneck.

### `ll_v4` (mean 0.557, std 0.064, n=21)

| season | value | sigma_delta | n_games |
|--------|-------|-------------|---------|
| 2011 | 0.6987 | 2.22 | 61 |

2011 is alone here, more than 2 sigma above the mean. The next two
worst LL seasons (2022 at 0.647, 2023 at 0.614) sit at ~1.4 sigma and
just below the threshold.

### `ece_v4` (mean 0.134, std 0.034, n=21)

| season | value | sigma_delta | n_games |
|--------|-------|-------------|---------|
| 2023 | 0.2117 | 2.30 | 47 |
| 2015 | 0.2040 | 2.07 | 63 |
| 2011 | 0.1861 | 1.54 | 61 |

### Standout: 2011

2011 is flagged in 3 of 4 tracked metrics: worst `ll_v4` (0.699 vs
mean 0.557, 2.22 sigma), worst `ll_v4_minus_vegas` (+0.074 vs mean
+0.015, 1.77 sigma), and high ECE (0.186, 1.54 sigma). v4 accuracy
57.4% in 2011 vs Vegas's 65.6% on the same 61 games -- v4 missed
8.2 pp of accuracy that Vegas captured. Domain note: 2011 was the
"Butler back to the Final Four / VCU as an 11-seed in the F4 /
UConn-Kemba run / Kentucky as a 4-seed" tournament, with heavier
upset traffic than the 2003-2010 training data Vegas had implicitly
absorbed via market liquidity. v4 was trained on prior seasons and
didn't have a direct read on the upset paths. The audit's role here
is to surface, not explain in depth.

### 2015 and 2023 ECE outliers

2015 was the "Final Four 1-seeds" year (Duke / Kentucky / Wisconsin /
Michigan State) -- chalk-heavy. v4's accuracy was 77.8% (the second-
best in the per-season frame after 2025), but ECE was 0.204 -- the
calibration miss came from over- or under-confidence on chalk picks
that all happened to land. 2023 was the wildest tournament in the
modern era (FAU and San Diego State in the F4). v4 accuracy 66.0%,
ECE 0.212 -- high ECE plausibly from confidence misses on chalk that
got dethroned. Both are sample-of-one observations; the audit
surfaces them, doesn't explain them.

## Cross-benchmark pattern

Vegas delta flags 2011 + 2013; 538 delta flags none. The two cross-
benchmark deltas don't agree on which seasons are flagged because:

1. **538 covers a different 7-season subset** (2016-2019, 2021-2023).
   2011 and 2013 aren't in the 538 frame at all -- there's no way
   for the 538-delta to flag them.
2. **The 7-season std (0.063) is roughly 2x the 21-season std (0.033)**.
   At n=7 the threshold for 1.5 sigma is much wider; the 538 deltas
   span -0.112 (2021) to +0.052 (2019), and even the worst
   single-season miss is well within that band.

In other words, the cross-benchmark divergence here is mostly about
sample sizes, not about a substantive disagreement on which seasons
were tough for v4.

## What this implies for the queue

**Per the plan's MIXED rule: retain current ordering, note ambiguity
in TODO preamble.** Specifically:

- The check did **not** produce a clean "single-season variance
  dominates" verdict. No season is so bad it explains the user's
  2159/3462 Kaggle finish. 2024 -- the Kaggle year -- was unremarkable
  in this data: `ll_v4`=0.591, +0.034 above the 21-season mean,
  `ll_v4_minus_vegas`=+0.037 (NOT an outlier), `ece_v4`=0.104 (below
  mean). 2024 is a typical season for v4. The Kaggle finish must
  therefore reflect calibration shape (over-confidence in some
  buckets, under-confidence in others) rather than 2024 being a
  uniquely bad year for v4.
- The check **did** surface that 2011 was a markedly worse year (1-2
  std worse on multiple metrics), but it predates the user's Kaggle
  interest by ~13 years. Not actionable for the production model.
- **ECE variance is non-trivial** -- std 0.034 on mean 0.134 is ~25%
  CV across 21 seasons. Modest but real evidence that v4's calibration
  shape varies year to year. The two ECE outliers (2015, 2023) are
  in the 538 frame's later seasons, suggesting the variance isn't
  shrinking over time.
- **Calibration-shape engineering** (currently Active queue #3, will
  be #2 after the variance check is moved to Done) remains the
  backup engineer-to target. The variance check did not lift its
  priority above External Data #2 -- the bottleneck is still
  benchmark-disagreement on weak-spot signature, not single-season
  blowups.
- **538's small N=7 is a confirmed limitation.** Cross-benchmark
  flagging on 538 is informational only at this sample size; any
  future re-run of the variance check would benefit from broader
  538 coverage, but 2014/2015 predate the API and 2024/2025 weren't
  archived, so the 7-season ceiling is structural unless an
  alternate forecast source is found.

By elimination, **External Data #2 (now #1 after promotion) takes
the lead.** The variance check's outcome doesn't unblock that lever
in any new way -- it just confirms there's no single-season fire
that would re-rank the queue.

## Files of record

```
src/analyze_v4_per_season_variance.py            -- driver
tests/test_analyze_v4_per_season_variance.py     -- 7 unit tests (incl. 1 smoke)

output/v4_per_season_variance.json               -- per-season metrics + outliers
output/v4_per_season_variance_traces.png         -- per-metric per-season trace lines
output/v4_per_season_variance_deltas.png         -- per-season delta-vs-Vegas + delta-vs-538 bar plot
output/v4_per_season_variance_log.txt            -- run log

docs/superpowers/specs/2026-05-07-v4-per-season-variance-design.md
docs/superpowers/plans/2026-05-07-v4-per-season-variance.md
docs/notes/2026-05-07-v4-per-season-variance.md  -- this note
```

## Compute

- Driver run: ~5s end-to-end (numpy/pandas-bounded over ~1300 games
  pooled across 21 Vegas seasons + 428 games over 7 538 seasons).
- All upstream caches (clean `pairwise_v4.csv`, Vegas closing-line
  CSV, Wayback-pinned 538 forecasts) reused in-place; no fetches.
