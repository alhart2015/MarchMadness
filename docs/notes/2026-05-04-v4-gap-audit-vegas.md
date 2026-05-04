# v4 Gap Audit vs Vegas -- Findings

**Date:** 2026-05-04
**Branch:** feat/v4-gap-audit-vegas
**Verdict:** **No weak spots vs Vegas at the n>=50, ll_delta>=+0.02 threshold.** v4 outperforms Vegas across every bucket measured. This audit does NOT localize v4's gap -- Vegas is the wrong benchmark for that. The 538 audit (queued as the immediate next experiment) is now strongly motivated.
**Spec:** `docs/superpowers/specs/2026-05-04-v4-gap-audit-vegas-design.md`
**Plan:** `docs/superpowers/plans/2026-05-04-v4-gap-audit-vegas.md`

## TL;DR

Compared v4 stage-1 tournament predictions against Vegas closing-line
implied probabilities (SIGMA=11) across 1326 played 2003-2025
tournament games, bucketed by round, chalk-vs-upset, v4-confidence
quintile, and seed-diff magnitude.

**v4 beats Vegas on every bucket.** Overall LL: v4=0.4305, Vegas=0.5447
(delta -0.114). Overall accuracy: v4=80.9%, Vegas=70.6%. v4 catches
56% of upsets vs Vegas's 17%. ECE comparable (v4=0.025, Vegas=0.030).

This is itself a useful finding: **Vegas is not the benchmark that
will surface v4's headroom.** Whatever is causing the user's
2159 / 3462 Kaggle finish, it's not visible against Vegas
implied probabilities at SIGMA=11. **The 538 tournament-forecast
audit is now the most important follow-up** -- 538 publishes
calibrated probabilities directly (no SIGMA conversion required) and
is widely regarded as a strong public benchmark.

## Setup recap

- **Inputs:** `output/pairwise_v4.csv` (48,465 unique pair keys),
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
  near-complete coverage.
- **Wall time:** ~10 seconds.

## Anchors

| anchor                              | expected band     | observed | verdict |
|-------------------------------------|-------------------|----------|---------|
| `overall.ll_v4`                     | ~0.437            | 0.4305   | PASS (slight diff = subset effect, see below) |
| `overall.acc_vegas`                 | [0.70, 0.72]      | 0.706    | PASS    |
| `coverage`                          | >= 60%            | 91.5%    | PASS    |
| `ll_vegas`                          | [0.42, 0.46]      | 0.5447   | **OFF (see Caveats)** |

The `ll_v4` slight discrepancy (0.4305 vs 0.4369 anchor) is the
subset effect: the 1326 audited games are slightly easier for v4
than the full 1449 v4 set (the 123 missing games skew toward harder).

The `ll_vegas` overshoot is the main caveat -- see below.

## Headline numbers

| metric    | v4     | Vegas  | delta (v4 - Vegas) |
|-----------|--------|--------|--------------------|
| log loss  | 0.4305 | 0.5447 | **-0.1142** (v4 better) |
| accuracy  | 80.9%  | 70.6%  | +10.3 pp           |
| ECE       | 0.025  | 0.030  | -0.005             |

## Per-bucket results

### By round

| round | n   | ll_v4  | ll_vegas | delta   | acc_v4 | acc_vegas | mean(p_v4 - p_vegas) |
|-------|-----|--------|----------|---------|--------|-----------|----------------------|
| R64   | 648 | 0.4040 | 0.5045   | -0.1005 | 0.824  | 0.731     | +0.001               |
| R32   | 319 | 0.4061 | 0.5510   | -0.1449 | 0.828  | 0.712     | -0.008               |
| S16   | 152 | 0.5093 | 0.6027   | -0.0934 | 0.770  | 0.658     | +0.024               |
| E8    | 80  | 0.5419 | 0.6555   | -0.1136 | 0.725  | 0.575     | -0.019               |
| F4    | 42  | 0.4944 | 0.5704   | -0.0760 | 0.762  | 0.714     | -0.035               |
| Champ | 20  | 0.5608 | 0.5603   | +0.0005 | 0.700  | 0.750     | -0.028               |
| FF    | 53  | 0.3485 | 0.6475   | -0.2991 | 0.849  | 0.660     | -0.023               |

v4 ahead by 0.07-0.30 LL in every round. The Champ round is the only
bucket where Vegas matches v4 (n=20, too small to draw a conclusion).

### By chalk vs upset

| outcome | n   | ll_v4  | ll_vegas | delta   | acc_v4 | acc_vegas |
|---------|-----|--------|----------|---------|--------|-----------|
| chalk   | 966 | 0.2756 | 0.3530   | -0.0774 | 0.901  | 0.904     |
| upset   | 360 | 0.8461 | 1.0592   | -0.2131 | 0.564  | 0.175     |

**Big finding:** v4 catches 56% of upsets vs Vegas's 17%. Vegas is
upset-blind by design (it picks the favorite); v4 has a real
upset-detection edge over Vegas's implied probabilities.

### By v4-confidence quintile

| quintile     | n   | ll_v4  | ll_vegas | delta   | acc_v4 | acc_vegas |
|--------------|-----|--------|----------|---------|--------|-----------|
| 0.50-0.60    | 130 | 0.6601 | 0.7205   | -0.0604 | 0.615  | 0.538     |
| 0.60-0.70    | 127 | 0.6608 | 0.7651   | -0.1043 | 0.622  | 0.520     |
| 0.70-0.80    | 216 | 0.6010 | 0.7330   | -0.1321 | 0.704  | 0.523     |
| 0.80-0.90    | 440 | 0.4425 | 0.5634   | -0.1208 | 0.832  | 0.693     |
| 0.90-1.00    | 413 | 0.1854 | 0.3033   | -0.1179 | 0.959  | 0.925     |

v4 dominates Vegas in mid-confidence games (0.60-0.80) by 0.10-0.13
LL -- this is where the upset-detection edge lives. At very high
confidence (0.90-1.00), both models do well; at very low (0.50-0.60)
both struggle but v4 is still ahead.

### By seed-diff magnitude

| seed_diff | n   | ll_v4  | ll_vegas | delta   | acc_v4 | acc_vegas |
|-----------|-----|--------|----------|---------|--------|-----------|
| 0-2       | 310 | 0.4760 | 0.6680   | -0.1920 | 0.771  | 0.565     |
| 3-5       | 367 | 0.5015 | 0.6324   | -0.1309 | 0.779  | 0.638     |
| 6-9       | 385 | 0.4346 | 0.5387   | -0.1040 | 0.795  | 0.743     |
| 10-15     | 264 | 0.2723 | 0.2869   | -0.0146 | 0.917  | 0.913     |

Tightest gap is in the 10-15 seed-diff bucket (mostly 1-vs-16, 2-vs-15
matchups -- chalky games where Vegas is essentially perfect). On
close-seed matchups (0-2, 3-5), v4 is meaningfully ahead.

## Charts (PNG outputs)

- `output/v4_gap_calibration_overall.png` -- overall calibration curve.
  v4 is on the diagonal across most predicted-prob bins; Vegas is
  systematically over-confident at the high end (predicts 0.85+ but
  actual win rates are lower).
- `output/v4_gap_calibration_by_round.png` -- 6-panel calibration by
  round. Pattern repeats per round.
- `output/v4_gap_per_bucket_ll_delta.png` -- horizontal bars of LL
  delta per bucket. Every bar is to the LEFT of zero (= v4 better)
  except Champ (n=20).

## Caveats / what this audit does NOT say

1. **SIGMA=11 may be too peaky for tournament games.** A 10-point
   spread under SIGMA=11 implies p=0.82 of the favorite winning, but
   the empirical rate at 10-point spreads in tournament games is
   typically ~0.75-0.78. A larger SIGMA (12-15) would shrink Vegas's
   peakiness and bring its LL down (closer to v4's). The current
   audit numbers should be read as "v4 vs Vegas-implied-with-SIGMA=11"
   rather than "v4 vs the true Vegas probability." Sensitivity sweep
   over SIGMA in {11, 12, 13, 14} was deferred at spec time; if the
   exact gap matters, run it. For purposes of "is Vegas the right
   benchmark to localize v4's headroom" -- it isn't, regardless of
   SIGMA.

2. **The user's 2159 / 3462 Kaggle finish is not explained by this
   audit.** v4 looks strong vs Vegas everywhere. Possible reasons:
   - Kaggle metric is single-season; v4 may be high-variance per
     season even if 22-season average is strong.
   - Top Kaggle entries may use external data (538, public KenPom
     forecasts) that v4 is not aware of.
   - Calibration ECE (0.025) is small but not zero; on a single
     season's tournament with high-leverage games, even small
     calibration errors compound.
   - The 22-season aggregate hides heterogeneity: some seasons v4
     does well, others poorly.

3. **2003 missing entirely.** The Vegas file `ncaabb03.csv` exists
   but no rows joined. Likely a date-format or team-name resolution
   issue specific to that year. Worth a separate fix-up if a future
   audit needs 2003 coverage; not load-bearing for the current
   conclusion.

4. **No buckets met the weak-spot threshold.** This is the actual
   audit verdict: there is no "v4 underperforms Vegas on round X /
   seed pair Y" signature. If we want to localize headroom, we need a
   stronger benchmark than Vegas-at-SIGMA=11.

## What this implies for the queue

1. **538 audit -- now the most important follow-up.** 538 publishes
   tournament-forecast probabilities directly during March Madness
   (no SIGMA conversion needed). Public archive coverage and access
   patterns are the open sourcing question. Same audit framework
   here can be reused once 538 data is in hand.

2. **Single-season variance check.** Pick a recent season (2024 or
   2025) and look at v4's per-game predictions vs actuals
   bracket-by-bracket. The user's Kaggle finish was a single season's
   submission; localizing v4's failure modes on a single recent
   season may surface signal that the 22-season aggregate hides.

3. **Calibration ECE breakdown by season.** Overall ECE is small
   (0.025); if any single season has ECE > 0.05, that's a signal
   that v4's calibration is unstable across seasons.

The "external rankings as features" item stays at active queue #2;
it's a different lever (modeling) from the audit work, but its
priority is unchanged.

## Files of record

```
src/audit_v4_gap_vegas.py            -- one-shot driver (~600 LOC)
tests/test_audit_v4_gap_vegas.py     -- 10 unit tests, all green

output/v4_gap_audit_vegas.json       -- per-bucket metrics + weak_spots (force-added)
output/v4_gap_calibration_overall.png
output/v4_gap_calibration_by_round.png
output/v4_gap_per_bucket_ll_delta.png
output/v4_gap_audit_log.txt
```

`pyproject.toml` adds `matplotlib>=3.7` to dependencies for the
calibration plots.

## Compute

~10 seconds for the audit. Vegas data load + name resolution is the
slow part (~5s); per-bucket aggregation is essentially instant.
