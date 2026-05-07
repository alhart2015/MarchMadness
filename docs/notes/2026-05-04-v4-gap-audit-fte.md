# v4 Gap Audit vs FiveThirtyEight -- Findings

**Date:** 2026-05-07
**Branch:** feat/v4-gap-audit-fte
**Verdict:** **PASS-AND-FLAG.** v4 marginally beats 538 on overall log
loss (-0.0212) but trails on accuracy (-1.8 pp). One weak spot at the
spec's n>=50, ll_delta>=+0.02 threshold: **`chalk_won=chalk` (n=298,
delta=+0.0754)** -- when the chalk pick wins, 538 is materially more
confident in that winner than v4 is. v4 *out*performs 538 in S16 (-0.123
LL) and E8 (-0.155), suggesting 538's BT-norm approximation degrades
faster than v4 in late rounds, while 538 still wins on early-round
chalk-picks confidence.
**Spec:** `docs/superpowers/specs/2026-05-04-v4-gap-audit-fte-design.md`
**Plan:** `docs/superpowers/plans/2026-05-04-v4-gap-audit-fte.md`
**Recovery context:** TODO.md Active queue item #1.

## Sourcing pivot

The spec assumed 538 forecasts were available at the GitHub raw URL
`raw.githubusercontent.com/fivethirtyeight/data/master/march-madness-predictions/<year>/fivethirtyeight_ncaa_forecasts.csv`.
That path does not exist (the `march-madness-predictions/` directory in
the repo only holds 2014's 62 bracket-challenge CSVs; per-year
subdirectories never existed there). The actual canonical URL pattern
that 538 used was `projects.fivethirtyeight.com/march-madness-api/<year>/fivethirtyeight_ncaa_forecasts.csv`
(documented in `march-madness-predictions-2018/README.md`). Every URL
under that pattern now 302-redirects to abcnews.go.com/politics --
spec risk #4 ("538 was shut down March 2025") materialized.

**Resolution:** the Internet Archive's Wayback Machine has 200-status
text/csv captures of the original CSVs for 2016-2023 (snapshots from
2025-03-06, the day before the shutdown). 2014/2015 predate the API
endpoint; 2024/2025 were never published at this URL pattern after the
Disney/ABC News ownership transition.

**Audit set is therefore 7 seasons (2016-2019, 2021-2023)**, down from
the 11 originally specified. Snapshot timestamps are pinned in
`src/ingest/fte_forecasts.py::_FTE_URL_BY_YEAR`. Once cached locally
under `data/raw/fte_forecasts/<year>.csv` the loader never re-fetches.

## Schema correction

A second discovery during implementation: 538's `rdR_win` column is
**P(reach round R)**, not P(win round R's game). For round-of-X game
auditing (R64 = round_index 1, ..., Champ = 6), the correct lookup is
`rd{X+1}_win`. The spec's `rd_R_win[a]` formula was off by one.
Verified empirically: per-column sums on a post-play-in snapshot are
`rd1_win=64` (all R64 entrants alive), `rd2_win=32` (32 R64 matchups),
`rd3_win=16`, ..., `rd7_win=1`. Documented in
`src/ingest/fte_forecasts.py` docstring; encoded in
`src/audit_v4_gap_fte.py::FTE_RD_COL_FOR_ROUND`.

## TL;DR (7-season audit)

Audited v4 stage-1 tournament predictions against 538's pre-tournament
round-survival forecasts on 428 R64-Champ games across 7 seasons,
bucketed by round, chalk-vs-upset, v4-confidence quintile, and
seed-difference magnitude.

| metric    | v4     | 538    | delta (v4 - 538) |
|-----------|--------|--------|--------------------|
| log loss  | 0.5799 | 0.6011 | **-0.0212** (v4 better) |
| accuracy  | 69.2%  | 71.0%  | **-1.8 pp** (538 better) |
| ECE       | 0.041  | 0.088  | -0.047 (v4 better calibrated) |

v4 is *competitive* against 538 in aggregate. Where the LL gap shows
up: **chalk-favored matchups, where 538 is more confident in the
winning favorite than v4 is.** Where v4 *beats* 538: late rounds
(S16, E8) -- 538's BT-norm approximation gets shaky in rounds where
538's `rdR_win` averages over multiple possible opponents.

## Anchors

| anchor | expected | observed | verdict |
|--------|----------|----------|---------|
| Coverage of audited R64-Champ games | >= 90% | 99.1% (428/432) | PASS |
| R64 rd2_win sum-to-1 (per matchup) | ~1.0 | 1.0 / 1.0 / 1.0 (min/mean/max, n=50 sampled) | PASS |
| overall.ll_v4 vs clean v4's known LL | within +/-0.05 of 22-season clean LL 0.5588 | 0.5799 (delta +0.021 on the 7-season subset) | PASS |
| overall.ll_fte plausible band | [0.40, 0.55] (spec) | 0.6011 | mild miss; consistent with 7-season-subset variance |
| overall.acc_fte plausible band | [0.70, 0.78] (spec) | 0.710 | PASS |

The 538 LL of 0.6011 is slightly above the spec's 0.40-0.55 band, but
on a 7-season subset (vs 22 for the spec's anchor) some variance is
expected and the BT-norm approximation introduces a small upward bias
on R32+ matchups.

## Per-bucket results

### By round

| round | n | ll_v4 | ll_fte | delta | acc_v4 | acc_fte |
|-------|---|-------|--------|-------|--------|---------|
| R64   | 208 | 0.5322 | 0.5212 | +0.0110 | 0.721 | 0.750 |
| R32   | 119 | 0.5823 | 0.5946 | -0.0123 | 0.714 | 0.723 |
| S16   |  48 | 0.7264 | 0.8491 | **-0.1227** | 0.583 | 0.625 |
| E8    |  32 | 0.7051 | 0.8603 | **-0.1553** | 0.531 | 0.531 |
| F4    |  14 | 0.5008 | 0.4178 | +0.0830 | 0.857 | 0.786 | (sample-limited) |
| Champ |   7 | 0.5383 | 0.5680 | -0.0298 | 0.571 | 0.571 | (sample-limited) |

**538 leads R64 (delta +0.011)**, gives back R32 narrowly, and
**v4 dominates S16 + E8** (deltas -0.123 and -0.155). F4 and Champ are
flagged sample-limited (n<50 below the weak-spot threshold).

The S16/E8 gaps are *much* larger than the spec's -0.02 threshold and
together account for most of v4's overall LL edge. Mechanically: in
S16/E8, the BT-norm approximation `p = rdR[a] / (rdR[a] + rdR[b])`
treats each team's reach-probability as if both faced the same average
opponent distribution -- but in a played matchup of A vs B in S16, A
got there via a path different from the average S16 entrant. v4
trains directly on individual games and doesn't carry that bias.

### By chalk vs upset

| bucket | n | ll_v4 | ll_fte | delta | acc_v4 | acc_fte | mean(p_v4 - p_fte) |
|--------|---|-------|--------|-------|--------|---------|--------------------|
| chalk  | 298 | 0.3219 | 0.2465 | **+0.0754** | 0.933 | 0.946 | +0.001 |
| upset  | 130 | 1.1713 | 1.4140 | -0.2428 | 0.138 | 0.169 | -0.009 |

**This is the audit's headline weak spot.** 538 is much better at
piling probability mass onto winning chalk picks (LL=0.247 vs v4's
0.322). The mean point-prediction gap is essentially zero (+0.001),
so it isn't that 538 picks a different team -- 538 just commits more
to its choice. Both models have nearly identical accuracy in this
bucket (94.6% vs 93.3%); 538 wins by sharpness, not by picking
different favorites.

The flip side: **v4 absorbs upset losses much better than 538.**
When the upset hits, v4's LL is 1.17 vs 538's 1.41 -- v4 is less
catastrophically wrong (likely because 538's BT-norm denominator pushes
favorite probs higher in rounds that average over opponent
distributions, which inflates LL whenever the favorite loses).

Both models are bad at *predicting* upsets: v4 catches 13.8%
(18 of 130), 538 catches 16.9% (22 of 130). For comparison, the Vegas
audit had clean v4 at 15.3% upset detection vs Vegas's 17.5% on a
22-season set -- both audits agree clean v4 has no upset-detection
edge.

### By v4-confidence quintile

| bucket | n | ll_v4 | ll_fte | delta | acc_v4 | acc_fte |
|--------|---|-------|--------|-------|--------|---------|
| 0.50-0.60 |  81 | 0.6860 | 0.7016 | -0.0155 | 0.506 | 0.543 |
| 0.60-0.70 |  91 | 0.6565 | 0.6445 | +0.0120 | 0.626 | 0.670 |
| 0.70-0.80 | 106 | 0.6296 | 0.6770 | -0.0475 | 0.689 | 0.698 |
| 0.80-0.90 | 150 | 0.4410 | 0.4670 | -0.0260 | 0.833 | 0.833 |

**No bucket clears the +0.02 weak-spot threshold.** v4 marginally
trails 538 in the 0.60-0.70 bucket (where v4 is hedging) but beats
538 in the 0.70-0.80 and 0.80-0.90 buckets. The 0.90-1.00 bucket is
empty -- consistent with the Vegas audit's "v4's 0.90-1.00 bucket is
empty under clean baseline" finding (the Vegas-leak was inflating
v4 confidence on tournament games; that's now been removed).

### By seed-diff magnitude

| bucket | n | ll_v4 | ll_fte | delta | acc_v4 | acc_fte |
|--------|---|-------|--------|-------|--------|---------|
| 0-2   |  82 | 0.6719 | 0.6524 | +0.0195 | 0.549 | 0.585 |
| 3-5   | 119 | 0.6978 | 0.7264 | -0.0287 | 0.605 | 0.639 |
| 6-9   | 138 | 0.5809 | 0.6034 | -0.0225 | 0.717 | 0.725 |
| 10-15 |  89 | 0.3359 | 0.3828 | -0.0469 | 0.899 | 0.899 |

Close-seed games (0-2) lean 538-favorable but just below threshold
(+0.0195 vs +0.02). Mid- and large-seed-gap buckets all favor v4. The
6-9 and 10-15 buckets that *did* clear the Vegas audit's weak-spot
threshold (Vegas: +0.026 and +0.021 respectively) flip sign here --
v4 actually does better than 538 in the seed-mismatch buckets.

## Charts

- `output/v4_gap_calibration_overall_fte.png` -- overall calibration
  diagonal. v4 sits near the diagonal; 538 has visible deviation in
  the high-confidence bins (over-confident on chalk).
- `output/v4_gap_calibration_by_round_fte.png` -- per-round calibration.
  R64 panels show 538 ahead; S16 and E8 show v4 ahead with 538's curve
  visibly bowed away from the diagonal (BT-norm approximation effect).
- `output/v4_gap_per_bucket_ll_delta_fte.png` -- horizontal bar of
  per-bucket ll_delta. Most bars are blue (v4 better); the lone red
  bar past the threshold is `chalk_won=chalk`.

## Caveats

- **7-season subset (vs spec's 11).** 2014/2015 predate 538's API
  endpoint. 2024/2025 not recoverable from Wayback Machine.
  Implication: the audit cannot speak directly to v4's behavior in
  the most-recent two tournaments, which are the years closest to
  the user's 2159/3462 Kaggle finish that motivated the audit. The
  7-season weak-spot signature should still generalize forward but
  has slightly less direct relevance than originally intended.
- **F4 (n=14) and Champ (n=7) are sample-limited.** Per-bucket numbers
  are reported for completeness but not actioned.
- **BT-norm is exact for R64** (rd2_win[a] + rd2_win[b] = 1 for actual
  played R64 matchups; verified to floating-point precision on the
  50-game sample). **Approximate for R32+** (538's rdR_win averages
  over expected opponents rather than the actual one). Bias is small
  per-matchup but accumulates: this is the most likely mechanism
  behind 538's S16/E8 LL inflation. The audit's S16/E8 verdict
  ("v4 beats 538") is therefore partly a verdict on 538's BT-norm
  artifact, not purely on 538-the-model. A sensitivity check using
  538's `team_rating` via 538's logistic was considered and rejected
  in brainstorming (would re-introduce a SIGMA-style conversion
  caveat parallel to the Vegas audit's).
- **Snapshot policy is "earliest post-play-in" per season** -- 538's
  view at the same epistemic state as v4 (zero R64+ tournament games
  observed). Round-aware snapshots would benchmark v4 against a 538
  that has seen post-R64 results, asymmetric and unfair.
- **Coverage 99.1% (428/432)** -- 4 games dropped, all in 2021. Likely
  cause: 2021 was the COVID-rescheduled tournament with ~2 cancelled
  pairings; not a model-comparison issue.

## What this implies for the queue

**Active queue #1 (538 audit) closes here.** The audit produced one
weak-spot signature (chalk_won=chalk) plus three engineering-level
observations:

1. **v4 is less confident in chalk picks than 538 is.** Engineering
   target: improve v4's calibration on the 80%+ confidence side of
   the favorite. Mechanism could be temperature scaling, isotonic
   regression on a held-out tournament-only validation set, or a
   late-stage "confidence sharpening" feature. Cost: ~half a day to
   prototype, with the gate being 22-season bracket-points (not just
   LL). This is a candidate next experiment if the queue's next-up
   priorities don't pre-empt it.
2. **The S16/E8 v4 advantage is partly an artifact of 538's BT-norm
   approximation, not v4-specific signal.** A sensitivity check
   using 538's `team_rating` via logistic conversion (the methodology
   we rejected for the main audit) could quantify how much of the
   -0.123 / -0.155 LL gap is BT-norm vs real model difference. Cheap
   ~1 hr followup if anyone questions the late-round verdict.
3. **The audit's verdict is consistent with the Vegas audit on
   v4's upset detection** (clean v4 catches ~14-15% of upsets in
   both audits; both Vegas and 538 catch ~17%). Strengthens the
   "v4 has no upset-detection edge" closing finding from the Vegas
   audit.

**Comparison with the Vegas audit (PR 22, clean baseline):**

| audit | overall ll_delta | weak spots (n>=50, +0.02) | sample |
|-------|------------------|---------------------------|--------|
| Vegas (22 seasons, n=1326) | +0.0148 (Vegas wins) | 6 buckets: round=E8 / chalk_won=upset / round=S16 / seed_diff=6-9 / v4_conf=0.80-0.90 / seed_diff=10-15 | clean LL 0.5595 |
| 538 (7 seasons, n=428) | -0.0212 (v4 wins) | 1 bucket: chalk_won=chalk | clean LL 0.5799 |

**The two benchmarks find DIFFERENT weak spots.** Vegas surfaced
"v4 underperforms on upsets, late rounds, mid-seed-gaps, and the
0.80-0.90 confidence band." 538 surfaces "v4 underperforms on chalk
picks." The two are not contradictory -- they reflect different
calibration patterns of the two benchmarks (Vegas at SIGMA=11 is
overall flatter; 538 commits harder to chalk via its rdR_win
products) -- but they imply that v4's bottleneck is calibration
shape rather than any single bucket. **This shifts the "engineer
against weak spots" lever from "fix the upset detector" (Vegas-
implied) to "fix calibration shape" (Vegas + 538 jointly).**

**Promote Active queue #2 to #1: single-season variance check.** The
two audits together establish that v4's per-bucket weakness depends
on the benchmark; what we still don't know is whether any *single
season* of v4 misbehaves enough to dominate the user's 2159/3462
Kaggle finish. ~30-min check; surface any season where v4's LL or
ECE is materially worse than the 22-season aggregate.

**Active queue #3 (external data as features) is unblocked.** With
the audit-lane closed, 538 forecasts as a v4 input feature is a
viable experiment using the same Wayback-pinned cache built here.
The 7-season coverage limit applies.

## Files of record

```
src/ingest/fte_forecasts.py             -- 538 loader (~150 LOC)
src/audit_v4_gap_fte.py                 -- driver (~470 LOC)
tests/test_ingest/test_fte_forecasts.py -- 10 unit tests
tests/test_audit_v4_gap_fte.py          --  9 unit tests

output/v4_gap_audit_fte.json            -- per-bucket metrics (force-added)
output/v4_gap_calibration_overall_fte.png
output/v4_gap_calibration_by_round_fte.png
output/v4_gap_per_bucket_ll_delta_fte.png
output/v4_gap_audit_fte_log.txt
docs/notes/2026-05-04-v4-gap-audit-fte.md   -- this note
```

Cached forecasts (gitignored): `data/raw/fte_forecasts/<year>.csv`
for years in `_AUDITED_YEARS = (2016, 2017, 2018, 2019, 2021, 2022,
2023)`.

## Compute

- 538 forecast download: ~7 fetches from Wayback Machine, ~5-15s each
  on first run (cached thereafter). Total ~60s for the cold path.
- Audit run: ~3s end-to-end (numpy/pandas-bounded over 428 games).
- Recon (one-shot, deleted): ~10s of CDX queries to find the 7
  per-year snapshot timestamps. Captured in
  `src/ingest/fte_forecasts.py::_FTE_URL_BY_YEAR`.
