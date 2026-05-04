# v4 Gap Audit vs FiveThirtyEight Tournament Forecasts -- Design

**Date:** 2026-05-04
**Branch:** feat/v4-gap-audit-fte
**Predecessors:**
- v4 gap audit vs Vegas (NO weak spots; v4 beats Vegas everywhere):
  `docs/notes/2026-05-04-v4-gap-audit-vegas.md`
- v9-C production swap (current production stage-2):
  `docs/superpowers/specs/2026-05-01-v9c-production-swap-design.md`

## Motivation

The Vegas audit (PR 18) ran the bucket-and-compute-metrics framework
against Vegas closing-line implied probabilities (SIGMA=11) and found
**no weak spots** -- v4 beats Vegas on every bucket measured. That
result is itself a strong "v4 is competitive" finding, but it does
NOT localize v4's headroom. The user's 2159 / 3462 Kaggle finish
(2/3 of entries beat v4 on the same data) shows real headroom exists
somewhere; Vegas-at-SIGMA=11 just doesn't surface it.

FiveThirtyEight's tournament forecasts are the next-strongest public
benchmark. Two reasons they may localize v4's headroom where Vegas
did not:

1. **Calibrated probabilities, no SIGMA conversion.** 538 publishes
   per-team round-survival probabilities directly (`rd1_win` ..
   `rd7_win`). No spread-to-prob assumption. This removes the SIGMA=11
   caveat that softened the Vegas audit's claims.
2. **Different signal mix.** 538's model is built on power ratings
   that fold in opponent-adjusted efficiency, season trajectory, and
   their own preseason prior. Different feature mix from v4 means
   different residual patterns -- buckets where 538 outperforms v4
   are candidate weak-spot signatures for v4 that the Vegas
   benchmark missed.

If 538 also beats v4 on no bucket, we have a strong "v4 is
competitive against multiple public benchmarks" finding that closes
the audit lane -- single-season variance check (TODO #2) and
external-data-as-features (TODO #3) become the active levers.
If 538 beats v4 on specific buckets, we have weak-spot signatures
to engineer against.

## Goals

- Build per-tournament-game records for played seasons in 538's
  archive (2014-2019, 2021-2025): `(season, round, team_a, team_b,
  p_v4, p_fte, winner)`. Tournament-only -- regular-season is out
  of scope.
- Compute per-bucket aggregate metrics: log loss, accuracy,
  calibration curve, ECE, per-bucket sample count.
- Compare v4 to 538. Identify weak-spot buckets at the same threshold
  the Vegas audit used (`n>=50, ll_delta>=+0.02`).
- Ship a single audit note + a structured JSON of the per-bucket
  numbers + a small set of calibration / per-bucket bar PNGs.

## Non-Goals (deliberately deferred)

- **Acting on findings (feature engineering, post-processing).** This
  audit produces a *map* of weak spots; engineering against any of
  them is its own follow-up experiment with its own spec.
- **Adding 538 round-survival probabilities as a v4 input feature.**
  Audit-only here; "external data as features" is TODO active queue
  item #3 and is its own follow-up.
- **Comparing 538's `team_rating` (per-team strength) via 538's own
  logistic.** We use round-survival probabilities directly, per the
  brainstorming decision. The rating-via-logistic path is methodically
  similar to Vegas-via-SIGMA and would re-introduce the conversion
  caveat we're trying to escape.
- **Live / round-aware 538 snapshots.** We use the earliest snapshot
  per season (pre-tournament). Round-aware snapshots would benchmark
  v4 against a 538 that knows post-R64 results -- asymmetric and
  unfair.
- **Women's tournament audit.** v4 is mens-only; we filter 538 to
  `gender='mens'`.
- **2020 audit.** No tournament was played; not in 538's archive
  either.
- **2026 audit.** No outcomes yet.
- **Per-team diagnostics.** Bucket-level only; per-team variance is
  too small for ~720 games / 11 seasons to surface clean signal.
- **Reusable diagnostic module.** Audit-as-script for now; if we run
  this after every model change, factor a reusable
  `src/audit/v4_vs_external.py` then.

## Approach

### Architecture

```
src/ingest/fte_forecasts.py            -- 538 data loader (new)
  |
  +-- _fte_forecasts_url(year)          -- raw GitHub URL builder
  +-- download_fte_forecasts(year)      -- HTTP GET with cache fallback
                                            to data/raw/fte_forecasts/<year>.csv
  +-- load_fte_forecasts(years)         -- load + concat + filter
                                            (gender='mens', earliest snapshot)
  +-- resolve_to_team_id(df, mapping)   -- name -> Kaggle TeamID via
                                            existing build_team_mapping +
                                            data/team_name_overrides.csv

src/audit_v4_gap_fte.py                -- one-shot driver (new)
  |
  +-- load v4 pairwise probabilities (output/pairwise_v4.csv, dedup'd)
  +-- load 538 forecasts via load_fte_forecasts
  +-- load tournament outcomes (MNCAATourneyCompactResults)
  +-- load tournament seeds (MNCAATourneySeeds)
  +-- join 538 to tournament games on (Season, sorted_team_pair)
  +-- compute 538 implied prob via BT-normalization of rd_R_win:
        p_fte_A = rd_R_win[A] / (rd_R_win[A] + rd_R_win[B])
      where R is the round of the game
  +-- bucket each game by:
        - round (R64, R32, S16, E8, F4, Champ; FF aggregate)
        - higher-vs-lower-seed binary (favorite_won?)
        - v4-confidence quintile (0.5..0.6, ..0.7, ..0.8, ..0.9, >0.9)
        - seed-diff magnitude (|seed_a - seed_b| in {0-2, 3-5, 6-9, 10-15})
  +-- per-bucket aggregate: count, ll_v4, ll_fte, ll_delta,
      acc_v4, acc_fte, calibration (predicted bin -> empirical rate)
  +-- write output/v4_gap_audit_fte.json
  +-- emit calibration plots:
        output/v4_gap_calibration_overall_fte.png
        output/v4_gap_calibration_by_round_fte.png
        output/v4_gap_per_bucket_ll_delta_fte.png
  +-- print top-N "v4 underperforms 538" bucket signatures
```

### Data sourcing

**Source:** `fivethirtyeight/data` GitHub repository, path
`march-madness-predictions/<year>/fivethirtyeight_ncaa_forecasts.csv`.
Public, stable; one CSV per tournament year covering 2014-2025
(no 2020). Raw URL pattern:

```
https://raw.githubusercontent.com/fivethirtyeight/data/master/march-madness-predictions/<year>/fivethirtyeight_ncaa_forecasts.csv
```

The loader maintains an explicit `_FTE_URL_BY_YEAR` dict so any
year-specific path quirk (538 reorganized the repo at least once
historically) is patchable without touching the URL builder. The
default value for each year is the pattern above; the implementer
verifies all 11 URLs return HTTP 200 and the expected schema in the
first task of the plan, populating the dict with overrides as
needed.

**Caching:** Auto-download missing years to
`data/raw/fte_forecasts/<year>.csv` (gitignored; re-fetchable). On
subsequent runs, read from cache. CI / fresh-clone behavior: if the
cache is empty, the loader fetches; tests that don't need network
can pass a fixture path.

**Coverage:** 11 seasons (2014, 2015, 2016, 2017, 2018, 2019, 2021,
2022, 2023, 2024, 2025). 2020 absent (no tournament). 67 games per
year, 737 total (693 R64-through-Champ games audited, 44 First Four
games excluded).

**Schema (typical 538 NCAA forecasts CSV):**

```
gender, forecast_date, playin_flag, rd1_win, rd2_win, rd3_win,
rd4_win, rd5_win, rd6_win, rd7_win, team_alive, team_id, team_name,
team_rating, team_region, team_seed
```

We use `gender, forecast_date, team_name, team_seed, rd1_win, ...,
rd7_win`. Filter to `gender='mens'`.

**Snapshot policy: earliest post-play-in snapshot per season.** 538
publishes multiple pre-tournament snapshots; the very-earliest
contain play-in teams where `rd1_win` is `P(win play-in)`, not
`P(win R64 game)`. We need the snapshot whose `rd1_win` already
represents R64 advancement. Detection rule: pick the earliest
`forecast_date` snapshot per season that contains exactly 64 unique
team_ids (the post-play-in field), OR equivalently, the earliest
snapshot where `playin_flag=0` (or missing) for every team. The
loader asserts both conditions agree; if they disagree on any year,
fail loudly so the human can disambiguate that year manually.

**Play-in handling.** First Four games (DayNum 134-135) are excluded
from the audit, mirroring the Vegas audit's `FF` aggregate handling.
With the post-play-in snapshot, the 64 teams in the audit each have
a clean `rd1_win = P(win their R64 game)`.

Schema-version risk: if a year's CSV uses different column names
(historical 538 reformatted twice), the loader normalizes by
column-name lookup with explicit fallbacks, raising a clear error
if the canonical columns are missing.

### Team-name resolution

538 uses standard college-basketball names ("Connecticut", "North
Carolina", "Saint Mary's") which differ from Kaggle Mania team names
in some cases (e.g., "Saint Mary's" vs "St Mary's CA"). Resolution
goes through the existing
`src/ingest/team_mapping.py::build_team_mapping` fuzzy matcher with
the canonical thresholds from `config.yaml`
(`auto_accept_threshold=85`, `review_threshold=70`). Manual overrides
flow through `data/team_name_overrides.csv`.

The audit driver fails fast with a clear error if any team in 538
data goes unresolved, naming the unresolved team(s) so an override
row can be added. Per repository convention, names are NEVER
persisted as join keys; resolution happens once at load time and
all downstream joins are on `TeamID`.

### Per-matchup probability via BT normalization

For an actual played matchup `(team_a, team_b)` in round R (with
R in {1, 2, 3, 4, 5, 6} mapping to {R64, R32, S16, E8, F4, Champ}):

```python
p_fte_a = rd_R_win[team_a] / (rd_R_win[team_a] + rd_R_win[team_b])
```

`rd_R_win[team_x]` is 538's published probability that team_x wins
its round-R game (= reaches round R+1).

**R64 case (R=1) is exact.** rd1_win[A] + rd1_win[B] = 1 by 538's
bracket consistency: A's R64 opponent is B, so A advancing past R64
is the same event as B not advancing. The BT denominator is
exactly 1 and `p_fte_a = rd1_win[a]`.

**R32+ case (R>=2) is a BT-style approximation.** In rounds beyond
R64, 538's `rd_R_win` averages over multiple possible opponents in
round R. For an actual played matchup (A, B), the BT normalization
asks "given both teams reach R, what's the relative chance A wins?"
which assumes both teams have similar opponent-difficulty profiles.
This is approximate; bias direction is unclear in aggregate but
small in practice (sub-1% per matchup).

This approximation is documented in the findings note as a caveat,
parallel to the SIGMA=11 caveat in the Vegas audit. The alternative
methods (using `team_rating` via 538's logistic, using conditional
cascades) were considered and rejected in brainstorming -- BT-norm
keeps the audit faithful to 538's published numbers while staying
methodologically simple.

Implementation note: rd0 / round-of-128 / play-in is not used.
First Four games in our data carry round="FF" (DayNum 134-135) and
are not in any rd_R_win column directly; we exclude FF games from
the audit, mirroring the Vegas audit's treatment.

### Joining 538 to tournament games

Each Kaggle tournament game has `Season, DayNum, WTeamID, LTeamID`.
The matching 538 forecast is uniquely identified by `(Season,
team_id)` (one row per team per season after snapshot filtering).
Join procedure:

1. Resolve each 538 row's `team_name` to Kaggle `TeamID` via
   `build_team_mapping`.
2. For each tournament game `(Season, DayNum, WTeamID, LTeamID)`,
   look up the 538 row for `(Season, WTeamID)` and `(Season,
   LTeamID)`. If both present, the game joins.
3. Determine the round from `DayNum` via `ROUND_BY_DAYNUM` (existing
   convention); pick the corresponding `rd_R_win` column.
4. Compute `p_fte_a` (a = team_a, where team_a < team_b in the
   canonical pair ordering) via BT normalization on `rd_R_win`.

Coverage caveat: any Kaggle team missing from 538's CSV (rare;
typically only First Four play-in losers in some years) drops the
game. Report the join coverage rate; expect >= 95% post-FF-trim.

### Bucket definitions

| bucket | values | sample counts (11 seasons, 693 audited games) |
|--------|--------|-----------------------------------------------|
| `round` | R64, R32, S16, E8, F4, Champ | 32, 16, 8, 4, 2, 1 per season; 352, 176, 88, 44, 22, 11 total |
| `higher_seed_won` (binary) | True (chalk), False (upset) | ~70/30 split |
| `v4_confidence_quintile` | (0.50-0.60], ..., (0.90-1.00] | quintiles by predicted prob for the favored side |
| `seed_diff_bucket` | |seed_a - seed_b| in {0-2, 3-5, 6-9, 10-15} | based on seeded slot pairings |

Same buckets as the Vegas audit. F4 (n~22) and Champ (n~11) will fall
below the n>=50 weak-spot threshold; explicitly flagged
"sample-limited" in findings rather than treated as a verdict.

### Per-bucket metrics

For each (bucket, value) cell:

- `n_games`: count of games where both v4 and 538 have probabilities.
- `ll_v4`, `ll_fte`: log loss on the actual winner.
- `ll_delta = ll_v4 - ll_fte`: positive = v4 worse than 538.
- `acc_v4`, `acc_fte`: accuracy on chalk pick.
- `calibration_v4`, `calibration_fte`: list of `(predicted_bin,
  empirical_rate, n)` entries, predicted bins of width 0.05 over
  [0.5, 1.0]. ECE computed as `sum_bin (n_bin / n_total) *
  |predicted_bin_mid - empirical_rate|`.
- `p_v4_minus_fte_mean`: mean of `p_v4 - p_fte` -- positive means
  v4 is more confident in the favored side than 538 in this bucket.

Findings note prioritizes buckets with `n_games >= 50` and `ll_delta
>= 0.02` as "v4 specifically underperforms 538 here", matching the
Vegas audit threshold.

### Charts

Three PNGs (matplotlib, already a project dep):

1. **Overall calibration** (`v4_gap_calibration_overall_fte.png`) --
   predicted-prob bin (x) vs empirical win-rate (y), one line per
   model (v4 vs 538). Diagonal reference.
2. **Calibration by round** (`v4_gap_calibration_by_round_fte.png`)
   -- 6 panels (R64 .. Champ), same axes. Per-round calibration
   pattern.
3. **Per-bucket LL delta** (`v4_gap_per_bucket_ll_delta_fte.png`) --
   horizontal bar chart of `ll_v4 - ll_fte` per bucket, sorted
   descending. Highlights worst buckets.

Force-add to git alongside the findings note.

### Output schema

`output/v4_gap_audit_fte.json`:

```json
{
  "config": {
    "v4_pairwise": "output/pairwise_v4.csv",
    "fte_cache_dir": "data/raw/fte_forecasts",
    "seasons": [2014, 2015, 2016, 2017, 2018, 2019, 2021, 2022, 2023, 2024, 2025],
    "snapshot_policy": "earliest_per_season"
  },
  "join_coverage": {
    "n_tournament_games": NNNN,
    "n_with_v4": NNNN,
    "n_with_fte": NNNN,
    "n_both": NNNN,
    "missing_fte_seasons": {2020: 67, ...},
    "unresolved_fte_names": []
  },
  "overall": {
    "n_games": NNNN,
    "ll_v4": ..., "ll_fte": ...,
    "acc_v4": ..., "acc_fte": ...,
    "ece_v4": ..., "ece_fte": ...,
    "calibration_v4": [{"bin": [0.5, 0.55], "n": ..., "empirical": ...}, ...],
    "calibration_fte": [...]
  },
  "by_round": {
    "R64": { ... per-bucket metrics ... },
    ...
  },
  "by_higher_seed_won": {
    "True": { ... },
    "False": { ... }
  },
  "by_v4_confidence_quintile": {
    "0.50-0.60": { ... }, ...
  },
  "by_seed_diff": {
    "0-2": { ... }, ...
  },
  "weak_spots": [
    {
      "bucket": "round=R32, v4_confidence=0.70-0.80",
      "n_games": NN,
      "ll_v4": ..., "ll_fte": ...,
      "ll_delta": +0.05,
      "interpretation": "v4 is over-confident on R32 mid-confidence picks; ..."
    }
  ]
}
```

## Anchor / sanity checks

- **Overall log loss subset matches v4's known LL on the 11-season
  subset.** v4's standalone LL on the full 22-season tournament-game
  set is 0.4369. The audit's `overall.ll_v4` should be close (likely
  within +/- 0.01) to v4's LL recomputed over the same 11-season
  subset. If far off, the join is broken.
- **538 LL is plausible.** 538's published forecasts on tournament
  games are well-calibrated; expect `overall.ll_fte` in the 0.45-0.55
  band. Far above 0.55 = BT normalization or join is wrong; far
  below 0.40 = suspicious (would imply 538 outperforms v4
  dramatically, which would be a very strong claim worth verifying).
- **538 accuracy is plausible.** 538 picks the favorite right
  ~70-75% historically. `overall.acc_fte` should land in that band.
- **R64 BT-norm sums-to-1 invariant.** For any R64 matchup (A, B),
  `rd1_win[A] + rd1_win[B]` should be 1.0 within tolerance (538
  publishes rounded probabilities; tolerance `<= 1e-2` is realistic,
  `<= 1e-3` after a re-normalization step). The audit asserts within
  tolerance; failure means we picked the wrong snapshot or the
  forecast is mid-tournament.
- **Calibration diagonal.** v4's overall calibration plot should be
  *roughly* on the diagonal (matching the Vegas audit's finding).
  538's plot is also expected to be near-diagonal (well-calibrated
  by construction).
- **Bucket sums.** `sum(per-round n_games) = overall.n_games`.
  Regression guard.

## Falsification / what would make us re-think

- **538 join coverage < 90% post-FF-trim.** If we can't recover 538
  forecasts for most tournament games, the audit is unreliable.
  Halt; investigate the team-name resolution before trusting any
  per-bucket numbers.
- **All buckets show `|ll_delta| < 0.01`.** v4 is roughly 538-tier
  everywhere -- which means audit-lane levers (v4 vs Vegas, v4 vs
  538) have produced no localizable headroom. Active queue items #2
  (single-season variance) and #3 (external data as features)
  become the next levers; ensemble exploration is closed for now.
- **v4 dominates 538 across the board (parallel to Vegas finding).**
  Strong "v4 is competitive against multiple public benchmarks"
  result. Findings note says so explicitly; same conclusion as
  above for next steps.
- **538 dominates v4 across the board.** Unexpected; would indicate
  v4 has a systemic miscalibration or a feature bug. Triple-check
  the join, the BT normalization, and v4's LL anchor before
  trusting the finding.
- **One bucket dominates the LL gap.** E.g., entire LL_delta
  concentrated in F4+Champ rounds (combined n~33). Findings note
  must explicitly call out fragile vs durable signatures.

## Test plan

- `tests/test_ingest/test_fte_forecasts.py` (new):
  - Unit: snapshot filtering -- earliest forecast_date per
    (gender, season, team_id) is selected.
  - Unit: gender filter -- only `gender='mens'` rows kept.
  - Unit: cache hit / cache miss behavior on a fixture path.
  - Unit: schema-version normalization -- a CSV with renamed columns
    raises a clear error or is mapped via the explicit fallback table.
  - Unit: name resolution -- a synthetic 538 row resolves to the
    expected `TeamID` via the existing matcher (asserts the dataframe
    schema returned matches downstream consumers' expectations).

- `tests/test_audit_v4_gap_fte.py` (new, mirrors
  `tests/test_audit_v4_gap_vegas.py`):
  - Unit: `_bt_norm(0.5, 0.5) == 0.5`, `_bt_norm(0.8, 0.2) == 0.8`,
    `_bt_norm(0.0, 0.0)` handled (return 0.5 with a warning, since
    both teams have zero round-R survival -- unreachable in practice
    for the actual played matchup but defensive code).
  - Unit: R64 sum-to-1 invariant -- on a fixture pair with
    `rd1_win[A]=0.6, rd1_win[B]=0.4`, BT-norm gives 0.6 / 1.0 == 0.6
    (matches direct probability). On a fixture pair with
    `rd1_win[A]=0.601, rd1_win[B]=0.401` (538 rounding), BT-norm
    gives 0.601 / 1.002 ~ 0.5998 -- within tolerance of 0.6.
  - Unit: round inference from DayNum is the same as the Vegas audit
    (regression guard if the audit modules drift).
  - Unit: bucketing -- a synthetic 100-game DataFrame is bucketed
    into the expected counts per round / seed_diff bucket.
  - Unit: per-bucket metric computation -- on a hand-computed
    fixture, `ll_v4` matches the formula and `ece` is computed
    correctly.
  - Smoke: end-to-end on a 2-season subset writes the JSON with the
    expected top-level keys and the anchor v4-LL within tolerance of
    a hand-computed reference.

Existing test suite must remain green.

## Risks

1. **538 schema drift across years.** Older years (2014-2017) may
   use different column names than later years. Mitigation: explicit
   column-mapping table per year, with a clear error on missing
   canonical columns. Spot-check 3 representative years (2014, 2019,
   2024) at implementation time.
2. **Team-name resolution gaps.** Less likely than Vegas (538 uses
   cleaner CBB names) but not zero. Mitigation: same fuzzy-matcher
   pipeline, explicit list of unresolved names in the JSON output,
   `data/team_name_overrides.csv` for manual corrections.
3. **GitHub raw-URL availability.** The `fivethirtyeight/data` repo
   is public and stable, but the raw URL could rate-limit or 404 on
   an unexpected sub-path. Mitigation: cache aggressively (fetch
   once, reuse from `data/raw/fte_forecasts/<year>.csv`); loader
   raises a clear error with the attempted URL on failure.
4. **538 was shut down March 2025.** ABC News owns the brand; the
   data archive may eventually be taken offline. Cache mitigates
   ongoing-availability risk; if the archive disappears mid-project,
   we have local copies.
5. **BT-norm approximation bias for R32+.** The R32+ buckets are the
   approximation cells. Mitigation: documented caveat in the
   findings note; if a finding looks knife-edge on R32+ alone,
   followup with rating-via-logistic as a sensitivity check.
6. **Smaller sample (11 seasons vs 22).** F4 and Champ buckets fall
   below n=50. Mitigation: explicit "sample-limited" flag in
   findings; weak-spot threshold preserves cross-audit comparability
   with Vegas.
7. **Calibration plot interpretability.** Same as Vegas audit: ECE
   single-number can hide per-bin behavior, plots fix this, both
   reported.
8. **Bucket multiplicity / cherry-picking.** Same as Vegas audit:
   ALL cells reported in JSON; only top-3-to-5 weak-spots
   highlighted as call-to-action; no p-hacking.

## Lessons from prior experiments carried forward

- **Diagnostic-first.** Single-script audit before any feature
  engineering. Same paid-for-itself logic as the BT and HBT gates,
  and the Vegas audit.
- **Reuse existing infrastructure.** `build_team_mapping`,
  `data/team_name_overrides.csv`, the bucket / metrics framework
  from the Vegas audit, the round-from-DayNum mapping, the
  calibration / ECE math. New code is only the 538 ingest module
  and the BT-norm probability helper; everything else is parallel
  to the Vegas audit.
- **Force-add output artifacts** (`output/` is gitignored) per
  precedent.
- **Anchor before trusting.** Overall LL_v4 must match v4's known
  baseline on the 11-season subset. R64 BT-norm sum-to-1 invariant
  must hold before any per-bucket metric is trusted.
- **Cache external data locally.** Avoid network in tests; ensures
  reproducibility across machines.

## Out-of-scope follow-ups (post-538-audit)

- **Single-season v4 variance check** (TODO active queue #2).
- **External rankings as v4 input features** (TODO #3) -- 538's
  team_rating and rd_R_win as candidate features, gated by the same
  cheap-falsification pattern (correlation, standalone LL,
  blend-headroom) used for prior feature additions.
- **Sensitivity check: 538 BT-norm vs rating-via-logistic.** If any
  R32+ weak spot is knife-edge, run the alternative methodology to
  see if the finding is method-stable.
- **Top public Kaggle entry comparison.** Same audit shape; sourcing
  question is whether per-game prediction CSVs are public for top
  finishers.

## File-touch summary

```
new   docs/superpowers/specs/2026-05-04-v4-gap-audit-fte-design.md
new   docs/superpowers/plans/2026-05-04-v4-gap-audit-fte.md
new   src/ingest/fte_forecasts.py
new   src/audit_v4_gap_fte.py
new   tests/test_ingest/test_fte_forecasts.py
new   tests/test_audit_v4_gap_fte.py
new   output/v4_gap_audit_fte.json                   (force-added)
new   output/v4_gap_calibration_overall_fte.png      (force-added)
new   output/v4_gap_calibration_by_round_fte.png     (force-added)
new   output/v4_gap_per_bucket_ll_delta_fte.png      (force-added)
new   docs/notes/2026-05-04-v4-gap-audit-fte.md      (findings)
new   data/raw/fte_forecasts/<year>.csv              (cached, gitignored)

edit  TODO.md                                        (move 538 audit
                                                      from active queue
                                                      #1 to "tried and
                                                      rejected" or "done"
                                                      as appropriate)
edit  .gitignore                                     (add data/raw/fte_forecasts/
                                                      if not already ignored
                                                      under data/cache/ rules)
```
