# v4 Per-Season Variance Check -- Design

**Date:** 2026-05-07
**Branch:** feat/v4-per-season-variance
**Predecessors:**
- v4 gap audit vs Vegas (PR 22; clean baseline): `docs/notes/2026-05-04-v4-gap-audit-vegas.md`
- v4 gap audit vs 538 (PR 29): `docs/notes/2026-05-04-v4-gap-audit-fte.md`

## Motivation

Both audits ran on aggregate metrics across 22 seasons (Vegas) and 7
seasons (538) and found weak-spot signatures that disagree:

- Vegas surfaced 6 weak spots (upsets, late rounds, mid-seed-gap,
  0.80-0.90 confidence band).
- 538 surfaced 1 weak spot (chalk picks).

The two-benchmark disagreement implies the bottleneck is calibration
*shape* rather than any single bucket. But there's an unresolved
question: **the user's 2159 / 3462 Kaggle finish is a single-season
result, while v4's audit numbers are 22-season aggregates.** A model
that calibrates well on the average tournament may still misbehave on
specific tournaments. If a single outlier season dominates v4's
single-season Kaggle exposure, fixing aggregate calibration won't
help -- the right move is to investigate the outlier.

This is the cheap-falsification gate before committing engineering
budget to calibration-shape work (TODO active queue #3): does v4's
22-season-average story hide high-variance per-season behavior?

## Goals

- Per-season metrics for v4 across 22 LOSO seasons (2003-2025 minus
  2020), restricted to R64-Champ tournament games (excluding First
  Four play-in games).
- Per-season metrics for **two cross-benchmark comparisons** (the
  better signal than v4 alone): v4 vs Vegas-implied probabilities
  on the 22 seasons where Vegas data is available, and v4 vs 538
  forecasts on the 7 seasons where 538 data is available.
- Identify outlier seasons (>= 1.5 sigma from the cross-season mean
  on either the v4-vs-Vegas delta or the v4-vs-538 delta), plus
  intra-v4 outliers (>= 1.5 sigma from v4's own per-season LL/ECE
  mean) as a sanity check.
- Verdict shape (informs the queue): variance flat -> bottleneck is
  aggregate calibration; one or two outlier seasons -> investigate
  outliers before fixing aggregate; trend (e.g., recent seasons
  drifting worse) -> separate analysis question.

## Non-Goals (deliberately deferred)

- Acting on the verdict. This is a diagnostic gate; the next
  experiment (calibration-shape engineering, outlier investigation,
  or trend deep-dive) gets its own spec.
- Per-team variance within seasons. Only per-season-aggregate
  variance is in scope.
- Modeling the variance (e.g., regressing season-level LL on tournament
  difficulty proxies). Visualize and identify; don't model yet.
- Single-bucket variance (e.g., per-season chalk_won=chalk LL).
  The audits already broke aggregate metrics by bucket; this analysis
  breaks aggregate metrics by season. Per-(season, bucket) is a
  finer-grained followup if a bucket-level outlier surfaces here.
- Bracket-points per-season variance. The audits use LL as the cheap
  metric; bracket points is the production metric and a per-season
  bracket-points trace would be a useful followup, but adds the
  v9-C / v8 stage-2 pipeline as a dependency. Out of scope for this
  ~30-min check.
- Re-running an entire audit per season. We reuse the audit drivers'
  data-load helpers (`_build_vegas_lookup`, `_load_v4_lookup`,
  `_resolve_fte_team_ids`, etc.) and just compute per-season metrics
  on top.

## Approach

### Architecture

```
src/analyze_v4_per_season_variance.py    -- one-shot driver (~200 LOC)
  |
  +-- _per_season_metrics(audit_df) -> pd.DataFrame  (n_games, ll_v4,
  |                                                   acc_v4, ece_v4,
  |                                                   plus _vegas/_fte
  |                                                   suffixed)
  +-- _flag_outliers(df, sigma=1.5) -> dict
  +-- run_analysis(...) -> dict
  |       reuses:
  |         - src/audit_v4_gap_vegas.py: _build_vegas_lookup,
  |             _load_v4_lookup, _load_seeds_lookup,
  |             _build_per_game_audit_df (for v4 + Vegas join),
  |             _calibration_table, _ece, _round_from_daynum
  |         - src/audit_v4_gap_fte.py: _build_fte_lookup,
  |             _resolve_fte_team_ids, _build_per_game_audit_df
  |             (for v4 + 538 join), FTE_RD_COL_FOR_ROUND
  |         - src/ingest/fte_forecasts.py: load_fte_forecasts,
  |             _AUDITED_YEARS
  +-- writes:
        output/v4_per_season_variance.json
        output/v4_per_season_variance_traces.png       (LL + acc + ECE per season)
        output/v4_per_season_variance_deltas.png       (v4 - Vegas, v4 - 538 deltas per season)
        output/v4_per_season_variance_log.txt
```

### Data flow

1. **Build the v4 vs Vegas per-game DF** by reusing the Vegas
   audit's pipeline: `_load_v4_lookup` -> `_build_vegas_lookup` ->
   `_build_per_game_audit_df`. This gives a frame with columns
   `(season, daynum, team_a, team_b, p_v4, p_vegas, winner_is_a,
   round, ...)` over 22 seasons of R64-Champ games.

2. **Build the v4 vs 538 per-game DF** by reusing the 538 audit's
   pipeline: `_load_v4_lookup` -> `load_fte_forecasts` ->
   `_resolve_fte_team_ids` -> `_build_fte_lookup` ->
   `_build_per_game_audit_df`. Gives an analogous frame over the 7
   seasons in `_AUDITED_YEARS`.

3. **Per-season aggregation:**
   - From v4-vs-Vegas frame, compute per-season `(n_games, ll_v4,
     ll_vegas, acc_v4, acc_vegas, ece_v4, ece_vegas)`.
   - From v4-vs-538 frame, compute per-season `(n_games_fte, ll_fte,
     acc_fte, ece_fte)`.
   - Merge on `season` (left join from Vegas frame -- 538-side
     columns are NaN for non-audited seasons).

4. **Outlier flagging:**
   - For each metric `m` in `{ll_v4, ll_v4_minus_vegas,
     ll_v4_minus_fte, ece_v4}`, compute `mean_m`, `std_m` across
     non-NaN seasons.
   - Flag any season where `(m - mean_m) / std_m >= 1.5` (positive =
     v4 worse than typical for that season's metric).
   - Report all per-season values + the sigma-delta + the boolean
     outlier flag.

5. **Plots:**
   - **Traces plot**: 3 panels (LL, accuracy, ECE), x = season, three
     lines per panel (v4, Vegas, 538). Highlights v4's per-season
     trajectory with the two benchmarks for context.
   - **Deltas plot**: 2 panels (LL_delta_vs_Vegas, LL_delta_vs_538),
     x = season, horizontal bars colored red where outlier-flagged.
     Shows where v4 is materially worse than typical *vs each
     benchmark* (the more informative view per the brainstorm).

6. **JSON output:** per-season records + outlier list + cross-season
   summary (means, stds).

### Verdict criteria

The findings note picks ONE of:

- **Variance flat.** No season is >= 1.5 sigma on any
  cross-benchmark delta or on intra-v4 LL/ECE. Conclusion:
  v4's calibration is uniform across tournaments; aggregate calibration
  is the bottleneck. **Recommendation:** promote calibration-shape
  engineering (TODO #3) to the active queue's #1.

- **Outlier season(s).** 1-2 seasons exceed 1.5 sigma. Conclusion:
  v4 has season-specific calibration failures. **Recommendation:**
  next experiment investigates what's distinctive about the outlier
  season(s) -- feature drift, opponent-strength variance,
  historically-anomalous tournament patterns. Aggregate calibration
  fix may be premature.

- **Trend.** 3+ consecutive seasons drift in one direction. Conclusion:
  v4 has gradually-degrading calibration (data-pipeline drift?
  rule changes? COVID era effects?). **Recommendation:** trend deep-
  dive as a separate spec.

(Mixed cases get reported as such; the verdict picks the closest
match and notes ambiguity.)

## Anchor / sanity checks

- **22-season-aggregate ll_v4 from this analysis matches the Vegas
  audit's clean number.** Per `docs/notes/2026-05-04-v4-gap-audit-vegas.md`,
  v4's overall LL on the Vegas audit's 1326-game subset is 0.5595.
  Aggregating per-season LL via `np.average(per_season_ll,
  weights=n_games)` should reproduce ~0.5595 within floating-point
  tolerance. If far off, the per-season aggregator is broken.
- **7-season ll_v4 on the 538-overlap subset matches the 538 audit's
  number 0.5799.** Same check on the smaller sample.
- **n_games per season** approximately 60-65 (4 regions x 15 R32-Champ
  games + 32 R64 games = 63 expected, with FF excluded). 2021 will
  be lower (~50) due to COVID-era tournament cancellations.
- **Vegas LL and 538 LL anchors** match their respective audit
  numbers (0.5447 for Vegas on 1326 games, 0.6011 for 538 on 428
  games).

## Falsification / what would make us re-think

- **Per-season aggregator disagrees with audit aggregates.** Stop and
  debug; the analysis is wrong before any verdict can be trusted.
- **Every season is an outlier.** 1.5 sigma is a moderate threshold;
  if everything trips it, the std estimate is itself unstable
  (probably one extreme season dominating it). Report and reduce
  threshold.
- **All cross-benchmark deltas are large but match each other across
  seasons.** Suggests v4 has systematic miscalibration shape that's
  consistent across tournaments. Aggregate-calibration view is
  vindicated; 1.5-sigma flagging finds nothing because there's no
  per-season variance to find. This is the "variance flat" verdict
  by another name.
- **Vegas-derived deltas and 538-derived deltas disagree on which
  season is the outlier.** Plausible if Vegas and 538 have different
  per-season calibration tendencies themselves. Findings note
  flags as ambiguous; no single-outlier verdict.

## Test plan

- `tests/test_analyze_v4_per_season_variance.py` (new):
  - Unit: `_per_season_metrics` on a synthetic 3-season frame
    correctly aggregates LL / acc / ECE per season.
  - Unit: `_flag_outliers` flags a hand-constructed outlier at
    1.5+ sigma; does not flag values within 1 sigma.
  - Unit: weighted aggregation invariant -- per-season LL aggregated
    with `n_games` weights equals overall LL on the same frame.
  - Smoke: end-to-end on a 2-season subset writes the JSON and PNGs
    without raising.

Existing test suite must remain green.

## Risks

1. **Pipeline reuse coupling.** This script imports private helpers
   from `src/audit_v4_gap_vegas.py` and `src/audit_v4_gap_fte.py`
   (functions prefixed with `_`). If those modules' internals change,
   this analysis breaks. Mitigation: use the public-ish entrypoints
   where possible (`run_audit` etc.) -- but those are not parameterizable
   per-season. Alternative: factor a shared `src/audit/per_game_join.py`
   utility module. **Decision:** import the helpers directly with a
   clear comment marking the cross-module coupling. This is a one-off
   diagnostic; a refactor to share infrastructure is a separate
   followup if more diagnostics in this family follow.
2. **Sigma threshold sensitivity.** 1.5 sigma is a defensible cutoff
   but not the only one. Mitigation: report all per-season values +
   their sigma-deltas in the JSON, so the verdict can be revisited
   without re-running. The threshold is documented in code and
   findings note.
3. **Small-N seasons.** 2021's COVID-era tournament had fewer games;
   per-season metrics on n~50 are noisier than on n~63. Mitigation:
   per-season records include `n_games`; outliers are reported with
   their counts so a small-N artifact is visible.
4. **Mixed-baseline weight comparison.** v4-vs-Vegas covers 22
   seasons; v4-vs-538 covers 7. Outlier flagging on the v4-vs-538
   delta uses a 7-season mean/std; this is intrinsically less
   stable than the 22-season Vegas delta. Mitigation: report the
   sample size used for each sigma calculation; the findings note
   weighs Vegas-derived flags more heavily.
5. **Anchor failure.** If the per-season aggregator's weighted mean
   doesn't reproduce the audit's overall numbers, the analysis is
   wrong. Halt and debug before trusting any verdict.

## Lessons from prior experiments carried forward

- **Diagnostic-first.** Cheap analysis before committing engineering
  budget. Same paid-for-itself logic as Vegas audit (PR 18/22),
  538 audit (PR 29), BT/HBT/Massey/Colley diagnostic gates.
- **Reuse existing infrastructure.** Both audits' data-load + joining
  pipelines are reusable; the only new code is per-season aggregation
  + outlier flagging + plotting.
- **Force-add output artifacts.** `output/` is gitignored per
  precedent.
- **Anchor before trusting.** Per-season weighted aggregate must
  reproduce audit overall numbers.

## Out-of-scope follow-ups (post-variance-check)

- **Calibration-shape engineering** (TODO active queue #3) -- gated
  on this check's verdict.
- **Outlier-season investigation** -- conditional on this check
  finding outliers.
- **Per-(season, bucket) heatmap** -- finer-grained variance view if
  this check finds bucket-level structure.
- **Bracket-points per-season trace** -- production-metric view,
  requires v9-C stage-2 pipeline integration.

## File-touch summary

```
new   docs/superpowers/specs/2026-05-07-v4-per-season-variance-design.md
new   docs/superpowers/plans/2026-05-07-v4-per-season-variance.md
new   src/analyze_v4_per_season_variance.py
new   tests/test_analyze_v4_per_season_variance.py
new   output/v4_per_season_variance.json                   (force-added)
new   output/v4_per_season_variance_traces.png             (force-added)
new   output/v4_per_season_variance_deltas.png             (force-added)
new   output/v4_per_season_variance_log.txt                (force-added)
new   docs/notes/2026-05-07-v4-per-season-variance.md      (findings)

edit  TODO.md  -- mark active-queue #1 done; promote next item based on
                  verdict.
```
