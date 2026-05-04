# v4 Vegas-feature leakage fix -- Design

**Date:** 2026-05-04
**Branch:** feat/v4-vegas-leak-fix
**Status:** spec

## Problem

`compute_vegas_features()` (`src/enhanced_model_v3.py:258`) and
`_build_vegas_team_records_with_dates()` (`src/enhanced_model_v3.py:379`)
aggregate over the full Vegas-line dataframe with no date/round filter.
`load_vegas_lines()` (`src/enhanced_model_v3.py:193`) loads the entire
season Nov-Apr including NCAA tournament games. Result: for each
(season S, team T), the features

- `vegas_avg_spread`
- `vegas_avg_margin`
- `vegas_ats_pct`
- `vegas_power_rating`
- `vegas_consistency`
- `vegas_late_spread_delta` (computed by `compute_vegas_trend()` from
  `_build_vegas_team_records_with_dates`)

include season-S NCAA tournament outcomes in their aggregates.

In the LOSO CV used to generate `output/pairwise_v4.csv`, the model
trains on seasons != S but the test-time feature row for season S has
season S's tournament games already baked into these aggregates. The
training data has the same property (each train season's teams have
features computed on their tournament games), so the model learns to
USE the leak; at test time, it gets to use it again.

## Quantified leak (current state)

Direct measurement on the live `data/raw/vegas_lines/`:

| team-season              | n_games | n_tourney | full margin | reg-only margin | leak  |
|--------------------------|---------|-----------|-------------|-----------------|-------|
| 2024 UConn (champs)      | 41      | 9         | +18.13      | +16.16          | +1.98 |
| 2024 Purdue (runner-up)  | 42      | 11        | +13.46      | +11.48          | +1.98 |
| 2023 UConn (champs)      | 38      | 5         | +14.42      | +13.42          | +1.00 |
| 2018 Virginia (lost R64) | 31      | 1         | +11.87      | +12.70          | -0.83 |

The leak is **correlated with tournament success**: champions look
stronger than they were going in; first-round losers look weaker.
`vegas_late_spread_delta` is even more leaky -- its 30-day window for
tournament teams is dominated by conf-tourney + NCAA games.

## Indirect evidence the audit's verdict is wrong

PR 18 audit reports v4 LOSO accuracy 80.9% on 1326 tournament games and
"v4 beats Vegas everywhere." Top public tournament models (KenPom,
538) hit ~70-75%. v4's user finished 2159 / 3462 in actual Kaggle.
A model that genuinely beats Vegas across all buckets does not finish
in the bottom half of a real tournament prediction contest.

## Scope

**In scope (this PR):**

1. New helper `filter_vegas_to_pre_tournament(vegas_df, seasons_csv_path)`
   in `src/enhanced_model_v3.py`. Returns a copy of `vegas_df` with
   rows where `daynum >= 134` dropped.
2. Wire it in: in `prepare_loso_inputs()` and `main()` of
   `enhanced_model_v3.py`, after `load_vegas_lines()` and before any
   call to `compute_vegas_features()` or
   `_build_vegas_team_records_with_dates()`, replace the raw `vegas_df`
   with the filtered version.
3. Unit tests for the filter behavior and integration tests showing
   the leak is closed for known team-seasons.

**Out of scope (separate PRs, tracked in TODO.md):**

- Regenerate `output/pairwise_v4.csv` via clean LOSO.
- Re-run `audit_v4_gap_vegas.py` against the clean pairwise CSV.
- Massey / KenPom audit (separate leakage class).
- Re-run swap-candidate evaluations (v8 vs v9-C, plain BT bracket
  points, etc.) against the clean baseline.

**Explicitly NOT changed:**

- `load_vegas_lines()` itself stays unchanged. Two callers depend on
  its current "full season" behavior:
  1. `audit_v4_gap_vegas.py` -- needs tournament-game lines to compare
     v4 against Vegas implied probabilities.
  2. `_build_r64_lines()` -- needs tournament-game lines for R64 line
     blending in `src/bracket/line_blending.py`.
  Filtering inside `load_vegas_lines()` would break both.

- `compute_vegas_features()` and `_build_vegas_team_records_with_dates()`
  signatures unchanged. They simply receive a pre-filtered dataframe.

## Filter design

Boundary chosen for "tournament" is `daynum >= 134`, matching the
existing `ROUND_BY_DAYNUM` table in
`src/audit_v4_gap_vegas.py` (134-135 = First Four, ..., 154 = Champ).
This drops the First Four, R64, R32, S16, E8, F4, and Champ. Conference
tournaments (DayNum < 134) stay in the dataset -- they carry legitimate
late-season signal already used by the existing `vegas_late_spread_delta`
feature, and they are NOT the games we are predicting.

Date-to-daynum conversion uses `MSeasons.csv`'s `DayZero` per season,
matching the convention already used by `audit_v4_gap_vegas.py`'s
`_build_day_zero_map()`.

```python
def filter_vegas_to_pre_tournament(
    vegas_df: pd.DataFrame,
    seasons_csv_path: Path = MANIA_DIR / "MSeasons.csv",
) -> pd.DataFrame:
    """Drop rows whose daynum (date - DayZero[season]) is >= 134.

    134 is the First Four day. Anything from the First Four onward is
    NCAA tournament and must not feed into v4's per-team-per-season
    Vegas aggregates, or it leaks tournament outcomes into LOSO test
    features.

    Returns a copy with the same schema as `vegas_df`. Rows whose
    season is missing from MSeasons.csv or whose date is unparseable
    are kept (defensive: do not silently drop legitimate
    regular-season rows due to a data hiccup; print a warning).
    """
```

## Success criteria

1. **Unit tests pass**:
   - Synthetic vegas_df with a mix of regular-season and tournament
     daynums: filtered output keeps regular-season rows, drops
     tournament-day rows.
   - Schema preserved: input columns == output columns.
   - Empty input -> empty output, no exception.
   - Missing day_zero for a row's season -> row kept + warning.

2. **Integration test (smoke)**: build vegas features with and
   without the filter; for 2024 UConn (TeamID 1163), the
   `vegas_avg_margin` from the filtered build matches the regular-
   season-only number (~+16.16, within 0.05); the unfiltered build
   matches the leaky number (~+18.13).

3. **Existing audit tests continue to pass**:
   `tests/test_audit_v4_gap_vegas.py` (10 tests) -- these use
   `load_vegas_lines()` directly and should be unchanged.

4. **No regression in non-Vegas paths**: `pytest tests/test_features
   tests/test_ingest tests/test_integration.py` still green.

## Anchors (post-fix expectations)

After this PR lands, the *next* PR will regenerate
`output/pairwise_v4.csv`. Expected directional shifts (predictions, not
guarantees):

| metric                              | current (leaky)        | expected (clean)              |
|-------------------------------------|------------------------|-------------------------------|
| v4 LOSO log loss                    | 0.4369 (22-season avg) | higher (worse), perhaps 0.45-0.47 |
| v4 LOSO accuracy                    | ~80.4% per-season avg  | lower, perhaps 73-77%         |
| v4 vs Vegas LL delta (audit)        | -0.114 (v4 better)     | smaller magnitude, possibly closer to 0 |
| v4 upset capture rate               | 56%                    | lower, perhaps 30-40%         |

If the regenerated v4 still cleanly beats Vegas across all buckets
after the fix, that's a real finding. If not, the audit's headline
needs to be retracted.

## What this does NOT fix

- Massey ordinals: if `data["massey"]` (loaded in `load_all_data()`)
  contains end-of-season-INCLUDING-tournament rankings, that's a
  separate leak class with its own fix path. Audit deferred to a
  separate PR.
- KenPom snapshots: same potential issue, same deferral.
- `feature_matrix_v2.py` (used by `enhanced_model_v2.py`): also calls
  `compute_vegas_features` and may have the same leak. v2 is not in
  the v4 production path so its fix is lower priority.

## Files of record

```
docs/superpowers/specs/2026-05-04-v4-vegas-leak-fix-design.md  -- this
docs/superpowers/plans/2026-05-04-v4-vegas-leak-fix.md         -- plan

src/enhanced_model_v3.py                                       -- add filter helper, wire it in
tests/test_vegas_leak_filter.py                                -- new unit + smoke tests

TODO.md                                                        -- record contamination + recovery roadmap
```
