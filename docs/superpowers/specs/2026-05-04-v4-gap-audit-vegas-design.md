# v4 Gap Audit vs Vegas Closing Lines -- Design

**Date:** 2026-05-04
**Branch:** feat/v4-gap-audit-vegas
**Predecessors:**
- HBT (NO-GO, framing-corrected): `docs/notes/2026-05-03-hierarchical-bt.md`
- Plain BT bracket-points re-test (NO-GO): `docs/notes/2026-05-04-bt-bracket-points.md`
- v9-C production swap (current production stage-2):
  `docs/superpowers/specs/2026-05-01-v9c-production-swap-design.md`

## Motivation

Two recent ensemble experiments (HBT, plain BT vs bracket points) both
failed. The PR 17 finding identified the actual problem: structural
diversity is necessary but not sufficient -- a stage-1 candidate also
needs **per-disagreement accuracy** in the regions where it differs
from v4. Plain BT was right only 27.9% of the time on v4-disagreements,
so v9-C cannot extract upset signal from it regardless of metric.

This shifts the question. Rather than testing more ensemble candidates
blind, **localize where v4 specifically loses** -- by bucket and by
calibration -- so we know what kind of signal would help. The user's
2159 / 3462 Kaggle finish confirms there is real headroom; we need to
find which buckets it's in.

This experiment audits v4's tournament predictions against Vegas
closing-line implied probabilities, broken down by round, higher-vs-
lower-seed status, v4-confidence band, and seed-difference magnitude.

## Goals

- Build per-tournament-game records for 2003-2025: `(season, day,
  team_a, team_b, p_v4, p_vegas, winner)`. Tournament-only -- regular-
  season games are out of scope.
- Compute per-bucket aggregate metrics: log loss, accuracy, calibration
  curve (predicted-prob vs empirical win-rate), per-bucket sample
  count.
- Compare v4 to Vegas. Identify 3-5 concrete bucket signatures where
  v4 specifically underperforms Vegas. Frame each finding as a
  candidate target for future feature engineering or post-processing
  ("v4 is over-confident on `<bucket>`; Vegas is more accurate by
  `<X>` log-loss").
- Ship a single audit note + a structured JSON of the per-bucket
  numbers + a small set of calibration / per-bucket bar PNGs.

## Non-Goals (deliberately deferred)

- **538 tournament-forecast audit.** Sourcing question (data
  accessibility across 22 seasons, scraping vs API). **NOT skipped --
  scheduled as the immediate follow-up audit on its own branch.**
  Reason for sequencing Vegas first: Vegas data is already ingested
  (`data/raw/vegas_lines/`) with team-name resolution already
  validated by `src/enhanced_model_v2.py:compute_vegas_features`,
  so Vegas-first is a quick win that surfaces the bucket structure;
  538 then tells us whether v4 also loses on the buckets where Vegas
  agrees with v4 (covering a different angle).
- **Roster-injury data.** External-data follow-up after Vegas + 538.
- **Top public Kaggle entry comparison.** Sourcing-availability
  question; deprioritized vs Vegas + 538.
- **Building a reusable diagnostic module.** Audit-as-script for now;
  if we find ourselves running the audit after every model change,
  factor a reusable `src/audit/v4_vs_external.py` then.
- **Acting on findings (feature engineering, post-processing).**
  This audit produces a *map* of weak spots; engineering against any
  of them is its own follow-up experiment with its own spec.
- **2026 audit.** No tournament outcomes for 2026 yet; the audit
  covers played seasons only (2003-2025, excluding 2020).
- **Per-team diagnostics.** Bucket-level only -- per-team variance is
  too small for ~1400 games / 22 seasons to surface signal cleanly.

## Approach

### Architecture

```
src/audit_v4_gap_vegas.py            -- one-shot driver
  |
  +-- load v4 pairwise probabilities (output/pairwise_v4.csv, dedup'd)
  +-- load Vegas closing lines (reuse load_vegas_lines() from enhanced_model_v2)
  +-- load tournament outcomes (MNCAATourneyCompactResults)
  +-- load tournament seeds (MNCAATourneySeeds)
  +-- resolve Vegas team names -> Kaggle TeamIDs (reuse existing
      _build_vegas_name_to_kaggle_map + _resolve_vegas_name)
  +-- join Vegas lines to tournament games on (Season, date, team-pair)
  +-- compute Vegas implied prob = norm.cdf(line / SIGMA), SIGMA=11.0
      (matching existing src/blend_sweep.py and src/alternate_bracket.py)
  +-- bucket each game by:
        - round (R64, R32, S16, E8, F4, Champ)
        - higher-vs-lower-seed binary (favorite_won?)
        - v4-confidence quintile (0.5..0.55, ..0.65, ..0.75, ..0.85, >0.85)
        - seed-diff magnitude (|seed_a - seed_b|, bucketed coarsely)
  +-- per-bucket aggregate: count, ll_v4, ll_vegas, ll_delta,
      acc_v4, acc_vegas, calibration (predicted bin -> empirical rate)
  +-- write output/v4_gap_audit_vegas.json
  +-- emit calibration plots:
        output/v4_gap_calibration_overall.png
        output/v4_gap_calibration_by_round.png
        output/v4_gap_per_bucket_ll_delta.png
  +-- print top-N "v4 underperforms Vegas" bucket signatures
```

### Vegas implied probability

Standard CBB convention from existing modules:

```python
SIGMA = 11.0  # CBB sigma for spread-to-prob via N(0, SIGMA)
p_home_wins = scipy.stats.norm.cdf(line / SIGMA)
```

`line` is positive when home is favored (per The Prediction Tracker
schema: `line` = points home is favored by; e.g., `line=5.5` means
home -5.5).

After resolving home_id and road_id to Kaggle TeamIDs, the canonical
`(team_a, team_b)` pair has `team_a < team_b`, with `p_a_wins`:

- If `home_id < road_id`: `p_a_wins = norm.cdf(line / SIGMA)`
- If `home_id > road_id`: `p_a_wins = norm.cdf(-line / SIGMA) = 1 - norm.cdf(line / SIGMA)`

For neutral-court tournament games, the home/road labels in the Vegas
data still encode which team the spread refers to; the math doesn't
change.

### Joining Vegas to tournament games

Each tournament game in `MNCAATourneyCompactResults.csv` has
`Season, DayNum, WTeamID, LTeamID, WScore, LScore, WLoc`. To get the
Vegas line for that game, we need to find the corresponding row in
the Vegas data.

`load_vegas_lines()` returns rows with `season, date, home, road,
hscore, rscore, line`. The joining key is the unordered team pair on
the actual game date. Approach:

1. Convert Vegas `date` (`MM/DD/YYYY` strings) to `(season,
   day_offset)` where `season` follows the Kaggle convention
   (season ending in March/April of year Y -> Season=Y) and
   `day_offset` is the integer days since season start. Kaggle's
   `DayNum=0` is roughly Nov-1 of season-start year (`Season=2024
   day 0 = Nov 1, 2023`; verify against `MSeasons.csv DayZero`).
2. Map Vegas (home, road) names to (home_id, road_id) Kaggle
   TeamIDs via `_build_vegas_name_to_kaggle_map` +
   `_resolve_vegas_name` (existing).
3. For each tournament game `(Season, DayNum, WTeamID, LTeamID)`,
   find the Vegas row with matching season + day_offset (within +/- 1
   day for slack) and matching unordered team pair. If found, attach
   the line to the tournament game.

Coverage caveat: Vegas data may not include every tournament game in
every year (especially older seasons). Report the join coverage rate
and only audit games where both v4 and Vegas have predictions.

### Bucket definitions

| bucket | values | sample counts (rough, 22 seasons * 63 games = ~1386) |
|--------|--------|------------------------------------------------------|
| `round` | R64, R32, S16, E8, F4, Champ | 32, 16, 8, 4, 2, 1 per season; 704, 352, 176, 88, 44, 22 total |
| `higher_seed_won` (binary) | True (chalk), False (upset) | ~70/30 split historically |
| `v4_confidence_quintile` | (0.50-0.60], (0.60-0.70], (0.70-0.80], (0.80-0.90], (0.90-1.00] | quintiles by predicted prob for the favored side |
| `seed_diff_bucket` | |seed_a - seed_b| in {0-2, 3-5, 6-9, 10-15} | based on seeded slot pairings |

For each bucket cell with N >= 30 games, report metrics. Smaller cells
get reported but flagged as low-N.

### Per-bucket metrics

For each (bucket, value) cell:

- `n_games`: count of games where both v4 and Vegas have probabilities.
- `ll_v4`, `ll_vegas`: log loss on the actual winner.
- `ll_delta = ll_v4 - ll_vegas`: positive = v4 worse than Vegas.
- `acc_v4`, `acc_vegas`: accuracy on chalk pick.
- `calibration_v4`, `calibration_vegas`: list of `(predicted_bin,
  empirical_rate, n)` entries, with predicted bins of width 0.05 over
  [0.5, 1.0]. ECE (expected calibration error) computed as
  `sum_bin (n_bin / n_total) * |predicted_bin_mid - empirical_rate|`.
- `p_v4_minus_vegas_mean`: mean of `p_v4 - p_vegas` -- positive means
  v4 is more confident in the favored side than Vegas in this bucket.

Findings note prioritizes buckets with `n_games >= 50` and `ll_delta
>= 0.02` as "v4 specifically underperforms Vegas here."

### Charts

PNGs via matplotlib (already a project dep via XGBoost stack):

1. **Overall calibration** -- predicted-prob bin (x) vs empirical
   win-rate (y), one line per model (v4 vs Vegas). Diagonal reference.
2. **Calibration by round** -- 6 panels, one per round, same axes.
   Identifies if v4 is well-calibrated overall but wrong per-round.
3. **Per-bucket LL delta** -- horizontal bar chart of `ll_v4 -
   ll_vegas` per bucket. Sorted descending. Highlights worst buckets.

Commit the PNGs alongside the findings note. Each is small (~50-150
KB).

### Output schema

`output/v4_gap_audit_vegas.json`:

```json
{
  "config": {
    "v4_pairwise": "output/pairwise_v4.csv",
    "seasons": [2003, 2004, ..., 2025],
    "sigma": 11.0
  },
  "join_coverage": {
    "n_tournament_games": NNNN,
    "n_with_v4": NNNN,
    "n_with_vegas": NNNN,
    "n_both": NNNN,
    "missing_vegas_seasons": {2003: NN, ...}
  },
  "overall": {
    "n_games": NNNN,
    "ll_v4": ..., "ll_vegas": ...,
    "acc_v4": ..., "acc_vegas": ...,
    "ece_v4": ..., "ece_vegas": ...,
    "calibration_v4": [{"bin": [0.5, 0.55], "n": ..., "empirical": ...}, ...],
    "calibration_vegas": [...]
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
      "bucket": "round=R64, v4_confidence>0.90",
      "n_games": ...,
      "ll_v4": ..., "ll_vegas": ...,
      "ll_delta": +0.05,
      "interpretation": "v4 is over-confident on R64 chalk picks; ..."
    },
    ... 3-5 entries
  ]
}
```

## Anchor / sanity checks

- **Overall log loss matches v4's known LL on tournament games.**
  v4's standalone LL on the 1449 played 2003-2025 tournament games is
  0.4369 (per the plain-BT diagnostic). The audit's `overall.ll_v4`
  must agree to floating-point precision. If not, there's a bug in
  the join or the LL calc.
- **Vegas LL is plausible.** Public CBB-Vegas calibration gives LL
  around 0.42-0.46 on tournament games. The audit's
  `overall.ll_vegas` should land in that band; far outside =
  spread-to-prob conversion or join is wrong.
- **Vegas accuracy is plausible.** Vegas predicts the favorite right
  about 70-72% in CBB tournament games. `overall.acc_vegas` should
  agree.
- **Calibration diagonal.** A well-calibrated model has `empirical
  ~ predicted_bin_mid`. v4's overall calibration plot should be
  *roughly* on the diagonal (with known over/under-confidence
  patterns visible). If v4 plots *way* off-diagonal, the LL number
  is hiding a calibration problem worth flagging.
- **Bucket sums.** `sum(per-round n_games) = overall.n_games`.
  Regression guard.

## Falsification / what would make us re-think

- **Vegas join coverage < 60%.** If we can't recover Vegas lines for
  most tournament games, the audit is unreliable. Halt; investigate
  the date / team-name resolution before trusting any per-bucket
  numbers.
- **All buckets show `ll_delta < 0.01`.** v4 is roughly Vegas-tier
  everywhere -- which means our headroom isn't in calibration
  /per-bucket weakness; it's somewhere else (specific upsets, late
  rounds, etc.). The audit still produces value as a "v4 is at
  Vegas-level" proof point, but the immediate followups would be
  different.
- **One bucket dominates the LL gap.** E.g., if the entire LL_delta
  is concentrated in S16+ rounds (small samples), the finding is
  fragile. Findings note must explicitly call out fragile vs durable
  signatures.

## Test plan

- `tests/test_audit_v4_gap_vegas.py`:
  - Unit: `spread_to_prob(0.0) == 0.5`, `spread_to_prob(11.0) ~ 0.84`
    (`norm.cdf(1.0)`), `spread_to_prob(-5.5) ~ 0.31`. Anchor on the
    SIGMA convention.
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

1. **Vegas team-name resolution gaps in older seasons.** Common in
   2003-2008. Mitigation: reuse the existing fuzzy matcher; report
   per-season coverage; only audit games with both predictions.
   Findings note's "join_coverage" section makes any coverage hole
   visible.
2. **Vegas date-to-Season alignment errors.** A tournament game on
   Mar 17, 2024 belongs to `Season=2024` (Kaggle convention). The
   conversion needs `MSeasons.csv` `DayZero` to map dates to
   `(Season, DayNum)`. Sanity-check by spot-checking a few known
   games.
3. **Spread-to-prob conversion choice (SIGMA=11).** This is
   approximate (true CBB SD varies ~10-12). Findings note can flag
   the convention but the existing codebase already uses SIGMA=11
   so we stay consistent. Sensitivity sweep over SIGMA in {10, 11,
   12} can be a follow-up if any finding looks knife-edge.
4. **Calibration plot interpretability.** ECE is a single number
   that hides per-bin behavior. Plots fix this; we report both ECE
   and the per-bin table.
5. **Bucket multiplicity / cherry-picking.** With four bucketing
   axes, we'll see ~20+ cells. Findings note must report ALL cells
   (in the JSON), and only the top-3-to-5 "weak spots" are
   highlighted as call-to-action. No p-hacking on the cells.

## Lessons from prior experiments carried forward

- **Diagnostic-first.** Single-script audit before any feature
  engineering. Same paid-for-itself logic as the BT and HBT gates.
- **Reuse existing infrastructure.** `load_vegas_lines`,
  `_build_vegas_name_to_kaggle_map`, `_resolve_vegas_name` all
  exist. Standard SIGMA=11 already used in two places.
- **Force-add output artifacts** (`output/` is gitignored) per
  precedent.
- **Anchor before trusting.** Overall LL_v4 must match the known
  0.4369 baseline.

## Out-of-scope follow-ups (post-Vegas-audit)

- **538 tournament-forecast audit.** Same buckets, same metrics,
  different external benchmark. Sourcing question is the gating
  unknown; sourcing investigation is the first task on that
  follow-up branch.
- **Sensitivity sweep over SIGMA** (10, 11, 12) for spread-to-prob.
- **Per-team Vegas vs v4 outliers** (which teams does v4
  systematically over-predict?).
- **Acting on a weak-spot finding.** Each weak-spot signature is a
  candidate target for feature engineering / post-processing /
  v4-retraining; each is its own follow-up experiment.

## File-touch summary

```
new   docs/superpowers/specs/2026-05-04-v4-gap-audit-vegas-design.md
new   docs/superpowers/plans/2026-05-04-v4-gap-audit-vegas.md
new   src/audit_v4_gap_vegas.py
new   tests/test_audit_v4_gap_vegas.py
new   output/v4_gap_audit_vegas.json                  (force-added)
new   output/v4_gap_calibration_overall.png           (force-added)
new   output/v4_gap_calibration_by_round.png          (force-added)
new   output/v4_gap_per_bucket_ll_delta.png           (force-added)
new   docs/notes/2026-05-04-v4-gap-audit-vegas.md     (findings)

edit  TODO.md                                         (mark active queue
                                                       item #1 in progress
                                                       / done; add 538 audit
                                                       as #2 since external
                                                       data fits there)
```
