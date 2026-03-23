# Model v3 Design Spec — Structural Improvements for 2027 Tournament

## Problem Statement

The v2 model achieved 0.456 log loss and 79% accuracy on LOSO CV, with strong
R64 performance (78%, perfectly calibrated) but degraded R32 predictions (62.5%,
overconfident). The model uses season-level aggregate features that miss
late-season dynamics, and trains on only ~1,400 tournament games.

## Goals

- Improve R32+ calibration without degrading R64 performance
- Generalize across historical seasons, not overfit to 2026 failure modes
- Maintain the existing pipeline structure (XGBoost + Platt + LOSO CV + Monte Carlo)
- Create `enhanced_model_v3.py` alongside v2 as a baseline

## Non-Goals

- Architecture rethink (ensemble, two-stage model) — see TODO.md for future work
- Seed-based matchup history features — too blunt, model should learn from real features
- Mid-tournament adjustment — bracket is locked pre-tournament

## Changes

### 1. New Features (+10, total ~62)

**Opponent-adjusted late-season metrics (4 features):**
- `late_adj_oe` — offensive efficiency, last 30 days, against top-100 opponents
- `late_adj_de` — defensive efficiency, same filter
- `late_adj_em` — efficiency margin (oe - de)
- `late_sos` — average opponent strength in the late-season window

"Top-100" defined as: end-of-season KenPom `KADJ EM RANK` <= 100 from `KenPom
Barttorvik.csv`. For seasons/teams missing KenPom data, fall back to Massey
composite rank. Using end-of-season ranking retroactively is fine — the top 100
is stable enough late-season that contemporary vs final ranking is negligible.

Note: The 30-day window for `late_adj_*` is deliberately shorter than the 45-day
window for trajectory features (below). Trajectory needs more data points for a
stable slope; late-season quality is about the most recent stretch.

These features use raw efficiency (points per 100 possessions against the
filtered opponents), not iterative opponent adjustment — the small sample size
(3-8 games per team in the window) makes iterative convergence unstable. This
is distinct from the existing `rolling_oe`/`rolling_de` features which cover
*all* opponents in the window, not just quality opponents.

These are the highest-priority additions. The v2 model's biggest misses
(Gonzaga, Virginia, Louisville, Texas Tech) were teams whose season averages
looked strong but whose late-season performance against quality opponents
told a different story.

**Late-season trajectory (2 features):**
- `efficiency_trend` — linear slope of per-game adjusted efficiency margin
  over the final 45 days
- `margin_trend` — slope of actual scoring margin over same window

Captures "peaking vs fading" independent of absolute level.

**Conference tournament performance (2-3 features):**
- `conf_tourney_wins` — number of conference tournament wins (0-4)
- `conf_tourney_champ` — binary flag
- `conf_tourney_margin` — average scoring margin in conference tournament games
  (requires joining `MConferenceTourneyGames.csv` with
  `MRegularSeasonCompactResults.csv` on Season/DayNum/TeamIDs to get scores;
  if conf tourney games are not in the results file, drop this feature and
  keep the other two)

These are exploratory — we add them and let XGBoost's feature importance
and CV performance determine if they're signal or noise.

**Vegas line trend (1 feature):**
- `vegas_late_spread_delta` — average Vegas spread in last 30 days minus
  season average spread. Negative = market sees the team getting stronger.

Note: The v2 Vegas pipeline does not parse game dates (it uses index order for
recency weighting). This feature requires parsing the `MM/DD/YYYY` date column
in the Vegas CSVs to split games into "last 30 days" vs "season average."

### 2. Expanded Training Data

**Primary (weight = 1.0):**
Tournament games, same as v2. ~1,400 games across 22 seasons.

**Supplemental (weight = 0.25):**
Regular season games from February 1 onward, filtered to matchups where both
teams finished in the top 80 of KenPom (tournament-caliber). Estimated
~3,000-5,000 additional games across 22 seasons.

Rationale: The model needs more examples of "good team vs good team" matchups
to learn well. Late-season regular season games between strong teams are the
closest proxy for tournament games. The 0.25 weight ensures they inform the
model without drowning out actual tournament signal.

XGBoost supports `sample_weight` natively. Note: `train_model` in
`src/models/train.py` currently does not accept `sample_weight` — the function
signature and `CalibratedClassifierCV.fit()` call need modification to pass
weights through.

Important: Platt calibration (via `CalibratedClassifierCV`) should be
restricted to tournament-game rows only. Supplemental games inform the XGBoost
base model but should not influence the probability calibration step, since
calibration should reflect tournament dynamics specifically.

LOSO CV continues to evaluate on tournament games only — supplemental games
are training data only.

Acceptance criterion: If supplemental training data degrades LOSO tournament
log loss compared to v2 baseline, drop it and use tournament-only training.

### 3. Recency Weighting (Experimental)

Test during CV: apply a per-season weight multiplier where recent seasons
count more (e.g., 5% decay per year back from the evaluation season).

Rationale: The game has evolved (pace, three-point volume, transfer portal).
Recent seasons may be more informative about current dynamics.

Approach: Run LOSO CV with and without recency weighting. Keep only if it
improves log loss. The weight stacks multiplicatively with the
tournament/supplemental weight.

### 4. R64 Line Blending (Post-Processing)

After the model generates pairwise probabilities, blend R64 predictions with
game-specific Vegas closing lines:

```
blended_prob = (1 - w) * model_prob + w * vegas_implied_prob
```

- Default `w = 0.35` (from 2026 blend sweep analysis)
- Stored as a config parameter for easy tuning
- Only applied to R64 games where a closing line is available
- R32+ games use model probabilities unchanged
- Vegas implied probability computed via probit model: `P = Phi(spread / 11.0)`

This is a bracket generation step, not a model change. Clean separation from
training.

### 5. What Stays the Same

- XGBoost binary classifier with Platt scaling calibration
- Leave-one-season-out cross-validation
- Optuna hyperparameter tuning (30 trials)
- Monte Carlo bracket simulation (10,000 iterations)
- All existing 52 features (kept, 10 new added alongside)
- Symmetric matchup pair representation (feature differences)
- Pipeline structure and output format

## Implementation Notes

- New file: `src/enhanced_model_v3.py` (v2 preserved as baseline)
- New feature functions go in `src/features/` modules
- Conference tournament data available in `MConferenceTourneyGames.csv`
- Late-season filtering uses game dates already present in detailed results
- Top-80 filtering for supplemental training uses KenPom rankings or
  Massey composite as proxy

## Evaluation

- Compare v3 LOSO CV metrics against v2 baseline (log loss, accuracy, Brier, AUC)
- Per-season breakdown to check for consistency and variance
- Feature importance analysis to validate new features contribute; if any of
  the 10 new features rank in the bottom quartile of importance AND removing
  them improves CV stability, drop them
- Retroactively score 2026 bracket to check if v3 would have done better
- Report recency weighting results (with vs without)
- Report supplemental training data results (with vs without)
- Ablation: run v3 with only the new features (no training methodology
  changes) and vice versa, to attribute improvement to features vs training
