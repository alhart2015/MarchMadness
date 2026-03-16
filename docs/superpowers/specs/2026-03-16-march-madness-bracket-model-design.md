# March Madness Bracket Prediction Model — Design Spec

## Overview

A reusable Python system that ranks NCAA men's basketball teams by building custom adjusted efficiency ratings from raw game data, ensembles them with established external ranking systems, and generates optimized tournament brackets via Monte Carlo simulation.

**Goals:**
- Build custom power ratings (adjusted efficiency) from scratch for educational depth
- Improve accuracy by incorporating proven external signals (Massey Ordinals, Barttorvik)
- Produce win probabilities for any matchup and simulate full brackets
- Reusable year over year: swap in new season data, retrain, generate brackets
- Maximize chances of winning a bracket pool

## Architecture

Four-stage pipeline, each stage producing artifacts consumed by the next:

```
[1. Data Ingestion] → [2. Feature Engineering] → [3. Model Training] → [4. Bracket Generation]
     ↓                       ↓                         ↓                       ↓
  Raw CSVs &            Team-season             Trained models &          Filled bracket(s)
  API responses         feature matrix          win probability            + analysis
                                                 function
```

### Project Structure

```
MarchMadness/
├── data/
│   ├── raw/              # Kaggle CSVs, API downloads (gitignored)
│   ├── processed/        # Cleaned/merged dataframes (gitignored)
│   └── cache/            # cbbd API response cache
├── src/
│   ├── ingest/           # Data download & loading
│   ├── features/         # Feature engineering & custom ratings
│   ├── models/           # Model training & evaluation
│   └── bracket/          # Simulation & bracket output
├── notebooks/            # Exploratory analysis & visualization
├── models/               # Saved trained models (gitignored)
├── output/               # Generated brackets & reports
├── tests/
└── config.yaml           # Seasons, hyperparams, data source toggles
```

Each stage is independently runnable via CLI (`python -m src.ingest`, `python -m src.features`, etc.). Intermediate results are cached to disk as parquet files. `config.yaml` controls seasons, features, hyperparameters, and simulation settings.

## Stage 1: Data Ingestion

Three data sources, each with its own loader module:

### Source 1: Kaggle March Mania Dataset (Primary)

Downloaded via `kaggle` CLI or manually. Key files:

| File | Description |
|------|-------------|
| `MRegularSeasonDetailedResults.csv` | Box scores for every game since 2003 |
| `MNCAATourneyDetailedResults.csv` | Tournament box scores |
| `MNCAATourneySeeds.csv` | Seeds by year |
| `MMasseyOrdinals.csv` | 100+ ranking systems, daily snapshots |
| `MTeams.csv` | Team IDs and names |
| `MTeamConferences.csv` | Team-conference mappings by season |

### Source 2: cbbd (CollegeBasketballData.com API)

Free API key registration at CollegeBasketballData.com. Pulls Barttorvik-derived stats: adjusted efficiency (off/def), adjusted tempo, four factors, Barthag. Used primarily for current season data not yet in the Kaggle dataset. Responses cached locally with 24-hour TTL.

### Source 3: Massey Ratings Composite CSV

Direct download from masseyratings.com/cb/compare.htm. Provides current-day composite ranking across all systems. Lightweight late-season supplement.

### Data Unification

All sources joined on Kaggle's `TeamID` as the canonical key. A team name mapping utility resolves names across sources:

- **Algorithm:** `rapidfuzz` (token_sort_ratio) with a configurable confidence threshold (default 85). Matches above threshold are auto-accepted; matches between 70-85 are flagged for manual review; below 70 are left unmatched.
- **Manual overrides:** A hand-curated `data/team_name_overrides.csv` (checked into git) handles known problem cases (e.g., "UConn" → "Connecticut", "St. Mary's" → "Saint Mary's (CA)"). Overrides take precedence over fuzzy matching.
- **Validation:** After mapping, a validation step checks for unmapped teams and duplicate mappings. The pipeline halts with a clear error if any tournament team is unmapped.
- **The resolved mapping is persisted as `data/processed/team_id_mapping.csv`** and reused across runs. Re-matching only runs when new source data is detected.

## Stage 2: Feature Engineering

### Stage 2A: Custom Adjusted Efficiency Ratings

For each team-season, compute:

- **Adjusted Offensive Efficiency (AdjOE):** Points scored per 100 possessions, adjusted for opponent defensive strength
- **Adjusted Defensive Efficiency (AdjDE):** Points allowed per 100 possessions, adjusted for opponent offensive strength
- **Adjusted Tempo:** Possessions per 40 minutes, adjusted for opponent pace
- **Net Efficiency (AdjEM):** AdjOE - AdjDE (the power rating)

**Method:** Iterative ridge regression on point differential per possession. Start with raw efficiencies, iteratively adjust for strength of schedule until convergence (10-20 iterations). Home court advantage included as a fixed offset (~3.5 points, estimated from data).

**Possessions estimated from box scores:**
```
possessions ≈ FGA - OR + TO + 0.475 * FTA
```

**Recency weighting:** Exponential decay applied to game-level data before regression. Half-life default of 30 days (configurable in `config.yaml`). Applied as sample weights in the ridge regression — more recent games contribute more to the efficiency estimates.

**Convergence:** Fixed iteration count of 15 (configurable), which empirically matches KenPom's approach. Ratings typically stabilize within 10 iterations; 15 provides margin. A convergence diagnostic (max rating change across all teams) is logged each iteration for verification.

### Stage 2B: Full Feature Matrix

| Feature | Source | Description |
|---------|--------|-------------|
| `adj_oe`, `adj_de`, `adj_em` | Custom (2A) | Own efficiency ratings |
| `adj_tempo` | Custom (2A) | Adjusted pace |
| `off_efg`, `off_to_rate`, `off_or_rate`, `off_ft_rate` | Kaggle box scores | Offensive four factors |
| `def_efg`, `def_to_rate`, `def_or_rate`, `def_ft_rate` | Kaggle box scores | Defensive four factors |
| `seed` | Kaggle seeds | Tournament seed (1-16) |
| `conf_strength` | Derived | Average AdjEM of conference opponents |
| `massey_composite_rank` | Massey Ordinals | Composite rank from 100+ systems |
| `top_n_system_ranks` | Massey Ordinals | Ranks from 5 pre-selected systems: Pomeroy, Sagarin, BPI, T-Rank, RPI |
| `win_pct_last_10` | Kaggle results | Recent form indicator |
| `road_win_pct` | Kaggle results | Performance away from home |

For model training, each row is a **matchup pair**: features are the difference between Team A and Team B values, with the target being win/loss.

## Stage 3: Model Training & Evaluation

### Training Data

Every NCAA tournament game from 2003-2025 (~1,400 games with detailed box scores). Features computed from regular season data only — no leakage from tournament games.

### Model: XGBoost Binary Classifier

- **Input:** Matchup feature vector (Team A features - Team B features, plus seed delta)
- **Output:** P(Team A wins)
- **Symmetry:** Each game included twice (A vs B and B vs A with flipped target) to prevent positional bias

**Why XGBoost:** Handles feature interactions naturally, robust to correlated features, well-calibrated probabilities with tuning, proven performer on Kaggle for this competition.

### Calibration

Post-hoc Platt scaling (logistic regression on model outputs) to ensure predicted probabilities are well-calibrated.

### Evaluation

- **Primary method:** Leave-one-season-out cross-validation (train on all years except one, predict that year's tournament, repeat)
- **Primary metric:** Log loss / Brier score (Kaggle competition metric, measures probability calibration)
- **Secondary metrics:** Accuracy, AUC, bracket score (ESPN-style 1-2-4-8-16-32 points per round)
- **Baselines:** Seed-only model (always pick better seed), logistic regression on seed delta

### Hyperparameter Tuning

Bayesian optimization via Optuna over learning rate, max depth, subsample, colsample_bytree, using leave-one-season-out CV as the objective.

## Stage 4: Bracket Generation & Simulation

### Monte Carlo Simulation

- Takes the actual tournament bracket as a structured CSV (`data/raw/bracket_YYYY.csv`) with columns: `Region, Seed, TeamID, TeamName`. The bracket structure (which seeds play which) is hardcoded per the standard NCAA format (1v16, 8v9, etc.). First Four play-in games are pre-resolved before the main bracket — the winner is slotted into the appropriate seed position.
- Computes P(Team A beats Team B) for each possible matchup using the trained model
- Simulates the entire tournament 10,000+ times, sampling each game outcome from win probabilities
- Tracks advancement frequency per team per round

### Bracket Selection Strategies

Three modes:

1. **Chalk bracket:** Always pick the higher win-probability team. Simple baseline.

2. **Expected value bracket:** Pick the team that maximizes expected points under standard ESPN scoring (1-2-4-8-16-32) per bracket slot. Accounts for the upside of correctly picking upsets in later rounds.

3. **Pool-optimized bracket (stretch goal):** Maximize P(winning the pool) using public pick percentages from ESPN/Yahoo as a proxy for opponent behavior. Uses a greedy slot-by-slot algorithm: for each bracket slot, pick the team that maximizes `P(team reaches slot) * points_for_slot * (1 - public_pick_pct)^(pool_size - 1)`. This is an approximation — full bracket-space optimization is intractable. Marked as stretch goal; chalk and EV brackets are the core deliverables.

### Output

- Bracket printed to console in readable tree format
- CSV export: each matchup with teams, win probabilities, and pick
- Summary report: team advancement probabilities, most likely Final Four, champion, highest-value upsets
- Optional stretch goal: visual bracket HTML page

### Current Season Workflow

1. Download latest Kaggle data + pull current season from cbbd API
2. Recompute features for all current teams
3. Load trained model (or retrain)
4. Input actual bracket matchups (manually or parsed from bracket CSV)
5. Run simulation, generate bracket(s)

## Data Validation

Each stage validates its inputs before proceeding:

- **Post-ingestion:** Check for expected columns, no null TeamIDs, game counts per season within reasonable bounds (2,000-6,000 D1 games per season). Flag seasons with missing detailed box scores.
- **Post-feature engineering:** Every tournament team must have a complete feature vector. Log warnings for teams with fewer than 15 games (small sample). Validate efficiency ratings are in reasonable ranges (e.g., AdjEM between -40 and +40).
- **Pre-simulation:** Every team in the bracket input must exist in the feature matrix. Halt with clear error if not.
- **cbbd API fallback:** If the API is down or returns incomplete data, log a warning and proceed with Kaggle + Massey data only. The pipeline should never hard-fail due to an optional data source.

## Massey Ordinals Optimization

The `MMasseyOrdinals.csv` file can be hundreds of MB. At load time, filter to only the latest available snapshot per season (last day with data, typically Selection Sunday) rather than loading all daily snapshots. For the current season, use the most recent available date.

## Technical Decisions

- **Language:** Python 3.11+
- **Key dependencies:** pandas, numpy, scikit-learn, xgboost, optuna, rapidfuzz, cbbd, pyyaml
- **Data format:** Parquet for intermediate storage (fast, typed, compact)
- **Config-driven:** `config.yaml` for all tunable parameters
- **No database:** File-based storage is sufficient for this scale
- **Testing:** pytest, focused on data pipeline correctness and model reproducibility
- **Logging:** Python `logging` module, structured per-stage logs for debugging data issues
- **Model versioning:** Saved models include a metadata sidecar JSON with training date, config hash, seasons used, and feature list. Filenames follow `models/xgb_{YYYY}_{config_hash}.pkl` convention.
