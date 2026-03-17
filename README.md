# March Madness Bracket Prediction Model

A machine learning system that predicts NCAA tournament outcomes using team efficiency ratings, historical performance, and Vegas betting lines. Generates optimized bracket picks via Monte Carlo simulation.

**2026 Model Performance (Leave-One-Season-Out CV, 22 seasons):**

| Metric | Value |
|--------|-------|
| Log Loss | **0.456** |
| Accuracy | **79.0%** |
| Brier Score | 0.148 |
| AUC | 0.870 |

## How It Works

### Data Sources
- **Kaggle March Machine Learning Mania** — 124,529 game-level box scores (2003-2026), tournament results, seeds, and 197 ranking systems via Massey Ordinals
- **KenPom/Barttorvik** — season-level adjusted efficiency ratings, four factors, talent, experience, strength of schedule
- **The Prediction Tracker** — 95,319 games of Vegas closing point spreads (2003-2026)

### Features (52 total)
- **Custom adjusted efficiency ratings** — iterative opponent-adjusted offensive/defensive efficiency with recency weighting, computed from raw box scores
- **Four factors** — eFG%, turnover rate, offensive rebound rate, free throw rate (offense and defense)
- **Rolling form** — 30-day efficiency, last-10 win%, scoring momentum
- **Massey Ordinals ensemble** — ranks from 7 top systems (KenPom, Sagarin, Massey, Wolfe, Dolphin, Colley, RPI) plus composite
- **KenPom/Barttorvik aggregates** — KADJ EM, BARTHAG, talent rating, experience, elite SOS
- **Vegas features** — average spread, power rating, against-the-spread %, consistency

### Model
XGBoost binary classifier with Platt scaling calibration. Trained on symmetric matchup pairs (feature differences between teams). Hyperparameters tuned via Optuna with leave-one-season-out cross-validation.

### Bracket Generation
Monte Carlo simulation (10,000 iterations) with pre-computed pairwise win probabilities for all 2,016 possible matchups. Supports chalk (highest probability) and expected value (maximize points) bracket selection strategies.

## Project Structure

```
src/
  enhanced_model.py       # Core model with game-level features
  enhanced_model_v2.py    # + Vegas lines integration
  generate_bracket_real.py # Bracket generation with actual 2026 bracket
  kaggle_submission.py    # Kaggle competition submission generator
  pool_optimizer.py       # Alternative pool format optimizer
  run_pipeline.py         # End-to-end training pipeline
  ingest/                 # Data loading (Kaggle, cbbd API, Massey, Vegas)
  features/               # Feature engineering (efficiency, four factors, feature matrix)
  models/                 # XGBoost training, evaluation, Optuna tuning
  bracket/                # Monte Carlo simulator, strategies, output
tests/                    # 54 unit + integration tests
output/
  bracket.html            # Interactive bracket viewer (open in browser)
  bracket_2026.csv        # Team advancement probabilities
  submission_stage2.csv   # Kaggle competition submission
```

## Quick Start

```bash
# Install
pip install -e ".[dev]"

# Run the full pipeline (train + evaluate + generate bracket)
python src/enhanced_model_v2.py

# Generate Kaggle submission
python src/kaggle_submission.py

# Optimize alternative pool picks
python src/pool_optimizer.py

# Run tests
pytest tests/ -v
```

## Interactive Bracket Viewer

Open `output/bracket.html` in a browser for a full interactive bracket with:
- Win probabilities for every matchup
- Click to override model picks with your own
- Color-coded picks (blue = model, green = your override, orange = upset)
- Copy button for easy transcription
- Hover tooltips with advancement probabilities

## Model Evolution

| Version | Features | Log Loss | Accuracy | Key Addition |
|---------|----------|----------|----------|-------------|
| v1 | 34 | 0.570 | 70.5% | KenPom/Barttorvik season aggregates |
| v2 | 46 | 0.558 | 71.5% | + game-level box scores, rolling efficiency, Massey Ordinals |
| v3 | 52 | **0.456** | **79.0%** | + Vegas closing lines (18% log loss reduction) |

## 2026 Predictions

**Championship probabilities:**

| Seed | Team | Champ% |
|------|------|--------|
| 2 | Iowa St. | 15.9% |
| 1 | Arizona | 14.4% |
| 1 | Duke | 14.4% |
| 2 | Purdue | 10.8% |
| 2 | Houston | 9.3% |

Key insight: Duke has the easiest path to the Final Four but is an underdog in championship matchups against Iowa St. (31%), Arizona (39%), and Purdue (43%) — likely due to being the least experienced team in the field (rank 363/363).
