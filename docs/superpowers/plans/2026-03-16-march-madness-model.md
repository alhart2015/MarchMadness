# March Madness Bracket Prediction Model — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reusable Python system that produces NCAA tournament bracket predictions via custom efficiency ratings, external ranking ensembles, and Monte Carlo simulation.

**Architecture:** Four-stage pipeline (Ingest → Features → Model → Bracket). Each stage runs independently via CLI, caches intermediate results as parquet, and is configured via `config.yaml`.

**Tech Stack:** Python 3.11+, pandas, numpy, scikit-learn, xgboost, optuna, rapidfuzz, cbbd, pyyaml, pytest

---

## File Structure

```
MarchMadness/
├── pyproject.toml
├── .gitignore
├── config.yaml
├── data/
│   ├── raw/                        # gitignored
│   ├── processed/                  # gitignored
│   ├── cache/                      # gitignored
│   └── team_name_overrides.csv     # checked in
├── src/
│   ├── __init__.py
│   ├── config.py                   # Load/validate config.yaml
│   ├── ingest/
│   │   ├── __init__.py
│   │   ├── __main__.py             # CLI: python -m src.ingest
│   │   ├── kaggle_loader.py        # Load Kaggle CSVs
│   │   ├── cbbd_loader.py          # CollegeBasketballData API
│   │   ├── massey_loader.py        # Massey composite CSV
│   │   ├── team_mapping.py         # Fuzzy name matching + overrides
│   │   └── validation.py           # Post-ingestion checks
│   ├── features/
│   │   ├── __init__.py
│   │   ├── __main__.py             # CLI: python -m src.features
│   │   ├── efficiency.py           # Iterative adjusted efficiency ratings
│   │   ├── four_factors.py         # Compute four factors from box scores
│   │   └── feature_matrix.py       # Assemble full feature matrix + validation
│   ├── models/
│   │   ├── __init__.py
│   │   ├── __main__.py             # CLI: python -m src.models
│   │   ├── matchup.py              # Build symmetric matchup training data
│   │   ├── train.py                # XGBoost training + Platt calibration
│   │   ├── evaluate.py             # Leave-one-season-out CV + metrics
│   │   └── baselines.py            # Seed-only + logistic regression baselines
│   └── bracket/
│       ├── __init__.py
│       ├── __main__.py             # CLI: python -m src.bracket
│       ├── simulator.py            # Monte Carlo tournament simulation
│       ├── strategies.py           # Chalk + EV bracket selection
│       └── output.py               # Console, CSV, summary report
├── tests/
│   ├── conftest.py                 # Shared fixtures (sample data)
│   ├── test_config.py
│   ├── test_ingest/
│   │   ├── test_kaggle_loader.py
│   │   ├── test_cbbd_loader.py
│   │   ├── test_massey_loader.py
│   │   ├── test_team_mapping.py
│   │   └── test_validation.py
│   ├── test_features/
│   │   ├── test_efficiency.py
│   │   ├── test_four_factors.py
│   │   └── test_feature_matrix.py
│   ├── test_models/
│   │   ├── test_matchup.py
│   │   ├── test_train.py
│   │   └── test_evaluate.py
│   └── test_bracket/
│       ├── test_simulator.py
│       ├── test_strategies.py
│       └── test_output.py
├── models/                         # gitignored
├── output/                         # gitignored
└── notebooks/
```

---

## Chunk 1: Project Scaffolding + Data Ingestion

### Task 1: Project Setup

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `config.yaml`
- Create: `src/__init__.py`
- Create: `tests/conftest.py`

- [ ] **Step 1: Create pyproject.toml**

```toml
[project]
name = "march-madness"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "pandas>=2.0",
    "numpy>=1.24",
    "pyarrow>=14.0",
    "scikit-learn>=1.3",
    "xgboost>=2.0",
    "optuna>=3.4",
    "rapidfuzz>=3.0",
    "cbbd>=1.20",
    "pyyaml>=6.0",
    "requests>=2.31",
    "joblib>=1.3",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.4",
    "pytest-cov>=4.1",
]

[build-system]
requires = ["setuptools>=68.0"]
build-backend = "setuptools.build_meta"

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["."]
```

- [ ] **Step 2: Create .gitignore**

```
data/raw/
data/processed/
data/cache/
models/
output/
__pycache__/
*.pyc
.env
*.egg-info/
dist/
build/
.venv/
```

- [ ] **Step 3: Create config.yaml**

```yaml
# March Madness Bracket Model Configuration

data:
  kaggle_dir: "data/raw/kaggle"
  cache_dir: "data/cache"
  processed_dir: "data/processed"
  team_overrides: "data/team_name_overrides.csv"

seasons:
  train_start: 2003
  train_end: 2025
  predict_season: 2026

efficiency:
  iterations: 15
  recency_half_life_days: 30
  home_court_advantage: 3.5
  ridge_alpha: 1.0

matching:
  auto_accept_threshold: 85
  review_threshold: 70
  algorithm: "token_sort_ratio"

model:
  random_seed: 42
  n_simulations: 10000

massey:
  systems:
    - "POM"   # Pomeroy / KenPom
    - "SAG"   # Sagarin
    - "BPI"   # ESPN BPI
    - "TRK"   # T-Rank / Barttorvik
    - "RPI"   # RPI

bracket:
  scoring: [1, 2, 4, 8, 16, 32]  # points per round
  strategies: ["chalk", "expected_value"]
```

- [ ] **Step 4: Create src/__init__.py** (empty file)

- [ ] **Step 5: Create src/config.py**

```python
"""Load and validate config.yaml."""

import logging
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

_DEFAULT_CONFIG_PATH = Path("config.yaml")


def load_config(path: Path | None = None) -> dict[str, Any]:
    """Load config from YAML file."""
    config_path = path or _DEFAULT_CONFIG_PATH
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path) as f:
        config = yaml.safe_load(f)
    _validate(config)
    return config


def _validate(config: dict[str, Any]) -> None:
    """Validate required config keys exist."""
    required_sections = ["data", "seasons", "efficiency", "model"]
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required config section: {section}")
    if config["seasons"]["train_start"] >= config["seasons"]["train_end"]:
        raise ValueError("train_start must be before train_end")
```

- [ ] **Step 6: Create tests/conftest.py with sample data fixtures**

```python
"""Shared test fixtures."""

import pandas as pd
import pytest


@pytest.fixture
def sample_detailed_results():
    """Minimal detailed results DataFrame matching Kaggle schema."""
    return pd.DataFrame({
        "Season": [2023, 2023, 2023, 2023],
        "DayNum": [11, 11, 15, 15],
        "WTeamID": [1101, 1102, 1101, 1103],
        "WScore": [75, 80, 68, 90],
        "LTeamID": [1103, 1104, 1102, 1104],
        "LScore": [60, 70, 65, 55],
        "WLoc": ["H", "A", "N", "H"],
        "NumOT": [0, 0, 0, 0],
        "WFGM": [28, 30, 25, 35],
        "WFGA": [55, 60, 52, 62],
        "WFGM3": [8, 10, 7, 12],
        "WFGA3": [20, 25, 18, 28],
        "WFTM": [11, 10, 11, 8],
        "WFTA": [15, 14, 14, 10],
        "WOR": [10, 12, 8, 14],
        "WDR": [22, 20, 24, 25],
        "WAst": [15, 18, 12, 20],
        "WTO": [12, 10, 14, 8],
        "WStl": [7, 5, 6, 9],
        "WBlk": [3, 4, 2, 5],
        "WPF": [18, 16, 20, 14],
        "LFGM": [22, 26, 24, 20],
        "LFGA": [58, 62, 55, 60],
        "LFGM3": [6, 8, 7, 5],
        "LFGA3": [18, 22, 20, 16],
        "LFTM": [10, 10, 10, 10],
        "LFTA": [14, 13, 12, 15],
        "LOR": [8, 10, 9, 7],
        "LDR": [20, 18, 22, 19],
        "LAst": [12, 14, 13, 10],
        "LTO": [15, 13, 12, 16],
        "LStl": [5, 6, 7, 4],
        "LBlk": [2, 3, 3, 2],
        "LPF": [16, 15, 17, 12],
    })


@pytest.fixture
def sample_teams():
    """Minimal teams DataFrame."""
    return pd.DataFrame({
        "TeamID": [1101, 1102, 1103, 1104],
        "TeamName": ["Abilene Chr", "Air Force", "Akron", "Alabama"],
    })


@pytest.fixture
def sample_seeds():
    """Minimal seeds DataFrame."""
    return pd.DataFrame({
        "Season": [2023, 2023, 2023, 2023],
        "Seed": ["W01", "W02", "W03", "W04"],
        "TeamID": [1101, 1102, 1103, 1104],
    })


@pytest.fixture
def sample_config(tmp_path):
    """Minimal config dict for testing."""
    return {
        "data": {
            "kaggle_dir": str(tmp_path / "raw"),
            "cache_dir": str(tmp_path / "cache"),
            "processed_dir": str(tmp_path / "processed"),
            "team_overrides": str(tmp_path / "overrides.csv"),
        },
        "seasons": {
            "train_start": 2003,
            "train_end": 2025,
            "predict_season": 2026,
        },
        "efficiency": {
            "iterations": 15,
            "recency_half_life_days": 30,
            "home_court_advantage": 3.5,
            "ridge_alpha": 1.0,
        },
        "matching": {
            "auto_accept_threshold": 85,
            "review_threshold": 70,
            "algorithm": "token_sort_ratio",
        },
        "model": {
            "random_seed": 42,
            "n_simulations": 10000,
        },
        "massey": {
            "systems": ["POM", "SAG", "BPI", "TRK", "RPI"],
        },
        "bracket": {
            "scoring": [1, 2, 4, 8, 16, 32],
            "strategies": ["chalk", "expected_value"],
        },
    }
```

- [ ] **Step 7: Create test_config.py and verify**

```python
"""Tests for config loading."""

from pathlib import Path

import pytest
import yaml

from src.config import load_config


def test_load_config_valid(tmp_path, sample_config):
    config_path = tmp_path / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(sample_config, f)
    result = load_config(config_path)
    assert result["seasons"]["train_start"] == 2003


def test_load_config_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "nonexistent.yaml")


def test_load_config_invalid_seasons(tmp_path, sample_config):
    sample_config["seasons"]["train_start"] = 2026
    sample_config["seasons"]["train_end"] = 2003
    config_path = tmp_path / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(sample_config, f)
    with pytest.raises(ValueError, match="train_start must be before"):
        load_config(config_path)
```

Run: `pytest tests/test_config.py -v`
Expected: 3 tests PASS

- [ ] **Step 8: Commit**

```bash
git add pyproject.toml .gitignore config.yaml src/__init__.py src/config.py tests/conftest.py tests/test_config.py
git commit -m "feat: project scaffolding with config, dependencies, and test fixtures"
```

---

### Task 2: Kaggle Data Loader

**Files:**
- Create: `src/ingest/__init__.py`
- Create: `src/ingest/kaggle_loader.py`
- Create: `tests/test_ingest/test_kaggle_loader.py`

- [ ] **Step 1: Write the failing tests**

```python
"""Tests for Kaggle data loading."""

import pandas as pd
import pytest

from src.ingest.kaggle_loader import load_kaggle_data


@pytest.fixture
def kaggle_dir(tmp_path, sample_detailed_results, sample_teams, sample_seeds):
    """Create a fake Kaggle data directory with CSVs."""
    d = tmp_path / "kaggle"
    d.mkdir()
    sample_detailed_results.to_csv(d / "MRegularSeasonDetailedResults.csv", index=False)
    sample_teams.to_csv(d / "MTeams.csv", index=False)
    sample_seeds.to_csv(d / "MNCAATourneySeeds.csv", index=False)
    # Create minimal compact results
    compact = sample_detailed_results[["Season", "DayNum", "WTeamID", "WScore", "LTeamID", "LScore", "WLoc", "NumOT"]]
    compact.to_csv(d / "MRegularSeasonCompactResults.csv", index=False)
    compact.to_csv(d / "MNCAATourneyCompactResults.csv", index=False)
    sample_detailed_results.to_csv(d / "MNCAATourneyDetailedResults.csv", index=False)
    # Minimal conferences
    pd.DataFrame({
        "ConfAbbrev": ["big12", "mwc", "mac", "sec"],
        "Description": ["Big 12", "Mountain West", "Mid-American", "SEC"],
    }).to_csv(d / "MConferences.csv", index=False)
    pd.DataFrame({
        "Season": [2023, 2023, 2023, 2023],
        "TeamID": [1101, 1102, 1103, 1104],
        "ConfAbbrev": ["big12", "mwc", "mac", "sec"],
    }).to_csv(d / "MTeamConferences.csv", index=False)
    # Minimal Massey ordinals
    pd.DataFrame({
        "Season": [2023, 2023, 2023, 2023],
        "RankingDayNum": [128, 128, 128, 128],
        "SystemName": ["POM", "POM", "POM", "POM"],
        "TeamID": [1101, 1102, 1103, 1104],
        "OrdinalRank": [1, 5, 50, 100],
    }).to_csv(d / "MMasseyOrdinals.csv", index=False)
    return d


def test_load_kaggle_data_returns_dict(kaggle_dir):
    data = load_kaggle_data(str(kaggle_dir))
    assert isinstance(data, dict)
    assert "teams" in data
    assert "regular_season" in data
    assert "tourney_results" in data
    assert "seeds" in data
    assert "massey" in data
    assert "conferences" in data
    assert "team_conferences" in data


def test_load_kaggle_data_teams_shape(kaggle_dir):
    data = load_kaggle_data(str(kaggle_dir))
    assert len(data["teams"]) == 4
    assert "TeamID" in data["teams"].columns
    assert "TeamName" in data["teams"].columns


def test_load_kaggle_data_massey_filtered(kaggle_dir):
    """Massey ordinals should be filtered to latest day per season."""
    data = load_kaggle_data(str(kaggle_dir))
    # Only one day in our test data, so all rows kept
    assert len(data["massey"]) == 4


def test_load_kaggle_data_missing_dir():
    with pytest.raises(FileNotFoundError):
        load_kaggle_data("/nonexistent/path")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_ingest/test_kaggle_loader.py -v`
Expected: FAIL (module not found)

- [ ] **Step 3: Implement kaggle_loader.py**

```python
"""Load Kaggle March Mania CSV files into DataFrames."""

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

_REQUIRED_FILES = {
    "teams": "MTeams.csv",
    "regular_season": "MRegularSeasonDetailedResults.csv",
    "tourney_results": "MNCAATourneyDetailedResults.csv",
    "seeds": "MNCAATourneySeeds.csv",
    "massey": "MMasseyOrdinals.csv",
    "conferences": "MConferences.csv",
    "team_conferences": "MTeamConferences.csv",
}


def load_kaggle_data(kaggle_dir: str) -> dict[str, pd.DataFrame]:
    """Load all required Kaggle CSVs from the given directory.

    Returns a dict mapping dataset name to DataFrame.
    Massey ordinals are filtered to the latest snapshot per season.
    """
    kaggle_path = Path(kaggle_dir)
    if not kaggle_path.exists():
        raise FileNotFoundError(f"Kaggle directory not found: {kaggle_dir}")

    data = {}
    for name, filename in _REQUIRED_FILES.items():
        filepath = kaggle_path / filename
        if not filepath.exists():
            raise FileNotFoundError(f"Required file missing: {filepath}")
        logger.info("Loading %s from %s", name, filepath)
        data[name] = pd.read_csv(filepath)

    # Filter Massey ordinals to latest day per season
    massey = data["massey"]
    latest_day = massey.groupby("Season")["RankingDayNum"].transform("max")
    data["massey"] = massey[massey["RankingDayNum"] == latest_day].copy()
    logger.info(
        "Massey ordinals filtered: %d rows (latest day per season)",
        len(data["massey"]),
    )

    return data
```

- [ ] **Step 4: Create src/ingest/__init__.py** (empty file)

- [ ] **Step 5: Create tests/test_ingest/__init__.py** (empty file)

- [ ] **Step 6: Run tests to verify they pass**

Run: `pytest tests/test_ingest/test_kaggle_loader.py -v`
Expected: 4 tests PASS

- [ ] **Step 7: Commit**

```bash
git add src/ingest/__init__.py src/ingest/kaggle_loader.py tests/test_ingest/__init__.py tests/test_ingest/test_kaggle_loader.py
git commit -m "feat: Kaggle CSV loader with Massey ordinals filtering"
```

---

### Task 3: Team Name Mapping

**Files:**
- Create: `src/ingest/team_mapping.py`
- Create: `data/team_name_overrides.csv`
- Create: `tests/test_ingest/test_team_mapping.py`

- [ ] **Step 1: Write the failing tests**

```python
"""Tests for team name fuzzy matching and overrides."""

import pandas as pd
import pytest

from src.ingest.team_mapping import build_team_mapping, apply_overrides


@pytest.fixture
def kaggle_teams():
    return pd.DataFrame({
        "TeamID": [1101, 1102, 1103],
        "TeamName": ["Connecticut", "Saint Mary's (CA)", "Miami (FL)"],
    })


@pytest.fixture
def external_names():
    return ["UConn", "Saint Mary's", "Miami FL"]


@pytest.fixture
def overrides_path(tmp_path):
    overrides = pd.DataFrame({
        "external_name": ["UConn"],
        "kaggle_team_id": [1101],
    })
    path = tmp_path / "overrides.csv"
    overrides.to_csv(path, index=False)
    return path


def test_apply_overrides(kaggle_teams, overrides_path):
    result = apply_overrides(["UConn", "Unknown Team"], overrides_path)
    assert result["UConn"] == 1101
    assert "Unknown Team" not in result


def test_build_team_mapping_with_overrides(kaggle_teams, external_names, overrides_path):
    mapping = build_team_mapping(
        kaggle_teams=kaggle_teams,
        external_names=external_names,
        overrides_path=str(overrides_path),
        auto_threshold=85,
        review_threshold=70,
    )
    # UConn should map via override
    assert mapping["UConn"] == 1101


def test_build_team_mapping_fuzzy(kaggle_teams):
    mapping = build_team_mapping(
        kaggle_teams=kaggle_teams,
        external_names=["Saint Mary's"],
        overrides_path=None,
        auto_threshold=80,
        review_threshold=60,
    )
    # Should fuzzy match to Saint Mary's (CA)
    assert mapping["Saint Mary's"] == 1102


def test_build_team_mapping_no_match(kaggle_teams):
    mapping = build_team_mapping(
        kaggle_teams=kaggle_teams,
        external_names=["Totally Unknown University"],
        overrides_path=None,
        auto_threshold=85,
        review_threshold=70,
    )
    assert "Totally Unknown University" not in mapping
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_ingest/test_team_mapping.py -v`
Expected: FAIL

- [ ] **Step 3: Implement team_mapping.py**

```python
"""Team name fuzzy matching between data sources."""

import logging
from pathlib import Path

import pandas as pd
from rapidfuzz import fuzz, process

logger = logging.getLogger(__name__)


def apply_overrides(
    external_names: list[str], overrides_path: str | Path | None
) -> dict[str, int]:
    """Load manual overrides CSV and return {external_name: kaggle_team_id}."""
    if overrides_path is None:
        return {}
    path = Path(overrides_path)
    if not path.exists():
        logger.warning("Overrides file not found: %s", path)
        return {}
    df = pd.read_csv(path)
    return dict(zip(df["external_name"], df["kaggle_team_id"]))


def build_team_mapping(
    kaggle_teams: pd.DataFrame,
    external_names: list[str],
    overrides_path: str | None,
    auto_threshold: int = 85,
    review_threshold: int = 70,
) -> dict[str, int]:
    """Map external team names to Kaggle TeamIDs.

    Returns dict of {external_name: TeamID} for matched teams.
    Names below review_threshold are dropped. Names between
    review_threshold and auto_threshold are logged as warnings.
    """
    # Apply overrides first
    overrides = apply_overrides(external_names, overrides_path)
    mapping = dict(overrides)

    # Build choices dict: {kaggle_name: team_id}
    choices = dict(zip(kaggle_teams["TeamName"], kaggle_teams["TeamID"]))

    remaining = [n for n in external_names if n not in mapping]
    for name in remaining:
        result = process.extractOne(
            name, choices.keys(), scorer=fuzz.token_sort_ratio
        )
        if result is None:
            continue
        match_name, score, _ = result
        if score >= auto_threshold:
            mapping[name] = choices[match_name]
        elif score >= review_threshold:
            logger.warning(
                "Low-confidence match: '%s' -> '%s' (score=%d). Review manually.",
                name,
                match_name,
                score,
            )
            mapping[name] = choices[match_name]
        else:
            logger.info("No match for '%s' (best: '%s', score=%d)", name, match_name, score)

    return mapping
```

- [ ] **Step 4: Create data/team_name_overrides.csv**

```csv
external_name,kaggle_team_id
UConn,1163
St. Mary's,1348
Miami FL,1246
NC State,1314
```

Note: TeamIDs above are based on common Kaggle IDs. Verify against actual `MTeams.csv` after downloading data.

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_ingest/test_team_mapping.py -v`
Expected: 4 tests PASS

- [ ] **Step 6: Commit**

```bash
git add src/ingest/team_mapping.py data/team_name_overrides.csv tests/test_ingest/test_team_mapping.py
git commit -m "feat: team name fuzzy matching with overrides and confidence tiers"
```

---

### Task 4: Ingestion Validation

**Files:**
- Create: `src/ingest/validation.py`
- Create: `tests/test_ingest/test_validation.py`

- [ ] **Step 1: Write the failing tests**

```python
"""Tests for post-ingestion data validation."""

import pandas as pd
import pytest

from src.ingest.validation import validate_ingested_data, ValidationError


def test_validate_passes_good_data(sample_detailed_results, sample_teams, sample_seeds):
    data = {
        "teams": sample_teams,
        "regular_season": sample_detailed_results,
        "seeds": sample_seeds,
    }
    # Should not raise
    validate_ingested_data(data)


def test_validate_fails_null_team_ids(sample_detailed_results, sample_teams, sample_seeds):
    bad_results = sample_detailed_results.copy()
    bad_results.loc[0, "WTeamID"] = None
    data = {
        "teams": sample_teams,
        "regular_season": bad_results,
        "seeds": sample_seeds,
    }
    with pytest.raises(ValidationError, match="null TeamID"):
        validate_ingested_data(data)


def test_validate_fails_missing_columns(sample_teams, sample_seeds):
    bad_results = pd.DataFrame({"Season": [2023], "WTeamID": [1101]})
    data = {
        "teams": sample_teams,
        "regular_season": bad_results,
        "seeds": sample_seeds,
    }
    with pytest.raises(ValidationError, match="Missing columns"):
        validate_ingested_data(data)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_ingest/test_validation.py -v`
Expected: FAIL

- [ ] **Step 3: Implement validation.py**

```python
"""Post-ingestion data validation."""

import logging

import pandas as pd

logger = logging.getLogger(__name__)

_REQUIRED_RESULT_COLS = [
    "Season", "DayNum", "WTeamID", "WScore", "LTeamID", "LScore",
    "WLoc", "WFGM", "WFGA", "WFGM3", "WFGA3", "WFTM", "WFTA",
    "WOR", "WDR", "WAst", "WTO", "WStl", "WBlk", "WPF",
    "LFGM", "LFGA", "LFGM3", "LFGA3", "LFTM", "LFTA",
    "LOR", "LDR", "LAst", "LTO", "LStl", "LBlk", "LPF",
]


class ValidationError(Exception):
    pass


def validate_ingested_data(data: dict[str, pd.DataFrame]) -> None:
    """Validate ingested data for completeness and quality."""
    results = data["regular_season"]

    # Check required columns
    missing = set(_REQUIRED_RESULT_COLS) - set(results.columns)
    if missing:
        raise ValidationError(f"Missing columns in regular_season: {missing}")

    # Check for null TeamIDs
    for col in ["WTeamID", "LTeamID"]:
        if results[col].isna().any():
            raise ValidationError(f"Found null TeamID in column {col}")

    # Check game counts per season
    for season, group in results.groupby("Season"):
        n_games = len(group)
        if n_games < 100:
            logger.warning("Season %d has only %d games (expected 2000-6000)", season, n_games)

    logger.info("Ingestion validation passed: %d seasons, %d games", results["Season"].nunique(), len(results))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_ingest/test_validation.py -v`
Expected: 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/ingest/validation.py tests/test_ingest/test_validation.py
git commit -m "feat: post-ingestion data validation with column and null checks"
```

---

### Task 5: cbbd API Loader

**Files:**
- Create: `src/ingest/cbbd_loader.py`
- Create: `tests/test_ingest/test_cbbd_loader.py`

- [ ] **Step 1: Write the failing tests**

```python
"""Tests for cbbd API loader."""

import json
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

import pandas as pd
import pytest

from src.ingest.cbbd_loader import load_cbbd_data, _get_cache_path, _is_cache_valid


@pytest.fixture
def mock_cbbd_response():
    """Simulated cbbd API team ratings response."""
    return [
        {"team": "Duke", "year": 2026, "barthag": 0.95, "adj_o": 120.5, "adj_d": 90.2, "adj_t": 68.1,
         "efg_o": 0.55, "efg_d": 0.44, "tov_o": 0.15, "tov_d": 0.20, "orb_o": 0.33, "orb_d": 0.24, "ftr_o": 0.36, "ftr_d": 0.27},
        {"team": "UNC", "year": 2026, "barthag": 0.88, "adj_o": 115.0, "adj_d": 95.0, "adj_t": 70.0,
         "efg_o": 0.52, "efg_d": 0.47, "tov_o": 0.17, "tov_d": 0.18, "orb_o": 0.30, "orb_d": 0.27, "ftr_o": 0.33, "ftr_d": 0.30},
    ]


def test_cache_path(tmp_path):
    path = _get_cache_path(tmp_path, 2026)
    assert "cbbd_2026" in str(path)


def test_cache_validity(tmp_path):
    cache_file = tmp_path / "test_cache.json"
    cache_file.write_text("[]")
    assert _is_cache_valid(cache_file, ttl_hours=24)
    # File from far in the past would be invalid — tested via mocking


def test_load_cbbd_data_returns_dataframe(tmp_path, mock_cbbd_response):
    with patch("src.ingest.cbbd_loader._fetch_from_api", return_value=mock_cbbd_response):
        result = load_cbbd_data(season=2026, cache_dir=str(tmp_path))
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 2
    assert "team" in result.columns
    assert "adj_o" in result.columns


def test_load_cbbd_data_uses_cache(tmp_path, mock_cbbd_response):
    # First call: fetches from API
    with patch("src.ingest.cbbd_loader._fetch_from_api", return_value=mock_cbbd_response) as mock_fetch:
        load_cbbd_data(season=2026, cache_dir=str(tmp_path))
        assert mock_fetch.call_count == 1

    # Second call: should use cache
    with patch("src.ingest.cbbd_loader._fetch_from_api", return_value=mock_cbbd_response) as mock_fetch:
        load_cbbd_data(season=2026, cache_dir=str(tmp_path))
        assert mock_fetch.call_count == 0


def test_load_cbbd_data_api_failure_returns_none(tmp_path):
    with patch("src.ingest.cbbd_loader._fetch_from_api", side_effect=Exception("API down")):
        result = load_cbbd_data(season=2026, cache_dir=str(tmp_path))
    assert result is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_ingest/test_cbbd_loader.py -v`
Expected: FAIL

- [ ] **Step 3: Implement cbbd_loader.py**

```python
"""Load team ratings from CollegeBasketballData.com API (Barttorvik data)."""

import json
import logging
import time
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def _get_cache_path(cache_dir: str | Path, season: int) -> Path:
    return Path(cache_dir) / f"cbbd_{season}.json"


def _is_cache_valid(cache_path: Path, ttl_hours: int = 24) -> bool:
    if not cache_path.exists():
        return False
    age_hours = (time.time() - cache_path.stat().st_mtime) / 3600
    return age_hours < ttl_hours


def _fetch_from_api(season: int) -> list[dict]:
    """Fetch team ratings from cbbd API."""
    try:
        import cbbd
        config = cbbd.Configuration()
        api = cbbd.RatingsApi(cbbd.ApiClient(config))
        response = api.get_ratings(season=season)
        return [r.to_dict() for r in response]
    except ImportError:
        # Fallback: direct HTTP request to the API
        import requests
        resp = requests.get(
            f"https://api.collegebasketballdata.com/ratings",
            params={"season": season},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()


def load_cbbd_data(season: int, cache_dir: str, ttl_hours: int = 24) -> pd.DataFrame | None:
    """Load Barttorvik-derived ratings from cbbd API with caching.

    Returns DataFrame with team ratings, or None if API is unavailable.
    """
    cache_path = _get_cache_path(cache_dir, season)
    Path(cache_dir).mkdir(parents=True, exist_ok=True)

    # Check cache
    if _is_cache_valid(cache_path, ttl_hours):
        logger.info("Using cached cbbd data: %s", cache_path)
        with open(cache_path) as f:
            data = json.load(f)
        return pd.DataFrame(data)

    # Fetch from API
    try:
        logger.info("Fetching cbbd data for season %d", season)
        data = _fetch_from_api(season)
        # Cache response
        with open(cache_path, "w") as f:
            json.dump(data, f)
        logger.info("Cached cbbd response: %d teams", len(data))
        return pd.DataFrame(data)
    except Exception as e:
        logger.warning("cbbd API unavailable: %s. Proceeding without.", e)
        return None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_ingest/test_cbbd_loader.py -v`
Expected: 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/ingest/cbbd_loader.py tests/test_ingest/test_cbbd_loader.py
git commit -m "feat: cbbd API loader with 24h cache and graceful fallback"
```

---

### Task 6: Massey Composite Loader

**Files:**
- Create: `src/ingest/massey_loader.py`
- Create: `tests/test_ingest/test_massey_loader.py`

- [ ] **Step 1: Write the failing tests**

```python
"""Tests for Massey Ratings composite loader."""

from unittest.mock import patch

import pandas as pd
import pytest

from src.ingest.massey_loader import load_massey_composite, parse_massey_csv


@pytest.fixture
def sample_massey_csv():
    return "Team,1,2,3,Comp\nDuke,1,2,1,1\nUNC,3,4,5,4\n"


def test_parse_massey_csv(sample_massey_csv):
    result = parse_massey_csv(sample_massey_csv)
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 2
    assert "Team" in result.columns
    assert "Comp" in result.columns


def test_load_massey_composite_returns_dataframe(tmp_path, sample_massey_csv):
    with patch("src.ingest.massey_loader._download_massey_csv", return_value=sample_massey_csv):
        result = load_massey_composite(cache_dir=str(tmp_path))
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 2


def test_load_massey_composite_failure_returns_none(tmp_path):
    with patch("src.ingest.massey_loader._download_massey_csv", side_effect=Exception("Network error")):
        result = load_massey_composite(cache_dir=str(tmp_path))
    assert result is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_ingest/test_massey_loader.py -v`
Expected: FAIL

- [ ] **Step 3: Implement massey_loader.py**

```python
"""Load Massey Ratings composite rankings."""

import logging
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

logger = logging.getLogger(__name__)

_MASSEY_URL = "https://masseyratings.com/cb/compare.csv"


def _download_massey_csv() -> str:
    """Download composite rankings CSV from Massey Ratings."""
    resp = requests.get(_MASSEY_URL, timeout=30)
    resp.raise_for_status()
    return resp.text


def parse_massey_csv(csv_text: str) -> pd.DataFrame:
    """Parse the Massey composite CSV text into a DataFrame."""
    return pd.read_csv(StringIO(csv_text))


def load_massey_composite(cache_dir: str | None = None) -> pd.DataFrame | None:
    """Load current Massey composite rankings.

    Returns DataFrame with team names and composite rank, or None on failure.
    """
    try:
        csv_text = _download_massey_csv()
        df = parse_massey_csv(csv_text)

        if cache_dir:
            cache_path = Path(cache_dir) / "massey_composite.csv"
            Path(cache_dir).mkdir(parents=True, exist_ok=True)
            with open(cache_path, "w") as f:
                f.write(csv_text)
            logger.info("Cached Massey composite: %s", cache_path)

        logger.info("Loaded Massey composite: %d teams", len(df))
        return df
    except Exception as e:
        logger.warning("Massey Ratings unavailable: %s. Proceeding without.", e)
        return None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_ingest/test_massey_loader.py -v`
Expected: 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/ingest/massey_loader.py tests/test_ingest/test_massey_loader.py
git commit -m "feat: Massey composite rankings loader with graceful fallback"
```

---

### Task 7: Ingestion CLI Entry Point

**Files:**
- Create: `src/ingest/__main__.py`

- [ ] **Step 1: Implement __main__.py**

```python
"""CLI entry point: python -m src.ingest"""

import logging
from pathlib import Path

from src.config import load_config
from src.ingest.kaggle_loader import load_kaggle_data
from src.ingest.cbbd_loader import load_cbbd_data
from src.ingest.massey_loader import load_massey_composite
from src.ingest.team_mapping import build_team_mapping
from src.ingest.validation import validate_ingested_data

logging.basicConfig(level=logging.INFO, format="%(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    config = load_config()
    kaggle_dir = config["data"]["kaggle_dir"]
    processed_dir = Path(config["data"]["processed_dir"])
    cache_dir = config["data"]["cache_dir"]
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Source 1: Kaggle (required)
    logger.info("Loading Kaggle data from %s", kaggle_dir)
    data = load_kaggle_data(kaggle_dir)

    logger.info("Validating ingested data")
    validate_ingested_data(data)

    # Source 2: cbbd API (optional)
    predict_season = config["seasons"]["predict_season"]
    cbbd_data = load_cbbd_data(season=predict_season, cache_dir=cache_dir)
    if cbbd_data is not None:
        data["cbbd"] = cbbd_data
        # Build team mapping for cbbd names -> Kaggle IDs
        cbbd_mapping = build_team_mapping(
            kaggle_teams=data["teams"],
            external_names=cbbd_data["team"].tolist(),
            overrides_path=config["data"]["team_overrides"],
            auto_threshold=config["matching"]["auto_accept_threshold"],
            review_threshold=config["matching"]["review_threshold"],
        )
        cbbd_data["TeamID"] = cbbd_data["team"].map(cbbd_mapping)
        data["cbbd"] = cbbd_data

    # Source 3: Massey composite (optional)
    massey_composite = load_massey_composite(cache_dir=cache_dir)
    if massey_composite is not None:
        data["massey_composite"] = massey_composite

    # Save processed data as parquet
    for name, df in data.items():
        output_path = processed_dir / f"{name}.parquet"
        df.to_parquet(output_path, index=False)
        logger.info("Saved %s -> %s (%d rows)", name, output_path, len(df))

    logger.info("Ingestion complete")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
git add src/ingest/__main__.py
git commit -m "feat: ingestion CLI wiring Kaggle, cbbd, Massey, and team mapping"
```

---

## Chunk 2: Feature Engineering

### Task 8: Four Factors Computation

**Files:**
- Create: `src/features/__init__.py`
- Create: `src/features/four_factors.py`
- Create: `tests/test_features/__init__.py`
- Create: `tests/test_features/test_four_factors.py`

- [ ] **Step 1: Write the failing tests**

```python
"""Tests for four factors computation."""

import pandas as pd
import pytest

from src.features.four_factors import compute_four_factors, estimate_possessions


def test_estimate_possessions():
    # FGA=60, OR=10, TO=12, FTA=15
    # possessions = 60 - 10 + 12 + 0.475 * 15 = 69.125
    result = estimate_possessions(fga=60, offensive_rebounds=10, turnovers=12, fta=15)
    assert abs(result - 69.125) < 0.01


def test_compute_four_factors(sample_detailed_results):
    result = compute_four_factors(sample_detailed_results, season=2023)
    assert isinstance(result, pd.DataFrame)
    assert "TeamID" in result.columns
    assert "off_efg" in result.columns
    assert "def_efg" in result.columns
    assert "off_to_rate" in result.columns
    assert "off_or_rate" in result.columns
    assert "off_ft_rate" in result.columns
    # Should have one row per team
    assert len(result) == 4


def test_four_factors_efg_range(sample_detailed_results):
    result = compute_four_factors(sample_detailed_results, season=2023)
    # eFG% should be between 0 and 1
    assert (result["off_efg"] >= 0).all()
    assert (result["off_efg"] <= 1).all()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_features/test_four_factors.py -v`
Expected: FAIL

- [ ] **Step 3: Implement four_factors.py**

```python
"""Compute four factors from detailed box score data."""

import pandas as pd


def estimate_possessions(fga: float, offensive_rebounds: float, turnovers: float, fta: float) -> float:
    """Estimate possessions using the standard formula."""
    return fga - offensive_rebounds + turnovers + 0.475 * fta


def compute_four_factors(detailed_results: pd.DataFrame, season: int) -> pd.DataFrame:
    """Compute season-level four factors for each team.

    Returns DataFrame with columns:
        TeamID, off_efg, off_to_rate, off_or_rate, off_ft_rate,
        def_efg, def_to_rate, def_or_rate, def_ft_rate
    """
    df = detailed_results[detailed_results["Season"] == season].copy()

    # Build per-game stats from winner perspective
    winner_off = pd.DataFrame({
        "TeamID": df["WTeamID"],
        "FGM": df["WFGM"], "FGA": df["WFGA"],
        "FGM3": df["WFGM3"], "FTM": df["WFTM"],
        "FTA": df["WFTA"], "OR": df["WOR"],
        "TO": df["WTO"],
        "opp_DR": df["LDR"],  # opponent defensive rebounds
        # Opponent stats for defense
        "opp_FGM": df["LFGM"], "opp_FGA": df["LFGA"],
        "opp_FGM3": df["LFGM3"], "opp_FTM": df["LFTM"],
        "opp_FTA": df["LFTA"], "opp_OR": df["LOR"],
        "opp_TO": df["LTO"],
        "DR": df["WDR"],  # own defensive rebounds
    })

    # Build per-game stats from loser perspective
    loser_off = pd.DataFrame({
        "TeamID": df["LTeamID"],
        "FGM": df["LFGM"], "FGA": df["LFGA"],
        "FGM3": df["LFGM3"], "FTM": df["LFTM"],
        "FTA": df["LFTA"], "OR": df["LOR"],
        "TO": df["LTO"],
        "opp_DR": df["WDR"],
        "opp_FGM": df["WFGM"], "opp_FGA": df["WFGA"],
        "opp_FGM3": df["WFGM3"], "opp_FTM": df["WFTM"],
        "opp_FTA": df["WFTA"], "opp_OR": df["WOR"],
        "opp_TO": df["WTO"],
        "DR": df["LDR"],
    })

    all_games = pd.concat([winner_off, loser_off], ignore_index=True)

    # Aggregate per team
    agg = all_games.groupby("TeamID").sum()

    # Offensive four factors
    agg["off_efg"] = (agg["FGM"] + 0.5 * agg["FGM3"]) / agg["FGA"]
    agg["off_to_rate"] = agg["TO"] / (agg["FGA"] + 0.475 * agg["FTA"] + agg["TO"])
    agg["off_or_rate"] = agg["OR"] / (agg["OR"] + agg["opp_DR"])
    agg["off_ft_rate"] = agg["FTM"] / agg["FGA"]

    # Defensive four factors (opponent's offensive stats)
    agg["def_efg"] = (agg["opp_FGM"] + 0.5 * agg["opp_FGM3"]) / agg["opp_FGA"]
    agg["def_to_rate"] = agg["opp_TO"] / (agg["opp_FGA"] + 0.475 * agg["opp_FTA"] + agg["opp_TO"])
    agg["def_or_rate"] = agg["opp_OR"] / (agg["opp_OR"] + agg["DR"])
    agg["def_ft_rate"] = agg["opp_FTM"] / agg["opp_FGA"]

    result_cols = [
        "off_efg", "off_to_rate", "off_or_rate", "off_ft_rate",
        "def_efg", "def_to_rate", "def_or_rate", "def_ft_rate",
    ]
    return agg[result_cols].reset_index()
```

- [ ] **Step 4: Create empty __init__.py files**

Create: `src/features/__init__.py`, `tests/test_features/__init__.py`

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_features/test_four_factors.py -v`
Expected: 3 tests PASS

- [ ] **Step 6: Commit**

```bash
git add src/features/__init__.py src/features/four_factors.py tests/test_features/__init__.py tests/test_features/test_four_factors.py
git commit -m "feat: four factors computation from detailed box scores"
```

---

### Task 9: Custom Adjusted Efficiency Ratings

**Files:**
- Create: `src/features/efficiency.py`
- Create: `tests/test_features/test_efficiency.py`

- [ ] **Step 1: Write the failing tests**

```python
"""Tests for adjusted efficiency rating computation."""

import numpy as np
import pandas as pd
import pytest

from src.features.efficiency import compute_adjusted_efficiency


@pytest.fixture
def game_data():
    """6-team round-robin with known strength ordering."""
    games = []
    teams = [1, 2, 3, 4, 5, 6]
    # Stronger teams score more, weaker teams score less
    # Team 1 is best, team 6 is worst
    np.random.seed(42)
    for i, t1 in enumerate(teams):
        for j, t2 in enumerate(teams):
            if t1 >= t2:
                continue
            # Higher-ranked (lower index) team wins more
            t1_score = 75 - i * 3 + np.random.randint(-3, 4)
            t2_score = 75 - j * 3 + np.random.randint(-3, 4)
            if t1_score == t2_score:
                t1_score += 1
            if t1_score > t2_score:
                games.append({"WTeamID": t1, "LTeamID": t2, "WScore": t1_score, "LScore": t2_score,
                              "WFGA": 60, "WOR": 10, "WTO": 12, "WFTA": 15, "WLoc": "N",
                              "LFGA": 60, "LOR": 10, "LTO": 12, "LFTA": 15,
                              "Season": 2023, "DayNum": 50})
            else:
                games.append({"WTeamID": t2, "LTeamID": t1, "WScore": t2_score, "LScore": t1_score,
                              "WFGA": 60, "WOR": 10, "WTO": 12, "WFTA": 15, "WLoc": "N",
                              "LFGA": 60, "LOR": 10, "LTO": 12, "LFTA": 15,
                              "Season": 2023, "DayNum": 50})
    return pd.DataFrame(games)


def test_compute_adjusted_efficiency_returns_all_teams(game_data):
    result = compute_adjusted_efficiency(game_data, season=2023, iterations=15, hca=3.5, half_life_days=30, ridge_alpha=1.0)
    assert isinstance(result, pd.DataFrame)
    assert "TeamID" in result.columns
    assert "adj_oe" in result.columns
    assert "adj_de" in result.columns
    assert "adj_em" in result.columns
    assert "adj_tempo" in result.columns
    assert len(result) == 6


def test_adj_em_is_oe_minus_de(game_data):
    result = compute_adjusted_efficiency(game_data, season=2023, iterations=15, hca=3.5, half_life_days=30, ridge_alpha=1.0)
    diff = abs(result["adj_em"] - (result["adj_oe"] - result["adj_de"]))
    assert (diff < 0.01).all()


def test_efficiency_ratings_reasonable_range(game_data):
    result = compute_adjusted_efficiency(game_data, season=2023, iterations=15, hca=3.5, half_life_days=30, ridge_alpha=1.0)
    # AdjEM should be in a reasonable range for college basketball
    assert (result["adj_em"] > -50).all()
    assert (result["adj_em"] < 50).all()


def test_strongest_team_ranked_first(game_data):
    result = compute_adjusted_efficiency(game_data, season=2023, iterations=15, hca=3.5, half_life_days=30, ridge_alpha=1.0)
    # Team 1 is best (highest scores), team 6 is worst
    # Best team should have highest AdjEM
    best_team = result.iloc[0]["TeamID"]  # sorted by adj_em desc
    assert best_team in [1, 2]  # allow small tolerance for randomness in fixture


def test_worst_team_ranked_last(game_data):
    result = compute_adjusted_efficiency(game_data, season=2023, iterations=15, hca=3.5, half_life_days=30, ridge_alpha=1.0)
    worst_team = result.iloc[-1]["TeamID"]
    assert worst_team in [5, 6]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_features/test_efficiency.py -v`
Expected: FAIL

- [ ] **Step 3: Implement efficiency.py**

```python
"""Iterative adjusted efficiency ratings."""

import logging

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from src.features.four_factors import estimate_possessions

logger = logging.getLogger(__name__)


def compute_adjusted_efficiency(
    detailed_results: pd.DataFrame,
    season: int,
    iterations: int = 15,
    hca: float = 3.5,
    half_life_days: float = 30.0,
    ridge_alpha: float = 1.0,
) -> pd.DataFrame:
    """Compute adjusted offensive/defensive efficiency for all teams in a season.

    Uses iterative ridge regression on per-possession point differential,
    adjusting for opponent strength and home court advantage.

    Returns DataFrame with: TeamID, adj_oe, adj_de, adj_em, adj_tempo
    """
    df = detailed_results[detailed_results["Season"] == season].copy()

    # Estimate possessions per game
    df["w_poss"] = df.apply(
        lambda r: estimate_possessions(r["WFGA"], r["WOR"], r["WTO"], r["WFTA"]), axis=1
    )
    df["l_poss"] = df.apply(
        lambda r: estimate_possessions(r["LFGA"], r["LOR"], r["LTO"], r["LFTA"]), axis=1
    )
    df["possessions"] = (df["w_poss"] + df["l_poss"]) / 2

    # Build game-level records (two rows per game: one per team)
    winner_rows = pd.DataFrame({
        "TeamID": df["WTeamID"],
        "OppID": df["LTeamID"],
        "points_scored": df["WScore"],
        "points_allowed": df["LScore"],
        "possessions": df["possessions"],
        "home": (df["WLoc"] == "H").astype(int),
        "day_num": df["DayNum"],
    })
    loser_rows = pd.DataFrame({
        "TeamID": df["LTeamID"],
        "OppID": df["WTeamID"],
        "points_scored": df["LScore"],
        "points_allowed": df["WScore"],
        "possessions": df["possessions"],
        "home": (df["WLoc"] == "A").astype(int),
        "day_num": df["DayNum"],
    })
    games = pd.concat([winner_rows, loser_rows], ignore_index=True)

    # Offensive/defensive efficiency per game (points per 100 possessions)
    games["oe"] = 100 * games["points_scored"] / games["possessions"]
    games["de"] = 100 * games["points_allowed"] / games["possessions"]
    games["tempo"] = games["possessions"]

    # Recency weights: exponential decay from last game day
    max_day = games["day_num"].max()
    decay_rate = np.log(2) / half_life_days
    games["weight"] = np.exp(-decay_rate * (max_day - games["day_num"]))

    # Initialize ratings as raw averages
    teams = games["TeamID"].unique()
    team_oe = games.groupby("TeamID")["oe"].mean().to_dict()
    team_de = games.groupby("TeamID")["de"].mean().to_dict()
    league_avg_oe = games["oe"].mean()
    league_avg_de = games["de"].mean()

    # Iterative adjustment
    for iteration in range(iterations):
        # For each game, compute expected OE/DE based on opponent strength
        games["opp_de_rating"] = games["OppID"].map(team_de).fillna(league_avg_de)
        games["opp_oe_rating"] = games["OppID"].map(team_oe).fillna(league_avg_oe)

        # Adjusted OE = raw OE * (league_avg_de / opp_de_rating)
        # Plus home court advantage adjustment
        # HCA produces neutral-court equivalents: home teams have inflated raw OE
        # (subtract to deflate) and deflated raw DE (add to inflate). Away teams
        # get the opposite adjustment via their own rows in the symmetric records.
        home_adj = games["home"] * hca / 2  # split HCA between off and def

        games["adj_oe_game"] = games["oe"] * (league_avg_de / games["opp_de_rating"]) - home_adj
        games["adj_de_game"] = games["de"] * (league_avg_oe / games["opp_oe_rating"]) + home_adj

        # Weighted average per team
        prev_oe = dict(team_oe)
        for team in teams:
            mask = games["TeamID"] == team
            w = games.loc[mask, "weight"]
            team_oe[team] = np.average(games.loc[mask, "adj_oe_game"], weights=w)
            team_de[team] = np.average(games.loc[mask, "adj_de_game"], weights=w)

        # Convergence diagnostic
        max_change = max(abs(team_oe[t] - prev_oe[t]) for t in teams)
        logger.debug("Iteration %d: max rating change = %.4f", iteration + 1, max_change)

    # Compute adjusted tempo
    team_tempo = {}
    league_avg_tempo = games["tempo"].mean()
    for team in teams:
        mask = games["TeamID"] == team
        opp_tempos = games.loc[mask, "OppID"].map(
            games.groupby("TeamID")["tempo"].mean().to_dict()
        ).fillna(league_avg_tempo)
        raw_tempos = games.loc[mask, "tempo"]
        team_tempo[team] = np.average(
            raw_tempos * (league_avg_tempo / opp_tempos),
            weights=games.loc[mask, "weight"],
        )

    # Assemble results
    result = pd.DataFrame({
        "TeamID": list(teams),
        "adj_oe": [team_oe[t] for t in teams],
        "adj_de": [team_de[t] for t in teams],
        "adj_em": [team_oe[t] - team_de[t] for t in teams],
        "adj_tempo": [team_tempo[t] for t in teams],
    })

    return result.sort_values("adj_em", ascending=False).reset_index(drop=True)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_features/test_efficiency.py -v`
Expected: 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/features/efficiency.py tests/test_features/test_efficiency.py
git commit -m "feat: iterative adjusted efficiency ratings with recency weighting"
```

---

### Task 10: Feature Matrix Assembly

**Files:**
- Create: `src/features/feature_matrix.py`
- Create: `tests/test_features/test_feature_matrix.py`

- [ ] **Step 1: Write the failing tests**

```python
"""Tests for feature matrix assembly."""

import pandas as pd
import pytest

from src.features.feature_matrix import build_feature_matrix


@pytest.fixture
def efficiency_ratings():
    return pd.DataFrame({
        "TeamID": [1101, 1102, 1103, 1104],
        "adj_oe": [115.0, 110.0, 105.0, 100.0],
        "adj_de": [95.0, 98.0, 102.0, 108.0],
        "adj_em": [20.0, 12.0, 3.0, -8.0],
        "adj_tempo": [68.0, 65.0, 70.0, 72.0],
    })


@pytest.fixture
def four_factors():
    return pd.DataFrame({
        "TeamID": [1101, 1102, 1103, 1104],
        "off_efg": [0.55, 0.52, 0.48, 0.45],
        "off_to_rate": [0.15, 0.17, 0.19, 0.21],
        "off_or_rate": [0.32, 0.30, 0.28, 0.25],
        "off_ft_rate": [0.35, 0.33, 0.30, 0.28],
        "def_efg": [0.45, 0.48, 0.50, 0.53],
        "def_to_rate": [0.20, 0.18, 0.16, 0.14],
        "def_or_rate": [0.25, 0.28, 0.30, 0.33],
        "def_ft_rate": [0.28, 0.30, 0.33, 0.35],
    })


@pytest.fixture
def massey_ranks():
    return pd.DataFrame({
        "TeamID": [1101, 1102, 1103, 1104,
                    1101, 1102, 1103, 1104],
        "SystemName": ["POM", "POM", "POM", "POM",
                        "SAG", "SAG", "SAG", "SAG"],
        "OrdinalRank": [1, 5, 30, 80,
                        2, 6, 28, 75],
    })


def test_build_feature_matrix_shape(efficiency_ratings, four_factors, sample_seeds, massey_ranks, sample_detailed_results):
    result = build_feature_matrix(
        efficiency=efficiency_ratings,
        four_factors=four_factors,
        seeds=sample_seeds,
        massey=massey_ranks,
        results=sample_detailed_results,
        season=2023,
        massey_systems=["POM", "SAG"],
    )
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 4
    assert "TeamID" in result.columns
    assert "seed" in result.columns
    assert "adj_em" in result.columns
    assert "massey_POM" in result.columns


def test_build_feature_matrix_no_nulls(efficiency_ratings, four_factors, sample_seeds, massey_ranks, sample_detailed_results):
    result = build_feature_matrix(
        efficiency=efficiency_ratings,
        four_factors=four_factors,
        seeds=sample_seeds,
        massey=massey_ranks,
        results=sample_detailed_results,
        season=2023,
        massey_systems=["POM", "SAG"],
    )
    # Tournament teams should have no nulls in critical features
    assert not result[["adj_em", "seed"]].isna().any().any()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_features/test_feature_matrix.py -v`
Expected: FAIL

- [ ] **Step 3: Implement feature_matrix.py**

```python
"""Assemble the full feature matrix for each team-season."""

import logging
import re

import pandas as pd

logger = logging.getLogger(__name__)


def _parse_seed_number(seed_str: str) -> int:
    """Extract numeric seed from strings like 'W01', 'X16a'."""
    match = re.search(r"(\d+)", seed_str)
    return int(match.group(1)) if match else 16


def _compute_recent_form(results: pd.DataFrame, season: int, last_n: int = 10) -> pd.DataFrame:
    """Compute win% over the last N games of the season."""
    df = results[results["Season"] == season].copy()
    max_day = df["DayNum"].max()

    # Get all games sorted by day
    winner_games = df[["WTeamID", "DayNum"]].rename(columns={"WTeamID": "TeamID"}).assign(win=1)
    loser_games = df[["LTeamID", "DayNum"]].rename(columns={"LTeamID": "TeamID"}).assign(win=0)
    all_games = pd.concat([winner_games, loser_games]).sort_values("DayNum")

    records = []
    for team_id, group in all_games.groupby("TeamID"):
        tail = group.tail(last_n)
        records.append({"TeamID": team_id, "win_pct_last_10": tail["win"].mean()})
    return pd.DataFrame(records)


def _compute_road_win_pct(results: pd.DataFrame, season: int) -> pd.DataFrame:
    """Compute road + neutral win percentage."""
    df = results[results["Season"] == season].copy()

    # Away wins: winner was away
    away_wins = df[df["WLoc"] == "A"][["WTeamID"]].rename(columns={"WTeamID": "TeamID"}).assign(win=1)
    # Neutral wins
    neutral_wins = df[df["WLoc"] == "N"][["WTeamID"]].rename(columns={"WTeamID": "TeamID"}).assign(win=1)
    # Away losses: loser was home (winner was away), so loser is home team
    away_losses = df[df["WLoc"] == "H"][["LTeamID"]].rename(columns={"LTeamID": "TeamID"}).assign(win=0)
    # Neutral losses
    neutral_losses = df[df["WLoc"] == "N"][["LTeamID"]].rename(columns={"LTeamID": "TeamID"}).assign(win=0)

    # Road games = away + neutral
    road = pd.concat([away_wins, neutral_wins, away_losses, neutral_losses])
    result = road.groupby("TeamID")["win"].mean().reset_index()
    result.columns = ["TeamID", "road_win_pct"]
    return result


def _compute_conf_strength(results: pd.DataFrame, team_conferences: pd.DataFrame, efficiency: pd.DataFrame, season: int) -> pd.DataFrame:
    """Compute average AdjEM of conference opponents."""
    conf = team_conferences[team_conferences["Season"] == season][["TeamID", "ConfAbbrev"]]
    merged = conf.merge(efficiency[["TeamID", "adj_em"]], on="TeamID", how="left")

    conf_avg = merged.groupby("ConfAbbrev")["adj_em"].mean().reset_index()
    conf_avg.columns = ["ConfAbbrev", "conf_strength"]

    return conf.merge(conf_avg, on="ConfAbbrev")[["TeamID", "conf_strength"]]


def build_feature_matrix(
    efficiency: pd.DataFrame,
    four_factors: pd.DataFrame,
    seeds: pd.DataFrame,
    massey: pd.DataFrame,
    results: pd.DataFrame,
    season: int,
    massey_systems: list[str],
    team_conferences: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build the full feature matrix for a given season.

    Merges efficiency ratings, four factors, seeds, Massey rankings,
    and derived features into a single DataFrame with one row per team.
    """
    # Start with efficiency ratings
    matrix = efficiency.copy()

    # Merge four factors
    matrix = matrix.merge(four_factors, on="TeamID", how="left")

    # Parse seeds
    season_seeds = seeds[seeds["Season"] == season].copy()
    season_seeds["seed"] = season_seeds["Seed"].apply(_parse_seed_number)
    matrix = matrix.merge(season_seeds[["TeamID", "seed"]], on="TeamID", how="left")

    # Massey rankings: pivot each system into its own column
    # Filter by system name; caller is responsible for pre-filtering by season
    season_massey = massey[massey["SystemName"].isin(massey_systems)]
    for system in massey_systems:
        sys_ranks = season_massey[season_massey["SystemName"] == system][["TeamID", "OrdinalRank"]]
        sys_ranks = sys_ranks.rename(columns={"OrdinalRank": f"massey_{system}"})
        matrix = matrix.merge(sys_ranks, on="TeamID", how="left")

    # Composite Massey rank (average across systems)
    massey_cols = [f"massey_{s}" for s in massey_systems]
    existing_massey_cols = [c for c in massey_cols if c in matrix.columns]
    if existing_massey_cols:
        matrix["massey_composite_rank"] = matrix[existing_massey_cols].mean(axis=1)

    # Recent form
    recent = _compute_recent_form(results, season)
    matrix = matrix.merge(recent, on="TeamID", how="left")

    # Road win percentage
    road = _compute_road_win_pct(results, season)
    matrix = matrix.merge(road, on="TeamID", how="left")

    # Conference strength (if conference data available)
    if team_conferences is not None:
        conf = _compute_conf_strength(results, team_conferences, efficiency, season)
        matrix = matrix.merge(conf, on="TeamID", how="left")

    # Validate: warn about tournament teams with missing features
    tourney_teams = season_seeds["TeamID"].unique()
    tourney_matrix = matrix[matrix["TeamID"].isin(tourney_teams)]
    null_counts = tourney_matrix.isna().sum()
    if null_counts.any():
        for col in null_counts[null_counts > 0].index:
            logger.warning("Feature '%s' has %d nulls among tournament teams", col, null_counts[col])

    return matrix
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_features/test_feature_matrix.py -v`
Expected: 2 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/features/feature_matrix.py tests/test_features/test_feature_matrix.py
git commit -m "feat: feature matrix assembly with seeds, Massey, form, and road record"
```

---

### Task 11: Features CLI Entry Point

**Files:**
- Create: `src/features/__main__.py`

- [ ] **Step 1: Implement __main__.py**

```python
"""CLI entry point: python -m src.features"""

import logging
from pathlib import Path

import pandas as pd

from src.config import load_config
from src.features.efficiency import compute_adjusted_efficiency
from src.features.four_factors import compute_four_factors
from src.features.feature_matrix import build_feature_matrix

logging.basicConfig(level=logging.INFO, format="%(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    config = load_config()
    processed_dir = Path(config["data"]["processed_dir"])

    # Load ingested data
    results = pd.read_parquet(processed_dir / "regular_season.parquet")
    seeds = pd.read_parquet(processed_dir / "seeds.parquet")
    massey = pd.read_parquet(processed_dir / "massey.parquet")
    team_conferences = pd.read_parquet(processed_dir / "team_conferences.parquet")

    start = config["seasons"]["train_start"]
    end = config["seasons"]["predict_season"]
    eff_cfg = config["efficiency"]
    massey_systems = config["massey"]["systems"]

    all_features = []
    for season in range(start, end + 1):
        season_results = results[results["Season"] == season]
        if len(season_results) == 0:
            logger.warning("No data for season %d, skipping", season)
            continue

        logger.info("Computing features for season %d", season)

        efficiency = compute_adjusted_efficiency(
            season_results, season=season,
            iterations=eff_cfg["iterations"],
            hca=eff_cfg["home_court_advantage"],
            half_life_days=eff_cfg["recency_half_life_days"],
            ridge_alpha=eff_cfg["ridge_alpha"],
        )

        ff = compute_four_factors(season_results, season=season)

        season_massey = massey[massey["Season"] == season] if "Season" in massey.columns else massey

        matrix = build_feature_matrix(
            efficiency=efficiency,
            four_factors=ff,
            seeds=seeds,
            massey=season_massey,
            results=results,
            season=season,
            massey_systems=massey_systems,
            team_conferences=team_conferences,
        )
        matrix["Season"] = season
        all_features.append(matrix)

    combined = pd.concat(all_features, ignore_index=True)
    output_path = processed_dir / "feature_matrix.parquet"
    combined.to_parquet(output_path, index=False)
    logger.info("Feature matrix saved: %s (%d rows, %d seasons)", output_path, len(combined), combined["Season"].nunique())


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
git add src/features/__main__.py
git commit -m "feat: features CLI entry point computing all seasons"
```

---

## Chunk 3: Model Training & Evaluation

### Task 12: Matchup Training Data Builder

**Files:**
- Create: `src/models/__init__.py`
- Create: `src/models/matchup.py`
- Create: `tests/test_models/__init__.py`
- Create: `tests/test_models/test_matchup.py`

- [ ] **Step 1: Write the failing tests**

```python
"""Tests for matchup training data construction."""

import pandas as pd
import pytest

from src.models.matchup import build_matchup_data

FEATURE_COLS = ["adj_em", "adj_oe", "adj_de", "seed"]


@pytest.fixture
def feature_matrix():
    return pd.DataFrame({
        "TeamID": [1, 2, 3, 4],
        "Season": [2023, 2023, 2023, 2023],
        "adj_em": [20.0, 12.0, 3.0, -8.0],
        "adj_oe": [115.0, 110.0, 105.0, 100.0],
        "adj_de": [95.0, 98.0, 102.0, 108.0],
        "seed": [1, 2, 3, 4],
    })


@pytest.fixture
def tourney_results():
    return pd.DataFrame({
        "Season": [2023, 2023],
        "WTeamID": [1, 2],
        "LTeamID": [4, 3],
    })


def test_build_matchup_data_symmetric(feature_matrix, tourney_results):
    X, y = build_matchup_data(feature_matrix, tourney_results, FEATURE_COLS)
    # 2 games * 2 (symmetric) = 4 rows
    assert len(X) == 4
    assert len(y) == 4


def test_build_matchup_data_labels(feature_matrix, tourney_results):
    X, y = build_matchup_data(feature_matrix, tourney_results, FEATURE_COLS)
    # Half should be wins (1), half losses (0)
    assert y.sum() == 2
    assert (y == 0).sum() == 2


def test_build_matchup_data_feature_differences(feature_matrix, tourney_results):
    X, y = build_matchup_data(feature_matrix, tourney_results, FEATURE_COLS)
    # First row: team 1 vs team 4 (winner perspective)
    # adj_em diff should be 20 - (-8) = 28
    first_win_row = X[y == 1].iloc[0]
    assert abs(first_win_row["adj_em"]) > 0  # non-zero difference
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_models/test_matchup.py -v`
Expected: FAIL

- [ ] **Step 3: Implement matchup.py**

```python
"""Build symmetric matchup training data from tournament results."""

import pandas as pd
import numpy as np


def build_matchup_data(
    feature_matrix: pd.DataFrame,
    tourney_results: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[pd.DataFrame, pd.Series]:
    """Build training data for matchup prediction.

    Each tournament game produces two rows (A vs B and B vs A).
    Features are the difference: team_A_features - team_B_features.
    Target is 1 if team A won, 0 otherwise.

    Returns (X, y) where X is the feature difference DataFrame
    and y is the binary target Series.
    """
    rows = []
    labels = []

    for _, game in tourney_results.iterrows():
        season = game["Season"]
        w_id = game["WTeamID"]
        l_id = game["LTeamID"]

        w_features = feature_matrix[
            (feature_matrix["TeamID"] == w_id) & (feature_matrix["Season"] == season)
        ][feature_cols]
        l_features = feature_matrix[
            (feature_matrix["TeamID"] == l_id) & (feature_matrix["Season"] == season)
        ][feature_cols]

        if w_features.empty or l_features.empty:
            continue

        w_vals = w_features.iloc[0].values
        l_vals = l_features.iloc[0].values

        # Winner perspective: W - L, label = 1
        rows.append(w_vals - l_vals)
        labels.append(1)

        # Loser perspective: L - W, label = 0
        rows.append(l_vals - w_vals)
        labels.append(0)

    X = pd.DataFrame(rows, columns=feature_cols)
    y = pd.Series(labels, name="win")
    return X, y
```

- [ ] **Step 4: Create empty __init__.py files**

Create: `src/models/__init__.py`, `tests/test_models/__init__.py`

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_models/test_matchup.py -v`
Expected: 3 tests PASS

- [ ] **Step 6: Commit**

```bash
git add src/models/__init__.py src/models/matchup.py tests/test_models/__init__.py tests/test_models/test_matchup.py
git commit -m "feat: symmetric matchup training data builder"
```

---

### Task 13: Model Training + Platt Calibration

**Files:**
- Create: `src/models/train.py`
- Create: `tests/test_models/test_train.py`

- [ ] **Step 1: Write the failing tests**

```python
"""Tests for model training and calibration."""

import numpy as np
import pandas as pd
import pytest

from src.models.train import train_model, predict_matchup


@pytest.fixture
def training_data():
    """Synthetic matchup data with clear signal."""
    np.random.seed(42)
    n = 200
    # Feature: positive = team A is stronger
    X = pd.DataFrame({
        "adj_em": np.random.randn(n) * 10,
        "seed": np.random.randn(n) * 3,
    })
    # Label: team A wins when features are positive (with noise)
    y = pd.Series((X["adj_em"] + np.random.randn(n) * 3 > 0).astype(int), name="win")
    return X, y


def test_train_model_returns_pipeline(training_data):
    X, y = training_data
    model = train_model(X, y, random_seed=42)
    assert hasattr(model, "predict_proba")


def test_predict_matchup_returns_probability(training_data):
    X, y = training_data
    model = train_model(X, y, random_seed=42)
    prob = predict_matchup(model, X.iloc[[0]])
    assert 0.0 <= prob <= 1.0


def test_model_predicts_strong_team_wins(training_data):
    X, y = training_data
    model = train_model(X, y, random_seed=42)
    # Very strong team A: should have high win prob
    strong = pd.DataFrame({"adj_em": [25.0], "seed": [-5.0]})
    prob = predict_matchup(model, strong)
    assert prob > 0.7

    # Very weak team A: should have low win prob
    weak = pd.DataFrame({"adj_em": [-25.0], "seed": [5.0]})
    prob = predict_matchup(model, weak)
    assert prob < 0.3
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_models/test_train.py -v`
Expected: FAIL

- [ ] **Step 3: Implement train.py**

```python
"""XGBoost model training with Platt calibration."""

import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold

logger = logging.getLogger(__name__)


def train_model(
    X: pd.DataFrame,
    y: pd.Series,
    random_seed: int = 42,
    xgb_params: dict | None = None,
) -> CalibratedClassifierCV:
    """Train XGBoost classifier with Platt scaling calibration.

    Returns a CalibratedClassifierCV wrapping the XGBoost model.
    """
    params = {
        "n_estimators": 300,
        "max_depth": 4,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": random_seed,
        "eval_metric": "logloss",
    }
    if xgb_params:
        params.update(xgb_params)

    base_model = xgb.XGBClassifier(**params)

    # Platt scaling via 5-fold CV
    calibrated = CalibratedClassifierCV(
        base_model, method="sigmoid", cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=random_seed)
    )
    calibrated.fit(X, y)

    logger.info("Model trained: %d samples, %d features", len(X), X.shape[1])
    return calibrated


def predict_matchup(model: CalibratedClassifierCV, X: pd.DataFrame) -> float:
    """Predict P(team A wins) for a single matchup feature vector."""
    proba = model.predict_proba(X)
    return float(proba[0, 1])


def save_model(
    model: CalibratedClassifierCV,
    output_dir: str,
    config: dict,
    feature_cols: list[str],
    seasons: list[int],
) -> Path:
    """Save model + metadata sidecar JSON."""
    import joblib

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    config_hash = hashlib.md5(json.dumps(config, sort_keys=True).encode()).hexdigest()[:8]
    year = max(seasons)
    model_name = f"xgb_{year}_{config_hash}"

    model_path = output_path / f"{model_name}.pkl"
    meta_path = output_path / f"{model_name}_meta.json"

    joblib.dump(model, model_path)

    meta = {
        "training_date": datetime.now().isoformat(),
        "config_hash": config_hash,
        "seasons": seasons,
        "feature_cols": feature_cols,
        "n_features": len(feature_cols),
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    logger.info("Model saved: %s", model_path)
    return model_path
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_models/test_train.py -v`
Expected: 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/models/train.py tests/test_models/test_train.py
git commit -m "feat: XGBoost training with Platt calibration and model persistence"
```

---

### Task 14: Evaluation + Baselines

**Files:**
- Create: `src/models/evaluate.py`
- Create: `src/models/baselines.py`
- Create: `tests/test_models/test_evaluate.py`

- [ ] **Step 1: Write the failing tests**

```python
"""Tests for model evaluation."""

import numpy as np
import pandas as pd
import pytest

from src.models.evaluate import compute_log_loss, compute_brier_score, leave_one_season_out_cv
from src.models.baselines import seed_baseline_prob


def test_compute_log_loss_perfect():
    y_true = pd.Series([1, 0, 1])
    y_prob = np.array([0.99, 0.01, 0.99])
    loss = compute_log_loss(y_true, y_prob)
    assert loss < 0.05


def test_compute_log_loss_random():
    y_true = pd.Series([1, 0, 1, 0])
    y_prob = np.array([0.5, 0.5, 0.5, 0.5])
    loss = compute_log_loss(y_true, y_prob)
    assert abs(loss - 0.693) < 0.01  # ln(2) ≈ 0.693


def test_compute_brier_score_perfect():
    y_true = pd.Series([1, 0, 1])
    y_prob = np.array([1.0, 0.0, 1.0])
    score = compute_brier_score(y_true, y_prob)
    assert score == 0.0


def test_compute_brier_score_random():
    y_true = pd.Series([1, 0, 1, 0])
    y_prob = np.array([0.5, 0.5, 0.5, 0.5])
    score = compute_brier_score(y_true, y_prob)
    assert abs(score - 0.25) < 0.01


def test_seed_baseline_prob():
    # 1-seed vs 16-seed: should strongly favor 1-seed
    prob = seed_baseline_prob(1, 16)
    assert prob > 0.9

    # Equal seeds: should be ~0.5
    prob = seed_baseline_prob(8, 8)
    assert abs(prob - 0.5) < 0.01

    # 16-seed vs 1-seed: should strongly favor 1-seed (low prob for team A)
    prob = seed_baseline_prob(16, 1)
    assert prob < 0.1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_models/test_evaluate.py -v`
Expected: FAIL

- [ ] **Step 3: Implement baselines.py**

```python
"""Baseline models for comparison."""

import numpy as np


def seed_baseline_prob(seed_a: int, seed_b: int) -> float:
    """Predict P(team A wins) based purely on seed difference.

    Uses a logistic function fitted to historical seed-vs-seed outcomes.
    Coefficient of ~0.15 per seed difference is a reasonable approximation.
    """
    seed_diff = seed_b - seed_a  # positive means A is better seed
    return float(1 / (1 + np.exp(-0.15 * seed_diff)))
```

- [ ] **Step 4: Implement evaluate.py**

```python
"""Model evaluation with leave-one-season-out CV."""

import logging

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss as sklearn_log_loss, roc_auc_score

from src.models.matchup import build_matchup_data
from src.models.train import train_model

logger = logging.getLogger(__name__)


def compute_log_loss(y_true: pd.Series, y_prob: np.ndarray) -> float:
    """Compute log loss (cross-entropy)."""
    return float(sklearn_log_loss(y_true, y_prob))


def compute_brier_score(y_true: pd.Series, y_prob: np.ndarray) -> float:
    """Compute Brier score (mean squared error of probabilities)."""
    return float(np.mean((y_prob - y_true.values) ** 2))


def leave_one_season_out_cv(
    feature_matrix: pd.DataFrame,
    tourney_results: pd.DataFrame,
    feature_cols: list[str],
    random_seed: int = 42,
    xgb_params: dict | None = None,
) -> dict:
    """Run leave-one-season-out cross-validation.

    For each season, trains on all other seasons and evaluates
    on the held-out season's tournament games.

    Returns dict with per-season and aggregate metrics.
    """
    seasons = sorted(tourney_results["Season"].unique())
    results = []

    for holdout_season in seasons:
        # Split
        train_tourney = tourney_results[tourney_results["Season"] != holdout_season]
        test_tourney = tourney_results[tourney_results["Season"] == holdout_season]

        if len(test_tourney) == 0:
            continue

        X_train, y_train = build_matchup_data(feature_matrix, train_tourney, feature_cols)
        X_test, y_test = build_matchup_data(feature_matrix, test_tourney, feature_cols)

        if len(X_train) == 0 or len(X_test) == 0:
            continue

        model = train_model(X_train, y_train, random_seed=random_seed, xgb_params=xgb_params)
        y_prob = model.predict_proba(X_test)[:, 1]

        season_loss = compute_log_loss(y_test, y_prob)
        season_brier = compute_brier_score(y_test, y_prob)
        season_acc = float((y_prob.round() == y_test).mean())
        season_auc = float(roc_auc_score(y_test, y_prob))

        results.append({
            "season": holdout_season,
            "log_loss": season_loss,
            "brier_score": season_brier,
            "accuracy": season_acc,
            "auc": season_auc,
            "n_games": len(test_tourney),
        })
        logger.info("Season %d: log_loss=%.4f, brier=%.4f, acc=%.3f, auc=%.3f", holdout_season, season_loss, season_brier, season_acc, season_auc)

    results_df = pd.DataFrame(results)
    return {
        "per_season": results_df,
        "mean_log_loss": float(results_df["log_loss"].mean()),
        "mean_brier_score": float(results_df["brier_score"].mean()),
        "mean_accuracy": float(results_df["accuracy"].mean()),
        "mean_auc": float(results_df["auc"].mean()),
    }
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_models/test_evaluate.py -v`
Expected: 5 tests PASS

- [ ] **Step 6: Commit**

```bash
git add src/models/evaluate.py src/models/baselines.py tests/test_models/test_evaluate.py
git commit -m "feat: evaluation with LOSO CV, log loss, Brier score, and seed baseline"
```

---

### Task 15: Hyperparameter Tuning with Optuna

**Files:**
- Create: `src/models/tuning.py`
- Create: `tests/test_models/test_tuning.py`

- [ ] **Step 1: Write the failing tests**

```python
"""Tests for Optuna hyperparameter tuning."""

import numpy as np
import pandas as pd
import pytest

from src.models.tuning import tune_hyperparameters


@pytest.fixture
def training_data():
    np.random.seed(42)
    n = 200
    X = pd.DataFrame({
        "adj_em": np.random.randn(n) * 10,
        "seed": np.random.randn(n) * 3,
    })
    y = pd.Series((X["adj_em"] + np.random.randn(n) * 3 > 0).astype(int), name="win")
    return X, y


def test_tune_returns_params(training_data):
    X, y = training_data
    best_params = tune_hyperparameters(X, y, n_trials=5, random_seed=42)
    assert isinstance(best_params, dict)
    assert "max_depth" in best_params
    assert "learning_rate" in best_params


def test_tune_params_in_range(training_data):
    X, y = training_data
    best_params = tune_hyperparameters(X, y, n_trials=5, random_seed=42)
    assert 2 <= best_params["max_depth"] <= 8
    assert 0.01 <= best_params["learning_rate"] <= 0.3
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_models/test_tuning.py -v`
Expected: FAIL

- [ ] **Step 3: Implement tuning.py**

```python
"""Bayesian hyperparameter optimization via Optuna."""

import logging

import numpy as np
import pandas as pd
import optuna
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold

logger = logging.getLogger(__name__)

# Suppress Optuna's verbose logging
optuna.logging.set_verbosity(optuna.logging.WARNING)


def tune_hyperparameters(
    X: pd.DataFrame,
    y: pd.Series,
    n_trials: int = 50,
    random_seed: int = 42,
) -> dict:
    """Find optimal XGBoost hyperparameters using Optuna.

    Uses 5-fold stratified CV with log loss as the objective.
    Returns dict of best hyperparameters.
    """
    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 2, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "random_state": random_seed,
            "eval_metric": "logloss",
        }

        model = xgb.XGBClassifier(**params)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_seed)

        losses = []
        for train_idx, val_idx in cv.split(X, y):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            model.fit(X_train, y_train)
            y_prob = model.predict_proba(X_val)[:, 1]
            losses.append(log_loss(y_val, y_prob))

        return np.mean(losses)

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=random_seed),
    )
    study.optimize(objective, n_trials=n_trials)

    logger.info("Best trial: log_loss=%.4f, params=%s", study.best_value, study.best_params)
    return study.best_params
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_models/test_tuning.py -v`
Expected: 2 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/models/tuning.py tests/test_models/test_tuning.py
git commit -m "feat: Optuna hyperparameter tuning for XGBoost"
```

---

### Task 16: Models CLI Entry Point

**Files:**
- Create: `src/models/__main__.py`

- [ ] **Step 1: Implement __main__.py**

```python
"""CLI entry point: python -m src.models"""

import logging
from pathlib import Path

import pandas as pd

from src.config import load_config
from src.models.matchup import build_matchup_data
from src.models.train import train_model, save_model
from src.models.evaluate import leave_one_season_out_cv
from src.models.tuning import tune_hyperparameters

logging.basicConfig(level=logging.INFO, format="%(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

FEATURE_COLS = [
    "adj_oe", "adj_de", "adj_em", "adj_tempo",
    "off_efg", "off_to_rate", "off_or_rate", "off_ft_rate",
    "def_efg", "def_to_rate", "def_or_rate", "def_ft_rate",
    "seed", "massey_composite_rank",
    "win_pct_last_10", "road_win_pct",
]


def main() -> None:
    config = load_config()
    processed_dir = Path(config["data"]["processed_dir"])

    feature_matrix = pd.read_parquet(processed_dir / "feature_matrix.parquet")
    tourney_results = pd.read_parquet(processed_dir / "tourney_results.parquet")

    # Filter feature cols to those actually present
    available_cols = [c for c in FEATURE_COLS if c in feature_matrix.columns]
    logger.info("Using %d features: %s", len(available_cols), available_cols)

    # Build training data
    X, y = build_matchup_data(feature_matrix, tourney_results, available_cols)

    # Hyperparameter tuning
    logger.info("Tuning hyperparameters with Optuna (50 trials)")
    best_params = tune_hyperparameters(X, y, n_trials=50, random_seed=config["model"]["random_seed"])
    logger.info("Best params: %s", best_params)

    # Evaluate via LOSO CV with tuned params
    logger.info("Running leave-one-season-out cross-validation")
    cv_results = leave_one_season_out_cv(
        feature_matrix, tourney_results, available_cols,
        random_seed=config["model"]["random_seed"],
        xgb_params=best_params,
    )
    logger.info(
        "CV Results: log_loss=%.4f, accuracy=%.3f, auc=%.3f",
        cv_results["mean_log_loss"],
        cv_results["mean_accuracy"],
        cv_results["mean_auc"],
    )

    # Train final model on all data with tuned params
    logger.info("Training final model on all seasons")
    model = train_model(X, y, random_seed=config["model"]["random_seed"], xgb_params=best_params)

    seasons = sorted(feature_matrix["Season"].unique().tolist())
    save_model(model, "models", config, available_cols, seasons)

    logger.info("Model training complete")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
git add src/models/__main__.py
git commit -m "feat: models CLI with LOSO CV evaluation and final model training"
```

---

## Chunk 4: Bracket Simulation & Output

### Task 17: Monte Carlo Tournament Simulator

**Files:**
- Create: `src/bracket/__init__.py`
- Create: `src/bracket/simulator.py`
- Create: `tests/test_bracket/__init__.py`
- Create: `tests/test_bracket/test_simulator.py`

- [ ] **Step 1: Write the failing tests**

```python
"""Tests for Monte Carlo bracket simulation."""

import pandas as pd
import pytest

from src.bracket.simulator import (
    load_bracket,
    simulate_tournament,
    get_advancement_probabilities,
    FIRST_ROUND_MATCHUPS,
)


@pytest.fixture
def bracket_df():
    """64-team 4-region bracket."""
    regions = ["East", "West", "South", "Midwest"]
    rows = []
    team_id = 101
    for region in regions:
        for seed in range(1, 17):
            rows.append({"Region": region, "Seed": seed, "TeamID": team_id, "TeamName": f"Team{team_id}"})
            team_id += 1
    return pd.DataFrame(rows)


@pytest.fixture
def mock_predict():
    """Predict function: higher seed always wins with 70% prob."""
    def predict(team_a_features, team_b_features):
        seed_a = team_a_features["seed"]
        seed_b = team_b_features["seed"]
        if seed_a < seed_b:
            return 0.7
        elif seed_a > seed_b:
            return 0.3
        else:
            return 0.5
    return predict


def test_first_round_matchups():
    assert FIRST_ROUND_MATCHUPS == [(1, 16), (8, 9), (5, 12), (4, 13), (6, 11), (3, 14), (7, 10), (2, 15)]


def test_simulate_tournament_returns_results(bracket_df, mock_predict):
    feature_matrix = pd.DataFrame({
        "TeamID": list(range(101, 165)),
        "seed": list(range(1, 17)) * 4,
        "adj_em": [30 - (i % 16) * 2 for i in range(64)],
    })
    results = simulate_tournament(
        bracket=bracket_df,
        feature_matrix=feature_matrix,
        predict_fn=mock_predict,
        feature_cols=["seed", "adj_em"],
        n_simulations=100,
        random_seed=42,
    )
    assert "advancement_counts" in results
    assert "champions" in results
    # 1-seeds should collectively dominate championships
    one_seeds = [101, 117, 133, 149]  # 1-seeds from each region
    champ_counts = results["champions"]
    one_seed_wins = sum(champ_counts.get(t, 0) for t in one_seeds)
    assert one_seed_wins > 50  # should win majority of 100 sims


def test_get_advancement_probabilities():
    counts = {101: {1: 100, 2: 80, 3: 50, 4: 20}, 102: {1: 100, 2: 20, 3: 5, 4: 0}}
    probs = get_advancement_probabilities(counts, n_simulations=100)
    assert probs[101][4] == 0.2
    assert probs[102][3] == 0.05
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_bracket/test_simulator.py -v`
Expected: FAIL

- [ ] **Step 3: Implement simulator.py**

```python
"""Monte Carlo tournament bracket simulation."""

import logging
from collections import defaultdict
from typing import Callable

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Standard NCAA first-round matchups by seed within a region
FIRST_ROUND_MATCHUPS = [
    (1, 16), (8, 9), (5, 12), (4, 13),
    (6, 11), (3, 14), (7, 10), (2, 15),
]


def load_bracket(path: str) -> pd.DataFrame:
    """Load bracket CSV with columns: Region, Seed, TeamID, TeamName."""
    df = pd.read_csv(path)
    required = {"Region", "Seed", "TeamID", "TeamName"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Bracket CSV missing columns: {missing}")
    return df


def _get_team_features(team_id: int, feature_matrix: pd.DataFrame, feature_cols: list[str]) -> dict:
    """Look up features for a team."""
    row = feature_matrix[feature_matrix["TeamID"] == team_id]
    if row.empty:
        raise ValueError(f"Team {team_id} not found in feature matrix")
    return row.iloc[0][feature_cols].to_dict()


def _simulate_game(
    team_a: int,
    team_b: int,
    feature_matrix: pd.DataFrame,
    predict_fn: Callable,
    feature_cols: list[str],
    rng: np.random.Generator,
) -> int:
    """Simulate a single game, return winning team ID."""
    a_features = _get_team_features(team_a, feature_matrix, feature_cols)
    b_features = _get_team_features(team_b, feature_matrix, feature_cols)
    prob_a_wins = predict_fn(a_features, b_features)
    return team_a if rng.random() < prob_a_wins else team_b


def _simulate_region(
    region_teams: pd.DataFrame,
    feature_matrix: pd.DataFrame,
    predict_fn: Callable,
    feature_cols: list[str],
    rng: np.random.Generator,
    advancement: dict[int, dict[int, int]],
) -> int:
    """Simulate a single region through 4 rounds, return regional champion TeamID."""
    # Map seed -> TeamID
    seed_to_team = dict(zip(region_teams["Seed"], region_teams["TeamID"]))

    # Round 1: 8 games from FIRST_ROUND_MATCHUPS
    round_winners = []
    for seed_a, seed_b in FIRST_ROUND_MATCHUPS:
        team_a = seed_to_team[seed_a]
        team_b = seed_to_team[seed_b]
        winner = _simulate_game(team_a, team_b, feature_matrix, predict_fn, feature_cols, rng)
        round_winners.append(winner)
        advancement[winner][1] = advancement[winner].get(1, 0) + 1

    # Rounds 2-4
    current = round_winners
    for round_num in range(2, 5):
        next_round = []
        for i in range(0, len(current), 2):
            winner = _simulate_game(current[i], current[i + 1], feature_matrix, predict_fn, feature_cols, rng)
            next_round.append(winner)
            advancement[winner][round_num] = advancement[winner].get(round_num, 0) + 1
        current = next_round

    return current[0]  # regional champion


def simulate_tournament(
    bracket: pd.DataFrame,
    feature_matrix: pd.DataFrame,
    predict_fn: Callable,
    feature_cols: list[str],
    n_simulations: int = 10000,
    random_seed: int = 42,
) -> dict:
    """Run Monte Carlo simulation of the full tournament.

    Returns dict with:
        advancement_counts: {team_id: {round: count}}
        champions: {team_id: count}
        n_simulations: int
    """
    rng = np.random.default_rng(random_seed)
    regions = sorted(bracket["Region"].unique())
    advancement = defaultdict(lambda: defaultdict(int))
    champions = defaultdict(int)

    for sim in range(n_simulations):
        # Simulate each region (rounds 1-4: R64, R32, S16, E8)
        regional_champs = []
        for region in regions:
            region_teams = bracket[bracket["Region"] == region].copy()
            champ = _simulate_region(
                region_teams, feature_matrix, predict_fn, feature_cols, rng, advancement
            )
            regional_champs.append(champ)
            # Round 5 = reached Final Four (won region)
            advancement[champ][5] = advancement[champ].get(5, 0) + 1

        # Semifinals (round 5 winners → round 6 = won semifinal)
        semi1 = _simulate_game(regional_champs[0], regional_champs[1], feature_matrix, predict_fn, feature_cols, rng)
        semi2 = _simulate_game(regional_champs[2], regional_champs[3], feature_matrix, predict_fn, feature_cols, rng)

        # Championship (round 6 = won championship)
        champion = _simulate_game(semi1, semi2, feature_matrix, predict_fn, feature_cols, rng)
        advancement[champion][6] = advancement[champion].get(6, 0) + 1
        champions[champion] += 1

        if (sim + 1) % 1000 == 0:
            logger.info("Completed %d / %d simulations", sim + 1, n_simulations)

    return {
        "advancement_counts": dict(advancement),
        "champions": dict(champions),
        "n_simulations": n_simulations,
    }


def get_advancement_probabilities(
    advancement_counts: dict[int, dict[int, int]],
    n_simulations: int,
) -> dict[int, dict[int, float]]:
    """Convert raw counts to probabilities."""
    probs = {}
    for team_id, rounds in advancement_counts.items():
        probs[team_id] = {r: count / n_simulations for r, count in rounds.items()}
    return probs
```

- [ ] **Step 4: Create empty __init__.py files**

Create: `src/bracket/__init__.py`, `tests/test_bracket/__init__.py`

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_bracket/test_simulator.py -v`
Expected: 4 tests PASS

- [ ] **Step 6: Commit**

```bash
git add src/bracket/__init__.py src/bracket/simulator.py tests/test_bracket/__init__.py tests/test_bracket/test_simulator.py
git commit -m "feat: Monte Carlo tournament simulator with region-based bracket structure"
```

---

### Task 18: Bracket Selection Strategies

**Files:**
- Create: `src/bracket/strategies.py`
- Create: `tests/test_bracket/test_strategies.py`

- [ ] **Step 1: Write the failing tests**

```python
"""Tests for bracket selection strategies."""

import pandas as pd
import pytest

from src.bracket.strategies import chalk_bracket, expected_value_bracket
from src.bracket.simulator import FIRST_ROUND_MATCHUPS


@pytest.fixture
def bracket_df():
    """Single-region 16-team bracket for strategy testing."""
    return pd.DataFrame({
        "Region": ["East"] * 16,
        "Seed": list(range(1, 17)),
        "TeamID": list(range(101, 117)),
        "TeamName": [f"Team{i}" for i in range(1, 17)],
    })


@pytest.fixture
def advancement_probs():
    """Advancement probabilities for 16 teams across 4 region rounds."""
    probs = {}
    for i in range(16):
        team_id = 101 + i
        seed = i + 1
        # Better seeds get higher advancement probs
        base = max(0.98 - seed * 0.05, 0.02)
        probs[team_id] = {
            1: min(base * 1.2, 0.99),
            2: base * 0.8,
            3: base * 0.5,
            4: base * 0.3,
        }
    return probs


def test_chalk_bracket_structure(bracket_df, advancement_probs):
    picks = chalk_bracket(bracket_df, advancement_probs)
    # Round 1: exactly 8 winners from 8 matchups
    assert len(picks[1]) == 8
    # Round 2: exactly 4 winners
    assert len(picks[2]) == 4
    # All round 2 picks must be in round 1
    assert all(t in picks[1] for t in picks[2])
    # Round 4: regional champion
    assert len(picks[4]) == 1
    assert picks[4][0] in picks[3]


def test_chalk_bracket_picks_favorites(bracket_df, advancement_probs):
    picks = chalk_bracket(bracket_df, advancement_probs)
    # 1-seed should beat 16-seed in round 1
    assert 101 in picks[1]
    # 1-seed should be regional champion
    assert 101 in picks[4]


def test_expected_value_bracket_structure(bracket_df, advancement_probs):
    scoring = [1, 2, 4, 8]
    picks = expected_value_bracket(bracket_df, advancement_probs, scoring=scoring)
    # Same structural constraints as chalk
    assert len(picks[1]) == 8
    assert len(picks[2]) == 4
    assert all(t in picks[1] for t in picks[2])
    assert len(picks[4]) == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_bracket/test_strategies.py -v`
Expected: FAIL

- [ ] **Step 3: Implement strategies.py**

```python
"""Bracket selection strategies that produce structurally valid brackets.

A valid bracket respects the bracket tree: picks in round N+1 must be
a subset of picks in round N, and each matchup slot has exactly one winner.
"""

import pandas as pd

from src.bracket.simulator import FIRST_ROUND_MATCHUPS


def _pick_slot_winner(
    team_a: int,
    team_b: int,
    probs: dict[int, dict[int, float]],
    round_num: int,
    scorer: str = "prob",
    points: int = 1,
) -> int:
    """Pick the winner of a single bracket slot.

    scorer="prob": pick higher advancement probability (chalk).
    scorer="ev": pick higher expected value (prob * points).
    """
    prob_a = probs.get(team_a, {}).get(round_num, 0.0)
    prob_b = probs.get(team_b, {}).get(round_num, 0.0)
    if scorer == "ev":
        val_a = prob_a * points
        val_b = prob_b * points
    else:
        val_a = prob_a
        val_b = prob_b
    return team_a if val_a >= val_b else team_b


def _fill_region(
    region_teams: pd.DataFrame,
    probs: dict[int, dict[int, float]],
    scorer: str = "prob",
    scoring: list[int] | None = None,
) -> dict[int, list[int]]:
    """Fill bracket picks for a single region (4 rounds)."""
    if scoring is None:
        scoring = [1, 2, 4, 8]

    seed_to_team = dict(zip(region_teams["Seed"], region_teams["TeamID"]))
    picks = {}

    # Round 1: known matchups from bracket structure
    round1_winners = []
    for seed_a, seed_b in FIRST_ROUND_MATCHUPS:
        team_a = seed_to_team[seed_a]
        team_b = seed_to_team[seed_b]
        winner = _pick_slot_winner(team_a, team_b, probs, 1, scorer, scoring[0])
        round1_winners.append(winner)
    picks[1] = round1_winners

    # Rounds 2-4: winners play each other in bracket order
    current = round1_winners
    for rnd in range(2, 5):
        next_round = []
        pts = scoring[rnd - 1] if rnd - 1 < len(scoring) else scoring[-1]
        for i in range(0, len(current), 2):
            winner = _pick_slot_winner(current[i], current[i + 1], probs, rnd, scorer, pts)
            next_round.append(winner)
        picks[rnd] = next_round
        current = next_round

    return picks


def chalk_bracket(
    bracket: pd.DataFrame,
    advancement_probs: dict[int, dict[int, float]],
) -> dict[int, list[int]]:
    """Chalk bracket: pick the higher-probability team in each slot.

    Returns {round_number: [team_ids advancing that round]}.
    Respects bracket structure — all picks are consistent across rounds.
    """
    regions = sorted(bracket["Region"].unique())
    all_picks = {r: [] for r in range(1, 7)}

    # Fill each region (rounds 1-4)
    regional_champs = []
    for region in regions:
        region_teams = bracket[bracket["Region"] == region]
        region_picks = _fill_region(region_teams, advancement_probs, scorer="prob")
        for rnd, teams in region_picks.items():
            all_picks[rnd].extend(teams)
        regional_champs.append(region_picks[4][0])

    # Semifinals (round 5)
    semi1 = _pick_slot_winner(regional_champs[0], regional_champs[1], advancement_probs, 5, "prob")
    semi2 = _pick_slot_winner(regional_champs[2], regional_champs[3], advancement_probs, 5, "prob")
    all_picks[5] = [semi1, semi2]

    # Championship (round 6)
    champ = _pick_slot_winner(semi1, semi2, advancement_probs, 6, "prob")
    all_picks[6] = [champ]

    return all_picks


def expected_value_bracket(
    bracket: pd.DataFrame,
    advancement_probs: dict[int, dict[int, float]],
    scoring: list[int] | None = None,
) -> dict[int, list[int]]:
    """Expected value bracket: pick teams maximizing expected points per slot.

    Returns {round_number: [team_ids advancing that round]}.
    """
    if scoring is None:
        scoring = [1, 2, 4, 8, 16, 32]

    regions = sorted(bracket["Region"].unique())
    all_picks = {r: [] for r in range(1, 7)}

    regional_champs = []
    for region in regions:
        region_teams = bracket[bracket["Region"] == region]
        region_picks = _fill_region(region_teams, advancement_probs, scorer="ev", scoring=scoring)
        for rnd, teams in region_picks.items():
            all_picks[rnd].extend(teams)
        regional_champs.append(region_picks[4][0])

    # Semifinals
    semi1 = _pick_slot_winner(regional_champs[0], regional_champs[1], advancement_probs, 5, "ev", scoring[4])
    semi2 = _pick_slot_winner(regional_champs[2], regional_champs[3], advancement_probs, 5, "ev", scoring[4])
    all_picks[5] = [semi1, semi2]

    # Championship
    champ = _pick_slot_winner(semi1, semi2, advancement_probs, 6, "ev", scoring[5])
    all_picks[6] = [champ]

    return all_picks
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_bracket/test_strategies.py -v`
Expected: 2 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/bracket/strategies.py tests/test_bracket/test_strategies.py
git commit -m "feat: chalk and expected-value bracket selection strategies"
```

---

### Task 19: Bracket Output

**Files:**
- Create: `src/bracket/output.py`
- Create: `tests/test_bracket/test_output.py`

- [ ] **Step 1: Write the failing tests**

```python
"""Tests for bracket output formatting."""

import pandas as pd
import pytest

from src.bracket.output import format_advancement_table, export_bracket_csv


@pytest.fixture
def advancement_probs():
    return {
        101: {1: 0.95, 2: 0.70, 3: 0.40, 4: 0.20, 5: 0.10, 6: 0.05},
        102: {1: 0.80, 2: 0.50, 3: 0.20, 4: 0.08, 5: 0.03, 6: 0.01},
    }


@pytest.fixture
def teams():
    return pd.DataFrame({"TeamID": [101, 102], "TeamName": ["Duke", "UNC"]})


def test_format_advancement_table(advancement_probs, teams):
    table = format_advancement_table(advancement_probs, teams)
    assert isinstance(table, str)
    assert "Duke" in table
    assert "UNC" in table


def test_export_bracket_csv(advancement_probs, teams, tmp_path):
    output_path = tmp_path / "bracket.csv"
    export_bracket_csv(advancement_probs, teams, str(output_path))
    df = pd.read_csv(output_path)
    assert len(df) == 2
    assert "TeamName" in df.columns
    assert "R1" in df.columns
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_bracket/test_output.py -v`
Expected: FAIL

- [ ] **Step 3: Implement output.py**

```python
"""Bracket output formatting and export."""

import pandas as pd

ROUND_NAMES = {1: "R64", 2: "R32", 3: "S16", 4: "E8", 5: "F4", 6: "Champ"}


def format_advancement_table(
    advancement_probs: dict[int, dict[int, float]],
    teams: pd.DataFrame,
) -> str:
    """Format advancement probabilities as a readable table string."""
    team_map = dict(zip(teams["TeamID"], teams["TeamName"]))
    rows = []

    for team_id, rounds in sorted(
        advancement_probs.items(),
        key=lambda x: x[1].get(6, 0),
        reverse=True,
    ):
        name = team_map.get(team_id, str(team_id))
        probs = "  ".join(
            f"{ROUND_NAMES.get(r, f'R{r}')}: {rounds.get(r, 0):.1%}"
            for r in range(1, 7)
        )
        rows.append(f"{name:25s} {probs}")

    header = f"{'Team':25s} " + "  ".join(f"{ROUND_NAMES.get(r, f'R{r}'):>6s}" for r in range(1, 7))
    return header + "\n" + "-" * len(header) + "\n" + "\n".join(rows)


def export_bracket_csv(
    advancement_probs: dict[int, dict[int, float]],
    teams: pd.DataFrame,
    output_path: str,
) -> None:
    """Export advancement probabilities to CSV."""
    team_map = dict(zip(teams["TeamID"], teams["TeamName"]))
    rows = []
    for team_id, rounds in advancement_probs.items():
        row = {"TeamID": team_id, "TeamName": team_map.get(team_id, str(team_id))}
        for r in range(1, 7):
            row[f"R{r}"] = rounds.get(r, 0.0)
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("R6", ascending=False)
    df.to_csv(output_path, index=False)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_bracket/test_output.py -v`
Expected: 2 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/bracket/output.py tests/test_bracket/test_output.py
git commit -m "feat: bracket output with advancement table and CSV export"
```

---

### Task 20: Bracket CLI Entry Point

**Files:**
- Create: `src/bracket/__main__.py`

- [ ] **Step 1: Implement __main__.py**

```python
"""CLI entry point: python -m src.bracket"""

import logging
from pathlib import Path

import joblib
import pandas as pd

from src.config import load_config
from src.models.train import predict_matchup
from src.bracket.simulator import load_bracket, simulate_tournament, get_advancement_probabilities
from src.bracket.strategies import chalk_bracket, expected_value_bracket
from src.bracket.output import format_advancement_table, export_bracket_csv

logging.basicConfig(level=logging.INFO, format="%(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    config = load_config()
    processed_dir = Path(config["data"]["processed_dir"])
    predict_season = config["seasons"]["predict_season"]

    # Load model
    import glob
    model_files = sorted(glob.glob("models/xgb_*.pkl"))
    if not model_files:
        raise FileNotFoundError("No trained model found in models/. Run 'python -m src.models' first.")
    model_path = model_files[-1]
    meta_path = model_path.replace(".pkl", "_meta.json")

    import json
    with open(meta_path) as f:
        meta = json.load(f)
    feature_cols = meta["feature_cols"]

    model = joblib.load(model_path)
    logger.info("Loaded model: %s (%d features)", model_path, len(feature_cols))

    # Load data
    feature_matrix = pd.read_parquet(processed_dir / "feature_matrix.parquet")
    current_features = feature_matrix[feature_matrix["Season"] == predict_season]
    teams = pd.read_parquet(processed_dir / "teams.parquet")

    # Load bracket
    bracket_path = f"data/raw/bracket_{predict_season}.csv"
    bracket = load_bracket(bracket_path)

    # Validate all bracket teams have features
    missing = set(bracket["TeamID"]) - set(current_features["TeamID"])
    if missing:
        raise ValueError(f"Bracket teams missing from feature matrix: {missing}")

    # Define predict function for simulator
    def predict_fn(a_features: dict, b_features: dict) -> float:
        diff = {col: a_features[col] - b_features[col] for col in feature_cols}
        X = pd.DataFrame([diff])
        return predict_matchup(model, X)

    # Simulate
    logger.info("Running %d simulations", config["model"]["n_simulations"])
    sim_results = simulate_tournament(
        bracket=bracket,
        feature_matrix=current_features,
        predict_fn=predict_fn,
        feature_cols=feature_cols,
        n_simulations=config["model"]["n_simulations"],
        random_seed=config["model"]["random_seed"],
    )

    # Advancement probabilities
    probs = get_advancement_probabilities(sim_results["advancement_counts"], sim_results["n_simulations"])

    # Output
    print("\n" + "=" * 80)
    print("ADVANCEMENT PROBABILITIES")
    print("=" * 80)
    print(format_advancement_table(probs, teams))

    # Champion probabilities
    champ_probs = {tid: count / sim_results["n_simulations"] for tid, count in sim_results["champions"].items()}
    team_map = dict(zip(teams["TeamID"], teams["TeamName"]))
    print("\n" + "=" * 80)
    print("CHAMPION PROBABILITIES")
    print("=" * 80)
    for tid, prob in sorted(champ_probs.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {team_map.get(tid, str(tid)):25s} {prob:.1%}")

    # Strategies
    scoring = config["bracket"]["scoring"]
    for strategy_name in config["bracket"]["strategies"]:
        if strategy_name == "chalk":
            picks = chalk_bracket(bracket, probs)
        elif strategy_name == "expected_value":
            picks = expected_value_bracket(bracket, probs, scoring=scoring)
        else:
            logger.warning("Unknown strategy: %s", strategy_name)
            continue

        print(f"\n{'=' * 80}")
        print(f"BRACKET: {strategy_name.upper()}")
        print(f"{'=' * 80}")
        champion = picks[6][0] if picks.get(6) else None
        if champion:
            print(f"  Champion: {team_map.get(champion, str(champion))}")
        final_four = picks.get(5, [])[:4]
        print(f"  Final Four: {[team_map.get(t, str(t)) for t in final_four]}")

    # Export
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    export_bracket_csv(probs, teams, str(output_dir / f"bracket_{predict_season}.csv"))
    logger.info("Bracket exported to output/bracket_%d.csv", predict_season)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
git add src/bracket/__main__.py
git commit -m "feat: bracket CLI with simulation, strategies, and output"
```

---

### Task 21: Integration Smoke Test

**Files:**
- Create: `tests/test_integration.py`

- [ ] **Step 1: Write an end-to-end test with synthetic data**

```python
"""Integration test: full pipeline with synthetic data."""

import numpy as np
import pandas as pd
import pytest

from src.features.efficiency import compute_adjusted_efficiency
from src.features.four_factors import compute_four_factors
from src.features.feature_matrix import build_feature_matrix
from src.models.matchup import build_matchup_data
from src.models.train import train_model, predict_matchup
from src.bracket.simulator import simulate_tournament, get_advancement_probabilities
from src.bracket.strategies import chalk_bracket


def _generate_synthetic_season(season: int, n_teams: int = 16, games_per_team: int = 10) -> pd.DataFrame:
    """Generate synthetic box score data."""
    np.random.seed(season)
    teams = list(range(1, n_teams + 1))
    games = []
    for _ in range(n_teams * games_per_team // 2):
        t1, t2 = np.random.choice(teams, 2, replace=False)
        # Stronger teams (lower ID) score more on average
        s1 = int(75 - t1 + np.random.randint(-10, 11))
        s2 = int(75 - t2 + np.random.randint(-10, 11))
        if s1 == s2:
            s1 += 1
        w, l = (t1, t2) if s1 > s2 else (t2, t1)
        ws, ls = max(s1, s2), min(s1, s2)
        games.append({
            "Season": season, "DayNum": np.random.randint(1, 132),
            "WTeamID": w, "WScore": ws, "LTeamID": l, "LScore": ls, "WLoc": "N", "NumOT": 0,
            "WFGM": 28, "WFGA": 58, "WFGM3": 8, "WFGA3": 20, "WFTM": 10, "WFTA": 14,
            "WOR": 10, "WDR": 22, "WAst": 14, "WTO": 12, "WStl": 6, "WBlk": 3, "WPF": 17,
            "LFGM": 24, "LFGA": 58, "LFGM3": 6, "LFGA3": 18, "LFTM": 10, "LFTA": 14,
            "LOR": 8, "LDR": 20, "LAst": 12, "LTO": 14, "LStl": 5, "LBlk": 2, "LPF": 16,
        })
    return pd.DataFrame(games)


def test_full_pipeline_synthetic():
    """Test the complete pipeline from raw data to bracket output."""
    # Generate 3 seasons of data
    all_results = pd.concat([_generate_synthetic_season(s) for s in [2021, 2022, 2023]])
    seeds = pd.DataFrame({
        "Season": [s for s in [2021, 2022, 2023] for _ in range(16)],
        "Seed": [f"W{i:02d}" for _ in range(3) for i in range(1, 17)],
        "TeamID": list(range(1, 17)) * 3,
    })
    massey = pd.DataFrame({
        "Season": [s for s in [2021, 2022, 2023] for _ in range(16)],
        "RankingDayNum": [128] * 48,
        "SystemName": ["POM"] * 48,
        "TeamID": list(range(1, 17)) * 3,
        "OrdinalRank": list(range(1, 17)) * 3,
    })
    tourney = pd.DataFrame({
        "Season": [2021, 2021, 2022, 2022],
        "WTeamID": [1, 2, 1, 3],
        "LTeamID": [16, 15, 14, 13],
    })

    feature_cols = ["adj_em", "adj_oe", "adj_de", "seed"]
    all_features = []
    for season in [2021, 2022, 2023]:
        season_data = all_results[all_results["Season"] == season]
        eff = compute_adjusted_efficiency(season_data, season, iterations=5, hca=3.5, half_life_days=30, ridge_alpha=1.0)
        ff = compute_four_factors(season_data, season)
        season_massey = massey[massey["Season"] == season]
        matrix = build_feature_matrix(eff, ff, seeds, season_massey, all_results, season, ["POM"])
        matrix["Season"] = season
        all_features.append(matrix)
    full_matrix = pd.concat(all_features, ignore_index=True)

    # Train model
    X, y = build_matchup_data(full_matrix, tourney, feature_cols)
    assert len(X) > 0
    model = train_model(X, y, random_seed=42)

    # Simulate bracket for 2023 — need 4 regions, 64 teams
    # Reuse the 16 teams across 4 regions (same features, different bracket slots)
    regions = ["East", "West", "South", "Midwest"]
    bracket_rows = []
    for region in regions:
        for seed in range(1, 17):
            bracket_rows.append({
                "Region": region,
                "Seed": seed,
                "TeamID": seed,  # reuse same team IDs across regions for simplicity
                "TeamName": f"Team{seed}",
            })
    bracket = pd.DataFrame(bracket_rows)
    current = full_matrix[full_matrix["Season"] == 2023]

    def predict_fn(a_feats, b_feats):
        diff = {c: a_feats[c] - b_feats[c] for c in feature_cols}
        return predict_matchup(model, pd.DataFrame([diff]))

    results = simulate_tournament(bracket, current, predict_fn, feature_cols, n_simulations=100, random_seed=42)

    assert len(results["champions"]) > 0
    probs = get_advancement_probabilities(results["advancement_counts"], 100)
    picks = chalk_bracket(bracket, probs)
    assert len(picks) > 0
    # Champion should be a single team
    assert len(picks[6]) == 1
```

- [ ] **Step 2: Run the integration test**

Run: `pytest tests/test_integration.py -v`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_integration.py
git commit -m "test: end-to-end integration test with synthetic data"
```

---

### Task 22: Run Full Test Suite

> **Note:** Pool-optimized bracket strategy (spec stretch goal) is intentionally deferred. Chalk and expected-value brackets are the core deliverables. Pool optimization can be added as a follow-up task.

- [ ] **Step 1: Run all tests**

Run: `pytest tests/ -v --tb=short`
Expected: All tests PASS

- [ ] **Step 2: Fix any failures, then commit if fixes were needed**
