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
