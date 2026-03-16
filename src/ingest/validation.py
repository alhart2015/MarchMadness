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
