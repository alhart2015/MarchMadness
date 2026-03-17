"""Load the alternative Kaggle dataset (KenPom/Barttorvik pre-computed ratings)."""

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

_FILES = {
    "kenpom": "KenPom Barttorvik.csv",
    "matchups": "Tournament Matchups.csv",
    "resumes": "Resumes.csv",
    "ratings_538": "538 Ratings.csv",
    "public_picks": "Public Picks.csv",
}


def load_kaggle2026_data(kaggle_dir: str) -> dict[str, pd.DataFrame]:
    """Load all dataset files from the alternative Kaggle directory.

    Returns a dict mapping dataset name to DataFrame:
      - "kenpom":       KenPom Barttorvik per-team-season ratings
      - "matchups":     Tournament Matchups (one row per team per game)
      - "resumes":      Team tournament resumes (NET, ELO, WAB, quad wins)
      - "ratings_538":  FiveThirtyEight power ratings (2016-2024 only)
      - "public_picks": ESPN public pick percentages (2025 only)
    """
    base = Path(kaggle_dir)
    if not base.exists():
        raise FileNotFoundError(f"Kaggle directory not found: {kaggle_dir}")

    data: dict[str, pd.DataFrame] = {}
    for name, filename in _FILES.items():
        filepath = base / filename
        if not filepath.exists():
            logger.warning("Optional file missing, skipping: %s", filepath)
            data[name] = pd.DataFrame()
            continue
        logger.info("Loading %s from %s", name, filepath)
        data[name] = pd.read_csv(filepath)

    # Normalise column names: strip leading/trailing whitespace
    for name, df in data.items():
        if not df.empty:
            df.columns = df.columns.str.strip()

    logger.info(
        "Loaded: kenpom=%d rows, matchups=%d rows, resumes=%d rows, "
        "538=%d rows, picks=%d rows",
        len(data["kenpom"]),
        len(data["matchups"]),
        len(data["resumes"]),
        len(data["ratings_538"]),
        len(data["public_picks"]),
    )
    return data
