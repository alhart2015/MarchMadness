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
