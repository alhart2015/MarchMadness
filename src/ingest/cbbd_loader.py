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
