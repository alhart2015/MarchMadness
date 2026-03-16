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
