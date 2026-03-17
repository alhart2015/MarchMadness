"""Parse Tournament Matchups.csv into paired game results (winner / loser per game).

The source file has one row per team per game.  Rows are ordered so that every
two consecutive rows (sorted descending by BY YEAR NO) share the same
CURRENT ROUND value and represent the two opponents in that game.  The team
with the higher SCORE wins.
"""

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def build_tournament_results(matchups: pd.DataFrame) -> pd.DataFrame:
    """Convert per-team-per-game rows into per-game winner/loser rows.

    Parameters
    ----------
    matchups : DataFrame
        The raw Tournament Matchups.csv DataFrame as loaded by
        ``load_kaggle2026_data``.

    Returns
    -------
    DataFrame with columns:
        Season, WTeamID, WTeam, WSeed, WScore, LTeamID, LTeam, LSeed, LScore, Round
    where *Season* maps to the YEAR column and *Round* maps to CURRENT ROUND
    (the round in which this game was played, e.g. 64 = first round).
    """
    if matchups.empty:
        return pd.DataFrame()

    required = {"YEAR", "BY YEAR NO", "TEAM NO", "TEAM", "SEED", "SCORE", "CURRENT ROUND"}
    missing = required - set(matchups.columns)
    if missing:
        raise ValueError(f"Tournament Matchups missing columns: {missing}")

    # Sort descending so consecutive pairs are opponents (matches the raw layout)
    df = matchups.sort_values(["YEAR", "BY YEAR NO"], ascending=[True, False]).copy()

    records = []
    for year, group in df.groupby("YEAR"):
        rows = group.reset_index(drop=True)
        if len(rows) % 2 != 0:
            logger.warning("Year %d has odd number of matchup rows (%d); last row dropped", year, len(rows))
            rows = rows.iloc[:-1]

        for i in range(0, len(rows), 2):
            a = rows.iloc[i]
            b = rows.iloc[i + 1]

            # Sanity check: both rows should be in the same round
            if a["CURRENT ROUND"] != b["CURRENT ROUND"]:
                logger.warning(
                    "Year %d row %d: CURRENT ROUND mismatch (%s vs %s), skipping",
                    year, i, a["CURRENT ROUND"], b["CURRENT ROUND"],
                )
                continue

            # Determine winner by score
            if a["SCORE"] > b["SCORE"]:
                winner, loser = a, b
            elif b["SCORE"] > a["SCORE"]:
                winner, loser = b, a
            else:
                # Overtime tie edge case — skip or treat first team as winner
                logger.warning(
                    "Year %d: tied score (%s vs %s) in round %s, skipping",
                    year, a["TEAM"], b["TEAM"], a["CURRENT ROUND"],
                )
                continue

            records.append({
                "Season": int(year),
                "WTeamID": int(winner["TEAM NO"]),
                "WTeam": winner["TEAM"],
                "WSeed": int(winner["SEED"]),
                "WScore": int(winner["SCORE"]),
                "LTeamID": int(loser["TEAM NO"]),
                "LTeam": loser["TEAM"],
                "LSeed": int(loser["SEED"]),
                "LScore": int(loser["SCORE"]),
                "Round": int(a["CURRENT ROUND"]),
            })

    results = pd.DataFrame(records)
    logger.info(
        "Built %d tournament games across %d seasons",
        len(results),
        results["Season"].nunique() if not results.empty else 0,
    )
    return results
