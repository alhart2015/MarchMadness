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
