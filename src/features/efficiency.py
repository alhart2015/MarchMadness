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
