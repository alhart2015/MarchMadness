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
