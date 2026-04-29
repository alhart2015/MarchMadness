"""Quality-wins-vs-tournament-field features.

For each (Season, TeamID), counts regular-season games against teams that
*made the same season's tournament field*. The intuition is that beating
tournament-bound opponents is a stronger signal than generic SOS, which
averages over an entire schedule that may include many non-tournament teams.

Features per (Season, TeamID):
  qw_wins:    wins vs same-season tournament field
  qw_losses:  losses vs same-season tournament field
  qw_games:   wins + losses vs tournament field
  qw_winpct:  qw_wins / max(qw_games, 1); 0.5 if qw_games == 0
  qw_record_minus_overall_pct: qw_winpct minus the team's overall regular-season
                               win pct (positive = does better vs tournament
                               teams than against everyone else)

Inputs:
  reg_season:  pd.DataFrame from MRegularSeasonCompactResults.csv with
               Season, WTeamID, LTeamID.
  tourney_seeds: pd.DataFrame from MNCAATourneySeeds.csv with Season, TeamID.

Note on leakage:
  The "tournament field" for a year is known on Selection Sunday, which is
  the day before R64 tips off. For LOSO backtest and real bracket
  prediction, this is fair to use as a feature.
"""
import pandas as pd


def compute_quality_wins(
    reg_season: pd.DataFrame,
    tourney_seeds: pd.DataFrame,
) -> pd.DataFrame:
    """Return one row per (Season, TeamID) with quality-wins features."""
    field_by_season = (
        tourney_seeds.groupby("Season")["TeamID"]
        .apply(lambda s: set(s.astype(int)))
        .to_dict()
    )

    rs = reg_season[["Season", "WTeamID", "LTeamID"]].copy()
    rs["Season"] = rs["Season"].astype(int)
    rs["WTeamID"] = rs["WTeamID"].astype(int)
    rs["LTeamID"] = rs["LTeamID"].astype(int)

    # Tag each game with whether each team is in the field for that season.
    rs["w_in_field"] = rs.apply(
        lambda r: r["WTeamID"] in field_by_season.get(r["Season"], set()), axis=1
    )
    rs["l_in_field"] = rs.apply(
        lambda r: r["LTeamID"] in field_by_season.get(r["Season"], set()), axis=1
    )

    # For each season-team perspective, win/loss counts overall and vs field.
    win_rows = rs[["Season", "WTeamID", "l_in_field"]].rename(
        columns={"WTeamID": "TeamID", "l_in_field": "opp_in_field"}
    )
    win_rows["won"] = 1
    loss_rows = rs[["Season", "LTeamID", "w_in_field"]].rename(
        columns={"LTeamID": "TeamID", "w_in_field": "opp_in_field"}
    )
    loss_rows["won"] = 0

    games = pd.concat([win_rows, loss_rows], ignore_index=True)

    # Aggregate by (Season, TeamID).
    by_team = games.groupby(["Season", "TeamID"]).agg(
        total_games=("won", "size"),
        total_wins=("won", "sum"),
        qw_games=("opp_in_field", "sum"),
    ).reset_index()
    qw_wins = (
        games[games["opp_in_field"] & (games["won"] == 1)]
        .groupby(["Season", "TeamID"])
        .size()
        .rename("qw_wins")
        .reset_index()
    )
    out = by_team.merge(qw_wins, on=["Season", "TeamID"], how="left")
    out["qw_wins"] = out["qw_wins"].fillna(0).astype(int)
    out["qw_games"] = out["qw_games"].astype(int)
    out["qw_losses"] = out["qw_games"] - out["qw_wins"]
    out["qw_winpct"] = out.apply(
        lambda r: r["qw_wins"] / r["qw_games"] if r["qw_games"] > 0 else 0.5,
        axis=1,
    )
    out["overall_winpct"] = out["total_wins"] / out["total_games"].clip(lower=1)
    out["qw_record_minus_overall_pct"] = out["qw_winpct"] - out["overall_winpct"]

    return out[[
        "Season", "TeamID", "qw_wins", "qw_losses", "qw_games",
        "qw_winpct", "qw_record_minus_overall_pct",
    ]]
