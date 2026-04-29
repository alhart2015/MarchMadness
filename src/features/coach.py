"""Coach tournament-history features.

For each (Season, TeamID) in the tournament field, identifies the head coach
and computes their career tournament stats AS-OF Season-1 (no leakage):
  - coach_career_games:    cumulative tournament games coached
  - coach_career_wins:     cumulative tournament wins
  - coach_career_winpct:   wins / max(games, 1)
  - coach_career_f4_apps:  cumulative F4-or-better appearances (rounds_won >= 4)
  - coach_career_champs:   cumulative championships
  - coach_career_seasons:  cumulative tournament appearances (distinct seasons)

Inputs:
  team_coaches:    pd.DataFrame from MTeamCoaches.csv with Season, TeamID,
                   FirstDayNum, LastDayNum, CoachName.
  tourney_results: pd.DataFrame from MNCAATourneyCompactResults.csv with
                   Season, DayNum, WTeamID, LTeamID.

Notes:
  Tournament rounds are derived from DayNum:
    134-135 -> First Four (round 0; not counted in F4 apps but counts as a game)
    136-137 -> R64 (round 1)
    138-139 -> R32 (round 2)
    143-144 -> S16 (round 3)
    145-146 -> E8  (round 4)
    152     -> F4  (round 5)
    154     -> Champ (round 6)
  A coach's "F4 apps" = seasons with rounds_won >= 4 (reached the F4).
"""
import pandas as pd


DAY_TO_ROUND = [
    (134, 135, 0),  # First Four
    (136, 137, 1),  # R64
    (138, 139, 2),  # R32
    (143, 144, 3),  # S16
    (145, 146, 4),  # E8
    (152, 152, 5),  # F4
    (154, 154, 6),  # Champ
]


def day_to_round(day):
    for lo, hi, r in DAY_TO_ROUND:
        if lo <= day <= hi:
            return r
    return None


def coach_for_team_season(team_coaches, season, team_id):
    """Return the post-season coach for (season, team_id), or None."""
    rows = team_coaches[(team_coaches["Season"] == season) &
                         (team_coaches["TeamID"] == team_id)]
    if rows.empty:
        return None
    return rows.sort_values("LastDayNum").iloc[-1]["CoachName"]


def compute_coach_features(
    tourney_results: pd.DataFrame,
    team_coaches: pd.DataFrame,
) -> pd.DataFrame:
    """Return a DataFrame with one row per (Season, TeamID) in the tournament,
    with coach career stats AS-OF the start of that Season."""
    # Build per-game records: (season, team_id, won, round)
    records = []
    for _, g in tourney_results.iterrows():
        season = int(g["Season"])
        day = int(g["DayNum"])
        rnd = day_to_round(day)
        if rnd is None:
            continue
        records.append({"Season": season, "TeamID": int(g["WTeamID"]),
                        "won": 1, "round": rnd})
        records.append({"Season": season, "TeamID": int(g["LTeamID"]),
                        "won": 0, "round": rnd})

    games_df = pd.DataFrame(records)

    # Attach coach name to each game row.
    coach_lookup = team_coaches.sort_values("LastDayNum").groupby(
        ["Season", "TeamID"]).tail(1)[["Season", "TeamID", "CoachName"]]
    games_df = games_df.merge(coach_lookup, on=["Season", "TeamID"], how="left")
    games_df = games_df[games_df["CoachName"].notna()].copy()

    # Per-coach per-season aggregates.
    # max_won_round = the latest round in which the coach's team WON a game.
    games_df["round_if_won"] = games_df["round"].where(games_df["won"] == 1)
    season_agg = games_df.groupby(["CoachName", "Season"]).agg(
        games=("won", "size"),
        wins=("won", "sum"),
        max_won_round=("round_if_won", "max"),
    ).reset_index()
    # F4 appearance = won an E8 game (so reached F4) -> max_won_round >= 4.
    season_agg["f4_app"] = (season_agg["max_won_round"].fillna(0) >= 4).astype(int)
    # Champion = won a round-6 game (the championship final).
    season_agg["champ"] = (season_agg["max_won_round"].fillna(0) >= 6).astype(int)

    # Cumulative-through-prior-season: shift by one season per coach.
    season_agg = season_agg.sort_values(["CoachName", "Season"]).reset_index(drop=True)
    cum_cols = ["games", "wins", "f4_app", "champ"]
    for c in cum_cols:
        season_agg[f"cum_{c}"] = (
            season_agg.groupby("CoachName")[c]
            .cumsum()
            .shift(1)
            .fillna(0)
            .astype(int)
        )
    # Cumulative count of distinct prior tournament-appearance seasons.
    season_agg["cum_seasons"] = (
        season_agg.groupby("CoachName").cumcount()
    )

    # Now produce one row per (Season, TeamID) in the tournament field, with
    # the coach's cumulative-through-prior-year stats.
    field = games_df.drop_duplicates(["Season", "TeamID", "CoachName"])[
        ["Season", "TeamID", "CoachName"]
    ]
    out = field.merge(
        season_agg[["CoachName", "Season", "cum_games", "cum_wins",
                     "cum_f4_app", "cum_champ", "cum_seasons"]],
        on=["CoachName", "Season"], how="left",
    )
    out["coach_career_games"] = out["cum_games"].fillna(0).astype(int)
    out["coach_career_wins"] = out["cum_wins"].fillna(0).astype(int)
    out["coach_career_winpct"] = out["coach_career_wins"] / out["coach_career_games"].clip(lower=1)
    out["coach_career_f4_apps"] = out["cum_f4_app"].fillna(0).astype(int)
    out["coach_career_champs"] = out["cum_champ"].fillna(0).astype(int)
    out["coach_career_seasons"] = out["cum_seasons"].fillna(0).astype(int)

    return out[["Season", "TeamID",
                "coach_career_games", "coach_career_wins",
                "coach_career_winpct", "coach_career_f4_apps",
                "coach_career_champs", "coach_career_seasons"]]
