"""Team-program tournament history features.

Two new XGB features for v4:
  - team_seed_residual_mean_10yr: shrunk mean of per-seed residuals
    over prior-10-year window. Captures program continuity (Duke / UNC /
    Kansas / UConn).
  - team_seed_residual_ewma_hl2: shrunk EWMA at HL=2 over the same
    window. Captures program emergence/momentum (Bennett-Virginia,
    Drew-Baylor) and recent form.

Both are keyed on TeamID (NOT coach, unlike coach_career_*), filling
the program-DNA gap in v4's feature stack. See
docs/superpowers/specs/2026-05-09-team-seed-residual-design.md.
"""
from __future__ import annotations

import pandas as pd

from src.features.coach import day_to_round


def _rounds_won_per_team_season(
    tourney_results: pd.DataFrame,
    max_season: int | None,
) -> pd.DataFrame:
    """Return a DataFrame [(Season, TeamID, rounds_won)] for every team
    that appeared in any tournament in tourney_results.

    rounds_won = number of games the team WON in that tournament. Champion
    = 6 wins (R64 + R32 + S16 + E8 + F4 + Champ). R64 loser = 0 wins.
    First Four wins (DayNum 134-135) count toward rounds_won; this
    matches src/features/coach.py's convention.

    If max_season is provided, asserts no input row has Season > max_season.
    """
    if max_season is not None:
        bad = tourney_results[tourney_results["Season"] > max_season]
        assert bad.empty, (
            f"Leak guard: {len(bad)} rows have Season > max_season "
            f"({max_season}). team_history features must be computed "
            f"on a leak-free tourney_results subset."
        )

    # Per-game records: (season, team_id, won, round)
    rows = []
    for _, g in tourney_results.iterrows():
        rnd = day_to_round(int(g["DayNum"]))
        if rnd is None:
            continue
        season = int(g["Season"])
        rows.append({"Season": season, "TeamID": int(g["WTeamID"]), "won": 1})
        rows.append({"Season": season, "TeamID": int(g["LTeamID"]), "won": 0})
    if not rows:
        return pd.DataFrame(columns=["Season", "TeamID", "rounds_won"])
    df = pd.DataFrame(rows)
    out = df.groupby(["Season", "TeamID"])["won"].sum().reset_index()
    out = out.rename(columns={"won": "rounds_won"})
    out["rounds_won"] = out["rounds_won"].astype(int)
    return out
