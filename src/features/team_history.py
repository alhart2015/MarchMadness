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


def _extract_seed_num(seed_str: str) -> int:
    """'W01' -> 1; 'X16a' -> 16; 'Y08' -> 8."""
    digits = "".join(ch for ch in seed_str if ch.isdigit())
    return int(digits)


def compute_per_seed_baseline(
    tourney_results: pd.DataFrame,
    seeds: pd.DataFrame,
    max_season: int,
) -> dict[int | str, float]:
    """Per-seed average rounds_won across all (Season, TeamID) rows
    with Season <= max_season.

    Returns: dict mapping seed_num (int) -> expected rounds_won. Includes
    a special key '__fallback__' = overall mean rounds_won, used by callers
    when a queried seed has 0 historical observations.

    Asserts no input tourney_results row has Season > max_season.
    """
    # Leak guard: fail loudly if input violates max_season constraint
    bad = tourney_results[tourney_results["Season"] > max_season]
    assert bad.empty, (
        f"Leak guard: {len(bad)} rows have Season > max_season "
        f"({max_season}) in compute_per_seed_baseline."
    )

    # Now we can safely pre-filter and compute rounds_won
    rounds_won = _rounds_won_per_team_season(
        tourney_results[tourney_results["Season"] <= max_season],
        max_season=max_season,
    )
    seeds_in_window = seeds[seeds["Season"] <= max_season].copy()
    seeds_in_window["seed_num"] = seeds_in_window["Seed"].apply(_extract_seed_num)
    joined = rounds_won.merge(
        seeds_in_window[["Season", "TeamID", "seed_num"]],
        on=["Season", "TeamID"],
        how="left",
    )
    # Drop team-seasons without a seed (shouldn't happen for tournament rows,
    # but defensive).
    joined = joined.dropna(subset=["seed_num"])
    joined["seed_num"] = joined["seed_num"].astype(int)

    baseline: dict[int | str, float] = {}
    for seed in range(1, 17):
        sub = joined[joined["seed_num"] == seed]
        if len(sub) > 0:
            baseline[seed] = float(sub["rounds_won"].mean())
    baseline["__fallback__"] = float(joined["rounds_won"].mean()) if len(joined) > 0 else 0.0
    return baseline


def shrunk_mean(residuals: list[float], k: int = 3) -> float:
    """Bayesian shrinkage of mean(residuals) toward 0 with k pseudo-obs.

    For empty input, returns 0.0 ("no evidence").
    For n observations: (sum(residuals) + k * 0) / (n + k).
    """
    n = len(residuals)
    if n == 0:
        return 0.0
    return float(sum(residuals)) / (n + k)


def shrunk_ewma(
    residuals_with_age: list[tuple[int, float]],
    half_life: float = 2.0,
    k: int = 3,
) -> float:
    """Bayesian-shrunk exponentially-weighted mean of residuals.

    residuals_with_age: list of (years_ago, residual) tuples. years_ago=1
    is the most recent prior season; larger = more remote.

    Weights: w(a) = 0.5 ** ((a - 1) / half_life). w(1) = 1.0.

    Computes weighted_mean = sum(w * r) / sum(w), then applies n-based
    shrinkage: (n * weighted_mean + k * 0) / (n + k). This decouples
    "which residuals matter" (EWMA weights) from "how confident we are
    in the estimate" (raw n).

    Returns 0.0 for empty input.
    """
    n = len(residuals_with_age)
    if n == 0:
        return 0.0
    weights = [0.5 ** ((a - 1) / half_life) for (a, _) in residuals_with_age]
    weight_sum = sum(weights)
    if weight_sum == 0:
        return 0.0
    weighted_mean = sum(
        w * r for (w, (_, r)) in zip(weights, residuals_with_age)
    ) / weight_sum
    return float(n * weighted_mean) / (n + k)
