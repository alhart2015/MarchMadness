"""Late-season and tournament-context features for model v3."""
import numpy as np
import pandas as pd

from src.features.four_factors import estimate_possessions


def compute_late_season_metrics(
    detailed_results: pd.DataFrame,
    season: int,
    top_n_teams: set[int],
    late_window_days: int = 30,
) -> pd.DataFrame:
    """Compute opponent-adjusted efficiency over late season vs quality opponents.

    Uses raw efficiency (not iterative adjustment) since the filtered sample
    is too small (3-8 games) for convergence.
    """
    df = detailed_results[detailed_results["Season"] == season].copy()
    if df.empty:
        return pd.DataFrame(columns=["TeamID", "Season", "late_adj_oe",
                                      "late_adj_de", "late_adj_em", "late_sos"])

    max_day = df["DayNum"].max()
    cutoff = max_day - late_window_days
    df = df[df["DayNum"] >= cutoff]

    if df.empty:
        return pd.DataFrame(columns=["TeamID", "Season", "late_adj_oe",
                                      "late_adj_de", "late_adj_em", "late_sos"])

    records = []
    for _, g in df.iterrows():
        w_id, l_id = int(g["WTeamID"]), int(g["LTeamID"])
        w_poss = estimate_possessions(g["WFGA"], g["WOR"], g["WTO"], g["WFTA"])
        l_poss = estimate_possessions(g["LFGA"], g["LOR"], g["LTO"], g["LFTA"])
        poss = (w_poss + l_poss) / 2
        if poss <= 0:
            continue

        w_oe = 100 * g["WScore"] / poss
        w_de = 100 * g["LScore"] / poss
        l_oe = 100 * g["LScore"] / poss
        l_de = 100 * g["WScore"] / poss

        if l_id in top_n_teams:
            records.append({"TeamID": w_id, "opp_id": l_id, "oe": w_oe, "de": w_de})
        if w_id in top_n_teams:
            records.append({"TeamID": l_id, "opp_id": w_id, "oe": l_oe, "de": l_de})

    if not records:
        return pd.DataFrame(columns=["TeamID", "Season", "late_adj_oe",
                                      "late_adj_de", "late_adj_em", "late_sos"])

    rec_df = pd.DataFrame(records)

    rows = []
    for tid, grp in rec_df.groupby("TeamID"):
        if len(grp) < 1:
            continue
        rows.append({
            "TeamID": int(tid),
            "Season": season,
            "late_adj_oe": grp["oe"].mean(),
            "late_adj_de": grp["de"].mean(),
            "late_adj_em": grp["oe"].mean() - grp["de"].mean(),
            "late_sos": grp["opp_id"].nunique(),
        })

    return pd.DataFrame(rows)


def compute_trajectory_features(
    detailed_results: pd.DataFrame,
    season: int,
    trajectory_window_days: int = 45,
) -> pd.DataFrame:
    """Compute linear trend slopes of efficiency and margin over late season."""
    df = detailed_results[detailed_results["Season"] == season].copy()
    if df.empty:
        return pd.DataFrame(columns=["TeamID", "Season",
                                      "efficiency_trend", "margin_trend"])

    max_day = df["DayNum"].max()
    cutoff = max_day - trajectory_window_days
    df = df[df["DayNum"] >= cutoff]

    records = []
    for _, g in df.iterrows():
        w_poss = estimate_possessions(g["WFGA"], g["WOR"], g["WTO"], g["WFTA"])
        l_poss = estimate_possessions(g["LFGA"], g["LOR"], g["LTO"], g["LFTA"])
        poss = (w_poss + l_poss) / 2
        if poss <= 0:
            continue

        w_em = 100 * (g["WScore"] - g["LScore"]) / poss
        l_em = -w_em
        w_margin = g["WScore"] - g["LScore"]
        l_margin = -w_margin

        records.append({"TeamID": int(g["WTeamID"]), "DayNum": g["DayNum"],
                        "em": w_em, "margin": w_margin})
        records.append({"TeamID": int(g["LTeamID"]), "DayNum": g["DayNum"],
                        "em": l_em, "margin": l_margin})

    if not records:
        return pd.DataFrame(columns=["TeamID", "Season",
                                      "efficiency_trend", "margin_trend"])

    rec_df = pd.DataFrame(records)

    rows = []
    for tid, grp in rec_df.groupby("TeamID"):
        if len(grp) < 3:
            continue
        days = grp["DayNum"].values.astype(float)
        em_slope = np.polyfit(days, grp["em"].values, 1)[0]
        margin_slope = np.polyfit(days, grp["margin"].values, 1)[0]
        rows.append({
            "TeamID": int(tid),
            "Season": season,
            "efficiency_trend": em_slope,
            "margin_trend": margin_slope,
        })

    return pd.DataFrame(rows)
