"""Generate Kaggle March Machine Learning Mania 2026 submission files.

Produces two CSV files:
  - output/submission_stage1.csv  (seasons 2022-2025, ~519k rows)
  - output/submission_stage2.csv  (season 2026, ~132k rows)

Both men's (IDs 1101-1481) and women's (IDs 3101-3481) predictions.

Strategy:
  - Men's: adjusted efficiency + four factors + rolling stats + Massey ordinals
    + KenPom/Barttorvik features -> XGBoost trained on tournament history
  - Women's: adjusted efficiency + four factors + rolling stats -> XGBoost
    trained on women's tournament history

Usage:
    python src/kaggle_submission.py
"""

import re
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ── Path setup ───────────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent          # src/
_ROOT = _HERE.parent                             # project root
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

MANIA_DIR  = _ROOT / "data" / "raw" / "march-machine-learning-2026"
KAGGLE_DIR = _ROOT / "data" / "raw" / "kaggle"
OUTPUT_DIR = _ROOT / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MASSEY_SYSTEMS = ["POM", "SAG", "MOR", "WOL", "DOL", "COL", "RPI"]


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _parse_seed_number(seed_str: str) -> int:
    match = re.search(r"(\d+)", str(seed_str))
    return int(match.group(1)) if match else 16


def estimate_possessions(fga, off_reb, to, fta):
    return fga - off_reb + to + 0.475 * fta


def train_xgb_model(X, y, random_seed=42):
    """Train XGBoost with Platt calibration."""
    params = {
        "n_estimators": 300,
        "max_depth": 4,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": random_seed,
        "eval_metric": "logloss",
    }
    base_model = xgb.XGBClassifier(**params)
    calibrated = CalibratedClassifierCV(
        base_model,
        method="sigmoid",
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=random_seed),
    )
    calibrated.fit(X, y)
    return calibrated


# ═══════════════════════════════════════════════════════════════════════════════
# FEATURE COMPUTATION (generic: works for both men's and women's)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_adjusted_efficiency_fast(df_season):
    """Compute adjusted efficiency for all teams in a single season's games.

    Replicates the logic from src/features/efficiency.py but takes a
    pre-filtered DataFrame (single season) to avoid re-filtering.
    """
    df = df_season.copy()
    if len(df) == 0:
        return pd.DataFrame(columns=["TeamID", "adj_oe", "adj_de", "adj_em", "adj_tempo"])

    # Estimate possessions
    df["w_poss"] = estimate_possessions(df["WFGA"], df["WOR"], df["WTO"], df["WFTA"])
    df["l_poss"] = estimate_possessions(df["LFGA"], df["LOR"], df["LTO"], df["LFTA"])
    df["possessions"] = (df["w_poss"] + df["l_poss"]) / 2

    # Build symmetric game records
    winner_rows = pd.DataFrame({
        "TeamID": df["WTeamID"].values,
        "OppID": df["LTeamID"].values,
        "points_scored": df["WScore"].values,
        "points_allowed": df["LScore"].values,
        "possessions": df["possessions"].values,
        "home": (df["WLoc"] == "H").astype(int).values,
        "day_num": df["DayNum"].values,
    })
    loser_rows = pd.DataFrame({
        "TeamID": df["LTeamID"].values,
        "OppID": df["WTeamID"].values,
        "points_scored": df["LScore"].values,
        "points_allowed": df["WScore"].values,
        "possessions": df["possessions"].values,
        "home": (df["WLoc"] == "A").astype(int).values,
        "day_num": df["DayNum"].values,
    })
    games = pd.concat([winner_rows, loser_rows], ignore_index=True)

    # Per-game efficiency
    games["oe"] = 100 * games["points_scored"] / games["possessions"]
    games["de"] = 100 * games["points_allowed"] / games["possessions"]
    games["tempo"] = games["possessions"]

    # Recency weights
    max_day = games["day_num"].max()
    decay_rate = np.log(2) / 30.0
    games["weight"] = np.exp(-decay_rate * (max_day - games["day_num"]))

    teams = games["TeamID"].unique()
    team_oe = games.groupby("TeamID")["oe"].mean().to_dict()
    team_de = games.groupby("TeamID")["de"].mean().to_dict()
    league_avg_oe = games["oe"].mean()
    league_avg_de = games["de"].mean()

    hca = 3.5
    for _iteration in range(10):
        games["opp_de_rating"] = games["OppID"].map(team_de).fillna(league_avg_de)
        games["opp_oe_rating"] = games["OppID"].map(team_oe).fillna(league_avg_oe)

        home_adj = games["home"] * hca / 2
        games["adj_oe_game"] = games["oe"] * (league_avg_de / games["opp_de_rating"]) - home_adj
        games["adj_de_game"] = games["de"] * (league_avg_oe / games["opp_oe_rating"]) + home_adj

        for team in teams:
            mask = games["TeamID"] == team
            w = games.loc[mask, "weight"]
            if w.sum() > 0:
                team_oe[team] = np.average(games.loc[mask, "adj_oe_game"], weights=w)
                team_de[team] = np.average(games.loc[mask, "adj_de_game"], weights=w)

    # Tempo
    team_tempo = {}
    league_avg_tempo = games["tempo"].mean()
    raw_tempo_means = games.groupby("TeamID")["tempo"].mean().to_dict()
    for team in teams:
        mask = games["TeamID"] == team
        opp_tempos = games.loc[mask, "OppID"].map(raw_tempo_means).fillna(league_avg_tempo)
        raw_tempos = games.loc[mask, "tempo"]
        w = games.loc[mask, "weight"]
        if w.sum() > 0:
            team_tempo[team] = np.average(
                raw_tempos * (league_avg_tempo / opp_tempos), weights=w
            )
        else:
            team_tempo[team] = league_avg_tempo

    return pd.DataFrame({
        "TeamID": list(teams),
        "adj_oe": [team_oe[t] for t in teams],
        "adj_de": [team_de[t] for t in teams],
        "adj_em": [team_oe[t] - team_de[t] for t in teams],
        "adj_tempo": [team_tempo[t] for t in teams],
    })


def compute_four_factors_fast(df_season):
    """Compute season-level four factors for all teams."""
    df = df_season.copy()
    if len(df) == 0:
        return pd.DataFrame(columns=[
            "TeamID", "off_efg", "off_to_rate", "off_or_rate", "off_ft_rate",
            "def_efg", "def_to_rate", "def_or_rate", "def_ft_rate",
        ])

    winner_off = pd.DataFrame({
        "TeamID": df["WTeamID"], "FGM": df["WFGM"], "FGA": df["WFGA"],
        "FGM3": df["WFGM3"], "FTM": df["WFTM"], "FTA": df["WFTA"],
        "OR": df["WOR"], "TO": df["WTO"], "opp_DR": df["LDR"],
        "opp_FGM": df["LFGM"], "opp_FGA": df["LFGA"],
        "opp_FGM3": df["LFGM3"], "opp_FTM": df["LFTM"],
        "opp_FTA": df["LFTA"], "opp_OR": df["LOR"], "opp_TO": df["LTO"],
        "DR": df["WDR"],
    })
    loser_off = pd.DataFrame({
        "TeamID": df["LTeamID"], "FGM": df["LFGM"], "FGA": df["LFGA"],
        "FGM3": df["LFGM3"], "FTM": df["LFTM"], "FTA": df["LFTA"],
        "OR": df["LOR"], "TO": df["LTO"], "opp_DR": df["WDR"],
        "opp_FGM": df["WFGM"], "opp_FGA": df["WFGA"],
        "opp_FGM3": df["WFGM3"], "opp_FTM": df["WFTM"],
        "opp_FTA": df["WFTA"], "opp_OR": df["WOR"], "opp_TO": df["WTO"],
        "DR": df["LDR"],
    })
    all_games = pd.concat([winner_off, loser_off], ignore_index=True)
    agg = all_games.groupby("TeamID").sum()

    agg["off_efg"] = (agg["FGM"] + 0.5 * agg["FGM3"]) / agg["FGA"]
    agg["off_to_rate"] = agg["TO"] / (agg["FGA"] + 0.475 * agg["FTA"] + agg["TO"])
    agg["off_or_rate"] = agg["OR"] / (agg["OR"] + agg["opp_DR"])
    agg["off_ft_rate"] = agg["FTM"] / agg["FGA"]

    agg["def_efg"] = (agg["opp_FGM"] + 0.5 * agg["opp_FGM3"]) / agg["opp_FGA"]
    agg["def_to_rate"] = agg["opp_TO"] / (agg["opp_FGA"] + 0.475 * agg["opp_FTA"] + agg["opp_TO"])
    agg["def_or_rate"] = agg["opp_OR"] / (agg["opp_OR"] + agg["DR"])
    agg["def_ft_rate"] = agg["opp_FTM"] / agg["opp_FGA"]

    result_cols = [
        "off_efg", "off_to_rate", "off_or_rate", "off_ft_rate",
        "def_efg", "def_to_rate", "def_or_rate", "def_ft_rate",
    ]
    return agg[result_cols].reset_index()


def compute_rolling_and_form(df_season):
    """Compute rolling efficiency (last 30 days) and form features."""
    if len(df_season) == 0:
        return {}

    max_day = df_season["DayNum"].max()
    recent_games = df_season[df_season["DayNum"] > max_day - 30]

    # Rolling efficiency
    recent_rows = []
    for _, g in recent_games.iterrows():
        w_poss = estimate_possessions(g["WFGA"], g["WOR"], g["WTO"], g["WFTA"])
        l_poss = estimate_possessions(g["LFGA"], g["LOR"], g["LTO"], g["LFTA"])
        poss = (w_poss + l_poss) / 2
        if poss > 0:
            recent_rows.append({
                "TeamID": g["WTeamID"], "pts": g["WScore"],
                "pts_allowed": g["LScore"], "poss": poss, "win": 1,
            })
            recent_rows.append({
                "TeamID": g["LTeamID"], "pts": g["LScore"],
                "pts_allowed": g["WScore"], "poss": poss, "win": 0,
            })

    rolling_eff = {}
    if recent_rows:
        recent_df = pd.DataFrame(recent_rows)
        for tid, grp in recent_df.groupby("TeamID"):
            rolling_eff[tid] = {
                "rolling_oe": 100 * grp["pts"].sum() / grp["poss"].sum(),
                "rolling_de": 100 * grp["pts_allowed"].sum() / grp["poss"].sum(),
                "win_pct_30d": grp["win"].mean(),
            }

    # All-season form features
    w_games = df_season[["WTeamID", "WScore", "LScore", "DayNum"]].rename(
        columns={"WTeamID": "TeamID", "WScore": "pts", "LScore": "pts_allowed"}
    ).assign(win=1)
    l_games = df_season[["LTeamID", "LScore", "WScore", "DayNum"]].rename(
        columns={"LTeamID": "TeamID", "LScore": "pts", "WScore": "pts_allowed"}
    ).assign(win=0)
    all_games = pd.concat([w_games, l_games]).sort_values("DayNum")

    form_features = {}
    for tid, grp in all_games.groupby("TeamID"):
        last10 = grp.tail(10)
        season_avg_pts = grp["pts"].mean()
        last10_avg_pts = last10["pts"].mean()
        form_features[tid] = {
            "win_pct_last10": last10["win"].mean(),
            "avg_mov_last10": (last10["pts"] - last10["pts_allowed"]).mean(),
            "scoring_trend": last10_avg_pts - season_avg_pts,
            "season_win_pct": grp["win"].mean(),
            "season_avg_mov": (grp["pts"] - grp["pts_allowed"]).mean(),
        }

    # Merge rolling and form
    combined = {}
    all_tids = set(rolling_eff.keys()) | set(form_features.keys())
    for tid in all_tids:
        combined[tid] = {}
        if tid in rolling_eff:
            combined[tid].update(rolling_eff[tid])
        if tid in form_features:
            combined[tid].update(form_features[tid])
    return combined


def compute_conf_strength(conferences_season, eff):
    """Compute conference average adj_em."""
    if conferences_season.empty or eff.empty:
        return {}
    merged = conferences_season.merge(eff[["TeamID", "adj_em"]], on="TeamID", how="left")
    conf_avg = merged.groupby("ConfAbbrev")["adj_em"].mean()
    team_conf = dict(zip(conferences_season["TeamID"], conferences_season["ConfAbbrev"]))
    result = {}
    for tid, conf_abbrev in team_conf.items():
        if conf_abbrev in conf_avg.index:
            result[tid] = conf_avg[conf_abbrev]
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# BUILD FEATURE MATRIX FOR ALL D1 TEAMS
# ═══════════════════════════════════════════════════════════════════════════════

def build_all_team_features(
    reg_season, seeds, conferences, seasons,
    massey=None, kenpom=None, kp_to_kaggle=None,
    massey_systems=None, kp_cols=None, gender="M",
):
    """Build feature matrix covering ALL D1 teams for given seasons.

    Returns DataFrame with one row per (TeamID, Season).
    """
    if massey_systems is None:
        massey_systems = MASSEY_SYSTEMS
    if kp_cols is None:
        kp_cols = [
            "KADJ EM", "KADJ O", "KADJ D", "BARTHAG", "TALENT", "EXP",
            "ELITE SOS", "WAB", "EFG%", "EFG%D", "TOV%", "TOV%D",
            "OREB%", "DREB%", "FTR", "FTRD", "K TEMPO",
        ]

    all_rows = []

    for season in seasons:
        t0 = time.time()
        season_reg = reg_season[reg_season["Season"] == season]
        if len(season_reg) == 0:
            continue

        # 1. Adjusted efficiency
        try:
            eff = compute_adjusted_efficiency_fast(season_reg)
        except Exception as e:
            print(f"    WARNING: Efficiency failed for {gender} {season}: {e}")
            eff = pd.DataFrame(columns=["TeamID", "adj_oe", "adj_de", "adj_em", "adj_tempo"])

        # 2. Four factors
        try:
            ff = compute_four_factors_fast(season_reg)
        except Exception as e:
            print(f"    WARNING: Four factors failed for {gender} {season}: {e}")
            ff = pd.DataFrame(columns=["TeamID"])

        # 3. Rolling + form features
        try:
            form_dict = compute_rolling_and_form(season_reg)
        except Exception:
            form_dict = {}

        # 4. Massey ordinals (men only)
        massey_features = {}
        if massey is not None and not massey.empty:
            season_massey = massey[massey["Season"] == season]
            if not season_massey.empty:
                for system in massey_systems:
                    sys_ranks = season_massey[season_massey["SystemName"] == system]
                    for _, row in sys_ranks.iterrows():
                        tid = int(row["TeamID"])
                        if tid not in massey_features:
                            massey_features[tid] = {}
                        massey_features[tid][f"massey_{system}"] = row["OrdinalRank"]
                for tid in massey_features:
                    ranks = [v for k, v in massey_features[tid].items() if k.startswith("massey_")]
                    if ranks:
                        massey_features[tid]["massey_composite"] = np.mean(ranks)

        # 5. Conference strength
        season_conf = conferences[conferences["Season"] == season] if conferences is not None else pd.DataFrame()
        conf_strength = compute_conf_strength(season_conf, eff)

        # 6. Seeds
        season_seeds = seeds[seeds["Season"] == season]
        seed_map = {}
        for _, row in season_seeds.iterrows():
            seed_map[int(row["TeamID"])] = _parse_seed_number(row["Seed"])

        # 7. KenPom/Barttorvik (men only)
        kp_features = {}
        if kenpom is not None and kp_to_kaggle is not None:
            kp_season = kenpom[kenpom["YEAR"] == season]
            for _, row in kp_season.iterrows():
                kp_id = int(row["TEAM NO"])
                kaggle_id = kp_to_kaggle.get(kp_id)
                if kaggle_id is None:
                    continue
                feats = {}
                for col in kp_cols:
                    if col in row.index and pd.notna(row[col]):
                        feats[f"kp_{col}"] = row[col]
                kp_features[kaggle_id] = feats

        # Collect all team IDs that appeared this season
        all_team_ids = set()
        all_team_ids.update(eff["TeamID"].values if not eff.empty else [])
        all_team_ids.update(ff["TeamID"].values if not ff.empty and "TeamID" in ff.columns else [])
        all_team_ids.update(form_dict.keys())
        all_team_ids.update(seed_map.keys())
        all_team_ids.update(massey_features.keys())
        all_team_ids.update(kp_features.keys())

        for tid in all_team_ids:
            row_data = {"TeamID": int(tid), "Season": season}

            # Seed
            if tid in seed_map:
                row_data["seed"] = seed_map[tid]

            # Adjusted efficiency
            if not eff.empty:
                eff_row = eff[eff["TeamID"] == tid]
                if not eff_row.empty:
                    for col in ["adj_oe", "adj_de", "adj_em", "adj_tempo"]:
                        row_data[col] = eff_row.iloc[0][col]

            # Four factors
            if not ff.empty and "TeamID" in ff.columns:
                ff_row = ff[ff["TeamID"] == tid]
                if not ff_row.empty:
                    for col in ff.columns:
                        if col != "TeamID":
                            row_data[col] = ff_row.iloc[0][col]

            # Rolling + form
            if tid in form_dict:
                row_data.update(form_dict[tid])

            # Massey
            if tid in massey_features:
                row_data.update(massey_features[tid])

            # Conference strength
            if tid in conf_strength:
                row_data["conf_strength"] = conf_strength[tid]

            # KenPom
            if tid in kp_features:
                row_data.update(kp_features[tid])

            all_rows.append(row_data)

        elapsed = time.time() - t0
        print(f"    {gender} Season {season}: {sum(1 for r in all_rows if r['Season']==season)} teams ({elapsed:.1f}s)")

    return pd.DataFrame(all_rows)


# ═══════════════════════════════════════════════════════════════════════════════
# BUILD TRAINING DATA FROM TOURNAMENT RESULTS
# ═══════════════════════════════════════════════════════════════════════════════

def build_matchup_training_data(feature_matrix, tourney_results, feature_cols):
    """Build symmetric matchup training data from tournament results.

    For each game: winner_features - loser_features -> label 1
                   loser_features - winner_features -> label 0
    """
    # Build lookup: (TeamID, Season) -> feature vector
    fm_indexed = feature_matrix.set_index(["TeamID", "Season"])

    rows = []
    labels = []

    for _, game in tourney_results.iterrows():
        season = game["Season"]
        w_id = game["WTeamID"]
        l_id = game["LTeamID"]

        try:
            w_feats = fm_indexed.loc[(w_id, season), feature_cols].values.astype(float)
            l_feats = fm_indexed.loc[(l_id, season), feature_cols].values.astype(float)
        except KeyError:
            continue

        rows.append(w_feats - l_feats)
        labels.append(1)
        rows.append(l_feats - w_feats)
        labels.append(0)

    if not rows:
        return pd.DataFrame(columns=feature_cols), pd.Series(dtype=float)

    X = pd.DataFrame(rows, columns=feature_cols)
    y = pd.Series(labels, name="win")
    return X, y


def get_feature_cols(fm):
    """Return list of numeric feature columns."""
    exclude = {"TeamID", "Season", "seed"}
    cols = [c for c in fm.columns if c not in exclude]
    numeric_cols = fm[cols].select_dtypes(include="number").columns.tolist()
    return numeric_cols


# ═══════════════════════════════════════════════════════════════════════════════
# PREDICTION
# ═══════════════════════════════════════════════════════════════════════════════

def predict_all_pairs(submission_df, feature_matrix, feature_cols, model, gender_filter=None):
    """Fill in predictions for all ID rows matching the gender filter.

    gender_filter: 'M' for men (TeamA < 3000), 'W' for women, None for all
    """
    # Parse IDs
    parsed = submission_df["ID"].str.split("_", expand=True)
    parsed.columns = ["Season", "TeamA", "TeamB"]
    parsed["Season"] = parsed["Season"].astype(int)
    parsed["TeamA"] = parsed["TeamA"].astype(int)
    parsed["TeamB"] = parsed["TeamB"].astype(int)

    # Filter by gender
    if gender_filter == "M":
        mask = parsed["TeamA"] < 3000
    elif gender_filter == "W":
        mask = parsed["TeamA"] >= 3000
    else:
        mask = pd.Series(True, index=parsed.index)

    indices = parsed[mask].index
    if len(indices) == 0:
        return submission_df

    # Build lookup
    fm_indexed = {}
    for _, row in feature_matrix.iterrows():
        key = (int(row["TeamID"]), int(row["Season"]))
        fm_indexed[key] = row[feature_cols].values.astype(float)

    # Batch prediction: collect all feature diffs, then predict in one go
    batch_indices = []
    batch_diffs = []
    fallback_indices = []

    for idx in indices:
        season = parsed.loc[idx, "Season"]
        team_a = parsed.loc[idx, "TeamA"]
        team_b = parsed.loc[idx, "TeamB"]

        a_feats = fm_indexed.get((team_a, season))
        b_feats = fm_indexed.get((team_b, season))

        if a_feats is not None and b_feats is not None:
            batch_indices.append(idx)
            batch_diffs.append(a_feats - b_feats)
        else:
            fallback_indices.append(idx)

    # Batch predict
    if batch_diffs:
        X_batch = pd.DataFrame(batch_diffs, columns=feature_cols)
        # Handle NaN: fill with 0 (difference of missing features = no info)
        X_batch = X_batch.fillna(0.0)
        probs = model.predict_proba(X_batch)[:, 1]
        for i, idx in enumerate(batch_indices):
            submission_df.loc[idx, "Pred"] = probs[i]

    # Fallback: 0.5 for teams without features
    for idx in fallback_indices:
        submission_df.loc[idx, "Pred"] = 0.5

    return submission_df


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    overall_start = time.time()

    print("=" * 70)
    print("KAGGLE MARCH MACHINE LEARNING MANIA 2026 — SUBMISSION GENERATOR")
    print("=" * 70)

    # ── Load data ────────────────────────────────────────────────────────────
    print("\n[1/8] Loading data...")

    # Men's data
    m_reg = pd.read_csv(MANIA_DIR / "MRegularSeasonDetailedResults.csv")
    m_tourney = pd.read_csv(MANIA_DIR / "MNCAATourneyDetailedResults.csv")
    m_seeds = pd.read_csv(MANIA_DIR / "MNCAATourneySeeds.csv")
    m_teams = pd.read_csv(MANIA_DIR / "MTeams.csv")
    m_conf = pd.read_csv(MANIA_DIR / "MTeamConferences.csv")
    m_spellings = pd.read_csv(MANIA_DIR / "MTeamSpellings.csv", encoding="latin-1")

    # Women's data
    w_reg = pd.read_csv(MANIA_DIR / "WRegularSeasonDetailedResults.csv")
    w_tourney = pd.read_csv(MANIA_DIR / "WNCAATourneyDetailedResults.csv")
    w_seeds = pd.read_csv(MANIA_DIR / "WNCAATourneySeeds.csv")
    w_conf = pd.read_csv(MANIA_DIR / "WTeamConferences.csv")

    # Massey ordinals (men only — filter to target systems)
    print("  Loading Massey Ordinals...")
    massey_chunks = []
    for chunk in pd.read_csv(MANIA_DIR / "MMasseyOrdinals.csv", chunksize=500_000):
        filtered = chunk[chunk["SystemName"].isin(MASSEY_SYSTEMS)]
        massey_chunks.append(filtered)
    massey_all = pd.concat(massey_chunks, ignore_index=True)
    # Keep latest snapshot per season
    max_days = massey_all.groupby("Season")["RankingDayNum"].max().reset_index()
    max_days.columns = ["Season", "MaxDay"]
    massey_all = massey_all.merge(max_days, on="Season")
    massey = massey_all[massey_all["RankingDayNum"] == massey_all["MaxDay"]].drop(columns=["MaxDay"])

    # KenPom/Barttorvik
    kenpom = pd.read_csv(KAGGLE_DIR / "KenPom Barttorvik.csv")

    # Sample submissions
    sample_s1 = pd.read_csv(MANIA_DIR / "SampleSubmissionStage1.csv")
    sample_s2 = pd.read_csv(MANIA_DIR / "SampleSubmissionStage2.csv")

    print(f"  Men's RS games   : {len(m_reg):,}")
    print(f"  Men's tourney    : {len(m_tourney):,}")
    print(f"  Women's RS games : {len(w_reg):,}")
    print(f"  Women's tourney  : {len(w_tourney):,}")
    print(f"  Massey (filtered): {len(massey):,}")
    print(f"  KenPom rows      : {len(kenpom):,}")
    print(f"  Stage 1 rows     : {len(sample_s1):,}")
    print(f"  Stage 2 rows     : {len(sample_s2):,}")

    # ── Build KenPom -> Kaggle mapping ───────────────────────────────────────
    print("\n[2/8] Building KenPom -> Kaggle team ID mapping...")
    from src.enhanced_model import build_kenpom_to_kaggle_map
    kp_to_kaggle = build_kenpom_to_kaggle_map(kenpom, m_teams, m_spellings)
    print(f"  Mapped {len(kp_to_kaggle)} KenPom teams")

    # ── Men's features ───────────────────────────────────────────────────────
    # Stage1 needs 2022-2025, Stage2 needs 2026
    # Training needs historical tournament seasons (2003+)
    # Build features for ALL seasons we need for training + submission
    print("\n[3/8] Computing men's features (all D1 teams, 2003-2026)...")
    men_seasons = sorted(s for s in m_reg["Season"].unique() if s >= 2003)
    men_fm = build_all_team_features(
        reg_season=m_reg,
        seeds=m_seeds,
        conferences=m_conf,
        seasons=men_seasons,
        massey=massey,
        kenpom=kenpom,
        kp_to_kaggle=kp_to_kaggle,
        gender="M",
    )
    print(f"  Men's feature matrix: {len(men_fm):,} rows, {len(men_fm.columns)} cols")

    # ── Women's features ─────────────────────────────────────────────────────
    print("\n[4/8] Computing women's features (all D1 teams, 2010-2026)...")
    women_seasons = sorted(s for s in w_reg["Season"].unique() if s >= 2010)
    women_fm = build_all_team_features(
        reg_season=w_reg,
        seeds=w_seeds,
        conferences=w_conf,
        seasons=women_seasons,
        massey=None,  # No women's Massey ordinals
        kenpom=None,  # No women's KenPom
        kp_to_kaggle=None,
        gender="W",
    )
    print(f"  Women's feature matrix: {len(women_fm):,} rows, {len(women_fm.columns)} cols")

    # ── Train men's model ────────────────────────────────────────────────────
    print("\n[5/8] Training men's model...")

    men_feature_cols = get_feature_cols(men_fm)
    print(f"  Men's feature columns ({len(men_feature_cols)}): {men_feature_cols[:10]}...")

    # Filter tournament results to seasons in our feature matrix
    men_fm_seasons = set(men_fm["Season"].unique())
    m_tourney_filtered = m_tourney[m_tourney["Season"].isin(men_fm_seasons)]
    print(f"  Training on {len(m_tourney_filtered)} tournament games")

    X_men, y_men = build_matchup_training_data(men_fm, m_tourney_filtered, men_feature_cols)
    print(f"  Training samples: {len(X_men):,} (win rate: {y_men.mean():.3f})")

    # Drop columns with >30% NaN
    if not X_men.empty:
        null_fracs = X_men.isna().mean()
        drop_cols = null_fracs[null_fracs > 0.30].index.tolist()
        if drop_cols:
            print(f"  Dropping {len(drop_cols)} high-NaN columns: {drop_cols[:10]}...")
            men_feature_cols = [c for c in men_feature_cols if c not in drop_cols]
            X_men = X_men[men_feature_cols]

    # Fill remaining NaN
    men_medians = X_men.median()
    X_men = X_men.fillna(men_medians)

    men_model = train_xgb_model(X_men, y_men)
    print("  Men's model trained.")

    # ── Train women's model ──────────────────────────────────────────────────
    print("\n[6/8] Training women's model...")

    women_feature_cols = get_feature_cols(women_fm)
    print(f"  Women's feature columns ({len(women_feature_cols)}): {women_feature_cols[:10]}...")

    women_fm_seasons = set(women_fm["Season"].unique())
    w_tourney_filtered = w_tourney[w_tourney["Season"].isin(women_fm_seasons)]
    print(f"  Training on {len(w_tourney_filtered)} tournament games")

    X_women, y_women = build_matchup_training_data(women_fm, w_tourney_filtered, women_feature_cols)
    print(f"  Training samples: {len(X_women):,} (win rate: {y_women.mean():.3f})")

    # Drop columns with >30% NaN
    if not X_women.empty:
        null_fracs = X_women.isna().mean()
        drop_cols = null_fracs[null_fracs > 0.30].index.tolist()
        if drop_cols:
            print(f"  Dropping {len(drop_cols)} high-NaN columns: {drop_cols[:10]}...")
            women_feature_cols = [c for c in women_feature_cols if c not in drop_cols]
            X_women = X_women[women_feature_cols]

    women_medians = X_women.median()
    X_women = X_women.fillna(women_medians)

    women_model = train_xgb_model(X_women, y_women)
    print("  Women's model trained.")

    # ── Generate predictions ─────────────────────────────────────────────────
    print("\n[7/8] Generating predictions...")

    for stage_name, sample_df, out_name in [
        ("Stage 1", sample_s1, "submission_stage1.csv"),
        ("Stage 2", sample_s2, "submission_stage2.csv"),
    ]:
        print(f"\n  --- {stage_name} ({len(sample_df):,} rows) ---")
        sub = sample_df.copy()
        sub["Pred"] = 0.5  # default

        # Men's predictions
        print(f"    Predicting men's matchups...")
        sub = predict_all_pairs(sub, men_fm, men_feature_cols, men_model, gender_filter="M")

        # Women's predictions
        print(f"    Predicting women's matchups...")
        sub = predict_all_pairs(sub, women_fm, women_feature_cols, women_model, gender_filter="W")

        # Clip to [0.01, 0.99]
        sub["Pred"] = sub["Pred"].clip(0.01, 0.99)

        # Write
        out_path = OUTPUT_DIR / out_name
        sub.to_csv(out_path, index=False)
        print(f"    Written: {out_path}")

    # ── Validation ───────────────────────────────────────────────────────────
    print("\n[8/8] Validating submissions...")

    for stage_name, sample_path, out_name in [
        ("Stage 1", MANIA_DIR / "SampleSubmissionStage1.csv", "submission_stage1.csv"),
        ("Stage 2", MANIA_DIR / "SampleSubmissionStage2.csv", "submission_stage2.csv"),
    ]:
        sample = pd.read_csv(sample_path)
        sub = pd.read_csv(OUTPUT_DIR / out_name)

        n_expected = len(sample)
        n_actual = len(sub)
        n_nan = sub["Pred"].isna().sum()
        pred_min = sub["Pred"].min()
        pred_max = sub["Pred"].max()
        pred_mean = sub["Pred"].mean()
        pred_std = sub["Pred"].std()

        # Check IDs match
        ids_match = (sample["ID"].values == sub["ID"].values).all()

        # Count non-0.5 predictions (model-based)
        n_non_default = (sub["Pred"] != 0.5).sum()
        pct_modeled = 100 * n_non_default / n_actual

        # Parse men/women split
        parsed = sub["ID"].str.split("_", expand=True)
        team_a = parsed[1].astype(int)
        men_mask = team_a < 3000
        women_mask = team_a >= 3000

        men_preds = sub.loc[men_mask, "Pred"]
        women_preds = sub.loc[women_mask, "Pred"]

        print(f"\n  {stage_name}:")
        print(f"    Rows: {n_actual:,} (expected {n_expected:,}) {'OK' if n_actual == n_expected else 'MISMATCH!'}")
        print(f"    IDs match sample: {'YES' if ids_match else 'NO!'}")
        print(f"    NaN predictions: {n_nan}")
        print(f"    Pred range: [{pred_min:.4f}, {pred_max:.4f}]")
        print(f"    Pred mean: {pred_mean:.4f}, std: {pred_std:.4f}")
        print(f"    Model-based predictions: {n_non_default:,} / {n_actual:,} ({pct_modeled:.1f}%)")
        print(f"    Men's  — count: {len(men_preds):,}, mean: {men_preds.mean():.4f}, std: {men_preds.std():.4f}")
        print(f"    Women's — count: {len(women_preds):,}, mean: {women_preds.mean():.4f}, std: {women_preds.std():.4f}")

    elapsed = time.time() - overall_start
    print(f"\nTotal elapsed: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print("\nDone! Submission files written to:")
    print(f"  {OUTPUT_DIR / 'submission_stage1.csv'}")
    print(f"  {OUTPUT_DIR / 'submission_stage2.csv'}")


if __name__ == "__main__":
    main()
