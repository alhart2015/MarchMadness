"""Enhanced March Madness prediction model.

Combines game-level features from the Kaggle March Machine Learning Mania dataset
with season-aggregate KenPom/Barttorvik stats. Runs end-to-end: feature engineering,
leave-one-season-out CV, Optuna tuning, 2026 bracket prediction, and HTML update.

Usage
-----
    python src/enhanced_model.py
"""

import json
import logging
import re
import sys
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

# ── Suppress noisy warnings ─────────────────────────────────────────────────
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ── Logging setup ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Path setup ───────────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent          # src/
_ROOT = _HERE.parent                             # project root
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

MANIA_DIR   = _ROOT / "data" / "raw" / "march-machine-learning-2026"
KAGGLE_DIR  = _ROOT / "data" / "raw" / "kaggle"
BRACKET_CSV = _ROOT / "data" / "raw" / "bracket_2026.csv"
OUTPUT_DIR  = _ROOT / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Constants ────────────────────────────────────────────────────────────────
MASSEY_SYSTEMS = ["POM", "SAG", "MOR", "WOL", "DOL", "COL", "RPI"]
FIRST_ROUND_PAIRS = [(1, 16), (8, 9), (5, 12), (4, 13), (6, 11), (3, 14), (7, 10), (2, 15)]
ROUND_NAMES = {1: "R64", 2: "R32", 3: "S16", 4: "E8", 5: "F4", 6: "Champ"}


# ═════════════════════════════════════════════════════════════════════════════
# STEP 1: LOAD ALL DATA
# ═════════════════════════════════════════════════════════════════════════════

def load_all_data() -> dict:
    """Load all datasets from both Kaggle Mania and KenPom/Barttorvik sources."""
    print("\n" + "=" * 70)
    print("STEP 1 — Loading all data sources")
    print("=" * 70)

    data = {}

    # ── Kaggle March Mania dataset ───────────────────────────────────────
    data["reg_season"] = pd.read_csv(MANIA_DIR / "MRegularSeasonDetailedResults.csv")
    data["tourney"] = pd.read_csv(MANIA_DIR / "MNCAATourneyDetailedResults.csv")
    data["seeds"] = pd.read_csv(MANIA_DIR / "MNCAATourneySeeds.csv")
    data["teams"] = pd.read_csv(MANIA_DIR / "MTeams.csv")
    data["conferences"] = pd.read_csv(MANIA_DIR / "MTeamConferences.csv")
    data["spellings"] = pd.read_csv(
        MANIA_DIR / "MTeamSpellings.csv", encoding="latin-1"
    )

    # ── Massey Ordinals (load only needed systems + latest day per season) ─
    print("  Loading Massey Ordinals (filtering to target systems)...")
    massey_chunks = []
    for chunk in pd.read_csv(MANIA_DIR / "MMasseyOrdinals.csv", chunksize=500_000):
        filtered = chunk[chunk["SystemName"].isin(MASSEY_SYSTEMS)]
        massey_chunks.append(filtered)
    massey_all = pd.concat(massey_chunks, ignore_index=True)

    # Keep only latest snapshot per season
    max_days = massey_all.groupby("Season")["RankingDayNum"].max().reset_index()
    max_days.columns = ["Season", "MaxDay"]
    massey_all = massey_all.merge(max_days, on="Season")
    data["massey"] = massey_all[massey_all["RankingDayNum"] == massey_all["MaxDay"]].drop(
        columns=["MaxDay"]
    )

    # ── KenPom/Barttorvik ────────────────────────────────────────────────
    data["kenpom"] = pd.read_csv(KAGGLE_DIR / "KenPom Barttorvik.csv")

    print(f"  Regular season games : {len(data['reg_season']):,}")
    print(f"  Tournament games     : {len(data['tourney']):,}")
    print(f"  Seeds                : {len(data['seeds']):,}")
    print(f"  Teams                : {len(data['teams']):,}")
    print(f"  Massey (filtered)    : {len(data['massey']):,}")
    print(f"  KenPom/Barttorvik    : {len(data['kenpom']):,}")

    return data


# ═════════════════════════════════════════════════════════════════════════════
# STEP 2: TEAM ID MAPPING
# ═════════════════════════════════════════════════════════════════════════════

def build_kenpom_to_kaggle_map(
    kenpom: pd.DataFrame,
    teams: pd.DataFrame,
    spellings: pd.DataFrame,
) -> dict:
    """Build mapping from KenPom TEAM NO to Kaggle TeamID using name matching."""
    from rapidfuzz import fuzz, process

    # Build spelling -> kaggle_id lookup (lowercase)
    spell_map = {}
    for _, row in spellings.iterrows():
        spell_map[str(row["TeamNameSpelling"]).lower().strip()] = int(row["TeamID"])

    # Build kaggle_name -> kaggle_id lookup
    kaggle_names = {}
    for _, row in teams.iterrows():
        kaggle_names[row["TeamName"]] = int(row["TeamID"])
        kaggle_names[row["TeamName"].lower()] = int(row["TeamID"])

    # Get unique KenPom team entries
    kp_teams = kenpom[["TEAM NO", "TEAM"]].drop_duplicates("TEAM NO")

    mapping = {}  # kenpom_team_no -> kaggle_team_id

    for _, row in kp_teams.iterrows():
        kp_id = int(row["TEAM NO"])
        kp_name = str(row["TEAM"]).strip()

        # Try exact match in spellings first
        if kp_name.lower() in spell_map:
            mapping[kp_id] = spell_map[kp_name.lower()]
            continue

        # Try exact match in MTeams
        if kp_name in kaggle_names:
            mapping[kp_id] = kaggle_names[kp_name]
            continue

        # Fuzzy match against MTeams names
        result = process.extractOne(
            kp_name, list(teams["TeamName"].values), scorer=fuzz.token_sort_ratio
        )
        if result and result[1] >= 75:
            match_name = result[0]
            mapping[kp_id] = kaggle_names[match_name]
        else:
            # Try fuzzy against spellings
            result2 = process.extractOne(
                kp_name.lower(),
                list(spellings["TeamNameSpelling"].values),
                scorer=fuzz.token_sort_ratio,
            )
            if result2 and result2[1] >= 75:
                mapping[kp_id] = spell_map.get(result2[0].lower(), -1)

    return mapping


# ═════════════════════════════════════════════════════════════════════════════
# STEP 2: COMPUTE GAME-LEVEL FEATURES
# ═════════════════════════════════════════════════════════════════════════════

def _parse_seed_number(seed_str: str) -> int:
    """Extract numeric seed from strings like 'W01', 'X16a'."""
    match = re.search(r"(\d+)", str(seed_str))
    return int(match.group(1)) if match else 16


def compute_all_features(data: dict) -> pd.DataFrame:
    """Compute the full feature matrix: one row per team per season.

    Combines game-level features from Kaggle Mania with season-aggregate
    KenPom/Barttorvik stats.
    """
    print("\n" + "=" * 70)
    print("STEP 2 — Computing game-level features")
    print("=" * 70)

    reg = data["reg_season"]
    seeds = data["seeds"]
    massey = data["massey"]
    conferences = data["conferences"]
    kenpom = data["kenpom"]
    teams = data["teams"]
    spellings = data["spellings"]

    # Determine seasons range (use regular season data as reference)
    seasons = sorted(reg["Season"].unique())
    # Only process seasons where we have detailed results (2003+)
    seasons = [s for s in seasons if s >= 2003]

    # ── Build KenPom -> Kaggle ID mapping ────────────────────────────────
    print("  Building team ID mapping (KenPom -> Kaggle)...")
    kp_to_kaggle = build_kenpom_to_kaggle_map(kenpom, teams, spellings)
    print(f"  Mapped {len(kp_to_kaggle)} KenPom teams to Kaggle IDs")

    all_rows = []

    for season in seasons:
        season_reg = reg[reg["Season"] == season]
        if len(season_reg) == 0:
            continue

        # ── 2a: Adjusted efficiency (full season) ───────────────────────
        from src.features.efficiency import compute_adjusted_efficiency
        try:
            eff = compute_adjusted_efficiency(reg, season, iterations=10)
        except Exception as e:
            logger.warning("Efficiency computation failed for %d: %s", season, e)
            eff = pd.DataFrame(columns=["TeamID", "adj_oe", "adj_de", "adj_em", "adj_tempo"])

        # ── 2b: Four factors ─────────────────────────────────────────────
        from src.features.four_factors import compute_four_factors, estimate_possessions
        try:
            ff = compute_four_factors(reg, season)
        except Exception as e:
            logger.warning("Four factors failed for %d: %s", season, e)
            ff = pd.DataFrame(columns=["TeamID"])

        # ── 2c: Rolling efficiency (last 30 days) ───────────────────────
        max_day = season_reg["DayNum"].max()
        recent_games = season_reg[season_reg["DayNum"] > max_day - 30]

        # Build per-game records for recent window
        recent_rows = []
        for _, g in recent_games.iterrows():
            w_poss = estimate_possessions(g["WFGA"], g["WOR"], g["WTO"], g["WFTA"])
            l_poss = estimate_possessions(g["LFGA"], g["LOR"], g["LTO"], g["LFTA"])
            poss = (w_poss + l_poss) / 2
            if poss > 0:
                recent_rows.append({
                    "TeamID": g["WTeamID"], "pts": g["WScore"],
                    "pts_allowed": g["LScore"], "poss": poss, "win": 1,
                    "DayNum": g["DayNum"],
                })
                recent_rows.append({
                    "TeamID": g["LTeamID"], "pts": g["LScore"],
                    "pts_allowed": g["WScore"], "poss": poss, "win": 0,
                    "DayNum": g["DayNum"],
                })

        recent_df = pd.DataFrame(recent_rows)
        rolling_eff = {}
        if not recent_df.empty:
            for tid, grp in recent_df.groupby("TeamID"):
                rolling_eff[tid] = {
                    "rolling_oe": 100 * grp["pts"].sum() / grp["poss"].sum(),
                    "rolling_de": 100 * grp["pts_allowed"].sum() / grp["poss"].sum(),
                    "win_pct_30d": grp["win"].mean(),
                }

        # ── 2d: Recent form features ────────────────────────────────────
        # All games for the season
        w_games = season_reg[["WTeamID", "WScore", "LScore", "DayNum"]].rename(
            columns={"WTeamID": "TeamID", "WScore": "pts", "LScore": "pts_allowed"}
        ).assign(win=1)
        l_games = season_reg[["LTeamID", "LScore", "WScore", "DayNum"]].rename(
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

        # ── 2e: Massey ordinals ──────────────────────────────────────────
        season_massey = massey[massey["Season"] == season]
        massey_features = {}
        if not season_massey.empty:
            for system in MASSEY_SYSTEMS:
                sys_ranks = season_massey[season_massey["SystemName"] == system]
                for _, row in sys_ranks.iterrows():
                    tid = int(row["TeamID"])
                    if tid not in massey_features:
                        massey_features[tid] = {}
                    massey_features[tid][f"massey_{system}"] = row["OrdinalRank"]

            # Composite rank
            for tid, feats in massey_features.items():
                ranks = [v for k, v in feats.items() if k.startswith("massey_")]
                if ranks:
                    massey_features[tid]["massey_composite"] = np.mean(ranks)

        # ── 2f: Conference strength ──────────────────────────────────────
        season_conf = conferences[conferences["Season"] == season]
        conf_strength = {}
        if not season_conf.empty and not eff.empty:
            conf_merged = season_conf.merge(eff[["TeamID", "adj_em"]], on="TeamID", how="left")
            conf_avg = conf_merged.groupby("ConfAbbrev")["adj_em"].mean()
            team_conf = dict(zip(season_conf["TeamID"], season_conf["ConfAbbrev"]))
            for tid, conf_abbrev in team_conf.items():
                if conf_abbrev in conf_avg.index:
                    conf_strength[tid] = conf_avg[conf_abbrev]

        # ── 2g: KenPom/Barttorvik season aggregates ─────────────────────
        kp_season = kenpom[kenpom["YEAR"] == season]
        kp_features = {}
        kp_cols = [
            "KADJ EM", "KADJ O", "KADJ D", "BARTHAG", "TALENT", "EXP",
            "ELITE SOS", "WAB", "EFG%", "EFG%D", "TOV%", "TOV%D",
            "OREB%", "DREB%", "FTR", "FTRD", "K TEMPO",
        ]
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

        # ── 2h: Seed features ───────────────────────────────────────────
        season_seeds = seeds[seeds["Season"] == season]
        seed_map = {}
        for _, row in season_seeds.iterrows():
            seed_map[int(row["TeamID"])] = _parse_seed_number(row["Seed"])

        # ── Assemble features for each team in this season ───────────────
        all_team_ids = set()
        all_team_ids.update(eff["TeamID"].values if not eff.empty else [])
        all_team_ids.update(seed_map.keys())

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
            if not ff.empty:
                ff_row = ff[ff["TeamID"] == tid]
                if not ff_row.empty:
                    for col in ff.columns:
                        if col != "TeamID":
                            row_data[col] = ff_row.iloc[0][col]

            # Rolling efficiency
            if tid in rolling_eff:
                row_data.update(rolling_eff[tid])

            # Recent form
            if tid in form_features:
                row_data.update(form_features[tid])

            # Massey ordinals
            if tid in massey_features:
                row_data.update(massey_features[tid])

            # Conference strength
            if tid in conf_strength:
                row_data["conf_strength"] = conf_strength[tid]

            # KenPom/Barttorvik
            if tid in kp_features:
                row_data.update(kp_features[tid])

            all_rows.append(row_data)

        if season % 5 == 0 or season == seasons[-1]:
            print(f"  Season {season}: {len([r for r in all_rows if r['Season']==season])} teams")

    feature_matrix = pd.DataFrame(all_rows)

    # ── Filter to tournament teams only (have a seed) ────────────────────
    tourney_fm = feature_matrix[feature_matrix["seed"].notna()].copy()
    print(f"\n  Full feature matrix : {len(feature_matrix):,} rows")
    print(f"  Tournament teams    : {len(tourney_fm):,} rows")
    print(f"  Feature columns     : {len(tourney_fm.columns)} total")
    print(f"  Seasons             : {int(tourney_fm['Season'].min())}-{int(tourney_fm['Season'].max())}")

    return tourney_fm


# ═════════════════════════════════════════════════════════════════════════════
# STEP 3: GET FEATURE COLUMNS
# ═════════════════════════════════════════════════════════════════════════════

def get_feature_cols(fm: pd.DataFrame) -> list:
    """Return list of numeric feature column names."""
    exclude = {"TeamID", "Season", "seed"}
    cols = [c for c in fm.columns if c not in exclude]
    numeric_cols = fm[cols].select_dtypes(include="number").columns.tolist()
    return numeric_cols


# ═════════════════════════════════════════════════════════════════════════════
# STEP 4: BUILD MATCHUP DATA & TRAIN/EVALUATE
# ═════════════════════════════════════════════════════════════════════════════

def build_matchup_data_from_kaggle(
    feature_matrix: pd.DataFrame,
    tourney_results: pd.DataFrame,
    feature_cols: list,
) -> tuple:
    """Build symmetric matchup training data from Kaggle Mania tournament results.

    Uses Kaggle TeamIDs directly (from MNCAATourneyDetailedResults.csv).
    """
    rows = []
    labels = []
    seasons_list = []

    for _, game in tourney_results.iterrows():
        season = game["Season"]
        w_id = game["WTeamID"]
        l_id = game["LTeamID"]

        w_feats = feature_matrix[
            (feature_matrix["TeamID"] == w_id) & (feature_matrix["Season"] == season)
        ]
        l_feats = feature_matrix[
            (feature_matrix["TeamID"] == l_id) & (feature_matrix["Season"] == season)
        ]

        if w_feats.empty or l_feats.empty:
            continue

        w_vals = w_feats.iloc[0][feature_cols].values.astype(float)
        l_vals = l_feats.iloc[0][feature_cols].values.astype(float)

        # Winner perspective
        rows.append(w_vals - l_vals)
        labels.append(1)
        seasons_list.append(season)

        # Loser perspective
        rows.append(l_vals - w_vals)
        labels.append(0)
        seasons_list.append(season)

    X = pd.DataFrame(rows, columns=feature_cols)
    y = pd.Series(labels, name="win")
    s = pd.Series(seasons_list, name="season")
    return X, y, s


def leave_one_season_out_cv(
    feature_matrix: pd.DataFrame,
    tourney_results: pd.DataFrame,
    feature_cols: list,
    xgb_params: dict = None,
    random_seed: int = 42,
) -> dict:
    """Run LOSO CV using Kaggle Mania tournament results."""
    from sklearn.metrics import log_loss as sklearn_log_loss, roc_auc_score
    from src.models.train import train_model

    seasons = sorted(tourney_results["Season"].unique())
    # Only evaluate seasons where we have features (2003+)
    seasons = [s for s in seasons if s >= 2003]

    results = []

    for holdout in seasons:
        train_tourney = tourney_results[tourney_results["Season"] != holdout]
        test_tourney = tourney_results[tourney_results["Season"] == holdout]

        if len(test_tourney) == 0:
            continue

        X_train, y_train, _ = build_matchup_data_from_kaggle(
            feature_matrix, train_tourney, feature_cols
        )
        X_test, y_test, _ = build_matchup_data_from_kaggle(
            feature_matrix, test_tourney, feature_cols
        )

        if len(X_train) == 0 or len(X_test) == 0:
            continue

        # Fill NaN with training median
        medians = X_train.median()
        X_train = X_train.fillna(medians)
        X_test = X_test.fillna(medians)

        model = train_model(X_train, y_train, random_seed=random_seed, xgb_params=xgb_params)
        y_prob = model.predict_proba(X_test)[:, 1]

        season_loss = float(sklearn_log_loss(y_test, y_prob))
        season_brier = float(np.mean((y_prob - y_test.values) ** 2))
        season_acc = float((y_prob.round() == y_test).mean())
        try:
            season_auc = float(roc_auc_score(y_test, y_prob))
        except ValueError:
            season_auc = 0.5

        results.append({
            "season": holdout,
            "log_loss": season_loss,
            "brier_score": season_brier,
            "accuracy": season_acc,
            "auc": season_auc,
            "n_games": len(test_tourney),
        })

    results_df = pd.DataFrame(results)
    return {
        "per_season": results_df,
        "mean_log_loss": float(results_df["log_loss"].mean()),
        "mean_brier_score": float(results_df["brier_score"].mean()),
        "mean_accuracy": float(results_df["accuracy"].mean()),
        "mean_auc": float(results_df["auc"].mean()),
    }


# ═════════════════════════════════════════════════════════════════════════════
# STEP 5: BRACKET SIMULATION
# ═════════════════════════════════════════════════════════════════════════════

def precompute_win_probs(bracket, feature_matrix, feature_cols, model):
    """Pre-compute P(a beats b) for all bracket team pairs."""
    team_ids = bracket["TeamID"].tolist()
    n = len(team_ids)

    feat_lookup = {}
    for tid in team_ids:
        row = feature_matrix[feature_matrix["TeamID"] == tid]
        if row.empty:
            raise ValueError(f"TeamID {tid} not found in feature matrix")
        feat_lookup[tid] = row.iloc[0][feature_cols].values.astype(float)

    pairs = [(team_ids[i], team_ids[j]) for i in range(n) for j in range(n) if i != j]
    rows_list = [feat_lookup[a] - feat_lookup[b] for a, b in pairs]

    X_batch = pd.DataFrame(rows_list, columns=feature_cols).fillna(0.0)
    proba_batch = model.predict_proba(X_batch)[:, 1]

    return {(a, b): float(p) for (a, b), p in zip(pairs, proba_batch)}


def simulate_tournament_fast(bracket, win_prob, n_simulations=10_000, random_seed=42):
    """Monte Carlo tournament simulation with pre-computed probabilities."""
    rng = np.random.default_rng(random_seed)
    regions = sorted(bracket["Region"].unique())

    region_teams = {}
    for region in regions:
        rdf = bracket[bracket["Region"] == region]
        region_teams[region] = dict(zip(rdf["Seed"].astype(int), rdf["TeamID"].astype(int)))

    advancement = defaultdict(lambda: defaultdict(int))
    champions = defaultdict(int)

    def play(a, b):
        p = win_prob.get((a, b), 0.5)
        return a if rng.random() < p else b

    def sim_region(seed_to_tid):
        r1 = []
        for sa, sb in FIRST_ROUND_PAIRS:
            w = play(seed_to_tid[sa], seed_to_tid[sb])
            r1.append(w)
            advancement[w][1] += 1
        current = r1
        for rnd in range(2, 5):
            nxt = []
            for i in range(0, len(current), 2):
                w = play(current[i], current[i + 1])
                nxt.append(w)
                advancement[w][rnd] += 1
            current = nxt
        return current[0]

    for sim in range(n_simulations):
        rc = []
        for region in regions:
            champ = sim_region(region_teams[region])
            rc.append(champ)
            advancement[champ][5] += 1

        # Regions sorted: East(0), Midwest(1), South(2), West(3)
        # Semis: East vs South, West vs Midwest
        semi1 = play(rc[0], rc[2])
        semi2 = play(rc[3], rc[1])
        champion = play(semi1, semi2)
        advancement[champion][6] += 1
        champions[champion] += 1

        if (sim + 1) % 2000 == 0:
            print(f"    ...{sim + 1:,} / {n_simulations:,} simulations done")

    return {
        "advancement_counts": dict(advancement),
        "champions": dict(champions),
        "n_simulations": n_simulations,
    }


def get_advancement_probabilities(counts, n_sim):
    return {
        tid: {r: c / n_sim for r, c in rounds.items()}
        for tid, rounds in counts.items()
    }


# ═════════════════════════════════════════════════════════════════════════════
# STEP 6: BRACKET OUTPUT
# ═════════════════════════════════════════════════════════════════════════════

def build_bracket_compact_json(bracket, advancement_probs, win_prob):
    """Build the compact JSON structure needed by bracket.html."""
    id_to_seed = dict(zip(bracket["TeamID"], bracket["Seed"]))
    id_to_name = dict(zip(bracket["TeamID"], bracket["TeamName"]))

    # Build bracket structure: {region: [matchup_pair, ...]}
    bracket_data = {}
    for region in ["East", "West", "South", "Midwest"]:
        rdf = bracket[bracket["Region"] == region]
        seed_to_tid = dict(zip(rdf["Seed"].astype(int), rdf["TeamID"].astype(int)))

        matchups = []
        for sa, sb in FIRST_ROUND_PAIRS:
            ta, tb = seed_to_tid[sa], seed_to_tid[sb]
            t1 = {
                "id": int(ta),
                "name": id_to_name.get(ta, str(ta)),
                "seed": int(sa),
                "adv": {},
            }
            t2 = {
                "id": int(tb),
                "name": id_to_name.get(tb, str(tb)),
                "seed": int(sb),
                "adv": {},
            }
            # Add advancement probabilities
            round_keys = ["R1", "R2", "R3", "R4", "R5", "R6"]
            for t_obj, tid in [(t1, ta), (t2, tb)]:
                probs = advancement_probs.get(tid, {})
                for rnd_num, rk in enumerate(round_keys, 1):
                    if rnd_num in probs:
                        t_obj["adv"][rk] = round(probs[rnd_num], 4)

            matchups.append([t1, t2])
        bracket_data[region] = matchups

    # Build pairwise probabilities: "id1_id2" -> P(id1 wins), where id1 < id2
    pairwise = {}
    team_ids = bracket["TeamID"].tolist()
    for i in range(len(team_ids)):
        for j in range(i + 1, len(team_ids)):
            a, b = team_ids[i], team_ids[j]
            lo, hi = min(a, b), max(a, b)
            key = f"{lo}_{hi}"
            # P(lo wins)
            p = win_prob.get((lo, hi), 0.5)
            pairwise[key] = round(p, 4)

    return {
        "bracket": bracket_data,
        "pairwise": pairwise,
        "matchup_order": [list(p) for p in FIRST_ROUND_PAIRS],
    }


def print_champion_probs(advancement_probs, bracket, top_n=15):
    id_to_name = dict(zip(bracket["TeamID"], bracket["TeamName"]))
    id_to_seed = dict(zip(bracket["TeamID"], bracket["Seed"]))

    sorted_teams = sorted(
        advancement_probs.items(),
        key=lambda x: x[1].get(6, 0),
        reverse=True,
    )

    print(f"\n{'Rank':>4}  {'Seed':>4}  {'Team':<28}  {'Champ%':>8}  {'F4%':>7}  {'E8%':>7}")
    print("-" * 65)
    for rank, (tid, probs) in enumerate(sorted_teams[:top_n], start=1):
        name = id_to_name.get(tid, str(tid))
        seed = id_to_seed.get(tid, "?")
        print(
            f"{rank:>4}  {seed:>4}  {name:<28}  "
            f"{probs.get(6, 0):>8.2%}  {probs.get(5, 0):>7.2%}  {probs.get(4, 0):>7.2%}"
        )


def print_advancement_table(advancement_probs, bracket, top_n=30):
    id_to_name = dict(zip(bracket["TeamID"], bracket["TeamName"]))
    id_to_seed = dict(zip(bracket["TeamID"], bracket["Seed"]))
    id_to_region = dict(zip(bracket["TeamID"], bracket["Region"]))

    sorted_teams = sorted(
        advancement_probs.items(),
        key=lambda x: x[1].get(6, 0),
        reverse=True,
    )

    header = (
        f"{'#':>3}  {'Seed':>4}  {'Team':<26}  {'Region':<10}  "
        f"{'R64':>6}  {'R32':>6}  {'S16':>6}  {'E8':>6}  {'F4':>6}  {'Champ':>7}"
    )
    print(header)
    print("-" * len(header))

    for rank, (tid, probs) in enumerate(sorted_teams[:top_n], start=1):
        name = id_to_name.get(tid, str(tid))
        seed = id_to_seed.get(tid, "?")
        region = id_to_region.get(tid, "?")
        print(
            f"{rank:>3}  {seed:>4}  {name:<26}  {region:<10}  "
            f"{probs.get(1, 0):>6.1%}  {probs.get(2, 0):>6.1%}  "
            f"{probs.get(3, 0):>6.1%}  {probs.get(4, 0):>6.1%}  "
            f"{probs.get(5, 0):>6.1%}  {probs.get(6, 0):>7.2%}"
        )


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    # ── Step 1: Load all data ────────────────────────────────────────────
    data = load_all_data()

    # ── Step 2: Compute features ─────────────────────────────────────────
    feature_matrix = compute_all_features(data)

    feature_cols = get_feature_cols(feature_matrix)
    print(f"\n  Feature columns ({len(feature_cols)}):")
    for i in range(0, len(feature_cols), 6):
        print(f"    {', '.join(feature_cols[i:i+6])}")

    # ── Step 3: Build matchup training data ──────────────────────────────
    print("\n" + "=" * 70)
    print("STEP 3 — Building matchup training data")
    print("=" * 70)

    tourney = data["tourney"]
    # Only use seasons present in feature matrix
    fm_seasons = set(feature_matrix["Season"].unique())
    tourney_filtered = tourney[tourney["Season"].isin(fm_seasons)]

    X_all, y_all, s_all = build_matchup_data_from_kaggle(
        feature_matrix, tourney_filtered, feature_cols
    )

    # Drop columns with > 30% NaN
    if not X_all.empty:
        null_fracs = X_all.isna().mean()
        drop_cols = null_fracs[null_fracs > 0.30].index.tolist()
        if drop_cols:
            print(f"  Dropping {len(drop_cols)} high-NaN columns: {drop_cols}")
            feature_cols = [c for c in feature_cols if c not in drop_cols]
            X_all = X_all[feature_cols]

    # Fill remaining NaN with median
    medians = X_all.median()
    X_all = X_all.fillna(medians)

    print(f"  Training samples : {len(X_all):,}  (win rate {y_all.mean():.3f})")
    print(f"  Features used    : {len(feature_cols)}")

    # ── Step 4: Leave-one-season-out CV (default params) ─────────────────
    print("\n" + "=" * 70)
    print("STEP 4 — Leave-one-season-out CV (default XGBoost params)")
    print("=" * 70)

    # Fill NaN in feature matrix for CV
    fm_filled = feature_matrix.copy()
    for col in feature_cols:
        if col in fm_filled.columns:
            fm_filled[col] = fm_filled[col].fillna(fm_filled[col].median())

    cv_default = leave_one_season_out_cv(
        fm_filled, tourney_filtered, feature_cols, random_seed=42
    )

    print(f"\n  Default params CV results:")
    print(f"  Mean Log Loss  : {cv_default['mean_log_loss']:.4f}")
    print(f"  Mean Brier     : {cv_default['mean_brier_score']:.4f}")
    print(f"  Mean Accuracy  : {cv_default['mean_accuracy']:.3f}")
    print(f"  Mean AUC       : {cv_default['mean_auc']:.4f}")

    # ── Step 5: Optuna hyperparameter tuning ─────────────────────────────
    print("\n" + "=" * 70)
    print("STEP 5 — Optuna hyperparameter tuning (30 trials)")
    print("=" * 70)

    from src.models.tuning import tune_hyperparameters
    best_params = tune_hyperparameters(X_all, y_all, n_trials=30, random_seed=42)
    print(f"  Best params: {best_params}")

    # ── Step 6: Re-evaluate with tuned params ────────────────────────────
    print("\n" + "=" * 70)
    print("STEP 6 — Leave-one-season-out CV (tuned params)")
    print("=" * 70)

    cv_tuned = leave_one_season_out_cv(
        fm_filled, tourney_filtered, feature_cols,
        xgb_params=best_params, random_seed=42,
    )

    per_season = cv_tuned["per_season"]
    print(f"\n{'Season':>8}  {'LogLoss':>9}  {'Brier':>7}  {'Accuracy':>9}  {'AUC':>7}  {'#Games':>7}")
    print("-" * 58)
    for _, row in per_season.sort_values("season").iterrows():
        print(
            f"  {int(row['season']):>6}  {row['log_loss']:>9.4f}  "
            f"{row['brier_score']:>7.4f}  {row['accuracy']:>9.3f}  "
            f"{row['auc']:>7.4f}  {int(row['n_games']):>7}"
        )
    print("-" * 58)
    print(
        f"  {'MEAN':>6}  {cv_tuned['mean_log_loss']:>9.4f}  "
        f"{cv_tuned['mean_brier_score']:>7.4f}  {cv_tuned['mean_accuracy']:>9.3f}  "
        f"{cv_tuned['mean_auc']:>7.4f}"
    )

    # ── Comparison with baseline ─────────────────────────────────────────
    print("\n" + "=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)
    baseline_ll = 0.57
    baseline_acc = 0.705
    new_ll = cv_tuned["mean_log_loss"]
    new_acc = cv_tuned["mean_accuracy"]

    print(f"\n  {'Metric':<20}  {'Baseline':>10}  {'Enhanced':>10}  {'Delta':>10}")
    print(f"  {'-'*55}")
    print(f"  {'Log Loss':<20}  {baseline_ll:>10.4f}  {new_ll:>10.4f}  {new_ll - baseline_ll:>+10.4f}")
    print(f"  {'Accuracy':<20}  {baseline_acc:>10.3f}  {new_acc:>10.3f}  {new_acc - baseline_acc:>+10.3f}")
    print(f"  {'Brier Score':<20}  {'N/A':>10}  {cv_tuned['mean_brier_score']:>10.4f}")
    print(f"  {'AUC':<20}  {'N/A':>10}  {cv_tuned['mean_auc']:>10.4f}")

    if new_ll < baseline_ll:
        pct = (baseline_ll - new_ll) / baseline_ll * 100
        print(f"\n  >>> Enhanced model improves log loss by {pct:.1f}% <<<")
    else:
        print(f"\n  Note: Log loss did not improve (baseline {baseline_ll:.4f} vs {new_ll:.4f})")

    # ── Step 7: Train final model on all data ────────────────────────────
    print("\n" + "=" * 70)
    print("STEP 7 — Training final model on all historical data")
    print("=" * 70)

    from src.models.train import train_model
    final_model = train_model(X_all, y_all, random_seed=42, xgb_params=best_params)
    print("  Final model trained successfully.")

    # ── Step 8: Generate 2026 predictions ────────────────────────────────
    print("\n" + "=" * 70)
    print("STEP 8 — Loading 2026 bracket and preparing predictions")
    print("=" * 70)

    # Load actual bracket
    bracket_raw = pd.read_csv(BRACKET_CSV)
    bracket_raw["Seed"] = bracket_raw["Seed"].astype(int)
    bracket_raw["TeamID"] = bracket_raw["TeamID"].astype(int)

    # The bracket uses KenPom TEAM NO as TeamID. Map to Kaggle TeamID.
    kp_to_kaggle = build_kenpom_to_kaggle_map(
        data["kenpom"], data["teams"], data["spellings"]
    )

    bracket_kaggle = bracket_raw.copy()
    bracket_kaggle["KenPomID"] = bracket_kaggle["TeamID"]
    bracket_kaggle["TeamID"] = bracket_kaggle["KenPomID"].map(kp_to_kaggle)

    # Check for unmapped teams
    unmapped = bracket_kaggle[bracket_kaggle["TeamID"].isna()]
    if len(unmapped) > 0:
        print(f"  WARNING: {len(unmapped)} teams could not be mapped to Kaggle IDs:")
        for _, row in unmapped.iterrows():
            print(f"    {row['TeamName']} (KenPom ID={row['KenPomID']})")

    bracket_kaggle = bracket_kaggle.dropna(subset=["TeamID"])
    bracket_kaggle["TeamID"] = bracket_kaggle["TeamID"].astype(int)
    print(f"  Bracket teams mapped: {len(bracket_kaggle)} / 64")

    # Get 2026 features
    fm_2026 = fm_filled[fm_filled["Season"] == 2026].copy()
    bracket_team_ids = set(bracket_kaggle["TeamID"].tolist())
    fm_2026_tourney = fm_2026[fm_2026["TeamID"].isin(bracket_team_ids)].copy()

    # Fill any remaining NaN
    for col in feature_cols:
        if col in fm_2026_tourney.columns:
            col_median = fm_filled[col].median()
            fm_2026_tourney[col] = fm_2026_tourney[col].fillna(col_median)

    missing_ids = bracket_team_ids - set(fm_2026_tourney["TeamID"].tolist())
    if missing_ids:
        print(f"  WARNING: {len(missing_ids)} bracket teams missing from 2026 feature matrix")
        # Create placeholder rows with median features for missing teams
        for tid in missing_ids:
            placeholder = {col: fm_filled[col].median() for col in feature_cols}
            placeholder["TeamID"] = tid
            placeholder["Season"] = 2026
            placeholder["seed"] = bracket_kaggle[bracket_kaggle["TeamID"] == tid]["Seed"].iloc[0]
            fm_2026_tourney = pd.concat(
                [fm_2026_tourney, pd.DataFrame([placeholder])], ignore_index=True
            )
        print(f"  Added placeholder features for {len(missing_ids)} missing teams")
    else:
        print(f"  All {len(bracket_team_ids)} bracket teams found in feature matrix.")

    # ── Step 9: Monte Carlo simulation ───────────────────────────────────
    print("\n" + "=" * 70)
    print("STEP 9 — Pre-computing pairwise win probabilities")
    print("=" * 70)

    n_teams = len(bracket_kaggle)
    print(f"  Computing {n_teams} x {n_teams - 1} = {n_teams * (n_teams - 1):,} pair probabilities...")
    win_prob = precompute_win_probs(bracket_kaggle, fm_2026_tourney, feature_cols, final_model)
    print(f"  Done. Lookup table has {len(win_prob):,} entries.")

    print("\n" + "=" * 70)
    print("STEP 10 — Monte Carlo simulation (10,000 iterations)")
    print("=" * 70)

    print("  Running simulation...")
    sim_results = simulate_tournament_fast(
        bracket=bracket_kaggle,
        win_prob=win_prob,
        n_simulations=10_000,
        random_seed=42,
    )
    print("  Simulation complete.")

    advancement_probs = get_advancement_probabilities(
        sim_results["advancement_counts"],
        sim_results["n_simulations"],
    )

    # ── Results ──────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("RESULTS — Championship Probabilities (Top 15)")
    print("=" * 70)
    print_champion_probs(advancement_probs, bracket_kaggle, top_n=15)

    print("\n" + "=" * 70)
    print("RESULTS — Advancement Probabilities (Top 30)")
    print("=" * 70)
    print_advancement_table(advancement_probs, bracket_kaggle, top_n=30)

    # ── Bracket picks ────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("RESULTS — Bracket Picks")
    print("=" * 70)

    from src.bracket.strategies import chalk_bracket, expected_value_bracket

    chalk_picks = chalk_bracket(bracket_kaggle, advancement_probs)
    ev_picks = expected_value_bracket(bracket_kaggle, advancement_probs)

    id_to_name = dict(zip(bracket_kaggle["TeamID"], bracket_kaggle["TeamName"]))
    id_to_seed = dict(zip(bracket_kaggle["TeamID"], bracket_kaggle["Seed"]))

    # Print chalk champion
    if chalk_picks.get(6):
        chalk_champ = chalk_picks[6][0]
        print(f"\n  Chalk champion: ({id_to_seed.get(chalk_champ, '?'):>2}) "
              f"{id_to_name.get(chalk_champ, str(chalk_champ))}  "
              f"{advancement_probs.get(chalk_champ, {}).get(6, 0):.2%}")

    if ev_picks.get(6):
        ev_champ = ev_picks[6][0]
        print(f"  EV champion   : ({id_to_seed.get(ev_champ, '?'):>2}) "
              f"{id_to_name.get(ev_champ, str(ev_champ))}  "
              f"{advancement_probs.get(ev_champ, {}).get(6, 0):.2%}")

    # Print Final Four
    if chalk_picks.get(5):
        print(f"\n  Final Four (chalk):")
        for tid in chalk_picks[5]:
            name = id_to_name.get(tid, str(tid))
            seed = id_to_seed.get(tid, "?")
            print(f"    ({seed:>2}) {name}")

    # ── Step 11: Export results ──────────────────────────────────────────
    print("\n" + "=" * 70)
    print("STEP 11 — Exporting results")
    print("=" * 70)

    # Export advancement probabilities CSV
    from src.bracket.output import export_bracket_csv
    csv_path = str(OUTPUT_DIR / "bracket_2026_real.csv")
    export_bracket_csv(advancement_probs, bracket_kaggle, csv_path)
    print(f"  Saved: {csv_path}")

    # Export bracket structure
    bracket_csv_path = str(OUTPUT_DIR / "bracket_2026_real_structure.csv")
    bracket_kaggle.to_csv(bracket_csv_path, index=False)
    print(f"  Saved: {bracket_csv_path}")

    # Export pairwise probabilities JSON
    pairwise_json = {}
    for (a, b), p in win_prob.items():
        lo, hi = min(a, b), max(a, b)
        key = f"{lo}_{hi}"
        if key not in pairwise_json:
            pairwise_json[key] = round(p if a < b else 1 - p, 4)

    pairwise_path = str(OUTPUT_DIR / "pairwise_probs.json")
    with open(pairwise_path, "w") as f:
        json.dump(pairwise_json, f)
    print(f"  Saved: {pairwise_path}")

    # Export bracket data JSON
    bracket_data = {}
    for _, row in bracket_kaggle.iterrows():
        tid = int(row["TeamID"])
        probs = advancement_probs.get(tid, {})
        bracket_data[str(tid)] = {
            "name": row["TeamName"],
            "seed": int(row["Seed"]),
            "region": row["Region"],
            "advancement": {
                ROUND_NAMES.get(r, f"R{r}"): round(p, 4)
                for r, p in probs.items()
            },
        }
    bracket_data_path = str(OUTPUT_DIR / "bracket_data.json")
    with open(bracket_data_path, "w") as f:
        json.dump(bracket_data, f, indent=2)
    print(f"  Saved: {bracket_data_path}")

    # Export compact JSON for bracket.html
    compact = build_bracket_compact_json(bracket_kaggle, advancement_probs, win_prob)
    compact_path = str(OUTPUT_DIR / "bracket_compact.json")
    with open(compact_path, "w") as f:
        json.dump(compact, f, separators=(",", ":"))
    print(f"  Saved: {compact_path}")

    # ── Step 12: Update bracket.html ─────────────────────────────────────
    print("\n" + "=" * 70)
    print("STEP 12 — Updating bracket.html")
    print("=" * 70)

    html_path = OUTPUT_DIR / "bracket.html"
    if html_path.exists():
        html_content = html_path.read_text(encoding="utf-8")
        compact_json_str = json.dumps(compact, separators=(",", ":"))

        # Find the `const RAW = ...;` line and replace the JSON data
        import re as re_mod
        pattern = r"const RAW = .+?;"
        replacement = f"const RAW = {compact_json_str};"

        if "const RAW = " in html_content:
            # Replace just the RAW data line
            lines = html_content.split("\n")
            new_lines = []
            for line in lines:
                if "const RAW = " in line:
                    new_lines.append(f"const RAW = {compact_json_str};")
                else:
                    new_lines.append(line)
            html_content = "\n".join(new_lines)
            html_path.write_text(html_content, encoding="utf-8")
            print(f"  Updated: {html_path}")
        else:
            print(f"  WARNING: Could not find 'const RAW = ' in bracket.html")
    else:
        print(f"  WARNING: bracket.html not found at {html_path}")

    # ── Final summary ────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"\n  Enhanced Model Performance (LOSO CV):")
    print(f"    Log Loss  : {cv_tuned['mean_log_loss']:.4f}  (baseline: {baseline_ll:.4f})")
    print(f"    Accuracy  : {cv_tuned['mean_accuracy']:.3f}  (baseline: {baseline_acc:.3f})")
    print(f"    Brier     : {cv_tuned['mean_brier_score']:.4f}")
    print(f"    AUC       : {cv_tuned['mean_auc']:.4f}")
    print(f"\n  Features used: {len(feature_cols)}")
    print(f"  Training games: {len(tourney_filtered):,}")
    print(f"  XGBoost params: {best_params}")

    if chalk_picks.get(6):
        chalk_champ = chalk_picks[6][0]
        print(f"\n  2026 Predicted Champion: ({id_to_seed.get(chalk_champ, '?')}) "
              f"{id_to_name.get(chalk_champ, str(chalk_champ))}")

    print("\n" + "=" * 70)
    print("  Done! Enhanced model complete.")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
