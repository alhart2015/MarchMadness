"""Enhanced March Madness prediction model v3 -- late-season features, weighted training, line blending.

Extends v2 by adding:
  - Late-season efficiency metrics (vs top-100 opponents)
  - Trajectory / momentum features (efficiency & margin trend slopes)
  - Conference tournament performance features
  - Vegas line trend (late-season spread delta)
  - Weighted training data (tournament + supplemental late-season reg season)
  - R64 game-specific Vegas line blending into pairwise probabilities

Usage
-----
    python src/enhanced_model_v3.py
"""

import json
import logging
import re
import sys
import time
import warnings
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

# -- Suppress noisy warnings -------------------------------------------------
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# -- Logging setup ------------------------------------------------------------
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# -- Path setup ---------------------------------------------------------------
_HERE = Path(__file__).resolve().parent          # src/
_ROOT = _HERE.parent                             # project root
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

MANIA_DIR   = _ROOT / "data" / "raw" / "march-machine-learning-2026"
KAGGLE_DIR  = _ROOT / "data" / "raw" / "kaggle"
VEGAS_DIR   = _ROOT / "data" / "raw" / "vegas_lines"
BRACKET_CSV = _ROOT / "data" / "raw" / "bracket_2026.csv"
OUTPUT_DIR  = _ROOT / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# -- Constants ----------------------------------------------------------------
MASSEY_SYSTEMS = ["POM", "SAG", "MOR", "WOL", "DOL", "COL", "RPI"]
FIRST_ROUND_PAIRS = [(1, 16), (8, 9), (5, 12), (4, 13), (6, 11), (3, 14), (7, 10), (2, 15)]
ROUND_NAMES = {1: "R64", 2: "R32", 3: "S16", 4: "E8", 5: "F4", 6: "Champ"}


# =============================================================================
# NEW v3 IMPORTS
# =============================================================================

from src.features.late_season import (
    compute_late_season_metrics,
    compute_trajectory_features,
    compute_conf_tourney_features,
    compute_vegas_trend,
)
from src.features.coach import compute_coach_features
from src.features.team_history import compute_team_history_features
# quality_wins (src/features/quality_wins.py) was tested in a backtest
# (22 seasons): added 0 R64/R32 signal, dropped F4 accuracy by 9pp. Dropped
# from the pipeline. Kept as a module for reference / future variants.
from src.bracket.line_blending import blend_r64_probs
from src.models.matchup import build_weighted_matchup_data


# =============================================================================
# ABLATION HELPERS
# =============================================================================

def apply_feature_drop(feature_cols, drop_env):
    """Filter feature_cols by names listed in MM_FEATURE_DROP env-var string.

    Returns (filtered_cols, missing_names_set). Unknown names are returned in
    `missing` for the caller to log -- not raised, so a typo does not abort
    a multi-hour LOSO retrain.
    """
    if not drop_env:
        return list(feature_cols), set()
    drop = {c.strip() for c in drop_env.split(",") if c.strip()}
    present = drop & set(feature_cols)
    missing = drop - set(feature_cols)
    filtered = [c for c in feature_cols if c not in present]
    return filtered, missing


def apply_output_suffix(path, suffix):
    """Insert `suffix` before the final extension of `path`. Empty suffix = no-op.

    Uses os.path.splitext so only the trailing extension is split, even if
    intermediate directory names contain dots.
    """
    if not suffix:
        return path
    import os
    root, ext = os.path.splitext(path)
    return f"{root}{suffix}{ext}"


# =============================================================================
# VEGAS LINE PROCESSING
# =============================================================================

def _build_vegas_name_to_kaggle_map(
    teams: pd.DataFrame,
    spellings: pd.DataFrame,
) -> dict:
    """Build a lookup mapping normalised team names -> Kaggle TeamID.

    Combines MTeamSpellings (primary lookup) and MTeams (fallback).
    Returns {lowercase_name: kaggle_id}.
    """
    name_to_id: Dict[str, int] = {}

    # Primary: MTeamSpellings
    for _, row in spellings.iterrows():
        name = str(row["TeamNameSpelling"]).lower().strip().replace("_", " ")
        tid = int(row["TeamID"])
        name_to_id[name] = tid

    # Also add the canonical MTeams names
    for _, row in teams.iterrows():
        name = str(row["TeamName"]).lower().strip()
        tid = int(row["TeamID"])
        if name not in name_to_id:
            name_to_id[name] = tid

    return name_to_id


def _resolve_vegas_name(
    raw_name: str,
    name_to_id: dict,
    fuzzy_cache: dict,
) -> int | None:
    """Resolve a Prediction Tracker team name to a Kaggle TeamID.

    Uses exact match first, then falls back to rapidfuzz.
    Returns TeamID or None if no match found.
    """
    # Normalise
    norm = raw_name.lower().strip().replace("_", " ")

    # Exact match
    if norm in name_to_id:
        return name_to_id[norm]

    # Check cache
    if norm in fuzzy_cache:
        return fuzzy_cache[norm]

    # Fuzzy match
    from rapidfuzz import fuzz, process
    result = process.extractOne(
        norm,
        list(name_to_id.keys()),
        scorer=fuzz.token_sort_ratio,
    )
    if result and result[1] >= 80:
        tid = name_to_id[result[0]]
        fuzzy_cache[norm] = tid
        return tid

    fuzzy_cache[norm] = None
    return None


def _vegas_file_to_season(filename: str) -> int:
    """Convert filename like 'ncaabb25.csv' to Kaggle season year (2026).

    The Prediction Tracker file ncaabbYY.csv covers the season that *starts*
    in fall of 20YY and ends in spring of 20(YY+1). Kaggle's Season field
    uses the spring year. So ncaabb25 -> 2026, ncaabb03 -> 2004.
    """
    match = re.search(r"ncaabb(\d{2})\.csv", filename)
    if not match:
        return 0
    yy = int(match.group(1))
    return 2000 + yy + 1


def filter_vegas_to_pre_tournament(
    vegas_df: pd.DataFrame,
    seasons_csv_path: Path | None = None,
) -> pd.DataFrame:
    """Drop rows whose daynum (date - DayZero[season]) is >= 134.

    134 is the First Four day in Kaggle's DayNum convention. Anything
    from the First Four onward is NCAA tournament and must NOT feed
    into v4's per-team-per-season Vegas aggregates -- otherwise it
    leaks tournament outcomes into LOSO test features.

    Returns a copy with the same schema as `vegas_df`. Rows whose
    season is missing from MSeasons.csv or whose date is unparseable
    are KEPT (defensive: a data hiccup must not silently delete
    legitimate regular-season rows). Both cases emit a warning.

    Spec: docs/superpowers/specs/2026-05-04-v4-vegas-leak-fix-design.md
    """
    if seasons_csv_path is None:
        seasons_csv_path = MANIA_DIR / "MSeasons.csv"

    if vegas_df.empty:
        return vegas_df.copy()

    seasons = pd.read_csv(seasons_csv_path)
    day_zero: dict[int, datetime] = {}
    for _, r in seasons.iterrows():
        try:
            day_zero[int(r["Season"])] = datetime.strptime(
                str(r["DayZero"]).strip(), "%m/%d/%Y"
            )
        except (ValueError, TypeError):
            continue

    out_mask = []
    n_unknown_season = 0
    n_unparseable_date = 0
    for season, date_str in zip(vegas_df["season"], vegas_df["date"]):
        dz = day_zero.get(int(season))
        if dz is None:
            n_unknown_season += 1
            out_mask.append(True)
            continue
        try:
            d = datetime.strptime(str(date_str).strip(), "%m/%d/%Y")
        except (ValueError, TypeError):
            n_unparseable_date += 1
            out_mask.append(True)
            continue
        daynum = (d - dz).days
        out_mask.append(daynum < 134)

    if n_unknown_season:
        unknown_seasons = sorted({int(s) for s in vegas_df["season"]
                                   if int(s) not in day_zero})
        print(f"  warning: {n_unknown_season} Vegas rows have unknown "
              f"DayZero (seasons {unknown_seasons}); keeping them")
    if n_unparseable_date:
        print(f"  warning: {n_unparseable_date} Vegas rows have "
              f"unparseable dates; keeping them")

    return vegas_df.loc[pd.Series(out_mask, index=vegas_df.index)].copy()


def load_vegas_lines() -> pd.DataFrame:
    """Load all Vegas lines CSVs and return a unified DataFrame.

    Columns: season, date, home, road, hscore, rscore, line, neutral
    """
    all_frames = []

    for fpath in sorted(VEGAS_DIR.glob("ncaabb*.csv")):
        season = _vegas_file_to_season(fpath.name)
        if season == 0:
            continue

        try:
            df = pd.read_csv(fpath, encoding="latin-1")
        except Exception as e:
            logger.warning("Failed to read %s: %s", fpath, e)
            continue

        # Strip quotes from column names & values
        df.columns = [c.strip().strip('"') for c in df.columns]

        # Keep only the columns we need
        keep = ["date", "home", "hscore", "road", "rscore", "line"]
        if "neutral" in df.columns:
            keep.append("neutral")

        # Check required columns exist
        missing = [c for c in ["date", "home", "road", "line"] if c not in df.columns]
        if missing:
            logger.warning("Skipping %s -- missing columns: %s", fpath.name, missing)
            continue

        df = df[[c for c in keep if c in df.columns]].copy()
        df["season"] = season

        # Parse line as numeric, coerce errors to NaN
        df["line"] = pd.to_numeric(df["line"], errors="coerce")

        # Parse scores
        for col in ["hscore", "rscore"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Parse neutral
        if "neutral" in df.columns:
            df["neutral"] = pd.to_numeric(df["neutral"], errors="coerce").fillna(0).astype(int)
        else:
            df["neutral"] = 0

        # Clean team names: strip quotes and whitespace
        for col in ["home", "road"]:
            df[col] = df[col].astype(str).str.strip().str.strip('"')

        all_frames.append(df)

    if not all_frames:
        return pd.DataFrame()

    combined = pd.concat(all_frames, ignore_index=True)
    # Drop rows without a valid line
    combined = combined.dropna(subset=["line"])

    return combined


def compute_vegas_features(
    vegas_df: pd.DataFrame,
    teams: pd.DataFrame,
    spellings: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    """Compute per-team per-season aggregate features from Vegas lines.

    Returns:
        (features_df, name_resolution) where features_df has columns:
            TeamID, Season, vegas_avg_spread, vegas_avg_margin, vegas_ats_pct,
            vegas_power_rating, vegas_consistency, vegas_game_count
        and name_resolution is {vegas_name: kaggle_team_id}.
    """
    print("  Building Vegas team name -> Kaggle ID mapping...")
    name_to_id = _build_vegas_name_to_kaggle_map(teams, spellings)
    fuzzy_cache: dict = {}

    # Pre-resolve all unique team names
    all_names = set(vegas_df["home"].unique()) | set(vegas_df["road"].unique())
    name_resolution: dict = {}
    resolved = 0
    for name in all_names:
        tid = _resolve_vegas_name(name, name_to_id, fuzzy_cache)
        if tid is not None:
            name_resolution[name] = tid
            resolved += 1

    print(f"  Resolved {resolved} / {len(all_names)} unique Vegas team names to Kaggle IDs")

    # Map team names to IDs in the dataframe
    vegas_df = vegas_df.copy()
    vegas_df["home_id"] = vegas_df["home"].map(name_resolution)
    vegas_df["road_id"] = vegas_df["road"].map(name_resolution)

    # Drop rows where either team couldn't be matched
    matched = vegas_df.dropna(subset=["home_id", "road_id"]).copy()
    matched["home_id"] = matched["home_id"].astype(int)
    matched["road_id"] = matched["road_id"].astype(int)

    print(f"  Games with both teams matched: {len(matched):,} / {len(vegas_df):,}")

    # Compute actual margin (home perspective)
    matched["actual_margin"] = matched["hscore"] - matched["rscore"]

    # Build per-team game records (from each team's perspective)
    # team_spread < 0 => team is favored (the more negative, the bigger favorite)
    # team_spread > 0 => team is an underdog
    # For home team: team_spread = -line (if line=15 and home favored, home_spread = -15)
    # For road team: team_spread = +line (if line=15 and home favored, road_spread = +15)

    home_records = pd.DataFrame({
        "TeamID": matched["home_id"],
        "Season": matched["season"],
        "team_spread": -matched["line"],
        "actual_margin": matched["actual_margin"],
        "neutral": matched["neutral"],
    })

    road_records = pd.DataFrame({
        "TeamID": matched["road_id"],
        "Season": matched["season"],
        "team_spread": matched["line"],
        "actual_margin": -matched["actual_margin"],
        "neutral": matched["neutral"],
    })

    all_records = pd.concat([home_records, road_records], ignore_index=True)

    # Drop records with NaN in key fields
    all_records = all_records.dropna(subset=["team_spread", "actual_margin"])

    # Compute per-team per-season features
    rows = []
    for (tid, season), grp in all_records.groupby(["TeamID", "Season"]):
        n = len(grp)
        if n < 3:
            # Skip teams with too few games (unreliable)
            continue

        spreads = grp["team_spread"].values
        margins = grp["actual_margin"].values

        # Against-the-spread: cover when actual_margin + team_spread > 0
        ats_margin = margins + spreads  # positive = covered the spread
        ats_wins = np.sum(ats_margin > 0)
        ats_losses = np.sum(ats_margin < 0)
        ats_total = ats_wins + ats_losses
        ats_pct = ats_wins / ats_total if ats_total > 0 else 0.5

        # Recency-weighted average spread (exponential decay, half-life ~30 games)
        decay = np.exp(-np.log(2) * np.arange(n)[::-1] / max(n / 2, 1))
        decay /= decay.sum()
        weighted_spread = np.sum(spreads * decay)

        rows.append({
            "TeamID": int(tid),
            "Season": int(season),
            "vegas_avg_spread": np.mean(spreads),
            "vegas_avg_margin": np.mean(margins),
            "vegas_ats_pct": ats_pct,
            "vegas_power_rating": weighted_spread,
            "vegas_consistency": np.std(margins - (-spreads)),
            "vegas_game_count": n,
        })

    result = pd.DataFrame(rows)
    print(f"  Computed Vegas features for {len(result):,} team-seasons")

    # Summary by season
    if not result.empty:
        season_counts = result.groupby("Season").size()
        print(f"  Seasons covered: {int(season_counts.index.min())}-{int(season_counts.index.max())}")
        print(f"  Teams per season: {season_counts.mean():.0f} avg, {season_counts.min()}-{season_counts.max()} range")

    return result, name_resolution


# =============================================================================
# v3 HELPER FUNCTIONS
# =============================================================================

def _build_vegas_team_records_with_dates(vegas_df, name_resolution):
    """Build per-team game records with dates from raw Vegas data.

    Returns DataFrame with columns: TeamID, Season, date, team_spread
    """
    records = []
    for _, row in vegas_df.iterrows():
        home_id = name_resolution.get(row["home"])
        road_id = name_resolution.get(row["road"])
        if home_id is None or road_id is None:
            continue
        line = row["line"]
        if pd.isna(line):
            continue
        records.append({"TeamID": home_id, "Season": row["season"],
                        "date": row["date"], "team_spread": -line})
        records.append({"TeamID": road_id, "Season": row["season"],
                        "date": row["date"], "team_spread": line})
    return pd.DataFrame(records)


def _build_r64_lines(vegas_df, name_resolution, bracket_kaggle):
    """Extract R64 game-specific Vegas lines for bracket teams.

    Returns {(team_a_id, team_b_id): spread} for neutral-site games
    involving two bracket teams.
    """
    latest = vegas_df[vegas_df["neutral"] == 1].copy()
    if latest.empty:
        return {}
    bracket_ids = set(bracket_kaggle["TeamID"].tolist())
    r64_lines = {}
    for _, row in latest.iterrows():
        home_id = name_resolution.get(row["home"])
        road_id = name_resolution.get(row["road"])
        line = row["line"]
        if home_id is None or road_id is None or pd.isna(line):
            continue
        if home_id in bracket_ids and road_id in bracket_ids:
            r64_lines[(home_id, road_id)] = line
    return r64_lines


def _get_top_n_team_ids(kenpom, kp_to_kaggle, massey, season, n=100):
    """Get top-N team IDs for a season from KenPom KADJ EM RANK, with Massey fallback."""
    kp_season = kenpom[kenpom["YEAR"] == season]
    if not kp_season.empty and "KADJ EM RANK" in kp_season.columns:
        top_kp = kp_season[kp_season["KADJ EM RANK"] <= n]
        top_ids = set()
        for _, row in top_kp.iterrows():
            kp_id = int(row["TEAM NO"])
            kaggle_id = kp_to_kaggle.get(kp_id)
            if kaggle_id is not None:
                top_ids.add(kaggle_id)
        if top_ids:
            return top_ids

    # Fallback: Massey composite for seasons without KenPom
    if massey is not None and not massey.empty:
        season_massey = massey[massey["Season"] == season]
        if not season_massey.empty:
            # Use POM system first, then average ranks across systems
            composite = season_massey.groupby("TeamID")["OrdinalRank"].mean().nsmallest(n)
            return set(composite.index.astype(int))

    return set()


# =============================================================================
# REUSE EXISTING MODEL INFRASTRUCTURE
# =============================================================================

# Import functions from the original enhanced model
from src.enhanced_model import (
    load_all_data,
    build_kenpom_to_kaggle_map,
    compute_all_features,
    get_feature_cols,
    build_matchup_data_from_kaggle,
    leave_one_season_out_cv,
    precompute_win_probs,
    simulate_tournament_fast,
    get_advancement_probabilities,
    build_bracket_compact_json,
    print_champion_probs,
    print_advancement_table,
)


# =============================================================================
# WEIGHTED LOSO CV
# =============================================================================

def leave_one_season_out_cv_weighted(
    feature_matrix: pd.DataFrame,
    tourney_results: pd.DataFrame,
    regular_results: pd.DataFrame,
    feature_cols: list,
    top_n_team_ids_by_season: dict,
    xgb_params: dict = None,
    random_seed: int = 42,
    supplemental_weight: float = 0.25,
    allowed_holdouts: list[int] | None = None,
) -> dict:
    """Run LOSO CV using weighted matchup data (tournament + supplemental).

    If allowed_holdouts is provided, restrict the iteration to those
    seasons (used by diagnostic gates for cheap subsets); training on
    each iteration still uses ALL non-holdout seasons.
    """
    from sklearn.metrics import log_loss as sklearn_log_loss, roc_auc_score
    from src.models.train import train_model

    seasons = sorted(tourney_results["Season"].unique())
    seasons = [s for s in seasons if s >= 2003]
    if allowed_holdouts is not None:
        seasons = [s for s in seasons if s in set(allowed_holdouts)]

    results = []

    for holdout in seasons:
        train_tourney = tourney_results[tourney_results["Season"] != holdout]
        test_tourney = tourney_results[tourney_results["Season"] == holdout]

        if len(test_tourney) == 0:
            continue

        # Build training set top-N IDs (union across all training seasons)
        train_top_ids = set()
        for s in train_tourney["Season"].unique():
            train_top_ids |= top_n_team_ids_by_season.get(int(s), set())

        # Weighted training data
        train_reg = regular_results[regular_results["Season"] != holdout]
        X_train, y_train, w_train = build_weighted_matchup_data(
            feature_matrix, train_tourney, train_reg, feature_cols,
            top_n_team_ids=train_top_ids,
            supplemental_weight=supplemental_weight,
        )

        # Test data: tournament only (no weights needed for evaluation)
        X_test, y_test, _ = build_matchup_data_from_kaggle(
            feature_matrix, test_tourney, feature_cols
        )

        if len(X_train) == 0 or len(X_test) == 0:
            continue

        # Fill NaN with training median
        medians = X_train.median()
        X_train = X_train.fillna(medians)
        X_test = X_test.fillna(medians)

        model = train_model(
            X_train, y_train, random_seed=random_seed,
            xgb_params=xgb_params, sample_weight=w_train,
        )
        y_prob = model.predict_proba(X_test)[:, 1]

        # Optional: save full pairwise probs for the year's field (backtest).
        import os as _os
        from pathlib import Path as _Path
        _pw_out = _os.environ.get("MM_PAIRWISE_OUT")
        if _pw_out:
            from src.models.matchup import build_matchup_features, expand_feature_cols
            _field = sorted(set(test_tourney["WTeamID"]) | set(test_tourney["LTeamID"]))
            _fm_yr = feature_matrix[feature_matrix["Season"] == holdout].set_index("TeamID")
            _have_feats = [t for t in _field if t in _fm_yr.index]
            _pair_rows, _pair_ids = [], []
            for _i in range(len(_have_feats)):
                for _j in range(_i + 1, len(_have_feats)):
                    _a, _b = _have_feats[_i], _have_feats[_j]
                    _av = _fm_yr.loc[_a, feature_cols].values.astype(float)
                    _bv = _fm_yr.loc[_b, feature_cols].values.astype(float)
                    _pair_rows.append(build_matchup_features(_av, _bv))
                    _pair_ids.append((_a, _b))
            if _pair_rows:
                _pdf = pd.DataFrame(_pair_rows, columns=expand_feature_cols(feature_cols)).fillna(medians)
                _pp = model.predict_proba(_pdf)[:, 1]
                _out = pd.DataFrame({
                    "season": holdout,
                    "team_a": [a for a, _ in _pair_ids],
                    "team_b": [b for _, b in _pair_ids],
                    "p_a_wins": _pp,
                })
                _out.to_csv(_pw_out, mode="a", index=False,
                            header=not _Path(_pw_out).exists())

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


# =============================================================================
# MAIN
# =============================================================================

def prepare_loso_inputs() -> dict:
    """Build the v3/v4 feature matrix, training data, and per-season top-80
    team ID sets for use by any LOSO-loop trainer. This is the data-setup
    half of v4's main() extracted as a callable so parallel trainers (e.g.,
    the LR stage-1 in src/train_lr_stage1.py) can reuse the byte-identical
    inputs.

    Returns dict with keys:
        feature_matrix      -- pd.DataFrame with TeamID, Season, all features
        tourney_filtered    -- pd.DataFrame of tournament results filtered
                               to seasons present in feature_matrix
        regular_results     -- pd.DataFrame of regular-season results
                               (data["reg_season"] in v4 main)
        feature_cols        -- list of feature column names (post-NaN-prune)
        top_80_by_season    -- dict[int -> set[int]] of top-80 team IDs
                               per season, used by the weighted matchup
                               builder to mark supplemental rows
        feature_medians     -- pd.Series of per-feature medians from the
                               weighted-matchup X_all (used to fill NaNs
                               in apply-time pair construction)
    """
    import os as _os

    # -- Step 1: Load all base data ----------------------------------------
    data = load_all_data()

    # Load conference tournament data (new in v3)
    data["conf_tourney"] = pd.read_csv(
        MANIA_DIR / "MConferenceTourneyGames.csv"
    )
    print(f"  Conference tourney   : {len(data['conf_tourney']):,}")

    # Load team-coach mapping (for v4 coach features)
    data["team_coaches"] = pd.read_csv(MANIA_DIR / "MTeamCoaches.csv")
    print(f"  Team-coach rows      : {len(data['team_coaches']):,}")

    # -- Build KenPom -> Kaggle map early (needed for top-N filtering) -----
    kp_to_kaggle = build_kenpom_to_kaggle_map(
        data["kenpom"], data["teams"], data["spellings"]
    )
    print(f"  KenPom->Kaggle map   : {len(kp_to_kaggle)} teams mapped")

    # -- Step 2: Compute base features (same as enhanced_model.py) ---------
    feature_matrix = compute_all_features(data)

    # -- Step 3: Load and compute Vegas features ---------------------------
    print("\n" + "=" * 70)
    print("STEP 3 -- Loading and computing Vegas line features")
    print("=" * 70)

    vegas_df = load_vegas_lines()
    print(f"  Loaded {len(vegas_df):,} Vegas line records across {vegas_df['season'].nunique()} seasons")

    # Drop NCAA tournament games before per-team-per-season aggregation.
    # Otherwise season S tournament outcomes leak into season S feature
    # rows at LOSO test time. Keep `vegas_df` (full) for the R64 line-
    # blending consumer downstream, which intentionally uses tournament
    # lines. Spec: docs/superpowers/specs/2026-05-04-v4-vegas-leak-fix-design.md
    vegas_df_pre_tourney = filter_vegas_to_pre_tournament(vegas_df)
    print(f"  Filtered to {len(vegas_df_pre_tourney):,} pre-tournament rows "
          f"({len(vegas_df) - len(vegas_df_pre_tourney):,} tournament rows dropped)")

    vegas_features, name_resolution = compute_vegas_features(
        vegas_df_pre_tourney, data["teams"], data["spellings"]
    )

    # -- Step 3a: Merge Vegas features into feature matrix -----------------
    print("\n" + "=" * 70)
    print("STEP 3a -- Merging Vegas features into feature matrix")
    print("=" * 70)

    pre_merge_cols = len(feature_matrix.columns)
    feature_matrix = feature_matrix.merge(
        vegas_features,
        on=["TeamID", "Season"],
        how="left",
    )
    post_merge_cols = len(feature_matrix.columns)

    # Stats on merge coverage
    vegas_cols = [c for c in feature_matrix.columns if c.startswith("vegas_")]
    n_with_vegas = feature_matrix[vegas_cols[0]].notna().sum() if vegas_cols else 0
    n_total = len(feature_matrix)
    print(f"  Added {post_merge_cols - pre_merge_cols} Vegas feature columns")
    print(f"  Teams with Vegas data: {n_with_vegas} / {n_total} ({100*n_with_vegas/n_total:.1f}%)")
    print(f"  Vegas feature columns: {vegas_cols}")

    # -- Step 3b: Compute new v3 features ----------------------------------
    print("\n" + "=" * 70)
    print("STEP 3b -- Computing late-season, trajectory, conf tourney, and Vegas trend features")
    print("=" * 70)

    # Build Vegas team records with dates for the trend computation.
    # Use the pre-tournament-filtered df for the same reason as above.
    vegas_team_records = _build_vegas_team_records_with_dates(vegas_df_pre_tourney, name_resolution)

    seasons = sorted(feature_matrix["Season"].unique())
    all_late_season = []
    all_trajectory = []
    all_conf_tourney = []
    all_vegas_trend = []

    for season in seasons:
        # Top-100 teams for this season (for late-season quality opponent filter)
        top_100 = _get_top_n_team_ids(
            data["kenpom"], kp_to_kaggle, data["massey"], season, n=100
        )

        # Late-season metrics (vs top-100 opponents)
        if top_100:
            late_df = compute_late_season_metrics(
                data["reg_season"], season, top_100
            )
            if not late_df.empty:
                all_late_season.append(late_df)

        # Trajectory features
        traj_df = compute_trajectory_features(data["reg_season"], season)
        if not traj_df.empty:
            all_trajectory.append(traj_df)

        # Conference tournament features
        conf_df = compute_conf_tourney_features(data["conf_tourney"], season)
        if not conf_df.empty:
            all_conf_tourney.append(conf_df)

        # Vegas trend
        if not vegas_team_records.empty:
            vtrend_df = compute_vegas_trend(vegas_team_records, season)
            if not vtrend_df.empty:
                all_vegas_trend.append(vtrend_df)

    # Merge all new features into feature matrix
    new_feature_names = []

    if all_late_season:
        late_season_df = pd.concat(all_late_season, ignore_index=True)
        feature_matrix = feature_matrix.merge(
            late_season_df, on=["TeamID", "Season"], how="left"
        )
        new_feature_names.extend(["late_adj_oe", "late_adj_de", "late_adj_em", "late_sos"])
        print(f"  Late-season metrics: {len(late_season_df):,} team-seasons")

    if all_trajectory:
        trajectory_df = pd.concat(all_trajectory, ignore_index=True)
        feature_matrix = feature_matrix.merge(
            trajectory_df, on=["TeamID", "Season"], how="left"
        )
        new_feature_names.extend(["efficiency_trend", "margin_trend"])
        print(f"  Trajectory features: {len(trajectory_df):,} team-seasons")

    if all_conf_tourney:
        conf_tourney_df = pd.concat(all_conf_tourney, ignore_index=True)
        feature_matrix = feature_matrix.merge(
            conf_tourney_df, on=["TeamID", "Season"], how="left"
        )
        new_feature_names.extend(["conf_tourney_wins", "conf_tourney_champ"])
        print(f"  Conf tourney features: {len(conf_tourney_df):,} team-seasons")

    if all_vegas_trend:
        vegas_trend_df = pd.concat(all_vegas_trend, ignore_index=True)
        feature_matrix = feature_matrix.merge(
            vegas_trend_df, on=["TeamID", "Season"], how="left"
        )
        new_feature_names.extend(["vegas_late_spread_delta"])
        print(f"  Vegas trend features: {len(vegas_trend_df):,} team-seasons")

    # Coach tournament-history features (cross-season cumulative).
    coach_df = compute_coach_features(data["tourney"], data["team_coaches"])
    if not coach_df.empty:
        feature_matrix = feature_matrix.merge(
            coach_df, on=["TeamID", "Season"], how="left"
        )
        coach_feat_names = ["coach_career_games", "coach_career_wins",
                             "coach_career_winpct", "coach_career_f4_apps",
                             "coach_career_champs", "coach_career_seasons"]
        new_feature_names.extend(coach_feat_names)
        print(f"  Coach features: {len(coach_df):,} team-seasons")

    # Team-program tournament-history features (cross-season, team-keyed).
    # Spec: docs/superpowers/specs/2026-05-09-team-seed-residual-design.md
    th_df = compute_team_history_features(
        tournament_field=feature_matrix[["Season", "TeamID"]].drop_duplicates(),
        tourney_results=data["tourney"],
        seeds=data["seeds"],
        window_years=10,
    )
    if not th_df.empty:
        feature_matrix = feature_matrix.merge(
            th_df, on=["TeamID", "Season"], how="left",
        )
        th_feat_names = ["team_seed_residual_mean_10yr",
                         "team_seed_residual_ewma_hl2"]
        new_feature_names.extend(th_feat_names)
        print(f"  Team history features: {len(th_df):,} team-seasons")

    # quality_wins block removed: 22-season backtest showed -93 bracket pts vs
    # the v3+coach baseline, with F4 accuracy dropping 9pp. Signal already
    # captured by KenPom/SOS features.

    print(f"  New v3 feature columns: {new_feature_names}")

    # -- Step 4: Get feature columns and prepare data ----------------------
    feature_cols = get_feature_cols(feature_matrix)
    print(f"\n  Total feature columns ({len(feature_cols)}):")
    for i in range(0, len(feature_cols), 6):
        print(f"    {', '.join(feature_cols[i:i+6])}")

    # ABLATION HOOK: drop features named in MM_FEATURE_DROP env var.
    _drop_env = _os.environ.get("MM_FEATURE_DROP", "")
    if _drop_env:
        _before = len(feature_cols)
        feature_cols, _missing = apply_feature_drop(feature_cols, _drop_env)
        if _missing:
            print(f"  ABLATION WARNING: MM_FEATURE_DROP names not in feature_cols: {sorted(_missing)}")
        print(f"  ABLATION: dropped {_before - len(feature_cols)} features (drop list: {_drop_env}); remaining: {len(feature_cols)}")

    # -- Step 5: Build weighted matchup training data ----------------------
    print("\n" + "=" * 70)
    print("STEP 5 -- Building weighted matchup training data")
    print("=" * 70)

    tourney = data["tourney"]
    fm_seasons = set(feature_matrix["Season"].unique())
    tourney_filtered = tourney[tourney["Season"].isin(fm_seasons)]

    # Build top-80 team IDs per season for weighted matchup data
    top_80_by_season = {}
    for season in sorted(fm_seasons):
        top_80_by_season[int(season)] = _get_top_n_team_ids(
            data["kenpom"], kp_to_kaggle, data["massey"], int(season), n=80
        )
    all_top_80 = set()
    for ids in top_80_by_season.values():
        all_top_80 |= ids

    X_all, y_all, weights_all = build_weighted_matchup_data(
        feature_matrix, tourney_filtered, data["reg_season"], feature_cols,
        top_n_team_ids=all_top_80,
        supplemental_weight=0.25,
    )

    # Drop columns with > 30% NaN. X_all has expanded matchup columns
    # (<feat>_diff and <feat>_avg); we drop the raw feature if either of
    # its expanded forms is too sparse, then rebuild the expanded list.
    if not X_all.empty:
        from src.models.matchup import expand_feature_cols as _expand
        null_fracs = X_all.isna().mean()
        drop_expanded = null_fracs[null_fracs > 0.30].index.tolist()
        drop_raw = {c.removesuffix("_diff") for c in drop_expanded
                    if c.endswith("_diff")}
        if drop_raw:
            print(f"  Dropping {len(drop_raw)} high-NaN raw features: {sorted(drop_raw)}")
            feature_cols = [c for c in feature_cols if c not in drop_raw]
            X_all = X_all[_expand(feature_cols)]

    # Fill remaining NaN with median
    medians = X_all.median()
    X_all = X_all.fillna(medians)

    n_tourney = int((weights_all >= 1.0).sum())
    n_supplemental = int((weights_all < 1.0).sum())
    print(f"  Training samples : {len(X_all):,}  (tourney: {n_tourney}, supplemental: {n_supplemental})")
    print(f"  Features used    : {len(feature_cols)}")

    return {
        "feature_matrix": feature_matrix,
        "tourney_filtered": tourney_filtered,
        "regular_results": data["reg_season"],
        "feature_cols": feature_cols,
        "top_80_by_season": top_80_by_season,
        "feature_medians": medians,
        # Additional artifacts retained so v4's main() can continue without
        # rebuilding them. Not part of the public contract for stage-1
        # ensemble trainers (those should rely only on the documented keys
        # above), but kept here so this extraction is a pure refactor.
        "_data": data,
        "_kp_to_kaggle": kp_to_kaggle,
        "_vegas_df": vegas_df,
        "_name_resolution": name_resolution,
        "_X_all": X_all,
        "_y_all": y_all,
        "_weights_all": weights_all,
    }


def main():
    overall_start = time.time()

    import os as _os
    _output_suffix = _os.environ.get("MM_OUTPUT_SUFFIX", "")
    if _output_suffix:
        print(f"  ABLATION: output suffix = '{_output_suffix}'")

    print("\n" + "=" * 70)
    print("ENHANCED MODEL v3 -- Late-Season Features, Weighted Training, Line Blending")
    print("=" * 70)

    inputs = prepare_loso_inputs()
    feature_matrix = inputs["feature_matrix"]
    tourney_filtered = inputs["tourney_filtered"]
    regular_results = inputs["regular_results"]
    feature_cols = inputs["feature_cols"]
    top_80_by_season = inputs["top_80_by_season"]
    medians = inputs["feature_medians"]
    # Internal artifacts that the rest of main() still needs.
    data = inputs["_data"]
    kp_to_kaggle = inputs["_kp_to_kaggle"]
    vegas_df = inputs["_vegas_df"]
    name_resolution = inputs["_name_resolution"]
    X_all = inputs["_X_all"]
    y_all = inputs["_y_all"]
    weights_all = inputs["_weights_all"]

    # -- Step 6: LOSO CV with default params (weighted) --------------------
    print("\n" + "=" * 70)
    print("STEP 6 -- Leave-one-season-out CV (default params, weighted)")
    print("=" * 70)

    # Fill NaN in feature matrix for CV
    fm_filled = feature_matrix.copy()
    for col in feature_cols:
        if col in fm_filled.columns:
            fm_filled[col] = fm_filled[col].fillna(fm_filled[col].median())

    import os as _os_step6
    if _os_step6.environ.get("MM_SKIP_DEFAULT_LOSO"):
        # Default-params LOSO is a sanity-check baseline only; its pairwise
        # rows are dedup'd away (keep="last") by every downstream consumer
        # of pairwise_v4.csv. Skipping it ~halves clean-regen runtime.
        print("  MM_SKIP_DEFAULT_LOSO set -- skipping default-params LOSO.")
    else:
        cv_default = leave_one_season_out_cv_weighted(
            fm_filled, tourney_filtered, data["reg_season"], feature_cols,
            top_n_team_ids_by_season=top_80_by_season,
            random_seed=42,
            supplemental_weight=0.25,
        )

        print(f"\n  Default params CV results (weighted):")
        print(f"  Mean Log Loss  : {cv_default['mean_log_loss']:.4f}")
        print(f"  Mean Brier     : {cv_default['mean_brier_score']:.4f}")
        print(f"  Mean Accuracy  : {cv_default['mean_accuracy']:.3f}")
        print(f"  Mean AUC       : {cv_default['mean_auc']:.4f}")

    # -- Step 7: Optuna hyperparameter tuning ------------------------------
    print("\n" + "=" * 70)
    print("STEP 7 -- Optuna hyperparameter tuning (30 trials)")
    print("=" * 70)

    import os
    if os.environ.get("MM_TUNED_PARAMS_V3"):
        import json as _json
        best_params = _json.loads(os.environ["MM_TUNED_PARAMS_V3"])
        print(f"  Using cached tuned params from env (skipping Optuna): {best_params}")
    else:
        from src.models.tuning import tune_hyperparameters
        best_params = tune_hyperparameters(X_all, y_all, n_trials=30, random_seed=42)
        print(f"  Best params: {best_params}")
        # Persist tuned params so downstream tools (ablation driver, etc.)
        # can reuse them via MM_TUNED_PARAMS_V3 without re-running Optuna.
        # Not suffixed: shared input across ablations, not a per-run output.
        import json as _json
        _params_path = "output/v4_tuned_params.json"
        with open(_params_path, "w") as _f:
            _json.dump(best_params, _f, indent=2)
        print(f"  Saved tuned params to {_params_path}")

    # -- Step 8: Re-evaluate with tuned params (weighted) ------------------
    print("\n" + "=" * 70)
    print("STEP 8 -- Leave-one-season-out CV (tuned params, weighted)")
    print("=" * 70)

    cv_tuned = leave_one_season_out_cv_weighted(
        fm_filled, tourney_filtered, data["reg_season"], feature_cols,
        top_n_team_ids_by_season=top_80_by_season,
        xgb_params=best_params, random_seed=42,
        supplemental_weight=0.25,
    )
    cv_path = apply_output_suffix("output/cv_per_season_v3.csv", _output_suffix)
    cv_tuned["per_season"].to_csv(cv_path, index=False)
    print(f"  Saved: {cv_path}")

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

    # -- Model comparison --------------------------------------------------
    print("\n" + "=" * 70)
    print("MODEL COMPARISON: v2 (enhanced + Vegas) vs v3 (+ late-season, weighted)")
    print("=" * 70)

    # The baseline from enhanced_model_v2.py
    baseline_v2_ll = 0.4560
    baseline_v2_acc = 0.735
    new_ll = cv_tuned["mean_log_loss"]
    new_acc = cv_tuned["mean_accuracy"]

    print(f"\n  {'Metric':<20}  {'v2 (+Vegas)':>14}  {'v3 (+late,wt)':>14}  {'Delta':>10}")
    print(f"  {'-'*63}")
    print(f"  {'Log Loss':<20}  {baseline_v2_ll:>14.4f}  {new_ll:>14.4f}  {new_ll - baseline_v2_ll:>+10.4f}")
    print(f"  {'Accuracy':<20}  {baseline_v2_acc:>14.3f}  {new_acc:>14.3f}  {new_acc - baseline_v2_acc:>+10.3f}")
    print(f"  {'Brier Score':<20}  {'N/A':>14}  {cv_tuned['mean_brier_score']:>14.4f}")
    print(f"  {'AUC':<20}  {'N/A':>14}  {cv_tuned['mean_auc']:>14.4f}")

    if new_ll < baseline_v2_ll:
        pct = (baseline_v2_ll - new_ll) / baseline_v2_ll * 100
        print(f"\n  >>> v3 model improves log loss by {pct:.1f}% over v2 <<<")
    else:
        print(f"\n  Note: Log loss did not improve (v2 {baseline_v2_ll:.4f} vs v3 {new_ll:.4f})")

    # -- Step 9: Train final model (with weights) -------------------------
    print("\n" + "=" * 70)
    print("STEP 9 -- Training final model on all historical data (weighted)")
    print("=" * 70)

    from src.models.train import train_model
    final_model = train_model(
        X_all, y_all, random_seed=42, xgb_params=best_params,
        sample_weight=weights_all,
    )
    print("  Final model trained successfully (with sample weights).")

    # -- Step 10: Generate 2026 predictions --------------------------------
    print("\n" + "=" * 70)
    print("STEP 10 -- Loading 2026 bracket and preparing predictions")
    print("=" * 70)

    # Load actual bracket
    bracket_raw = pd.read_csv(BRACKET_CSV)
    bracket_raw["Seed"] = bracket_raw["Seed"].astype(int)
    bracket_raw["TeamID"] = bracket_raw["TeamID"].astype(int)

    # Map KenPom TEAM NO -> Kaggle TeamID (kp_to_kaggle already built above)
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

    # -- Step 11: Monte Carlo simulation -----------------------------------
    print("\n" + "=" * 70)
    print("STEP 11 -- Pre-computing pairwise win probabilities")
    print("=" * 70)

    n_teams = len(bracket_kaggle)
    print(f"  Computing {n_teams} x {n_teams - 1} = {n_teams * (n_teams - 1):,} pair probabilities...")
    win_prob = precompute_win_probs(bracket_kaggle, fm_2026_tourney, feature_cols, final_model)
    print(f"  Done. Lookup table has {len(win_prob):,} entries.")

    # -- Step 11b: R64 line blending ---------------------------------------
    r64_lines = _build_r64_lines(vegas_df, name_resolution, bracket_kaggle)
    if r64_lines:
        print(f"  Blending {len(r64_lines)} R64 game-specific Vegas lines (weight=0.35)...")
        win_prob = blend_r64_probs(win_prob, r64_lines, weight=0.35)
        print(f"  R64 line blending complete.")
    else:
        print(f"  No R64 lines found for blending.")

    print("\n" + "=" * 70)
    print("STEP 12 -- Monte Carlo simulation (10,000 iterations)")
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

    # -- Results ----------------------------------------------------------
    print("\n" + "=" * 70)
    print("RESULTS -- Championship Probabilities (Top 15)")
    print("=" * 70)
    print_champion_probs(advancement_probs, bracket_kaggle, top_n=15)

    print("\n" + "=" * 70)
    print("RESULTS -- Advancement Probabilities (Top 30)")
    print("=" * 70)
    print_advancement_table(advancement_probs, bracket_kaggle, top_n=30)

    # -- Bracket picks ----------------------------------------------------
    print("\n" + "=" * 70)
    print("RESULTS -- Bracket Picks")
    print("=" * 70)

    from src.bracket.strategies import chalk_bracket, expected_value_bracket

    chalk_picks = chalk_bracket(bracket_kaggle, advancement_probs)
    ev_picks = expected_value_bracket(bracket_kaggle, advancement_probs)

    id_to_name = dict(zip(bracket_kaggle["TeamID"], bracket_kaggle["TeamName"]))
    id_to_seed = dict(zip(bracket_kaggle["TeamID"], bracket_kaggle["Seed"]))

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

    if chalk_picks.get(5):
        print(f"\n  Final Four (chalk):")
        for tid in chalk_picks[5]:
            name = id_to_name.get(tid, str(tid))
            seed = id_to_seed.get(tid, "?")
            print(f"    ({seed:>2}) {name}")

    # -- Step 13: Export results -------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 13 -- Exporting results")
    print("=" * 70)

    # Export advancement probabilities CSV
    from src.bracket.output import export_bracket_csv
    csv_path = apply_output_suffix(str(OUTPUT_DIR / "bracket_2026_real.csv"), _output_suffix)
    export_bracket_csv(advancement_probs, bracket_kaggle, csv_path)
    print(f"  Saved: {csv_path}")

    # Export bracket structure
    bracket_csv_path = apply_output_suffix(str(OUTPUT_DIR / "bracket_2026_real_structure.csv"), _output_suffix)
    bracket_kaggle.to_csv(bracket_csv_path, index=False)
    print(f"  Saved: {bracket_csv_path}")

    # Export pairwise probabilities JSON
    pairwise_json = {}
    for (a, b), p in win_prob.items():
        lo, hi = min(a, b), max(a, b)
        key = f"{lo}_{hi}"
        if key not in pairwise_json:
            pairwise_json[key] = round(p if a < b else 1 - p, 4)

    pairwise_path = apply_output_suffix(str(OUTPUT_DIR / "pairwise_probs.json"), _output_suffix)
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
    bracket_data_path = apply_output_suffix(str(OUTPUT_DIR / "bracket_data.json"), _output_suffix)
    with open(bracket_data_path, "w") as f:
        json.dump(bracket_data, f, indent=2)
    print(f"  Saved: {bracket_data_path}")

    # Export compact JSON for bracket.html
    compact = build_bracket_compact_json(bracket_kaggle, advancement_probs, win_prob)
    compact_path = apply_output_suffix(str(OUTPUT_DIR / "bracket_compact.json"), _output_suffix)
    with open(compact_path, "w") as f:
        json.dump(compact, f, separators=(",", ":"))
    print(f"  Saved: {compact_path}")

    # -- Step 14: Update bracket.html -------------------------------------
    print("\n" + "=" * 70)
    print("STEP 14 -- Updating bracket.html")
    print("=" * 70)

    html_path = Path(apply_output_suffix(str(OUTPUT_DIR / "bracket.html"), _output_suffix))
    if html_path.exists():
        html_content = html_path.read_text(encoding="utf-8")
        compact_json_str = json.dumps(compact, separators=(",", ":"))

        if "const RAW = " in html_content:
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

    # -- Step 15: Regenerate Kaggle submission -----------------------------
    print("\n" + "=" * 70)
    print("STEP 15 -- Regenerating Kaggle submission files")
    print("=" * 70)

    _regenerate_kaggle_submission(
        data, feature_matrix, feature_cols, fm_filled, best_params,
        output_suffix=_output_suffix,
    )

    # -- Final summary ----------------------------------------------------
    elapsed = time.time() - overall_start
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"\n  Enhanced Model v3 Performance (LOSO CV, weighted):")
    print(f"    Log Loss  : {cv_tuned['mean_log_loss']:.4f}  (v2 baseline: {baseline_v2_ll:.4f})")
    print(f"    Accuracy  : {cv_tuned['mean_accuracy']:.3f}  (v2 baseline: {baseline_v2_acc:.3f})")
    print(f"    Brier     : {cv_tuned['mean_brier_score']:.4f}")
    print(f"    AUC       : {cv_tuned['mean_auc']:.4f}")
    # Reconstruct summary locals from inputs (these were locals of the old
    # monolithic main() before prepare_loso_inputs() was extracted; the
    # extraction kept the artifacts addressable via the returned dict).
    _v3_prefixes = ("late_", "efficiency_trend", "margin_trend", "conf_tourney_",
                    "vegas_late_spread_delta", "coach_")
    new_feature_names = [c for c in feature_cols if c.startswith(_v3_prefixes)]
    n_tourney = int((weights_all >= 1.0).sum())
    n_supplemental = int((weights_all < 1.0).sum())
    print(f"\n  Features used: {len(feature_cols)}")
    print(f"  New v3 features: {new_feature_names}")
    print(f"  Vegas features: {[c for c in feature_cols if c.startswith('vegas_')]}")
    print(f"  Training games: {len(X_all):,} (tourney: {n_tourney}, supplemental: {n_supplemental})")
    print(f"  XGBoost params: {best_params}")

    if chalk_picks.get(6):
        chalk_champ = chalk_picks[6][0]
        print(f"\n  2026 Predicted Champion: ({id_to_seed.get(chalk_champ, '?')}) "
              f"{id_to_name.get(chalk_champ, str(chalk_champ))}")

    print(f"\n  Total runtime: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print("\n" + "=" * 70)
    print("  Done! Enhanced model v3 complete.")
    print("=" * 70 + "\n")


def _regenerate_kaggle_submission(data, feature_matrix, feature_cols, fm_filled, best_params, output_suffix=""):
    """Regenerate Kaggle submission files using the v3 model.

    This rebuilds the men's model with all v3 features and produces updated
    submission files. Women's model remains unchanged (no Vegas/late-season data).
    """
    from src.kaggle_submission import (
        build_all_team_features,
        build_matchup_training_data,
        get_feature_cols as ks_get_feature_cols,
        train_xgb_model,
        predict_all_pairs,
        compute_adjusted_efficiency_fast,
        compute_four_factors_fast,
        compute_rolling_and_form,
        compute_conf_strength,
    )

    # -- Load women's data (reuse from kaggle_submission) ------------------
    w_reg = pd.read_csv(MANIA_DIR / "WRegularSeasonDetailedResults.csv")
    w_tourney = pd.read_csv(MANIA_DIR / "WNCAATourneyDetailedResults.csv")
    w_seeds = pd.read_csv(MANIA_DIR / "WNCAATourneySeeds.csv")
    w_conf = pd.read_csv(MANIA_DIR / "WTeamConferences.csv")

    # Sample submissions
    sample_s1 = pd.read_csv(MANIA_DIR / "SampleSubmissionStage1.csv")
    sample_s2 = pd.read_csv(MANIA_DIR / "SampleSubmissionStage2.csv")

    # -- Men's model: use existing v3 feature matrix -----------------------
    print("  Building men's feature matrix for Kaggle submission...")

    # Build full men's feature matrix (all D1 teams) using kaggle_submission infrastructure
    m_reg = data["reg_season"]
    m_seeds = data["seeds"]
    m_teams = data["teams"]
    m_conf = data["conferences"]
    m_spellings = data["spellings"]

    kp_to_kaggle = build_kenpom_to_kaggle_map(data["kenpom"], m_teams, m_spellings)

    men_seasons = sorted(s for s in m_reg["Season"].unique() if s >= 2003)
    men_fm = build_all_team_features(
        reg_season=m_reg,
        seeds=m_seeds,
        conferences=m_conf,
        seasons=men_seasons,
        massey=data["massey"],
        kenpom=data["kenpom"],
        kp_to_kaggle=kp_to_kaggle,
        gender="M",
    )

    # Load Vegas features and merge. Filter NCAA tournament games out
    # before per-team-per-season aggregation -- otherwise season S
    # tournament outcomes leak into season S feature rows. Spec:
    # docs/superpowers/specs/2026-05-04-v4-vegas-leak-fix-design.md
    vegas_df = load_vegas_lines()
    vegas_df_pre_tourney = filter_vegas_to_pre_tournament(vegas_df)
    vegas_features, _ = compute_vegas_features(vegas_df_pre_tourney, m_teams, m_spellings)
    men_fm = men_fm.merge(vegas_features, on=["TeamID", "Season"], how="left")

    men_feature_cols = ks_get_feature_cols(men_fm)
    print(f"  Men's feature columns: {len(men_feature_cols)}")

    # Build training data
    m_tourney = data["tourney"]
    men_fm_seasons = set(men_fm["Season"].unique())
    m_tourney_filtered = m_tourney[m_tourney["Season"].isin(men_fm_seasons)]

    X_men, y_men = build_matchup_training_data(men_fm, m_tourney_filtered, men_feature_cols)

    # Drop columns with >30% NaN. X_men columns are expanded (<feat>_diff,
    # <feat>_avg); we drop the raw feature if either of its expanded forms
    # is too sparse, then rebuild the expanded column list.
    if not X_men.empty:
        from src.models.matchup import expand_feature_cols as _expand
        null_fracs = X_men.isna().mean()
        drop_expanded = null_fracs[null_fracs > 0.30].index.tolist()
        drop_raw = {c.removesuffix("_diff") for c in drop_expanded
                    if c.endswith("_diff")}
        if drop_raw:
            print(f"  Dropping {len(drop_raw)} high-NaN raw features: "
                  f"{sorted(drop_raw)[:10]}...")
            men_feature_cols = [c for c in men_feature_cols if c not in drop_raw]
            X_men = X_men[_expand(men_feature_cols)]

    men_medians = X_men.median()
    X_men = X_men.fillna(men_medians)

    men_model = train_xgb_model(X_men, y_men)
    print(f"  Men's model trained ({len(X_men)} samples, {len(men_feature_cols)} features)")

    # -- Women's model (unchanged) ----------------------------------------
    print("  Building women's feature matrix...")
    women_seasons = sorted(s for s in w_reg["Season"].unique() if s >= 2010)
    women_fm = build_all_team_features(
        reg_season=w_reg,
        seeds=w_seeds,
        conferences=w_conf,
        seasons=women_seasons,
        massey=None,
        kenpom=None,
        kp_to_kaggle=None,
        gender="W",
    )

    women_feature_cols = ks_get_feature_cols(women_fm)
    women_fm_seasons = set(women_fm["Season"].unique())
    w_tourney_filtered = w_tourney[w_tourney["Season"].isin(women_fm_seasons)]

    X_women, y_women = build_matchup_training_data(women_fm, w_tourney_filtered, women_feature_cols)

    if not X_women.empty:
        from src.models.matchup import expand_feature_cols as _expand
        null_fracs = X_women.isna().mean()
        drop_expanded = null_fracs[null_fracs > 0.30].index.tolist()
        drop_raw = {c.removesuffix("_diff") for c in drop_expanded
                    if c.endswith("_diff")}
        if drop_raw:
            women_feature_cols = [c for c in women_feature_cols if c not in drop_raw]
            X_women = X_women[_expand(women_feature_cols)]

    women_medians = X_women.median()
    X_women = X_women.fillna(women_medians)

    women_model = train_xgb_model(X_women, y_women)
    print(f"  Women's model trained ({len(X_women)} samples, {len(women_feature_cols)} features)")

    # -- Generate predictions ----------------------------------------------
    print("  Generating predictions...")

    for stage_name, sample_df, out_name in [
        ("Stage 1", sample_s1, "submission_stage1.csv"),
        ("Stage 2", sample_s2, "submission_stage2.csv"),
    ]:
        print(f"    {stage_name} ({len(sample_df):,} rows)...")
        sub = sample_df.copy()
        sub["Pred"] = 0.5

        sub = predict_all_pairs(sub, men_fm, men_feature_cols, men_model, gender_filter="M")
        sub = predict_all_pairs(sub, women_fm, women_feature_cols, women_model, gender_filter="W")

        sub["Pred"] = sub["Pred"].clip(0.01, 0.99)

        out_path = Path(apply_output_suffix(str(OUTPUT_DIR / out_name), output_suffix))
        sub.to_csv(out_path, index=False)
        print(f"    Written: {out_path}")


if __name__ == "__main__":
    main()
