"""Assemble the team-season feature matrix from the KenPom/Barttorvik dataset.

Core features
-------------
Efficiency (KenPom)
    KADJ O, KADJ D, KADJ EM   — adjusted offensive/defensive/net efficiency
    K TEMPO / KADJ T           — tempo
    BARTHAG                    — power rating (expected win % vs avg D1 opponent)

Barttorvik four factors (offensive + defensive)
    EFG%, EFG%D  — effective field-goal percentage
    TOV%, TOV%D  — turnover rate
    OREB%, DREB% — rebounding rates
    FTR, FTRD    — free-throw rate

Additional efficiency
    BADJ EM, BADJ O, BADJ D    — Barttorvik adjusted margin / off / def
    PPPO, PPPD                 — points per possession
    EXP, TALENT                — roster experience and talent index
    ELITE SOS, WAB             — schedule strength and wins-above-bubble

Seed
    SEED — tournament seeding (numeric)

Resume metrics (from Resumes.csv)
    NET_RPI, ELO, WAB_RANK, Q1_W, Q2_W, Q3_Q4_L

538 Power Rating (optional, 2016-2024)
    FTE_POWER

Conference strength
    CONF_AVG_KADJ_EM — mean KADJ EM of conference members that season
"""

import logging

import pandas as pd

logger = logging.getLogger(__name__)

# ── Core feature columns from KenPom Barttorvik.csv ──────────────────────────
_KENPOM_FEATURES = [
    # KenPom adjusted efficiency
    "KADJ O", "KADJ D", "KADJ EM",
    # Tempo
    "K TEMPO", "KADJ T",
    # Barttorvik efficiency
    "BADJ EM", "BADJ O", "BADJ D",
    # Power rating
    "BARTHAG",
    # Four factors — offensive
    "EFG%", "TOV%", "OREB%", "FTR",
    # Four factors — defensive
    "EFG%D", "TOV%D", "DREB%", "FTRD",
    # Shooting
    "2PT%", "3PT%", "2PT%D", "3PT%D",
    # Possession / efficiency
    "PPPO", "PPPD",
    # Roster / strength
    "EXP", "TALENT", "ELITE SOS", "WAB",
]

_RESUME_FEATURES = ["NET_RPI", "ELO", "WAB_RANK", "Q1_W", "Q2_W", "Q3_Q4_L"]


def _build_conf_strength(kenpom: pd.DataFrame, season: int) -> pd.DataFrame:
    """Return a DataFrame of TeamID → CONF_AVG_KADJ_EM for the given season."""
    df = kenpom[kenpom["YEAR"] == season][["TEAM NO", "CONF", "KADJ EM"]].copy()
    df = df.rename(columns={"TEAM NO": "TeamID"})
    conf_avg = df.groupby("CONF")["KADJ EM"].mean().reset_index()
    conf_avg.columns = ["CONF", "CONF_AVG_KADJ_EM"]
    return df[["TeamID", "CONF"]].merge(conf_avg, on="CONF")[["TeamID", "CONF_AVG_KADJ_EM"]]


def build_feature_matrix_v2(
    kenpom: pd.DataFrame,
    resumes: pd.DataFrame,
    ratings_538: pd.DataFrame,
) -> pd.DataFrame:
    """Build a full feature matrix across all available seasons.

    Parameters
    ----------
    kenpom : DataFrame
        KenPom Barttorvik.csv — one row per team per season.
    resumes : DataFrame
        Resumes.csv — one row per team per season.
    ratings_538 : DataFrame
        538 Ratings.csv — one row per team per season (may be empty).

    Returns
    -------
    DataFrame with columns: TeamID, Season, SEED, <feature cols>
    One row per (team, season).  Tournament-only teams are kept; non-tournament
    teams (SEED == 0 or NaN) are also kept so that conference-strength
    calculations are accurate, but callers can filter them out.
    """
    if kenpom.empty:
        raise ValueError("kenpom DataFrame is empty; cannot build feature matrix")

    # ── 1. Build per-team-season base from KenPom ────────────────────────────
    available_kp_features = [c for c in _KENPOM_FEATURES if c in kenpom.columns]
    missing = set(_KENPOM_FEATURES) - set(available_kp_features)
    if missing:
        logger.warning("KenPom columns not found (will be skipped): %s", sorted(missing))

    base_cols = ["YEAR", "TEAM NO", "TEAM", "CONF", "SEED"] + available_kp_features
    base = kenpom[base_cols].copy()
    base = base.rename(columns={"YEAR": "Season", "TEAM NO": "TeamID", "TEAM": "TeamName"})

    # ── 2. Conference strength ────────────────────────────────────────────────
    conf_frames = []
    for season in base["Season"].unique():
        conf_frames.append(_build_conf_strength(kenpom, season))
    if conf_frames:
        conf_df = pd.concat(conf_frames, ignore_index=True)
        # conf_df is per-team-season but only has TeamID — need Season too
        # Rebuild with Season tag
        conf_list = []
        for season in base["Season"].unique():
            c = _build_conf_strength(kenpom, season).copy()
            c["Season"] = season
            conf_list.append(c)
        conf_all = pd.concat(conf_list, ignore_index=True)
        base = base.merge(conf_all, on=["TeamID", "Season"], how="left")

    # ── 3. Resume metrics ─────────────────────────────────────────────────────
    if not resumes.empty:
        res = resumes[["YEAR", "TEAM NO", "NET RPI", "ELO", "WAB RANK",
                        "Q1 W", "Q2 W", "Q3 Q4 L"]].copy()
        res = res.rename(columns={
            "YEAR": "Season",
            "TEAM NO": "TeamID",
            "NET RPI": "NET_RPI",
            "WAB RANK": "WAB_RANK",
            "Q1 W": "Q1_W",
            "Q2 W": "Q2_W",
            "Q3 Q4 L": "Q3_Q4_L",
        })
        base = base.merge(res, on=["TeamID", "Season"], how="left")
    else:
        logger.warning("Resumes DataFrame is empty; resume features will be NaN")
        for col in _RESUME_FEATURES:
            base[col] = float("nan")

    # ── 4. 538 power ratings (optional) ──────────────────────────────────────
    if not ratings_538.empty:
        fte = ratings_538[["YEAR", "TEAM NO", "POWER RATING"]].copy()
        fte = fte.rename(columns={
            "YEAR": "Season",
            "TEAM NO": "TeamID",
            "POWER RATING": "FTE_POWER",
        })
        base = base.merge(fte, on=["TeamID", "Season"], how="left")
    else:
        base["FTE_POWER"] = float("nan")

    # ── 5. Final clean-up ─────────────────────────────────────────────────────
    # Drop the raw CONF column (we already derived CONF_AVG_KADJ_EM)
    if "CONF" in base.columns:
        base = base.drop(columns=["CONF"])

    # Log null summary for tournament teams (SEED > 0)
    tourney_mask = base["SEED"].notna() & (base["SEED"] > 0)
    tourney_df = base[tourney_mask]
    null_counts = tourney_df.isna().sum()
    noisy_cols = null_counts[null_counts > 0]
    if not noisy_cols.empty:
        for col, cnt in noisy_cols.items():
            logger.debug("Feature '%s' has %d nulls among tournament teams", col, cnt)

    logger.info(
        "Feature matrix built: %d rows, %d feature columns, seasons %s–%s",
        len(base),
        len(base.columns) - 3,  # subtract TeamID, Season, TeamName
        int(base["Season"].min()),
        int(base["Season"].max()),
    )
    return base


def get_feature_cols(feature_matrix: pd.DataFrame) -> list[str]:
    """Return the list of numeric feature column names (excludes ID/name cols)."""
    exclude = {"TeamID", "Season", "TeamName", "SEED"}
    cols = [c for c in feature_matrix.columns if c not in exclude]
    # Keep only columns that are actually numeric
    numeric_cols = feature_matrix[cols].select_dtypes(include="number").columns.tolist()
    return numeric_cols
