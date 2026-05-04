"""Colley-matrix ratings: per-team-per-season ratings from solving
the standard Colley system (2I + diag(T) - A) x = b on regular-season
W/L data. No margin used; no venue used.

See docs/superpowers/specs/2026-05-03-colley-matrix-feature-design.md.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_PRODUCER_VERSION = "v1"


def _solve_one_season(games_df: pd.DataFrame) -> dict[int, float]:
    """Solve Colley (2I + diag(T) - A) x = b for one season.

    Required columns of games_df: WTeamID, LTeamID. (WScore, LScore,
    WLoc, NumOT, DayNum are ignored -- Colley is W/L only.)

    Returns {TeamID: colley_rating}. Sum of ratings is n/2 by construction.
    """
    team_ids = sorted(set(games_df["WTeamID"].tolist()) | set(games_df["LTeamID"].tolist()))
    n = len(team_ids)
    idx = {tid: i for i, tid in enumerate(team_ids)}

    # Start with C = 2 * I (the +2 Bayesian prior); each game adds 1 to
    # the diagonal of both participants and -1 to their off-diagonal entry.
    C = 2.0 * np.eye(n, dtype=np.float64)
    b = np.ones(n, dtype=np.float64)  # +1 prior contribution per team

    for w, l in zip(
        games_df["WTeamID"].to_numpy(),
        games_df["LTeamID"].to_numpy(),
    ):
        wi = idx[int(w)]
        li = idx[int(l)]
        C[wi, wi] += 1.0
        C[li, li] += 1.0
        C[wi, li] -= 1.0
        C[li, wi] -= 1.0
        b[wi] += 0.5
        b[li] -= 0.5

    cond = np.linalg.cond(C)
    if cond > 1e10:
        logger.warning("Colley matrix is ill-conditioned (cond=%.2e); "
                       "season may have a disconnected component", cond)

    x = np.linalg.solve(C, b)
    return {tid: float(x[idx[tid]]) for tid in team_ids}


def compute_colley_ratings(
    reg_season: pd.DataFrame,
    seasons: list[int] | None = None,
) -> pd.DataFrame:
    """Compute Colley ratings per (Season, TeamID).

    Parameters
    ----------
    reg_season : DataFrame
        Kaggle MRegularSeasonCompactResults (or DetailedResults superset).
        Required columns: Season, WTeamID, LTeamID.
    seasons : list of int or None
        Restrict to these seasons. None = all seasons present in reg_season.

    Returns
    -------
    DataFrame with columns [Season, TeamID, colley_rating], one row per
    (team, season) where the team appeared in the season's regular-season
    schedule. Sum of colley_rating within a season equals n/2.
    """
    required = {"Season", "WTeamID", "LTeamID"}
    missing = required - set(reg_season.columns)
    if missing:
        raise ValueError(f"reg_season missing required columns: {sorted(missing)}")

    season_iter = sorted(reg_season["Season"].unique()) if seasons is None else sorted(seasons)
    rows = []
    for season in season_iter:
        games = reg_season[reg_season["Season"] == season]
        if len(games) == 0:
            continue
        ratings = _solve_one_season(games)
        for tid, r in ratings.items():
            rows.append({"Season": int(season), "TeamID": int(tid), "colley_rating": r})
    return pd.DataFrame(rows)


def _hash_input(reg_season: pd.DataFrame) -> str:
    """Stable content hash of the relevant columns of the input frame."""
    cols = ["Season", "WTeamID", "LTeamID"]
    h = hashlib.sha256()
    for c in cols:
        if c in reg_season.columns:
            h.update(reg_season[c].astype(str).str.cat(sep="|").encode("ascii", errors="replace"))
    return h.hexdigest()[:16]


def load_colley_ratings(
    reg_season: pd.DataFrame,
    cache_dir: str | Path = "data/cache",
) -> pd.DataFrame:
    """Cached wrapper around compute_colley_ratings.

    Reads from <cache_dir>/colley_ratings.parquet on cache hit.
    Cache hit requires the sidecar metadata at
    <cache_dir>/colley_ratings.meta.json to match the current
    (_PRODUCER_VERSION, n_input_rows, sha_input).
    """
    cache_dir = Path(cache_dir)
    parquet_path = cache_dir / "colley_ratings.parquet"
    meta_path = cache_dir / "colley_ratings.meta.json"

    expected_meta = {
        "producer_version": _PRODUCER_VERSION,
        "n_input_rows": int(len(reg_season)),
        "sha_input": _hash_input(reg_season),
    }

    if parquet_path.exists() and meta_path.exists():
        try:
            actual_meta = json.loads(meta_path.read_text())
        except (json.JSONDecodeError, OSError):
            actual_meta = {}
        if all(actual_meta.get(k) == expected_meta[k] for k in expected_meta):
            logger.info("Colley cache hit: %s", parquet_path)
            return pd.read_parquet(parquet_path)
        logger.info("Colley cache stale (metadata mismatch); rebuilding")

    df = compute_colley_ratings(reg_season)
    cache_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(parquet_path, index=False)
    meta_path.write_text(json.dumps({**expected_meta, "written_at_n_rows": len(df)}, indent=2))
    logger.info("Colley cache written: %s (%d rows)", parquet_path, len(df))
    return df
