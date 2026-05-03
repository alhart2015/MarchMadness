"""Two-clause falsification gate for massey_mov_rating.

Clause 1 -- non-redundancy: per-season Pearson correlation between
massey_mov_rating and (adj_em, massey_composite). Pass if mean |corr|
< 0.95 and max |corr| < 0.97 against BOTH baselines.

Clause 2 -- no-harm headroom: 3-season subset {2019, 2022, 2024}.
Train v4 with massey_mov on, compute LL on holdout games. Pass if
mean LL with massey <= mean LL without massey + 0.001.

See docs/superpowers/specs/2026-05-03-massey-matrix-feature-design.md.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

GATE_SUBSET_SEASONS = [2019, 2022, 2024]
CORR_MEAN_MAX = 0.95
CORR_PER_SEASON_MAX = 0.97
LL_HEADROOM_MAX = 0.001


def clause1_correlations(feature_matrix: pd.DataFrame) -> dict:
    """Compute per-season correlations of massey_mov_rating vs adj_em
    and vs massey_composite. Returns a dict for output JSON.
    """
    needed = {"Season", "TeamID", "massey_mov_rating", "adj_em", "massey_composite"}
    missing = needed - set(feature_matrix.columns)
    if missing:
        raise ValueError(f"feature_matrix missing columns for clause 1: {sorted(missing)}")

    seasons = sorted(feature_matrix["Season"].unique())
    rows = []
    for season in seasons:
        sub = feature_matrix[feature_matrix["Season"] == season]
        sub = sub.dropna(subset=["massey_mov_rating", "adj_em", "massey_composite"])
        if len(sub) < 50:
            logger.warning("Season %d: only %d teams with all 3 cols; skipping",
                           season, len(sub))
            continue
        r_em = float(sub["massey_mov_rating"].corr(sub["adj_em"]))
        r_comp = float(sub["massey_mov_rating"].corr(sub["massey_composite"]))
        rows.append({"season": int(season), "n_teams": int(len(sub)),
                     "corr_vs_adj_em": r_em, "corr_vs_massey_composite": r_comp})

    df = pd.DataFrame(rows)
    summary = {
        "per_season": rows,
        "mean_abs_corr_vs_adj_em": float(df["corr_vs_adj_em"].abs().mean()),
        "max_abs_corr_vs_adj_em": float(df["corr_vs_adj_em"].abs().max()),
        "mean_abs_corr_vs_massey_composite": float(df["corr_vs_massey_composite"].abs().mean()),
        "max_abs_corr_vs_massey_composite": float(df["corr_vs_massey_composite"].abs().max()),
    }
    summary["pass"] = bool(
        summary["mean_abs_corr_vs_adj_em"] < CORR_MEAN_MAX
        and summary["max_abs_corr_vs_adj_em"] < CORR_PER_SEASON_MAX
        and summary["mean_abs_corr_vs_massey_composite"] < CORR_MEAN_MAX
        and summary["max_abs_corr_vs_massey_composite"] < CORR_PER_SEASON_MAX
    )
    return summary
