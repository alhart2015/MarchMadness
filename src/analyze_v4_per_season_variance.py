"""Per-season variance check for v4 across 22 LOSO seasons.

Cheap diagnostic gate before committing engineering budget to
calibration-shape work. Surfaces whether v4's 22-season-aggregate
metrics hide high-variance per-season behavior.

Spec: docs/superpowers/specs/2026-05-07-v4-per-season-variance-design.md

Outputs:
    output/v4_per_season_variance.json
    output/v4_per_season_variance_traces.png
    output/v4_per_season_variance_deltas.png
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Reuse audit drivers' data-load + join helpers. Cross-module coupling
# is intentional per the spec; this is a one-off diagnostic.
from src.audit_v4_gap_vegas import (  # noqa: E402
    DATA,
    _build_day_zero_map,
    _build_per_game_audit_df as _build_audit_df_vegas,
    _build_vegas_lookup,
    _calibration_table,
    _ece,
    _load_seeds_lookup,
    _load_v4_lookup,
    _vegas_to_seasonday,
)
from src.audit_v4_gap_fte import (  # noqa: E402
    _build_fte_lookup,
    _build_per_game_audit_df as _build_audit_df_fte,
    _resolve_fte_team_ids,
)
from src.enhanced_model_v2 import (  # noqa: E402
    _build_vegas_name_to_kaggle_map,
    _resolve_vegas_name,
    load_vegas_lines,
)
from src.ingest.fte_forecasts import _AUDITED_YEARS, load_fte_forecasts  # noqa: E402

logger = logging.getLogger(__name__)

DEFAULT_OUT_JSON = "output/v4_per_season_variance.json"
DEFAULT_OUT_DIR = "output"
DEFAULT_FTE_CACHE = Path("data/raw/fte_forecasts")
DEFAULT_SIGMA = 1.5


# ---------------------------------------------------------------------------
# Per-season aggregation
# ---------------------------------------------------------------------------


def _per_season_metrics(df: pd.DataFrame, ref_label: str) -> pd.DataFrame:
    """Per-season aggregate of v4-vs-<ref> metrics.

    df expected columns: season, p_v4, p_<ref_label>, winner_is_a.
    Returns columns: season, n_games, ll_v4, ll_<ref>, ll_v4_minus_<ref>,
    acc_v4, acc_<ref>, ece_v4, ece_<ref>.
    """
    eps = 1e-15
    ref_col = f"p_{ref_label}"
    rows = []
    for season, sub in df.groupby("season"):
        winner = sub["winner_is_a"].to_numpy()
        p_v4 = sub["p_v4"].to_numpy()
        p_ref = sub[ref_col].to_numpy()

        p_v4_w = np.where(winner == 1, p_v4, 1 - p_v4)
        p_ref_w = np.where(winner == 1, p_ref, 1 - p_ref)
        ll_v4 = float(-np.mean(np.log(np.clip(p_v4_w, eps, 1 - eps))))
        ll_ref = float(-np.mean(np.log(np.clip(p_ref_w, eps, 1 - eps))))

        acc_v4 = float(((p_v4 >= 0.5).astype(int) == winner).mean())
        acc_ref = float(((p_ref >= 0.5).astype(int) == winner).mean())

        cal_v4 = _calibration_table(p_v4, winner)
        cal_ref = _calibration_table(p_ref, winner)

        rows.append({
            "season": int(season),
            "n_games": int(len(sub)),
            "ll_v4": ll_v4,
            f"ll_{ref_label}": ll_ref,
            f"ll_v4_minus_{ref_label}": ll_v4 - ll_ref,
            "acc_v4": acc_v4,
            f"acc_{ref_label}": acc_ref,
            "ece_v4": _ece(cal_v4),
            f"ece_{ref_label}": _ece(cal_ref),
        })
    return pd.DataFrame(rows).sort_values("season").reset_index(drop=True)


def _flag_outliers(
    df: pd.DataFrame, columns: list[str], sigma: float = DEFAULT_SIGMA,
) -> dict:
    """Flag rows where (column - mean) / std >= sigma. Sample std (ddof=1).

    Returns: {column: [{season, value, sigma_delta, n_games}, ...]}.
    Empty list for missing columns or series too short to estimate std.
    """
    out: dict[str, list[dict]] = {}
    for col in columns:
        if col not in df.columns:
            continue
        vals = df[col].dropna()
        if len(vals) < 2:
            out[col] = []
            continue
        mean = float(vals.mean())
        std = float(vals.std(ddof=1))
        if std == 0.0 or not np.isfinite(std):
            out[col] = []
            continue
        flagged = []
        for _, row in df.iterrows():
            if pd.isna(row[col]):
                continue
            z = (float(row[col]) - mean) / std
            if z >= sigma:
                flagged.append({
                    "season": int(row["season"]),
                    "value": float(row[col]),
                    "sigma_delta": float(z),
                    "n_games": int(row["n_games"]),
                })
        flagged.sort(key=lambda x: -x["sigma_delta"])
        out[col] = flagged
    return out


def _pick_verdict(
    df: pd.DataFrame, outliers: dict, sigma: float,
) -> dict:
    """Pick one of {flat, outlier, trend, mixed} based on outlier counts.

    - flat: no flags on any tracked column.
    - outlier: 1-2 distinct seasons flagged across cross-benchmark deltas
      (ll_v4_minus_vegas, ll_v4_minus_fte).
    - trend: 3+ flagged seasons consecutive (monotonic season order) on
      a cross-benchmark delta.
    - mixed: anything else.
    """
    cross_keys = ["ll_v4_minus_vegas", "ll_v4_minus_fte"]
    intra_keys = ["ll_v4", "ece_v4"]

    flagged_cross_seasons: set[int] = set()
    for k in cross_keys:
        for entry in outliers.get(k, []):
            flagged_cross_seasons.add(int(entry["season"]))

    flagged_intra_seasons: set[int] = set()
    for k in intra_keys:
        for entry in outliers.get(k, []):
            flagged_intra_seasons.add(int(entry["season"]))

    all_flagged = flagged_cross_seasons | flagged_intra_seasons

    if not all_flagged:
        return {
            "label": "flat",
            "summary": (
                f"No season exceeds {sigma} sigma on any cross-benchmark "
                f"delta or intra-v4 metric. Aggregate calibration is the "
                f"likely bottleneck."
            ),
            "outlier_seasons": [],
        }

    # Detect trend: 3+ consecutive seasons on a cross-benchmark delta.
    sorted_seasons = sorted(flagged_cross_seasons)
    is_trend = False
    if len(sorted_seasons) >= 3:
        all_seasons = sorted(df["season"].unique())
        season_index = {s: i for i, s in enumerate(all_seasons)}
        sorted_idx = sorted(season_index[s] for s in sorted_seasons)
        for i in range(len(sorted_idx) - 2):
            if sorted_idx[i + 1] == sorted_idx[i] + 1 and sorted_idx[i + 2] == sorted_idx[i] + 2:
                is_trend = True
                break

    if is_trend:
        return {
            "label": "trend",
            "summary": (
                f"3+ consecutive seasons flagged on cross-benchmark delta. "
                f"Investigate gradual calibration drift (data pipeline, "
                f"rule changes, era effects)."
            ),
            "outlier_seasons": sorted_seasons,
        }

    if 1 <= len(all_flagged) <= 2:
        return {
            "label": "outlier",
            "summary": (
                f"{len(all_flagged)} season(s) exceed {sigma} sigma. "
                f"Investigate what's distinctive about these tournaments "
                f"before fixing aggregate calibration."
            ),
            "outlier_seasons": sorted(all_flagged),
        }

    return {
        "label": "mixed",
        "summary": (
            f"{len(all_flagged)} seasons flagged but no clean trend. "
            f"Findings note must call out the pattern."
        ),
        "outlier_seasons": sorted(all_flagged),
    }
