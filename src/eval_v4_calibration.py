"""Eval driver for v4 calibration: post-hoc temperature scaling on
canonical output/pairwise_v8.csv (Phase 1) + conditional retrain
(Phase 2).

Spec: docs/superpowers/specs/2026-05-08-v4-calibration-temperature-scaling-design.md
Plan: docs/superpowers/plans/2026-05-08-v4-calibration-temperature-scaling.md

Phase 1 sub-experiments:
    1. Global T sweep over T_GRID -- 7 cells, anchor T=1.0 reproduces 2069.
    2. Per-round T sequential greedy over (R64, R32, S16, E8, F4_NCG).

Phase 2 (only if Phase 1 PASS or MARGINAL):
    Scale v4 stage-1 with the winning T configuration, retrain v8 LOSO,
    re-score 22-season bracket points.

Verdict bands (carried from R64-blend / BT-bracket-points / v9-C):
    delta >= +25 with drop_best_delta >= 0 and wins >= 6 -> PASS
    delta in [+10, +25)  OR PASS-magnitude with drop_best_delta < 0 -> MARGINAL
    delta < +10 -> FAIL
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Mapping

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.apply_temperature_scaling import (  # noqa: E402
    ROUND_BUCKETS,
    assign_round_buckets,
    scale_pairwise,
)
from src.score_chalk_brackets import score_pairwise_path  # noqa: E402

logger = logging.getLogger(__name__)

T_GRID = [0.7, 0.85, 1.0, 1.15, 1.3, 1.5, 2.0]
ROUND_ORDER = ["R64", "R32", "S16", "E8", "F4_NCG"]
PASS_BAR = 25
MARGINAL_BAR = 10
DATA = Path("data/raw/march-machine-learning-2026")


# ---------------------------------------------------------------------------
# Pure helpers (testable in isolation; Task 2)
# ---------------------------------------------------------------------------


def _drop_best_season_delta(per_season_delta: Mapping[int, float]) -> float:
    """Sum of per-season deltas minus the largest single positive
    contribution. If every season is a loss (no positives), return the
    raw total -- nothing to 'drop'.
    """
    vals = list(per_season_delta.values())
    if not vals:
        return 0.0
    total = float(sum(vals))
    best = max(vals)
    if best <= 0:
        return total
    return total - float(best)


def _classify_verdict(
    delta_total: float,
    drop_best_delta: float,
    wins: int,
) -> str:
    """PASS / MARGINAL / FAIL per the spec decision matrix.

    PASS:
        delta_total >= +25 AND drop_best_delta >= 0 AND wins >= 6.
    MARGINAL:
        delta_total in [+10, +25),
        OR delta_total >= +25 with drop_best_delta < 0
           (single-season concentration demotes PASS),
        OR delta_total >= +25 with wins < 6
           (insufficiently broad win count).
    FAIL:
        delta_total < +10.
    """
    if delta_total >= PASS_BAR:
        if drop_best_delta < 0 or wins < 6:
            return "MARGINAL"
        return "PASS"
    if delta_total >= MARGINAL_BAR:
        return "MARGINAL"
    return "FAIL"


def _summarize_cell(
    per_season_delta: Mapping[int, float],
    baseline_total: float,
) -> dict:
    """Pack the per-season delta dict into the standard cell summary."""
    items = list(per_season_delta.items())
    delta_total = float(sum(v for _, v in items))
    wins = sum(1 for _, v in items if v > 0)
    losses = sum(1 for _, v in items if v < 0)
    ties = sum(1 for _, v in items if v == 0)
    if items:
        biggest_season, biggest_value = max(items, key=lambda kv: abs(kv[1]))
    else:
        biggest_season, biggest_value = None, 0.0
    drop_best_delta = _drop_best_season_delta(per_season_delta)
    return {
        "total": float(baseline_total + delta_total),
        "delta_total": delta_total,
        "wins": int(wins),
        "losses": int(losses),
        "ties": int(ties),
        "biggest_swing_value": float(biggest_value),
        "biggest_swing_season": (int(biggest_season) if biggest_season is not None else None),
        "drop_best_season_delta": float(drop_best_delta),
        "per_season_delta": {int(s): float(v) for s, v in items},
    }


def _anchor_check(df_actual: pd.DataFrame, baseline_csv: str) -> dict:
    """Compare df_actual to baseline_csv on (season, team_a, team_b).
    Returns {matches: bool, max_abs_diff: float, n_rows: int}.
    Mirrors src/sweep_bt_bracket_points._anchor_check semantics."""
    a = df_actual.drop_duplicates(["season", "team_a", "team_b"], keep="last")
    b = pd.read_csv(baseline_csv).drop_duplicates(
        ["season", "team_a", "team_b"], keep="last"
    )
    merged = a.merge(b, on=["season", "team_a", "team_b"],
                     suffixes=("_actual", "_expected"))
    if len(merged) != len(a) or len(merged) != len(b):
        return {
            "matches": False,
            "max_abs_diff": float("nan"),
            "n_only_actual": int(len(a) - len(merged)),
            "n_only_expected": int(len(b) - len(merged)),
            "n_rows": int(len(merged)),
        }
    diff = (merged["p_a_wins_actual"] - merged["p_a_wins_expected"]).abs()
    return {
        "matches": bool(diff.max() < 1e-9),
        "max_abs_diff": float(diff.max()),
        "n_rows": int(len(merged)),
    }


# ---------------------------------------------------------------------------
# Sweep-side scaffolding (filled in Task 3)
# ---------------------------------------------------------------------------


def _score_pairwise_df(df: pd.DataFrame, scratch_dir: Path) -> dict:
    """Write df to a tempfile in scratch_dir and call score_pairwise_path.
    Returns {'total_pts': float, 'per_season_pts': {int: float}}.
    The tempfile is cleaned up before return.
    """
    fd, tmp_path = tempfile.mkstemp(prefix="calib_", suffix=".csv", dir=str(scratch_dir))
    os.close(fd)
    try:
        df.to_csv(tmp_path, index=False)
        return score_pairwise_path(tmp_path)
    finally:
        try:
            Path(tmp_path).unlink()
        except FileNotFoundError:
            pass


def main(argv: list[str] | None = None) -> int:
    """Filled in Task 4."""
    raise NotImplementedError("see Task 4")


if __name__ == "__main__":
    sys.exit(main())
