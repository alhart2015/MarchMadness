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


def _per_season_delta(
    score: dict,
    baseline_per_season: Mapping[int, float],
) -> dict[int, float]:
    """Per-season delta = score - baseline, keyed on baseline's seasons."""
    return {
        int(s): float(score["per_season_pts"][s]) - float(baseline_per_season[s])
        for s in baseline_per_season
    }


# ---------------------------------------------------------------------------
# Phase 1 sweeps
# ---------------------------------------------------------------------------


def run_global_T_sweep(
    v8_csv: str,
    T_grid: list[float],
    baseline_total: float,
    scratch_dir: Path | None = None,
) -> dict:
    """Sweep a single global T over T_grid, scoring 22-season bracket
    points per cell. Returns a dict with 'cells' (per-T summaries),
    'anchor' (T=1.0 cell summary), 'best_cell' (the highest-total cell),
    and 'verdict' ('PASS' | 'MARGINAL' | 'FAIL').

    The anchor: applying T=1.0 must produce a frame byte-equal to v8_csv
    on (season, team_a, team_b, p_a_wins). HALT (raise) if it doesn't.
    """
    if scratch_dir is None:
        scratch_dir = Path(tempfile.gettempdir())
    scratch_dir.mkdir(parents=True, exist_ok=True)

    df_baseline = pd.read_csv(v8_csv)
    baseline_score = score_pairwise_path(v8_csv)
    baseline_per_season = baseline_score["per_season_pts"]
    if abs(baseline_score["total_pts"] - baseline_total) > 1e-6:
        raise RuntimeError(
            f"baseline mismatch: score_pairwise_path returned "
            f"{baseline_score['total_pts']} but caller passed "
            f"baseline_total={baseline_total}"
        )

    # Anchor verification: scale at T=1.0 and check byte-equal.
    anchor_df = scale_pairwise(df_baseline, T=1.0)
    anchor_check = _anchor_check(anchor_df, v8_csv)
    if not anchor_check["matches"]:
        raise RuntimeError(
            f"global T anchor FAILED: T=1.0 scaling did not reproduce "
            f"{v8_csv} byte-equal -- max_abs_diff={anchor_check['max_abs_diff']}"
        )
    anchor_summary = {
        "matches": True,
        "max_abs_diff": float(anchor_check["max_abs_diff"]),
        "total": float(baseline_score["total_pts"]),
    }

    cells = []
    for T in T_grid:
        scaled = scale_pairwise(df_baseline, T=float(T))
        score = _score_pairwise_df(scaled, scratch_dir)
        per_season_delta = _per_season_delta(score, baseline_per_season)
        summary = _summarize_cell(per_season_delta, baseline_total=baseline_total)
        summary["T"] = float(T)
        cells.append(summary)
        logger.info(
            "global T=%.3f -> total=%.1f delta=%+.1f W/L/T=%d/%d/%d drop_best=%+.1f",
            T, summary["total"], summary["delta_total"],
            summary["wins"], summary["losses"], summary["ties"],
            summary["drop_best_season_delta"],
        )

    best_cell = max(cells, key=lambda c: c["delta_total"])
    verdict = _classify_verdict(
        delta_total=best_cell["delta_total"],
        drop_best_delta=best_cell["drop_best_season_delta"],
        wins=best_cell["wins"],
    )
    return {
        "anchor": anchor_summary,
        "cells": cells,
        "best_cell": best_cell,
        "verdict": verdict,
    }


def run_per_round_greedy(
    v8_csv: str,
    T_grid: list[float],
    round_order: list[str],
    baseline_total: float,
    scratch_dir: Path | None = None,
) -> dict:
    """Sequential greedy per-round T sweep.

    For each round R in round_order:
        Hold all other rounds at their best-found T (rounds before R)
        or 1.0 (rounds after R). Sweep T_R over T_grid. Pick best total.
        Fix T_R; advance to next round.

    Returns: {anchor, greedy_chain (one entry per round with chosen T +
              total + per-cell summaries), winning_T (dict), winning_cell
              (full summary), verdict}.
    """
    if scratch_dir is None:
        scratch_dir = Path(tempfile.gettempdir())
    scratch_dir.mkdir(parents=True, exist_ok=True)

    df_baseline = pd.read_csv(v8_csv)
    baseline_score = score_pairwise_path(v8_csv)
    baseline_per_season = baseline_score["per_season_pts"]
    if abs(baseline_score["total_pts"] - baseline_total) > 1e-6:
        raise RuntimeError(
            f"baseline mismatch: {baseline_score['total_pts']} vs "
            f"baseline_total={baseline_total}"
        )

    # Resolve buckets for all rows up front.
    slots_df = pd.read_csv(DATA / "MNCAATourneySlots.csv")
    seeds_df = pd.read_csv(DATA / "MNCAATourneySeeds.csv")
    df_baseline = df_baseline.copy()
    df_baseline["round_bucket"] = assign_round_buckets(df_baseline, slots_df, seeds_df)
    n_total = len(df_baseline)
    df_resolved = df_baseline.dropna(subset=["round_bucket"]).copy()
    n_resolved = len(df_resolved)
    n_dropped = n_total - n_resolved
    logger.info(
        "per-round: %d/%d rows have a resolved round bucket (%d dropped)",
        n_resolved, n_total, n_dropped,
    )

    # Anchor: all-1 per-round dict reproduces resolved-row p_a_wins
    # byte-equal. Direct value comparison (not _anchor_check) because
    # _anchor_check exits early on row-count mismatch (full v8 vs
    # resolved subset) without checking values -- a T=1 implementation
    # bug could slip through that path.
    T_anchor = {b: 1.0 for b in ROUND_BUCKETS}
    anchor_df = scale_pairwise(df_resolved, T=T_anchor)
    max_diff = float(np.abs(
        anchor_df["p_a_wins"].to_numpy() - df_resolved["p_a_wins"].to_numpy()
    ).max())
    if max_diff > 1e-9:
        raise RuntimeError(
            f"per-round anchor FAILED: T=1.0 scaling produced max_abs_diff="
            f"{max_diff} on {n_resolved} resolved rows"
        )
    anchor_summary = {
        "matches": True,
        "max_abs_diff": max_diff,
        "total": float(baseline_score["total_pts"]),
        "n_resolved": int(n_resolved),
        "n_dropped": int(n_dropped),
    }

    # Greedy loop.
    best_T = {b: 1.0 for b in ROUND_BUCKETS}
    chain = []
    for r_idx, round_name in enumerate(round_order):
        cells = []
        for T in T_grid:
            cand_T = dict(best_T)
            cand_T[round_name] = float(T)
            scaled_resolved = scale_pairwise(df_resolved, T=cand_T)
            # Re-attach the dropped NA rows (which carry the v8 baseline
            # probabilities unchanged) for full-bracket scoring.
            scaled_full = pd.concat(
                [scaled_resolved.drop(columns=["round_bucket"]),
                 df_baseline[df_baseline["round_bucket"].isna()].drop(
                     columns=["round_bucket"])],
                ignore_index=True,
            )
            assert len(scaled_full) == n_total, (
                f"row count drift: {len(scaled_full)} vs {n_total}"
            )
            score = _score_pairwise_df(scaled_full, scratch_dir)
            per_season_delta = _per_season_delta(score, baseline_per_season)
            summary = _summarize_cell(per_season_delta, baseline_total=baseline_total)
            summary["T"] = float(T)
            summary["round"] = round_name
            cells.append(summary)
        # Pick best.
        best_cell = max(cells, key=lambda c: c["delta_total"])
        best_T[round_name] = float(best_cell["T"])
        logger.info(
            "per-round step %d/%d (%s): picked T=%.3f, total=%.1f, delta=%+.1f",
            r_idx + 1, len(round_order), round_name,
            best_cell["T"], best_cell["total"], best_cell["delta_total"],
        )
        chain.append({
            "round": round_name,
            "picked_T": float(best_cell["T"]),
            "total_after_step": float(best_cell["total"]),
            "delta_total_after_step": float(best_cell["delta_total"]),
            "all_cells": cells,
        })

    # Re-derive the full summary at winning_T (equal by construction to
    # chain[-1]['total_after_step']).
    winning_full = scale_pairwise(df_resolved, T=best_T)
    winning_full = pd.concat(
        [winning_full.drop(columns=["round_bucket"]),
         df_baseline[df_baseline["round_bucket"].isna()].drop(columns=["round_bucket"])],
        ignore_index=True,
    )
    assert len(winning_full) == n_total, (
        f"row count drift: {len(winning_full)} vs {n_total}"
    )
    winning_score = _score_pairwise_df(winning_full, scratch_dir)
    winning_per_season_delta = _per_season_delta(winning_score, baseline_per_season)
    winning_summary = _summarize_cell(
        winning_per_season_delta, baseline_total=baseline_total
    )
    winning_summary["T"] = dict(best_T)

    verdict = _classify_verdict(
        delta_total=winning_summary["delta_total"],
        drop_best_delta=winning_summary["drop_best_season_delta"],
        wins=winning_summary["wins"],
    )
    return {
        "anchor": anchor_summary,
        "greedy_chain": chain,
        "winning_T": dict(best_T),
        "winning_cell": winning_summary,
        "verdict": verdict,
    }


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
