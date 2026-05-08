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


# ---------------------------------------------------------------------------
# Phase 2 (filled in Task 5)
# ---------------------------------------------------------------------------


def run_phase2(
    v4_csv: str,
    winning_config: dict,
    baseline_v8_csv: str,
    out_csv: str,
) -> dict:
    """Filled in Task 5."""
    raise NotImplementedError("see Task 5")


# ---------------------------------------------------------------------------
# Reliability plot
# ---------------------------------------------------------------------------


def _plot_reliability(
    v8_baseline_df: pd.DataFrame,
    v8_global_df: "pd.DataFrame | None",
    v8_perround_df: "pd.DataFrame | None",
    out_path: str,
    n_bins: int = 10,
) -> None:
    """3-line reliability diagram (predicted prob vs empirical win rate).
    Each frame must have p_a_wins; we treat the symmetric pair frame as
    a per-row probability and use the matching outcome from
    MNCAATourneyCompactResults to compute empirical win rate per bin."""
    results = pd.read_csv(DATA / "MNCAATourneyCompactResults.csv")
    # Build a (season, min_id, max_id) -> outcome (1 if min_id beat max_id else 0).
    outcomes = {}
    for _, r in results.iterrows():
        s = int(r["Season"])
        w, l = int(r["WTeamID"]), int(r["LTeamID"])
        a, b = (w, l) if w < l else (l, w)
        outcomes[(s, a, b)] = 1 if w == a else 0

    def _bin(df, label):
        sub = df.copy()
        sub["pair_key"] = list(zip(sub["season"], sub["team_a"], sub["team_b"]))
        sub = sub[sub["pair_key"].apply(
            lambda k: (int(k[0]), int(k[1]), int(k[2])) in outcomes
            if k[1] < k[2] else (int(k[0]), int(k[2]), int(k[1])) in outcomes
        )]
        bins = np.linspace(0.0, 1.0, n_bins + 1)
        sub["bin"] = pd.cut(sub["p_a_wins"], bins, include_lowest=True, right=True)
        per_bin = []
        for b_interval, grp in sub.groupby("bin"):
            mid = float(b_interval.mid) if hasattr(b_interval, "mid") else 0.5
            outs = []
            for _, row in grp.iterrows():
                key = (int(row["season"]), int(row["team_a"]), int(row["team_b"]))
                key_norm = (key[0], min(key[1], key[2]), max(key[1], key[2]))
                if key_norm not in outcomes:
                    continue
                a_won = outcomes[key_norm]
                if row["team_a"] == key_norm[1]:
                    outs.append(a_won)
                else:
                    outs.append(1 - a_won)
            if outs:
                per_bin.append((mid, float(np.mean(outs)), len(outs)))
        return per_bin

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="ideal")
    for (df, label, color) in [
        (v8_baseline_df, "v8 baseline", "C0"),
        (v8_global_df, "v8 + global T", "C1"),
        (v8_perround_df, "v8 + per-round T", "C2"),
    ]:
        if df is None:
            continue
        pts = _bin(df, label)
        if not pts:
            continue
        xs, ys, ns = zip(*pts)
        ax.plot(xs, ys, "-o", color=color, label=f"{label} (n_bins={len(pts)})")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("predicted P(team_a wins)")
    ax.set_ylabel("empirical win rate")
    ax.set_title("v4 calibration: temperature scaling reliability (10 bins)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Eval v4 calibration via temperature scaling.")
    p.add_argument("--v8-csv", default="output/pairwise_v8.csv")
    p.add_argument("--v4-csv", default="output/pairwise_v4.csv")
    p.add_argument("--baseline-total", type=float, default=2069.0)
    p.add_argument("--out-json", default="output/v4_calibration_eval.json")
    p.add_argument("--out-log", default="output/v4_calibration_eval_log.txt")
    p.add_argument("--out-plot", default="output/v4_calibration_reliability.png")
    p.add_argument("--out-dir", default="output")
    p.add_argument(
        "--phase",
        choices=["phase1", "phase2", "auto"],
        default="auto",
        help="phase1=skip Phase 2 always; phase2=run Phase 2 unconditionally "
             "with a manually-specified winning T; auto=run Phase 1, "
             "trigger Phase 2 only if PASS or MARGINAL.",
    )
    p.add_argument(
        "--phase2-T-config",
        default=None,
        help="(phase=phase2 only) JSON-encoded winning T config. "
             "Either a scalar (e.g. '1.15') or a per-round dict "
             "(e.g. '{\"R64\":1.15,\"R32\":1.0,\"S16\":0.85,\"E8\":1.5,\"F4_NCG\":1.0}').",
    )
    args = p.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)

    # Tee logs to file + stdout.
    log_handler = logging.FileHandler(args.out_log, mode="w")
    log_stream = logging.StreamHandler(sys.stdout)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for h in (log_handler, log_stream):
        h.setFormatter(fmt)
        logging.getLogger().addHandler(h)
    logging.getLogger().setLevel(logging.INFO)

    t_start = time.time()
    summary = {
        "spec": "docs/superpowers/specs/2026-05-08-v4-calibration-temperature-scaling-design.md",
        "plan": "docs/superpowers/plans/2026-05-08-v4-calibration-temperature-scaling.md",
        "baseline_total": float(args.baseline_total),
        "v8_csv": str(args.v8_csv),
    }

    if args.phase in ("phase1", "auto"):
        logger.info("===== PHASE 1: global T sweep =====")
        global_out = run_global_T_sweep(
            v8_csv=args.v8_csv,
            T_grid=T_GRID,
            baseline_total=args.baseline_total,
            scratch_dir=out_dir,
        )
        summary["phase1_global"] = global_out

        logger.info("===== PHASE 1: per-round greedy =====")
        perround_out = run_per_round_greedy(
            v8_csv=args.v8_csv,
            T_grid=T_GRID,
            round_order=ROUND_ORDER,
            baseline_total=args.baseline_total,
            scratch_dir=out_dir,
        )
        summary["phase1_perround"] = perround_out

        # Write the winning frames out for force-add.
        df_v8 = pd.read_csv(args.v8_csv)

        best_global_T = float(global_out["best_cell"]["T"])
        global_winner_df = scale_pairwise(df_v8, T=best_global_T)
        global_winner_path = (
            out_dir / f"pairwise_v8_calibrated_global_T{best_global_T:.2f}.csv"
        )
        global_winner_df.to_csv(global_winner_path, index=False)
        summary["phase1_global"]["winner_csv"] = str(global_winner_path)

        slots_df = pd.read_csv(DATA / "MNCAATourneySlots.csv")
        seeds_df = pd.read_csv(DATA / "MNCAATourneySeeds.csv")
        df_v8_bucketed = df_v8.copy()
        df_v8_bucketed["round_bucket"] = assign_round_buckets(
            df_v8_bucketed, slots_df, seeds_df
        )
        df_v8_resolved = df_v8_bucketed.dropna(subset=["round_bucket"]).copy()
        T_winner = perround_out["winning_T"]
        perround_resolved = scale_pairwise(df_v8_resolved, T=T_winner)
        perround_winner_df = pd.concat(
            [perround_resolved.drop(columns=["round_bucket"]),
             df_v8_bucketed[df_v8_bucketed["round_bucket"].isna()].drop(
                 columns=["round_bucket"])],
            ignore_index=True,
        )
        perround_filename = (
            "pairwise_v8_calibrated_perround_"
            + "_".join(f"{T_winner[r]:.2f}" for r in ROUND_ORDER)
            + ".csv"
        )
        perround_winner_path = out_dir / perround_filename
        perround_winner_df.to_csv(perround_winner_path, index=False)
        summary["phase1_perround"]["winner_csv"] = str(perround_winner_path)

        # Reliability plot.
        _plot_reliability(
            v8_baseline_df=df_v8,
            v8_global_df=global_winner_df,
            v8_perround_df=perround_winner_df,
            out_path=str(args.out_plot),
        )
        summary["plot"] = str(args.out_plot)

        # Decide overall Phase 1 verdict from the better of the two cells.
        global_delta = global_out["best_cell"]["delta_total"]
        perround_delta = perround_out["winning_cell"]["delta_total"]
        if perround_delta > global_delta:
            best_phase1 = perround_out["winning_cell"]
            best_kind = "per-round"
        else:
            best_phase1 = global_out["best_cell"]
            best_kind = "global"
        phase1_verdict = _classify_verdict(
            delta_total=best_phase1["delta_total"],
            drop_best_delta=best_phase1["drop_best_season_delta"],
            wins=best_phase1["wins"],
        )
        summary["phase1_overall"] = {
            "verdict": phase1_verdict,
            "best_kind": best_kind,
            "best_cell": best_phase1,
        }
        logger.info(
            "PHASE 1 OVERALL: verdict=%s best_kind=%s delta=%+.1f drop_best=%+.1f wins=%d",
            phase1_verdict, best_kind,
            best_phase1["delta_total"],
            best_phase1["drop_best_season_delta"],
            best_phase1["wins"],
        )

    # Phase 2 (Task 5).
    if args.phase == "phase2" or (
        args.phase == "auto"
        and summary["phase1_overall"]["verdict"] in ("PASS", "MARGINAL")
    ):
        logger.info("===== PHASE 2: retrain v8 on rescaled v4 =====")
        if args.phase == "phase2":
            if args.phase2_T_config is None:
                raise SystemExit("--phase=phase2 requires --phase2-T-config")
            T_cfg = json.loads(args.phase2_T_config)
        else:
            best_kind = summary["phase1_overall"]["best_kind"]
            if best_kind == "global":
                T_cfg = float(global_out["best_cell"]["T"])
            else:
                T_cfg = perround_out["winning_T"]
        phase2_out = run_phase2(
            v4_csv=args.v4_csv,
            winning_config=T_cfg,
            baseline_v8_csv=args.v8_csv,
            out_csv=str(out_dir / "pairwise_v8_phase2.csv"),
        )
        summary["phase2"] = phase2_out
        logger.info(
            "PHASE 2: verdict=%s delta=%+.1f drop_best=%+.1f wins=%d",
            phase2_out["verdict"],
            phase2_out["cell"]["delta_total"],
            phase2_out["cell"]["drop_best_season_delta"],
            phase2_out["cell"]["wins"],
        )
    elif args.phase == "auto":
        logger.info("PHASE 2 SKIPPED: Phase 1 verdict was %s", summary["phase1_overall"]["verdict"])
        summary["phase2"] = {"skipped": True, "reason": "Phase 1 NO-GO"}

    summary["wall_seconds"] = time.time() - t_start
    Path(args.out_json).write_text(json.dumps(summary, indent=2, default=str))
    logger.info("wrote %s", args.out_json)
    logger.info("wall: %.1f seconds", summary["wall_seconds"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
