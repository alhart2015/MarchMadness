"""Eval driver for the R64 closing-line override experiment.

Spec: docs/superpowers/specs/2026-05-07-v4-r64-line-blend-design.md
Strategy: docs/notes/2026-05-07-v4-kaggle-gap-strategy.md

Pipeline:
    1. SIGMA sweep on the cheap R64 LL gate (mode=hard).
    2. Pick min-LL sigma.
    3. For each mode in [hard, mean] at picked sigma:
         apply override -> v4_overridden.csv
         train v8 on UNMODIFIED v4, apply to v4_overridden -> v8_overridden.csv
         score v8_overridden -> total + per-season brkt pts
    4. Anchor: v4-only (no override) v8 == output/pairwise_v8.csv (byte-equal,
       2069 brkt pts). HALT if anchor fails.
    5. Pick verdict per spec decision matrix.

Outputs:
    output/pairwise_v4_r64lineblend_<mode>_sigma<S>.csv
    output/pairwise_v8_r64lineblend_<mode>_sigma<S>.csv
    output/r64_line_blend_eval.json
    output/r64_line_blend_calibration.png
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.build_r64_line_override import (  # noqa: E402
    apply_r64_override,
    sigma_sweep_ll,
)
from src.train_stage2 import (  # noqa: E402
    DATA,
    build_v8_pairwise,
    load_per_game_data,
)
from src.score_chalk_brackets import score_pairwise_path  # noqa: E402

logger = logging.getLogger(__name__)

DEFAULT_SIGMAS = [9.0, 10.0, 11.0, 12.0, 13.0]
DEFAULT_MODES = ["hard", "mean"]
PASS_BAR = 25
MARGINAL_BAR = 10
CONCENTRATION_THRESHOLD = 0.5  # >50% single-season concentration demotes PASS


# ---------------------------------------------------------------------------
# Train-on-unmodified, apply-to-overridden discipline
# ---------------------------------------------------------------------------


def _run_v8_on_overridden(
    v4_csv_unmodified: str,
    v4_csv_overridden: str,
    out_csv: str,
) -> str:
    """Train v8 stage-2 on per-game data derived from UNMODIFIED v4,
    then apply the trained-per-LOSO-season models to the OVERRIDDEN
    pairwise frame for the corresponding season.

    `train_stage2.build_v8_pairwise` already supports this discipline:
    it takes `per_game` (training source) and `pairwise_v4_csv` (apply
    target) as independent inputs. We pass UNMODIFIED to the former
    and OVERRIDDEN to the latter.
    """
    print(f"[r64-eval] training v8 on UNMODIFIED v4 ({v4_csv_unmodified}), "
          f"applying to OVERRIDDEN ({v4_csv_overridden}) ...")
    seeds_csv = str(DATA / "MNCAATourneySeeds.csv")
    results_csv = str(DATA / "MNCAATourneyCompactResults.csv")
    per_game = load_per_game_data(v4_csv_unmodified, results_csv, seeds_csv)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    build_v8_pairwise(per_game, v4_csv_overridden, seeds_csv, out_csv)
    return out_csv


# ---------------------------------------------------------------------------
# Anchor check + scoring
# ---------------------------------------------------------------------------


def _anchor_check(actual_csv: str, expected_csv: str) -> dict:
    """Verify two pairwise CSVs match on (season, team_a, team_b, p_a_wins).
    Returns {matches, max_abs_diff, n_rows}.
    """
    a = pd.read_csv(actual_csv).drop_duplicates(
        ["season", "team_a", "team_b"], keep="last"
    )
    b = pd.read_csv(expected_csv).drop_duplicates(
        ["season", "team_a", "team_b"], keep="last"
    )
    merged = a.merge(b, on=["season", "team_a", "team_b"],
                     suffixes=("_actual", "_expected"))
    n_only_a = len(a) - len(merged)
    n_only_b = len(b) - len(merged)
    if n_only_a != 0 or n_only_b != 0:
        return {
            "matches": False,
            "max_abs_diff": float("nan"),
            "n_only_actual": int(n_only_a),
            "n_only_expected": int(n_only_b),
            "n_rows_overlap": int(len(merged)),
        }
    diff = (merged["p_a_wins_actual"] - merged["p_a_wins_expected"]).abs()
    max_diff = float(diff.max())
    return {
        "matches": bool(max_diff < 1e-9),
        "max_abs_diff": max_diff,
        "n_rows": int(len(merged)),
    }


def _score(pairwise_csv: str) -> dict:
    """Wrap score_chalk_brackets -> {total_pts, per_season}."""
    s = score_pairwise_path(pairwise_csv)
    return {
        "total_pts": float(s["total_pts"]),
        "per_season": {int(k): float(v) for k, v in s["per_season_pts"].items()},
    }


# ---------------------------------------------------------------------------
# Verdict logic
# ---------------------------------------------------------------------------


def _pick_verdict(cells: list[dict], baseline_total: float) -> dict:
    """Per the spec decision matrix:
    - PASS: best cell delta >= +25 AND single-season concentration <= 50%
    - MARGINAL: best cell delta in [+10, +25), OR PASS-magnitude
      with concentration > 50%
    - FAIL: best cell delta < +10 across all cells
    """
    if not cells:
        return {"label": "FAIL", "summary": "no cells", "best_mode": None}
    best = max(cells, key=lambda c: c["delta_total"])
    delta = best["delta_total"]

    if delta < MARGINAL_BAR:
        return {
            "label": "FAIL",
            "summary": (f"best cell delta {delta:+.0f} below MARGINAL bar "
                        f"({MARGINAL_BAR}). Close the data-hypothesis lane; "
                        f"calibration-shape engineering becomes #1."),
            "best_mode": best["mode"],
            "best_delta": float(delta),
        }

    # Single-season concentration check.
    biggest_swing_abs = abs(float(best.get("biggest_swing_value", 0.0)))
    concentration = (biggest_swing_abs / abs(delta)) if delta != 0 else 0.0
    concentration_demote = (concentration > CONCENTRATION_THRESHOLD)

    if delta >= PASS_BAR and not concentration_demote:
        return {
            "label": "PASS",
            "summary": (f"best cell delta {delta:+.0f} >= PASS bar "
                        f"({PASS_BAR}); robust profile (max single-season "
                        f"swing {biggest_swing_abs:.0f} = "
                        f"{concentration:.0%} of total). Proceed to "
                        f"Phase 2: re-train v8 on overridden frame."),
            "best_mode": best["mode"],
            "best_delta": float(delta),
            "concentration": float(concentration),
        }

    if delta >= PASS_BAR and concentration_demote:
        return {
            "label": "MARGINAL",
            "summary": (f"best cell delta {delta:+.0f} >= PASS bar but max "
                        f"single-season swing {biggest_swing_abs:.0f} = "
                        f"{concentration:.0%} of total (>50%); demoted to "
                        f"MARGINAL. Retain code; no production swap."),
            "best_mode": best["mode"],
            "best_delta": float(delta),
            "concentration": float(concentration),
        }

    return {
        "label": "MARGINAL",
        "summary": (f"best cell delta {delta:+.0f} in [+{MARGINAL_BAR}, "
                    f"+{PASS_BAR}). Retain code; no production swap. "
                    f"Calibration-shape engineering takes lead by "
                    f"elimination."),
        "best_mode": best["mode"],
        "best_delta": float(delta),
    }


# ---------------------------------------------------------------------------
# Calibration plot
# ---------------------------------------------------------------------------


def _plot_calibration(
    v4_pw: pd.DataFrame, override_pws: dict[str, pd.DataFrame],
    results_df: pd.DataFrame, out_path: Path,
) -> None:
    """Compare R64 calibration of v4-only vs each override mode.
    `override_pws` is {mode_label: pw_df}.
    """
    # Build R64 winner lookup.
    r64_winners = {}
    for _, g in results_df.iterrows():
        season = int(g["Season"])
        daynum = int(g["DayNum"])
        if season < 2003 or season > 2025:
            continue
        # _round_from_daynum lives in audit_v4_gap_vegas; re-import locally.
        from src.audit_v4_gap_vegas import _round_from_daynum
        if _round_from_daynum(daynum) != "R64":
            continue
        w, l = int(g["WTeamID"]), int(g["LTeamID"])
        a, b = (w, l) if w < l else (l, w)
        r64_winners[(season, a, b)] = 1 if w == a else 0

    def _r64_calibration(pw_df):
        edges = np.linspace(0.0, 1.0, 11)
        rows = []
        for _, r in pw_df.iterrows():
            key = (int(r["season"]), int(r["team_a"]), int(r["team_b"]))
            if key not in r64_winners:
                continue
            rows.append({"p": float(r["p_a_wins"]), "y": int(r64_winners[key])})
        df = pd.DataFrame(rows)
        out = []
        for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
            if i == len(edges) - 2:
                mask = (df["p"] >= lo) & (df["p"] <= hi)
            else:
                mask = (df["p"] >= lo) & (df["p"] < hi)
            n = int(mask.sum())
            empirical = float(df.loc[mask, "y"].mean()) if n > 0 else None
            out.append({"mid": float((lo + hi) / 2), "n": n,
                        "empirical": empirical})
        return out

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.5, label="ideal")
    for label, pw in [("v4-only", v4_pw)] + list(override_pws.items()):
        cal = _r64_calibration(pw)
        xs = [c["mid"] for c in cal if c["empirical"] is not None]
        ys = [c["empirical"] for c in cal if c["empirical"] is not None]
        ax.plot(xs, ys, marker="o", label=label)
    ax.set_xlabel("predicted P(team_a wins)")
    ax.set_ylabel("empirical rate")
    ax.set_title("R64 calibration: v4 vs override modes")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main eval driver
# ---------------------------------------------------------------------------


def run_eval(
    v4_csv: str = "output/pairwise_v4.csv",
    v8_baseline_csv: str = "output/pairwise_v8.csv",
    sigmas: list[float] | None = None,
    modes: list[str] | None = None,
    out_dir: str | Path = "output",
    out_json: str = "output/r64_line_blend_eval.json",
) -> dict:
    if sigmas is None:
        sigmas = list(DEFAULT_SIGMAS)
    if modes is None:
        modes = list(DEFAULT_MODES)

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("R64 LINE BLEND -- EVAL")
    print("=" * 70)

    # --- 1. SIGMA sweep ---
    print()
    print("STEP 1: SIGMA sweep (mode=hard, R64-only LL)")
    sigma_rows = sigma_sweep_ll(v4_csv, sigmas, mode="hard")
    picked_sigma = float(min(sigma_rows, key=lambda r: r["ll"])["sigma"])
    print(f"  picked sigma = {picked_sigma}")

    # --- 2. Anchor: score v8 baseline ---
    print()
    print("STEP 2: anchor -- score canonical v8 baseline")
    baseline_score = _score(v8_baseline_csv)
    print(f"  v8 baseline total: {baseline_score['total_pts']:.0f} brkt pts")

    # --- 3. Per-mode override + train+apply v8 + score ---
    cells = []
    override_pws_for_plot = {}
    for mode in modes:
        print()
        print("=" * 70)
        print(f"CELL  mode={mode}  sigma={picked_sigma}")
        print("=" * 70)

        v4_overridden = str(out_path /
            f"pairwise_v4_r64lineblend_{mode}_sigma{int(picked_sigma)}.csv")
        v8_overridden = str(out_path /
            f"pairwise_v8_r64lineblend_{mode}_sigma{int(picked_sigma)}.csv")

        t0 = time.time()
        cov_stats = apply_r64_override(
            v4_csv=v4_csv, mode=mode, sigma=picked_sigma,
            out_csv=v4_overridden,
        )
        _run_v8_on_overridden(v4_csv, v4_overridden, v8_overridden)

        cell_score = _score(v8_overridden)
        deltas = {
            s: cell_score["per_season"][s] - baseline_score["per_season"].get(s, 0.0)
            for s in cell_score["per_season"]
        }
        wins = sum(1 for d in deltas.values() if d > 0)
        losses = sum(1 for d in deltas.values() if d < 0)
        ties = sum(1 for d in deltas.values() if d == 0)
        delta_total = cell_score["total_pts"] - baseline_score["total_pts"]
        max_swing_season, max_swing_val = max(
            deltas.items(), key=lambda kv: abs(kv[1]), default=(None, 0.0)
        )

        elapsed = time.time() - t0
        print(f"  total: {cell_score['total_pts']:.0f}  "
              f"delta: {delta_total:+.0f}  "
              f"W/L/T: {wins}/{losses}/{ties}  "
              f"biggest_swing: {max_swing_season} ({max_swing_val:+.0f})  "
              f"({elapsed:.1f}s)")

        cells.append({
            "mode": mode,
            "sigma": picked_sigma,
            "v4_csv": v4_overridden,
            "v8_csv": v8_overridden,
            "coverage": cov_stats,
            "total_pts": cell_score["total_pts"],
            "delta_total": float(delta_total),
            "per_season_pts": cell_score["per_season"],
            "per_season_delta": {int(k): float(v) for k, v in deltas.items()},
            "wins": wins, "losses": losses, "ties": ties,
            "biggest_swing_season": int(max_swing_season) if max_swing_season is not None else None,
            "biggest_swing_value": float(max_swing_val),
            "fold_seconds": round(elapsed, 1),
        })
        override_pws_for_plot[f"{mode} (sigma={int(picked_sigma)})"] = pd.read_csv(v4_overridden)

    # --- 4. Anchor invariance check (v4-only against canonical baseline) ---
    print()
    print("STEP 4: anchor invariance -- v8 on UNMODIFIED v4 must == baseline")
    v8_anchor_csv = str(out_path / "pairwise_v8_r64lineblend_v4only.csv")
    _run_v8_on_overridden(v4_csv, v4_csv, v8_anchor_csv)
    anchor = _anchor_check(v8_anchor_csv, v8_baseline_csv)
    print(f"  matches={anchor['matches']}  max_abs_diff={anchor.get('max_abs_diff', 'n/a')}")
    if not anchor["matches"]:
        print("  *** ANCHOR FAILED *** verdict suppressed; investigate.")

    # --- 5. Verdict ---
    verdict = _pick_verdict(cells, baseline_total=baseline_score["total_pts"])
    print()
    print("=" * 70)
    print(f"VERDICT: {verdict['label']}")
    print("=" * 70)
    print(f"  {verdict['summary']}")

    # --- 6. Plots + JSON ---
    if cells:
        _plot_calibration(
            v4_pw=pd.read_csv(v4_csv).drop_duplicates(
                ["season", "team_a", "team_b"], keep="last"),
            override_pws=override_pws_for_plot,
            results_df=pd.read_csv(DATA / "MNCAATourneyCompactResults.csv"),
            out_path=out_path / "r64_line_blend_calibration.png",
        )

    summary = {
        "config": {
            "v4_csv": v4_csv,
            "v8_baseline_csv": v8_baseline_csv,
            "sigmas": sigmas,
            "modes": modes,
            "picked_sigma": picked_sigma,
        },
        "sigma_sweep": sigma_rows,
        "v8_baseline": baseline_score,
        "anchor_check": anchor,
        "cells": cells,
        "verdict": verdict,
    }
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  saved {out_json}")
    return summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--v4", default="output/pairwise_v4.csv")
    parser.add_argument("--v8-baseline", default="output/pairwise_v8.csv")
    parser.add_argument("--sigmas", default=",".join(f"{s:.1f}" for s in DEFAULT_SIGMAS))
    parser.add_argument("--modes", default=",".join(DEFAULT_MODES))
    parser.add_argument("--out-dir", default="output")
    parser.add_argument("--out-json", default="output/r64_line_blend_eval.json")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    sigmas = [float(s) for s in args.sigmas.split(",") if s.strip()]
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    run_eval(
        v4_csv=args.v4, v8_baseline_csv=args.v8_baseline,
        sigmas=sigmas, modes=modes,
        out_dir=args.out_dir, out_json=args.out_json,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
