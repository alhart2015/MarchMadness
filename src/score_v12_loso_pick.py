"""v12 LOSO cell picker.

Given a set of v12 stage-2 pairwise frames produced by train_stage2_v10
under different (N, hparams) settings, pick one cell per test season Y
based on training-season scores (under the v13 toss-up blend, alpha=0.6),
then concatenate the picked rows into a single pairwise frame.

Spec: docs/superpowers/specs/2026-05-14-v12-stage2-v4-feature-diffs-design.md
Plan: docs/superpowers/plans/2026-05-14-v12-stage2-v4-feature-diffs.md
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from src.blend_v4_v8 import BlendEvaluator
from src.score_v13_blend import make_blend

V13_BASELINE = 2106.0
DEFAULT_TOSS_UP_ALPHA = 0.6
DEFAULT_TOSS_UP_UPPER_EDGE = 0.55


def _cell_name_from_path(path: str) -> str:
    """e.g. 'output/pairwise_v12_n10_v8.csv' -> 'n10_v8'."""
    stem = Path(path).stem
    m = re.match(r"pairwise_v12_(.+)", stem)
    if m:
        return m.group(1)
    return stem


def score_cells(
    cell_frames: Dict[str, pd.DataFrame],
    v4_df: pd.DataFrame,
    alpha: float = DEFAULT_TOSS_UP_ALPHA,
    upper_edge: float = DEFAULT_TOSS_UP_UPPER_EDGE,
    evaluator: BlendEvaluator = None,
) -> Dict[str, Dict[int, float]]:
    """For each cell, compute the v13-style blend with v4 and score per-season.

    Returns {cell_name: {season: brkt_pts}}."""
    ev = evaluator or BlendEvaluator()
    out: Dict[str, Dict[int, float]] = {}
    for name, frame in cell_frames.items():
        blend = make_blend(frame, v4_df, toss_up_alpha=alpha, toss_up_upper_edge=upper_edge)
        per_season = ev.score_probs_df(blend)
        out[name] = {int(s): float(p) for s, p in per_season.items()}
    return out


def pick_cell_per_season(
    cell_totals_per_season: Dict[str, Dict[int, float]],
) -> Dict[int, str]:
    """For each test season Y, pick the cell with highest summed brkt pts
    across all training seasons {T : T != Y}."""
    cell_names = list(cell_totals_per_season.keys())
    if not cell_names:
        return {}
    all_seasons = sorted(cell_totals_per_season[cell_names[0]].keys())
    picks: Dict[int, str] = {}
    for test_season in all_seasons:
        train_seasons = [s for s in all_seasons if s != test_season]
        cell_train_totals = {
            name: sum(cell_totals_per_season[name].get(s, 0.0) for s in train_seasons)
            for name in cell_names
        }
        picks[test_season] = max(cell_train_totals, key=cell_train_totals.get)
    return picks


def concatenate_picked_rows(
    cell_frames: Dict[str, pd.DataFrame],
    picks: Dict[int, str],
) -> pd.DataFrame:
    """Return a frame with one row per (season, team_a, team_b) drawn from
    the picked cell for that season."""
    parts = []
    for season in sorted(picks.keys()):
        cell = picks[season]
        frame = cell_frames[cell]
        parts.append(frame[frame["season"] == season])
    return pd.concat(parts, ignore_index=True)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--cells", required=True,
                   help="Comma-separated paths to per-cell pairwise CSVs.")
    p.add_argument("--v4", default="output/pairwise_v4.csv")
    p.add_argument("--alpha", type=float, default=DEFAULT_TOSS_UP_ALPHA)
    p.add_argument("--upper-edge", type=float, default=DEFAULT_TOSS_UP_UPPER_EDGE)
    p.add_argument("--out", default="output/pairwise_v12.csv",
                   help="Picked-cell frame (stage-2 output, pre-blend).")
    p.add_argument("--blend-out", default="output/pairwise_v12_blend.csv",
                   help="v13-style blended frame on the picked-cell stage-2.")
    p.add_argument("--summary-out", default="output/v12_loso_pick_summary.json")
    args = p.parse_args(argv)

    cell_paths = [s for s in args.cells.split(",") if s.strip()]
    cell_frames = {
        _cell_name_from_path(path): pd.read_csv(path) for path in cell_paths
    }
    print(f"Loaded {len(cell_frames)} cells: {list(cell_frames.keys())}")

    v4 = pd.read_csv(args.v4).drop_duplicates(["season", "team_a", "team_b"], keep="last")
    ev = BlendEvaluator()

    print("\nScoring cells under v13 blend...")
    cell_per_season = score_cells(cell_frames, v4, args.alpha, args.upper_edge, ev)
    for name, ps in cell_per_season.items():
        print(f"  {name}: total={sum(ps.values()):.0f} brkt pts (mean={sum(ps.values())/len(ps):.1f}/season)")

    picks = pick_cell_per_season(cell_per_season)
    pick_counts = {}
    for _, cell in picks.items():
        pick_counts[cell] = pick_counts.get(cell, 0) + 1
    print("\nLOSO pick distribution:")
    for cell, n in sorted(pick_counts.items(), key=lambda kv: -kv[1]):
        print(f"  {cell}: {n}/22 seasons")

    picked = concatenate_picked_rows(cell_frames, picks)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    picked.to_csv(args.out, index=False)
    print(f"\nWrote {args.out} ({len(picked):,} rows)")

    blend = make_blend(picked, v4, toss_up_alpha=args.alpha, toss_up_upper_edge=args.upper_edge)
    blend.to_csv(args.blend_out, index=False)
    final_per_season = ev.score_probs_df(blend)
    total = sum(final_per_season.values())
    print(f"Wrote {args.blend_out}")
    print(f"\nv12 picked-cell @ v13 blend: {total:.0f} brkt pts")
    print(f"  vs v13 baseline 2106:        {total - V13_BASELINE:+.0f}")

    summary = {
        "picks_per_season": {int(s): c for s, c in picks.items()},
        "pick_counts": pick_counts,
        "cell_totals_per_season": cell_per_season,
        "v12_total": float(total),
        "v13_baseline": V13_BASELINE,
        "delta_vs_v13": float(total - V13_BASELINE),
        "alpha": args.alpha,
        "upper_edge": args.upper_edge,
        "final_per_season": {int(s): float(p) for s, p in final_per_season.items()},
    }
    with open(args.summary_out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {args.summary_out}")

    return total


if __name__ == "__main__":
    main()
