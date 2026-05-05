"""Per-season bracket-points breakdown for v9-C clean re-run.

The v9-C 15-cell sweep driver (sweep_v9_weights.py) emits 22-season
totals only. This script reads the v9-C winning cell's per-cell
pairwise CSV plus the clean v8 pairwise CSV, scores each season
individually via score_pairwise_path, and writes a per-season
comparison CSV. Used in the recovery-step-5 v9-C clean-rerun
findings note to show v9-C's W/L spread (matches PR 9's "6W-3L-13T"
profile reporting).

Inputs (CLI args, all required):
  --v9c-pairwise   Path to v9-C winning cell's pairwise CSV
                   (e.g. output/v9c_sweep/pairwise_v9_WU1.25_WM0.csv).
  --v8-pairwise    Path to clean v8 pairwise CSV
                   (e.g. output/pairwise_v8.csv).
  --output         Output CSV path
                   (e.g. output/v9c_clean_per_season.csv).

Output schema:
  season, v8_pts, v9c_pts, delta, winner
where delta = v9c_pts - v8_pts and winner in {'v8', 'v9c', 'tie'}
with tie when abs(delta) < 0.5.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.score_chalk_brackets import score_pairwise_path

TIE_THRESHOLD = 0.5


def _winner(delta: float) -> str:
    if abs(delta) < TIE_THRESHOLD:
        return "tie"
    return "v9c" if delta > 0 else "v8"


def build_breakdown(v9c_pairwise: str, v8_pairwise: str) -> pd.DataFrame:
    """Score each pairwise CSV per-season, return a comparison DataFrame."""
    v9c = score_pairwise_path(v9c_pairwise)["per_season_pts"]
    v8 = score_pairwise_path(v8_pairwise)["per_season_pts"]
    seasons = sorted(set(v9c) | set(v8))
    rows = []
    for s in seasons:
        v8_pts = float(v8.get(s, 0.0))
        v9c_pts = float(v9c.get(s, 0.0))
        delta = v9c_pts - v8_pts
        rows.append({
            "season": int(s),
            "v8_pts": v8_pts,
            "v9c_pts": v9c_pts,
            "delta": delta,
            "winner": _winner(delta),
        })
    return pd.DataFrame(rows, columns=["season", "v8_pts", "v9c_pts",
                                       "delta", "winner"])


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--v9c-pairwise", required=True)
    parser.add_argument("--v8-pairwise", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    df = build_breakdown(args.v9c_pairwise, args.v8_pairwise)
    df.to_csv(args.output, index=False)
    print(f"Wrote {len(df)} seasons to {args.output}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
