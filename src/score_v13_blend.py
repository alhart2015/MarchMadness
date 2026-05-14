"""CLI for the v13 production blend: v4 x v8-ensemble, toss-up bucket only.

Produces output/pairwise_v13.csv by blending stage-1 v4 with a stage-2
v8 frame at alpha=0.6 when max(p_v4, 1-p_v4) < 0.55, else pure v4.

Architecture motivation, audit findings, and the 22-season LOSO backtest
result (2106 brkt pts vs 2034 same-env v8 single-seed baseline) live in
TODO.md under the 2026-05-14 Done entry.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.blend_v4_v8 import BlendEvaluator


def bucket_for_p(p: float, upper_edge: float = 0.55) -> int:
    """Toss-up bucket = confidence (=max(p, 1-p)) below upper_edge. Else bucket 1."""
    conf = max(p, 1.0 - p)
    return 0 if conf < upper_edge else 1


def make_blend(v8_df: pd.DataFrame, v4_df: pd.DataFrame,
               toss_up_alpha: float = 0.6,
               toss_up_upper_edge: float = 0.55) -> pd.DataFrame:
    """Produce the v13 blended pairwise frame.

    Toss-up rows (|p_v4 - 0.5| < toss_up_upper_edge - 0.5): blend at toss_up_alpha.
    Other rows: pure v4.
    """
    v4_local = v4_df.drop_duplicates(["season", "team_a", "team_b"], keep="last")
    merged = v8_df.merge(v4_local, on=["season", "team_a", "team_b"], suffixes=["_v8", "_v4"])
    is_toss_up = merged["p_a_wins_v4"].apply(
        lambda p: max(p, 1.0 - p) < toss_up_upper_edge
    )
    alpha = is_toss_up.astype(float) * toss_up_alpha
    merged["p_a_wins"] = alpha * merged["p_a_wins_v8"] + (1 - alpha) * merged["p_a_wins_v4"]
    return merged[["season", "team_a", "team_b", "p_a_wins"]]


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--v8", default="output/pairwise_v8_ens30.csv",
                    help="Stage-2 pairwise CSV (default: 30-seed ensemble)")
    p.add_argument("--v4", default="output/pairwise_v4.csv",
                    help="Stage-1 pairwise CSV (default: canonical v4)")
    p.add_argument("--alpha", type=float, default=0.6)
    p.add_argument("--upper-edge", type=float, default=0.55)
    p.add_argument("--out", default="output/pairwise_v13.csv")
    args = p.parse_args(argv)

    v8 = pd.read_csv(args.v8)
    v4 = pd.read_csv(args.v4)
    blended = make_blend(v8, v4, args.alpha, args.upper_edge)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    blended.to_csv(args.out, index=False)

    ev = BlendEvaluator()
    pts = ev.score_probs_df(blended)
    total = sum(pts.values())
    print(f"v13 (alpha={args.alpha}, edge={args.upper_edge}): {total:.0f} brkt pts over "
          f"{len(pts)} seasons")
    print(f"Output: {args.out}")
    return total


if __name__ == "__main__":
    main()
