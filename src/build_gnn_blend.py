"""Build the v4 + GNN-Phase-2 blended pairwise frame for the v8 retrain.

Uses the cheating-ideal w_v4 from the Task G LL-blend gate (default 0.80).
Output schema matches output/pairwise_v4.csv exactly so train_stage2 can
consume it as a drop-in stage-1 frame.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairwise-v4", default="output/pairwise_v4.csv")
    parser.add_argument("--pairwise-gnn", default="output/pairwise_gnn_phase2.csv")
    parser.add_argument("--out", default="output/pairwise_v4_with_gnn_blend.csv")
    parser.add_argument("--w-v4", type=float, default=0.80,
                        help="Weight on v4 in p_blend = w*v4 + (1-w)*gnn. Default 0.80 (Task G SAGE optimal).")
    args = parser.parse_args(argv)

    v4 = pd.read_csv(args.pairwise_v4)
    v4 = v4.drop_duplicates(["season", "team_a", "team_b"], keep="last")
    gnn = pd.read_csv(args.pairwise_gnn)
    gnn = gnn.drop_duplicates(["season", "team_a", "team_b"], keep="last")

    merged = v4.merge(gnn, on=["season", "team_a", "team_b"],
                      suffixes=("_v4", "_gnn"), how="inner")

    w = args.w_v4
    merged["p_a_wins"] = w * merged["p_a_wins_v4"] + (1 - w) * merged["p_a_wins_gnn"]

    out = merged[["season", "team_a", "team_b", "p_a_wins"]]
    out.to_csv(args.out, index=False)

    n_v4 = len(v4)
    n_gnn = len(gnn)
    n_blend = len(out)
    print(f"v4 rows:       {n_v4}")
    print(f"gnn rows:      {n_gnn}")
    print(f"blended rows:  {n_blend}")
    print(f"w_v4:          {w}")
    print(f"wrote:         {args.out}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
