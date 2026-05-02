"""Average two pairwise-prediction CSVs into an ensemble pairwise CSV.

Spec: docs/superpowers/specs/2026-05-01-ensemble-stage1-design.md

Pure utility: no model. Joins on (season, team_a, team_b) and computes
    p_ensemble = w_a * p_a + w_b * p_b
The output schema is identical to pairwise_v4.csv:
    season, team_a, team_b, p_a_wins.

Anchor: --weights 1.0,0.0 reproduces input A row-for-row; the
LOSO experiment depends on this anchor passing before any
ensemble-vs-baseline numbers are trusted.
"""
import argparse
import sys
from pathlib import Path
from typing import Tuple

import pandas as pd

SCHEMA = ["season", "team_a", "team_b", "p_a_wins"]
JOIN_KEYS = ["season", "team_a", "team_b"]


def average_pairwise_csvs(
    in_a: str, in_b: str, out: str, weights: Tuple[float, float] = (0.5, 0.5)
) -> None:
    """Average two pairwise CSVs and write the result.

    in_a, in_b: paths to CSVs with columns season, team_a, team_b, p_a_wins.
                team_a < team_b (canonical orientation).
    out:        path to write the averaged CSV (same schema).
    weights:    (w_a, w_b) -- must sum to 1.0 within 1e-9.
    """
    w_a, w_b = float(weights[0]), float(weights[1])
    if abs(w_a + w_b - 1.0) > 1e-9:
        raise ValueError(
            f"weights must sum to 1; got {w_a} + {w_b} = {w_a + w_b}"
        )

    df_a = pd.read_csv(in_a)
    df_b = pd.read_csv(in_b)

    for label, df in [("a", df_a), ("b", df_b)]:
        missing = set(SCHEMA) - set(df.columns)
        if missing:
            raise ValueError(
                f"input {label} ({in_a if label == 'a' else in_b}) missing "
                f"columns: {sorted(missing)}"
            )

    # Inner join + coverage check.
    merged = df_a.merge(
        df_b, on=JOIN_KEYS, suffixes=("_a", "_b"), how="outer", indicator=True
    )
    only_a = (merged["_merge"] == "left_only").sum()
    only_b = (merged["_merge"] == "right_only").sum()
    if only_a or only_b:
        raise ValueError(
            f"join coverage failed: {only_a} rows only in A, "
            f"{only_b} rows only in B; the ensemble requires "
            "byte-identical pair coverage across inputs"
        )
    merged = merged.drop(columns=["_merge"])

    merged["p_a_wins"] = w_a * merged["p_a_wins_a"] + w_b * merged["p_a_wins_b"]

    out_df = (
        merged[SCHEMA]
        .sort_values(JOIN_KEYS)
        .reset_index(drop=True)
    )
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out, index=False)


def _parse_weights(s: str) -> Tuple[float, float]:
    parts = s.split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(
            f"weights must be 'w_a,w_b' (two comma-separated floats); got {s!r}"
        )
    return float(parts[0]), float(parts[1])


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Average two pairwise-prediction CSVs."
    )
    parser.add_argument("--in-a", required=True, help="first input CSV")
    parser.add_argument("--in-b", required=True, help="second input CSV")
    parser.add_argument("--out", required=True, help="output CSV")
    parser.add_argument(
        "--weights", type=_parse_weights, default=(0.5, 0.5),
        help="comma-separated weights 'w_a,w_b' (must sum to 1; default 0.5,0.5)"
    )
    args = parser.parse_args(argv)
    average_pairwise_csvs(args.in_a, args.in_b, args.out, args.weights)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
