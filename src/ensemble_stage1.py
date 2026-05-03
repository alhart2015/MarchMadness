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

    # Inputs from v3's LOSO loop may have duplicate (season, team_a, team_b)
    # rows from default + tuned passes; take the last write per pair, matching
    # train_upset_model.load_per_game_data_with_upset and eval_stage1.
    df_a = df_a.drop_duplicates(subset=JOIN_KEYS, keep="last")
    df_b = df_b.drop_duplicates(subset=JOIN_KEYS, keep="last")

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


def blend_pairwise_csvs(
    inputs: list,
    weights: list,
    out: str,
) -> None:
    """K-way generalization of average_pairwise_csvs.

    inputs:  list of paths to pairwise CSVs (schema season, team_a, team_b, p_a_wins).
             All inputs must share identical (season, team_a, team_b) coverage.
    weights: list of per-input non-negative weights, len == len(inputs),
             must sum to 1.0 within 1e-9.
    out:     path to write the blended CSV (same schema as inputs).
    """
    if len(weights) != len(inputs):
        raise ValueError(
            f"weights count ({len(weights)}) != inputs count ({len(inputs)})"
        )
    w_sum = sum(float(w) for w in weights)
    if abs(w_sum - 1.0) > 1e-9:
        raise ValueError(
            f"weights must sum to 1; got {w_sum:.6f}"
        )

    dfs = [pd.read_csv(p) for p in inputs]
    for i, df in enumerate(dfs):
        missing = set(SCHEMA) - set(df.columns)
        if missing:
            raise ValueError(
                f"input {i} ({inputs[i]}) missing columns: {sorted(missing)}"
            )
        dfs[i] = df.drop_duplicates(subset=JOIN_KEYS, keep="last")

    # Inner-join all inputs on (season, team_a, team_b); coverage check.
    base = dfs[0][JOIN_KEYS + ["p_a_wins"]].rename(columns={"p_a_wins": "p_0"})
    for i, df in enumerate(dfs[1:], start=1):
        rhs = df[JOIN_KEYS + ["p_a_wins"]].rename(columns={"p_a_wins": f"p_{i}"})
        base = base.merge(rhs, on=JOIN_KEYS, how="outer", indicator=True)
        only_left = (base["_merge"] == "left_only").sum()
        only_right = (base["_merge"] == "right_only").sum()
        if only_left or only_right:
            raise ValueError(
                f"input {i} coverage mismatch: {only_left} rows only in prior "
                f"inputs, {only_right} rows only in input {i}; the blend "
                "requires identical (season, team_a, team_b) coverage"
            )
        base = base.drop(columns=["_merge"])

    # Weighted sum of the per-input p columns.
    p_blend = sum(
        float(weights[i]) * base[f"p_{i}"]
        for i in range(len(inputs))
    )
    base["p_a_wins"] = p_blend

    out_df = (
        base[SCHEMA]
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
