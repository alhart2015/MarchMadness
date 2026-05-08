"""Post-hoc temperature scaling on pairwise probability frames.

Spec:  docs/superpowers/specs/2026-05-08-v4-calibration-temperature-scaling-design.md
Plan:  docs/superpowers/plans/2026-05-08-v4-calibration-temperature-scaling.md

Public API:
    scale_pairwise(df, T)
        df: DataFrame with columns (season, team_a, team_b, p_a_wins).
            For per-round T, must also have a 'round_bucket' column with
            values among {'R64','R32','S16','E8','F4_NCG'} (rounds 5+6
            collapsed -- see _round_int_to_bucket).
        T:  float or dict[str, float] keyed on round bucket.
        Returns: a NEW DataFrame, same shape, with p_a_wins rescaled.

    assign_round_buckets(df, slots_df, seeds_df)
        df: DataFrame with (season, team_a, team_b).
        slots_df, seeds_df: Kaggle MNCAATourneySlots / MNCAATourneySeeds.
        Returns: pd.Series indexed like df, dtype=object, values in
                 {'R64','R32','S16','E8','F4_NCG'} or pd.NA for rows
                 with no resolvable round (play-in pairs, missing seed
                 mappings).

CLI:
    python -m src.apply_temperature_scaling \\
        --in output/pairwise_v8.csv \\
        --T 1.15 \\
        --out output/pairwise_v8_T1.15.csv

(Per-round CLI not provided -- the eval driver wires per-round dicts.)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Mapping, Union

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.train_upset_model import build_pair_round_lookup  # noqa: E402

ROUND_BUCKETS = ("R64", "R32", "S16", "E8", "F4_NCG")
_CLIP = 1e-9


def _round_int_to_bucket(rnd: int) -> str:
    """Map round int from build_pair_round_lookup (1..6) to bucket key.

    Rounds 5 (F4) and 6 (Champ) are collapsed into 'F4_NCG' for n>=66
    per knob (44 + 22). NCG alone at n=22 is too noisy for a 7-cell
    grid search -- see spec section 'Architecture / Per-round scaling'.
    Round 0 (play-in) is not a tournament round and never reaches this
    function via the public path; raise rather than guess.
    """
    if rnd == 1:
        return "R64"
    if rnd == 2:
        return "R32"
    if rnd == 3:
        return "S16"
    if rnd == 4:
        return "E8"
    if rnd in (5, 6):
        return "F4_NCG"
    raise ValueError(f"unexpected round int: {rnd!r}")


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, _CLIP, 1.0 - _CLIP)
    return np.log(p / (1.0 - p))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def scale_pairwise(
    df: pd.DataFrame,
    T: Union[float, Mapping[str, float]],
) -> pd.DataFrame:
    """Return a NEW DataFrame with p_a_wins rescaled by T.

    Scalar T:
        p_out = sigmoid(logit(p_in) / T) for every row.

    Per-round T (dict):
        Per row, dispatch on row['round_bucket']. KeyError if any row's
        bucket is not in T.
    """
    out = df.copy()
    p = out["p_a_wins"].to_numpy(dtype=float)
    if isinstance(T, (int, float)):
        out["p_a_wins"] = _sigmoid(_logit(p) / float(T))
        return out
    # Per-round dict.
    if "round_bucket" not in out.columns:
        raise KeyError("scale_pairwise per-round mode requires 'round_bucket' column")
    z = _logit(p)
    new_p = np.empty_like(p)
    bucket_arr = out["round_bucket"].to_numpy()
    # Normalize T into a plain dict for repeated lookups.
    T_map = dict(T)
    for i, b in enumerate(bucket_arr):
        # Trigger a KeyError early with the missing bucket name in the
        # message -- pytest matches on bucket names.
        if b not in T_map:
            raise KeyError(b)
        new_p[i] = _sigmoid(z[i] / float(T_map[b]))
    out["p_a_wins"] = new_p
    return out


def assign_round_buckets(
    df: pd.DataFrame,
    slots_df: pd.DataFrame,
    seeds_df: pd.DataFrame,
) -> pd.Series:
    """For each row in df, return its round bucket.

    Uses build_pair_round_lookup per (season). Returns pd.NA for rows
    whose (team_a, team_b) pair has no resolvable round in their season
    (play-in pairs, or pairs with seeds that don't map).
    """
    out = pd.Series(pd.NA, index=df.index, dtype=object)
    for season in df["season"].unique():
        lookup = build_pair_round_lookup(int(season), slots_df, seeds_df)
        mask = df["season"] == season
        for idx in df[mask].index:
            a = int(df.at[idx, "team_a"])
            b = int(df.at[idx, "team_b"])
            key = (min(a, b), max(a, b))
            rnd = lookup.get(key)
            if rnd is None:
                continue
            try:
                out.at[idx] = _round_int_to_bucket(int(rnd))
            except ValueError:
                # Round 0 (play-in) -- leave as pd.NA.
                continue
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Apply temperature scaling.")
    p.add_argument("--in", dest="inp", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--T", type=float, required=True,
                   help="scalar temperature (per-round T not supported via CLI)")
    args = p.parse_args(argv)
    df = pd.read_csv(args.inp)
    out = scale_pairwise(df, T=args.T)
    out.to_csv(args.out, index=False)
    print(f"wrote {args.out} ({len(out)} rows) at T={args.T}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
