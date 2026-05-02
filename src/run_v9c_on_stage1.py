"""Run v9-C stage-2 on a stage-1 pairwise CSV.

Spec: docs/superpowers/specs/2026-05-01-ensemble-stage1-design.md

Wraps src.train_upset_model.build_v9_pairwise so the experiment can hold
the v9-C config (W_UPSET=1.25, W_MISS=0.0, feature_set='v9c') constant
while only the stage-1 input varies between
    --pairwise-in output/pairwise_v4.csv          (baseline)
    --pairwise-in output/pairwise_ensemble.csv    (experiment)
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.train_upset_model import (
    build_v9_pairwise,
    load_per_game_data_with_upset,
)

DATA = Path("data/raw/march-machine-learning-2026")
W_UPSET = 1.25
W_MISS = 0.0
FEATURE_SET = "v9c"


def run_v9c(pairwise_in: str, pairwise_out: str) -> dict:
    """Run v9-C stage-2 on a stage-1 pairwise CSV.

    Mirrors src.sweep_v9_weights.run_single_cell, minus the per-cell
    metrics dict. Writes pairwise_out (same schema as pairwise_in:
    season, team_a, team_b, p_a_wins) and returns a small summary.
    """
    seeds_csv = str(DATA / "MNCAATourneySeeds.csv")
    results_csv = str(DATA / "MNCAATourneyCompactResults.csv")
    slots_csv = str(DATA / "MNCAATourneySlots.csv")

    print(f"loading per-game data from {pairwise_in} ...")
    per_game = load_per_game_data_with_upset(
        pairwise_in, results_csv, seeds_csv,
    )
    print(f"  rows: {len(per_game):,}; seasons: {per_game.season.nunique()}")

    print(f"applying v9-C (W_UPSET={W_UPSET}, W_MISS={W_MISS}, "
          f"feature_set='{FEATURE_SET}') -> {pairwise_out}")
    Path(pairwise_out).parent.mkdir(parents=True, exist_ok=True)
    build_v9_pairwise(
        per_game, pairwise_in, seeds_csv, pairwise_out,
        slots_csv=slots_csv,
        w_upset=W_UPSET, w_miss=W_MISS,
        feature_set=FEATURE_SET,
    )
    out_df = pd.read_csv(pairwise_out)
    print(f"  wrote {len(out_df):,} rows")
    return {
        "n_rows": len(out_df),
        "n_seasons": out_df["season"].nunique(),
        "out": pairwise_out,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--pairwise-in", required=True)
    parser.add_argument("--pairwise-out", required=True)
    args = parser.parse_args(argv)
    run_v9c(args.pairwise_in, args.pairwise_out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
