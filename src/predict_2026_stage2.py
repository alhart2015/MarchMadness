"""Apply stage-2 corrector to v4's 2026 pairwise predictions.

Trains stage 2 on ALL 22 LOSO seasons of v4 out-of-fold data, then
adjusts each 2026 pair-pair prediction. Writes a new
output/pairwise_probs_v8_2026.json so postmortem_full.py can score
the v8 bracket against actuals.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Make `src.<...>` importable when running this file directly from the repo root.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.train_stage2 import (
    DATA, fit_stage2, load_per_game_data, parse_seed, stage2_features,
)


def main():
    print("Loading stage-2 training data (all 22 LOSO seasons)...")
    per_game = load_per_game_data(
        "output/pairwise_v4.csv",
        str(DATA / "MNCAATourneyCompactResults.csv"),
        str(DATA / "MNCAATourneySeeds.csv"),
    )
    print(f"  {len(per_game):,} rows")

    print("Training stage 2 on all 22 seasons...")
    model = fit_stage2(stage2_features(per_game), per_game["label"].values)

    print("Loading v4 2026 pairwise predictions...")
    with open("output/pairwise_probs_v4.json") as f:
        v4_probs = json.load(f)
    print(f"  {len(v4_probs):,} pair-pairs")

    print("Loading 2026 seeds...")
    seeds_df = pd.read_csv(DATA / "MNCAATourneySeeds.csv")
    seeds_df["seed_int"] = seeds_df["Seed"].apply(parse_seed)
    seeds_2026 = {int(r.TeamID): r.seed_int for _, r in seeds_df.iterrows()
                  if r.Season == 2026 and r.seed_int is not None}
    print(f"  {len(seeds_2026)} seeds")

    print("Applying stage 2 to each 2026 pair...")
    adjusted = {}
    skipped = 0
    feat_rows = []
    keys_with_seeds = []
    for key, p_stage1 in v4_probs.items():
        # Keys like "1181_1373"; smaller ID first.
        a_str, b_str = key.split("_")
        a, b = int(a_str), int(b_str)
        seed_a = seeds_2026.get(a)
        seed_b = seeds_2026.get(b)
        if seed_a is None or seed_b is None:
            adjusted[key] = float(p_stage1)  # passthrough
            skipped += 1
            continue
        feat_rows.append([float(p_stage1), seed_a, seed_b, abs(seed_a - seed_b)])
        keys_with_seeds.append(key)

    if feat_rows:
        X = np.array(feat_rows)
        p_stage12 = model.predict_proba(X)[:, 1]
        for key, p_new in zip(keys_with_seeds, p_stage12):
            adjusted[key] = round(float(p_new), 4)

    print(f"  Adjusted: {len(keys_with_seeds):,}; passthrough (no seeds): {skipped}")

    out_path = "output/pairwise_probs_v8_2026.json"
    with open(out_path, "w") as f:
        json.dump(adjusted, f)
    print(f"Saved: {out_path}")

    # Also update the canonical pairwise_probs.json so postmortem_full picks it up.
    with open("output/pairwise_probs.json", "w") as f:
        json.dump(adjusted, f)
    print("Overwrote: output/pairwise_probs.json")


if __name__ == "__main__":
    main()
