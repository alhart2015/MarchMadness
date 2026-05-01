"""Apply v9-C stage-2 corrector to v4's 2026 pairwise predictions.

Trains v9-C on ALL 22 LOSO seasons of v4 out-of-fold data with the
PR 9 winning weights (W_UPSET=1.25, W_MISS=0.0, feature_set='v9c'),
then applies it to v4's 2026 raw-pair JSON predictions. Writes
output/pairwise_probs_v9c_2026.json (versioned snapshot) and
overwrites output/pairwise_probs.json (canonical for analysis
scripts).

Production-swap path established in PR 9. See:
- docs/notes/2026-05-01-v9c-feature-stripped.md (LOSO findings)
- docs/superpowers/specs/2026-05-01-v9c-production-swap-design.md
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.train_upset_model import (
    DATA,
    build_pair_round_lookup,
    compute_sample_weights,
    fit_upset_model,
    load_per_game_data_with_upset,
    parse_seed,
    upset_features,
)

# Production weights from PR 9 winning cell.
PROD_W_UPSET = 1.25
PROD_W_MISS = 0.0
PROD_FEATURE_SET = "v9c"


def main():
    print("Loading v9-C training data (all 22 LOSO seasons)...")
    per_game = load_per_game_data_with_upset(
        "output/pairwise_v4.csv",
        str(DATA / "MNCAATourneyCompactResults.csv"),
        str(DATA / "MNCAATourneySeeds.csv"),
    )
    print(f"  {len(per_game):,} rows ({per_game.season.nunique()} seasons)")

    print(f"Training v9-C (W_UPSET={PROD_W_UPSET}, W_MISS={PROD_W_MISS}, "
          f"feature_set={PROD_FEATURE_SET}) on all seasons...")
    X = upset_features(per_game, feature_set=PROD_FEATURE_SET)
    y = per_game["label"].values
    w = compute_sample_weights(per_game, w_upset=PROD_W_UPSET,
                               w_miss=PROD_W_MISS)
    model = fit_upset_model(X, y, w)

    print("Loading v4 2026 pairwise predictions...")
    with open("output/pairwise_probs_v4.json") as f:
        v4_probs = json.load(f)
    print(f"  {len(v4_probs):,} pair-pairs")

    print("Loading 2026 seeds + slots...")
    seeds_df = pd.read_csv(DATA / "MNCAATourneySeeds.csv")
    seeds_df["seed_int"] = seeds_df["Seed"].apply(parse_seed)
    seeds_2026 = {int(r.TeamID): r.seed_int for _, r in seeds_df.iterrows()
                  if r.Season == 2026 and r.seed_int is not None}
    slots_df = pd.read_csv(DATA / "MNCAATourneySlots.csv")
    pair_round_2026 = build_pair_round_lookup(2026, slots_df, seeds_df)
    print(f"  {len(seeds_2026)} seeds; "
          f"{len(pair_round_2026)} pair-round entries")

    print("Applying v9-C to each 2026 pair...")
    adjusted = {}
    skipped = 0
    feat_rows = []
    keys_with_seeds = []
    for key, p_stage1 in v4_probs.items():
        a_str, b_str = key.split("_")
        a, b = int(a_str), int(b_str)
        seed_a = seeds_2026.get(a)
        seed_b = seeds_2026.get(b)
        if seed_a is None or seed_b is None:
            adjusted[key] = float(p_stage1)  # passthrough
            skipped += 1
            continue
        a_canon, b_canon = (a, b) if a < b else (b, a)
        rnd = float(pair_round_2026.get((a_canon, b_canon), 0))
        feat_rows.append({
            "p_stage1": float(p_stage1),
            "seed_a": float(seed_a),
            "seed_b": float(seed_b),
            "abs_seed_diff": float(abs(seed_a - seed_b)),
            "round": rnd,
        })
        keys_with_seeds.append(key)

    if feat_rows:
        X_apply = upset_features(pd.DataFrame(feat_rows),
                                 feature_set=PROD_FEATURE_SET)
        p_v9c = model.predict_proba(X_apply)[:, 1]
        for key, p_new in zip(keys_with_seeds, p_v9c):
            adjusted[key] = round(float(p_new), 4)

    print(f"  Adjusted: {len(keys_with_seeds):,}; "
          f"passthrough (no seeds): {skipped}")

    out_path = "output/pairwise_probs_v9c_2026.json"
    with open(out_path, "w") as f:
        json.dump(adjusted, f)
    print(f"Saved: {out_path}")

    # Overwrite the canonical pairwise_probs.json so analysis scripts
    # pick up v9-C corrections.
    with open("output/pairwise_probs.json", "w") as f:
        json.dump(adjusted, f)
    print("Overwrote: output/pairwise_probs.json")


if __name__ == "__main__":
    main()
