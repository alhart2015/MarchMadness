"""Upset-aware stage-2 model (v9): replacement for v8 with upset-weighted training.

Like v8 (src/train_stage2.py), v9 is a small XGBoost trained on v4's
out-of-fold pairwise predictions plus seed-pair context, predicting the
actual game outcome under double-LOSO. The differentiator is the loss:
training rows are weighted to emphasize upsets (higher seed lost) and
high-confidence-miss rows (where v4 was wrong with high confidence).

Inputs (4 features, identical to v8):
    p_v4_stage1, seed_a, seed_b, abs_seed_diff
Target:
    label = 1 if A beat B else 0 (symmetric: each game contributes 2 rows).

Sample weights:
    w = 1.0
    if higher_seed_team_in_this_game_lost: w *= W_UPSET   (default 3.0)
    w *= 1 + W_MISS * residual ** 2                       (default W_MISS = 4.0)
    where residual = label - p_v4_for_this_perspective.

Same-seed games (rare; F4 / Champ): no upset flag; W_UPSET multiplier
skipped. W_MISS multiplier still applies.

Outputs:
    output/pairwise_v9.csv -- v9-adjusted pairwise probs across 22 LOSO
        seasons. Same schema as pairwise_v8.csv.
    output/v9_eval.csv     -- per-season comparison row (v4, v8, v9).

Spec: docs/superpowers/specs/2026-04-30-upset-detection-design.md
"""
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import log_loss as sklearn_log_loss

DATA = Path("data/raw/march-machine-learning-2026")
OUTPUT = Path("output")

SEASONS_TO_BACKTEST = list(range(2003, 2026))  # 2003..2025 (excluding 2020 implicitly via data)

# Sample weighting hyperparameters. Tunable via narrow sweep.
W_UPSET = 3.0
W_MISS = 4.0


def parse_seed(seed_str):
    """Pull the integer seed out of strings like 'W01', 'W11a', 'X16b'.

    Copied from src/train_stage2.py: same Kaggle seed-string format.
    """
    if not isinstance(seed_str, str):
        return None
    digits = "".join(c for c in seed_str if c.isdigit())
    return int(digits) if digits else None


def load_per_game_data_with_upset(
    pairwise_csv: str, results_csv: str, seeds_csv: str
) -> pd.DataFrame:
    """Build per-played-game training rows for v9.

    Each row: (season, team_a, team_b, p_stage1, seed_a, seed_b,
              abs_seed_diff, upset, label). Symmetric: each game produces
              two rows (a=W,b=L; a=L,b=W). The upset flag is per-game
              (independent of A/B perspective) -- True iff the higher-
              seeded team lost. Same-seed games are flagged upset=False.

    Adapted from src/train_stage2.py:load_per_game_data: identical except
    for the added upset column.
    """
    pw = pd.read_csv(pairwise_csv)
    pw["pair_key"] = list(zip(pw["season"], pw["team_a"], pw["team_b"]))
    # Last write wins (default + tuned LOSO each appended); take final row per pair.
    pw = pw.drop_duplicates("pair_key", keep="last")
    pw_lookup = {(s, a, b): float(p)
                 for s, a, b, p in zip(pw.season, pw.team_a, pw.team_b, pw.p_a_wins)}

    results = pd.read_csv(results_csv)
    seeds = pd.read_csv(seeds_csv)
    seeds["seed_int"] = seeds["Seed"].apply(parse_seed)
    seed_lookup = {(int(r["Season"]), int(r["TeamID"])): r["seed_int"]
                   for _, r in seeds.iterrows() if r["seed_int"] is not None}

    rows = []
    for _, g in results.iterrows():
        season = int(g["Season"])
        if season not in SEASONS_TO_BACKTEST:
            continue
        w, l = int(g["WTeamID"]), int(g["LTeamID"])
        # pairwise CSV stores pairs as (min_id, max_id, p_min_wins).
        a, b = (w, l) if w < l else (l, w)
        p_a_wins = pw_lookup.get((season, a, b))
        if p_a_wins is None:
            continue
        # Map back to (W, L) perspective.
        p_w = p_a_wins if a == w else (1.0 - p_a_wins)
        seed_w = seed_lookup.get((season, w))
        seed_l = seed_lookup.get((season, l))
        if seed_w is None or seed_l is None:
            continue

        # Upset flag (per-game; same value for both symmetric rows): True
        # iff the higher-seeded team lost. Same-seed games: False.
        # Lower seed_int = better seed (1 is the top seed).
        if seed_w == seed_l:
            is_upset = False
        else:
            is_upset = seed_w > seed_l  # winner had a worse seed than loser

        # Symmetric pair: A=W (label=1), then A=L (label=0).
        rows.append({
            "season": season, "team_a": w, "team_b": l,
            "p_stage1": p_w,
            "seed_a": seed_w, "seed_b": seed_l,
            "abs_seed_diff": abs(seed_w - seed_l),
            "upset": is_upset,
            "label": 1,
        })
        rows.append({
            "season": season, "team_a": l, "team_b": w,
            "p_stage1": 1.0 - p_w,
            "seed_a": seed_l, "seed_b": seed_w,
            "abs_seed_diff": abs(seed_w - seed_l),
            "upset": is_upset,
            "label": 0,
        })
    return pd.DataFrame(rows)
