"""Two-stage model: train a stage-2 corrector on top of v4 (stage-1).

Stage 1 = v4 (v3 + coach features). Produces P(A beats B) for each pair.
Stage 2 = small XGBoost trained on (p_stage1, seed_a, seed_b, |seed_diff|)
          -> actual outcome. Learns to adjust stage-1 predictions where v4
          has structured errors (e.g., over-confidence in upset-prone seed
          pairings).

Leakage discipline (double-LOSO):
  For each test season Y:
    - Stage 1's prediction for Y is from pairwise_v4.csv (already
      out-of-fold: v4's LOSO trained on all-but-Y).
    - Stage 2 is trained on all-other-seasons' (p_stage1, context, label)
      tuples and applied to Y. So stage 2 never sees Y's data either.

Outputs:
  - Per-season comparison: stage-1-only vs stage-1+stage-2 log loss
    and accuracy.
  - output/pairwise_v8.csv: stage-2-adjusted pairwise probs across all
    22 LOSO seasons (for the 22-year bracket-points backtest).
"""
import math
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import log_loss as sklearn_log_loss

DATA = Path("data/raw/march-machine-learning-2026")
OUTPUT = Path("output")

SEASONS_TO_BACKTEST = list(range(2003, 2026))   # 2003..2025 (excluding 2020)


def parse_seed(seed_str):
    """Pull the integer seed out of strings like 'W01', 'W11a', 'X16b'."""
    if not isinstance(seed_str, str):
        return None
    digits = "".join(c for c in seed_str if c.isdigit())
    return int(digits) if digits else None


def load_per_game_data(pairwise_csv: str, results_csv: str, seeds_csv: str) -> pd.DataFrame:
    """Build per-played-game training rows for stage 2.

    Each row: (season, team_a, team_b, p_stage1, seed_a, seed_b, abs_seed_diff,
              label). Symmetric: each game produces two rows (a=W,b=L; a=L,b=W).
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

        # Symmetric pair: A=W (label=1), then A=L (label=0).
        rows.append({
            "season": season, "team_a": w, "team_b": l,
            "p_stage1": p_w,
            "seed_a": seed_w, "seed_b": seed_l,
            "abs_seed_diff": abs(seed_w - seed_l),
            "label": 1,
        })
        rows.append({
            "season": season, "team_a": l, "team_b": w,
            "p_stage1": 1.0 - p_w,
            "seed_a": seed_l, "seed_b": seed_w,
            "abs_seed_diff": abs(seed_w - seed_l),
            "label": 0,
        })
    return pd.DataFrame(rows)


def stage2_features(df: pd.DataFrame) -> np.ndarray:
    """Pull the stage-2 input matrix from a per-game DataFrame."""
    return df[["p_stage1", "seed_a", "seed_b", "abs_seed_diff"]].values


def fit_stage2(X: np.ndarray, y: np.ndarray, seed: int = 42) -> xgb.XGBClassifier:
    """Small XGBoost. Limited capacity to avoid overfitting on ~3000 rows."""
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=1.0,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=seed,
        eval_metric="logloss",
    )
    model.fit(X, y)
    return model


def double_loso_eval(per_game: pd.DataFrame):
    """For each test season, train stage 2 on all-other-seasons and evaluate on it.

    Returns DataFrame with per-season stage-1 vs stage-1+2 metrics.
    """
    seasons = sorted(per_game["season"].unique())
    results = []

    for test_season in seasons:
        train = per_game[per_game.season != test_season]
        test = per_game[per_game.season == test_season]
        if len(train) == 0 or len(test) == 0:
            continue

        X_train = stage2_features(train)
        y_train = train["label"].values
        X_test = stage2_features(test)
        y_test = test["label"].values

        model = fit_stage2(X_train, y_train)
        p_stage12 = model.predict_proba(X_test)[:, 1]
        p_stage1 = test["p_stage1"].values

        # Keep only one row per game (symmetric pair has two rows). Use
        # rows where label=1 (winner perspective) -- log loss is symmetric
        # but this is cleaner reporting.
        is_winner = test["label"].values == 1
        ll_s1 = sklearn_log_loss(y_test[is_winner], p_stage1[is_winner],
                                  labels=[0, 1])
        ll_s12 = sklearn_log_loss(y_test[is_winner], p_stage12[is_winner],
                                   labels=[0, 1])
        acc_s1 = float((p_stage1[is_winner] > 0.5).mean())
        acc_s12 = float((p_stage12[is_winner] > 0.5).mean())

        results.append({
            "season": test_season,
            "n_games": int(is_winner.sum()),
            "ll_stage1": ll_s1, "ll_stage12": ll_s12,
            "acc_stage1": acc_s1, "acc_stage12": acc_s12,
            "ll_delta": ll_s12 - ll_s1,
            "acc_delta": acc_s12 - acc_s1,
        })

    return pd.DataFrame(results).sort_values("season")


def build_v8_pairwise(per_game: pd.DataFrame, pairwise_v4_csv: str, seeds_csv: str,
                      out_path: str):
    """For each LOSO season, train stage 2 on other-seasons and apply to ALL pair-pairs
    in that season's field. Save the adjusted pairwise CSV for backtest scoring."""
    pw = pd.read_csv(pairwise_v4_csv).drop_duplicates(
        ["season", "team_a", "team_b"], keep="last"
    )
    seeds = pd.read_csv(seeds_csv)
    seeds["seed_int"] = seeds["Seed"].apply(parse_seed)
    seed_lookup = {(int(r["Season"]), int(r["TeamID"])): r["seed_int"]
                   for _, r in seeds.iterrows() if r["seed_int"] is not None}

    out_rows = []
    for season in sorted(pw.season.unique()):
        train = per_game[per_game.season != season]
        if len(train) == 0:
            continue
        model = fit_stage2(stage2_features(train), train["label"].values)

        season_pw = pw[pw.season == season].copy()
        # Build feature rows for stage 2.
        feat_rows = []
        keep = []
        for _, r in season_pw.iterrows():
            seed_a = seed_lookup.get((int(r["season"]), int(r["team_a"])))
            seed_b = seed_lookup.get((int(r["season"]), int(r["team_b"])))
            if seed_a is None or seed_b is None:
                keep.append(False)
                continue
            feat_rows.append([float(r["p_a_wins"]), seed_a, seed_b,
                              abs(seed_a - seed_b)])
            keep.append(True)

        if not feat_rows:
            # Fall back: keep stage-1 probs unchanged for this season.
            for _, r in season_pw.iterrows():
                out_rows.append({
                    "season": season, "team_a": int(r.team_a),
                    "team_b": int(r.team_b), "p_a_wins": float(r.p_a_wins),
                })
            continue

        X = np.array(feat_rows)
        p_stage12 = model.predict_proba(X)[:, 1]

        i = 0
        for (idx, r), keep_row in zip(season_pw.iterrows(), keep):
            if keep_row:
                p = float(p_stage12[i])
                i += 1
            else:
                p = float(r["p_a_wins"])
            out_rows.append({"season": season, "team_a": int(r.team_a),
                              "team_b": int(r.team_b), "p_a_wins": p})

    pd.DataFrame(out_rows).to_csv(out_path, index=False)


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairwise-in", default="output/pairwise_v4.csv",
                        help="Stage-1 pairwise CSV (input). Default: output/pairwise_v4.csv")
    parser.add_argument("--pairwise-out", default="output/pairwise_v8.csv",
                        help="Stage-2-adjusted pairwise CSV (output). Default: output/pairwise_v8.csv")
    args = parser.parse_args(argv)

    print("=" * 80)
    print("STAGE 2 TRAINING (double-LOSO on stage-1 predictions)")
    print(f"  Input:  {args.pairwise_in}")
    print(f"  Output: {args.pairwise_out}")
    print("=" * 80)

    per_game = load_per_game_data(
        args.pairwise_in,
        str(DATA / "MNCAATourneyCompactResults.csv"),
        str(DATA / "MNCAATourneySeeds.csv"),
    )
    print(f"  Per-game training rows: {len(per_game):,} (across "
          f"{per_game.season.nunique()} seasons)")

    eval_df = double_loso_eval(per_game)
    print(f"\n{'Season':>6}  {'N':>3}  {'LL_s1':>7}  {'LL_s12':>7}  "
          f"{'dLL':>7}  {'Acc_s1':>7}  {'Acc_s12':>7}  {'dAcc':>6}")
    print("-" * 75)
    for _, r in eval_df.iterrows():
        tag = ""
        if r["ll_delta"] < -0.005:
            tag = "  s12 better"
        elif r["ll_delta"] > 0.005:
            tag = "  s1 better"
        print(f"  {int(r.season):>4}  {int(r.n_games):>3}  "
              f"{r.ll_stage1:>7.3f}  {r.ll_stage12:>7.3f}  "
              f"{r.ll_delta:>+7.3f}  "
              f"{r.acc_stage1*100:>5.1f}%  {r.acc_stage12*100:>5.1f}%  "
              f"{r.acc_delta*100:>+5.1f}pp{tag}")
    print("-" * 75)
    n_total = eval_df["n_games"].sum()
    mean_ll_s1 = (eval_df["ll_stage1"] * eval_df["n_games"]).sum() / n_total
    mean_ll_s12 = (eval_df["ll_stage12"] * eval_df["n_games"]).sum() / n_total
    mean_acc_s1 = (eval_df["acc_stage1"] * eval_df["n_games"]).sum() / n_total
    mean_acc_s12 = (eval_df["acc_stage12"] * eval_df["n_games"]).sum() / n_total
    print(f"  {'WT MEAN':>4}        "
          f"{mean_ll_s1:>7.3f}  {mean_ll_s12:>7.3f}  "
          f"{mean_ll_s12 - mean_ll_s1:>+7.3f}  "
          f"{mean_acc_s1*100:>5.1f}%  {mean_acc_s12*100:>5.1f}%  "
          f"{(mean_acc_s12 - mean_acc_s1)*100:>+5.1f}pp")

    print(f"\nWriting stage-2-adjusted pairwise to {args.pairwise_out} ...")
    build_v8_pairwise(
        per_game,
        args.pairwise_in,
        str(DATA / "MNCAATourneySeeds.csv"),
        args.pairwise_out,
    )
    print("  Done.")


if __name__ == "__main__":
    main()
