"""Upset-aware stage-2 model (v9-B): feature-extended fallback over v9-A.

Like v8 (src/train_stage2.py), v9 is a small XGBoost trained on v4's
out-of-fold pairwise predictions plus seed-pair context, predicting the
actual game outcome under double-LOSO. The differentiator is the loss:
training rows are weighted to emphasize upsets (higher seed lost) and
high-confidence-miss rows (where v4 was wrong with high confidence).

Inputs (7 features for v9-B):
    p_v4_stage1, seed_a, seed_b, abs_seed_diff,
    round (1..6 for R64..Champ; 0 at apply time -- pairwise_v4.csv has
      no DayNum so build_v9_pairwise always passes 0.0 for this column),
    v4_confidence (|p_stage1 - 0.5|),
    is_a_higher_seed (1.0 if seed_a < seed_b else 0.0).
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
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import log_loss as sklearn_log_loss

# Path setup: allow `python src/train_upset_model.py` invocation by ensuring
# the project root is on sys.path before importing from `src.*`.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.models.matchup import day_to_round

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
              abs_seed_diff, upset, round, label). Symmetric: each game produces
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
            "round": day_to_round(int(g["DayNum"])),
            "label": 1,
        })
        rows.append({
            "season": season, "team_a": l, "team_b": w,
            "p_stage1": 1.0 - p_w,
            "seed_a": seed_l, "seed_b": seed_w,
            "abs_seed_diff": abs(seed_w - seed_l),
            "upset": is_upset,
            "round": day_to_round(int(g["DayNum"])),
            "label": 0,
        })
    return pd.DataFrame(rows)


def compute_sample_weights(
    df: pd.DataFrame, w_upset: float = W_UPSET, w_miss: float = W_MISS
) -> np.ndarray:
    """Per-row training weight for v9.

    For each row:
        w = 1.0
        if upset: w *= w_upset                        # upset multiplier
        w *= 1 + w_miss * (label - p_stage1) ** 2     # miss multiplier
                                                      # residual is per-perspective

    Returns: np.ndarray of length len(df), aligned with df rows.
    """
    base = np.ones(len(df), dtype=float)
    upset_factor = np.where(df["upset"].values, w_upset, 1.0)
    residual = df["label"].values.astype(float) - df["p_stage1"].values.astype(float)
    miss_factor = 1.0 + w_miss * (residual ** 2)
    return base * upset_factor * miss_factor


def upset_features(df: pd.DataFrame) -> np.ndarray:
    """Pull the v9-B input matrix from a per-game DataFrame.

    7 features:
      p_stage1, seed_a, seed_b, abs_seed_diff,
      round (1..6 for R64..Champ; 0 if unknown DayNum -- always 0 at
        apply time because pairwise_v4.csv has no DayNum),
      v4_confidence (|p_stage1 - 0.5|),
      is_a_higher_seed (1.0 if seed_a < seed_b else 0.0).
    """
    p = df["p_stage1"].values.astype(float)
    sa = df["seed_a"].values.astype(float)
    sb = df["seed_b"].values.astype(float)
    diff = df["abs_seed_diff"].values.astype(float)
    rnd = df["round"].values.astype(float)
    conf = np.abs(p - 0.5)
    higher = (sa < sb).astype(float)
    return np.column_stack([p, sa, sb, diff, rnd, conf, higher])


def fit_upset_model(
    X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray, seed: int = 42
) -> xgb.XGBClassifier:
    """Small XGBoost trained with upset-aware sample weights.

    Same shape as v8's fit_stage2 (src/train_stage2.py): n_estimators=100,
    max_depth=3, lr=0.05. Adds sample_weight at fit time. Capacity stays
    low (~3000 training rows in the full backtest).
    """
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
    model.fit(X, y, sample_weight=sample_weight)
    return model


def double_loso_eval(
    per_game: pd.DataFrame,
    w_upset: float = W_UPSET,
    w_miss: float = W_MISS,
) -> pd.DataFrame:
    """For each test season, train v9 on all-other-seasons and evaluate on it.

    Mirrors src/train_stage2.py:double_loso_eval. Differences:
    - Uses upset_features / fit_upset_model / compute_sample_weights.
    - Reports v8-style metrics (log loss, accuracy) but for v9.

    Weights are forwarded to compute_sample_weights; defaults preserve
    canonical 3.0 / 4.0 behavior.

    Returns DataFrame with per-season metrics:
        season, n_games, ll_v9, acc_v9.

    Per-row stage-1 (v4) predictions remain available in per_game; the
    caller is responsible for joining v4 / v8 numbers if it wants the
    full head-to-head table.
    """
    seasons = sorted(per_game["season"].unique())
    results = []

    for test_season in seasons:
        train = per_game[per_game.season != test_season]
        test = per_game[per_game.season == test_season]
        if len(train) == 0 or len(test) == 0:
            continue

        X_train = upset_features(train)
        y_train = train["label"].values
        w_train = compute_sample_weights(train, w_upset=w_upset, w_miss=w_miss)
        X_test = upset_features(test)
        y_test = test["label"].values

        model = fit_upset_model(X_train, y_train, w_train)
        p_v9 = model.predict_proba(X_test)[:, 1]

        # Keep only one row per game (winner perspective) for clean reporting,
        # matching v8's convention.
        is_winner = test["label"].values == 1
        if is_winner.sum() == 0:
            continue
        ll_v9 = sklearn_log_loss(y_test[is_winner], p_v9[is_winner], labels=[0, 1])
        acc_v9 = float((p_v9[is_winner] > 0.5).mean())

        results.append({
            "season": test_season,
            "n_games": int(is_winner.sum()),
            "ll_v9": ll_v9, "acc_v9": acc_v9,
        })

    return pd.DataFrame(results).sort_values("season").reset_index(drop=True)


def build_v9_pairwise(
    per_game: pd.DataFrame,
    pairwise_v4_csv: str,
    seeds_csv: str,
    out_path: str,
) -> None:
    """For each LOSO season, train v9 on other-seasons' per-game rows and
    apply to every pair in that season's pairwise_v4.csv. Writes a CSV in
    v8-compatible schema (season, team_a, team_b, p_a_wins) with team_a <
    team_b on every row.

    Mirrors src/train_stage2.py:build_v8_pairwise. Differences: feeds
    sample weights to fit_upset_model, no other functional change.
    """
    pw = pd.read_csv(pairwise_v4_csv).drop_duplicates(
        ["season", "team_a", "team_b"], keep="last"
    )
    seeds = pd.read_csv(seeds_csv)
    seeds["seed_int"] = seeds["Seed"].apply(parse_seed)
    seed_lookup = {(int(r["Season"]), int(r["TeamID"])): r["seed_int"]
                   for _, r in seeds.iterrows() if r["seed_int"] is not None}

    out_rows = []
    for season in sorted(pw.season.unique()):
        season_pw = pw[pw.season == season].copy()
        season_pw["seed_a"] = [seed_lookup.get((int(s), int(a)))
                               for s, a in zip(season_pw["season"], season_pw["team_a"])]
        season_pw["seed_b"] = [seed_lookup.get((int(s), int(b)))
                               for s, b in zip(season_pw["season"], season_pw["team_b"])]
        valid_mask = season_pw["seed_a"].notna() & season_pw["seed_b"].notna()

        train = per_game[per_game.season != season]
        if len(train) > 0 and valid_mask.any():
            X_train = upset_features(train)
            y_train = train["label"].values
            w_train = compute_sample_weights(train)
            model = fit_upset_model(X_train, y_train, w_train)

            apply_df = season_pw[valid_mask].copy()
            apply_df["abs_seed_diff"] = (apply_df["seed_a"] - apply_df["seed_b"]).abs()
            apply_df["round"] = 0.0  # unknown at apply time -- pairwise CSV has no DayNum
            apply_df["p_stage1"] = apply_df["p_a_wins"]
            p_v9 = model.predict_proba(upset_features(apply_df))[:, 1]
            v9_by_index = pd.Series(p_v9, index=apply_df.index)
        else:
            v9_by_index = pd.Series(dtype=float)

        for idx, r in season_pw.iterrows():
            p = float(v9_by_index[idx]) if idx in v9_by_index.index else float(r["p_a_wins"])
            out_rows.append({
                "season": int(season), "team_a": int(r["team_a"]),
                "team_b": int(r["team_b"]), "p_a_wins": p,
            })

    pd.DataFrame(out_rows).to_csv(out_path, index=False)


def main():
    print("=" * 80)
    print("V9 TRAINING (upset-aware stage-2 on v4 OOF predictions)")
    print(f"  W_UPSET={W_UPSET}, W_MISS={W_MISS}")
    print("=" * 80)

    pairwise_v4 = str(OUTPUT / "pairwise_v4.csv")
    pairwise_v8 = str(OUTPUT / "pairwise_v8.csv")
    seeds_csv = str(DATA / "MNCAATourneySeeds.csv")
    results_csv = str(DATA / "MNCAATourneyCompactResults.csv")

    per_game = load_per_game_data_with_upset(pairwise_v4, results_csv, seeds_csv)
    print(f"  Per-game training rows: {len(per_game):,} "
          f"(across {per_game.season.nunique()} seasons; "
          f"{int(per_game['upset'].sum() / 2)} upset games)")

    # Per-season log loss / accuracy -- v9 alone, v4 and v8 joined for context.
    eval_v9 = double_loso_eval(per_game)

    # v4 / v8 per-season stats from the same per_game frame for v4 (p_stage1)
    # plus pairwise_v8.csv joined back for v8.
    pw_v4 = per_game[per_game.label == 1]
    v4_stats = (
        pw_v4.groupby("season")
        .apply(lambda g: pd.Series({
            "n": len(g),
            "ll_v4": sklearn_log_loss(g["label"].values, g["p_stage1"].values,
                                      labels=[0, 1]),
            "acc_v4": float((g["p_stage1"].values > 0.5).mean()),
        }), include_groups=False)
        .reset_index()
    )

    if Path(pairwise_v8).exists():
        v8_pw = pd.read_csv(pairwise_v8).drop_duplicates(
            ["season", "team_a", "team_b"], keep="last"
        )
        v8_lookup = {(int(s), int(a), int(b)): float(p)
                     for s, a, b, p in zip(v8_pw.season, v8_pw.team_a,
                                           v8_pw.team_b, v8_pw.p_a_wins)}
        v8_per_season = []
        for season, g in pw_v4.groupby("season"):
            ps = []
            for _, row in g.iterrows():
                a, b = int(row.team_a), int(row.team_b)
                if a < b:
                    p = v8_lookup.get((int(season), a, b))
                else:
                    raw = v8_lookup.get((int(season), b, a))
                    p = (1.0 - raw) if raw is not None else None
                if p is None:
                    p = float(row.p_stage1)  # pass-through
                ps.append(p)
            ps_arr = np.array(ps)
            v8_per_season.append({
                "season": int(season),
                "ll_v8": sklearn_log_loss(g["label"].values, ps_arr, labels=[0, 1]),
                "acc_v8": float((ps_arr > 0.5).mean()),
            })
        v8_stats = pd.DataFrame(v8_per_season)
    else:
        print("  (output/pairwise_v8.csv not found; skipping v8 column.)")
        v8_stats = pd.DataFrame(columns=["season", "ll_v8", "acc_v8"])

    merged = (
        v4_stats.merge(v8_stats, on="season", how="left")
        .merge(eval_v9, on="season", how="left")
    )

    print(f"\n{'Season':>6}  {'N':>3}  {'LL_v4':>6}  {'LL_v8':>6}  {'LL_v9':>6}  "
          f"{'Acc_v4':>6}  {'Acc_v8':>6}  {'Acc_v9':>6}")
    print("-" * 72)
    for _, r in merged.iterrows():
        ll_v8 = f"{r.ll_v8:>6.3f}" if not pd.isna(r.get('ll_v8', np.nan)) else "    --"
        acc_v8 = f"{r.acc_v8 * 100:>5.1f}%" if not pd.isna(r.get('acc_v8', np.nan)) else "    --"
        print(f"  {int(r.season):>4}  {int(r.n):>3}  "
              f"{r.ll_v4:>6.3f}  {ll_v8}  {r.ll_v9:>6.3f}  "
              f"{r.acc_v4 * 100:>5.1f}%  {acc_v8}  {r.acc_v9 * 100:>5.1f}%")
    print("-" * 72)
    n_total = merged["n"].sum()
    mean_ll_v4 = (merged["ll_v4"] * merged["n"]).sum() / n_total
    mean_ll_v9 = (merged["ll_v9"] * merged["n"]).sum() / n_total
    mean_acc_v4 = (merged["acc_v4"] * merged["n"]).sum() / n_total
    mean_acc_v9 = (merged["acc_v9"] * merged["n"]).sum() / n_total
    if "ll_v8" in merged.columns and merged["ll_v8"].notna().any():
        v8_mask = merged["ll_v8"].notna()
        n_v8 = merged.loc[v8_mask, "n"].sum()
        mean_ll_v8 = (merged.loc[v8_mask, "ll_v8"] * merged.loc[v8_mask, "n"]).sum() / n_v8
        mean_acc_v8 = (merged.loc[v8_mask, "acc_v8"] * merged.loc[v8_mask, "n"]).sum() / n_v8
        ll_v8_str = f"{mean_ll_v8:>6.3f}"
        acc_v8_str = f"{mean_acc_v8 * 100:>5.1f}%"
    else:
        ll_v8_str = "    --"
        acc_v8_str = "    --"
    print(f"  {'WT MEAN':>6}        "
          f"{mean_ll_v4:>6.3f}  {ll_v8_str}  {mean_ll_v9:>6.3f}  "
          f"{mean_acc_v4 * 100:>5.1f}%  {acc_v8_str}  {mean_acc_v9 * 100:>5.1f}%")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    eval_path = OUTPUT / "v9_eval.csv"
    merged.to_csv(eval_path, index=False)
    print(f"\nWrote per-season eval to {eval_path}")

    pairwise_v9 = OUTPUT / "pairwise_v9.csv"
    print(f"Writing v9-adjusted pairwise to {pairwise_v9} ...")
    build_v9_pairwise(per_game, pairwise_v4, seeds_csv, str(pairwise_v9))
    print("  Done.")


if __name__ == "__main__":
    main()
