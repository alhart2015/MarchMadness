"""Experimental stage-2 trainer: feature-set toggles + multi-seed ensembles.

A near-clone of train_stage2.py with two extensions: (1) FEATURE_SETS
selectable via --features, (2) multi-seed ensemble averaging via --seeds.
With --features v8 --seeds 42, output reproduces train_stage2.py byte-equal
(anchor invariance enforced by tests; do not remove that guarantee).

Double-LOSO discipline (unchanged from v8): for each test season Y, stage-2
trains on all-other-seasons per-game data and applies to Y's full pairwise
field. pairwise_v4.csv is already LOSO out-of-fold across 22 seasons.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import log_loss as sklearn_log_loss

from src.bracket.expected_round import expected_round_for_pair

DATA = Path("data/raw/march-machine-learning-2026")
OUTPUT = Path("output")

SEASONS_TO_BACKTEST = list(range(2003, 2026))

_V8_BASE = ["p_stage1", "seed_a", "seed_b", "abs_seed_diff"]

# v12 (v13 enriched with top-N v4 feature diffs) -- top-N is read from the
# Phase-0 ranking artifact at import time. Sets are only populated if the
# ranking exists, so train_stage2_v10 stays importable without v12 inputs.
_V12_RANKING_PATH = OUTPUT / "v4_feature_importance.csv"
_V12_FM_DEFAULT_PATH = OUTPUT / "v4_feature_matrix.parquet"
_V12_DIFF_PREFIX = "diff_"


def _v12_top_n_features(n: int) -> List[str]:
    ranking = pd.read_csv(_V12_RANKING_PATH)
    return ranking.head(n)["feature_name"].tolist()


def _v12_diff_cols(n: int) -> List[str]:
    return [f"{_V12_DIFF_PREFIX}{name}" for name in _v12_top_n_features(n)]


def _build_feature_sets() -> dict:
    sets = {
        "v8":      _V8_BASE,
        "v10a":    _V8_BASE + ["expected_round"],
        "v10b":    _V8_BASE + ["expected_round", "v4_logit"],
        "v10c":    _V8_BASE + ["expected_round", "min_seed", "max_seed"],
        "v10":     _V8_BASE + ["expected_round", "v4_logit", "min_seed", "max_seed"],
        "v10a_oh": _V8_BASE + ["er_r64", "er_r32", "er_s16", "er_e8", "er_f4", "er_champ"],
    }
    if _V12_RANKING_PATH.exists():
        for n in (5, 10, 15):
            sets[f"v12_n{n}"] = _V8_BASE + ["expected_round"] + _v12_diff_cols(n)
    return sets


FEATURE_SETS = _build_feature_sets()

HPARAM_SETS = {
    "v8":     dict(n_estimators=100, max_depth=3, learning_rate=0.05, subsample=0.9,
                   colsample_bytree=1.0, reg_alpha=0.1, reg_lambda=1.0),
    "v10cap": dict(n_estimators=200, max_depth=4, learning_rate=0.05, subsample=0.9,
                   colsample_bytree=1.0, reg_alpha=0.1, reg_lambda=1.0),
}


def parse_seed(seed_str):
    if not isinstance(seed_str, str):
        return None
    digits = "".join(c for c in seed_str if c.isdigit())
    return int(digits) if digits else None


def _logit(p, eps=1e-6):
    p_clipped = min(1.0 - eps, max(eps, float(p)))
    return math.log(p_clipped / (1.0 - p_clipped))


def _build_v4_feature_lookup(fm_df: pd.DataFrame) -> dict:
    """Index v4 feature matrix by (Season, TeamID) -> {feat_name: value}.
    Returns ({}, []) if fm_df is None."""
    if fm_df is None:
        return {}, []
    feat_cols = [c for c in fm_df.columns if c not in ("Season", "TeamID")]
    lookup = {
        (int(r["Season"]), int(r["TeamID"])): {c: float(r[c]) for c in feat_cols}
        for _, r in fm_df.iterrows()
    }
    return lookup, feat_cols


def load_per_game_data(
    pairwise_csv: str,
    results_csv: str,
    seeds_csv: str,
    slots_csv: str,
    v4_feature_matrix_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Per-played-game training rows, symmetric (W and L perspective each).

    If `v4_feature_matrix_df` is provided, also emits `diff_<feat>` columns
    for each v4 raw feature in the matrix. The label=1 row sees
    `feat_w - feat_l`; the label=0 row sees `feat_l - feat_w`."""
    pw = pd.read_csv(pairwise_csv)
    pw["pair_key"] = list(zip(pw["season"], pw["team_a"], pw["team_b"]))
    pw = pw.drop_duplicates("pair_key", keep="last")
    pw_lookup = {(s, a, b): float(p)
                 for s, a, b, p in zip(pw.season, pw.team_a, pw.team_b, pw.p_a_wins)}

    results = pd.read_csv(results_csv)
    seeds = pd.read_csv(seeds_csv)
    seeds["seed_int"] = seeds["Seed"].apply(parse_seed)
    seed_lookup = {(int(r["Season"]), int(r["TeamID"])): r["seed_int"]
                   for _, r in seeds.iterrows() if r["seed_int"] is not None}

    v4_lookup, v4_feat_cols = _build_v4_feature_lookup(v4_feature_matrix_df)

    rows = []
    for _, g in results.iterrows():
        season = int(g["Season"])
        if season not in SEASONS_TO_BACKTEST:
            continue
        w, l = int(g["WTeamID"]), int(g["LTeamID"])
        a, b = (w, l) if w < l else (l, w)
        p_a_wins = pw_lookup.get((season, a, b))
        if p_a_wins is None:
            continue
        p_w = p_a_wins if a == w else (1.0 - p_a_wins)
        seed_w = seed_lookup.get((season, w))
        seed_l = seed_lookup.get((season, l))
        if seed_w is None or seed_l is None:
            continue
        # Play-in self-pairs (e.g., W16a vs W16b) have no defined bracket
        # round -- sentinel 0 keeps the row in the training set (v8 anchor
        # invariance) and lets XGB split on it if useful.
        er = expected_round_for_pair(season, w, l, slots_csv, seeds_csv)
        if er is None:
            er = 0

        # v4 diffs (signed, winner perspective). If either team is missing
        # from the v4 FM, skip the diff fields entirely -- v12 feature sets
        # will fail the column-select if the columns are absent, so this
        # only kicks in for rows that v12 wouldn't have selected anyway.
        w_feats = v4_lookup.get((season, w)) if v4_lookup else None
        l_feats = v4_lookup.get((season, l)) if v4_lookup else None
        if v4_lookup and (w_feats is None or l_feats is None):
            continue
        w_diff = {f"{_V12_DIFF_PREFIX}{c}": w_feats[c] - l_feats[c]
                  for c in v4_feat_cols} if v4_lookup else {}

        for label, p_team, seed_self, seed_opp in (
            (1, p_w, seed_w, seed_l),
            (0, 1.0 - p_w, seed_l, seed_w),
        ):
            row = {
                "season": season,
                "team_a": (w if label == 1 else l),
                "team_b": (l if label == 1 else w),
                "p_stage1": p_team,
                "seed_a": seed_self,
                "seed_b": seed_opp,
                "abs_seed_diff": abs(seed_self - seed_opp),
                "expected_round": er,
                "v4_logit": _logit(p_team),
                "min_seed": min(seed_self, seed_opp),
                "max_seed": max(seed_self, seed_opp),
                "label": label,
            }
            if w_diff:
                # Label=0 (loser perspective) flips the sign on every diff.
                sign = 1 if label == 1 else -1
                row.update({k: sign * v for k, v in w_diff.items()})
            rows.append(row)
    return pd.DataFrame(rows)


def _augment_one_hot_round(df: pd.DataFrame) -> pd.DataFrame:
    """Add er_r64..er_champ binary columns derived from `expected_round`."""
    out = df.copy()
    for i, name in enumerate(["er_r64", "er_r32", "er_s16", "er_e8", "er_f4", "er_champ"], start=1):
        out[name] = (out["expected_round"] == i).astype(int)
    return out


def stage2_features(df: pd.DataFrame, feature_set: str) -> np.ndarray:
    cols = FEATURE_SETS[feature_set]
    if any(c.startswith("er_") for c in cols):
        df = _augment_one_hot_round(df)
    return df[cols].values


def fit_stage2(
    X: np.ndarray,
    y: np.ndarray,
    seed: int = 42,
    hparams: str = "v8",
) -> xgb.XGBClassifier:
    """Defaults to identical hyperparameters as the canonical train_stage2.py.
    Set hparams='v10cap' for the higher-capacity variant."""
    params = HPARAM_SETS[hparams]
    model = xgb.XGBClassifier(
        random_state=seed,
        eval_metric="logloss",
        **params,
    )
    model.fit(X, y)
    return model


def fit_stage2_ensemble(X: np.ndarray, y: np.ndarray, seeds, hparams: str = "v8"):
    """Train one XGB per seed; return list of fitted classifiers."""
    return [fit_stage2(X, y, seed=int(s), hparams=hparams) for s in seeds]


def predict_ensemble(models, X: np.ndarray) -> np.ndarray:
    """Average P(label=1) across the ensemble."""
    probs = np.zeros(X.shape[0], dtype=float)
    for m in models:
        probs += m.predict_proba(X)[:, 1]
    return probs / len(models)


def double_loso_eval(per_game: pd.DataFrame, feature_set: str, hparams: str = "v8"):
    seasons = sorted(per_game["season"].unique())
    results = []
    for test_season in seasons:
        train = per_game[per_game.season != test_season]
        test = per_game[per_game.season == test_season]
        if len(train) == 0 or len(test) == 0:
            continue

        X_train = stage2_features(train, feature_set)
        y_train = train["label"].values
        X_test = stage2_features(test, feature_set)
        y_test = test["label"].values

        model = fit_stage2(X_train, y_train, hparams=hparams)
        p_s12 = model.predict_proba(X_test)[:, 1]
        p_s1 = test["p_stage1"].values

        is_winner = test["label"].values == 1
        ll_s1 = sklearn_log_loss(y_test[is_winner], p_s1[is_winner], labels=[0, 1])
        ll_s12 = sklearn_log_loss(y_test[is_winner], p_s12[is_winner], labels=[0, 1])
        acc_s1 = float((p_s1[is_winner] > 0.5).mean())
        acc_s12 = float((p_s12[is_winner] > 0.5).mean())

        results.append({
            "season": test_season,
            "n_games": int(is_winner.sum()),
            "ll_stage1": ll_s1, "ll_stage12": ll_s12,
            "acc_stage1": acc_s1, "acc_stage12": acc_s12,
            "ll_delta": ll_s12 - ll_s1,
            "acc_delta": acc_s12 - acc_s1,
        })

    return pd.DataFrame(results).sort_values("season")


def build_pairwise(
    per_game: pd.DataFrame,
    pairwise_v4_csv: str,
    seeds_csv: str,
    slots_csv: str,
    out_path: str,
    feature_set: str,
    seeds=(42,),
    hparams: str = "v8",
    v4_feature_matrix_df: Optional[pd.DataFrame] = None,
):
    """For each LOSO season, train stage-2 on other-seasons and apply to ALL
    pairs in that season's field. Save the adjusted pairwise CSV.

    If multiple `seeds` are passed, trains one stage-2 per seed and averages
    their pairwise probabilities. Single-seed call reproduces v8 byte-equal
    when feature_set='v8'.

    For v12 feature sets, `v4_feature_matrix_df` must be provided -- each
    pair's diff_<feat> = (team_a feature) - (team_b feature)."""
    pw = pd.read_csv(pairwise_v4_csv).drop_duplicates(
        ["season", "team_a", "team_b"], keep="last"
    )
    seeds_df = pd.read_csv(seeds_csv)
    seeds_df["seed_int"] = seeds_df["Seed"].apply(parse_seed)
    seed_lookup = {(int(r["Season"]), int(r["TeamID"])): r["seed_int"]
                   for _, r in seeds_df.iterrows() if r["seed_int"] is not None}

    v4_lookup, v4_feat_cols = _build_v4_feature_lookup(v4_feature_matrix_df)

    out_rows = []
    for season in sorted(pw.season.unique()):
        print(f"  [build_pairwise] season={season} ...", flush=True)
        train = per_game[per_game.season != season]
        if len(train) == 0:
            continue
        X_train = stage2_features(train, feature_set)
        y_train = train["label"].values
        if len(seeds) == 1:
            model = fit_stage2(X_train, y_train, seed=int(seeds[0]), hparams=hparams)
            models = [model]
        else:
            models = fit_stage2_ensemble(X_train, y_train, seeds, hparams=hparams)

        season_pw = pw[pw.season == season].copy()
        feat_rows: List[Sequence[float]] = []
        keep: List[bool] = []
        for _, r in season_pw.iterrows():
            seed_a = seed_lookup.get((int(r["season"]), int(r["team_a"])))
            seed_b = seed_lookup.get((int(r["season"]), int(r["team_b"])))
            if seed_a is None or seed_b is None:
                keep.append(False)
                continue
            # v4 diffs for this pair. Skip the row if v12 is requested but
            # either team is missing from the FM (defensive; should not happen
            # for the canonical FM which covers all tournament-field teams).
            a_feats = v4_lookup.get((season, int(r["team_a"]))) if v4_lookup else None
            b_feats = v4_lookup.get((season, int(r["team_b"]))) if v4_lookup else None
            if v4_lookup and (a_feats is None or b_feats is None):
                keep.append(False)
                continue

            er = expected_round_for_pair(
                int(r["season"]), int(r["team_a"]), int(r["team_b"]),
                slots_csv, seeds_csv,
            )
            if er is None:
                er = 0
            p1 = float(r["p_a_wins"])
            row_dict = {
                "p_stage1": p1,
                "seed_a": seed_a,
                "seed_b": seed_b,
                "abs_seed_diff": abs(seed_a - seed_b),
                "expected_round": er,
                "v4_logit": _logit(p1),
                "min_seed": min(seed_a, seed_b),
                "max_seed": max(seed_a, seed_b),
                # one-hot encoding of expected_round (er in 1..6 maps to er_r64..er_champ)
                "er_r64":   int(er == 1),
                "er_r32":   int(er == 2),
                "er_s16":   int(er == 3),
                "er_e8":    int(er == 4),
                "er_f4":    int(er == 5),
                "er_champ": int(er == 6),
            }
            if a_feats is not None:
                for c in v4_feat_cols:
                    row_dict[f"{_V12_DIFF_PREFIX}{c}"] = a_feats[c] - b_feats[c]
            cols = FEATURE_SETS[feature_set]
            feat_rows.append([row_dict[c] for c in cols])
            keep.append(True)

        if not feat_rows:
            for _, r in season_pw.iterrows():
                out_rows.append({
                    "season": season, "team_a": int(r.team_a),
                    "team_b": int(r.team_b), "p_a_wins": float(r.p_a_wins),
                })
            continue

        X = np.array(feat_rows)
        p_s12 = predict_ensemble(models, X)

        i = 0
        for (_, r), keep_row in zip(season_pw.iterrows(), keep):
            if keep_row:
                p = float(p_s12[i])
                i += 1
            else:
                p = float(r["p_a_wins"])
            out_rows.append({"season": season, "team_a": int(r.team_a),
                              "team_b": int(r.team_b), "p_a_wins": p})

    pd.DataFrame(out_rows).to_csv(out_path, index=False)


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairwise-in", default="output/pairwise_v4.csv")
    parser.add_argument("--pairwise-out", default="output/pairwise_v10.csv")
    parser.add_argument("--features", choices=list(FEATURE_SETS), default="v10")
    parser.add_argument("--seeds", default="42",
                        help="Comma-separated XGB random_state values. If multiple, the "
                             "stage-2 pairwise probs are averaged across the ensemble.")
    parser.add_argument("--hparams", choices=list(HPARAM_SETS), default="v8",
                        help="XGB hyperparameter set. 'v8' matches canonical train_stage2; "
                             "'v10cap' bumps depth=4 and n_estimators=200.")
    parser.add_argument("--v4-feature-matrix",
                        default=str(_V12_FM_DEFAULT_PATH),
                        help="v4 per-(Season, TeamID) feature parquet for v12 diff joins. "
                             "Only loaded when --features starts with 'v12_'.")
    args = parser.parse_args(argv)
    seeds = tuple(int(s) for s in args.seeds.split(",") if s.strip())

    v4_fm_df = None
    if args.features.startswith("v12_"):
        v4_fm_df = pd.read_parquet(args.v4_feature_matrix)
        print(f"  Loaded v4 feature matrix: {len(v4_fm_df):,} rows, "
              f"{len([c for c in v4_fm_df.columns if c not in ('Season', 'TeamID')])} features")

    print("=" * 80)
    print(f"STAGE 2 v10 (feature_set={args.features})")
    print(f"  Input:  {args.pairwise_in}")
    print(f"  Output: {args.pairwise_out}")
    print(f"  Features ({len(FEATURE_SETS[args.features])}): {FEATURE_SETS[args.features]}")
    print("=" * 80)

    per_game = load_per_game_data(
        args.pairwise_in,
        str(DATA / "MNCAATourneyCompactResults.csv"),
        str(DATA / "MNCAATourneySeeds.csv"),
        str(DATA / "MNCAATourneySlots.csv"),
        v4_feature_matrix_df=v4_fm_df,
    )
    print(f"  Per-game training rows: {len(per_game):,} (across "
          f"{per_game.season.nunique()} seasons)")

    eval_df = double_loso_eval(per_game, args.features, hparams=args.hparams)
    print(f"\n{'Season':>6}  {'N':>3}  {'LL_s1':>7}  {'LL_s12':>7}  "
          f"{'dLL':>7}  {'Acc_s1':>7}  {'Acc_s12':>7}  {'dAcc':>6}")
    print("-" * 75)
    for _, r in eval_df.iterrows():
        tag = "  s12 better" if r["ll_delta"] < -0.005 else ("  s1 better" if r["ll_delta"] > 0.005 else "")
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

    print(f"\nWriting stage-2-adjusted pairwise to {args.pairwise_out} "
          f"(seeds={seeds}, hparams={args.hparams}) ...")
    build_pairwise(
        per_game,
        args.pairwise_in,
        str(DATA / "MNCAATourneySeeds.csv"),
        str(DATA / "MNCAATourneySlots.csv"),
        args.pairwise_out,
        args.features,
        seeds=seeds,
        hparams=args.hparams,
        v4_feature_matrix_df=v4_fm_df,
    )
    print("  Done.")


if __name__ == "__main__":
    main()
