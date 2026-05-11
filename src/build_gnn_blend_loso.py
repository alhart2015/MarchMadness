"""LOSO-realistic v4+GNN blend.

For each holdout season S, fit w_v4 on the 21 other seasons' tournament
games (minimizing log-loss over the union), then apply that w_v4 to S's
pairwise frame. Produces output/pairwise_v4_with_gnn_blend_loso.csv with
the same schema as pairwise_v4.csv.

This is the methodologically clean version of build_gnn_blend.py (which
uses a single cheating-ideal w fit on all test outcomes).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path("data/raw/march-machine-learning-2026")


def _load_lookup(path: str) -> dict:
    df = pd.read_csv(path).drop_duplicates(["season", "team_a", "team_b"], keep="last")
    return {(int(s), int(a), int(b)): float(p)
            for s, a, b, p in zip(df.season, df.team_a, df.team_b, df.p_a_wins)}


def load_per_game_outcomes(
    pairwise_v4: str, pairwise_gnn: str, results_csv: str
) -> pd.DataFrame:
    """Return DataFrame of (season, p_v4_w, p_gnn_w) for every played tournament game."""
    v4 = _load_lookup(pairwise_v4)
    gnn = _load_lookup(pairwise_gnn)
    results = pd.read_csv(results_csv)
    rows = []
    for _, g in results.iterrows():
        s, w, l = int(g["Season"]), int(g["WTeamID"]), int(g["LTeamID"])
        if s < 2003 or s > 2025 or s == 2020:
            continue
        a, b = (w, l) if w < l else (l, w)
        p_v4 = v4.get((s, a, b))
        p_gnn = gnn.get((s, a, b))
        if p_v4 is None or p_gnn is None:
            continue
        p_v4_w = p_v4 if a == w else 1.0 - p_v4
        p_gnn_w = p_gnn if a == w else 1.0 - p_gnn
        rows.append({"season": s, "p_v4_w": p_v4_w, "p_gnn_w": p_gnn_w})
    return pd.DataFrame(rows)


def fit_w_loso(games: pd.DataFrame, holdout_season: int) -> float:
    """Grid-search w on all games NOT in holdout_season."""
    train = games[games["season"] != holdout_season]
    eps = 1e-15
    ws = np.linspace(0.0, 1.0, 101)
    best_ll = np.inf
    best_w = 1.0
    for w in ws:
        p_blend = w * train["p_v4_w"].values + (1 - w) * train["p_gnn_w"].values
        p_blend = np.clip(p_blend, eps, 1 - eps)
        ll = float(-np.log(p_blend).mean())
        if ll < best_ll:
            best_ll = ll
            best_w = float(w)
    return best_w


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairwise-v4", default="output/pairwise_v4.csv")
    parser.add_argument("--pairwise-gnn", default="output/pairwise_gnn_phase2.csv")
    parser.add_argument("--results", default=str(DATA / "MNCAATourneyCompactResults.csv"))
    parser.add_argument("--out", default="output/pairwise_v4_with_gnn_blend_loso.csv")
    parser.add_argument("--ws-out", default="output/gnn_blend_loso_weights.csv",
                        help="Per-season LOSO-fit w_v4 (audit trail)")
    args = parser.parse_args(argv)

    games = load_per_game_outcomes(args.pairwise_v4, args.pairwise_gnn, args.results)
    seasons = sorted(games["season"].unique())

    loso_ws = {}
    print(f"{'season':>6}  {'n_train':>7}  {'w_v4':>5}")
    for s in seasons:
        w = fit_w_loso(games, s)
        loso_ws[s] = w
        n_train = int((games["season"] != s).sum())
        print(f"{s:>6}  {n_train:>7}  {w:>5.2f}")

    pd.DataFrame(
        [{"season": s, "w_v4": w} for s, w in sorted(loso_ws.items())]
    ).to_csv(args.ws_out, index=False)
    print(f"\nWrote per-season weights to {args.ws_out}")

    v4 = pd.read_csv(args.pairwise_v4).drop_duplicates(
        ["season", "team_a", "team_b"], keep="last"
    )
    gnn = pd.read_csv(args.pairwise_gnn).drop_duplicates(
        ["season", "team_a", "team_b"], keep="last"
    )
    merged = v4.merge(gnn, on=["season", "team_a", "team_b"],
                      suffixes=("_v4", "_gnn"), how="inner")
    merged["w_v4_loso"] = merged["season"].map(loso_ws)
    merged["p_a_wins"] = (
        merged["w_v4_loso"] * merged["p_a_wins_v4"]
        + (1 - merged["w_v4_loso"]) * merged["p_a_wins_gnn"]
    )
    out = merged[["season", "team_a", "team_b", "p_a_wins"]]
    out.to_csv(args.out, index=False)
    print(f"Wrote {len(out)} blended pairs to {args.out}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
