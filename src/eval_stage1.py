"""Per-season LOSO log loss + accuracy from a pairwise CSV.

Spec: docs/superpowers/specs/2026-05-01-ensemble-stage1-design.md

Reads a pairwise CSV (season, team_a, team_b, p_a_wins; team_a < team_b)
and the Kaggle MNCAATourneyCompactResults.csv, computes per-season log
loss + accuracy on tournament games, then weighted-mean across seasons
(weight = n_games per season).

Used to compare pairwise_v4.csv vs pairwise_ensemble.csv as the stage-1
only head-to-head before the v9-C correction step.
"""
import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path("data/raw/march-machine-learning-2026")


def evaluate_pairwise(
    pairwise_csv: str,
    results_csv: str = str(DATA / "MNCAATourneyCompactResults.csv"),
    results_df: pd.DataFrame = None,
) -> dict:
    """Compute per-season log loss + accuracy for a pairwise CSV.

    pairwise_csv may have duplicate (season, team_a, team_b) rows from
    multiple LOSO passes (default + tuned); we take the last write per
    pair, matching `train_upset_model.load_per_game_data_with_upset`.
    """
    pw = pd.read_csv(pairwise_csv)
    pw = pw.drop_duplicates(subset=["season", "team_a", "team_b"], keep="last")
    pw_lookup = {}
    for s, a, b, p in zip(pw.season, pw.team_a, pw.team_b, pw.p_a_wins):
        pw_lookup[(int(s), int(a), int(b))] = float(p)

    if results_df is None:
        results_df = pd.read_csv(results_csv)

    per_season = {}
    eps = 1e-15
    for season, group in results_df.groupby("Season"):
        season = int(season)
        if season < 2003:
            continue
        ll_terms = []
        correct = 0
        for _, g in group.iterrows():
            w, l = int(g["WTeamID"]), int(g["LTeamID"])
            a, b = (w, l) if w < l else (l, w)
            p = pw_lookup.get((season, a, b))
            if p is None:
                continue
            p_w = p if a == w else 1.0 - p
            p_w = min(max(p_w, eps), 1.0 - eps)
            ll_terms.append(-math.log(p_w))
            correct += 1 if p_w > 0.5 else 0
        if not ll_terms:
            continue
        per_season[season] = {
            "n_games": len(ll_terms),
            "log_loss": float(np.mean(ll_terms)),
            "accuracy": float(correct / len(ll_terms)),
        }

    if not per_season:
        return {
            "per_season": {},
            "weighted_mean_log_loss": float("nan"),
            "weighted_mean_accuracy": float("nan"),
            "total_games": 0,
        }

    total_n = sum(s["n_games"] for s in per_season.values())
    wm_ll = sum(s["log_loss"] * s["n_games"] for s in per_season.values()) / total_n
    wm_acc = sum(s["accuracy"] * s["n_games"] for s in per_season.values()) / total_n
    return {
        "per_season": per_season,
        "weighted_mean_log_loss": float(wm_ll),
        "weighted_mean_accuracy": float(wm_acc),
        "total_games": total_n,
    }


def _print_table(name: str, result: dict) -> None:
    print(f"\n=== {name} ===")
    print(f"{'season':>6}  {'n':>4}  {'log_loss':>9}  {'accuracy':>8}")
    for s, m in sorted(result["per_season"].items()):
        print(f"{s:>6}  {m['n_games']:>4}  {m['log_loss']:>9.4f}  "
              f"{m['accuracy']:>8.3f}")
    print(f"{'WMEAN':>6}  {result['total_games']:>4}  "
          f"{result['weighted_mean_log_loss']:>9.4f}  "
          f"{result['weighted_mean_accuracy']:>8.3f}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--pairwise", required=True, help="pairwise CSV")
    parser.add_argument("--label", default=None, help="optional label for table header")
    args = parser.parse_args(argv)
    res = evaluate_pairwise(args.pairwise)
    label = args.label or args.pairwise
    _print_table(label, res)
    return 0


if __name__ == "__main__":
    sys.exit(main())
