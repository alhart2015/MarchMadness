"""Multi-year backtest analysis: compare v1/v2/v3 LOSO CV per season.

Reads per-season CV results saved by each model script:
  output/cv_per_season_v1.csv
  output/cv_per_season_v2.csv
  output/cv_per_season_v3.csv

For each season 2003-2025, reports log loss + accuracy by version.
Counts wins per version and identifies the seasons where each version
materially out-/under-performed the others.
"""
from pathlib import Path

import pandas as pd


VERSION_CSVS = {
    "v1": "output/cv_per_season_v1.csv",
    "v2": "output/cv_per_season_v2.csv",
    "v3": "output/cv_per_season_v3.csv",
}

LL_TIE_THRESHOLD = 0.01    # log-loss diff smaller than this counts as a tie
ACC_TIE_THRESHOLD = 0.02   # accuracy diff smaller than this counts as a tie


def main():
    dfs = {}
    for v, path in VERSION_CSVS.items():
        if not Path(path).exists():
            print(f"[skip] {v}: {path} not found")
            continue
        df = pd.read_csv(path).sort_values("season").reset_index(drop=True)
        dfs[v] = df

    if len(dfs) < 2:
        print("Need at least 2 versions for comparison.")
        return

    # Merge into one wide table on Season
    base = next(iter(dfs.values()))[["season", "n_games"]].copy()
    for v, df in dfs.items():
        base[f"ll_{v}"] = df["log_loss"].values
        base[f"acc_{v}"] = df["accuracy"].values
        base[f"brier_{v}"] = df["brier_score"].values

    print("=" * 100)
    print("MULTI-YEAR BACKTEST: LOG LOSS BY SEASON")
    print("=" * 100)
    versions = list(dfs.keys())
    print(f"{'Season':>6}  {'N':>3}  | " +
          "  ".join(f"{v:>6}" for v in versions) + "   | best")
    print("-" * (15 + 8 * len(versions) + 8))
    for _, row in base.iterrows():
        ll = {v: row[f"ll_{v}"] for v in versions}
        best = min(ll, key=ll.get)
        cells = []
        for v in versions:
            tag = " *" if v == best else "  "
            cells.append(f"{ll[v]:.3f}{tag}")
        print(f"{int(row['season']):>6}  {int(row['n_games']):>3}  | " +
              "  ".join(f"{c:>6}" for c in cells) + f"   | {best}")

    print(f"\n{'MEAN':>6}       | " +
          "  ".join(f"{base[f'll_{v}'].mean():>6.3f}" for v in versions))
    print(f"{'MEDIAN':>6}       | " +
          "  ".join(f"{base[f'll_{v}'].median():>6.3f}" for v in versions))

    # Win counts (best log loss per season)
    print(f"\n{'=' * 100}\nLOG-LOSS HEAD-TO-HEAD (best per season; tie = within {LL_TIE_THRESHOLD})")
    print("=" * 100)
    wins = {v: 0 for v in versions}
    ties = 0
    for _, row in base.iterrows():
        ll = {v: row[f"ll_{v}"] for v in versions}
        sorted_v = sorted(ll, key=ll.get)
        if ll[sorted_v[1]] - ll[sorted_v[0]] < LL_TIE_THRESHOLD:
            ties += 1
        else:
            wins[sorted_v[0]] += 1
    n = len(base)
    for v in versions:
        print(f"  {v}: {wins[v]:>3} wins ({wins[v]/n:>5.1%})")
    print(f"  ties: {ties:>3} ({ties/n:>5.1%})")

    # Accuracy
    print(f"\n{'=' * 100}\nACCURACY BY SEASON")
    print("=" * 100)
    print(f"{'Season':>6}  {'N':>3}  | " +
          "  ".join(f"{v:>7}" for v in versions) + "   | best")
    print("-" * (15 + 9 * len(versions) + 8))
    for _, row in base.iterrows():
        acc = {v: row[f"acc_{v}"] for v in versions}
        best = max(acc, key=acc.get)
        cells = []
        for v in versions:
            tag = " *" if v == best else "  "
            cells.append(f"{acc[v]*100:.1f}%{tag}")
        print(f"{int(row['season']):>6}  {int(row['n_games']):>3}  | " +
              "  ".join(f"{c:>7}" for c in cells) + f"   | {best}")
    print(f"\n{'MEAN':>6}       | " +
          "  ".join(f"{base[f'acc_{v}'].mean()*100:>5.1f}%" for v in versions))

    # Pairwise season-over-season delta
    if "v1" in dfs and "v3" in dfs:
        print(f"\n{'=' * 100}\nv1 vs v3 PER-SEASON DELTA (positive = v3 better at log loss)")
        print("=" * 100)
        base["ll_diff_v3_v1"] = base["ll_v1"] - base["ll_v3"]
        base["acc_diff_v3_v1"] = base["acc_v3"] - base["acc_v1"]
        for _, row in base.iterrows():
            d_ll = row["ll_diff_v3_v1"]
            d_acc = row["acc_diff_v3_v1"]
            tag = ""
            if d_ll > LL_TIE_THRESHOLD:
                tag = "v3 better LL"
            elif d_ll < -LL_TIE_THRESHOLD:
                tag = "v1 better LL"
            print(f"  {int(row['season']):>4}: LL delta {d_ll:>+7.3f}, "
                  f"acc delta {d_acc*100:>+5.1f}pp   {tag}")
        print(f"\n  Mean LL delta (v1-v3) = {base['ll_diff_v3_v1'].mean():+.3f}")
        print(f"  Mean acc delta (v3-v1) = {base['acc_diff_v3_v1'].mean()*100:+.2f}pp")
        v3_better_ll = (base["ll_diff_v3_v1"] > LL_TIE_THRESHOLD).sum()
        v1_better_ll = (base["ll_diff_v3_v1"] < -LL_TIE_THRESHOLD).sum()
        print(f"  v3 better at log loss in {v3_better_ll} / {len(base)} seasons")
        print(f"  v1 better at log loss in {v1_better_ll} / {len(base)} seasons")


if __name__ == "__main__":
    main()
