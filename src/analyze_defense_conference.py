"""Empirical check: do defensive teams and certain conferences outperform
seed expectation in the tournament?

Uses KenPom Barttorvik data (2008-2026, 1215 team-season rows) which has
KADJ O / KADJ D ranks, conference, seed, and the ROUND the team reached.

For each tournament team we compute:
  expected_round   = expected exit round for a team of that seed (historical)
  actual_round     = actual exit round in that year
  perf_delta       = actual - expected (positive = overperformed)

Then test whether perf_delta correlates with:
  - defensive emphasis (KADJ D rank vs KADJ O rank)
  - conference (power vs mid-major)
  - mid-major elite-rating flag (top KenPom from a small conference)

ROUND encoding: 1=champ, 2=runner-up, 4=F4 loser, 8=E8 loser, 16=S16 loser,
32=R32 loser, 64=R64 loser, 68=First Four loser, 0=missed tournament.
We convert to a "rounds won" scale for clearer interpretation.
"""
import numpy as np
import pandas as pd

KP_PATH = "data/raw/kaggle/KenPom Barttorvik.csv"

# Map ROUND to "rounds won" (= 6 means won championship, 0 = lost in R64)
ROUND_TO_WINS = {1: 6, 2: 5, 4: 4, 8: 3, 16: 2, 32: 1, 64: 0, 68: -1}

# Power conferences (rough; based on tournament autobid history)
POWER_CONFS = {"ACC", "B10", "B12", "BE", "SEC", "P12"}


def main():
    df = pd.read_csv(KP_PATH)
    df = df[df["ROUND"].isin(ROUND_TO_WINS)].copy()
    df["rounds_won"] = df["ROUND"].map(ROUND_TO_WINS)

    # Expected rounds_won by seed (historical mean).
    by_seed = df.groupby("SEED")["rounds_won"].mean().rename("expected_rounds_won")
    df = df.merge(by_seed, on="SEED")
    df["perf_delta"] = df["rounds_won"] - df["expected_rounds_won"]

    print("=" * 80)
    print(f"KENPOM TOURNAMENT TEAMS: {len(df)} team-seasons, {df.YEAR.nunique()} years"
          f" ({df.YEAR.min()}-{df.YEAR.max()})")
    print("=" * 80)

    print("\nExpected rounds-won by seed (baseline):")
    for seed, row in by_seed.items():
        print(f"  Seed {int(seed):2d}: {row:.2f} rounds won on average")

    # Defensive vs offensive emphasis
    print("\n" + "=" * 80)
    print("DEFENSE-FIRST vs OFFENSE-FIRST")
    print("=" * 80)
    df["def_minus_off_rank"] = df["KADJ O RANK"] - df["KADJ D RANK"]
    # Higher def_minus_off_rank means defense ranks BETTER than offense (defense-first).

    print(f"\n  Mean perf_delta when defense ranks BETTER than offense: "
          f"{df[df['def_minus_off_rank'] > 0]['perf_delta'].mean():+.3f}")
    print(f"  Mean perf_delta when offense ranks BETTER than defense: "
          f"{df[df['def_minus_off_rank'] < 0]['perf_delta'].mean():+.3f}")
    print(f"  Mean perf_delta when balanced (within 5 ranks):              "
          f"{df[df['def_minus_off_rank'].abs() <= 5]['perf_delta'].mean():+.3f}")

    # Stratified by seed group: high-seed (1-4), mid-seed (5-12), low-seed (13-16)
    print("\n  By seed group:")
    for label, mask in [
        ("Seeds 1-4   ", df.SEED.between(1, 4)),
        ("Seeds 5-12  ", df.SEED.between(5, 12)),
        ("Seeds 13-16 ", df.SEED.between(13, 16)),
    ]:
        sub = df[mask]
        n = len(sub)
        d_better = sub[sub["def_minus_off_rank"] > 0]["perf_delta"].mean()
        o_better = sub[sub["def_minus_off_rank"] < 0]["perf_delta"].mean()
        d_n = (sub["def_minus_off_rank"] > 0).sum()
        o_n = (sub["def_minus_off_rank"] < 0).sum()
        print(f"    {label}  n={n:>3}: D-first ({d_n:>3} teams) {d_better:+.3f}"
              f"  vs  O-first ({o_n:>3} teams) {o_better:+.3f}")

    # Correlation
    corr = df[["KADJ D RANK", "KADJ O RANK", "rounds_won", "perf_delta"]].corr()
    print("\n  Pearson correlations with rounds_won:")
    print(f"    KADJ D RANK (lower=better defense): {corr.loc['KADJ D RANK', 'rounds_won']:+.3f}")
    print(f"    KADJ O RANK (lower=better offense): {corr.loc['KADJ O RANK', 'rounds_won']:+.3f}")
    print(f"  Pearson correlations with perf_delta:")
    print(f"    KADJ D RANK: {corr.loc['KADJ D RANK', 'perf_delta']:+.3f}")
    print(f"    KADJ O RANK: {corr.loc['KADJ O RANK', 'perf_delta']:+.3f}")

    # Conference analysis
    print("\n" + "=" * 80)
    print("CONFERENCE QUALITY")
    print("=" * 80)
    df["is_power"] = df["CONF"].isin(POWER_CONFS)
    p = df[df.is_power]["perf_delta"].mean()
    np_ = df[~df.is_power]["perf_delta"].mean()
    p_n = df.is_power.sum()
    np_n = (~df.is_power).sum()
    print(f"\n  Power-conference teams (n={p_n}, ACC/B10/B12/BE/SEC/P12): "
          f"perf_delta = {p:+.3f}")
    print(f"  Non-power teams        (n={np_n}, all others):                "
          f"perf_delta = {np_:+.3f}")

    print("\n  By seed group, power vs non-power:")
    for label, mask in [
        ("Seeds 1-4   ", df.SEED.between(1, 4)),
        ("Seeds 5-12  ", df.SEED.between(5, 12)),
        ("Seeds 13-16 ", df.SEED.between(13, 16)),
    ]:
        sub = df[mask]
        if len(sub) == 0:
            continue
        p = sub[sub.is_power]["perf_delta"].mean()
        np_ = sub[~sub.is_power]["perf_delta"].mean()
        p_n = sub.is_power.sum()
        np_n = (~sub.is_power).sum()
        print(f"    {label}  power ({p_n:>3}) {p:+.3f}  vs  non-power ({np_n:>3}) {np_:+.3f}")

    # Per-conference perf
    print("\n  Top 12 conferences by mean perf_delta (min 20 team-seasons):")
    grp = df.groupby("CONF").agg(n=("perf_delta", "size"),
                                   mean_perf=("perf_delta", "mean"),
                                   mean_seed=("SEED", "mean")).reset_index()
    grp = grp[grp["n"] >= 20].sort_values("mean_perf", ascending=False)
    print(f"    {'CONF':<8} {'n':>3}  {'mean_seed':>9}  {'perf_delta':>11}")
    for _, row in grp.iterrows():
        print(f"    {row['CONF']:<8} {int(row['n']):>3}  {row['mean_seed']:>9.1f}  {row['mean_perf']:>+11.3f}")

    # Mid-major top-rated teams: highly-rated KenPom but in non-power conference
    print("\n" + "=" * 80)
    print("MID-MAJOR ELITE TEAMS (KADJ EM RANK <= 25, non-power conference)")
    print("=" * 80)
    midmajor_elite = df[(df["KADJ EM RANK"] <= 25) & (~df.is_power)].copy()
    midmajor_elite = midmajor_elite[["YEAR", "TEAM", "CONF", "SEED",
                                       "KADJ EM RANK", "rounds_won",
                                       "expected_rounds_won", "perf_delta"]]
    if len(midmajor_elite):
        print(f"  n={len(midmajor_elite)} team-seasons. Mean perf_delta = "
              f"{midmajor_elite.perf_delta.mean():+.3f}")
        print(f"  (Comparison: power-conference teams with KADJ EM RANK <= 25: "
              f"{df[(df['KADJ EM RANK'] <= 25) & df.is_power]['perf_delta'].mean():+.3f})")
        print(f"\n  Top mid-major elites and how they fared:")
        midmajor_elite = midmajor_elite.sort_values("YEAR")
        for _, r in midmajor_elite.head(40).iterrows():
            print(f"    {int(r.YEAR)} {r.TEAM:<20} ({r.CONF}, seed {int(r.SEED)}): "
                  f"reached round {int(r.rounds_won)}/6, expected "
                  f"{r.expected_rounds_won:.1f}, delta {r.perf_delta:+.2f}")


if __name__ == "__main__":
    main()
