"""Optimize team selection for alternative bracket pool format.

Pool rules:
- Pick 10 teams
- At most 1 per seed for seeds 1-4, unlimited for 5-16
- Points per win = team's seed number
- Bonus: +6 for E8 win, +10 for F4 win, +18 for Championship win
"""

import pandas as pd


def compute_expected_points(merged: pd.DataFrame, e8_bonus: int = 6, f4_bonus: int = 10, champ_bonus: int = 18) -> pd.DataFrame:
    """Compute expected pool points per team including deep-run bonuses."""
    df = merged.copy()
    df["expected_wins"] = df["R1"] + df["R2"] + df["R3"] + df["R4"] + df["R5"] + df["R6"]
    df["base_pts"] = df["Seed"] * df["expected_wins"]
    df["bonus_pts"] = e8_bonus * df["R4"] + f4_bonus * df["R5"] + champ_bonus * df["R6"]
    df["expected_points"] = df["base_pts"] + df["bonus_pts"]
    return df


def optimize_pool(merged: pd.DataFrame, n_picks: int = 10) -> list[dict]:
    """Find the n_picks teams maximizing expected points.

    Constraint: at most 1 team per seed for seeds 1-4.
    """
    df = compute_expected_points(merged)

    # Best candidate per seed 1-4
    top_seeds = []
    for seed in range(1, 5):
        seed_df = df[df["Seed"] == seed].sort_values("expected_points", ascending=False)
        if len(seed_df) > 0:
            top_seeds.append(seed_df.iloc[0])

    # All seed 5+ candidates sorted by expected points
    s5plus = df[df["Seed"] >= 5].sort_values("expected_points", ascending=False).to_dict("records")

    best_score = 0
    best_combo = None

    # Try all 16 combinations of including/excluding seeds 1-4
    for mask in range(16):
        chosen = []
        for i in range(len(top_seeds)):
            if mask & (1 << i):
                t = top_seeds[i]
                chosen.append({
                    "TeamName": t["TeamName"], "Seed": int(t["Seed"]),
                    "Region": t["Region"],
                    "expected_points": float(t["expected_points"]),
                    "expected_wins": float(t["expected_wins"]),
                    "base_pts": float(t["base_pts"]),
                    "bonus_pts": float(t["bonus_pts"]),
                })
        remaining = n_picks - len(chosen)
        fill = s5plus[:remaining]
        total = sum(t["expected_points"] for t in chosen) + sum(t["expected_points"] for t in fill)
        if total > best_score:
            best_score = total
            best_combo = chosen + fill

    return best_combo


def main():
    probs = pd.read_csv("output/bracket_2026.csv")
    struct = pd.read_csv("output/bracket_2026_structure.csv")
    merged = probs.merge(struct[["TeamID", "Seed", "Region"]], on="TeamID")

    combo = optimize_pool(merged)
    total = sum(t["expected_points"] for t in combo)

    print("=" * 65)
    print("OPTIMAL 10-TEAM POOL (with E8/F4/Champ bonuses)")
    print(f"Total Expected Points: {total:.1f}")
    print("=" * 65)
    print()
    print(f"{'Pick':>4s}  {'Seed':>4s}  {'Team':25s} {'Base':>6s} {'Bonus':>6s} {'Total':>6s}")
    print("-" * 60)
    tb = tw = tt = 0
    for i, t in enumerate(combo):
        print(f"{i+1:4d}  {int(t['Seed']):4d}  {t['TeamName']:25s} {t['base_pts']:6.1f} {t['bonus_pts']:6.1f} {t['expected_points']:6.1f}")
        tb += t["base_pts"]
        tw += t["bonus_pts"]
        tt += t["expected_points"]
    print("-" * 60)
    print(f"      TOTAL                            {tb:6.1f} {tw:6.1f} {tt:6.1f}")


if __name__ == "__main__":
    main()
