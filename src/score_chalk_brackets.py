"""Score each model version's chalk bracket per season against actuals.

For each model version (v1/v2/v3) and each LOSO season (2003-2025):
  1. Load pairwise probs from output/pairwise_v<N>.csv.
  2. Walk the bracket slot structure, picking the chalk winner at each slot.
  3. Score chalk's round-R advancers against actual round-R winners.

Bracket scoring (per round of advancement, 1/2/4/8/16/32 weighting):
  R1=R64, R2=R32, R3=S16, R4=E8, R5=F4, R6=Champ.

Inputs:
  output/pairwise_v1.csv, pairwise_v2.csv, pairwise_v3.csv  (from LOSO patches)
  data/raw/march-machine-learning-2026/MNCAATourneySlots.csv
  data/raw/march-machine-learning-2026/MNCAATourneySeeds.csv
  data/raw/march-machine-learning-2026/MNCAATourneyCompactResults.csv
"""
from collections import defaultdict
from pathlib import Path

import pandas as pd

DATA = Path("data/raw/march-machine-learning-2026")
ROUND_BY_PREFIX = {"R1": "R64", "R2": "R32", "R3": "S16",
                   "R4": "E8", "R5": "F4", "R6": "Champ"}
ROUND_PTS = {"R64": 1, "R32": 2, "S16": 4, "E8": 8, "F4": 16, "Champ": 32}
ROUND_ORDER = ["R64", "R32", "S16", "E8", "F4", "Champ"]


def load_pairwise(path):
    """Return {season: {(a,b): p_a_wins}} where a < b."""
    if not Path(path).exists():
        return {}
    df = pd.read_csv(path)
    out = {}
    for season, group in df.groupby("season"):
        d = {}
        for _, row in group.iterrows():
            a, b = int(row["team_a"]), int(row["team_b"])
            if a < b:
                d[(a, b)] = float(row["p_a_wins"])
            else:
                d[(b, a)] = 1.0 - float(row["p_a_wins"])
        out[int(season)] = d
    return out


def get_prob(probs_for_season, a, b):
    """P(a beats b)."""
    if a < b:
        return probs_for_season.get((a, b), 0.5)
    return 1.0 - probs_for_season.get((b, a), 0.5)


def chalk_pick(probs_for_season, a, b):
    return a if get_prob(probs_for_season, a, b) >= 0.5 else b


def round_of_slot(slot):
    """Map slot string like 'R1W3' or 'W11' or 'R6CH' to round name."""
    if slot.startswith("R"):
        return ROUND_BY_PREFIX.get(slot[:2])
    return None  # play-in


def resolve_slot(slot, slots_df, seed_to_team, slot_winners,
                 actual_winners_per_slot):
    """Resolve a slot to (team_a, team_b)."""
    row = slots_df[slots_df.Slot == slot].iloc[0]
    strong = row["StrongSeed"]
    weak = row["WeakSeed"]
    team_a = _resolve_seed_or_slot(strong, seed_to_team, slot_winners,
                                   actual_winners_per_slot)
    team_b = _resolve_seed_or_slot(weak, seed_to_team, slot_winners,
                                   actual_winners_per_slot)
    return team_a, team_b


def _resolve_seed_or_slot(s, seed_to_team, slot_winners, actual_winners_per_slot):
    """Resolve a seed string or slot reference to a team ID."""
    if s in seed_to_team:
        return seed_to_team[s]
    if s in slot_winners:
        return slot_winners[s]
    if s in actual_winners_per_slot:
        return actual_winners_per_slot[s]
    return None


def actual_winners_by_slot(season, slots_df, seed_to_team, results):
    """For a season, derive {slot: winning_team_id} for every slot using the
    actual game results. We do this by topologically resolving slots round by
    round, picking the winner from each game's two participants."""
    # Round prefix order: play-in slots (no R prefix) first, then R1..R6.
    season_slots = slots_df[slots_df.Season == season]
    play_in_slots = season_slots[~season_slots.Slot.str.startswith("R")]
    round_slots = {f"R{r}": season_slots[season_slots.Slot.str.startswith(f"R{r}")]
                   for r in range(1, 7)}

    season_results = results[results.Season == season]
    # Build an index: for each (team_a, team_b) frozenset, look up the winner.
    pair_winner = {}
    for _, g in season_results.iterrows():
        pair_winner[frozenset({int(g.WTeamID), int(g.LTeamID)})] = int(g.WTeamID)

    slot_winners = {}

    # Play-ins
    for _, row in play_in_slots.iterrows():
        a = seed_to_team.get(row.StrongSeed)
        b = seed_to_team.get(row.WeakSeed)
        if a is None or b is None:
            continue
        winner = pair_winner.get(frozenset({a, b}))
        if winner is not None:
            slot_winners[row.Slot] = winner

    # R1..R6 in order
    for r in range(1, 7):
        for _, row in round_slots[f"R{r}"].iterrows():
            a = _resolve_seed_or_slot(row.StrongSeed, seed_to_team, slot_winners, {})
            b = _resolve_seed_or_slot(row.WeakSeed, seed_to_team, slot_winners, {})
            if a is None or b is None:
                continue
            winner = pair_winner.get(frozenset({a, b}))
            if winner is not None:
                slot_winners[row.Slot] = winner

    return slot_winners


def chalk_winners_by_slot(season, slots_df, seed_to_team, probs_for_season,
                          actual_play_in_winners):
    """Walk each slot in round order; chalk-pick the winner. Use ACTUAL
    play-in winners to resolve play-in seeds (since pools usually skip those)."""
    season_slots = slots_df[slots_df.Season == season]
    round_slots = {f"R{r}": season_slots[season_slots.Slot.str.startswith(f"R{r}")]
                   for r in range(1, 7)}

    chalk_winners = {}
    # Play-ins resolved to actual winner so chalk seed lookups work uniformly.
    chalk_winners.update(actual_play_in_winners)

    for r in range(1, 7):
        for _, row in round_slots[f"R{r}"].iterrows():
            a = _resolve_seed_or_slot(row.StrongSeed, seed_to_team, chalk_winners, {})
            b = _resolve_seed_or_slot(row.WeakSeed, seed_to_team, chalk_winners, {})
            if a is None or b is None:
                continue
            chalk_winners[row.Slot] = chalk_pick(probs_for_season, a, b)
    return chalk_winners


def score_season(season, slots_df, seeds_df, results_df, probs_for_season):
    """Return dict of round_name -> (chalk_correct, n, points)."""
    season_seeds = seeds_df[seeds_df.Season == season]
    seed_to_team = dict(zip(season_seeds.Seed, season_seeds.TeamID.astype(int)))

    # Actual play-in winners (from results, days 134-135 typically)
    actual_all = actual_winners_by_slot(season, slots_df, seed_to_team, results_df)
    play_in_slots = slots_df[(slots_df.Season == season) &
                              (~slots_df.Slot.str.startswith("R"))]["Slot"].tolist()
    actual_play_in = {s: actual_all[s] for s in play_in_slots if s in actual_all}

    chalk = chalk_winners_by_slot(season, slots_df, seed_to_team,
                                  probs_for_season, actual_play_in)

    out = {}
    for r in range(1, 7):
        round_name = ROUND_BY_PREFIX[f"R{r}"]
        slots_in_round = slots_df[(slots_df.Season == season) &
                                  slots_df.Slot.str.startswith(f"R{r}")]["Slot"].tolist()
        chalk_winners = {chalk[s] for s in slots_in_round if s in chalk}
        actual_winners = {actual_all[s] for s in slots_in_round if s in actual_all}
        correct = len(chalk_winners & actual_winners)
        n = len(actual_winners)
        out[round_name] = (correct, n, correct * ROUND_PTS[round_name])
    return out


def _score_pairwise_with_details(slots_df, seeds_df, results_df, path):
    """Internal: score a pairwise CSV, return full per-round details.
    Returns {season: {round_name: (correct, n, points)}}.
    """
    if not Path(path).exists():
        return {}
    probs_by_season = load_pairwise(path)
    by_season = {}
    for season, probs in probs_by_season.items():
        by_season[int(season)] = score_season(season, slots_df, seeds_df,
                                               results_df, probs)
    return by_season


def score_pairwise_path(path):
    """Score the chalk bracket implied by `path` against actuals across all
    seasons present. Returns {"total_pts": float, "per_season_pts": {int: float}}.
    Raises FileNotFoundError if `path` does not exist.
    """
    if not Path(path).exists():
        raise FileNotFoundError(path)

    slots_df = pd.read_csv(DATA / "MNCAATourneySlots.csv")
    seeds_df = pd.read_csv(DATA / "MNCAATourneySeeds.csv")
    results_df = pd.read_csv(DATA / "MNCAATourneyCompactResults.csv")

    by_season = _score_pairwise_with_details(slots_df, seeds_df, results_df, path)
    per_season = {}
    for season, rounds_dict in by_season.items():
        total_pts = sum(t[2] for t in rounds_dict.values())
        per_season[int(season)] = float(total_pts)
    return {
        "total_pts": float(sum(per_season.values())),
        "per_season_pts": per_season,
    }


def main():
    slots_df = pd.read_csv(DATA / "MNCAATourneySlots.csv")
    seeds_df = pd.read_csv(DATA / "MNCAATourneySeeds.csv")
    results_df = pd.read_csv(DATA / "MNCAATourneyCompactResults.csv")

    versions = ["v1", "v2", "v4", "v8"]
    by_version_season = {}
    for v in versions:
        path = f"output/pairwise_{v}.csv"
        details = _score_pairwise_with_details(slots_df, seeds_df, results_df, path)
        if details:
            by_version_season[v] = details

    pw = by_version_season
    if not pw:
        print("No pairwise probs found. Run the model scripts with MM_PAIRWISE_OUT first.")
        return

    common_seasons = sorted(set.intersection(*(set(d.keys()) for d in pw.values())))
    print(f"Scoring {len(common_seasons)} seasons across {len(pw)} versions: {list(pw)}")

    # Per-season totals
    print("\n" + "=" * 90)
    print("BRACKET POINTS BY SEASON (1/2/4/8/16/32 per round; max 192)")
    print("=" * 90)
    print(f"{'Season':>6}  | " + "  ".join(f"{v:>8}" for v in pw) + "  | best")
    print("-" * (12 + 10 * len(pw)))
    totals = {v: 0 for v in pw}
    season_winners = defaultdict(int)
    for s in common_seasons:
        cells = []
        scores = {}
        for v in pw:
            tot = sum(t[2] for t in by_version_season[v][s].values())
            scores[v] = tot
            totals[v] += tot
            cells.append(f"{tot:>3}/192")
        best = max(scores, key=scores.get)
        season_winners[best] += 1
        print(f"{s:>6}  | " + "  ".join(f"{c:>8}" for c in cells) + f"  | {best}")
    print("-" * (12 + 10 * len(pw)))
    print(f"{'TOTAL':>6}  | " + "  ".join(f"{totals[v]:>4}" for v in pw))
    print(f"{'MEAN':>6}   | " +
          "  ".join(f"{totals[v]/len(common_seasons):>6.1f}" for v in pw))
    print(f"\nWins per version: " +
          ", ".join(f"{v}={season_winners.get(v, 0)}" for v in pw))

    # Per-round average across seasons
    print(f"\n{'=' * 90}\nMEAN POINTS PER ROUND PER SEASON")
    print("=" * 90)
    print(f"{'Round':<6}  | " + "  ".join(f"{v:>8}" for v in pw))
    print("-" * (12 + 10 * len(pw)))
    for rnd in ROUND_ORDER:
        cells = []
        for v in pw:
            avg = sum(by_version_season[v][s][rnd][2] for s in common_seasons) / len(common_seasons)
            cells.append(f"{avg:>4.1f}")
        print(f"{rnd:<6}  | " + "  ".join(f"{c:>8}" for c in cells))

    # Per-round average accuracy (correct / n)
    print(f"\n{'=' * 90}\nMEAN ACCURACY PER ROUND")
    print("=" * 90)
    print(f"{'Round':<6}  | " + "  ".join(f"{v:>8}" for v in pw))
    print("-" * (12 + 10 * len(pw)))
    for rnd in ROUND_ORDER:
        cells = []
        for v in pw:
            corrects = [by_version_season[v][s][rnd][0] for s in common_seasons]
            ns = [by_version_season[v][s][rnd][1] for s in common_seasons]
            acc = sum(corrects) / max(sum(ns), 1)
            cells.append(f"{acc*100:>5.1f}%")
        print(f"{rnd:<6}  | " + "  ".join(f"{c:>8}" for c in cells))


if __name__ == "__main__":
    main()
