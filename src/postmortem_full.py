"""Full tournament postmortem through the championship.

Extends src/postmortem.py (which stops at R32) with the actual S16/E8/F4/Champ
results. Computes per-round log loss and accuracy, expected vs actual correct
picks, high-confidence misses, and bracket points scored under our chalk
strategy.
"""
import json
import math
from collections import defaultdict


with open("output/pairwise_probs.json") as f:
    probs = json.load(f)


def get_prob(id_a, id_b):
    """P(team A beats team B) per the model's pairwise probability table."""
    key = f"{min(id_a, id_b)}_{max(id_a, id_b)}"
    p = probs.get(key, 0.5)
    return p if id_a < id_b else 1 - p


teams = {
    "Duke": 1181, "Siena": 1373, "Ohio St.": 1326, "TCU": 1395,
    "St. Johns": 1385, "Northern Iowa": 1320, "Kansas": 1242, "Cal Baptist": 1465,
    "Louisville": 1257, "South Florida": 1378, "Michigan St.": 1277, "North Dakota St.": 1295,
    "UCLA": 1417, "UCF": 1416, "Connecticut": 1163, "Furman": 1202,
    "Arizona": 1112, "LIU Brooklyn": 1254, "Villanova": 1437, "Utah St.": 1429,
    "Wisconsin": 1458, "High Point": 1219, "Arkansas": 1116, "Hawaii": 1218,
    "BYU": 1140, "Texas": 1400, "Gonzaga": 1211, "Kennesaw St.": 1244,
    "Miami FL": 1274, "Missouri": 1281, "Purdue": 1345, "Queens": 1474,
    "Florida": 1196, "Prairie View": 1341, "Clemson": 1155, "Iowa": 1234,
    "Vanderbilt": 1435, "McNeese St.": 1270, "Nebraska": 1304, "Troy": 1407,
    "North Carolina": 1314, "VCU": 1433, "Illinois": 1228, "Penn": 1335,
    "Saint Marys": 1388, "Texas A&M": 1401, "Houston": 1222, "Idaho": 1225,
    "Michigan": 1276, "Howard": 1224, "Georgia": 1208, "Saint Louis": 1387,
    "Texas Tech": 1403, "Akron": 1103, "Alabama": 1104, "Hofstra": 1220,
    "Tennessee": 1397, "Miami OH": 1275, "Virginia": 1438, "Wright St.": 1460,
    "Kentucky": 1246, "Santa Clara": 1365, "Iowa St.": 1235, "Tennessee St.": 1398,
}


def chalk_pick(a, b):
    p = get_prob(teams[a], teams[b])
    return a if p >= 0.5 else b


# All actual 2026 tournament results: (winner, loser, round)
all_results = [
    # -- R64 ---------------------------------------------------------------
    ("Duke", "Siena", "R64"), ("TCU", "Ohio St.", "R64"),
    ("St. Johns", "Northern Iowa", "R64"), ("Kansas", "Cal Baptist", "R64"),
    ("Louisville", "South Florida", "R64"), ("Michigan St.", "North Dakota St.", "R64"),
    ("UCLA", "UCF", "R64"), ("Connecticut", "Furman", "R64"),
    ("Arizona", "LIU Brooklyn", "R64"), ("Utah St.", "Villanova", "R64"),
    ("High Point", "Wisconsin", "R64"), ("Arkansas", "Hawaii", "R64"),
    ("Texas", "BYU", "R64"), ("Gonzaga", "Kennesaw St.", "R64"),
    ("Miami FL", "Missouri", "R64"), ("Purdue", "Queens", "R64"),
    ("Florida", "Prairie View", "R64"), ("Iowa", "Clemson", "R64"),
    ("Vanderbilt", "McNeese St.", "R64"), ("Nebraska", "Troy", "R64"),
    ("VCU", "North Carolina", "R64"), ("Illinois", "Penn", "R64"),
    ("Texas A&M", "Saint Marys", "R64"), ("Houston", "Idaho", "R64"),
    ("Michigan", "Howard", "R64"), ("Saint Louis", "Georgia", "R64"),
    ("Texas Tech", "Akron", "R64"), ("Alabama", "Hofstra", "R64"),
    ("Tennessee", "Miami OH", "R64"), ("Virginia", "Wright St.", "R64"),
    ("Kentucky", "Santa Clara", "R64"), ("Iowa St.", "Tennessee St.", "R64"),
    # -- R32 ---------------------------------------------------------------
    ("Duke", "TCU", "R32"), ("Nebraska", "Vanderbilt", "R32"),
    ("Illinois", "VCU", "R32"), ("Houston", "Texas A&M", "R32"),
    ("Arkansas", "High Point", "R32"), ("Texas", "Gonzaga", "R32"),
    ("Michigan", "Saint Louis", "R32"), ("Michigan St.", "Louisville", "R32"),
    ("Alabama", "Texas Tech", "R32"), ("Connecticut", "UCLA", "R32"),
    ("Arizona", "Utah St.", "R32"), ("Iowa", "Florida", "R32"),
    ("Tennessee", "Virginia", "R32"), ("St. Johns", "Kansas", "R32"),
    ("Iowa St.", "Kentucky", "R32"), ("Purdue", "Miami FL", "R32"),
    # -- S16 ---------------------------------------------------------------
    ("Duke", "St. Johns", "S16"), ("Connecticut", "Michigan St.", "S16"),
    ("Arizona", "Arkansas", "S16"), ("Purdue", "Texas", "S16"),
    ("Iowa", "Nebraska", "S16"), ("Illinois", "Houston", "S16"),
    ("Michigan", "Alabama", "S16"), ("Tennessee", "Iowa St.", "S16"),
    # -- E8 ----------------------------------------------------------------
    ("Connecticut", "Duke", "E8"), ("Arizona", "Purdue", "E8"),
    ("Illinois", "Iowa", "E8"), ("Michigan", "Tennessee", "E8"),
    # -- F4 ----------------------------------------------------------------
    ("Connecticut", "Illinois", "F4"), ("Michigan", "Arizona", "F4"),
    # -- Championship ------------------------------------------------------
    ("Michigan", "Connecticut", "Champ"),
]

# Region structure for building our chalk bracket end-to-end.
regions_matchups = {
    "East": [
        ("Duke", "Siena"), ("Ohio St.", "TCU"),
        ("St. Johns", "Northern Iowa"), ("Kansas", "Cal Baptist"),
        ("Louisville", "South Florida"), ("Michigan St.", "North Dakota St."),
        ("UCLA", "UCF"), ("Connecticut", "Furman"),
    ],
    "West": [
        ("Arizona", "LIU Brooklyn"), ("Villanova", "Utah St."),
        ("Wisconsin", "High Point"), ("Arkansas", "Hawaii"),
        ("BYU", "Texas"), ("Gonzaga", "Kennesaw St."),
        ("Miami FL", "Missouri"), ("Purdue", "Queens"),
    ],
    "South": [
        ("Florida", "Prairie View"), ("Clemson", "Iowa"),
        ("Vanderbilt", "McNeese St."), ("Nebraska", "Troy"),
        ("North Carolina", "VCU"), ("Illinois", "Penn"),
        ("Saint Marys", "Texas A&M"), ("Houston", "Idaho"),
    ],
    "Midwest": [
        ("Michigan", "Howard"), ("Georgia", "Saint Louis"),
        ("Texas Tech", "Akron"), ("Alabama", "Hofstra"),
        ("Tennessee", "Miami OH"), ("Virginia", "Wright St."),
        ("Kentucky", "Santa Clara"), ("Iowa St.", "Tennessee St."),
    ],
}

# Build our chalk bracket through the championship.
bracket = {}
region_winners = []
for region, matchups in regions_matchups.items():
    r64w = [chalk_pick(a, b) for a, b in matchups]
    bracket[(region, "R64")] = r64w
    r32w = [chalk_pick(r64w[i], r64w[i + 1]) for i in range(0, 8, 2)]
    bracket[(region, "R32")] = r32w  # our S16 picks (4 per region)
    s16w = [chalk_pick(r32w[i], r32w[i + 1]) for i in range(0, 4, 2)]
    bracket[(region, "S16")] = s16w  # our E8 picks (2 per region)
    e8w = chalk_pick(s16w[0], s16w[1])
    bracket[(region, "E8")] = [e8w]  # our F4 pick (1 per region)
    region_winners.append(e8w)

f4_left = chalk_pick(region_winners[0], region_winners[1])    # East/West winner
f4_right = chalk_pick(region_winners[2], region_winners[3])   # South/Midwest winner
bracket[("FF", "F4")] = [f4_left, f4_right]                   # our Champ-game picks
bracket[("FF", "Champ")] = [chalk_pick(f4_left, f4_right)]    # our champion pick

# Bracket-pick advancement maps for scoring. For each round, a "round pick" is a
# team we said would *advance from* that round's game (i.e., reach the next round).
# round_picks[round] = list of teams we picked to advance.
round_picks = {
    "R64": [t for region in regions_matchups for t in bracket[(region, "R64")]],
    "R32": [t for region in regions_matchups for t in bracket[(region, "R32")]],
    "S16": [t for region in regions_matchups for t in bracket[(region, "S16")]],
    "E8": [bracket[(region, "E8")][0] for region in regions_matchups],
    "F4": list(bracket[("FF", "F4")]),
    "Champ": list(bracket[("FF", "Champ")]),
}

# Actual advancers per round (winners of each round's games).
actual_advancers = defaultdict(set)
for w, l, rnd in all_results:
    actual_advancers[rnd].add(w)

# Bracket scoring: 1, 2, 4, 8, 16, 32 per game in R64 -> Champ.
scoring = {"R64": 1, "R32": 2, "S16": 4, "E8": 8, "F4": 16, "Champ": 32}

# -- Per-round game-level analysis --------------------------------------------
print("=" * 80)
print("FULL POSTMORTEM: 2026 NCAA TOURNAMENT (67 GAMES)")
print("=" * 80)

per_round_stats = {}

for rnd in ["R64", "R32", "S16", "E8", "F4", "Champ"]:
    games = [(w, l) for w, l, r in all_results if r == rnd]
    correct = wrong = 0
    ll_sum = 0.0
    expected_wins = 0.0
    lucky = []
    unlucky = []

    for winner, loser in games:
        p_winner = get_prob(teams[winner], teams[loser])
        our_pick = chalk_pick(winner, loser)
        p_pick = max(p_winner, 1 - p_winner)
        got_it = our_pick == winner

        if got_it:
            correct += 1
        else:
            wrong += 1

        expected_wins += p_pick
        ll_sum += -math.log(max(min(p_winner, 0.999), 0.001))

        if got_it and p_pick < 0.58:
            lucky.append((winner, loser, p_pick))
        elif not got_it and p_pick > 0.60:
            unlucky.append((our_pick, winner, p_pick))

    n = len(games)
    avg_ll = ll_sum / n if n else 0.0
    per_round_stats[rnd] = {
        "n": n,
        "correct": correct,
        "expected": expected_wins,
        "log_loss": avg_ll,
    }

    print(f"\n{rnd:6s}: {correct}/{n} ({correct/n:.1%}) | Log loss: {avg_ll:.3f}"
          f" | Expected {expected_wins:.1f}, got {correct}, "
          f"delta {correct - expected_wins:+.1f}")

    if unlucky:
        print(f"  HIGH-CONFIDENCE BUSTS:")
        for pick, actual, p in sorted(unlucky, key=lambda x: -x[2]):
            print(f"    Picked {pick:18s} ({p:.1%}) but {actual:18s} won")
    if lucky:
        print(f"  CLOSE CALLS WE GOT RIGHT:")
        for w, l, p in sorted(lucky, key=lambda x: x[2]):
            print(f"    {w:18s} over {l:18s} ({p:.1%})")

# -- Overall ------------------------------------------------------------------
all_correct = sum(1 for w, l, r in all_results if chalk_pick(w, l) == w)
all_n = len(all_results)
all_expected = 0.0
all_ll = 0.0
for w, l, r in all_results:
    p = get_prob(teams[w], teams[l])
    all_expected += max(p, 1 - p)
    all_ll += -math.log(max(min(p, 0.999), 0.001))

print(f"\n{'=' * 80}")
print(f"OVERALL: {all_correct}/{all_n} ({all_correct/all_n:.1%}) "
      f"| Log loss: {all_ll/all_n:.3f}")
print(f"  Expected {all_expected:.1f} correct, got {all_correct}, "
      f"delta {all_correct - all_expected:+.1f}")
print(f"  CV log loss reference: 0.456")

# -- Bracket scoring ----------------------------------------------------------
print(f"\n{'=' * 80}")
print(f"BRACKET POINTS (1/2/4/8/16/32 per game)")
print(f"{'=' * 80}")

total_points = 0
max_points_round = {}
for rnd in ["R64", "R32", "S16", "E8", "F4", "Champ"]:
    actual = actual_advancers[rnd]
    picks = round_picks[rnd]
    correct = sum(1 for t in picks if t in actual)
    pts = correct * scoring[rnd]
    total_points += pts
    max_points_round[rnd] = len(picks) * scoring[rnd]
    print(f"  {rnd:6s}: {correct}/{len(picks)} correct x {scoring[rnd]:>2d} = {pts:>3d} pts"
          f"  (max {max_points_round[rnd]})")

print(f"  {'-' * 50}")
total_max = sum(max_points_round.values())
print(f"  TOTAL: {total_points} / {total_max} pts ({total_points/total_max:.1%})")

# -- Championship-path autopsy ------------------------------------------------
print(f"\n{'=' * 80}")
print(f"CHAMPIONSHIP PATH AUTOPSY")
print(f"{'=' * 80}")
print(f"  Our region winners (F4 picks):")
# Map each E8 winner to its region by membership in the R64 team set.
region_of = {t: r for r in regions_matchups
             for matchup in regions_matchups[r] for t in matchup}
actual_region_winner = {}
for w, l, rnd in all_results:
    if rnd == "E8":
        actual_region_winner[region_of[w]] = w

for region in regions_matchups:
    pick = bracket[(region, "E8")][0]
    actual = actual_region_winner.get(region)
    status = "CORRECT" if pick == actual else f"WRONG (actual: {actual})"
    print(f"    {region:8s}: picked {pick:15s} -> {status}")

print(f"\n  Our F4 game picks (who reaches title game):")
print(f"    East/West:   picked {bracket[('FF', 'F4')][0]:15s}")
print(f"    South/Midwest: picked {bracket[('FF', 'F4')][1]:15s}")
print(f"  Actual title game: Michigan vs Connecticut")

print(f"\n  Our champion pick: {bracket[('FF', 'Champ')][0]}")
print(f"  Actual champion:   Michigan")

# Probabilities the model assigned to the actual champ matchup and final winners
ucm = get_prob(teams["Michigan"], teams["Connecticut"])
print(f"\n  Model's P(Michigan beats UConn head-to-head): {ucm:.1%}")
print(f"  Model's P(Duke beats UConn head-to-head): "
      f"{get_prob(teams['Duke'], teams['Connecticut']):.1%}")
print(f"  Model's P(Duke beats Michigan head-to-head): "
      f"{get_prob(teams['Duke'], teams['Michigan']):.1%}")

# -- Calibration drift summary ------------------------------------------------
print(f"\n{'=' * 80}")
print(f"CALIBRATION BY ROUND (log loss; CV reference = 0.456)")
print(f"{'=' * 80}")
print(f"  {'Round':<8} {'N':>3}  {'Acc':>6}  {'LogLoss':>8}  {'vs CV':>8}")
for rnd in ["R64", "R32", "S16", "E8", "F4", "Champ"]:
    s = per_round_stats[rnd]
    delta = s["log_loss"] - 0.456
    print(f"  {rnd:<8} {s['n']:>3}  {s['correct']/s['n']:>6.1%}  "
          f"{s['log_loss']:>8.3f}  {delta:>+8.3f}")
