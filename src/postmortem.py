"""Full tournament postmortem through R32."""
import json
import math

with open("output/pairwise_probs.json") as f:
    probs = json.load(f)

def get_prob(id_a, id_b):
    key = f"{min(id_a,id_b)}_{max(id_a,id_b)}"
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

# All actual results: (winner, loser, round)
all_results = [
    # R64
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
    # R32 Saturday
    ("Duke", "TCU", "R32"), ("Nebraska", "Vanderbilt", "R32"),
    ("Illinois", "VCU", "R32"), ("Houston", "Texas A&M", "R32"),
    ("Arkansas", "High Point", "R32"), ("Texas", "Gonzaga", "R32"),
    ("Michigan", "Saint Louis", "R32"), ("Michigan St.", "Louisville", "R32"),
    # R32 Sunday
    ("Alabama", "Texas Tech", "R32"), ("Connecticut", "UCLA", "R32"),
    ("Arizona", "Utah St.", "R32"), ("Iowa", "Florida", "R32"),
    ("Tennessee", "Virginia", "R32"), ("St. Johns", "Kansas", "R32"),
    ("Iowa St.", "Kentucky", "R32"), ("Purdue", "Miami FL", "R32"),
]

# Build our chalk bracket
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

bracket = {}
for region, matchups in regions_matchups.items():
    r64w = [chalk_pick(a, b) for a, b in matchups]
    bracket[(region, "R64")] = r64w
    r32w = [chalk_pick(r64w[i], r64w[i+1]) for i in range(0, 8, 2)]
    bracket[(region, "R32")] = r32w
    s16w = [chalk_pick(r32w[i], r32w[i+1]) for i in range(0, 4, 2)]
    bracket[(region, "S16")] = s16w
    e8w = chalk_pick(s16w[0], s16w[1])
    bracket[(region, "E8")] = [e8w]

e8s = [bracket[(r, "E8")][0] for r in ["East", "West", "South", "Midwest"]]
f4_1 = chalk_pick(e8s[0], e8s[1])
f4_2 = chalk_pick(e8s[2], e8s[3])
bracket[("FF", "F4")] = [f4_1, f4_2]
bracket[("FF", "Champ")] = [chalk_pick(f4_1, f4_2)]

# Score each game
print("=" * 80)
print("POSTMORTEM: MODEL PERFORMANCE THROUGH R32 (48 GAMES)")
print("=" * 80)

for rnd in ["R64", "R32"]:
    games = [(w, l) for w, l, r in all_results if r == rnd]
    correct = 0
    wrong = 0
    ll_sum = 0
    expected_wins = 0
    lucky = []
    unlucky = []

    for winner, loser in games:
        p_winner = get_prob(teams[winner], teams[loser])
        p_pick = max(p_winner, 1 - p_winner)  # prob of our chalk pick
        our_pick = chalk_pick(winner, loser)
        got_it = (our_pick == winner)

        if got_it:
            correct += 1
        else:
            wrong += 1

        expected_wins += p_pick

        # Log loss (from winner's perspective)
        ll_sum += -math.log(max(min(p_winner, 0.999), 0.001))

        # Lucky = we picked correctly but our pick was < 60% favorite
        # Unlucky = we picked wrong and our pick was > 60% favorite
        if got_it and p_pick < 0.58:
            lucky.append((winner, loser, p_winner if our_pick == winner else 1 - p_winner))
        elif not got_it and p_pick > 0.60:
            unlucky.append((our_pick, winner, p_pick))

    n = len(games)
    print(f"\n{rnd}: {correct}/{n} ({correct/n:.1%}) | Log loss: {ll_sum/n:.3f}")
    print(f"  Expected correct: {expected_wins:.1f} | Actual: {correct} | Delta: {correct - expected_wins:+.1f}")

    if lucky:
        print(f"\n  LUCKY wins (close calls we got right):")
        for w, l, p in sorted(lucky, key=lambda x: x[2]):
            print(f"    {w:15s} over {l:15s} (our confidence: {p:.1%})")

    if unlucky:
        print(f"\n  UNLUCKY losses (confident picks that busted):")
        for pick, actual, p in sorted(unlucky, key=lambda x: -x[2]):
            print(f"    Picked {pick:15s} ({p:.1%}) but {actual:15s} won")

# Overall
print(f"\n{'=' * 80}")
all_correct = sum(1 for w, l, r in all_results if chalk_pick(w, l) == w)
all_n = len(all_results)
all_expected = 0
all_ll = 0
for w, l, r in all_results:
    p = get_prob(teams[w], teams[l])
    pp = max(p, 1-p)
    all_expected += pp
    all_ll += -math.log(max(min(p, 0.999), 0.001))

print(f"OVERALL: {all_correct}/{all_n} ({all_correct/all_n:.1%}) | Log loss: {all_ll/all_n:.3f}")
print(f"  Expected correct: {all_expected:.1f} | Actual: {all_correct} | Delta: {all_correct - all_expected:+.1f}")

# Bracket points
r64_pts = sum(10 for w, l, r in all_results if r == "R64" and chalk_pick(w, l) == w)

# For R32 bracket scoring: did we pick the correct team to advance to S16?
# We need to check if the R32 winner matches our S16 pick for that slot
s16_actual = set()
r32_bracket_pts = 0
for w, l, r in all_results:
    if r == "R32":
        s16_actual.add(w)

# Check our R32 bracket picks (the teams we picked to reach S16)
our_s16_picks = []
for region in ["East", "West", "South", "Midwest"]:
    our_s16_picks.extend(bracket[(region, "R32")])

r32_correct = sum(1 for t in our_s16_picks if t in s16_actual)
r32_bracket_pts = r32_correct * 20

print(f"\nBRACKET SCORING:")
print(f"  R64: {r64_pts // 10}/32 correct x 10 = {r64_pts} pts")
print(f"  R32: {r32_correct}/16 correct x 20 = {r32_bracket_pts} pts")
print(f"  Total: {r64_pts + r32_bracket_pts} pts")

# Sweet 16 field vs our picks
print(f"\n{'=' * 80}")
print(f"SWEET 16 FIELD vs OUR BRACKET")
print(f"{'=' * 80}")

actual_s16 = {
    "East": ["Duke", "St. Johns", "Michigan St.", "Connecticut"],
    "West": ["Arizona", "Arkansas", "Texas", "Purdue"],
    "South": ["Iowa", "Nebraska", "Illinois", "Houston"],
    "Midwest": ["Michigan", "Alabama", "Tennessee", "Iowa St."],
}

for region in ["East", "West", "South", "Midwest"]:
    our = bracket[(region, "R32")]
    actual = actual_s16[region]
    our_e8 = bracket[(region, "E8")][0]
    print(f"\n  {region}:")
    print(f"    Our S16:    {our}")
    print(f"    Actual S16: {actual}")
    matches = sum(1 for t in our if t in actual)
    print(f"    Correct: {matches}/4")
    alive = "ALIVE" if our_e8 in actual else "DEAD"
    print(f"    Our E8 pick ({our_e8}): {alive}")

# Future bracket status
print(f"\n{'=' * 80}")
print(f"REMAINING BRACKET PICKS STATUS")
print(f"{'=' * 80}")

all_alive = set()
for region in actual_s16:
    all_alive.update(actual_s16[region])

pts_map = {"S16": 40, "E8": 80, "F4": 160, "Champ": 320}
alive_pts = 0
dead_pts = 0

for rnd, pts in pts_map.items():
    if rnd in ("F4", "Champ"):
        picks = bracket[("FF", rnd)]
    else:
        picks = []
        for region in ["East", "West", "South", "Midwest"]:
            if (region, rnd) in bracket:
                picks.extend(bracket[(region, rnd)])

    a = [t for t in picks if t in all_alive]
    d = [t for t in picks if t not in all_alive]
    round_alive = len(a) * pts
    round_dead = len(d) * pts
    alive_pts += round_alive
    dead_pts += round_dead

    if a or d:
        print(f"\n  {rnd} ({pts} pts each):")
        if a:
            print(f"    ALIVE: {', '.join(a)} ({round_alive} pts)")
        if d:
            print(f"    DEAD:  {', '.join(d)} ({round_dead} pts)")

print(f"\n  Points still achievable: {alive_pts}")
print(f"  Points already lost:     {dead_pts}")
print(f"  Current score:           {r64_pts + r32_bracket_pts}")
print(f"  Max possible:            {r64_pts + r32_bracket_pts + alive_pts}")
print(f"  Theoretical max:         {r64_pts + r32_bracket_pts + alive_pts + dead_pts + r64_pts + r32_bracket_pts}")

# Luck analysis summary
print(f"\n{'=' * 80}")
print(f"LUCK ANALYSIS")
print(f"{'=' * 80}")

# For every game, compute P(our bracket pick is correct)
# Sum those = expected bracket correct picks
# Compare to actual
r64_exp = 0
r64_act = 0
for w, l, r in all_results:
    if r != "R64":
        continue
    our = chalk_pick(w, l)
    p = get_prob(teams[our], teams[w if our != w else l])
    r64_exp += p
    if our == w:
        r64_act += 1

# R32 is trickier - expected S16 picks correct
# Use advancement probability R2 from the model output
print(f"\n  R64: Expected {r64_exp:.1f} correct, got {r64_act} ({r64_act - r64_exp:+.1f})")

# Overall narrative
delta = all_correct - all_expected
if delta > 1.5:
    verdict = "LUCKY - outperforming the model's expectations"
elif delta < -1.5:
    verdict = "UNLUCKY - underperforming the model's expectations"
else:
    verdict = "NEUTRAL - tracking expected value closely"
print(f"\n  Game-level verdict: {verdict}")
print(f"  (Expected {all_expected:.1f} correct picks, got {all_correct}, delta {delta:+.1f})")
