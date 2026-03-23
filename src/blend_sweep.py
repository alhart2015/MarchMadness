"""Sweep Vegas blend weights to find optimal mix for 2026 bracket."""
import json
from scipy.stats import norm

SIGMA = 11.0

with open("output/pairwise_probs.json") as f:
    probs = json.load(f)

def get_model_prob(id_a, id_b):
    key = f"{min(id_a,id_b)}_{max(id_a,id_b)}"
    p = probs.get(key, 0.5)
    return p if id_a < id_b else 1 - p

def spread_to_prob(spread):
    return norm.cdf(spread / SIGMA)

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

r64_lines = {
    ("Duke", "Siena"): 28, ("Ohio St.", "TCU"): 2,
    ("Louisville", "South Florida"): 4.5, ("Michigan St.", "North Dakota St."): 16,
    ("Georgia", "Saint Louis"): 2, ("Gonzaga", "Kennesaw St."): 21,
    ("Houston", "Idaho"): 23.5, ("Illinois", "Penn"): 25.5,
    ("Michigan", "Howard"): 31, ("Nebraska", "Troy"): 13.5,
    ("North Carolina", "VCU"): 2.5, ("Saint Marys", "Texas A&M"): 3,
    ("Vanderbilt", "McNeese St."): 12, ("Wisconsin", "High Point"): 10,
    ("Arkansas", "Hawaii"): 15, ("BYU", "Texas"): 2.5,
    ("Arizona", "LIU Brooklyn"): 31, ("Villanova", "Utah St."): -2,
    ("Clemson", "Iowa"): -1.5, ("Connecticut", "Furman"): 20.5,
    ("Florida", "Prairie View"): 35.5, ("Iowa St.", "Tennessee St."): 25,
    ("Kansas", "Cal Baptist"): 14, ("Kentucky", "Santa Clara"): 3,
    ("Miami FL", "Missouri"): 1, ("Purdue", "Queens"): 25.5,
    ("St. Johns", "Northern Iowa"): 10.5, ("Tennessee", "Miami OH"): 12,
    ("Texas Tech", "Akron"): 7.5, ("UCLA", "UCF"): 5.5,
    ("Virginia", "Wright St."): 18, ("Alabama", "Hofstra"): 12,
}

r32_lines = {
    ("Duke", "TCU"): 12.5, ("Nebraska", "Vanderbilt"): -1.5,
    ("Illinois", "VCU"): 11, ("Houston", "Texas A&M"): 10,
    ("Arkansas", "High Point"): 12, ("Gonzaga", "Texas"): 6.5,
    ("Michigan", "Saint Louis"): 12.5, ("Michigan St.", "Louisville"): 5,
    ("Alabama", "Texas Tech"): 3.5, ("Connecticut", "UCLA"): 6,
    ("Arizona", "Utah St."): 10, ("Florida", "Iowa"): 4,
    ("Tennessee", "Virginia"): -2, ("St. Johns", "Kansas"): 1.5,
    ("Iowa St.", "Kentucky"): 8, ("Purdue", "Miami FL"): 5,
}

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

all_results = [
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
    ("Duke", "TCU", "R32"), ("Nebraska", "Vanderbilt", "R32"),
    ("Illinois", "VCU", "R32"), ("Houston", "Texas A&M", "R32"),
    ("Arkansas", "High Point", "R32"), ("Texas", "Gonzaga", "R32"),
    ("Michigan", "Saint Louis", "R32"), ("Michigan St.", "Louisville", "R32"),
    ("Alabama", "Texas Tech", "R32"), ("Connecticut", "UCLA", "R32"),
    ("Arizona", "Utah St.", "R32"), ("Iowa", "Florida", "R32"),
    ("Tennessee", "Virginia", "R32"), ("St. Johns", "Kansas", "R32"),
    ("Iowa St.", "Kentucky", "R32"), ("Purdue", "Miami FL", "R32"),
]


def get_vegas_prob(a, b):
    for lines in [r32_lines, r64_lines]:
        if (a, b) in lines:
            return spread_to_prob(lines[(a, b)])
        elif (b, a) in lines:
            return 1 - spread_to_prob(lines[(b, a)])
    return None


def build_and_score(r64_vw, r32_vw, shrinkage):
    """Build bracket with given Vegas weights and score it."""

    def prob_fn(a, b):
        model_p = get_model_prob(teams[a], teams[b])
        vegas_p = get_vegas_prob(a, b)
        if vegas_p is not None:
            # Determine if this is an R64 or R32 game
            is_r64 = (a, b) in r64_lines or (b, a) in r64_lines
            vw = r64_vw if is_r64 else r32_vw
            blended = (1 - vw) * model_p + vw * vegas_p
            return blended
        else:
            return model_p * (1 - shrinkage) + 0.5 * shrinkage

    bracket = {}
    for region, matchups in regions_matchups.items():
        r64w = []
        for a, b in matchups:
            p = prob_fn(a, b)
            r64w.append(a if p >= 0.5 else b)
        bracket[(region, "R64")] = r64w
        r32w = [r64w[i] if prob_fn(r64w[i], r64w[i+1]) >= 0.5 else r64w[i+1]
                for i in range(0, 8, 2)]
        bracket[(region, "R32")] = r32w

    # Score
    r64_c = 0
    for w, l, rnd in all_results:
        if rnd != "R64":
            continue
        for region, matchups in regions_matchups.items():
            for i, (a, b) in enumerate(matchups):
                if (w == a and l == b) or (w == b and l == a):
                    if bracket[(region, "R64")][i] == w:
                        r64_c += 1

    s16_actual = set(w for w, l, rnd in all_results if rnd == "R32")
    r32_c = sum(1 for region in regions_matchups
                for pick in bracket[(region, "R32")]
                if pick in s16_actual)

    return r64_c, r32_c, r64_c * 10 + r32_c * 20


# Sweep
print("VEGAS WEIGHT SWEEP (R64 weight, R32 weight, shrinkage for unlined games)")
print("=" * 80)
print(f"{'R64_vw':>7s} {'R32_vw':>7s} {'Shrink':>7s} {'R64':>5s} {'R32':>5s} {'Total':>6s}")
print("-" * 80)

best = (0, 0, 0, 0, 0, 0)
results = []

for r64_vw_pct in range(0, 105, 5):
    r64_vw = r64_vw_pct / 100
    for r32_vw_pct in range(0, 105, 5):
        r32_vw = r32_vw_pct / 100
        for shrink_pct in [0, 5, 10, 15, 20]:
            shrink = shrink_pct / 100
            r64_c, r32_c, total = build_and_score(r64_vw, r32_vw, shrink)
            results.append((r64_vw, r32_vw, shrink, r64_c, r32_c, total))
            if total > best[5] or (total == best[5] and r64_c + r32_c > best[3] + best[4]):
                best = (r64_vw, r32_vw, shrink, r64_c, r32_c, total)

# Print top results
results.sort(key=lambda x: (-x[5], -(x[3] + x[4])))
seen = set()
count = 0
for r64_vw, r32_vw, shrink, r64_c, r32_c, total in results:
    if total not in seen or count < 20:
        print(f"  {r64_vw:5.0%}   {r32_vw:5.0%}   {shrink:5.0%}    {r64_c:2d}    {r32_c:2d}    {total:4d}")
        seen.add(total)
        count += 1
    if count >= 20:
        break

print(f"\n{'=' * 80}")
print(f"BEST:  R64 weight={best[0]:.0%}, R32 weight={best[1]:.0%}, shrink={best[2]:.0%}")
print(f"       R64: {best[3]}/32, R32: {best[4]}/16, Total: {best[5]} pts")
print(f"\nOurs:  R64: 25/32, R32: 10/16, Total: 450 pts")
print(f"Delta: {best[5] - 450:+d} pts")

# Show what the best bracket picks differently
print(f"\n{'=' * 80}")
print(f"BEST BRACKET vs OUR BRACKET - Specific differences")
print(f"{'=' * 80}")

def build_full(r64_vw, r32_vw, shrinkage):
    def prob_fn(a, b):
        model_p = get_model_prob(teams[a], teams[b])
        vegas_p = get_vegas_prob(a, b)
        if vegas_p is not None:
            is_r64 = (a, b) in r64_lines or (b, a) in r64_lines
            vw = r64_vw if is_r64 else r32_vw
            return (1 - vw) * model_p + vw * vegas_p
        return model_p * (1 - shrinkage) + 0.5 * shrinkage

    bracket = {}
    for region, matchups in regions_matchups.items():
        r64w = [a if prob_fn(a, b) >= 0.5 else b for a, b in matchups]
        bracket[(region, "R64")] = r64w
        r32w = [r64w[i] if prob_fn(r64w[i], r64w[i+1]) >= 0.5 else r64w[i+1]
                for i in range(0, 8, 2)]
        bracket[(region, "R32")] = r32w
    return bracket

ours = build_full(0, 0, 0)
best_b = build_full(best[0], best[1], best[2])
s16_actual = set(w for w, l, rnd in all_results if rnd == "R32")

for region, matchups in regions_matchups.items():
    for i, (a, b) in enumerate(matchups):
        o = ours[(region, "R64")][i]
        n = best_b[(region, "R64")][i]
        if o != n:
            actual = None
            for w, l, rnd in all_results:
                if rnd == "R64" and ((w == a and l == b) or (w == b and l == a)):
                    actual = w
            was_right = "RIGHT" if n == actual else "WRONG"
            model_p = get_model_prob(teams[a], teams[b])
            vegas_p = get_vegas_prob(a, b) or 0
            print(f"  R64 {region}: {o:15s} -> {n:15s} (actual: {actual:15s}) [{was_right}]")
            print(f"       Model: {model_p:.1%} for {a}, Vegas: {vegas_p:.1%} for {a}")

    for i in range(4):
        o = ours[(region, "R32")][i]
        n = best_b[(region, "R32")][i]
        if o != n:
            was_right = "RIGHT" if n in s16_actual else "WRONG"
            print(f"  R32 {region}: {o:15s} -> {n:15s} [{was_right}]")
