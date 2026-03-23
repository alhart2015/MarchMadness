"""Build alternate 2026 brackets using game-specific Vegas lines.

Approach:
  v0 (baseline): Our original chalk bracket (model pairwise probs only)
  v1 (R64 lines): Blend model probs with R64 game-specific Vegas lines
                   (info available pre-tournament) + calibration shrinkage for R32+
  v2 (all lines): Blend model probs with game-specific lines for ALL rounds
                   (uses info NOT available pre-tournament - hindsight test)

For each version, pick chalk (higher prob side) and score against actuals.
"""
import json
import math
from scipy.stats import norm

SIGMA = 11.0  # spread-to-probability conversion parameter for CBB

with open("output/pairwise_probs.json") as f:
    probs = json.load(f)


def get_model_prob(id_a, id_b):
    """P(team_a beats team_b) from our XGBoost model."""
    key = f"{min(id_a,id_b)}_{max(id_a,id_b)}"
    p = probs.get(key, 0.5)
    return p if id_a < id_b else 1 - p


def spread_to_prob(spread):
    """Convert a point spread to win probability for the favorite.
    spread > 0 means the 'home' (first listed) team is favored.
    Returns P(home team wins).
    """
    return norm.cdf(spread / SIGMA)


def blend(model_p, vegas_p, vegas_weight=0.5):
    """Blend model and Vegas probabilities."""
    return (1 - vegas_weight) * model_p + vegas_weight * vegas_p


def shrink_toward_50(p, factor=0.15):
    """Shrink a probability toward 0.5 to reduce overconfidence.
    factor=0 means no shrinkage, factor=1 means everything becomes 0.5.
    """
    return p * (1 - factor) + 0.5 * factor


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

# R64 game-specific Vegas lines (from pre-tournament)
# Format: (home_team, road_team, line) where line > 0 = home favored
# home = higher seed in neutral site tournament games
r64_lines = {
    # Thursday 03/19
    ("Duke", "Siena"): 28, ("Ohio St.", "TCU"): 2,
    ("Louisville", "South Florida"): 4.5, ("Michigan St.", "North Dakota St."): 16,
    ("Georgia", "Saint Louis"): 2, ("Gonzaga", "Kennesaw St."): 21,
    ("Houston", "Idaho"): 23.5, ("Illinois", "Penn"): 25.5,
    ("Michigan", "Howard"): 31, ("Nebraska", "Troy"): 13.5,
    ("North Carolina", "VCU"): 2.5, ("Saint Marys", "Texas A&M"): 3,
    ("Vanderbilt", "McNeese St."): 12, ("Wisconsin", "High Point"): 10,
    ("Arkansas", "Hawaii"): 15, ("BYU", "Texas"): 2.5,
    # Friday 03/20
    ("Arizona", "LIU Brooklyn"): 31, ("Villanova", "Utah St."): -2,
    ("Clemson", "Iowa"): -1.5, ("Connecticut", "Furman"): 20.5,
    ("Florida", "Prairie View"): 35.5, ("Iowa St.", "Tennessee St."): 25,
    ("Kansas", "Cal Baptist"): 14, ("Kentucky", "Santa Clara"): 3,
    ("Miami FL", "Missouri"): 1, ("Purdue", "Queens"): 25.5,
    ("St. Johns", "Northern Iowa"): 10.5, ("Tennessee", "Miami OH"): 12,
    ("Texas Tech", "Akron"): 7.5, ("UCLA", "UCF"): 5.5,
    ("Virginia", "Wright St."): 18, ("Alabama", "Hofstra"): 12,
}

# R32 game-specific Vegas lines (NOT available pre-tournament)
r32_lines = {
    # Saturday 03/21
    ("Duke", "TCU"): 12.5, ("Nebraska", "Vanderbilt"): -1.5,
    ("Illinois", "VCU"): 11, ("Houston", "Texas A&M"): 10,
    ("Arkansas", "High Point"): 12, ("Gonzaga", "Texas"): 6.5,
    ("Michigan", "Saint Louis"): 12.5, ("Michigan St.", "Louisville"): 5,
    # Sunday 03/22 (from ESPN data)
    ("Alabama", "Texas Tech"): 3.5, ("Connecticut", "UCLA"): 6,
    ("Arizona", "Utah St."): 10, ("Florida", "Iowa"): 4,
    ("Tennessee", "Virginia"): -2, ("St. Johns", "Kansas"): 1.5,
    ("Iowa St.", "Kentucky"): 8, ("Purdue", "Miami FL"): 5,
}

# All actual results
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
    # R32
    ("Duke", "TCU", "R32"), ("Nebraska", "Vanderbilt", "R32"),
    ("Illinois", "VCU", "R32"), ("Houston", "Texas A&M", "R32"),
    ("Arkansas", "High Point", "R32"), ("Texas", "Gonzaga", "R32"),
    ("Michigan", "Saint Louis", "R32"), ("Michigan St.", "Louisville", "R32"),
    ("Alabama", "Texas Tech", "R32"), ("Connecticut", "UCLA", "R32"),
    ("Arizona", "Utah St.", "R32"), ("Iowa", "Florida", "R32"),
    ("Tennessee", "Virginia", "R32"), ("St. Johns", "Kansas", "R32"),
    ("Iowa St.", "Kentucky", "R32"), ("Purdue", "Miami FL", "R32"),
]

# Bracket structure
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


def get_r64_vegas_prob(a, b):
    """Get Vegas implied probability for R64 game. Returns P(a wins)."""
    if (a, b) in r64_lines:
        return spread_to_prob(r64_lines[(a, b)])
    elif (b, a) in r64_lines:
        return 1 - spread_to_prob(r64_lines[(b, a)])
    return None


def get_r32_vegas_prob(a, b):
    """Get Vegas implied probability for R32 game. Returns P(a wins)."""
    if (a, b) in r32_lines:
        return spread_to_prob(r32_lines[(a, b)])
    elif (b, a) in r32_lines:
        return 1 - spread_to_prob(r32_lines[(b, a)])
    return None


def build_bracket(prob_fn):
    """Build a full chalk bracket using the given probability function.
    prob_fn(a, b) -> P(a beats b)
    Returns dict of picks per round.
    """
    bracket = {}
    for region, matchups in regions_matchups.items():
        r64w = []
        for a, b in matchups:
            p = prob_fn(a, b)
            r64w.append(a if p >= 0.5 else b)
        bracket[(region, "R64")] = r64w

        r32w = []
        for i in range(0, 8, 2):
            p = prob_fn(r64w[i], r64w[i + 1])
            r32w.append(r64w[i] if p >= 0.5 else r64w[i + 1])
        bracket[(region, "R32")] = r32w

        s16w = []
        for i in range(0, 4, 2):
            p = prob_fn(r32w[i], r32w[i + 1])
            s16w.append(r32w[i] if p >= 0.5 else r32w[i + 1])
        bracket[(region, "S16")] = s16w

        e8w_p = prob_fn(s16w[0], s16w[1])
        bracket[(region, "E8")] = [s16w[0] if e8w_p >= 0.5 else s16w[1]]

    e8s = [bracket[(r, "E8")][0] for r in ["East", "West", "South", "Midwest"]]
    p1 = prob_fn(e8s[0], e8s[1])
    f4_1 = e8s[0] if p1 >= 0.5 else e8s[1]
    p2 = prob_fn(e8s[2], e8s[3])
    f4_2 = e8s[2] if p2 >= 0.5 else e8s[3]
    bracket[("FF", "F4")] = [f4_1, f4_2]

    pc = prob_fn(f4_1, f4_2)
    bracket[("FF", "Champ")] = [f4_1 if pc >= 0.5 else f4_2]

    return bracket


def score_bracket(bracket):
    """Score a bracket against actual results. Returns (r64_pts, r32_pts, details)."""
    # R64: did we pick the right winner for each matchup?
    r64_correct = 0
    r64_details = []
    for w, l, rnd in all_results:
        if rnd != "R64":
            continue
        # Find which matchup this corresponds to
        for region, matchups in regions_matchups.items():
            for i, (a, b) in enumerate(matchups):
                if (w == a and l == b) or (w == b and l == a):
                    our_pick = bracket[(region, "R64")][i]
                    got_it = (our_pick == w)
                    if got_it:
                        r64_correct += 1
                    r64_details.append((w, l, our_pick, got_it))

    # R32: did we pick the right team to advance to S16?
    s16_actual = set(w for w, l, rnd in all_results if rnd == "R32")
    r32_correct = 0
    r32_details = []
    for region in ["East", "West", "South", "Midwest"]:
        for pick in bracket[(region, "R32")]:
            got_it = pick in s16_actual
            if got_it:
                r32_correct += 1
            r32_details.append((pick, got_it))

    return r64_correct, r32_correct, r64_details, r32_details


# ── v0: Original model-only bracket ──────────────────────────────────────────
def v0_prob(a, b):
    return get_model_prob(teams[a], teams[b])

# ── v1: Blend model + R64 Vegas lines + R32 calibration shrinkage ────────────
def v1_prob(a, b):
    model_p = get_model_prob(teams[a], teams[b])
    vegas_p = get_r64_vegas_prob(a, b)
    if vegas_p is not None:
        # Blend model and Vegas 50/50 for R64 games
        return blend(model_p, vegas_p, vegas_weight=0.5)
    else:
        # Non-R64 game: shrink model prob toward 50% to reduce overconfidence
        return shrink_toward_50(model_p, factor=0.15)

# ── v2: Blend model + ALL game-specific lines (hindsight) ───────────────────
def v2_prob(a, b):
    model_p = get_model_prob(teams[a], teams[b])
    # Try R32 lines first (more specific), then R64
    vegas_p = get_r32_vegas_prob(a, b)
    if vegas_p is None:
        vegas_p = get_r64_vegas_prob(a, b)
    if vegas_p is not None:
        return blend(model_p, vegas_p, vegas_weight=0.5)
    else:
        return shrink_toward_50(model_p, factor=0.15)


# ── v3: Vegas-only (no model at all) ────────────────────────────────────────
def v3_prob(a, b):
    vegas_p = get_r32_vegas_prob(a, b)
    if vegas_p is None:
        vegas_p = get_r64_vegas_prob(a, b)
    if vegas_p is not None:
        return vegas_p
    else:
        # Fallback to model if no line available
        return get_model_prob(teams[a], teams[b])


versions = [
    ("v0: Model only (our bracket)", v0_prob),
    ("v1: Model + R64 lines + shrinkage", v1_prob),
    ("v2: Model + ALL lines (hindsight)", v2_prob),
    ("v3: Vegas lines only (hindsight)", v3_prob),
]

print("=" * 85)
print("ALTERNATE BRACKET COMPARISON")
print("=" * 85)

for name, prob_fn in versions:
    b = build_bracket(prob_fn)
    r64_c, r32_c, r64_d, r32_d = score_bracket(b)

    r64_pts = r64_c * 10
    r32_pts = r32_c * 20
    total = r64_pts + r32_pts

    print(f"\n{'─' * 85}".encode('ascii', 'replace').decode())
    print(f"  {name}")
    print(f"  R64: {r64_c}/32 ({r64_c*10} pts) | R32: {r32_c}/16 ({r32_c*20} pts) | Total: {total} pts")

    # Show where this bracket differs from v0
    b0 = build_bracket(v0_prob)
    diffs_r64 = []
    for region, matchups in regions_matchups.items():
        for i, (a, bp) in enumerate(matchups):
            p0 = b0[(region, "R64")][i]
            p1 = b[(region, "R64")][i]
            if p0 != p1:
                actual_winner = None
                for w, l, rnd in all_results:
                    if rnd == "R64" and ((w == a and l == bp) or (w == bp and l == a)):
                        actual_winner = w
                v0_right = (p0 == actual_winner)
                v1_right = (p1 == actual_winner)
                delta = (10 if v1_right else 0) - (10 if v0_right else 0)
                diffs_r64.append((p0, p1, actual_winner, delta))

    diffs_r32 = []
    s16_actual = set(w for w, l, rnd in all_results if rnd == "R32")
    for region in ["East", "West", "South", "Midwest"]:
        for i in range(4):
            p0 = b0[(region, "R32")][i]
            p1 = b[(region, "R32")][i]
            if p0 != p1:
                v0_right = p0 in s16_actual
                v1_right = p1 in s16_actual
                delta = (20 if v1_right else 0) - (20 if v0_right else 0)
                diffs_r32.append((p0, p1, v0_right, v1_right, delta))

    if diffs_r64 or diffs_r32:
        print(f"\n  Differences from original bracket:")
        net = 0
        for old, new, actual, delta in diffs_r64:
            marker = "+" if delta > 0 else "-" if delta < 0 else "="
            print(f"    R64: {old:15s} -> {new:15s} (actual: {actual:15s}) [{marker}{abs(delta)} pts]")
            net += delta
        for old, new, v0r, v1r, delta in diffs_r32:
            marker = "+" if delta > 0 else "-" if delta < 0 else "="
            status = f"{'correct' if v1r else 'wrong':>7s}"
            print(f"    R32: {old:15s} -> {new:15s} ({status}) [{marker}{abs(delta)} pts]")
            net += delta
        print(f"    Net impact: {net:+d} pts")
    else:
        print(f"\n  (identical to original bracket)")

# Also show the champion pick for each version
print(f"\n{'=' * 85}")
print(f"CHAMPION PICKS:")
print(f"{'=' * 85}")
for name, prob_fn in versions:
    b = build_bracket(prob_fn)
    champ = b[("FF", "Champ")][0]
    f4 = b[("FF", "F4")]
    print(f"  {name}")
    print(f"    F4: {f4[0]} vs {f4[1]} -> Champion: {champ}")
