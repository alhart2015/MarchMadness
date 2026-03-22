import json

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

# Build full chalk bracket
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

eliminated = {
    "Siena", "Ohio St.", "Northern Iowa", "Cal Baptist", "South Florida",
    "North Dakota St.", "UCF", "Furman",
    "LIU Brooklyn", "Villanova", "Wisconsin", "Hawaii",
    "BYU", "Kennesaw St.", "Missouri", "Queens",
    "Prairie View", "Clemson", "McNeese St.", "Troy",
    "North Carolina", "Penn", "Saint Marys", "Idaho",
    "Howard", "Georgia", "Akron", "Hofstra",
    "Miami OH", "Wright St.", "Santa Clara", "Tennessee St.",
    "TCU", "Louisville", "High Point", "Gonzaga",
    "Vanderbilt", "VCU", "Texas A&M", "Saint Louis",
}

pts_map = {"R32": 20, "S16": 40, "E8": 80, "F4": 160, "Champ": 320}

# Full bracket from S16 on
print("OUR FULL CHALK BRACKET")
print("=" * 65)
for region in ["East", "West", "South", "Midwest"]:
    s16 = bracket[(region, "S16")]
    e8 = bracket[(region, "E8")]
    s16s = ["ALIVE" if t not in eliminated else "DEAD" for t in s16]
    e8s_str = "ALIVE" if e8[0] not in eliminated else "DEAD"
    print(f"  {region:8s} S16: {s16[0]:15s}[{s16s[0]:5s}]  {s16[1]:15s}[{s16s[1]:5s}]")
    print(f"           E8:  {e8[0]:15s}[{e8s_str:5s}]")

f4 = bracket[("FF", "F4")]
champ = bracket[("FF", "Champ")]
f4s = ["ALIVE" if t not in eliminated else "DEAD" for t in f4]
cs = "ALIVE" if champ[0] not in eliminated else "DEAD"
print(f"\n  Final Four: {f4[0]:15s}[{f4s[0]:5s}]  {f4[1]:15s}[{f4s[1]:5s}]")
print(f"  Champion:   {champ[0]:15s}[{cs:5s}]")

# Points riding on each remaining team
print(f"\n{'=' * 65}")
print("POINTS RIDING ON EACH REMAINING BRACKET PICK")
print("=" * 65)

team_stakes = {}
for rnd, pts in pts_map.items():
    if rnd in ("F4", "Champ"):
        picks = bracket[("FF", rnd)]
    else:
        picks = []
        for region in ["East", "West", "South", "Midwest"]:
            if (region, rnd) in bracket:
                picks.extend(bracket[(region, rnd)])
    for t in picks:
        if t not in eliminated:
            if t not in team_stakes:
                team_stakes[t] = []
            team_stakes[t].append((rnd, pts))

team_totals = [(t, sum(p for _, p in r), r) for t, r in team_stakes.items()]
team_totals.sort(key=lambda x: x[1], reverse=True)

for t, total, rounds in team_totals:
    rnd_str = " + ".join(f"{r}({p})" for r, p in rounds)
    print(f"  {t:15s}: {total:4d} pts  ({rnd_str})")

total_alive_pts = sum(t[1] for t in team_totals)
print(f"\n  Total pts still in play: {total_alive_pts}")
print(f"  Current score:           350")
print(f"  Max possible:            {350 + total_alive_pts}")

# Today's pending games ranked by impact
print(f"\n{'=' * 65}")
print("TODAY'S GAMES RANKED BY BRACKET IMPACT")
print("=" * 65)

pending = [
    ("St. Johns", "Kansas"),
    ("UCLA", "Connecticut"),
    ("Arizona", "Utah St."),
    ("Miami FL", "Purdue"),
    ("Florida", "Iowa"),
    ("Texas Tech", "Alabama"),
    ("Tennessee", "Virginia"),
    ("Kentucky", "Iowa St."),
]

impacts = []
for a, b in pending:
    pick = chalk_pick(a, b)
    other = b if pick == a else a
    p = get_prob(teams[a], teams[b])
    pick_p = p if pick == a else 1 - p
    stakes = team_stakes.get(pick, [])
    total_risk = sum(p for _, p in stakes)
    impacts.append((pick, other, pick_p, total_risk, stakes))

impacts.sort(key=lambda x: x[3], reverse=True)

print(f"\n  {'Our Pick':15s} {'vs':3s} {'Opponent':15s} {'Win%':>5s} {'At Risk':>7s}  Rounds")
print(f"  {'-'*60}")
for pick, other, pick_p, risk, stakes in impacts:
    rnd_str = " + ".join(f"{r}({p})" for r, p in stakes)
    print(f"  {pick:15s} vs  {other:15s} {pick_p:4.0%}  {risk:5d} pts  {rnd_str}")
