"""Trace our chalk bracket picks vs actual tournament results."""
import json

with open("output/pairwise_probs.json") as f:
    probs = json.load(f)


def get_prob(id_a, id_b):
    key = f"{min(id_a,id_b)}_{max(id_a,id_b)}"
    p = probs.get(key)
    if p is None:
        return None
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

# R64 matchups in bracket order (1v16, 8v9, 5v12, 4v13, 6v11, 3v14, 7v10, 2v15)
r64_matchups = {
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

# Actual R64 winners
r64_winners = {
    "Duke", "TCU", "St. Johns", "Kansas", "Louisville", "Michigan St.", "UCLA", "Connecticut",
    "Arizona", "Utah St.", "High Point", "Arkansas", "Texas", "Gonzaga", "Miami FL", "Purdue",
    "Florida", "Iowa", "Vanderbilt", "Nebraska", "VCU", "Illinois", "Texas A&M", "Houston",
    "Michigan", "Saint Louis", "Texas Tech", "Alabama", "Tennessee", "Virginia", "Kentucky", "Iowa St.",
}

# Actual R32 winners (Saturday only, Sunday TBD)
r32_winners = {
    "Duke", "Nebraska", "Illinois", "Houston", "Arkansas", "Texas", "Michigan", "Michigan St.",
}


def chalk_pick(a, b):
    p = get_prob(teams[a], teams[b])
    return a if p >= 0.5 else b


def actual_winner_of(a, b):
    for w in list(r32_winners) + list(r64_winners):
        if w == a or w == b:
            if w in r32_winners and (a in r64_winners and b in r64_winners):
                return w
    return None


# Build our chalk bracket
print("=" * 80)
print("BRACKET SCORECARD: Our Chalk Picks vs Reality")
print("=" * 80)

total_correct = {"R64": 0, "R32": 0}
total_wrong = {"R64": 0, "R32": 0}
total_dead = {"R64": 0, "R32": 0}
total_pending = {"R32": 0}

for region in ["East", "West", "South", "Midwest"]:
    print(f"\n{'-' * 80}")
    print(f"  {region.upper()} REGION")
    print(f"{'-' * 80}")

    matchups = r64_matchups[region]

    # R64
    our_r64_picks = []
    print(f"\n  R64:")
    for a, b in matchups:
        pick = chalk_pick(a, b)
        our_r64_picks.append(pick)
        actual = a if a in r64_winners else b
        status = "OK" if pick == actual else "WRONG"
        if status == "OK":
            total_correct["R64"] += 1
        else:
            total_wrong["R64"] += 1
        marker = "  " if status == "OK" else "X "
        print(f"    {marker} {a:15s} vs {b:15s} | Pick: {pick:15s} | Won: {actual:15s}")

    # R32 (pairs: 0+1, 2+3, 4+5, 6+7)
    print(f"\n  R32:")
    our_s16_picks = []
    for i in range(0, 8, 2):
        our_a = our_r64_picks[i]
        our_b = our_r64_picks[i + 1]

        # Actual R32 matchup
        act_a = matchups[i][0] if matchups[i][0] in r64_winners else matchups[i][1]
        act_b = matchups[i + 1][0] if matchups[i + 1][0] in r64_winners else matchups[i + 1][1]

        our_pick = chalk_pick(our_a, our_b)
        our_s16_picks.append(our_pick)

        # Determine status
        act_winner = None
        if act_a in r32_winners:
            act_winner = act_a
        elif act_b in r32_winners:
            act_winner = act_b

        if act_winner:
            if our_pick == act_winner:
                status = "CORRECT"
                total_correct["R32"] += 1
            elif our_pick not in r64_winners:
                status = "DEAD (R64 pick lost)"
                total_dead["R32"] += 1
            else:
                status = "WRONG"
                total_wrong["R32"] += 1
        else:
            if our_pick not in r64_winners:
                status = "DEAD (R64 pick lost)"
                total_dead["R32"] += 1
            else:
                status = "pending"
                total_pending["R32"] += 1

        marker = "  " if "CORRECT" in status or "pending" in status else "X "
        our_matchup = f"{our_a} vs {our_b}"
        act_matchup = f"{act_a} vs {act_b}"
        print(f"    {marker} Bracket: {our_matchup:35s} -> {our_pick:15s}")
        print(f"       Actual:  {act_matchup:35s} -> {str(act_winner or 'TBD'):15s}  [{status}]")

print(f"\n{'=' * 80}")
print(f"SUMMARY")
print(f"{'=' * 80}")
print(f"  R64:  {total_correct['R64']}/32 correct")
print(f"  R32:  {total_correct['R32']} correct, {total_wrong['R32']} wrong, "
      f"{total_dead['R32']} dead from R64, {total_pending['R32']} pending")
s16_alive = total_correct["R32"] + total_pending["R32"]
s16_dead = total_wrong["R32"] + total_dead["R32"]
print(f"\n  Sweet 16 picks: {total_correct['R32']} locked in, {total_pending['R32']} pending, {s16_dead} busted")
