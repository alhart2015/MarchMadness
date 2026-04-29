"""Score v1/v2/v3 model versions against actual 2026 tournament results.

Reads three pairwise-probability files (one per version) and the actual
results from src/postmortem_full.py. For each version, reports:
  - Per-round log loss and accuracy on actual matchups
  - Bracket points under the chalk strategy
  - Which busted picks (high-confidence misses) were specific to that version
  - Pairwise disagreements between versions

Inputs:
  output/pairwise_probs_v1.json   (produced by src/enhanced_model.py)
  output/pairwise_probs_v2.json   (produced by src/enhanced_model_v2.py)
  output/pairwise_probs_v3.json   (produced by src/enhanced_model_v3.py)
"""
import json
import math
from pathlib import Path

# Reuse actuals + team map from postmortem_full
from postmortem_full import (
    all_results,
    teams,
    regions_matchups,
)


PROB_FILES = {
    "v1": "output/pairwise_probs_v1.json",
    "v2": "output/pairwise_probs_v2.json",
    "v3": "output/pairwise_probs_v3.json",
}


def load_probs(path):
    with open(path) as f:
        return json.load(f)


def get_prob(probs, id_a, id_b):
    key = f"{min(id_a, id_b)}_{max(id_a, id_b)}"
    p = probs.get(key, 0.5)
    return p if id_a < id_b else 1 - p


def chalk_pick(probs, a, b):
    p = get_prob(probs, teams[a], teams[b])
    return a if p >= 0.5 else b


def build_chalk_bracket(probs):
    bracket = {}
    region_winners = []
    for region, matchups in regions_matchups.items():
        r64w = [chalk_pick(probs, a, b) for a, b in matchups]
        bracket[(region, "R64")] = r64w
        r32w = [chalk_pick(probs, r64w[i], r64w[i + 1]) for i in range(0, 8, 2)]
        bracket[(region, "R32")] = r32w
        s16w = [chalk_pick(probs, r32w[i], r32w[i + 1]) for i in range(0, 4, 2)]
        bracket[(region, "S16")] = s16w
        e8w = chalk_pick(probs, s16w[0], s16w[1])
        bracket[(region, "E8")] = [e8w]
        region_winners.append(e8w)
    f4_l = chalk_pick(probs, region_winners[0], region_winners[1])
    f4_r = chalk_pick(probs, region_winners[2], region_winners[3])
    bracket[("FF", "F4")] = [f4_l, f4_r]
    bracket[("FF", "Champ")] = [chalk_pick(probs, f4_l, f4_r)]
    return bracket


def round_picks_from_bracket(bracket):
    return {
        "R64": [t for r in regions_matchups for t in bracket[(r, "R64")]],
        "R32": [t for r in regions_matchups for t in bracket[(r, "R32")]],
        "S16": [t for r in regions_matchups for t in bracket[(r, "S16")]],
        "E8":  [bracket[(r, "E8")][0] for r in regions_matchups],
        "F4":  list(bracket[("FF", "F4")]),
        "Champ": list(bracket[("FF", "Champ")]),
    }


def actual_advancers_by_round(results):
    out = {}
    for w, _, rnd in results:
        out.setdefault(rnd, set()).add(w)
    return out


SCORING = {"R64": 1, "R32": 2, "S16": 4, "E8": 8, "F4": 16, "Champ": 32}
ROUND_ORDER = ["R64", "R32", "S16", "E8", "F4", "Champ"]


def score_version(label, probs):
    """Return per-round + overall stats for one model version."""
    bracket = build_chalk_bracket(probs)
    picks = round_picks_from_bracket(bracket)
    actual = actual_advancers_by_round(all_results)

    per_round = {}
    bracket_points = {}
    total_pts = 0
    overall_correct = 0
    overall_n = 0
    overall_ll = 0.0

    for rnd in ROUND_ORDER:
        games = [(w, l) for w, l, r in all_results if r == rnd]
        n = len(games)
        correct = 0
        ll = 0.0
        for w, l in games:
            p_w = get_prob(probs, teams[w], teams[l])
            ll += -math.log(max(min(p_w, 0.999), 0.001))
            if chalk_pick(probs, w, l) == w:
                correct += 1
        per_round[rnd] = {
            "n": n,
            "correct": correct,
            "log_loss": ll / n if n else 0.0,
        }
        bp = sum(1 for t in picks[rnd] if t in actual[rnd]) * SCORING[rnd]
        bracket_points[rnd] = bp
        total_pts += bp
        overall_correct += correct
        overall_n += n
        overall_ll += ll

    return {
        "label": label,
        "bracket": bracket,
        "picks": picks,
        "per_round": per_round,
        "bracket_points": bracket_points,
        "total_points": total_pts,
        "overall_acc": overall_correct / overall_n,
        "overall_ll": overall_ll / overall_n,
        "champion": bracket[("FF", "Champ")][0],
        "f4": list(bracket[("FF", "F4")]),
    }


def main():
    versions = {}
    raw_probs = {}
    for label, path in PROB_FILES.items():
        if not Path(path).exists():
            print(f"[skip] {label}: {path} not found")
            continue
        raw_probs[label] = load_probs(path)
        versions[label] = score_version(label, raw_probs[label])

    if not versions:
        print("No version probs available. Run the model scripts first.")
        return

    print("=" * 80)
    print("MODEL VERSION COMPARISON ON 2026 TOURNAMENT")
    print("=" * 80)

    # Per-round table
    print(f"\n{'Round':<7} {'N':>3} | " +
          " | ".join(f"{v:>14}" for v in versions))
    print("-" * (12 + 17 * len(versions)))
    for rnd in ROUND_ORDER:
        n = versions[next(iter(versions))]["per_round"][rnd]["n"]
        cells = []
        for label in versions:
            s = versions[label]["per_round"][rnd]
            cells.append(f"{s['correct']}/{n} LL={s['log_loss']:.3f}")
        print(f"{rnd:<7} {n:>3} | " + " | ".join(f"{c:>14}" for c in cells))

    # Overall
    print(f"\n{'OVERALL':<7}     | " +
          " | ".join(f"{versions[v]['overall_acc']*100:>5.1f}% LL={versions[v]['overall_ll']:.3f}".rjust(14)
                     for v in versions))

    # Bracket points
    print(f"\n{'BRACKET POINTS (1/2/4/8/16/32 per game)':<40}")
    print(f"{'Round':<7} | " + " | ".join(f"{v:>10}" for v in versions))
    print("-" * (10 + 13 * len(versions)))
    for rnd in ROUND_ORDER:
        cells = [f"{versions[v]['bracket_points'][rnd]:>3} pts" for v in versions]
        print(f"{rnd:<7} | " + " | ".join(f"{c:>10}" for c in cells))
    cells = [f"{versions[v]['total_points']:>3}/192" for v in versions]
    print(f"{'TOTAL':<7} | " + " | ".join(f"{c:>10}" for c in cells))

    # Final-Four / Champ picks
    print(f"\nF4 + CHAMP PICKS:")
    for v, info in versions.items():
        print(f"  {v}: F4={info['f4']}, Champ={info['champion']}")
    print(f"  ACTUAL: F4=['Connecticut', 'Michigan', 'Illinois', 'Arizona'], Champ=Michigan")

    # Per-game pick disagreements
    if len(versions) >= 2:
        print(f"\nPER-GAME PICK DISAGREEMENTS (chalk pick differs across versions):")
        print(f"  {'Round':>5}  {'Game':<35}  " +
              "  ".join(f"{v:>10}" for v in versions) + "   actual")
        for w, l, rnd in all_results:
            picks = {v: chalk_pick(raw_probs[v], w, l) for v in versions}
            confs = {v: get_prob(raw_probs[v], teams[picks[v]],
                                  teams[w if picks[v] != w else l])
                     for v in versions}
            if len(set(picks.values())) > 1:
                cells = [f"{picks[v]}({confs[v]:.0%})" for v in versions]
                print(f"  {rnd:>5}  {w + ' vs ' + l:<35}  " +
                      "  ".join(f"{c:>14}" for c in cells) +
                      f"   {w}")

    # Where each version uniquely got it right
    print(f"\nGAMES WHERE ONLY ONE VERSION GOT THE PICK RIGHT:")
    for w, l, rnd in all_results:
        right = {v for v in versions if chalk_pick(raw_probs[v], w, l) == w}
        if len(right) == 1 and len(versions) > 1:
            only = next(iter(right))
            wrong = sorted(set(versions) - right)
            print(f"  {rnd:>5}: {w} over {l} -- only {only} got it right "
                  f"({', '.join(wrong)} picked the loser)")


if __name__ == "__main__":
    main()
