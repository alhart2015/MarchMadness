"""Gate diagnostic: residual correlation + ideal-weight search of v4 vs GNN.

Spec: docs/superpowers/specs/2026-05-01-bayesian-stage1-design.md
GNN Phase 2 plan: BT-class LL-blend gate.

Loads the canonical pairwise CSVs (v4 dedup last-write-wins; GNN as is),
joins to MNCAATourneyCompactResults.csv, and computes:
  - per-game residuals for both models
  - Pearson r(residual_v4, residual_gnn)
  - cheating ideal-weight search: argmin_w mean(-log(w*p_v4 + (1-w)*p_gnn))
  - disagreement breakdown
  - per-season breakdown (cheating ideal weight per season)
Then applies the gate's three clauses and prints a verdict.

Note on r-clause direction: the GNN Phase 2 plan text says ``r >= 0.60``
but the BT-class precedent (src/diagnose_bt_vs_v4.py) uses
``r_residual < 0.60`` (low residual correlation = complementary signals
= good). We follow the BT-class convention exactly per the plan's
"match the BT-class convention exactly" instruction.

Gate (all three must pass):
  1. r_residual < 0.60       (signals not too redundant)
  2. 0.40 <= optimal_w <= 0.85 (non-degenerate; GNN plan stricter than BT)
  3. headroom > 0.005         (blend strictly better than v4 alone)
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path("data/raw/march-machine-learning-2026")
DEFAULT_DIAGNOSTIC_OUT = "output/diag_gnn_vs_v4.json"
DEFAULT_CURVE_OUT = "output/diag_gnn_vs_v4_curve.csv"
DEFAULT_PER_SEASON_OUT = "output/cv_per_season_gnn_phase2_blend.csv"

# Gate thresholds. r/headroom from BT-class precedent; w lower bound
# tightened to 0.40 per the GNN Phase 2 plan.
GATE_R_MAX = 0.60
GATE_W_LOW = 0.40
GATE_W_HIGH = 0.85
GATE_HEADROOM_MIN = 0.005


def _write_curve(path: str, ll_at_w: list) -> None:
    """Write the LL(w) blend curve to a 2-column CSV.

    Format: header `w,ll_blend`; 101 data rows for w in [0.00, 1.00] step
    0.01. Both columns formatted to 6 decimals.
    """
    ws = np.linspace(0.0, 1.0, 101)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write("w,ll_blend\n")
        for w, ll in zip(ws, ll_at_w):
            f.write(f"{w:.6f},{ll:.6f}\n")


def _load_pairwise_lookup(path: str) -> dict:
    pw = pd.read_csv(path)
    pw = pw.drop_duplicates(subset=["season", "team_a", "team_b"], keep="last")
    return {(int(s), int(a), int(b)): float(p)
            for s, a, b, p in zip(pw.season, pw.team_a, pw.team_b, pw.p_a_wins)}


def _gather_winner_probs(
    v4: dict,
    gnn: dict,
    results_df: pd.DataFrame,
    season_filter=None,
):
    """Walk results, return (v4_p_w, gnn_p_w) numpy arrays of P(actual winner).

    season_filter, if not None, is a set/iterable of seasons to keep.
    Always restricts to 2003 <= season <= 2025.
    """
    v4_p_w, gnn_p_w = [], []
    for _, g in results_df.iterrows():
        s, w, l = int(g["Season"]), int(g["WTeamID"]), int(g["LTeamID"])
        if s < 2003 or s > 2025:
            continue
        if season_filter is not None and s not in season_filter:
            continue
        a, b = (w, l) if w < l else (l, w)
        p_v4 = v4.get((s, a, b))
        p_gnn = gnn.get((s, a, b))
        if p_v4 is None or p_gnn is None:
            continue
        v4_p_w.append(p_v4 if a == w else 1.0 - p_v4)
        gnn_p_w.append(p_gnn if a == w else 1.0 - p_gnn)
    return np.array(v4_p_w), np.array(gnn_p_w)


def _diagnostic_from_arrays(v4_p_w: np.ndarray, gnn_p_w: np.ndarray) -> dict:
    """Compute the standard diagnostic dict from two parallel P(winner) arrays."""
    n = len(v4_p_w)
    eps = 1e-15

    if n == 0:
        return {
            "n_games": 0,
            "ll_v4": float("nan"),
            "ll_gnn": float("nan"),
            "acc_v4": float("nan"),
            "acc_gnn": float("nan"),
            "r_residual": float("nan"),
            "disagree_n": 0,
            "both_correct": 0,
            "v4_only_correct": 0,
            "gnn_only_correct": 0,
            "both_wrong": 0,
            "optimal_w": float("nan"),
            "optimal_ll": float("nan"),
            "headroom": float("nan"),
            "ll_at_w": [float("nan")] * 101,
        }

    v4_clip = np.clip(v4_p_w, eps, 1 - eps)
    gnn_clip = np.clip(gnn_p_w, eps, 1 - eps)

    # Standalone log loss + accuracy
    ll_v4 = float(-np.mean(np.log(v4_clip)))
    ll_gnn = float(-np.mean(np.log(gnn_clip)))
    acc_v4 = float((v4_p_w > 0.5).mean())
    acc_gnn = float((gnn_p_w > 0.5).mean())

    # Residual correlation (1 - p_for_actual_winner)
    v4_res = 1 - v4_p_w
    gnn_res = 1 - gnn_p_w
    if n > 1 and v4_res.std() > 0 and gnn_res.std() > 0:
        r_residual = float(np.corrcoef(v4_res, gnn_res)[0, 1])
    else:
        r_residual = float("nan")

    # Disagreement
    v4_pick = (v4_p_w > 0.5).astype(int)
    gnn_pick = (gnn_p_w > 0.5).astype(int)
    disagree_n = int((v4_pick != gnn_pick).sum())
    both_correct = int(((v4_pick == 1) & (gnn_pick == 1)).sum())
    v4_only = int(((v4_pick == 1) & (gnn_pick == 0)).sum())
    gnn_only = int(((v4_pick == 0) & (gnn_pick == 1)).sum())
    both_wrong = int(((v4_pick == 0) & (gnn_pick == 0)).sum())

    # Ideal-weight search (cheating: tune on test outcomes)
    ws = np.linspace(0.0, 1.0, 101)
    ll_at_w = []
    for w in ws:
        p_blend = w * v4_p_w + (1 - w) * gnn_p_w
        p_blend = np.clip(p_blend, eps, 1 - eps)
        ll_at_w.append(float(-np.mean(np.log(p_blend))))
    ll_at_w = np.array(ll_at_w)
    optimal_idx = int(np.argmin(ll_at_w))
    optimal_w = float(ws[optimal_idx])
    optimal_ll = float(ll_at_w[optimal_idx])
    headroom = ll_v4 - optimal_ll  # positive = ensemble beats v4 alone

    return {
        "n_games": n,
        "ll_v4": ll_v4,
        "ll_gnn": ll_gnn,
        "acc_v4": acc_v4,
        "acc_gnn": acc_gnn,
        "r_residual": r_residual,
        "disagree_n": disagree_n,
        "both_correct": both_correct,
        "v4_only_correct": v4_only,
        "gnn_only_correct": gnn_only,
        "both_wrong": both_wrong,
        "optimal_w": optimal_w,
        "optimal_ll": optimal_ll,
        "headroom": float(headroom),
        "ll_at_w": ll_at_w.tolist(),
    }


def compute_diagnostic(
    pairwise_v4_csv: str,
    pairwise_gnn_csv: str,
    results_csv: str = str(DATA / "MNCAATourneyCompactResults.csv"),
    results_df: pd.DataFrame = None,
) -> dict:
    """Compute the gate diagnostic across all 2003-2025 tournament games."""
    v4 = _load_pairwise_lookup(pairwise_v4_csv)
    gnn = _load_pairwise_lookup(pairwise_gnn_csv)
    if results_df is None:
        results_df = pd.read_csv(results_csv)

    v4_p_w, gnn_p_w = _gather_winner_probs(v4, gnn, results_df)
    return _diagnostic_from_arrays(v4_p_w, gnn_p_w)


def compute_per_season_diagnostic(
    pairwise_v4_csv: str,
    pairwise_gnn_csv: str,
    results_csv: str = str(DATA / "MNCAATourneyCompactResults.csv"),
    results_df: pd.DataFrame = None,
) -> pd.DataFrame:
    """Per-season r_residual, optimal_w, headroom (cheating per-season ideal weight).

    Returns a DataFrame with columns:
      season, n_games, ll_v4, ll_gnn, r_residual, optimal_w, optimal_ll, headroom
    """
    v4 = _load_pairwise_lookup(pairwise_v4_csv)
    gnn = _load_pairwise_lookup(pairwise_gnn_csv)
    if results_df is None:
        results_df = pd.read_csv(results_csv)

    rows = []
    seasons = sorted(int(s) for s in results_df["Season"].unique())
    for season in seasons:
        if season < 2003 or season > 2025:
            continue
        v4_p_w, gnn_p_w = _gather_winner_probs(
            v4, gnn, results_df, season_filter={season}
        )
        if len(v4_p_w) == 0:
            continue
        d = _diagnostic_from_arrays(v4_p_w, gnn_p_w)
        rows.append({
            "season": season,
            "n_games": d["n_games"],
            "ll_v4": d["ll_v4"],
            "ll_gnn": d["ll_gnn"],
            "r_residual": d["r_residual"],
            "optimal_w": d["optimal_w"],
            "optimal_ll": d["optimal_ll"],
            "headroom": d["headroom"],
        })
    return pd.DataFrame(
        rows,
        columns=[
            "season", "n_games", "ll_v4", "ll_gnn",
            "r_residual", "optimal_w", "optimal_ll", "headroom",
        ],
    )


def check_gate(diag: dict) -> dict:
    """Apply the three-clause gate and return {pass, reason}."""
    failures = []
    if not (diag["r_residual"] < GATE_R_MAX):
        failures.append(
            f"residual correlation {diag['r_residual']:.3f} >= {GATE_R_MAX}"
        )
    if not (GATE_W_LOW <= diag["optimal_w"] <= GATE_W_HIGH):
        failures.append(
            f"optimal weight {diag['optimal_w']:.2f} outside "
            f"[{GATE_W_LOW}, {GATE_W_HIGH}]"
        )
    if not (diag["headroom"] > GATE_HEADROOM_MIN):
        failures.append(
            f"headroom {diag['headroom']:.4f} <= {GATE_HEADROOM_MIN}"
        )
    if failures:
        return {"pass": False, "reason": "; ".join(failures)}
    return {"pass": True, "reason": "all three clauses cleared"}


def print_report(diag: dict, gate: dict) -> None:
    print("=" * 70)
    print("GNN vs v4 GATE DIAGNOSTIC")
    print("=" * 70)
    print(f"  n tournament games: {diag['n_games']}")
    print(f"\n  Standalone log loss:")
    print(f"    v4:  {diag['ll_v4']:.4f}   acc: {diag['acc_v4']:.3f}")
    print(f"    GNN: {diag['ll_gnn']:.4f}   acc: {diag['acc_gnn']:.3f}")
    print(f"\n  Pearson r(residual_v4, residual_gnn) = {diag['r_residual']:.4f}")
    print(f"\n  Disagreement on predicted winner:")
    n = diag['n_games']
    print(f"    disagree:      {diag['disagree_n']}/{n} "
          f"({100*diag['disagree_n']/n:.1f}%)")
    print(f"    both correct:  {diag['both_correct']}/{n}")
    print(f"    v4 only:       {diag['v4_only_correct']}/{n}")
    print(f"    GNN only:      {diag['gnn_only_correct']}/{n}")
    print(f"    both wrong:    {diag['both_wrong']}/{n}")
    print(f"\n  Optimal-weight search (cheating; no LOSO):")
    print(f"    log loss at w=1.0 (v4):       {diag['ll_v4']:.4f}")
    print(f"    log loss at w=0.5:            {diag['ll_at_w'][50]:.4f}")
    print(f"    log loss at w=0.0 (GNN):      {diag['ll_gnn']:.4f}")
    print(f"    log loss at optimal w={diag['optimal_w']:.2f}: "
          f"{diag['optimal_ll']:.4f}")
    print(f"    headroom vs v4 alone:         {diag['headroom']:+.4f}")
    print(f"\n=== VERDICT ===")
    if gate["pass"]:
        print(f"  GATE PASSED: {gate['reason']}")
    else:
        print(f"  GATE FAILED: {gate['reason']}")


def print_per_season(per_season: pd.DataFrame) -> None:
    if per_season.empty:
        print("\n  (no per-season rows)")
        return
    print("\n  Per-season breakdown (cheating per-season optimal w):")
    print("    season  n   ll_v4    ll_gnn   r_res    w*     opt_ll   headroom")
    for _, r in per_season.iterrows():
        print(f"    {int(r['season']):>4}    {int(r['n_games']):>2}  "
              f"{r['ll_v4']:.4f}  {r['ll_gnn']:.4f}  {r['r_residual']:+.3f}  "
              f"{r['optimal_w']:.2f}  {r['optimal_ll']:.4f}  "
              f"{r['headroom']:+.4f}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--pairwise-v4", default="output/pairwise_v4.csv")
    parser.add_argument("--pairwise-gnn",
                        default="output/pairwise_gnn_phase2.csv")
    parser.add_argument("--out-json", default=DEFAULT_DIAGNOSTIC_OUT)
    parser.add_argument("--curve-out", default=DEFAULT_CURVE_OUT,
                        help="Where to write the LL(w) blend curve (CSV)")
    parser.add_argument("--per-season-out", default=DEFAULT_PER_SEASON_OUT,
                        help="Where to write the per-season diagnostic CSV")
    args = parser.parse_args(argv)

    diag = compute_diagnostic(args.pairwise_v4, args.pairwise_gnn)
    gate = check_gate(diag)
    print_report(diag, gate)

    per_season = compute_per_season_diagnostic(
        args.pairwise_v4, args.pairwise_gnn,
    )
    print_per_season(per_season)

    _write_curve(args.curve_out, diag["ll_at_w"])
    print(f"\n  saved {args.curve_out}")

    Path(args.per_season_out).parent.mkdir(parents=True, exist_ok=True)
    per_season.to_csv(args.per_season_out, index=False, float_format="%.6f")
    print(f"  saved {args.per_season_out}")

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w") as f:
        # Drop the per-w log-loss curve to keep the JSON small; keep
        # the headline numbers.
        slim = {k: v for k, v in diag.items() if k != "ll_at_w"}
        json.dump({"diagnostic": slim, "gate": gate}, f, indent=2)
    print(f"  saved {args.out_json}")
    return 0 if gate["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
