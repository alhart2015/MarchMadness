"""Gate diagnostic: residual correlation + ideal-weight search of v4 vs BT.

Spec: docs/superpowers/specs/2026-05-01-bayesian-stage1-design.md

Loads the canonical pairwise CSVs (v4 dedup last-write-wins; BT as is),
joins to MNCAATourneyCompactResults.csv, and computes:
  - per-game residuals for both models
  - Pearson r(residual_v4, residual_bt)
  - cheating ideal-weight search: argmin_w mean(-log(w*p_v4 + (1-w)*p_bt))
  - disagreement breakdown
Then applies the gate's three clauses and prints a verdict.

Used as the falsification gate before any v9-C correction step or
22-season bracket-points backtest.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path("data/raw/march-machine-learning-2026")
DEFAULT_DIAGNOSTIC_OUT = "output/diag_bt_vs_v4.json"

# Gate thresholds (from spec).
GATE_R_MAX = 0.60
GATE_W_LOW = 0.30
GATE_W_HIGH = 0.85
GATE_HEADROOM_MIN = 0.005


def _load_pairwise_lookup(path: str) -> dict:
    pw = pd.read_csv(path)
    pw = pw.drop_duplicates(subset=["season", "team_a", "team_b"], keep="last")
    return {(int(s), int(a), int(b)): float(p)
            for s, a, b, p in zip(pw.season, pw.team_a, pw.team_b, pw.p_a_wins)}


def compute_diagnostic(
    pairwise_v4_csv: str,
    pairwise_bt_csv: str,
    results_csv: str = str(DATA / "MNCAATourneyCompactResults.csv"),
    results_df: pd.DataFrame = None,
) -> dict:
    """Compute the gate diagnostic. Returns a dict with all numbers."""
    v4 = _load_pairwise_lookup(pairwise_v4_csv)
    bt = _load_pairwise_lookup(pairwise_bt_csv)
    if results_df is None:
        results_df = pd.read_csv(results_csv)

    v4_p_w, bt_p_w = [], []
    for _, g in results_df.iterrows():
        s, w, l = int(g["Season"]), int(g["WTeamID"]), int(g["LTeamID"])
        if s < 2003 or s > 2025:
            continue
        a, b = (w, l) if w < l else (l, w)
        p_v4 = v4.get((s, a, b))
        p_bt = bt.get((s, a, b))
        if p_v4 is None or p_bt is None:
            continue
        v4_p_w.append(p_v4 if a == w else 1.0 - p_v4)
        bt_p_w.append(p_bt if a == w else 1.0 - p_bt)

    v4_p_w = np.array(v4_p_w)
    bt_p_w = np.array(bt_p_w)
    n = len(v4_p_w)

    eps = 1e-15
    v4_clip = np.clip(v4_p_w, eps, 1 - eps)
    bt_clip = np.clip(bt_p_w, eps, 1 - eps)

    # Standalone log loss + accuracy
    ll_v4 = float(-np.mean(np.log(v4_clip)))
    ll_bt = float(-np.mean(np.log(bt_clip)))
    acc_v4 = float((v4_p_w > 0.5).mean())
    acc_bt = float((bt_p_w > 0.5).mean())

    # Residual correlation (1 - p_for_actual_winner)
    v4_res = 1 - v4_p_w
    bt_res = 1 - bt_p_w
    if n > 1 and v4_res.std() > 0 and bt_res.std() > 0:
        r_residual = float(np.corrcoef(v4_res, bt_res)[0, 1])
    else:
        r_residual = float("nan")

    # Disagreement
    v4_pick = (v4_p_w > 0.5).astype(int)
    bt_pick = (bt_p_w > 0.5).astype(int)
    disagree_n = int((v4_pick != bt_pick).sum())
    both_correct = int(((v4_pick == 1) & (bt_pick == 1)).sum())
    v4_only = int(((v4_pick == 1) & (bt_pick == 0)).sum())
    bt_only = int(((v4_pick == 0) & (bt_pick == 1)).sum())
    both_wrong = int(((v4_pick == 0) & (bt_pick == 0)).sum())

    # Ideal-weight search (cheating: tune on test outcomes)
    ws = np.linspace(0.0, 1.0, 101)
    ll_at_w = []
    for w in ws:
        p_blend = w * v4_p_w + (1 - w) * bt_p_w
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
        "ll_bt": ll_bt,
        "acc_v4": acc_v4,
        "acc_bt": acc_bt,
        "r_residual": r_residual,
        "disagree_n": disagree_n,
        "both_correct": both_correct,
        "v4_only_correct": v4_only,
        "bt_only_correct": bt_only,
        "both_wrong": both_wrong,
        "optimal_w": optimal_w,
        "optimal_ll": optimal_ll,
        "headroom": float(headroom),
        "ll_at_w": ll_at_w.tolist(),
    }


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
    print("BT vs v4 GATE DIAGNOSTIC")
    print("=" * 70)
    print(f"  n tournament games: {diag['n_games']}")
    print(f"\n  Standalone log loss:")
    print(f"    v4: {diag['ll_v4']:.4f}   acc: {diag['acc_v4']:.3f}")
    print(f"    BT: {diag['ll_bt']:.4f}   acc: {diag['acc_bt']:.3f}")
    print(f"\n  Pearson r(residual_v4, residual_bt) = {diag['r_residual']:.4f}")
    print(f"\n  Disagreement on predicted winner:")
    n = diag['n_games']
    print(f"    disagree:      {diag['disagree_n']}/{n} "
          f"({100*diag['disagree_n']/n:.1f}%)")
    print(f"    both correct:  {diag['both_correct']}/{n}")
    print(f"    v4 only:       {diag['v4_only_correct']}/{n}")
    print(f"    BT only:       {diag['bt_only_correct']}/{n}")
    print(f"    both wrong:    {diag['both_wrong']}/{n}")
    print(f"\n  Optimal-weight search (cheating; no LOSO):")
    print(f"    log loss at w=1.0 (v4):       {diag['ll_v4']:.4f}")
    print(f"    log loss at w=0.5:            {diag['ll_at_w'][50]:.4f}")
    print(f"    log loss at w=0.0 (BT):       {diag['ll_bt']:.4f}")
    print(f"    log loss at optimal w={diag['optimal_w']:.2f}: "
          f"{diag['optimal_ll']:.4f}")
    print(f"    headroom vs v4 alone:         {diag['headroom']:+.4f}")
    print(f"\n=== VERDICT ===")
    if gate["pass"]:
        print(f"  GATE PASSED: {gate['reason']}")
        print(f"  -> Proceed to v9-C correction + bracket-points backtest")
    else:
        print(f"  GATE FAILED: {gate['reason']}")
        print(f"  -> Stop. Write findings note. No v9-C compute.")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--pairwise-v4", default="output/pairwise_v4.csv")
    parser.add_argument("--pairwise-bt", default="output/pairwise_bt.csv")
    parser.add_argument("--out-json", default=DEFAULT_DIAGNOSTIC_OUT)
    args = parser.parse_args(argv)

    diag = compute_diagnostic(args.pairwise_v4, args.pairwise_bt)
    gate = check_gate(diag)
    print_report(diag, gate)

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w") as f:
        # Drop the per-w log-loss curve to keep the JSON small; keep
        # the headline numbers.
        slim = {k: v for k, v in diag.items() if k != "ll_at_w"}
        json.dump({"diagnostic": slim, "gate": gate}, f, indent=2)
    print(f"\n  saved {args.out_json}")
    return 0 if gate["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
