"""Pre-sweep falsification gate for the feature-view diversity ensemble.

Spec: docs/superpowers/specs/2026-05-02-feature-view-ensemble-design.md

3-clause gate:
  1. Per-peer LL ceiling: each peer's weighted-mean per-game LL on
     played tournament games is within PEER_LL_CEILING_DELTA of v4's.
  2. Inter-peer residual correlation: pearson r(resid_A, resid_B) on
     played-game rows, where resid = p_winner_won - 1, is < RESID_CORR_MAX.
  3. Best-blend LL headroom: optimal 2-blend of peer A and peer B beats
     v4 standalone by >= HEADROOM_MIN.

If any clause fails, the gate fails. Exits nonzero on FAIL so a wrapper
can short-circuit the sweep.

Mirrors src/diagnose_v9d.py and src/diagnose_bt_vs_v4.py in shape.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar, minimize

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

DATA = Path("data/raw/march-machine-learning-2026")
DEFAULT_DIAGNOSTIC_OUT = "output/diag_feature_view_ensemble.json"

# Clause thresholds. Each maps to a prior-experiment failure mode.
PEER_LL_CEILING_DELTA = 0.025
RESID_CORR_MAX = 0.60
HEADROOM_MIN = 0.001

EPS = 1e-15


def _winner_log_loss(p_winner: np.ndarray) -> float:
    """Mean -log(p_winner) over rows with epsilon clipping."""
    p = np.clip(np.asarray(p_winner, dtype=float), EPS, 1 - EPS)
    return float(-np.log(p).mean())


def compute_pairwise_ll(
    pairwise_csv: str, results_csv: str
) -> tuple[float, np.ndarray, np.ndarray]:
    """Join a pairwise probability CSV with played tournament games and
    compute the winner-perspective log loss.

    Returns: (weighted_mean_ll, p_winner_won, labels)
    Pairwise schema: season, team_a, team_b, p_a_wins (team_a < team_b).
    Results schema: Season, DayNum, WTeamID, LTeamID, ...

    For a played game (W, L):
      a = min(W, L), b = max(W, L) so the pair matches the pairwise CSV.
      p_winner_won = p_a_wins if W < L else 1 - p_a_wins.
    """
    pairwise = pd.read_csv(pairwise_csv)
    results = pd.read_csv(results_csv)

    res = results[["Season", "WTeamID", "LTeamID"]].copy()
    res["team_a"] = np.minimum(res["WTeamID"], res["LTeamID"])
    res["team_b"] = np.maximum(res["WTeamID"], res["LTeamID"])
    res["winner_was_a"] = res["WTeamID"] < res["LTeamID"]
    res = res.rename(columns={"Season": "season"})[
        ["season", "team_a", "team_b", "winner_was_a"]
    ]

    merged = res.merge(
        pairwise[["season", "team_a", "team_b", "p_a_wins"]],
        on=["season", "team_a", "team_b"],
        how="inner",
    )

    p_winner_won = np.where(
        merged["winner_was_a"],
        merged["p_a_wins"],
        1.0 - merged["p_a_wins"],
    ).astype(float)
    labels = np.ones(len(merged), dtype=int)
    return _winner_log_loss(p_winner_won), p_winner_won, labels


def optimal_2blend(
    p_winner_a: np.ndarray, p_winner_b: np.ndarray
) -> tuple[float, float]:
    """Find w in [0, 1] minimizing LL(w * p_a + (1-w) * p_b).

    Returns (w_opt, ll_opt).
    """
    def loss(w):
        blend = w * p_winner_a + (1 - w) * p_winner_b
        return _winner_log_loss(blend)
    result = minimize_scalar(loss, bounds=(0.0, 1.0), method="bounded")
    return float(result.x), float(result.fun)


def optimal_3blend(
    p_winner_v4: np.ndarray,
    p_winner_a: np.ndarray,
    p_winner_b: np.ndarray,
) -> tuple[tuple[float, float, float], float]:
    """Find (w_v4, w_a, w_b) on the simplex minimizing LL of the blend.

    Returns ((w_v4, w_a, w_b), ll_opt). Used for E2 ensemble materialization
    in Task 9; not part of any gate clause.
    """
    def loss(w):
        w0, w1, w2 = w[0], w[1], w[2]
        blend = w0 * p_winner_v4 + w1 * p_winner_a + w2 * p_winner_b
        return _winner_log_loss(blend)

    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
    bounds = [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]
    x0 = np.array([1.0 / 3, 1.0 / 3, 1.0 / 3])
    result = minimize(
        loss, x0, method="SLSQP", bounds=bounds, constraints=constraints,
    )
    return (
        (float(result.x[0]), float(result.x[1]), float(result.x[2])),
        float(result.fun),
    )


def residual_correlation(
    p_winner_a: np.ndarray, p_winner_b: np.ndarray
) -> float:
    """Pearson r between residuals (p - 1) on aligned played-game rows.

    All rows are winner-perspective so the label is always 1; residuals
    are p - 1, and Pearson r is invariant to the constant shift, so
    r(resid_A, resid_B) == r(p_a, p_b).
    """
    pa = np.asarray(p_winner_a, dtype=float)
    pb = np.asarray(p_winner_b, dtype=float)
    if pa.std() == 0 or pb.std() == 0:
        return 0.0
    return float(np.corrcoef(pa, pb)[0, 1])


def compute_gate(
    pairwise_v4_csv: str,
    pairwise_peer_a_csv: str,
    pairwise_peer_b_csv: str,
    results_csv: str = str(DATA / "MNCAATourneyCompactResults.csv"),
) -> dict:
    """Compute all three clause values plus optimal 2-blend / 3-blend.

    Returns a dict suitable for json.dump.
    """
    ll_v4, p_v4, _ = compute_pairwise_ll(pairwise_v4_csv, results_csv)
    ll_a, p_a, _ = compute_pairwise_ll(pairwise_peer_a_csv, results_csv)
    ll_b, p_b, _ = compute_pairwise_ll(pairwise_peer_b_csv, results_csv)

    n_v4 = len(p_v4)
    n_a = len(p_a)
    n_b = len(p_b)
    if not (n_v4 == n_a == n_b):
        raise ValueError(
            f"played-game coverage mismatch: v4={n_v4}, peer_a={n_a}, "
            f"peer_b={n_b}; the gate requires identical coverage"
        )

    rho_resid = residual_correlation(p_a, p_b)
    w_opt_2blend, ll_opt_2blend = optimal_2blend(p_a, p_b)
    w_opt_3blend, ll_opt_3blend = optimal_3blend(p_v4, p_a, p_b)
    headroom = ll_v4 - ll_opt_2blend

    return {
        "n_played_games": int(n_v4),
        "ll_v4": float(ll_v4),
        "ll_peer_a": float(ll_a),
        "ll_peer_b": float(ll_b),
        "ll_2blend_optimal": float(ll_opt_2blend),
        "ll_3blend_optimal": float(ll_opt_3blend),
        "w_2blend_optimal": float(w_opt_2blend),
        "w_3blend_optimal": list(w_opt_3blend),
        "rho_residual": float(rho_resid),
        "headroom_2blend_vs_v4": float(headroom),
        "clauses": {
            "per_peer_ll_ceiling": {
                "threshold": float(PEER_LL_CEILING_DELTA),
                "ll_v4": float(ll_v4),
                "ll_peer_a": float(ll_a),
                "ll_peer_b": float(ll_b),
                "delta_a": float(ll_a - ll_v4),
                "delta_b": float(ll_b - ll_v4),
                "pass": (ll_a - ll_v4 <= PEER_LL_CEILING_DELTA)
                        and (ll_b - ll_v4 <= PEER_LL_CEILING_DELTA),
            },
            "residual_correlation": {
                "threshold": float(RESID_CORR_MAX),
                "rho": float(rho_resid),
                "pass": rho_resid < RESID_CORR_MAX,
            },
            "blend_headroom": {
                "threshold": float(HEADROOM_MIN),
                "headroom": float(headroom),
                "pass": headroom >= HEADROOM_MIN,
            },
        },
    }


def check_gate(diag: dict) -> dict:
    """All clauses must pass for the gate to clear."""
    failed = [name for name, c in diag["clauses"].items() if not c["pass"]]
    if not failed:
        return {"pass": True, "failed_clauses": [], "reason": "all clauses pass"}
    return {
        "pass": False,
        "failed_clauses": failed,
        "reason": f"failed clauses: {failed}",
    }


def print_report(diag: dict, gate: dict) -> None:
    print("=" * 70)
    print("FEATURE-VIEW ENSEMBLE PRE-SWEEP GATE")
    print("=" * 70)
    print(f"  n played games: {diag['n_played_games']}")
    print(f"\n  Per-game LL on played tournament games:")
    print(f"    v4:                         {diag['ll_v4']:.4f}")
    print(f"    peer_A (team-strength):     {diag['ll_peer_a']:.4f}  "
          f"(delta_a {diag['clauses']['per_peer_ll_ceiling']['delta_a']:+.4f})")
    print(f"    peer_B (form+market):       {diag['ll_peer_b']:.4f}  "
          f"(delta_b {diag['clauses']['per_peer_ll_ceiling']['delta_b']:+.4f})")
    print(f"    2-blend optimal:            {diag['ll_2blend_optimal']:.4f}  "
          f"(headroom vs v4 {diag['headroom_2blend_vs_v4']:+.4f})")
    print(f"    3-blend optimal (v4,A,B):   {diag['ll_3blend_optimal']:.4f}")
    print(f"\n  optimal weights:")
    print(f"    2-blend (A, B):   ({diag['w_2blend_optimal']:.3f}, "
          f"{1 - diag['w_2blend_optimal']:.3f})")
    print(f"    3-blend (v4,A,B): "
          f"({diag['w_3blend_optimal'][0]:.3f}, "
          f"{diag['w_3blend_optimal'][1]:.3f}, "
          f"{diag['w_3blend_optimal'][2]:.3f})")
    print(f"\n  rho(resid_A, resid_B): {diag['rho_residual']:+.3f}")
    print(f"\n  Clause checks:")
    for name, c in diag["clauses"].items():
        verdict = "PASS" if c["pass"] else "FAIL"
        print(f"    {name}: {verdict} (threshold {c['threshold']})")
    print(f"\n=== VERDICT ===")
    if gate["pass"]:
        print(f"  GATE PASSED: {gate['reason']}")
        print(f"  -> Proceed to materialize E1 + E2 ensemble CSVs and run sweeps.")
    else:
        print(f"  GATE FAILED: {gate['reason']}")
        print(f"  -> Stop. Write findings note. No sweep.")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--pairwise-v4", default="output/pairwise_v4.csv")
    parser.add_argument("--pairwise-peer-a", default="output/pairwise_peer_a.csv")
    parser.add_argument("--pairwise-peer-b", default="output/pairwise_peer_b.csv")
    parser.add_argument(
        "--results-csv",
        default=str(DATA / "MNCAATourneyCompactResults.csv"),
    )
    parser.add_argument("--out-json", default=DEFAULT_DIAGNOSTIC_OUT)
    args = parser.parse_args(argv)

    diag = compute_gate(
        pairwise_v4_csv=args.pairwise_v4,
        pairwise_peer_a_csv=args.pairwise_peer_a,
        pairwise_peer_b_csv=args.pairwise_peer_b,
        results_csv=args.results_csv,
    )
    gate = check_gate(diag)
    print_report(diag, gate)

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump({"diagnostic": diag, "gate": gate}, f, indent=2)
    print(f"\n  saved {args.out_json}")
    return 0 if gate["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
