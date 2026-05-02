"""Per-season Bradley-Terry stage-1 trainer.

Spec: docs/superpowers/specs/2026-05-01-bayesian-stage1-design.md

For each season Y in 2003-2025, fits team strengths from Y's regular-
season games via L2-regularized logistic regression with team-indicator
+ home-court design matrix. Mathematically equivalent to MAP Bradley-
Terry under a Gaussian prior on strengths.

The fit is per-season -- team strengths are season-specific parameters
fit on season-specific data. No cross-season learning, no leakage from
tournament outcomes (we only use regular-season games).

Output: appends to output/pairwise_bt.csv with rows
    (season, team_a, team_b, p_a_wins),  team_a < team_b
covering all unordered pairs of tournament-field teams in each held-
out season. Same schema as pairwise_v4.csv.
"""
import argparse
import math
import sys
import time
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.linear_model import LogisticRegression

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

DATA = Path("data/raw/march-machine-learning-2026")
DEFAULT_PAIRWISE_OUT = "output/pairwise_bt.csv"
DEFAULT_C = 10.0
SEASONS = list(range(2003, 2026))  # 2003..2025; 2020 absent in data


def extract_home_court_value(wloc: str) -> int:
    """WLoc -> home-court column value relative to the *winner*:
        H -> +1 (winner was home)
        A -> -1 (winner was away)
        N ->  0 (neutral)
    """
    if wloc == "H":
        return 1
    if wloc == "A":
        return -1
    return 0


def build_design_matrix(
    games: pd.DataFrame, team_ids: Sequence[int]
) -> tuple[sp.csr_matrix, np.ndarray]:
    """Build (X, y) for L2-LR Bradley-Terry.

    games: DataFrame with WTeamID, LTeamID, WLoc columns.
    team_ids: ordered list of team IDs; column index = team_ids.index(tid).

    Each game is one row. The row has +1 in the winner's column,
    -1 in the loser's column, and the home-court signal in the final
    column (extract_home_court_value(WLoc)). Label y = 1 (we always
    encode the +1-winner perspective).
    """
    team_idx = {int(tid): i for i, tid in enumerate(team_ids)}
    n_teams = len(team_ids)
    n_games = len(games)
    n_cols = n_teams + 1

    rows = np.empty(3 * n_games, dtype=np.int64)
    cols = np.empty(3 * n_games, dtype=np.int64)
    vals = np.empty(3 * n_games, dtype=np.float64)

    for k, (_, g) in enumerate(games.iterrows()):
        w, l = int(g["WTeamID"]), int(g["LTeamID"])
        wloc = g["WLoc"]
        rows[3 * k]     = k
        cols[3 * k]     = team_idx[w]
        vals[3 * k]     = 1.0
        rows[3 * k + 1] = k
        cols[3 * k + 1] = team_idx[l]
        vals[3 * k + 1] = -1.0
        rows[3 * k + 2] = k
        cols[3 * k + 2] = n_teams
        vals[3 * k + 2] = float(extract_home_court_value(wloc))

    X = sp.csr_matrix((vals, (rows, cols)), shape=(n_games, n_cols))
    y = np.ones(n_games, dtype=np.int64)
    return X, y


def fit_bradley_terry(
    X: sp.csr_matrix, y: np.ndarray, C: float = DEFAULT_C
) -> np.ndarray:
    """Fit L2 logistic regression and return the coefficient vector.

    Returns array of length n_cols = (n_teams + 1).
    Indices 0..n_teams-1: per-team strengths.
    Index n_teams: home-court coefficient.
    """
    # n_classes=2 with labels [0, 1]: y is all 1s, but the all-1 case
    # yields a degenerate fit. Inject a single artificial label-0 row
    # at the *all-zero* feature vector so LogisticRegression is happy
    # without distorting any team's evidence.
    # (sklearn refuses to fit with a single class. The zero row + C=10
    # adds negligible bias to the actual coefficients.)
    n_cols = X.shape[1]
    zero_row = sp.csr_matrix((1, n_cols))
    X_aug = sp.vstack([X, zero_row], format="csr")
    y_aug = np.concatenate([y, [0]])

    model = LogisticRegression(
        penalty="l2",
        solver="lbfgs",
        fit_intercept=False,
        C=C,
        max_iter=2000,
    )
    model.fit(X_aug, y_aug)
    return model.coef_.ravel()


def predict_pairwise_for_field(
    season: int,
    field: Iterable[int],
    team_ids: Sequence[int],
    strengths: np.ndarray,
) -> List[dict]:
    """For each unordered pair (a, b) in the field with a < b, compute
    p_a_wins = sigmoid(s_a - s_b). NO home-court term -- tournament is
    neutral. Returns a list of dict rows ready for pd.DataFrame.

    Teams in the field but missing from team_ids (e.g., a team that
    appears in the tournament but never played a regular-season game --
    extremely rare but possible at the edge) are skipped.
    """
    team_idx = {int(tid): i for i, tid in enumerate(team_ids)}
    field_sorted = sorted(set(int(t) for t in field if int(t) in team_idx))
    rows = []
    for i in range(len(field_sorted)):
        for j in range(i + 1, len(field_sorted)):
            a, b = field_sorted[i], field_sorted[j]
            s_a = strengths[team_idx[a]]
            s_b = strengths[team_idx[b]]
            p = 1.0 / (1.0 + math.exp(-(s_a - s_b)))
            rows.append({
                "season": season,
                "team_a": a,
                "team_b": b,
                "p_a_wins": p,
            })
    return rows


def run_bt_loso(
    out_csv: str = DEFAULT_PAIRWISE_OUT,
    C: float = DEFAULT_C,
    seasons: Iterable[int] = SEASONS,
) -> dict:
    """Per-season BT fits over the configured season range.

    Writes (season, team_a, team_b, p_a_wins) rows for each season's
    tournament field to out_csv (overwrites any existing file).
    Returns a summary dict with per-season metrics.
    """
    print("=" * 70)
    print("BRADLEY-TERRY STAGE-1 PER-SEASON TRAINER")
    print("=" * 70)
    print(f"  C={C}, out_csv={out_csv}")

    reg = pd.read_csv(DATA / "MRegularSeasonCompactResults.csv")
    seeds = pd.read_csv(DATA / "MNCAATourneySeeds.csv")
    results = pd.read_csv(DATA / "MNCAATourneyCompactResults.csv")

    if Path(out_csv).exists():
        Path(out_csv).unlink()
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)

    summary = []
    overall_start = time.time()

    for season in seasons:
        t0 = time.time()
        season_games = reg[reg["Season"] == season]
        if len(season_games) == 0:
            print(f"  [{season}] no regular-season games, skipping")
            continue

        team_ids = sorted(set(season_games["WTeamID"].astype(int)) |
                          set(season_games["LTeamID"].astype(int)))
        X, y = build_design_matrix(season_games, team_ids)
        coefs = fit_bradley_terry(X, y, C=C)
        strengths = coefs[: len(team_ids)]
        h_coef = float(coefs[len(team_ids)])

        # Field = teams that played in this season's tournament.
        season_results = results[results["Season"] == season]
        field = sorted(set(season_results["WTeamID"].astype(int)) |
                       set(season_results["LTeamID"].astype(int)))
        if not field:
            # Fall back to seeded teams if no tournament results yet.
            season_seeds = seeds[seeds["Season"] == season]
            field = sorted(season_seeds["TeamID"].astype(int).tolist())

        rows = predict_pairwise_for_field(season, field, team_ids, strengths)
        out_df = pd.DataFrame(rows)
        write_header = not Path(out_csv).exists()
        out_df.to_csv(out_csv, mode="a", index=False, header=write_header)

        # Per-season tournament log loss for visibility.
        ll, acc, n_eval = _score_tournament_games(
            season_results, dict(zip(team_ids, strengths))
        )
        summary.append({
            "season": season,
            "n_teams": len(team_ids),
            "n_games": len(season_games),
            "n_pairs_written": len(rows),
            "n_eval_games": n_eval,
            "h_coef": h_coef,
            "log_loss": ll,
            "accuracy": acc,
            "fold_seconds": round(time.time() - t0, 1),
        })
        print(f"  [{season}] teams={len(team_ids):>3} games={len(season_games):>5} "
              f"h={h_coef:>+5.3f} ll={ll:.4f} acc={acc:.3f} "
              f"pairs={len(rows):>5} ({time.time() - t0:.1f}s)")

    overall = time.time() - overall_start
    print(f"\nDONE in {overall:.1f}s; pairwise CSV: {out_csv}")
    return {"per_season": pd.DataFrame(summary), "out_csv": out_csv}


def _score_tournament_games(
    results: pd.DataFrame, strengths_by_id: dict
) -> tuple[float, float, int]:
    """Per-season tournament log loss + accuracy from fitted strengths.
    Returns (log_loss, accuracy, n_games_evaluated).
    """
    eps = 1e-15
    ll_terms = []
    correct = 0
    for _, g in results.iterrows():
        w, l = int(g["WTeamID"]), int(g["LTeamID"])
        s_w = strengths_by_id.get(w)
        s_l = strengths_by_id.get(l)
        if s_w is None or s_l is None:
            continue
        p_w = 1.0 / (1.0 + math.exp(-(s_w - s_l)))
        p_w = min(max(p_w, eps), 1 - eps)
        ll_terms.append(-math.log(p_w))
        correct += 1 if p_w > 0.5 else 0
    n = len(ll_terms)
    if n == 0:
        return float("nan"), float("nan"), 0
    return float(np.mean(ll_terms)), float(correct / n), n


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", default=DEFAULT_PAIRWISE_OUT)
    parser.add_argument("--c", type=float, default=DEFAULT_C,
                        help=f"L2 inverse-regularization (default: {DEFAULT_C})")
    args = parser.parse_args(argv)
    run_bt_loso(out_csv=args.out, C=args.c)
    return 0


if __name__ == "__main__":
    sys.exit(main())
