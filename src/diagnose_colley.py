"""Two-clause falsification gate for colley_rating.

Clause 1 -- non-redundancy: per-season Pearson correlation between
colley_rating and (adj_em, massey_composite, season_win_pct). Pass
if mean |corr| < 0.95 AND max |corr| < 0.97 against ALL THREE
baselines.

Clause 2 -- no-harm headroom: 3-season subset {2019, 2022, 2024}.
Train v4 with colley_rating in feature_cols vs without; pass if
mean LL with - mean LL without <= +0.001.

See docs/superpowers/specs/2026-05-03-colley-matrix-feature-design.md.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Path setup: when invoked as `python src/diagnose_colley.py`, ensure
# the project root is on sys.path so `from src.enhanced_model import ...`
# resolves. Mirrors src/enhanced_model_v3.py and src/diagnose_massey_mov.py.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

logger = logging.getLogger(__name__)

GATE_SUBSET_SEASONS = [2019, 2022, 2024]
CORR_MEAN_MAX = 0.95
CORR_PER_SEASON_MAX = 0.97
LL_HEADROOM_MAX = 0.001
CLAUSE1_BASELINES = ["adj_em", "massey_composite", "season_win_pct"]


def clause1_correlations(feature_matrix: pd.DataFrame) -> dict:
    """Compute per-season Pearson correlations of colley_rating vs
    each baseline in CLAUSE1_BASELINES. Returns aggregate summary."""
    needed = {"Season", "TeamID", "colley_rating"} | set(CLAUSE1_BASELINES)
    missing = needed - set(feature_matrix.columns)
    if missing:
        raise ValueError(f"feature_matrix missing columns for clause 1: {sorted(missing)}")

    seasons = sorted(feature_matrix["Season"].unique())
    rows = []
    for season in seasons:
        sub = feature_matrix[feature_matrix["Season"] == season]
        sub = sub.dropna(subset=["colley_rating", *CLAUSE1_BASELINES])
        if len(sub) < 50:
            logger.warning("Season %d: only %d teams with all required cols; skipping",
                           season, len(sub))
            continue
        row = {"season": int(season), "n_teams": int(len(sub))}
        for baseline in CLAUSE1_BASELINES:
            row[f"corr_vs_{baseline}"] = float(sub["colley_rating"].corr(sub[baseline]))
        rows.append(row)

    df = pd.DataFrame(rows)
    summary = {"per_season": rows}
    all_pass = True
    for baseline in CLAUSE1_BASELINES:
        col = f"corr_vs_{baseline}"
        mean_abs = float(df[col].abs().mean())
        max_abs = float(df[col].abs().max())
        summary[f"mean_abs_{col}"] = mean_abs
        summary[f"max_abs_{col}"] = max_abs
        if mean_abs >= CORR_MEAN_MAX or max_abs >= CORR_PER_SEASON_MAX:
            all_pass = False
    summary["pass"] = bool(all_pass)
    return summary


def clause2_headroom(seasons: list[int] = GATE_SUBSET_SEASONS) -> dict:
    """Run LOSO on the 3-season subset twice (with / without
    colley_rating in feature_cols). Pass if mean(LL_with) -
    mean(LL_without) <= LL_HEADROOM_MAX.

    Toggles colley_rating in feature_cols (NOT in the matrix) so
    train/test splits are byte-identical between arms.
    """
    from src.enhanced_model_v3 import (
        leave_one_season_out_cv_weighted,
        prepare_loso_inputs,
    )

    inputs = prepare_loso_inputs()
    fm = inputs["feature_matrix"]
    tourney = inputs["tourney_filtered"]
    regular = inputs["regular_results"]
    feature_cols_full = inputs["feature_cols"]
    top_80 = inputs["top_80_by_season"]

    if "colley_rating" not in fm.columns:
        raise RuntimeError(
            "colley_rating not in feature_matrix; ensure Task 4 wire-in is committed"
        )
    if "colley_rating" not in feature_cols_full:
        raise RuntimeError(
            "colley_rating not in feature_cols; check get_feature_cols include logic"
        )

    cols_with = list(feature_cols_full)
    cols_without = [c for c in feature_cols_full if c != "colley_rating"]

    res_with = leave_one_season_out_cv_weighted(
        fm, tourney, regular, cols_with, top_80, allowed_holdouts=seasons,
    )
    res_without = leave_one_season_out_cv_weighted(
        fm, tourney, regular, cols_without, top_80, allowed_holdouts=seasons,
    )

    df_with = res_with["per_season"]
    df_without = res_without["per_season"]
    per_with = {int(r["season"]): r for _, r in df_with.iterrows()}
    per_without = {int(r["season"]): r for _, r in df_without.iterrows()}

    per_season = []
    for season in seasons:
        rw = per_with.get(int(season))
        rwo = per_without.get(int(season))
        if rw is None or rwo is None:
            continue
        per_season.append({
            "season": int(season),
            "ll_with": float(rw["log_loss"]),
            "ll_without": float(rwo["log_loss"]),
            "ll_delta": float(rw["log_loss"] - rwo["log_loss"]),
        })

    mean_with = float(np.mean([r["ll_with"] for r in per_season]))
    mean_without = float(np.mean([r["ll_without"] for r in per_season]))
    delta = mean_with - mean_without
    return {
        "subset_seasons": list(seasons),
        "per_season": per_season,
        "mean_ll_with_colley": mean_with,
        "mean_ll_without_colley": mean_without,
        "mean_ll_delta": delta,
        "pass": bool(delta <= LL_HEADROOM_MAX),
    }


def main():
    from src.enhanced_model import compute_all_features, load_all_data
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    print("=" * 70)
    print("Colley gate: clause 1 (non-redundancy)")
    print("=" * 70)

    data = load_all_data()
    feature_matrix = compute_all_features(data)
    c1 = clause1_correlations(feature_matrix)
    print(json.dumps({k: v for k, v in c1.items() if k != "per_season"}, indent=2))
    print(f"  CLAUSE 1: {'PASS' if c1['pass'] else 'FAIL'}")

    if not c1["pass"]:
        result = {"clause1": c1, "clause2": None, "gate_pass": False}
        Path("output").mkdir(exist_ok=True)
        Path("output/diag_colley.json").write_text(json.dumps(result, indent=2))
        print("\nSTOPPING: clause 1 failed; clause 2 not run.")
        sys.exit(1)

    print()
    print("=" * 70)
    print("Colley gate: clause 2 (LL headroom on 3-season subset)")
    print("=" * 70)
    c2 = clause2_headroom()
    print(json.dumps({k: v for k, v in c2.items() if k != "per_season"}, indent=2))
    print(f"  CLAUSE 2: {'PASS' if c2['pass'] else 'FAIL'}")

    gate_pass = c1["pass"] and c2["pass"]
    result = {"clause1": c1, "clause2": c2, "gate_pass": gate_pass}
    Path("output").mkdir(exist_ok=True)
    Path("output/diag_colley.json").write_text(json.dumps(result, indent=2))

    print()
    print("=" * 70)
    print(f"AGGREGATE GATE: {'PASS -- proceed to full LOSO backtest' if gate_pass else 'FAIL -- stop, write findings'}")
    print("=" * 70)
    sys.exit(0 if gate_pass else 1)


if __name__ == "__main__":
    main()
