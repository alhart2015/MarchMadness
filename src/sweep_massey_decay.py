"""Sweep half_life_days for Massey-MOV solver and measure clause-1
correlations vs adj_em (which itself uses 30-day half-life decay) and
massey_composite.

Hypothesis to falsify: time-decay weighting can push the Massey-vs-adj_em
correlation below the 0.95 mean threshold from the original gate (which
hit 0.957 with no Massey decay).

For each half_life in {None, 7, 14, 30, 60, 120}:
  - rebuild Massey ratings
  - merge into v4 feature matrix (without re-running compute_all_features)
  - compute per-season Pearson correlations vs adj_em and massey_composite
  - report mean and max-abs across 24 seasons

Output: stdout table + output/sweep_massey_decay.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.enhanced_model import compute_all_features, load_all_data  # noqa: E402
from src.features.massey_matrix import compute_massey_mov_ratings  # noqa: E402

HALF_LIFE_VALUES: list[float | None] = [None, 7.0, 14.0, 30.0, 60.0, 120.0]


def correlations_vs_baselines(fm: pd.DataFrame) -> dict:
    """Per-season Pearson correlations of massey_mov_rating vs (adj_em,
    massey_composite). Returns aggregate summary."""
    seasons = sorted(fm["Season"].unique())
    rows = []
    for season in seasons:
        sub = fm[fm["Season"] == season]
        sub = sub.dropna(subset=["massey_mov_rating", "adj_em", "massey_composite"])
        if len(sub) < 50:
            continue
        rows.append({
            "season": int(season),
            "corr_vs_adj_em": float(sub["massey_mov_rating"].corr(sub["adj_em"])),
            "corr_vs_massey_composite": float(
                sub["massey_mov_rating"].corr(sub["massey_composite"])
            ),
        })
    df = pd.DataFrame(rows)
    return {
        "n_seasons": len(rows),
        "mean_abs_corr_vs_adj_em": float(df["corr_vs_adj_em"].abs().mean()),
        "max_abs_corr_vs_adj_em": float(df["corr_vs_adj_em"].abs().max()),
        "mean_abs_corr_vs_massey_composite": float(
            df["corr_vs_massey_composite"].abs().mean()
        ),
        "max_abs_corr_vs_massey_composite": float(
            df["corr_vs_massey_composite"].abs().max()
        ),
        "per_season": rows,
    }


def main():
    print("Loading v4 base feature matrix (no massey_mov_rating)...")
    data = load_all_data()
    base_fm = compute_all_features(data)
    reg = data["reg_season"]
    print(f"  base feature matrix: {len(base_fm)} rows, {len(base_fm.columns)} cols")
    # Confirm baselines exist.
    for col in ("adj_em", "massey_composite"):
        if col not in base_fm.columns:
            raise SystemExit(f"baseline column '{col}' missing from feature_matrix")

    results = []
    for hl in HALF_LIFE_VALUES:
        label = "uniform" if hl is None else f"hl={hl}d"
        print(f"\n--- {label} ---")
        massey = compute_massey_mov_ratings(reg, mov_cap=21, half_life_days=hl)
        # Merge by (TeamID, Season). Use a fresh copy so we don't pollute the
        # base across iterations.
        fm = base_fm.merge(massey, on=["TeamID", "Season"], how="left")
        summary = correlations_vs_baselines(fm)
        summary["half_life_days"] = hl
        summary["label"] = label
        # Concise stdout: drop per-season detail
        compact = {k: v for k, v in summary.items() if k != "per_season"}
        print(json.dumps(compact, indent=2))
        clause1_pass = (
            summary["mean_abs_corr_vs_adj_em"] < 0.95
            and summary["max_abs_corr_vs_adj_em"] < 0.97
            and summary["mean_abs_corr_vs_massey_composite"] < 0.95
            and summary["max_abs_corr_vs_massey_composite"] < 0.97
        )
        summary["clause1_pass"] = bool(clause1_pass)
        print(f"  clause1 pass: {clause1_pass}")
        results.append(summary)

    Path("output").mkdir(exist_ok=True)
    Path("output/sweep_massey_decay.json").write_text(json.dumps(results, indent=2))
    print("\nWrote output/sweep_massey_decay.json")

    # Final summary table.
    print("\n=== summary table ===")
    header = f"{'label':<12} {'mean_em':>9} {'max_em':>9} {'mean_comp':>10} {'max_comp':>9} {'pass':>6}"
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r['label']:<12} "
            f"{r['mean_abs_corr_vs_adj_em']:>9.4f} "
            f"{r['max_abs_corr_vs_adj_em']:>9.4f} "
            f"{r['mean_abs_corr_vs_massey_composite']:>10.4f} "
            f"{r['max_abs_corr_vs_massey_composite']:>9.4f} "
            f"{'PASS' if r['clause1_pass'] else 'FAIL':>6}"
        )


if __name__ == "__main__":
    main()
