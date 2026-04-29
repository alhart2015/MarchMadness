"""Driver: run v4 LOSO + 2026 prediction with each named feature drop
group, then with named individual features in Pass 2. Writes a per-row
results CSV.

Spec: docs/superpowers/specs/2026-04-29-v4-feature-ablation.md
Plan: docs/superpowers/plans/2026-04-29-v4-feature-ablation.md
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

from src.score_chalk_brackets import score_pairwise_path


GROUP_ABLATIONS = {
    "late_season": ["late_adj_oe", "late_adj_de", "late_adj_em", "late_sos"],
    "trajectory":  ["efficiency_trend", "margin_trend"],
    "conf_tourney": ["conf_tourney_wins", "conf_tourney_champ"],
    "vegas_trend": ["vegas_late_spread_delta"],
    "coach": ["coach_career_games", "coach_career_wins",
              "coach_career_winpct", "coach_career_f4_apps",
              "coach_career_champs", "coach_career_seasons"],
}

BUST_TEAMS = [
    {"name": "Vanderbilt", "bust_round": "R32", "advance_key": "S16"},
    {"name": "Iowa State", "bust_round": "S16", "advance_key": "E8"},
    {"name": "Texas Tech", "bust_round": "R32", "advance_key": "S16"},
    {"name": "Duke",       "bust_round": "E8",  "advance_key": "F4"},
]

OUTPUT_DIR = Path("output")
ABLATION_DIR = OUTPUT_DIR / "ablation"
RESULTS_CSV = OUTPUT_DIR / "ablation_v4_results.csv"


def parse_advance_probs(bracket_data_path, team_name, advance_key):
    """Read bracket_data.json and return advancement[advance_key] for the
    team whose 'name' matches team_name (case-insensitive). None if absent.
    """
    data = json.loads(Path(bracket_data_path).read_text())
    target = team_name.lower().strip()
    for tid, entry in data.items():
        if entry.get("name", "").lower().strip() == target:
            return entry.get("advancement", {}).get(advance_key)
    return None


def parse_loso_logloss(cv_per_season_path):
    """Return mean log_loss across seasons from cv_per_season_v3<sfx>.csv."""
    df = pd.read_csv(cv_per_season_path)
    return float(df["log_loss"].mean())


def build_results_row(ablation, team, bust_round, advance_key,
                      p_advance_baseline, p_advance_ablated,
                      loso_baseline, loso_ablated,
                      bracket_pts_baseline, bracket_pts_ablated):
    """Assemble one CSV row. delta_pp is in percentage points."""
    return {
        "ablation": ablation,
        "team": team,
        "bust_round": bust_round,
        "advance_key": advance_key,
        "p_advance_baseline": p_advance_baseline,
        "p_advance_ablated": p_advance_ablated,
        "delta_pp": (p_advance_ablated - p_advance_baseline) * 100
            if (p_advance_baseline is not None and p_advance_ablated is not None)
            else None,
        "loso_logloss_baseline": loso_baseline,
        "loso_logloss_ablated": loso_ablated,
        "loso_logloss_delta": loso_ablated - loso_baseline,
        "bracket_pts_baseline": bracket_pts_baseline,
        "bracket_pts_ablated": bracket_pts_ablated,
        "bracket_pts_delta": bracket_pts_ablated - bracket_pts_baseline,
    }


def run_pipeline(tag, drop_features, tuned_params_json):
    """Invoke enhanced_model_v3.py as a subprocess with the env-var hooks set.
    Returns paths to the suffixed artifacts.
    """
    suffix = "" if tag == "baseline" else f"_{tag}"
    pairwise_csv = ABLATION_DIR / f"pairwise{suffix}.csv"
    pairwise_csv.parent.mkdir(parents=True, exist_ok=True)
    if pairwise_csv.exists():
        pairwise_csv.unlink()  # MM_PAIRWISE_OUT appends; start clean.

    env = os.environ.copy()
    env["MM_FEATURE_DROP"] = ",".join(drop_features)
    env["MM_OUTPUT_SUFFIX"] = suffix
    env["MM_PAIRWISE_OUT"] = str(pairwise_csv)
    env["MM_TUNED_PARAMS_V3"] = tuned_params_json

    print(f"\n>>> ABLATION: {tag} (drop: {drop_features or 'NONE'})")
    print(f"    suffix={suffix} pairwise={pairwise_csv}")
    subprocess.run(
        [sys.executable, "src/enhanced_model_v3.py"],
        env=env, check=True,
    )

    return {
        "pairwise_csv": pairwise_csv,
        "cv_per_season": OUTPUT_DIR / f"cv_per_season_v3{suffix}.csv",
        "bracket_data": OUTPUT_DIR / f"bracket_data{suffix}.json",
    }


def collect_metrics(artifacts):
    return {
        "loso_logloss": parse_loso_logloss(artifacts["cv_per_season"]),
        "bracket_pts": score_pairwise_path(str(artifacts["pairwise_csv"]))["total_pts"],
        "advance_probs": {
            b["name"]: parse_advance_probs(
                artifacts["bracket_data"], b["name"], b["advance_key"]
            )
            for b in BUST_TEAMS
        },
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pass-num", type=int, choices=[1, 2], required=True,
                   help="1 = group ablations, 2 = individual feature drill-down")
    p.add_argument("--features", nargs="*", default=None,
                   help="Pass 2: individual feature names to ablate one at a time")
    p.add_argument("--tuned-params", required=True,
                   help="Path to JSON with v4's Optuna best_params (passed to MM_TUNED_PARAMS_V3)")
    p.add_argument("--baseline-only", action="store_true",
                   help="Run only the no-drop baseline, skip ablations")
    return p.parse_args()


def main():
    args = parse_args()
    tuned_params_json = Path(args.tuned_params).read_text().strip()

    # Always run baseline first.
    results = []
    print("=" * 70 + "\nBASELINE (no drops)\n" + "=" * 70)
    base_artifacts = run_pipeline("baseline", [], tuned_params_json)
    base_metrics = collect_metrics(base_artifacts)

    if args.baseline_only:
        print(f"\nBaseline LOSO log loss : {base_metrics['loso_logloss']:.4f}")
        print(f"Baseline 22yr brkt pts : {base_metrics['bracket_pts']:.1f}")
        for n, p in base_metrics["advance_probs"].items():
            print(f"  {n:>12s}: {p:.3f}")
        return

    if args.pass_num == 1:
        ablations = list(GROUP_ABLATIONS.items())
    else:
        if not args.features:
            sys.exit("Pass 2 requires --features f1 f2 ...")
        ablations = [(f"drop_{f}", [f]) for f in args.features]

    for tag, features in ablations:
        artifacts = run_pipeline(f"drop_{tag}", features, tuned_params_json)
        m = collect_metrics(artifacts)
        for b in BUST_TEAMS:
            results.append(build_results_row(
                ablation=tag,
                team=b["name"],
                bust_round=b["bust_round"],
                advance_key=b["advance_key"],
                p_advance_baseline=base_metrics["advance_probs"][b["name"]],
                p_advance_ablated=m["advance_probs"][b["name"]],
                loso_baseline=base_metrics["loso_logloss"],
                loso_ablated=m["loso_logloss"],
                bracket_pts_baseline=base_metrics["bracket_pts"],
                bracket_pts_ablated=m["bracket_pts"],
            ))

    df = pd.DataFrame(results)
    if RESULTS_CSV.exists():
        existing = pd.read_csv(RESULTS_CSV)
        df = pd.concat([existing, df], ignore_index=True)
    df.to_csv(RESULTS_CSV, index=False)
    print(f"\nWrote {len(df)} rows to {RESULTS_CSV}")


if __name__ == "__main__":
    main()
