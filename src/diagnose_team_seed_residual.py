"""Phase 1 diagnostic for team-seed-residual features.

Produces 5 sanity-check artifacts to output/team_seed_residual_diagnostic.{json,log}:
  1. Per-seed empirical baseline table
  2. 9-champion (2015-2024) residual values with hand-computable cross-check
  3. Pearson correlation matrix vs incumbent v4 features
  4. Distribution percentiles (5/25/50/75/95) of each new feature
  5. Top-10 / bottom-10 (Season, TeamName, value) pairs

Usage:
    python -m src.diagnose_team_seed_residual

Spec: docs/superpowers/specs/2026-05-09-team-seed-residual-design.md
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.features.team_history import (
    compute_per_seed_baseline,
    compute_team_history_features,
    compute_team_residuals_in_window,
    shrunk_ewma,
    shrunk_mean,
)

DATA_DIR = Path("data/raw/march-machine-learning-2026")
OUT_JSON = Path("output/team_seed_residual_diagnostic.json")
OUT_LOG = Path("output/team_seed_residual_diagnostic.log")

CHAMPIONS_2015_2024 = [
    (2015, "Duke"),
    (2016, "Villanova"),
    (2017, "North Carolina"),
    (2018, "Villanova"),
    (2019, "Virginia"),
    (2021, "Baylor"),
    (2022, "Kansas"),
    (2023, "Connecticut"),
    (2024, "Connecticut"),
]


def _emit_per_seed_baseline(tr, seeds, max_season, log, jdict):
    # Pre-filter to satisfy compute_per_seed_baseline's leak guard
    tr_filtered = tr[tr["Season"] <= max_season]
    baseline = compute_per_seed_baseline(tr_filtered, seeds, max_season=max_season)
    table = []
    for seed in range(1, 17):
        if seed in baseline:
            sub = tr[tr["Season"] <= max_season]
            from src.features.team_history import _rounds_won_per_team_season, _extract_seed_num
            rw = _rounds_won_per_team_season(sub, max_season=None)
            seed_rows = seeds[seeds["Season"] <= max_season].copy()
            seed_rows["seed_num"] = seed_rows["Seed"].apply(_extract_seed_num)
            joined = rw.merge(seed_rows[["Season", "TeamID", "seed_num"]],
                              on=["Season", "TeamID"], how="left")
            n = int((joined["seed_num"] == seed).sum())
            table.append({"seed": seed, "n_observations": n,
                          "expected_rounds_won": baseline[seed]})
    jdict["per_seed_baseline"] = table
    jdict["fallback_baseline"] = baseline["__fallback__"]
    log.append(f"\n=== Per-seed baseline (Season <= {max_season}) ===")
    log.append(f"{'seed':>4} {'n':>5} {'E[rounds_won]':>15}")
    for row in table:
        log.append(f"{row['seed']:>4} {row['n_observations']:>5} {row['expected_rounds_won']:>15.3f}")
    log.append(f"  (fallback for missing seeds: {baseline['__fallback__']:.3f})")


def _emit_champion_residuals(tr, seeds, teams, log, jdict):
    log.append("\n=== 9-champion residual values ===")
    log.append(f"{'Yr':>4} {'Team':<18} | {'cont':>6} {'mom':>6} | priors")
    log.append("-" * 90)
    champ_records = []
    for season, team_name in CHAMPIONS_2015_2024:
        team_row = teams[teams["TeamName"] == team_name]
        if team_row.empty:
            log.append(f"{season:>4} {team_name:<18} | (TeamName not in MTeams.csv)")
            continue
        team_id = int(team_row.iloc[0]["TeamID"])
        # Pre-filter to satisfy leak guard
        tr_filtered = tr[tr["Season"] < season]
        baseline = compute_per_seed_baseline(tr_filtered, seeds, max_season=season - 1)
        residuals = compute_team_residuals_in_window(
            season=season, team_id=team_id, window_years=10,
            baseline=baseline, tourney_results=tr, seeds=seeds,
        )
        mean_v = shrunk_mean([r for (_, _, r) in residuals], k=3)
        ewma_v = shrunk_ewma([(a, r) for (a, _, r) in residuals],
                             half_life=2, k=3)
        priors_str = ", ".join(
            f"yr{a}/sd{s}/r={r:+.2f}" for (a, s, r) in residuals
        )
        log.append(f"{season:>4} {team_name:<18} | {mean_v:>6.2f} {ewma_v:>6.2f} | {priors_str}")
        champ_records.append({
            "season": season, "team_name": team_name, "team_id": team_id,
            "team_seed_residual_mean_10yr": mean_v,
            "team_seed_residual_ewma_hl2": ewma_v,
            "priors": [{"years_ago": a, "prior_seed": s, "residual": r}
                       for (a, s, r) in residuals],
        })
    jdict["champion_residuals"] = champ_records


def _emit_correlation(features_df, incumbent_csv, log, jdict):
    if not Path(incumbent_csv).exists():
        log.append(f"\n=== Correlation matrix: SKIPPED (no {incumbent_csv}) ===")
        return
    incumbent = pd.read_csv(incumbent_csv)
    cols_to_check = ["adj_em", "kp_TALENT", "kp_BARTHAG",
                     "coach_career_f4_apps", "coach_career_winpct",
                     "coach_career_seasons", "season_win_pct", "conf_strength"]
    available = [c for c in cols_to_check if c in incumbent.columns]
    joined = features_df.merge(
        incumbent[["Season", "TeamID"] + available],
        on=["Season", "TeamID"], how="inner",
    )
    log.append("\n=== Pearson correlation: new features vs incumbents ===")
    log.append(f"{'feature':<40} | {' '.join(f'{c:>20}' for c in available)}")
    table = {}
    for new_col in ["team_seed_residual_mean_10yr", "team_seed_residual_ewma_hl2"]:
        corrs = {c: float(joined[new_col].corr(joined[c])) for c in available}
        log.append(f"{new_col:<40} | " + " ".join(f"{corrs[c]:>20.3f}" for c in available))
        table[new_col] = corrs
        # Flag high-correlation entries
        for c, v in corrs.items():
            if abs(v) > 0.85:
                log.append(f"  FLAG: |corr({new_col}, {c})| = {abs(v):.3f} > 0.85")
    jdict["correlation_matrix"] = table


def _emit_distribution(features_df, log, jdict):
    log.append("\n=== Distribution percentiles ===")
    log.append(f"{'feature':<40} | {'p05':>7} {'p25':>7} {'p50':>7} {'p75':>7} {'p95':>7}")
    table = {}
    for col in ["team_seed_residual_mean_10yr", "team_seed_residual_ewma_hl2"]:
        v = features_df[col].values
        pcts = [float(np.percentile(v, p)) for p in (5, 25, 50, 75, 95)]
        log.append(f"{col:<40} | " + " ".join(f"{x:>+7.3f}" for x in pcts))
        table[col] = dict(zip(["p05", "p25", "p50", "p75", "p95"], pcts))
    jdict["distribution_percentiles"] = table


def _emit_top_bottom(features_df, teams, log, jdict, n=10):
    log.append("\n=== Top-10 / bottom-10 by each feature ===")
    teams_lookup = teams.set_index("TeamID")["TeamName"].to_dict()
    table = {}
    for col in ["team_seed_residual_mean_10yr", "team_seed_residual_ewma_hl2"]:
        sorted_df = features_df.sort_values(col, ascending=False)
        top = sorted_df.head(n).copy()
        bot = sorted_df.tail(n).copy()
        top["TeamName"] = top["TeamID"].map(teams_lookup)
        bot["TeamName"] = bot["TeamID"].map(teams_lookup)
        log.append(f"\n  Top-{n} {col}:")
        for _, r in top.iterrows():
            log.append(f"    {int(r['Season']):>4} {r['TeamName']:<22} {r[col]:>+7.3f}")
        log.append(f"  Bottom-{n} {col}:")
        for _, r in bot.iterrows():
            log.append(f"    {int(r['Season']):>4} {r['TeamName']:<22} {r[col]:>+7.3f}")
        table[col] = {
            "top": [{"season": int(r["Season"]), "team_name": r["TeamName"], "value": float(r[col])}
                    for _, r in top.iterrows()],
            "bottom": [{"season": int(r["Season"]), "team_name": r["TeamName"], "value": float(r[col])}
                       for _, r in bot.iterrows()],
        }
    jdict["top_bottom_n"] = table


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--incumbent-csv", default="output/v4_team_features.csv",
                        help="CSV of incumbent v4 team features for correlation check (skipped if absent)")
    parser.add_argument("--seasons", default="2003-2024",
                        help="LOSO season range as 'min-max', inclusive")
    args = parser.parse_args(argv)

    s_min, s_max = (int(x) for x in args.seasons.split("-"))

    tr = pd.read_csv(DATA_DIR / "MNCAATourneyDetailedResults.csv")
    seeds = pd.read_csv(DATA_DIR / "MNCAATourneySeeds.csv")
    teams = pd.read_csv(DATA_DIR / "MTeams.csv")

    log: list[str] = [
        "=== Team-seed-residual Phase 1 diagnostic ===",
        f"Data: {DATA_DIR}",
        f"Tournament rows: {len(tr):,}",
        f"Seed rows: {len(seeds):,}",
        f"LOSO seasons: {s_min}-{s_max}",
    ]
    jdict: dict = {
        "spec": "docs/superpowers/specs/2026-05-09-team-seed-residual-design.md",
        "n_tourney_rows": len(tr),
        "n_seed_rows": len(seeds),
        "loso_seasons": list(range(s_min, s_max + 1)),
    }

    _emit_per_seed_baseline(tr, seeds, max_season=s_max - 1, log=log, jdict=jdict)
    _emit_champion_residuals(tr, seeds, teams, log=log, jdict=jdict)

    # Build features for all (Season, TeamID) in LOSO seasons for distribution + top/bottom
    field = seeds[seeds["Season"].between(s_min, s_max)][["Season", "TeamID"]].drop_duplicates()
    features_df = compute_team_history_features(
        tournament_field=field, tourney_results=tr, seeds=seeds, window_years=10,
    )
    log.append(f"\nFeature DataFrame: {len(features_df):,} rows over seasons {s_min}-{s_max}")

    _emit_correlation(features_df, args.incumbent_csv, log=log, jdict=jdict)
    _emit_distribution(features_df, log=log, jdict=jdict)
    _emit_top_bottom(features_df, teams, log=log, jdict=jdict, n=10)

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(jdict, indent=2))
    OUT_LOG.write_text("\n".join(log) + "\n")
    print(f"Wrote {OUT_JSON} and {OUT_LOG}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
