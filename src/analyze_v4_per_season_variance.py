"""Per-season variance check for v4 across 22 LOSO seasons.

Cheap diagnostic gate before committing engineering budget to
calibration-shape work. Surfaces whether v4's 22-season-aggregate
metrics hide high-variance per-season behavior.

Spec: docs/superpowers/specs/2026-05-07-v4-per-season-variance-design.md

Outputs:
    output/v4_per_season_variance.json
    output/v4_per_season_variance_traces.png
    output/v4_per_season_variance_deltas.png
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Reuse audit drivers' data-load + join helpers. Cross-module coupling
# is intentional per the spec; this is a one-off diagnostic.
from src.audit_v4_gap_vegas import (  # noqa: E402
    DATA,
    _build_day_zero_map,
    _build_per_game_audit_df as _build_audit_df_vegas,
    _build_vegas_lookup,
    _calibration_table,
    _ece,
    _load_seeds_lookup,
    _load_v4_lookup,
    _vegas_to_seasonday,
)
from src.audit_v4_gap_fte import (  # noqa: E402
    _build_fte_lookup,
    _build_per_game_audit_df as _build_audit_df_fte,
    _resolve_fte_team_ids,
)
from src.enhanced_model_v2 import (  # noqa: E402
    _build_vegas_name_to_kaggle_map,
    _resolve_vegas_name,
    load_vegas_lines,
)
from src.ingest.fte_forecasts import _AUDITED_YEARS, load_fte_forecasts  # noqa: E402

logger = logging.getLogger(__name__)

DEFAULT_OUT_JSON = "output/v4_per_season_variance.json"
DEFAULT_OUT_DIR = "output"
DEFAULT_FTE_CACHE = Path("data/raw/fte_forecasts")
DEFAULT_SIGMA = 1.5


# ---------------------------------------------------------------------------
# Per-season aggregation
# ---------------------------------------------------------------------------


def _per_season_metrics(df: pd.DataFrame, ref_label: str) -> pd.DataFrame:
    """Per-season aggregate of v4-vs-<ref> metrics.

    df expected columns: season, p_v4, p_<ref_label>, winner_is_a.
    Returns columns: season, n_games, ll_v4, ll_<ref>, ll_v4_minus_<ref>,
    acc_v4, acc_<ref>, ece_v4, ece_<ref>.
    """
    eps = 1e-15
    ref_col = f"p_{ref_label}"
    rows = []
    for season, sub in df.groupby("season"):
        winner = sub["winner_is_a"].to_numpy()
        p_v4 = sub["p_v4"].to_numpy()
        p_ref = sub[ref_col].to_numpy()

        p_v4_w = np.where(winner == 1, p_v4, 1 - p_v4)
        p_ref_w = np.where(winner == 1, p_ref, 1 - p_ref)
        ll_v4 = float(-np.mean(np.log(np.clip(p_v4_w, eps, 1 - eps))))
        ll_ref = float(-np.mean(np.log(np.clip(p_ref_w, eps, 1 - eps))))

        acc_v4 = float(((p_v4 >= 0.5).astype(int) == winner).mean())
        acc_ref = float(((p_ref >= 0.5).astype(int) == winner).mean())

        cal_v4 = _calibration_table(p_v4, winner)
        cal_ref = _calibration_table(p_ref, winner)

        rows.append({
            "season": int(season),
            "n_games": int(len(sub)),
            "ll_v4": ll_v4,
            f"ll_{ref_label}": ll_ref,
            f"ll_v4_minus_{ref_label}": ll_v4 - ll_ref,
            "acc_v4": acc_v4,
            f"acc_{ref_label}": acc_ref,
            "ece_v4": _ece(cal_v4),
            f"ece_{ref_label}": _ece(cal_ref),
        })
    return pd.DataFrame(rows).sort_values("season").reset_index(drop=True)


def _flag_outliers(
    df: pd.DataFrame, columns: list[str], sigma: float = DEFAULT_SIGMA,
) -> dict:
    """Flag rows where (column - mean) / std >= sigma. Sample std (ddof=1).

    Returns: {column: [{season, value, sigma_delta, n_games}, ...]}.
    Empty list for missing columns or series too short to estimate std.
    """
    out: dict[str, list[dict]] = {}
    for col in columns:
        if col not in df.columns:
            continue
        vals = df[col].dropna()
        if len(vals) < 2:
            out[col] = []
            continue
        mean = float(vals.mean())
        std = float(vals.std(ddof=1))
        if std == 0.0 or not np.isfinite(std):
            out[col] = []
            continue
        flagged = []
        for _, row in df.iterrows():
            if pd.isna(row[col]):
                continue
            z = (float(row[col]) - mean) / std
            if z >= sigma:
                flagged.append({
                    "season": int(row["season"]),
                    "value": float(row[col]),
                    "sigma_delta": float(z),
                    "n_games": int(row["n_games"]),
                })
        flagged.sort(key=lambda x: -x["sigma_delta"])
        out[col] = flagged
    return out


def _pick_verdict(
    df: pd.DataFrame, outliers: dict, sigma: float,
) -> dict:
    """Pick one of {flat, outlier, trend, mixed} based on outlier counts.

    - flat: no flags on any tracked column.
    - outlier: 1-2 distinct seasons flagged across cross-benchmark deltas
      (ll_v4_minus_vegas, ll_v4_minus_fte).
    - trend: 3+ flagged seasons consecutive (monotonic season order) on
      a cross-benchmark delta.
    - mixed: anything else.
    """
    cross_keys = ["ll_v4_minus_vegas", "ll_v4_minus_fte"]
    intra_keys = ["ll_v4", "ece_v4"]

    flagged_cross_seasons: set[int] = set()
    for k in cross_keys:
        for entry in outliers.get(k, []):
            flagged_cross_seasons.add(int(entry["season"]))

    flagged_intra_seasons: set[int] = set()
    for k in intra_keys:
        for entry in outliers.get(k, []):
            flagged_intra_seasons.add(int(entry["season"]))

    all_flagged = flagged_cross_seasons | flagged_intra_seasons

    if not all_flagged:
        return {
            "label": "flat",
            "summary": (
                f"No season exceeds {sigma} sigma on any cross-benchmark "
                f"delta or intra-v4 metric. Aggregate calibration is the "
                f"likely bottleneck."
            ),
            "outlier_seasons": [],
        }

    # Detect trend: 3+ consecutive seasons on a cross-benchmark delta.
    sorted_seasons = sorted(flagged_cross_seasons)
    is_trend = False
    if len(sorted_seasons) >= 3:
        all_seasons = sorted(df["season"].unique())
        season_index = {s: i for i, s in enumerate(all_seasons)}
        sorted_idx = sorted(season_index[s] for s in sorted_seasons)
        for i in range(len(sorted_idx) - 2):
            if sorted_idx[i + 1] == sorted_idx[i] + 1 and sorted_idx[i + 2] == sorted_idx[i] + 2:
                is_trend = True
                break

    if is_trend:
        return {
            "label": "trend",
            "summary": (
                f"3+ consecutive seasons flagged on cross-benchmark delta. "
                f"Investigate gradual calibration drift (data pipeline, "
                f"rule changes, era effects)."
            ),
            "outlier_seasons": sorted_seasons,
        }

    if 1 <= len(all_flagged) <= 2:
        return {
            "label": "outlier",
            "summary": (
                f"{len(all_flagged)} season(s) exceed {sigma} sigma. "
                f"Investigate what's distinctive about these tournaments "
                f"before fixing aggregate calibration."
            ),
            "outlier_seasons": sorted(all_flagged),
        }

    return {
        "label": "mixed",
        "summary": (
            f"{len(all_flagged)} seasons flagged but no clean trend. "
            f"Findings note must call out the pattern."
        ),
        "outlier_seasons": sorted(all_flagged),
    }


# ---------------------------------------------------------------------------
# Data load (reuse audit drivers' join pipelines)
# ---------------------------------------------------------------------------


def _build_vegas_per_game_df(v4_csv: str) -> pd.DataFrame:
    """Build per-game (season, p_v4, p_vegas, winner_is_a, round, ...)
    frame by reusing the Vegas audit's pipeline. 22 seasons of R64-Champ.
    """
    logger.info("loading v4 pairwise + Vegas lines + tournament outcomes ...")
    v4_lookup = _load_v4_lookup(v4_csv)
    results = pd.read_csv(DATA / "MNCAATourneyCompactResults.csv")
    seeds_lookup = _load_seeds_lookup(DATA / "MNCAATourneySeeds.csv")
    day_zero = _build_day_zero_map(DATA / "MSeasons.csv")

    vegas_df = load_vegas_lines()
    teams = pd.read_csv(DATA / "MTeams.csv")
    spellings = pd.read_csv(DATA / "MTeamSpellings.csv", encoding="latin-1")
    name_to_id = _build_vegas_name_to_kaggle_map(teams, spellings)

    fuzzy_cache: dict = {}
    all_names = set(vegas_df["home"].unique()) | set(vegas_df["road"].unique())
    name_resolution = {}
    for name in all_names:
        tid = _resolve_vegas_name(name, name_to_id, fuzzy_cache)
        if tid is not None:
            name_resolution[name] = tid

    vegas_df = _vegas_to_seasonday(vegas_df, day_zero)
    vegas_lookup = _build_vegas_lookup(vegas_df, name_resolution)

    df = _build_audit_df_vegas(v4_lookup, vegas_lookup, results, seeds_lookup)
    # Vegas audit includes FF in by_round buckets but downstream metrics
    # filter out small-n; for variance check we exclude FF + OTHER explicitly.
    df = df[df["round"].isin(["R64", "R32", "S16", "E8", "F4", "Champ"])]
    return df.reset_index(drop=True)


def _build_fte_per_game_df(
    v4_csv: str, fte_cache_dir: Path,
) -> pd.DataFrame:
    """Build per-game (season, p_v4, p_fte, winner_is_a, round, ...) frame
    by reusing the 538 audit's pipeline. 7 seasons of R64-Champ."""
    logger.info("loading 538 forecasts ...")
    v4_lookup = _load_v4_lookup(v4_csv)
    fte_df = load_fte_forecasts(years=_AUDITED_YEARS, cache_dir=fte_cache_dir)
    teams = pd.read_csv(DATA / "MTeams.csv")
    spellings = pd.read_csv(DATA / "MTeamSpellings.csv", encoding="latin-1")
    fte_resolved, _ = _resolve_fte_team_ids(fte_df, teams, spellings)
    fte_lookup = _build_fte_lookup(fte_resolved)
    results = pd.read_csv(DATA / "MNCAATourneyCompactResults.csv")
    seeds_lookup = _load_seeds_lookup(DATA / "MNCAATourneySeeds.csv")
    df = _build_audit_df_fte(v4_lookup, fte_lookup, results, seeds_lookup)
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def _plot_traces(merged: pd.DataFrame, out_path: Path) -> None:
    """3 panels (LL, accuracy, ECE), x = season, lines for v4 / Vegas / 538."""
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    metrics = [
        ("ll", ["ll_v4", "ll_vegas", "ll_fte"], "log loss (lower is better)"),
        ("acc", ["acc_v4", "acc_vegas", "acc_fte"], "accuracy"),
        ("ece", ["ece_v4", "ece_vegas", "ece_fte"], "ECE"),
    ]
    labels = {"ll_v4": "v4", "ll_vegas": "Vegas", "ll_fte": "538",
              "acc_v4": "v4", "acc_vegas": "Vegas", "acc_fte": "538",
              "ece_v4": "v4", "ece_vegas": "Vegas", "ece_fte": "538"}
    colors = {"v4": "C0", "Vegas": "C1", "538": "C2"}
    for ax, (_kind, cols, ylabel) in zip(axes, metrics):
        for col in cols:
            if col not in merged.columns:
                continue
            sub = merged[["season", col]].dropna()
            if len(sub) == 0:
                continue
            label = labels[col]
            ax.plot(sub["season"], sub[col], marker="o", label=label,
                    color=colors[label])
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)
    axes[-1].set_xlabel("season")
    fig.suptitle("Per-season metrics: v4 vs Vegas vs 538")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _plot_deltas(merged: pd.DataFrame, outliers: dict, out_path: Path) -> None:
    """2 panels (LL_v4 - LL_vegas, LL_v4 - LL_fte) per season; bars red for
    outlier-flagged seasons."""
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for ax, (col, title) in zip(axes, [
        ("ll_v4_minus_vegas", "v4 - Vegas LL per season (positive = v4 worse)"),
        ("ll_v4_minus_fte",   "v4 - 538  LL per season (positive = v4 worse)"),
    ]):
        if col not in merged.columns:
            ax.text(0.5, 0.5, f"{col} not available", ha="center", va="center")
            ax.axis("off")
            continue
        sub = merged[["season", col]].dropna()
        flagged_seasons = {e["season"] for e in outliers.get(col, [])}
        colors_per_bar = ["red" if int(s) in flagged_seasons else "steelblue"
                          for s in sub["season"]]
        ax.bar(sub["season"], sub[col], color=colors_per_bar)
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_title(title)
        ax.set_ylabel("delta LL")
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("season")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


_TRACKED_COLUMNS = ["ll_v4_minus_vegas", "ll_v4_minus_fte", "ll_v4", "ece_v4"]


def run_analysis(
    v4_csv: str = "output/pairwise_v4.csv",
    out_dir: str | Path = DEFAULT_OUT_DIR,
    out_json: str = DEFAULT_OUT_JSON,
    fte_cache_dir: str | Path = DEFAULT_FTE_CACHE,
    sigma_threshold: float = DEFAULT_SIGMA,
) -> dict:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("v4 PER-SEASON VARIANCE CHECK")
    print("=" * 70)

    vegas_df = _build_vegas_per_game_df(v4_csv)
    print(f"  v4-vs-Vegas frame: {len(vegas_df)} games "
          f"across {vegas_df['season'].nunique()} seasons")

    fte_df = _build_fte_per_game_df(v4_csv, Path(fte_cache_dir))
    print(f"  v4-vs-538   frame: {len(fte_df)} games "
          f"across {fte_df['season'].nunique()} seasons")

    print("computing per-season metrics ...")
    per_season_vegas = _per_season_metrics(vegas_df, ref_label="vegas")
    per_season_fte = _per_season_metrics(fte_df, ref_label="fte")

    # Merge: left-join 538 onto Vegas (538 is a 7-season subset of Vegas's 22).
    merged = per_season_vegas.merge(
        per_season_fte.drop(columns=["ll_v4", "acc_v4", "ece_v4", "n_games"]),
        on="season", how="left",
    )

    outliers = _flag_outliers(merged, _TRACKED_COLUMNS, sigma=sigma_threshold)

    # Sanity anchors -- weighted aggregate matches audit overall numbers.
    eps = 1e-15

    def _weighted_ll(per_season: pd.DataFrame, col: str) -> float:
        sub = per_season[["n_games", col]].dropna()
        return float(np.average(sub[col], weights=sub["n_games"]))

    anchors = {
        "weighted_ll_v4_vs_vegas_audit_subset": _weighted_ll(per_season_vegas, "ll_v4"),
        "weighted_ll_vegas_vs_audit": _weighted_ll(per_season_vegas, "ll_vegas"),
        "weighted_ll_v4_vs_fte_audit_subset": _weighted_ll(per_season_fte, "ll_v4"),
        "weighted_ll_fte_vs_audit": _weighted_ll(per_season_fte, "ll_fte"),
    }

    verdict = _pick_verdict(merged, outliers, sigma=sigma_threshold)

    summary = {
        "config": {
            "v4_pairwise": str(v4_csv),
            "fte_cache_dir": str(fte_cache_dir),
            "sigma_threshold": sigma_threshold,
            "tracked_columns": _TRACKED_COLUMNS,
        },
        "per_season": merged.to_dict(orient="records"),
        "cross_season_summary": {
            col: {
                "mean": float(merged[col].dropna().mean()) if col in merged.columns else None,
                "std": float(merged[col].dropna().std(ddof=1)) if col in merged.columns else None,
                "n": int(merged[col].dropna().shape[0]) if col in merged.columns else 0,
            }
            for col in _TRACKED_COLUMNS
        },
        "outliers": outliers,
        "anchors": anchors,
        "verdict": verdict,
    }

    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)

    _plot_traces(merged, out_path / "v4_per_season_variance_traces.png")
    _plot_deltas(merged, outliers, out_path / "v4_per_season_variance_deltas.png")

    print()
    print("=" * 70)
    print(f"VERDICT: {verdict['label'].upper()}")
    print("=" * 70)
    print(f"  {verdict['summary']}")
    if verdict["outlier_seasons"]:
        print(f"  outlier seasons: {verdict['outlier_seasons']}")
    print()
    print("ANCHORS (weighted per-season mean -- compare to audit overall):")
    for k, v in anchors.items():
        print(f"  {k:55s} = {v:.4f}")
    print()
    print(f"  saved {out_json}")
    print(f"  saved 2 PNGs in {out_path}/")

    return summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--v4", default="output/pairwise_v4.csv")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--out-json", default=DEFAULT_OUT_JSON)
    parser.add_argument("--fte-cache", default=str(DEFAULT_FTE_CACHE))
    parser.add_argument("--sigma", type=float, default=DEFAULT_SIGMA)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    run_analysis(
        v4_csv=args.v4,
        out_dir=args.out_dir,
        out_json=args.out_json,
        fte_cache_dir=args.fte_cache,
        sigma_threshold=args.sigma,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
