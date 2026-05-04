"""Audit v4's tournament-game predictions vs Vegas closing-line implied
probabilities, broken down by round, higher-vs-lower-seed status,
v4-confidence quintile, and seed-difference magnitude.

Spec: docs/superpowers/specs/2026-05-04-v4-gap-audit-vegas-design.md

Output:
    output/v4_gap_audit_vegas.json
    output/v4_gap_calibration_overall.png
    output/v4_gap_calibration_by_round.png
    output/v4_gap_per_bucket_ll_delta.png
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.stats import norm

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.enhanced_model_v2 import (
    load_vegas_lines,
    _build_vegas_name_to_kaggle_map,
    _resolve_vegas_name,
)

SIGMA = 11.0  # CBB spread-to-prob sigma; matches src/blend_sweep.py + alternate_bracket.py
DATA = Path("data/raw/march-machine-learning-2026")
DEFAULT_OUT_JSON = "output/v4_gap_audit_vegas.json"
DEFAULT_OUT_DIR = "output"

# Tournament round inference from Kaggle DayNum.
ROUND_BY_DAYNUM = {
    134: "FF", 135: "FF",
    136: "R64", 137: "R64",
    138: "R32", 139: "R32",
    143: "S16", 144: "S16",
    145: "E8",  146: "E8",
    152: "F4",  153: "F4",
    154: "Champ",
}

CONFIDENCE_BIN_EDGES = [0.50, 0.60, 0.70, 0.80, 0.90, 1.00]
CONFIDENCE_BIN_LABELS = [
    "0.50-0.60", "0.60-0.70", "0.70-0.80", "0.80-0.90", "0.90-1.00",
]

ROUND_ORDER = ["FF", "R64", "R32", "S16", "E8", "F4", "Champ"]

WEAK_SPOT_MIN_N = 50
WEAK_SPOT_MIN_LL_DELTA = 0.02


def _spread_to_prob(spread: float, sigma: float = SIGMA) -> float:
    """Convert a closing point spread to win probability for the home/
    favored side via N(0, sigma).

    Positive spread = home favored.
    """
    return float(norm.cdf(float(spread) / sigma))


def _v4_confidence_quintile(p_a: float) -> str:
    """Map predicted prob (for either side) to confidence quintile of the
    favored side (>= 0.5)."""
    p_fav = max(float(p_a), 1.0 - float(p_a))
    for lo, hi, label in zip(
        CONFIDENCE_BIN_EDGES[:-1],
        CONFIDENCE_BIN_EDGES[1:],
        CONFIDENCE_BIN_LABELS,
    ):
        if lo <= p_fav <= hi:
            return label
    return CONFIDENCE_BIN_LABELS[-1]


def _seed_diff_bucket(d: int) -> str:
    d = int(abs(d))
    if d <= 2:
        return "0-2"
    if d <= 5:
        return "3-5"
    if d <= 9:
        return "6-9"
    return "10-15"


def _round_from_daynum(daynum: int) -> str:
    return ROUND_BY_DAYNUM.get(int(daynum), "OTHER")


_SEED_NUMBER_RE = re.compile(r"[A-Z](\d+)")


def _extract_seed_number(seed_str: str) -> int:
    """Seed string like 'W01', 'X16a' -> integer 1, 16."""
    m = _SEED_NUMBER_RE.match(str(seed_str))
    if not m:
        raise ValueError(f"unrecognized seed format: {seed_str!r}")
    return int(m.group(1))


def _calibration_table(
    p_pred: np.ndarray,
    y_actual: np.ndarray,
    n_bins: int = 10,
) -> list[dict]:
    """Per-bin (predicted-mid, empirical-rate, n) over [0, 1]."""
    p_pred = np.asarray(p_pred, dtype=np.float64)
    y_actual = np.asarray(y_actual, dtype=np.int64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    out = []
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        if i == n_bins - 1:
            mask = (p_pred >= lo) & (p_pred <= hi)
        else:
            mask = (p_pred >= lo) & (p_pred < hi)
        n = int(mask.sum())
        empirical = float(y_actual[mask].mean()) if n > 0 else None
        out.append({
            "bin": [float(lo), float(hi)],
            "mid": float((lo + hi) / 2),
            "n": n,
            "empirical": empirical,
        })
    return out


def _ece(cal_table: list[dict]) -> float:
    """Expected calibration error, weighted by per-bin counts."""
    n_total = sum(b["n"] for b in cal_table)
    if n_total == 0:
        return float("nan")
    s = 0.0
    for b in cal_table:
        if b["empirical"] is None:
            continue
        s += (b["n"] / n_total) * abs(b["mid"] - b["empirical"])
    return float(s)


def _compute_bucket_metrics(df: pd.DataFrame, bucket_col: str) -> dict:
    """For each unique value of df[bucket_col], compute LL + acc + cal."""
    out = {}
    for value, sub in df.groupby(bucket_col):
        n = len(sub)
        if n == 0:
            continue
        eps = 1e-15
        winner = sub["winner_is_a"].to_numpy()
        p_v4 = sub["p_v4"].to_numpy()
        p_ve = sub["p_vegas"].to_numpy()

        p_v4_w = np.where(winner == 1, p_v4, 1 - p_v4)
        p_ve_w = np.where(winner == 1, p_ve, 1 - p_ve)
        ll_v4 = float(-np.mean(np.log(np.clip(p_v4_w, eps, 1 - eps))))
        ll_ve = float(-np.mean(np.log(np.clip(p_ve_w, eps, 1 - eps))))

        acc_v4 = float(((p_v4 >= 0.5).astype(int) == winner).mean())
        acc_ve = float(((p_ve >= 0.5).astype(int) == winner).mean())

        cal_v4 = _calibration_table(p_v4, winner)
        cal_ve = _calibration_table(p_ve, winner)

        out[str(value)] = {
            "n_games": int(n),
            "ll_v4": ll_v4,
            "ll_vegas": ll_ve,
            "ll_delta": float(ll_v4 - ll_ve),
            "acc_v4": acc_v4,
            "acc_vegas": acc_ve,
            "ece_v4": _ece(cal_v4),
            "ece_vegas": _ece(cal_ve),
            "mean_p_v4_minus_vegas": float((p_v4 - p_ve).mean()),
            "calibration_v4": cal_v4,
            "calibration_vegas": cal_ve,
        }
    return out


# ---------------------------------------------------------------------------
# Data join
# ---------------------------------------------------------------------------


def _parse_mdy(s) -> datetime | None:
    """Parse 'MM/DD/YYYY' or return None for missing/malformed values."""
    if s is None:
        return None
    try:
        s_str = str(s).strip()
    except Exception:
        return None
    if not s_str or s_str.lower() in ("nan", "nat", "none"):
        return None
    try:
        return datetime.strptime(s_str, "%m/%d/%Y")
    except ValueError:
        return None


def _build_day_zero_map(seasons_csv: Path) -> dict[int, datetime]:
    """{Season: DayZero (datetime)}."""
    df = pd.read_csv(seasons_csv)
    out = {}
    for _, r in df.iterrows():
        dz = _parse_mdy(r["DayZero"])
        if dz is not None:
            out[int(r["Season"])] = dz
    return out


def _vegas_to_seasonday(
    vegas_df: pd.DataFrame,
    day_zero_by_season: dict[int, datetime],
) -> pd.DataFrame:
    """Add daynum column to vegas_df by parsing date and looking up
    DayZero per row's season. Rows with unparseable dates get NaN
    daynum and are filtered downstream."""
    vegas_df = vegas_df.copy()
    parsed = vegas_df["date"].apply(_parse_mdy)
    vegas_df["date_parsed"] = parsed
    n_bad = int(parsed.isna().sum())
    if n_bad:
        print(f"  warning: {n_bad} Vegas rows have unparseable dates; dropping")
    vegas_df = vegas_df.dropna(subset=["date_parsed"]).copy()

    daynums = []
    for season, date in zip(vegas_df["season"], vegas_df["date_parsed"]):
        dz = day_zero_by_season.get(int(season))
        if dz is None:
            daynums.append(np.nan)
        else:
            daynums.append((date - dz).days)
    vegas_df["daynum"] = pd.Series(daynums, index=vegas_df.index)
    return vegas_df


def _build_vegas_lookup(
    vegas_df: pd.DataFrame,
    name_resolution: dict[str, int],
) -> dict:
    """{(season, daynum, min_id, max_id): p_a_wins} where team_a is the
    one with the smaller TeamID. Drops rows where either team can't be
    resolved or daynum is NaN."""
    lookup = {}
    for _, row in vegas_df.iterrows():
        season = int(row["season"])
        if pd.isna(row["daynum"]):
            continue
        daynum = int(row["daynum"])
        home_id = name_resolution.get(row["home"])
        road_id = name_resolution.get(row["road"])
        line = row["line"]
        if home_id is None or road_id is None:
            continue
        if pd.isna(line):
            continue

        a, b = (home_id, road_id) if home_id < road_id else (road_id, home_id)
        # Vegas line is for home team. If home is team_a (smaller id),
        # p_a_wins = norm.cdf(line/sigma). Else p_a_wins = 1 - norm.cdf(line/sigma).
        p_home = _spread_to_prob(float(line))
        if home_id == a:
            p_a_wins = p_home
        else:
            p_a_wins = 1.0 - p_home
        lookup[(season, daynum, int(a), int(b))] = float(p_a_wins)
    return lookup


def _find_vegas_p(
    lookup: dict, season: int, daynum: int, a: int, b: int, day_slack: int = 1,
) -> float | None:
    """Look up Vegas p_a_wins with +/- day_slack tolerance."""
    for delta in range(-day_slack, day_slack + 1):
        key = (season, daynum + delta, a, b)
        if key in lookup:
            return lookup[key]
    return None


def _build_per_game_audit_df(
    v4_lookup: dict,
    vegas_lookup: dict,
    results_df: pd.DataFrame,
    seeds_lookup: dict,
) -> pd.DataFrame:
    """Per-game DataFrame: columns season, daynum, team_a, team_b, p_v4,
    p_vegas, winner_is_a, round, higher_seed_won, seed_a, seed_b,
    seed_diff_bucket, v4_confidence_quintile."""
    rows = []
    for _, g in results_df.iterrows():
        season = int(g["Season"])
        daynum = int(g["DayNum"])
        if season < 2003 or season > 2025:
            continue
        # Skip play-in games (FF) -- not part of the standard 64-team bracket
        # for our v4 pairwise (which is over the tournament field including
        # play-in winners). Keep the option open: the audit includes FF if
        # both predictions exist, but defaults to skipping.
        w, l = int(g["WTeamID"]), int(g["LTeamID"])
        a, b = (w, l) if w < l else (l, w)
        p_v4 = v4_lookup.get((season, a, b))
        if p_v4 is None:
            continue
        p_vegas = _find_vegas_p(vegas_lookup, season, daynum, a, b)
        if p_vegas is None:
            continue

        winner_is_a = 1 if w == a else 0

        seed_a = seeds_lookup.get((season, a))
        seed_b = seeds_lookup.get((season, b))
        if seed_a is None or seed_b is None:
            # Skip games where either team isn't seeded (shouldn't happen
            # for tournament games, but defensive)
            continue
        seed_a_num = _extract_seed_number(seed_a)
        seed_b_num = _extract_seed_number(seed_b)
        seed_diff = abs(seed_a_num - seed_b_num)

        # Higher seed won = team with higher seed number won (= lower seed
        # in basketball lingo, which is the "underdog"). We invert: True
        # means CHALK won (lower seed number = better seed); False = upset.
        # Convention here: "higher_seed_won" labels chalk True / upset False.
        seed_a_won = (winner_is_a == 1)
        seed_b_won = not seed_a_won
        if seed_a_num == seed_b_num:
            chalk_won = True  # toss-up; treat as chalk
        elif seed_a_num < seed_b_num:
            # team_a is the better seed (lower number)
            chalk_won = seed_a_won
        else:
            chalk_won = seed_b_won

        rows.append({
            "season": season,
            "daynum": daynum,
            "team_a": a,
            "team_b": b,
            "p_v4": p_v4,
            "p_vegas": p_vegas,
            "winner_is_a": winner_is_a,
            "round": _round_from_daynum(daynum),
            "seed_a_num": seed_a_num,
            "seed_b_num": seed_b_num,
            "seed_diff": seed_diff,
            "seed_diff_bucket": _seed_diff_bucket(seed_diff),
            "chalk_won": "chalk" if chalk_won else "upset",
            "v4_confidence_quintile": _v4_confidence_quintile(p_v4),
        })
    return pd.DataFrame(rows)


def _load_v4_lookup(v4_csv: str) -> dict:
    """Dedup'd dict: {(season, team_a, team_b): p_a_wins} where a < b."""
    df = pd.read_csv(v4_csv).drop_duplicates(
        ["season", "team_a", "team_b"], keep="last"
    )
    return {
        (int(s), int(a), int(b)): float(p)
        for s, a, b, p in zip(df.season, df.team_a, df.team_b, df.p_a_wins)
    }


def _load_seeds_lookup(seeds_csv: Path) -> dict:
    """{(Season, TeamID): seed_str}."""
    df = pd.read_csv(seeds_csv)
    return {(int(r["Season"]), int(r["TeamID"])): r["Seed"]
            for _, r in df.iterrows()}


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def _plot_calibration_overall(audit_df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    for label, prob_col in [("v4", "p_v4"), ("Vegas", "p_vegas")]:
        cal = _calibration_table(
            audit_df[prob_col].to_numpy(),
            audit_df["winner_is_a"].to_numpy(),
            n_bins=10,
        )
        xs, ys, ns = [], [], []
        for b in cal:
            if b["empirical"] is None:
                continue
            xs.append(b["mid"])
            ys.append(b["empirical"])
            ns.append(b["n"])
        ax.plot(xs, ys, marker="o", label=label)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="diagonal")
    ax.set_xlabel("predicted P(team_a wins)")
    ax.set_ylabel("empirical win rate")
    ax.set_title("v4 vs Vegas calibration on 2003-2025 tournament games")
    ax.legend(loc="upper left")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _plot_calibration_by_round(audit_df: pd.DataFrame, out_path: Path) -> None:
    rounds = [r for r in ROUND_ORDER if r in audit_df["round"].unique()]
    n_rounds = len(rounds)
    cols = 3
    rows = (n_rounds + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = np.array(axes).reshape(-1)
    for i, round_name in enumerate(rounds):
        ax = axes[i]
        sub = audit_df[audit_df["round"] == round_name]
        for label, prob_col in [("v4", "p_v4"), ("Vegas", "p_vegas")]:
            cal = _calibration_table(
                sub[prob_col].to_numpy(), sub["winner_is_a"].to_numpy(), n_bins=8,
            )
            xs, ys = [], []
            for b in cal:
                if b["empirical"] is None:
                    continue
                xs.append(b["mid"])
                ys.append(b["empirical"])
            ax.plot(xs, ys, marker="o", label=label)
        ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_title(f"{round_name} (n={len(sub)})")
        ax.legend(loc="upper left", fontsize=8)
    for j in range(n_rounds, len(axes)):
        axes[j].axis("off")
    fig.suptitle("v4 vs Vegas calibration by round")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _plot_per_bucket_ll_delta(by_axis: dict, out_path: Path) -> None:
    """Horizontal bar chart of ll_delta (v4 - Vegas) per bucket cell.
    by_axis: dict of axis_name -> {value: cell_metrics}."""
    rows = []
    for axis_name, axis_dict in by_axis.items():
        for val, cell in axis_dict.items():
            if cell["n_games"] < WEAK_SPOT_MIN_N:
                continue
            rows.append({
                "label": f"{axis_name}={val} (n={cell['n_games']})",
                "ll_delta": cell["ll_delta"],
            })
    if not rows:
        return
    df = pd.DataFrame(rows).sort_values("ll_delta", ascending=True)
    fig, ax = plt.subplots(figsize=(8, max(4, 0.35 * len(df))))
    colors = ["red" if d > 0 else "steelblue" for d in df["ll_delta"]]
    ax.barh(df["label"], df["ll_delta"], color=colors)
    ax.axvline(0, color="black", linewidth=0.7)
    ax.axvline(WEAK_SPOT_MIN_LL_DELTA, color="red", linestyle="--", alpha=0.5,
               label=f"weak-spot threshold (+{WEAK_SPOT_MIN_LL_DELTA})")
    ax.set_xlabel("ll_v4 - ll_vegas (positive = v4 worse)")
    ax.set_title("Per-bucket LL delta (n >= %d)" % WEAK_SPOT_MIN_N)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _identify_weak_spots(by_axes: dict) -> list[dict]:
    """Top buckets sorted by ll_delta with n >= WEAK_SPOT_MIN_N and
    ll_delta >= WEAK_SPOT_MIN_LL_DELTA."""
    rows = []
    for axis_name, axis_dict in by_axes.items():
        for val, cell in axis_dict.items():
            if cell["n_games"] < WEAK_SPOT_MIN_N:
                continue
            if cell["ll_delta"] < WEAK_SPOT_MIN_LL_DELTA:
                continue
            rows.append({
                "axis": axis_name,
                "value": val,
                "n_games": cell["n_games"],
                "ll_v4": cell["ll_v4"],
                "ll_vegas": cell["ll_vegas"],
                "ll_delta": cell["ll_delta"],
                "acc_v4": cell["acc_v4"],
                "acc_vegas": cell["acc_vegas"],
                "mean_p_v4_minus_vegas": cell["mean_p_v4_minus_vegas"],
            })
    rows.sort(key=lambda r: r["ll_delta"], reverse=True)
    return rows[:10]


def run_audit(
    v4_csv: str = "output/pairwise_v4.csv",
    out_dir: str | Path = DEFAULT_OUT_DIR,
    out_json: str = DEFAULT_OUT_JSON,
) -> dict:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("v4 vs Vegas TOURNAMENT AUDIT")
    print("=" * 70)

    # Load inputs
    print("loading v4 pairwise ...", flush=True)
    v4_lookup = _load_v4_lookup(v4_csv)
    print(f"  {len(v4_lookup):,} unique (season, a, b) keys")

    print("loading tournament results + seeds + seasons ...", flush=True)
    results = pd.read_csv(DATA / "MNCAATourneyCompactResults.csv")
    seeds_lookup = _load_seeds_lookup(DATA / "MNCAATourneySeeds.csv")
    day_zero = _build_day_zero_map(DATA / "MSeasons.csv")
    print(f"  {len(results):,} tournament games across "
          f"{results['Season'].nunique()} seasons")

    print("loading Vegas lines ...", flush=True)
    t0 = time.time()
    vegas_df = load_vegas_lines()
    print(f"  {len(vegas_df):,} Vegas-line rows in {time.time() - t0:.1f}s")

    print("resolving Vegas team names ...", flush=True)
    teams = pd.read_csv(DATA / "MTeams.csv")
    spellings = pd.read_csv(DATA / "MTeamSpellings.csv", encoding="latin-1")
    name_to_id = _build_vegas_name_to_kaggle_map(teams, spellings)
    fuzzy_cache = {}
    all_names = set(vegas_df["home"].unique()) | set(vegas_df["road"].unique())
    name_resolution = {}
    for name in all_names:
        tid = _resolve_vegas_name(name, name_to_id, fuzzy_cache)
        if tid is not None:
            name_resolution[name] = tid
    print(f"  resolved {len(name_resolution)} / {len(all_names)} unique names")

    print("converting Vegas dates to (season, daynum) ...", flush=True)
    vegas_df = _vegas_to_seasonday(vegas_df, day_zero)
    print("building Vegas lookup ...", flush=True)
    vegas_lookup = _build_vegas_lookup(vegas_df, name_resolution)
    print(f"  {len(vegas_lookup):,} (season, daynum, a, b) entries")

    print("joining v4 + Vegas + tournament outcomes ...", flush=True)
    audit_df = _build_per_game_audit_df(
        v4_lookup, vegas_lookup, results, seeds_lookup,
    )
    print(f"  {len(audit_df)} games joined")

    # Coverage report
    n_tourney = int(((results["Season"] >= 2003) & (results["Season"] <= 2025))
                    .sum())
    coverage = {
        "n_tournament_games_2003_2025": n_tourney,
        "n_joined": int(len(audit_df)),
        "coverage_pct": float(len(audit_df) / max(n_tourney, 1) * 100.0),
        "by_season": {
            int(s): int(((results["Season"] == s)
                         & (results["Season"] >= 2003)
                         & (results["Season"] <= 2025)).sum()
                        - len(audit_df[audit_df["season"] == s]))
            for s in sorted(audit_df["season"].unique())
        },
    }
    coverage["missing_by_season"] = {
        int(s): int(((results["Season"] == s).sum())
                    - len(audit_df[audit_df["season"] == s]))
        for s in sorted(set(results["Season"].unique())
                        & set(range(2003, 2026)))
    }
    print(f"  coverage: {coverage['coverage_pct']:.1f}%")

    if coverage["coverage_pct"] < 60.0:
        print(f"  *** WARNING: coverage below 60% halt threshold ***")

    # Overall metrics
    print("computing overall + per-bucket metrics ...", flush=True)
    eps = 1e-15
    p_v4 = audit_df["p_v4"].to_numpy()
    p_ve = audit_df["p_vegas"].to_numpy()
    winner = audit_df["winner_is_a"].to_numpy()
    p_v4_w = np.where(winner == 1, p_v4, 1 - p_v4)
    p_ve_w = np.where(winner == 1, p_ve, 1 - p_ve)

    cal_v4_overall = _calibration_table(p_v4, winner)
    cal_ve_overall = _calibration_table(p_ve, winner)

    overall = {
        "n_games": int(len(audit_df)),
        "ll_v4": float(-np.mean(np.log(np.clip(p_v4_w, eps, 1 - eps)))),
        "ll_vegas": float(-np.mean(np.log(np.clip(p_ve_w, eps, 1 - eps)))),
        "acc_v4": float(((p_v4 >= 0.5).astype(int) == winner).mean()),
        "acc_vegas": float(((p_ve >= 0.5).astype(int) == winner).mean()),
        "ece_v4": _ece(cal_v4_overall),
        "ece_vegas": _ece(cal_ve_overall),
        "calibration_v4": cal_v4_overall,
        "calibration_vegas": cal_ve_overall,
    }

    by_round = _compute_bucket_metrics(audit_df, "round")
    by_chalk_upset = _compute_bucket_metrics(audit_df, "chalk_won")
    by_confidence = _compute_bucket_metrics(audit_df, "v4_confidence_quintile")
    by_seed_diff = _compute_bucket_metrics(audit_df, "seed_diff_bucket")

    by_axes = {
        "round": by_round,
        "chalk_won": by_chalk_upset,
        "v4_confidence_quintile": by_confidence,
        "seed_diff_bucket": by_seed_diff,
    }

    weak_spots = _identify_weak_spots(by_axes)

    summary = {
        "config": {
            "v4_pairwise": v4_csv,
            "sigma": SIGMA,
            "seasons": sorted(audit_df["season"].unique().tolist()),
            "weak_spot_min_n": WEAK_SPOT_MIN_N,
            "weak_spot_min_ll_delta": WEAK_SPOT_MIN_LL_DELTA,
        },
        "join_coverage": coverage,
        "overall": overall,
        "by_round": by_round,
        "by_chalk_won": by_chalk_upset,
        "by_v4_confidence_quintile": by_confidence,
        "by_seed_diff_bucket": by_seed_diff,
        "weak_spots": weak_spots,
    }

    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)

    # Plots
    print("emitting plots ...", flush=True)
    _plot_calibration_overall(audit_df, out_path / "v4_gap_calibration_overall.png")
    _plot_calibration_by_round(audit_df, out_path / "v4_gap_calibration_by_round.png")
    _plot_per_bucket_ll_delta(by_axes, out_path / "v4_gap_per_bucket_ll_delta.png")

    # Print headline + weak spots
    print()
    print("=" * 70)
    print("OVERALL")
    print("=" * 70)
    print(f"  n_games  : {overall['n_games']}")
    print(f"  ll_v4    : {overall['ll_v4']:.4f}")
    print(f"  ll_vegas : {overall['ll_vegas']:.4f}")
    print(f"  acc_v4   : {overall['acc_v4']:.3f}")
    print(f"  acc_vegas: {overall['acc_vegas']:.3f}")
    print(f"  ece_v4   : {overall['ece_v4']:.4f}")
    print(f"  ece_vegas: {overall['ece_vegas']:.4f}")
    print()

    print("=" * 70)
    print(f"WEAK SPOTS (n >= {WEAK_SPOT_MIN_N}, ll_delta >= {WEAK_SPOT_MIN_LL_DELTA})")
    print("=" * 70)
    if not weak_spots:
        print("  none -- v4 is at Vegas-tier across all sufficiently-large buckets")
    else:
        for w in weak_spots:
            print(
                f"  {w['axis']}={w['value']:>10}  n={w['n_games']:>4}  "
                f"ll_v4={w['ll_v4']:.4f}  ll_vegas={w['ll_vegas']:.4f}  "
                f"delta={w['ll_delta']:+.4f}  "
                f"mean_(p_v4-p_vegas)={w['mean_p_v4_minus_vegas']:+.3f}"
            )

    print()
    print(f"  saved {out_json}")
    print(f"  saved 3 PNGs in {out_path}/")
    return summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--v4", default="output/pairwise_v4.csv")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--out-json", default=DEFAULT_OUT_JSON)
    args = parser.parse_args(argv)

    run_audit(v4_csv=args.v4, out_dir=args.out_dir, out_json=args.out_json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
