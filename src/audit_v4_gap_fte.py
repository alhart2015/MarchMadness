"""Audit v4's tournament-game predictions vs FiveThirtyEight pre-
tournament round-survival forecasts, broken down by round, chalk-vs-
upset, v4-confidence quintile, and seed-difference magnitude.

Spec: docs/superpowers/specs/2026-05-04-v4-gap-audit-fte-design.md

Output:
    output/v4_gap_audit_fte.json
    output/v4_gap_calibration_overall_fte.png
    output/v4_gap_calibration_by_round_fte.png
    output/v4_gap_per_bucket_ll_delta_fte.png

Sourcing: 538 forecasts loaded via Wayback Machine for 7 seasons
(2016-2019, 2021-2023). 2014/2015 predate 538's API endpoint;
2024/2025 not archived. See src/ingest/fte_forecasts.py for detail.

Round-column mapping: 538's rdR_win is P(reach round R), so to audit
round-of-X (R64=1, ..., Champ=6) we read rd{X+1}_win.
"""
from __future__ import annotations

import argparse
import json
import logging
import re
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

from src.ingest.fte_forecasts import _AUDITED_YEARS, load_fte_forecasts
from src.ingest.team_mapping import build_team_mapping

logger = logging.getLogger(__name__)

DATA = Path("data/raw/march-machine-learning-2026")
DEFAULT_FTE_CACHE = Path("data/raw/fte_forecasts")
DEFAULT_OUT_JSON = "output/v4_gap_audit_fte.json"
DEFAULT_OUT_DIR = "output"

# Tournament round inference from Kaggle DayNum (matches Vegas audit).
ROUND_BY_DAYNUM = {
    134: "FF", 135: "FF",
    136: "R64", 137: "R64",
    138: "R32", 139: "R32",
    143: "S16", 144: "S16",
    145: "E8",  146: "E8",
    152: "F4",  153: "F4",
    154: "Champ",
}
ROUND_ORDER = ["R64", "R32", "S16", "E8", "F4", "Champ"]

# Map game-round to 538's rd_R_win column. rdR_win = P(reach round R) so
# to audit round-of-X we read rd{X+1}_win.
FTE_RD_COL_FOR_ROUND: dict[str, str] = {
    "R64": "rd2_win",
    "R32": "rd3_win",
    "S16": "rd4_win",
    "E8": "rd5_win",
    "F4": "rd6_win",
    "Champ": "rd7_win",
}

CONFIDENCE_BIN_EDGES = [0.50, 0.60, 0.70, 0.80, 0.90, 1.00]
CONFIDENCE_BIN_LABELS = [
    "0.50-0.60", "0.60-0.70", "0.70-0.80", "0.80-0.90", "0.90-1.00",
]

WEAK_SPOT_MIN_N = 50
WEAK_SPOT_MIN_LL_DELTA = 0.02


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def _bt_norm(p_a: float, p_b: float) -> float:
    """Bradley-Terry normalization: P(A wins | both reach round R) =
    rd_{R+1}_win[A] / (rd_{R+1}_win[A] + rd_{R+1}_win[B]).

    Defensive: if both inputs are zero, return 0.5 -- unreachable for
    actual played matchups in our pipeline (both teams must be alive in
    the post-play-in snapshot to be in the joined dataframe), but keeps
    the function total.
    """
    p_a = float(p_a)
    p_b = float(p_b)
    s = p_a + p_b
    if s <= 0.0:
        logger.warning("_bt_norm: both inputs zero (%s, %s); returning 0.5",
                       p_a, p_b)
        return 0.5
    return p_a / s


def _v4_confidence_quintile(p_a: float) -> str:
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
    m = _SEED_NUMBER_RE.match(str(seed_str))
    if not m:
        raise ValueError(f"unrecognized seed format: {seed_str!r}")
    return int(m.group(1))


def _calibration_table(
    p_pred: np.ndarray,
    y_actual: np.ndarray,
    n_bins: int = 10,
) -> list[dict]:
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
        p_ft = sub["p_fte"].to_numpy()

        p_v4_w = np.where(winner == 1, p_v4, 1 - p_v4)
        p_ft_w = np.where(winner == 1, p_ft, 1 - p_ft)
        ll_v4 = float(-np.mean(np.log(np.clip(p_v4_w, eps, 1 - eps))))
        ll_ft = float(-np.mean(np.log(np.clip(p_ft_w, eps, 1 - eps))))

        acc_v4 = float(((p_v4 >= 0.5).astype(int) == winner).mean())
        acc_ft = float(((p_ft >= 0.5).astype(int) == winner).mean())

        cal_v4 = _calibration_table(p_v4, winner)
        cal_ft = _calibration_table(p_ft, winner)

        out[str(value)] = {
            "n_games": int(n),
            "ll_v4": ll_v4,
            "ll_fte": ll_ft,
            "ll_delta": float(ll_v4 - ll_ft),
            "acc_v4": acc_v4,
            "acc_fte": acc_ft,
            "ece_v4": _ece(cal_v4),
            "ece_fte": _ece(cal_ft),
            "mean_p_v4_minus_fte": float((p_v4 - p_ft).mean()),
            "calibration_v4": cal_v4,
            "calibration_fte": cal_ft,
        }
    return out


# ---------------------------------------------------------------------------
# Data load
# ---------------------------------------------------------------------------


def _load_v4_lookup(v4_csv: str) -> dict:
    """{(season, team_a, team_b): p_a_wins} where a < b. Dedup via keep=last."""
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


def _resolve_fte_team_ids(
    fte_df: pd.DataFrame,
    teams_df: pd.DataFrame,
    spellings_df: pd.DataFrame,
    overrides_path: str | Path | None = "data/team_name_overrides.csv",
) -> tuple[pd.DataFrame, list[str]]:
    """Resolve 538's team_name to Kaggle TeamID.

    Layered resolution:
      1) Exact case-insensitive lookup in MTeamSpellings (curated by Kaggle).
         Catches "Louisiana State" -> LSU, "North Carolina State" -> NC State,
         "Brigham Young" -> BYU, etc. -- variants where the Kaggle TeamName
         is an abbreviation that fuzzy-matching dangerously confuses with
         a different team's name.
      2) build_team_mapping fuzzy fallback for any name not in spellings
         (defensive; on the 7 audited seasons all 176 unique 538 names
         resolve via spellings alone).

    Returns (fte_df_with_TeamID, unresolved_names).
    """
    spelling_map = {
        str(s).lower(): int(t)
        for s, t in zip(spellings_df["TeamNameSpelling"], spellings_df["TeamID"])
    }
    fte_names = sorted(fte_df["team_name"].astype(str).unique())

    # Layer 1: spellings exact match
    name_to_id: dict[str, int] = {}
    for name in fte_names:
        tid = spelling_map.get(name.lower())
        if tid is not None:
            name_to_id[name] = tid

    # Layer 2: fuzzy via build_team_mapping for any leftovers
    leftover = [n for n in fte_names if n not in name_to_id]
    if leftover:
        fuzzy_map = build_team_mapping(
            kaggle_teams=teams_df,
            external_names=leftover,
            overrides_path=str(overrides_path) if overrides_path else None,
        )
        name_to_id.update(fuzzy_map)

    out = fte_df.copy()
    out["TeamID"] = out["team_name"].astype(str).map(name_to_id)
    unresolved = sorted(out[out["TeamID"].isna()]["team_name"].unique())
    out = out.dropna(subset=["TeamID"]).copy()
    out["TeamID"] = out["TeamID"].astype(int)
    return out, list(unresolved)


def _build_fte_lookup(fte_df: pd.DataFrame) -> dict:
    """{(Season, TeamID): {rd_col: prob}} -- O(1) lookup by (season, team)."""
    out = {}
    rd_cols = [f"rd{r}_win" for r in range(1, 8)]
    for _, row in fte_df.iterrows():
        key = (int(row["Season"]), int(row["TeamID"]))
        out[key] = {c: float(row[c]) for c in rd_cols}
    return out


# ---------------------------------------------------------------------------
# Per-game audit DF
# ---------------------------------------------------------------------------


def _build_per_game_audit_df(
    v4_lookup: dict,
    fte_lookup: dict,
    results_df: pd.DataFrame,
    seeds_lookup: dict,
) -> pd.DataFrame:
    """Per-game DataFrame for the audit set.

    Columns: season, daynum, team_a, team_b, p_v4, p_fte, winner_is_a,
    round, seed_a_num, seed_b_num, seed_diff, seed_diff_bucket,
    chalk_won (string 'chalk'/'upset'), v4_confidence_quintile.
    """
    rows = []
    for _, g in results_df.iterrows():
        season = int(g["Season"])
        if season not in _AUDITED_YEARS:
            continue
        daynum = int(g["DayNum"])
        round_label = _round_from_daynum(daynum)
        if round_label not in FTE_RD_COL_FOR_ROUND:
            # Skip FF and OTHER -- 538 doesn't carry play-in matchups in
            # rd_R_win and we explicitly drop them from the audit.
            continue
        w, l = int(g["WTeamID"]), int(g["LTeamID"])
        a, b = (w, l) if w < l else (l, w)

        p_v4 = v4_lookup.get((season, a, b))
        if p_v4 is None:
            continue

        fte_row_a = fte_lookup.get((season, a))
        fte_row_b = fte_lookup.get((season, b))
        if fte_row_a is None or fte_row_b is None:
            continue
        fte_col = FTE_RD_COL_FOR_ROUND[round_label]
        rd_a = fte_row_a[fte_col]
        rd_b = fte_row_b[fte_col]
        if (rd_a + rd_b) <= 0.0:
            # Defensive: both teams have zero round-R survival prob.
            # Shouldn't happen for actual played matchups; skip if it does.
            continue
        p_fte = _bt_norm(rd_a, rd_b)

        winner_is_a = 1 if w == a else 0

        seed_a = seeds_lookup.get((season, a))
        seed_b = seeds_lookup.get((season, b))
        if seed_a is None or seed_b is None:
            continue
        seed_a_num = _extract_seed_number(seed_a)
        seed_b_num = _extract_seed_number(seed_b)
        seed_diff = abs(seed_a_num - seed_b_num)

        # Same convention as Vegas audit: chalk_won == "chalk" iff better
        # (lower-numbered) seed won.
        seed_a_won = (winner_is_a == 1)
        if seed_a_num == seed_b_num:
            chalk_won = True
        elif seed_a_num < seed_b_num:
            chalk_won = seed_a_won
        else:
            chalk_won = not seed_a_won

        rows.append({
            "season": season,
            "daynum": daynum,
            "team_a": a,
            "team_b": b,
            "p_v4": p_v4,
            "p_fte": p_fte,
            "winner_is_a": winner_is_a,
            "round": round_label,
            "seed_a_num": seed_a_num,
            "seed_b_num": seed_b_num,
            "seed_diff": seed_diff,
            "seed_diff_bucket": _seed_diff_bucket(seed_diff),
            "chalk_won": "chalk" if chalk_won else "upset",
            "v4_confidence_quintile": _v4_confidence_quintile(p_v4),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def _plot_calibration_overall(audit_df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    for label, prob_col in [("v4", "p_v4"), ("538", "p_fte")]:
        cal = _calibration_table(
            audit_df[prob_col].to_numpy(),
            audit_df["winner_is_a"].to_numpy(),
            n_bins=10,
        )
        xs, ys = [], []
        for b in cal:
            if b["empirical"] is None:
                continue
            xs.append(b["mid"])
            ys.append(b["empirical"])
        ax.plot(xs, ys, marker="o", label=label)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="diagonal")
    ax.set_xlabel("predicted P(team_a wins)")
    ax.set_ylabel("empirical win rate")
    ax.set_title("v4 vs 538 calibration on 7 audited tournaments")
    ax.legend(loc="upper left")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _plot_calibration_by_round(audit_df: pd.DataFrame, out_path: Path) -> None:
    rounds = [r for r in ROUND_ORDER if r in audit_df["round"].unique()]
    n_rounds = len(rounds)
    cols = 3
    rows_grid = (n_rounds + cols - 1) // cols
    fig, axes = plt.subplots(rows_grid, cols, figsize=(4 * cols, 4 * rows_grid))
    axes = np.array(axes).reshape(-1)
    for i, round_name in enumerate(rounds):
        ax = axes[i]
        sub = audit_df[audit_df["round"] == round_name]
        for label, prob_col in [("v4", "p_v4"), ("538", "p_fte")]:
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
    fig.suptitle("v4 vs 538 calibration by round")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _plot_per_bucket_ll_delta(by_axis: dict, out_path: Path) -> None:
    rows_data = []
    for axis_name, axis_dict in by_axis.items():
        for val, cell in axis_dict.items():
            if cell["n_games"] < WEAK_SPOT_MIN_N:
                continue
            rows_data.append({
                "label": f"{axis_name}={val} (n={cell['n_games']})",
                "ll_delta": cell["ll_delta"],
            })
    if not rows_data:
        # Empty plot -- write a placeholder so the file exists.
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "no buckets meet n>=50 threshold",
                ha="center", va="center")
        ax.axis("off")
        fig.savefig(out_path, dpi=120)
        plt.close(fig)
        return
    df = pd.DataFrame(rows_data).sort_values("ll_delta", ascending=True)
    fig, ax = plt.subplots(figsize=(8, max(4, 0.35 * len(df))))
    colors = ["red" if d > 0 else "steelblue" for d in df["ll_delta"]]
    ax.barh(df["label"], df["ll_delta"], color=colors)
    ax.axvline(0, color="black", linewidth=0.7)
    ax.axvline(WEAK_SPOT_MIN_LL_DELTA, color="red", linestyle="--", alpha=0.5,
               label=f"weak-spot threshold (+{WEAK_SPOT_MIN_LL_DELTA})")
    ax.set_xlabel("ll_v4 - ll_fte (positive = v4 worse)")
    ax.set_title(f"Per-bucket LL delta (n >= {WEAK_SPOT_MIN_N})")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _identify_weak_spots(by_axes: dict) -> list[dict]:
    out = []
    for axis_name, axis_dict in by_axes.items():
        for val, cell in axis_dict.items():
            if cell["n_games"] < WEAK_SPOT_MIN_N:
                continue
            if cell["ll_delta"] < WEAK_SPOT_MIN_LL_DELTA:
                continue
            out.append({
                "axis": axis_name,
                "value": val,
                "n_games": cell["n_games"],
                "ll_v4": cell["ll_v4"],
                "ll_fte": cell["ll_fte"],
                "ll_delta": cell["ll_delta"],
                "acc_v4": cell["acc_v4"],
                "acc_fte": cell["acc_fte"],
                "mean_p_v4_minus_fte": cell["mean_p_v4_minus_fte"],
            })
    out.sort(key=lambda r: r["ll_delta"], reverse=True)
    return out[:10]


def run_audit(
    v4_csv: str = "output/pairwise_v4.csv",
    out_dir: str | Path = DEFAULT_OUT_DIR,
    out_json: str = DEFAULT_OUT_JSON,
    fte_cache_dir: str | Path = DEFAULT_FTE_CACHE,
) -> dict:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("v4 vs 538 TOURNAMENT AUDIT (7 seasons via Wayback Machine)")
    print("=" * 70)

    print("loading v4 pairwise ...", flush=True)
    v4_lookup = _load_v4_lookup(v4_csv)
    print(f"  {len(v4_lookup):,} unique (season, a, b) keys")

    print("loading 538 forecasts ...", flush=True)
    fte_df = load_fte_forecasts(
        years=_AUDITED_YEARS, cache_dir=Path(fte_cache_dir),
    )
    print(f"  {len(fte_df)} (season, team) rows across {fte_df['Season'].nunique()} seasons")

    print("resolving 538 team names to Kaggle TeamIDs ...", flush=True)
    teams = pd.read_csv(DATA / "MTeams.csv")
    spellings = pd.read_csv(DATA / "MTeamSpellings.csv", encoding="latin-1")
    fte_df_resolved, unresolved = _resolve_fte_team_ids(fte_df, teams, spellings)
    if unresolved:
        print(f"  *** {len(unresolved)} unresolved names: {unresolved[:5]}{'...' if len(unresolved) > 5 else ''}")
        print("  (add overrides in data/team_name_overrides.csv)")
    print(f"  resolved {len(fte_df_resolved)} of {len(fte_df)} rows")

    fte_lookup = _build_fte_lookup(fte_df_resolved)

    print("loading tournament results + seeds ...", flush=True)
    results = pd.read_csv(DATA / "MNCAATourneyCompactResults.csv")
    seeds_lookup = _load_seeds_lookup(DATA / "MNCAATourneySeeds.csv")
    print(f"  {len(results):,} tournament games across "
          f"{results['Season'].nunique()} seasons")

    print("joining v4 + 538 + outcomes ...", flush=True)
    audit_df = _build_per_game_audit_df(
        v4_lookup, fte_lookup, results, seeds_lookup,
    )
    print(f"  {len(audit_df)} games joined")

    # Coverage report (over the 7 audited seasons, R64-Champ only)
    audited_results = results[results["Season"].isin(_AUDITED_YEARS)].copy()
    audited_results["round"] = audited_results["DayNum"].map(_round_from_daynum)
    n_audit_eligible = int(
        audited_results["round"].isin(list(FTE_RD_COL_FOR_ROUND.keys())).sum()
    )
    coverage = {
        "audited_seasons": list(_AUDITED_YEARS),
        "n_eligible_games": n_audit_eligible,
        "n_joined": int(len(audit_df)),
        "coverage_pct": float(len(audit_df) / max(n_audit_eligible, 1) * 100.0),
        "unresolved_fte_names": unresolved,
        "by_season": {
            int(s): {
                "eligible": int(((audited_results["Season"] == s)
                                 & audited_results["round"].isin(list(FTE_RD_COL_FOR_ROUND.keys()))).sum()),
                "joined": int(len(audit_df[audit_df["season"] == s])),
            }
            for s in sorted(_AUDITED_YEARS)
        },
    }
    print(f"  coverage: {coverage['coverage_pct']:.1f}% ({coverage['n_joined']} / {coverage['n_eligible_games']})")
    if coverage["coverage_pct"] < 90.0:
        print(f"  *** WARNING: coverage below 90% ***")

    # Overall metrics
    print("computing overall + per-bucket metrics ...", flush=True)
    eps = 1e-15
    p_v4 = audit_df["p_v4"].to_numpy()
    p_ft = audit_df["p_fte"].to_numpy()
    winner = audit_df["winner_is_a"].to_numpy()
    p_v4_w = np.where(winner == 1, p_v4, 1 - p_v4)
    p_ft_w = np.where(winner == 1, p_ft, 1 - p_ft)

    cal_v4_overall = _calibration_table(p_v4, winner)
    cal_ft_overall = _calibration_table(p_ft, winner)

    overall = {
        "n_games": int(len(audit_df)),
        "ll_v4": float(-np.mean(np.log(np.clip(p_v4_w, eps, 1 - eps)))),
        "ll_fte": float(-np.mean(np.log(np.clip(p_ft_w, eps, 1 - eps)))),
        "acc_v4": float(((p_v4 >= 0.5).astype(int) == winner).mean()),
        "acc_fte": float(((p_ft >= 0.5).astype(int) == winner).mean()),
        "ece_v4": _ece(cal_v4_overall),
        "ece_fte": _ece(cal_ft_overall),
        "calibration_v4": cal_v4_overall,
        "calibration_fte": cal_ft_overall,
    }

    by_round = _compute_bucket_metrics(audit_df, "round")
    by_chalk = _compute_bucket_metrics(audit_df, "chalk_won")
    by_conf = _compute_bucket_metrics(audit_df, "v4_confidence_quintile")
    by_seed_diff = _compute_bucket_metrics(audit_df, "seed_diff_bucket")

    by_axes = {
        "round": by_round,
        "chalk_won": by_chalk,
        "v4_confidence_quintile": by_conf,
        "seed_diff_bucket": by_seed_diff,
    }

    weak_spots = _identify_weak_spots(by_axes)

    # Anchor: rd2_win pair sum-to-1 invariant on R64 games.
    r64_audit = audit_df[audit_df["round"] == "R64"]
    if len(r64_audit):
        # For R64 the BT-norm denominator should be ~1.0 (sum of rd2_win
        # for a played matchup of A and B is exactly 1 in 538's bracket
        # consistency, modulo rounding). We sample-check denominators.
        denoms = []
        for _, row in r64_audit.head(50).iterrows():
            a_rd = fte_lookup[(int(row["season"]), int(row["team_a"]))]["rd2_win"]
            b_rd = fte_lookup[(int(row["season"]), int(row["team_b"]))]["rd2_win"]
            denoms.append(a_rd + b_rd)
        anchor = {
            "r64_rd2_sum_min": float(min(denoms)),
            "r64_rd2_sum_max": float(max(denoms)),
            "r64_rd2_sum_mean": float(np.mean(denoms)),
            "n_sampled": len(denoms),
        }
    else:
        anchor = {"note": "no R64 games in audit"}

    summary = {
        "config": {
            "v4_pairwise": v4_csv,
            "fte_cache_dir": str(fte_cache_dir),
            "audited_seasons": list(_AUDITED_YEARS),
            "snapshot_policy": "earliest_post_playin_per_season",
            "weak_spot_min_n": WEAK_SPOT_MIN_N,
            "weak_spot_min_ll_delta": WEAK_SPOT_MIN_LL_DELTA,
        },
        "join_coverage": coverage,
        "anchor_r64_sum_to_one": anchor,
        "overall": overall,
        "by_round": by_round,
        "by_chalk_won": by_chalk,
        "by_v4_confidence_quintile": by_conf,
        "by_seed_diff_bucket": by_seed_diff,
        "weak_spots": weak_spots,
    }

    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)

    print("emitting plots ...", flush=True)
    _plot_calibration_overall(audit_df, out_path / "v4_gap_calibration_overall_fte.png")
    _plot_calibration_by_round(audit_df, out_path / "v4_gap_calibration_by_round_fte.png")
    _plot_per_bucket_ll_delta(by_axes, out_path / "v4_gap_per_bucket_ll_delta_fte.png")

    print()
    print("=" * 70)
    print("OVERALL")
    print("=" * 70)
    print(f"  n_games : {overall['n_games']}")
    print(f"  ll_v4   : {overall['ll_v4']:.4f}")
    print(f"  ll_fte  : {overall['ll_fte']:.4f}")
    print(f"  acc_v4  : {overall['acc_v4']:.3f}")
    print(f"  acc_fte : {overall['acc_fte']:.3f}")
    print(f"  ece_v4  : {overall['ece_v4']:.4f}")
    print(f"  ece_fte : {overall['ece_fte']:.4f}")
    print()
    print("ANCHOR R64 rd2_win sum-to-1:")
    print(f"  {anchor}")
    print()

    print("=" * 70)
    print(f"WEAK SPOTS (n >= {WEAK_SPOT_MIN_N}, ll_delta >= +{WEAK_SPOT_MIN_LL_DELTA})")
    print("=" * 70)
    if not weak_spots:
        print("  none -- v4 is at 538-tier across all sufficiently-large buckets")
    else:
        for w in weak_spots:
            print(
                f"  {w['axis']}={w['value']:>14}  n={w['n_games']:>4}  "
                f"ll_v4={w['ll_v4']:.4f}  ll_fte={w['ll_fte']:.4f}  "
                f"delta={w['ll_delta']:+.4f}  "
                f"mean(p_v4 - p_fte)={w['mean_p_v4_minus_fte']:+.3f}"
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
    parser.add_argument("--fte-cache", default=str(DEFAULT_FTE_CACHE))
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    run_audit(
        v4_csv=args.v4,
        out_dir=args.out_dir,
        out_json=args.out_json,
        fte_cache_dir=args.fte_cache,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
