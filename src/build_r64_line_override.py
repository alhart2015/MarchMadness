"""Apply-time override of v4's R64 pairwise probabilities with Vegas
closing-line implied probabilities. 32 R64 games per tournament.

Spec: docs/superpowers/specs/2026-05-07-v4-r64-line-blend-design.md
Strategy: docs/notes/2026-05-07-v4-kaggle-gap-strategy.md

Cross-module coupling: imports private helpers from
src.audit_v4_gap_vegas (Vegas join pipeline) and src.enhanced_model_v2
(Vegas line loader / name resolution); intentional per the spec, this
is a one-off diagnostic.

Outputs:
    output/pairwise_v4_r64lineblend_<mode>_sigma<sigma>.csv
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.audit_v4_gap_vegas import (  # noqa: E402
    DATA,
    _build_day_zero_map,
    _find_vegas_p,
    _round_from_daynum,
    _spread_to_prob,
    _vegas_to_seasonday,
)
from src.enhanced_model_v2 import (  # noqa: E402
    _build_vegas_name_to_kaggle_map,
    _resolve_vegas_name,
    load_vegas_lines,
)

logger = logging.getLogger(__name__)

VALID_MODES = {"hard", "mean"}
DEFAULT_SIGMA = 11.0


def _build_vegas_lookup_at_sigma(
    vegas_df: pd.DataFrame,
    name_resolution: dict,
    sigma: float,
) -> dict:
    """Sigma-parameterized variant of `audit_v4_gap_vegas._build_vegas_lookup`.

    The audit driver's `_build_vegas_lookup` calls `_spread_to_prob(line)`
    with the module-level `SIGMA=11` constant baked in, so changing sigma
    requires either monkey-patching the constant or duplicating the
    builder. We pick the latter: a tiny local copy that takes `sigma` as
    a function argument so we can sweep cheaply.

    Returns: {(season, daynum, min_id, max_id): p_a_wins (float)}.
    Drops rows where either team can't be resolved or daynum is NaN.

    Intentional duplication-by-parameterization: the audit driver is
    locked to its SIGMA constant for reproducibility; this builder
    exists so the override path can sweep sigma without touching the
    audit driver.
    """
    lookup: dict = {}
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
        p_home = _spread_to_prob(float(line), sigma=sigma)
        if home_id == a:
            p_a_wins = p_home
        else:
            p_a_wins = 1.0 - p_home
        lookup[(season, daynum, int(a), int(b))] = float(p_a_wins)
    return lookup


def _build_r64_pair_index(season: int, results_df: pd.DataFrame) -> dict:
    """For a single season, return {(season, team_a, team_b): daynum} for
    each of the 32 R64 games. team_a < team_b convention.

    R64 detected by `_round_from_daynum(daynum) == "R64"`.
    """
    out: dict = {}
    for _, g in results_df.iterrows():
        if int(g["Season"]) != int(season):
            continue
        daynum = int(g["DayNum"])
        if _round_from_daynum(daynum) != "R64":
            continue
        w, l = int(g["WTeamID"]), int(g["LTeamID"])
        a, b = (w, l) if w < l else (l, w)
        out[(int(season), a, b)] = daynum
    return out


def _apply_overrides(
    v4_df: pd.DataFrame,
    vegas_lookup: dict,
    r64_pairs: dict,
    mode: str,
    sigma: float,
) -> tuple[pd.DataFrame, dict]:
    """Return (modified_v4_df, stats).

    `vegas_lookup` is the *already-built* lookup at the desired sigma --
    shape `{(season, daynum, a, b): p_a_wins (float)}` produced by
    `_build_vegas_lookup_at_sigma`. The `sigma` kwarg here is purely
    informational (it gets recorded in stats); the actual probability
    is read from the precomputed lookup.

    For each (season, a, b, daynum) in r64_pairs:
      - look up Vegas prob via `_find_vegas_p(vegas_lookup, season, daynum, a, b)`
        (which has +/- 1 day slack tolerance baked in)
      - if found, replace v4_df row's p_a_wins per `mode`:
          - hard: p_vegas
          - mean: 0.5 * p_v4 + 0.5 * p_vegas
      - if not found (or v4 row absent), leave v4_df unchanged and
        increment n_missing_line.
    """
    if mode not in VALID_MODES:
        raise ValueError(f"mode must be one of {VALID_MODES}, got {mode!r}")

    out = v4_df.copy()
    n_target = len(r64_pairs)
    n_overridden = 0
    n_missing = 0

    # Build a lookup from (season, a, b) -> row index for fast updates.
    key_to_idx: dict = {}
    for idx, row in out.iterrows():
        key_to_idx[(int(row["season"]), int(row["team_a"]), int(row["team_b"]))] = idx

    for (season, a, b), daynum in r64_pairs.items():
        idx = key_to_idx.get((int(season), int(a), int(b)))
        if idx is None:
            # R64 pair missing from v4_df; treat as missing-line (rare).
            n_missing += 1
            continue
        p_vegas = _find_vegas_p(vegas_lookup, int(season), int(daynum), int(a), int(b))
        if p_vegas is None:
            n_missing += 1
            continue
        p_v4 = float(out.at[idx, "p_a_wins"])
        if mode == "hard":
            out.at[idx, "p_a_wins"] = float(p_vegas)
        else:  # mean
            out.at[idx, "p_a_wins"] = 0.5 * p_v4 + 0.5 * float(p_vegas)
        n_overridden += 1

    stats = {
        "n_r64_target": int(n_target),
        "n_overridden": int(n_overridden),
        "n_missing_line": int(n_missing),
        "coverage_pct": float(n_overridden / max(n_target, 1) * 100.0),
        "mode": mode,
        "sigma": float(sigma),
    }
    return out, stats


def apply_r64_override(
    v4_csv: str,
    mode: str,
    sigma: float,
    out_csv: str,
) -> dict:
    """End-to-end: read v4_csv, build R64 pair index from tournament
    results, build Vegas lookup from on-disk lines at the requested
    sigma, apply override per `mode`, write `out_csv`.

    Returns coverage stats dict.
    """
    if mode not in VALID_MODES:
        raise ValueError(f"mode must be one of {VALID_MODES}, got {mode!r}")

    print(f"[r64-override] reading v4 pairwise from {v4_csv} ...")
    v4_df = pd.read_csv(v4_csv).drop_duplicates(
        ["season", "team_a", "team_b"], keep="last"
    )
    print(f"  {len(v4_df):,} rows across {v4_df['season'].nunique()} seasons")

    print("[r64-override] loading tournament results + day-zero map ...")
    results = pd.read_csv(DATA / "MNCAATourneyCompactResults.csv")
    day_zero = _build_day_zero_map(DATA / "MSeasons.csv")

    print("[r64-override] building R64 pair index ...")
    r64_pairs: dict = {}
    for season in sorted(v4_df["season"].unique()):
        season_pairs = _build_r64_pair_index(int(season), results)
        r64_pairs.update(season_pairs)
    print(f"  {len(r64_pairs)} R64 pairs across "
          f"{len({s for s, _, _ in r64_pairs})} seasons")

    print("[r64-override] loading + resolving Vegas lines ...")
    vegas_df = load_vegas_lines()
    teams = pd.read_csv(DATA / "MTeams.csv")
    spellings = pd.read_csv(DATA / "MTeamSpellings.csv", encoding="latin-1")
    name_to_id = _build_vegas_name_to_kaggle_map(teams, spellings)
    fuzzy_cache: dict = {}
    all_names = set(vegas_df["home"].unique()) | set(vegas_df["road"].unique())
    name_resolution: dict = {}
    for name in all_names:
        tid = _resolve_vegas_name(name, name_to_id, fuzzy_cache)
        if tid is not None:
            name_resolution[name] = tid
    vegas_df = _vegas_to_seasonday(vegas_df, day_zero)

    print(f"[r64-override] building Vegas lookup at sigma={sigma} ...")
    vegas_lookup = _build_vegas_lookup_at_sigma(vegas_df, name_resolution, sigma=sigma)
    print(f"  {len(vegas_lookup):,} Vegas (season, daynum, a, b) entries")

    print(f"[r64-override] applying overrides (mode={mode}, sigma={sigma}) ...")
    out_df, stats = _apply_overrides(
        v4_df, vegas_lookup, r64_pairs, mode=mode, sigma=sigma,
    )

    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_csv, index=False)
    print(f"[r64-override] wrote {len(out_df):,} rows to {out_csv}")
    print(f"  coverage: {stats['n_overridden']}/{stats['n_r64_target']} "
          f"({stats['coverage_pct']:.1f}%)  missing: {stats['n_missing_line']}")

    return stats


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--v4", default="output/pairwise_v4.csv")
    parser.add_argument("--mode", choices=sorted(VALID_MODES), default="hard")
    parser.add_argument("--sigma", type=float, default=DEFAULT_SIGMA)
    parser.add_argument("--out-csv", required=True)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    apply_r64_override(
        v4_csv=args.v4, mode=args.mode, sigma=args.sigma,
        out_csv=args.out_csv,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())


# ---------------------------------------------------------------------------
# SIGMA sensitivity sweep (cheap R64-only LL)
# ---------------------------------------------------------------------------


def _compute_r64_ll(
    v4_csv: str,
    mode: str,
    sigma: float,
    results_df: pd.DataFrame,
    vegas_df: pd.DataFrame,
    name_resolution: dict,
) -> dict:
    """Compute weighted log-loss of the override frame on R64 games only.

    Builds a vegas lookup at the requested sigma, applies the override,
    and computes mean LL over the R64 games where we have outcomes.

    Returns: {sigma, mode, n_games, ll, n_overridden, n_missing}.
    """
    eps = 1e-15

    # Build lookup at this sigma.
    vegas_lookup = _build_vegas_lookup_at_sigma(vegas_df, name_resolution, sigma)

    # Build R64 pair index across all seasons in the v4 frame.
    v4_df = pd.read_csv(v4_csv).drop_duplicates(
        ["season", "team_a", "team_b"], keep="last"
    )
    r64_pairs: dict = {}
    for season in sorted(v4_df["season"].unique()):
        r64_pairs.update(_build_r64_pair_index(int(season), results_df))

    out_df, stats = _apply_overrides(
        v4_df, vegas_lookup, r64_pairs, mode=mode, sigma=sigma,
    )

    # Score on the R64 games we have outcomes for. Build a results lookup
    # once (per-iteration filtering would be O(R64 pairs * games)).
    results_lookup: dict = {}
    for _, g in results_df.iterrows():
        season = int(g["Season"])
        daynum = int(g["DayNum"])
        w, l = int(g["WTeamID"]), int(g["LTeamID"])
        a, b = (w, l) if w < l else (l, w)
        results_lookup[(season, daynum, a, b)] = (1 if w == a else 0)

    out_lookup = {(int(r["season"]), int(r["team_a"]), int(r["team_b"])):
                  float(r["p_a_wins"])
                  for _, r in out_df.iterrows()}

    p_list = []
    for (season, a, b), daynum in r64_pairs.items():
        winner_is_a = results_lookup.get((season, daynum, a, b))
        if winner_is_a is None:
            continue
        p = out_lookup.get((season, a, b))
        if p is None:
            continue
        p_list.append(p if winner_is_a == 1 else 1.0 - p)

    p_arr = np.clip(np.array(p_list), eps, 1.0 - eps)
    ll = float(-np.mean(np.log(p_arr)))
    return {
        "sigma": float(sigma),
        "mode": mode,
        "n_games": len(p_list),
        "ll": ll,
        "n_overridden": stats["n_overridden"],
        "n_missing": stats["n_missing_line"],
    }


def sigma_sweep_ll(
    v4_csv: str,
    sigmas: list[float],
    mode: str = "hard",
) -> list[dict]:
    """Compute R64-only LL across sigmas. Returns list of dicts (one per
    sigma). Loads Vegas + Kaggle data once, then iterates.
    """
    print(f"[r64-override] SIGMA sweep (mode={mode}, sigmas={sigmas}) ...")

    results = pd.read_csv(DATA / "MNCAATourneyCompactResults.csv")
    day_zero = _build_day_zero_map(DATA / "MSeasons.csv")

    vegas_df = load_vegas_lines()
    teams = pd.read_csv(DATA / "MTeams.csv")
    spellings = pd.read_csv(DATA / "MTeamSpellings.csv", encoding="latin-1")
    name_to_id = _build_vegas_name_to_kaggle_map(teams, spellings)
    fuzzy_cache: dict = {}
    all_names = set(vegas_df["home"].unique()) | set(vegas_df["road"].unique())
    name_resolution: dict = {}
    for name in all_names:
        tid = _resolve_vegas_name(name, name_to_id, fuzzy_cache)
        if tid is not None:
            name_resolution[name] = tid
    vegas_df = _vegas_to_seasonday(vegas_df, day_zero)

    rows = []
    for sigma in sigmas:
        row = _compute_r64_ll(
            v4_csv, mode, sigma, results, vegas_df, name_resolution,
        )
        rows.append(row)
        print(f"  sigma={sigma:>5.1f}  ll={row['ll']:.4f}  "
              f"n_games={row['n_games']}  overridden={row['n_overridden']}  "
              f"missing={row['n_missing']}")
    return rows
