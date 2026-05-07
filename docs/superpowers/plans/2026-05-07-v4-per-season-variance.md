# v4 Per-Season Variance Check -- Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Per-season metrics + cross-benchmark deltas for v4 across 22 LOSO seasons; flag outliers at 1.5 sigma; produce a verdict that informs whether the next experiment should be calibration-shape engineering, an outlier-season investigation, or a trend deep-dive.

**Architecture:** One new script `src/analyze_v4_per_season_variance.py` reusing data-load helpers from `src/audit_v4_gap_vegas.py` and `src/audit_v4_gap_fte.py`. New code: per-season aggregation, outlier flagging, traces + deltas plots, verdict logic. Audit-driver helpers are imported with private-`_` prefix; this is intentional cross-module coupling per the spec's Risk #1 (one-off diagnostic; refactor only if more diagnostics in this family follow).

**Tech Stack:** pandas, numpy, matplotlib, pytest.

**Spec:** `docs/superpowers/specs/2026-05-07-v4-per-season-variance-design.md`

---

## File Structure

**Created (committed):**

- `src/analyze_v4_per_season_variance.py` -- driver (~250 LOC)
  - Public: `run_analysis(v4_csv, out_dir, out_json, fte_cache_dir, sigma_threshold) -> dict`
  - Private: `_per_season_metrics(df, ref_label) -> pd.DataFrame`
  - Private: `_flag_outliers(df, columns, sigma) -> dict`
  - Private: `_pick_verdict(df, outliers, sigma) -> dict`
  - Private: `_plot_traces(merged_df, out_path) -> None`
  - Private: `_plot_deltas(merged_df, outliers, out_path) -> None`
- `tests/test_analyze_v4_per_season_variance.py` -- 5 unit tests + 1 smoke (~150 LOC).

**Generated (committed via `git add -f`):**

- `output/v4_per_season_variance.json`
- `output/v4_per_season_variance_traces.png`
- `output/v4_per_season_variance_deltas.png`
- `output/v4_per_season_variance_log.txt`
- `docs/notes/2026-05-07-v4-per-season-variance.md`

**Modified:**

- `TODO.md` -- mark active queue #1 done; promote next item per verdict.

---

## Phase 1: Per-season aggregation + outlier flagging (with tests)

### Task 1: Write helpers and unit tests

**Files:**
- Create: `src/analyze_v4_per_season_variance.py`
- Create: `tests/test_analyze_v4_per_season_variance.py`

- [ ] **Step 1: Write failing unit tests**

Create `tests/test_analyze_v4_per_season_variance.py`:

```python
"""Unit tests for src/analyze_v4_per_season_variance.py."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analyze_v4_per_season_variance import (
    _flag_outliers,
    _per_season_metrics,
    _pick_verdict,
)


def _three_season_fixture() -> pd.DataFrame:
    """Synthetic per-game v4 vs Vegas frame, 3 seasons x 4 games each.
    Season 2010 v4 wins big, 2011 ties, 2012 v4 loses (constructed).
    """
    rows = [
        # 2010: v4 perfectly confident in winners (LL=0)
        {"season": 2010, "p_v4": 0.99, "p_vegas": 0.60, "winner_is_a": 1},
        {"season": 2010, "p_v4": 0.99, "p_vegas": 0.60, "winner_is_a": 1},
        {"season": 2010, "p_v4": 0.01, "p_vegas": 0.40, "winner_is_a": 0},
        {"season": 2010, "p_v4": 0.01, "p_vegas": 0.40, "winner_is_a": 0},
        # 2011: tied
        {"season": 2011, "p_v4": 0.70, "p_vegas": 0.70, "winner_is_a": 1},
        {"season": 2011, "p_v4": 0.50, "p_vegas": 0.50, "winner_is_a": 1},
        {"season": 2011, "p_v4": 0.30, "p_vegas": 0.30, "winner_is_a": 0},
        {"season": 2011, "p_v4": 0.50, "p_vegas": 0.50, "winner_is_a": 0},
        # 2012: v4 wrongly confident (LL much worse than Vegas)
        {"season": 2012, "p_v4": 0.99, "p_vegas": 0.55, "winner_is_a": 0},
        {"season": 2012, "p_v4": 0.99, "p_vegas": 0.55, "winner_is_a": 0},
        {"season": 2012, "p_v4": 0.01, "p_vegas": 0.45, "winner_is_a": 1},
        {"season": 2012, "p_v4": 0.01, "p_vegas": 0.45, "winner_is_a": 1},
    ]
    return pd.DataFrame(rows)


def test_per_season_metrics_aggregates_correctly():
    """Per-season LL/acc/ECE on the 3-season fixture."""
    df = _three_season_fixture()
    out = _per_season_metrics(df, ref_label="vegas")
    assert list(out["season"]) == [2010, 2011, 2012]
    assert (out["n_games"] == 4).all()
    # 2010: v4 LL ~ -log(0.99) ~ 0.01
    assert out.loc[out["season"] == 2010, "ll_v4"].iloc[0] == pytest.approx(
        -np.log(0.99), abs=1e-3
    )
    # 2012: v4 LL ~ -log(0.01) ~ 4.6 (catastrophically wrong)
    assert out.loc[out["season"] == 2012, "ll_v4"].iloc[0] > 4.0
    # ll_v4_minus_vegas: 2010 negative (v4 better), 2012 positive (v4 worse)
    assert out.loc[out["season"] == 2010, "ll_v4_minus_vegas"].iloc[0] < 0
    assert out.loc[out["season"] == 2012, "ll_v4_minus_vegas"].iloc[0] > 0


def test_per_season_metrics_weighted_aggregate_matches_overall():
    """Invariant: weighted average of per-season LL (by n_games) equals
    the overall LL on the same frame."""
    df = _three_season_fixture()
    per_season = _per_season_metrics(df, ref_label="vegas")
    weighted_ll = float(np.average(per_season["ll_v4"],
                                    weights=per_season["n_games"]))
    eps = 1e-15
    winner = df["winner_is_a"].to_numpy()
    p_v4 = df["p_v4"].to_numpy()
    p_v4_w = np.where(winner == 1, p_v4, 1 - p_v4)
    overall_ll = float(-np.mean(np.log(np.clip(p_v4_w, eps, 1 - eps))))
    assert weighted_ll == pytest.approx(overall_ll, abs=1e-6)


def test_flag_outliers_flags_high_sigma_value():
    """One value 2.5 sigma above the mean is flagged at threshold 1.5."""
    df = pd.DataFrame({
        "season": list(range(2000, 2010)),
        "n_games": [60] * 10,
        # 9 values with mean 0.5, std small; one outlier at 1.5
        "ll_v4_minus_vegas": [0.05, -0.05, 0.05, -0.05, 0.05, -0.05, 0.05, -0.05, 0.05, 1.5],
    })
    out = _flag_outliers(df, columns=["ll_v4_minus_vegas"], sigma=1.5)
    assert "ll_v4_minus_vegas" in out
    assert len(out["ll_v4_minus_vegas"]) == 1
    assert out["ll_v4_minus_vegas"][0]["season"] == 2009
    assert out["ll_v4_minus_vegas"][0]["sigma_delta"] >= 1.5


def test_flag_outliers_skips_missing_column_and_short_series():
    """Missing columns are skipped; series shorter than 2 returns empty."""
    df = pd.DataFrame({
        "season": [2000, 2001],
        "n_games": [60, 60],
        "ll_v4": [0.5, 0.6],
    })
    # Column not present -> skipped
    out = _flag_outliers(df, columns=["ll_v4", "nonexistent"], sigma=1.5)
    assert "nonexistent" not in out
    # Series too short / std too small -> no outliers flagged
    assert out["ll_v4"] == []


def test_pick_verdict_flat_when_no_outliers():
    """No outliers on any tracked metric -> verdict='flat'."""
    df = pd.DataFrame({
        "season": [2000, 2001, 2002],
        "n_games": [60, 60, 60],
        "ll_v4": [0.55, 0.56, 0.54],
        "ll_v4_minus_vegas": [0.01, -0.01, 0.0],
        "ll_v4_minus_fte": [None, None, None],
        "ece_v4": [0.04, 0.05, 0.04],
    })
    outliers = {
        "ll_v4_minus_vegas": [],
        "ll_v4_minus_fte": [],
        "ll_v4": [],
        "ece_v4": [],
    }
    verdict = _pick_verdict(df, outliers, sigma=1.5)
    assert verdict["label"] == "flat"


def test_pick_verdict_outlier_when_one_or_two_seasons_flagged():
    """One outlier season on the v4-vs-Vegas delta -> verdict='outlier'."""
    df = pd.DataFrame({
        "season": list(range(2000, 2010)),
        "n_games": [60] * 10,
        "ll_v4": [0.55] * 10,
        "ll_v4_minus_vegas": [0.0] * 9 + [0.2],
        "ll_v4_minus_fte": [None] * 10,
        "ece_v4": [0.04] * 10,
    })
    outliers = {
        "ll_v4_minus_vegas": [{"season": 2009, "value": 0.2,
                                "sigma_delta": 3.0, "n_games": 60}],
        "ll_v4_minus_fte": [],
        "ll_v4": [],
        "ece_v4": [],
    }
    verdict = _pick_verdict(df, outliers, sigma=1.5)
    assert verdict["label"] == "outlier"
    assert 2009 in verdict["outlier_seasons"]
```

Run: `cd .claude/worktrees/feat-v4-per-season-variance && python -m pytest -v tests/test_analyze_v4_per_season_variance.py` -- expect ImportError on the `from src.analyze_v4_per_season_variance import ...` line.

- [ ] **Step 2: Implement the helpers in `src/analyze_v4_per_season_variance.py`**

Create `src/analyze_v4_per_season_variance.py`:

```python
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
        # Consecutive in season order means each is in the input df and
        # adjacent in the sorted unique-season list.
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
```

(More functions added in Task 2 -- this is enough to make the unit tests pass.)

- [ ] **Step 3: Run unit tests -- expect PASS**

```bash
cd .claude/worktrees/feat-v4-per-season-variance
python -m pytest -v tests/test_analyze_v4_per_season_variance.py
```

Expected: 6 passed (5 unit + 0 smoke yet).

If any test fails, fix the implementation.

- [ ] **Step 4: Commit Phase 1**

```bash
git add src/analyze_v4_per_season_variance.py tests/test_analyze_v4_per_season_variance.py
git commit -m "feat(per-season-variance): aggregator + outlier flagging

- _per_season_metrics: weighted-aggregate-invariant verified by test
- _flag_outliers: 1.5-sigma threshold on each tracked column
- _pick_verdict: flat / outlier / trend / mixed dispatch
- 6 unit tests covering aggregation, outlier detection, verdict picks

Spec: docs/superpowers/specs/2026-05-07-v4-per-season-variance-design.md"
```

---

## Phase 2: Driver wiring (data load + plots + JSON)

### Task 2: Wire the driver, plots, and `run_analysis`

**Files:**
- Modify: `src/analyze_v4_per_season_variance.py` (append driver + plots)

- [ ] **Step 1: Add `_build_vegas_per_game_df` and `_build_fte_per_game_df` helpers**

Append to `src/analyze_v4_per_season_variance.py`:

```python
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
```

- [ ] **Step 2: Add the two plotting functions**

Append:

```python
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
```

- [ ] **Step 3: Add `run_analysis` and CLI**

Append:

```python
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
```

- [ ] **Step 4: Add a smoke test for `run_analysis` end-to-end**

Append to `tests/test_analyze_v4_per_season_variance.py`:

```python
def test_run_analysis_smoke(tmp_path):
    """Run the full analysis on real data; verify output files exist
    and JSON is well-formed. Exercises the full join + write path."""
    pytest.importorskip("matplotlib")
    out_dir = tmp_path / "output"
    out_dir.mkdir()
    out_json = out_dir / "v4_per_season_variance.json"

    # Skip if the required data isn't present (fresh-clone case).
    if not Path("output/pairwise_v4.csv").exists():
        pytest.skip("output/pairwise_v4.csv not present")
    if not Path("data/raw/march-machine-learning-2026/MTeams.csv").exists():
        pytest.skip("Kaggle data not unpacked")

    from src.analyze_v4_per_season_variance import run_analysis
    summary = run_analysis(
        v4_csv="output/pairwise_v4.csv",
        out_dir=out_dir,
        out_json=str(out_json),
        fte_cache_dir="data/raw/fte_forecasts",
        sigma_threshold=1.5,
    )
    assert out_json.exists()
    assert (out_dir / "v4_per_season_variance_traces.png").exists()
    assert (out_dir / "v4_per_season_variance_deltas.png").exists()
    data = json.loads(out_json.read_text())
    assert "per_season" in data
    assert "outliers" in data
    assert "verdict" in data
    assert data["verdict"]["label"] in {"flat", "outlier", "trend", "mixed"}
    # At least 22 Vegas-covered seasons.
    assert len(data["per_season"]) >= 20
```

Add `import json` and `from pathlib import Path` at the top of the test file if not already present.

- [ ] **Step 5: Run all tests**

```bash
cd .claude/worktrees/feat-v4-per-season-variance
python -m pytest -v tests/test_analyze_v4_per_season_variance.py
```

Expected: 7 passed (6 unit + 1 smoke).

If the smoke test fails, debug. The unit tests should still pass.

- [ ] **Step 6: Commit Phase 2**

```bash
git add src/analyze_v4_per_season_variance.py tests/test_analyze_v4_per_season_variance.py
git commit -m "feat(per-season-variance): driver + plots + smoke test

- _build_vegas_per_game_df: reuses Vegas audit's join pipeline (22 seasons)
- _build_fte_per_game_df:   reuses 538 audit's join pipeline (7 seasons)
- _plot_traces: 3 panels (LL/acc/ECE) x season with v4/Vegas/538 lines
- _plot_deltas: 2 panels (v4-Vegas, v4-538 LL deltas) with outliers in red
- run_analysis: orchestrator + JSON + verdict
- Smoke test verifies output files + JSON shape"
```

---

## Phase 3: Run on real data, anchor verification, force-add outputs

### Task 3: end-to-end execution + anchor checks

**Files:**
- Run: `src/analyze_v4_per_season_variance.py`
- Force-add: `output/v4_per_season_variance.{json,_traces.png,_deltas.png,_log.txt}`

- [ ] **Step 1: Run the analysis on real data**

```bash
cd .claude/worktrees/feat-v4-per-season-variance
python src/analyze_v4_per_season_variance.py \
    --v4 output/pairwise_v4.csv \
    --out-dir output/ \
    --out-json output/v4_per_season_variance.json \
    --fte-cache data/raw/fte_forecasts \
    2>&1 | tee output/v4_per_season_variance_log.txt
```

Estimated wall time: 30-90 seconds (Vegas join is the heaviest step;
538 cache is already populated from PR 29).

- [ ] **Step 2: Verify anchors**

Open `output/v4_per_season_variance.json`. From the `anchors` block:

- `weighted_ll_v4_vs_vegas_audit_subset` should be approximately **0.5595**
  (the Vegas audit's overall ll_v4 on the same 1326-game subset).
  Tolerance: +/- 0.001 (floating-point only; weighting must reproduce).
- `weighted_ll_vegas_vs_audit` should be approximately **0.5447**.
- `weighted_ll_v4_vs_fte_audit_subset` should be approximately **0.5799**.
- `weighted_ll_fte_vs_audit` should be approximately **0.6011**.

If any anchor is off by more than 0.001, halt and debug -- the per-season
aggregator is broken.

Inspect `verdict.label` and `outliers`:
- If `flat`: no per-season outliers; the 1.5-sigma threshold finds nothing.
- If `outlier`: 1-2 seasons exceed threshold on at least one cross-benchmark
  delta. Note which seasons.
- If `trend`: 3+ consecutive seasons drift in one direction.
- If `mixed`: report-as-such.

- [ ] **Step 3: Force-add output artifacts + commit Phase 3**

```bash
git add -f \
    output/v4_per_season_variance.json \
    output/v4_per_season_variance_traces.png \
    output/v4_per_season_variance_deltas.png \
    output/v4_per_season_variance_log.txt
git commit -m "data(per-season-variance): force-add JSON + traces + deltas

Real-data run; anchors PASS (weighted per-season aggregates reproduce
audit overall numbers within FP precision). Verdict: <label>."
```

(Replace `<label>` with the actual verdict from Step 2.)

---

## Phase 4: Findings note + TODO update + push PR

### Task 4: findings note + TODO + final commit + PR

**Files:**
- Create: `docs/notes/2026-05-07-v4-per-season-variance.md`
- Modify: `TODO.md`

- [ ] **Step 1: Inspect findings + write note**

Open `output/v4_per_season_variance.json`. Note the verdict label, the
top-2 outlier seasons (if any), the cross-season summary (means, stds),
and the per-season trace shape.

Create `docs/notes/2026-05-07-v4-per-season-variance.md` using the audit
findings notes (`docs/notes/2026-05-04-v4-gap-audit-vegas.md`,
`docs/notes/2026-05-04-v4-gap-audit-fte.md`) as templates. Required
sections:

```markdown
# v4 Per-Season Variance Check -- Findings

**Date:** 2026-05-07
**Branch:** feat/v4-per-season-variance
**Verdict:** [FLAT / OUTLIER / TREND / MIXED]
**Spec:** `docs/superpowers/specs/2026-05-07-v4-per-season-variance-design.md`
**Plan:** `docs/superpowers/plans/2026-05-07-v4-per-season-variance.md`

## TL;DR

[1 paragraph: verdict + key per-season finding + queue implication.]

## Anchors

[Table: weighted per-season aggregate vs audit overall, all 4 metrics.]

## Per-season metrics

[Compact table from JSON's per_season list.]

## Outliers (if any)

[Per outlier: season, metric, value, sigma_delta, n_games. Plus
narrative on what the outlier season was -- COVID, an upset-heavy
year, etc.]

## Cross-benchmark pattern

[Whether v4-vs-Vegas and v4-vs-538 deltas agree on which seasons are
outliers; flag any disagreements.]

## What this implies for the queue

[Concrete recommendation: which active-queue item moves up, what
specific experiment is unblocked or motivated.]

## Files of record

```
src/analyze_v4_per_season_variance.py
tests/test_analyze_v4_per_season_variance.py
output/v4_per_season_variance.json
output/v4_per_season_variance_traces.png
output/v4_per_season_variance_deltas.png
```
```

Fill in all bracketed placeholders from the actual JSON. Be specific
about the queue implication -- this note is the artifact future
sessions will read.

- [ ] **Step 2: Update `TODO.md`**

In the "Active queue" section:

- Move active queue #1 (single-season variance check) to "Done" with
  the verdict + key numbers + findings-note path. Use the 538 audit's
  TODO Done entry as the structural template.
- Promote items #2 and #3 to #1 and #2 based on the verdict:
  - If FLAT -> #1 calibration-shape engineering (was #3).
  - If OUTLIER -> #1 outlier-season investigation (new spec to be
    written), #2 calibration-shape engineering, #3 external data.
  - If TREND -> #1 trend deep-dive (new spec).
  - If MIXED -> retain current ordering, note ambiguity in preamble.
- Update the re-prioritization preamble to reflect what the variance
  check settled.

- [ ] **Step 3: Run full pytest sweep**

```bash
cd .claude/worktrees/feat-v4-per-season-variance
python -m pytest -q
```

Expected: all green. State which subsets ran in the final commit message.

- [ ] **Step 4: Final commit**

```bash
git add docs/notes/2026-05-07-v4-per-season-variance.md TODO.md
git commit -m "docs(per-season-variance): findings + TODO update

[One-line verdict.]

Findings: docs/notes/2026-05-07-v4-per-season-variance.md"
```

- [ ] **Step 5: Push branch and open PR**

```bash
git push -u origin feat/v4-per-season-variance
gh pr create --title "feat(per-season-variance): [verdict shorthand]" --body "$(cat <<'EOF'
## Summary

Per-season variance check for v4 across 22 LOSO seasons. Cheap diagnostic
gate before committing engineering budget to calibration-shape work.

## Verdict

[Verdict label + 1-line summary.]

## Anchors PASS

- weighted per-season ll_v4 (Vegas subset, 1326 games) reproduces audit's 0.5595
- weighted per-season ll_v4 (538 subset, 428 games) reproduces audit's 0.5799

## Test plan

- [x] pytest tests/test_analyze_v4_per_season_variance.py
- [x] pytest -q full suite
- [x] python src/analyze_v4_per_season_variance.py end-to-end
- [x] manual inspection of traces + deltas PNGs

## Files

- src/analyze_v4_per_season_variance.py
- tests/test_analyze_v4_per_season_variance.py
- output/v4_per_season_variance.{json,_traces.png,_deltas.png,_log.txt}
- docs/superpowers/specs/2026-05-07-v4-per-season-variance-design.md
- docs/superpowers/plans/2026-05-07-v4-per-season-variance.md
- docs/notes/2026-05-07-v4-per-season-variance.md
EOF
)"
```

---

## Risks (carried from spec, restated for the executor)

1. **Cross-module coupling.** This script imports private helpers from
   `src/audit_v4_gap_vegas.py` and `src/audit_v4_gap_fte.py`. If those
   modules are refactored, this analysis breaks. Mitigation: clear
   `noqa: E402` + comment in the import block; one-off diagnostic;
   don't make this a long-lived dependency.
2. **538 cache miss.** If `data/raw/fte_forecasts/<year>.csv` is not
   already populated, `load_fte_forecasts(allow_download=True)` fetches
   from Wayback Machine on the first run (~7 fetches, ~5-15s each).
   Mitigation: cache aggressively; cache-hit path is instantaneous on
   repeat runs.
3. **Anchor failure on real data.** Most likely cause: the per-season
   aggregator's weighting differs from the audit driver's overall
   aggregation. Spot-check by computing `len(audit_df)` per season vs
   per_season_n_games sum.
4. **2021 small-N.** 2021's COVID-era tournament had 4 cancellations;
   per-season metrics on n~50 are noisier than n~63. Findings note
   should explicitly call out 2021's reduced sample size if it's
   flagged as an outlier.
5. **538-only flag with low N.** 538 covers 7 seasons. Outlier flagging
   on `ll_v4_minus_fte` uses a 7-season mean/std -- intrinsically
   less stable than the 22-season Vegas std. Findings note should
   weigh Vegas-derived flags more heavily.
