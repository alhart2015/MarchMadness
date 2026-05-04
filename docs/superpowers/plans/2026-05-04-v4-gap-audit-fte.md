# v4 Gap Audit vs FiveThirtyEight -- Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a per-bucket diagnostic of where v4 specifically underperforms FiveThirtyEight's pre-tournament round-survival forecasts on tournament games. Single audit note + JSON + 3 PNGs. Mirrors the Vegas audit (PR 18) framework.

**Architecture:** Two new files. `src/ingest/fte_forecasts.py` downloads + caches + normalizes 538's per-team round-survival CSVs. `src/audit_v4_gap_fte.py` reuses the bucketing/metrics/plotting pattern from `src/audit_v4_gap_vegas.py` but consumes 538 data instead of Vegas spreads, and computes per-matchup probability via Bradley-Terry-style normalization on `rd_R_win` columns.

**Tech Stack:** pandas, numpy, scipy, requests (HTTP for 538 download), matplotlib (PNGs), pytest. No new external dependencies; `requests` is already in the project's transitive deps via cbbd.

**Spec:** `docs/superpowers/specs/2026-05-04-v4-gap-audit-fte-design.md`

---

## File Structure

**Created (committed):**

- `src/ingest/fte_forecasts.py` -- 538 forecast loader (~150 LOC).
  - Public: `load_fte_forecasts(years, cache_dir, allow_download=True) -> pd.DataFrame`
  - Public: `download_fte_forecasts(year, cache_dir) -> Path`
  - Private: `_fte_url_for_year(year) -> str`
  - Private: `_normalize_schema(df, year) -> pd.DataFrame`
  - Private: `_select_post_playin_snapshot(df) -> pd.DataFrame`
  - Module-level: `_FTE_URL_BY_YEAR` (dict of year -> URL with overrides)
  - Module-level: `_REQUIRED_COLS` (canonical column set)
- `src/audit_v4_gap_fte.py` -- driver script (~400 LOC).
  - Public: `run_audit(v4_csv, out_dir, out_json, fte_cache_dir) -> dict`
  - Private: `_bt_norm(p_a, p_b) -> float`
  - Private: `_round_index(round_label) -> int` (R64=1..Champ=6)
  - Private: helpers reused / paralleled from `src/audit_v4_gap_vegas.py`
    (`_v4_confidence_quintile`, `_seed_diff_bucket`, `_round_from_daynum`,
    `_calibration_table`, `_ece`, `_compute_bucket_metrics`).
- `tests/test_ingest/test_fte_forecasts.py` -- 5+ unit tests (~120 LOC).
- `tests/test_audit_v4_gap_fte.py` -- 6+ unit tests (~180 LOC).
- Spec + plan (this file).

**Generated (committed via `git add -f`):**

- `output/v4_gap_audit_fte.json` -- per-bucket metrics.
- `output/v4_gap_calibration_overall_fte.png`
- `output/v4_gap_calibration_by_round_fte.png`
- `output/v4_gap_per_bucket_ll_delta_fte.png`
- `output/v4_gap_audit_fte_log.txt`
- `docs/notes/2026-05-04-v4-gap-audit-fte.md` (findings).

**Generated (gitignored):**

- `data/raw/fte_forecasts/<year>.csv` -- cached 538 forecasts.

**Modified:**

- `.gitignore` -- add `data/raw/fte_forecasts/` if not already covered by an upstream rule.
- `TODO.md` -- mark 538 audit done with verdict; promote item #2 (single-season variance) or #3 (external data as features) to the new #1.

---

## Phase 1: 538 forecast loader + URL verification

### Task 1: Verify 538 URLs and schema across the 11 audit years

**Files:**
- Create: `scripts/verify_fte_urls.py` (one-shot, NOT committed)

This is a one-shot reconnaissance script -- run it, capture findings, populate `_FTE_URL_BY_YEAR` in Task 2. Do not commit it.

- [ ] **Step 1: Write the verification script**

Create `scripts/verify_fte_urls.py`:

```python
"""One-shot: verify 538 forecast URLs return HTTP 200 and contain the
expected canonical columns.

Run: python scripts/verify_fte_urls.py
Output: per-year status table; flag any year that needs an URL override.
NOT committed.
"""
from __future__ import annotations

import io
import sys
import urllib.error
import urllib.request

import pandas as pd

YEARS = [2014, 2015, 2016, 2017, 2018, 2019, 2021, 2022, 2023, 2024, 2025]

URL_TEMPLATE = (
    "https://raw.githubusercontent.com/fivethirtyeight/data/master/"
    "march-madness-predictions/{year}/fivethirtyeight_ncaa_forecasts.csv"
)

CANONICAL_COLS = {
    "gender", "forecast_date", "rd1_win", "rd2_win", "rd3_win",
    "rd4_win", "rd5_win", "rd6_win", "rd7_win",
    "team_id", "team_name", "team_seed",
}


def main() -> int:
    rc = 0
    for year in YEARS:
        url = URL_TEMPLATE.format(year=year)
        print(f"--- {year} ---")
        print(f"  url: {url}")
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                status = resp.status
                body = resp.read()
        except urllib.error.HTTPError as e:
            print(f"  HTTP {e.code}: {e.reason}")
            rc = 1
            continue
        except Exception as e:
            print(f"  ERROR: {type(e).__name__}: {e}")
            rc = 1
            continue
        print(f"  http: {status}, bytes: {len(body)}")
        try:
            df = pd.read_csv(io.BytesIO(body))
        except Exception as e:
            print(f"  CSV parse failed: {e}")
            rc = 1
            continue
        cols = set(df.columns)
        missing = CANONICAL_COLS - cols
        unexpected_close = {c for c in cols if any(
            c != canon and c.lower().replace(" ", "_") == canon for canon in CANONICAL_COLS
        )}
        print(f"  rows: {len(df)}, columns: {len(cols)}")
        if missing:
            print(f"  MISSING canonical columns: {sorted(missing)}")
            rc = 1
        if unexpected_close:
            print(f"  near-match columns (renamed?): {sorted(unexpected_close)}")
        gendervals = sorted(df["gender"].unique()) if "gender" in cols else []
        print(f"  gender values: {gendervals}")
        if "forecast_date" in cols:
            dates = sorted(df["forecast_date"].unique())
            print(f"  forecast_dates: {len(dates)}, first={dates[0]}, last={dates[-1]}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Run the verification script**

```bash
python scripts/verify_fte_urls.py | tee /tmp/fte_url_verification.txt
```

Expected: all 11 years HTTP 200, canonical columns present. Some years may use renamed columns (e.g., `team_id` vs `team_no`) -- record those for normalization in Task 2.

- [ ] **Step 3: Capture the findings**

Read `/tmp/fte_url_verification.txt`. Note any URL overrides needed and any column renames. These feed `_FTE_URL_BY_YEAR` and `_normalize_schema` in Task 2.

If any year fails, halt and investigate before proceeding -- the audit's coverage depends on having all 11 years.

- [ ] **Step 4: Delete the script**

```bash
rm scripts/verify_fte_urls.py
rmdir scripts 2>/dev/null || true
```

This was reconnaissance, not committed code.

### Task 2: Implement `src/ingest/fte_forecasts.py`

**Files:**
- Create: `src/ingest/fte_forecasts.py`
- Create: `tests/test_ingest/test_fte_forecasts.py`

- [ ] **Step 1: Write failing unit tests for the loader**

Create `tests/test_ingest/test_fte_forecasts.py`:

```python
"""Unit tests for src/ingest/fte_forecasts.py."""
from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import pytest

from src.ingest.fte_forecasts import (
    _FTE_URL_BY_YEAR,
    _REQUIRED_COLS,
    _fte_url_for_year,
    _normalize_schema,
    _select_post_playin_snapshot,
    load_fte_forecasts,
)


def _synthetic_pre_post_playin_df():
    """Two snapshots: 2024-03-19 (pre-play-in, 68 teams) and 2024-03-21
    (post-play-in, 64 teams). Mens only."""
    rows = []
    # Pre-play-in snapshot: 68 teams, 4 with playin_flag=1
    for tid in range(1, 69):
        rows.append({
            "gender": "mens",
            "forecast_date": "2024-03-19",
            "playin_flag": 1 if tid > 64 else 0,
            "team_id": tid,
            "team_name": f"Team {tid}",
            "team_seed": "01",
            "rd1_win": 0.5,
            "rd2_win": 0.25,
            "rd3_win": 0.125,
            "rd4_win": 0.0625,
            "rd5_win": 0.03125,
            "rd6_win": 0.015625,
            "rd7_win": 0.0078125,
        })
    # Post-play-in snapshot: 64 teams, all playin_flag=0
    for tid in range(1, 65):
        rows.append({
            "gender": "mens",
            "forecast_date": "2024-03-21",
            "playin_flag": 0,
            "team_id": tid,
            "team_name": f"Team {tid}",
            "team_seed": "01",
            "rd1_win": 0.5,
            "rd2_win": 0.25,
            "rd3_win": 0.125,
            "rd4_win": 0.0625,
            "rd5_win": 0.03125,
            "rd6_win": 0.015625,
            "rd7_win": 0.0078125,
        })
    return pd.DataFrame(rows)


def test_url_for_each_year_resolves():
    for y in [2014, 2019, 2021, 2025]:
        url = _fte_url_for_year(y)
        assert url.startswith("https://raw.githubusercontent.com/")
        assert str(y) in url


def test_url_year_2020_raises():
    """No 2020 tournament; loader should refuse this year up-front."""
    with pytest.raises((KeyError, ValueError)):
        _fte_url_for_year(2020)


def test_normalize_schema_required_columns_present():
    df = _synthetic_pre_post_playin_df()
    out = _normalize_schema(df, year=2024)
    for col in _REQUIRED_COLS:
        assert col in out.columns, f"missing canonical column: {col}"


def test_normalize_schema_missing_required_col_raises():
    df = _synthetic_pre_post_playin_df().drop(columns=["rd5_win"])
    with pytest.raises(ValueError, match="rd5_win"):
        _normalize_schema(df, year=2024)


def test_select_post_playin_snapshot_picks_64_team_field():
    """Earliest snapshot with exactly 64 unique team_ids and no
    playin_flag=1 row."""
    df = _synthetic_pre_post_playin_df()
    out = _select_post_playin_snapshot(df)
    assert len(out) == 64
    assert (out["playin_flag"] == 0).all()
    assert out["forecast_date"].unique().tolist() == ["2024-03-21"]


def test_select_post_playin_snapshot_disagreement_fails_loud():
    """If the 64-team rule and the playin_flag=0 rule disagree, raise."""
    df = _synthetic_pre_post_playin_df()
    # Mutate: post-play-in snapshot has 65 rows but all playin_flag=0
    extra = df[df["forecast_date"] == "2024-03-21"].iloc[0:1].copy()
    extra["team_id"] = 999
    extra["team_name"] = "Team 999"
    df_bad = pd.concat([df, extra], ignore_index=True)
    with pytest.raises(ValueError, match="post-play-in"):
        _select_post_playin_snapshot(df_bad)


def test_load_fte_forecasts_uses_cache(tmp_path):
    """If cache file exists, loader reads from disk and does not download."""
    cache = tmp_path / "fte_forecasts"
    cache.mkdir()
    df = _synthetic_pre_post_playin_df()
    df.to_csv(cache / "2024.csv", index=False)
    out = load_fte_forecasts(years=[2024], cache_dir=cache,
                             allow_download=False)
    assert len(out) == 64
    assert "Season" in out.columns
    assert (out["Season"] == 2024).all()


def test_load_fte_forecasts_filters_to_mens(tmp_path):
    """Women's rows are dropped before snapshot selection."""
    cache = tmp_path / "fte_forecasts"
    cache.mkdir()
    df = _synthetic_pre_post_playin_df()
    womens = df.head(2).copy()
    womens["gender"] = "womens"
    pd.concat([df, womens], ignore_index=True).to_csv(
        cache / "2024.csv", index=False)
    out = load_fte_forecasts(years=[2024], cache_dir=cache,
                             allow_download=False)
    assert (out["gender"] == "mens").all() if "gender" in out.columns else True
    assert len(out) == 64
```

Run: `pytest -v tests/test_ingest/test_fte_forecasts.py` -- expect ImportError.

- [ ] **Step 2: Implement `src/ingest/fte_forecasts.py`**

```python
"""Load FiveThirtyEight pre-tournament round-survival forecasts.

Source: https://github.com/fivethirtyeight/data/tree/master/march-madness-predictions
One CSV per tournament year, one row per team per forecast snapshot.

Public API:
    load_fte_forecasts(years, cache_dir, allow_download=True) -> pd.DataFrame
        Returns one row per (Season, team_id) with columns
        [Season, team_id, team_name, team_seed, rd1_win, ..., rd7_win,
         forecast_date, playin_flag, gender].
        Filtered to gender='mens', earliest post-play-in snapshot per
        season (rd1_win = P(win R64 game)).

    download_fte_forecasts(year, cache_dir) -> Path
        HTTP GET the CSV for `year` and write to
        cache_dir / f'{year}.csv'. Returns the path.

Spec: docs/superpowers/specs/2026-05-04-v4-gap-audit-fte-design.md
"""
from __future__ import annotations

import io
import logging
import urllib.error
import urllib.request
from pathlib import Path
from typing import Iterable

import pandas as pd

logger = logging.getLogger(__name__)

# Per-year URL overrides; default is the URL_TEMPLATE pattern, populated
# from the Task 1 verification run. If a year needs a different path
# (e.g., 538 reorganized the repo for that year), override here.
_URL_TEMPLATE = (
    "https://raw.githubusercontent.com/fivethirtyeight/data/master/"
    "march-madness-predictions/{year}/fivethirtyeight_ncaa_forecasts.csv"
)
_FTE_URL_BY_YEAR: dict[int, str] = {
    # year: override_url  (empty = use _URL_TEMPLATE)
    # populated after Task 1 reconnaissance
}

_AUDITED_YEARS = (2014, 2015, 2016, 2017, 2018, 2019,
                  2021, 2022, 2023, 2024, 2025)

_REQUIRED_COLS = (
    "gender", "forecast_date",
    "rd1_win", "rd2_win", "rd3_win", "rd4_win",
    "rd5_win", "rd6_win", "rd7_win",
    "team_id", "team_name", "team_seed",
)

# Schema renames seen across years (from Task 1 reconnaissance).
# Map old-name -> canonical-name. If a year has a column whose canonical
# name is missing but an alias is present, _normalize_schema renames it.
_COLUMN_ALIASES: dict[str, str] = {
    # populated from Task 1 if any aliases were observed; example:
    # "team_no": "team_id",
}


def _fte_url_for_year(year: int) -> str:
    if year not in _AUDITED_YEARS:
        raise ValueError(
            f"Year {year} not in audited set {_AUDITED_YEARS}. "
            f"(2020 has no tournament.)"
        )
    return _FTE_URL_BY_YEAR.get(year) or _URL_TEMPLATE.format(year=year)


def download_fte_forecasts(year: int, cache_dir: Path) -> Path:
    """Download 538 forecasts CSV for `year` to cache_dir/{year}.csv.
    Returns the cached file path.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_path = cache_dir / f"{year}.csv"
    url = _fte_url_for_year(year)
    logger.info("downloading 538 forecasts: %s -> %s", url, out_path)
    try:
        with urllib.request.urlopen(url, timeout=60) as resp:
            body = resp.read()
    except urllib.error.HTTPError as e:
        raise RuntimeError(
            f"HTTP {e.code} fetching {url}: {e.reason}"
        ) from e
    out_path.write_bytes(body)
    return out_path


def _normalize_schema(df: pd.DataFrame, year: int) -> pd.DataFrame:
    """Apply per-year column aliases and verify all required cols present."""
    df = df.rename(columns=_COLUMN_ALIASES)
    missing = [c for c in _REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"538 forecasts CSV for year {year} missing required columns: "
            f"{missing}. Columns present: {sorted(df.columns)}"
        )
    return df


def _select_post_playin_snapshot(df: pd.DataFrame) -> pd.DataFrame:
    """Per season, pick the earliest snapshot where the field is exactly
    64 teams AND every row has playin_flag=0 (or playin_flag missing).

    Raises ValueError if no such snapshot exists or if the two conditions
    disagree on any candidate snapshot.
    """
    if "playin_flag" not in df.columns:
        df = df.assign(playin_flag=0)

    out_rows = []
    for season, season_df in df.groupby("Season"):
        snap_dates = sorted(season_df["forecast_date"].unique())
        chosen = None
        for d in snap_dates:
            snap = season_df[season_df["forecast_date"] == d]
            n_teams = snap["team_id"].nunique()
            all_post = (snap["playin_flag"] == 0).all()
            if n_teams == 64 and all_post:
                chosen = snap
                break
            if n_teams == 64 and not all_post:
                raise ValueError(
                    f"season={season} forecast_date={d}: 64 teams but "
                    f"some playin_flag != 0 -- post-play-in detection "
                    f"rules disagree, halt for manual review"
                )
            if n_teams != 64 and all_post:
                # Could be a 65+ snapshot with some duplicates; warn loudly
                logger.warning(
                    "season=%s forecast_date=%s: all playin_flag=0 but "
                    "team count = %d != 64; skipping",
                    season, d, n_teams,
                )
        if chosen is None:
            raise ValueError(
                f"season={season}: no post-play-in snapshot found "
                f"(checked {len(snap_dates)} forecast_dates)"
            )
        out_rows.append(chosen)
    return pd.concat(out_rows, ignore_index=True)


def load_fte_forecasts(
    years: Iterable[int],
    cache_dir: str | Path,
    allow_download: bool = True,
) -> pd.DataFrame:
    """Load 538 pre-tournament forecasts for the given years.

    Returns one DataFrame with columns:
        Season, forecast_date, gender, playin_flag, team_id, team_name,
        team_seed, rd1_win .. rd7_win.
    Filtered to gender='mens', earliest post-play-in snapshot per season.
    """
    cache_dir = Path(cache_dir)
    frames = []
    for year in years:
        path = cache_dir / f"{year}.csv"
        if not path.exists():
            if not allow_download:
                raise FileNotFoundError(
                    f"538 forecast cache miss for {year} at {path} and "
                    f"allow_download=False"
                )
            download_fte_forecasts(year, cache_dir)
        raw = pd.read_csv(path)
        raw = _normalize_schema(raw, year=year)
        raw = raw[raw["gender"] == "mens"].copy()
        raw["Season"] = int(year)
        frames.append(raw)
    if not frames:
        return pd.DataFrame(columns=["Season"] + list(_REQUIRED_COLS))
    df = pd.concat(frames, ignore_index=True)
    df = _select_post_playin_snapshot(df)
    return df.reset_index(drop=True)
```

- [ ] **Step 3: Run the loader unit tests**

```bash
pytest -v tests/test_ingest/test_fte_forecasts.py
```

Expected: PASS.

If any test fails, fix the implementation -- the tests assert the canonical contract that the audit driver depends on.

- [ ] **Step 4: Smoke-test the loader against real data**

```bash
mkdir -p data/raw/fte_forecasts
python -c "
from pathlib import Path
from src.ingest.fte_forecasts import load_fte_forecasts
df = load_fte_forecasts(
    years=[2024],
    cache_dir=Path('data/raw/fte_forecasts'),
    allow_download=True,
)
print(df.shape)
print(df.columns.tolist())
print(df['Season'].unique())
print('rd1_win sum check:', df['rd1_win'].sum())
print('team count:', df['team_id'].nunique())
print('forecast_dates:', df['forecast_date'].unique())
"
```

Expected:
- Shape: `(64, ~14)`
- 64 unique team_ids
- One forecast_date (the earliest post-play-in)
- `rd1_win` sum should be approximately 32 (each R64 matchup contributes ~1.0; 32 matchups -> sum ~32). Tolerance: within 0.5 absolute due to 538 rounding.

If the smoke test fails, debug before continuing.

- [ ] **Step 5: Commit Phase 1**

```bash
git add src/ingest/fte_forecasts.py tests/test_ingest/test_fte_forecasts.py
git commit -m "feat(fte-forecasts): loader for 538 round-survival CSVs

- HTTP fetch + per-year cache under data/raw/fte_forecasts/
- gender='mens' filter
- earliest post-play-in snapshot per season (rd1_win = R64 prob)
- column-alias normalization with explicit _REQUIRED_COLS check
- 7 unit tests covering URL/schema/snapshot/cache behavior

Spec: docs/superpowers/specs/2026-05-04-v4-gap-audit-fte-design.md"
```

---

## Phase 2: Audit driver + findings

### Task 3: Implement `src/audit_v4_gap_fte.py`

**Files:**
- Create: `src/audit_v4_gap_fte.py`
- Create: `tests/test_audit_v4_gap_fte.py`

- [ ] **Step 1: Write failing unit tests**

Create `tests/test_audit_v4_gap_fte.py`:

```python
"""Unit tests for src/audit_v4_gap_fte.py."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.audit_v4_gap_fte import (
    _bt_norm,
    _calibration_table,
    _compute_bucket_metrics,
    _ece,
    _round_from_daynum,
    _round_index,
    _seed_diff_bucket,
    _v4_confidence_quintile,
)


def test_bt_norm_basic():
    """Standard BT normalization."""
    assert _bt_norm(0.6, 0.4) == pytest.approx(0.6)
    assert _bt_norm(0.5, 0.5) == pytest.approx(0.5)
    assert _bt_norm(0.8, 0.2) == pytest.approx(0.8)


def test_bt_norm_with_538_rounding():
    """538 publishes rounded probs that don't exactly sum to 1; BT-norm
    handles by dividing by the actual sum."""
    out = _bt_norm(0.601, 0.401)
    assert 0.595 < out < 0.605
    expected = 0.601 / (0.601 + 0.401)
    assert out == pytest.approx(expected)


def test_bt_norm_zero_zero_safe():
    """Both teams have zero R-round survival prob (never reaches R in
    538's view). BT-norm returns 0.5 with a warning rather than NaN."""
    out = _bt_norm(0.0, 0.0)
    assert out == pytest.approx(0.5)


def test_round_index_mapping():
    """R64=1, R32=2, S16=3, E8=4, F4=5, Champ=6."""
    assert _round_index("R64") == 1
    assert _round_index("R32") == 2
    assert _round_index("S16") == 3
    assert _round_index("E8") == 4
    assert _round_index("F4") == 5
    assert _round_index("Champ") == 6


def test_round_from_daynum_canonical():
    """Same convention as Vegas audit: 134-135 FF, 136-137 R64, etc."""
    assert _round_from_daynum(136) == "R64"
    assert _round_from_daynum(138) == "R32"
    assert _round_from_daynum(143) == "S16"
    assert _round_from_daynum(145) == "E8"
    assert _round_from_daynum(152) == "F4"
    assert _round_from_daynum(154) == "Champ"
    assert _round_from_daynum(134) == "FF"


def test_seed_diff_bucket_boundaries():
    assert _seed_diff_bucket(0) == "0-2"
    assert _seed_diff_bucket(3) == "3-5"
    assert _seed_diff_bucket(6) == "6-9"
    assert _seed_diff_bucket(10) == "10-15"
    assert _seed_diff_bucket(15) == "10-15"


def test_v4_confidence_quintile_boundaries():
    assert _v4_confidence_quintile(0.55) == "0.50-0.60"
    assert _v4_confidence_quintile(0.61) == "0.60-0.70"
    assert _v4_confidence_quintile(0.95) == "0.90-1.00"
    # Below-0.5 mirrors to favored side
    assert _v4_confidence_quintile(0.40) == "0.50-0.60"
    assert _v4_confidence_quintile(0.05) == "0.90-1.00"


def test_calibration_table_perfect():
    rng = np.random.default_rng(0)
    n = 5000
    p = rng.uniform(0.5, 1.0, size=n)
    y = (rng.random(n) < p).astype(int)
    table = _calibration_table(p, y, n_bins=10)
    assert _ece(table) < 0.03


def test_compute_bucket_metrics_aggregates_correctly():
    """Three games, one bucket; LL + accuracy hand-computed."""
    df = pd.DataFrame([
        {"bucket": "R64", "p_v4": 0.8, "p_fte": 0.7, "winner_is_a": 1},
        {"bucket": "R64", "p_v4": 0.6, "p_fte": 0.55, "winner_is_a": 1},
        {"bucket": "R64", "p_v4": 0.4, "p_fte": 0.5, "winner_is_a": 0},
    ])
    by = _compute_bucket_metrics(df, "bucket")
    cell = by["R64"]
    assert cell["n_games"] == 3
    expected_ll = -np.mean([np.log(0.8), np.log(0.6), np.log(0.6)])
    assert cell["ll_v4"] == pytest.approx(expected_ll)
    # acc_v4: 0.8>0.5 hit; 0.6>0.5 hit; 0.4<0.5 hit (winner_is_a=0)
    assert cell["acc_v4"] == pytest.approx(1.0)
```

Run: `pytest -v tests/test_audit_v4_gap_fte.py` -- expect ImportError.

- [ ] **Step 2: Implement `src/audit_v4_gap_fte.py`**

This file is large; mirror `src/audit_v4_gap_vegas.py` closely. Three differences from the Vegas driver:

1. Loads 538 forecasts via `load_fte_forecasts(...)` instead of `load_vegas_lines(...)`.
2. Computes `p_fte` per matchup via `_bt_norm(rd_R_win[a], rd_R_win[b])` where R is read from the joined game's round.
3. Outputs use `_fte` suffixes (JSON keys say `ll_fte`, `acc_fte`, `ece_fte`; PNG filenames have `_fte` suffix).

Top of file:

```python
"""Audit v4's tournament-game predictions vs FiveThirtyEight pre-
tournament round-survival forecasts, broken down by round, higher-vs-
lower-seed status, v4-confidence quintile, and seed-difference
magnitude.

Spec: docs/superpowers/specs/2026-05-04-v4-gap-audit-fte-design.md

Output:
    output/v4_gap_audit_fte.json
    output/v4_gap_calibration_overall_fte.png
    output/v4_gap_calibration_by_round_fte.png
    output/v4_gap_per_bucket_ll_delta_fte.png
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

from src.ingest.fte_forecasts import load_fte_forecasts
from src.ingest.team_mapping import build_team_mapping
from src.config import load_config

logger = logging.getLogger(__name__)

DATA = Path("data/raw/march-machine-learning-2026")
DEFAULT_FTE_CACHE = Path("data/raw/fte_forecasts")
DEFAULT_OUT_JSON = "output/v4_gap_audit_fte.json"
DEFAULT_OUT_DIR = "output"

AUDITED_YEARS = (2014, 2015, 2016, 2017, 2018, 2019,
                 2021, 2022, 2023, 2024, 2025)

ROUND_BY_DAYNUM = {
    134: "FF", 135: "FF",
    136: "R64", 137: "R64",
    138: "R32", 139: "R32",
    143: "S16", 144: "S16",
    145: "E8",  146: "E8",
    152: "F4",  153: "F4",
    154: "Champ",
}
ROUND_INDEX = {"R64": 1, "R32": 2, "S16": 3, "E8": 4, "F4": 5, "Champ": 6}
ROUND_ORDER = ["R64", "R32", "S16", "E8", "F4", "Champ"]

CONFIDENCE_BIN_EDGES = [0.50, 0.60, 0.70, 0.80, 0.90, 1.00]
CONFIDENCE_BIN_LABELS = ["0.50-0.60", "0.60-0.70", "0.70-0.80",
                         "0.80-0.90", "0.90-1.00"]

WEAK_SPOT_MIN_N = 50
WEAK_SPOT_MIN_LL_DELTA = 0.02


def _bt_norm(p_a: float, p_b: float) -> float:
    """Bradley-Terry normalization: P(A wins | both reach round R) =
    rd_R_win[A] / (rd_R_win[A] + rd_R_win[B]).

    Defensive: if both inputs are zero (538 says neither team reaches R),
    return 0.5 -- this is unreachable for actual played matchups but
    keeps the function total.
    """
    p_a = float(p_a)
    p_b = float(p_b)
    s = p_a + p_b
    if s <= 0.0:
        logger.warning("_bt_norm: both inputs zero (%s, %s); returning 0.5",
                       p_a, p_b)
        return 0.5
    return p_a / s


def _round_index(round_label: str) -> int:
    return ROUND_INDEX[round_label]


def _round_from_daynum(daynum: int) -> str:
    return ROUND_BY_DAYNUM.get(int(daynum), "OTHER")


def _v4_confidence_quintile(p_a: float) -> str:
    p_fav = max(p_a, 1.0 - p_a)
    for lo, hi, label in zip(CONFIDENCE_BIN_EDGES[:-1],
                             CONFIDENCE_BIN_EDGES[1:],
                             CONFIDENCE_BIN_LABELS):
        if lo <= p_fav <= hi:
            return label
    return CONFIDENCE_BIN_LABELS[-1]


def _seed_diff_bucket(d: int) -> str:
    if d <= 2:
        return "0-2"
    if d <= 5:
        return "3-5"
    if d <= 9:
        return "6-9"
    return "10-15"


def _calibration_table(p_pred, y_actual, n_bins: int = 10):
    edges = np.linspace(0, 1, n_bins + 1)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (p_pred >= lo) & (p_pred < hi if hi < 1.0 else p_pred <= hi)
        n = int(mask.sum())
        empirical = float(y_actual[mask].mean()) if n else None
        out.append({
            "bin": [float(lo), float(hi)],
            "mid": float((lo + hi) / 2),
            "n": n,
            "empirical": empirical,
        })
    return out


def _ece(cal_table) -> float:
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
    out = {}
    for value, sub in df.groupby(bucket_col):
        n = len(sub)
        if n == 0:
            continue
        eps = 1e-15
        p_v4_w = np.where(sub["winner_is_a"] == 1, sub["p_v4"], 1 - sub["p_v4"])
        p_fte_w = np.where(sub["winner_is_a"] == 1, sub["p_fte"], 1 - sub["p_fte"])
        ll_v4 = float(-np.mean(np.log(np.clip(p_v4_w, eps, 1 - eps))))
        ll_fte = float(-np.mean(np.log(np.clip(p_fte_w, eps, 1 - eps))))
        acc_v4 = float(((sub["p_v4"] >= 0.5).astype(int) == sub["winner_is_a"]).mean())
        acc_fte = float(((sub["p_fte"] >= 0.5).astype(int) == sub["winner_is_a"]).mean())
        cal_v4 = _calibration_table(sub["p_v4"].to_numpy(),
                                    sub["winner_is_a"].to_numpy())
        cal_fte = _calibration_table(sub["p_fte"].to_numpy(),
                                     sub["winner_is_a"].to_numpy())
        out[str(value)] = {
            "n_games": int(n),
            "ll_v4": ll_v4,
            "ll_fte": ll_fte,
            "ll_delta": ll_v4 - ll_fte,
            "acc_v4": acc_v4,
            "acc_fte": acc_fte,
            "ece_v4": _ece(cal_v4),
            "ece_fte": _ece(cal_fte),
            "mean_p_v4_minus_fte": float((sub["p_v4"] - sub["p_fte"]).mean()),
            "calibration_v4": cal_v4,
            "calibration_fte": cal_fte,
        }
    return out
```

Continuing the same file with the join + main:

```python
def _resolve_fte_team_ids(fte_df: pd.DataFrame, mapping: pd.DataFrame) -> pd.DataFrame:
    """Add Kaggle TeamID column via the existing fuzzy-matcher mapping.
    Fails loud if any 538 team_name is unresolved.
    """
    # mapping: DataFrame with columns [TeamID, ext_name, source, ...]
    # 538 names are in fte_df['team_name']; resolve via the mapping built
    # for 'fte' source. Assumes build_team_mapping was called with 538 names.
    fte_to_kaggle = dict(zip(mapping["ext_name"], mapping["TeamID"]))
    out = fte_df.copy()
    out["TeamID"] = out["team_name"].map(fte_to_kaggle)
    unresolved = out[out["TeamID"].isna()]["team_name"].unique()
    if len(unresolved):
        raise ValueError(
            f"{len(unresolved)} 538 team names unresolved to TeamID; "
            f"add overrides in data/team_name_overrides.csv. "
            f"First few: {sorted(unresolved)[:10]}"
        )
    out["TeamID"] = out["TeamID"].astype(int)
    return out


def _build_per_game_audit_df(
    v4_df: pd.DataFrame,
    fte_df: pd.DataFrame,
    results_df: pd.DataFrame,
    seeds_df: pd.DataFrame,
) -> pd.DataFrame:
    """Per-game DataFrame keyed by (Season, DayNum, team_a, team_b).

    v4_df: pairwise probabilities, columns [Season, TeamID_a, TeamID_b, p_v4]
           with TeamID_a < TeamID_b.
    fte_df: per-team round-survival, columns [Season, TeamID, rd1_win..rd7_win].
    results_df: MNCAATourneyCompactResults [Season, DayNum, WTeamID, LTeamID].
    seeds_df: MNCAATourneySeeds [Season, TeamID, seed_int].

    Output columns:
        Season, DayNum, round, team_a, team_b, p_v4, p_fte,
        winner_is_a, seed_a, seed_b, seed_diff,
        higher_seed_won, v4_confidence_quintile, seed_diff_bucket
    """
    # Filter to audited seasons + non-FF games
    res = results_df[results_df["Season"].isin(AUDITED_YEARS)].copy()
    res["round"] = res["DayNum"].map(_round_from_daynum)
    res = res[res["round"].isin(ROUND_ORDER)].copy()  # drops FF

    # Canonical team_a < team_b
    res["team_a"] = np.minimum(res["WTeamID"], res["LTeamID"])
    res["team_b"] = np.maximum(res["WTeamID"], res["LTeamID"])
    res["winner_is_a"] = (res["WTeamID"] == res["team_a"]).astype(int)

    # Attach v4 prob (p_v4 for team_a)
    v4_keyed = v4_df.rename(columns={"TeamID_a": "team_a", "TeamID_b": "team_b"})
    df = res.merge(v4_keyed[["Season", "team_a", "team_b", "p_v4"]],
                   on=["Season", "team_a", "team_b"], how="left")

    # Attach 538 rd_R_win for both teams (one row per (Season, TeamID))
    rd_cols = [f"rd{r}_win" for r in range(1, 8)]
    fte_a = fte_df[["Season", "TeamID"] + rd_cols].rename(
        columns={"TeamID": "team_a", **{c: f"a_{c}" for c in rd_cols}})
    fte_b = fte_df[["Season", "TeamID"] + rd_cols].rename(
        columns={"TeamID": "team_b", **{c: f"b_{c}" for c in rd_cols}})
    df = df.merge(fte_a, on=["Season", "team_a"], how="left")
    df = df.merge(fte_b, on=["Season", "team_b"], how="left")

    # Compute p_fte for team_a in each game's round
    def pick_rd(row, side):
        r = ROUND_INDEX[row["round"]]
        return row[f"{side}_rd{r}_win"]
    df["a_rd_R"] = df.apply(lambda r: pick_rd(r, "a"), axis=1)
    df["b_rd_R"] = df.apply(lambda r: pick_rd(r, "b"), axis=1)
    df["p_fte"] = df.apply(
        lambda r: _bt_norm(r["a_rd_R"], r["b_rd_R"])
        if pd.notna(r["a_rd_R"]) and pd.notna(r["b_rd_R"])
        else float("nan"),
        axis=1,
    )

    # Attach seeds + derived buckets
    seeds_keyed = seeds_df.rename(columns={"TeamID": "team_a", "seed_int": "seed_a"})
    df = df.merge(seeds_keyed[["Season", "team_a", "seed_a"]],
                  on=["Season", "team_a"], how="left")
    seeds_keyed_b = seeds_df.rename(columns={"TeamID": "team_b", "seed_int": "seed_b"})
    df = df.merge(seeds_keyed_b[["Season", "team_b", "seed_b"]],
                  on=["Season", "team_b"], how="left")
    df["seed_diff"] = (df["seed_a"] - df["seed_b"]).abs().astype(int)
    df["seed_diff_bucket"] = df["seed_diff"].map(_seed_diff_bucket)
    df["higher_seed_won"] = (
        ((df["seed_a"] < df["seed_b"]) & (df["winner_is_a"] == 1))
        | ((df["seed_a"] > df["seed_b"]) & (df["winner_is_a"] == 0))
    ).astype(bool)
    df["v4_confidence_quintile"] = df["p_v4"].map(_v4_confidence_quintile)

    # Drop rows missing v4 or fte probability
    return df.dropna(subset=["p_v4", "p_fte"]).reset_index(drop=True)


def _save_calibration_overall_plot(df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    for label, col, color in [("v4", "p_v4", "C0"), ("538", "p_fte", "C1")]:
        cal = _calibration_table(df[col].to_numpy(), df["winner_is_a"].to_numpy())
        xs = [b["mid"] for b in cal if b["empirical"] is not None]
        ys = [b["empirical"] for b in cal if b["empirical"] is not None]
        ax.plot(xs, ys, marker="o", label=label, color=color)
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.5, label="diagonal")
    ax.set_xlabel("predicted probability")
    ax.set_ylabel("empirical win rate")
    ax.set_title("v4 vs 538 calibration (overall)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _save_calibration_by_round_plot(df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(12, 8), sharex=True, sharey=True)
    for ax, rnd in zip(axes.flat, ROUND_ORDER):
        sub = df[df["round"] == rnd]
        for label, col, color in [("v4", "p_v4", "C0"), ("538", "p_fte", "C1")]:
            if len(sub) == 0:
                continue
            cal = _calibration_table(sub[col].to_numpy(),
                                     sub["winner_is_a"].to_numpy())
            xs = [b["mid"] for b in cal if b["empirical"] is not None]
            ys = [b["empirical"] for b in cal if b["empirical"] is not None]
            ax.plot(xs, ys, marker="o", label=label, color=color)
        ax.plot([0, 1], [0, 1], "k--", linewidth=0.5)
        ax.set_title(f"{rnd} (n={len(sub)})")
        ax.set_xlabel("predicted")
        ax.set_ylabel("empirical")
    axes[0, 0].legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _save_per_bucket_bar_plot(by_bucket_blocks, out_path: Path) -> None:
    """Horizontal bar of ll_delta = ll_v4 - ll_fte per bucket cell."""
    rows = []
    for prefix, blk in by_bucket_blocks.items():
        for value, cell in blk.items():
            rows.append((f"{prefix}={value}", cell["ll_delta"], cell["n_games"]))
    rows.sort(key=lambda r: r[1])  # ascending; most v4-favorable on top
    labels = [r[0] for r in rows]
    deltas = [r[1] for r in rows]
    counts = [r[2] for r in rows]
    fig, ax = plt.subplots(figsize=(8, max(4, 0.3 * len(rows))))
    bars = ax.barh(labels, deltas)
    for bar, n in zip(bars, counts):
        ax.text(bar.get_width(), bar.get_y() + bar.get_height() / 2,
                f"  n={n}", va="center", fontsize=8)
    ax.axvline(0, color="k", linewidth=0.5)
    ax.set_xlabel("ll_v4 - ll_fte (positive = v4 worse)")
    ax.set_title("v4 vs 538 per-bucket log-loss delta")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _extract_weak_spots(by_blocks: dict, min_n: int, min_delta: float) -> list[dict]:
    out = []
    for prefix, blk in by_blocks.items():
        for value, cell in blk.items():
            if cell["n_games"] >= min_n and cell["ll_delta"] >= min_delta:
                out.append({
                    "bucket": f"{prefix}={value}",
                    "n_games": cell["n_games"],
                    "ll_v4": cell["ll_v4"],
                    "ll_fte": cell["ll_fte"],
                    "ll_delta": cell["ll_delta"],
                    "acc_v4": cell["acc_v4"],
                    "acc_fte": cell["acc_fte"],
                })
    out.sort(key=lambda c: -c["ll_delta"])
    return out


def run_audit(v4_csv: Path, out_dir: Path, out_json: Path,
              fte_cache_dir: Path) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("loading v4 pairwise probabilities from %s", v4_csv)
    v4 = pd.read_csv(v4_csv)
    # v4 pairwise CSV has TeamID_a/TeamID_b/Season/p_v4 (or similar);
    # adapt column names per the actual schema in output/pairwise_v4.csv.
    # The Vegas audit's _join_v4_vegas_outcomes is the reference for field names.
    v4 = v4[v4["Season"].isin(AUDITED_YEARS)].copy()

    logger.info("loading 538 forecasts (cache=%s)", fte_cache_dir)
    fte = load_fte_forecasts(years=AUDITED_YEARS, cache_dir=fte_cache_dir)

    # Build name-to-TeamID mapping for 538 names.
    # Use existing build_team_mapping pattern; the 538 'team_name' column
    # is the external name source.
    config = load_config()
    fte_names = sorted(fte["team_name"].unique())
    mapping = build_team_mapping(
        external_names=fte_names,
        source="fte",
        config=config,
    )
    fte = _resolve_fte_team_ids(fte, mapping)

    logger.info("loading tourney outcomes + seeds")
    results = pd.read_csv(DATA / "MNCAATourneyCompactResults.csv")
    seeds = pd.read_csv(DATA / "MNCAATourneySeeds.csv")
    seeds["seed_int"] = seeds["Seed"].str.extract(r"(\d+)").astype(int)

    df = _build_per_game_audit_df(v4, fte, results, seeds)

    by_blocks = {
        "round": _compute_bucket_metrics(df, "round"),
        "higher_seed_won": _compute_bucket_metrics(df, "higher_seed_won"),
        "v4_confidence_quintile": _compute_bucket_metrics(df, "v4_confidence_quintile"),
        "seed_diff_bucket": _compute_bucket_metrics(df, "seed_diff_bucket"),
    }

    # Overall metrics: treat the whole df as one bucket
    df_one = df.copy()
    df_one["__all"] = "all"
    overall = _compute_bucket_metrics(df_one, "__all")["all"]

    # Coverage stats
    total_audited_games = len(
        results[results["Season"].isin(AUDITED_YEARS)]
            .assign(round=lambda r: r["DayNum"].map(_round_from_daynum))
            .query("round in @ROUND_ORDER")
    )
    coverage = {
        "n_tournament_games": int(total_audited_games),
        "n_with_v4": int(df["p_v4"].notna().sum()),
        "n_with_fte": int(df["p_fte"].notna().sum()),
        "n_both": int(len(df)),
        "audited_seasons": list(AUDITED_YEARS),
    }

    weak_spots = _extract_weak_spots(by_blocks, WEAK_SPOT_MIN_N,
                                     WEAK_SPOT_MIN_LL_DELTA)

    audit = {
        "config": {
            "v4_pairwise": str(v4_csv),
            "fte_cache_dir": str(fte_cache_dir),
            "seasons": list(AUDITED_YEARS),
            "snapshot_policy": "earliest_post_playin_per_season",
            "weak_spot_min_n": WEAK_SPOT_MIN_N,
            "weak_spot_min_ll_delta": WEAK_SPOT_MIN_LL_DELTA,
        },
        "join_coverage": coverage,
        "overall": overall,
        "by_round": by_blocks["round"],
        "by_higher_seed_won": by_blocks["higher_seed_won"],
        "by_v4_confidence_quintile": by_blocks["v4_confidence_quintile"],
        "by_seed_diff": by_blocks["seed_diff_bucket"],
        "weak_spots": weak_spots,
    }

    Path(out_json).write_text(json.dumps(audit, indent=2, default=float))

    _save_calibration_overall_plot(df, out_dir / "v4_gap_calibration_overall_fte.png")
    _save_calibration_by_round_plot(df, out_dir / "v4_gap_calibration_by_round_fte.png")
    _save_per_bucket_bar_plot(by_blocks, out_dir / "v4_gap_per_bucket_ll_delta_fte.png")

    return audit


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--v4", default="output/pairwise_v4.csv")
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    p.add_argument("--out-json", default=DEFAULT_OUT_JSON)
    p.add_argument("--fte-cache", default=str(DEFAULT_FTE_CACHE))
    return p.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    args = _parse_args()
    audit = run_audit(
        v4_csv=Path(args.v4),
        out_dir=Path(args.out_dir),
        out_json=Path(args.out_json),
        fte_cache_dir=Path(args.fte_cache),
    )
    print(json.dumps(audit["overall"], indent=2, default=float))
    print(f"weak_spots: {len(audit['weak_spots'])}")
```

**Implementation note:** `build_team_mapping` may have a slightly different signature than shown above; consult `src/ingest/team_mapping.py` and adapt if needed. The principle (one call per external source, fuzzy-match against canonical Kaggle teams, manual overrides via `data/team_name_overrides.csv`) is fixed.

**Implementation note 2:** The `v4_csv` schema (column names) may not be `TeamID_a` / `TeamID_b` / `p_v4`. Read the actual `output/pairwise_v4.csv` column names and adapt the merge in `_build_per_game_audit_df`. The Vegas audit's analogous code is the reference.

- [ ] **Step 3: Run unit tests**

```bash
pytest -v tests/test_audit_v4_gap_fte.py
```

Expected: all tests PASS. If tests fail, fix the implementation.

- [ ] **Step 4: Run the full suite for regressions**

```bash
pytest -v
```

Expected: all green. The audit driver doesn't touch existing modules, so any new failure would indicate an import-time side-effect bug.

- [ ] **Step 5: Run the audit on real data**

```bash
mkdir -p data/raw/fte_forecasts
python src/audit_v4_gap_fte.py \
    --v4 output/pairwise_v4.csv \
    --out-dir output/ \
    --out-json output/v4_gap_audit_fte.json \
    --fte-cache data/raw/fte_forecasts \
    2>&1 | tee output/v4_gap_audit_fte_log.txt
```

Estimated wall time: 20-60 seconds (first run downloads 11 CSVs from GitHub; subsequent runs use cache).

- [ ] **Step 6: Verify anchors**

From `output/v4_gap_audit_fte.json`:

- `overall.n_games` should be approximately 660-693. Lower than 693 means coverage gaps; investigate `join_coverage`.
- `overall.ll_v4` should be in roughly [0.42, 0.45] (close to v4's known 0.4369 on the 22-season set, possibly different on the 11-season subset). If wildly off (>0.5 or <0.4), halt and debug.
- `overall.ll_fte` should be in [0.40, 0.55]. Outside that, suspect BT-norm or join.
- `overall.acc_fte` should be in [0.70, 0.78].
- For at least 5 different R64 matchups, hand-check that `rd1_win[a] + rd1_win[b] is approximately 1` in the cached CSVs. If far off, the snapshot selection is wrong.

If any anchor fails, halt and debug before writing the findings note.

- [ ] **Step 7: Force-add output artifacts**

```bash
git add -f \
    output/v4_gap_audit_fte.json \
    output/v4_gap_calibration_overall_fte.png \
    output/v4_gap_calibration_by_round_fte.png \
    output/v4_gap_per_bucket_ll_delta_fte.png \
    output/v4_gap_audit_fte_log.txt
```

- [ ] **Step 8: Update .gitignore for the cache dir**

Confirm `data/raw/fte_forecasts/` is gitignored (existing rules under `data/raw/*` may already cover it). If not:

```bash
echo "" >> .gitignore
echo "# 538 forecast cache (re-fetchable from GitHub)" >> .gitignore
echo "data/raw/fte_forecasts/" >> .gitignore
git add .gitignore
```

If it's already covered by an existing rule, skip this step.

- [ ] **Step 9: Commit Phase 2 audit driver + outputs**

```bash
git add src/audit_v4_gap_fte.py tests/test_audit_v4_gap_fte.py
git commit -m "feat(v4-gap-audit-fte): driver + outputs

- Mirrors src/audit_v4_gap_vegas.py, swaps Vegas spread-to-prob
  for BT normalization on 538's rd_R_win columns.
- Same buckets (round, chalk-vs-upset, v4-confidence quintile,
  seed-diff magnitude), same weak-spot threshold (n>=50,
  ll_delta>=+0.02) for cross-audit comparability.
- 9 unit tests covering BT-norm math, round indexing, bucketing,
  metrics, calibration, ECE.
- Output: output/v4_gap_audit_fte.json + 3 PNG calibration plots.

Spec:  docs/superpowers/specs/2026-05-04-v4-gap-audit-fte-design.md
Plan:  docs/superpowers/plans/2026-05-04-v4-gap-audit-fte.md"
```

### Task 4: Write findings note + update TODO.md

**Files:**
- Create: `docs/notes/2026-05-04-v4-gap-audit-fte.md`
- Modify: `TODO.md`

- [ ] **Step 1: Inspect findings**

Open `output/v4_gap_audit_fte.json`. Look at:

- `overall` block: headline LL, accuracy, ECE for v4 vs 538.
- `weak_spots` array: top buckets sorted descending by `ll_delta`.
  Each entry has `bucket`, `n_games`, `ll_v4`, `ll_fte`, `ll_delta`,
  `acc_v4`, `acc_fte`.
- `by_round`: per-round tables. F4 (n~22) and Champ (n~11) are
  sample-limited.
- `by_higher_seed_won`: chalk vs upset breakdown.
- `by_v4_confidence_quintile`: where v4 is over- or under-confident.
- `by_seed_diff`: seed-magnitude buckets.

- [ ] **Step 2: Write the findings note**

Create `docs/notes/2026-05-04-v4-gap-audit-fte.md` using the Vegas-audit findings note (`docs/notes/2026-05-04-v4-gap-audit-vegas.md`) as a template:

```markdown
# v4 Gap Audit vs FiveThirtyEight -- Findings

**Date:** 2026-05-04
**Branch:** feat/v4-gap-audit-fte
**Verdict:** [No weak spots vs 538 at the n>=50, ll_delta>=+0.02 threshold |
              N weak-spot signatures identified |
              538 dominates v4 across the board]
**Spec:** `docs/superpowers/specs/2026-05-04-v4-gap-audit-fte-design.md`
**Plan:** `docs/superpowers/plans/2026-05-04-v4-gap-audit-fte.md`

## TL;DR

[1-paragraph summary: where does v4 stand vs 538 across the 11-season
audit set? Headline LL/acc numbers + the verdict's implication for the
queue.]

## Setup recap

- Inputs: `output/pairwise_v4.csv`, 538 forecasts cached at
  `data/raw/fte_forecasts/<year>.csv`, `MNCAATourneyCompactResults`
  for outcomes, `MNCAATourneySeeds` for seeds.
- 538 snapshot: earliest post-play-in per season.
- Per-matchup probability: BT normalization on `rd_R_win[a]` and
  `rd_R_win[b]` where R is the game's round (1-6 = R64..Champ).
- Coverage: NN.N% (NNN of 693 audited 2014-2025-minus-2020 games).
- Wall time: ~NNs.

## Anchors

[Table: anchor / expected band / observed / verdict.]

## Headline numbers

| metric | v4 | 538 | delta (v4 - 538) |
|--------|----|-----|------------------|
| log loss | NN.NN | NN.NN | NN.NN |
| accuracy | NN.N% | NN.N% | +N.N pp |
| ECE | NN.NN | NN.NN | NN.NN |

## Per-bucket results

### By round
[Table from JSON's by_round, including F4/Champ flagged sample-limited.]

### By chalk vs upset
[Table from JSON's by_higher_seed_won.]

### By v4-confidence quintile
[Table from JSON's by_v4_confidence_quintile.]

### By seed-diff magnitude
[Table from JSON's by_seed_diff.]

## Charts

- `output/v4_gap_calibration_overall_fte.png` -- overall calibration.
- `output/v4_gap_calibration_by_round_fte.png` -- 6 panels per round.
- `output/v4_gap_per_bucket_ll_delta_fte.png` -- per-bucket bars,
  sorted ascending; bars to the LEFT of zero = v4 better.

## Caveats

- BT normalization is exact for R64 (rd1_win[a] + rd1_win[b] = 1) but
  approximate for R32+ (averages over 538's expected opponent
  distribution rather than the actual played opponent). Bias is
  small (sub-1% per matchup) but documented.
- 11-season sample (vs Vegas audit's 22). F4 (n=NN) and Champ
  (n=NN) buckets are sample-limited and explicitly flagged.
- Snapshot policy is "earliest post-play-in", representing 538's
  view at the same epistemic state as v4 (zero tournament games
  observed).

## What this implies for the queue

[Concrete: which active-queue item moves up, which experiments are
closed by this finding, what's the next audit/lever to pull.]

## Files of record

```
src/ingest/fte_forecasts.py            -- 538 loader (~150 LOC)
src/audit_v4_gap_fte.py                -- driver (~400 LOC)
tests/test_ingest/test_fte_forecasts.py -- 7 unit tests
tests/test_audit_v4_gap_fte.py          -- 9 unit tests

output/v4_gap_audit_fte.json           -- per-bucket metrics (force-added)
output/v4_gap_calibration_overall_fte.png
output/v4_gap_calibration_by_round_fte.png
output/v4_gap_per_bucket_ll_delta_fte.png
output/v4_gap_audit_fte_log.txt
```

## Compute

[Wall time, network usage, cache hit/miss for the 538 archive.]
```

Fill in all NN/NN.NN placeholders from the actual JSON. Be specific in the verdict and the queue-implication section -- this note is the artifact future-you / a future agent will read to understand what the audit concluded.

- [ ] **Step 3: Update `TODO.md`**

In the "Active queue" section:

- Move the 538 audit from item #1 to "Tried and rejected" (if no weak spots) or to "Done" (if weak spots found and now driving engineering followups). Use the Vegas audit's TODO entry as the structural template -- include the verdict, key numbers, code-retention pointers, and findings-note path.
- Promote item #2 (single-season variance) or item #3 (external data as features) to the new #1 based on what the findings imply.
- Update the re-prioritization preamble to reflect what the 538 audit settled.

- [ ] **Step 4: Final pytest sweep**

```bash
pytest -v
```

Expected: all green. State which subset(s) you ran in the final commit message.

- [ ] **Step 5: Final commit**

```bash
git add docs/notes/2026-05-04-v4-gap-audit-fte.md TODO.md
git commit -m "docs(v4-gap-audit-fte): findings + TODO update

[One-line verdict: no weak spots / N weak-spot signatures / etc.]

Findings: docs/notes/2026-05-04-v4-gap-audit-fte.md"
```

- [ ] **Step 6: Push the branch + open a PR**

```bash
git push -u origin feat/v4-gap-audit-fte
gh pr create --title "feat(v4-gap-audit-fte): [verdict]" --body "$(cat <<'EOF'
## Summary
- Audit v4 stage-1 vs 538 pre-tournament round-survival forecasts.
- 11 seasons (2014-2025 minus 2020), 693 audited games.
- Mirrors Vegas audit (PR 18) bucket framework; per-matchup probability
  via BT normalization on rd_R_win.

## Verdict
[One-line verdict from the findings note.]

## Test plan
- [x] pytest tests/test_ingest/test_fte_forecasts.py
- [x] pytest tests/test_audit_v4_gap_fte.py
- [x] pytest -v (full suite)
- [x] python src/audit_v4_gap_fte.py --v4 output/pairwise_v4.csv

## Files
- src/ingest/fte_forecasts.py
- src/audit_v4_gap_fte.py
- tests/test_ingest/test_fte_forecasts.py
- tests/test_audit_v4_gap_fte.py
- output/v4_gap_audit_fte.json
- output/v4_gap_calibration_*.png
- docs/superpowers/specs/2026-05-04-v4-gap-audit-fte-design.md
- docs/superpowers/plans/2026-05-04-v4-gap-audit-fte.md
- docs/notes/2026-05-04-v4-gap-audit-fte.md
EOF
)"
```

---

## Risks (carried from spec, restated for the executor)

1. **538 schema drift across years.** The Task 1 reconnaissance flushes any column-rename oddities up-front. If a year fails the schema check after Phase 1, populate `_COLUMN_ALIASES` in `src/ingest/fte_forecasts.py` and re-run the loader smoke test (Task 2 Step 4) before continuing.
2. **Team-name resolution gaps.** If `_resolve_fte_team_ids` raises with unresolved 538 names, add override rows to `data/team_name_overrides.csv` (one row per 538-name -> Kaggle TeamID) and re-run.
3. **GitHub raw-URL availability.** Cache aggressively; once cached, the audit doesn't need network. If GitHub is unreachable on the first run, halt and retry later -- there's no fallback source.
4. **Anchor failure on real data.** Most likely cause: snapshot selection picked a pre-play-in snapshot (rd1_win is wrong for play-in winners). Spot-check `rd1_win[a] + rd1_win[b]` for 5 R64 matchups; if much different from 1.0, the snapshot is wrong.
5. **R64 sum-to-1 approximate, not exact.** 538 publishes rounded probabilities; tolerance to 1e-2 is realistic. The driver doesn't enforce strict sum-to-1, just normalizes via BT division.
