"""Load FiveThirtyEight pre-tournament round-survival forecasts.

538 published per-team round-survival CSVs at
projects.fivethirtyeight.com/march-madness-api/<year>/fivethirtyeight_ncaa_forecasts.csv
from 2016 through 2023. The endpoint went dark with 538's March 2025
shutdown (every URL now 302-redirects to abcnews.go.com/politics).
2014/2015 forecasts predate the API endpoint; 2024/2025 forecasts
were not published at this URL pattern after the Disney/ABC News
ownership transition. The audit is therefore restricted to 7 seasons
(2016-2019, 2021-2023).

This loader fetches archived copies via the Internet Archive's Wayback
Machine; per-year snapshot timestamps are pinned in _FTE_URL_BY_YEAR
(captured 2025-03-06, the last day before the shutdown). Once cached
locally under data/raw/fte_forecasts/<year>.csv the loader never
re-fetches.

Snapshot semantics: 538 published one CSV per tournament year that was
updated daily over the course of the tournament. Each CSV contains
multiple forecast snapshots (forecast_date column). The schema carries
all 68 teams in every snapshot; play-in losers persist with rd_R_win = 0
after their elimination. The "earliest post-play-in snapshot" is
detected as the earliest forecast_date with exactly 64 teams having
rd1_win > 0 (i.e., the 4 play-in losers eliminated, 64 R64 entrants
alive). The loader drops the 4 play-in losers from the returned frame.

Column semantics (verified against 2016 data, sums per post-play-in
snapshot): rdR_win is P(reach round R), NOT P(win round R's game).

    column   meaning                       sum across 64 R64 entrants
    rd1_win  P(reach R64) = 1 post-playin  64.0  (all alive)
    rd2_win  P(reach R32) = P(win R64)     32.0  (32 R64 matchups)
    rd3_win  P(reach S16) = P(win R32)     16.0
    rd4_win  P(reach E8)                    8.0
    rd5_win  P(reach F4)                    4.0
    rd6_win  P(reach Champ game)            2.0
    rd7_win  P(win Champ)                   1.0

The audit driver therefore reads `rd{round_index + 1}_win` for an
actual played game, where round_index in {1..6} = {R64..Champ}.

Public API:
    load_fte_forecasts(years, cache_dir, allow_download=True) -> pd.DataFrame
        Returns one row per (Season, team_id) for the post-play-in
        snapshot of each requested season, gender='mens' only.

    download_fte_forecasts(year, cache_dir) -> Path
        HTTP GET via the pinned Wayback URL for `year`, write to
        cache_dir / f'{year}.csv', return the path.

Spec: docs/superpowers/specs/2026-05-04-v4-gap-audit-fte-design.md
"""
from __future__ import annotations

import logging
import urllib.error
import urllib.request
from pathlib import Path
from typing import Iterable

import pandas as pd

logger = logging.getLogger(__name__)

_AUDITED_YEARS = (2016, 2017, 2018, 2019, 2021, 2022, 2023)

# Per-year Wayback Machine replay URLs. The 'id_' flag returns raw
# archived bytes (no Wayback HTML wrapper). Snapshot timestamps captured
# via CDX recon on 2026-05-07; all are 200-status text/csv, taken on
# 2025-03-06 (one day before 538's shutdown).
_FTE_URL_BY_YEAR: dict[int, str] = {
    2016: "https://web.archive.org/web/20250306225623id_/https://projects.fivethirtyeight.com/march-madness-api/2016/fivethirtyeight_ncaa_forecasts.csv",
    2017: "https://web.archive.org/web/20250306234318id_/https://projects.fivethirtyeight.com/march-madness-api/2017/fivethirtyeight_ncaa_forecasts.csv",
    2018: "https://web.archive.org/web/20250306225144id_/https://projects.fivethirtyeight.com/march-madness-api/2018/fivethirtyeight_ncaa_forecasts.csv",
    2019: "https://web.archive.org/web/20250306183225id_/https://projects.fivethirtyeight.com/march-madness-api/2019/fivethirtyeight_ncaa_forecasts.csv",
    2021: "https://web.archive.org/web/20250306225145id_/https://projects.fivethirtyeight.com/march-madness-api/2021/fivethirtyeight_ncaa_forecasts.csv",
    2022: "https://web.archive.org/web/20250306225144id_/https://projects.fivethirtyeight.com/march-madness-api/2022/fivethirtyeight_ncaa_forecasts.csv",
    2023: "https://web.archive.org/web/20250306221915id_/https://projects.fivethirtyeight.com/march-madness-api/2023/fivethirtyeight_ncaa_forecasts.csv",
}

_REQUIRED_COLS = (
    "gender", "forecast_date",
    "rd1_win", "rd2_win", "rd3_win", "rd4_win",
    "rd5_win", "rd6_win", "rd7_win",
    "team_id", "team_name", "team_seed", "playin_flag",
)


def _fte_url_for_year(year: int) -> str:
    if year not in _FTE_URL_BY_YEAR:
        raise ValueError(
            f"Year {year} not in audited set {_AUDITED_YEARS}. "
            f"(2020 had no tournament; 2014/2015 predate 538's API "
            f"endpoint; 2024/2025 not recoverable from Wayback.)"
        )
    return _FTE_URL_BY_YEAR[year]


def download_fte_forecasts(year: int, cache_dir: Path) -> Path:
    """Download 538 forecasts CSV for `year` via Wayback to cache_dir/{year}.csv."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_path = cache_dir / f"{year}.csv"
    url = _fte_url_for_year(year)
    logger.info("downloading 538 forecasts (wayback): %s -> %s", url, out_path)
    try:
        with urllib.request.urlopen(url, timeout=120) as resp:
            body = resp.read()
    except urllib.error.HTTPError as e:
        raise RuntimeError(
            f"HTTP {e.code} fetching {url}: {e.reason}"
        ) from e
    out_path.write_bytes(body)
    return out_path


def _normalize_schema(df: pd.DataFrame, year: int) -> pd.DataFrame:
    """Verify all required canonical columns are present.

    No renames needed: every recoverable year (2016-2023) uses the same
    16-column schema. Kept as a stub for future schema-drift mitigation.
    """
    missing = [c for c in _REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"538 forecasts CSV for year {year} missing required columns: "
            f"{missing}. Columns present: {sorted(df.columns)}"
        )
    return df


def _select_post_playin_snapshot(df: pd.DataFrame) -> pd.DataFrame:
    """Per season: pick earliest forecast_date snapshot with exactly 64
    teams having rd1_win > 0 (i.e., the 4 play-in losers eliminated and
    the R64 field set), then drop the 4 play-in losers (rd1_win == 0)
    from the returned frame.

    Raises ValueError if no snapshot reaches the 64-alive state.
    """
    out_rows = []
    for season, season_df in df.groupby("Season"):
        snap_dates = sorted(season_df["forecast_date"].unique())
        chosen = None
        for d in snap_dates:
            snap = season_df[season_df["forecast_date"] == d]
            n_alive = int((snap["rd1_win"] > 0).sum())
            if n_alive == 64:
                chosen = snap[snap["rd1_win"] > 0].copy()
                break
        if chosen is None:
            raise ValueError(
                f"season={season}: no post-play-in snapshot found "
                f"(checked {len(snap_dates)} forecast_dates; need exactly "
                f"64 teams with rd1_win > 0)"
            )
        out_rows.append(chosen)
    return pd.concat(out_rows, ignore_index=True)


def load_fte_forecasts(
    years: Iterable[int],
    cache_dir: str | Path,
    allow_download: bool = True,
) -> pd.DataFrame:
    """Load 538 pre-tournament forecasts for the given years.

    Returns one DataFrame; columns include Season, forecast_date, gender,
    playin_flag, team_id, team_name, team_seed, rd1_win..rd7_win, plus
    any other columns 538 published. Filtered to gender='mens' and the
    earliest post-play-in snapshot per season.
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
        return pd.DataFrame(columns=["Season", *_REQUIRED_COLS])
    df = pd.concat(frames, ignore_index=True)
    df = _select_post_playin_snapshot(df)
    return df.reset_index(drop=True)
