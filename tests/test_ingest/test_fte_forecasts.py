"""Unit tests for src/ingest/fte_forecasts.py."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.ingest.fte_forecasts import (
    _AUDITED_YEARS,
    _FTE_URL_BY_YEAR,
    _REQUIRED_COLS,
    _fte_url_for_year,
    _normalize_schema,
    _select_post_playin_snapshot,
    load_fte_forecasts,
)


def _synthetic_forecasts_df(season: int = 2024,
                             include_post: bool = True,
                             include_womens: bool = False) -> pd.DataFrame:
    """Two snapshots reflecting real 538 schema:
      - 2024-03-19 (pre-play-in): 68 teams, all with rd1_win > 0
      - 2024-03-21 (post-play-in): 68 teams, 4 play-in losers have
        rd1_win = 0 (and zero in higher rounds), 64 with rd1_win > 0
    8 play-in teams are flagged via playin_flag=1 (statically).
    Tids 65-68 are the play-in losers in the post snapshot.
    """
    rows = []

    def _row(date: str, tid: int, lost: bool):
        return {
            "gender": "mens",
            "forecast_date": date,
            "playin_flag": 1 if tid > 60 else 0,
            "team_id": tid,
            "team_name": f"Team {tid}",
            "team_seed": "01",
            "rd1_win": 0.0 if lost else 0.5,
            "rd2_win": 0.0 if lost else 0.25,
            "rd3_win": 0.0 if lost else 0.125,
            "rd4_win": 0.0 if lost else 0.0625,
            "rd5_win": 0.0 if lost else 0.03125,
            "rd6_win": 0.0 if lost else 0.015625,
            "rd7_win": 0.0 if lost else 0.0078125,
        }

    # Pre-play-in: all 68 alive
    for tid in range(1, 69):
        rows.append(_row("2024-03-19", tid, lost=False))
    # Post-play-in: 4 losers (tid 65-68) get rd*_win=0
    if include_post:
        losers = {65, 66, 67, 68}
        for tid in range(1, 69):
            rows.append(_row("2024-03-21", tid, lost=tid in losers))
    if include_womens:
        for tid in range(1, 69):
            r = _row("2024-03-19", tid, lost=False)
            r["gender"] = "womens"
            rows.append(r)
    df = pd.DataFrame(rows)
    return df


def test_url_for_each_year_resolves():
    for y in [2016, 2019, 2021, 2023]:
        url = _fte_url_for_year(y)
        assert url.startswith("https://web.archive.org/web/"), url
        assert str(y) in url


def test_url_year_unrecoverable_raises():
    """Years outside the audited set raise. Covers pre-API (2014, 2015),
    no-tournament (2020), and post-shutdown (2024, 2025)."""
    for y in [2014, 2015, 2020, 2024, 2025]:
        with pytest.raises((KeyError, ValueError)):
            _fte_url_for_year(y)


def test_audited_years_matches_url_dict():
    """Defensive: the public _AUDITED_YEARS tuple must agree with the
    keys of _FTE_URL_BY_YEAR. A drift would silently break the loader."""
    assert tuple(sorted(_FTE_URL_BY_YEAR.keys())) == tuple(sorted(_AUDITED_YEARS))


def test_normalize_schema_required_columns_present():
    df = _synthetic_forecasts_df()
    out = _normalize_schema(df, year=2024)
    for col in _REQUIRED_COLS:
        assert col in out.columns, f"missing canonical column: {col}"


def test_normalize_schema_missing_required_col_raises():
    df = _synthetic_forecasts_df().drop(columns=["rd5_win"])
    with pytest.raises(ValueError, match="rd5_win"):
        _normalize_schema(df, year=2024)


def test_select_post_playin_snapshot_picks_post_playin():
    """Earliest snapshot with exactly 64 teams having rd1_win > 0 wins;
    the 4 play-in losers are dropped from the returned dataframe."""
    df = _synthetic_forecasts_df()
    df = df.assign(Season=2024)
    out = _select_post_playin_snapshot(df)
    assert len(out) == 64
    assert (out["rd1_win"] > 0).all()
    assert out["forecast_date"].unique().tolist() == ["2024-03-21"]
    # Play-in losers (tids 65-68) are dropped
    assert set(out["team_id"]) == set(range(1, 65))


def test_select_post_playin_snapshot_no_resolution_raises():
    """If no snapshot has exactly 64 teams alive, raise rather than
    silently returning a wrong-epoch snapshot."""
    df = _synthetic_forecasts_df(include_post=False)  # only pre-play-in (68 alive)
    df = df.assign(Season=2024)
    with pytest.raises(ValueError, match="post-play-in"):
        _select_post_playin_snapshot(df)


def test_load_fte_forecasts_uses_cache(tmp_path):
    """If cache file exists, loader reads from disk and does not download."""
    cache = tmp_path / "fte_forecasts"
    cache.mkdir()
    df = _synthetic_forecasts_df()
    df.to_csv(cache / "2024.csv", index=False)
    # 2024 isn't in _AUDITED_YEARS, so we patch a recoverable year instead.
    # Use 2023, write the synthetic frame under 2023.csv, and load that.
    (cache / "2023.csv").write_text((cache / "2024.csv").read_text())
    out = load_fte_forecasts(years=[2023], cache_dir=cache,
                             allow_download=False)
    assert len(out) == 64
    assert "Season" in out.columns
    assert (out["Season"] == 2023).all()


def test_load_fte_forecasts_filters_to_mens(tmp_path):
    """Womens rows are dropped before snapshot selection."""
    cache = tmp_path / "fte_forecasts"
    cache.mkdir()
    df = _synthetic_forecasts_df(include_womens=True)
    df.to_csv(cache / "2023.csv", index=False)
    out = load_fte_forecasts(years=[2023], cache_dir=cache,
                             allow_download=False)
    assert (out["gender"] == "mens").all()
    assert len(out) == 64


def test_load_fte_forecasts_missing_cache_no_download_raises(tmp_path):
    """allow_download=False and no cache present should raise FileNotFoundError."""
    cache = tmp_path / "fte_forecasts"
    cache.mkdir()
    with pytest.raises(FileNotFoundError):
        load_fte_forecasts(years=[2023], cache_dir=cache,
                           allow_download=False)
