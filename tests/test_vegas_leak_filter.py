"""Unit + smoke tests for filter_vegas_to_pre_tournament.

Spec: docs/superpowers/specs/2026-05-04-v4-vegas-leak-fix-design.md
"""
from pathlib import Path

import pandas as pd
import pytest

from src.enhanced_model_v3 import filter_vegas_to_pre_tournament

_MANIA = Path("data/raw/march-machine-learning-2026")
_VEGAS = Path("data/raw/vegas_lines")
_HAVE_REAL_DATA = (_MANIA / "MTeams.csv").exists() and _VEGAS.exists()


def _make_seasons_csv(tmp_path: Path) -> Path:
    """Minimal MSeasons.csv with DayZero=11/01/2024 for season 2025."""
    p = tmp_path / "MSeasons.csv"
    pd.DataFrame({
        "Season": [2024, 2025],
        "DayZero": ["10/30/2023", "11/01/2024"],
        "RegionW": ["W", "W"], "RegionX": ["X", "X"],
        "RegionY": ["Y", "Y"], "RegionZ": ["Z", "Z"],
    }).to_csv(p, index=False)
    return p


def test_drops_tournament_rows(tmp_path):
    """Rows with daynum >= 134 (NCAA tournament First Four onward) are dropped."""
    seasons_csv = _make_seasons_csv(tmp_path)
    # DayZero for season 2025 is 11/01/2024.
    # 2024-11-15 = day 14, 2025-03-05 = day 124, 2025-03-14 = day 133, 2025-04-07 = day 157.
    df = pd.DataFrame({
        "season": [2025, 2025, 2025, 2025],
        "date": ["11/15/2024", "03/05/2025", "03/14/2025", "04/07/2025"],
        "home": ["A", "B", "C", "D"],
        "road": ["E", "F", "G", "H"],
        "line": [3.0, -2.0, 5.0, 1.0],
        "hscore": [70, 80, 65, 75],
        "rscore": [60, 70, 75, 70],
        "neutral": [0, 0, 1, 1],
    })
    out = filter_vegas_to_pre_tournament(df, seasons_csv_path=seasons_csv)
    # daynums: 14, 124, 133, 157. Drop the one with daynum >= 134 (157 -> April 7).
    assert len(out) == 3
    assert "04/07/2025" not in out["date"].tolist()
    # Schema preserved.
    assert list(out.columns) == list(df.columns)


def test_drops_first_four_day_134(tmp_path):
    """Boundary: daynum == 134 is the First Four day; must be dropped."""
    seasons_csv = _make_seasons_csv(tmp_path)
    # Season 2025 DayZero=11/01/2024. day 133 = 03/14/2025, day 134 = 03/15/2025.
    df = pd.DataFrame({
        "season": [2025, 2025],
        "date": ["03/14/2025", "03/15/2025"],
        "home": ["A", "B"], "road": ["C", "D"],
        "line": [1.0, 2.0],
        "hscore": [70, 70], "rscore": [60, 60], "neutral": [1, 1],
    })
    out = filter_vegas_to_pre_tournament(df, seasons_csv_path=seasons_csv)
    assert len(out) == 1
    assert out["date"].iloc[0] == "03/14/2025"


def test_empty_input(tmp_path):
    """Empty input -> empty output, schema preserved."""
    seasons_csv = _make_seasons_csv(tmp_path)
    df = pd.DataFrame(columns=["season", "date", "home", "road", "line",
                                 "hscore", "rscore", "neutral"])
    out = filter_vegas_to_pre_tournament(df, seasons_csv_path=seasons_csv)
    assert len(out) == 0
    assert list(out.columns) == list(df.columns)


def test_unknown_season_kept_with_warning(tmp_path, capsys):
    """A row whose season has no DayZero entry is kept with a warning."""
    seasons_csv = _make_seasons_csv(tmp_path)
    df = pd.DataFrame({
        "season": [9999],
        "date": ["03/15/2025"],
        "home": ["A"], "road": ["B"], "line": [1.0],
        "hscore": [70], "rscore": [60], "neutral": [1],
    })
    out = filter_vegas_to_pre_tournament(df, seasons_csv_path=seasons_csv)
    assert len(out) == 1
    captured = capsys.readouterr()
    assert "9999" in captured.out or "9999" in captured.err


def test_unparseable_date_kept_with_warning(tmp_path, capsys):
    """A row with an unparseable date is kept (defensive, do not silently drop)."""
    seasons_csv = _make_seasons_csv(tmp_path)
    df = pd.DataFrame({
        "season": [2025],
        "date": ["not-a-date"],
        "home": ["A"], "road": ["B"], "line": [1.0],
        "hscore": [70], "rscore": [60], "neutral": [0],
    })
    out = filter_vegas_to_pre_tournament(df, seasons_csv_path=seasons_csv)
    assert len(out) == 1
    captured = capsys.readouterr()
    assert "unparseable" in captured.out.lower() or "unparseable" in captured.err.lower()


@pytest.mark.skipif(not _HAVE_REAL_DATA, reason="raw Mania/Vegas data not present in this checkout")
def test_smoke_real_data_2024_uconn():
    """Integration: with real Vegas data, 2024 UConn (TeamID 1163) has
    vegas_avg_margin near +16.16 (regular-season-only) after filter,
    not +18.13 (full season)."""
    from src.enhanced_model_v3 import load_vegas_lines, compute_vegas_features
    DATA = Path("data/raw/march-machine-learning-2026")
    teams = pd.read_csv(DATA / "MTeams.csv")
    spellings = pd.read_csv(DATA / "MTeamSpellings.csv", encoding="latin-1")
    vegas_df = load_vegas_lines()
    vegas_df_filtered = filter_vegas_to_pre_tournament(vegas_df)
    feats, _ = compute_vegas_features(vegas_df_filtered, teams, spellings)
    row = feats[(feats["TeamID"] == 1163) & (feats["Season"] == 2024)]
    assert len(row) == 1, "expected exactly one (UConn 2024) row"
    margin = float(row["vegas_avg_margin"].iloc[0])
    # Regular-season-only is ~+16.16. Filtered should be within 0.10 of that.
    assert 16.0 < margin < 16.30, f"expected ~16.16 reg-only, got {margin:.2f}"
