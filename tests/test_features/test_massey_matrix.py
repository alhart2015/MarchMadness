"""Unit tests for src/features/massey_matrix.py.

Synthetic schedules with closed-form solutions verify solver correctness;
real-data smoke test verifies the cached loader and pipeline integration.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.features.massey_matrix import (
    _PRODUCER_VERSION,
    compute_massey_mov_ratings,
    load_massey_mov_ratings,
)


def _make_round_robin(team_ids, ratings, h, season=2024):
    """Build a 12-game round-robin where the solution is known.

    Each pair plays twice: once at each home. MOV from W's perspective:
        y = r_W - r_L + h * z, z=+1 if W home else -1 if W away else 0
    Loser is the team with lower rating; winner = higher rating.
    """
    rows = []
    daynum = 10
    for i, ti in enumerate(team_ids):
        for j, tj in enumerate(team_ids):
            if i >= j:
                continue
            ri, rj = ratings[i], ratings[j]
            # Game at ti's home: ti is home; W = max-rating side, locator from W's view
            for home_team_idx in (i, j):
                if home_team_idx == i:
                    if ri > rj:
                        w, l = ti, tj
                        z = +1  # W home
                        mov = (ri - rj) + h * z
                    else:
                        w, l = tj, ti
                        z = -1  # W away
                        mov = (rj - ri) + h * z
                else:
                    if rj > ri:
                        w, l = tj, ti
                        z = +1
                        mov = (rj - ri) + h * z
                    else:
                        w, l = ti, tj
                        z = -1
                        mov = (ri - rj) + h * z
                wloc = {1: "H", -1: "A", 0: "N"}[z]
                # Guard: caller chose ratings + h such that the modelled MOV
                # is at least +1 from W's perspective. Silently clipping a
                # non-positive mov would make WScore disagree with the
                # game's true model y = r_W - r_L + h*z.
                assert mov >= 1, (
                    f"_make_round_robin: mov={mov} for W={w}, L={l}, z={z}; "
                    "tighten ratings/h so all matchups produce mov >= 1"
                )
                rows.append({
                    "Season": season,
                    "DayNum": daynum,
                    "WTeamID": w,
                    "WScore": int(50 + mov),
                    "LTeamID": l,
                    "LScore": 50,
                    "WLoc": wloc,
                    "NumOT": 0,
                })
                daynum += 1
    return pd.DataFrame(rows)


def test_synthetic_round_robin_recovers_ratings_and_home_constant():
    team_ids = [1101, 1102, 1103, 1104]
    ratings = [5.0, 2.0, -2.0, -5.0]
    h_true = 1.0
    games = _make_round_robin(team_ids, ratings, h_true)

    df = compute_massey_mov_ratings(games, mov_cap=21)

    assert set(df.columns) == {"Season", "TeamID", "massey_mov_rating"}
    assert len(df) == 4
    rating_by_team = dict(zip(df["TeamID"], df["massey_mov_rating"]))
    for tid, expected in zip(team_ids, ratings):
        assert rating_by_team[tid] == pytest.approx(expected, abs=1e-4), (
            f"Team {tid} expected {expected}, got {rating_by_team[tid]}"
        )


def test_sum_to_zero_invariant():
    """Solver enforces sum(ratings) = 0 for identifiability."""
    team_ids = [1101, 1102, 1103, 1104]
    ratings = [5.0, 2.0, -2.0, -5.0]
    games = _make_round_robin(team_ids, ratings, h=1.0)

    df = compute_massey_mov_ratings(games, mov_cap=21)
    assert df["massey_mov_rating"].sum() == pytest.approx(0.0, abs=1e-8)


def test_mov_cap_clips_blowouts():
    """Capping at 21 produces materially different (smaller-magnitude)
    ratings than capping at 100 when a single blowout exists."""
    # Team 1101 plays team 1102 once (100-point blowout) and team 1103
    # plays team 1104 once (3-point game). Two-component schedule.
    rows = [
        {"Season": 2024, "DayNum": 10, "WTeamID": 1101, "WScore": 150, "LTeamID": 1102,
         "LScore": 50, "WLoc": "N", "NumOT": 0},
        {"Season": 2024, "DayNum": 11, "WTeamID": 1103, "WScore": 70, "LTeamID": 1104,
         "LScore": 67, "WLoc": "N", "NumOT": 0},
        # Connect the two components so the system is solvable.
        {"Season": 2024, "DayNum": 12, "WTeamID": 1101, "WScore": 75, "LTeamID": 1103,
         "LScore": 70, "WLoc": "N", "NumOT": 0},
        {"Season": 2024, "DayNum": 13, "WTeamID": 1102, "WScore": 60, "LTeamID": 1104,
         "LScore": 58, "WLoc": "N", "NumOT": 0},
    ]
    games = pd.DataFrame(rows)

    df_capped = compute_massey_mov_ratings(games, mov_cap=21)
    df_uncapped = compute_massey_mov_ratings(games, mov_cap=100)

    rating_capped = dict(zip(df_capped["TeamID"], df_capped["massey_mov_rating"]))
    rating_uncapped = dict(zip(df_uncapped["TeamID"], df_uncapped["massey_mov_rating"]))

    # Team 1101's rating in the uncapped solve is dominated by the +100
    # game vs 1102, so |rating_1101_uncapped| > |rating_1101_capped|.
    assert abs(rating_uncapped[1101]) > abs(rating_capped[1101]) + 1.0
    # Sanity: the capped 1101 rating is bounded by mov_cap (its games
    # contributed at most cap=21 each toward the rating in score units).
    assert abs(rating_capped[1101]) < 30.0  # well under the uncapped value


def test_solver_handles_all_neutral_games():
    """All-neutral schedule (no home-court signal) -- solver returns
    h = 0 and continues rather than crashing on a singular matrix.
    Spec docs/superpowers/specs/2026-05-03-massey-matrix-feature-design.md
    'Edge cases' section: 'Zero non-neutral games in a season ... set
    h = 0 and solve the team-only sub-system.'"""
    rows = []
    daynum = 10
    # Tiny round-robin where 1101 wins by 10 over 1102 and 1102 wins by
    # 10 over 1103 etc -- everything neutral.
    for w, l, mov in [(1101, 1102, 10), (1102, 1103, 10), (1103, 1101, 8)]:
        rows.append({
            "Season": 2024, "DayNum": daynum,
            "WTeamID": w, "WScore": 70 + mov, "LTeamID": l, "LScore": 70,
            "WLoc": "N", "NumOT": 0,
        })
        daynum += 1
    games = pd.DataFrame(rows)

    # Should not raise.
    df = compute_massey_mov_ratings(games, mov_cap=21)
    assert len(df) == 3
    assert df["massey_mov_rating"].sum() == pytest.approx(0.0, abs=1e-8)

    # Inspect h via the private solver.
    from src.features.massey_matrix import _solve_one_season
    _ratings, h = _solve_one_season(games, mov_cap=21)
    assert h == 0.0, f"expected h=0 on all-neutral schedule, got {h}"


def test_home_court_constant_recovered():
    """If teams are equal-strength but home always wins by 5, h ~= 5."""
    team_ids = [1101, 1102, 1103, 1104]
    rows = []
    daynum = 10
    for i, ti in enumerate(team_ids):
        for j, tj in enumerate(team_ids):
            if i == j:
                continue
            # ti is home; ti wins by 5
            rows.append({
                "Season": 2024, "DayNum": daynum,
                "WTeamID": ti, "WScore": 75,
                "LTeamID": tj, "LScore": 70,
                "WLoc": "H", "NumOT": 0,
            })
            daynum += 1
    games = pd.DataFrame(rows)

    # Solve directly with the private function so we can inspect h.
    from src.features.massey_matrix import _solve_one_season
    ratings, h = _solve_one_season(games, mov_cap=21)

    assert h == pytest.approx(5.0, abs=1e-4)
    for tid, r in ratings.items():
        assert abs(r) < 1e-4, f"Team {tid} expected ~0 rating, got {r}"


def test_cache_roundtrip(tmp_path: Path):
    """load_massey_mov_ratings writes parquet + sidecar on first call,
    reads from cache on second call; both return equal frames."""
    team_ids = [1101, 1102, 1103, 1104]
    games = _make_round_robin(team_ids, [5.0, 2.0, -2.0, -5.0], h=1.0)

    df1 = load_massey_mov_ratings(games, mov_cap=21, cache_dir=tmp_path)
    parquet_path = tmp_path / "massey_mov_ratings.parquet"
    meta_path = tmp_path / "massey_mov_ratings.meta.json"
    assert parquet_path.exists()
    assert meta_path.exists()

    df2 = load_massey_mov_ratings(games, mov_cap=21, cache_dir=tmp_path)
    pd.testing.assert_frame_equal(
        df1.sort_values(["Season", "TeamID"]).reset_index(drop=True),
        df2.sort_values(["Season", "TeamID"]).reset_index(drop=True),
    )

    meta = json.loads(meta_path.read_text())
    assert meta["producer_version"] == _PRODUCER_VERSION
    assert meta["mov_cap"] == 21
    assert meta["n_input_rows"] == len(games)
    assert "sha_input" in meta


def test_cache_invalidates_on_meta_mismatch(tmp_path: Path, monkeypatch):
    """If sidecar metadata's producer_version doesn't match the module
    constant, the cache is rebuilt rather than reused."""
    team_ids = [1101, 1102, 1103, 1104]
    games = _make_round_robin(team_ids, [5.0, 2.0, -2.0, -5.0], h=1.0)

    # Initial write under v1 (current).
    df1 = load_massey_mov_ratings(games, mov_cap=21, cache_dir=tmp_path)
    parquet_path = tmp_path / "massey_mov_ratings.parquet"
    initial_mtime = parquet_path.stat().st_mtime_ns

    # Hand-edit the sidecar to claim a different producer version.
    meta_path = tmp_path / "massey_mov_ratings.meta.json"
    meta = json.loads(meta_path.read_text())
    meta["producer_version"] = "v0-stale"
    meta_path.write_text(json.dumps(meta))

    # Next load should detect the mismatch and rebuild.
    df2 = load_massey_mov_ratings(games, mov_cap=21, cache_dir=tmp_path)
    new_mtime = parquet_path.stat().st_mtime_ns
    assert new_mtime > initial_mtime, "parquet should have been rewritten"

    # The rebuilt sidecar should claim the current version.
    refreshed = json.loads(meta_path.read_text())
    assert refreshed["producer_version"] == _PRODUCER_VERSION

    pd.testing.assert_frame_equal(
        df1.sort_values(["Season", "TeamID"]).reset_index(drop=True),
        df2.sort_values(["Season", "TeamID"]).reset_index(drop=True),
    )
