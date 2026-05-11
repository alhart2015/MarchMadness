"""Phase 1 data loading and train/test splits."""
from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import pandas as pd

# Phase 1 split cutoffs (DayNum-based; approximate March 1 boundary).
# DayNum 0 is roughly the first Monday of November; March 1 of the following
# year is approximately DayNum 120. Selection Sunday is approximately DayNum 132,
# tournament starts at DayNum 134 (First Four).
TRAIN_CUTOFF_DAYNUM = 120
TEST_END_DAYNUM = 134


def load_rs_games(data_dir: Path, season: int) -> pd.DataFrame:
    """Load regular-season games for one season."""
    path = Path(data_dir) / "MRegularSeasonCompactResults.csv"
    df = pd.read_csv(path)
    return df[df["Season"] == season].reset_index(drop=True)


def split_phase1(
    games: pd.DataFrame,
    train_cutoff_daynum: int = TRAIN_CUTOFF_DAYNUM,
    test_end_daynum: int = TEST_END_DAYNUM,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Partition games into Phase 1 train (early-season) and test (late-season)."""
    train = games[games["DayNum"] < train_cutoff_daynum].reset_index(drop=True)
    test = games[
        (games["DayNum"] >= train_cutoff_daynum) & (games["DayNum"] < test_end_daynum)
    ].reset_index(drop=True)
    return train, test


def build_team_index(games: pd.DataFrame) -> dict[int, int]:
    """Build a contiguous TeamID -> node_idx mapping over all teams in `games`."""
    teams = sorted(set(games["WTeamID"]).union(set(games["LTeamID"])))
    return {team_id: idx for idx, team_id in enumerate(teams)}


def build_global_team_index(
    data_dir: Path, seasons: Iterable[int]
) -> dict[int, int]:
    """Build a contiguous TeamID -> node_idx mapping spanning multiple seasons.

    LOSO training shares one ``nn.Embedding(num_teams, hidden_dim)`` across all
    seasons, so the team index must cover the union of teams that appear in any
    requested season's regular-season games. Returns a deterministic mapping
    sorted ascending by TeamID, with contiguous indices 0..N-1.
    """
    team_ids: set[int] = set()
    for season in seasons:
        games = load_rs_games(data_dir, season)
        team_ids.update(games["WTeamID"].tolist())
        team_ids.update(games["LTeamID"].tolist())
    sorted_ids = sorted(team_ids)
    return {int(team_id): idx for idx, team_id in enumerate(sorted_ids)}
