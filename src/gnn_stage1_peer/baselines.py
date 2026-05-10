"""Scalar Massey-composite baseline for Phase 1 gate."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

# Default Massey systems used for the composite. Subset of v4's massey_systems.
# Coverage in MMasseyOrdinals.csv: POM, MOR, DOL span 2003-2026 (full); MAS
# covers 20/24 seasons (gaps); SAG ends in 2023 (absent for 2024-2026). The
# composite degrades gracefully -- pandas groupby+mean averages over whichever
# systems are present per season, so 2024+ is effectively a 3-system mean.
DEFAULT_SYSTEMS = ("POM", "MAS", "SAG", "MOR", "DOL")


def load_massey_composite(
    data_dir: Path,
    season: int,
    ranking_day: int = 133,
    systems: tuple[str, ...] = DEFAULT_SYSTEMS,
) -> dict[int, float]:
    """Load Massey composite rank as `{team_id: mean_rank}` for one season.

    Composite = mean of OrdinalRank across `systems`, evaluated at the latest
    RankingDayNum <= `ranking_day` per (season, system, team).
    """
    path = Path(data_dir) / "MMasseyOrdinals.csv"
    df = pd.read_csv(path)
    df = df[(df["Season"] == season) & (df["RankingDayNum"] <= ranking_day)]
    df = df[df["SystemName"].isin(systems)]
    # Take latest day per (system, team)
    df = df.sort_values("RankingDayNum").groupby(["SystemName", "TeamID"]).tail(1)
    composite = df.groupby("TeamID")["OrdinalRank"].mean().astype(float)
    return composite.to_dict()


def predict_massey_logit(
    team_a: int, team_b: int, massey_ranks: dict[int, float], scale: float = 0.05
) -> float:
    """Predict logit p(A wins) given Massey composite ranks. Lower rank = better."""
    if team_a not in massey_ranks or team_b not in massey_ranks:
        return 0.0
    return -scale * (massey_ranks[team_a] - massey_ranks[team_b])


def evaluate_massey_baseline(
    test_games: pd.DataFrame,
    season: int,
    data_dir: Path,
    scale: float = 0.05,
    ranking_day: int = 133,
) -> dict:
    """Evaluate Massey baseline on a test split. Symmetric over orientations."""
    ranks = load_massey_composite(data_dir, season, ranking_day)
    nll_sum = 0.0
    correct = 0
    n = 0
    for _, g in test_games.iterrows():
        for (a, b, label) in (
            (int(g["WTeamID"]), int(g["LTeamID"]), 1.0),
            (int(g["LTeamID"]), int(g["WTeamID"]), 0.0),
        ):
            logit = predict_massey_logit(a, b, ranks, scale)
            p = 1.0 / (1.0 + math.exp(-logit))
            # BCE per sample: -[y log p + (1-y) log(1-p)]
            eps = 1e-12
            nll_sum += -(label * math.log(max(p, eps)) + (1.0 - label) * math.log(max(1.0 - p, eps)))
            correct += int((p >= 0.5) == (label >= 0.5))
            n += 1
    return {"ll": nll_sum / max(n, 1), "accuracy": correct / max(n, 1), "n": n}
