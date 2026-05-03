"""Unit tests for src/train_peer_stage1.py.

These tests exercise the public CLI of the per-peer stage-1 trainer
on synthetic-shape inputs (feature_matrix, tourney results) -- the full
LOSO loop on real data is exercised in Task 7's real-data run.
"""
import numpy as np
import pandas as pd
import pytest


def _make_synthetic_feature_matrix(n_teams: int = 8, season: int = 2024):
    """Build a feature_matrix with both PEER_A and PEER_B columns."""
    from src.feature_views import PEER_A_FEATURES, PEER_B_FEATURES

    rng = np.random.default_rng(0)
    team_ids = list(range(1, n_teams + 1))
    rows = []
    for tid in team_ids:
        row = {"TeamID": tid, "Season": season, "seed": (tid % 16) + 1}
        for c in list(PEER_A_FEATURES) + list(PEER_B_FEATURES):
            row[c] = float(rng.standard_normal())
        rows.append(row)
    return pd.DataFrame(rows)


def test_select_peer_features_returns_only_peer_a():
    from src.feature_views import PEER_A_FEATURES, PEER_B_FEATURES
    from src.train_peer_stage1 import select_peer_features

    fm = _make_synthetic_feature_matrix()
    all_cols = [c for c in fm.columns if c not in {"TeamID", "Season", "seed"}]
    selected = select_peer_features(all_cols, peer="a")
    assert set(selected) == set(PEER_A_FEATURES)
    assert set(selected).isdisjoint(set(PEER_B_FEATURES))


def test_select_peer_features_returns_only_peer_b():
    from src.feature_views import PEER_A_FEATURES, PEER_B_FEATURES
    from src.train_peer_stage1 import select_peer_features

    fm = _make_synthetic_feature_matrix()
    all_cols = [c for c in fm.columns if c not in {"TeamID", "Season", "seed"}]
    selected = select_peer_features(all_cols, peer="b")
    assert set(selected) == set(PEER_B_FEATURES)
    assert set(selected).isdisjoint(set(PEER_A_FEATURES))


def test_select_peer_features_unknown_peer_raises():
    from src.train_peer_stage1 import select_peer_features

    with pytest.raises(ValueError, match="peer"):
        select_peer_features(["adj_oe"], peer="c")


def test_dump_pairwise_for_season_writes_documented_schema(tmp_path):
    """The OOF pairwise CSV must match v4's schema:
    columns = (season, team_a, team_b, p_a_wins) with team_a < team_b.
    """
    from src.train_peer_stage1 import dump_pairwise_for_season

    rng = np.random.default_rng(0)
    team_ids = [10, 20, 30]
    feature_lookup = {tid: rng.standard_normal(5) for tid in team_ids}

    class _StubModel:
        def predict_proba(self, X):
            n = len(X)
            return np.column_stack([np.full(n, 0.4), np.full(n, 0.6)])

    out = tmp_path / "pairwise_peer_test.csv"
    n = dump_pairwise_for_season(
        season=2024,
        field_team_ids=team_ids,
        feature_lookup=feature_lookup,
        model=_StubModel(),
        out_csv=str(out),
    )
    assert n == 3  # C(3, 2) = 3 pairs
    df = pd.read_csv(out)
    assert list(df.columns) == ["season", "team_a", "team_b", "p_a_wins"]
    assert (df["team_a"] < df["team_b"]).all()
