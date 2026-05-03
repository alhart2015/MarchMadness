"""Unit tests for src/feature_views.py.

The module defines the disjoint feature partition used by the
feature-view diversity ensemble (PEER_A: team strength;
PEER_B: form + market + meta). Validates partition disjointness and
exhaustiveness against any caller-supplied feature list.
"""
import pytest


def test_partition_disjoint():
    from src.feature_views import PEER_A_FEATURES, PEER_B_FEATURES

    a = set(PEER_A_FEATURES)
    b = set(PEER_B_FEATURES)
    assert a & b == set(), f"PEER_A and PEER_B overlap: {sorted(a & b)}"


def test_partition_lists_are_immutable_tuples():
    """Lists are exposed as tuples so downstream code can't mutate them."""
    from src.feature_views import PEER_A_FEATURES, PEER_B_FEATURES

    assert isinstance(PEER_A_FEATURES, tuple)
    assert isinstance(PEER_B_FEATURES, tuple)


def test_validate_partition_passes_when_complete_and_disjoint():
    from src.feature_views import (
        PEER_A_FEATURES, PEER_B_FEATURES, validate_partition,
    )

    all_cols = list(PEER_A_FEATURES) + list(PEER_B_FEATURES)
    validate_partition(all_cols)  # must not raise


def test_validate_partition_raises_on_missing_feature():
    """A column in all_cols that's in neither PEER_A nor PEER_B is
    a partition gap and must raise ValueError naming the column.
    """
    from src.feature_views import (
        PEER_A_FEATURES, PEER_B_FEATURES, validate_partition,
    )

    all_cols = list(PEER_A_FEATURES) + list(PEER_B_FEATURES) + ["new_feature_xyz"]
    with pytest.raises(ValueError, match="new_feature_xyz"):
        validate_partition(all_cols)


def test_validate_partition_raises_on_extra_peer_feature():
    """A column in PEER_A but not in all_cols means PEER_A drifted past
    v4's actual columns. Must raise.
    """
    from src.feature_views import (
        PEER_A_FEATURES, PEER_B_FEATURES, validate_partition,
    )

    # Drop one PEER_A feature from all_cols.
    all_cols = list(PEER_A_FEATURES[1:]) + list(PEER_B_FEATURES)
    missing = PEER_A_FEATURES[0]
    with pytest.raises(ValueError, match=missing):
        validate_partition(all_cols)
