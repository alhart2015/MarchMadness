"""Tests for src/sweep_v9_weights.py (15-cell W_UPSET / W_MISS sweep)."""

import pytest

from src.sweep_v9_weights import GRID, validate_grid


def test_grid_contains_anchor_cell():
    """The anchor cell (W_UPSET=1.0, W_MISS=0.0) MUST be in the grid -- it
    is the v8 reproduction sanity check.
    """
    assert (1.0, 0.0) in GRID


def test_grid_has_15_unique_cells():
    """Spec calls for 5 * 3 = 15 cells."""
    assert len(GRID) == 15
    assert len(set(GRID)) == 15


def test_validate_grid_passes_with_anchor_cell():
    """validate_grid raises iff anchor is missing."""
    validate_grid([(1.0, 0.0), (1.5, 1.0)])  # contains anchor; should not raise


def test_validate_grid_raises_without_anchor_cell():
    with pytest.raises(ValueError, match="anchor cell"):
        validate_grid([(1.5, 1.0), (2.0, 0.0)])
