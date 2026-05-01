"""15-cell W_UPSET / W_MISS tuning sweep over the v9-A trainer.

Grid: W_UPSET in {1.0, 1.25, 1.5, 1.75, 2.0} x W_MISS in {0.0, 0.5, 1.0}.

For each cell, run double-LOSO across 22 seasons (2003..2025), build
v9-adjusted pairwise probabilities, score with score_pairwise_path
against MNCAATourneyCompactResults.csv, and write one row to
output/v9_sweep_results.csv.

Anchor cell (1.0, 0.0) must be present in the grid -- it is the v8
reproduction sanity check.

Spec:  docs/superpowers/specs/2026-05-01-v9-weight-sweep.md
"""
import sys
from pathlib import Path
from typing import Iterable, List, Tuple

# Path setup: allow `python src/sweep_v9_weights.py` invocation.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

W_UPSET_VALUES = [1.0, 1.25, 1.5, 1.75, 2.0]
W_MISS_VALUES = [0.0, 0.5, 1.0]
GRID: List[Tuple[float, float]] = [
    (wu, wm) for wu in W_UPSET_VALUES for wm in W_MISS_VALUES
]
ANCHOR_CELL: Tuple[float, float] = (1.0, 0.0)


def validate_grid(grid: Iterable[Tuple[float, float]]) -> None:
    """Raise ValueError if the anchor cell (1.0, 0.0) is missing.

    The anchor is the v8 reproduction sanity check: at uniform weights
    the v9-A trainer should reproduce v8 within 1 bracket point. Without
    the anchor, the sweep cannot be sanity-checked.
    """
    cells = set((float(wu), float(wm)) for wu, wm in grid)
    if ANCHOR_CELL not in cells:
        raise ValueError(
            f"anchor cell {ANCHOR_CELL} missing from grid; sweep is invalid "
            "(no v8 reproduction sanity check possible)"
        )
