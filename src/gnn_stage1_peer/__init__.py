"""Graph Neural Network stage-1 peer model.

Phase 1 sanity check vs scalar Massey on regular-season game prediction.
See docs/superpowers/specs/2026-05-09-non-tabular-model-class-scoping-design.md
and docs/superpowers/plans/2026-05-09-non-tabular-model-class-scoping-phase1.md.
"""

from .training import run_phase1_one_season

__all__ = ["run_phase1_one_season"]
