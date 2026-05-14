"""Unit tests for src/eval_r64_line_blend.py."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def _real_data_present() -> bool:
    # Note: the canonical Vegas dir is `data/raw/vegas_lines` (see
    # enhanced_model_v2.VEGAS_DIR). The plan's draft used `data/raw/vegas`
    # which doesn't exist; we use the real path so the smoke test exercises
    # the train-on-unmodified-apply-to-unmodified anchor path.
    return (
        Path("output/pairwise_v4.csv").exists()
        and Path("output/pairwise_v8.csv").exists()
        and Path("data/raw/march-machine-learning-2026/MTeams.csv").exists()
        and Path("data/raw/vegas_lines").exists()
    )


def test_anchor_check_matches_baseline():
    """If a cell's v8 csv equals the canonical baseline byte-for-byte,
    _anchor_check returns matches=True with max_abs_diff=0."""
    from src.eval_r64_line_blend import _anchor_check

    # Build two identical CSVs in memory via tmp files.
    df = pd.DataFrame({
        "season": [2024, 2024, 2025, 2025],
        "team_a": [1, 1, 2, 2],
        "team_b": [2, 3, 3, 4],
        "p_a_wins": [0.55, 0.6, 0.5, 0.7],
    })
    import tempfile
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".csv") as f1:
        df.to_csv(f1.name, index=False)
        a = f1.name
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".csv") as f2:
        df.to_csv(f2.name, index=False)
        b = f2.name
    res = _anchor_check(a, b)
    assert res["matches"] is True
    assert res["max_abs_diff"] == 0.0
    Path(a).unlink(); Path(b).unlink()


def test_anchor_check_flags_difference():
    """A 1e-3 difference in p_a_wins triggers matches=False."""
    from src.eval_r64_line_blend import _anchor_check

    a = pd.DataFrame({"season": [2024], "team_a": [1], "team_b": [2],
                      "p_a_wins": [0.55]})
    b = a.copy()
    b["p_a_wins"] = 0.551  # 0.001 difference
    import tempfile
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".csv") as f1:
        a.to_csv(f1.name, index=False)
        ap = f1.name
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".csv") as f2:
        b.to_csv(f2.name, index=False)
        bp = f2.name
    res = _anchor_check(ap, bp)
    assert res["matches"] is False
    assert res["max_abs_diff"] == pytest.approx(0.001, abs=1e-9)
    Path(ap).unlink(); Path(bp).unlink()


def test_pick_verdict_pass_when_delta_above_25():
    """Per the spec decision matrix: delta_total >= +25 with robust profile -> PASS."""
    from src.eval_r64_line_blend import _pick_verdict

    cells = [
        {"mode": "hard", "delta_total": 30, "wins": 8, "losses": 6, "ties": 8,
         "biggest_swing_value": 12.0},
        {"mode": "mean", "delta_total": 12, "wins": 7, "losses": 7, "ties": 8,
         "biggest_swing_value": 8.0},
    ]
    v = _pick_verdict(cells, baseline_total=2069)
    assert v["label"] == "PASS"
    assert v["best_mode"] == "hard"


def test_pick_verdict_marginal_when_delta_in_band():
    from src.eval_r64_line_blend import _pick_verdict

    cells = [
        {"mode": "hard", "delta_total": 15, "wins": 6, "losses": 6, "ties": 10,
         "biggest_swing_value": 5.0},
        {"mode": "mean", "delta_total": 8, "wins": 5, "losses": 7, "ties": 10,
         "biggest_swing_value": 4.0},
    ]
    v = _pick_verdict(cells, baseline_total=2069)
    assert v["label"] == "MARGINAL"


def test_pick_verdict_fail_when_below_10():
    from src.eval_r64_line_blend import _pick_verdict

    cells = [
        {"mode": "hard", "delta_total": 5, "wins": 5, "losses": 7, "ties": 10,
         "biggest_swing_value": 3.0},
        {"mode": "mean", "delta_total": -5, "wins": 4, "losses": 8, "ties": 10,
         "biggest_swing_value": 3.0},
    ]
    v = _pick_verdict(cells, baseline_total=2069)
    assert v["label"] == "FAIL"


def test_pick_verdict_pass_concentration_demotes_to_marginal():
    """If delta=+30 but >50% comes from a single season, demote to MARGINAL."""
    from src.eval_r64_line_blend import _pick_verdict

    cells = [
        # max single swing 18 of 30 = 60% concentration -> demote.
        {"mode": "hard", "delta_total": 30, "wins": 5, "losses": 5, "ties": 12,
         "biggest_swing_value": 18.0},
    ]
    v = _pick_verdict(cells, baseline_total=2069)
    assert v["label"] == "MARGINAL"


# --- real-data smoke (skip on fresh clone) ---


def test_run_eval_anchor_reproduces_canonical(tmp_path):
    """Smoke: run_eval with sigmas=[11.0], modes=[] (no override modes,
    just the v4-only anchor cell) reproduces the canonical v8 baseline.

    Baseline value (2034) reflects the current-XGB regenerated canonical
    pairwise_v8.csv. Previous era (XGB 2.x) scored 2069 -- see
    TODO.md 2026-05-14 entry on XGB env drift."""
    if not _real_data_present():
        pytest.skip("real data not present")
    from src.eval_r64_line_blend import run_eval

    out_dir = tmp_path / "output"
    out_dir.mkdir()
    out_json = out_dir / "r64_line_blend_eval.json"
    summary = run_eval(
        v4_csv="output/pairwise_v4.csv",
        v8_baseline_csv="output/pairwise_v8.csv",
        sigmas=[11.0],
        modes=[],  # anchor only
        out_dir=out_dir,
        out_json=str(out_json),
    )
    # Anchor must match exactly.
    assert summary["anchor_check"]["matches"] is True
    assert abs(summary["v8_baseline"]["total_pts"] - 2034) < 1e-6
