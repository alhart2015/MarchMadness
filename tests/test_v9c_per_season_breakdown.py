"""Smoke test for v9c_per_season_breakdown.py.

Patches score_pairwise_path to return synthetic per-season totals so
the test does not depend on full Kaggle Tourney CSV files being
present. The fidelity of score_pairwise_path itself is covered by
tests/test_score_chalk_brackets.py.
"""
import sys
from pathlib import Path

import pandas as pd
import pytest


def test_v9c_per_season_breakdown_cli_smoke(tmp_path, monkeypatch):
    """main() writes a CSV with the documented schema and correct deltas."""
    v9c_csv = tmp_path / "v9c.csv"
    v8_csv = tmp_path / "v8.csv"
    out_csv = tmp_path / "out.csv"
    # Schema-only fixtures; scoring is patched.
    pd.DataFrame([{"season": 2024, "team_a": 1101, "team_b": 1102,
                   "p_a_wins": 0.6}]).to_csv(v9c_csv, index=False)
    pd.DataFrame([{"season": 2024, "team_a": 1101, "team_b": 1102,
                   "p_a_wins": 0.55}]).to_csv(v8_csv, index=False)

    import src.v9c_per_season_breakdown as mod

    def fake_score(path):
        # Path discriminator: v9c fixture has "v9c" in filename; v8 doesn't.
        if "v9c" in Path(path).name:
            return {"total_pts": 100.0,
                    "per_season_pts": {2024: 50.0, 2023: 50.0}}
        return {"total_pts": 90.0,
                "per_season_pts": {2024: 60.0, 2023: 30.0}}

    monkeypatch.setattr(mod, "score_pairwise_path", fake_score)
    monkeypatch.setattr(sys, "argv", [
        "_",
        "--v9c-pairwise", str(v9c_csv),
        "--v8-pairwise", str(v8_csv),
        "--output", str(out_csv),
    ])

    mod.main()

    out = pd.read_csv(out_csv)
    assert list(out.columns) == ["season", "v8_pts", "v9c_pts",
                                 "delta", "winner"]
    assert sorted(out["season"].tolist()) == [2023, 2024]
    rows = {int(r.season): r for _, r in out.iterrows()}
    # 2024: v9c 50, v8 60 -> delta -10, winner v8
    assert rows[2024]["delta"] == pytest.approx(-10.0)
    assert rows[2024]["winner"] == "v8"
    # 2023: v9c 50, v8 30 -> delta +20, winner v9c
    assert rows[2023]["delta"] == pytest.approx(20.0)
    assert rows[2023]["winner"] == "v9c"
