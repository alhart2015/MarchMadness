"""Smoke test for src/predict_2026_v9c.py.

Builds tiny synthetic Kaggle-shaped inputs for one historical season
and 2026, monkeypatches input file paths, then runs main() and
asserts the output JSON files exist and have plausible content.
"""
import json
import sys
from pathlib import Path

import pandas as pd
import pytest


def _write_synthetic_inputs(tmp_path: Path) -> dict:
    """Create the input files predict_2026_v9c.main() reads.

    One historical season (2024) with two played games gives the
    trainer a per-game training row to fit on. 2026 has 4 seeded
    teams forming the same bracket-walk shape.
    """
    data_dir = tmp_path / "data" / "raw" / "march-machine-learning-2026"
    data_dir.mkdir(parents=True)
    out_dir = tmp_path / "output"
    out_dir.mkdir()

    # Pairwise v4: one historical season + 2026 (the 2026 rows are
    # not actually consumed by main() -- v4 2026 lives in JSON --
    # but the trainer needs LOSO data, which load_per_game_data_with_upset
    # reads from this CSV.)
    pd.DataFrame({
        "season":  [2024, 2024, 2024],
        "team_a":  [1, 1, 2],
        "team_b":  [2, 3, 3],
        "p_a_wins": [0.7, 0.6, 0.55],
    }).to_csv(out_dir / "pairwise_v4.csv", index=False)

    # Seeds: 2024 + 2026, four teams each.
    pd.DataFrame({
        "Season": [2024, 2024, 2024, 2024, 2026, 2026, 2026, 2026],
        "Seed":   ["W01", "W08", "W09", "W16",
                   "W01", "W08", "W09", "W16"],
        "TeamID": [1, 2, 3, 4, 1, 2, 3, 4],
    }).to_csv(data_dir / "MNCAATourneySeeds.csv", index=False)

    # Slots: same bracket shape for both years (R1 + R2).
    pd.DataFrame({
        "Season": [2024]*3 + [2026]*3,
        "Slot":   ["R1W1", "R1W8", "R2W1",  "R1W1", "R1W8", "R2W1"],
        "StrongSeed": ["W01", "W08", "R1W1", "W01", "W08", "R1W1"],
        "WeakSeed":   ["W16", "W09", "R1W8", "W16", "W09", "R1W8"],
    }).to_csv(data_dir / "MNCAATourneySlots.csv", index=False)

    # Compact results: two played games in 2024 so trainer has rows.
    pd.DataFrame({
        "Season": [2024, 2024],
        "DayNum": [136, 138],
        "WTeamID": [1, 1],
        "WScore": [70, 75],
        "LTeamID": [2, 3],
        "LScore": [60, 65],
    }).to_csv(data_dir / "MNCAATourneyCompactResults.csv", index=False)

    # v4 2026 raw predictions: 6 pair-pairs across the 4 seeded teams.
    v4_2026 = {
        "1_2": 0.62,
        "1_3": 0.58,
        "1_4": 0.78,
        "2_3": 0.51,
        "2_4": 0.66,
        "3_4": 0.60,
    }
    with open(out_dir / "pairwise_probs_v4.json", "w") as f:
        json.dump(v4_2026, f)

    return {
        "data_dir": data_dir,
        "out_dir": out_dir,
        "pairwise_v4_csv": str(out_dir / "pairwise_v4.csv"),
        "results_csv": str(data_dir / "MNCAATourneyCompactResults.csv"),
        "seeds_csv": str(data_dir / "MNCAATourneySeeds.csv"),
        "slots_csv": str(data_dir / "MNCAATourneySlots.csv"),
        "v4_json": str(out_dir / "pairwise_probs_v4.json"),
        "canonical_json": str(out_dir / "pairwise_probs.json"),
        "v9c_versioned_json": str(out_dir / "pairwise_probs_v9c_2026.json"),
    }


def test_predict_2026_v9c_smoke(tmp_path, monkeypatch):
    """End-to-end smoke: synthetic inputs -> main() -> two JSON outputs
    written with the expected schema and value range.
    """
    paths = _write_synthetic_inputs(tmp_path)

    # Run main() with cwd = tmp_path so the relative paths in the
    # script resolve against our synthetic files.
    monkeypatch.chdir(tmp_path)

    # The script imports DATA from train_upset_model, which is set at
    # import time to "data/raw/march-machine-learning-2026". Patch it
    # to point at our synthetic data dir before main() runs.
    import src.train_upset_model as tum
    monkeypatch.setattr(tum, "DATA", paths["data_dir"])
    import src.predict_2026_v9c as p2026
    monkeypatch.setattr(p2026, "DATA", paths["data_dir"])

    p2026.main()

    # Both output files exist.
    assert Path(paths["canonical_json"]).exists()
    assert Path(paths["v9c_versioned_json"]).exists()

    # Versioned and canonical have the same content (script writes the
    # same dict twice).
    canon = json.loads(Path(paths["canonical_json"]).read_text())
    versioned = json.loads(Path(paths["v9c_versioned_json"]).read_text())
    assert canon == versioned

    # All 6 input pair-pair keys are present in the output.
    expected_keys = {"1_2", "1_3", "1_4", "2_3", "2_4", "3_4"}
    assert set(canon.keys()) == expected_keys

    # All probabilities are in (0, 1) (not 0 or 1 exactly; xgboost
    # almost never produces hard 0/1 with regularization).
    for k, p in canon.items():
        assert 0.0 < p < 1.0, f"pair {k} probability out of range: {p}"
