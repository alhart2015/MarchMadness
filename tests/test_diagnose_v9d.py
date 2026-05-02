"""Unit tests for src/diagnose_v9d.py.

Gate function tests use plain-dict inputs (no real-data dependency).
A small integration test exercises compute_gate end-to-end on
synthetic fixtures to pin the call chain.
"""
import pandas as pd
import pytest


def _write_pairwise(path, rows):
    df = pd.DataFrame(rows, columns=["season", "team_a", "team_b", "p_a_wins"])
    df.to_csv(path, index=False)


def _write_seeds(path, rows):
    df = pd.DataFrame(rows, columns=["Season", "Seed", "TeamID"])
    df.to_csv(path, index=False)


def _write_results(path, rows):
    df = pd.DataFrame(rows, columns=["Season", "DayNum", "WTeamID", "LTeamID"])
    df.to_csv(path, index=False)


def test_check_gate_passes_when_headroom_above_threshold():
    """LL_v9c - LL_v9d >= GATE_LL_HEADROOM_MIN -> pass."""
    from src.diagnose_v9d import check_gate, GATE_LL_HEADROOM_MIN

    diag = {
        "ll_v9c": 0.45, "ll_v9d": 0.44,
        "headroom": 0.01, "threshold": GATE_LL_HEADROOM_MIN,
    }
    out = check_gate(diag)
    assert out["pass"] is True
    assert "headroom" in out["reason"].lower()


def test_check_gate_fails_when_headroom_below_threshold():
    """Headroom below threshold -> fail."""
    from src.diagnose_v9d import check_gate, GATE_LL_HEADROOM_MIN

    diag = {
        "ll_v9c": 0.45, "ll_v9d": 0.45 - (GATE_LL_HEADROOM_MIN - 0.0001),
        "headroom": GATE_LL_HEADROOM_MIN - 0.0001,
        "threshold": GATE_LL_HEADROOM_MIN,
    }
    out = check_gate(diag)
    assert out["pass"] is False


def test_check_gate_fails_when_v9d_is_worse():
    """Negative headroom -> fail."""
    from src.diagnose_v9d import check_gate, GATE_LL_HEADROOM_MIN

    diag = {
        "ll_v9c": 0.44, "ll_v9d": 0.45,
        "headroom": -0.01, "threshold": GATE_LL_HEADROOM_MIN,
    }
    out = check_gate(diag)
    assert out["pass"] is False


def test_check_gate_passes_at_threshold_exactly():
    """Headroom == threshold -> pass (strict >=)."""
    from src.diagnose_v9d import check_gate, GATE_LL_HEADROOM_MIN

    diag = {
        "ll_v9c": 0.45, "ll_v9d": 0.45 - GATE_LL_HEADROOM_MIN,
        "headroom": GATE_LL_HEADROOM_MIN,
        "threshold": GATE_LL_HEADROOM_MIN,
    }
    out = check_gate(diag)
    assert out["pass"] is True


def test_compute_gate_returns_expected_keys(tmp_path):
    """compute_gate runs end-to-end on synthetic inputs and returns
    a dict with all expected fields. Assertions on the LL values
    themselves are intentionally loose -- the unit test pins shape,
    not a magic number.
    """
    from src.diagnose_v9d import compute_gate

    pw_v4 = tmp_path / "pw_v4.csv"
    pw_bt = tmp_path / "pw_bt.csv"
    seeds = tmp_path / "seeds.csv"
    results = tmp_path / "results.csv"
    # 4 teams, 2 seasons, 2 games per season -- enough rows for
    # double_loso_eval to fit and predict.
    _write_pairwise(pw_v4, [
        (2022, 1, 2, 0.7), (2022, 3, 4, 0.6),
        (2023, 1, 2, 0.55), (2023, 3, 4, 0.5),
    ])
    _write_pairwise(pw_bt, [
        (2022, 1, 2, 0.6), (2022, 3, 4, 0.55),
        (2023, 1, 2, 0.5), (2023, 3, 4, 0.45),
    ])
    _write_seeds(seeds, [
        (2022, "W01", 1), (2022, "W08", 2), (2022, "X01", 3), (2022, "X08", 4),
        (2023, "W01", 1), (2023, "W08", 2), (2023, "X01", 3), (2023, "X08", 4),
    ])
    _write_results(results, [
        (2022, 136, 1, 2), (2022, 138, 3, 4),
        (2023, 136, 1, 2), (2023, 138, 3, 4),
    ])

    diag = compute_gate(
        pairwise_v4_csv=str(pw_v4),
        pairwise_bt_csv=str(pw_bt),
        results_csv=str(results),
        seeds_csv=str(seeds),
    )
    assert set(diag.keys()) >= {
        "n_games_v9c", "n_games_v9d",
        "ll_v9c", "ll_v9d", "headroom", "threshold",
    }
    # Same per-game frame underlies both evals -> n_games equal.
    assert diag["n_games_v9c"] == diag["n_games_v9d"]
    # LLs are both finite, non-negative.
    assert diag["ll_v9c"] >= 0 and diag["ll_v9d"] >= 0
    assert diag["threshold"] == 0.001
