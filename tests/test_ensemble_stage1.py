"""Unit tests for src/ensemble_stage1.py."""
from pathlib import Path
import pandas as pd
import pytest


SCHEMA = ["season", "team_a", "team_b", "p_a_wins"]


def _write_csv(path, rows):
    pd.DataFrame(rows, columns=SCHEMA).to_csv(path, index=False)


def test_average_simple(tmp_path):
    from src.ensemble_stage1 import average_pairwise_csvs

    a = tmp_path / "a.csv"
    b = tmp_path / "b.csv"
    out = tmp_path / "ens.csv"
    _write_csv(a, [
        (2003, 1104, 1112, 0.20),
        (2003, 1104, 1113, 0.40),
    ])
    _write_csv(b, [
        (2003, 1104, 1112, 0.60),
        (2003, 1104, 1113, 0.80),
    ])

    average_pairwise_csvs(str(a), str(b), str(out), weights=(0.5, 0.5))

    df = pd.read_csv(out).sort_values(["season", "team_a", "team_b"]).reset_index(drop=True)
    assert list(df.columns) == SCHEMA
    assert df.loc[0, "p_a_wins"] == pytest.approx(0.40)
    assert df.loc[1, "p_a_wins"] == pytest.approx(0.60)


def test_anchor_weights_1_0_reproduces_first(tmp_path):
    """--weights 1.0,0.0 must reproduce input A row-for-row."""
    from src.ensemble_stage1 import average_pairwise_csvs

    a = tmp_path / "a.csv"
    b = tmp_path / "b.csv"
    out = tmp_path / "ens.csv"
    _write_csv(a, [
        (2003, 1104, 1112, 0.123456789),
        (2004, 1101, 1115, 0.987654321),
    ])
    _write_csv(b, [
        (2003, 1104, 1112, 0.5),
        (2004, 1101, 1115, 0.5),
    ])

    average_pairwise_csvs(str(a), str(b), str(out), weights=(1.0, 0.0))

    df_a = pd.read_csv(a).sort_values(["season", "team_a", "team_b"]).reset_index(drop=True)
    df_o = pd.read_csv(out).sort_values(["season", "team_a", "team_b"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(df_a, df_o)


def test_join_one_to_one_required(tmp_path):
    """If A has a (season, a, b) absent in B (or vice versa), error out."""
    from src.ensemble_stage1 import average_pairwise_csvs

    a = tmp_path / "a.csv"
    b = tmp_path / "b.csv"
    out = tmp_path / "ens.csv"
    _write_csv(a, [(2003, 1104, 1112, 0.5), (2003, 1104, 1113, 0.5)])
    _write_csv(b, [(2003, 1104, 1112, 0.5)])  # missing the 1113 pair

    with pytest.raises(ValueError, match="join coverage"):
        average_pairwise_csvs(str(a), str(b), str(out), weights=(0.5, 0.5))


def test_weights_sum_validation(tmp_path):
    """Weights must sum to 1.0 (within float tolerance) so output is a probability."""
    from src.ensemble_stage1 import average_pairwise_csvs

    a = tmp_path / "a.csv"
    b = tmp_path / "b.csv"
    out = tmp_path / "ens.csv"
    _write_csv(a, [(2003, 1104, 1112, 0.5)])
    _write_csv(b, [(2003, 1104, 1112, 0.5)])

    with pytest.raises(ValueError, match="sum to 1"):
        average_pairwise_csvs(str(a), str(b), str(out), weights=(0.5, 0.6))


def test_cli_invocation(tmp_path):
    """Smoke: src/ensemble_stage1.py with --weights and CSV paths runs."""
    import subprocess
    a = tmp_path / "a.csv"
    b = tmp_path / "b.csv"
    out = tmp_path / "ens.csv"
    _write_csv(a, [(2003, 1104, 1112, 0.3)])
    _write_csv(b, [(2003, 1104, 1112, 0.7)])

    cmd = [
        "python", "src/ensemble_stage1.py",
        "--in-a", str(a), "--in-b", str(b),
        "--out", str(out), "--weights", "0.5,0.5",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    df = pd.read_csv(out)
    assert df.loc[0, "p_a_wins"] == pytest.approx(0.5)
