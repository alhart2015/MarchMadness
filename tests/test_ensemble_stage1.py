"""Unit tests for src/ensemble_stage1.py."""
from pathlib import Path
import numpy as np
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


def test_blend_pairwise_csvs_three_inputs(tmp_path):
    """Three input CSVs at uniform 1/3 weights produce row-wise mean
    of p_a_wins. Schema and join coverage same as average_pairwise_csvs.
    """
    from src.ensemble_stage1 import blend_pairwise_csvs

    csvs = []
    p_values = [0.6, 0.4, 0.5]
    for i, p in enumerate(p_values):
        path = tmp_path / f"in_{i}.csv"
        pd.DataFrame({
            "season": 2024, "team_a": 1, "team_b": 2,
            "p_a_wins": [p],
        }).to_csv(path, index=False)
        csvs.append(str(path))

    out = tmp_path / "out.csv"
    blend_pairwise_csvs(csvs, weights=[1/3, 1/3, 1/3], out=str(out))

    df = pd.read_csv(out)
    assert list(df.columns) == ["season", "team_a", "team_b", "p_a_wins"]
    assert df["p_a_wins"].iloc[0] == pytest.approx(np.mean(p_values))


def test_blend_pairwise_csvs_anchor_one_input(tmp_path):
    """Single input at weight 1.0 reproduces input row-for-row."""
    from src.ensemble_stage1 import blend_pairwise_csvs

    src = tmp_path / "src.csv"
    pd.DataFrame({
        "season": [2024, 2024],
        "team_a": [1, 1],
        "team_b": [2, 3],
        "p_a_wins": [0.6, 0.7],
    }).to_csv(src, index=False)

    out = tmp_path / "out.csv"
    blend_pairwise_csvs([str(src)], weights=[1.0], out=str(out))

    expected = pd.read_csv(src)
    actual = pd.read_csv(out)
    pd.testing.assert_frame_equal(
        actual.sort_values(["season", "team_a", "team_b"]).reset_index(drop=True),
        expected.sort_values(["season", "team_a", "team_b"]).reset_index(drop=True),
    )


def test_blend_pairwise_csvs_weight_count_mismatch_raises(tmp_path):
    from src.ensemble_stage1 import blend_pairwise_csvs

    pd.DataFrame({"season": 2024, "team_a": [1], "team_b": [2],
                  "p_a_wins": [0.5]}).to_csv(tmp_path / "a.csv", index=False)

    with pytest.raises(ValueError, match="weights"):
        blend_pairwise_csvs(
            [str(tmp_path / "a.csv")],
            weights=[0.5, 0.5],
            out=str(tmp_path / "out.csv"),
        )


def test_blend_pairwise_csvs_weights_must_sum_to_one(tmp_path):
    from src.ensemble_stage1 import blend_pairwise_csvs

    pd.DataFrame({"season": 2024, "team_a": [1], "team_b": [2],
                  "p_a_wins": [0.5]}).to_csv(tmp_path / "a.csv", index=False)

    with pytest.raises(ValueError, match="sum"):
        blend_pairwise_csvs(
            [str(tmp_path / "a.csv")],
            weights=[0.7],
            out=str(tmp_path / "out.csv"),
        )
