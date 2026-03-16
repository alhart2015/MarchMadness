"""Tests for bracket output formatting."""

import pandas as pd
import pytest

from src.bracket.output import format_advancement_table, export_bracket_csv


@pytest.fixture
def advancement_probs():
    return {
        101: {1: 0.95, 2: 0.70, 3: 0.40, 4: 0.20, 5: 0.10, 6: 0.05},
        102: {1: 0.80, 2: 0.50, 3: 0.20, 4: 0.08, 5: 0.03, 6: 0.01},
    }


@pytest.fixture
def teams():
    return pd.DataFrame({"TeamID": [101, 102], "TeamName": ["Duke", "UNC"]})


def test_format_advancement_table(advancement_probs, teams):
    table = format_advancement_table(advancement_probs, teams)
    assert isinstance(table, str)
    assert "Duke" in table
    assert "UNC" in table


def test_export_bracket_csv(advancement_probs, teams, tmp_path):
    output_path = tmp_path / "bracket.csv"
    export_bracket_csv(advancement_probs, teams, str(output_path))
    df = pd.read_csv(output_path)
    assert len(df) == 2
    assert "TeamName" in df.columns
    assert "R1" in df.columns
