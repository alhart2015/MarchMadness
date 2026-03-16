"""Tests for post-ingestion data validation."""

import pandas as pd
import pytest

from src.ingest.validation import validate_ingested_data, ValidationError


def test_validate_passes_good_data(sample_detailed_results, sample_teams, sample_seeds):
    data = {
        "teams": sample_teams,
        "regular_season": sample_detailed_results,
        "seeds": sample_seeds,
    }
    # Should not raise
    validate_ingested_data(data)


def test_validate_fails_null_team_ids(sample_detailed_results, sample_teams, sample_seeds):
    bad_results = sample_detailed_results.copy()
    bad_results.loc[0, "WTeamID"] = None
    data = {
        "teams": sample_teams,
        "regular_season": bad_results,
        "seeds": sample_seeds,
    }
    with pytest.raises(ValidationError, match="null TeamID"):
        validate_ingested_data(data)


def test_validate_fails_missing_columns(sample_teams, sample_seeds):
    bad_results = pd.DataFrame({"Season": [2023], "WTeamID": [1101]})
    data = {
        "teams": sample_teams,
        "regular_season": bad_results,
        "seeds": sample_seeds,
    }
    with pytest.raises(ValidationError, match="Missing columns"):
        validate_ingested_data(data)
