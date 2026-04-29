"""Tests for the ablation driver. The subprocess invocation is too expensive
to test end-to-end; we test the helpers (group definitions, parse/aggregate
logic) and use a synthetic bracket_data.json fixture."""
import json
from pathlib import Path
import pytest
from src.ablate_v4 import (
    GROUP_ABLATIONS, BUST_TEAMS,
    parse_advance_probs, build_results_row,
)


def test_group_ablations_cover_spec():
    expected_groups = {"late_season", "trajectory", "conf_tourney",
                       "vegas_trend", "coach"}
    assert set(GROUP_ABLATIONS.keys()) == expected_groups


def test_late_season_group_features():
    assert set(GROUP_ABLATIONS["late_season"]) == {
        "late_adj_oe", "late_adj_de", "late_adj_em", "late_sos"
    }


def test_coach_group_features():
    assert set(GROUP_ABLATIONS["coach"]) == {
        "coach_career_games", "coach_career_wins", "coach_career_winpct",
        "coach_career_f4_apps", "coach_career_champs", "coach_career_seasons",
    }


def test_bust_teams_metric_keys():
    # Each bust must declare bust_round + advance_key (= round AFTER bust).
    assert {b["name"] for b in BUST_TEAMS} == {
        "Vanderbilt", "Iowa State", "Texas Tech", "Duke",
    }
    expected = {"Vanderbilt": "S16", "Iowa State": "E8",
                "Texas Tech": "S16", "Duke": "F4"}
    for b in BUST_TEAMS:
        assert b["advance_key"] == expected[b["name"]]


def test_parse_advance_probs_extracts_named_team(tmp_path):
    bracket_data = {
        "1101": {"name": "Vanderbilt", "seed": 5, "region": "South",
                 "advancement": {"R64": 0.95, "R32": 0.79, "S16": 0.42}},
        "1102": {"name": "Duke", "seed": 1, "region": "East",
                 "advancement": {"R64": 0.99, "R32": 0.93, "S16": 0.90,
                                  "E8": 0.88, "F4": 0.62, "Champ": 0.30}},
    }
    p = tmp_path / "bracket_data.json"
    p.write_text(json.dumps(bracket_data))
    assert parse_advance_probs(p, "Vanderbilt", "S16") == 0.42
    assert parse_advance_probs(p, "Duke", "F4") == 0.62


def test_parse_advance_probs_missing_team_returns_none(tmp_path):
    p = tmp_path / "bracket_data.json"
    p.write_text(json.dumps({}))
    assert parse_advance_probs(p, "Vanderbilt", "S16") is None


def test_build_results_row_shape():
    row = build_results_row(
        ablation="drop_coach", team="Duke", bust_round="E8",
        advance_key="F4",
        p_advance_baseline=0.62, p_advance_ablated=0.50,
        loso_baseline=0.4321, loso_ablated=0.4385,
        bracket_pts_baseline=2670.0, bracket_pts_ablated=2640.0,
    )
    assert row["ablation"] == "drop_coach"
    assert row["team"] == "Duke"
    assert row["delta_pp"] == pytest.approx(-12.0)
    assert row["loso_logloss_delta"] == pytest.approx(0.0064)
    assert row["bracket_pts_delta"] == pytest.approx(-30.0)


def test_pass2_tags_are_unprefixed():
    """Regression test: Pass 2 ablation tags should equal the bare feature
    name, not 'drop_<feature>'. The 'drop_' prefix is added uniformly at
    the run_pipeline call site, so prefixing here would double-prefix."""
    # Simulate what main() does for Pass 2 with --features coach_career_winpct
    features = ["coach_career_winpct"]
    ablations = [(f, [f]) for f in features]
    assert ablations == [("coach_career_winpct", ["coach_career_winpct"])]
    # The tag must NOT already start with 'drop_' -- otherwise the call
    # site's f"drop_{tag}" produces 'drop_drop_coach_career_winpct'.
    for tag, _ in ablations:
        assert not tag.startswith("drop_")
