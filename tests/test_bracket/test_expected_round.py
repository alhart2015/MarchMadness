"""Tests for src.bracket.expected_round."""
from src.bracket.expected_round import (
    _strip_play_in_suffix,
    expected_round_for_pair,
)


def test_strip_play_in_suffix():
    assert _strip_play_in_suffix("W11a") == "W11"
    assert _strip_play_in_suffix("W11b") == "W11"
    assert _strip_play_in_suffix("W11") == "W11"
    assert _strip_play_in_suffix("W01") == "W01"


def _seed_team(season, seeds_df, seed_string):
    row = seeds_df[(seeds_df.Season == season) & (seeds_df.Seed == seed_string)]
    if row.empty:
        return None
    return int(row.iloc[0].TeamID)


def test_same_region_r64_pair_2024():
    """A 1-seed vs 16-seed in the same region meets at R64."""
    import pandas as pd
    seeds = pd.read_csv("data/raw/march-machine-learning-2026/MNCAATourneySeeds.csv")
    one = _seed_team(2024, seeds, "W01")
    sixteen = _seed_team(2024, seeds, "W16")
    assert one is not None and sixteen is not None
    assert expected_round_for_pair(2024, one, sixteen) == 1


def test_same_region_r32_pair_2024():
    """A 1-seed vs 8-seed in the same region meets at R32."""
    import pandas as pd
    seeds = pd.read_csv("data/raw/march-machine-learning-2026/MNCAATourneySeeds.csv")
    one = _seed_team(2024, seeds, "W01")
    eight = _seed_team(2024, seeds, "W08")
    assert one is not None and eight is not None
    assert expected_round_for_pair(2024, one, eight) == 2


def test_same_region_s16_pair_2024():
    """A 1-seed vs 4-seed in the same region meets at S16."""
    import pandas as pd
    seeds = pd.read_csv("data/raw/march-machine-learning-2026/MNCAATourneySeeds.csv")
    one = _seed_team(2024, seeds, "W01")
    four = _seed_team(2024, seeds, "W04")
    assert one is not None and four is not None
    assert expected_round_for_pair(2024, one, four) == 3


def test_same_region_e8_pair_2024():
    """A 1-seed vs 2-seed in the same region meets at E8 (region final)."""
    import pandas as pd
    seeds = pd.read_csv("data/raw/march-machine-learning-2026/MNCAATourneySeeds.csv")
    one = _seed_team(2024, seeds, "W01")
    two = _seed_team(2024, seeds, "W02")
    assert one is not None and two is not None
    assert expected_round_for_pair(2024, one, two) == 4


def test_cross_region_f4_wx_2024():
    """W vs X always meets at F4 in modern bracket structure."""
    import pandas as pd
    seeds = pd.read_csv("data/raw/march-machine-learning-2026/MNCAATourneySeeds.csv")
    w1 = _seed_team(2024, seeds, "W01")
    x1 = _seed_team(2024, seeds, "X01")
    assert w1 is not None and x1 is not None
    assert expected_round_for_pair(2024, w1, x1) == 5


def test_cross_region_champ_2024():
    """W vs Y always meets at Champ (different halves of bracket)."""
    import pandas as pd
    seeds = pd.read_csv("data/raw/march-machine-learning-2026/MNCAATourneySeeds.csv")
    w1 = _seed_team(2024, seeds, "W01")
    y1 = _seed_team(2024, seeds, "Y01")
    assert w1 is not None and y1 is not None
    assert expected_round_for_pair(2024, w1, y1) == 6


def test_unknown_team_returns_none():
    assert expected_round_for_pair(2024, 9999999, 9999998) is None


def test_unknown_season_returns_none():
    import pandas as pd
    seeds = pd.read_csv("data/raw/march-machine-learning-2026/MNCAATourneySeeds.csv")
    w1 = _seed_team(2024, seeds, "W01")
    w16 = _seed_team(2024, seeds, "W16")
    assert expected_round_for_pair(1980, w1, w16) is None


def test_play_in_seed_returns_same_round_as_canonical():
    """A play-in team (W11a) should have the same expected_round vs other
    seeds as a non-play-in W11 would."""
    import pandas as pd
    seeds = pd.read_csv("data/raw/march-machine-learning-2026/MNCAATourneySeeds.csv")
    # Find a play-in seed
    play_ins = seeds[seeds.Seed.str.endswith("a") | seeds.Seed.str.endswith("b")]
    if play_ins.empty:
        return  # No play-ins in this archive; nothing to test.
    row = play_ins.iloc[0]
    season = int(row.Season)
    team_play_in = int(row.TeamID)
    base_seed = row.Seed[:-1]  # strip a/b
    # Find the opposing 1-seed in the same region.
    region = base_seed[0]
    opposing_seed = f"{region}01"
    opp_row = seeds[(seeds.Season == season) & (seeds.Seed == opposing_seed)]
    if opp_row.empty:
        return
    opp_team = int(opp_row.iloc[0].TeamID)
    r_play_in = expected_round_for_pair(season, team_play_in, opp_team)
    assert r_play_in is not None
    # Sanity: a 11-seed vs 1-seed in the same region meets at S16 (round 3).
    if base_seed.endswith("11"):
        assert r_play_in == 3
