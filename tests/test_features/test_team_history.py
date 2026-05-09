"""Unit tests for src/features/team_history.py.

Synthetic toy tournaments verify rounds_won counting and per-seed
baseline computation; real Kaggle data smoke-tests the integrator.
"""
import pytest
import pandas as pd

from src.features.team_history import _rounds_won_per_team_season


def _toy_tourney(rows):
    """Build a MNCAATourney*Results-shaped DataFrame from compact tuples.

    rows: list of (season, daynum, w_team_id, l_team_id) tuples.
    """
    return pd.DataFrame(
        rows,
        columns=["Season", "DayNum", "WTeamID", "LTeamID"],
    )


def test_rounds_won_counts_games_won_per_team_season():
    # Season 2024: team 100 wins R64 (day 136) + R32 (day 138) + S16 (day 143),
    # loses E8 (day 145) -> rounds_won = 3.
    # team 200 loses R64 -> rounds_won = 0.
    # team 300 wins championship (days 136, 138, 143, 145, 152, 154) -> rounds_won = 6.
    tr = _toy_tourney([
        (2024, 136, 100, 200),  # team 100 wins R64
        (2024, 138, 100, 201),  # team 100 wins R32
        (2024, 143, 100, 202),  # team 100 wins S16
        (2024, 145, 203, 100),  # team 100 LOSES E8
        (2024, 136, 300, 301),  # team 300 wins R64
        (2024, 138, 300, 302),  # team 300 wins R32
        (2024, 143, 300, 303),  # team 300 wins S16
        (2024, 145, 300, 304),  # team 300 wins E8
        (2024, 152, 300, 305),  # team 300 wins F4
        (2024, 154, 300, 306),  # team 300 wins Champ
    ])
    out = _rounds_won_per_team_season(tr, max_season=None)
    out = out.set_index(["Season", "TeamID"])["rounds_won"].to_dict()
    assert out[(2024, 100)] == 3
    assert out[(2024, 200)] == 0
    assert out[(2024, 300)] == 6


def test_rounds_won_leak_guard_fires_when_input_has_future_season():
    tr = _toy_tourney([
        (2023, 136, 100, 200),
        (2024, 136, 300, 400),  # future-season row
    ])
    with pytest.raises(AssertionError, match="Leak guard"):
        _rounds_won_per_team_season(tr, max_season=2023)


def _toy_seeds(rows):
    """Build a MNCAATourneySeeds-shaped DataFrame from (season, seed_str, team_id) tuples."""
    return pd.DataFrame(rows, columns=["Season", "Seed", "TeamID"])


def test_per_seed_baseline_basic():
    """1-seeds: avg of (4, 1, 6) = 11/3; 16-seeds: avg 0."""
    from src.features.team_history import compute_per_seed_baseline
    tr = _toy_tourney([
        # 2021: team 100 (1-seed) wins R64, R32, S16, E8 → 4 wins (loses F4)
        (2021, 136, 100, 116), (2021, 138, 100, 117),
        (2021, 143, 100, 118), (2021, 145, 100, 119),
        (2021, 152, 120, 100),  # team 100 loses F4
        # 2022: team 200 (1-seed) wins R64 only → 1 win
        (2022, 136, 200, 216), (2022, 138, 217, 200),
        # 2023: team 300 (1-seed) wins championship → 6 wins
        (2023, 136, 300, 316), (2023, 138, 300, 317),
        (2023, 143, 300, 318), (2023, 145, 300, 319),
        (2023, 152, 300, 320), (2023, 154, 300, 321),
        # 16-seeds: all lose R64 (0 wins each)
    ])
    seeds = _toy_seeds([
        (2021, "W01", 100), (2021, "W16", 116),
        (2022, "X01", 200), (2022, "X16", 216),
        (2023, "Y01", 300), (2023, "Y16", 316),
    ])
    baseline = compute_per_seed_baseline(tr, seeds, max_season=2024)
    # 1-seeds: rounds_won = (4 + 1 + 6) / 3 = 11/3 = 3.667
    assert baseline[1] == pytest.approx(11.0 / 3.0)
    # 16-seeds: rounds_won = (0 + 0 + 0) / 3 = 0
    assert baseline[16] == 0.0


def test_per_seed_baseline_leak_guard():
    """compute_per_seed_baseline propagates the leak assertion."""
    from src.features.team_history import compute_per_seed_baseline
    tr = _toy_tourney([
        (2023, 136, 100, 116),
        (2024, 136, 200, 216),  # future
    ])
    seeds = _toy_seeds([
        (2023, "W01", 100), (2023, "W16", 116),
        (2024, "X01", 200), (2024, "X16", 216),
    ])
    with pytest.raises(AssertionError, match="Leak guard"):
        compute_per_seed_baseline(tr, seeds, max_season=2023)


def test_per_seed_baseline_missing_seed_falls_back_to_overall_mean():
    """A seed with no observations gets the overall-mean rounds_won."""
    from src.features.team_history import compute_per_seed_baseline
    # Three team-seasons total: 100 (1-seed, 2 wins), 116 (16-seed, 0 wins),
    # 117 (8-seed, 0 wins). Overall mean = (2 + 0 + 0) / 3 = 0.667.
    tr = _toy_tourney([
        (2021, 136, 100, 116), (2021, 138, 100, 117),
    ])
    seeds = _toy_seeds([
        (2021, "W01", 100), (2021, "W16", 116), (2021, "W08", 117),
    ])
    baseline = compute_per_seed_baseline(tr, seeds, max_season=2022)
    # 1-seed observed (avg 2.0), 8-seed observed (avg 0.0), 16-seed observed (avg 0.0).
    # Seeds 2-7, 9-15 missing → callers fall back via __fallback__.
    assert baseline["__fallback__"] == pytest.approx(2.0 / 3.0)
    assert 2 not in baseline  # not observed → caller uses fallback
    assert baseline[1] == 2.0
    assert baseline[8] == 0.0
    assert baseline[16] == 0.0


def test_shrunk_mean_empty_returns_zero():
    from src.features.team_history import shrunk_mean
    assert shrunk_mean([], k=3) == 0.0


def test_shrunk_mean_single_value_shrinks_toward_zero():
    """Single +6 with k=3: (6 + 0) / (1 + 3) = 1.5."""
    from src.features.team_history import shrunk_mean
    assert shrunk_mean([6.0], k=3) == pytest.approx(1.5)


def test_shrunk_mean_at_n_equals_k_is_halfway():
    """3 obs averaging 4.0, k=3 → 4.0 * 3 / 6 = 2.0 (halfway between 0 and 4)."""
    from src.features.team_history import shrunk_mean
    assert shrunk_mean([4.0, 4.0, 4.0], k=3) == pytest.approx(2.0)


def test_shrunk_mean_at_large_n_approaches_raw_mean():
    """100 obs averaging 1.0, k=3 → 100/103 ≈ 0.97."""
    from src.features.team_history import shrunk_mean
    assert shrunk_mean([1.0] * 100, k=3) == pytest.approx(100.0 / 103.0)


def test_shrunk_ewma_empty_returns_zero():
    from src.features.team_history import shrunk_ewma
    assert shrunk_ewma([], half_life=2, k=3) == 0.0


def test_shrunk_ewma_single_recent_observation_matches_shrunk_mean():
    """1 obs at year_ago=1 with HL=2: weight = 1.0, weighted_mean = value.
    Then n-based shrinkage: (1 * value + 3 * 0) / (1 + 3) = value/4.
    Equivalent to shrunk_mean([value], k=3)."""
    from src.features.team_history import shrunk_ewma
    assert shrunk_ewma([(1, 2.0)], half_life=2, k=3) == pytest.approx(0.5)


def test_shrunk_ewma_weights_decay_correctly():
    """4 obs at years_ago = (1, 3, 5, 9) all with residual 2.0, HL=2.
    Weights: w(1)=1.0, w(3)=0.5, w(5)=0.25, w(9)=0.0625.
    Weighted mean = 2.0 (constant residual). After n-shrinkage with n=4, k=3:
    (4 * 2.0 + 3 * 0) / (4 + 3) = 8/7 ≈ 1.143."""
    from src.features.team_history import shrunk_ewma
    out = shrunk_ewma([(1, 2.0), (3, 2.0), (5, 2.0), (9, 2.0)],
                      half_life=2, k=3)
    assert out == pytest.approx(8.0 / 7.0)


def test_shrunk_ewma_recent_negatives_dominate_old_positive():
    """UConn 2023 walkthrough from spec: years_ago = (9, 7, 2, 1) with
    residuals (+5, +0.3, -1, -1.5). Weights: 0.0625, 0.125, ~0.7071, 1.0.
    Weighted mean ≈ -0.98. n=4, k=3 → (4 * -0.98 + 0) / 7 ≈ -0.56."""
    from src.features.team_history import shrunk_ewma
    out = shrunk_ewma(
        [(9, 5.0), (7, 0.3), (2, -1.0), (1, -1.5)],
        half_life=2, k=3,
    )
    # Hand-computed: weights = [0.0625, 0.125, 0.5**0.5=0.7071..., 1.0]
    # weighted_sum = 0.0625*5 + 0.125*0.3 + 0.7071*(-1) + 1.0*(-1.5)
    #              = 0.3125 + 0.0375 - 0.7071 - 1.5 = -1.8571
    # weight_sum = 1.8946
    # weighted_mean = -0.9802
    # shrunk = 4 * -0.9802 / 7 = -0.5601
    assert out == pytest.approx(-0.5601, abs=1e-3)
