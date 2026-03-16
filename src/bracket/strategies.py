"""Bracket selection strategies that produce structurally valid brackets.

A valid bracket respects the bracket tree: picks in round N+1 must be
a subset of picks in round N, and each matchup slot has exactly one winner.
"""

import pandas as pd

from src.bracket.simulator import FIRST_ROUND_MATCHUPS


def _pick_slot_winner(
    team_a: int,
    team_b: int,
    probs: dict[int, dict[int, float]],
    round_num: int,
    scorer: str = "prob",
    points: int = 1,
) -> int:
    """Pick the winner of a single bracket slot.

    scorer="prob": pick higher advancement probability (chalk).
    scorer="ev": pick higher expected value (prob * points).
    """
    prob_a = probs.get(team_a, {}).get(round_num, 0.0)
    prob_b = probs.get(team_b, {}).get(round_num, 0.0)
    if scorer == "ev":
        val_a = prob_a * points
        val_b = prob_b * points
    else:
        val_a = prob_a
        val_b = prob_b
    return team_a if val_a >= val_b else team_b


def _fill_region(
    region_teams: pd.DataFrame,
    probs: dict[int, dict[int, float]],
    scorer: str = "prob",
    scoring: list[int] | None = None,
) -> dict[int, list[int]]:
    """Fill bracket picks for a single region (4 rounds)."""
    if scoring is None:
        scoring = [1, 2, 4, 8]

    seed_to_team = dict(zip(region_teams["Seed"], region_teams["TeamID"]))
    picks = {}

    # Round 1: known matchups from bracket structure
    round1_winners = []
    for seed_a, seed_b in FIRST_ROUND_MATCHUPS:
        team_a = seed_to_team[seed_a]
        team_b = seed_to_team[seed_b]
        winner = _pick_slot_winner(team_a, team_b, probs, 1, scorer, scoring[0])
        round1_winners.append(winner)
    picks[1] = round1_winners

    # Rounds 2-4: winners play each other in bracket order
    current = round1_winners
    for rnd in range(2, 5):
        next_round = []
        pts = scoring[rnd - 1] if rnd - 1 < len(scoring) else scoring[-1]
        for i in range(0, len(current), 2):
            winner = _pick_slot_winner(current[i], current[i + 1], probs, rnd, scorer, pts)
            next_round.append(winner)
        picks[rnd] = next_round
        current = next_round

    return picks


def chalk_bracket(
    bracket: pd.DataFrame,
    advancement_probs: dict[int, dict[int, float]],
) -> dict[int, list[int]]:
    """Chalk bracket: pick the higher-probability team in each slot.

    Returns {round_number: [team_ids advancing that round]}.
    Respects bracket structure — all picks are consistent across rounds.
    """
    regions = sorted(bracket["Region"].unique())
    all_picks = {r: [] for r in range(1, 7)}

    # Fill each region (rounds 1-4)
    regional_champs = []
    for region in regions:
        region_teams = bracket[bracket["Region"] == region]
        region_picks = _fill_region(region_teams, advancement_probs, scorer="prob")
        for rnd, teams in region_picks.items():
            all_picks[rnd].extend(teams)
        regional_champs.append(region_picks[4][0])

    # Semifinals (round 5) — only if at least 2 regions
    if len(regional_champs) >= 2:
        semi1 = _pick_slot_winner(regional_champs[0], regional_champs[1], advancement_probs, 5, "prob")
        semi2 = _pick_slot_winner(regional_champs[2], regional_champs[3], advancement_probs, 5, "prob") if len(regional_champs) >= 4 else semi1
        all_picks[5] = [semi1, semi2]

        # Championship (round 6)
        champ = _pick_slot_winner(semi1, semi2, advancement_probs, 6, "prob")
        all_picks[6] = [champ]

    return all_picks


def expected_value_bracket(
    bracket: pd.DataFrame,
    advancement_probs: dict[int, dict[int, float]],
    scoring: list[int] | None = None,
) -> dict[int, list[int]]:
    """Expected value bracket: pick teams maximizing expected points per slot.

    Returns {round_number: [team_ids advancing that round]}.
    """
    if scoring is None:
        scoring = [1, 2, 4, 8, 16, 32]

    regions = sorted(bracket["Region"].unique())
    all_picks = {r: [] for r in range(1, 7)}

    regional_champs = []
    for region in regions:
        region_teams = bracket[bracket["Region"] == region]
        region_picks = _fill_region(region_teams, advancement_probs, scorer="ev", scoring=scoring)
        for rnd, teams in region_picks.items():
            all_picks[rnd].extend(teams)
        regional_champs.append(region_picks[4][0])

    # Semifinals — only if at least 2 regions
    if len(regional_champs) >= 2:
        pts5 = scoring[4] if len(scoring) > 4 else scoring[-1]
        pts6 = scoring[5] if len(scoring) > 5 else scoring[-1]
        semi1 = _pick_slot_winner(regional_champs[0], regional_champs[1], advancement_probs, 5, "ev", pts5)
        semi2 = _pick_slot_winner(regional_champs[2], regional_champs[3], advancement_probs, 5, "ev", pts5) if len(regional_champs) >= 4 else semi1
        all_picks[5] = [semi1, semi2]

        # Championship
        champ = _pick_slot_winner(semi1, semi2, advancement_probs, 6, "ev", pts6)
        all_picks[6] = [champ]

    return all_picks
