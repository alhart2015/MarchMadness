"""Derive `expected_round` for a (Season, TeamID_a, TeamID_b) tournament pair.

Returned round is the smallest bracket round at which the two teams could
meet under the bracket structure recorded in MNCAATourneySlots.csv:

    1 = R64 (first-round meeting; same seed-pair slot)
    2 = R32
    3 = S16
    4 = E8  (region final)
    5 = F4  (national semifinal, cross-region)
    6 = Champ (cross-region, opposite halves of bracket)

For played tournament games, this is exactly the round the game occurred
in (two teams alive at their first-possible-meeting round and meeting later
would imply they both advanced through that round without playing each
other, which the bracket structure forbids -- they would have already met).

The lookup is built from the slots CSV directly, so any change to bracket
shape in a future season is absorbed automatically.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Dict, FrozenSet, Optional, Tuple

import pandas as pd

# Round-prefix -> integer mapping. The slot identifier 'R{n}...' carries
# the round in its first two characters.
_ROUND_FROM_PREFIX = {"R1": 1, "R2": 2, "R3": 3, "R4": 4, "R5": 5, "R6": 6}


def _strip_play_in_suffix(seed_string: str) -> str:
    """Normalize 'W11a' / 'W11b' to 'W11'.

    Play-in games happen before R64 to decide which of two ties takes the
    main-draw seed slot. For round-lookup purposes, both play-in teams share
    the same expected meeting round with everyone else: whichever of them
    survives takes the canonical seed (e.g., W11) into R64.
    """
    if seed_string and seed_string[-1] in ("a", "b"):
        return seed_string[:-1]
    return seed_string


def _resolve_seed_set(
    node: str,
    slots_by_slot_id: Dict[str, Tuple[str, str]],
) -> FrozenSet[str]:
    """Walk down the slot tree from a node to the set of seed strings in its sub-tree.

    `node` may be a slot identifier ('R3W1') or a seed string ('W01', 'W16a').
    Returns the frozenset of canonical seed strings reachable from that node.
    """
    if node not in slots_by_slot_id:
        # Leaf: a seed string. Normalize play-in suffix.
        return frozenset({_strip_play_in_suffix(node)})
    strong, weak = slots_by_slot_id[node]
    return _resolve_seed_set(strong, slots_by_slot_id) | _resolve_seed_set(
        weak, slots_by_slot_id
    )


def _round_from_slot_id(slot_id: str) -> Optional[int]:
    """Extract the round integer from a slot id like 'R3W1' or play-in 'W11'.

    Returns None for play-in (non-R-prefixed) slots.
    """
    return _ROUND_FROM_PREFIX.get(slot_id[:2])


def build_pair_to_round_for_season(
    slots_df: pd.DataFrame,
    season: int,
) -> Dict[FrozenSet[str], int]:
    """Build {frozenset({seed_a, seed_b}) -> round_int} for a single season.

    Walks every R1..R6 slot once: each slot defines a (StrongSeed sub-tree,
    WeakSeed sub-tree) cross-pairing. Every (s_strong, s_weak) pair across
    those two sub-trees has its first-possible-meeting round set to that
    slot's round.
    """
    season_slots = slots_df[slots_df.Season == season]
    slots_by_slot_id: Dict[str, Tuple[str, str]] = {
        row.Slot: (row.StrongSeed, row.WeakSeed)
        for row in season_slots.itertuples(index=False)
    }
    pair_round: Dict[FrozenSet[str], int] = {}
    for slot_id, (strong, weak) in slots_by_slot_id.items():
        r = _round_from_slot_id(slot_id)
        if r is None:
            continue
        left = _resolve_seed_set(strong, slots_by_slot_id)
        right = _resolve_seed_set(weak, slots_by_slot_id)
        for s_a in left:
            for s_b in right:
                if s_a == s_b:
                    continue
                key = frozenset({s_a, s_b})
                # Slots are visited bottom-up by tree depth in practice, but
                # be defensive: keep the smallest round assigned.
                cur = pair_round.get(key)
                if cur is None or r < cur:
                    pair_round[key] = r
    return pair_round


def build_team_to_seed_for_season(
    seeds_df: pd.DataFrame,
    season: int,
) -> Dict[int, str]:
    """{TeamID -> seed string} for a season, keeping play-in suffix intact."""
    s = seeds_df[seeds_df.Season == season]
    return {int(r.TeamID): str(r.Seed) for r in s.itertuples(index=False)}


@lru_cache(maxsize=None)
def _cached_lookup(slots_path: str, seeds_path: str) -> Dict[int, Dict]:
    slots = pd.read_csv(slots_path)
    seeds = pd.read_csv(seeds_path)
    out: Dict[int, Dict] = {}
    for season in sorted(slots.Season.unique()):
        season = int(season)
        out[season] = {
            "pair_round": build_pair_to_round_for_season(slots, season),
            "team_seed": build_team_to_seed_for_season(seeds, season),
        }
    return out


def expected_round_for_pair(
    season: int,
    team_a: int,
    team_b: int,
    slots_csv: str = "data/raw/march-machine-learning-2026/MNCAATourneySlots.csv",
    seeds_csv: str = "data/raw/march-machine-learning-2026/MNCAATourneySeeds.csv",
) -> Optional[int]:
    """Return 1..6 (R64..Champ) or None if either team is missing in seeds."""
    lookup = _cached_lookup(slots_csv, seeds_csv)
    if season not in lookup:
        return None
    team_seed = lookup[season]["team_seed"]
    s_a = team_seed.get(int(team_a))
    s_b = team_seed.get(int(team_b))
    if s_a is None or s_b is None:
        return None
    pair_round = lookup[season]["pair_round"]
    return pair_round.get(frozenset({_strip_play_in_suffix(s_a), _strip_play_in_suffix(s_b)}))
