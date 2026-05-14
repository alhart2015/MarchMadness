"""Fast in-memory v4 (stage-1) x v8 (stage-2) blend evaluation.

Background:
  v8 stage-2 was treated as a fixed object across 8 prior same-data-peer
  experiments. The discovery in this work: a linear blend
      p_final = alpha * p_v8 + (1 - alpha) * p_v4
  with the right alpha (selected by LOSO discipline on bracket points)
  produces a structurally stable improvement over current-env v8 single-
  seed. Mechanism: v8 stage-2 injects noise that flips some chalk picks
  in the wrong direction; v4 stage-1 acts as a regularizer that pulls
  the noisy stage-2 back toward sanity in those games.

Usage:
  from src.blend_v4_v8 import BlendEvaluator
  ev = BlendEvaluator()  # preloads all bracket structure
  per_season = ev.score(v8_df, v4_df, alpha=0.6)
  total = ev.loso_alpha_total(v8_df, v4_df, alphas)
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

DATA = Path("data/raw/march-machine-learning-2026")
ROUND_BY_PREFIX = {"R1": "R64", "R2": "R32", "R3": "S16",
                   "R4": "E8", "R5": "F4", "R6": "Champ"}
ROUND_PTS = {"R64": 1, "R32": 2, "S16": 4, "E8": 8, "F4": 16, "Champ": 32}


def _slot_round(slot: str):
    if slot.startswith("R"):
        return ROUND_BY_PREFIX.get(slot[:2])
    return None


def _resolve_seed_or_slot(s, seed_to_team, slot_winners):
    if s in seed_to_team:
        return seed_to_team[s]
    return slot_winners.get(s)


class BlendEvaluator:
    """Pre-loads bracket structure and actual outcomes; evaluates pairwise
    probability tables (or pre-blended frames) in memory without round-
    tripping to CSV. ~100x faster than calling score_pairwise_path repeatedly.
    """

    def __init__(self,
                 slots_csv: str = str(DATA / "MNCAATourneySlots.csv"),
                 seeds_csv: str = str(DATA / "MNCAATourneySeeds.csv"),
                 results_csv: str = str(DATA / "MNCAATourneyCompactResults.csv")):
        self.slots = pd.read_csv(slots_csv)
        self.seeds = pd.read_csv(seeds_csv)
        self.results = pd.read_csv(results_csv)
        self._precompute_per_season()

    def _precompute_per_season(self):
        """For each season, cache:
          - seed_to_team: dict
          - round_slot_ids: dict of round_name -> list of R*-prefixed slot ids
          - play_in_slot_ids: list
          - actual_winners_by_slot: dict
        """
        self.by_season = {}
        for season in sorted(self.slots.Season.unique()):
            season_slots = self.slots[self.slots.Season == season]
            season_seeds = self.seeds[self.seeds.Season == season]
            seed_to_team = dict(zip(season_seeds.Seed, season_seeds.TeamID.astype(int)))
            round_slot_ids = {
                ROUND_BY_PREFIX[f"R{r}"]: season_slots[
                    season_slots.Slot.str.startswith(f"R{r}")
                ]["Slot"].tolist()
                for r in range(1, 7)
            }
            play_in_slots = season_slots[~season_slots.Slot.str.startswith("R")]
            slot_defs = {
                row.Slot: (row.StrongSeed, row.WeakSeed)
                for row in season_slots.itertuples(index=False)
            }

            # Actual outcomes per (season, frozenset(team_a, team_b))
            season_results = self.results[self.results.Season == season]
            pair_winner = {}
            for g in season_results.itertuples(index=False):
                pair_winner[frozenset({int(g.WTeamID), int(g.LTeamID)})] = int(g.WTeamID)

            # Resolve actual winners through the bracket tree
            actual_winners = {}
            for _, row in play_in_slots.iterrows():
                a = seed_to_team.get(row.StrongSeed)
                b = seed_to_team.get(row.WeakSeed)
                if a is None or b is None:
                    continue
                w = pair_winner.get(frozenset({a, b}))
                if w is not None:
                    actual_winners[row.Slot] = w
            for r in range(1, 7):
                for slot_id in round_slot_ids[ROUND_BY_PREFIX[f"R{r}"]]:
                    strong, weak = slot_defs[slot_id]
                    a = _resolve_seed_or_slot(strong, seed_to_team, actual_winners)
                    b = _resolve_seed_or_slot(weak, seed_to_team, actual_winners)
                    if a is None or b is None:
                        continue
                    w = pair_winner.get(frozenset({a, b}))
                    if w is not None:
                        actual_winners[slot_id] = w

            self.by_season[int(season)] = {
                "seed_to_team": seed_to_team,
                "round_slot_ids": round_slot_ids,
                "slot_defs": slot_defs,
                "play_in_slots": play_in_slots["Slot"].tolist(),
                "actual_winners": actual_winners,
            }

    def _score_season(self, season: int, probs: Dict):
        """probs: {(a, b): p_a_wins} with a < b. Returns (correct, n, pts) per round."""
        ctx = self.by_season[season]
        seed_to_team = ctx["seed_to_team"]
        slot_defs = ctx["slot_defs"]
        actual = ctx["actual_winners"]
        play_in_slot_ids = ctx["play_in_slots"]

        def get_p(a, b):
            if a < b:
                return probs.get((a, b), 0.5)
            return 1.0 - probs.get((b, a), 0.5)

        # Chalk bracket: start with actual play-in winners (pools skip these)
        chalk = {sid: actual[sid] for sid in play_in_slot_ids if sid in actual}
        for r in range(1, 7):
            for sid in ctx["round_slot_ids"][ROUND_BY_PREFIX[f"R{r}"]]:
                strong, weak = slot_defs[sid]
                a = _resolve_seed_or_slot(strong, seed_to_team, chalk)
                b = _resolve_seed_or_slot(weak, seed_to_team, chalk)
                if a is None or b is None:
                    continue
                p = get_p(a, b)
                chalk[sid] = a if p >= 0.5 else b

        round_pts = {}
        for r in range(1, 7):
            rn = ROUND_BY_PREFIX[f"R{r}"]
            slot_ids = ctx["round_slot_ids"][rn]
            chalk_w = {chalk[s] for s in slot_ids if s in chalk}
            actual_w = {actual[s] for s in slot_ids if s in actual}
            correct = len(chalk_w & actual_w)
            n = len(actual_w)
            round_pts[rn] = (correct, n, correct * ROUND_PTS[rn])
        return round_pts

    def score_probs_df(self, probs_df: pd.DataFrame) -> Dict[int, float]:
        """Convert a (season, team_a, team_b, p_a_wins) DataFrame into
        per-season bracket points."""
        out: Dict[int, float] = {}
        for season, g in probs_df.groupby("season"):
            season = int(season)
            if season not in self.by_season:
                continue
            probs: Dict = {}
            # team_a < team_b is the canonical orientation in the pairwise file
            for r in g.itertuples(index=False):
                a, b = int(r.team_a), int(r.team_b)
                if a < b:
                    probs[(a, b)] = float(r.p_a_wins)
                else:
                    probs[(b, a)] = 1.0 - float(r.p_a_wins)
            rp = self._score_season(season, probs)
            out[season] = float(sum(t[2] for t in rp.values()))
        return out

    def score_blend(self,
                    v8_df: pd.DataFrame,
                    v4_df: pd.DataFrame,
                    alpha: float) -> Dict[int, float]:
        """Linear blend at scalar alpha. Both frames assumed pre-sorted on
        (season, team_a, team_b) with the same key set."""
        merged = self._merge_aligned(v8_df, v4_df)
        merged = merged.copy()
        merged["p_a_wins"] = alpha * merged["p_a_wins_v8"] + (1 - alpha) * merged["p_a_wins_v4"]
        return self.score_probs_df(merged)

    def score_blend_by_bucket(self,
                              v8_df: pd.DataFrame,
                              v4_df: pd.DataFrame,
                              alpha_per_bucket: Sequence[float],
                              bucket_for_p: callable) -> Dict[int, float]:
        """alpha is a function of v4's prob: bucket_for_p(p_v4) -> int index."""
        merged = self._merge_aligned(v8_df, v4_df).copy()
        merged["bucket"] = merged["p_a_wins_v4"].apply(bucket_for_p)
        alpha_arr = np.array(alpha_per_bucket)
        a = alpha_arr[merged["bucket"].values]
        merged["p_a_wins"] = a * merged["p_a_wins_v8"].values + (1 - a) * merged["p_a_wins_v4"].values
        return self.score_probs_df(merged)

    def _merge_aligned(self, v8_df, v4_df):
        cache_key = (id(v8_df), id(v4_df))
        if not hasattr(self, "_merge_cache"):
            self._merge_cache = {}
        if cache_key in self._merge_cache:
            return self._merge_cache[cache_key]
        v4_local = v4_df.drop_duplicates(["season", "team_a", "team_b"], keep="last")
        merged = v8_df.merge(v4_local, on=["season", "team_a", "team_b"], suffixes=["_v8", "_v4"])
        self._merge_cache[cache_key] = merged
        return merged
