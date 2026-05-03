"""Disjoint feature partition for the feature-view diversity ensemble.

Spec: docs/superpowers/specs/2026-05-02-feature-view-ensemble-design.md

PEER_A: team-strength view (full-season measures of team level).
PEER_B: form + market + meta view (recent-form, market, meta-features).

Together the two lists must partition v4's full feature set: disjoint
(every feature in at most one list) and exhaustive (every feature in
at least one list). The validate_partition helper enforces this against
a caller-supplied list (typically v4's get_feature_cols output).

Imported by:
  src/train_peer_stage1.py
  src/diagnose_feature_view_ensemble.py
  tests/test_feature_views.py
"""
from __future__ import annotations


PEER_A_FEATURES: tuple[str, ...] = (
    # Adjusted efficiency
    "adj_oe", "adj_de", "adj_em", "adj_tempo",
    # Four factors (offensive)
    "off_efg", "off_ft_rate", "off_or_rate", "off_to_rate",
    # Four factors (defensive)
    "def_efg", "def_ft_rate", "def_or_rate", "def_to_rate",
    # KenPom (full-season)
    "kp_BARTHAG", "kp_DREB%", "kp_EFG%", "kp_EFG%D",
    "kp_ELITE SOS", "kp_EXP", "kp_FTR", "kp_FTRD",
    "kp_K TEMPO", "kp_KADJ D", "kp_KADJ EM", "kp_KADJ O",
    "kp_OREB%", "kp_TALENT", "kp_TOV%", "kp_TOV%D", "kp_WAB",
    # Massey orderings
    "massey_COL", "massey_DOL", "massey_MOR", "massey_POM",
    "massey_RPI", "massey_SAG", "massey_WOL", "massey_composite",
    # Conference + full-season summary
    "conf_strength", "season_avg_mov", "season_win_pct",
)


PEER_B_FEATURES: tuple[str, ...] = (
    # Late-season efficiency
    "late_adj_oe", "late_adj_de", "late_adj_em", "late_sos",
    # Trajectory
    "efficiency_trend", "margin_trend", "scoring_trend",
    # Rolling form
    "rolling_oe", "rolling_de",
    "win_pct_last10", "win_pct_30d", "avg_mov_last10",
    # Conference tournament
    "conf_tourney_wins", "conf_tourney_champ",
    # Coach meta
    "coach_career_games", "coach_career_wins", "coach_career_winpct",
    "coach_career_f4_apps", "coach_career_champs", "coach_career_seasons",
    # Vegas market
    "vegas_avg_spread", "vegas_avg_margin", "vegas_ats_pct",
    "vegas_power_rating", "vegas_consistency", "vegas_game_count",
    "vegas_late_spread_delta",
)


def validate_partition(all_cols: list[str]) -> None:
    """Assert PEER_A | PEER_B exactly equals set(all_cols).

    Raises ValueError listing the problematic features if any of:
      - a column in all_cols is in neither peer list (partition gap),
      - a column in PEER_A or PEER_B is missing from all_cols (peer
        list drifted past v4's actual columns).
    """
    all_set = set(all_cols)
    a_set = set(PEER_A_FEATURES)
    b_set = set(PEER_B_FEATURES)
    union = a_set | b_set

    missing_from_peers = sorted(all_set - union)
    extra_in_peers = sorted(union - all_set)

    errs = []
    if missing_from_peers:
        errs.append(
            f"features in all_cols not assigned to any peer: "
            f"{missing_from_peers}"
        )
    if extra_in_peers:
        errs.append(
            f"features in PEER_A or PEER_B but missing from all_cols: "
            f"{extra_in_peers}"
        )
    if errs:
        raise ValueError("; ".join(errs))
