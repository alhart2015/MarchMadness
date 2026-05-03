"""XGBoost stage-1 trainer restricted to a single feature-view peer.

Spec: docs/superpowers/specs/2026-05-02-feature-view-ensemble-design.md

Mirrors src/enhanced_model_v3.py's LOSO loop. For each held-out season,
trains an XGBoost classifier on every-other-season's weighted matchup
data using only one peer's features (PEER_A or PEER_B), then dumps OOF
pairwise probabilities for the held-out season's full field. Uses the
exact same hyperparameters as v4's classifier so peer LL is comparable
to v4's standalone LL on equal footing.

v4 hyperparameter source: src/models/train.py train_model() defaults.
  n_estimators=300, max_depth=4, learning_rate=0.05,
  subsample=0.8, colsample_bytree=0.8
v4 also wraps XGBoost in CalibratedClassifierCV (Platt sigmoid scaling);
we mirror that here via src/models/train.train_model().

Output schema (matches output/pairwise_v4.csv):
    season, team_a, team_b, p_a_wins   (team_a < team_b)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.enhanced_model_v3 import prepare_loso_inputs
from src.feature_views import (
    PEER_A_FEATURES, PEER_B_FEATURES, validate_partition,
)
from src.models.matchup import build_matchup_features, build_weighted_matchup_data
from src.models.train import train_model

DEFAULT_PAIRWISE_OUT_A = "output/pairwise_peer_a.csv"
DEFAULT_PAIRWISE_OUT_B = "output/pairwise_peer_b.csv"


def select_peer_features(all_cols: list[str], peer: str) -> list[str]:
    """Return the subset of all_cols that belongs to the named peer.

    peer in {'a', 'b'}; raises ValueError otherwise.
    """
    if peer == "a":
        peer_set = set(PEER_A_FEATURES)
    elif peer == "b":
        peer_set = set(PEER_B_FEATURES)
    else:
        raise ValueError(f"peer must be 'a' or 'b'; got {peer!r}")
    return [c for c in all_cols if c in peer_set]


def dump_pairwise_for_season(
    season: int,
    field_team_ids: Iterable[int],
    feature_lookup: dict,
    model,
    out_csv: str,
    train_medians: "pd.Series | None" = None,
    peer_cols: "list[str] | None" = None,
) -> int:
    """Append (season, team_a, team_b, p_a_wins) rows for the season to out_csv.

    field_team_ids: iterable of team IDs in this season's tournament.
    feature_lookup: dict[team_id -> np.ndarray of raw features for the
        peer-restricted feature list].
    model: a fitted classifier with predict_proba(X) -> [N, 2].
    out_csv: appended-to (header written only on first call when file
        doesn't exist).
    train_medians: per-diff-column medians computed on the training fold's
        X_train (already in matchup diff-space). Mirrors v4's predict-time
        fillna(medians) at enhanced_model_v3.py:548. Must be passed together
        with peer_cols so the matchup DataFrame is named correctly.
    peer_cols: list of raw feature column names (used to name the diff
        columns via expand_feature_cols). Required when train_medians is
        passed.

    Returns the number of pair rows written.
    """
    from src.models.matchup import expand_feature_cols

    if (train_medians is None) != (peer_cols is None):
        raise ValueError(
            "train_medians and peer_cols must be provided together; "
            "passing one without the other is a likely bug"
        )

    field = sorted(set(int(t) for t in field_team_ids if t in feature_lookup))
    if len(field) < 2:
        return 0

    pair_rows = []
    pair_ids = []
    for i in range(len(field)):
        for j in range(i + 1, len(field)):
            a, b = field[i], field[j]
            av = feature_lookup[a]
            bv = feature_lookup[b]
            pair_rows.append(build_matchup_features(av, bv))
            pair_ids.append((a, b))

    # Mirror v4's predict-time NaN fill (enhanced_model_v3.py:548):
    # build a named DataFrame in diff-space, fill with training-fold medians,
    # then predict. If train_medians is not provided (e.g., unit tests using
    # stub models without NaN), fall back to a plain numpy array.
    if train_medians is not None and peer_cols is not None:
        diff_cols = expand_feature_cols(peer_cols)
        X_df = pd.DataFrame(pair_rows, columns=diff_cols).fillna(train_medians)
        p = model.predict_proba(X_df)[:, 1]
    else:
        X = np.array(pair_rows, dtype=float)
        p = model.predict_proba(X)[:, 1]

    out_df = pd.DataFrame({
        "season": season,
        "team_a": [a for a, _ in pair_ids],
        "team_b": [b for _, b in pair_ids],
        "p_a_wins": p,
    })
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    write_header = not Path(out_csv).exists()
    out_df.to_csv(out_csv, mode="a", index=False, header=write_header)
    return len(out_df)


def run_peer_loso(peer: str, out_csv: str | None = None) -> dict:
    """22-season LOSO loop training XGBoost on the named peer's feature
    subset only. For each held-out season, train on every-other-season's
    weighted matchup data, then dump pairwise probs for the held-out
    season's full field to out_csv.
    """
    if peer not in ("a", "b"):
        raise ValueError(f"peer must be 'a' or 'b'; got {peer!r}")

    if out_csv is None:
        out_csv = DEFAULT_PAIRWISE_OUT_A if peer == "a" else DEFAULT_PAIRWISE_OUT_B

    print("=" * 70)
    print(f"PEER STAGE-1 LOSO TRAINER (peer={peer.upper()})")
    print("=" * 70)
    inputs = prepare_loso_inputs()
    feature_matrix = inputs["feature_matrix"]
    tourney = inputs["tourney_filtered"]
    regular = inputs["regular_results"]
    feature_cols = inputs["feature_cols"]
    top_80_by_season = inputs["top_80_by_season"]

    # Sanity: partition must validate against v4's actual feature list
    # before we restrict by peer; catches drift between v4 and feature_views.
    validate_partition(feature_cols)
    peer_cols = select_peer_features(feature_cols, peer=peer)
    print(f"  feature_cols total: {len(feature_cols)}")
    print(f"  peer_cols restricted to PEER_{peer.upper()}: {len(peer_cols)}")

    # Wipe any prior partial output so the run produces a clean file.
    if Path(out_csv).exists():
        Path(out_csv).unlink()

    seasons = sorted(set(int(s) for s in tourney["Season"].unique()))
    # Mirror v4's filter (enhanced_model_v3.py:487): skip pre-2003 seasons so
    # OOF row coverage matches pairwise_v4.csv exactly.
    seasons = [s for s in seasons if s >= 2003]
    total_pairs = 0
    for season in seasons:
        # Training rows: every season except the held-out one.
        train_tourney = tourney[tourney["Season"] != season]
        train_regular = regular[regular["Season"] != season]

        # Union of top-80 IDs across all training seasons (not the held-out
        # season) to mark supplemental regular-season rows. Mirrors how
        # prepare_loso_inputs() builds all_top_80 for the full dataset.
        train_top_ids: set[int] = set()
        for s in sorted(train_tourney["Season"].unique()):
            train_top_ids |= top_80_by_season.get(int(s), set())

        X_train, y_train, sample_weight = build_weighted_matchup_data(
            feature_matrix=feature_matrix,
            tourney_results=train_tourney,
            regular_results=train_regular,
            feature_cols=peer_cols,
            top_n_team_ids=train_top_ids,
            supplemental_weight=0.25,
        )

        # Mirror v4's guard (enhanced_model_v3.py:516-517): skip degenerate
        # folds with no training data.
        if X_train.empty:
            print(f"  season {season}: skipped (empty training fold)")
            continue

        # Fill NaN with training-fold median before fitting; store medians
        # for reuse at predict time (mirrors v4 lines 519-522).
        train_medians = X_train.median()
        X_train = X_train.fillna(train_medians)

        # Mirror v4's explicit random_seed=42 (enhanced_model_v3.py:524-526).
        model = train_model(X_train, y_train, sample_weight=sample_weight,
                            random_seed=42)

        # Build the per-team feature lookup for the held-out season.
        season_fm = feature_matrix[feature_matrix["Season"] == season]
        feature_lookup = {
            int(row["TeamID"]): row[peer_cols].values.astype(float)
            for _, row in season_fm.iterrows()
        }

        # Field = teams that appeared in the season's tournament.
        season_tourney = tourney[tourney["Season"] == season]
        field_ids = set(season_tourney["WTeamID"]).union(
            set(season_tourney["LTeamID"])
        )

        n = dump_pairwise_for_season(
            season=season,
            field_team_ids=field_ids,
            feature_lookup=feature_lookup,
            model=model,
            out_csv=out_csv,
            train_medians=train_medians,
            peer_cols=peer_cols,
        )
        total_pairs += n
        print(f"  season {season}: {n} pairs (cumulative {total_pairs})")

    return {"total_pairs": total_pairs, "out_csv": out_csv}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--peer", choices=("a", "b"), required=True,
        help="which peer's feature subset to train on",
    )
    parser.add_argument(
        "--output", default=None,
        help="output CSV path (defaults: pairwise_peer_a.csv or pairwise_peer_b.csv)",
    )
    args = parser.parse_args(argv)

    summary = run_peer_loso(peer=args.peer, out_csv=args.output)
    print(f"\nwrote {summary['total_pairs']} pair rows to {summary['out_csv']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
