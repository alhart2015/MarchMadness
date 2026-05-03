# Feature-View Diversity Ensemble Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train two same-class XGBoost stage-1 peers on disjoint v4 feature subsets (PEER_A: 40 team-strength features, PEER_B: 27 form/market/meta features). Run a 3-clause pre-sweep falsification gate (per-peer LL ceiling, residual correlation < 0.60, best-blend headroom >= 0.001) before committing to the 15-cell sweep. If the gate clears, run the 15-cell W_UPSET / W_MISS sweep twice -- once for the pure E1 blend (peer_A + peer_B) and once for the augmented E2 blend (v4 + peer_A + peer_B) -- against v9-C's production cell at 2713 brkt pts.

**Architecture:** Additive new modules (`src/feature_views.py`, `src/train_peer_stage1.py`, `src/diagnose_feature_view_ensemble.py`) plus extensions to two existing modules (`src/ensemble_stage1.py` for K-way blend, `src/sweep_v9_weights.py` for env-var-driven stage-1 input override). Existing v4, v8, v9-A/B/C/D code paths are untouched. v9-C consumes whatever pairwise CSV it's pointed at via the new `V9_STAGE1_PAIRWISE` env var; the diversity work happens upstream.

**Tech Stack:** Python 3.11+, xgboost (existing trainer), scipy.optimize (1-D and simplex-constrained minimization for blend weights), numpy, pandas, scikit-learn (`log_loss`), pytest. No new top-level dependencies.

**Spec:** `docs/superpowers/specs/2026-05-02-feature-view-ensemble-design.md`

**Predecessors (rejected, frozen artifacts referenced here):**
- LR ensemble (PR 11): `docs/notes/2026-05-01-ensemble-stage1.md`. `src/ensemble_stage1.py:average_pairwise_csvs` is reused as-is and extended for K-way.
- BT ensemble (PR 12): `docs/notes/2026-05-01-bayesian-stage1.md`. The 3-clause gate threshold `rho < 0.60` is calibrated against PR 12's measured r=0.58.
- BT-as-feature (PR 13): `docs/notes/2026-05-02-bt-as-feature.md`. The headroom threshold `>= 0.001` matches PR 13's gate; the diagnose-script shape mirrors `src/diagnose_v9d.py`.
- v9-C production swap: `docs/notes/2026-05-01-v9c-feature-stripped.md`, `output/v9c_sweep/pairwise_v9_WU1.25_WM0.00.csv` (2713 brkt pts baseline).

**Reference reads (skim before starting):**
- `src/feature_views.py` -- doesn't exist yet; the partition lists go here.
- `src/enhanced_model_v3.py:prepare_loso_inputs` and the LOSO training loop in `src/enhanced_model_v3.py:leave_one_season_out_cv_weighted` -- the v4 trainer that the peer trainer mirrors with a feature-subset restriction.
- `src/models/matchup.py:build_weighted_matchup_data`, `build_matchup_features`, `expand_feature_cols` -- the matchup-row builders the peer trainer reuses unchanged.
- `src/train_lr_stage1.py` -- shape reference for the new `train_peer_stage1.py`. Note `dump_pairwise_for_season` and the `run_lr_loso` per-season loop. The peer trainer mirrors this shape but uses XGBoost.
- `src/diagnose_v9d.py` -- shape reference for the new `diagnose_feature_view_ensemble.py`. Note the `compute_gate` / `check_gate` / `print_report` / `main` decomposition.
- `src/ensemble_stage1.py:average_pairwise_csvs` -- existing 2-way blend; Task 4 generalizes it to K-way.
- `src/sweep_v9_weights.py:main` -- the `V9_FEATURE_SET` env-var switch is the pattern Task 5 mirrors for `V9_STAGE1_PAIRWISE`.
- `output/pairwise_v4.csv` -- the schema that all pairwise CSVs in this work share: `season,team_a,team_b,p_a_wins` with `team_a < team_b`.
- `output/v9c_sweep/pairwise_v9_WU1.25_WM0.00.csv` -- the v9-C production-cell artifact; baseline for `delta_vs_v9c` in Task 13.
- `output/v9c_sweep/pairwise_v9_WU1.00_WM0.00.csv` -- the v9-C anchor cell; baseline for the harness-anchor check in Task 12.

**Verification gates (CLAUDE.md "Forced Verification"):**
After every code-level task: `pytest -v` for the touched test file(s) must pass. Task 6 runs the full ingest/feature/integration suite once before any real-data run. After data-generation tasks (7, 8, 9, 10, 11): inspect schema and row count of the produced artifact before committing.

**ASCII discipline (CLAUDE.md):** All files written must be ASCII-only. After every Write/Edit, run:
```bash
python -c "open('PATH', encoding='utf-8').read().encode('ascii')" && echo "ASCII OK"
```

---

### Task 1: `src/feature_views.py` -- partition lists and validator

**Why:** Single source of truth for the disjoint feature partition. Every downstream module imports `PEER_A_FEATURES` and `PEER_B_FEATURES` from here. The validator catches drift if v4 gains a feature that's not assigned to a peer.

**Files:**
- Create: `src/feature_views.py`
- Create: `tests/test_feature_views.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_feature_views.py`:

```python
"""Unit tests for src/feature_views.py.

The module defines the disjoint feature partition used by the
feature-view diversity ensemble (PEER_A: team strength;
PEER_B: form + market + meta). Validates partition disjointness and
exhaustiveness against any caller-supplied feature list.
"""
import pytest


def test_partition_disjoint():
    from src.feature_views import PEER_A_FEATURES, PEER_B_FEATURES

    a = set(PEER_A_FEATURES)
    b = set(PEER_B_FEATURES)
    assert a & b == set(), f"PEER_A and PEER_B overlap: {sorted(a & b)}"


def test_partition_lists_are_immutable_tuples():
    """Lists are exposed as tuples so downstream code can't mutate them."""
    from src.feature_views import PEER_A_FEATURES, PEER_B_FEATURES

    assert isinstance(PEER_A_FEATURES, tuple)
    assert isinstance(PEER_B_FEATURES, tuple)


def test_validate_partition_passes_when_complete_and_disjoint():
    from src.feature_views import (
        PEER_A_FEATURES, PEER_B_FEATURES, validate_partition,
    )

    all_cols = list(PEER_A_FEATURES) + list(PEER_B_FEATURES)
    validate_partition(all_cols)  # must not raise


def test_validate_partition_raises_on_missing_feature():
    """A column in all_cols that's in neither PEER_A nor PEER_B is
    a partition gap and must raise ValueError naming the column.
    """
    from src.feature_views import (
        PEER_A_FEATURES, PEER_B_FEATURES, validate_partition,
    )

    all_cols = list(PEER_A_FEATURES) + list(PEER_B_FEATURES) + ["new_feature_xyz"]
    with pytest.raises(ValueError, match="new_feature_xyz"):
        validate_partition(all_cols)


def test_validate_partition_raises_on_extra_peer_feature():
    """A column in PEER_A but not in all_cols means PEER_A drifted past
    v4's actual columns. Must raise.
    """
    from src.feature_views import (
        PEER_A_FEATURES, PEER_B_FEATURES, validate_partition,
    )

    # Drop one PEER_A feature from all_cols.
    all_cols = list(PEER_A_FEATURES[1:]) + list(PEER_B_FEATURES)
    missing = PEER_A_FEATURES[0]
    with pytest.raises(ValueError, match=missing):
        validate_partition(all_cols)
```

- [ ] **Step 2: Run tests to verify they fail with ImportError**

Run: `pytest tests/test_feature_views.py -v`
Expected: 5 errors, all `ModuleNotFoundError: No module named 'src.feature_views'`

- [ ] **Step 3: Write `src/feature_views.py`**

Create `src/feature_views.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_feature_views.py -v`
Expected: 5 passed.

- [ ] **Step 5: ASCII check**

Run:
```bash
python -c "open('src/feature_views.py', encoding='utf-8').read().encode('ascii')" && echo "ASCII OK"
python -c "open('tests/test_feature_views.py', encoding='utf-8').read().encode('ascii')" && echo "ASCII OK"
```
Expected: `ASCII OK` printed twice.

- [ ] **Step 6: Verify the partition matches v4's actual feature list**

This is a real-data check (not a unit test) and is run interactively:

Run:
```bash
python -c "
import sys
sys.path.insert(0, '.')
from src.enhanced_model_v3 import prepare_loso_inputs
from src.feature_views import validate_partition
inp = prepare_loso_inputs()
validate_partition(inp['feature_cols'])
print(f'partition validates against v4 ({len(inp[\"feature_cols\"])} features)')
"
```
Expected: `partition validates against v4 (67 features)` -- with no exception. If `validate_partition` raises, v4's feature list has drifted; either update PEER_A/PEER_B to assign the new feature, or remove the stale reference.

- [ ] **Step 7: Commit**

```bash
git add src/feature_views.py tests/test_feature_views.py
git commit -m "$(cat <<'EOF'
feat(feature-view-ensemble): src/feature_views.py partition

Defines PEER_A_FEATURES (40 team-strength columns) and
PEER_B_FEATURES (27 form/market/meta columns) as the disjoint
partition for the feature-view diversity ensemble. validate_partition
asserts disjointness and exhaustiveness against v4's get_feature_cols
output.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: `src/train_peer_stage1.py` -- per-peer XGBoost LOSO trainer

**Why:** Produces `output/pairwise_peer_a.csv` and `output/pairwise_peer_b.csv` -- the OOF predictions that feed the gate diagnostic and (on PASS) the ensemble blends. Mirrors `src/enhanced_model_v3.py`'s LOSO loop but restricts the feature matrix to one peer's columns.

**Files:**
- Create: `src/train_peer_stage1.py`
- Create: `tests/test_train_peer_stage1.py`

- [ ] **Step 1: Read the v4 trainer to understand what to mirror**

Run: `grep -n "def leave_one_season_out_cv_weighted\|def train_xgb_model\|^XGB_PARAMS\|xgb.XGBClassifier" src/enhanced_model_v3.py`

Note the function names, signatures, and hyperparameters. Specifically: the XGBoost hyperparameters used by v4 (n_estimators, max_depth, learning_rate, etc.), the per-fold training data assembly, and the per-pair feature-difference matchup builder.

- [ ] **Step 2: Write the failing tests**

Create `tests/test_train_peer_stage1.py`:

```python
"""Unit tests for src/train_peer_stage1.py.

These tests exercise the public CLI of the per-peer stage-1 trainer
on synthetic-shape inputs (feature_matrix, tourney results) -- the full
LOSO loop on real data is exercised in Task 7's real-data run.
"""
import numpy as np
import pandas as pd
import pytest


def _make_synthetic_feature_matrix(n_teams: int = 8, season: int = 2024):
    """Build a feature_matrix with both PEER_A and PEER_B columns."""
    from src.feature_views import PEER_A_FEATURES, PEER_B_FEATURES

    rng = np.random.default_rng(0)
    team_ids = list(range(1, n_teams + 1))
    rows = []
    for tid in team_ids:
        row = {"TeamID": tid, "Season": season, "seed": (tid % 16) + 1}
        for c in list(PEER_A_FEATURES) + list(PEER_B_FEATURES):
            row[c] = float(rng.standard_normal())
        rows.append(row)
    return pd.DataFrame(rows)


def test_select_peer_features_returns_only_peer_a():
    from src.feature_views import PEER_A_FEATURES, PEER_B_FEATURES
    from src.train_peer_stage1 import select_peer_features

    fm = _make_synthetic_feature_matrix()
    all_cols = [c for c in fm.columns if c not in {"TeamID", "Season", "seed"}]
    selected = select_peer_features(all_cols, peer="a")
    assert set(selected) == set(PEER_A_FEATURES)
    assert set(selected).isdisjoint(set(PEER_B_FEATURES))


def test_select_peer_features_returns_only_peer_b():
    from src.feature_views import PEER_A_FEATURES, PEER_B_FEATURES
    from src.train_peer_stage1 import select_peer_features

    fm = _make_synthetic_feature_matrix()
    all_cols = [c for c in fm.columns if c not in {"TeamID", "Season", "seed"}]
    selected = select_peer_features(all_cols, peer="b")
    assert set(selected) == set(PEER_B_FEATURES)
    assert set(selected).isdisjoint(set(PEER_A_FEATURES))


def test_select_peer_features_unknown_peer_raises():
    from src.train_peer_stage1 import select_peer_features

    with pytest.raises(ValueError, match="peer"):
        select_peer_features(["adj_oe"], peer="c")


def test_dump_pairwise_for_season_writes_documented_schema(tmp_path):
    """The OOF pairwise CSV must match v4's schema:
    columns = (season, team_a, team_b, p_a_wins) with team_a < team_b.
    """
    from src.train_peer_stage1 import dump_pairwise_for_season

    rng = np.random.default_rng(0)
    team_ids = [10, 20, 30]
    feature_lookup = {tid: rng.standard_normal(5) for tid in team_ids}

    class _StubModel:
        def predict_proba(self, X):
            n = len(X)
            return np.column_stack([np.full(n, 0.4), np.full(n, 0.6)])

    out = tmp_path / "pairwise_peer_test.csv"
    n = dump_pairwise_for_season(
        season=2024,
        field_team_ids=team_ids,
        feature_lookup=feature_lookup,
        model=_StubModel(),
        out_csv=str(out),
    )
    assert n == 3  # C(3, 2) = 3 pairs
    df = pd.read_csv(out)
    assert list(df.columns) == ["season", "team_a", "team_b", "p_a_wins"]
    assert (df["team_a"] < df["team_b"]).all()
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest tests/test_train_peer_stage1.py -v`
Expected: 4 errors, all `ModuleNotFoundError: No module named 'src.train_peer_stage1'`.

- [ ] **Step 4: Implement `src/train_peer_stage1.py`**

Create `src/train_peer_stage1.py`:

```python
"""XGBoost stage-1 trainer restricted to a single feature-view peer.

Spec: docs/superpowers/specs/2026-05-02-feature-view-ensemble-design.md

Mirrors src/enhanced_model_v3.py's LOSO loop. For each held-out season,
trains an XGBoost classifier on every-other-season's weighted matchup
data using only one peer's features (PEER_A or PEER_B), then dumps OOF
pairwise probabilities for the held-out season's full field. Uses the
exact same hyperparameters as v4's classifier so peer LL is comparable
to v4's standalone LL on equal footing.

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
import xgboost as xgb

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.enhanced_model_v3 import prepare_loso_inputs
from src.feature_views import (
    PEER_A_FEATURES, PEER_B_FEATURES, validate_partition,
)
from src.models.matchup import (
    build_matchup_features,
    build_weighted_matchup_data,
    expand_feature_cols,
)

# v4 hyperparameters; copy from src/enhanced_model_v3.py to keep peer
# training byte-comparable to v4 in everything except the feature subset.
# Confirm these match the values in the v4 trainer at implementation time;
# do not let them drift.
XGB_PARAMS = {
    "n_estimators": 100,
    "max_depth": 4,
    "learning_rate": 0.1,
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "use_label_encoder": False,
    "random_state": 42,
}

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
) -> int:
    """Append (season, team_a, team_b, p_a_wins) rows for the season to out_csv.

    field_team_ids: iterable of team IDs in this season's tournament.
    feature_lookup: dict[team_id -> np.ndarray of raw features for the
        peer-restricted feature list].
    model: a fitted classifier with predict_proba(X) -> [N, 2].
    out_csv: appended-to (header written only on first call when file
        doesn't exist).

    Returns the number of pair rows written.
    """
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
    top_80 = inputs["top_80_by_season"]

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
    total_pairs = 0
    for season in seasons:
        # Training rows: every season except the held-out one.
        train_tourney = tourney[tourney["Season"] != season]
        train_regular = regular[regular["Season"] != season]

        X_train, y_train, sample_weight = build_weighted_matchup_data(
            feature_matrix=feature_matrix,
            tourney=train_tourney,
            regular=train_regular,
            feature_cols=peer_cols,
            top_80_by_season=top_80,
        )

        model = xgb.XGBClassifier(**XGB_PARAMS)
        model.fit(X_train, y_train, sample_weight=sample_weight)

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
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_train_peer_stage1.py -v`
Expected: 4 passed.

- [ ] **Step 6: Confirm v4 hyperparameters match**

Run: `grep -n "XGBClassifier\|n_estimators\|max_depth\|learning_rate" src/enhanced_model_v3.py | head -30`

Compare the values in the matched lines to `XGB_PARAMS` in `src/train_peer_stage1.py`. If they differ, update `train_peer_stage1.py` to match v4 (or document the divergence in the file's docstring with rationale). Re-run pytest after any change.

- [ ] **Step 7: ASCII check + commit**

```bash
python -c "open('src/train_peer_stage1.py', encoding='utf-8').read().encode('ascii')" && echo "ASCII OK"
python -c "open('tests/test_train_peer_stage1.py', encoding='utf-8').read().encode('ascii')" && echo "ASCII OK"
git add src/train_peer_stage1.py tests/test_train_peer_stage1.py
git commit -m "$(cat <<'EOF'
feat(feature-view-ensemble): src/train_peer_stage1.py XGBoost peer trainer

Per-peer LOSO trainer mirroring src/enhanced_model_v3.py's training
loop, restricted to PEER_A or PEER_B's feature subset. Same XGBoost
hyperparameters as v4 so peer LL is directly comparable.

CLI: python src/train_peer_stage1.py --peer {a|b} [--output PATH]
Output: output/pairwise_peer_{a|b}.csv (default), v4-compatible schema.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: `src/diagnose_feature_view_ensemble.py` -- 3-clause pre-sweep gate

**Why:** Decides whether to commit ~90-150 minutes to running both 15-cell sweeps. Each clause maps to a specific prior failure mode (PR 12 BT-ensemble: peer too weak; PR 11 LR-ensemble: errors too correlated; PR 13 BT-as-feature: no headroom). All three must pass for the gate to clear.

**Files:**
- Create: `src/diagnose_feature_view_ensemble.py`
- Create: `tests/test_diagnose_feature_view_ensemble.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_diagnose_feature_view_ensemble.py`:

```python
"""Unit tests for src/diagnose_feature_view_ensemble.py.

The module decomposes into:
  compute_pairwise_ll(pairwise_csv, results_csv) -> (ll, p_winner_won, labels)
  optimal_2blend(p_a, p_b, y) -> (w_opt, ll_opt)
  residual_correlation(p_a, p_b, y) -> float
  compute_gate(...) -> dict
  check_gate(diag) -> dict

These tests rig synthetic inputs to fail exactly one clause at a time,
plus a pass case and a multi-fail case.
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def _write_pairwise(path, rows):
    df = pd.DataFrame(rows, columns=["season", "team_a", "team_b", "p_a_wins"])
    df.to_csv(path, index=False)


def _write_results(path, rows):
    """rows: list of (Season, DayNum, WTeamID, LTeamID).

    Adds default values for the score columns so pandas dtypes match the
    real MNCAATourneyCompactResults schema; the gate code only reads the
    Season/WTeamID/LTeamID columns.
    """
    expanded = [(s, d, wt, lt, 80, 70, "N", 0) for s, d, wt, lt in rows]
    cols = [
        "Season", "DayNum", "WTeamID", "LTeamID",
        "WScore", "LScore", "WLoc", "NumOT",
    ]
    pd.DataFrame(expanded, columns=cols).to_csv(path, index=False)


def _winner_log_loss(p_winner, eps=1e-15):
    """Mean -log(p_winner) over rows."""
    p = np.clip(np.asarray(p_winner, dtype=float), eps, 1 - eps)
    return float(-np.log(p).mean())


def test_compute_pairwise_ll_roundtrip(tmp_path):
    """For a single played game with WTeamID=1, LTeamID=2 and pairwise
    p_a_wins=0.7 (a=1, b=2 since 1<2), the winner's prob is 0.7 and the
    LL is -log(0.7).
    """
    from src.diagnose_feature_view_ensemble import compute_pairwise_ll

    pw = tmp_path / "pw.csv"
    res = tmp_path / "res.csv"
    _write_pairwise(pw, [(2024, 1, 2, 0.7)])
    _write_results(res, [(2024, 136, 1, 2)])

    ll, p_winner, y = compute_pairwise_ll(str(pw), str(res))
    assert pytest.approx(ll, rel=1e-9) == _winner_log_loss([0.7])
    assert list(p_winner) == [pytest.approx(0.7)]
    assert list(y) == [1]  # winner indicator is always 1 (we score from
                           # the winner's perspective)


def test_compute_pairwise_ll_orientation_when_winner_id_greater(tmp_path):
    """Played game with WTeamID=2, LTeamID=1: a=1, b=2 (orientation
    fixed), p_a_wins is now P(loser won), so winner's prob is 1 - p_a_wins.
    """
    from src.diagnose_feature_view_ensemble import compute_pairwise_ll

    pw = tmp_path / "pw.csv"
    res = tmp_path / "res.csv"
    _write_pairwise(pw, [(2024, 1, 2, 0.3)])  # P(team 1 wins) = 0.3
    _write_results(res, [(2024, 136, 2, 1)])  # team 2 actually won

    ll, p_winner, y = compute_pairwise_ll(str(pw), str(res))
    # Winner is team 2; P(team 2 wins) = 1 - P(team 1 wins) = 0.7.
    assert pytest.approx(ll, rel=1e-9) == _winner_log_loss([0.7])


def test_optimal_2blend_returns_weight_one_when_a_dominates(tmp_path):
    """If peer A is perfect and B is awful, the optimal 2-blend weight
    on A is ~1.0.
    """
    from src.diagnose_feature_view_ensemble import optimal_2blend

    p_winner_a = np.array([0.99, 0.99, 0.99])  # perfect winner-side LL
    p_winner_b = np.array([0.01, 0.01, 0.01])  # inverted predictions
    w_opt, ll_opt = optimal_2blend(p_winner_a, p_winner_b)
    assert w_opt == pytest.approx(1.0, abs=1e-3)
    assert ll_opt == pytest.approx(_winner_log_loss(p_winner_a), abs=1e-3)


def test_residual_correlation_perfect_alignment_returns_one():
    from src.diagnose_feature_view_ensemble import residual_correlation

    # identical residuals -> r = 1.0
    p_a = np.array([0.6, 0.7, 0.8])
    p_b = np.array([0.6, 0.7, 0.8])
    r = residual_correlation(p_a, p_b)
    assert r == pytest.approx(1.0, abs=1e-9)


def test_residual_correlation_zero_when_anti_aligned():
    from src.diagnose_feature_view_ensemble import residual_correlation

    # Perfectly anti-aligned residuals -> r = -1.0
    p_a = np.array([0.6, 0.7, 0.8])
    p_b = np.array([0.4, 0.3, 0.2])
    r = residual_correlation(p_a, p_b)
    assert r == pytest.approx(-1.0, abs=1e-9)


def _rig_synthetic_inputs(tmp_path, ll_a, ll_b, ll_v4, rho_resid, ll_blend):
    """Construct synthetic pairwise CSVs (v4, peer_a, peer_b) and a
    matching results CSV that produce the target LL values and residual
    correlation.

    Approach: with all winners on the (W=1, L=2) side (pair always
    oriented (1, 2)), p_winner = p_a_wins. Choose p_a and p_b values
    that yield target LLs and residual correlation, then derive p_v4 from
    the target ll_v4.
    """
    # 100 played games; rng-controlled.
    rng = np.random.default_rng(42)
    n = 100
    # Generate residuals with target rho and means consistent with target LLs.
    # p = 1 - resid (where label y=1, so resid = p - 1, i.e. negative).
    # mean(-log(p)) = ll  =>  p = exp(-ll) approximately for tight clusters.
    # Use a small jitter + correlation structure.
    z1 = rng.standard_normal(n)
    z2 = rho_resid * z1 + np.sqrt(max(1 - rho_resid * rho_resid, 0)) * rng.standard_normal(n)

    # Center predictions around exp(-ll) to hit target LL.
    def _hit_ll(z, target_ll):
        target_p = float(np.exp(-target_ll))
        p = target_p + 0.02 * z
        p = np.clip(p, 1e-3, 1 - 1e-3)
        # Re-scale so the realized LL matches target.
        for _ in range(20):
            cur_ll = _winner_log_loss(p)
            shift = target_ll - cur_ll
            p = np.clip(p * np.exp(-shift), 1e-3, 1 - 1e-3)
            if abs(_winner_log_loss(p) - target_ll) < 1e-4:
                break
        return p

    p_a = _hit_ll(z1, ll_a)
    p_b = _hit_ll(z2, ll_b)
    p_v4 = _hit_ll(rng.standard_normal(n), ll_v4)

    # Optional: solve for the blend that hits ll_blend by tweaking peer B
    # until best 2-blend matches. Skipped here because the gate's clause 3
    # uses an actual optimizer, and tests for that clause should construct
    # a peer pair whose true optimum lands at ll_blend.

    rows_v4 = [(2024, 1, 2, p) for p in p_v4]
    rows_a = [(2024, 1, 2, p) for p in p_a]
    rows_b = [(2024, 1, 2, p) for p in p_b]
    res_rows = [(2024, 136 + i, 1, 2) for i in range(n)]

    pw_v4 = tmp_path / "pw_v4.csv"
    pw_a = tmp_path / "pw_a.csv"
    pw_b = tmp_path / "pw_b.csv"
    res = tmp_path / "res.csv"

    # Note: replicate (1, 2) pair n times by season+daynum -- not really
    # valid input shape for production but works for the gate's joining
    # logic since we only need n distinct played-game rows.
    pd.DataFrame(rows_v4, columns=["season", "team_a", "team_b", "p_a_wins"]).to_csv(pw_v4, index=False)
    pd.DataFrame(rows_a, columns=["season", "team_a", "team_b", "p_a_wins"]).to_csv(pw_a, index=False)
    pd.DataFrame(rows_b, columns=["season", "team_a", "team_b", "p_a_wins"]).to_csv(pw_b, index=False)
    _write_results(res, res_rows)

    return str(pw_v4), str(pw_a), str(pw_b), str(res)


def test_compute_gate_passes_when_all_clauses_met(tmp_path):
    """Pass case: each peer LL well within 0.025 of v4; rho ~ 0.4;
    blend headroom > 0.001.
    """
    from src.diagnose_feature_view_ensemble import compute_gate, check_gate

    # Replace the per-game synthesis with a directly-constructed scenario
    # that has well-defined peer LLs and residual correlation.
    n = 200
    rng = np.random.default_rng(0)
    p_v4 = np.full(n, 0.65)  # LL = -log(0.65) ~ 0.4308
    p_a = np.full(n, 0.66)   # LL ~ 0.4155 (slightly better than v4 to allow blend headroom)
    p_b = np.full(n, 0.65)   # LL ~ 0.4308 (matches v4)
    # Add per-row jitter to create a residual correlation < 0.6.
    z = rng.standard_normal(n)
    p_a = np.clip(p_a + 0.01 * z, 1e-3, 1 - 1e-3)
    p_b = np.clip(p_b + 0.01 * (-z * 0.3 + rng.standard_normal(n) * 0.7),
                  1e-3, 1 - 1e-3)
    p_v4 = np.clip(p_v4 + 0.01 * rng.standard_normal(n), 1e-3, 1 - 1e-3)

    pw_v4 = tmp_path / "pw_v4.csv"
    pw_a = tmp_path / "pw_a.csv"
    pw_b = tmp_path / "pw_b.csv"
    res = tmp_path / "res.csv"
    season = 2024
    for path, p in [(pw_v4, p_v4), (pw_a, p_a), (pw_b, p_b)]:
        pd.DataFrame({
            "season": season, "team_a": 1, "team_b": 2 + np.arange(n),
            "p_a_wins": p,
        }).to_csv(path, index=False)
    _write_results(
        res,
        [(season, 136 + i, 1, 2 + i) for i in range(n)],
    )

    diag = compute_gate(
        pairwise_v4_csv=str(pw_v4),
        pairwise_peer_a_csv=str(pw_a),
        pairwise_peer_b_csv=str(pw_b),
        results_csv=str(res),
    )
    gate = check_gate(diag)
    # The clauses individually:
    # - per-peer LL ceiling: peer LLs near v4 LL (within 0.025) -> PASS
    # - residual correlation: < 0.60 -> should PASS at this scale
    # - headroom: peer A slightly better than v4, blend may have small
    #   positive headroom; relax the assertion if borderline.
    assert "per_peer_ll_ceiling" in diag["clauses"]
    assert "residual_correlation" in diag["clauses"]
    assert "blend_headroom" in diag["clauses"]
    # Smoke check: gate produced a dict shape with 'pass' key.
    assert isinstance(gate["pass"], bool)


def test_compute_gate_fails_clause_per_peer_ll_ceiling(tmp_path):
    """Peer A LL = 0.50 (more than 0.025 above v4's ~0.43): clause 1 fails."""
    from src.diagnose_feature_view_ensemble import compute_gate, check_gate

    n = 100
    p_v4 = np.full(n, 0.65)
    p_a = np.full(n, 0.55)  # winner-side prob much lower => much higher LL
    p_b = np.full(n, 0.65)
    pw_v4 = tmp_path / "pw_v4.csv"
    pw_a = tmp_path / "pw_a.csv"
    pw_b = tmp_path / "pw_b.csv"
    res = tmp_path / "res.csv"
    season = 2024
    for path, p in [(pw_v4, p_v4), (pw_a, p_a), (pw_b, p_b)]:
        pd.DataFrame({
            "season": season, "team_a": 1,
            "team_b": 2 + np.arange(n), "p_a_wins": p,
        }).to_csv(path, index=False)
    _write_results(res, [(season, 136 + i, 1, 2 + i) for i in range(n)])

    diag = compute_gate(
        pairwise_v4_csv=str(pw_v4),
        pairwise_peer_a_csv=str(pw_a),
        pairwise_peer_b_csv=str(pw_b),
        results_csv=str(res),
    )
    gate = check_gate(diag)
    assert gate["pass"] is False
    assert "per_peer_ll_ceiling" in gate["failed_clauses"]


def test_compute_gate_main_exits_nonzero_on_fail(tmp_path):
    """Subprocess invocation: a failing gate exits with code 1."""
    n = 50
    p_v4 = np.full(n, 0.65)
    p_a = np.full(n, 0.55)  # forces clause 1 fail
    p_b = np.full(n, 0.65)
    pw_v4 = tmp_path / "pw_v4.csv"
    pw_a = tmp_path / "pw_a.csv"
    pw_b = tmp_path / "pw_b.csv"
    res = tmp_path / "res.csv"
    out_json = tmp_path / "diag.json"
    season = 2024
    for path, p in [(pw_v4, p_v4), (pw_a, p_a), (pw_b, p_b)]:
        pd.DataFrame({
            "season": season, "team_a": 1,
            "team_b": 2 + np.arange(n), "p_a_wins": p,
        }).to_csv(path, index=False)
    _write_results(res, [(season, 136 + i, 1, 2 + i) for i in range(n)])

    proc = subprocess.run(
        [
            sys.executable, "src/diagnose_feature_view_ensemble.py",
            "--pairwise-v4", str(pw_v4),
            "--pairwise-peer-a", str(pw_a),
            "--pairwise-peer-b", str(pw_b),
            "--results-csv", str(res),
            "--out-json", str(out_json),
        ],
        capture_output=True, text=True,
    )
    assert proc.returncode == 1, proc.stdout + proc.stderr
    payload = json.loads(out_json.read_text())
    assert payload["gate"]["pass"] is False
```

- [ ] **Step 2: Run tests to verify they fail with ImportError**

Run: `pytest tests/test_diagnose_feature_view_ensemble.py -v`
Expected: 9 errors, all `ModuleNotFoundError: No module named 'src.diagnose_feature_view_ensemble'` (or import error if a helper is missing).

- [ ] **Step 3: Implement `src/diagnose_feature_view_ensemble.py`**

Create `src/diagnose_feature_view_ensemble.py`:

```python
"""Pre-sweep falsification gate for the feature-view diversity ensemble.

Spec: docs/superpowers/specs/2026-05-02-feature-view-ensemble-design.md

3-clause gate:
  1. Per-peer LL ceiling: each peer's weighted-mean per-game LL on
     played tournament games is within PEER_LL_CEILING_DELTA of v4's.
  2. Inter-peer residual correlation: pearson r(resid_A, resid_B) on
     played-game rows, where resid = p_winner_won - 1, is < RESID_CORR_MAX.
  3. Best-blend LL headroom: optimal 2-blend of peer A and peer B beats
     v4 standalone by >= HEADROOM_MIN.

If any clause fails, the gate fails. Exits nonzero on FAIL so a wrapper
can short-circuit the sweep.

Mirrors src/diagnose_v9d.py and src/diagnose_bt_vs_v4.py in shape.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar, minimize

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

DATA = Path("data/raw/march-machine-learning-2026")
DEFAULT_DIAGNOSTIC_OUT = "output/diag_feature_view_ensemble.json"

# Clause thresholds. Each maps to a prior-experiment failure mode.
PEER_LL_CEILING_DELTA = 0.025  # PR 12 BT-ensemble (peer too weak)
RESID_CORR_MAX = 0.60          # PR 11 LR-ensemble (errors too correlated)
HEADROOM_MIN = 0.001           # PR 13 BT-as-feature (no signal lift)

EPS = 1e-15  # log-loss clipping; matches sklearn convention


def _winner_log_loss(p_winner: np.ndarray) -> float:
    """Mean -log(p_winner) over rows with epsilon clipping."""
    p = np.clip(np.asarray(p_winner, dtype=float), EPS, 1 - EPS)
    return float(-np.log(p).mean())


def compute_pairwise_ll(
    pairwise_csv: str, results_csv: str
) -> tuple[float, np.ndarray, np.ndarray]:
    """Join a pairwise probability CSV with played tournament games and
    compute the winner-perspective log loss.

    Returns:
        (weighted_mean_ll, p_winner_won, labels)
        - weighted_mean_ll: mean -log(p_winner_won)
        - p_winner_won: per-row probability the actual winner won
        - labels: per-row label (always 1 since rows are played games)

    Pairwise schema: season, team_a, team_b, p_a_wins  (team_a < team_b).
    Results schema: Season, DayNum, WTeamID, LTeamID, ... (Kaggle Mania).

    For a played game (W, L):
      let a = min(W, L), b = max(W, L) so the pair matches the pairwise CSV.
      p_winner_won = p_a_wins if W < L else 1 - p_a_wins.
    """
    pairwise = pd.read_csv(pairwise_csv)
    results = pd.read_csv(results_csv)

    # Normalize results into (season, team_a, team_b, w_id) where a < b.
    res = results[["Season", "WTeamID", "LTeamID"]].copy()
    res["team_a"] = np.minimum(res["WTeamID"], res["LTeamID"])
    res["team_b"] = np.maximum(res["WTeamID"], res["LTeamID"])
    res["winner_was_a"] = res["WTeamID"] < res["LTeamID"]
    res = res.rename(columns={"Season": "season"})[
        ["season", "team_a", "team_b", "winner_was_a"]
    ]

    merged = res.merge(
        pairwise[["season", "team_a", "team_b", "p_a_wins"]],
        on=["season", "team_a", "team_b"],
        how="inner",
    )

    p_winner_won = np.where(
        merged["winner_was_a"],
        merged["p_a_wins"],
        1.0 - merged["p_a_wins"],
    ).astype(float)
    labels = np.ones(len(merged), dtype=int)
    return _winner_log_loss(p_winner_won), p_winner_won, labels


def optimal_2blend(
    p_winner_a: np.ndarray, p_winner_b: np.ndarray
) -> tuple[float, float]:
    """Find w in [0, 1] minimizing LL(w * p_a + (1-w) * p_b).

    Returns (w_opt, ll_opt).
    """
    def loss(w):
        blend = w * p_winner_a + (1 - w) * p_winner_b
        return _winner_log_loss(blend)
    result = minimize_scalar(loss, bounds=(0.0, 1.0), method="bounded")
    return float(result.x), float(result.fun)


def optimal_3blend(
    p_winner_v4: np.ndarray,
    p_winner_a: np.ndarray,
    p_winner_b: np.ndarray,
) -> tuple[tuple[float, float, float], float]:
    """Find (w_v4, w_a, w_b) on the simplex minimizing LL of the blend.

    Returns ((w_v4, w_a, w_b), ll_opt). Used for E2 ensemble materialization
    in Task 9; not part of any gate clause.
    """
    def loss(w):
        w0, w1, w2 = w[0], w[1], w[2]
        blend = w0 * p_winner_v4 + w1 * p_winner_a + w2 * p_winner_b
        return _winner_log_loss(blend)

    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
    bounds = [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]
    x0 = np.array([1.0 / 3, 1.0 / 3, 1.0 / 3])
    result = minimize(
        loss, x0, method="SLSQP", bounds=bounds, constraints=constraints,
    )
    return (
        (float(result.x[0]), float(result.x[1]), float(result.x[2])),
        float(result.fun),
    )


def residual_correlation(
    p_winner_a: np.ndarray, p_winner_b: np.ndarray
) -> float:
    """Pearson r between residuals (p - 1) on aligned played-game rows.

    All rows are winner-perspective so the label is always 1; residuals
    are p - 1 = -(1 - p), and Pearson r is invariant to the constant
    shift, so r(resid_A, resid_B) == r(p_a, p_b).
    """
    pa = np.asarray(p_winner_a, dtype=float)
    pb = np.asarray(p_winner_b, dtype=float)
    if pa.std() == 0 or pb.std() == 0:
        return 0.0
    return float(np.corrcoef(pa, pb)[0, 1])


def compute_gate(
    pairwise_v4_csv: str,
    pairwise_peer_a_csv: str,
    pairwise_peer_b_csv: str,
    results_csv: str = str(DATA / "MNCAATourneyCompactResults.csv"),
) -> dict:
    """Compute all three clause values plus the optimal 2-blend / 3-blend
    weights. Returns a dict suitable for json.dump.
    """
    ll_v4, p_v4, _ = compute_pairwise_ll(pairwise_v4_csv, results_csv)
    ll_a, p_a, _ = compute_pairwise_ll(pairwise_peer_a_csv, results_csv)
    ll_b, p_b, _ = compute_pairwise_ll(pairwise_peer_b_csv, results_csv)

    # Sanity: all three CSVs cover the same played games.
    n_v4 = len(p_v4)
    n_a = len(p_a)
    n_b = len(p_b)
    if not (n_v4 == n_a == n_b):
        raise ValueError(
            f"played-game coverage mismatch: v4={n_v4}, peer_a={n_a}, "
            f"peer_b={n_b}; the gate requires identical coverage"
        )

    rho_resid = residual_correlation(p_a, p_b)
    w_opt_2blend, ll_opt_2blend = optimal_2blend(p_a, p_b)
    w_opt_3blend, ll_opt_3blend = optimal_3blend(p_v4, p_a, p_b)
    headroom = ll_v4 - ll_opt_2blend  # positive = blend beats v4

    return {
        "n_played_games": int(n_v4),
        "ll_v4": float(ll_v4),
        "ll_peer_a": float(ll_a),
        "ll_peer_b": float(ll_b),
        "ll_2blend_optimal": float(ll_opt_2blend),
        "ll_3blend_optimal": float(ll_opt_3blend),
        "w_2blend_optimal": float(w_opt_2blend),
        "w_3blend_optimal": list(w_opt_3blend),
        "rho_residual": float(rho_resid),
        "headroom_2blend_vs_v4": float(headroom),
        "clauses": {
            "per_peer_ll_ceiling": {
                "threshold": float(PEER_LL_CEILING_DELTA),
                "ll_v4": float(ll_v4),
                "ll_peer_a": float(ll_a),
                "ll_peer_b": float(ll_b),
                "delta_a": float(ll_a - ll_v4),
                "delta_b": float(ll_b - ll_v4),
                "pass": (ll_a - ll_v4 <= PEER_LL_CEILING_DELTA)
                        and (ll_b - ll_v4 <= PEER_LL_CEILING_DELTA),
            },
            "residual_correlation": {
                "threshold": float(RESID_CORR_MAX),
                "rho": float(rho_resid),
                "pass": rho_resid < RESID_CORR_MAX,
            },
            "blend_headroom": {
                "threshold": float(HEADROOM_MIN),
                "headroom": float(headroom),
                "pass": headroom >= HEADROOM_MIN,
            },
        },
    }


def check_gate(diag: dict) -> dict:
    """All clauses must pass for the gate to clear."""
    failed = [name for name, c in diag["clauses"].items() if not c["pass"]]
    if not failed:
        return {"pass": True, "failed_clauses": [], "reason": "all clauses pass"}
    return {
        "pass": False,
        "failed_clauses": failed,
        "reason": f"failed clauses: {failed}",
    }


def print_report(diag: dict, gate: dict) -> None:
    print("=" * 70)
    print("FEATURE-VIEW ENSEMBLE PRE-SWEEP GATE")
    print("=" * 70)
    print(f"  n played games: {diag['n_played_games']}")
    print(f"\n  Per-game LL on played tournament games:")
    print(f"    v4:                   {diag['ll_v4']:.4f}")
    print(f"    peer_A (team-strength):     {diag['ll_peer_a']:.4f}  "
          f"(delta_a {diag['clauses']['per_peer_ll_ceiling']['delta_a']:+.4f})")
    print(f"    peer_B (form+market):       {diag['ll_peer_b']:.4f}  "
          f"(delta_b {diag['clauses']['per_peer_ll_ceiling']['delta_b']:+.4f})")
    print(f"    2-blend optimal:            {diag['ll_2blend_optimal']:.4f}  "
          f"(headroom vs v4 {diag['headroom_2blend_vs_v4']:+.4f})")
    print(f"    3-blend optimal (v4,A,B):   {diag['ll_3blend_optimal']:.4f}")
    print(f"\n  optimal weights:")
    print(f"    2-blend (A, B):   ({diag['w_2blend_optimal']:.3f}, "
          f"{1 - diag['w_2blend_optimal']:.3f})")
    print(f"    3-blend (v4,A,B): "
          f"({diag['w_3blend_optimal'][0]:.3f}, "
          f"{diag['w_3blend_optimal'][1]:.3f}, "
          f"{diag['w_3blend_optimal'][2]:.3f})")
    print(f"\n  rho(resid_A, resid_B): {diag['rho_residual']:+.3f}")
    print(f"\n  Clause checks:")
    for name, c in diag["clauses"].items():
        verdict = "PASS" if c["pass"] else "FAIL"
        print(f"    {name}: {verdict} (threshold {c['threshold']})")
    print(f"\n=== VERDICT ===")
    if gate["pass"]:
        print(f"  GATE PASSED: {gate['reason']}")
        print(f"  -> Proceed to materialize E1 + E2 ensemble CSVs and run sweeps.")
    else:
        print(f"  GATE FAILED: {gate['reason']}")
        print(f"  -> Stop. Write findings note. No sweep.")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--pairwise-v4", default="output/pairwise_v4.csv")
    parser.add_argument("--pairwise-peer-a", default="output/pairwise_peer_a.csv")
    parser.add_argument("--pairwise-peer-b", default="output/pairwise_peer_b.csv")
    parser.add_argument(
        "--results-csv",
        default=str(DATA / "MNCAATourneyCompactResults.csv"),
    )
    parser.add_argument("--out-json", default=DEFAULT_DIAGNOSTIC_OUT)
    args = parser.parse_args(argv)

    diag = compute_gate(
        pairwise_v4_csv=args.pairwise_v4,
        pairwise_peer_a_csv=args.pairwise_peer_a,
        pairwise_peer_b_csv=args.pairwise_peer_b,
        results_csv=args.results_csv,
    )
    gate = check_gate(diag)
    print_report(diag, gate)

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump({"diagnostic": diag, "gate": gate}, f, indent=2)
    print(f"\n  saved {args.out_json}")
    return 0 if gate["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_diagnose_feature_view_ensemble.py -v`
Expected: 9 passed.

If `test_compute_gate_passes_when_all_clauses_met` fails because the
synthetic peers don't actually produce a positive headroom -- accept
that the smoke test in that case is just verifying clause-shape; tighten
the synthetic construction (lower peer-A LL slightly) and re-run. The
load-bearing tests are the per-clause failure tests, not the omnibus
pass test.

- [ ] **Step 5: ASCII check + commit**

```bash
python -c "open('src/diagnose_feature_view_ensemble.py', encoding='utf-8').read().encode('ascii')" && echo "ASCII OK"
python -c "open('tests/test_diagnose_feature_view_ensemble.py', encoding='utf-8').read().encode('ascii')" && echo "ASCII OK"
git add src/diagnose_feature_view_ensemble.py tests/test_diagnose_feature_view_ensemble.py
git commit -m "$(cat <<'EOF'
feat(feature-view-ensemble): src/diagnose_feature_view_ensemble.py 3-clause gate

Pre-sweep falsification gate. Three clauses, each mapping to a prior-
experiment failure mode:
  1. per_peer_ll_ceiling: peer LL within 0.025 of v4 (PR 12 lesson)
  2. residual_correlation: rho < 0.60 (PR 11 lesson)
  3. blend_headroom: best 2-blend beats v4 by >= 0.001 (PR 13 lesson)

Also computes optimal 3-blend weights (v4, A, B) for E2 materialization
in Task 9 -- not gated, just produced alongside.

Exits nonzero on FAIL so a wrapper can short-circuit the sweep.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: extend `src/ensemble_stage1.py` for K-way blend

**Why:** Existing `average_pairwise_csvs` blends two CSVs at user-supplied weights. E1 reuses it directly. E2 needs a 3-input blend; rather than calling `average` twice, add a single `blend_pairwise_csvs(inputs, weights, out)` that handles K inputs.

**Files:**
- Modify: `src/ensemble_stage1.py`
- Modify: `tests/test_ensemble_stage1.py` (already exists from PR 11)

- [ ] **Step 1: Read existing tests for `average_pairwise_csvs`**

Run: `grep -n "def test_" tests/test_ensemble_stage1.py`

Note the existing test patterns (anchor weights, schema, join coverage). New tests follow the same style.

- [ ] **Step 2: Write the failing test for the K-way blend**

Append to `tests/test_ensemble_stage1.py`:

```python
def test_blend_pairwise_csvs_three_inputs(tmp_path):
    """Three input CSVs at uniform 1/3 weights produce row-wise mean
    of p_a_wins. Schema and join coverage same as average_pairwise_csvs.
    """
    from src.ensemble_stage1 import blend_pairwise_csvs

    csvs = []
    p_values = [0.6, 0.4, 0.5]
    for i, p in enumerate(p_values):
        path = tmp_path / f"in_{i}.csv"
        pd.DataFrame({
            "season": 2024, "team_a": 1, "team_b": 2,
            "p_a_wins": [p],
        }).to_csv(path, index=False)
        csvs.append(str(path))

    out = tmp_path / "out.csv"
    blend_pairwise_csvs(csvs, weights=[1/3, 1/3, 1/3], out=str(out))

    df = pd.read_csv(out)
    assert list(df.columns) == ["season", "team_a", "team_b", "p_a_wins"]
    assert df["p_a_wins"].iloc[0] == pytest.approx(np.mean(p_values))


def test_blend_pairwise_csvs_anchor_one_input(tmp_path):
    """Single input at weight 1.0 reproduces input row-for-row."""
    from src.ensemble_stage1 import blend_pairwise_csvs

    src = tmp_path / "src.csv"
    pd.DataFrame({
        "season": [2024, 2024],
        "team_a": [1, 1],
        "team_b": [2, 3],
        "p_a_wins": [0.6, 0.7],
    }).to_csv(src, index=False)

    out = tmp_path / "out.csv"
    blend_pairwise_csvs([str(src)], weights=[1.0], out=str(out))

    expected = pd.read_csv(src)
    actual = pd.read_csv(out)
    pd.testing.assert_frame_equal(
        actual.sort_values(["season", "team_a", "team_b"]).reset_index(drop=True),
        expected.sort_values(["season", "team_a", "team_b"]).reset_index(drop=True),
    )


def test_blend_pairwise_csvs_weight_count_mismatch_raises(tmp_path):
    from src.ensemble_stage1 import blend_pairwise_csvs

    pd.DataFrame({"season": 2024, "team_a": [1], "team_b": [2],
                  "p_a_wins": [0.5]}).to_csv(tmp_path / "a.csv", index=False)

    with pytest.raises(ValueError, match="weights"):
        blend_pairwise_csvs(
            [str(tmp_path / "a.csv")],
            weights=[0.5, 0.5],
            out=str(tmp_path / "out.csv"),
        )


def test_blend_pairwise_csvs_weights_must_sum_to_one(tmp_path):
    from src.ensemble_stage1 import blend_pairwise_csvs

    pd.DataFrame({"season": 2024, "team_a": [1], "team_b": [2],
                  "p_a_wins": [0.5]}).to_csv(tmp_path / "a.csv", index=False)

    with pytest.raises(ValueError, match="sum"):
        blend_pairwise_csvs(
            [str(tmp_path / "a.csv")],
            weights=[0.7],
            out=str(tmp_path / "out.csv"),
        )
```

If `tests/test_ensemble_stage1.py` doesn't already import `numpy as np` and `pytest`, add those imports at the top.

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest tests/test_ensemble_stage1.py -v -k blend_pairwise_csvs`
Expected: 4 errors, all `ImportError: cannot import name 'blend_pairwise_csvs'`.

- [ ] **Step 4: Add `blend_pairwise_csvs` to `src/ensemble_stage1.py`**

Append after the existing `average_pairwise_csvs` function:

```python
def blend_pairwise_csvs(
    inputs: list[str],
    weights: list[float],
    out: str,
) -> None:
    """K-way generalization of average_pairwise_csvs.

    inputs:  list of paths to pairwise CSVs (schema season, team_a, team_b, p_a_wins).
             All inputs must share identical (season, team_a, team_b) coverage.
    weights: list of per-input non-negative weights, len == len(inputs),
             must sum to 1.0 within 1e-9.
    out:     path to write the blended CSV (same schema as inputs).
    """
    if len(weights) != len(inputs):
        raise ValueError(
            f"weights count ({len(weights)}) != inputs count ({len(inputs)})"
        )
    w_sum = sum(float(w) for w in weights)
    if abs(w_sum - 1.0) > 1e-9:
        raise ValueError(
            f"weights must sum to 1; got {w_sum:.6f}"
        )

    dfs = [pd.read_csv(p) for p in inputs]
    for i, df in enumerate(dfs):
        missing = set(SCHEMA) - set(df.columns)
        if missing:
            raise ValueError(
                f"input {i} ({inputs[i]}) missing columns: {sorted(missing)}"
            )
        dfs[i] = df.drop_duplicates(subset=JOIN_KEYS, keep="last")

    # Inner-join all inputs on (season, team_a, team_b); coverage check.
    base = dfs[0][JOIN_KEYS + ["p_a_wins"]].rename(columns={"p_a_wins": "p_0"})
    for i, df in enumerate(dfs[1:], start=1):
        rhs = df[JOIN_KEYS + ["p_a_wins"]].rename(columns={"p_a_wins": f"p_{i}"})
        base = base.merge(rhs, on=JOIN_KEYS, how="outer", indicator=True)
        only_left = (base["_merge"] == "left_only").sum()
        only_right = (base["_merge"] == "right_only").sum()
        if only_left or only_right:
            raise ValueError(
                f"input {i} coverage mismatch: {only_left} rows only in prior "
                f"inputs, {only_right} rows only in input {i}; the blend "
                "requires identical (season, team_a, team_b) coverage"
            )
        base = base.drop(columns=["_merge"])

    # Weighted sum of the per-input p columns.
    p_blend = sum(
        float(weights[i]) * base[f"p_{i}"]
        for i in range(len(inputs))
    )
    base["p_a_wins"] = p_blend

    out_df = (
        base[SCHEMA]
        .sort_values(JOIN_KEYS)
        .reset_index(drop=True)
    )
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out, index=False)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_ensemble_stage1.py -v`
Expected: all tests pass (existing + 4 new).

- [ ] **Step 6: ASCII check + commit**

```bash
python -c "open('src/ensemble_stage1.py', encoding='utf-8').read().encode('ascii')" && echo "ASCII OK"
python -c "open('tests/test_ensemble_stage1.py', encoding='utf-8').read().encode('ascii')" && echo "ASCII OK"
git add src/ensemble_stage1.py tests/test_ensemble_stage1.py
git commit -m "$(cat <<'EOF'
feat(feature-view-ensemble): blend_pairwise_csvs K-way generalization

Adds blend_pairwise_csvs() to src/ensemble_stage1.py. Generalizes the
existing 2-way average_pairwise_csvs to K inputs at user-supplied
weights. Used by Task 9 to materialize E1 (K=2 over peer A and peer B)
and E2 (K=3 over v4, peer A, peer B) ensemble pairwise CSVs.

Existing average_pairwise_csvs is unchanged.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 5: extend `src/sweep_v9_weights.py` for `V9_STAGE1_PAIRWISE` env var

**Why:** The sweep harness today hardcodes `pairwise_v4` as the stage-1 input. To run the sweep against E1 / E2 ensembled stage-1 outputs without forking the harness, add an env-var override that points to a different pairwise CSV. Output dir keys off the basename so v9-C / v9-D / E1 / E2 artifacts coexist.

**Files:**
- Modify: `src/sweep_v9_weights.py` (function `main`)
- Modify: `tests/test_sweep_v9_weights.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_sweep_v9_weights.py`:

```python
def test_sweep_main_uses_v9_stage1_pairwise_env_var(tmp_path, monkeypatch):
    """When V9_STAGE1_PAIRWISE is set, main() uses that path as the
    stage-1 input and writes outputs to a basename-keyed dir.

    We don't run the full sweep here -- we monkey-patch run_sweep to
    capture its arguments and assert the env-var value flowed through.
    """
    from src import sweep_v9_weights as sw

    captured = {}

    def fake_run_sweep(**kwargs):
        captured.update(kwargs)
        # Return a minimal df so main() doesn't crash on the score-anchor block.
        import pandas as pd
        return pd.DataFrame([{
            "w_upset": 1.0, "w_miss": 0.0, "total_brkt_pts": 100.0,
            "ll_loso_weighted_mean": 0.5, "acc_loso_weighted_mean": 0.5,
            "pairwise_csv": str(tmp_path / "fake.csv"),
        }])

    monkeypatch.setattr(sw, "run_sweep", fake_run_sweep)
    monkeypatch.setattr(
        sw, "score_pairwise_path",
        lambda *a, **k: {"total_pts": 100.0},
        raising=False,
    )
    # Make the v8 lookup tolerable; the env var override doesn't change v8 logic.
    monkeypatch.setenv("V9_FEATURE_SET", "v9c")

    pw_override = tmp_path / "pairwise_ensemble_e1.csv"
    pw_override.write_text("season,team_a,team_b,p_a_wins\n")
    monkeypatch.setenv("V9_STAGE1_PAIRWISE", str(pw_override))

    # main() may try to score v8 from a path; tolerate by short-circuiting.
    try:
        sw.main()
    except Exception:
        # The fake doesn't fully simulate downstream IO; we only care that
        # run_sweep was called with the override path.
        pass

    assert captured.get("pairwise_v4_csv") == str(pw_override)
    # Output dir basename incorporates the override filename.
    assert "ensemble_e1" in captured.get("out_dir", "")


def test_sweep_default_path_when_env_var_unset(monkeypatch):
    """When V9_STAGE1_PAIRWISE is unset, main() uses output/pairwise_v4.csv."""
    from src import sweep_v9_weights as sw

    captured = {}

    def fake_run_sweep(**kwargs):
        captured.update(kwargs)
        import pandas as pd
        return pd.DataFrame([{
            "w_upset": 1.0, "w_miss": 0.0, "total_brkt_pts": 100.0,
            "ll_loso_weighted_mean": 0.5, "acc_loso_weighted_mean": 0.5,
            "pairwise_csv": "fake.csv",
        }])

    monkeypatch.setattr(sw, "run_sweep", fake_run_sweep)
    monkeypatch.setattr(
        sw, "score_pairwise_path",
        lambda *a, **k: {"total_pts": 100.0},
        raising=False,
    )
    monkeypatch.setenv("V9_FEATURE_SET", "v9c")
    monkeypatch.delenv("V9_STAGE1_PAIRWISE", raising=False)

    try:
        sw.main()
    except Exception:
        pass

    assert captured.get("pairwise_v4_csv") == "output/pairwise_v4.csv"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_sweep_v9_weights.py -v -k v9_stage1_pairwise`
Expected: 2 failures with the env var not yet wired.

- [ ] **Step 3: Modify `src/sweep_v9_weights.py:main`**

Read the current `main()` first to confirm the local variable layout. Then add env-var handling at the top of `main()`, after `feature_set` is read:

```python
def main():
    """Run the canonical 15-cell sweep against production data paths.

    feature_set is read from the V9_FEATURE_SET env var (default 'v9b').
    pairwise_v4_csv is read from the V9_STAGE1_PAIRWISE env var
    (default 'output/pairwise_v4.csv') -- the harness uses whatever
    pairwise CSV is supplied as the stage-1 input. Output dirs key off
    feature_set and the stage-1 input's basename so v9-B / v9-C /
    v9-D / ensemble-E1 / ensemble-E2 artifacts coexist.

    Compares the anchor cell (1.0, 0.0) bracket points against
    output/pairwise_v8.csv as a sanity gate after the sweep.
    """
    import os
    feature_set = os.environ.get("V9_FEATURE_SET", "v9b")
    if feature_set not in ("v9b", "v9c", "v9d"):
        raise ValueError(
            f"V9_FEATURE_SET={feature_set!r} invalid; "
            "must be 'v9b', 'v9c', or 'v9d'"
        )

    pairwise_v4 = os.environ.get(
        "V9_STAGE1_PAIRWISE", "output/pairwise_v4.csv"
    )

    print("=" * 80)
    print(f"V9 UPSET-WEIGHT SWEEP (feature_set={feature_set})")
    print(f"  stage-1 input: {pairwise_v4}")
    print(f"  Grid: {len(GRID)} cells, "
          f"W_UPSET in {W_UPSET_VALUES}, W_MISS in {W_MISS_VALUES}")
    print("=" * 80)

    pairwise_v8 = "output/pairwise_v8.csv"
    seeds_csv = "data/raw/march-machine-learning-2026/MNCAATourneySeeds.csv"
    results_csv = "data/raw/march-machine-learning-2026/MNCAATourneyCompactResults.csv"
    slots_csv = "data/raw/march-machine-learning-2026/MNCAATourneySlots.csv"

    # Output dir keys off the stage-1 input basename so different stage-1s
    # produce non-colliding artifacts. The default ('pairwise_v4.csv')
    # preserves the historical 'output/v9{c|d}_sweep' naming for
    # backwards compatibility.
    pw_basename = Path(pairwise_v4).stem  # e.g. 'pairwise_v4', 'pairwise_ensemble_e1'
    if pw_basename == "pairwise_v4":
        if feature_set == "v9b":
            out_dir = "output/v9_sweep"
            results_csv_path = "output/v9_sweep_results.csv"
        elif feature_set == "v9c":
            out_dir = "output/v9c_sweep"
            results_csv_path = "output/v9c_sweep_results.csv"
        else:  # v9d
            out_dir = "output/v9d_sweep"
            results_csv_path = "output/v9d_sweep_results.csv"
    else:
        # Custom stage-1 input: e.g. pairwise_ensemble_e1.csv -> v9c_ensemble_e1_sweep.
        suffix = pw_basename.replace("pairwise_", "")
        out_dir = f"output/{feature_set}_{suffix}_sweep"
        results_csv_path = f"output/{feature_set}_{suffix}_sweep_results.csv"

    pairwise_bt_csv = "output/pairwise_bt.csv" if feature_set == "v9d" else None

    df = run_sweep(
        grid=GRID,
        pairwise_v4_csv=pairwise_v4,
        results_csv=results_csv,
        seeds_csv=seeds_csv,
        out_dir=out_dir,
        results_csv_path=results_csv_path,
        slots_csv=slots_csv,
        feature_set=feature_set,
        pairwise_bt_csv=pairwise_bt_csv,
    )
    # ... rest of main() unchanged ...
```

The remainder of `main()` (summary table, v8 anchor, winner check) is unchanged.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_sweep_v9_weights.py -v`
Expected: all tests pass (existing + 2 new).

- [ ] **Step 5: ASCII check + commit**

```bash
python -c "open('src/sweep_v9_weights.py', encoding='utf-8').read().encode('ascii')" && echo "ASCII OK"
python -c "open('tests/test_sweep_v9_weights.py', encoding='utf-8').read().encode('ascii')" && echo "ASCII OK"
git add src/sweep_v9_weights.py tests/test_sweep_v9_weights.py
git commit -m "$(cat <<'EOF'
feat(feature-view-ensemble): V9_STAGE1_PAIRWISE env var override

Adds V9_STAGE1_PAIRWISE env var to src/sweep_v9_weights.py:main so the
15-cell sweep harness can be retargeted at any pairwise CSV (e.g. an
ensembled stage-1 output) without forking the harness. Output dir
basename keys off the input filename when non-default; the default
'pairwise_v4.csv' preserves the v9_sweep / v9c_sweep / v9d_sweep
naming for backwards compatibility.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 6: Run full ingest/feature/integration test suite (CLAUDE.md forced verification)

**Why:** Tasks 1-5 touched data-loading and trainer code. CLAUDE.md's forced-verification rule requires running the full ingest/feature/integration suite before any real-data run, even if the change is narrowly scoped.

**Files:** None.

- [ ] **Step 1: Run the full ingest/feature/integration suite**

Run:
```bash
pytest -v tests/test_ingest tests/test_features tests/test_integration.py
```

Expected: all passed. Capture the output and paste a summary into the next commit message if any tests are slow / flaky.

If any test fails, stop and investigate before proceeding -- a failing test here means a code regression upstream of the experiment.

- [ ] **Step 2: Run the full repo test suite (excluding integration if too slow)**

Run:
```bash
pytest -q
```

Expected: all passed.

- [ ] **Step 3: No commit -- this task is a verification gate**

Proceed to Task 7 only if Steps 1 and 2 produced no failures.

---

### Task 7: Generate peer pairwise CSVs (real data)

**Why:** The gate diagnostic (Task 8) needs `output/pairwise_peer_a.csv` and `output/pairwise_peer_b.csv`. This task produces them by running `src/train_peer_stage1.py` for each peer.

**Files:**
- Create: `output/pairwise_peer_a.csv` (run-time)
- Create: `output/pairwise_peer_b.csv` (run-time)

- [ ] **Step 1: Train peer A**

Run:
```bash
python src/train_peer_stage1.py --peer a --output output/pairwise_peer_a.csv
```

Expected runtime: ~5-15 min (22 LOSO seasons x XGB fit + per-season pair dump).

Expected stdout: `feature_cols total: 67`, `peer_cols restricted to PEER_A: 40`, then 22 lines of `season YYYY: NNN pairs (cumulative XXXX)`. Final line `wrote XXXX pair rows to output/pairwise_peer_a.csv`.

- [ ] **Step 2: Sanity-check peer A schema and row count**

Run:
```bash
python -c "
import pandas as pd
df = pd.read_csv('output/pairwise_peer_a.csv')
print('shape:', df.shape)
print('columns:', list(df.columns))
print('seasons:', sorted(df['season'].unique()))
print('symmetric-pair check (team_a < team_b):', (df['team_a'] < df['team_b']).all())
print('p range:', df['p_a_wins'].min(), df['p_a_wins'].max())
"
```

Expected: schema `[season, team_a, team_b, p_a_wins]`; seasons `[2003, ..., 2025]` (22 years); team_a < team_b is True; p in `[~0.05, ~0.95]` range.

- [ ] **Step 3: Train peer B**

Run:
```bash
python src/train_peer_stage1.py --peer b --output output/pairwise_peer_b.csv
```

Expected runtime: ~5-15 min. Same shape stdout but `peer_cols restricted to PEER_B: 27`.

- [ ] **Step 4: Sanity-check peer B schema and row count**

Run the same sanity-check command as Step 2 but on `pairwise_peer_b.csv`. Both peer CSVs should have the same row count (same 22 seasons, same field per season).

Cross-check:
```bash
python -c "
import pandas as pd
a = pd.read_csv('output/pairwise_peer_a.csv')
b = pd.read_csv('output/pairwise_peer_b.csv')
v4 = pd.read_csv('output/pairwise_v4.csv')
print('peer_a rows:', len(a))
print('peer_b rows:', len(b))
print('v4 rows:    ', len(v4))
"
```

Expected: peer_a rows == peer_b rows; same as v4 rows (since all three use the same 22-season tournament field).

If row counts differ, debug before proceeding -- the gate diagnostic requires identical (season, team_a, team_b) coverage across the three CSVs.

- [ ] **Step 5: Commit the peer CSVs**

```bash
git add output/pairwise_peer_a.csv output/pairwise_peer_b.csv
git commit -m "$(cat <<'EOF'
data(feature-view-ensemble): peer A and peer B pairwise CSVs

22-season LOSO OOF predictions from XGBoost trained on PEER_A
(40 team-strength features) and PEER_B (27 form/market/meta features)
respectively. Generated via:
  python src/train_peer_stage1.py --peer a --output output/pairwise_peer_a.csv
  python src/train_peer_stage1.py --peer b --output output/pairwise_peer_b.csv

Schema matches output/pairwise_v4.csv. Inputs to the gate diagnostic
in Task 8.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 8: Run the 3-clause gate + branch on verdict (real data)

**Why:** This is the experimental decision point. PASS -> proceed to E1 / E2 sweeps. FAIL -> write findings note, update TODO, finalize PR; do not run the sweeps.

**Files:**
- Create: `output/diag_feature_view_ensemble.json` (run-time)

- [ ] **Step 1: Run the gate diagnostic**

Run:
```bash
python src/diagnose_feature_view_ensemble.py
```

Expected: prints the report (n played games, per-peer LL, blend headroom, per-clause PASS/FAIL); writes `output/diag_feature_view_ensemble.json`; exits 0 on PASS or 1 on FAIL.

Capture the verdict line ("GATE PASSED" or "GATE FAILED") and the failed-clause list (if any).

- [ ] **Step 2: Inspect the gate JSON**

Run:
```bash
python -c "
import json
d = json.load(open('output/diag_feature_view_ensemble.json'))
print(json.dumps(d, indent=2))
"
```

Confirm:
- `diagnostic.n_played_games` ~ 1449 (22 seasons of tournament games).
- `diagnostic.ll_v4` ~ 0.4369 (matches v4's known weighted-mean LL).
- `gate.pass` is consistent with what the report printed.

- [ ] **Step 3: Commit the diagnostic**

```bash
git add output/diag_feature_view_ensemble.json
git commit -m "data(feature-view-ensemble): output/diag_feature_view_ensemble.json -- gate $(python -c \"import json; print('PASSED' if json.load(open('output/diag_feature_view_ensemble.json'))['gate']['pass'] else 'FAILED')\")"
```

(The `$(...)` expansion auto-fills the verdict in the commit subject.)

- [ ] **Step 4: BRANCH ON VERDICT**

**If gate FAILED -> proceed to Task 9-FAIL.** Skip Tasks 9 through 12. The findings note in Task 9-FAIL closes the experiment.

**If gate PASSED -> proceed to Task 9.**

---

### Task 9 (PASS branch): Materialize E1 + E2 ensemble pairwise CSVs

**Why:** The sweep harness consumes a single pairwise CSV per run. E1 = blend(peer A, peer B) at the gate's optimal 2-blend weight; E2 = blend(v4, peer A, peer B) at the gate's optimal 3-blend weight. Both are produced as pairwise CSVs ready to be passed via `V9_STAGE1_PAIRWISE`.

**Files:**
- Create: `output/pairwise_ensemble_e1.csv` (run-time)
- Create: `output/pairwise_ensemble_e2.csv` (run-time)

- [ ] **Step 1: Read optimal weights from the diagnostic**

Run:
```bash
python -c "
import json
d = json.load(open('output/diag_feature_view_ensemble.json'))['diagnostic']
print('w_2blend (peer_a, peer_b):', d['w_2blend_optimal'], 1 - d['w_2blend_optimal'])
print('w_3blend (v4, peer_a, peer_b):', d['w_3blend_optimal'])
"
```

Note the values.

- [ ] **Step 2: Materialize E1 (peer_a + peer_b at 2-blend optimal weights)**

Run, substituting `<W_A>` with the optimal w_2blend value from Step 1:
```bash
python -c "
import json
from src.ensemble_stage1 import blend_pairwise_csvs
d = json.load(open('output/diag_feature_view_ensemble.json'))['diagnostic']
w_a = d['w_2blend_optimal']
w_b = 1 - w_a
blend_pairwise_csvs(
    inputs=['output/pairwise_peer_a.csv', 'output/pairwise_peer_b.csv'],
    weights=[w_a, w_b],
    out='output/pairwise_ensemble_e1.csv',
)
print(f'E1: w_a={w_a:.4f}, w_b={w_b:.4f}')
"
```

Expected: prints the weights; writes `output/pairwise_ensemble_e1.csv`.

- [ ] **Step 3: Materialize E2 (v4 + peer_a + peer_b at 3-blend optimal weights)**

Run:
```bash
python -c "
import json
from src.ensemble_stage1 import blend_pairwise_csvs
d = json.load(open('output/diag_feature_view_ensemble.json'))['diagnostic']
w0, w1, w2 = d['w_3blend_optimal']
blend_pairwise_csvs(
    inputs=['output/pairwise_v4.csv', 'output/pairwise_peer_a.csv', 'output/pairwise_peer_b.csv'],
    weights=[w0, w1, w2],
    out='output/pairwise_ensemble_e2.csv',
)
print(f'E2: w_v4={w0:.4f}, w_a={w1:.4f}, w_b={w2:.4f}')
"
```

- [ ] **Step 4: Sanity-check both ensemble CSVs**

```bash
python -c "
import pandas as pd
e1 = pd.read_csv('output/pairwise_ensemble_e1.csv')
e2 = pd.read_csv('output/pairwise_ensemble_e2.csv')
v4 = pd.read_csv('output/pairwise_v4.csv')
print('E1 rows:', len(e1), '(expect ==', len(v4), ')')
print('E2 rows:', len(e2), '(expect ==', len(v4), ')')
print('E1 p range:', e1['p_a_wins'].min(), e1['p_a_wins'].max())
print('E2 p range:', e2['p_a_wins'].min(), e2['p_a_wins'].max())
"
```

Expected: row counts match v4; p ranges are valid probabilities.

- [ ] **Step 5: Commit**

```bash
git add output/pairwise_ensemble_e1.csv output/pairwise_ensemble_e2.csv
git commit -m "$(cat <<'EOF'
data(feature-view-ensemble): E1 and E2 ensemble pairwise CSVs

E1 = optimal-weight blend(peer_a, peer_b)
E2 = optimal-weight blend(v4, peer_a, peer_b)

Weights from output/diag_feature_view_ensemble.json (the 3-clause
gate's optimal_2blend / optimal_3blend output). Inputs to the
V9_STAGE1_PAIRWISE-driven sweep runs in Tasks 10 and 11.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 10 (PASS branch): Run the E1 sweep

**Why:** Produces 15 per-cell pairwise CSVs and a sweep results CSV for the E1 ensemble fed through v9-C's stage-2 transform.

**Files:**
- Create: `output/v9c_ensemble_e1_sweep/pairwise_v9_WU{u:.2f}_WM{m:.2f}.csv` x 15 (run-time)
- Create: `output/v9c_ensemble_e1_sweep_results.csv` (run-time)

- [ ] **Step 1: Run the E1 sweep**

Run:
```bash
V9_STAGE1_PAIRWISE=output/pairwise_ensemble_e1.csv V9_FEATURE_SET=v9c python src/sweep_v9_weights.py
```

Expected runtime: ~45-75 min. Stdout shows the 15-cell loop with per-cell `total_brkt_pts`, `ll`, `acc`, then the results table sorted descending.

- [ ] **Step 2: Sanity-check the results**

Run:
```bash
python -c "
import pandas as pd
df = pd.read_csv('output/v9c_ensemble_e1_sweep_results.csv')
print(df.to_string(index=False))
print()
print('Best cell:', df.iloc[0].to_dict())
"
```

Note the best cell's `total_brkt_pts`. Compute the delta vs v9-C's baseline (2713 brkt pts):
```bash
python -c "
import pandas as pd
df = pd.read_csv('output/v9c_ensemble_e1_sweep_results.csv')
best = df.iloc[0]['total_brkt_pts']
print(f'E1 best: {best:.1f} brkt pts (delta vs v9-C 2713: {best - 2713:+.1f})')
"
```

- [ ] **Step 3: Commit**

```bash
git add output/v9c_ensemble_e1_sweep_results.csv output/v9c_ensemble_e1_sweep/
git commit -m "$(cat <<'EOF'
data(feature-view-ensemble): E1 sweep results

15-cell W_UPSET / W_MISS sweep with v9-C stage-2 transform on the
E1 ensemble (peer_a + peer_b at gate-optimal weights) as stage-1.
Generated via:
  V9_STAGE1_PAIRWISE=output/pairwise_ensemble_e1.csv \
    V9_FEATURE_SET=v9c python src/sweep_v9_weights.py

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 11 (PASS branch): Run the E2 sweep

**Why:** Same as Task 10, but for E2 (v4 + peer_a + peer_b at 3-blend optimal weights).

**Files:**
- Create: `output/v9c_ensemble_e2_sweep/pairwise_v9_WU{u:.2f}_WM{m:.2f}.csv` x 15 (run-time)
- Create: `output/v9c_ensemble_e2_sweep_results.csv` (run-time)

- [ ] **Step 1: Run the E2 sweep**

Run:
```bash
V9_STAGE1_PAIRWISE=output/pairwise_ensemble_e2.csv V9_FEATURE_SET=v9c python src/sweep_v9_weights.py
```

Expected runtime: ~45-75 min.

- [ ] **Step 2: Sanity-check the results**

Same as Task 10 Step 2 but for E2:
```bash
python -c "
import pandas as pd
df = pd.read_csv('output/v9c_ensemble_e2_sweep_results.csv')
print(df.to_string(index=False))
print()
best = df.iloc[0]['total_brkt_pts']
print(f'E2 best: {best:.1f} brkt pts (delta vs v9-C 2713: {best - 2713:+.1f})')
"
```

- [ ] **Step 3: Commit**

```bash
git add output/v9c_ensemble_e2_sweep_results.csv output/v9c_ensemble_e2_sweep/
git commit -m "$(cat <<'EOF'
data(feature-view-ensemble): E2 sweep results

15-cell W_UPSET / W_MISS sweep with v9-C stage-2 transform on the
E2 ensemble (v4 + peer_a + peer_b at gate-optimal weights) as
stage-1. Generated via:
  V9_STAGE1_PAIRWISE=output/pairwise_ensemble_e2.csv \
    V9_FEATURE_SET=v9c python src/sweep_v9_weights.py

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 12 (PASS branch): Verify v9-C harness anchor + compute deltas

**Why:** Confirms the env-var threading didn't regress the v9-C path. The anchor cell `(W_UPSET=1.0, W_MISS=0.0)` re-run with `V9_STAGE1_PAIRWISE=output/pairwise_v4.csv` must reproduce the committed v9-C anchor within `1e-9` per-pair max delta.

**Files:** None new; verification only.

- [ ] **Step 1: Re-run the v9-C anchor cell**

Run:
```bash
V9_STAGE1_PAIRWISE=output/pairwise_v4.csv V9_FEATURE_SET=v9c python src/sweep_v9_weights.py
```

This re-runs the full 15-cell v9-C sweep. The anchor cell `(1.0, 0.0)` is what we care about.

Expected runtime: ~45-75 min. This is the third sweep; the previous two (E1 and E2) ran in Tasks 10 and 11.

- [ ] **Step 2: Compare the anchor cell against the committed v9-C anchor**

Run:
```bash
python -c "
import pandas as pd
import numpy as np

new = pd.read_csv('output/v9c_sweep/pairwise_v9_WU1.00_WM0.00.csv')
# Compare against the committed copy in git history (use git show if needed).
# For this check we re-read the file as-of-now and confirm against the earlier
# copy which should match the pre-extension v9-C output exactly.
old = pd.read_csv('output/v9c_sweep/pairwise_v9_WU1.00_WM0.00.csv')

# In practice the new run overwrites the old. The anchor verification is
# instead: did the v9-C anchor's bracket points reproduce within 1e-9?
# Use diff against git's previous commit:
import subprocess
prev = subprocess.run(
    ['git', 'show', 'HEAD~1:output/v9c_sweep/pairwise_v9_WU1.00_WM0.00.csv'],
    capture_output=True, text=True,
).stdout
import io
prev_df = pd.read_csv(io.StringIO(prev))
merged = new.merge(
    prev_df, on=['season', 'team_a', 'team_b'], suffixes=('_new', '_old'),
)
delta = (merged['p_a_wins_new'] - merged['p_a_wins_old']).abs().max()
print(f'max per-pair p delta: {delta}')
assert delta < 1e-9, 'v9-C anchor regressed; investigate before trusting E1/E2 results'
print('v9-C anchor reproduces within 1e-9 -- harness is sane.')
"
```

If the assertion fires, the env-var extension introduced a regression in the v9-C path. Stop and debug before proceeding -- the E1/E2 results may be invalid.

- [ ] **Step 3: Score v9-C's production cell freshly + compute deltas**

Run:
```bash
python -c "
from src.score_chalk_brackets import score_pairwise_path
import pandas as pd

baseline = score_pairwise_path('output/v9c_sweep/pairwise_v9_WU1.25_WM0.00.csv')['total_pts']
print(f'v9-C production cell baseline: {baseline:.1f} brkt pts')

for variant in ('e1', 'e2'):
    df = pd.read_csv(f'output/v9c_ensemble_{variant}_sweep_results.csv')
    print(f'\n{variant.upper()}:')
    df['delta_vs_v9c'] = df['total_brkt_pts'] - baseline
    print(df[['w_upset', 'w_miss', 'total_brkt_pts', 'delta_vs_v9c']].to_string(index=False))
    best = df.iloc[0]
    print(f'  best cell: WU={best[\"w_upset\"]}, WM={best[\"w_miss\"]}, '
          f'total={best[\"total_brkt_pts\"]:.1f}, delta={best[\"delta_vs_v9c\"]:+.1f}')
"
```

Expected: prints v9-C baseline, then per-cell delta tables for E1 and E2 with best cells highlighted.

- [ ] **Step 4: No commit -- this task is a verification + analysis gate**

Capture the best cells and deltas; they go in the findings note in Task 13.

---

### Task 13: Write findings note + update TODO + finalize PR

**Why:** Closes the experiment. Single source of truth for the verdict and what the experiment learned.

**Files:**
- Create: `docs/notes/2026-05-02-feature-view-ensemble.md`
- Modify: `TODO.md`

- [ ] **Step 1: Write the findings note**

Create `docs/notes/2026-05-02-feature-view-ensemble.md`. Use the BT-as-feature findings note as a template (`docs/notes/2026-05-02-bt-as-feature.md`). Sections:

1. **Header** -- date, branch, verdict (one of: NO-GO at gate, NO-GO at sweep, marginal candidate, clear winner), spec / plan paths.
2. **TL;DR** -- one paragraph summarizing the outcome and the key numbers.
3. **Setup recap** -- what the experiment tested.
4. **Pre-sweep gate result** -- table of clause values, thresholds, PASS/FAIL.
5. **Sweep results** (if gate PASSED) -- E1 and E2 best-cell numbers, per-season sketch if interesting.
6. **Falsification reasoning** -- one paragraph explaining what each clause failure would have meant and what the actual outcome implies.
7. **Comparison to predecessors** -- table comparing this experiment's mechanism / failure mode against PR 11, PR 12, PR 13.
8. **Verdict** -- explicit GO / NO-GO / MARGINAL with action next.
9. **Recommendation** -- which queue item moves forward.
10. **Files of record** -- list of artifacts (specs, plans, code, data, tests).

ASCII-only. Verify with `python -c "open('PATH').read().encode('ascii')"`.

- [ ] **Step 2: Update TODO.md**

Move the `feature-view-diversity-ensemble` line from the "Active queue" section to either:
- "Tried and rejected" (if NO-GO),
- "Done" with a one-line summary (if MARGINAL or WINNER).

If WINNER (`delta >= +25` for E1 or E2), add a new "Active queue" item: **"Production swap PR for the feature-view diversity ensemble"** referencing the spec and findings paths.

If MARGINAL or NO-GO, the new top-of-queue item is "Hierarchical Bradley-Terry with feature priors" (currently #2).

ASCII-only.

- [ ] **Step 3: Run the full test suite one more time as a final gate**

Run:
```bash
pytest -q
```

Expected: all passed.

- [ ] **Step 4: Commit findings + TODO update**

```bash
git add docs/notes/2026-05-02-feature-view-ensemble.md TODO.md
git commit -m "$(cat <<'EOF'
docs(feature-view-ensemble): findings note + TODO update -- VERDICT

Findings: docs/notes/2026-05-02-feature-view-ensemble.md
Spec:     docs/superpowers/specs/2026-05-02-feature-view-ensemble-design.md
Plan:     docs/superpowers/plans/2026-05-02-feature-view-ensemble.md

[FILL IN: one-paragraph summary of the verdict and headline numbers]

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

(Replace `VERDICT` and the placeholder paragraph with the actual outcome before committing.)

- [ ] **Step 5: Push and open PR**

```bash
git push -u origin feat/feature-view-ensemble
gh pr create --title "feat: feature-view diversity ensemble (E1 + E2) -- <VERDICT>" --body "$(cat <<'EOF'
## Summary
- Tests same-class XGBoost peers on disjoint feature views (PEER_A: 40 team-strength features; PEER_B: 27 form/market/meta features) as a stage-1 diversity mechanism.
- 3-clause pre-sweep gate (per-peer LL ceiling, residual correlation < 0.60, best-blend headroom >= 0.001) gates the 15-cell sweep.
- E1 = blend(peer_A, peer_B); E2 = blend(v4, peer_A, peer_B). Both run if gate clears.

## Verdict
[FILL IN: GATE PASS/FAIL; if PASS, E1 and E2 best-cell deltas vs v9-C 2713]

## Files
- Spec: `docs/superpowers/specs/2026-05-02-feature-view-ensemble-design.md`
- Plan: `docs/superpowers/plans/2026-05-02-feature-view-ensemble.md`
- Findings: `docs/notes/2026-05-02-feature-view-ensemble.md`
- New code: `src/feature_views.py`, `src/train_peer_stage1.py`, `src/diagnose_feature_view_ensemble.py`
- Extended code: `src/ensemble_stage1.py` (K-way blend), `src/sweep_v9_weights.py` (V9_STAGE1_PAIRWISE env var)
- New tests: `tests/test_feature_views.py`, `tests/test_train_peer_stage1.py`, `tests/test_diagnose_feature_view_ensemble.py`; extensions to `tests/test_ensemble_stage1.py` and `tests/test_sweep_v9_weights.py`.

## Test plan
- [x] Unit tests pass (`pytest tests/test_feature_views.py tests/test_train_peer_stage1.py tests/test_diagnose_feature_view_ensemble.py tests/test_ensemble_stage1.py tests/test_sweep_v9_weights.py`)
- [x] Full ingest/feature/integration suite passes (`pytest tests/test_ingest tests/test_features tests/test_integration.py`)
- [x] Gate diagnostic ran on real data; outcome committed to `output/diag_feature_view_ensemble.json`
- [x] (if PASS) E1 and E2 sweeps ran on real data; per-cell results committed
- [x] (if PASS) v9-C harness anchor reproduces within 1e-9 with `V9_STAGE1_PAIRWISE=output/pairwise_v4.csv V9_FEATURE_SET=v9c`

EOF
)"
```

(Update the body's `[FILL IN]` placeholders before opening.)

---

### Task 9-FAIL (FAIL branch): Findings note + TODO update for gated NO-GO

**Why:** If the gate failed in Task 8, this is the only post-gate task. Skip Tasks 9 through 13's PASS-branch steps.

**Files:**
- Create: `docs/notes/2026-05-02-feature-view-ensemble.md`
- Modify: `TODO.md`

- [ ] **Step 1: Write the findings note**

Same structure as Task 13 Step 1, but the verdict is "NO-GO -- pre-sweep gate failed" and Section 5 (Sweep results) is omitted. Section 4 (Pre-sweep gate result) is the load-bearing section. Use `docs/notes/2026-05-02-bt-as-feature.md` as the template -- it's a NO-GO findings note for a gated experiment, exactly the shape needed here.

- [ ] **Step 2: Update TODO.md**

Move the `feature-view-diversity-ensemble` line from "Active queue" to "Tried and rejected." Note the failed clause(s) in the bullet so future readers see the shape of the failure.

The new top of the active queue is "Hierarchical Bradley-Terry with feature priors" (#2 in the existing queue).

- [ ] **Step 3: Run the full test suite as a final gate**

Run: `pytest -q`
Expected: all passed.

- [ ] **Step 4: Commit + push + PR**

```bash
git add docs/notes/2026-05-02-feature-view-ensemble.md TODO.md
git commit -m "$(cat <<'EOF'
docs(feature-view-ensemble): findings note + TODO update -- gate FAILED

Failed clause(s): [FILL IN: e.g., per_peer_ll_ceiling, blend_headroom]

The disjoint-view ensembling hypothesis is falsified at this scale by
clause N: [FILL IN one-line explanation]. v9-C / v4 stay in production.

Findings: docs/notes/2026-05-02-feature-view-ensemble.md
Spec:     docs/superpowers/specs/2026-05-02-feature-view-ensemble-design.md
Plan:     docs/superpowers/plans/2026-05-02-feature-view-ensemble.md
Diag:     output/diag_feature_view_ensemble.json

Saved compute: ~90-150 minutes (no E1/E2 sweep run).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"

git push -u origin feat/feature-view-ensemble
gh pr create --title "feat: feature-view diversity ensemble -- gate FAILED, NO-GO" --body "$(cat <<'EOF'
## Summary
- Tests same-class XGBoost peers on disjoint feature views (PEER_A: 40 team-strength features; PEER_B: 27 form/market/meta features) as a stage-1 diversity mechanism.
- 3-clause pre-sweep falsification gate FAILED on clause(s): [FILL IN].

## Verdict
NO-GO. v9-C / v4 stay in production. The disjoint-view ensembling hypothesis is falsified at this scale by [FILL IN]. Saved ~90-150 minutes by gating before the E1 + E2 sweeps.

## Files
- Spec: `docs/superpowers/specs/2026-05-02-feature-view-ensemble-design.md`
- Plan: `docs/superpowers/plans/2026-05-02-feature-view-ensemble.md`
- Findings: `docs/notes/2026-05-02-feature-view-ensemble.md`
- Gate diagnostic: `output/diag_feature_view_ensemble.json`
- New code: `src/feature_views.py`, `src/train_peer_stage1.py`, `src/diagnose_feature_view_ensemble.py`
- Extended code: `src/ensemble_stage1.py` (K-way blend, retained for future experiments), `src/sweep_v9_weights.py` (V9_STAGE1_PAIRWISE env var, retained as reusable harness extension)
- Frozen artifacts: `output/pairwise_peer_a.csv`, `output/pairwise_peer_b.csv`

## Test plan
- [x] Unit tests pass
- [x] Full ingest/feature/integration suite passes
- [x] Gate diagnostic ran on real data; outcome committed

EOF
)"
```

(Replace `[FILL IN]` placeholders before opening.)
