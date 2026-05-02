# Bayesian / Bradley-Terry Stage-1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a per-season Bradley-Terry stage-1 model (binary outcomes, regular-season games only) and gate any expensive backtest on a cheap residual-correlation + ideal-weight diagnostic vs v4. If the gate clears, run the v9-C correction step and the 22-season bracket-points head-to-head; otherwise stop and write findings.

**Architecture:** Per-season MAP Bradley-Terry implemented as `sklearn.linear_model.LogisticRegression` with team-indicator + home-court design matrix on `MRegularSeasonCompactResults.csv`. New scripts `src/train_bt_stage1.py` and `src/diagnose_bt_vs_v4.py` plus a conditional re-use of PR 11's ensemble / v9-C / scoring tooling. Falsification-first: 30-second diagnostic runs before any 3-hour backtest.

**Tech Stack:** Python 3.11+, scikit-learn (`LogisticRegression`), scipy.sparse (CSR matrix), numpy, pandas, pytest. No new top-level dependencies.

**Spec:** `docs/superpowers/specs/2026-05-01-bayesian-stage1-design.md`

**Reference reads (skim before starting):**
- `data/raw/march-machine-learning-2026/MRegularSeasonCompactResults.csv` -- schema is `Season,DayNum,WTeamID,WScore,LTeamID,LScore,WLoc,NumOT`. ~199k rows total across all seasons. The trainer loads this filtered to one season at a time.
- `output/pairwise_v4.csv` -- the canonical-format reference with schema `season,team_a,team_b,p_a_wins` where `team_a < team_b`. Has duplicates from default+tuned LOSO passes; consumers dedup last-write-wins.
- `data/raw/march-machine-learning-2026/MNCAATourneyCompactResults.csv` -- played tournament games. Used by the diagnostic to compute residuals and to enumerate the per-season tournament field.
- The LR experiment's findings note `docs/notes/2026-05-01-ensemble-stage1.md` (on PR 11 / `feat/ensemble-stage1` -- not yet on `main`) -- the diagnostic numbers (r=0.77, optimal w=0.93, headroom 0.0006) are the falsification baseline this experiment is gated against.

**PR 11 dependency note:** Tasks 1-4 do not depend on any code from PR 11. Task 5 (the conditional v9-C bracket-points head-to-head) reuses three modules that live on PR 11 / `feat/ensemble-stage1`: `src/ensemble_stage1.py`, `src/run_v9c_on_stage1.py`, `src/eval_stage1.py`. If PR 11 has merged into `main` by Task 5 time, rebase this branch onto `main` and they appear automatically. If PR 11 is still open or rejected, Task 5's first step cherry-picks those three commits. Either way, do not block Tasks 1-4 on PR 11.

**Verification gates (CLAUDE.md "Forced Verification"):**
After every code-level task: `pytest -v` at repo root must pass. After data-generation tasks (2, 4, conditional 5b/5c): inspect schema and row count of the produced CSV before committing.

---

### Task 1: Build `src/train_bt_stage1.py` (per-season BT trainer)

**Why:** Per-season Bradley-Terry MAP via L2 logistic regression with team-indicator + home-court features. Spec section "Per-season Bradley-Terry by MAP / L2 logistic regression."

**Files:**
- Create: `src/train_bt_stage1.py`
- Test: `tests/test_train_bt_stage1.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_train_bt_stage1.py`:

```python
"""Unit tests for src/train_bt_stage1.py.

Per-season Bradley-Terry trainer: design-matrix builder, home-court
extraction, and a synthetic-data recovery check. Full 22-season run
is exercised in Task 2, not here.
"""
import numpy as np
import pandas as pd
import pytest


def test_extract_home_court_value():
    """WLoc -> home-court column value: H -> +1 (winner home), A -> -1
    (winner away), N -> 0 (neutral)."""
    from src.train_bt_stage1 import extract_home_court_value

    assert extract_home_court_value("H") == 1
    assert extract_home_court_value("A") == -1
    assert extract_home_court_value("N") == 0


def test_build_design_matrix_shape():
    """3 teams, 4 games -> design matrix shape (4, 3+1)."""
    from src.train_bt_stage1 import build_design_matrix

    games = pd.DataFrame([
        {"WTeamID": 1101, "LTeamID": 1102, "WLoc": "H"},
        {"WTeamID": 1102, "LTeamID": 1103, "WLoc": "A"},
        {"WTeamID": 1101, "LTeamID": 1103, "WLoc": "N"},
        {"WTeamID": 1103, "LTeamID": 1101, "WLoc": "H"},
    ])
    team_ids = [1101, 1102, 1103]

    X, y = build_design_matrix(games, team_ids)

    # X: (4 games, 3 teams + 1 home-court column)
    assert X.shape == (4, 4)
    # y: all 1s (we encode one row per game with winner=+1)
    assert (y == 1).all()
    # Convert sparse to dense for inspection.
    X_dense = X.toarray()
    # Game 0: 1101 beats 1102 at home -> winner_col=0, loser_col=1, hc=+1
    assert X_dense[0, 0] == 1 and X_dense[0, 1] == -1 and X_dense[0, 3] == 1
    # Game 1: 1102 beats 1103 away -> winner_col=1, loser_col=2, hc=-1
    assert X_dense[1, 1] == 1 and X_dense[1, 2] == -1 and X_dense[1, 3] == -1
    # Game 2: neutral -> hc=0
    assert X_dense[2, 0] == 1 and X_dense[2, 2] == -1 and X_dense[2, 3] == 0


def test_recover_strengths_synthetic():
    """With teams of true strengths s = (0, 1, 2) and many simulated
    games, the fitted strengths should be in the right ORDER and within
    ~0.5 of the true differences after L2 regularization shrinks them."""
    from src.train_bt_stage1 import build_design_matrix, fit_bradley_terry

    rng = np.random.default_rng(42)
    true_s = np.array([0.0, 1.0, 2.0])
    n_games = 4000
    rows = []
    for _ in range(n_games):
        i, j = rng.choice(3, size=2, replace=False)
        # P(i beats j) = sigmoid(s_i - s_j); home-court irrelevant here.
        p = 1.0 / (1.0 + np.exp(-(true_s[i] - true_s[j])))
        winner = i if rng.random() < p else j
        loser = j if winner == i else i
        rows.append({
            "WTeamID": 1101 + winner,
            "LTeamID": 1101 + loser,
            "WLoc": "N",
        })
    games = pd.DataFrame(rows)
    team_ids = [1101, 1102, 1103]

    X, y = build_design_matrix(games, team_ids)
    coefs = fit_bradley_terry(X, y, C=10.0)
    s_fit = coefs[:3]
    # Order: team 0 weakest, team 2 strongest.
    assert s_fit[0] < s_fit[1] < s_fit[2]
    # Pairwise differences within ~0.5 of truth.
    assert abs((s_fit[2] - s_fit[0]) - 2.0) < 0.5
    assert abs((s_fit[1] - s_fit[0]) - 1.0) < 0.5


def test_predict_pairwise_for_field():
    """Given fitted strengths, predict_pairwise produces P(a beats b) =
    sigmoid(s_a - s_b) for every unordered pair (a < b) in the field."""
    from src.train_bt_stage1 import predict_pairwise_for_field

    team_ids = [1101, 1102, 1103]
    s = np.array([0.0, 1.0, 2.0])
    field = [1101, 1102, 1103]

    rows = predict_pairwise_for_field(season=2003, field=field,
                                       team_ids=team_ids, strengths=s)
    df = pd.DataFrame(rows)
    assert list(df.columns) == ["season", "team_a", "team_b", "p_a_wins"]
    assert len(df) == 3  # 3 unordered pairs from 3 teams
    assert (df["team_a"] < df["team_b"]).all()
    # P(1101 beats 1102) = sigmoid(0 - 1) = ~0.269
    p_1101_1102 = float(df[(df.team_a == 1101) & (df.team_b == 1102)].p_a_wins.iloc[0])
    assert abs(p_1101_1102 - 1.0 / (1.0 + np.exp(1.0))) < 1e-6
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_train_bt_stage1.py -v`
Expected: 4 ImportError failures (`No module named 'src.train_bt_stage1'`).

- [ ] **Step 3: Implement `src/train_bt_stage1.py`**

Create `src/train_bt_stage1.py`:

```python
"""Per-season Bradley-Terry stage-1 trainer.

Spec: docs/superpowers/specs/2026-05-01-bayesian-stage1-design.md

For each season Y in 2003-2025, fits team strengths from Y's regular-
season games via L2-regularized logistic regression with team-indicator
+ home-court design matrix. Mathematically equivalent to MAP Bradley-
Terry under a Gaussian prior on strengths.

The fit is per-season -- team strengths are season-specific parameters
fit on season-specific data. No cross-season learning, no leakage from
tournament outcomes (we only use regular-season games).

Output: appends to output/pairwise_bt.csv with rows
    (season, team_a, team_b, p_a_wins),  team_a < team_b
covering all unordered pairs of tournament-field teams in each held-
out season. Same schema as pairwise_v4.csv.
"""
import argparse
import math
import sys
import time
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.linear_model import LogisticRegression

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

DATA = Path("data/raw/march-machine-learning-2026")
DEFAULT_PAIRWISE_OUT = "output/pairwise_bt.csv"
DEFAULT_C = 10.0
SEASONS = list(range(2003, 2026))  # 2003..2025; 2020 absent in data


def extract_home_court_value(wloc: str) -> int:
    """WLoc -> home-court column value relative to the *winner*:
        H -> +1 (winner was home)
        A -> -1 (winner was away)
        N ->  0 (neutral)
    """
    if wloc == "H":
        return 1
    if wloc == "A":
        return -1
    return 0


def build_design_matrix(
    games: pd.DataFrame, team_ids: Sequence[int]
) -> tuple[sp.csr_matrix, np.ndarray]:
    """Build (X, y) for L2-LR Bradley-Terry.

    games: DataFrame with WTeamID, LTeamID, WLoc columns.
    team_ids: ordered list of team IDs; column index = team_ids.index(tid).

    Each game is one row. The row has +1 in the winner's column,
    -1 in the loser's column, and the home-court signal in the final
    column (extract_home_court_value(WLoc)). Label y = 1 (we always
    encode the +1-winner perspective).
    """
    team_idx = {int(tid): i for i, tid in enumerate(team_ids)}
    n_teams = len(team_ids)
    n_games = len(games)
    n_cols = n_teams + 1

    rows = np.empty(3 * n_games, dtype=np.int64)
    cols = np.empty(3 * n_games, dtype=np.int64)
    vals = np.empty(3 * n_games, dtype=np.float64)

    for k, (_, g) in enumerate(games.iterrows()):
        w, l = int(g["WTeamID"]), int(g["LTeamID"])
        wloc = g["WLoc"]
        rows[3 * k]     = k
        cols[3 * k]     = team_idx[w]
        vals[3 * k]     = 1.0
        rows[3 * k + 1] = k
        cols[3 * k + 1] = team_idx[l]
        vals[3 * k + 1] = -1.0
        rows[3 * k + 2] = k
        cols[3 * k + 2] = n_teams
        vals[3 * k + 2] = float(extract_home_court_value(wloc))

    X = sp.csr_matrix((vals, (rows, cols)), shape=(n_games, n_cols))
    y = np.ones(n_games, dtype=np.int64)
    return X, y


def fit_bradley_terry(
    X: sp.csr_matrix, y: np.ndarray, C: float = DEFAULT_C
) -> np.ndarray:
    """Fit L2 logistic regression and return the coefficient vector.

    Returns array of length n_cols = (n_teams + 1).
    Indices 0..n_teams-1: per-team strengths.
    Index n_teams: home-court coefficient.
    """
    # n_classes=2 with labels [0, 1]: y is all 1s, but the all-1 case
    # yields a degenerate fit. Inject a single artificial label-0 row
    # at the *all-zero* feature vector so LogisticRegression is happy
    # without distorting any team's evidence.
    # (sklearn refuses to fit with a single class. The zero row + C=10
    # adds negligible bias to the actual coefficients.)
    n_cols = X.shape[1]
    zero_row = sp.csr_matrix((1, n_cols))
    X_aug = sp.vstack([X, zero_row], format="csr")
    y_aug = np.concatenate([y, [0]])

    model = LogisticRegression(
        penalty="l2",
        solver="lbfgs",
        fit_intercept=False,
        C=C,
        max_iter=2000,
    )
    model.fit(X_aug, y_aug)
    return model.coef_.ravel()


def predict_pairwise_for_field(
    season: int,
    field: Iterable[int],
    team_ids: Sequence[int],
    strengths: np.ndarray,
) -> List[dict]:
    """For each unordered pair (a, b) in the field with a < b, compute
    p_a_wins = sigmoid(s_a - s_b). NO home-court term -- tournament is
    neutral. Returns a list of dict rows ready for pd.DataFrame.

    Teams in the field but missing from team_ids (e.g., a team that
    appears in the tournament but never played a regular-season game --
    extremely rare but possible at the edge) are skipped.
    """
    team_idx = {int(tid): i for i, tid in enumerate(team_ids)}
    field_sorted = sorted(set(int(t) for t in field if int(t) in team_idx))
    rows = []
    for i in range(len(field_sorted)):
        for j in range(i + 1, len(field_sorted)):
            a, b = field_sorted[i], field_sorted[j]
            s_a = strengths[team_idx[a]]
            s_b = strengths[team_idx[b]]
            p = 1.0 / (1.0 + math.exp(-(s_a - s_b)))
            rows.append({
                "season": season,
                "team_a": a,
                "team_b": b,
                "p_a_wins": p,
            })
    return rows


def run_bt_loso(
    out_csv: str = DEFAULT_PAIRWISE_OUT,
    C: float = DEFAULT_C,
    seasons: Iterable[int] = SEASONS,
) -> dict:
    """Per-season BT fits over the configured season range.

    Writes (season, team_a, team_b, p_a_wins) rows for each season's
    tournament field to out_csv (overwrites any existing file).
    Returns a summary dict with per-season metrics.
    """
    print("=" * 70)
    print("BRADLEY-TERRY STAGE-1 PER-SEASON TRAINER")
    print("=" * 70)
    print(f"  C={C}, out_csv={out_csv}")

    reg = pd.read_csv(DATA / "MRegularSeasonCompactResults.csv")
    seeds = pd.read_csv(DATA / "MNCAATourneySeeds.csv")
    results = pd.read_csv(DATA / "MNCAATourneyCompactResults.csv")

    if Path(out_csv).exists():
        Path(out_csv).unlink()
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)

    summary = []
    overall_start = time.time()

    for season in seasons:
        t0 = time.time()
        season_games = reg[reg["Season"] == season]
        if len(season_games) == 0:
            print(f"  [{season}] no regular-season games, skipping")
            continue

        team_ids = sorted(set(season_games["WTeamID"].astype(int)) |
                          set(season_games["LTeamID"].astype(int)))
        X, y = build_design_matrix(season_games, team_ids)
        coefs = fit_bradley_terry(X, y, C=C)
        strengths = coefs[: len(team_ids)]
        h_coef = float(coefs[len(team_ids)])

        # Field = teams that played in this season's tournament.
        season_results = results[results["Season"] == season]
        field = sorted(set(season_results["WTeamID"].astype(int)) |
                       set(season_results["LTeamID"].astype(int)))
        if not field:
            # Fall back to seeded teams if no tournament results yet.
            season_seeds = seeds[seeds["Season"] == season]
            field = sorted(season_seeds["TeamID"].astype(int).tolist())

        rows = predict_pairwise_for_field(season, field, team_ids, strengths)
        out_df = pd.DataFrame(rows)
        write_header = not Path(out_csv).exists()
        out_df.to_csv(out_csv, mode="a", index=False, header=write_header)

        # Per-season tournament log loss for visibility.
        ll, acc, n_eval = _score_tournament_games(
            season_results, dict(zip(team_ids, strengths))
        )
        summary.append({
            "season": season,
            "n_teams": len(team_ids),
            "n_games": len(season_games),
            "n_pairs_written": len(rows),
            "n_eval_games": n_eval,
            "h_coef": h_coef,
            "log_loss": ll,
            "accuracy": acc,
            "fold_seconds": round(time.time() - t0, 1),
        })
        print(f"  [{season}] teams={len(team_ids):>3} games={len(season_games):>5} "
              f"h={h_coef:>+5.3f} ll={ll:.4f} acc={acc:.3f} "
              f"pairs={len(rows):>5} ({time.time() - t0:.1f}s)")

    overall = time.time() - overall_start
    print(f"\nDONE in {overall:.1f}s; pairwise CSV: {out_csv}")
    return {"per_season": pd.DataFrame(summary), "out_csv": out_csv}


def _score_tournament_games(
    results: pd.DataFrame, strengths_by_id: dict
) -> tuple[float, float, int]:
    """Per-season tournament log loss + accuracy from fitted strengths.
    Returns (log_loss, accuracy, n_games_evaluated).
    """
    eps = 1e-15
    ll_terms = []
    correct = 0
    for _, g in results.iterrows():
        w, l = int(g["WTeamID"]), int(g["LTeamID"])
        s_w = strengths_by_id.get(w)
        s_l = strengths_by_id.get(l)
        if s_w is None or s_l is None:
            continue
        p_w = 1.0 / (1.0 + math.exp(-(s_w - s_l)))
        p_w = min(max(p_w, eps), 1 - eps)
        ll_terms.append(-math.log(p_w))
        correct += 1 if p_w > 0.5 else 0
    n = len(ll_terms)
    if n == 0:
        return float("nan"), float("nan"), 0
    return float(np.mean(ll_terms)), float(correct / n), n


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", default=DEFAULT_PAIRWISE_OUT)
    parser.add_argument("--c", type=float, default=DEFAULT_C,
                        help=f"L2 inverse-regularization (default: {DEFAULT_C})")
    args = parser.parse_args(argv)
    run_bt_loso(out_csv=args.out, C=args.c)
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

NOTE on the `fit_bradley_terry` zero-row trick: scikit-learn's
`LogisticRegression` cannot fit when `y` has only one unique class.
Since we encode every row from the +1-winner perspective, `y` is all
1s. Adding a single row of all zeros with label 0 makes sklearn
happy without distorting team strengths -- the zero row contributes
no evidence about any team or about home-court (all features are 0)
and only nudges the global intercept-equivalent toward 0, which we
don't have an intercept for anyway. With `C=10.0` regularization,
the bias from this row is negligible (~1/n_games of a real game's
contribution). The synthetic-data recovery test in Step 1 verifies
the recovered strengths are still within ~0.5 of truth.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_train_bt_stage1.py -v`
Expected: 4 PASSED.

- [ ] **Step 5: Run a 2-season smoke (no commit)**

```bash
python -c "
import sys; sys.path.insert(0, '.')
from src.train_bt_stage1 import run_bt_loso
out = run_bt_loso(out_csv='output/_smoke_pairwise_bt.csv', seasons=[2003, 2004])
import pandas as pd
df = pd.read_csv('output/_smoke_pairwise_bt.csv')
assert list(df.columns) == ['season','team_a','team_b','p_a_wins'], df.columns.tolist()
assert set(df['season'].unique()) == {2003, 2004}, df['season'].unique()
assert (df['team_a'] < df['team_b']).all()
assert df['p_a_wins'].between(0, 1).all()
print(f'smoke OK: {len(df)} rows over 2 seasons')
"
rm output/_smoke_pairwise_bt.csv
```

Expected: `smoke OK: <N> rows over 2 seasons` where N is around 4000-5000 (2 seasons x ~2000-2300 unique pairs each).

- [ ] **Step 6: Run full test suite**

Run: `pytest -v --ignore=tests/test_prepare_loso_inputs.py 2>&1 | tail -5`
Expected: all green.

(Skip the slow `test_prepare_loso_inputs.py` which only exists on PR 11 anyway -- it lives on `feat/ensemble-stage1` and not on this branch.)

- [ ] **Step 7: Commit**

```bash
git add src/train_bt_stage1.py tests/test_train_bt_stage1.py
git commit -m "$(cat <<'EOF'
feat(bt): src/train_bt_stage1.py + tests

Per-season Bradley-Terry stage-1 trainer. Implements MAP estimation
via L2-regularized logistic regression with team-indicator + home-
court design matrix on regular-season games. Per-season parameters,
no cross-season learning, no leakage from tournament outcomes.

Key choices documented in spec:
- C=10.0 (mild L2; tunable)
- Zero-row trick to satisfy sklearn's two-class requirement when
  every game encodes the +1-winner perspective
- fit_intercept=False, identifiability handled by L2 prior
- Home-court column from WLoc; tournament predictions drop it

Smoke + unit tests pass; full 22-season run lands in Task 2.

Spec: docs/superpowers/specs/2026-05-01-bayesian-stage1-design.md
EOF
)"
```

---

### Task 2: Run BT trainer end-to-end and commit `output/pairwise_bt.csv`

**Why:** Generate the BT pairwise predictions for all 22 seasons. Long-running step (~5-10 min based on the size of regular-season data per season). Committed because it is the input to the diagnostic gate in Task 4.

**Files:**
- Generate: `output/pairwise_bt.csv`

- [ ] **Step 1: Run the trainer**

Run: `python src/train_bt_stage1.py --out output/pairwise_bt.csv 2>&1 | tee /tmp/bt_run.log`
Expected: per-season `[YYYY] teams=N games=M h=+0.XXX ll=X.XXXX acc=X.XXX pairs=N (Ts)` lines for 2003-2025 (excluding 2020 if missing). Final summary `DONE in Xs`.

Sanity check the printed `h` (home-court) coefficients: each should be roughly in the range `+0.30` to `+0.50` (corresponding to a ~7-12% home-win-rate uplift). If you see negative or very large values, stop and inspect.

- [ ] **Step 2: Sanity-check the output**

```bash
python -c "
import pandas as pd
bt = pd.read_csv('output/pairwise_bt.csv')
print(f'bt: {len(bt):,} rows, {bt.season.nunique()} seasons')
assert list(bt.columns) == ['season','team_a','team_b','p_a_wins'], bt.columns.tolist()
assert (bt['team_a'] < bt['team_b']).all()
assert bt['p_a_wins'].between(0, 1).all()
print(f'min p_a_wins: {bt.p_a_wins.min():.4f}; max: {bt.p_a_wins.max():.4f}')
"
```

Expected: ~48,000 rows over 22 seasons (the field per season is ~64-68 teams; ~2000-2300 unique pairs; ~22 seasons). All probabilities in [0, 1].

- [ ] **Step 3: Pair-coverage match against pairwise_v4.csv (skip if pairwise_v4.csv not present)**

```bash
if [ -f output/pairwise_v4.csv ]; then
python -c "
import pandas as pd
v4 = pd.read_csv('output/pairwise_v4.csv').drop_duplicates(['season','team_a','team_b'], keep='last')
bt = pd.read_csv('output/pairwise_bt.csv')
print(f'v4 unique pairs: {len(v4):,}')
print(f'bt unique pairs: {len(bt):,}')
v4_keys = set(zip(v4.season, v4.team_a, v4.team_b))
bt_keys = set(zip(bt.season, bt.team_a, bt.team_b))
only_v4 = v4_keys - bt_keys
only_bt = bt_keys - v4_keys
print(f'only in v4: {len(only_v4)}; only in bt: {len(only_bt)}')
if only_v4 or only_bt:
    print(f'  example only_v4: {list(only_v4)[:3]}')
    print(f'  example only_bt: {list(only_bt)[:3]}')
"
else
  echo 'pairwise_v4.csv not present locally; coverage check deferred to Task 4 diagnostic.'
fi
```

Expected: zero pairs only in one side. If non-zero, diagnose before committing -- likely a play-in / First Four edge case in field enumeration. Acceptable resolution: enumerate the field from `MNCAATourneySeeds` (seeded teams) instead of `MNCAATourneyCompactResults` (played teams). Adjust `train_bt_stage1.py:run_bt_loso` field construction and re-run.

- [ ] **Step 4: Commit the CSV**

```bash
git add -f output/pairwise_bt.csv
git commit -m "$(cat <<'EOF'
data(bt): output/pairwise_bt.csv (22-season BT LOSO)

Generated by src/train_bt_stage1.py over seasons 2003-2025 (excluding
2020 implicitly via missing data). Schema matches pairwise_v4.csv:
season, team_a, team_b, p_a_wins (team_a < team_b).

Per-season fits: each season's strengths trained on that season's
regular-season games only, no cross-season leakage. Used by
src/diagnose_bt_vs_v4.py in Task 4 as the input to the gate
diagnostic.

Force-added (output/ is gitignored), mirroring the tracked
output/pairwise_v9.csv precedent.
EOF
)"
```

---

### Task 3: Build `src/diagnose_bt_vs_v4.py` (gate diagnostic)

**Why:** The falsification gate. Computes residual correlation, ideal-weight log-loss search, and a verdict line in seconds. If the gate fails, no v9-C compute happens.

**Files:**
- Create: `src/diagnose_bt_vs_v4.py`
- Test: `tests/test_diagnose_bt_vs_v4.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_diagnose_bt_vs_v4.py`:

```python
"""Unit tests for src/diagnose_bt_vs_v4.py."""
import math

import numpy as np
import pandas as pd
import pytest


def _write_pairwise(path, rows):
    df = pd.DataFrame(rows, columns=["season", "team_a", "team_b", "p_a_wins"])
    df.to_csv(path, index=False)


def test_compute_diagnostic_known_values(tmp_path):
    """Two games per season, known probabilities. Both models predict
    perfectly correlated residuals -> r close to +1; optimal weight
    indeterminate (any weighting gives same log loss when both are
    equally good)."""
    from src.diagnose_bt_vs_v4 import compute_diagnostic

    pw_a = tmp_path / "a.csv"
    pw_b = tmp_path / "b.csv"
    _write_pairwise(pw_a, [
        (2003, 1101, 1102, 0.9),
        (2003, 1103, 1104, 0.4),
    ])
    _write_pairwise(pw_b, [
        (2003, 1101, 1102, 0.9),
        (2003, 1103, 1104, 0.4),
    ])
    results = pd.DataFrame([
        {"Season": 2003, "WTeamID": 1101, "LTeamID": 1102, "DayNum": 136},
        {"Season": 2003, "WTeamID": 1104, "LTeamID": 1103, "DayNum": 136},
    ])

    out = compute_diagnostic(str(pw_a), str(pw_b), results_df=results)
    # Identical predictors -> residual correlation = 1.
    assert out["r_residual"] == pytest.approx(1.0)
    # Both correct: agreement on predicted winner.
    assert out["disagree_n"] == 0
    # Optimal weight: any value gives same loss; picker should still
    # pick something deterministic. Just assert it's in [0, 1].
    assert 0.0 <= out["optimal_w"] <= 1.0


def test_optimal_weight_picks_better_model(tmp_path):
    """If A is much better than B on the only game, optimal weight is
    biased toward A (close to 1.0)."""
    from src.diagnose_bt_vs_v4 import compute_diagnostic

    pw_a = tmp_path / "a.csv"
    pw_b = tmp_path / "b.csv"
    # 5 games where A is consistently right and B consistently wrong.
    _write_pairwise(pw_a, [
        (2003, 1100 + i, 1200 + i, 0.95) for i in range(5)
    ])
    _write_pairwise(pw_b, [
        (2003, 1100 + i, 1200 + i, 0.20) for i in range(5)
    ])
    results = pd.DataFrame([
        {"Season": 2003, "WTeamID": 1100 + i, "LTeamID": 1200 + i, "DayNum": 136}
        for i in range(5)
    ])

    out = compute_diagnostic(str(pw_a), str(pw_b), results_df=results)
    assert out["optimal_w"] >= 0.9, f"optimal_w should heavily favor A, got {out['optimal_w']}"


def test_gate_logic_each_clause(tmp_path):
    """check_gate flips on each clause individually."""
    from src.diagnose_bt_vs_v4 import check_gate

    # Pass case
    base = {"r_residual": 0.4, "optimal_w": 0.6, "headroom": 0.01}
    assert check_gate(base)["pass"] is True

    # Fail r
    diag = dict(base, r_residual=0.7)
    assert check_gate(diag)["pass"] is False
    assert "correlation" in check_gate(diag)["reason"].lower()

    # Fail optimal_w (degenerate v4-dominant)
    diag = dict(base, optimal_w=0.92)
    assert check_gate(diag)["pass"] is False
    assert "weight" in check_gate(diag)["reason"].lower()

    # Fail optimal_w (degenerate bt-dominant)
    diag = dict(base, optimal_w=0.05)
    assert check_gate(diag)["pass"] is False

    # Fail headroom
    diag = dict(base, headroom=0.001)
    assert check_gate(diag)["pass"] is False
    assert "headroom" in check_gate(diag)["reason"].lower()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_diagnose_bt_vs_v4.py -v`
Expected: 3 ImportError failures.

- [ ] **Step 3: Implement `src/diagnose_bt_vs_v4.py`**

Create `src/diagnose_bt_vs_v4.py`:

```python
"""Gate diagnostic: residual correlation + ideal-weight search of v4 vs BT.

Spec: docs/superpowers/specs/2026-05-01-bayesian-stage1-design.md

Loads the canonical pairwise CSVs (v4 dedup last-write-wins; BT as is),
joins to MNCAATourneyCompactResults.csv, and computes:
  - per-game residuals for both models
  - Pearson r(residual_v4, residual_bt)
  - cheating ideal-weight search: argmin_w mean(-log(w*p_v4 + (1-w)*p_bt))
  - disagreement breakdown
Then applies the gate's three clauses and prints a verdict.

Used as the falsification gate before any v9-C correction step or
22-season bracket-points backtest.
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path("data/raw/march-machine-learning-2026")
DEFAULT_DIAGNOSTIC_OUT = "output/diag_bt_vs_v4.json"

# Gate thresholds (from spec).
GATE_R_MAX = 0.60
GATE_W_LOW = 0.30
GATE_W_HIGH = 0.85
GATE_HEADROOM_MIN = 0.005


def _load_pairwise_lookup(path: str) -> dict:
    pw = pd.read_csv(path)
    pw = pw.drop_duplicates(subset=["season", "team_a", "team_b"], keep="last")
    return {(int(s), int(a), int(b)): float(p)
            for s, a, b, p in zip(pw.season, pw.team_a, pw.team_b, pw.p_a_wins)}


def compute_diagnostic(
    pairwise_v4_csv: str,
    pairwise_bt_csv: str,
    results_csv: str = str(DATA / "MNCAATourneyCompactResults.csv"),
    results_df: pd.DataFrame = None,
) -> dict:
    """Compute the gate diagnostic. Returns a dict with all numbers."""
    v4 = _load_pairwise_lookup(pairwise_v4_csv)
    bt = _load_pairwise_lookup(pairwise_bt_csv)
    if results_df is None:
        results_df = pd.read_csv(results_csv)

    v4_p_w, bt_p_w = [], []
    for _, g in results_df.iterrows():
        s, w, l = int(g["Season"]), int(g["WTeamID"]), int(g["LTeamID"])
        if s < 2003 or s > 2025:
            continue
        a, b = (w, l) if w < l else (l, w)
        p_v4 = v4.get((s, a, b))
        p_bt = bt.get((s, a, b))
        if p_v4 is None or p_bt is None:
            continue
        v4_p_w.append(p_v4 if a == w else 1.0 - p_v4)
        bt_p_w.append(p_bt if a == w else 1.0 - p_bt)

    v4_p_w = np.array(v4_p_w)
    bt_p_w = np.array(bt_p_w)
    n = len(v4_p_w)

    eps = 1e-15
    v4_clip = np.clip(v4_p_w, eps, 1 - eps)
    bt_clip = np.clip(bt_p_w, eps, 1 - eps)

    # Standalone log loss + accuracy
    ll_v4 = float(-np.mean(np.log(v4_clip)))
    ll_bt = float(-np.mean(np.log(bt_clip)))
    acc_v4 = float((v4_p_w > 0.5).mean())
    acc_bt = float((bt_p_w > 0.5).mean())

    # Residual correlation (1 - p_for_actual_winner)
    v4_res = 1 - v4_p_w
    bt_res = 1 - bt_p_w
    if n > 1 and v4_res.std() > 0 and bt_res.std() > 0:
        r_residual = float(np.corrcoef(v4_res, bt_res)[0, 1])
    else:
        r_residual = float("nan")

    # Disagreement
    v4_pick = (v4_p_w > 0.5).astype(int)
    bt_pick = (bt_p_w > 0.5).astype(int)
    disagree_n = int((v4_pick != bt_pick).sum())
    both_correct = int(((v4_pick == 1) & (bt_pick == 1)).sum())
    v4_only = int(((v4_pick == 1) & (bt_pick == 0)).sum())
    bt_only = int(((v4_pick == 0) & (bt_pick == 1)).sum())
    both_wrong = int(((v4_pick == 0) & (bt_pick == 0)).sum())

    # Ideal-weight search (cheating: tune on test outcomes)
    ws = np.linspace(0.0, 1.0, 101)
    ll_at_w = []
    for w in ws:
        p_blend = w * v4_p_w + (1 - w) * bt_p_w
        p_blend = np.clip(p_blend, eps, 1 - eps)
        ll_at_w.append(float(-np.mean(np.log(p_blend))))
    ll_at_w = np.array(ll_at_w)
    optimal_idx = int(np.argmin(ll_at_w))
    optimal_w = float(ws[optimal_idx])
    optimal_ll = float(ll_at_w[optimal_idx])
    headroom = ll_v4 - optimal_ll  # positive = ensemble beats v4 alone

    return {
        "n_games": n,
        "ll_v4": ll_v4,
        "ll_bt": ll_bt,
        "acc_v4": acc_v4,
        "acc_bt": acc_bt,
        "r_residual": r_residual,
        "disagree_n": disagree_n,
        "both_correct": both_correct,
        "v4_only_correct": v4_only,
        "bt_only_correct": bt_only,
        "both_wrong": both_wrong,
        "optimal_w": optimal_w,
        "optimal_ll": optimal_ll,
        "headroom": float(headroom),
        "ll_at_w": ll_at_w.tolist(),
    }


def check_gate(diag: dict) -> dict:
    """Apply the three-clause gate and return {pass, reason}."""
    failures = []
    if not (diag["r_residual"] < GATE_R_MAX):
        failures.append(
            f"residual correlation {diag['r_residual']:.3f} >= {GATE_R_MAX}"
        )
    if not (GATE_W_LOW <= diag["optimal_w"] <= GATE_W_HIGH):
        failures.append(
            f"optimal weight {diag['optimal_w']:.2f} outside "
            f"[{GATE_W_LOW}, {GATE_W_HIGH}]"
        )
    if not (diag["headroom"] > GATE_HEADROOM_MIN):
        failures.append(
            f"headroom {diag['headroom']:.4f} <= {GATE_HEADROOM_MIN}"
        )
    if failures:
        return {"pass": False, "reason": "; ".join(failures)}
    return {"pass": True, "reason": "all three clauses cleared"}


def print_report(diag: dict, gate: dict) -> None:
    print("=" * 70)
    print("BT vs v4 GATE DIAGNOSTIC")
    print("=" * 70)
    print(f"  n tournament games: {diag['n_games']}")
    print(f"\n  Standalone log loss:")
    print(f"    v4: {diag['ll_v4']:.4f}   acc: {diag['acc_v4']:.3f}")
    print(f"    BT: {diag['ll_bt']:.4f}   acc: {diag['acc_bt']:.3f}")
    print(f"\n  Pearson r(residual_v4, residual_bt) = {diag['r_residual']:.4f}")
    print(f"\n  Disagreement on predicted winner:")
    n = diag['n_games']
    print(f"    disagree:      {diag['disagree_n']}/{n} "
          f"({100*diag['disagree_n']/n:.1f}%)")
    print(f"    both correct:  {diag['both_correct']}/{n}")
    print(f"    v4 only:       {diag['v4_only_correct']}/{n}")
    print(f"    BT only:       {diag['bt_only_correct']}/{n}")
    print(f"    both wrong:    {diag['both_wrong']}/{n}")
    print(f"\n  Optimal-weight search (cheating; no LOSO):")
    print(f"    log loss at w=1.0 (v4):       {diag['ll_v4']:.4f}")
    print(f"    log loss at w=0.5:            {diag['ll_at_w'][50]:.4f}")
    print(f"    log loss at w=0.0 (BT):       {diag['ll_bt']:.4f}")
    print(f"    log loss at optimal w={diag['optimal_w']:.2f}: "
          f"{diag['optimal_ll']:.4f}")
    print(f"    headroom vs v4 alone:         {diag['headroom']:+.4f}")
    print(f"\n=== VERDICT ===")
    if gate["pass"]:
        print(f"  GATE PASSED: {gate['reason']}")
        print(f"  -> Proceed to v9-C correction + bracket-points backtest")
    else:
        print(f"  GATE FAILED: {gate['reason']}")
        print(f"  -> Stop. Write findings note. No v9-C compute.")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--pairwise-v4", default="output/pairwise_v4.csv")
    parser.add_argument("--pairwise-bt", default="output/pairwise_bt.csv")
    parser.add_argument("--out-json", default=DEFAULT_DIAGNOSTIC_OUT)
    args = parser.parse_args(argv)

    diag = compute_diagnostic(args.pairwise_v4, args.pairwise_bt)
    gate = check_gate(diag)
    print_report(diag, gate)

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w") as f:
        # Drop the per-w log-loss curve to keep the JSON small; keep
        # the headline numbers.
        slim = {k: v for k, v in diag.items() if k != "ll_at_w"}
        json.dump({"diagnostic": slim, "gate": gate}, f, indent=2)
    print(f"\n  saved {args.out_json}")
    return 0 if gate["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
```

NOTE: `main()` returns `0` if the gate passes and `1` if it fails. This
lets a downstream shell pipeline branch on `$?` if desired, but the
canonical use is to read the JSON and decide manually.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_diagnose_bt_vs_v4.py -v`
Expected: 3 PASSED.

- [ ] **Step 5: Run full test suite**

Run: `pytest -v --ignore=tests/test_prepare_loso_inputs.py 2>&1 | tail -5`
Expected: all green.

- [ ] **Step 6: Commit**

```bash
git add src/diagnose_bt_vs_v4.py tests/test_diagnose_bt_vs_v4.py
git commit -m "$(cat <<'EOF'
feat(bt): src/diagnose_bt_vs_v4.py + tests

Gate diagnostic for the BT vs v4 ensemble experiment. Loads the two
canonical pairwise CSVs, joins to tournament results, and computes:
  - standalone log loss + accuracy for both models
  - Pearson r(residual_v4, residual_bt)
  - disagreement breakdown
  - cheating ideal-weight search
  - three-clause gate verdict (r < 0.60, optimal w in [0.30, 0.85],
    headroom > 0.005)

Returns exit code 0 on pass / 1 on fail. Writes diagnostic JSON to
output/diag_bt_vs_v4.json.

Lesson from the LR experiment (PR 11): the same diagnostic on the
LR ensemble would have produced a NO-GO in 30 seconds (r=0.77,
w=0.93, headroom 0.0006). This module is the gate that prevents
that cycle from repeating.

Spec: docs/superpowers/specs/2026-05-01-bayesian-stage1-design.md
EOF
)"
```

---

### Task 4: Run the gate diagnostic; branch on the verdict

**Why:** Falsification step. Spends ~30 seconds determining whether the BT ensemble has a real shot. If the gate fails, this branch terminates with a NO-GO findings note (Task 6) and never burns v9-C / bracket-points compute.

**Files:**
- Generate: `output/diag_bt_vs_v4.json`

- [ ] **Step 1: Confirm `output/pairwise_v4.csv` is present locally**

```bash
ls -lh output/pairwise_v4.csv
```

If the file is missing, the diagnostic cannot run. Either:
- Wait for PR 11 to merge (which does not commit `pairwise_v4.csv` -- so this won't help; it isn't tracked anywhere).
- Generate it locally with `MM_PAIRWISE_OUT=output/pairwise_v4.csv python src/enhanced_model_v3.py` (~30+ minutes including Optuna tuning; you can interrupt after the default-params LOSO pass once `pairwise_v4.csv` has rows from all 22 seasons -- the Task 4 diagnostic only needs default-pass numbers).

- [ ] **Step 2: Run the diagnostic**

```bash
python src/diagnose_bt_vs_v4.py \
  --pairwise-v4 output/pairwise_v4.csv \
  --pairwise-bt output/pairwise_bt.csv \
  --out-json output/diag_bt_vs_v4.json \
  | tee /tmp/bt_diagnostic.log
```

Expected output: standalone numbers, residual correlation, disagreement breakdown, optimal-weight headroom, and the verdict line. Capture the verdict.

- [ ] **Step 3: Commit the diagnostic JSON**

```bash
git add -f output/diag_bt_vs_v4.json
git commit -m "$(cat <<'EOF'
data(bt): output/diag_bt_vs_v4.json -- gate diagnostic

Pre-flight diagnostic before any v9-C correction or bracket-points
backtest. Computed from output/pairwise_v4.csv (dedup'd) and
output/pairwise_bt.csv on the played 2003-2025 tournament games.

Reports residual correlation, disagreement, optimal-weight
headroom, and the gate verdict (pass / fail).
EOF
)"
```

- [ ] **Step 4: Branch on the verdict**

Read `/tmp/bt_diagnostic.log`'s "VERDICT" line:

- **GATE PASSED:** Proceed to Task 5 (the v9-C bracket-points head-to-head).
- **GATE FAILED:** Skip to Task 6 (write findings note + TODO update). The LR experiment's lesson is that running a 3-hour backtest after a failed gate just produces the same NO-GO with extra steps. Don't.

- [ ] **Step 5 (only if gate failed): record the falsification reason**

Note the specific failing clause(s) from the verdict line. They go directly into the findings note in Task 6 (e.g., "BT residual correlation 0.78 vs gate threshold 0.60" or "optimal weight 0.95 outside [0.30, 0.85]"). No code changes here -- the analysis is what the diagnostic JSON already captured.

---

### Task 5 (CONDITIONAL on Task 4 PASS): v9-C bracket-points head-to-head

**Skip this entire task if the Task 4 gate failed.** Proceed straight to Task 6.

**Why:** The decisive measurement. Mirrors the LR experiment's Tasks 5-9 pattern. Reuses PR 11's tooling (`ensemble_stage1.py`, `run_v9c_on_stage1.py`, `score_chalk_brackets.score_pairwise_path`).

**Files:**
- Possibly cherry-pick from `feat/ensemble-stage1`: `src/ensemble_stage1.py`, `src/run_v9c_on_stage1.py`, `src/eval_stage1.py`, and their tests.
- Generate: `output/pairwise_bt_ensemble.csv`, `output/pairwise_v9c_v4_baseline.csv` (if not already present from PR 11), `output/pairwise_v9c_bt_ensemble.csv`.

- [ ] **Step 1: Confirm or import the PR 11 modules**

```bash
ls src/ensemble_stage1.py src/run_v9c_on_stage1.py src/eval_stage1.py 2>&1
```

If all three are present (PR 11 has merged into main and you've rebased): proceed to Step 2.

If any are missing (PR 11 still open): cherry-pick the relevant commits from `feat/ensemble-stage1`:

```bash
git fetch origin feat/ensemble-stage1
# Pick the three feature commits (from `feat/ensemble-stage1`'s log):
#   d91f849 feat(ensemble): src/ensemble_stage1.py + tests
#   672c205 feat(ensemble): src/eval_stage1.py + tests
#   61ebd9c feat(ensemble): src/run_v9c_on_stage1.py + tests
#   f664b6b data(ensemble): output/pairwise_ensemble.csv (only if you also want the LR ensemble for comparison; usually skip)
git cherry-pick d91f849 672c205 61ebd9c
```

If the cherry-picks conflict, resolve in favor of the cherry-picked version (these are new files; conflicts shouldn't happen unless this branch added a same-named file). Verify with `pytest -v tests/test_ensemble_stage1.py tests/test_eval_stage1.py tests/test_run_v9c_on_stage1.py`.

- [ ] **Step 2: Read the optimal weight from the diagnostic JSON**

```bash
python -c "
import json
with open('output/diag_bt_vs_v4.json') as f:
    d = json.load(f)
w_v4 = d['diagnostic']['optimal_w']
w_bt = 1.0 - w_v4
print(f'OPTIMAL_W_V4={w_v4}')
print(f'OPTIMAL_W_BT={w_bt}')
"
```

Note: this weight was tuned on the test data without LOSO discipline. Acceptable for the head-to-head's first read since the verdict bands are coarse (+/- 25 brkt pts), but record this caveat in the findings note.

- [ ] **Step 3: Anchor checks (mandatory before head-to-head)**

```bash
# Anchor 1: --weights 1.0,0.0 reproduces dedup'd pairwise_v4.csv
python src/ensemble_stage1.py \
  --in-a output/pairwise_v4.csv \
  --in-b output/pairwise_bt.csv \
  --out output/_anchor_v4.csv \
  --weights 1.0,0.0
python -c "
import pandas as pd
v4 = pd.read_csv('output/pairwise_v4.csv').drop_duplicates(['season','team_a','team_b'], keep='last').sort_values(['season','team_a','team_b']).reset_index(drop=True)
chk = pd.read_csv('output/_anchor_v4.csv').sort_values(['season','team_a','team_b']).reset_index(drop=True)
diff = (v4['p_a_wins'] - chk['p_a_wins']).abs().max()
assert diff < 1e-9, f'anchor (1.0, 0.0) failed: {diff}'
print(f'anchor (1.0, 0.0) OK: max diff {diff:.2e}')
"
rm output/_anchor_v4.csv

# Anchor 2: --weights 0.0,1.0 reproduces pairwise_bt.csv
python src/ensemble_stage1.py \
  --in-a output/pairwise_v4.csv \
  --in-b output/pairwise_bt.csv \
  --out output/_anchor_bt.csv \
  --weights 0.0,1.0
python -c "
import pandas as pd
bt = pd.read_csv('output/pairwise_bt.csv').sort_values(['season','team_a','team_b']).reset_index(drop=True)
chk = pd.read_csv('output/_anchor_bt.csv').sort_values(['season','team_a','team_b']).reset_index(drop=True)
diff = (bt['p_a_wins'] - chk['p_a_wins']).abs().max()
assert diff < 1e-9, f'anchor (0.0, 1.0) failed: {diff}'
print(f'anchor (0.0, 1.0) OK: max diff {diff:.2e}')
"
rm output/_anchor_bt.csv
```

If either anchor fails: stop. Diagnose the join logic before trusting any head-to-head numbers.

- [ ] **Step 4: Generate the BT ensemble pairwise CSV**

Read the optimal weights from Step 2, then:

```bash
python src/ensemble_stage1.py \
  --in-a output/pairwise_v4.csv \
  --in-b output/pairwise_bt.csv \
  --out output/pairwise_bt_ensemble.csv \
  --weights <OPTIMAL_W_V4>,<OPTIMAL_W_BT>
```

Replace `<OPTIMAL_W_V4>` and `<OPTIMAL_W_BT>` with the values printed in Step 2 (e.g., `0.65,0.35`).

Verify schema:

```bash
python -c "
import pandas as pd
ens = pd.read_csv('output/pairwise_bt_ensemble.csv')
assert list(ens.columns) == ['season','team_a','team_b','p_a_wins']
assert ens['p_a_wins'].between(0, 1).all()
print(f'OK: {len(ens):,} rows')
"
```

- [ ] **Step 5: Run v9-C on v4 baseline (skip if `pairwise_v9c_v4_baseline.csv` already on this branch from PR 11)**

```bash
if [ -f output/pairwise_v9c_v4_baseline.csv ]; then
  echo "v4 baseline already exists; skipping"
else
  python src/run_v9c_on_stage1.py \
    --pairwise-in output/pairwise_v4.csv \
    --pairwise-out output/pairwise_v9c_v4_baseline.csv \
    2>&1 | tee /tmp/v9c_v4.log
fi
```

Expected: ~1-3 min runtime. ~48,465 rows in the output across 22 seasons.

- [ ] **Step 6: Run v9-C on the BT ensemble**

```bash
python src/run_v9c_on_stage1.py \
  --pairwise-in output/pairwise_bt_ensemble.csv \
  --pairwise-out output/pairwise_v9c_bt_ensemble.csv \
  2>&1 | tee /tmp/v9c_bt_ens.log
```

- [ ] **Step 7: Bracket-points head-to-head**

```bash
python -c "
from src.score_chalk_brackets import score_pairwise_path
import json

baseline = score_pairwise_path('output/pairwise_v9c_v4_baseline.csv')
experiment = score_pairwise_path('output/pairwise_v9c_bt_ensemble.csv')

print('=' * 70)
print('STAGE-1 + v9-C HEAD-TO-HEAD (BT ensemble vs v4 baseline)')
print('=' * 70)
print(f'{\"season\":>6}  {\"v4_base\":>10}  {\"bt_ens\":>10}  {\"delta\":>7}')
common = sorted(set(baseline['per_season_pts']) & set(experiment['per_season_pts']))
total_b, total_e = 0.0, 0.0
wins_b, wins_e, ties = 0, 0, 0
for s in common:
    b = baseline['per_season_pts'][s]
    e = experiment['per_season_pts'][s]
    total_b += b
    total_e += e
    if e > b: wins_e += 1
    elif b > e: wins_b += 1
    else: ties += 1
    print(f'{s:>6}  {b:>10.1f}  {e:>10.1f}  {e-b:>+7.1f}')
print('-' * 38)
print(f'{\"TOTAL\":>6}  {total_b:>10.1f}  {total_e:>10.1f}  {total_e-total_b:>+7.1f}')
print(f'\\nbt_ens W/L/T vs v4-baseline: {wins_e}/{wins_b}/{ties}')

delta = total_e - total_b
print(f'\\ntotal delta: {delta:+.1f} bracket points')
if delta >= 25:
    verdict = 'CLEAR WIN -- swap to BT ensemble (>= +25)'
elif delta >= 10:
    verdict = 'MARGINAL -- candidate (+10 to +25), do not swap'
else:
    verdict = 'NO-GO -- keep v4 (< +10)'
print(f'verdict: {verdict}')

with open('output/bt_head_to_head.json', 'w') as f:
    json.dump({
        'baseline': baseline,
        'experiment': experiment,
        'delta': delta,
        'verdict': verdict,
        'wins_b': wins_b, 'wins_e': wins_e, 'ties': ties,
    }, f, indent=2)
print('\\nsaved output/bt_head_to_head.json')
" 2>&1 | tee /tmp/bt_head_to_head.log
```

- [ ] **Step 8: Commit Task 5 artifacts**

```bash
git add -f output/pairwise_bt_ensemble.csv output/pairwise_v9c_bt_ensemble.csv output/bt_head_to_head.json
# Conditional: also add output/pairwise_v9c_v4_baseline.csv if it was generated above (not already present from PR 11):
test -f output/pairwise_v9c_v4_baseline.csv && git add -f output/pairwise_v9c_v4_baseline.csv

git commit -m "$(cat <<'EOF'
data(bt): v9-C bracket-points head-to-head, BT ensemble vs v4

output/pairwise_bt_ensemble.csv     -- (w_v4, w_bt) optimal blend
output/pairwise_v9c_bt_ensemble.csv -- v9-C on the BT ensemble
output/bt_head_to_head.json         -- per-season + total bracket-points

Verdict captured in /tmp/bt_head_to_head.log; falls into one of the
spec's verdict bands (>= +25 = clear win, +10 to +25 = marginal,
< +10 = no-go). Findings note in Task 6.
EOF
)"
```

---

### Task 6: Findings note + TODO update

**Why:** Closes the experiment with a written verdict. Always runs, regardless of whether the gate passed (the falsification is itself a finding worth documenting).

**Files:**
- Create: `docs/notes/2026-05-01-bayesian-stage1.md`
- Modify: `TODO.md`

- [ ] **Step 1: Draft the findings note**

Use `docs/notes/2026-05-01-ensemble-stage1.md` (on PR 11) as the template. Required sections:

1. Header (date, branch, predecessors).
2. **TL;DR** (the verdict in one line).
3. **Setup recap** (per-season BT, regular-season-only, binary outcomes, MAP via L2 LR with team-indicators + home court).
4. **Diagnostic gate result** (always present): r_residual, optimal_w, headroom, pass/fail. Cite `output/diag_bt_vs_v4.json`.
5. **(Conditional, only if gate passed) Stage-1 + v9-C bracket-points table** -- per-season + total + W/L/T from `output/bt_head_to_head.json`. Include the verdict band.
6. **(If gate failed) Falsification reasoning:** explicitly tie back to the LR experiment's diagnostic. Three possibilities to call out:
   - r >= 0.60 (BT errors still correlated with v4) -- "model-class diversity at independent training data also doesn't help."
   - optimal_w outside [0.30, 0.85] (BT degenerate) -- "BT alone too weak / too strong to contribute."
   - headroom <= 0.005 (no meaningful gain available) -- "even ideal blending can't recover bracket points."
7. **Verdict** (one of CLEAR WIN / MARGINAL / NO-GO).
8. **Recommendation:**
   - If CLEAR WIN: separate swap commit, regenerate canonical `pairwise_probs.json` with BT-ensemble + v9-C. Promote in TODO.
   - If MARGINAL: document as a candidate; v4 stays. Note the LOSO-disciplined weight follow-up.
   - If NO-GO: v4 stays. Promote either feature-view diversity (queue #3 in PR 11's TODO) or full-Bayesian (PyMC, latent strength + variance) as the next active queue item.
9. **Files of record** (mirroring the LR findings).

ASCII-only. Verify with `python -c "open('docs/notes/2026-05-01-bayesian-stage1.md', encoding='utf-8').read().encode('ascii')"`.

- [ ] **Step 2: Update TODO.md**

The TODO.md on this branch reflects pre-LR-experiment state (`feat/ensemble-stage1` rewrites it but PR 11 has not merged yet -- on `main`, item #1 is still "Ensemble of model classes"). Two cases:

- **PR 11 has merged into `main`:** the TODO already lists Bayesian as queue #1, NN as #2, feature-view diversity as #3. This branch's update marks "Bayesian / Bradley-Terry stage-1" as done with the verdict, citing the findings note. Renumber queue (NN -> #1, feature-view diversity -> #2).
- **PR 11 still open:** the TODO on this branch is still the pre-experiment version. This branch's update needs to add BOTH the LR rejection (mirror PR 11's TODO update) AND the BT result. Or, defer the TODO update until PR 11 merges and rebase. Cleanest: defer. Add a one-line note to the findings note: "TODO update lands once PR 11 merges and this branch rebases."

For all three verdict bands, the TODO entry should:
- State the gate result (r_residual, optimal_w, headroom).
- If the gate passed: state the bracket-points delta.
- Cite the findings note path.
- Mention that the BT trainer code stays on this branch as the experiment record.

- [ ] **Step 3: ASCII verification**

```bash
python -c "open('docs/notes/2026-05-01-bayesian-stage1.md', encoding='utf-8').read().encode('ascii')"
python -c "open('TODO.md', encoding='utf-8').read().encode('ascii')"
echo "ASCII OK"
```

- [ ] **Step 4: Run the full test suite one more time**

Run: `pytest -v --ignore=tests/test_prepare_loso_inputs.py 2>&1 | tail -5`
Expected: all green.

- [ ] **Step 5: Commit**

```bash
git add docs/notes/2026-05-01-bayesian-stage1.md
test -f TODO.md && git diff --quiet TODO.md || git add TODO.md
git commit -m "$(cat <<'EOF'
docs(bt): findings note + TODO update -- <VERDICT>

Per-season Bradley-Terry stage-1 (binary outcomes, regular-season
games only, MAP via L2 LR with team indicators + home court).
Diagnostic gate: r_residual=<X>, optimal_w=<X>, headroom=<X>.
Gate <PASS|FAIL>.

(If pass) Stage-1 + v9-C bracket-points: <delta> brkt pts vs v4
baseline over 22 LOSO seasons (W/L/T <a>/<b>/<c>). Verdict band:
<CLEAR WIN | MARGINAL | NO-GO>.

(If fail) Falsification reason: <which clause(s) failed>. The
diagnostic completed in seconds, saving ~3 hours of compute that
the LR experiment burned before learning the same lesson.

Findings: docs/notes/2026-05-01-bayesian-stage1.md
EOF
)"
```

- [ ] **Step 6: Push and open PR**

```bash
git push -u origin feat/bayesian-stage1
gh pr create --title "feat: Bayesian / Bradley-Terry stage-1 -- <VERDICT>" --body "$(cat <<'EOF'
## Summary

- Per-season Bradley-Terry stage-1 (binary outcomes, regular-season-only, MAP via L2 LR with team-indicators + home court). Disjoint training data + structurally different inductive bias from v4.
- Diagnostic gate: r(residual_v4, residual_bt) = <X>; optimal blend weight <w>; headroom <h>. Gate <PASS|FAIL>.
- (If gate passed) Bracket-points head-to-head: <delta> over 22 LOSO seasons. Verdict band: <CLEAR WIN | MARGINAL | NO-GO>.

## What landed

- `src/train_bt_stage1.py` -- per-season BT trainer + tests.
- `src/diagnose_bt_vs_v4.py` -- residual-correlation + ideal-weight gate + tests.
- `output/pairwise_bt.csv` -- 22-season BT predictions.
- `output/diag_bt_vs_v4.json` -- gate diagnostic.
- (If gate passed) `output/pairwise_bt_ensemble.csv`, `output/pairwise_v9c_bt_ensemble.csv`, `output/bt_head_to_head.json`.

Findings: `docs/notes/2026-05-01-bayesian-stage1.md`.
Spec: `docs/superpowers/specs/2026-05-01-bayesian-stage1-design.md`.
Plan: `docs/superpowers/plans/2026-05-01-bayesian-stage1.md`.

## Test plan

- [x] BT trainer unit tests + synthetic-data recovery
- [x] Diagnostic unit tests + gate-clause coverage
- [x] Anchor reproductions for `(1.0, 0.0)` and `(0.0, 1.0)` (if gate passed)
- [x] `pytest -v` -- all green
- [x] Standalone BT log loss in believable range; home-court coefficient `h` in [0.3, 0.5]

EOF
)"
```

---

## Self-Review

**Spec coverage check:**

- [x] `src/train_bt_stage1.py` -- Task 1
- [x] `src/diagnose_bt_vs_v4.py` -- Task 3
- [x] Per-season L2 LR fit on team-indicator + home-court matrix -- Task 1 implementation
- [x] Predict pairwise from fitted strengths -- Task 1 (`predict_pairwise_for_field`)
- [x] Diagnostic-first gate with three clauses -- Task 3 + Task 4
- [x] Conditional v9-C bracket-points head-to-head -- Task 5
- [x] Verdict bands (>= +25 / +10-+25 / < +10) -- Task 5 Step 7 + Task 6
- [x] Anchor checks for ensemble averaging -- Task 5 Step 3
- [x] Findings note + TODO update -- Task 6
- [x] PR 11 dependency mitigation (cherry-pick if not on main) -- Task 5 Step 1
- [x] No new top-level dependencies -- verified (sklearn + scipy.sparse already in pyproject.toml)
- [x] Standalone log loss / home-court sanity checks -- Task 2 Step 1 (printed per fold)

**Placeholder scan:** no TBD / TODO / "fill in details" markers in any task body. The Task 5 commit message and PR body include `<placeholders>` for verdict + numbers because those are unknown at plan-write time and will be filled in at run time -- they are intentional templates, not gaps.

**Type / signature consistency:**
- `extract_home_court_value(wloc: str) -> int` -- consistent in test (Task 1 Step 1) and implementation (Task 1 Step 3).
- `build_design_matrix(games, team_ids) -> (sparse, ndarray)` -- consistent.
- `fit_bradley_terry(X, y, C=10.0) -> ndarray` -- consistent.
- `predict_pairwise_for_field(season, field, team_ids, strengths) -> List[dict]` -- consistent.
- `compute_diagnostic(pairwise_v4_csv, pairwise_bt_csv, results_csv=..., results_df=...) -> dict` -- consistent across Task 3 (test/impl) and Task 4 invocation.
- `check_gate(diag) -> {pass, reason}` -- consistent.
- Gate threshold constants (`GATE_R_MAX=0.60`, `GATE_W_LOW=0.30`, `GATE_W_HIGH=0.85`, `GATE_HEADROOM_MIN=0.005`) match spec values verbatim.
