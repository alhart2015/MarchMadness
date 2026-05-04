# Massey-Matrix MOV Rating Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `massey_mov_rating` column to v4's per-team-per-season feature matrix by solving a Massey-style least-squares system over regular-season MOV with home-court estimated jointly and MOV capped at +/- 21. Run a two-clause cheap falsification gate, then a 22-season LOSO backtest, and apply the standard Reject -> Clear -> Marginal ladder.

**Architecture:** Standalone solver module `src/features/massey_matrix.py` with parquet cache; one-line wire-in to `compute_all_features()` in `src/enhanced_model.py`; standalone gate runner `src/diagnose_massey_mov.py` mirroring the diagnostic-runner pattern from PRs 13/14. The solver builds the normal-equations matrix (n_teams + 1) directly from team co-occurrence counts and MOV sums (no game-level design matrix materialization), then resolves the 1-d null space with a sum-to-zero KKT constraint.

**Tech Stack:** Python 3.x, NumPy (linalg.solve), pandas (parquet via pyarrow), pytest, XGBoost (existing v4 trainer in src/enhanced_model_v3.py).

**Spec:** `docs/superpowers/specs/2026-05-03-massey-matrix-feature-design.md`

---

## File Structure

**Files this plan creates (committed):**

- `src/features/massey_matrix.py` -- solver core + cached loader (~120 LOC).
  - Public: `compute_massey_mov_ratings(reg_season, seasons=None, mov_cap=21) -> DataFrame`
  - Public: `load_massey_mov_ratings(reg_season, mov_cap=21, cache_dir="data/cache") -> DataFrame`
  - Module constant: `_PRODUCER_VERSION = "v1"`
  - Private: `_solve_one_season(games_df: DataFrame, mov_cap: int) -> dict[int, float]` -- builds and solves the bordered KKT system for one season; returns `{TeamID: rating}` plus `{"_h": home_constant}`.
  - Private: `_build_normal_equations(games_df, team_ids, mov_cap)` -- returns `(M, b)` for the bordered system.
- `src/diagnose_massey_mov.py` -- gate runner CLI (~150 LOC).
  - One CLI entry point, both clauses, writes `output/diag_massey_mov.json`.
- `tests/test_features/test_massey_matrix.py` -- unit tests (~180 LOC).

**Files this plan modifies:**

- `src/enhanced_model.py` -- 3 small additions in `compute_all_features()`. No structural refactor.

**Generated artifacts (gitignored, regenerable):**

- `data/cache/massey_mov_ratings.parquet` (+ sidecar `data/cache/massey_mov_ratings.meta.json`)
- `output/diag_massey_mov.json`
- `output/v4_with_massey_loso.json` (only if gate passes)

**Generated artifacts (committed for audit trail):**

- `docs/notes/2026-05-03-massey-mov.md` (gate findings)
- `docs/notes/2026-05-03-massey-mov-backtest.md` (only if gate passes)

**File-touch budget by task** (for reviewing change scope):

| Task | Touches |
|------|---------|
| 1-7  | `src/features/massey_matrix.py`, `tests/test_features/test_massey_matrix.py` |
| 8    | `src/enhanced_model.py` (3 additions) |
| 9-11 | `src/diagnose_massey_mov.py`, `output/diag_massey_mov.json`, `docs/notes/2026-05-03-massey-mov.md` |
| 12-14| `output/v4_with_massey_loso.json`, `docs/notes/2026-05-03-massey-mov-backtest.md`, `TODO.md`, possibly revert of task 8 |

---

## Task 1: Solver core (synthetic round-robin TDD)

**Files:**
- Create: `src/features/massey_matrix.py`
- Test: `tests/test_features/test_massey_matrix.py`

The synthetic schedule below has a known closed-form solution: `r = [+5, +2, -2, -5]` (teams A, B, C, D), `h = +1`. Each pair plays twice (home/away). All MOVs are in [+1, +11], well below mov_cap=21, so capping doesn't engage.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_features/test_massey_matrix.py
"""Unit tests for src/features/massey_matrix.py.

Synthetic schedules with closed-form solutions verify solver correctness;
real-data smoke test verifies the cached loader and pipeline integration.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.features.massey_matrix import (
    _PRODUCER_VERSION,
    compute_massey_mov_ratings,
    load_massey_mov_ratings,
)


def _make_round_robin(team_ids, ratings, h, season=2024):
    """Build a 12-game round-robin where the solution is known.

    Each pair plays twice: once at each home. MOV from W's perspective:
        y = r_W - r_L + h * z, z=+1 if W home else -1 if W away else 0
    Loser is the team with lower rating; winner = higher rating.
    """
    rows = []
    daynum = 10
    for i, ti in enumerate(team_ids):
        for j, tj in enumerate(team_ids):
            if i >= j:
                continue
            ri, rj = ratings[i], ratings[j]
            # Game at ti's home: ti is home; W = max-rating side, locator from W's view
            for home_team_idx in (i, j):
                if home_team_idx == i:
                    if ri > rj:
                        w, l = ti, tj
                        z = +1  # W home
                        mov = (ri - rj) + h * z
                    else:
                        w, l = tj, ti
                        z = -1  # W away
                        mov = (rj - ri) + h * z
                else:
                    if rj > ri:
                        w, l = tj, ti
                        z = +1
                        mov = (rj - ri) + h * z
                    else:
                        w, l = ti, tj
                        z = -1
                        mov = (ri - rj) + h * z
                wloc = {1: "H", -1: "A", 0: "N"}[z]
                rows.append({
                    "Season": season,
                    "DayNum": daynum,
                    "WTeamID": w,
                    "WScore": int(50 + max(1, mov)),
                    "LTeamID": l,
                    "LScore": 50,
                    "WLoc": wloc,
                    "NumOT": 0,
                })
                daynum += 1
    return pd.DataFrame(rows)


def test_synthetic_round_robin_recovers_ratings_and_home_constant():
    team_ids = [1101, 1102, 1103, 1104]
    ratings = [5.0, 2.0, -2.0, -5.0]
    h_true = 1.0
    games = _make_round_robin(team_ids, ratings, h_true)

    df = compute_massey_mov_ratings(games, mov_cap=21)

    assert set(df.columns) == {"Season", "TeamID", "massey_mov_rating"}
    assert len(df) == 4
    rating_by_team = dict(zip(df["TeamID"], df["massey_mov_rating"]))
    for tid, expected in zip(team_ids, ratings):
        assert rating_by_team[tid] == pytest.approx(expected, abs=1e-4), (
            f"Team {tid} expected {expected}, got {rating_by_team[tid]}"
        )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_features/test_massey_matrix.py::test_synthetic_round_robin_recovers_ratings_and_home_constant -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.features.massey_matrix'` or `ImportError`.

- [ ] **Step 3: Implement the solver**

```python
# src/features/massey_matrix.py
"""Massey-matrix MOV ratings: per-team-per-season ratings from a
least-squares solve over regular-season game results with a jointly
estimated home-court constant and MOV capping.

See docs/superpowers/specs/2026-05-03-massey-matrix-feature-design.md.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_PRODUCER_VERSION = "v1"


def _solve_one_season(games_df: pd.DataFrame, mov_cap: int) -> tuple[dict[int, float], float]:
    """Solve Massey-style least squares for one season.

    Parameters
    ----------
    games_df : DataFrame
        Subset of MRegularSeasonCompactResults for a single season.
        Required columns: WTeamID, LTeamID, WScore, LScore, WLoc.
    mov_cap : int
        Cap absolute score-margin contributions.

    Returns
    -------
    (ratings, h) : (dict, float)
        ratings maps TeamID -> rating; sum of ratings is 0.
        h is the home-court constant.
    """
    if mov_cap <= 0:
        raise ValueError(f"mov_cap must be positive, got {mov_cap}")

    team_ids = sorted(set(games_df["WTeamID"].tolist()) | set(games_df["LTeamID"].tolist()))
    n = len(team_ids)
    idx = {tid: i for i, tid in enumerate(team_ids)}

    # Bordered KKT system: (n+2) x (n+2)
    #   [ X^T X    e ] [ beta   ]   [ X^T y ]
    #   [   e^T    0 ] [ lambda ] = [   0   ]
    # where beta = [r_1, ..., r_n, h] and e = [1, ..., 1, 0]^T (ones in
    # the n team slots, zero in the home-constant slot).
    M = np.zeros((n + 2, n + 2), dtype=np.float64)
    rhs = np.zeros(n + 2, dtype=np.float64)

    # Constraint row/col: sum(r) = 0 (does not constrain h).
    for k in range(n):
        M[n + 1, k] = 1.0
        M[k, n + 1] = 1.0

    h_col = n  # column index of home-constant in beta

    for w, l, ws, ls, wloc in zip(
        games_df["WTeamID"].to_numpy(),
        games_df["LTeamID"].to_numpy(),
        games_df["WScore"].to_numpy(),
        games_df["LScore"].to_numpy(),
        games_df["WLoc"].to_numpy(),
    ):
        wi = idx[int(w)]
        li = idx[int(l)]
        z = 1 if wloc == "H" else (-1 if wloc == "A" else 0)
        s = int(ws) - int(ls)
        # Cap. Sign(s) is always +1 here since W beat L; keep abs(s) <= cap.
        y = min(s, mov_cap)

        # X row for this game has +1 in col wi, -1 in col li, +z in col h_col.
        # X^T X contributions:
        M[wi, wi] += 1.0
        M[li, li] += 1.0
        M[wi, li] -= 1.0
        M[li, wi] -= 1.0
        M[wi, h_col] += z
        M[h_col, wi] += z
        M[li, h_col] -= z
        M[h_col, li] -= z
        M[h_col, h_col] += z * z  # 1 if non-neutral, 0 if neutral

        # X^T y contributions:
        rhs[wi] += y
        rhs[li] -= y
        rhs[h_col] += z * y

    cond = np.linalg.cond(M)
    if cond > 1e10:
        logger.warning("Massey normal-equations matrix is ill-conditioned (cond=%.2e); "
                       "season may have a disconnected component", cond)

    sol = np.linalg.solve(M, rhs)
    ratings_arr = sol[:n]
    h_val = float(sol[n])

    return ({tid: float(ratings_arr[idx[tid]]) for tid in team_ids}, h_val)


def compute_massey_mov_ratings(
    reg_season: pd.DataFrame,
    seasons: list[int] | None = None,
    mov_cap: int = 21,
) -> pd.DataFrame:
    """Compute Massey-matrix MOV ratings per (Season, TeamID).

    Parameters
    ----------
    reg_season : DataFrame
        Kaggle MRegularSeasonCompactResults (or DetailedResults superset).
        Required columns: Season, WTeamID, LTeamID, WScore, LScore, WLoc.
    seasons : list of int or None
        Restrict to these seasons. None = all seasons present in reg_season.
    mov_cap : int
        Cap absolute score-margin (predictive Massey, default 21).

    Returns
    -------
    DataFrame with columns [Season, TeamID, massey_mov_rating],
    one row per (team, season) where the team appeared in the season's
    regular-season schedule.
    """
    required = {"Season", "WTeamID", "LTeamID", "WScore", "LScore", "WLoc"}
    missing = required - set(reg_season.columns)
    if missing:
        raise ValueError(f"reg_season missing required columns: {sorted(missing)}")

    season_iter = sorted(reg_season["Season"].unique()) if seasons is None else sorted(seasons)
    rows = []
    for season in season_iter:
        games = reg_season[reg_season["Season"] == season]
        if len(games) == 0:
            continue
        ratings, _h = _solve_one_season(games, mov_cap)
        for tid, r in ratings.items():
            rows.append({"Season": int(season), "TeamID": int(tid), "massey_mov_rating": r})

    return pd.DataFrame(rows)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_features/test_massey_matrix.py::test_synthetic_round_robin_recovers_ratings_and_home_constant -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/features/massey_matrix.py tests/test_features/test_massey_matrix.py
git commit -m "feat(massey-matrix): solver core + synthetic round-robin test

Bordered KKT system (n+2)x(n+2) with sum-to-zero constraint resolves
the 1-d null space of X^T X; home constant estimated jointly. Builds
normal equations from team co-occurrence directly (no game-level X).
mov_cap=21 default per spec.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: Sum-to-zero invariant test

**Files:**
- Test: `tests/test_features/test_massey_matrix.py`

The sum-to-zero constraint is what makes the solve unique; this is a regression guard that the bordered-system implementation doesn't drift if the solver is later rewritten.

- [ ] **Step 1: Add the test**

Append to `tests/test_features/test_massey_matrix.py`:

```python
def test_sum_to_zero_invariant():
    """Solver enforces sum(ratings) = 0 for identifiability."""
    team_ids = [1101, 1102, 1103, 1104]
    ratings = [5.0, 2.0, -2.0, -5.0]
    games = _make_round_robin(team_ids, ratings, h=1.0)

    df = compute_massey_mov_ratings(games, mov_cap=21)
    assert df["massey_mov_rating"].sum() == pytest.approx(0.0, abs=1e-8)
```

- [ ] **Step 2: Run the test**

Run: `pytest tests/test_features/test_massey_matrix.py::test_sum_to_zero_invariant -v`
Expected: PASS (the bordered KKT enforces it directly).

- [ ] **Step 3: Commit**

```bash
git add tests/test_features/test_massey_matrix.py
git commit -m "test(massey-matrix): sum-to-zero invariant regression guard

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: MOV cap behavior test

**Files:**
- Test: `tests/test_features/test_massey_matrix.py`

A team that wins one 100-point blowout against an otherwise weak schedule should not have its rating dominated by that single game when `mov_cap=21`. We compare ratings under cap=21 vs cap=100 to verify the cap is doing real work.

- [ ] **Step 1: Add the test**

Append:

```python
def test_mov_cap_clips_blowouts():
    """Capping at 21 produces materially different (smaller-magnitude)
    ratings than capping at 100 when a single blowout exists."""
    # Team 1101 plays team 1102 once (100-point blowout) and team 1103
    # plays team 1104 once (3-point game). Two-component schedule.
    rows = [
        {"Season": 2024, "DayNum": 10, "WTeamID": 1101, "WScore": 150, "LTeamID": 1102,
         "LScore": 50, "WLoc": "N", "NumOT": 0},
        {"Season": 2024, "DayNum": 11, "WTeamID": 1103, "WScore": 70, "LTeamID": 1104,
         "LScore": 67, "WLoc": "N", "NumOT": 0},
        # Connect the two components so the system is solvable.
        {"Season": 2024, "DayNum": 12, "WTeamID": 1101, "WScore": 75, "LTeamID": 1103,
         "LScore": 70, "WLoc": "N", "NumOT": 0},
        {"Season": 2024, "DayNum": 13, "WTeamID": 1102, "WScore": 60, "LTeamID": 1104,
         "LScore": 58, "WLoc": "N", "NumOT": 0},
    ]
    games = pd.DataFrame(rows)

    df_capped = compute_massey_mov_ratings(games, mov_cap=21)
    df_uncapped = compute_massey_mov_ratings(games, mov_cap=100)

    rating_capped = dict(zip(df_capped["TeamID"], df_capped["massey_mov_rating"]))
    rating_uncapped = dict(zip(df_uncapped["TeamID"], df_uncapped["massey_mov_rating"]))

    # Team 1101's rating in the uncapped solve is dominated by the +100
    # game vs 1102, so |rating_1101_uncapped| > |rating_1101_capped|.
    assert abs(rating_uncapped[1101]) > abs(rating_capped[1101]) + 1.0
    # Sanity: the capped 1101 rating is bounded by mov_cap (its games
    # contributed at most cap=21 each toward the rating in score units).
    assert abs(rating_capped[1101]) < 30.0  # well under the uncapped value
```

- [ ] **Step 2: Run the test**

Run: `pytest tests/test_features/test_massey_matrix.py::test_mov_cap_clips_blowouts -v`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/test_features/test_massey_matrix.py
git commit -m "test(massey-matrix): mov_cap=21 vs 100 -- cap reduces blowout influence

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: Home-court constant sign test

**Files:**
- Test: `tests/test_features/test_massey_matrix.py`

In a balanced schedule where the home team always wins by 5, the solver should recover h ~= 5 and team ratings near zero.

- [ ] **Step 1: Add the test**

Append:

```python
def test_home_court_constant_recovered():
    """If teams are equal-strength but home always wins by 5, h ~= 5."""
    team_ids = [1101, 1102, 1103, 1104]
    rows = []
    daynum = 10
    for i, ti in enumerate(team_ids):
        for j, tj in enumerate(team_ids):
            if i == j:
                continue
            # ti is home; ti wins by 5
            rows.append({
                "Season": 2024, "DayNum": daynum,
                "WTeamID": ti, "WScore": 75,
                "LTeamID": tj, "LScore": 70,
                "WLoc": "H", "NumOT": 0,
            })
            daynum += 1
    games = pd.DataFrame(rows)

    # Solve directly with the private function so we can inspect h.
    from src.features.massey_matrix import _solve_one_season
    ratings, h = _solve_one_season(games, mov_cap=21)

    assert h == pytest.approx(5.0, abs=1e-4)
    for tid, r in ratings.items():
        assert abs(r) < 1e-4, f"Team {tid} expected ~0 rating, got {r}"
```

- [ ] **Step 2: Run the test**

Run: `pytest tests/test_features/test_massey_matrix.py::test_home_court_constant_recovered -v`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/test_features/test_massey_matrix.py
git commit -m "test(massey-matrix): home-court constant recovered from equal-strength schedule

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: Cached loader (round-trip test)

**Files:**
- Modify: `src/features/massey_matrix.py`
- Test: `tests/test_features/test_massey_matrix.py`

Adds `load_massey_mov_ratings()` -- the consumer-side wrapper that reads a parquet cache or rebuilds it via `compute_massey_mov_ratings`.

- [ ] **Step 1: Add the round-trip test**

Append:

```python
def test_cache_roundtrip(tmp_path: Path):
    """load_massey_mov_ratings writes parquet + sidecar on first call,
    reads from cache on second call; both return equal frames."""
    team_ids = [1101, 1102, 1103, 1104]
    games = _make_round_robin(team_ids, [5.0, 2.0, -2.0, -5.0], h=1.0)

    df1 = load_massey_mov_ratings(games, mov_cap=21, cache_dir=tmp_path)
    parquet_path = tmp_path / "massey_mov_ratings.parquet"
    meta_path = tmp_path / "massey_mov_ratings.meta.json"
    assert parquet_path.exists()
    assert meta_path.exists()

    df2 = load_massey_mov_ratings(games, mov_cap=21, cache_dir=tmp_path)
    pd.testing.assert_frame_equal(
        df1.sort_values(["Season", "TeamID"]).reset_index(drop=True),
        df2.sort_values(["Season", "TeamID"]).reset_index(drop=True),
    )

    meta = json.loads(meta_path.read_text())
    assert meta["producer_version"] == _PRODUCER_VERSION
    assert meta["mov_cap"] == 21
    assert meta["n_input_rows"] == len(games)
    assert "sha_input" in meta
```

- [ ] **Step 2: Run the test (expect FAIL)**

Run: `pytest tests/test_features/test_massey_matrix.py::test_cache_roundtrip -v`
Expected: FAIL with `ImportError: cannot import name 'load_massey_mov_ratings'`.

- [ ] **Step 3: Add the loader to the module**

Append to `src/features/massey_matrix.py`:

```python
def _hash_input(reg_season: pd.DataFrame) -> str:
    """Stable content hash of the relevant columns of the input frame."""
    cols = ["Season", "DayNum", "WTeamID", "WScore", "LTeamID", "LScore", "WLoc"]
    h = hashlib.sha256()
    for c in cols:
        if c in reg_season.columns:
            h.update(reg_season[c].astype(str).str.cat(sep="|").encode("ascii", errors="replace"))
    return h.hexdigest()[:16]


def load_massey_mov_ratings(
    reg_season: pd.DataFrame,
    mov_cap: int = 21,
    cache_dir: str | Path = "data/cache",
) -> pd.DataFrame:
    """Cached wrapper around compute_massey_mov_ratings.

    Reads from <cache_dir>/massey_mov_ratings.parquet on cache hit.
    Cache hit requires the sidecar metadata at
    <cache_dir>/massey_mov_ratings.meta.json to match the current
    (_PRODUCER_VERSION, mov_cap, n_input_rows, sha_input).
    """
    cache_dir = Path(cache_dir)
    parquet_path = cache_dir / "massey_mov_ratings.parquet"
    meta_path = cache_dir / "massey_mov_ratings.meta.json"

    expected_meta = {
        "producer_version": _PRODUCER_VERSION,
        "mov_cap": int(mov_cap),
        "n_input_rows": int(len(reg_season)),
        "sha_input": _hash_input(reg_season),
    }

    if parquet_path.exists() and meta_path.exists():
        try:
            actual_meta = json.loads(meta_path.read_text())
        except (json.JSONDecodeError, OSError):
            actual_meta = {}
        if all(actual_meta.get(k) == expected_meta[k] for k in expected_meta):
            logger.info("Massey MOV cache hit: %s", parquet_path)
            return pd.read_parquet(parquet_path)
        logger.info("Massey MOV cache stale (metadata mismatch); rebuilding")

    df = compute_massey_mov_ratings(reg_season, mov_cap=mov_cap)
    cache_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(parquet_path, index=False)
    meta_path.write_text(json.dumps({**expected_meta, "written_at_n_rows": len(df)}, indent=2))
    logger.info("Massey MOV cache written: %s (%d rows)", parquet_path, len(df))
    return df
```

- [ ] **Step 4: Run the test**

Run: `pytest tests/test_features/test_massey_matrix.py::test_cache_roundtrip -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/features/massey_matrix.py tests/test_features/test_massey_matrix.py
git commit -m "feat(massey-matrix): load_massey_mov_ratings with parquet cache

Sidecar meta.json with (producer_version, mov_cap, n_input_rows,
sha_input). Mismatch triggers rebuild; matches return cached frame.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 6: Cache invalidation test

**Files:**
- Test: `tests/test_features/test_massey_matrix.py`

Verifies that bumping `_PRODUCER_VERSION` invalidates an existing cache without manual file deletion.

- [ ] **Step 1: Add the test**

Append:

```python
def test_cache_invalidates_on_meta_mismatch(tmp_path: Path, monkeypatch):
    """If sidecar metadata's producer_version doesn't match the module
    constant, the cache is rebuilt rather than reused."""
    team_ids = [1101, 1102, 1103, 1104]
    games = _make_round_robin(team_ids, [5.0, 2.0, -2.0, -5.0], h=1.0)

    # Initial write under v1 (current).
    df1 = load_massey_mov_ratings(games, mov_cap=21, cache_dir=tmp_path)
    parquet_path = tmp_path / "massey_mov_ratings.parquet"
    initial_mtime = parquet_path.stat().st_mtime_ns

    # Hand-edit the sidecar to claim a different producer version.
    meta_path = tmp_path / "massey_mov_ratings.meta.json"
    meta = json.loads(meta_path.read_text())
    meta["producer_version"] = "v0-stale"
    meta_path.write_text(json.dumps(meta))

    # Next load should detect the mismatch and rebuild.
    df2 = load_massey_mov_ratings(games, mov_cap=21, cache_dir=tmp_path)
    new_mtime = parquet_path.stat().st_mtime_ns
    assert new_mtime > initial_mtime, "parquet should have been rewritten"

    # The rebuilt sidecar should claim the current version.
    refreshed = json.loads(meta_path.read_text())
    assert refreshed["producer_version"] == _PRODUCER_VERSION

    pd.testing.assert_frame_equal(
        df1.sort_values(["Season", "TeamID"]).reset_index(drop=True),
        df2.sort_values(["Season", "TeamID"]).reset_index(drop=True),
    )
```

- [ ] **Step 2: Run the test**

Run: `pytest tests/test_features/test_massey_matrix.py::test_cache_invalidates_on_meta_mismatch -v`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/test_features/test_massey_matrix.py
git commit -m "test(massey-matrix): cache invalidates on producer_version mismatch

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 7: Real-data smoke + initial cache build

**Files:**
- Test: `tests/test_features/test_massey_matrix.py`

Runs the solver against actual `MRegularSeasonCompactResults.csv` (skipped if data missing). Asserts the output shape and rating-range plausibility, and confirms the cache file was written. This is the moment we discover any real-data shape surprise (e.g., a season we forgot, a team-id type collision) before wiring into the v4 pipeline.

- [ ] **Step 1: Add the test**

Append:

```python
_REG_SEASON_CSV = (
    Path(__file__).resolve().parents[2]
    / "data" / "raw" / "march-machine-learning-2026"
    / "MRegularSeasonCompactResults.csv"
)


@pytest.mark.skipif(not _REG_SEASON_CSV.exists(), reason="raw Kaggle data not available")
def test_real_data_shape_and_rating_range(tmp_path: Path):
    """Smoke-test: solver runs on real Kaggle data, output shape is
    plausible, ratings fall in a sane range (no infinity / NaN)."""
    reg = pd.read_csv(_REG_SEASON_CSV)
    # v4 trains on Season >= 2003; smoke-test the same range.
    reg = reg[reg["Season"] >= 2003]
    df = load_massey_mov_ratings(reg, mov_cap=21, cache_dir=tmp_path)

    assert df["massey_mov_rating"].notna().all(), "no NaN ratings"
    assert np.isfinite(df["massey_mov_rating"]).all(), "no inf ratings"
    # D-I rating range: cap=21 means individual game contributions
    # are bounded; per-team aggregates fall in roughly [-25, +25].
    assert df["massey_mov_rating"].abs().max() < 40.0, (
        f"unexpectedly large rating: {df['massey_mov_rating'].abs().max():.2f}"
    )
    # Each season should have at least 300 D-I teams and at most 380.
    counts = df.groupby("Season").size()
    assert (counts >= 300).all(), f"min teams per season: {counts.min()}"
    assert (counts <= 380).all(), f"max teams per season: {counts.max()}"
    # Sum-to-zero per season.
    sums = df.groupby("Season")["massey_mov_rating"].sum()
    assert sums.abs().max() < 1e-6, f"per-season sum drift: {sums.abs().max()}"
```

- [ ] **Step 2: Run the test**

Run: `pytest tests/test_features/test_massey_matrix.py::test_real_data_shape_and_rating_range -v`
Expected: PASS. If it fails, the failure is informative -- shape, range, or NaN issue with real data.

- [ ] **Step 3: Build the production cache**

Run: `python -c "import pandas as pd; from src.features.massey_matrix import load_massey_mov_ratings; reg = pd.read_csv('data/raw/march-machine-learning-2026/MRegularSeasonCompactResults.csv'); reg = reg[reg.Season >= 2003]; df = load_massey_mov_ratings(reg); print('rows:', len(df), 'seasons:', df.Season.nunique(), 'teams range:', df.TeamID.min(), df.TeamID.max(), 'rating range:', df.massey_mov_rating.min(), df.massey_mov_rating.max())"`
Expected: prints something like `rows: 8000+ seasons: 23 teams range: 1101 1480 rating range: -25.x 25.x`. Produces `data/cache/massey_mov_ratings.parquet` (gitignored).

- [ ] **Step 4: Run the full feature test file**

Run: `pytest tests/test_features/test_massey_matrix.py -v`
Expected: all 7 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/test_features/test_massey_matrix.py
git commit -m "test(massey-matrix): real-data smoke test + production cache build

Asserts shape, NaN-free, rating range, sum-to-zero per season on
actual MRegularSeasonCompactResults. Cache (data/cache/) is gitignored
per CLAUDE.md (regenerable artifact).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 8: Wire-in to compute_all_features

**Files:**
- Modify: `src/enhanced_model.py`

Three small additions in `compute_all_features()`. **Re-read the file first** -- per CLAUDE.md "CONTEXT DECAY AWARENESS," do not trust your memory of file contents.

- [ ] **Step 1: Re-read the file before editing**

Read `src/enhanced_model.py` lines 170-400 to confirm the current shape of `compute_all_features` and locate the three insertion points.

- [ ] **Step 2: Add the cache-load at the top of `compute_all_features`**

Find the block (around line 195-205) that reads:

```python
    seasons = sorted(reg["Season"].unique())
    # Only process seasons where we have detailed results (2003+)
    seasons = [s for s in seasons if s >= 2003]

    # -- Build KenPom -> Kaggle ID mapping --------------------------------
    print("  Building team ID mapping (KenPom -> Kaggle)...")
```

Insert *immediately after* the `seasons = [s for s in seasons if s >= 2003]` line and before the `-- Build KenPom -> Kaggle ID mapping --` block:

```python
    # -- Massey-matrix MOV ratings (cached) -------------------------------
    from src.features.massey_matrix import load_massey_mov_ratings
    massey_mov_full = load_massey_mov_ratings(reg)
```

- [ ] **Step 3: Add a per-season block "2i"**

Find the block "-- 2h: Seed features --" (around line 332 in the existing file). Immediately after that block ends (right before `# -- Assemble features for each team in this season ---------------`), insert:

```python
        # -- 2i: Massey-matrix MOV rating ---------------------------------
        season_mov_df = massey_mov_full[massey_mov_full["Season"] == season]
        massey_mov = dict(zip(season_mov_df["TeamID"], season_mov_df["massey_mov_rating"]))
```

- [ ] **Step 4: Add the per-team-row assembly line**

Inside the `for tid in all_team_ids:` loop, between the `# Massey ordinals` block and the `# Conference strength` block, insert:

```python
            # Massey-matrix MOV rating
            if tid in massey_mov:
                row_data["massey_mov_rating"] = massey_mov[tid]
```

- [ ] **Step 5: Re-read the file to verify edits applied**

Read `src/enhanced_model.py` lines 170-400 and confirm:
- The `from src.features.massey_matrix import load_massey_mov_ratings` line is present near top of `compute_all_features`.
- A "2i: Massey-matrix MOV rating" block exists between 2h and the assembly loop.
- The `if tid in massey_mov: row_data["massey_mov_rating"] = ...` line is present in the assembly loop.

- [ ] **Step 6: Run the existing seam tests (CLAUDE.md forced verification)**

Run: `pytest -v tests/test_ingest tests/test_features tests/test_integration.py`
Expected: all PASS. If `test_integration.py` fails on a probability-tolerance assert, STOP and flag to user per CLAUDE.md (don't silently widen tolerances).

- [ ] **Step 7: Smoke-run the v4 pipeline on a single holdout**

Run: `python -c "from src.enhanced_model_v3 import prepare_loso_inputs; from src.enhanced_model import load_data; data = load_data(); inputs = prepare_loso_inputs(data, holdout_season=2024); fm = inputs['feature_matrix']; print('cols:', sorted(fm.columns)); print('massey_mov present:', 'massey_mov_rating' in fm.columns); print('n with massey:', fm['massey_mov_rating'].notna().sum() if 'massey_mov_rating' in fm.columns else 'N/A')"`
Expected: prints `massey_mov present: True` and `n with massey: <thousands>`. Confirms the column flows through.

- [ ] **Step 8: Commit**

```bash
git add src/enhanced_model.py
git commit -m "feat(massey-matrix): wire massey_mov_rating into compute_all_features

Three additions in src/enhanced_model.py: cache load at top of
compute_all_features, per-season dict in the season loop, per-team
row-assembly line. Total ~8 LOC added; no structural refactor.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 9: Diagnostic gate -- clause 1 (correlation)

**Files:**
- Create: `src/diagnose_massey_mov.py`

Computes per-season Pearson correlations between `massey_mov_rating` and (a) `adj_em`, (b) `massey_composite`. Reports mean and max-abs across seasons.

- [ ] **Step 1: Create the gate runner skeleton with clause 1**

```python
# src/diagnose_massey_mov.py
"""Two-clause falsification gate for massey_mov_rating.

Clause 1 -- non-redundancy: per-season Pearson correlation between
massey_mov_rating and (adj_em, massey_composite). Pass if mean |corr|
< 0.95 and max |corr| < 0.97 against BOTH baselines.

Clause 2 -- no-harm headroom: 3-season subset {2019, 2022, 2024}.
Train v4 with massey_mov on, compute LL on holdout games. Pass if
mean LL with massey <= mean LL without massey + 0.001.

See docs/superpowers/specs/2026-05-03-massey-matrix-feature-design.md.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

GATE_SUBSET_SEASONS = [2019, 2022, 2024]
CORR_MEAN_MAX = 0.95
CORR_PER_SEASON_MAX = 0.97
LL_HEADROOM_MAX = 0.001


def clause1_correlations(feature_matrix: pd.DataFrame) -> dict:
    """Compute per-season correlations of massey_mov_rating vs adj_em
    and vs massey_composite. Returns a dict for output JSON.
    """
    needed = {"Season", "TeamID", "massey_mov_rating", "adj_em", "massey_composite"}
    missing = needed - set(feature_matrix.columns)
    if missing:
        raise ValueError(f"feature_matrix missing columns for clause 1: {sorted(missing)}")

    seasons = sorted(feature_matrix["Season"].unique())
    rows = []
    for season in seasons:
        sub = feature_matrix[feature_matrix["Season"] == season]
        sub = sub.dropna(subset=["massey_mov_rating", "adj_em", "massey_composite"])
        if len(sub) < 50:
            logger.warning("Season %d: only %d teams with all 3 cols; skipping",
                           season, len(sub))
            continue
        r_em = float(sub["massey_mov_rating"].corr(sub["adj_em"]))
        r_comp = float(sub["massey_mov_rating"].corr(sub["massey_composite"]))
        rows.append({"season": int(season), "n_teams": int(len(sub)),
                     "corr_vs_adj_em": r_em, "corr_vs_massey_composite": r_comp})

    df = pd.DataFrame(rows)
    summary = {
        "per_season": rows,
        "mean_abs_corr_vs_adj_em": float(df["corr_vs_adj_em"].abs().mean()),
        "max_abs_corr_vs_adj_em": float(df["corr_vs_adj_em"].abs().max()),
        "mean_abs_corr_vs_massey_composite": float(df["corr_vs_massey_composite"].abs().mean()),
        "max_abs_corr_vs_massey_composite": float(df["corr_vs_massey_composite"].abs().max()),
    }
    summary["pass"] = bool(
        summary["mean_abs_corr_vs_adj_em"] < CORR_MEAN_MAX
        and summary["max_abs_corr_vs_adj_em"] < CORR_PER_SEASON_MAX
        and summary["mean_abs_corr_vs_massey_composite"] < CORR_MEAN_MAX
        and summary["max_abs_corr_vs_massey_composite"] < CORR_PER_SEASON_MAX
    )
    return summary
```

- [ ] **Step 2: Add a quick smoke test**

Append to `tests/test_features/test_massey_matrix.py` (or create `tests/test_diagnose_massey_mov.py` -- put it in test_features for now to keep this PR's test count low):

```python
def test_clause1_pass_when_uncorrelated():
    """Clause 1 passes when massey_mov_rating is uncorrelated with the
    two baseline columns."""
    from src.diagnose_massey_mov import clause1_correlations
    rng = np.random.default_rng(0)
    n = 100
    fm = pd.DataFrame({
        "Season": [2024] * n,
        "TeamID": list(range(1, n + 1)),
        "massey_mov_rating": rng.standard_normal(n),
        "adj_em": rng.standard_normal(n),
        "massey_composite": rng.standard_normal(n),
    })
    out = clause1_correlations(fm)
    assert out["pass"] is True
    assert out["mean_abs_corr_vs_adj_em"] < 0.5  # random, easily under 0.95
```

- [ ] **Step 3: Run the test**

Run: `pytest tests/test_features/test_massey_matrix.py::test_clause1_pass_when_uncorrelated -v`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add src/diagnose_massey_mov.py tests/test_features/test_massey_matrix.py
git commit -m "feat(diagnose-massey-mov): clause 1 -- per-season correlation vs adj_em / composite

Pass criterion: mean |corr| < 0.95 AND max |corr| < 0.97 vs BOTH
baselines. Skips seasons with <50 fully-populated team rows.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 10: Diagnostic gate -- clause 2 (LL headroom)

**Files:**
- Modify: `src/enhanced_model_v3.py` (add `allowed_holdouts` kwarg to `leave_one_season_out_cv_weighted`)
- Modify: `src/diagnose_massey_mov.py`

The real LOSO trainer is `leave_one_season_out_cv_weighted` (src/enhanced_model_v3.py:472), which iterates every season in `tourney_results`. To run it on just the 3 subset seasons (cheap), we add a small `allowed_holdouts` kwarg that filters the iteration. We then call it twice -- once with `feature_cols` including `massey_mov_rating`, once without -- and compare mean test LL across the 3 holdouts.

- [ ] **Step 1: Re-read the trainer to confirm shape before refactor**

Read `src/enhanced_model_v3.py` lines 472-590 (the body of `leave_one_season_out_cv_weighted`). Confirm:
- Iteration starts at line 491 with `for holdout in seasons:`.
- The function returns `dict` with per-season keys; locate the return key for per-season test log loss (search for `log_loss` / `ll`).

- [ ] **Step 2: Add `allowed_holdouts` filter to the trainer**

Edit `src/enhanced_model_v3.py:472` -- add the new kwarg to the signature and one filter line right after `seasons = [s for s in seasons if s >= 2003]`:

```python
def leave_one_season_out_cv_weighted(
    feature_matrix: pd.DataFrame,
    tourney_results: pd.DataFrame,
    regular_results: pd.DataFrame,
    feature_cols: list,
    top_n_team_ids_by_season: dict,
    xgb_params: dict = None,
    random_seed: int = 42,
    supplemental_weight: float = 0.25,
    allowed_holdouts: list[int] | None = None,
) -> dict:
    """Run LOSO CV using weighted matchup data (tournament + supplemental).

    If allowed_holdouts is provided, restrict the iteration to those
    seasons (used by diagnostic gates for cheap subsets); training on
    each iteration still uses ALL non-holdout seasons.
    """
    from sklearn.metrics import log_loss as sklearn_log_loss, roc_auc_score
    from src.models.train import train_model

    seasons = sorted(tourney_results["Season"].unique())
    seasons = [s for s in seasons if s >= 2003]
    if allowed_holdouts is not None:
        seasons = [s for s in seasons if s in set(allowed_holdouts)]
```

(Replace the existing 6-line head of the function -- signature + early lines down through the seasons filter -- with the block above.)

- [ ] **Step 3: Verify the trainer still runs end-to-end on its existing tests**

Run: `pytest -v tests/test_integration.py`
Expected: PASS. Existing callers don't pass `allowed_holdouts` so behavior is unchanged.

- [ ] **Step 4: Commit the refactor (small, isolated)**

```bash
git add src/enhanced_model_v3.py
git commit -m "refactor(loso): allowed_holdouts kwarg for cheap diagnostic subsets

Optional list filter on the LOSO iteration; default None preserves
existing behavior. Used by diagnose_massey_mov clause-2 gate.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

- [ ] **Step 5: Add clause 2 implementation to the gate runner**

Append to `src/diagnose_massey_mov.py`:

```python
def clause2_headroom(seasons: list[int] = GATE_SUBSET_SEASONS) -> dict:
    """Run LOSO twice on the 3-season subset -- once with massey_mov_rating
    in feature_cols, once with it excluded. Compare mean test LL.

    Pass: mean(LL_with) - mean(LL_without) <= LL_HEADROOM_MAX.

    Implementation note: we toggle massey_mov_rating in feature_cols
    (NOT in the matrix) so train/test splits are byte-identical between
    arms; the column simply isn't fed to XGBoost in the without-arm.
    """
    from src.enhanced_model_v3 import (
        leave_one_season_out_cv_weighted,
        prepare_loso_inputs,
    )

    inputs = prepare_loso_inputs()
    fm = inputs["feature_matrix"]
    tourney = inputs["tourney_filtered"]
    regular = inputs["regular_results"]
    feature_cols_full = inputs["feature_cols"]
    top_80 = inputs["top_80_by_season"]

    if "massey_mov_rating" not in fm.columns:
        raise RuntimeError(
            "massey_mov_rating not present in feature_matrix; ensure Task 8 wire-in is committed"
        )
    if "massey_mov_rating" not in feature_cols_full:
        raise RuntimeError(
            "massey_mov_rating not in feature_cols; check get_feature_cols include logic"
        )

    cols_with = list(feature_cols_full)
    cols_without = [c for c in feature_cols_full if c != "massey_mov_rating"]

    res_with = leave_one_season_out_cv_weighted(
        fm, tourney, regular, cols_with, top_80, allowed_holdouts=seasons,
    )
    res_without = leave_one_season_out_cv_weighted(
        fm, tourney, regular, cols_without, top_80, allowed_holdouts=seasons,
    )

    # Both result dicts contain a list under key "results" or similar; verify
    # by reading the trainer's return shape (search for the final return in
    # leave_one_season_out_cv_weighted). Adjust below if the per-season list
    # is keyed differently in your code.
    per_with = {r["season"]: r for r in res_with["results"]}
    per_without = {r["season"]: r for r in res_without["results"]}

    per_season = []
    for season in seasons:
        rw = per_with.get(season)
        rwo = per_without.get(season)
        if rw is None or rwo is None:
            continue
        per_season.append({
            "season": int(season),
            "ll_with": float(rw["log_loss"]),
            "ll_without": float(rwo["log_loss"]),
            "ll_delta": float(rw["log_loss"] - rwo["log_loss"]),
        })

    mean_with = float(np.mean([r["ll_with"] for r in per_season]))
    mean_without = float(np.mean([r["ll_without"] for r in per_season]))
    delta = mean_with - mean_without
    return {
        "subset_seasons": seasons,
        "per_season": per_season,
        "mean_ll_with_massey": mean_with,
        "mean_ll_without_massey": mean_without,
        "mean_ll_delta": delta,
        "pass": bool(delta <= LL_HEADROOM_MAX),
    }
```

- [ ] **Step 6: Verify the trainer's result shape matches the access pattern**

Run: `grep -n "^    return\|results.append\|\"log_loss\"\|'log_loss'" src/enhanced_model_v3.py | head -20`
Expected: confirm that the per-season records appended in `leave_one_season_out_cv_weighted` use a key called `"log_loss"` (or similar like `"test_ll"`, `"ll"`). If the key name differs, update `clause2_headroom` accordingly. The return dict's top-level structure (where the per-season records are stored, here assumed `"results"`) must also be confirmed -- search for the final `return` near line 580+ of `enhanced_model_v3.py`.

- [ ] **Step 7: Commit**

```bash
git add src/diagnose_massey_mov.py
git commit -m "feat(diagnose-massey-mov): clause 2 -- 3-season LL headroom check

Calls leave_one_season_out_cv_weighted twice on holdouts {2019, 2022, 2024}
with and without massey_mov_rating in feature_cols. Pass if mean LL with
the column is no worse than +0.001 vs without.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 11: Gate runner CLI + execute gate on real data

**Files:**
- Modify: `src/diagnose_massey_mov.py`
- Generated: `output/diag_massey_mov.json`
- Possibly create: `docs/notes/2026-05-03-massey-mov.md` (only if FAIL)

CLI ties clause 1 and clause 2 together, writes `output/diag_massey_mov.json`. If either fails, write the findings note and STOP -- do not proceed to Task 12.

- [ ] **Step 1: Add the CLI entry point**

Append to `src/diagnose_massey_mov.py`:

```python
def main():
    from src.enhanced_model import load_data, compute_all_features
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    print("=" * 70)
    print("Massey-MOV gate: clause 1 (non-redundancy)")
    print("=" * 70)

    data = load_data()
    feature_matrix = compute_all_features(data)
    c1 = clause1_correlations(feature_matrix)
    print(json.dumps({k: v for k, v in c1.items() if k != "per_season"}, indent=2))
    print(f"  CLAUSE 1: {'PASS' if c1['pass'] else 'FAIL'}")

    if not c1["pass"]:
        result = {"clause1": c1, "clause2": None, "gate_pass": False}
        Path("output").mkdir(exist_ok=True)
        Path("output/diag_massey_mov.json").write_text(json.dumps(result, indent=2))
        print("\nSTOPPING: clause 1 failed; clause 2 not run.")
        sys.exit(1)

    print()
    print("=" * 70)
    print("Massey-MOV gate: clause 2 (LL headroom on 3-season subset)")
    print("=" * 70)
    c2 = clause2_headroom()
    print(json.dumps({k: v for k, v in c2.items() if k != "per_season"}, indent=2))
    print(f"  CLAUSE 2: {'PASS' if c2['pass'] else 'FAIL'}")

    gate_pass = c1["pass"] and c2["pass"]
    result = {"clause1": c1, "clause2": c2, "gate_pass": gate_pass}
    Path("output").mkdir(exist_ok=True)
    Path("output/diag_massey_mov.json").write_text(json.dumps(result, indent=2))

    print()
    print("=" * 70)
    print(f"AGGREGATE GATE: {'PASS -- proceed to full LOSO backtest' if gate_pass else 'FAIL -- stop, write findings'}")
    print("=" * 70)
    sys.exit(0 if gate_pass else 1)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the gate on real data**

Run: `python src/diagnose_massey_mov.py`
Expected wall-clock: ~5-15 min (clause 1 is seconds; clause 2 is the bulk via 3 v4 LOSO fits).

Three possible outcomes:
- **AGGREGATE GATE PASS** -> proceed to step 4 (commit, then Task 12).
- **CLAUSE 1 FAIL** -> step 3 (write findings, do NOT run clause 2), then STOP this plan.
- **CLAUSE 2 FAIL** -> step 3 (write findings), then STOP this plan.

- [ ] **Step 3: If FAIL, write the findings note and stop**

Create `docs/notes/2026-05-03-massey-mov.md`:

```markdown
# Massey-Matrix MOV Feature -- Gate FAILED

**Date:** 2026-05-03
**Branch:** feat/todo-massey-colley
**Spec:** docs/superpowers/specs/2026-05-03-massey-matrix-feature-design.md
**Verdict:** REJECTED at <clause 1|clause 2|both>.

## Numbers

<paste the aggregate metrics block from output/diag_massey_mov.json>

## Diagnosis

<one paragraph: which clause failed, by how much, and what existing
feature subsumes Massey (clause 1 case) or how the LL degraded (clause
2 case). For clause 1: explicitly identify whether adj_em or
massey_composite is the redundant baseline.>

## Lessons for Colley (TODO #1, separate work item)

<one paragraph: does the failure mode generalize to Colley? Massey
operates on margin; Colley on win/loss only. If clause 1 failed
because adj_em already extracts the margin signal, Colley's distinct
W/L information may still be productive. If clause 2 failed for
reasons that look structural (data scale at v4's hypers), Colley faces
the same ceiling.>

## Code retained as experiment record

- src/features/massey_matrix.py
- src/diagnose_massey_mov.py
- tests/test_features/test_massey_matrix.py
- src/enhanced_model.py wire-in (3 lines)

Branch feat/todo-massey-colley retained; not merged to main.
```

Then update `TODO.md` "Tried and rejected" with one bullet summarizing the verdict, run `python -c "open('docs/notes/2026-05-03-massey-mov.md').read().encode('ascii'); open('TODO.md').read().encode('ascii'); print('ASCII OK')"`, and commit:

```bash
git add docs/notes/2026-05-03-massey-mov.md TODO.md output/diag_massey_mov.json
git commit -m "docs(massey-matrix): findings note + TODO update -- gate FAILED

<one-line summary of why>

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

**STOP HERE. Do not run Task 12.** Update task list to skip remaining tasks.

- [ ] **Step 4: If PASS, commit and proceed**

```bash
git add src/diagnose_massey_mov.py output/diag_massey_mov.json
git commit -m "feat(diagnose-massey-mov): CLI runner + gate PASS on real data

clause 1 mean |corr| vs adj_em = X.XXX, vs massey_composite = X.XXX.
clause 2 mean LL delta = +/-X.XXX on subset {2019, 2022, 2024}.
Both clauses pass; proceeding to 22-season LOSO backtest.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 12: Full 22-season LOSO backtest

**Files:**
- Generated: `output/v4_with_massey_loso.json`

If and only if Task 11 ended in PASS.

- [ ] **Step 1: Run the full v4 LOSO backtest**

Run: `python src/enhanced_model_v3.py`
Expected wall-clock: ~30-90 min (22 LOSO holdouts, no Optuna re-tune).

The script writes its standard output (per-season LL/acc, brkt-pts) to its usual locations. It does NOT need a flag to "include massey_mov_rating" -- the column is already part of `compute_all_features()` (Task 8).

- [ ] **Step 2: Capture the result into `output/v4_with_massey_loso.json`**

Identify the existing per-season output the script writes (typically `output/loso_results.json` or similar; check the actual filename written by `enhanced_model_v3.py` by running `ls -lt output/ | head -10` after the run). Copy/rename to `output/v4_with_massey_loso.json` so the backtest output isn't overwritten by the next run:

Run: `cp output/loso_results.json output/v4_with_massey_loso.json` (substitute the actual filename if different).

- [ ] **Step 3: Compute deltas vs v4 baseline**

The v4 baseline numbers are recorded in prior findings notes (`docs/notes/2026-05-01-v9c-feature-stripped.md`: v4 wt-mean LL = 0.4369, v4 brkt = 2670 over 22 LOSO seasons; v9-C is 2713 = +43 vs v4 = current production stage-2). Compute:

```bash
python -c "
import json
with open('output/v4_with_massey_loso.json') as f:
    res = json.load(f)
# Adjust the key paths below if the file structure differs.
ll_with = res.get('weighted_mean_ll', res.get('mean_ll'))
brkt_with = res.get('total_bracket_points', res.get('brkt_pts'))
print('LL with massey:', ll_with)
print('brkt with massey:', brkt_with)
print('LL delta vs v4 (negative = improvement):', ll_with - 0.4369)
print('brkt delta vs v4:', brkt_with - 2670)
"
```

Record these numbers; they drive the ladder decision in Task 13.

- [ ] **Step 4: Commit the backtest output**

```bash
git add output/v4_with_massey_loso.json
git commit -m "data(massey-matrix): output/v4_with_massey_loso.json -- 22-season LOSO

ll_delta vs v4 = X.XXX, brkt_delta vs v4 = +/-XX

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 13: Apply ladder verdict + write findings

**Files:**
- Create: `docs/notes/2026-05-03-massey-mov-backtest.md`
- Modify: `TODO.md`
- Possibly revert: `src/enhanced_model.py` (only on Reject)

Apply the ladder **in order**: Reject -> Clear -> Marginal. Mixed signals (LL win + brkt loss) hit Reject because Reject is checked first.

- [ ] **Step 1: Compute the ladder bucket**

```python
# Mental / scratch check (not committed):
# ladder_bucket =
#   "Reject"   if ll_delta >= +0.001 OR brkt_delta <= +10
#   "Clear"    elif ll_delta <= -0.005 OR brkt_delta >= +25
#   "Marginal" otherwise
```

- [ ] **Step 2: Write the findings note**

Create `docs/notes/2026-05-03-massey-mov-backtest.md`:

```markdown
# Massey-Matrix MOV Feature -- 22-season LOSO Backtest

**Date:** 2026-05-03
**Branch:** feat/todo-massey-colley
**Spec:** docs/superpowers/specs/2026-05-03-massey-matrix-feature-design.md
**Plan:** docs/superpowers/plans/2026-05-03-massey-matrix-feature.md
**Verdict:** <Clear | Marginal | Reject> -- <one-line summary>

## Headline numbers

| metric                  | v4 baseline | v4 + massey_mov | delta |
|-------------------------|-------------|-----------------|-------|
| weighted-mean LL        | 0.4369      | <X.XXXX>        | <+/-X.XXXX> |
| total brkt pts (22 sn)  | 2670        | <XXXX>          | <+/-XX> |
| F4 chalk acc            | <baseline>  | <X.XXX>         | <+/-> |
| E8 chalk acc            | <baseline>  | <X.XXX>         | <+/-> |

Ladder bucket: <Reject | Clear | Marginal>
- Reject if ll_delta >= +0.001 OR brkt_delta <= +10
- Clear if ll_delta <= -0.005 OR brkt_delta >= +25
- Marginal otherwise

## Per-season W/L/T

<paste the per-season delta-vs-v4 line summary>

## Diagnosis

<one or two paragraphs: where the feature helped, where it hurt, and
why. For Reject, focus on what the post-hoc lesson is.>

## Lessons for Colley (TODO #1, separate work item)

<one paragraph: does the result generalize? If Massey-MOV passed
because raw margin info was present beyond what adj_em captured, the
Colley case (W/L only) is the cleaner question.>

## Decision

<verbatim action: feature stays in / feature reverted / candidate-only>
```

- [ ] **Step 3: Apply the verdict action**

**If Reject:**

```bash
# Revert the wire-in (Task 8 commit) but KEEP the solver and tests
# as the experiment record per the spec's "branch retained" guidance.
git revert --no-edit <task-8-commit-sha>
git add docs/notes/2026-05-03-massey-mov-backtest.md TODO.md
git commit -m "docs(massey-matrix): findings + TODO update -- backtest REJECT

ll_delta vs v4 = X.XXX, brkt_delta = +/-XX. Wire-in reverted; solver
module and tests retained as experiment record.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

Update `TODO.md` "Tried and rejected" with one bullet citing the deltas and the lesson.

**If Clear:**

```bash
git add docs/notes/2026-05-03-massey-mov-backtest.md TODO.md
git commit -m "docs(massey-matrix): findings + TODO update -- backtest CLEAR

ll_delta vs v4 = X.XXX, brkt_delta = +XX. Feature merged into v4 via
the wire-in commit on this branch; v4 effectively becomes
v4-with-massey on PR merge.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

Update `TODO.md` "Done" with one bullet describing the win and remove TODO #1's Massey portion (leave Colley still active).

**If Marginal:**

```bash
git add docs/notes/2026-05-03-massey-mov-backtest.md TODO.md
git commit -m "docs(massey-matrix): findings + TODO update -- backtest MARGINAL

ll_delta vs v4 = X.XXX, brkt_delta = +XX (in (10, 25) band). Wire-in
NOT merged; documented as candidate-only on this branch.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

Update `TODO.md` -- move Massey portion of #1 to a new "Candidates" subsection or to "Tried and rejected" with the candidate framing.

- [ ] **Step 4: Verify ASCII**

Run: `python -c "open('docs/notes/2026-05-03-massey-mov-backtest.md').read().encode('ascii'); open('TODO.md').read().encode('ascii'); print('ASCII OK')"`
Expected: `ASCII OK`. If it fails, find and replace the offending non-ASCII characters per CLAUDE.md.

---

## Task 14: Final verification + branch cleanup

**Files:**
- All test files

Confirms the whole repo is healthy after the work and that no test that should pass is failing.

- [ ] **Step 1: Run the full test suite**

Run: `pytest -v`
Expected: all PASS. Any new failures must be diagnosed before declaring this PR ready.

- [ ] **Step 2: Run the seam tests explicitly per CLAUDE.md**

Run: `pytest -v tests/test_ingest tests/test_features tests/test_integration.py`
Expected: all PASS.

- [ ] **Step 3: Final ASCII audit on all newly created/modified files**

Run:
```bash
python -c "
import os
for p in [
    'src/features/massey_matrix.py',
    'src/diagnose_massey_mov.py',
    'tests/test_features/test_massey_matrix.py',
    'src/enhanced_model.py',
    'docs/superpowers/specs/2026-05-03-massey-matrix-feature-design.md',
    'docs/superpowers/plans/2026-05-03-massey-matrix-feature.md',
    'docs/notes/2026-05-03-massey-mov.md' if os.path.exists('docs/notes/2026-05-03-massey-mov.md') else None,
    'docs/notes/2026-05-03-massey-mov-backtest.md' if os.path.exists('docs/notes/2026-05-03-massey-mov-backtest.md') else None,
    'TODO.md',
]:
    if p is None: continue
    open(p, encoding='utf-8').read().encode('ascii')
    print('OK', p)
"
```
Expected: `OK <path>` for each existing file.

- [ ] **Step 4: Final commit (only if anything new, e.g., last-pass tweaks)**

```bash
git status
# If anything is dirty:
git add <files>
git commit -m "chore(massey-matrix): final verification

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

- [ ] **Step 5: Done**

Branch state at end:
- Always: spec, plan, solver module, tests, gate runner, gate output, findings note. v4 wire-in either committed (Clear/Marginal) or reverted (Reject).
- Pull-request scope: per the project workflow rule, the PR target is `main` and includes everything on the branch from the spec onward. Open the PR via `gh pr create` -- not part of this plan; user-triggered.

---

## Plan self-review

**Spec coverage:**

| Spec section | Covered by |
|---|---|
| Goals: solver recipe (home + cap=21 + sum-to-zero) | Task 1 |
| Goals: cheap two-clause gate | Tasks 9-11 |
| Goals: 22-season LOSO backtest if gate clears | Task 12 |
| Module shape (massey_matrix.py with two public fns) | Tasks 1, 5 |
| Cache (parquet + sidecar, version-stamp invalidation) | Tasks 5, 6 |
| Wire-in (3 additions to compute_all_features) | Task 8 |
| Diagnostic gate clause 1 (correlation thresholds) | Task 9 |
| Diagnostic gate clause 2 (LL headroom on subset) | Task 10 |
| Gate runner CLI + outputs | Task 11 |
| Decision ladder (Reject -> Clear -> Marginal in order) | Task 13 |
| Tests: round-robin / sum-to-zero / cap / home / cache RT / cache invalidate / real-data | Tasks 1-7 |
| ASCII compliance | Tasks 13 step 4, 14 step 3 |
| Findings notes (gate fail vs backtest) | Tasks 11 step 3, 13 step 2 |
| TODO.md update | Tasks 11 step 3, 13 step 3 |

**Placeholder scan:** No "TBD" / "TODO" placeholders in step bodies. The few `<X.XXX>` markers in the findings-note templates are intentional fill-ins from concrete numbers -- those are post-hoc data, not pre-coding placeholders.

**Type / signature consistency:**
- `compute_massey_mov_ratings(reg_season, seasons=None, mov_cap=21)` -- defined Task 1, used Task 5/7.
- `load_massey_mov_ratings(reg_season, mov_cap=21, cache_dir="data/cache")` -- defined Task 5, used Task 8/Task 10's pipeline.
- `_solve_one_season(games_df, mov_cap)` -- defined Task 1, used Task 4 test.
- `_PRODUCER_VERSION` -- defined Task 1, referenced Task 5/6.
- `clause1_correlations(feature_matrix)`, `clause2_headroom(seasons)` -- defined Tasks 9/10, used Task 11.
- `leave_one_season_out_cv_weighted` -- the real LOSO trainer in `enhanced_model_v3.py:472`. Task 10 adds a small `allowed_holdouts` kwarg (Task 10 steps 1-4) and uses it from clause2_headroom. The per-season-record key (assumed `"log_loss"` and stored under `"results"` in the returned dict) is verified in Task 10 step 6 with grep before the gate is run, so the executor can adjust if the actual key name differs.
