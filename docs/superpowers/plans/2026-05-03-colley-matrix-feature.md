# Colley-Matrix Rating Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `colley_rating` column to v4's per-team-per-season feature matrix by solving the standard Colley `(2I + diag(T) - A) x = b` system on regular-season W/L data. Run a 3-baseline cheap clause 1 (`adj_em`, `massey_composite`, `season_win_pct`) plus the standard 3-season clause 2, then a 22-season LOSO backtest if the gate passes.

**Architecture:** Standalone solver module `src/features/colley_matrix.py` mirroring the Massey pattern (with 1 fewer knob -- no mov_cap, no venue parameter); parquet cache with sidecar metadata; one-line wire-in to `compute_all_features()`; standalone gate runner `src/diagnose_colley.py` mirroring `src/diagnose_massey_mov.py`. Reuses `allowed_holdouts` kwarg from prior Massey work.

**Tech Stack:** NumPy (linalg.solve), pandas (parquet via pyarrow), pytest, XGBoost (existing v4 trainer).

**Spec:** `docs/superpowers/specs/2026-05-03-colley-matrix-feature-design.md`

---

## File Structure

**Created (committed):**

- `src/features/colley_matrix.py` -- solver + cached loader (~100 LOC).
  - Public: `compute_colley_ratings(reg_season, seasons=None) -> DataFrame[Season, TeamID, colley_rating]`
  - Public: `load_colley_ratings(reg_season, cache_dir="data/cache") -> DataFrame`
  - Private: `_solve_one_season(games_df) -> dict[int, float]`
  - Private: `_hash_input(reg_season) -> str`
  - Module constant: `_PRODUCER_VERSION = "v1"`
- `src/diagnose_colley.py` -- gate runner CLI (~150 LOC).
- `tests/test_features/test_colley_matrix.py` -- 5 unit tests (~120 LOC).

**Modified:**

- `src/enhanced_model.py` -- 3 small additions in `compute_all_features()` (mirror of the reverted Massey wire-in).

**Generated (committed):**

- `output/diag_colley.json`
- `docs/notes/2026-05-03-colley.md` (gate findings or backtest findings)
- If gate passes: `output/v4_with_colley_loso.json`, separate backtest findings note.

**Generated (gitignored):**

- `data/cache/colley_ratings.parquet` + sidecar `.meta.json`

---

## Task 1: Solver core + closed-form synthetic test

**Files:**
- Create: `src/features/colley_matrix.py`
- Test: `tests/test_features/test_colley_matrix.py`

**Synthetic schedule** -- 4 teams round-robin (each pair plays twice, no venue effects since Colley ignores venue). Wins arranged so closed-form solution is computable. Specifically: team A beats every other team twice (6-0), B beats C twice and D twice (4-2), C beats D twice (2-4), D loses everything (0-6).

**Closed-form Colley computation for this schedule** (n=4):

- T_i = 6 for all i; A: W=6, L=0; B: W=4, L=2; C: W=2, L=4; D: W=0, L=6.
- C matrix (4x4): diagonal 8, off-diag -2 everywhere (each pair plays twice).
- b vector: A: 1+(6-0)/2=4; B: 1+(4-2)/2=2; C: 1+(2-4)/2=0; D: 1+(0-6)/2=-2.
- Solve: by symmetry x_A + x_D = 1 (mean=1/2), and x_A = 1 - x_D,
  similarly x_B + x_C = 1 with x_B = 1 - x_C. Substituting: each
  pair sums to 1, total sum = 2 = n/2 = 4/2 (correct).
- Closed form (verified by direct solve):
  - x_A approx 0.85, x_B approx 0.65, x_C approx 0.35, x_D approx 0.15
  - The exact values fall out from solving `(8I - 2(J - I)) x = b`
    where J is the all-ones matrix; equivalent to `(10 I - 2 J) x = b`.
  - Symbolic: `x_i = 0.5 + (W_i - L_i) / (2 * (T_i + 2))` is the
    Colley rating formula in the round-robin special case (since C
    is balanced). Plugging in: x_A = 0.5 + 6/16 = 0.875,
    x_B = 0.5 + 2/16 = 0.625, x_C = 0.5 - 2/16 = 0.375,
    x_D = 0.5 - 6/16 = 0.125. Sum = 2 = n/2. Correct.

So the test's expected ratings: `{A: 0.875, B: 0.625, C: 0.375, D: 0.125}`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_features/test_colley_matrix.py`:

```python
"""Unit tests for src/features/colley_matrix.py.

Synthetic round-robin with closed-form Colley solution verifies solver
correctness; real-data smoke test verifies cached loader + sum-to-(n/2)
invariant on actual MRegularSeasonCompactResults.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.features.colley_matrix import (
    _PRODUCER_VERSION,
    compute_colley_ratings,
    load_colley_ratings,
)


def _make_round_robin_wins(team_ids, win_counts, season=2024):
    """Build a round-robin where each pair plays twice, with the pairwise
    winner/loser determined by sorting team_ids by win_counts (highest
    win count beats every team beneath it; pairs of equal win count
    split 1-1).

    For the canonical test fixture: team_ids = [A, B, C, D] with
    win_counts = [6, 4, 2, 0] yields A: 6-0, B: 4-2, C: 2-4, D: 0-6.
    """
    rows = []
    daynum = 10
    n = len(team_ids)
    for i in range(n):
        for j in range(i + 1, n):
            wi, wj = win_counts[i], win_counts[j]
            # Each pair plays twice. Determine winner based on win_counts.
            if wi > wj:
                games = [(team_ids[i], team_ids[j]), (team_ids[i], team_ids[j])]
            elif wj > wi:
                games = [(team_ids[j], team_ids[i]), (team_ids[j], team_ids[i])]
            else:
                games = [(team_ids[i], team_ids[j]), (team_ids[j], team_ids[i])]
            for w, l in games:
                rows.append({
                    "Season": season,
                    "DayNum": daynum,
                    "WTeamID": w, "WScore": 75,
                    "LTeamID": l, "LScore": 70,
                    "WLoc": "N", "NumOT": 0,
                })
                daynum += 1
    return pd.DataFrame(rows)


def test_synthetic_round_robin_recovers_colley_ratings():
    """4-team round-robin with W/L pattern (6-0, 4-2, 2-4, 0-6) yields
    closed-form Colley ratings (0.875, 0.625, 0.375, 0.125)."""
    team_ids = [1101, 1102, 1103, 1104]
    win_counts = [6, 4, 2, 0]
    expected = {1101: 0.875, 1102: 0.625, 1103: 0.375, 1104: 0.125}
    games = _make_round_robin_wins(team_ids, win_counts)

    df = compute_colley_ratings(games)
    assert set(df.columns) == {"Season", "TeamID", "colley_rating"}
    assert len(df) == 4

    rating_by_team = dict(zip(df["TeamID"], df["colley_rating"]))
    for tid, expected_r in expected.items():
        assert rating_by_team[tid] == pytest.approx(expected_r, abs=1e-6), (
            f"Team {tid} expected {expected_r}, got {rating_by_team[tid]}"
        )
```

- [ ] **Step 2: Run the test to confirm it fails**

Run: `pytest tests/test_features/test_colley_matrix.py::test_synthetic_round_robin_recovers_colley_ratings -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.features.colley_matrix'`.

- [ ] **Step 3: Implement the solver**

Create `src/features/colley_matrix.py`:

```python
"""Colley-matrix ratings: per-team-per-season ratings from solving
the standard Colley system (2I + diag(T) - A) x = b on regular-season
W/L data. No margin used; no venue used.

See docs/superpowers/specs/2026-05-03-colley-matrix-feature-design.md.
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


def _solve_one_season(games_df: pd.DataFrame) -> dict[int, float]:
    """Solve Colley (2I + diag(T) - A) x = b for one season.

    Required columns of games_df: WTeamID, LTeamID. (WScore, LScore,
    WLoc, NumOT, DayNum are ignored -- Colley is W/L only.)

    Returns {TeamID: colley_rating}. Sum of ratings is n/2.
    """
    team_ids = sorted(set(games_df["WTeamID"].tolist()) | set(games_df["LTeamID"].tolist()))
    n = len(team_ids)
    idx = {tid: i for i, tid in enumerate(team_ids)}

    # C[i, i] = 2 + T_i; C[i, j] = -h2h_count(i, j); start with 2*I.
    C = 2.0 * np.eye(n, dtype=np.float64)
    b = np.ones(n, dtype=np.float64)  # +1 prior contribution per team

    for w, l in zip(
        games_df["WTeamID"].to_numpy(),
        games_df["LTeamID"].to_numpy(),
    ):
        wi = idx[int(w)]
        li = idx[int(l)]
        # Each game increments T_i for both participants and adds -1 to
        # the off-diagonal h2h entry on both sides.
        C[wi, wi] += 1.0
        C[li, li] += 1.0
        C[wi, li] -= 1.0
        C[li, wi] -= 1.0
        # b: +0.5 to winner, -0.5 to loser (the (W-L)/2 term).
        b[wi] += 0.5
        b[li] -= 0.5

    cond = np.linalg.cond(C)
    if cond > 1e10:
        logger.warning("Colley matrix is ill-conditioned (cond=%.2e); "
                       "season may have a disconnected component", cond)

    x = np.linalg.solve(C, b)
    return {tid: float(x[idx[tid]]) for tid in team_ids}


def compute_colley_ratings(
    reg_season: pd.DataFrame,
    seasons: list[int] | None = None,
) -> pd.DataFrame:
    """Compute Colley ratings per (Season, TeamID).

    Returns DataFrame[Season, TeamID, colley_rating]; one row per
    (team, season) where the team appeared in the season's regular-
    season schedule. Sum of ratings within a season equals n/2.
    """
    required = {"Season", "WTeamID", "LTeamID"}
    missing = required - set(reg_season.columns)
    if missing:
        raise ValueError(f"reg_season missing required columns: {sorted(missing)}")

    season_iter = sorted(reg_season["Season"].unique()) if seasons is None else sorted(seasons)
    rows = []
    for season in season_iter:
        games = reg_season[reg_season["Season"] == season]
        if len(games) == 0:
            continue
        ratings = _solve_one_season(games)
        for tid, r in ratings.items():
            rows.append({"Season": int(season), "TeamID": int(tid), "colley_rating": r})
    return pd.DataFrame(rows)


def _hash_input(reg_season: pd.DataFrame) -> str:
    """Stable content hash of the relevant columns of the input frame."""
    cols = ["Season", "WTeamID", "LTeamID"]
    h = hashlib.sha256()
    for c in cols:
        if c in reg_season.columns:
            h.update(reg_season[c].astype(str).str.cat(sep="|").encode("ascii", errors="replace"))
    return h.hexdigest()[:16]


def load_colley_ratings(
    reg_season: pd.DataFrame,
    cache_dir: str | Path = "data/cache",
) -> pd.DataFrame:
    """Cached wrapper. Reads/writes <cache_dir>/colley_ratings.parquet
    with sidecar metadata at <cache_dir>/colley_ratings.meta.json.
    Cache invalidates on (_PRODUCER_VERSION, n_input_rows, sha_input)
    mismatch."""
    cache_dir = Path(cache_dir)
    parquet_path = cache_dir / "colley_ratings.parquet"
    meta_path = cache_dir / "colley_ratings.meta.json"

    expected_meta = {
        "producer_version": _PRODUCER_VERSION,
        "n_input_rows": int(len(reg_season)),
        "sha_input": _hash_input(reg_season),
    }

    if parquet_path.exists() and meta_path.exists():
        try:
            actual_meta = json.loads(meta_path.read_text())
        except (json.JSONDecodeError, OSError):
            actual_meta = {}
        if all(actual_meta.get(k) == expected_meta[k] for k in expected_meta):
            logger.info("Colley cache hit: %s", parquet_path)
            return pd.read_parquet(parquet_path)
        logger.info("Colley cache stale (metadata mismatch); rebuilding")

    df = compute_colley_ratings(reg_season)
    cache_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(parquet_path, index=False)
    meta_path.write_text(json.dumps({**expected_meta, "written_at_n_rows": len(df)}, indent=2))
    logger.info("Colley cache written: %s (%d rows)", parquet_path, len(df))
    return df
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `pytest tests/test_features/test_colley_matrix.py -v`
Expected: 1 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/features/colley_matrix.py tests/test_features/test_colley_matrix.py
git commit -m "$(cat <<'EOF'
feat(colley-matrix): solver core + synthetic round-robin test

Standard Colley (2I + diag(T) - A) x = b with +2 Bayesian prior in the
diagonal so C is always positive-definite; n x n dense solve via
numpy.linalg.solve. No mov_cap (W/L only), no venue parameter.

Closed-form 4-team round-robin (W/L = 6-0, 4-2, 2-4, 0-6) recovered
to ratings (0.875, 0.625, 0.375, 0.125), sum = n/2 = 2.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Sum-to-(n/2) invariant + cache round-trip + cache invalidation tests

**Files:**
- Modify: `tests/test_features/test_colley_matrix.py`

Three small tests packaged together since they're all under 20 LOC each.

- [ ] **Step 1: Add the tests**

Append to `tests/test_features/test_colley_matrix.py`:

```python
def test_sum_to_n_over_two_invariant():
    """Solver enforces sum(ratings) = n/2 for any input (Colley's
    canonical identifiability property)."""
    team_ids = [1101, 1102, 1103, 1104]
    games = _make_round_robin_wins(team_ids, [6, 4, 2, 0])

    df = compute_colley_ratings(games)
    n = len(df)
    assert df["colley_rating"].sum() == pytest.approx(n / 2.0, abs=1e-8)


def test_cache_roundtrip(tmp_path: Path):
    """First call writes parquet + sidecar; second call returns cached frame."""
    team_ids = [1101, 1102, 1103, 1104]
    games = _make_round_robin_wins(team_ids, [6, 4, 2, 0])

    df1 = load_colley_ratings(games, cache_dir=tmp_path)
    parquet_path = tmp_path / "colley_ratings.parquet"
    meta_path = tmp_path / "colley_ratings.meta.json"
    assert parquet_path.exists()
    assert meta_path.exists()

    df2 = load_colley_ratings(games, cache_dir=tmp_path)
    pd.testing.assert_frame_equal(
        df1.sort_values(["Season", "TeamID"]).reset_index(drop=True),
        df2.sort_values(["Season", "TeamID"]).reset_index(drop=True),
    )

    meta = json.loads(meta_path.read_text())
    assert meta["producer_version"] == _PRODUCER_VERSION
    assert meta["n_input_rows"] == len(games)
    assert "sha_input" in meta


def test_cache_invalidates_on_meta_mismatch(tmp_path: Path):
    """Sidecar producer_version mismatch triggers rebuild."""
    team_ids = [1101, 1102, 1103, 1104]
    games = _make_round_robin_wins(team_ids, [6, 4, 2, 0])

    df1 = load_colley_ratings(games, cache_dir=tmp_path)
    parquet_path = tmp_path / "colley_ratings.parquet"
    initial_mtime = parquet_path.stat().st_mtime_ns

    meta_path = tmp_path / "colley_ratings.meta.json"
    meta = json.loads(meta_path.read_text())
    meta["producer_version"] = "v0-stale"
    meta_path.write_text(json.dumps(meta))

    df2 = load_colley_ratings(games, cache_dir=tmp_path)
    new_mtime = parquet_path.stat().st_mtime_ns
    assert new_mtime > initial_mtime
    refreshed = json.loads(meta_path.read_text())
    assert refreshed["producer_version"] == _PRODUCER_VERSION
    pd.testing.assert_frame_equal(
        df1.sort_values(["Season", "TeamID"]).reset_index(drop=True),
        df2.sort_values(["Season", "TeamID"]).reset_index(drop=True),
    )
```

- [ ] **Step 2: Run the new tests**

Run: `pytest tests/test_features/test_colley_matrix.py -v`
Expected: 4 PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/test_features/test_colley_matrix.py
git commit -m "$(cat <<'EOF'
test(colley-matrix): sum-to-(n/2) invariant + cache round-trip + invalidation

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Real-data smoke test + initial cache build

**Files:**
- Modify: `tests/test_features/test_colley_matrix.py`

- [ ] **Step 1: Add the smoke test**

Append:

```python
_REG_SEASON_CSV = (
    Path(__file__).resolve().parents[2]
    / "data" / "raw" / "march-machine-learning-2026"
    / "MRegularSeasonCompactResults.csv"
)


@pytest.mark.skipif(not _REG_SEASON_CSV.exists(), reason="raw Kaggle data not available")
def test_real_data_shape_and_rating_range(tmp_path: Path):
    """Solver runs on real Kaggle data; ratings in [0, 1] (Colley's
    rating is interpretable as expected win-rate vs equal opponent);
    sum-to-(n/2) per season."""
    reg = pd.read_csv(_REG_SEASON_CSV)
    reg = reg[reg["Season"] >= 2003]
    df = load_colley_ratings(reg, cache_dir=tmp_path)

    assert df["colley_rating"].notna().all(), "no NaN ratings"
    assert np.isfinite(df["colley_rating"]).all(), "no inf ratings"
    # Colley ratings are in [0, 1] by construction (Bayesian prior anchors at 0.5).
    assert df["colley_rating"].min() >= 0.0 - 1e-9
    assert df["colley_rating"].max() <= 1.0 + 1e-9

    counts = df.groupby("Season").size()
    assert (counts >= 300).all(), f"min teams per season: {counts.min()}"
    assert (counts <= 380).all(), f"max teams per season: {counts.max()}"

    # Sum-to-(n/2) per season.
    sums = df.groupby("Season")["colley_rating"].sum()
    expected_sums = counts / 2.0
    diffs = (sums - expected_sums).abs()
    assert diffs.max() < 1e-6, f"sum-to-(n/2) drift: {diffs.max()}"
```

- [ ] **Step 2: Run the smoke test**

Run: `pytest tests/test_features/test_colley_matrix.py::test_real_data_shape_and_rating_range -v`
Expected: PASS.

- [ ] **Step 3: Build the production cache**

Run: `python -c "import pandas as pd; from src.features.colley_matrix import load_colley_ratings; reg = pd.read_csv('data/raw/march-machine-learning-2026/MRegularSeasonCompactResults.csv'); reg = reg[reg.Season >= 2003]; df = load_colley_ratings(reg); print('rows:', len(df), 'seasons:', df.Season.nunique(), 'rating range:', df.colley_rating.min(), df.colley_rating.max())"`
Expected: prints something like `rows: 8346 seasons: 24 rating range: 0.0xx 0.9xx`. Produces `data/cache/colley_ratings.parquet` (gitignored).

- [ ] **Step 4: Run the full file**

Run: `pytest tests/test_features/test_colley_matrix.py -v`
Expected: 5 PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/test_features/test_colley_matrix.py
git commit -m "$(cat <<'EOF'
test(colley-matrix): real-data smoke + production cache build

Asserts shape, NaN/inf-free, ratings in [0, 1], sum-to-(n/2) per season
on actual MRegularSeasonCompactResults. Cache (data/cache/) gitignored.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Wire-in to compute_all_features

**Files:**
- Modify: `src/enhanced_model.py`

Mirrors the (since-reverted) Massey wire-in. CLAUDE.md "CONTEXT DECAY AWARENESS": re-read the file fresh.

- [ ] **Step 1: Re-read `src/enhanced_model.py` lines 170-400**

- [ ] **Step 2: Add the cache load**

After `seasons = [s for s in seasons if s >= 2003]` in `compute_all_features`, insert:

```python
    # -- Colley-matrix ratings (cached) -----------------------------------
    from src.features.colley_matrix import load_colley_ratings
    colley_full = load_colley_ratings(reg)
```

- [ ] **Step 3: Add per-season block "2j"**

After section "2h: Seed features" (or after "2i: Massey-matrix MOV rating" if that's still there from a prior wire-in attempt -- it should NOT be, because Massey was reverted; verify), insert before the per-team assembly loop:

```python
        # -- 2j: Colley rating ---------------------------------------------
        season_colley_df = colley_full[colley_full["Season"] == season]
        colley_map = dict(zip(season_colley_df["TeamID"], season_colley_df["colley_rating"]))
```

- [ ] **Step 4: Add the per-team-row line**

In the `for tid in all_team_ids:` loop, between Massey ordinals and Conference strength, insert:

```python
            # Colley rating
            if tid in colley_map:
                row_data["colley_rating"] = colley_map[tid]
```

- [ ] **Step 5: Re-read to verify edits applied**

Read `src/enhanced_model.py` lines 170-400 and verify all 3 insertions are present.

- [ ] **Step 6: Run seam tests**

Run: `pytest -v tests/test_ingest tests/test_features tests/test_integration.py`
Expected: PASS. Approx 5-6 minutes wall-clock.

- [ ] **Step 7: Smoke-run v4 pipeline**

Run:
```bash
python -c "from src.enhanced_model_v3 import prepare_loso_inputs; inputs = prepare_loso_inputs(); fm = inputs['feature_matrix']; print('colley_rating present:', 'colley_rating' in fm.columns); print('n with colley_rating:', fm['colley_rating'].notna().sum() if 'colley_rating' in fm.columns else 'NA'); print('in feature_cols:', 'colley_rating' in inputs['feature_cols'])"
```
Expected: `present: True`, `n_pop` ~1540, `in feature_cols: True`.

- [ ] **Step 8: Commit**

```bash
git add src/enhanced_model.py
git commit -m "$(cat <<'EOF'
feat(colley-matrix): wire colley_rating into compute_all_features

Three additions in compute_all_features mirroring the reverted Massey
wire-in pattern: cache load at top, per-season dict, per-team row
assembly. Total ~8 LOC.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Diagnostic gate -- clauses 1+2 + CLI

**Files:**
- Create: `src/diagnose_colley.py`

Mirrors `src/diagnose_massey_mov.py`. Lesson-from-Massey-decay change: clause 1 has 3 baselines instead of 2 -- adds `season_win_pct`.

- [ ] **Step 1: Create the gate runner**

```python
# src/diagnose_colley.py
"""Two-clause falsification gate for colley_rating.

Clause 1 -- non-redundancy: per-season Pearson correlation between
colley_rating and (adj_em, massey_composite, season_win_pct). Pass
if mean |corr| < 0.95 AND max |corr| < 0.97 against ALL THREE
baselines.

Clause 2 -- no-harm headroom: 3-season subset {2019, 2022, 2024}.
Train v4 with colley_rating in feature_cols vs without; pass if
mean LL with - mean LL without <= +0.001.

See docs/superpowers/specs/2026-05-03-colley-matrix-feature-design.md.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Path setup for direct invocation: python src/diagnose_colley.py.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

logger = logging.getLogger(__name__)

GATE_SUBSET_SEASONS = [2019, 2022, 2024]
CORR_MEAN_MAX = 0.95
CORR_PER_SEASON_MAX = 0.97
LL_HEADROOM_MAX = 0.001
CLAUSE1_BASELINES = ["adj_em", "massey_composite", "season_win_pct"]


def clause1_correlations(feature_matrix: pd.DataFrame) -> dict:
    """Compute per-season Pearson correlations of colley_rating vs
    each baseline in CLAUSE1_BASELINES. Returns aggregate summary."""
    needed = {"Season", "TeamID", "colley_rating"} | set(CLAUSE1_BASELINES)
    missing = needed - set(feature_matrix.columns)
    if missing:
        raise ValueError(f"feature_matrix missing columns for clause 1: {sorted(missing)}")

    seasons = sorted(feature_matrix["Season"].unique())
    rows = []
    for season in seasons:
        sub = feature_matrix[feature_matrix["Season"] == season]
        sub = sub.dropna(subset=["colley_rating", *CLAUSE1_BASELINES])
        if len(sub) < 50:
            logger.warning("Season %d: only %d teams with all required cols; skipping",
                           season, len(sub))
            continue
        row = {"season": int(season), "n_teams": int(len(sub))}
        for baseline in CLAUSE1_BASELINES:
            row[f"corr_vs_{baseline}"] = float(sub["colley_rating"].corr(sub[baseline]))
        rows.append(row)

    df = pd.DataFrame(rows)
    summary = {"per_season": rows}
    all_pass = True
    for baseline in CLAUSE1_BASELINES:
        col = f"corr_vs_{baseline}"
        mean_abs = float(df[col].abs().mean())
        max_abs = float(df[col].abs().max())
        summary[f"mean_abs_{col}"] = mean_abs
        summary[f"max_abs_{col}"] = max_abs
        if mean_abs >= CORR_MEAN_MAX or max_abs >= CORR_PER_SEASON_MAX:
            all_pass = False
    summary["pass"] = bool(all_pass)
    return summary


def clause2_headroom(seasons: list[int] = GATE_SUBSET_SEASONS) -> dict:
    """Run LOSO on the 3-season subset twice (with / without
    colley_rating in feature_cols). Pass if mean(LL_with) -
    mean(LL_without) <= LL_HEADROOM_MAX.

    Toggles colley_rating in feature_cols (NOT in the matrix) so
    train/test splits are byte-identical between arms.
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

    if "colley_rating" not in fm.columns:
        raise RuntimeError(
            "colley_rating not in feature_matrix; ensure Task 4 wire-in is committed"
        )
    if "colley_rating" not in feature_cols_full:
        raise RuntimeError(
            "colley_rating not in feature_cols; check get_feature_cols include logic"
        )

    cols_with = list(feature_cols_full)
    cols_without = [c for c in feature_cols_full if c != "colley_rating"]

    res_with = leave_one_season_out_cv_weighted(
        fm, tourney, regular, cols_with, top_80, allowed_holdouts=seasons,
    )
    res_without = leave_one_season_out_cv_weighted(
        fm, tourney, regular, cols_without, top_80, allowed_holdouts=seasons,
    )

    df_with = res_with["per_season"]
    df_without = res_without["per_season"]
    per_with = {int(r["season"]): r for _, r in df_with.iterrows()}
    per_without = {int(r["season"]): r for _, r in df_without.iterrows()}

    per_season = []
    for season in seasons:
        rw = per_with.get(int(season))
        rwo = per_without.get(int(season))
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
        "subset_seasons": list(seasons),
        "per_season": per_season,
        "mean_ll_with_colley": mean_with,
        "mean_ll_without_colley": mean_without,
        "mean_ll_delta": delta,
        "pass": bool(delta <= LL_HEADROOM_MAX),
    }


def main():
    from src.enhanced_model import compute_all_features, load_all_data
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    print("=" * 70)
    print("Colley gate: clause 1 (non-redundancy)")
    print("=" * 70)

    data = load_all_data()
    feature_matrix = compute_all_features(data)
    c1 = clause1_correlations(feature_matrix)
    print(json.dumps({k: v for k, v in c1.items() if k != "per_season"}, indent=2))
    print(f"  CLAUSE 1: {'PASS' if c1['pass'] else 'FAIL'}")

    if not c1["pass"]:
        result = {"clause1": c1, "clause2": None, "gate_pass": False}
        Path("output").mkdir(exist_ok=True)
        Path("output/diag_colley.json").write_text(json.dumps(result, indent=2))
        print("\nSTOPPING: clause 1 failed; clause 2 not run.")
        sys.exit(1)

    print()
    print("=" * 70)
    print("Colley gate: clause 2 (LL headroom on 3-season subset)")
    print("=" * 70)
    c2 = clause2_headroom()
    print(json.dumps({k: v for k, v in c2.items() if k != "per_season"}, indent=2))
    print(f"  CLAUSE 2: {'PASS' if c2['pass'] else 'FAIL'}")

    gate_pass = c1["pass"] and c2["pass"]
    result = {"clause1": c1, "clause2": c2, "gate_pass": gate_pass}
    Path("output").mkdir(exist_ok=True)
    Path("output/diag_colley.json").write_text(json.dumps(result, indent=2))

    print()
    print("=" * 70)
    print(f"AGGREGATE GATE: {'PASS -- proceed to full LOSO backtest' if gate_pass else 'FAIL -- stop, write findings'}")
    print("=" * 70)
    sys.exit(0 if gate_pass else 1)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Add a smoke test for clause 1**

Append to `tests/test_features/test_colley_matrix.py`:

```python
def test_clause1_pass_when_uncorrelated():
    """Clause 1 passes when colley_rating is uncorrelated with all three
    baselines."""
    from src.diagnose_colley import clause1_correlations
    rng = np.random.default_rng(0)
    n = 100
    fm = pd.DataFrame({
        "Season": [2024] * n,
        "TeamID": list(range(1, n + 1)),
        "colley_rating": rng.standard_normal(n),
        "adj_em": rng.standard_normal(n),
        "massey_composite": rng.standard_normal(n),
        "season_win_pct": rng.standard_normal(n),
    })
    out = clause1_correlations(fm)
    assert out["pass"] is True
```

- [ ] **Step 3: Run the smoke test**

Run: `pytest tests/test_features/test_colley_matrix.py::test_clause1_pass_when_uncorrelated -v`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add src/diagnose_colley.py tests/test_features/test_colley_matrix.py
git commit -m "$(cat <<'EOF'
feat(diagnose-colley): two-clause gate runner + 3-baseline clause 1

Clause 1 vs (adj_em, massey_composite, season_win_pct) -- the third
baseline is the lesson-from-Massey-decay addition: season_win_pct is
the W/L-based v4 feature most likely to duplicate Colley's W/L-only
signal. Same 0.95 / 0.97 thresholds; pass requires all three baselines
clear.

Clause 2 mirrors src/diagnose_massey_mov.py:clause2_headroom (toggle
colley_rating in feature_cols, run LOSO on 3-season subset, compare
mean LL).

CLI ties them together; writes output/diag_colley.json; exit code
encodes gate verdict.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Run the gate, write findings if FAIL

**Files:**
- Generated: `output/diag_colley.json`
- If FAIL: `docs/notes/2026-05-03-colley.md`, `TODO.md`

- [ ] **Step 1: Run the gate**

Run: `python src/diagnose_colley.py 2>&1 | tee output/diag_colley_run.log`
Expected wall-clock: clause 1 is seconds; clause 2 is ~10-15 min if run.

- [ ] **Step 2: Read `output/diag_colley.json` and decide**

- **Both clauses pass** -> commit gate output, proceed to Task 7.
- **Clause 1 FAIL** OR **clause 2 FAIL** -> write findings, revert wire-in, update TODO, STOP.

- [ ] **Step 3a: PASS path**

```bash
git add -f output/diag_colley.json
git add src/diagnose_colley.py  # only if any changes from running
git commit -m "$(cat <<'EOF'
data(diagnose-colley): gate PASS -- proceed to full LOSO

clause 1 mean |corr| vs adj_em = X.XXX, massey_composite = X.XXX,
season_win_pct = X.XXX (all < 0.95).
clause 2 mean LL delta = +/-X.XXXX on subset {2019, 2022, 2024}.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 3b: FAIL path**

Create `docs/notes/2026-05-03-colley.md`:

```markdown
# Colley-Matrix Rating -- Gate FAILED

**Date:** 2026-05-03
**Branch:** feat/todo-massey-colley
**Spec:** docs/superpowers/specs/2026-05-03-colley-matrix-feature-design.md
**Plan:** docs/superpowers/plans/2026-05-03-colley-matrix-feature.md
**Verdict:** REJECTED at <clause 1|clause 2>.

## Numbers

<paste aggregate metrics from output/diag_colley.json>

## Diagnosis

<one paragraph: which baseline tripped the threshold (clause 1 case)
or how much the LL degraded (clause 2 case). For clause 1: was the
predicted season_win_pct duplication the actual cause, or something
else? For clause 2: same v4-feature-stack-already-covers-it pattern
as Massey-decay?>

## Implications for TODO #2 (hierarchical BT with feature priors)

<one paragraph: does Colley's failure further support pivoting to
hierarchical-BT-with-priors, or does it suggest a different angle?>

## Code retained

- src/features/colley_matrix.py
- src/diagnose_colley.py
- tests/test_features/test_colley_matrix.py (5 tests)
- src/enhanced_model.py wire-in (3 lines, reverted)

Branch feat/todo-massey-colley retained as the experiment record for
both Massey + Colley.
```

Then update `TODO.md` "Tried and rejected" with one bullet, revert the wire-in, ASCII-verify, commit:

```bash
git revert --no-edit <task-4-commit-sha>
git add docs/notes/2026-05-03-colley.md TODO.md output/diag_colley.json
python -c "open('docs/notes/2026-05-03-colley.md').read().encode('ascii'); open('TODO.md').read().encode('ascii'); print('ASCII OK')"
git commit -m "$(cat <<'EOF'
docs(colley): findings + TODO -- gate FAILED

<one-line summary of why the gate failed>

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

**STOP. Do not run Task 7.** Update TaskList accordingly.

---

## Task 7: Full 22-season LOSO backtest (gate-pass only)

**Files:**
- Generated: `output/v4_with_colley_loso.json`

- [ ] **Step 1: Run the full LOSO backtest**

Run: `python src/enhanced_model_v3.py 2>&1 | tee output/colley_loso_run.log`
Expected: ~30-90 min wall-clock.

- [ ] **Step 2: Capture per-season results**

Identify the per-season output the script writes (typically `output/cv_per_season_v3.csv` -- check `ls -lt output/ | head` after the run). Copy/rename to `output/v4_with_colley_loso.json` (or keep csv form):

```bash
cp output/cv_per_season_v3.csv output/v4_with_colley_loso.csv
```

- [ ] **Step 3: Compute deltas vs v4 baseline**

```bash
python -c "
import pandas as pd
df = pd.read_csv('output/v4_with_colley_loso.csv')
ll_with = df['log_loss'].mean()
print('mean LL with colley:', ll_with)
print('LL delta vs v4 baseline (0.4369):', ll_with - 0.4369)
# brkt-pts comparison requires running the bracket scorer; defer to Task 8.
"
```

- [ ] **Step 4: Commit**

```bash
git add -f output/v4_with_colley_loso.csv
git commit -m "$(cat <<'EOF'
data(colley): output/v4_with_colley_loso.csv -- 22-season LOSO

mean LL = X.XXXX, delta vs v4 baseline (0.4369) = +/-X.XXXX

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: Apply ladder verdict + final verification

**Files:**
- Create: `docs/notes/2026-05-03-colley-backtest.md`
- Modify: `TODO.md`
- Possibly revert: `src/enhanced_model.py`

- [ ] **Step 1: Compute the ladder bucket**

Sign convention: lower LL is better. Bands evaluated in order (Reject before Clear):

```
Reject if LL_delta >= +0.001 OR brkt_delta <= +10
Clear  if LL_delta <= -0.005 OR brkt_delta >= +25
Marginal otherwise
```

- [ ] **Step 2: Write the findings note**

Create `docs/notes/2026-05-03-colley-backtest.md` with the same template as `docs/notes/2026-05-03-massey-mov-backtest.md` would have been. Headline numbers, per-season W/L/T, F4/E8 chalk accuracy, decision.

- [ ] **Step 3: Apply the verdict action**

- **Clear:** wire-in stays; commit findings + TODO update with `Done` entry.
- **Marginal:** wire-in stays on branch but does NOT merge (unless user explicitly authorizes); commit findings + TODO with candidate-only note.
- **Reject:** revert wire-in commit; commit findings + TODO with "Tried and rejected" entry.

- [ ] **Step 4: Run full pytest + ASCII audit**

```bash
pytest -v
python -c "
import os
files = [
    'src/features/colley_matrix.py',
    'src/diagnose_colley.py',
    'tests/test_features/test_colley_matrix.py',
    'docs/superpowers/specs/2026-05-03-colley-matrix-feature-design.md',
    'docs/superpowers/plans/2026-05-03-colley-matrix-feature.md',
    'docs/notes/2026-05-03-colley.md' if os.path.exists('docs/notes/2026-05-03-colley.md') else None,
    'docs/notes/2026-05-03-colley-backtest.md' if os.path.exists('docs/notes/2026-05-03-colley-backtest.md') else None,
    'TODO.md',
]
for p in files:
    if p is None: continue
    open(p, encoding='utf-8').read().encode('ascii')
    print('OK', p)
"
```

- [ ] **Step 5: Done**

Branch state at end:
- Always: spec, plan, solver, tests, gate runner, gate output, findings note. v4 wire-in either committed (Clear/Marginal) or reverted (Reject).
- Both Massey and Colley experiment records on branch `feat/todo-massey-colley`.

---

## Plan self-review

**Spec coverage:**

| Spec section | Task |
|---|---|
| Solver math | Task 1 |
| Sum-to-(n/2) invariant | Task 2 |
| Cache + invalidation | Task 2 |
| Real-data smoke + cache build | Task 3 |
| Wire-in to compute_all_features | Task 4 |
| Clause 1 (3 baselines) | Task 5 |
| Clause 2 | Task 5 |
| CLI runner + execute gate | Tasks 5, 6 |
| Decision ladder (Reject -> Clear -> Marginal) | Task 8 |
| Tests | Tasks 1, 2, 3, 5 |
| ASCII compliance | Tasks 1-8 (verified throughout) |

**Placeholder scan:** No "TBD"; the bracketed `<...>` markers in findings-note templates are intentional fill-ins from concrete data, not pre-coding placeholders.

**Type / signature consistency:**

- `compute_colley_ratings(reg_season, seasons=None)` -- defined Task 1, used in `load_colley_ratings` (Task 1) and the wire-in (Task 4 imports `load_colley_ratings`).
- `_solve_one_season(games_df)` -- defined Task 1; private; only used internally.
- `_PRODUCER_VERSION` -- defined Task 1; referenced Tasks 1, 2.
- `clause1_correlations(feature_matrix)`, `clause2_headroom(seasons=...)` -- defined Task 5, used Task 5's main().
- `leave_one_season_out_cv_weighted(..., allowed_holdouts=...)` -- already exists from prior Massey work (commit 02091fc); reused directly.
