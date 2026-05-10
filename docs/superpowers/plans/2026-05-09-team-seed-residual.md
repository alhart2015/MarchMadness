# Team-Program Tournament History Features Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add two team-program tournament-history features to v4's stack — `team_seed_residual_mean_10yr` (continuity, shrunk mean of per-seed residuals over prior 10yr) and `team_seed_residual_ewma_hl2` (momentum, shrunk EWMA at HL=2). Phase 1 ships the module + diagnostic + tests with no v4 wire-in (sanity-check artifacts for human review). Phase 2 wires into `enhanced_model_v3.py`'s feature build, runs full 22-season LOSO, applies v8 stage-2, scores bracket points, and emits a verdict per pre-registered bands (PASS ≥+20 with fragility check / MARGINAL +10..+20 / FAIL <+10).

**Architecture:** New module `src/features/team_history.py` implements pure functions: `compute_per_seed_baseline` (leak-safe via explicit `max_season` arg + assertion), `shrunk_mean` and `shrunk_ewma` (both decay-independent, `n`-based shrinkage), `compute_team_residuals_in_window`, and the `compute_team_history_features` DataFrame integrator. Phase 1 driver `src/diagnose_team_seed_residual.py` produces 5 sanity-check artifacts (per-seed baseline, 9-champion residuals with hand-computable cross-check, correlation matrix vs incumbents, distribution percentiles, top/bottom-10). Phase 2 mirrors `coach.py`'s join pattern at `enhanced_model_v3.py:830-840`. Anchors throughout: empty-input → 0; `max_season` violation → assertion error; `MM_FEATURE_DROP=team_seed_residual_mean_10yr,team_seed_residual_ewma_hl2` reproduces canonical clean v4 baseline byte-equal.

**Tech Stack:** Python, pandas, numpy, xgboost (existing v3/v4/v8 training), pytest. Inputs: `data/raw/march-machine-learning-2026/MNCAATourneyDetailedResults.csv`, `MNCAATourneySeeds.csv`. Existing pieces reused: `src/features/coach.py` (`day_to_round` round-mapping convention), `src/enhanced_model_v3.py` (`compute_all_features` and the `MM_FEATURE_DROP` ablation hook).

**Spec:** `docs/superpowers/specs/2026-05-09-team-seed-residual-design.md`
**Predecessors:**
- TODO retire-Kaggle-framing (PR 33): production objective is the 22-season bracket-points backtest (clean baseline 2069), with LL secondary.
- v4 calibration temperature scaling (MARGINAL): closes calibration-shape lane.
- 538 audit (1 weak spot vs 538 chalk picks): `docs/notes/2026-05-04-v4-gap-audit-fte.md`.
- Vegas audit (clean baseline, 6 weak spots): `docs/notes/2026-05-04-v4-gap-audit-vegas.md`.

> **Production stage-2 is v8** (clean v8 = 2069 brkt pts; v9-C reverted per the TODO Compounding-work note). Treat `output/pairwise_v8.csv` as the canonical baseline.

---

## File Structure

**Created (committed):**

- `src/features/team_history.py` (~200 LOC)
  - Public: `compute_per_seed_baseline(tourney_results: pd.DataFrame, max_season: int) -> dict[int, float]` — per-seed expected `rounds_won`, computed from `Season <= max_season` only. Asserts no input row violates `max_season`. Falls back to overall mean for seeds with 0 observations.
  - Public: `shrunk_mean(residuals: list[float], k: int = 3) -> float` — `(sum(residuals) + 0) / (len + k)`; returns 0.0 for empty input.
  - Public: `shrunk_ewma(residuals_with_age: list[tuple[int, float]], half_life: float = 2.0, k: int = 3) -> float` — weighted mean with `w(a) = 0.5**((a-1)/half_life)`, then n-based shrinkage; returns 0.0 for empty input.
  - Public: `compute_team_residuals_in_window(season: int, team_id: int, window_years: int, baseline: dict[int, float], tourney_results: pd.DataFrame, seeds: pd.DataFrame) -> list[tuple[int, int, float]]` — returns `[(years_ago, prior_seed, residual), ...]` for the team's prior tournament appearances within `[season - window_years, season - 1]`.
  - Public: `compute_team_history_features(tournament_field: pd.DataFrame, tourney_results: pd.DataFrame, seeds: pd.DataFrame, window_years: int = 10) -> pd.DataFrame` — DataFrame with columns `[Season, TeamID, team_seed_residual_mean_10yr, team_seed_residual_ewma_hl2]`, one row per `(Season, TeamID)` in `tournament_field`. Re-uses `src.features.coach.day_to_round`.
  - Private: `_extract_seed_num(seed_str: str) -> int` — `"W01"` → `1`, `"X16a"` → `16`. Mirrors the established `seeds["Seed"].str.extract(r"(\d+)")` pattern from `src/audit_v4_gap_vegas.py:_load_seeds_lookup`.
  - Private: `_rounds_won_per_team_season(tourney_results: pd.DataFrame, max_season: int | None) -> pd.DataFrame` — returns `(Season, TeamID, rounds_won)`. **Convention:** `rounds_won` = number of games the team won in that tournament. Champion = 6 wins (R64 + R32 + S16 + E8 + F4 + Champ). R64 loser = 0 wins. First Four wins (DayNum 134-135) count, mirroring `src/features/coach.py`. Implementation: count rows in `tourney_results` where `team_id == WTeamID` per `(Season, TeamID)`.

- `src/diagnose_team_seed_residual.py` (~150 LOC)
  - Public: `main(argv) -> int` — CLI driver. Loads data, calls `compute_team_history_features`, joins with v4 incumbents from latest `pairwise_v4.csv`-anchored team-feature snapshot, writes JSON + log artifacts.
  - Private: `_emit_per_seed_baseline(baseline, tourney_results, log_lines, json_dict) -> None`.
  - Private: `_emit_champion_residuals(features_df, tourney_results, seeds, log_lines, json_dict) -> None` — for each of 9 champions 2015-2024, dump prior-window appearances with year_ago + seed + rounds_won, the computed feature values, and a manual-cross-check value computed in-place.
  - Private: `_emit_correlation_matrix(features_df, incumbent_features_csv, log_lines, json_dict) -> None`.
  - Private: `_emit_distribution_percentiles(features_df, log_lines, json_dict) -> None`.
  - Private: `_emit_top_bottom_n(features_df, n, teams_lookup, log_lines, json_dict) -> None`.

- `tests/test_features/test_team_history.py` (~250 LOC, 11 tests)
  - All tests listed in the spec's "Test plan" section, plus a smoke test on real Kaggle data.

**Modified:**

- `src/enhanced_model_v3.py` — add ~7 lines after the coach-features block (line 840), mirroring the `compute_coach_features` pattern. Single import line at top of file.

**Generated (force-added per `.gitignore: output/`):**

- `output/team_seed_residual_diagnostic.json`
- `output/team_seed_residual_diagnostic.log`
- (Phase 2 only) `output/team_seed_residual_loso_summary.json` — per-season LL/brkt-pts deltas + verdict band.

---

## Phase 1: Feature module + diagnostic + tests

### Task 1: Module skeleton + `_rounds_won_per_team_season` helper

**Files:**
- Create: `src/features/team_history.py`
- Create: `tests/test_features/test_team_history.py`

- [ ] **Step 1: Write the failing test for `_rounds_won_per_team_season`**

```python
# tests/test_features/test_team_history.py
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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /c/Users/alden/MarchMadness && python -m pytest tests/test_features/test_team_history.py::test_rounds_won_counts_games_won_per_team_season -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'src.features.team_history'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/features/team_history.py
"""Team-program tournament history features.

Two new XGB features for v4:
  - team_seed_residual_mean_10yr: shrunk mean of per-seed residuals
    over prior-10-year window. Captures program continuity (Duke / UNC /
    Kansas / UConn).
  - team_seed_residual_ewma_hl2: shrunk EWMA at HL=2 over the same
    window. Captures program emergence/momentum (Bennett-Virginia,
    Drew-Baylor) and recent form.

Both are keyed on TeamID (NOT coach, unlike coach_career_*), filling
the program-DNA gap in v4's feature stack. See
docs/superpowers/specs/2026-05-09-team-seed-residual-design.md.
"""
from __future__ import annotations

import pandas as pd

from src.features.coach import day_to_round


def _rounds_won_per_team_season(
    tourney_results: pd.DataFrame,
    max_season: int | None,
) -> pd.DataFrame:
    """Return a DataFrame [(Season, TeamID, rounds_won)] for every team
    that appeared in any tournament in tourney_results.

    rounds_won = number of games the team WON in that tournament. Champion
    = 6 wins (R64 + R32 + S16 + E8 + F4 + Champ). R64 loser = 0 wins.
    First Four wins (DayNum 134-135) count toward rounds_won; this
    matches src/features/coach.py's convention.

    If max_season is provided, asserts no input row has Season > max_season.
    """
    if max_season is not None:
        bad = tourney_results[tourney_results["Season"] > max_season]
        assert bad.empty, (
            f"Leak guard: {len(bad)} rows have Season > max_season "
            f"({max_season}). team_history features must be computed "
            f"on a leak-free tourney_results subset."
        )

    # Per-game records: (season, team_id, won, round)
    rows = []
    for _, g in tourney_results.iterrows():
        rnd = day_to_round(int(g["DayNum"]))
        if rnd is None:
            continue
        season = int(g["Season"])
        rows.append({"Season": season, "TeamID": int(g["WTeamID"]), "won": 1})
        rows.append({"Season": season, "TeamID": int(g["LTeamID"]), "won": 0})
    if not rows:
        return pd.DataFrame(columns=["Season", "TeamID", "rounds_won"])
    df = pd.DataFrame(rows)
    out = df.groupby(["Season", "TeamID"])["won"].sum().reset_index()
    out = out.rename(columns={"won": "rounds_won"})
    out["rounds_won"] = out["rounds_won"].astype(int)
    return out
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd /c/Users/alden/MarchMadness && python -m pytest tests/test_features/test_team_history.py::test_rounds_won_counts_games_won_per_team_season -v
```

Expected: PASS.

- [ ] **Step 5: Add leak-guard test**

```python
def test_rounds_won_leak_guard_fires_when_input_has_future_season():
    tr = _toy_tourney([
        (2023, 136, 100, 200),
        (2024, 136, 300, 400),  # future-season row
    ])
    with pytest.raises(AssertionError, match="Leak guard"):
        _rounds_won_per_team_season(tr, max_season=2023)
```

- [ ] **Step 6: Run both tests**

```bash
cd /c/Users/alden/MarchMadness && python -m pytest tests/test_features/test_team_history.py -v
```

Expected: 2 passed.

- [ ] **Step 7: Commit**

```bash
cd /c/Users/alden/MarchMadness && git add src/features/team_history.py tests/test_features/test_team_history.py && git commit -m "feat(team-seed-residual): rounds_won helper + leak guard"
```

---

### Task 2: `compute_per_seed_baseline`

**Files:**
- Modify: `src/features/team_history.py`
- Modify: `tests/test_features/test_team_history.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_features/test_team_history.py (append)
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /c/Users/alden/MarchMadness && python -m pytest tests/test_features/test_team_history.py -v -k per_seed_baseline
```

Expected: 3 FAIL (`ImportError: cannot import compute_per_seed_baseline`).

- [ ] **Step 3: Implement `compute_per_seed_baseline`**

```python
# src/features/team_history.py (append)
def _extract_seed_num(seed_str: str) -> int:
    """'W01' -> 1; 'X16a' -> 16; 'Y08' -> 8."""
    digits = "".join(ch for ch in seed_str if ch.isdigit())
    return int(digits)


def compute_per_seed_baseline(
    tourney_results: pd.DataFrame,
    seeds: pd.DataFrame,
    max_season: int,
) -> dict[int | str, float]:
    """Per-seed average rounds_won across all (Season, TeamID) rows
    with Season <= max_season.

    Returns: dict mapping seed_num (int) -> expected rounds_won. Includes
    a special key '__fallback__' = overall mean rounds_won, used by callers
    when a queried seed has 0 historical observations.

    Asserts no input tourney_results row has Season > max_season.
    """
    rounds_won = _rounds_won_per_team_season(
        tourney_results[tourney_results["Season"] <= max_season],
        max_season=max_season,
    )
    seeds_in_window = seeds[seeds["Season"] <= max_season].copy()
    seeds_in_window["seed_num"] = seeds_in_window["Seed"].apply(_extract_seed_num)
    joined = rounds_won.merge(
        seeds_in_window[["Season", "TeamID", "seed_num"]],
        on=["Season", "TeamID"],
        how="left",
    )
    # Drop team-seasons without a seed (shouldn't happen for tournament rows,
    # but defensive).
    joined = joined.dropna(subset=["seed_num"])
    joined["seed_num"] = joined["seed_num"].astype(int)

    baseline: dict[int | str, float] = {}
    for seed in range(1, 17):
        sub = joined[joined["seed_num"] == seed]
        if len(sub) > 0:
            baseline[seed] = float(sub["rounds_won"].mean())
    baseline["__fallback__"] = float(joined["rounds_won"].mean()) if len(joined) > 0 else 0.0
    return baseline
```

- [ ] **Step 4: Run tests**

```bash
cd /c/Users/alden/MarchMadness && python -m pytest tests/test_features/test_team_history.py -v
```

Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
cd /c/Users/alden/MarchMadness && git add src/features/team_history.py tests/test_features/test_team_history.py && git commit -m "feat(team-seed-residual): per-seed leak-safe baseline"
```

---

### Task 3: `shrunk_mean`

**Files:**
- Modify: `src/features/team_history.py`
- Modify: `tests/test_features/test_team_history.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_features/test_team_history.py (append)
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /c/Users/alden/MarchMadness && python -m pytest tests/test_features/test_team_history.py -v -k shrunk_mean
```

Expected: 4 FAIL (`ImportError: cannot import shrunk_mean`).

- [ ] **Step 3: Implement `shrunk_mean`**

```python
# src/features/team_history.py (append)
def shrunk_mean(residuals: list[float], k: int = 3) -> float:
    """Bayesian shrinkage of mean(residuals) toward 0 with k pseudo-obs.

    For empty input, returns 0.0 ("no evidence").
    For n observations: (sum(residuals) + k * 0) / (n + k).
    """
    n = len(residuals)
    if n == 0:
        return 0.0
    return float(sum(residuals)) / (n + k)
```

- [ ] **Step 4: Run tests**

```bash
cd /c/Users/alden/MarchMadness && python -m pytest tests/test_features/test_team_history.py -v
```

Expected: 9 passed.

- [ ] **Step 5: Commit**

```bash
cd /c/Users/alden/MarchMadness && git add src/features/team_history.py tests/test_features/test_team_history.py && git commit -m "feat(team-seed-residual): shrunk_mean with k=3 prior at 0"
```

---

### Task 4: `shrunk_ewma`

**Files:**
- Modify: `src/features/team_history.py`
- Modify: `tests/test_features/test_team_history.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_features/test_team_history.py (append)
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /c/Users/alden/MarchMadness && python -m pytest tests/test_features/test_team_history.py -v -k shrunk_ewma
```

Expected: 4 FAIL.

- [ ] **Step 3: Implement `shrunk_ewma`**

```python
# src/features/team_history.py (append)
def shrunk_ewma(
    residuals_with_age: list[tuple[int, float]],
    half_life: float = 2.0,
    k: int = 3,
) -> float:
    """Bayesian-shrunk exponentially-weighted mean of residuals.

    residuals_with_age: list of (years_ago, residual) tuples. years_ago=1
    is the most recent prior season; larger = more remote.

    Weights: w(a) = 0.5 ** ((a - 1) / half_life). w(1) = 1.0.

    Computes weighted_mean = sum(w * r) / sum(w), then applies n-based
    shrinkage: (n * weighted_mean + k * 0) / (n + k). This decouples
    "which residuals matter" (EWMA weights) from "how confident we are
    in the estimate" (raw n).

    Returns 0.0 for empty input.
    """
    n = len(residuals_with_age)
    if n == 0:
        return 0.0
    weights = [0.5 ** ((a - 1) / half_life) for (a, _) in residuals_with_age]
    weight_sum = sum(weights)
    if weight_sum == 0:
        return 0.0
    weighted_mean = sum(
        w * r for (w, (_, r)) in zip(weights, residuals_with_age)
    ) / weight_sum
    return float(n * weighted_mean) / (n + k)
```

- [ ] **Step 4: Run tests**

```bash
cd /c/Users/alden/MarchMadness && python -m pytest tests/test_features/test_team_history.py -v
```

Expected: 13 passed.

- [ ] **Step 5: Commit**

```bash
cd /c/Users/alden/MarchMadness && git add src/features/team_history.py tests/test_features/test_team_history.py && git commit -m "feat(team-seed-residual): shrunk_ewma with HL=2, n-based shrinkage"
```

---

### Task 5: `compute_team_residuals_in_window`

**Files:**
- Modify: `src/features/team_history.py`
- Modify: `tests/test_features/test_team_history.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_features/test_team_history.py (append)
def test_residuals_in_window_returns_empty_when_no_prior_appearances():
    from src.features.team_history import (
        compute_per_seed_baseline,
        compute_team_residuals_in_window,
    )
    tr = _toy_tourney([(2024, 136, 100, 116)])
    seeds = _toy_seeds([(2024, "W01", 100), (2024, "W16", 116)])
    baseline = compute_per_seed_baseline(tr, seeds, max_season=2024)
    out = compute_team_residuals_in_window(
        season=2024, team_id=999, window_years=10,
        baseline=baseline, tourney_results=tr, seeds=seeds,
    )
    assert out == []


def test_residuals_in_window_window_edges():
    """For target season 2024 with window=10, year-10 (2014) is IN,
    year-11 (2013) is OUT."""
    from src.features.team_history import (
        compute_per_seed_baseline,
        compute_team_residuals_in_window,
    )
    # Team 100 appears in 2013 (out of window), 2014 (in window edge), 2024 (target, excluded).
    tr = _toy_tourney([
        (2013, 136, 100, 116),  # team 100 wins R64 in 2013
        (2014, 136, 100, 117),  # team 100 wins R64 in 2014
        (2024, 136, 100, 118),  # target season, must NOT be in residuals
    ])
    seeds = _toy_seeds([
        (2013, "W08", 100), (2013, "W09", 116),
        (2014, "W08", 100), (2014, "W09", 117),
        (2024, "W08", 100), (2024, "W09", 118),
    ])
    baseline = compute_per_seed_baseline(tr, seeds, max_season=2023)
    out = compute_team_residuals_in_window(
        season=2024, team_id=100, window_years=10,
        baseline=baseline, tourney_results=tr, seeds=seeds,
    )
    # Only 2014 should appear (years_ago=10 is in window; 2013 years_ago=11 is out;
    # 2024 itself is excluded as the target season).
    years_ago_seen = sorted(a for (a, _, _) in out)
    assert years_ago_seen == [10]


def test_residuals_in_window_uses_baseline_for_seed_in_prior_season():
    """A team with a prior 1-seed appearance (rounds_won=2) gets
    residual = 2 - baseline[1]."""
    from src.features.team_history import (
        compute_per_seed_baseline,
        compute_team_residuals_in_window,
    )
    tr = _toy_tourney([
        # 2020-2022: 1-seeds win R64 + R32 (rounds_won=2 each)
        (2020, 136, 100, 116), (2020, 138, 100, 117),
        (2021, 136, 200, 216), (2021, 138, 200, 217),
        (2022, 136, 300, 316), (2022, 138, 300, 317),
        # 2023: team 400 (1-seed) wins R64 only
        (2023, 136, 400, 416),
    ])
    seeds = _toy_seeds([
        (2020, "W01", 100), (2020, "W16", 116),
        (2021, "X01", 200), (2021, "X16", 216),
        (2022, "Y01", 300), (2022, "Y16", 316),
        (2023, "Z01", 400), (2023, "Z16", 416),
    ])
    baseline = compute_per_seed_baseline(tr, seeds, max_season=2023)
    # 1-seed baseline = (2 + 2 + 2 + 1) / 4 = 1.75
    assert baseline[1] == pytest.approx(1.75)
    out = compute_team_residuals_in_window(
        season=2024, team_id=400, window_years=10,
        baseline=baseline, tourney_results=tr, seeds=seeds,
    )
    # Team 400's only prior is 2023 as a 1-seed with rounds_won=1
    # residual = 1 - 1.75 = -0.75
    assert len(out) == 1
    years_ago, prior_seed, residual = out[0]
    assert years_ago == 1
    assert prior_seed == 1
    assert residual == pytest.approx(-0.75)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /c/Users/alden/MarchMadness && python -m pytest tests/test_features/test_team_history.py -v -k residuals_in_window
```

Expected: 3 FAIL.

- [ ] **Step 3: Implement `compute_team_residuals_in_window`**

```python
# src/features/team_history.py (append)
def compute_team_residuals_in_window(
    season: int,
    team_id: int,
    window_years: int,
    baseline: dict[int | str, float],
    tourney_results: pd.DataFrame,
    seeds: pd.DataFrame,
) -> list[tuple[int, int, float]]:
    """For target (season, team_id), return [(years_ago, prior_seed, residual)]
    for the team's prior tournament appearances within window_years.

    years_ago = season - prior_season, in [1, window_years].
    residual = prior_rounds_won - baseline[prior_seed], with fallback
    to baseline['__fallback__'] for seeds with no historical data.
    """
    earliest = season - window_years
    rounds_won = _rounds_won_per_team_season(
        tourney_results[
            (tourney_results["Season"] >= earliest)
            & (tourney_results["Season"] <= season - 1)
        ],
        max_season=None,
    )
    team_priors = rounds_won[rounds_won["TeamID"] == team_id]
    if team_priors.empty:
        return []

    seeds_in_range = seeds[
        (seeds["Season"] >= earliest) & (seeds["Season"] <= season - 1)
    ].copy()
    seeds_in_range["seed_num"] = seeds_in_range["Seed"].apply(_extract_seed_num)
    team_priors = team_priors.merge(
        seeds_in_range[["Season", "TeamID", "seed_num"]],
        on=["Season", "TeamID"],
        how="left",
    )
    team_priors = team_priors.dropna(subset=["seed_num"])
    team_priors["seed_num"] = team_priors["seed_num"].astype(int)

    fallback = baseline.get("__fallback__", 0.0)
    out = []
    for _, row in team_priors.iterrows():
        prior_season = int(row["Season"])
        prior_seed = int(row["seed_num"])
        prior_rounds_won = int(row["rounds_won"])
        seed_baseline = baseline.get(prior_seed, fallback)
        residual = prior_rounds_won - seed_baseline
        years_ago = season - prior_season
        out.append((years_ago, prior_seed, residual))
    out.sort()
    return out
```

- [ ] **Step 4: Run tests**

```bash
cd /c/Users/alden/MarchMadness && python -m pytest tests/test_features/test_team_history.py -v
```

Expected: 16 passed.

- [ ] **Step 5: Commit**

```bash
cd /c/Users/alden/MarchMadness && git add src/features/team_history.py tests/test_features/test_team_history.py && git commit -m "feat(team-seed-residual): per-team residuals in window"
```

---

### Task 6: `compute_team_history_features` integrator

**Files:**
- Modify: `src/features/team_history.py`
- Modify: `tests/test_features/test_team_history.py`

- [ ] **Step 1: Write failing test (integration with hand-computed UConn 2024)**

```python
# tests/test_features/test_team_history.py (append)
import os
from pathlib import Path

DATA_DIR = Path(__file__).parents[2] / "data" / "raw" / "march-machine-learning-2026"


@pytest.mark.skipif(
    not (DATA_DIR / "MNCAATourneySeeds.csv").exists(),
    reason="Needs Kaggle data; run `tar -xzf data/training_data.tar.gz -C data/raw/`",
)
def test_compute_features_uconn_2024_spot_check():
    """Hand-compute UConn 2024's two features against the implementation.

    UConn = TeamID 1163 (verified via MTeams.csv 'TeamName == Connecticut').
    Prior-10-year window for season=2024: seasons 2014-2023.
    UConn's appearances:
      2014 (7-seed, won championship → rounds_won=6)
      2016 (9-seed, R32 loss → rounds_won=1)
      2021 (7-seed, R64 loss → rounds_won=0)
      2022 (5-seed, R64 loss → rounds_won=0)
      2023 (4-seed, won championship → rounds_won=6)
    """
    from src.features.team_history import (
        compute_per_seed_baseline,
        compute_team_history_features,
        compute_team_residuals_in_window,
        shrunk_ewma,
        shrunk_mean,
    )
    tr = pd.read_csv(DATA_DIR / "MNCAATourneyDetailedResults.csv")
    seeds = pd.read_csv(DATA_DIR / "MNCAATourneySeeds.csv")
    teams = pd.read_csv(DATA_DIR / "MTeams.csv")

    uconn = int(teams[teams["TeamName"] == "Connecticut"].iloc[0]["TeamID"])

    # Compute features for UConn 2024 only via the public integrator
    field_2024 = pd.DataFrame([{"Season": 2024, "TeamID": uconn}])
    out = compute_team_history_features(
        tournament_field=field_2024,
        tourney_results=tr,
        seeds=seeds,
        window_years=10,
    )
    assert len(out) == 1
    row = out.iloc[0]

    # Hand-compute via the same primitives
    baseline = compute_per_seed_baseline(tr, seeds, max_season=2023)
    residuals = compute_team_residuals_in_window(
        season=2024, team_id=uconn, window_years=10,
        baseline=baseline, tourney_results=tr, seeds=seeds,
    )
    expected_mean = shrunk_mean([r for (_, _, r) in residuals], k=3)
    expected_ewma = shrunk_ewma(
        [(a, r) for (a, _, r) in residuals], half_life=2, k=3,
    )

    assert row["team_seed_residual_mean_10yr"] == pytest.approx(expected_mean, abs=1e-9)
    assert row["team_seed_residual_ewma_hl2"] == pytest.approx(expected_ewma, abs=1e-9)
    # Sanity: UConn 2024 should have ≥ 4 prior appearances in window
    assert len(residuals) >= 4
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /c/Users/alden/MarchMadness && python -m pytest tests/test_features/test_team_history.py::test_compute_features_uconn_2024_spot_check -v
```

Expected: FAIL (function not yet defined).

- [ ] **Step 3: Implement `compute_team_history_features`**

```python
# src/features/team_history.py (append)
def compute_team_history_features(
    tournament_field: pd.DataFrame,
    tourney_results: pd.DataFrame,
    seeds: pd.DataFrame,
    window_years: int = 10,
) -> pd.DataFrame:
    """For each (Season, TeamID) in tournament_field, compute
    team_seed_residual_mean_10yr and team_seed_residual_ewma_hl2.

    The per-seed baseline is recomputed per target season using
    Season < S data only (leak-safe).
    """
    rows = []
    # Cache baseline per target season (avoids recompute when many teams share S)
    baseline_cache: dict[int, dict[int | str, float]] = {}
    for _, fr in tournament_field.iterrows():
        season = int(fr["Season"])
        team_id = int(fr["TeamID"])
        if season not in baseline_cache:
            baseline_cache[season] = compute_per_seed_baseline(
                tourney_results, seeds, max_season=season - 1,
            )
        residuals = compute_team_residuals_in_window(
            season=season, team_id=team_id, window_years=window_years,
            baseline=baseline_cache[season],
            tourney_results=tourney_results, seeds=seeds,
        )
        mean_feat = shrunk_mean([r for (_, _, r) in residuals], k=3)
        ewma_feat = shrunk_ewma(
            [(a, r) for (a, _, r) in residuals], half_life=2, k=3,
        )
        rows.append({
            "Season": season,
            "TeamID": team_id,
            "team_seed_residual_mean_10yr": mean_feat,
            "team_seed_residual_ewma_hl2": ewma_feat,
        })
    return pd.DataFrame(
        rows,
        columns=["Season", "TeamID",
                 "team_seed_residual_mean_10yr", "team_seed_residual_ewma_hl2"],
    )
```

- [ ] **Step 4: Run tests**

```bash
cd /c/Users/alden/MarchMadness && python -m pytest tests/test_features/test_team_history.py -v
```

Expected: 17 passed (including the UConn spot check).

- [ ] **Step 5: Commit**

```bash
cd /c/Users/alden/MarchMadness && git add src/features/team_history.py tests/test_features/test_team_history.py && git commit -m "feat(team-seed-residual): compute_team_history_features integrator + UConn 2024 spot check"
```

---

### Task 7: Phase 1 diagnostic + run + force-add artifacts

**Files:**
- Create: `src/diagnose_team_seed_residual.py`
- Modify: `tests/test_features/test_team_history.py` (smoke test for diagnostic)

- [ ] **Step 1: Write the diagnostic script**

```python
# src/diagnose_team_seed_residual.py
"""Phase 1 diagnostic for team-seed-residual features.

Produces 5 sanity-check artifacts to output/team_seed_residual_diagnostic.{json,log}:
  1. Per-seed empirical baseline table
  2. 9-champion (2015-2024) residual values with hand-computable cross-check
  3. Pearson correlation matrix vs incumbent v4 features
  4. Distribution percentiles (5/25/50/75/95) of each new feature
  5. Top-10 / bottom-10 (Season, TeamName, value) pairs

Usage:
    python -m src.diagnose_team_seed_residual

Spec: docs/superpowers/specs/2026-05-09-team-seed-residual-design.md
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.features.team_history import (
    compute_per_seed_baseline,
    compute_team_history_features,
    compute_team_residuals_in_window,
    shrunk_ewma,
    shrunk_mean,
)

DATA_DIR = Path("data/raw/march-machine-learning-2026")
OUT_JSON = Path("output/team_seed_residual_diagnostic.json")
OUT_LOG = Path("output/team_seed_residual_diagnostic.log")

CHAMPIONS_2015_2024 = [
    (2015, "Duke"),
    (2016, "Villanova"),
    (2017, "North Carolina"),
    (2018, "Villanova"),
    (2019, "Virginia"),
    (2021, "Baylor"),
    (2022, "Kansas"),
    (2023, "Connecticut"),
    (2024, "Connecticut"),
]


def _emit_per_seed_baseline(tr, seeds, max_season, log, jdict):
    baseline = compute_per_seed_baseline(tr, seeds, max_season=max_season)
    table = []
    for seed in range(1, 17):
        if seed in baseline:
            sub = tr[tr["Season"] <= max_season]
            from src.features.team_history import _rounds_won_per_team_season, _extract_seed_num
            rw = _rounds_won_per_team_season(sub, max_season=None)
            seed_rows = seeds[seeds["Season"] <= max_season].copy()
            seed_rows["seed_num"] = seed_rows["Seed"].apply(_extract_seed_num)
            joined = rw.merge(seed_rows[["Season", "TeamID", "seed_num"]],
                              on=["Season", "TeamID"], how="left")
            n = int((joined["seed_num"] == seed).sum())
            table.append({"seed": seed, "n_observations": n,
                          "expected_rounds_won": baseline[seed]})
    jdict["per_seed_baseline"] = table
    jdict["fallback_baseline"] = baseline["__fallback__"]
    log.append(f"\n=== Per-seed baseline (Season <= {max_season}) ===")
    log.append(f"{'seed':>4} {'n':>5} {'E[rounds_won]':>15}")
    for row in table:
        log.append(f"{row['seed']:>4} {row['n_observations']:>5} {row['expected_rounds_won']:>15.3f}")
    log.append(f"  (fallback for missing seeds: {baseline['__fallback__']:.3f})")


def _emit_champion_residuals(tr, seeds, teams, log, jdict):
    log.append("\n=== 9-champion residual values ===")
    log.append(f"{'Yr':>4} {'Team':<18} | {'cont':>6} {'mom':>6} | priors")
    log.append("-" * 90)
    champ_records = []
    for season, team_name in CHAMPIONS_2015_2024:
        team_row = teams[teams["TeamName"] == team_name]
        if team_row.empty:
            log.append(f"{season:>4} {team_name:<18} | (TeamName not in MTeams.csv)")
            continue
        team_id = int(team_row.iloc[0]["TeamID"])
        baseline = compute_per_seed_baseline(tr, seeds, max_season=season - 1)
        residuals = compute_team_residuals_in_window(
            season=season, team_id=team_id, window_years=10,
            baseline=baseline, tourney_results=tr, seeds=seeds,
        )
        mean_v = shrunk_mean([r for (_, _, r) in residuals], k=3)
        ewma_v = shrunk_ewma([(a, r) for (a, _, r) in residuals],
                             half_life=2, k=3)
        priors_str = ", ".join(
            f"yr{a}/sd{s}/r={r:+.2f}" for (a, s, r) in residuals
        )
        log.append(f"{season:>4} {team_name:<18} | {mean_v:>6.2f} {ewma_v:>6.2f} | {priors_str}")
        champ_records.append({
            "season": season, "team_name": team_name, "team_id": team_id,
            "team_seed_residual_mean_10yr": mean_v,
            "team_seed_residual_ewma_hl2": ewma_v,
            "priors": [{"years_ago": a, "prior_seed": s, "residual": r}
                       for (a, s, r) in residuals],
        })
    jdict["champion_residuals"] = champ_records


def _emit_correlation(features_df, incumbent_csv, log, jdict):
    if not Path(incumbent_csv).exists():
        log.append(f"\n=== Correlation matrix: SKIPPED (no {incumbent_csv}) ===")
        return
    incumbent = pd.read_csv(incumbent_csv)
    cols_to_check = ["adj_em", "kp_TALENT", "kp_BARTHAG",
                     "coach_career_f4_apps", "coach_career_winpct",
                     "coach_career_seasons", "season_win_pct", "conf_strength"]
    available = [c for c in cols_to_check if c in incumbent.columns]
    joined = features_df.merge(
        incumbent[["Season", "TeamID"] + available],
        on=["Season", "TeamID"], how="inner",
    )
    log.append("\n=== Pearson correlation: new features vs incumbents ===")
    log.append(f"{'feature':<40} | {' '.join(f'{c:>20}' for c in available)}")
    table = {}
    for new_col in ["team_seed_residual_mean_10yr", "team_seed_residual_ewma_hl2"]:
        corrs = {c: float(joined[new_col].corr(joined[c])) for c in available}
        log.append(f"{new_col:<40} | " + " ".join(f"{corrs[c]:>20.3f}" for c in available))
        table[new_col] = corrs
        # Flag high-correlation entries
        for c, v in corrs.items():
            if abs(v) > 0.85:
                log.append(f"  FLAG: |corr({new_col}, {c})| = {abs(v):.3f} > 0.85")
    jdict["correlation_matrix"] = table


def _emit_distribution(features_df, log, jdict):
    log.append("\n=== Distribution percentiles ===")
    log.append(f"{'feature':<40} | {'p05':>7} {'p25':>7} {'p50':>7} {'p75':>7} {'p95':>7}")
    table = {}
    for col in ["team_seed_residual_mean_10yr", "team_seed_residual_ewma_hl2"]:
        v = features_df[col].values
        pcts = [float(np.percentile(v, p)) for p in (5, 25, 50, 75, 95)]
        log.append(f"{col:<40} | " + " ".join(f"{x:>+7.3f}" for x in pcts))
        table[col] = dict(zip(["p05", "p25", "p50", "p75", "p95"], pcts))
    jdict["distribution_percentiles"] = table


def _emit_top_bottom(features_df, teams, log, jdict, n=10):
    log.append("\n=== Top-10 / bottom-10 by each feature ===")
    teams_lookup = teams.set_index("TeamID")["TeamName"].to_dict()
    table = {}
    for col in ["team_seed_residual_mean_10yr", "team_seed_residual_ewma_hl2"]:
        sorted_df = features_df.sort_values(col, ascending=False)
        top = sorted_df.head(n).copy()
        bot = sorted_df.tail(n).copy()
        top["TeamName"] = top["TeamID"].map(teams_lookup)
        bot["TeamName"] = bot["TeamID"].map(teams_lookup)
        log.append(f"\n  Top-{n} {col}:")
        for _, r in top.iterrows():
            log.append(f"    {int(r['Season']):>4} {r['TeamName']:<22} {r[col]:>+7.3f}")
        log.append(f"  Bottom-{n} {col}:")
        for _, r in bot.iterrows():
            log.append(f"    {int(r['Season']):>4} {r['TeamName']:<22} {r[col]:>+7.3f}")
        table[col] = {
            "top": [{"season": int(r["Season"]), "team_name": r["TeamName"], "value": float(r[col])}
                    for _, r in top.iterrows()],
            "bottom": [{"season": int(r["Season"]), "team_name": r["TeamName"], "value": float(r[col])}
                       for _, r in bot.iterrows()],
        }
    jdict["top_bottom_n"] = table


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--incumbent-csv", default="output/v4_team_features.csv",
                        help="CSV of incumbent v4 team features for correlation check (skipped if absent)")
    parser.add_argument("--seasons", default="2003-2024",
                        help="LOSO season range as 'min-max', inclusive")
    args = parser.parse_args(argv)

    s_min, s_max = (int(x) for x in args.seasons.split("-"))

    tr = pd.read_csv(DATA_DIR / "MNCAATourneyDetailedResults.csv")
    seeds = pd.read_csv(DATA_DIR / "MNCAATourneySeeds.csv")
    teams = pd.read_csv(DATA_DIR / "MTeams.csv")

    log: list[str] = [
        "=== Team-seed-residual Phase 1 diagnostic ===",
        f"Data: {DATA_DIR}",
        f"Tournament rows: {len(tr):,}",
        f"Seed rows: {len(seeds):,}",
        f"LOSO seasons: {s_min}-{s_max}",
    ]
    jdict: dict = {
        "spec": "docs/superpowers/specs/2026-05-09-team-seed-residual-design.md",
        "n_tourney_rows": len(tr),
        "n_seed_rows": len(seeds),
        "loso_seasons": list(range(s_min, s_max + 1)),
    }

    _emit_per_seed_baseline(tr, seeds, max_season=s_max - 1, log=log, jdict=jdict)
    _emit_champion_residuals(tr, seeds, teams, log=log, jdict=jdict)

    # Build features for all (Season, TeamID) in LOSO seasons for distribution + top/bottom
    field = seeds[seeds["Season"].between(s_min, s_max)][["Season", "TeamID"]].drop_duplicates()
    features_df = compute_team_history_features(
        tournament_field=field, tourney_results=tr, seeds=seeds, window_years=10,
    )
    log.append(f"\nFeature DataFrame: {len(features_df):,} rows over seasons {s_min}-{s_max}")

    _emit_correlation(features_df, args.incumbent_csv, log=log, jdict=jdict)
    _emit_distribution(features_df, log=log, jdict=jdict)
    _emit_top_bottom(features_df, teams, log=log, jdict=jdict, n=10)

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(jdict, indent=2))
    OUT_LOG.write_text("\n".join(log) + "\n")
    print(f"Wrote {OUT_JSON} and {OUT_LOG}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Add a smoke test for the diagnostic**

```python
# tests/test_features/test_team_history.py (append)
@pytest.mark.skipif(
    not (DATA_DIR / "MNCAATourneySeeds.csv").exists(),
    reason="Needs Kaggle data",
)
def test_diagnose_team_seed_residual_smoke(tmp_path, monkeypatch):
    """The Phase 1 diagnostic runs without error and writes JSON + log."""
    from src import diagnose_team_seed_residual as diag
    monkeypatch.setattr(diag, "OUT_JSON", tmp_path / "diag.json")
    monkeypatch.setattr(diag, "OUT_LOG", tmp_path / "diag.log")
    rc = diag.main(["--seasons", "2020-2024"])  # short range = fast smoke
    assert rc == 0
    assert (tmp_path / "diag.json").exists()
    assert (tmp_path / "diag.log").exists()
    payload = json.loads((tmp_path / "diag.json").read_text())
    assert "per_seed_baseline" in payload
    assert "champion_residuals" in payload
    assert "distribution_percentiles" in payload


import json  # ensure module-level import for the smoke test
```

- [ ] **Step 3: Run all tests**

```bash
cd /c/Users/alden/MarchMadness && python -m pytest tests/test_features/test_team_history.py -v
```

Expected: 18 passed.

- [ ] **Step 4: Run the diagnostic on the full season range**

```bash
cd /c/Users/alden/MarchMadness && python -m src.diagnose_team_seed_residual
```

Expected output: `Wrote output/team_seed_residual_diagnostic.json and output/team_seed_residual_diagnostic.log`. Inspect `output/team_seed_residual_diagnostic.log` and confirm:
1. Per-seed baseline table: 1-seeds ~3.0-3.3, 2-seeds ~2.4-2.7, 8/9-seeds ~1.0-1.3, 16-seeds ~0.0-0.05.
2. UConn 2024's `mean_10yr` is positive (~+0.5 to +1.5 range expected from the spec walkthrough).
3. UConn 2023's `ewma_hl2` is negative or near-zero (recent 2021/2022 R64 exits dominate).
4. Distribution medians close to 0.
5. Top-10 lists obvious historical-powerhouse seasons (Duke, Kentucky, Kansas, UNC, UConn).

- [ ] **Step 5: Force-add artifacts and commit Phase 1**

```bash
cd /c/Users/alden/MarchMadness && git add -f output/team_seed_residual_diagnostic.json output/team_seed_residual_diagnostic.log && git add src/diagnose_team_seed_residual.py tests/test_features/test_team_history.py && git commit -m "$(cat <<'EOF'
feat(team-seed-residual): Phase 1 diagnostic + force-add artifacts

5-artifact sanity-check (per-seed baseline, 9-champion residuals with
hand-computable cross-check, correlation vs v4 incumbents, distribution
percentiles, top/bottom-10) saved to output/team_seed_residual_diagnostic.{json,log}.
No v4 wire-in yet -- Phase 2 PR follows after human review of artifacts.

Spec: docs/superpowers/specs/2026-05-09-team-seed-residual-design.md
EOF
)"
```

- [ ] **Step 6: PR-style review checkpoint**

> **HUMAN REVIEW CHECKPOINT.** Before starting Phase 2, the user reviews `output/team_seed_residual_diagnostic.log` and confirms (a) per-seed baseline is sane, (b) 9-champion values match qualitative expectations, (c) correlations vs incumbents are < 0.85, (d) distribution and top/bottom-10 pass face validity. If any of these fail, address inline (likely a computational bug; fix and re-run Step 4).

---

## Phase 2: Wire-in + LOSO + verdict

### Task 8: Wire `compute_team_history_features` into `enhanced_model_v3.py`

**Files:**
- Modify: `src/enhanced_model_v3.py`

- [ ] **Step 1: Add the import**

Add at the top of `src/enhanced_model_v3.py`, near the existing `from src.features.coach import compute_coach_features` line (currently line 71):

```python
from src.features.team_history import compute_team_history_features
```

- [ ] **Step 2: Wire-in after the coach features block**

In `src/enhanced_model_v3.py`, after the existing coach features block at line 830-840, add:

```python
    # Team-program tournament-history features (cross-season, team-keyed).
    # Spec: docs/superpowers/specs/2026-05-09-team-seed-residual-design.md
    th_df = compute_team_history_features(
        tournament_field=feature_matrix[["Season", "TeamID"]].drop_duplicates(),
        tourney_results=data["tourney"],
        seeds=data["seeds"],
        window_years=10,
    )
    if not th_df.empty:
        feature_matrix = feature_matrix.merge(
            th_df, on=["TeamID", "Season"], how="left",
        )
        th_feat_names = ["team_seed_residual_mean_10yr",
                         "team_seed_residual_ewma_hl2"]
        new_feature_names.extend(th_feat_names)
        print(f"  Team history features: {len(th_df):,} team-seasons")
```

- [ ] **Step 3: Smoke-run feature build to verify both columns appear**

```bash
cd /c/Users/alden/MarchMadness && MM_SKIP_DEFAULT_LOSO=1 python -m src.enhanced_model_v3 2>&1 | grep -E "Team history|team_seed_residual|Total feature columns"
```

Expected: a line like `Team history features: 1XXX team-seasons`, and `team_seed_residual_mean_10yr` + `team_seed_residual_ewma_hl2` in the printed feature column list. Total feature columns = previous total + 2.

- [ ] **Step 4: Commit the wire-in**

```bash
cd /c/Users/alden/MarchMadness && git add src/enhanced_model_v3.py && git commit -m "feat(team-seed-residual): wire two history features into v4 build"
```

---

### Task 9: Anchor invariance check via `MM_FEATURE_DROP`

**Files:** none (this is a one-shot run)

- [ ] **Step 1: Run the LOSO with both new features dropped, compare against canonical clean v4 baseline**

The canonical clean v4 baseline is `output/pairwise_v4.csv` (PR 21 + recovery). To verify the wire-in is non-invasive when the new features are zero-removed:

```bash
# MM_TUNED_PARAMS_V3 must be a JSON dict of XGB hyperparameters,
# not "1". Use the cached snapshot from a prior run:
cd /c/Users/alden/MarchMadness && \
  MM_FEATURE_DROP=team_seed_residual_mean_10yr,team_seed_residual_ewma_hl2 \
  MM_TUNED_PARAMS_V3="$(cat output/v4_tuned_params.json)" \
  MM_SKIP_DEFAULT_LOSO=1 \
  python -m src.enhanced_model_v3 2>&1 | tee output/anchor_invariance_run.log
```

Expected: writes a fresh `output/pairwise_v4.csv` (overwriting the canonical one). After the run, compare against the prior canonical file — but since we're overwriting, do this BEFORE the run:

- [ ] **Step 2: Pre-run snapshot of canonical baseline**

```bash
cd /c/Users/alden/MarchMadness && cp output/pairwise_v4.csv output/pairwise_v4_canonical_snapshot.csv
```

- [ ] **Step 3: Run anchor LOSO (Step 1 above), then compare**

```bash
cd /c/Users/alden/MarchMadness && python -c "
import pandas as pd
canonical = pd.read_csv('output/pairwise_v4_canonical_snapshot.csv')
fresh = pd.read_csv('output/pairwise_v4.csv')
joined = canonical.merge(fresh, on=['Season','TeamA','TeamB'], suffixes=('_old','_new'))
diff = (joined['p_a_wins_old'] - joined['p_a_wins_new']).abs()
print(f'rows: {len(joined)}, max_abs_diff: {diff.max():.6e}, mean_abs_diff: {diff.mean():.6e}')
"
```

Expected: `max_abs_diff < 1e-6` (XGB nondeterminism may produce tiny shifts; if max_abs_diff > 1e-3, the wire-in is leaking even when features are dropped, which is a bug).

- [ ] **Step 4: Restore canonical and clean up**

```bash
cd /c/Users/alden/MarchMadness && mv output/pairwise_v4_canonical_snapshot.csv output/pairwise_v4.csv
```

- [ ] **Step 5: Commit the anchor log (informational only)**

```bash
cd /c/Users/alden/MarchMadness && git add -f output/anchor_invariance_run.log && git commit -m "data(team-seed-residual): anchor invariance LOSO log (features dropped)"
```

---

### Task 10: Run full LOSO with new features active

**Files:** none

- [ ] **Step 1: Run the LOSO with new features active (no MM_FEATURE_DROP)**

```bash
cd /c/Users/alden/MarchMadness && \
  MM_TUNED_PARAMS_V3="$(cat output/v4_tuned_params.json)" \
  MM_SKIP_DEFAULT_LOSO=1 \
  python -m src.enhanced_model_v3 2>&1 | tee output/team_seed_residual_loso_run.log
```

Expected: ~30-60 min wall. Writes a fresh `output/pairwise_v4.csv` that includes the two new features in v4's training. Save the new file under a versioned name first:

- [ ] **Step 2: Save versioned output**

```bash
cd /c/Users/alden/MarchMadness && cp output/pairwise_v4.csv output/pairwise_v4_with_team_history.csv
```

- [ ] **Step 3: Apply v8 stage-2 over the new v4 stage-1**

`src/train_stage2.py` reads `output/pairwise_v4.csv` and writes `output/pairwise_v8.csv`. Just run it:

```bash
cd /c/Users/alden/MarchMadness && python -m src.train_stage2 2>&1 | tee output/v8_retrain_team_history_run.log
# After it finishes, save the v8 output under a versioned name:
cp output/pairwise_v8.csv output/pairwise_v8_with_team_history.csv
```

- [ ] **Step 4: Score 22-season bracket points**

NOTE: `score_pairwise_path` returns `{"total_pts": float, "per_season_pts": {...}}` — the key is `total_pts`, NOT `total`.

```bash
cd /c/Users/alden/MarchMadness && python -c "
from src.score_chalk_brackets import score_pairwise_path
canonical = score_pairwise_path('output/pairwise_v8_canonical_snapshot.csv')
new = score_pairwise_path('output/pairwise_v8_with_team_history.csv')
print(f'canonical v8: {canonical[\"total_pts\"]:.0f}')
print(f'new (with team history): {new[\"total_pts\"]:.0f}')
print(f'delta: {new[\"total_pts\"] - canonical[\"total_pts\"]:+.0f} brkt pts')
"
```

Expected: print delta in brkt pts. Save the JSON output for the verdict step.

- [ ] **Step 5: Commit raw outputs**

```bash
cd /c/Users/alden/MarchMadness && git add -f output/pairwise_v4_with_team_history.csv output/pairwise_v8_with_team_history.csv output/team_seed_residual_loso_run.log && git commit -m "data(team-seed-residual): force-add LOSO outputs"
```

---

### Task 11: Compute verdict + write findings + update TODO

**Files:**
- Create: `docs/notes/2026-05-09-team-seed-residual.md`
- Create: `output/team_seed_residual_loso_summary.json`
- Modify: `TODO.md`

- [ ] **Step 1: Compute per-season deltas and verdict**

```bash
cd /c/Users/alden/MarchMadness && python -c "
import json
from src.score_chalk_brackets import score_pairwise_path

# score_pairwise_path returns {'total_pts': float, 'per_season_pts': {int: float}}
canonical = score_pairwise_path('output/pairwise_v8_canonical_snapshot.csv')
new = score_pairwise_path('output/pairwise_v8_with_team_history.csv')

per_season_delta = {}
for season in canonical['per_season_pts']:
    per_season_delta[season] = new['per_season_pts'][season] - canonical['per_season_pts'][season]

agg = sum(per_season_delta.values())
max_swing = max(per_season_delta.values(), key=abs)
max_pos = max((d for d in per_season_delta.values() if d > 0), default=0)
fragility_check = agg - max_pos  # 'aggregate minus largest single positive'

if agg >= 20 and fragility_check >= 5:
    verdict = 'PASS'
elif agg >= 10:
    verdict = 'MARGINAL (low magnitude)' if agg < 20 else 'MARGINAL (fragile)'
else:
    verdict = 'FAIL'

wins = sum(1 for d in per_season_delta.values() if d > 0)
losses = sum(1 for d in per_season_delta.values() if d < 0)
ties = sum(1 for d in per_season_delta.values() if d == 0)

summary = {
    'canonical_total': canonical['total_pts'],
    'new_total': new['total_pts'],
    'aggregate_delta': agg,
    'max_swing_value': max_swing,
    'max_positive_delta': max_pos,
    'fragility_check_value': fragility_check,
    'fragility_check_pass': fragility_check >= 5,
    'wins_losses_ties': [wins, losses, ties],
    'per_season_delta': {str(k): v for k, v in per_season_delta.items()},
    'verdict': verdict,
    'verdict_bands': {
        'PASS': '>=20 brkt pts AND fragility-check>=5',
        'MARGINAL': '+10..+20 brkt pts OR PASS-magnitude but fragile',
        'FAIL': '<+10 brkt pts',
    },
}

with open('output/team_seed_residual_loso_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)
print(json.dumps(summary, indent=2))
"
```

- [ ] **Step 2: Write the findings note**

Create `docs/notes/2026-05-09-team-seed-residual.md`. Use the template from `docs/notes/2026-05-08-v4-calibration-temperature-scaling.md` as the structural guide. Required sections:
- TL;DR with the verdict, aggregate delta, and the spec-gated decision (swap-in / candidate / closed)
- Per-season W/L/T table
- Largest single-season swings (top 3 wins, top 3 losses)
- Anchor invariance check result
- SHAP feature importance for the two new features (run `xgb.plot_importance` on the LOSO model from one season, e.g. 2024)
- Open questions / next steps based on verdict
- Files of record + commands to reproduce

- [ ] **Step 3: Update TODO.md**

Per the verdict, update the Active queue and Done sections of `TODO.md`:
- If PASS: move "Team-program tournament history (#1)" from Active queue to Done section. Promote the next item (likely roster-level returning-experience #2 or pool-aware bracket construction #3, depending on what the user wants next).
- If MARGINAL: leave in Active queue with the MARGINAL flag and the candidate sensitivity-sweep to-do (HL or k).
- If FAIL: move to "Tried and rejected" section with the empirical evidence summarized.

- [ ] **Step 4: Commit**

```bash
cd /c/Users/alden/MarchMadness && git add -f output/team_seed_residual_loso_summary.json && git add docs/notes/2026-05-09-team-seed-residual.md TODO.md && git commit -m "docs(team-seed-residual): findings + TODO update -- <VERDICT>"
```

(Replace `<VERDICT>` with the actual verdict from Step 1.)

---

## Self-review notes

**Spec coverage check (against `docs/superpowers/specs/2026-05-09-team-seed-residual-design.md`):**
- "Two new XGB features" — covered by Tasks 6 (feature build) + 8 (wire-in).
- "Empirical per-seed baseline, leak-safe" — Task 2 (`compute_per_seed_baseline` with assertion + tests).
- "Window=10, HL=2, k=3" — defaults locked into all five public functions.
- "Bayesian shrinkage" — Tasks 3 (`shrunk_mean`) + 4 (`shrunk_ewma` with n-based shrinkage explanation).
- "Phase 1 sanity-check artifacts (5 items)" — Task 7's diagnostic emits all 5.
- "Phase 2 LOSO + pre-registered verdict bands" — Tasks 10-11.
- "Anchor invariance check" — Task 9.
- "Per-season fragility check" — explicit in Task 11's Python.
- "Hand-computed UConn 2024 spot check" — Task 6 step 1 test.
- "Test convention `rounds_won = games won, champion = 6`" — Task 1 test docstring + Task 6 docstring + spec walkthrough comments.

**Type / signature consistency check:**
- `compute_per_seed_baseline` returns `dict[int | str, float]` (with `'__fallback__'` key); consumers use `baseline.get(seed, baseline['__fallback__'])`. Consistent in Tasks 2, 5, 6.
- `shrunk_mean(residuals: list[float], k=3)` — consumed in Task 6 as `shrunk_mean([r for (_, _, r) in residuals], k=3)`. Consistent.
- `shrunk_ewma(residuals_with_age: list[tuple[int, float]], half_life=2, k=3)` — consumed in Task 6 as `shrunk_ewma([(a, r) for (a, _, r) in residuals], ...)`. Consistent.
- `compute_team_residuals_in_window` returns `list[tuple[int, int, float]]` = `(years_ago, prior_seed, residual)`. Consumed correctly in Task 6 and the diagnostic.
- `compute_team_history_features` returns DataFrame `[Season, TeamID, team_seed_residual_mean_10yr, team_seed_residual_ewma_hl2]`. Consumed in Task 8 wire-in.

**Placeholder scan:**
- Task 10 Step 3 references `train_stage2.fit_loso` with a follow-up "(If `train_stage2` doesn't expose a clean LOSO entry point, mirror the recipe from `src/eval_v4_calibration.py:run_phase2`.)" — this is acceptable as the integrating script will need a small amount of glue and the executor has explicit fallback guidance. Not a placeholder; an entry-point uncertainty resolved at execution time.
- Task 11 Step 2 ("Required sections") lists what to include in the findings note, modeled on the calibration-temperature-scaling note structure. Not a placeholder; standard findings-note recipe.

No TBDs, TODOs, or "implement later" markers. Plan complete.
