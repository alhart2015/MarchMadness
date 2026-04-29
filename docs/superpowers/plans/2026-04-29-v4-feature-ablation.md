# v4 Feature Ablation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Identify which v3-or-coach feature drives v4's 2026 over-confidence on Vanderbilt (R32), Iowa St. (S16), Texas Tech (R32), and Duke (E8) by drop-and-retrain ablation, gated on a 22-season LOSO log-loss tolerance.

**Architecture:** Add two env-var hooks to the existing v4 pipeline (`enhanced_model_v3.py`) so it can run with arbitrary feature drops and write outputs to a per-ablation suffix. A new driver script (`src/ablate_v4.py`) invokes the pipeline as a subprocess for each ablation, captures the artifacts, and assembles a comparison CSV. The 22-season bracket-points metric is computed by reusing `src/score_chalk_brackets.py` against each ablation's pairwise CSV.

**Tech Stack:** Python, pandas, XGBoost, subprocess. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-04-29-v4-feature-ablation.md`

**Repo conventions to follow:**
- ASCII-only files (CLAUDE.md). Verify with `python -c "open('PATH').read().encode('ascii')"` after writing.
- `pytest -v` must pass before any task is marked done. For tasks that touch features or training, also run the relevant subset.
- Frequent commits, one per task.
- Reuse existing helpers; no new abstractions unless an existing one is wrong.

---

### Task 1: Add `MM_FEATURE_DROP` env-var hook to v4 pipeline

**Files:**
- Modify: `src/enhanced_model_v3.py:714-720` (the section where `feature_cols` is computed via `get_feature_cols(feature_matrix)`)
- Test: `tests/test_features/test_ablation_hooks.py` (create)

The hook reads a comma-separated list of feature names from `MM_FEATURE_DROP`, removes them from `feature_cols` after `get_feature_cols(...)`, warns on unknown names, and prints what was dropped. A dropped name absent from `feature_cols` is a warning, not an error -- otherwise typos in the driver's group definitions abort an entire 22-season retrain.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_features/test_ablation_hooks.py
"""Tests for v4 ablation env-var hooks (MM_FEATURE_DROP, MM_OUTPUT_SUFFIX).

The hooks are tiny pieces of logic embedded in enhanced_model_v3.main(),
so we test them as standalone helpers extracted into the same module
namespace. Drop logic: take a feature_cols list and an env var string,
return the filtered list plus the set of names that were unknown.
"""
import os
from src.enhanced_model_v3 import apply_feature_drop


def test_drop_env_empty_returns_unchanged():
    cols = ["a", "b", "c"]
    result, missing = apply_feature_drop(cols, "")
    assert result == cols
    assert missing == set()


def test_drop_env_removes_named_columns():
    cols = ["a", "b", "c", "d"]
    result, missing = apply_feature_drop(cols, "b,d")
    assert result == ["a", "c"]
    assert missing == set()


def test_drop_env_strips_whitespace():
    cols = ["a", "b", "c"]
    result, missing = apply_feature_drop(cols, " a , c ")
    assert result == ["b"]
    assert missing == set()


def test_drop_env_reports_unknown_names():
    cols = ["a", "b"]
    result, missing = apply_feature_drop(cols, "a,zzz")
    assert result == ["b"]
    assert missing == {"zzz"}


def test_drop_env_preserves_order():
    cols = ["d", "c", "b", "a"]
    result, _ = apply_feature_drop(cols, "c")
    assert result == ["d", "b", "a"]
```

- [ ] **Step 2: Run the test and verify it fails**

Run: `pytest tests/test_features/test_ablation_hooks.py -v`
Expected: FAIL with `ImportError: cannot import name 'apply_feature_drop'`.

- [ ] **Step 3: Implement `apply_feature_drop` in the v4 pipeline**

In `src/enhanced_model_v3.py`, add a module-level helper just below the existing imports block (after the `from src.bracket.line_blending import blend_r64_probs` line, around line 75):

```python
def apply_feature_drop(feature_cols, drop_env):
    """Filter feature_cols by names listed in MM_FEATURE_DROP env-var string.

    Returns (filtered_cols, missing_names_set). Unknown names are returned in
    `missing` for the caller to log -- not raised, so a typo does not abort
    a multi-hour LOSO retrain.
    """
    if not drop_env:
        return list(feature_cols), set()
    drop = {c.strip() for c in drop_env.split(",") if c.strip()}
    present = drop & set(feature_cols)
    missing = drop - set(feature_cols)
    filtered = [c for c in feature_cols if c not in present]
    return filtered, missing
```

- [ ] **Step 4: Run the test and verify it passes**

Run: `pytest tests/test_features/test_ablation_hooks.py -v`
Expected: PASS, all 5 tests.

- [ ] **Step 5: Wire the hook into `main()`**

Find the line `feature_cols = get_feature_cols(feature_matrix)` (currently around line 717). Immediately after it, insert:

```python
    # ABLATION HOOK: drop features named in MM_FEATURE_DROP env var.
    import os as _os
    _drop_env = _os.environ.get("MM_FEATURE_DROP", "")
    if _drop_env:
        _before = len(feature_cols)
        feature_cols, _missing = apply_feature_drop(feature_cols, _drop_env)
        if _missing:
            print(f"  ABLATION WARNING: MM_FEATURE_DROP names not in feature_cols: {sorted(_missing)}")
        print(f"  ABLATION: dropped {_before - len(feature_cols)} features (drop list: {_drop_env}); remaining: {len(feature_cols)}")
```

- [ ] **Step 6: Smoke-test the wiring**

Run a tiny invocation to confirm the hook fires without breaking the pipeline. We do *not* run a full LOSO -- just confirm the script starts up, prints the ABLATION line, and the dropped column does not appear in the printed feature list.

```bash
MM_FEATURE_DROP=coach_career_winpct python -c "from src.enhanced_model_v3 import apply_feature_drop; cols = ['a', 'coach_career_winpct', 'b']; print(apply_feature_drop(cols, 'coach_career_winpct'))"
```
Expected stdout: `(['a', 'b'], set())`.

- [ ] **Step 7: ASCII verification + commit**

```bash
python -c "open('src/enhanced_model_v3.py').read().encode('ascii')"
python -c "open('tests/test_features/test_ablation_hooks.py').read().encode('ascii')"
git add src/enhanced_model_v3.py tests/test_features/test_ablation_hooks.py
git commit -m "feat: add MM_FEATURE_DROP env-var hook for v4 ablation"
```

---

### Task 2: Add `MM_OUTPUT_SUFFIX` env-var hook to v4 pipeline

**Files:**
- Modify: `src/enhanced_model_v3.py` (multiple hard-coded output paths in `main()`)
- Test: `tests/test_features/test_ablation_hooks.py` (extend)

`MM_OUTPUT_SUFFIX` is a string appended to every output filename produced by `main()` (before the file extension). Default empty string = current behavior. With `MM_OUTPUT_SUFFIX=_drop_coach`, `output/cv_per_season_v3.csv` becomes `output/cv_per_season_v3_drop_coach.csv`. This stops parallel ablation runs from clobbering each other and from clobbering the canonical v4 outputs that `train_stage2.py`, `score_chalk_brackets.py`, etc., depend on.

The existing `MM_PAIRWISE_OUT` env var already lets the LOSO loop redirect the per-season pairwise CSV; the driver in Task 3 will set both env vars in tandem.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_features/test_ablation_hooks.py`:

```python
from src.enhanced_model_v3 import apply_output_suffix


def test_suffix_empty_returns_unchanged():
    assert apply_output_suffix("output/foo.csv", "") == "output/foo.csv"


def test_suffix_inserts_before_extension():
    assert apply_output_suffix("output/foo.csv", "_drop_coach") == "output/foo_drop_coach.csv"


def test_suffix_handles_json():
    assert apply_output_suffix("output/bracket_data.json", "_x") == "output/bracket_data_x.json"


def test_suffix_no_extension():
    # Edge case: path without extension. Just append.
    assert apply_output_suffix("output/foo", "_x") == "output/foo_x"


def test_suffix_handles_path_with_dot_in_directory():
    # e.g., output/v.4/foo.csv -- only the final ext should be split.
    assert apply_output_suffix("output/v.4/foo.csv", "_x") == "output/v.4/foo_x.csv"
```

- [ ] **Step 2: Run test, verify it fails**

Run: `pytest tests/test_features/test_ablation_hooks.py -v`
Expected: FAIL with `ImportError: cannot import name 'apply_output_suffix'`.

- [ ] **Step 3: Implement `apply_output_suffix`**

Add to `src/enhanced_model_v3.py` next to `apply_feature_drop`:

```python
def apply_output_suffix(path, suffix):
    """Insert `suffix` before the final extension of `path`. Empty suffix = no-op.

    Uses os.path.splitext so only the trailing extension is split, even if
    intermediate directory names contain dots.
    """
    if not suffix:
        return path
    import os
    root, ext = os.path.splitext(path)
    return f"{root}{suffix}{ext}"
```

- [ ] **Step 4: Run test, verify it passes**

Run: `pytest tests/test_features/test_ablation_hooks.py -v`
Expected: PASS, all 10 tests.

- [ ] **Step 5: Wire suffix through `main()` output paths**

In `main()`, near the top (just after the `overall_start = time.time()` line), read the suffix once:

```python
    import os as _os
    _output_suffix = _os.environ.get("MM_OUTPUT_SUFFIX", "")
    if _output_suffix:
        print(f"  ABLATION: output suffix = '{_output_suffix}'")
```

Then update each hard-coded `output/...` path to wrap with `apply_output_suffix`. The current paths are:

| Line (approx) | Original | Replacement |
|---------------|----------|-------------|
| 820 | `"output/cv_per_season_v3.csv"` | `apply_output_suffix("output/cv_per_season_v3.csv", _output_suffix)` |
| 1014 | `str(OUTPUT_DIR / "bracket_2026_real.csv")` | `apply_output_suffix(str(OUTPUT_DIR / "bracket_2026_real.csv"), _output_suffix)` |
| 1019 | `str(OUTPUT_DIR / "bracket_2026_real_structure.csv")` | same wrapping |
| 1031 | `str(OUTPUT_DIR / "pairwise_probs.json")` | same wrapping |
| 1050 | `str(OUTPUT_DIR / "bracket_data.json")` | same wrapping |
| 1057 | `str(OUTPUT_DIR / "bracket_compact.json")` | same wrapping |

Also grep for any further `output/` literal strings or `OUTPUT_DIR /` paths in the file you may have missed:

```bash
grep -n "output/\|OUTPUT_DIR" src/enhanced_model_v3.py
```

Wrap each one. The HTML / kaggle submission paths near the bottom of the file are likely targets too.

- [ ] **Step 6: Smoke-test the wiring**

Confirm the suffix logic works via the helper directly:

```bash
python -c "from src.enhanced_model_v3 import apply_output_suffix; print(apply_output_suffix('output/cv_per_season_v3.csv', '_drop_coach'))"
```
Expected: `output/cv_per_season_v3_drop_coach.csv`.

- [ ] **Step 7: ASCII verification + commit**

```bash
python -c "open('src/enhanced_model_v3.py').read().encode('ascii')"
git add src/enhanced_model_v3.py tests/test_features/test_ablation_hooks.py
git commit -m "feat: add MM_OUTPUT_SUFFIX env-var hook for v4 ablation"
```

---

### Task 3: Add reusable bracket-points scorer for arbitrary pairwise CSV

**Files:**
- Modify: `src/score_chalk_brackets.py` (extract a callable)
- Test: `tests/test_score_chalk_brackets.py` (create)

`src/score_chalk_brackets.py` currently hard-codes `output/pairwise_v1.csv`, `pairwise_v2.csv`, etc. in `main()`. Extract a function `score_pairwise_path(path) -> dict` that returns `{"total_pts": float, "per_season_pts": {2003: 113, ...}}` for any pairwise CSV. `main()` then composes that across all known model versions, behavior unchanged.

- [ ] **Step 1: Inspect current `main()` to identify the per-version loop**

Run: `grep -n "score_season\|for ver" src/score_chalk_brackets.py`

Note which lines walk per-season scoring for a single version. The new function lifts that loop.

- [ ] **Step 2: Write the failing test**

```python
# tests/test_score_chalk_brackets.py
"""Tests that score_pairwise_path runs cleanly on the canonical v4 file
and returns the structure the ablation driver expects."""
from pathlib import Path
import pytest
from src.score_chalk_brackets import score_pairwise_path


@pytest.mark.skipif(
    not Path("output/pairwise_v4.csv").exists(),
    reason="pairwise_v4.csv missing -- run v4 LOSO first",
)
def test_score_v4_returns_known_shape():
    result = score_pairwise_path("output/pairwise_v4.csv")
    assert "total_pts" in result
    assert "per_season_pts" in result
    assert isinstance(result["total_pts"], (int, float))
    assert isinstance(result["per_season_pts"], dict)
    # 22 seasons (2003-2024 typical), give or take 2.
    assert 18 <= len(result["per_season_pts"]) <= 25
    # v4's known mean is ~121 -- total across 22 seasons should be in range.
    assert 2000 <= result["total_pts"] <= 3500


def test_missing_pairwise_path_raises():
    with pytest.raises(FileNotFoundError):
        score_pairwise_path("output/this_does_not_exist.csv")
```

- [ ] **Step 3: Run test, verify it fails**

Run: `pytest tests/test_score_chalk_brackets.py -v`
Expected: FAIL with `ImportError: cannot import name 'score_pairwise_path'`.

- [ ] **Step 4: Extract `score_pairwise_path`**

In `src/score_chalk_brackets.py`, add (above `main()`):

```python
def score_pairwise_path(path):
    """Score the chalk bracket implied by `path` against actuals across all
    seasons present. Returns {"total_pts": float, "per_season_pts": dict}.
    Raises FileNotFoundError if `path` does not exist.
    """
    if not Path(path).exists():
        raise FileNotFoundError(path)

    slots_df = pd.read_csv(DATA / "MNCAATourneySlots.csv")
    seeds_df = pd.read_csv(DATA / "MNCAATourneySeeds.csv")
    results_df = pd.read_csv(DATA / "MNCAATourneyCompactResults.csv")

    probs_by_season = load_pairwise(path)
    per_season = {}
    for season, probs in probs_by_season.items():
        pts = score_season(season, slots_df, seeds_df, results_df, probs)
        per_season[int(season)] = float(pts)
    return {
        "total_pts": float(sum(per_season.values())),
        "per_season_pts": per_season,
    }
```

Then refactor the existing `main()` to call `score_pairwise_path(...)` for each version instead of duplicating the loop. Behavior and printed output stay the same.

- [ ] **Step 5: Run tests, verify they pass**

Run: `pytest tests/test_score_chalk_brackets.py -v`
Expected: PASS, both tests.

Then run `score_chalk_brackets.main()` and verify the printed totals match the prior numbers (v4 ~ 121.4 per-season mean, ~2670 total over 22 seasons):

```bash
python src/score_chalk_brackets.py | tail -20
```
Expected: same per-season and total numbers as before the refactor (within float-precision noise).

- [ ] **Step 6: ASCII verification + commit**

```bash
python -c "open('src/score_chalk_brackets.py').read().encode('ascii')"
git add src/score_chalk_brackets.py tests/test_score_chalk_brackets.py
git commit -m "refactor: extract score_pairwise_path for ablation reuse"
```

---

### Task 4: Write the ablation driver `src/ablate_v4.py`

**Files:**
- Create: `src/ablate_v4.py`
- Create: `tests/test_ablate_v4.py`

The driver:
1. Defines feature groups (Pass 1) and a configurable individual-feature list (Pass 2).
2. For each ablation, sets `MM_FEATURE_DROP`, `MM_OUTPUT_SUFFIX`, `MM_PAIRWISE_OUT`, and `MM_TUNED_PARAMS_V3` (so Optuna is *not* re-run -- v4's tuned params are reused) and invokes `python src/enhanced_model_v3.py` as a subprocess.
3. After each run, parses the suffixed `cv_per_season_v3<sfx>.csv` for LOSO log loss and the suffixed `bracket_data.json` for 2026 advancement probabilities.
4. Calls `score_pairwise_path` (Task 3) on the suffixed pairwise CSV to get 22-yr bracket points.
5. Writes `output/ablation_v4_results.csv` with one row per (ablation, bust_team).

Subprocess isolation matters: each ablation is a fresh Python process, so memory does not pile up across 5+ multi-hour runs. Use `subprocess.run(..., check=True)` with `MM_*` env vars merged into a copy of `os.environ`.

**Bust teams (2026 TeamIDs):**

The driver looks them up by name from `data/raw/march-machine-learning-2026/MTeams.csv` and the 2026 bracket. Hard-coded names + a name->TeamID resolution at startup is fine; no need to embed integer IDs.

| Team        | Eliminated round | "P advance past round" key |
|-------------|------------------|----------------------------|
| Vanderbilt  | R32              | `R32` (their P of advancing past R32 = reaching S16) |
| Iowa State  | S16              | `S16`                                                 |
| Texas Tech  | R32              | `R32`                                                 |
| Duke        | E8               | `E8`                                                  |

The pipeline writes `output/bracket_data<sfx>.json` with a top-level `{TeamID_str: {"name": ..., "advancement": {"R64": ..., "R32": ..., "S16": ..., "E8": ..., "F4": ..., "Champ": ...}}}` structure (verified at `enhanced_model_v3.py:1037-1053`). The "P advance past R32" we want is `bracket_data[tid]["advancement"]["S16"]` -- because the round key represents the probability of advancing *to* that round.

Read this carefully: a bust at R32 (lost in round 2) has its advancement under-estimated by the model's `S16` probability. So the metric we report is `P(team reaches the round AFTER its bust round)`. For each bust:

| Team       | Bust round | Reported metric           |
|------------|------------|---------------------------|
| Vanderbilt | R32        | `advancement["S16"]`      |
| Iowa State | S16        | `advancement["E8"]`       |
| Texas Tech | R32        | `advancement["S16"]`      |
| Duke       | E8         | `advancement["F4"]`       |

This gives "the model said this team would survive their bust round with probability X". The numbers cited in the spec (Vanderbilt 0.79, Iowa St. 0.82, Texas Tech 0.86, Duke 0.88) are this metric.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_ablate_v4.py
"""Tests for the ablation driver. The subprocess invocation is too expensive
to test end-to-end; we test the helpers (group definitions, parse/aggregate
logic) and use a synthetic bracket_data.json fixture."""
import json
from pathlib import Path
import pytest
from src.ablate_v4 import (
    GROUP_ABLATIONS, BUST_TEAMS,
    parse_advance_probs, build_results_row,
)


def test_group_ablations_cover_spec():
    expected_groups = {"late_season", "trajectory", "conf_tourney",
                       "vegas_trend", "coach"}
    assert set(GROUP_ABLATIONS.keys()) == expected_groups


def test_late_season_group_features():
    assert set(GROUP_ABLATIONS["late_season"]) == {
        "late_adj_oe", "late_adj_de", "late_adj_em", "late_sos"
    }


def test_coach_group_features():
    assert set(GROUP_ABLATIONS["coach"]) == {
        "coach_career_games", "coach_career_wins", "coach_career_winpct",
        "coach_career_f4_apps", "coach_career_champs", "coach_career_seasons",
    }


def test_bust_teams_metric_keys():
    # Each bust must declare bust_round + advance_key (= round AFTER bust).
    assert {b["name"] for b in BUST_TEAMS} == {
        "Vanderbilt", "Iowa State", "Texas Tech", "Duke",
    }
    expected = {"Vanderbilt": "S16", "Iowa State": "E8",
                "Texas Tech": "S16", "Duke": "F4"}
    for b in BUST_TEAMS:
        assert b["advance_key"] == expected[b["name"]]


def test_parse_advance_probs_extracts_named_team(tmp_path):
    bracket_data = {
        "1101": {"name": "Vanderbilt", "seed": 5, "region": "South",
                 "advancement": {"R64": 0.95, "R32": 0.79, "S16": 0.42}},
        "1102": {"name": "Duke", "seed": 1, "region": "East",
                 "advancement": {"R64": 0.99, "R32": 0.93, "S16": 0.90,
                                  "E8": 0.88, "F4": 0.62, "Champ": 0.30}},
    }
    p = tmp_path / "bracket_data.json"
    p.write_text(json.dumps(bracket_data))
    assert parse_advance_probs(p, "Vanderbilt", "S16") == 0.42
    assert parse_advance_probs(p, "Duke", "F4") == 0.62


def test_parse_advance_probs_missing_team_returns_none(tmp_path):
    p = tmp_path / "bracket_data.json"
    p.write_text(json.dumps({}))
    assert parse_advance_probs(p, "Vanderbilt", "S16") is None


def test_build_results_row_shape():
    row = build_results_row(
        ablation="drop_coach", team="Duke", bust_round="E8",
        advance_key="F4",
        p_advance_baseline=0.62, p_advance_ablated=0.50,
        loso_baseline=0.4321, loso_ablated=0.4385,
        bracket_pts_baseline=2670.0, bracket_pts_ablated=2640.0,
    )
    assert row["ablation"] == "drop_coach"
    assert row["team"] == "Duke"
    assert row["delta_pp"] == pytest.approx(-12.0)
    assert row["loso_logloss_delta"] == pytest.approx(0.0064)
    assert row["bracket_pts_delta"] == pytest.approx(-30.0)
```

- [ ] **Step 2: Run test, verify it fails**

Run: `pytest tests/test_ablate_v4.py -v`
Expected: FAIL with `ModuleNotFoundError: src.ablate_v4`.

- [ ] **Step 3: Implement the driver**

```python
# src/ablate_v4.py
"""Driver: run v4 LOSO + 2026 prediction with each named feature drop
group, then with named individual features in Pass 2. Writes a per-row
results CSV.

Spec: docs/superpowers/specs/2026-04-29-v4-feature-ablation.md
Plan: docs/superpowers/plans/2026-04-29-v4-feature-ablation.md
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

from src.score_chalk_brackets import score_pairwise_path


GROUP_ABLATIONS = {
    "late_season": ["late_adj_oe", "late_adj_de", "late_adj_em", "late_sos"],
    "trajectory":  ["efficiency_trend", "margin_trend"],
    "conf_tourney": ["conf_tourney_wins", "conf_tourney_champ"],
    "vegas_trend": ["vegas_late_spread_delta"],
    "coach": ["coach_career_games", "coach_career_wins",
              "coach_career_winpct", "coach_career_f4_apps",
              "coach_career_champs", "coach_career_seasons"],
}

BUST_TEAMS = [
    {"name": "Vanderbilt", "bust_round": "R32", "advance_key": "S16"},
    {"name": "Iowa State", "bust_round": "S16", "advance_key": "E8"},
    {"name": "Texas Tech", "bust_round": "R32", "advance_key": "S16"},
    {"name": "Duke",       "bust_round": "E8",  "advance_key": "F4"},
]

OUTPUT_DIR = Path("output")
ABLATION_DIR = OUTPUT_DIR / "ablation"
RESULTS_CSV = OUTPUT_DIR / "ablation_v4_results.csv"


def parse_advance_probs(bracket_data_path, team_name, advance_key):
    """Read bracket_data.json and return advancement[advance_key] for the
    team whose 'name' matches team_name (case-insensitive). None if absent.
    """
    data = json.loads(Path(bracket_data_path).read_text())
    target = team_name.lower().strip()
    for tid, entry in data.items():
        if entry.get("name", "").lower().strip() == target:
            return entry.get("advancement", {}).get(advance_key)
    return None


def parse_loso_logloss(cv_per_season_path):
    """Return mean log_loss across seasons from cv_per_season_v3<sfx>.csv."""
    df = pd.read_csv(cv_per_season_path)
    return float(df["log_loss"].mean())


def build_results_row(ablation, team, bust_round, advance_key,
                      p_advance_baseline, p_advance_ablated,
                      loso_baseline, loso_ablated,
                      bracket_pts_baseline, bracket_pts_ablated):
    """Assemble one CSV row. delta_pp is in percentage points."""
    return {
        "ablation": ablation,
        "team": team,
        "bust_round": bust_round,
        "advance_key": advance_key,
        "p_advance_baseline": p_advance_baseline,
        "p_advance_ablated": p_advance_ablated,
        "delta_pp": (p_advance_ablated - p_advance_baseline) * 100
            if (p_advance_baseline is not None and p_advance_ablated is not None)
            else None,
        "loso_logloss_baseline": loso_baseline,
        "loso_logloss_ablated": loso_ablated,
        "loso_logloss_delta": loso_ablated - loso_baseline,
        "bracket_pts_baseline": bracket_pts_baseline,
        "bracket_pts_ablated": bracket_pts_ablated,
        "bracket_pts_delta": bracket_pts_ablated - bracket_pts_baseline,
    }


def run_pipeline(tag, drop_features, tuned_params_json):
    """Invoke enhanced_model_v3.py as a subprocess with the env-var hooks set.
    Returns paths to the suffixed artifacts.
    """
    suffix = "" if tag == "baseline" else f"_{tag}"
    pairwise_csv = ABLATION_DIR / f"pairwise{suffix}.csv"
    pairwise_csv.parent.mkdir(parents=True, exist_ok=True)
    if pairwise_csv.exists():
        pairwise_csv.unlink()  # MM_PAIRWISE_OUT appends; start clean.

    env = os.environ.copy()
    env["MM_FEATURE_DROP"] = ",".join(drop_features)
    env["MM_OUTPUT_SUFFIX"] = suffix
    env["MM_PAIRWISE_OUT"] = str(pairwise_csv)
    env["MM_TUNED_PARAMS_V3"] = tuned_params_json

    print(f"\n>>> ABLATION: {tag} (drop: {drop_features or 'NONE'})")
    print(f"    suffix={suffix} pairwise={pairwise_csv}")
    subprocess.run(
        [sys.executable, "src/enhanced_model_v3.py"],
        env=env, check=True,
    )

    return {
        "pairwise_csv": pairwise_csv,
        "cv_per_season": OUTPUT_DIR / f"cv_per_season_v3{suffix}.csv",
        "bracket_data": OUTPUT_DIR / f"bracket_data{suffix}.json",
    }


def collect_metrics(artifacts):
    return {
        "loso_logloss": parse_loso_logloss(artifacts["cv_per_season"]),
        "bracket_pts": score_pairwise_path(str(artifacts["pairwise_csv"]))["total_pts"],
        "advance_probs": {
            b["name"]: parse_advance_probs(
                artifacts["bracket_data"], b["name"], b["advance_key"]
            )
            for b in BUST_TEAMS
        },
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pass-num", type=int, choices=[1, 2], required=True,
                   help="1 = group ablations, 2 = individual feature drill-down")
    p.add_argument("--features", nargs="*", default=None,
                   help="Pass 2: individual feature names to ablate one at a time")
    p.add_argument("--tuned-params", required=True,
                   help="Path to JSON with v4's Optuna best_params (passed to MM_TUNED_PARAMS_V3)")
    p.add_argument("--baseline-only", action="store_true",
                   help="Run only the no-drop baseline, skip ablations")
    return p.parse_args()


def main():
    args = parse_args()
    tuned_params_json = Path(args.tuned_params).read_text().strip()

    # Always run baseline first.
    results = []
    print("=" * 70 + "\nBASELINE (no drops)\n" + "=" * 70)
    base_artifacts = run_pipeline("baseline", [], tuned_params_json)
    base_metrics = collect_metrics(base_artifacts)

    if args.baseline_only:
        print(f"\nBaseline LOSO log loss : {base_metrics['loso_logloss']:.4f}")
        print(f"Baseline 22yr brkt pts : {base_metrics['bracket_pts']:.1f}")
        for n, p in base_metrics["advance_probs"].items():
            print(f"  {n:>12s}: {p:.3f}")
        return

    if args.pass_num == 1:
        ablations = list(GROUP_ABLATIONS.items())
    else:
        if not args.features:
            sys.exit("Pass 2 requires --features f1 f2 ...")
        ablations = [(f, [f]) for f in args.features]

    for tag, features in ablations:
        artifacts = run_pipeline(f"drop_{tag}", features, tuned_params_json)
        m = collect_metrics(artifacts)
        for b in BUST_TEAMS:
            results.append(build_results_row(
                ablation=tag,
                team=b["name"],
                bust_round=b["bust_round"],
                advance_key=b["advance_key"],
                p_advance_baseline=base_metrics["advance_probs"][b["name"]],
                p_advance_ablated=m["advance_probs"][b["name"]],
                loso_baseline=base_metrics["loso_logloss"],
                loso_ablated=m["loso_logloss"],
                bracket_pts_baseline=base_metrics["bracket_pts"],
                bracket_pts_ablated=m["bracket_pts"],
            ))

    df = pd.DataFrame(results)
    # Append if results CSV already exists from a prior pass; else create.
    if RESULTS_CSV.exists():
        existing = pd.read_csv(RESULTS_CSV)
        df = pd.concat([existing, df], ignore_index=True)
    df.to_csv(RESULTS_CSV, index=False)
    print(f"\nWrote {len(df)} rows to {RESULTS_CSV}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run unit tests, verify they pass**

Run: `pytest tests/test_ablate_v4.py -v`
Expected: PASS, all 7 tests.

- [ ] **Step 5: Save v4's tuned params to a file the driver can pass through**

The driver expects a JSON file with v4's Optuna best_params. v4 has been run before; the tuned params were either printed to stdout or are sitting in a stale env. Find them by running v4 once and capturing the printed `Best params: {...}` line, then saving the dict to `output/v4_tuned_params.json`. Or, if they're already on disk somewhere, locate them.

```bash
grep -rn "Best params\|MM_TUNED_PARAMS_V3" output/ src/ 2>&1 | head -20
```

If nothing surfaces, run v4 once with `n_trials=30` (this is the slow path; ~30 min) and capture from stdout. Save to:

```bash
# example after capture:
cat > output/v4_tuned_params.json <<'EOF'
{"n_estimators": 350, "max_depth": 4, "learning_rate": 0.04, ...}
EOF
```

State the params source explicitly in your commit message.

- [ ] **Step 6: ASCII verification + commit**

```bash
python -c "open('src/ablate_v4.py').read().encode('ascii')"
python -c "open('tests/test_ablate_v4.py').read().encode('ascii')"
git add src/ablate_v4.py tests/test_ablate_v4.py output/v4_tuned_params.json
git commit -m "feat: v4 ablation driver (subprocess-based, group + individual)"
```

---

### Task 5: Run Pass 1 (5 group ablations + baseline)

**Files:** none modified; produces `output/ablation_v4_results.csv` and `output/ablation/pairwise_*.csv`.

This is a long-running step. A single LOSO + 2026 + Optuna-skipped pipeline run on this repo takes [TBD by previous v4 run -- estimate 20-60 min on a workstation]. Six runs (1 baseline + 5 groups) = [120-360 min]. Schedule accordingly.

Each ablation is fully independent. If you want parallelism, launch each subprocess in a separate terminal (each writes its own suffixed artifacts). The driver as written runs them sequentially -- simpler, deterministic, and avoids RAM blowups from 5 concurrent XGBoost trainings.

- [ ] **Step 1: Sanity-check on baseline only**

```bash
python src/ablate_v4.py --pass-num 1 --tuned-params output/v4_tuned_params.json --baseline-only
```

Expected: prints `Baseline LOSO log loss` ~ `0.4320` (matching v4's known number) and `Baseline 22yr brkt pts` ~ `2670` (~121.4 mean x 22 seasons). Vanderbilt advance prob ~ 0.79, Iowa St ~ 0.82, Texas Tech ~ 0.86, Duke ~ 0.88.

If the baseline numbers do not match v4's known values within 0.001 log loss / 5 bracket pts / 1pp on the bust teams, **stop and investigate**. Most likely cause: the suffix wiring in Task 2 missed an output path so the ablation is writing into the canonical v4 outputs and reading them back. Fix before proceeding.

- [ ] **Step 2: Run the full Pass 1**

```bash
python src/ablate_v4.py --pass-num 1 --tuned-params output/v4_tuned_params.json 2>&1 | tee output/ablation/pass1.log
```

- [ ] **Step 3: Verify the results CSV shape**

```bash
python -c "import pandas as pd; df = pd.read_csv('output/ablation_v4_results.csv'); print(df.shape); print(df.head(20))"
```
Expected: 20 rows (5 ablations x 4 bust teams), all numeric columns populated.

- [ ] **Step 4: Commit raw Pass 1 artifacts**

```bash
git add output/ablation_v4_results.csv output/ablation/pass1.log
# Don't commit the per-ablation pairwise CSVs (they're large and reproducible).
echo "output/ablation/pairwise_*.csv" >> .gitignore
echo "output/ablation/cv_per_season_*.csv" >> .gitignore
git add .gitignore
git commit -m "data: v4 ablation Pass 1 results (5 groups vs baseline)"
```

---

### Task 6: Decision point -- review Pass 1, decide on Pass 2

**Files:** none modified; this is an analysis step. The next code task depends on its output.

Open `output/ablation_v4_results.csv`. For each group ablation, check:

- **Did any bust team's `delta_pp` improve by >= +5 pp?** (More negative = worse for that team's advance prob, which is what we *want* on a bust -- the team got over-rated, dropping its advance prob is the goal.) Re-read the metric: `delta_pp = (p_advance_ablated - p_advance_baseline) * 100`. A *negative* delta on a bust team means removing the feature reduced the model's confidence in the bust team -- that is the *desirable* direction. We want to find groups where `delta_pp <= -5` for one or more busts.
- **Did `loso_logloss_delta` stay below `+0.005`?** (More positive = worse calibration across 22 seasons.) A group that helps 2026 by 5pp but hurts 22-yr log loss by +0.01 is overfitting to 2026 and is not a fix.
- **What about `bracket_pts_delta`?** Across 22 seasons, the 2670-ish baseline can absorb -50 noise. A drop > -100 means the group is providing real signal.

A **suspicious group** = `delta_pp <= -5` on at least one bust AND `loso_logloss_delta <= +0.005`. Drill down on each suspicious group's individual features.

- [ ] **Step 1: Tabulate Pass 1 verdict in a markdown note**

Create `docs/notes/2026-04-29-ablation-v4-pass1.md` (a short scratch note, not a spec). For each of the 5 groups, one line:

```
late_season: max bust delta = -2.1pp (Iowa St); LOSO delta = +0.001; brkt pts = -8 -- NOT SUSPICIOUS
coach:       max bust delta = -8.4pp (Duke);    LOSO delta = +0.002; brkt pts = -18 -- SUSPICIOUS, drill down
...
```

Then list the individual features inside any suspicious group(s). For example, if `coach` is suspicious, the Pass 2 list is:
- `coach_career_games`
- `coach_career_wins`
- `coach_career_winpct`
- `coach_career_f4_apps`
- `coach_career_champs`
- `coach_career_seasons`

If multiple groups are suspicious, the Pass 2 list combines all individual features from each suspicious group.

If **no group** is suspicious, skip Tasks 7-8 and write the final note (Task 9) with the verdict "no single feature dominates -- next step is architectural (ensemble / upset model)". State this explicitly so the work has a clean stopping point.

- [ ] **Step 2: Commit the Pass 1 note**

```bash
git add docs/notes/2026-04-29-ablation-v4-pass1.md
git commit -m "docs: v4 ablation Pass 1 verdict"
```

---

### Task 7: Run Pass 2 (drill-down on suspicious group features)

**Files:** none modified; appends rows to `output/ablation_v4_results.csv`.

- [ ] **Step 1: Run Pass 2 with the feature list from Task 6**

Replace `<features...>` with the individual feature names from your Pass 1 verdict.

```bash
python src/ablate_v4.py --pass-num 2 --tuned-params output/v4_tuned_params.json --features <features...> 2>&1 | tee output/ablation/pass2.log
```

- [ ] **Step 2: Verify the appended results**

```bash
python -c "import pandas as pd; df = pd.read_csv('output/ablation_v4_results.csv'); print(df['ablation'].value_counts())"
```
Expected: 5 group ablations + N individual ablations, each with 4 rows.

- [ ] **Step 3: Commit Pass 2 artifacts**

```bash
git add output/ablation_v4_results.csv output/ablation/pass2.log
git commit -m "data: v4 ablation Pass 2 drill-down"
```

---

### Task 8: Write the final findings note

**Files:**
- Create: `docs/notes/2026-04-29-ablation-v4-findings.md`

This is the deliverable from the spec. Structure:

```markdown
# v4 Feature Ablation Findings (2026-04-29)

**Spec:** `docs/superpowers/specs/2026-04-29-v4-feature-ablation.md`
**Plan:** `docs/superpowers/plans/2026-04-29-v4-feature-ablation.md`

## Verdict

[One sentence: which feature(s), if any, are responsible for v4's 2026 over-confidence on the four busts.]

## Pass 1: Group Ablations

| Group | Vandy delta | Iowa St delta | TT delta | Duke delta | LOSO delta | 22yr brkt pts delta |
|-------|-------------|---------------|----------|------------|------------|---------------------|
| late_season | ... | ... | ... | ... | ... | ... |
| ... |

[Brief commentary per row.]

## Pass 2: Drill-down [if applicable]

[Same table for individual features.]

## Recommendation

[One of the three branches from the spec's acceptance criteria:]

- "Feature/group X is the culprit. Follow-up spec: tune/replace/remove X."
- "Multiple features share blame. Follow-up: sensitivity / interaction analysis."
- "No single feature dominates. Next step is architectural -- ensemble or upset-detection model from TODO.md."

## Artifacts

- `output/ablation_v4_results.csv` -- raw per-row results
- `output/ablation/pairwise_<tag>.csv` -- per-ablation 22-season pairwise predictions
- `output/ablation/pass1.log`, `pass2.log` -- pipeline run logs
```

- [ ] **Step 1: Draft the note from the results CSV**

Open `output/ablation_v4_results.csv`. For each ablation's 4 rows, pull the bust deltas, the (single) LOSO delta, and the (single) bracket-points delta into the table. Write the verdict and recommendation explicitly -- one sentence each, not "TBD".

- [ ] **Step 2: ASCII verification + commit**

```bash
python -c "open('docs/notes/2026-04-29-ablation-v4-findings.md').read().encode('ascii')"
git add docs/notes/2026-04-29-ablation-v4-findings.md
git commit -m "docs: v4 feature ablation findings (2026-04-29)"
```

- [ ] **Step 3: Update TODO.md**

Replace the active-queue entry #1 ("Feature ablation on v4's 2026 high-confidence misses") with one of:

- A new "Done" entry summarizing the verdict and pointing to the findings note. Move the next-action item (e.g., "tune coach features") to a new active-queue position based on the recommendation.
- If the verdict was "no single feature dominates", remove the ablation entry from the active queue (it's resolved) and promote one of the architectural items (ensemble or upset model) to position #1.

```bash
git add TODO.md
git commit -m "docs: TODO update after v4 ablation findings"
```

---

### Task 9: Final verification

**Files:** none modified.

- [ ] **Step 1: Run the test suite**

```bash
pytest -v
```
Expected: all green. Tests added in this plan (`tests/test_features/test_ablation_hooks.py`, `tests/test_score_chalk_brackets.py`, `tests/test_ablate_v4.py`) plus the existing suite must all pass.

- [ ] **Step 2: Confirm canonical v4 outputs are intact**

The whole point of `MM_OUTPUT_SUFFIX` is to leave canonical paths alone. Confirm:

```bash
git diff --stat output/pairwise_v4.csv output/cv_per_season_v3.csv output/bracket_data.json output/pairwise_probs.json
```
Expected: empty (no changes from the pre-ablation baseline). If any of those files diff, an output path was missed in Task 2; the ablation runs polluted the canonical artifacts and need to be re-run after the suffix is fixed.

- [ ] **Step 3: Push the branch**

```bash
git push -u origin wip/post-pr4
```

(The user opens the PR.)

---

## Self-review checklist

- [x] Spec coverage: every section of `2026-04-29-v4-feature-ablation.md` is addressed by a task. Two-pass design = Tasks 5 + 7 with decision in Task 6. Drop-and-retrain method = Tasks 1+4. Group + individual features = `GROUP_ABLATIONS` in Task 4. Acceptance "no single feature dominates" branch = Task 6 Step 1 explicit handling.
- [x] No placeholders. The only "TBD" in the plan is a runtime estimate in Task 5 -- that one is genuinely unknown and the engineer will discover it on the baseline run.
- [x] Type consistency: `apply_feature_drop`, `apply_output_suffix`, `score_pairwise_path`, `parse_advance_probs`, `parse_loso_logloss`, `build_results_row`, `run_pipeline`, `collect_metrics` -- names match across tasks. `BUST_TEAMS` schema (`name`/`bust_round`/`advance_key`) is used identically in test, driver, and findings note.
- [x] CLAUDE.md rules: ASCII verification appears in every file-touching task; pytest gate appears in Task 9; symbol/string searches mentioned in Task 2 Step 5 (the grep for residual `output/` paths).
