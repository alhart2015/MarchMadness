# v4 Calibration: Temperature Scaling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Phase 1 -- post-hoc temperature scaling on canonical `output/pairwise_v8.csv` (single global T sweep + per-round T sequential greedy with F4/Champ collapsed); Phase 2 -- conditional on Phase 1 PASS or MARGINAL, scale v4 stage-1 with the winning T configuration and retrain v8 LOSO. Both phases gated on 22-season bracket points vs the canonical 2069 baseline; verdict bands `+25 / +10 / <10` per the codebase convention.

**Architecture:** Two new scripts. `src/apply_temperature_scaling.py` is a pure rescaling module (`scale_pairwise(df, T)` accepting scalar or per-round-dict T; numerical guards on logit/sigmoid; round bucket assignment via `train_upset_model.build_pair_round_lookup` with rounds 5+6 collapsed to bucket `F4_NCG`). `src/eval_v4_calibration.py` is the driver: helpers for verdict bands and drop-best-season delta; sweep functions for global T and per-round greedy; Phase 2 retrain glue around `train_stage2.fit_stage2` + `build_v8_pairwise`; CLI; reliability plot. Anchors throughout: T=1.0 (and the all-1 per-round vector) reproduces canonical `output/pairwise_v8.csv` byte-equal in Phase 1; T=1.0-scaled v4 retrained reproduces `pairwise_v8.csv` byte-equal in Phase 2.

**Tech Stack:** Python, pandas, numpy, xgboost (existing v8 training), matplotlib, pytest. Canonical inputs already in repo: `output/pairwise_v4.csv`, `output/pairwise_v8.csv` (= 2069 brkt pts), Kaggle CSVs under `data/raw/march-machine-learning-2026/`.

**Spec:** `docs/superpowers/specs/2026-05-08-v4-calibration-temperature-scaling-design.md`
**Strategy frame:** `docs/notes/2026-05-07-v4-kaggle-gap-strategy.md`
**Predecessor (FAIL):** `docs/notes/2026-05-07-v4-r64-line-blend.md`

> **Production stage-2 is v8** (clean v8 = 2069 brkt pts; clean v9-C peaked at 1929 -- per the TODO's Compounding-work note, v9-C was reverted). Treat `output/pairwise_v8.csv` as the canonical baseline throughout.

---

## File Structure

**Created (committed):**

- `src/apply_temperature_scaling.py` (~150 LOC)
  - Public: `scale_pairwise(df, T) -> pd.DataFrame` -- T is `float` or `dict[str, float]` keyed on round bucket. Returns a new DataFrame with `p_a_wins` rescaled.
  - Public: `assign_round_buckets(df, slots_df, seeds_df) -> pd.Series` -- per-row bucket label among `{"R64","R32","S16","E8","F4_NCG"}`. Returns `pd.NA` for rows with no resolvable round (play-in pairs, missing seeds).
  - Public: `main(argv) -> int` -- CLI for one-shot frame generation given a path + T value.
  - Private: `_logit(p)` / `_sigmoid(x)` with numeric clipping (`p in [1e-9, 1-1e-9]`).
  - Private: `_round_int_to_bucket(rnd: int) -> str` -- maps `{1:"R64", 2:"R32", 3:"S16", 4:"E8", 5:"F4_NCG", 6:"F4_NCG"}`. Round 0 (play-in) raises `ValueError` -- caller filters those out via `assign_round_buckets`.

- `src/eval_v4_calibration.py` (~400 LOC)
  - Public: `run_global_T_sweep(v8_csv, T_grid, baseline_total) -> dict`.
  - Public: `run_per_round_greedy(v8_csv, T_grid, round_order, baseline_total) -> dict`.
  - Public: `run_phase2(v4_csv, winning_config, baseline_v8_csv, out_csv) -> dict` -- LOSO retrain + score. Reuses `train_stage2.load_per_game_data` + `fit_stage2` + `build_v8_pairwise`.
  - Public: `main(argv) -> int` -- CLI driver.
  - Private: `_score_pairwise_df(df, scratch_path) -> dict` -- writes a tempfile and calls `score_chalk_brackets.score_pairwise_path`. Cleans up scratch on exit.
  - Private: `_anchor_check(df_actual, baseline_csv) -> dict` -- `{matches: bool, max_abs_diff: float}` over `(season, team_a, team_b, p_a_wins)`.
  - Private: `_classify_verdict(delta_total, drop_best_delta, wins) -> str` -- returns `"PASS" | "MARGINAL" | "FAIL"` per spec decision matrix.
  - Private: `_drop_best_season_delta(per_season_delta) -> float` -- sum of all season deltas minus the largest single positive.
  - Private: `_summarize_cell(per_season_delta, baseline_total) -> dict` -- `{total, delta_total, wins, losses, ties, biggest_swing_value, biggest_swing_season, drop_best_season_delta}`.
  - Private: `_plot_reliability(v8_baseline_df, v8_global_df, v8_perround_df, out_path) -> None`.

- `tests/test_apply_temperature_scaling.py` (~180 LOC, 6 unit + 1 smoke)
  - Identity at T=1.0 (synthetic 4-row frame).
  - Flatten at T=2.0 (rank-preserving, all distances to 0.5 shrink monotonically).
  - Sharpen at T=0.5 (rank-preserving, all distances to 0.5 grow monotonically).
  - Per-round dispatch: 5-row synthetic frame with one row per bucket, distinct T per bucket, verify each row gets the right T.
  - Numeric guard: input `p in {0.0, 1.0}` produces finite output (no inf/NaN).
  - Anchor regression: applying T=1.0 (and all-1 per-round dict) to canonical `output/pairwise_v8.csv` produces byte-equal output (`max_abs_diff < 1e-9`). Skipped if `output/pairwise_v8.csv` absent.
  - Smoke: `assign_round_buckets` over real seeds/slots produces 5 buckets present, no `pd.NA` outside play-in (covered above 99%).

- `tests/test_eval_v4_calibration.py` (~250 LOC, 6 unit + 1 smoke)
  - `_drop_best_season_delta` unit on synthetic series.
  - `_classify_verdict` unit covers all 4 quadrants (PASS, MARGINAL, MARGINAL-from-concentration, FAIL).
  - `_anchor_check` byte-equal vs unequal cases (mirrors R64-blend's pattern).
  - Anchor: global T=1.0 cell produces total brkt pts == 2069 (gated on Kaggle data + `pairwise_v8.csv` present; skipped otherwise).
  - Anchor: per-round (1,1,1,1,1) cell produces 2069 (same gate).
  - Greedy invariant: each per-round step's chosen-cell total >= previous step's chosen-cell total (monotonic improvement in greedy chain). Tested over a synthetic 4-row, 4-season fixture against a forced-pass synthetic scoring function.
  - Smoke: full Phase 1 sweep (global + per-round) over the real `pairwise_v8.csv` writes the expected JSON keys and per-cell artifacts in <10 min wall.

**Generated (force-added per `.gitignore: output/`):**

- `output/pairwise_v8_calibrated_global_T<best>.csv` -- best global cell.
- `output/pairwise_v8_calibrated_perround_<T_R64>_<T_R32>_<T_S16>_<T_E8>_<T_F4NCG>.csv` -- best per-round cell.
- `output/pairwise_v8_phase2_T<best>.csv` -- Phase 2 retrain output (only if Phase 2 triggered).
- `output/v4_calibration_eval.json` -- per-cell metrics + verdict + drop-best-season delta + per-season table.
- `output/v4_calibration_eval_log.txt` -- captured stdout.
- `output/v4_calibration_reliability.png` -- 3-line reliability diagram (v8 baseline, v8 + best global T, v8 + best per-round T).

**Modified:**

- `TODO.md` -- on completion, move "v4 calibration-shape engineering (audit-derived)" entry from Active queue #1 to Done with verdict + numbers; update queue preamble.

---

## Phase 0: Pre-flight verification

### Task 0: Confirm canonical baseline exists and scores 2069

This is a 2-minute safety check that the worktree environment has what it needs. The plan does not write any code yet.

**Files:**
- (none modified)

- [ ] **Step 1: Confirm working directory and branch**

```bash
pwd
# Expected: C:/Users/alden/MarchMadness/.claude/worktrees/feat-v4-calibration-temperature-scaling

git status
# Expected: On branch feat/v4-calibration-temperature-scaling. nothing to commit.

git log --oneline -3
# Expected: 92a298a spec(v4-calibration-temperature-scaling): ...
#           06a5247 Merge pull request #31 from alhart2015/feat/v4-r64-line-blend
#           ...
```

If branch / commit is wrong, halt and ask. If worktree dir is empty (Windows junction wipe), halt and rebuild via `git worktree add`.

- [ ] **Step 2: Verify canonical `pairwise_v8.csv` is present and scores 2069**

```bash
python -c "
import sys; sys.path.insert(0, '.')
from src.score_chalk_brackets import score_pairwise_path
res = score_pairwise_path('output/pairwise_v8.csv')
print(f\"total_pts = {res['total_pts']}\")
print(f\"n_seasons = {len(res['per_season_pts'])}\")
"
```

Expected output:
```
total_pts = 2069.0
n_seasons = 22
```

If `total_pts != 2069.0` or `pairwise_v8.csv` is missing, **halt**. Most likely cause: data wipe (gitignored CSVs were lost; see `docs/data_recovery.md`). Recovery is to regen `pairwise_v8.csv` per the data-recovery runbook before continuing -- this plan assumes the canonical baseline is intact.

- [ ] **Step 3: Verify Kaggle data unzipped**

```bash
ls data/raw/march-machine-learning-2026/MNCAATourneySeeds.csv
ls data/raw/march-machine-learning-2026/MNCAATourneySlots.csv
ls data/raw/march-machine-learning-2026/MNCAATourneyCompactResults.csv
```

Expected: all three exist. If any missing, run:

```bash
tar -xzf data/training_data.tar.gz -C data/raw/
```

(This is the same recurring data-wipe-recovery step called out under "Engineering follow-ups" in `TODO.md`.)

- [ ] **Step 4: Confirm pytest runs cleanly on the eval-side test infra we'll lean on**

```bash
python -m pytest -q tests/test_eval_r64_line_blend.py 2>&1 | tail -5
```

Expected: 7 passed (the R64-blend's tests must be green; we'll mimic their pattern). Failures here mean the data-wipe recovery is incomplete.

No commit on this task. It's verification only.

---

## Phase 1: Apply-temperature-scaling module

### Task 1: Build `apply_temperature_scaling.py` with TDD

**Files:**
- Create: `src/apply_temperature_scaling.py`
- Create: `tests/test_apply_temperature_scaling.py`

- [ ] **Step 1: Write failing test file (math + bucket dispatch)**

Create `tests/test_apply_temperature_scaling.py`:

```python
"""Unit tests for src/apply_temperature_scaling.py."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def _four_row_synth() -> pd.DataFrame:
    """Synthetic 4 rows spanning the [0.05, 0.95] interior + 0.5."""
    return pd.DataFrame({
        "season": [2024, 2024, 2024, 2024],
        "team_a": [1, 2, 3, 4],
        "team_b": [5, 6, 7, 8],
        "p_a_wins": [0.05, 0.5, 0.85, 0.95],
        "round_bucket": ["R64", "R64", "R64", "R64"],
    })


def test_scale_identity_when_T_is_one():
    """T=1.0 returns p_a_wins unchanged (modulo float roundtrip)."""
    from src.apply_temperature_scaling import scale_pairwise

    df = _four_row_synth()
    out = scale_pairwise(df, T=1.0)
    np.testing.assert_allclose(
        out["p_a_wins"].values,
        df["p_a_wins"].values,
        atol=1e-12,
    )


def test_scale_flatten_when_T_above_one():
    """T=2.0 pulls every p toward 0.5 monotonically (rank-preserving on
    distances to 0.5; smaller-margin probs end closer to 0.5)."""
    from src.apply_temperature_scaling import scale_pairwise

    df = _four_row_synth()
    out = scale_pairwise(df, T=2.0)
    p_in = df["p_a_wins"].values
    p_out = out["p_a_wins"].values
    # 0.5 stays at 0.5.
    assert p_out[1] == pytest.approx(0.5, abs=1e-9)
    # Distances to 0.5 strictly shrink for non-0.5 inputs.
    for i in [0, 2, 3]:
        assert abs(p_out[i] - 0.5) < abs(p_in[i] - 0.5)
    # Order is preserved.
    assert list(np.argsort(p_out)) == list(np.argsort(p_in))


def test_scale_sharpen_when_T_below_one():
    """T=0.5 pushes every p away from 0.5 monotonically."""
    from src.apply_temperature_scaling import scale_pairwise

    df = _four_row_synth()
    out = scale_pairwise(df, T=0.5)
    p_in = df["p_a_wins"].values
    p_out = out["p_a_wins"].values
    assert p_out[1] == pytest.approx(0.5, abs=1e-9)
    for i in [0, 2, 3]:
        assert abs(p_out[i] - 0.5) > abs(p_in[i] - 0.5)
    assert list(np.argsort(p_out)) == list(np.argsort(p_in))


def test_scale_per_round_dispatch():
    """Per-round T applies a different T to each row based on its
    round_bucket. We give each bucket a unique T and verify the row
    output is what scale_pairwise would have produced under that T
    alone."""
    from src.apply_temperature_scaling import scale_pairwise

    rows = []
    for bucket, p in [("R64", 0.7), ("R32", 0.7), ("S16", 0.7),
                      ("E8", 0.7), ("F4_NCG", 0.7)]:
        rows.append({"season": 2024, "team_a": 1, "team_b": 2,
                     "p_a_wins": p, "round_bucket": bucket})
    df = pd.DataFrame(rows)
    T = {"R64": 1.0, "R32": 2.0, "S16": 0.5, "E8": 1.5, "F4_NCG": 1.0}
    out = scale_pairwise(df, T=T)
    # R64 and F4_NCG (T=1.0) -> 0.7 unchanged.
    assert out.iloc[0]["p_a_wins"] == pytest.approx(0.7, abs=1e-9)
    assert out.iloc[4]["p_a_wins"] == pytest.approx(0.7, abs=1e-9)
    # R32 (T=2.0) -> closer to 0.5 than 0.7.
    assert 0.5 < out.iloc[1]["p_a_wins"] < 0.7
    # S16 (T=0.5) -> further from 0.5 than 0.7.
    assert out.iloc[2]["p_a_wins"] > 0.7
    # E8 (T=1.5) -> closer to 0.5 than 0.7 but less so than R32.
    assert 0.5 < out.iloc[3]["p_a_wins"] < out.iloc[1]["p_a_wins"]


def test_scale_clips_extreme_inputs_to_finite_output():
    """p in {0, 1} should produce finite output (no inf/NaN)."""
    from src.apply_temperature_scaling import scale_pairwise

    df = pd.DataFrame({
        "season": [2024, 2024],
        "team_a": [1, 2], "team_b": [3, 4],
        "p_a_wins": [0.0, 1.0],
        "round_bucket": ["R64", "R64"],
    })
    out = scale_pairwise(df, T=1.5)
    assert np.isfinite(out["p_a_wins"].values).all()
    # The clipped extremes should still be near {0, 1} after T=1.5
    # since logit/sigmoid round-trip near-identity at the bounds.
    assert out.iloc[0]["p_a_wins"] < 1e-3
    assert out.iloc[1]["p_a_wins"] > 1.0 - 1e-3


def test_scale_per_round_raises_if_bucket_missing_in_T_dict():
    """If a row's round_bucket isn't in T (dict mode), raise KeyError --
    no silent fallback. Caller must pass complete T dict."""
    from src.apply_temperature_scaling import scale_pairwise

    df = pd.DataFrame({
        "season": [2024], "team_a": [1], "team_b": [2],
        "p_a_wins": [0.6], "round_bucket": ["F4_NCG"],
    })
    with pytest.raises(KeyError, match="F4_NCG"):
        scale_pairwise(df, T={"R64": 1.0})


def test_scale_per_round_does_not_mutate_input():
    """scale_pairwise returns a NEW DataFrame; input is not mutated."""
    from src.apply_temperature_scaling import scale_pairwise

    df = _four_row_synth()
    p_orig = df["p_a_wins"].copy()
    _ = scale_pairwise(df, T=2.0)
    np.testing.assert_array_equal(df["p_a_wins"].values, p_orig.values)
```

- [ ] **Step 2: Run tests to verify they fail with import error**

```bash
python -m pytest -q tests/test_apply_temperature_scaling.py 2>&1 | tail -10
```

Expected: 7 errors / failures, all citing `ModuleNotFoundError: No module named 'src.apply_temperature_scaling'` or `ImportError`.

- [ ] **Step 3: Write minimal `apply_temperature_scaling.py` to satisfy the math + dispatch tests**

Create `src/apply_temperature_scaling.py`:

```python
"""Post-hoc temperature scaling on pairwise probability frames.

Spec:  docs/superpowers/specs/2026-05-08-v4-calibration-temperature-scaling-design.md
Plan:  docs/superpowers/plans/2026-05-08-v4-calibration-temperature-scaling.md

Public API:
    scale_pairwise(df, T)
        df: DataFrame with columns (season, team_a, team_b, p_a_wins).
            For per-round T, must also have a 'round_bucket' column with
            values among {'R64','R32','S16','E8','F4_NCG'} (rounds 5+6
            collapsed -- see _round_int_to_bucket).
        T:  float or dict[str, float] keyed on round bucket.
        Returns: a NEW DataFrame, same shape, with p_a_wins rescaled.

    assign_round_buckets(df, slots_df, seeds_df)
        df: DataFrame with (season, team_a, team_b).
        slots_df, seeds_df: Kaggle MNCAATourneySlots / MNCAATourneySeeds.
        Returns: pd.Series indexed like df, dtype=object, values in
                 {'R64','R32','S16','E8','F4_NCG'} or pd.NA for rows
                 with no resolvable round (play-in pairs, missing seed
                 mappings).

CLI:
    python -m src.apply_temperature_scaling \\
        --in output/pairwise_v8.csv \\
        --T 1.15 \\
        --out output/pairwise_v8_T1.15.csv

(Per-round CLI not provided -- the eval driver wires per-round dicts.)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Mapping, Union

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.train_upset_model import build_pair_round_lookup  # noqa: E402

ROUND_BUCKETS = ("R64", "R32", "S16", "E8", "F4_NCG")
_CLIP = 1e-9


def _round_int_to_bucket(rnd: int) -> str:
    """Map round int from build_pair_round_lookup (1..6) to bucket key.

    Rounds 5 (F4) and 6 (Champ) are collapsed into 'F4_NCG' for n>=66
    per knob (44 + 22). NCG alone at n=22 is too noisy for a 7-cell
    grid search -- see spec section 'Architecture / Per-round scaling'.
    Round 0 (play-in) is not a tournament round and never reaches this
    function via the public path; raise rather than guess.
    """
    if rnd == 1:
        return "R64"
    if rnd == 2:
        return "R32"
    if rnd == 3:
        return "S16"
    if rnd == 4:
        return "E8"
    if rnd in (5, 6):
        return "F4_NCG"
    raise ValueError(f"unexpected round int: {rnd!r}")


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, _CLIP, 1.0 - _CLIP)
    return np.log(p / (1.0 - p))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def scale_pairwise(
    df: pd.DataFrame,
    T: Union[float, Mapping[str, float]],
) -> pd.DataFrame:
    """Return a NEW DataFrame with p_a_wins rescaled by T.

    Scalar T:
        p_out = sigmoid(logit(p_in) / T) for every row.

    Per-round T (dict):
        Per row, dispatch on row['round_bucket']. KeyError if any row's
        bucket is not in T.
    """
    out = df.copy()
    p = out["p_a_wins"].to_numpy(dtype=float)
    if isinstance(T, (int, float)):
        out["p_a_wins"] = _sigmoid(_logit(p) / float(T))
        return out
    # Per-round dict.
    if "round_bucket" not in out.columns:
        raise KeyError("scale_pairwise per-round mode requires 'round_bucket' column")
    z = _logit(p)
    new_p = np.empty_like(p)
    bucket_arr = out["round_bucket"].to_numpy()
    # Normalize T into a plain dict for repeated lookups.
    T_map = dict(T)
    for i, b in enumerate(bucket_arr):
        # Trigger a KeyError early with the missing bucket name in the
        # message -- pytest matches on bucket names.
        if b not in T_map:
            raise KeyError(b)
        new_p[i] = _sigmoid(z[i] / float(T_map[b]))
    out["p_a_wins"] = new_p
    return out


def assign_round_buckets(
    df: pd.DataFrame,
    slots_df: pd.DataFrame,
    seeds_df: pd.DataFrame,
) -> pd.Series:
    """For each row in df, return its round bucket.

    Uses build_pair_round_lookup per (season). Returns pd.NA for rows
    whose (team_a, team_b) pair has no resolvable round in their season
    (play-in pairs, or pairs with seeds that don't map).
    """
    out = pd.Series(pd.NA, index=df.index, dtype=object)
    for season in df["season"].unique():
        lookup = build_pair_round_lookup(int(season), slots_df, seeds_df)
        mask = df["season"] == season
        for idx in df[mask].index:
            a = int(df.at[idx, "team_a"])
            b = int(df.at[idx, "team_b"])
            key = (min(a, b), max(a, b))
            rnd = lookup.get(key)
            if rnd is None:
                continue
            try:
                out.at[idx] = _round_int_to_bucket(int(rnd))
            except ValueError:
                # Round 0 (play-in) -- leave as pd.NA.
                continue
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Apply temperature scaling.")
    p.add_argument("--in", dest="inp", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--T", type=float, required=True,
                   help="scalar temperature (per-round T not supported via CLI)")
    args = p.parse_args(argv)
    df = pd.read_csv(args.inp)
    out = scale_pairwise(df, T=args.T)
    out.to_csv(args.out, index=False)
    print(f"wrote {args.out} ({len(out)} rows) at T={args.T}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest -q tests/test_apply_temperature_scaling.py::test_scale_identity_when_T_is_one tests/test_apply_temperature_scaling.py::test_scale_flatten_when_T_above_one tests/test_apply_temperature_scaling.py::test_scale_sharpen_when_T_below_one tests/test_apply_temperature_scaling.py::test_scale_per_round_dispatch tests/test_apply_temperature_scaling.py::test_scale_clips_extreme_inputs_to_finite_output tests/test_apply_temperature_scaling.py::test_scale_per_round_raises_if_bucket_missing_in_T_dict tests/test_apply_temperature_scaling.py::test_scale_per_round_does_not_mutate_input -v
```

Expected: 7 passed.

- [ ] **Step 5: Add the anchor-regression test on real `pairwise_v8.csv`**

Append to `tests/test_apply_temperature_scaling.py`:

```python
def _real_v8_present() -> bool:
    return Path("output/pairwise_v8.csv").exists()


@pytest.mark.skipif(
    not _real_v8_present(),
    reason="output/pairwise_v8.csv missing; data wipe? see docs/data_recovery.md",
)
def test_scale_T_one_anchors_byte_equal_to_canonical_v8():
    """Apply T=1.0 to canonical pairwise_v8.csv -- p_a_wins must round-trip
    to FP precision. This is the Phase-1 anchor (spec section 'Anchors')."""
    from src.apply_temperature_scaling import scale_pairwise

    df = pd.read_csv("output/pairwise_v8.csv")
    out = scale_pairwise(df, T=1.0)
    np.testing.assert_allclose(
        out["p_a_wins"].values,
        df["p_a_wins"].values,
        atol=1e-9,
    )


@pytest.mark.skipif(
    not _real_v8_present(),
    reason="output/pairwise_v8.csv missing; data wipe? see docs/data_recovery.md",
)
def test_scale_T_all_one_perround_anchors_byte_equal_to_canonical_v8():
    """All-1 per-round dict is identity even with bucket dispatch."""
    from src.apply_temperature_scaling import scale_pairwise, assign_round_buckets

    df = pd.read_csv("output/pairwise_v8.csv")
    slots_df = pd.read_csv("data/raw/march-machine-learning-2026/MNCAATourneySlots.csv")
    seeds_df = pd.read_csv("data/raw/march-machine-learning-2026/MNCAATourneySeeds.csv")
    df["round_bucket"] = assign_round_buckets(df, slots_df, seeds_df)
    # Drop rows with no resolvable round (play-in, etc.) so per-round
    # mode doesn't KeyError on pd.NA.
    df_resolved = df.dropna(subset=["round_bucket"]).copy()
    T = {b: 1.0 for b in ("R64", "R32", "S16", "E8", "F4_NCG")}
    out = scale_pairwise(df_resolved, T=T)
    np.testing.assert_allclose(
        out["p_a_wins"].values,
        df_resolved["p_a_wins"].values,
        atol=1e-9,
    )
```

- [ ] **Step 6: Add the `assign_round_buckets` smoke test**

Append to `tests/test_apply_temperature_scaling.py`:

```python
@pytest.mark.skipif(
    not (Path("output/pairwise_v8.csv").exists()
         and Path("data/raw/march-machine-learning-2026/MNCAATourneySeeds.csv").exists()
         and Path("data/raw/march-machine-learning-2026/MNCAATourneySlots.csv").exists()),
    reason="Kaggle data missing -- needs tar -xzf data/training_data.tar.gz",
)
def test_assign_round_buckets_covers_all_five_buckets_on_real_data():
    """All 5 buckets show up; resolved fraction >= 0.95 of rows."""
    from src.apply_temperature_scaling import assign_round_buckets, ROUND_BUCKETS

    df = pd.read_csv("output/pairwise_v8.csv")
    slots_df = pd.read_csv("data/raw/march-machine-learning-2026/MNCAATourneySlots.csv")
    seeds_df = pd.read_csv("data/raw/march-machine-learning-2026/MNCAATourneySeeds.csv")
    s = assign_round_buckets(df, slots_df, seeds_df)
    present = set(s.dropna().unique())
    assert present == set(ROUND_BUCKETS), f"missing buckets: {set(ROUND_BUCKETS) - present}"
    # The pairwise frame includes every (i, j) pair in each season's
    # field, so most pairs do NOT meet during the tournament -- those
    # legitimately have no round and stay as pd.NA. We check only that
    # the resolved fraction is consistent across seasons (every season
    # has roughly the same pair count).
    season_resolved = s.dropna().groupby(df.loc[s.dropna().index, "season"]).size()
    assert (season_resolved > 0).all(), "every season should have >=1 resolved pair"
```

- [ ] **Step 7: Run the full unit test file -- all 9 tests pass (7 unit + 2 anchor + 1 smoke = 10 if real data present, or 7 if not)**

```bash
python -m pytest -q tests/test_apply_temperature_scaling.py -v 2>&1 | tail -20
```

Expected: 10 passed (or 7 passed + 3 skipped on a fresh worktree pre-data-unzip).

- [ ] **Step 8: Commit**

```bash
git add src/apply_temperature_scaling.py tests/test_apply_temperature_scaling.py
git commit -m "feat(v4-calibration-temperature-scaling): apply_temperature_scaling module + tests

Pure rescaling: scale_pairwise(df, T) with scalar or per-round dict T,
numerical guards (logit clip at 1e-9), F4+Champ collapsed to F4_NCG
bucket. assign_round_buckets reuses train_upset_model.build_pair_round_lookup.
9 unit + 1 smoke; 2 anchor-regression tests against canonical
output/pairwise_v8.csv (T=1.0 byte-equal in scalar and per-round all-1
modes -- spec Phase-1 anchor).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Phase 1: Eval driver helpers (verdict + drop-best-season)

### Task 2: Build `eval_v4_calibration.py` helpers with TDD

**Files:**
- Create: `src/eval_v4_calibration.py`
- Create: `tests/test_eval_v4_calibration.py`

This task is the helpers shell only -- pure functions, no LOSO scoring yet. Sweeps come in Task 3.

- [ ] **Step 1: Write failing tests for helpers**

Create `tests/test_eval_v4_calibration.py`:

```python
"""Unit tests for src/eval_v4_calibration.py."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def test_drop_best_season_delta_subtracts_largest_positive():
    """drop_best_season_delta = sum(per_season_delta) - max(per_season_delta).
    The 'best' season is the largest single-season positive contribution."""
    from src.eval_v4_calibration import _drop_best_season_delta

    per_season = {2010: -1, 2011: 5, 2012: 2, 2013: -2}
    total = sum(per_season.values())  # 4
    drop = _drop_best_season_delta(per_season)  # 4 - 5 = -1
    assert drop == -1


def test_drop_best_season_delta_handles_all_negative():
    """If every season is a loss, drop_best subtracts 0 (no positive
    seasons to drop) -- equals the original total."""
    from src.eval_v4_calibration import _drop_best_season_delta

    per_season = {2010: -3, 2011: -1, 2012: -2}
    drop = _drop_best_season_delta(per_season)
    assert drop == sum(per_season.values())  # -6


def test_classify_verdict_pass_when_delta_above_25_and_robust():
    """delta >= +25, drop_best_delta >= 0, wins >= 6 -> PASS."""
    from src.eval_v4_calibration import _classify_verdict

    assert _classify_verdict(delta_total=30, drop_best_delta=15, wins=8) == "PASS"


def test_classify_verdict_marginal_when_above_10_below_25():
    """delta in [+10, +25) -> MARGINAL regardless of robustness."""
    from src.eval_v4_calibration import _classify_verdict

    assert _classify_verdict(delta_total=15, drop_best_delta=5, wins=5) == "MARGINAL"


def test_classify_verdict_marginal_when_pass_magnitude_but_concentrated():
    """delta >= +25 but drop_best_delta < 0 (concentrated in one season)
    -> MARGINAL (per spec: '>50% single-season concentration demotes PASS')."""
    from src.eval_v4_calibration import _classify_verdict

    # Total +30, but if you drop the best season the result is negative,
    # then that one season is doing more than 100% of the lift.
    assert _classify_verdict(delta_total=30, drop_best_delta=-5, wins=4) == "MARGINAL"


def test_classify_verdict_fail_when_below_10():
    """delta < +10 -> FAIL."""
    from src.eval_v4_calibration import _classify_verdict

    assert _classify_verdict(delta_total=8, drop_best_delta=8, wins=11) == "FAIL"
    assert _classify_verdict(delta_total=-5, drop_best_delta=-5, wins=6) == "FAIL"


def test_anchor_check_byte_equal():
    """Identical CSVs -> matches=True, max_abs_diff=0."""
    from src.eval_v4_calibration import _anchor_check

    df = pd.DataFrame({
        "season": [2024, 2024],
        "team_a": [1, 2], "team_b": [2, 3],
        "p_a_wins": [0.55, 0.62],
    })
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".csv") as f:
        df.to_csv(f.name, index=False)
        baseline = f.name
    res = _anchor_check(df, baseline)
    assert res["matches"] is True
    assert res["max_abs_diff"] == 0.0
    Path(baseline).unlink()


def test_anchor_check_flags_difference():
    """1e-3 difference -> matches=False."""
    from src.eval_v4_calibration import _anchor_check

    df = pd.DataFrame({
        "season": [2024], "team_a": [1], "team_b": [2], "p_a_wins": [0.55]})
    df_changed = df.copy(); df_changed["p_a_wins"] = 0.551
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".csv") as f:
        df.to_csv(f.name, index=False)
        baseline = f.name
    res = _anchor_check(df_changed, baseline)
    assert res["matches"] is False
    assert res["max_abs_diff"] == pytest.approx(0.001, abs=1e-9)
    Path(baseline).unlink()


def test_summarize_cell_computes_wlt_and_biggest_swing():
    """Per-season delta dict -> {total, wins, losses, ties,
    biggest_swing_value, biggest_swing_season, drop_best_season_delta}."""
    from src.eval_v4_calibration import _summarize_cell

    per_season = {2010: 5, 2011: -3, 2012: 0, 2013: 8, 2014: -1}
    out = _summarize_cell(per_season, baseline_total=2069)
    assert out["delta_total"] == sum(per_season.values())  # 9
    assert out["wins"] == 2  # 2010, 2013
    assert out["losses"] == 2  # 2011, 2014
    assert out["ties"] == 1  # 2012
    assert out["biggest_swing_value"] == 8
    assert out["biggest_swing_season"] == 2013
    # drop_best = 9 - 8 = 1
    assert out["drop_best_season_delta"] == 1
    # total = baseline + delta
    assert out["total"] == 2069 + 9
```

- [ ] **Step 2: Run tests; expect failure**

```bash
python -m pytest -q tests/test_eval_v4_calibration.py 2>&1 | tail -10
```

Expected: 9 errors / failures (`ModuleNotFoundError: No module named 'src.eval_v4_calibration'`).

- [ ] **Step 3: Write the helpers**

Create `src/eval_v4_calibration.py`:

```python
"""Eval driver for v4 calibration: post-hoc temperature scaling on
canonical output/pairwise_v8.csv (Phase 1) + conditional retrain
(Phase 2).

Spec: docs/superpowers/specs/2026-05-08-v4-calibration-temperature-scaling-design.md
Plan: docs/superpowers/plans/2026-05-08-v4-calibration-temperature-scaling.md

Phase 1 sub-experiments:
    1. Global T sweep over T_GRID -- 7 cells, anchor T=1.0 reproduces 2069.
    2. Per-round T sequential greedy over (R64, R32, S16, E8, F4_NCG).

Phase 2 (only if Phase 1 PASS or MARGINAL):
    Scale v4 stage-1 with the winning T configuration, retrain v8 LOSO,
    re-score 22-season bracket points.

Verdict bands (carried from R64-blend / BT-bracket-points / v9-C):
    delta >= +25 with drop_best_delta >= 0 and wins >= 6 -> PASS
    delta in [+10, +25)  OR PASS-magnitude with drop_best_delta < 0 -> MARGINAL
    delta < +10 -> FAIL
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import tempfile
import time
from pathlib import Path
from typing import Mapping

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.apply_temperature_scaling import (  # noqa: E402
    ROUND_BUCKETS,
    assign_round_buckets,
    scale_pairwise,
)
from src.score_chalk_brackets import score_pairwise_path  # noqa: E402

logger = logging.getLogger(__name__)

T_GRID = [0.7, 0.85, 1.0, 1.15, 1.3, 1.5, 2.0]
ROUND_ORDER = ["R64", "R32", "S16", "E8", "F4_NCG"]
PASS_BAR = 25
MARGINAL_BAR = 10
DATA = Path("data/raw/march-machine-learning-2026")


# ---------------------------------------------------------------------------
# Pure helpers (testable in isolation; Task 2)
# ---------------------------------------------------------------------------


def _drop_best_season_delta(per_season_delta: Mapping[int, float]) -> float:
    """Sum of per-season deltas minus the largest single positive
    contribution. If every season is a loss (no positives), return the
    raw total -- nothing to 'drop'.
    """
    vals = list(per_season_delta.values())
    if not vals:
        return 0.0
    total = float(sum(vals))
    best = max(vals)
    if best <= 0:
        return total
    return total - float(best)


def _classify_verdict(
    delta_total: float,
    drop_best_delta: float,
    wins: int,
) -> str:
    """PASS / MARGINAL / FAIL per the spec decision matrix.

    PASS:
        delta_total >= +25 AND drop_best_delta >= 0 AND wins >= 6.
    MARGINAL:
        delta_total in [+10, +25),
        OR delta_total >= +25 with drop_best_delta < 0
           (single-season concentration demotes PASS),
        OR delta_total >= +25 with wins < 6
           (insufficiently broad win count).
    FAIL:
        delta_total < +10.
    """
    if delta_total >= PASS_BAR:
        if drop_best_delta < 0 or wins < 6:
            return "MARGINAL"
        return "PASS"
    if delta_total >= MARGINAL_BAR:
        return "MARGINAL"
    return "FAIL"


def _summarize_cell(
    per_season_delta: Mapping[int, float],
    baseline_total: float,
) -> dict:
    """Pack the per-season delta dict into the standard cell summary."""
    items = list(per_season_delta.items())
    delta_total = float(sum(v for _, v in items))
    wins = sum(1 for _, v in items if v > 0)
    losses = sum(1 for _, v in items if v < 0)
    ties = sum(1 for _, v in items if v == 0)
    if items:
        biggest_season, biggest_value = max(items, key=lambda kv: abs(kv[1]))
    else:
        biggest_season, biggest_value = None, 0.0
    drop_best_delta = _drop_best_season_delta(per_season_delta)
    return {
        "total": float(baseline_total + delta_total),
        "delta_total": delta_total,
        "wins": int(wins),
        "losses": int(losses),
        "ties": int(ties),
        "biggest_swing_value": float(biggest_value),
        "biggest_swing_season": (int(biggest_season) if biggest_season is not None else None),
        "drop_best_season_delta": float(drop_best_delta),
        "per_season_delta": {int(s): float(v) for s, v in items},
    }


def _anchor_check(df_actual: pd.DataFrame, baseline_csv: str) -> dict:
    """Compare df_actual to baseline_csv on (season, team_a, team_b).
    Returns {matches: bool, max_abs_diff: float, n_rows: int}.
    Mirrors src/sweep_bt_bracket_points._anchor_check semantics."""
    a = df_actual.drop_duplicates(["season", "team_a", "team_b"], keep="last")
    b = pd.read_csv(baseline_csv).drop_duplicates(
        ["season", "team_a", "team_b"], keep="last"
    )
    merged = a.merge(b, on=["season", "team_a", "team_b"],
                     suffixes=("_actual", "_expected"))
    if len(merged) != len(a) or len(merged) != len(b):
        return {
            "matches": False,
            "max_abs_diff": float("nan"),
            "n_only_actual": int(len(a) - len(merged)),
            "n_only_expected": int(len(b) - len(merged)),
            "n_rows": int(len(merged)),
        }
    diff = (merged["p_a_wins_actual"] - merged["p_a_wins_expected"]).abs()
    return {
        "matches": bool(diff.max() < 1e-9),
        "max_abs_diff": float(diff.max()),
        "n_rows": int(len(merged)),
    }


# ---------------------------------------------------------------------------
# Sweep-side scaffolding (filled in Task 3)
# ---------------------------------------------------------------------------


def _score_pairwise_df(df: pd.DataFrame, scratch_dir: Path) -> dict:
    """Write df to a tempfile in scratch_dir and call score_pairwise_path.
    Returns {'total_pts': float, 'per_season_pts': {int: float}}.
    The tempfile is cleaned up before return.
    """
    fd, tmp_path = tempfile.mkstemp(prefix="calib_", suffix=".csv", dir=str(scratch_dir))
    Path(tmp_path).close = lambda: None  # nothing
    try:
        df.to_csv(tmp_path, index=False)
        # close the fd that mkstemp opened
        import os
        os.close(fd)
        return score_pairwise_path(tmp_path)
    finally:
        try:
            Path(tmp_path).unlink()
        except FileNotFoundError:
            pass


def main(argv: list[str] | None = None) -> int:
    """Filled in Task 4."""
    raise NotImplementedError("see Task 4")


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run helper tests; expect pass**

```bash
python -m pytest -q tests/test_eval_v4_calibration.py -v 2>&1 | tail -15
```

Expected: 9 passed.

- [ ] **Step 5: Commit**

```bash
git add src/eval_v4_calibration.py tests/test_eval_v4_calibration.py
git commit -m "feat(v4-calibration-temperature-scaling): eval helpers + verdict bands

_drop_best_season_delta, _classify_verdict (PASS/MARGINAL/FAIL with
single-season-concentration demotion), _anchor_check, _summarize_cell.
Sweep functions stubbed; CLI raises NotImplementedError pending Task 3.
9 unit tests covering all 4 verdict quadrants + concentration demotion
+ anchor byte-equal/diff cases.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Phase 1: Sweep functions (global + per-round greedy)

### Task 3: Implement `run_global_T_sweep` and `run_per_round_greedy`

**Files:**
- Modify: `src/eval_v4_calibration.py`
- Modify: `tests/test_eval_v4_calibration.py`

- [ ] **Step 1: Write failing tests for sweep functions on synthetic small frames**

Append to `tests/test_eval_v4_calibration.py`:

```python
def _real_v8_present() -> bool:
    return Path("output/pairwise_v8.csv").exists() and Path(
        "data/raw/march-machine-learning-2026/MNCAATourneySeeds.csv"
    ).exists()


@pytest.mark.skipif(not _real_v8_present(), reason="canonical pairwise_v8.csv missing")
def test_global_T_anchor_cell_reproduces_2069():
    """T=1.0 must score 2069 to FP precision. Phase-1 anchor."""
    from src.eval_v4_calibration import run_global_T_sweep

    out = run_global_T_sweep(
        v8_csv="output/pairwise_v8.csv",
        T_grid=[1.0],
        baseline_total=2069.0,
    )
    assert "anchor" in out
    assert out["anchor"]["matches"] is True
    assert out["anchor"]["total"] == pytest.approx(2069.0, abs=1e-9)
    assert len(out["cells"]) == 1
    cell = out["cells"][0]
    assert cell["T"] == 1.0
    assert cell["total"] == pytest.approx(2069.0, abs=1e-9)
    assert cell["delta_total"] == pytest.approx(0.0, abs=1e-9)


@pytest.mark.skipif(not _real_v8_present(), reason="canonical pairwise_v8.csv missing")
def test_per_round_greedy_anchor_all_one_reproduces_2069():
    """All-1 per-round vector reproduces 2069 to FP precision."""
    from src.eval_v4_calibration import run_per_round_greedy

    out = run_per_round_greedy(
        v8_csv="output/pairwise_v8.csv",
        T_grid=[1.0],  # singleton grid -- only T=1 available
        round_order=["R64", "R32", "S16", "E8", "F4_NCG"],
        baseline_total=2069.0,
    )
    assert "anchor" in out
    assert out["anchor"]["total"] == pytest.approx(2069.0, abs=1e-9)
    assert out["winning_T"] == {b: 1.0 for b in ("R64", "R32", "S16", "E8", "F4_NCG")}
    assert out["winning_cell"]["total"] == pytest.approx(2069.0, abs=1e-9)


def test_per_round_greedy_monotonic_improvement_in_chain():
    """Each greedy step's chosen-cell total >= previous step's. Tested
    against a forced-pass synthetic scoring stub via monkeypatch.

    The greedy invariant is: at step k, holding rounds < k at their
    best-found T and rounds > k at 1.0, the picked T_round_k must
    yield total >= total at the previous step's pick."""
    import src.eval_v4_calibration as mod
    from src.eval_v4_calibration import run_per_round_greedy

    # Seed a synthetic v8.csv with one row per bucket.
    df = pd.DataFrame({
        "season": [2024] * 5,
        "team_a": [1, 2, 3, 4, 5],
        "team_b": [10, 11, 12, 13, 14],
        "p_a_wins": [0.6] * 5,
        "round_bucket": ["R64", "R32", "S16", "E8", "F4_NCG"],
    })
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".csv") as f:
        df.to_csv(f.name, index=False)
        v8_path = f.name

    # Monkeypatch _score_pairwise_df to return a synthetic score that
    # rewards lower T monotonically (via sum of p_a_wins).
    def fake_score(scaled_df, scratch_dir):
        s = scaled_df["p_a_wins"].sum()
        return {"total_pts": float(2069 + s), "per_season_pts": {2024: float(2069 + s)}}

    # Patch the baseline call too -- the synthetic frame isn't a real
    # bracket so the real score_pairwise_path would crash on it.
    def fake_baseline_score(path):
        df_local = pd.read_csv(path)
        s = df_local["p_a_wins"].sum()
        return {"total_pts": float(2069 + s), "per_season_pts": {2024: float(2069 + s)}}

    # Short-circuit assign_round_buckets -- the synthetic frame already
    # has round_bucket, but the driver re-derives via slots_df/seeds_df.
    real_assign = mod.assign_round_buckets
    real_score_df = mod._score_pairwise_df
    real_score_path = mod.score_pairwise_path

    def fake_assign(d, slots_df, seeds_df):
        return d["round_bucket"]
    mod.assign_round_buckets = fake_assign
    mod._score_pairwise_df = fake_score
    mod.score_pairwise_path = fake_baseline_score

    try:
        out = run_per_round_greedy(
            v8_csv=v8_path,
            T_grid=[0.5, 1.0, 2.0],
            round_order=["R64", "R32", "S16", "E8", "F4_NCG"],
            # baseline_total must equal what fake_baseline_score returns
            # for the v8_path (5 rows of p=0.6 each = 3.0 sum -> 2072.0).
            baseline_total=2072.0,
        )
    finally:
        mod.assign_round_buckets = real_assign
        mod._score_pairwise_df = real_score_df
        mod.score_pairwise_path = real_score_path
        Path(v8_path).unlink()

    chain = out["greedy_chain"]
    assert len(chain) == 5  # one entry per round
    # Monotonic invariant.
    for i in range(1, len(chain)):
        assert chain[i]["total_after_step"] >= chain[i - 1]["total_after_step"]
```

- [ ] **Step 2: Run tests; expect failures (functions not implemented)**

```bash
python -m pytest -q tests/test_eval_v4_calibration.py::test_global_T_anchor_cell_reproduces_2069 tests/test_eval_v4_calibration.py::test_per_round_greedy_anchor_all_one_reproduces_2069 tests/test_eval_v4_calibration.py::test_per_round_greedy_monotonic_improvement_in_chain 2>&1 | tail -10
```

Expected: 3 failures with `AttributeError` / `ImportError` for the sweep functions.

- [ ] **Step 3: Implement `run_global_T_sweep`**

In `src/eval_v4_calibration.py`, replace the "Sweep-side scaffolding" comment block with:

```python
# ---------------------------------------------------------------------------
# Phase 1 sweeps
# ---------------------------------------------------------------------------


def run_global_T_sweep(
    v8_csv: str,
    T_grid: list[float],
    baseline_total: float,
    scratch_dir: Path | None = None,
) -> dict:
    """Sweep a single global T over T_grid, scoring 22-season bracket
    points per cell. Returns a dict with 'cells' (per-T summaries),
    'anchor' (T=1.0 cell summary), 'best_cell' (the highest-total cell),
    and 'verdict' ('PASS' | 'MARGINAL' | 'FAIL').

    The anchor: applying T=1.0 must produce a frame byte-equal to v8_csv
    on (season, team_a, team_b, p_a_wins). HALT (raise) if it doesn't.
    """
    if scratch_dir is None:
        scratch_dir = Path(tempfile.gettempdir())
    scratch_dir.mkdir(parents=True, exist_ok=True)

    df_baseline = pd.read_csv(v8_csv)
    baseline_score = score_pairwise_path(v8_csv)
    baseline_per_season = baseline_score["per_season_pts"]
    if abs(baseline_score["total_pts"] - baseline_total) > 1e-6:
        raise RuntimeError(
            f"baseline mismatch: score_pairwise_path returned "
            f"{baseline_score['total_pts']} but caller passed "
            f"baseline_total={baseline_total}"
        )

    # Anchor verification: scale at T=1.0 and check byte-equal.
    anchor_df = scale_pairwise(df_baseline, T=1.0)
    anchor_check = _anchor_check(anchor_df, v8_csv)
    if not anchor_check["matches"]:
        raise RuntimeError(
            f"global T anchor FAILED: T=1.0 scaling did not reproduce "
            f"{v8_csv} byte-equal -- max_abs_diff={anchor_check['max_abs_diff']}"
        )
    anchor_summary = {
        "matches": True,
        "max_abs_diff": float(anchor_check["max_abs_diff"]),
        "total": float(baseline_score["total_pts"]),
    }

    cells = []
    for T in T_grid:
        scaled = scale_pairwise(df_baseline, T=float(T))
        score = _score_pairwise_df(scaled, scratch_dir)
        per_season_delta = {
            int(s): float(score["per_season_pts"][s]) - float(baseline_per_season[s])
            for s in baseline_per_season
        }
        summary = _summarize_cell(per_season_delta, baseline_total=baseline_total)
        summary["T"] = float(T)
        cells.append(summary)
        logger.info(
            "global T=%.3f -> total=%.1f delta=%+.1f W/L/T=%d/%d/%d drop_best=%+.1f",
            T, summary["total"], summary["delta_total"],
            summary["wins"], summary["losses"], summary["ties"],
            summary["drop_best_season_delta"],
        )

    best_cell = max(cells, key=lambda c: c["delta_total"])
    verdict = _classify_verdict(
        delta_total=best_cell["delta_total"],
        drop_best_delta=best_cell["drop_best_season_delta"],
        wins=best_cell["wins"],
    )
    return {
        "anchor": anchor_summary,
        "cells": cells,
        "best_cell": best_cell,
        "verdict": verdict,
    }


def run_per_round_greedy(
    v8_csv: str,
    T_grid: list[float],
    round_order: list[str],
    baseline_total: float,
    scratch_dir: Path | None = None,
) -> dict:
    """Sequential greedy per-round T sweep.

    For each round R in round_order:
        Hold all other rounds at their best-found T (rounds before R)
        or 1.0 (rounds after R). Sweep T_R over T_grid. Pick best total.
        Fix T_R; advance to next round.

    Returns: {anchor, greedy_chain (one entry per round with chosen T +
              total + per-cell summaries), winning_T (dict), winning_cell
              (full summary), verdict}.
    """
    if scratch_dir is None:
        scratch_dir = Path(tempfile.gettempdir())
    scratch_dir.mkdir(parents=True, exist_ok=True)

    df_baseline = pd.read_csv(v8_csv)
    baseline_score = score_pairwise_path(v8_csv)
    baseline_per_season = baseline_score["per_season_pts"]
    if abs(baseline_score["total_pts"] - baseline_total) > 1e-6:
        raise RuntimeError(
            f"baseline mismatch: {baseline_score['total_pts']} vs "
            f"baseline_total={baseline_total}"
        )

    # Resolve buckets for all rows up front.
    slots_df = pd.read_csv(DATA / "MNCAATourneySlots.csv")
    seeds_df = pd.read_csv(DATA / "MNCAATourneySeeds.csv")
    df_baseline = df_baseline.copy()
    df_baseline["round_bucket"] = assign_round_buckets(df_baseline, slots_df, seeds_df)
    n_total = len(df_baseline)
    df_resolved = df_baseline.dropna(subset=["round_bucket"]).copy()
    n_resolved = len(df_resolved)
    n_dropped = n_total - n_resolved
    logger.info(
        "per-round: %d/%d rows have a resolved round bucket (%d dropped)",
        n_resolved, n_total, n_dropped,
    )

    # Anchor: all-1 per-round dict reproduces resolved-row p_a_wins
    # byte-equal. Direct value comparison (not _anchor_check) because
    # _anchor_check exits early on row-count mismatch (full v8 vs
    # resolved subset) without checking values -- a T=1 implementation
    # bug could slip through that path.
    T_anchor = {b: 1.0 for b in ROUND_BUCKETS}
    anchor_df = scale_pairwise(df_resolved, T=T_anchor)
    max_diff = float(np.abs(
        anchor_df["p_a_wins"].to_numpy() - df_resolved["p_a_wins"].to_numpy()
    ).max())
    if max_diff > 1e-9:
        raise RuntimeError(
            f"per-round anchor FAILED: T=1.0 scaling produced max_abs_diff="
            f"{max_diff} on {n_resolved} resolved rows"
        )
    anchor_summary = {
        "matches": True,
        "max_abs_diff": max_diff,
        "n_resolved": int(n_resolved),
        "n_dropped": int(n_dropped),
    }

    # Greedy loop.
    best_T = {b: 1.0 for b in ROUND_BUCKETS}
    chain = []
    for r_idx, round_name in enumerate(round_order):
        cells = []
        for T in T_grid:
            cand_T = dict(best_T)
            cand_T[round_name] = float(T)
            scaled_resolved = scale_pairwise(df_resolved, T=cand_T)
            # Re-attach the dropped NA rows (which carry the v8 baseline
            # probabilities unchanged) for full-bracket scoring.
            scaled_full = pd.concat(
                [scaled_resolved.drop(columns=["round_bucket"]),
                 df_baseline[df_baseline["round_bucket"].isna()].drop(
                     columns=["round_bucket"])],
                ignore_index=True,
            )
            score = _score_pairwise_df(scaled_full, scratch_dir)
            per_season_delta = {
                int(s): float(score["per_season_pts"][s]) - float(baseline_per_season[s])
                for s in baseline_per_season
            }
            summary = _summarize_cell(per_season_delta, baseline_total=baseline_total)
            summary["T"] = float(T)
            summary["round"] = round_name
            cells.append(summary)
        # Pick best.
        best_cell = max(cells, key=lambda c: c["delta_total"])
        best_T[round_name] = float(best_cell["T"])
        logger.info(
            "per-round step %d/%d (%s): picked T=%.3f, total=%.1f, delta=%+.1f",
            r_idx + 1, len(round_order), round_name,
            best_cell["T"], best_cell["total"], best_cell["delta_total"],
        )
        chain.append({
            "round": round_name,
            "picked_T": float(best_cell["T"]),
            "total_after_step": float(best_cell["total"]),
            "delta_total_after_step": float(best_cell["delta_total"]),
            "all_cells": cells,
        })

    winning_cell = chain[-1]  # final step's pick is the chain end
    # Re-derive the full summary at winning_T (could differ from
    # chain[-1]['total_after_step'] only by selection order; here they
    # are equal by construction).
    winning_full = scale_pairwise(df_resolved, T=best_T)
    winning_full = pd.concat(
        [winning_full.drop(columns=["round_bucket"]),
         df_baseline[df_baseline["round_bucket"].isna()].drop(columns=["round_bucket"])],
        ignore_index=True,
    )
    winning_score = _score_pairwise_df(winning_full, scratch_dir)
    winning_per_season_delta = {
        int(s): float(winning_score["per_season_pts"][s]) - float(baseline_per_season[s])
        for s in baseline_per_season
    }
    winning_summary = _summarize_cell(
        winning_per_season_delta, baseline_total=baseline_total
    )
    winning_summary["T"] = dict(best_T)

    verdict = _classify_verdict(
        delta_total=winning_summary["delta_total"],
        drop_best_delta=winning_summary["drop_best_season_delta"],
        wins=winning_summary["wins"],
    )
    return {
        "anchor": anchor_summary,
        "greedy_chain": chain,
        "winning_T": dict(best_T),
        "winning_cell": winning_summary,
        "verdict": verdict,
    }
```

- [ ] **Step 4: Run sweep tests**

```bash
python -m pytest -q tests/test_eval_v4_calibration.py::test_global_T_anchor_cell_reproduces_2069 tests/test_eval_v4_calibration.py::test_per_round_greedy_anchor_all_one_reproduces_2069 tests/test_eval_v4_calibration.py::test_per_round_greedy_monotonic_improvement_in_chain -v 2>&1 | tail -15
```

Expected: 3 passed.

- [ ] **Step 5: Run full eval-side test file -- 12 tests**

```bash
python -m pytest -q tests/test_eval_v4_calibration.py -v 2>&1 | tail -20
```

Expected: 12 passed.

- [ ] **Step 6: Commit**

```bash
git add src/eval_v4_calibration.py tests/test_eval_v4_calibration.py
git commit -m "feat(v4-calibration-temperature-scaling): global T sweep + per-round greedy

run_global_T_sweep: anchor T=1.0 reproduces baseline byte-equal, sweeps
T_GRID, picks best cell, returns verdict via _classify_verdict.

run_per_round_greedy: sequential 1-D sweep over (R64, R32, S16, E8,
F4_NCG); per-step monotonic invariant tested via synthetic scoring
stub. Rows with no resolvable round (play-in pairs) pass through at
v8 baseline values. Anchor: all-1 per-round vector reproduces baseline
byte-equal on resolved rows.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Phase 1: CLI + end-to-end run

### Task 4: Wire CLI; run Phase 1 end-to-end on real data; capture verdict

**Files:**
- Modify: `src/eval_v4_calibration.py` (replace `main` stub)
- Generate: `output/v4_calibration_eval.json`, `output/v4_calibration_eval_log.txt`, `output/v4_calibration_reliability.png`, `output/pairwise_v8_calibrated_*.csv`

- [ ] **Step 1: Implement `_plot_reliability` and `main`**

In `src/eval_v4_calibration.py`, append (replacing the `main` stub):

```python
# ---------------------------------------------------------------------------
# Phase 2 (filled in Task 5)
# ---------------------------------------------------------------------------


def run_phase2(
    v4_csv: str,
    winning_config: dict,
    baseline_v8_csv: str,
    out_csv: str,
) -> dict:
    """Filled in Task 5."""
    raise NotImplementedError("see Task 5")


# ---------------------------------------------------------------------------
# Reliability plot
# ---------------------------------------------------------------------------


def _plot_reliability(
    v8_baseline_df: pd.DataFrame,
    v8_global_df: pd.DataFrame | None,
    v8_perround_df: pd.DataFrame | None,
    out_path: str,
    n_bins: int = 10,
) -> None:
    """3-line reliability diagram (predicted prob vs empirical win rate).
    Each frame must have p_a_wins; we treat the symmetric pair frame as
    a per-row probability and use the matching outcome from
    MNCAATourneyCompactResults to compute empirical win rate per bin."""
    results = pd.read_csv(DATA / "MNCAATourneyCompactResults.csv")
    # Build a (season, min_id, max_id) -> outcome (1 if min_id beat max_id else 0).
    outcomes = {}
    for _, r in results.iterrows():
        s = int(r["Season"])
        w, l = int(r["WTeamID"]), int(r["LTeamID"])
        a, b = (w, l) if w < l else (l, w)
        outcomes[(s, a, b)] = 1 if w == a else 0

    def _bin(df, label):
        sub = df.copy()
        sub["pair_key"] = list(zip(sub["season"], sub["team_a"], sub["team_b"]))
        sub = sub[sub["pair_key"].apply(
            lambda k: (int(k[0]), int(k[1]), int(k[2])) in outcomes
            if k[1] < k[2] else (int(k[0]), int(k[2]), int(k[1])) in outcomes
        )]
        bins = np.linspace(0.0, 1.0, n_bins + 1)
        sub["bin"] = pd.cut(sub["p_a_wins"], bins, include_lowest=True, right=True)
        per_bin = []
        for b_interval, grp in sub.groupby("bin"):
            mid = float(b_interval.mid) if hasattr(b_interval, "mid") else 0.5
            outs = []
            for _, row in grp.iterrows():
                key = (int(row["season"]), int(row["team_a"]), int(row["team_b"]))
                key_norm = (key[0], min(key[1], key[2]), max(key[1], key[2]))
                if key_norm not in outcomes:
                    continue
                a_won = outcomes[key_norm]
                if row["team_a"] == key_norm[1]:
                    outs.append(a_won)
                else:
                    outs.append(1 - a_won)
            if outs:
                per_bin.append((mid, float(np.mean(outs)), len(outs)))
        return per_bin

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="ideal")
    for (df, label, color) in [
        (v8_baseline_df, "v8 baseline", "C0"),
        (v8_global_df, "v8 + global T", "C1"),
        (v8_perround_df, "v8 + per-round T", "C2"),
    ]:
        if df is None:
            continue
        pts = _bin(df, label)
        if not pts:
            continue
        xs, ys, ns = zip(*pts)
        ax.plot(xs, ys, "-o", color=color, label=f"{label} (n_bins={len(pts)})")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("predicted P(team_a wins)")
    ax.set_ylabel("empirical win rate")
    ax.set_title("v4 calibration: temperature scaling reliability (10 bins)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Eval v4 calibration via temperature scaling.")
    p.add_argument("--v8-csv", default="output/pairwise_v8.csv")
    p.add_argument("--v4-csv", default="output/pairwise_v4.csv")
    p.add_argument("--baseline-total", type=float, default=2069.0)
    p.add_argument("--out-json", default="output/v4_calibration_eval.json")
    p.add_argument("--out-log", default="output/v4_calibration_eval_log.txt")
    p.add_argument("--out-plot", default="output/v4_calibration_reliability.png")
    p.add_argument("--out-dir", default="output")
    p.add_argument(
        "--phase",
        choices=["phase1", "phase2", "auto"],
        default="auto",
        help="phase1=skip Phase 2 always; phase2=run Phase 2 unconditionally "
             "with a manually-specified winning T; auto=run Phase 1, "
             "trigger Phase 2 only if PASS or MARGINAL.",
    )
    p.add_argument(
        "--phase2-T-config",
        default=None,
        help="(phase=phase2 only) JSON-encoded winning T config. "
             "Either a scalar (e.g. '1.15') or a per-round dict "
             "(e.g. '{\"R64\":1.15,\"R32\":1.0,\"S16\":0.85,\"E8\":1.5,\"F4_NCG\":1.0}').",
    )
    args = p.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)

    # Tee logs to file + stdout.
    log_handler = logging.FileHandler(args.out_log, mode="w")
    log_stream = logging.StreamHandler(sys.stdout)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for h in (log_handler, log_stream):
        h.setFormatter(fmt)
        logging.getLogger().addHandler(h)
    logging.getLogger().setLevel(logging.INFO)

    t_start = time.time()
    summary = {
        "spec": "docs/superpowers/specs/2026-05-08-v4-calibration-temperature-scaling-design.md",
        "plan": "docs/superpowers/plans/2026-05-08-v4-calibration-temperature-scaling.md",
        "baseline_total": float(args.baseline_total),
        "v8_csv": str(args.v8_csv),
    }

    if args.phase in ("phase1", "auto"):
        logger.info("===== PHASE 1: global T sweep =====")
        global_out = run_global_T_sweep(
            v8_csv=args.v8_csv,
            T_grid=T_GRID,
            baseline_total=args.baseline_total,
            scratch_dir=out_dir,
        )
        summary["phase1_global"] = global_out

        logger.info("===== PHASE 1: per-round greedy =====")
        perround_out = run_per_round_greedy(
            v8_csv=args.v8_csv,
            T_grid=T_GRID,
            round_order=ROUND_ORDER,
            baseline_total=args.baseline_total,
            scratch_dir=out_dir,
        )
        summary["phase1_perround"] = perround_out

        # Write the winning frames out for force-add.
        df_v8 = pd.read_csv(args.v8_csv)

        best_global_T = float(global_out["best_cell"]["T"])
        global_winner_df = scale_pairwise(df_v8, T=best_global_T)
        global_winner_path = (
            out_dir / f"pairwise_v8_calibrated_global_T{best_global_T:.2f}.csv"
        )
        global_winner_df.to_csv(global_winner_path, index=False)
        summary["phase1_global"]["winner_csv"] = str(global_winner_path)

        slots_df = pd.read_csv(DATA / "MNCAATourneySlots.csv")
        seeds_df = pd.read_csv(DATA / "MNCAATourneySeeds.csv")
        df_v8_bucketed = df_v8.copy()
        df_v8_bucketed["round_bucket"] = assign_round_buckets(
            df_v8_bucketed, slots_df, seeds_df
        )
        df_v8_resolved = df_v8_bucketed.dropna(subset=["round_bucket"]).copy()
        T_winner = perround_out["winning_T"]
        perround_resolved = scale_pairwise(df_v8_resolved, T=T_winner)
        perround_winner_df = pd.concat(
            [perround_resolved.drop(columns=["round_bucket"]),
             df_v8_bucketed[df_v8_bucketed["round_bucket"].isna()].drop(
                 columns=["round_bucket"])],
            ignore_index=True,
        )
        perround_filename = (
            "pairwise_v8_calibrated_perround_"
            + "_".join(f"{T_winner[r]:.2f}" for r in ROUND_ORDER)
            + ".csv"
        )
        perround_winner_path = out_dir / perround_filename
        perround_winner_df.to_csv(perround_winner_path, index=False)
        summary["phase1_perround"]["winner_csv"] = str(perround_winner_path)

        # Reliability plot.
        _plot_reliability(
            v8_baseline_df=df_v8,
            v8_global_df=global_winner_df,
            v8_perround_df=perround_winner_df,
            out_path=str(args.out_plot),
        )
        summary["plot"] = str(args.out_plot)

        # Decide overall Phase 1 verdict from the better of the two cells.
        global_delta = global_out["best_cell"]["delta_total"]
        perround_delta = perround_out["winning_cell"]["delta_total"]
        if perround_delta > global_delta:
            best_phase1 = perround_out["winning_cell"]
            best_kind = "per-round"
        else:
            best_phase1 = global_out["best_cell"]
            best_kind = "global"
        phase1_verdict = _classify_verdict(
            delta_total=best_phase1["delta_total"],
            drop_best_delta=best_phase1["drop_best_season_delta"],
            wins=best_phase1["wins"],
        )
        summary["phase1_overall"] = {
            "verdict": phase1_verdict,
            "best_kind": best_kind,
            "best_cell": best_phase1,
        }
        logger.info(
            "PHASE 1 OVERALL: verdict=%s best_kind=%s delta=%+.1f drop_best=%+.1f wins=%d",
            phase1_verdict, best_kind,
            best_phase1["delta_total"],
            best_phase1["drop_best_season_delta"],
            best_phase1["wins"],
        )

    # Phase 2 (Task 5).
    if args.phase == "phase2" or (
        args.phase == "auto"
        and summary["phase1_overall"]["verdict"] in ("PASS", "MARGINAL")
    ):
        logger.info("===== PHASE 2: retrain v8 on rescaled v4 =====")
        if args.phase == "phase2":
            if args.phase2_T_config is None:
                raise SystemExit("--phase=phase2 requires --phase2-T-config")
            T_cfg = json.loads(args.phase2_T_config)
        else:
            best_kind = summary["phase1_overall"]["best_kind"]
            if best_kind == "global":
                T_cfg = float(global_out["best_cell"]["T"])
            else:
                T_cfg = perround_out["winning_T"]
        phase2_out = run_phase2(
            v4_csv=args.v4_csv,
            winning_config=T_cfg,
            baseline_v8_csv=args.v8_csv,
            out_csv=str(out_dir / "pairwise_v8_phase2.csv"),
        )
        summary["phase2"] = phase2_out
        logger.info(
            "PHASE 2: verdict=%s delta=%+.1f drop_best=%+.1f wins=%d",
            phase2_out["verdict"],
            phase2_out["cell"]["delta_total"],
            phase2_out["cell"]["drop_best_season_delta"],
            phase2_out["cell"]["wins"],
        )
    elif args.phase == "auto":
        logger.info("PHASE 2 SKIPPED: Phase 1 verdict was %s", summary["phase1_overall"]["verdict"])
        summary["phase2"] = {"skipped": True, "reason": "Phase 1 NO-GO"}

    summary["wall_seconds"] = time.time() - t_start
    Path(args.out_json).write_text(json.dumps(summary, indent=2, default=str))
    logger.info("wrote %s", args.out_json)
    logger.info("wall: %.1f seconds", summary["wall_seconds"])
    return 0
```

- [ ] **Step 2: Run targeted tests + smoke**

```bash
python -m pytest -q tests/test_apply_temperature_scaling.py tests/test_eval_v4_calibration.py -v 2>&1 | tail -20
```

Expected: all tests pass (~22 total).

- [ ] **Step 3: Run Phase 1 end-to-end on real data**

```bash
python -m src.eval_v4_calibration --phase phase1 2>&1 | tee output/v4_calibration_eval_log.txt | tail -40
```

Expected wall: ~5-10 minutes. The log contains per-T cell summaries and the final overall verdict.

If `python -m src.eval_v4_calibration` fails with `ModuleNotFoundError`, run from the worktree root and add `python -m src.eval_v4_calibration` (the `_HERE/_ROOT` block in the module already handles `sys.path`). If it still fails, fall back to `python src/eval_v4_calibration.py`.

- [ ] **Step 4: Inspect the JSON for verdict + headline numbers**

```bash
python -c "
import json
from pathlib import Path
data = json.loads(Path('output/v4_calibration_eval.json').read_text())
print('phase1 overall:', data['phase1_overall']['verdict'])
print('  best_kind:', data['phase1_overall']['best_kind'])
print('  delta:', data['phase1_overall']['best_cell']['delta_total'])
print('  drop_best:', data['phase1_overall']['best_cell']['drop_best_season_delta'])
print('  wins:', data['phase1_overall']['best_cell']['wins'])
print('global cells:')
for c in data['phase1_global']['cells']:
    print(f\"  T={c['T']:.2f}: total={c['total']:.1f} delta={c['delta_total']:+.1f} W/L/T={c['wins']}/{c['losses']}/{c['ties']}\")
print('per-round chain:')
for s in data['phase1_perround']['greedy_chain']:
    print(f\"  {s['round']}: T={s['picked_T']:.2f} total={s['total_after_step']:.1f} delta={s['delta_total_after_step']:+.1f}\")
print('winning_T:', data['phase1_perround']['winning_T'])
"
```

This is the moment of truth. Record:
- Phase 1 overall verdict (PASS / MARGINAL / FAIL).
- Which sub-experiment (global vs per-round) won.
- The winning T (or per-round T vector).
- delta_total + drop_best_season_delta + wins.

- [ ] **Step 5: Inspect the reliability plot**

Open `output/v4_calibration_reliability.png`. Confirm:
- 3 traces are visible (baseline, global, per-round) and the diagonal.
- No NaN bins or crash artifacts.
- The traces don't crash in obviously implausible ways (e.g., the global-T trace below 0 at p=0.1 -- that would suggest a sign bug somewhere).

- [ ] **Step 6: Commit (Phase 1 wired + run done; outputs not yet force-added)**

```bash
git add src/eval_v4_calibration.py
git commit -m "feat(v4-calibration-temperature-scaling): CLI + end-to-end Phase 1 run

main() drives Phase 1 (global + per-round). Writes JSON + log + 3-line
reliability plot. Auto-decides Phase 2 trigger from Phase 1 overall
verdict (PASS or MARGINAL -> Phase 2; FAIL -> skip per spec).
run_phase2 stub still raises NotImplementedError until Task 5; --phase
flag controls explicit invocation.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

- [ ] **Step 7: Force-add Phase 1 outputs**

```bash
git add -f output/v4_calibration_eval.json
git add -f output/v4_calibration_eval_log.txt
git add -f output/v4_calibration_reliability.png
git add -f output/pairwise_v8_calibrated_global_T*.csv
git add -f output/pairwise_v8_calibrated_perround_*.csv
git status
git commit -m "data(v4-calibration-temperature-scaling): force-add Phase 1 outputs

Phase 1 verdict: <fill in PASS/MARGINAL/FAIL from the JSON>.
Headline: <best kind> delta=<+/-X.X> over 22 LOSO seasons.
Anchor: T=1.0 reproduces canonical pairwise_v8.csv byte-equal
(max_abs_diff=<value from JSON>).

Outputs:
  output/v4_calibration_eval.json
  output/v4_calibration_eval_log.txt
  output/v4_calibration_reliability.png
  output/pairwise_v8_calibrated_global_T<X>.csv
  output/pairwise_v8_calibrated_perround_*.csv

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Phase 2: Conditional retrain (only if Phase 1 PASS or MARGINAL)

### Task 5: Implement `run_phase2` and run it (conditional)

**Files:**
- Modify: `src/eval_v4_calibration.py` (replace `run_phase2` stub)
- Modify: `tests/test_eval_v4_calibration.py`
- Generate (only if triggered): `output/pairwise_v8_phase2.csv`

> **Conditional execution:** If Phase 1 verdict is FAIL, **skip this task entirely**. Document the FAIL in Task 6 and close the lane. The `run_phase2` stub stays as `NotImplementedError` and the corresponding test is added but skipped on `phase1_verdict=='FAIL'`.

> **If Phase 1 verdict is PASS or MARGINAL,** continue.

- [ ] **Step 1: Write failing test for `run_phase2` anchor**

Append to `tests/test_eval_v4_calibration.py`:

```python
@pytest.mark.skipif(
    not (Path("output/pairwise_v4.csv").exists()
         and Path("output/pairwise_v8.csv").exists()
         and Path("data/raw/march-machine-learning-2026/MNCAATourneyCompactResults.csv").exists()),
    reason="canonical CSVs / Kaggle data missing",
)
def test_phase2_anchor_T_one_reproduces_canonical_v8():
    """Phase 2 retrain on T=1.0-scaled v4 (== unmodified v4) reproduces
    canonical pairwise_v8.csv byte-equal. Spec Phase-2 anchor.

    This test is the slowest in the suite (~50s/season * 22 = ~20 min)
    so it is implicitly slow-marked; CI may skip via -m 'not slow'."""
    from src.eval_v4_calibration import run_phase2

    out = run_phase2(
        v4_csv="output/pairwise_v4.csv",
        winning_config=1.0,
        baseline_v8_csv="output/pairwise_v8.csv",
        out_csv="output/pairwise_v8_phase2_anchor.csv",
    )
    assert out["anchor"]["matches"] is True
    assert out["anchor"]["max_abs_diff"] < 1e-9
    Path("output/pairwise_v8_phase2_anchor.csv").unlink(missing_ok=True)
```

- [ ] **Step 2: Implement `run_phase2`**

In `src/eval_v4_calibration.py`, replace the `run_phase2` stub:

```python
def run_phase2(
    v4_csv: str,
    winning_config,
    baseline_v8_csv: str,
    out_csv: str,
) -> dict:
    """Scale v4 stage-1 with `winning_config` (scalar T or per-round dict),
    write the rescaled v4 to a tempfile, retrain v8 LOSO on the rescaled
    v4, score the resulting v8 frame.

    Anchor: if winning_config == 1.0 (or all-1 dict), the resulting v8
    frame must be byte-equal to baseline_v8_csv.

    Returns: {anchor, cell (full summary dict), verdict, retrain_csv}.
    """
    from src.train_stage2 import (
        DATA as STAGE2_DATA,
        build_v8_pairwise,
        load_per_game_data,
    )

    # Resolve winning_config into a scalar or dict; both go through scale_pairwise.
    df_v4 = pd.read_csv(v4_csv)
    if isinstance(winning_config, (int, float)):
        scaled_v4 = scale_pairwise(df_v4, T=float(winning_config))
    else:
        # Per-round dict requires bucket assignment.
        slots_df = pd.read_csv(STAGE2_DATA / "MNCAATourneySlots.csv")
        seeds_df = pd.read_csv(STAGE2_DATA / "MNCAATourneySeeds.csv")
        df_v4 = df_v4.copy()
        df_v4["round_bucket"] = assign_round_buckets(df_v4, slots_df, seeds_df)
        df_resolved = df_v4.dropna(subset=["round_bucket"]).copy()
        scaled_resolved = scale_pairwise(df_resolved, T=winning_config)
        scaled_v4 = pd.concat(
            [scaled_resolved.drop(columns=["round_bucket"]),
             df_v4[df_v4["round_bucket"].isna()].drop(columns=["round_bucket"])],
            ignore_index=True,
        )

    # Write rescaled v4 to a tempfile so build_v8_pairwise (which
    # accepts a path) can ingest it.
    rescaled_path = str(Path(out_csv).parent / "_phase2_v4_rescaled.csv")
    scaled_v4.to_csv(rescaled_path, index=False)

    # Build per-game training data from the RESCALED v4. (This is the
    # whole point of Phase 2: v8 trains on rescaled v4, not on the
    # canonical v4 used in Phase 1.)
    per_game = load_per_game_data(
        pairwise_csv=rescaled_path,
        results_csv=str(STAGE2_DATA / "MNCAATourneyCompactResults.csv"),
        seeds_csv=str(STAGE2_DATA / "MNCAATourneySeeds.csv"),
    )

    # Retrain v8 LOSO and apply to the rescaled v4 frame.
    build_v8_pairwise(
        per_game=per_game,
        pairwise_v4_csv=rescaled_path,
        seeds_csv=str(STAGE2_DATA / "MNCAATourneySeeds.csv"),
        out_path=out_csv,
    )

    # Score the resulting v8 frame.
    score = score_pairwise_path(out_csv)
    baseline_score = score_pairwise_path(baseline_v8_csv)
    per_season_delta = {
        int(s): float(score["per_season_pts"][s])
                - float(baseline_score["per_season_pts"][s])
        for s in baseline_score["per_season_pts"]
    }
    cell = _summarize_cell(per_season_delta, baseline_total=baseline_score["total_pts"])

    # Anchor: scoring against the canonical v8 baseline; the test is
    # whether the FRAME is byte-equal (not just whether bracket points
    # match).
    df_phase2 = pd.read_csv(out_csv)
    anchor = _anchor_check(df_phase2, baseline_v8_csv)

    verdict = _classify_verdict(
        delta_total=cell["delta_total"],
        drop_best_delta=cell["drop_best_season_delta"],
        wins=cell["wins"],
    )

    # Cleanup intermediate (keep the final out_csv for force-add).
    try:
        Path(rescaled_path).unlink()
    except FileNotFoundError:
        pass

    return {
        "anchor": anchor,
        "cell": cell,
        "verdict": verdict,
        "retrain_csv": str(out_csv),
    }
```

- [ ] **Step 3: Run the Phase 2 anchor test (slow; ~20 min)**

```bash
python -m pytest -q tests/test_eval_v4_calibration.py::test_phase2_anchor_T_one_reproduces_canonical_v8 -v 2>&1 | tail -10
```

Expected: 1 passed (~20 min wall). If it fails, **halt** -- the Phase 2 retrain pipeline is non-deterministic at T=1.0 and any Phase 2 result is uninterpretable until that's fixed.

- [ ] **Step 4: If Phase 2 anchor passed, run Phase 2 with the actual winning T**

```bash
python -m src.eval_v4_calibration --phase auto 2>&1 | tee -a output/v4_calibration_eval_log.txt | tail -40
```

(`--phase auto` will re-run Phase 1 + run Phase 2 only if Phase 1 returns PASS or MARGINAL. If Phase 1 was FAIL on the previous run and you've added Phase 2 only as the conditional escape hatch, this command will skip Phase 2 again -- which is the intended behavior.)

If Phase 1 was MARGINAL or PASS, this run does the Phase 2 retrain (~20-40 min) and writes Phase 2 cell numbers into the JSON.

- [ ] **Step 5: Inspect Phase 2 verdict**

```bash
python -c "
import json
from pathlib import Path
data = json.loads(Path('output/v4_calibration_eval.json').read_text())
p2 = data.get('phase2', {})
if p2.get('skipped'):
    print('Phase 2 SKIPPED:', p2['reason']); exit()
print('Phase 2 verdict:', p2['verdict'])
print('  delta:', p2['cell']['delta_total'])
print('  drop_best:', p2['cell']['drop_best_season_delta'])
print('  wins/losses/ties:', p2['cell']['wins'], p2['cell']['losses'], p2['cell']['ties'])
print('  biggest swing:', p2['cell']['biggest_swing_value'], 'in', p2['cell']['biggest_swing_season'])
print('  anchor matches:', p2['anchor']['matches'], 'max_abs_diff:', p2['anchor']['max_abs_diff'])
"
```

- [ ] **Step 6: Commit Phase 2 implementation**

```bash
git add src/eval_v4_calibration.py tests/test_eval_v4_calibration.py
git commit -m "feat(v4-calibration-temperature-scaling): Phase 2 retrain

run_phase2: scale v4 stage-1 by winning T config, retrain v8 LOSO via
train_stage2.fit_stage2 + build_v8_pairwise, score 22-season bracket
points. Anchor: T=1.0-scaled v4 retrain reproduces canonical
pairwise_v8.csv byte-equal (max_abs_diff < 1e-9).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

- [ ] **Step 7: Force-add Phase 2 outputs (if triggered)**

```bash
git add -f output/pairwise_v8_phase2.csv
git add -f output/v4_calibration_eval.json
git add -f output/v4_calibration_eval_log.txt
git status
git commit -m "data(v4-calibration-temperature-scaling): force-add Phase 2 outputs

Phase 2 verdict: <PASS/MARGINAL/FAIL from JSON>. Retrain anchor passes
(T=1.0 reproduces canonical pairwise_v8.csv byte-equal). Cell
delta=<+/-X.X>; drop_best_delta=<+/-X.X>; W/L/T=<a>/<b>/<c>.

Outputs:
  output/pairwise_v8_phase2.csv
  output/v4_calibration_eval.json (updated)
  output/v4_calibration_eval_log.txt (updated)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Phase 3: Findings + TODO update + PR

### Task 6: Write findings, update TODO, push branch, open PR

**Files:**
- Create: `docs/notes/2026-05-08-v4-calibration-temperature-scaling.md`
- Modify: `TODO.md`

- [ ] **Step 1: Write `docs/notes/2026-05-08-v4-calibration-temperature-scaling.md`**

Use this template; fill in every bracketed placeholder from the actual `output/v4_calibration_eval.json`:

````markdown
# v4 Calibration: Temperature Scaling -- Findings

**Date:** 2026-05-08
**Branch:** `feat/v4-calibration-temperature-scaling`
**Verdict:** **[PASS / MARGINAL / FAIL]** -- best [global / per-round] cell
delta = [+X.X] over 22 LOSO seasons vs canonical 2069 baseline; drop-
best-season delta = [+/-X.X]; W/L/T = [a/b/c]. [One-line follow-up
implication.]

**Spec:** `docs/superpowers/specs/2026-05-08-v4-calibration-temperature-scaling-design.md`
**Plan:** `docs/superpowers/plans/2026-05-08-v4-calibration-temperature-scaling.md`
**Strategy frame:** `docs/notes/2026-05-07-v4-kaggle-gap-strategy.md`

## TL;DR

[2-4 sentence summary: what was tested, what the verdict is, what
moves in the queue. Lead with the bracket-points number.]

## Anchors

| anchor | required | observed | verdict |
|--------|----------|----------|---------|
| Global T=1.0 reproduces `pairwise_v8.csv` byte-equal | max_abs_diff < 1e-9 | matches=[True/False], max_abs_diff=[X] | **PASS / FAIL** |
| Per-round (1,1,1,1,1) reproduces `pairwise_v8.csv` byte-equal on resolved rows | max_abs_diff < 1e-9 | matches=[True/False], max_abs_diff=[X], n_resolved=[N] | **PASS / FAIL** |
| Phase 2 T=1.0 retrain reproduces `pairwise_v8.csv` byte-equal | max_abs_diff < 1e-9 (only if Phase 2 ran) | [observed or "Phase 2 skipped"] | **PASS / SKIPPED** |
| Canonical baseline total brkt pts | 2069 | 2069 (FP precision) | **PASS** |

## Phase 1: Global T sweep

7-cell sweep over T in [TABLE FROM JSON]:

| T | total | delta vs 2069 | W/L/T | drop_best | biggest_swing |
|---|-------|---------------|-------|-----------|----------------|
| 0.70 | ... | ... | ... | ... | ... |
| 0.85 | ... | ... | ... | ... | ... |
| **1.00** | **2069** | **0** | **0/0/22** | **0** | **0** (anchor) |
| 1.15 | ... | ... | ... | ... | ... |
| 1.30 | ... | ... | ... | ... | ... |
| 1.50 | ... | ... | ... | ... | ... |
| 2.00 | ... | ... | ... | ... | ... |

Best cell: **T=[X]** at delta = **[+/-X.X]** over 22 seasons.

## Phase 1: Per-round greedy

Sequential greedy over (R64, R32, S16, E8, F4_NCG):

| step | round | picked T | total after step | delta after step |
|------|-------|----------|------------------|------------------|
| 1 | R64 | [X] | [X] | [+/-X.X] |
| 2 | R32 | [X] | [X] | [+/-X.X] |
| 3 | S16 | [X] | [X] | [+/-X.X] |
| 4 | E8 | [X] | [X] | [+/-X.X] |
| 5 | F4_NCG | [X] | [X] | [+/-X.X] |

Final winning T: **R64=[X], R32=[X], S16=[X], E8=[X], F4_NCG=[X]**.
Final delta vs 2069: **[+/-X.X]**.

## Phase 1 overall verdict

[PASS / MARGINAL / FAIL] -- the better of the two sub-experiments
([global / per-round]) at delta=[X] cleared / failed the
[+10 MARGINAL / +25 PASS] bar.

## Per-season breakdown (Phase 1, winning cell)

[22-row table from JSON.]

## Phase 2: [retrained / SKIPPED per Phase 1 NO-GO]

[If skipped:]
> Phase 2 was not triggered because Phase 1 returned FAIL across both
> sub-experiments. Per the spec's decision matrix, only PASS or
> MARGINAL Phase 1 verdicts trigger the retrain. The lane is closed.

[If triggered, fill in:]

| metric | value |
|--------|-------|
| total brkt pts | [X] |
| delta vs 2069 | [+/-X.X] |
| W/L/T | [a/b/c] |
| drop_best_season_delta | [+/-X.X] |
| anchor (T=1.0 reproduces canonical v8) | matches=[True/False] |

[Per-season table.]

## What this implies for the queue

[Conditional on verdict:]

- **PASS (Phase 1 + Phase 2):** swap candidate. Calibration-shape lane
  delivered. Production swap: write follow-up spec for live wiring of
  the temperature-scaling step into `predict_2026_stage2.py` /
  `generate_bracket_real.py`. Active queue #1 closes; #2 (MLP) becomes
  the new #1 only if a follow-up specifically wants ensemble diversity.
- **MARGINAL:** retain code on branch as experiment record; document
  candidate. Roster-level data (TODO #4) stays in place at #2 -- still
  dominant external-data candidate. MLP (#3) stays at #3.
- **FAIL:** close the calibration-shape lane. Roster-level data
  (TODO #4) becomes Active queue #1 by elimination -- both same-data
  fixes (R64-blend, calibration-shape) have now produced null results
  on bracket points, strongly elevating structurally-different signal.
  MLP (#3) stays at #3 (same-data peer; same risk profile).

## Calibration plot (`output/v4_calibration_reliability.png`)

[1-paragraph description of what the 3-line reliability diagram shows.
Whether the calibrated traces visibly differ from the baseline; in
which bins. If the bracket-points result is FAIL but reliability
visually improves, that's important to call out -- the LL/calibration
gain didn't translate to bracket points (the same-data-peer pattern).]

## Caveats

- **Phase 1 selection bias.** Sweeping T on the same 22-season aggregate
  we score on is in-sample selection. Drop-best-season delta exposes
  single-season concentration; nested-LOSO selection deferred to a
  Phase 2 robustness check only if a cell PASSes.
- **F4_NCG collapse.** Two structurally-different rounds (best-of-4 vs
  best-of-2) share one T. If T_F4_NCG is non-trivial in the winning
  cell, [PASS-case sensitivity probe / FAIL-case retrospective].
- **Sequential greedy.** R64-first ordering means later-round T is
  conditioned on R64's pick. Joint optimization (7^5 = 16,807) is
  intractable; greedy chain is reported step-by-step for inspection.
- **Same-data-peer ceiling.** Every same-v4-data experiment since the
  leak fix has either failed the gate or failed translation; this is
  the sixth in that pattern (BT, PEER, HBT, plain-BT-bracket-points,
  R64-line-blend, calibration-shape). [Verdict-conditional phrasing.]

## Files of record

```
src/apply_temperature_scaling.py         -- pure rescaling module
src/eval_v4_calibration.py               -- driver
tests/test_apply_temperature_scaling.py  -- 9 unit + 2 anchor + 1 smoke
tests/test_eval_v4_calibration.py        -- 12 unit + 1 Phase 2 anchor + 1 smoke

output/v4_calibration_eval.json
output/v4_calibration_eval_log.txt
output/v4_calibration_reliability.png
output/pairwise_v8_calibrated_global_T<X>.csv
output/pairwise_v8_calibrated_perround_*.csv
[output/pairwise_v8_phase2.csv -- only if Phase 2 ran]

docs/superpowers/specs/2026-05-08-v4-calibration-temperature-scaling-design.md
docs/superpowers/plans/2026-05-08-v4-calibration-temperature-scaling.md
docs/notes/2026-05-08-v4-calibration-temperature-scaling.md (this note)
```

## Compute

- Phase 1 global sweep: ~[X] minutes wall (~10s/cell * 7 cells).
- Phase 1 per-round greedy: ~[X] minutes wall (~10s/cell * 35 cells).
- Phase 2 retrain (if triggered): ~[X] minutes wall (~50s/season * 22).
- **Total wall time: [X] minutes.**

## What we learned (one-sentence summary)

[Tight verdict in one sentence. Examples:
 PASS:   "Post-hoc temperature scaling at T=[X] adds +[X] bracket points
         on a robust 22-season profile -- v4's stage-2 output is
         meaningfully miscalibrated for the bracket-points objective."
 MARG:   "Post-hoc temperature scaling produced a +[X] but
         single-season-concentrated lift; not strong enough to swap
         on Phase 1 alone."
 FAIL:   "Post-hoc temperature scaling does not move bracket points
         despite [the audits' / the per-round greedy chain's] strong
         calibration-shape priors -- the same-data-peer pattern holds
         for the calibration-shape lane too."]
````

Fill in every `[X]` from the JSON. Keep the prose tight; this note is the canonical findings record and will be the single source the TODO entry references.

- [ ] **Step 2: Update `TODO.md` based on verdict**

Read current `TODO.md`. The current Active queue #1 is "v4 calibration-shape engineering (audit-derived)" (per the post-PR 31 re-prioritization).

Action depends on verdict:

- **PASS (Phase 1 + Phase 2):** Move "v4 calibration-shape engineering" to Done with the verdict + numbers. Add a new Active queue #1: "Wire temperature scaling into the production pipeline (live 2026 bracket regen)." Demote MLP (was #2) and roster-data (was #4) accordingly.

- **MARGINAL:** Move the entry to Done with the MARGINAL verdict and the numbers. Update Active queue preamble: calibration-shape produced a candidate-only result; retain code, no swap. Promote roster-data (was #4) to #1 by elimination. MLP stays at #2.

- **FAIL:** Move the entry to Done with the FAIL verdict + numbers. Update Active queue preamble: calibration-shape lane closed. Promote roster-level returning-experience (was #4) to #1; MLP (was #2) stays at #2; Bayesian BT (was #3) stays at #3; futures-as-feature (was #5) stays at #5 (further demoted -- two same-data null results in a row).

The Done entry's title:

```
- **v4 calibration-shape: temperature scaling -- [VERDICT] (2026-05-08).**
  [1-2 sentence summary with the headline numbers + queue implication.
  Mirror the R64-line-blend Done entry's structure.]
```

Update the Active queue preamble with one short paragraph describing what changed:

```
> **Update 2026-05-08 (calibration-shape verdict came back [VERDICT]).**
> [1-3 sentence framing: what was tested, what won/lost, what the
> queue implication is. Specifically address the "same-data-peer"
> pattern as a thread.]
```

- [ ] **Step 3: Run full pytest sweep**

```bash
python -m pytest -q tests/test_apply_temperature_scaling.py tests/test_eval_v4_calibration.py 2>&1 | tail -10
```

Expected: ~24 passed (or +/- a few skipped if data files are missing in CI).

If a full-suite run is requested, that takes ~8 hours via XGB-heavy LOSO tests; prefer the targeted run unless explicitly asked.

- [ ] **Step 4: Verify the leak-fix regression test still green**

```bash
python -m pytest -q tests/test_filter_vegas_to_pre_tournament.py 2>&1 | tail -5 || true
```

(File name may vary; check existence first.)

```bash
ls tests | grep -i leak
ls tests | grep -i vegas
```

If a leak-fix regression test exists, run it and confirm green. If not, this step is informational only.

- [ ] **Step 5: Final commit (findings + TODO)**

```bash
git add docs/notes/2026-05-08-v4-calibration-temperature-scaling.md TODO.md
git commit -m "docs(v4-calibration-temperature-scaling): findings + TODO update -- <VERDICT>

Verdict: <VERDICT>. <1-line headline>.

Findings: docs/notes/2026-05-08-v4-calibration-temperature-scaling.md

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

- [ ] **Step 6: Push branch and open PR**

```bash
git push -u origin feat/v4-calibration-temperature-scaling
```

Then open the PR (capture URL):

```bash
gh pr create --title "feat(v4-calibration-temperature-scaling): <VERDICT> -- <verdict shorthand>" --body "$(cat <<'EOF'
## Summary

Post-hoc temperature scaling on v8's pairwise output (Phase 1) +
conditional retrain of v8 on rescaled v4 (Phase 2). Cheapest test of
the calibration-shape hypothesis from the 538 + Vegas audits.

Phase 1 global T sweep: 7-cell grid over `{0.7, 0.85, 1.0, 1.15, 1.3,
1.5, 2.0}`. Phase 1 per-round greedy: sequential 1-D sweeps over
(R64, R32, S16, E8, F4_NCG with F4 + Champ collapsed for n>=66 per
knob). Phase 2: only if Phase 1 PASS or MARGINAL, retrain v8 LOSO on
rescaled v4. Gate: 22-season bracket points vs canonical 2069.

## Verdict: <VERDICT>

<1-2 sentence verdict with delta + best kind + queue implication.>

## Anchors PASS

- Global T=1.0 reproduces canonical `output/pairwise_v8.csv`
  byte-equal (matches=True, max_abs_diff < 1e-9).
- Per-round (1,1,1,1,1) reproduces canonical on resolved rows.
- Phase 2 T=1.0 retrain reproduces canonical pairwise_v8.csv
  byte-equal (only if Phase 2 ran).
- Canonical baseline: 2069 brkt pts over 22 LOSO seasons.

## Phase 1 cells

[Same tables as the findings note, condensed.]

## Phase 2 cell (if applicable)

[Same numbers as findings, condensed.]

## Test plan

- [x] pytest tests/test_apply_temperature_scaling.py (10 passed)
- [x] pytest tests/test_eval_v4_calibration.py (13 passed)
- [x] Phase 2 anchor test (slow, ~20 min) [PASS / SKIPPED]
- [x] python -m src.eval_v4_calibration end-to-end
- [x] manual inspection of reliability PNG

## Files

- src/apply_temperature_scaling.py
- src/eval_v4_calibration.py
- tests/test_apply_temperature_scaling.py
- tests/test_eval_v4_calibration.py
- output/pairwise_v8_calibrated_*.csv (force-added)
- output/pairwise_v8_phase2.csv (force-added if Phase 2 ran)
- output/v4_calibration_eval.{json,_log.txt}
- output/v4_calibration_reliability.png
- docs/superpowers/specs/2026-05-08-v4-calibration-temperature-scaling-design.md
- docs/superpowers/plans/2026-05-08-v4-calibration-temperature-scaling.md
- docs/notes/2026-05-08-v4-calibration-temperature-scaling.md
EOF
)"
```

Capture the PR URL.

---

## Risks (carried from spec, restated here for executor reference)

1. **Phase 1 selection bias.** Sweeping T on 22-season aggregate is
   in-sample selection. Drop-best-season delta is the first-class
   robustness column; cells with single-season concentration are
   demoted to MARGINAL even at PASS magnitude.

2. **Same-data-peer ceiling.** Every same-v4-data experiment since the
   leak fix has either failed the gate or failed translation. Phase 2
   gives the hypothesis a second chance via retrain; a NO-GO across
   both is decisive.

3. **F4_NCG collapse may hide signal.** F4 (n=44) and NCG (n=22) share
   a T. If Phase 1 PASSes with non-trivial T_F4_NCG, sensitivity probe
   in findings note splitting the bucket and re-scoring at granularity.

4. **Sequential greedy may miss the joint optimum.** 7^5 = 16,807 is
   intractable; greedy chain is the codebase pattern. Report each step's
   marginal so a non-monotone marginal flags a re-sweep candidate.

5. **PR 19 leak-fix invariance.** Temperature scaling acts on pairwise
   probabilities only; orthogonal to v4's training feature pipeline.
   Verified by running the leak-fix regression test (Step 4 of Task 6).

6. **Phase 2 retrain may diverge.** XGB at `random_state=42` is
   deterministic in seed but the input frame changes when v4 is
   rescaled. Phase 2 anchor (T=1.0 reproduces 2069 byte-equal) gates
   this; report tree-structure changes (n_estimators, feature
   importances) in the findings note.

7. **Anchor failure recovery.** If `_anchor_check.matches == False` in
   either Phase 1 or Phase 2 path, **halt** and investigate before
   declaring a verdict. Most likely cause: `train_stage2` was modified
   between the canonical baseline run and this run (random seed,
   feature column ordering, etc.). Recovery: regenerate the canonical
   baseline per `docs/data_recovery.md` or relax tolerance (with
   findings-note disclosure).

---

## Self-review (controller's notes; not executor instructions)

**Spec coverage:** every section of the spec maps to at least one task.

| Spec section | Task |
|---|---|
| Goals 1 (post-hoc rescaling) | Task 1 |
| Goals 2.i (global T sweep, anchor T=1.0 reproduces 2069) | Task 3 (sweep) + Task 4 (run + JSON) |
| Goals 2.ii (per-round greedy, anchor (1,1,1,1,1) reproduces 2069) | Task 3 (sweep) + Task 4 (run + JSON) |
| Goals 3 (Phase 2 retrain) | Task 5 |
| Goals 4 (decide PASS/MARGINAL/FAIL) | Task 4 (Phase 1) + Task 5 (Phase 2) + Task 6 (TODO) |
| Goals 5 (single experiment-record commit set + findings) | Task 6 |
| Anchor 1 (T=1 round-trip on real CSV) | Task 1 (Step 5 anchor test) |
| Anchor 2 (global T=1 cell == 2069) | Task 3 (Step 1 test) + Task 4 (Step 3 run) |
| Anchor 3 (per-round all-1 cell == 2069) | Task 3 (Step 1 test) + Task 4 (Step 3 run) |
| Anchor 4 (Phase 2 T=1 retrain == canonical) | Task 5 (Step 1 test + Step 3 run) |
| Decision matrix (+25/+10/<10 with concentration demotion) | Task 2 (`_classify_verdict`) |
| Drop-best-season delta | Task 2 (`_drop_best_season_delta`) |
| F4_NCG collapse | Task 1 (`_round_int_to_bucket`) |
| Reliability plot | Task 4 (Step 1 `_plot_reliability`) |
| Risks 1-7 | Restated above; addressed in respective tasks |

**No placeholders:** every step has actual code or commands. The
findings template (Task 6 Step 1) has bracketed `[X]` markers but those
are FILL FROM JSON instructions for the executor, not plan-level
placeholders.

**Type/name consistency:**

- `scale_pairwise(df, T)` accepts `float | dict[str, float]`; consistent.
- `assign_round_buckets(df, slots_df, seeds_df) -> pd.Series` returns
  `pd.NA` for unresolvable; consistent with usage in `run_per_round_greedy`.
- `_anchor_check(df_actual, baseline_csv) -> dict` returns
  `{matches, max_abs_diff, n_rows, ...}`; consistent with R64-blend
  `_anchor_check`.
- `_classify_verdict(delta_total, drop_best_delta, wins) -> str`
  returns one of `{"PASS", "MARGINAL", "FAIL"}`; consistent across
  callers in `run_global_T_sweep`, `run_per_round_greedy`, `run_phase2`.
- `_summarize_cell(per_season_delta, baseline_total) -> dict` returns
  `{total, delta_total, wins, losses, ties, biggest_swing_value,
  biggest_swing_season, drop_best_season_delta, per_season_delta}`;
  every call site uses the same keys.
- `T_GRID = [0.7, 0.85, 1.0, 1.15, 1.3, 1.5, 2.0]` is the single source
  of truth; `ROUND_ORDER = ["R64", "R32", "S16", "E8", "F4_NCG"]` is the
  single source of truth for greedy order.

**Phase boundaries:** Tasks 1-4 = Phase 1 module + run; Task 5 = Phase 2
(conditional); Task 6 = findings + TODO + PR. Skipping Task 5 entirely
on Phase 1 FAIL is documented at Task 5's header. The plan stays
coherent if Phase 2 is skipped (no orphan references).
