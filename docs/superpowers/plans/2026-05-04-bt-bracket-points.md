# Plain BT Bracket-Points Re-Test -- Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Score `(w * pairwise_v4 + (1 - w) * pairwise_bt) -> v9-C` against bracket points across `w in {0.60, 0.70, 0.80, 0.90, 0.95, 1.00}`. Verify `w=1.00` reproduces v4 + v9-C exactly. Apply the standard ladder.

**Architecture:** Single driver script `src/sweep_bt_bracket_points.py` loops the grid, calling `ensemble_stage1.py`, `run_v9c_on_stage1.py`, and `score_chalk_brackets.score_pairwise_path` -- all unchanged. JSON summary writes to `output/bt_bracket_sweep.json`.

**Tech Stack:** pandas, existing v9-C / scoring infrastructure, no new model code.

**Spec:** `docs/superpowers/specs/2026-05-04-bt-bracket-points-design.md`

---

## File Structure

**Created (committed):**

- `src/sweep_bt_bracket_points.py` -- driver script (~150 LOC).
  - Public: `run_sweep(weights, v4_csv, bt_csv, baseline_v9c_csv, out_dir, out_json) -> dict`
  - Private: `_score_pairwise(csv_path) -> dict` (per-season + total brkt pts, W/L/T)
  - Private: `_anchor_check(w1_csv, baseline_csv) -> dict` (assert match)
  - `main()` CLI: `--weights 0.6,0.7,0.8,0.9,0.95,1.0 --out-dir output/`
- `tests/test_sweep_bt_bracket_points.py` -- 4 unit tests (~80 LOC).
- `docs/superpowers/specs/2026-05-04-bt-bracket-points-design.md` (already created)
- `docs/superpowers/plans/2026-05-04-bt-bracket-points.md` (this file)

**Generated (committed via `git add -f`):**

- `output/pairwise_v4bt_w<W>.csv` x 6 cells (ensemble outputs).
- `output/pairwise_v9c_v4bt_w<W>.csv` x 6 cells (post-v9-C).
- `output/bt_bracket_sweep.json` (per-(w, season) numbers + verdict).
- `output/bt_bracket_log.txt` (stdout from the run).
- `docs/notes/2026-05-04-bt-bracket-points.md` (findings).

**Modified:**

- `TODO.md` -- move active queue item #2 to "Done" or "Tried and rejected" with verdict.

---

## Phase 1: Driver script + unit tests

Single phase. The experiment is a thin orchestration layer over existing modules.

### Task 1: Implement `src/sweep_bt_bracket_points.py`

**Files:**
- Create: `src/sweep_bt_bracket_points.py`
- Test: `tests/test_sweep_bt_bracket_points.py`

- [ ] **Step 1: Write failing unit tests**

```python
"""Unit tests for src/sweep_bt_bracket_points.py."""
from pathlib import Path

import pandas as pd
import pytest

from src.sweep_bt_bracket_points import (
    _anchor_check,
    _make_weight_pair,
    _score_pairwise,
    run_sweep,
)


def test_make_weight_pair_complements_to_one():
    assert _make_weight_pair(0.6) == (0.6, 0.4)
    assert _make_weight_pair(1.0) == (1.0, 0.0)
    assert _make_weight_pair(0.95) == (0.95, 0.05)


def test_anchor_check_matches_when_csvs_equal(tmp_path):
    csv_a = tmp_path / "a.csv"
    csv_b = tmp_path / "b.csv"
    df = pd.DataFrame([
        {"season": 2024, "team_a": 1, "team_b": 2, "p_a_wins": 0.7},
        {"season": 2024, "team_a": 1, "team_b": 3, "p_a_wins": 0.6},
    ])
    df.to_csv(csv_a, index=False)
    df.to_csv(csv_b, index=False)
    result = _anchor_check(str(csv_a), str(csv_b))
    assert result["matches"] is True
    assert result["max_abs_diff"] < 1e-12


def test_anchor_check_detects_mismatch(tmp_path):
    csv_a = tmp_path / "a.csv"
    csv_b = tmp_path / "b.csv"
    pd.DataFrame([
        {"season": 2024, "team_a": 1, "team_b": 2, "p_a_wins": 0.7},
    ]).to_csv(csv_a, index=False)
    pd.DataFrame([
        {"season": 2024, "team_a": 1, "team_b": 2, "p_a_wins": 0.65},
    ]).to_csv(csv_b, index=False)
    result = _anchor_check(str(csv_a), str(csv_b))
    assert result["matches"] is False
    assert result["max_abs_diff"] > 0.04


def test_score_pairwise_returns_per_season_and_total(tmp_path, monkeypatch):
    """Smoke: _score_pairwise wraps score_chalk_brackets without crashing
    on the existing pairwise_v9c_v4_baseline.csv."""
    csv = Path("output/pairwise_v9c_v4_baseline.csv")
    if not csv.exists():
        pytest.skip(f"{csv} not present")
    summary = _score_pairwise(str(csv))
    assert "total_pts" in summary
    assert "per_season" in summary
    assert summary["total_pts"] > 0
    assert len(summary["per_season"]) >= 20
```

Run: `pytest tests/test_sweep_bt_bracket_points.py` -- expect ImportError.

- [ ] **Step 2: Implement the driver**

`src/sweep_bt_bracket_points.py`:

```python
"""Drive the plain-BT bracket-points re-test sweep.

Spec: docs/superpowers/specs/2026-05-04-bt-bracket-points-design.md

For each w in --weights:
    1. ensemble_stage1.average_pairwise_csvs(v4, bt, w, 1-w) -> ensemble_csv
    2. run_v9c_on_stage1.run_v9c(ensemble_csv) -> v9c_csv
    3. score_chalk_brackets.score_pairwise_path(v9c_csv) -> per-season + total
Anchor: w=1.0 should produce a v9c_csv that matches the existing
output/pairwise_v9c_v4_baseline.csv exactly.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.ensemble_stage1 import average_pairwise_csvs
from src.run_v9c_on_stage1 import run_v9c
from src.score_chalk_brackets import score_pairwise_path


DEFAULT_WEIGHTS = [0.60, 0.70, 0.80, 0.90, 0.95, 1.00]
DEFAULT_OUT_JSON = "output/bt_bracket_sweep.json"


def _make_weight_pair(w_v4: float) -> tuple[float, float]:
    return (round(w_v4, 4), round(1.0 - w_v4, 4))


def _format_w(w: float) -> str:
    return f"{w:.2f}"


def _anchor_check(csv_actual: str, csv_expected: str) -> dict:
    """Verify csv_actual matches csv_expected on (season, team_a, team_b).
    Returns {matches: bool, max_abs_diff: float}."""
    a = pd.read_csv(csv_actual).drop_duplicates(
        ["season", "team_a", "team_b"], keep="last"
    )
    b = pd.read_csv(csv_expected).drop_duplicates(
        ["season", "team_a", "team_b"], keep="last"
    )
    merged = a.merge(b, on=["season", "team_a", "team_b"],
                     suffixes=("_actual", "_expected"))
    n_only_a = len(a) - len(merged)
    n_only_b = len(b) - len(merged)
    if n_only_a != 0 or n_only_b != 0:
        return {
            "matches": False,
            "max_abs_diff": float("nan"),
            "n_only_actual": n_only_a,
            "n_only_expected": n_only_b,
        }
    diff = (merged["p_a_wins_actual"] - merged["p_a_wins_expected"]).abs()
    return {
        "matches": bool(diff.max() < 1e-9),
        "max_abs_diff": float(diff.max()),
        "n_rows": len(merged),
    }


def _score_pairwise(csv_path: str) -> dict:
    """Wrap score_chalk_brackets.score_pairwise_path -> stable dict."""
    s = score_pairwise_path(csv_path)
    # score_pairwise_path may return its own shape; normalize.
    return {
        "total_pts": int(s["total_pts"]) if "total_pts" in s else int(sum(s.get("per_season", {}).values())),
        "per_season": {int(k): int(v) for k, v in s.get("per_season", {}).items()},
    }


def run_sweep(
    weights: list[float],
    v4_csv: str,
    bt_csv: str,
    baseline_v9c_csv: str,
    out_dir: str | Path,
    out_json: str = DEFAULT_OUT_JSON,
) -> dict:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Score the canonical v4 + v9-C baseline once.
    print("=" * 70)
    print(f"BASELINE: {baseline_v9c_csv}")
    print("=" * 70)
    baseline = _score_pairwise(baseline_v9c_csv)
    print(f"  total_pts: {baseline['total_pts']}")

    cells = []
    anchor_check = None
    for w in weights:
        w_v4, w_bt = _make_weight_pair(w)
        ens_csv = str(out_path / f"pairwise_v4bt_w{_format_w(w)}.csv")
        v9c_csv = str(out_path / f"pairwise_v9c_v4bt_w{_format_w(w)}.csv")

        print()
        print("=" * 70)
        print(f"CELL  w_v4={w_v4}  w_bt={w_bt}")
        print("=" * 70)

        t0 = time.time()
        average_pairwise_csvs(
            in_a=v4_csv, in_b=bt_csv,
            weights=(w_v4, w_bt),
            out=ens_csv,
        )
        run_v9c(pairwise_in=ens_csv, pairwise_out=v9c_csv)
        cell = _score_pairwise(v9c_csv)

        deltas = {s: cell["per_season"][s] - baseline["per_season"].get(s, 0)
                  for s in cell["per_season"]}
        wins = sum(1 for d in deltas.values() if d > 0)
        losses = sum(1 for d in deltas.values() if d < 0)
        ties = sum(1 for d in deltas.values() if d == 0)
        delta_total = cell["total_pts"] - baseline["total_pts"]

        print(f"  total_pts: {cell['total_pts']}  delta: {delta_total:+d}  "
              f"W/L/T: {wins}/{losses}/{ties}  ({time.time() - t0:.1f}s)")

        cells.append({
            "w_v4": w_v4, "w_bt": w_bt,
            "ensemble_csv": ens_csv, "v9c_csv": v9c_csv,
            "total_pts": cell["total_pts"],
            "delta_vs_baseline": delta_total,
            "per_season": cell["per_season"],
            "per_season_delta": deltas,
            "wins": wins, "losses": losses, "ties": ties,
        })

        if abs(w_v4 - 1.0) < 1e-9:
            anchor_check = _anchor_check(v9c_csv, baseline_v9c_csv)
            print(f"  ANCHOR CHECK: matches={anchor_check['matches']}, "
                  f"max_abs_diff={anchor_check.get('max_abs_diff', 'NaN'):.2e}")

    best = max(cells, key=lambda c: c["delta_vs_baseline"])
    if best["delta_vs_baseline"] >= 25:
        verdict = "CLEAR"
    elif best["delta_vs_baseline"] >= 10:
        verdict = "MARGINAL"
    else:
        verdict = "NO-GO"

    summary = {
        "config": {
            "weights": weights,
            "v4_pairwise": v4_csv,
            "bt_pairwise": bt_csv,
            "v9c_baseline": baseline_v9c_csv,
        },
        "anchor_check": anchor_check,
        "v4_baseline": baseline,
        "cells": cells,
        "best_cell": best,
        "verdict": verdict,
    }

    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print()
    print("=" * 70)
    print(f"VERDICT: {verdict}  best w_v4={best['w_v4']}  "
          f"delta={best['delta_vs_baseline']:+d}")
    print(f"  saved {out_json}")
    return summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--weights",
        default=",".join(f"{w:.2f}" for w in DEFAULT_WEIGHTS),
    )
    parser.add_argument("--v4", default="output/pairwise_v4.csv")
    parser.add_argument("--bt", default="output/pairwise_bt.csv")
    parser.add_argument(
        "--baseline-v9c",
        default="output/pairwise_v9c_v4_baseline.csv",
    )
    parser.add_argument("--out-dir", default="output")
    parser.add_argument("--out-json", default=DEFAULT_OUT_JSON)
    args = parser.parse_args(argv)

    weights = [float(x) for x in args.weights.split(",") if x.strip()]
    summary = run_sweep(
        weights=weights,
        v4_csv=args.v4,
        bt_csv=args.bt,
        baseline_v9c_csv=args.baseline_v9c,
        out_dir=args.out_dir,
        out_json=args.out_json,
    )
    return 0 if summary["verdict"] != "NO-GO" else 1


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 3: Run unit tests**

`pytest -v tests/test_sweep_bt_bracket_points.py` -- expect PASS.

**Phase 1 exit criterion:** 4 unit tests green.

---

## Phase 2: Run sweep on real data + apply ladder

### Task 2: Run the full sweep

- [ ] **Step 1: Verify inputs are present**

```bash
ls -la output/pairwise_v4.csv output/pairwise_bt.csv output/pairwise_v9c_v4_baseline.csv
```

All three force-added on prior PRs (or copied from main checkout if running in a fresh worktree). The baseline reference exists from the v9-C production swap (PR 9).

- [ ] **Step 2: Run the sweep**

```bash
python src/sweep_bt_bracket_points.py \
    --weights 0.60,0.70,0.80,0.90,0.95,1.00 \
    --out-dir output/ \
    --out-json output/bt_bracket_sweep.json \
    2>&1 | tee output/bt_bracket_log.txt
```

Expected wall time: ~10-15 min (six v9-C runs, ~1-2 min each).

- [ ] **Step 3: Verify the anchor**

The `w_v4=1.00` cell must produce a v9-C CSV that matches
`pairwise_v9c_v4_baseline.csv` to floating-point precision.
`output/bt_bracket_sweep.json` includes `anchor_check.matches`. If
false, investigate before trusting any other cell.

- [ ] **Step 4: Apply the verdict ladder**

Read `verdict` from the JSON. Three branches:

| verdict   | next                                                |
|-----------|-----------------------------------------------------|
| NO-GO     | proceed to Phase 3 findings (most likely outcome)   |
| MARGINAL  | proceed to Phase 3 findings; do NOT swap            |
| CLEAR     | proceed to Phase 3 findings; production swap is a separate follow-up commit |

**Phase 2 exit criterion:** `output/bt_bracket_sweep.json` written, anchor verified, verdict assigned.

---

## Phase 3: Findings note + TODO update

### Task 3: Write `docs/notes/2026-05-04-bt-bracket-points.md`

- [ ] **Step 1: Capture verdict + per-cell numbers + lesson**

Mirror the structure of `docs/notes/2026-05-01-bayesian-stage1.md`:

```markdown
# Plain BT Bracket-Points Re-Test -- Findings

**Date:** 2026-05-04
**Branch:** feat/bt-bracket-points
**Verdict:** [NO-GO | MARGINAL | CLEAR]
**Spec:** docs/superpowers/specs/2026-05-04-bt-bracket-points-design.md
**Plan:** docs/superpowers/plans/2026-05-04-bt-bracket-points.md

## TL;DR
[1-paragraph summary including how this updates the LL-gate's status
 as a screening tool.]

## Setup recap
[Inputs, weight grid, anchor outcome, total wall time.]

## Per-cell results
| w_v4 | w_bt | total_pts | delta | W | L | T |
|------|------|-----------|-------|---|---|---|
| 1.00 | 0.00 | NNNN      |  +0   | - | - | - |
| 0.95 | 0.05 | ...       |  ...  | ...
[full grid]

## Best cell
[w, total, delta, biggest single-season swing -- flag if delta is
 driven by one season vs durably distributed.]

## What the LL gate said vs what bracket points said
[Compare the gate verdict from PR 12 (NO-GO; w_opt=0.98, headroom 0)
 with the bracket-points verdict here. If they agree (NO-GO either
 way), the LL gate was right for plain BT specifically. If they
 disagree (LL NO-GO but bracket points CLEAR / MARGINAL), the LL
 gate is screening unsoundly and future stage-1 candidates should
 be evaluated on bracket points directly even when the LL gate
 says no.]

## Lesson
[What this implies for future stage-1 / ensemble experiments and
 for the active queue.]

## Files of record
```

- [ ] **Step 2: Update `TODO.md`**

Move active queue item #2 to:
- "Done" if CLEAR (with note that production swap is a separate followup).
- "Tried and rejected" if NO-GO or MARGINAL.

Keep item #1 (v4 gap audit) and the rest of the queue unchanged.

- [ ] **Step 3: Force-add output artifacts**

```bash
git add -f output/pairwise_v4bt_w*.csv output/pairwise_v9c_v4bt_w*.csv \
            output/bt_bracket_sweep.json output/bt_bracket_log.txt
```

- [ ] **Step 4: Verify ASCII on all written .md files**

```bash
for f in docs/superpowers/specs/2026-05-04-bt-bracket-points-design.md \
         docs/superpowers/plans/2026-05-04-bt-bracket-points.md \
         docs/notes/2026-05-04-bt-bracket-points.md \
         TODO.md; do
    python -c "open('$f').read().encode('ascii')" && echo "$f OK" || echo "$f FAIL"
done
```

- [ ] **Step 5: Run pytest**

```bash
pytest -v tests/test_sweep_bt_bracket_points.py
pytest -v   # full suite
```

State the runtime + pass count in the final commit message.

**Phase 3 exit criterion:** findings + TODO updated, all tests green, ASCII-clean, ready for PR.

---

## Risks (carried from spec)

1. **No cell beats baseline by >= +10.** Most likely outcome. Tightens the conclusion: plain BT does not help on bracket points either.
2. **Anchor at w=1.00 fails to reproduce baseline.** Halt and investigate.
3. **One-season skew.** A +30 delta could be one fluky bracket. Findings note must report W/L/T and the biggest single-season swing.
4. **v9-C re-runs non-deterministic.** Compare against the static `pairwise_v9c_v4_baseline.csv` reference, not against a re-run.

## Out-of-scope (carried from spec)

- Re-tuning v9-C's hyperparams.
- HBT cells against bracket points.
- BT-as-feature for v9-C (already tested NO-GO on PR 13).
- Per-season weight tuning.
- Production swap.
