# v9-C Clean Re-run Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Re-evaluate v9-C vs v8 stage-2 correctors against clean v4 (PR 21 output), apply revert-or-stay decision, document findings, expand step-5 marginal-rejections list.

**Architecture:** Re-run two existing LOSO drivers (`train_stage2.py`, `sweep_v9_weights.py` with `V9_FEATURE_SET=v9c`) on the clean `output/pairwise_v4.csv`. Add one small post-processing script (`v9c_per_season_breakdown.py`) for per-season W/L. Decision matrix on best-cell delta vs clean v8: `> 0` stays in production, `<= 0` reverts via `predict_2026_stage2.py`.

**Tech Stack:** Python 3.11, XGBoost (existing), pandas, pytest. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-05-04-v9c-clean-rerun-design.md` (commit 76e850d).

**Worktree:** `.claude/worktrees/feat-v9c-clean-rerun/` (branch `feat/v9c-clean-rerun`).

---

## File Structure

| File | Action | Responsibility |
|---|---|---|
| `data/raw/march-machine-learning-2026/` | Create junction | Bridge to main repo's Kaggle data dir (gitignored). |
| `output/pairwise_v8.csv` | Overwrite | Clean v8 LOSO OOF. Tracked artifact -- force-add. |
| `output/v9c_sweep/` | Overwrite | 15 per-cell pairwise CSVs (gitignored dir). |
| `output/v9c_sweep_results.csv` | Overwrite | Sweep results table (local-only). |
| `output/pairwise_v9.csv` | Overwrite | Winning v9-C cell's per-cell pairwise (tracked, force-add). |
| `output/v9c_clean_per_season.csv` | Create | New: per-season W/L breakdown (local-only). |
| `output/pairwise_probs.json` | Conditionally restore | If revert: re-written by `predict_2026_stage2.py`. Tracked. |
| `src/v9c_per_season_breakdown.py` | Create | New post-processing script (~50 LOC). |
| `tests/test_v9c_per_season_breakdown.py` | Create | One smoke test on synthetic 2-season fixture. |
| `docs/notes/2026-05-04-v9c-clean-rerun.md` | Create | Findings note (mirrors PR 22 / PR 21 structure). |
| `TODO.md` | Modify | Mark step 5 item 1 done; expand marginal-rejections list. |

---

## Phase 0: Worktree data setup

### Task 0.1: Create data junction so trainers can find Kaggle CSVs

**Files:** `data/raw/march-machine-learning-2026/` (new junction)

The worktree was created without the gitignored `data/raw/march-machine-learning-2026/` subdirectory. All scripts we'll run reference this path with `Path("data/raw/march-machine-learning-2026")` as a hardcoded relative root. We junction it to the main repo's copy of that subdir using a Windows directory junction, the safe primitive per the user's memory note (avoid PowerShell `DirectoryInfo.Delete()` on junctions; use `cmd //c rmdir` or `git worktree remove` to clean up).

- [ ] **Step 1: Verify the junction target exists in main repo**

```bash
ls "C:/Users/alden/MarchMadness/data/raw/march-machine-learning-2026/" | head -3
```

Expected output: filenames including `MNCAATourneySeeds.csv`, `MNCAATourneySlots.csv`, `MNCAATourneyCompactResults.csv`. If the main-repo path is empty or missing, halt and investigate — the recovery state of `data/` is broken.

- [ ] **Step 2: Verify the junction destination does NOT exist in worktree**

```bash
ls "C:/Users/alden/MarchMadness/.claude/worktrees/feat-v9c-clean-rerun/data/raw/march-machine-learning-2026/" 2>&1 | head -1
```

Expected output: `ls: cannot access ...: No such file or directory`. If it DOES exist (someone bootstrapped already), skip Step 3 and continue at Step 4 to verify access.

- [ ] **Step 3: Create the directory junction**

```bash
cmd //c "mklink /J C:\Users\alden\MarchMadness\.claude\worktrees\feat-v9c-clean-rerun\data\raw\march-machine-learning-2026 C:\Users\alden\MarchMadness\data\raw\march-machine-learning-2026"
```

Expected output: `Junction created for ... <<===>> ...`

- [ ] **Step 4: Verify access through the junction**

```bash
cd "C:/Users/alden/MarchMadness/.claude/worktrees/feat-v9c-clean-rerun" && wc -l "data/raw/march-machine-learning-2026/MNCAATourneySeeds.csv"
```

Expected output: a non-zero line count (e.g. `7000+ data/raw/...`). If 0 or "No such file", the junction is broken — `cmd //c rmdir` it and retry Step 3.

- [ ] **Step 5: Verify git status is unchanged by the junction**

```bash
git -C "C:/Users/alden/MarchMadness/.claude/worktrees/feat-v9c-clean-rerun" status --short
```

Expected output: empty (the junction target is in `data/raw/` which is gitignored). If new files appear, the junction is somehow not respected by gitignore — inspect before proceeding.

No commit for Phase 0 — junction is local environment setup, not branch state.

---

## Phase 1: Per-season breakdown script (TDD)

### Task 1.1: Write the failing smoke test

**Files:** Create `tests/test_v9c_per_season_breakdown.py`.

The script reads two pairwise CSVs (one for v8, one for v9-C winning cell), uses `score_chalk_brackets.score_pairwise_path()` to score each, joins on season, and emits a per-season comparison CSV. The test exercises the end-to-end CLI on a tiny synthetic fixture covering 2 seasons.

- [ ] **Step 1: Write the test**

```python
"""Smoke test for v9c_per_season_breakdown.py.

Patches score_pairwise_path to return synthetic per-season totals so
the test does not depend on full Kaggle Tourney CSV files being
present. The fidelity of score_pairwise_path itself is covered by
tests/test_score_chalk_brackets.py.
"""
import sys
from pathlib import Path

import pandas as pd
import pytest


def test_v9c_per_season_breakdown_cli_smoke(tmp_path, monkeypatch):
    """main() writes a CSV with the documented schema and correct deltas."""
    v9c_csv = tmp_path / "v9c.csv"
    v8_csv = tmp_path / "v8.csv"
    out_csv = tmp_path / "out.csv"
    # Schema-only fixtures; scoring is patched.
    pd.DataFrame([{"season": 2024, "team_a": 1101, "team_b": 1102,
                   "p_a_wins": 0.6}]).to_csv(v9c_csv, index=False)
    pd.DataFrame([{"season": 2024, "team_a": 1101, "team_b": 1102,
                   "p_a_wins": 0.55}]).to_csv(v8_csv, index=False)

    import src.v9c_per_season_breakdown as mod

    def fake_score(path):
        # Path discriminator: v9c fixture has "v9c" in name; v8 doesn't.
        if "v9c" in str(path):
            return {"total_pts": 100.0,
                    "per_season_pts": {2024: 50.0, 2023: 50.0}}
        return {"total_pts": 90.0,
                "per_season_pts": {2024: 60.0, 2023: 30.0}}

    monkeypatch.setattr(mod, "score_pairwise_path", fake_score)
    monkeypatch.setattr(sys, "argv", [
        "_",
        "--v9c-pairwise", str(v9c_csv),
        "--v8-pairwise", str(v8_csv),
        "--output", str(out_csv),
    ])

    mod.main()

    out = pd.read_csv(out_csv)
    assert list(out.columns) == ["season", "v8_pts", "v9c_pts",
                                 "delta", "winner"]
    assert sorted(out["season"].tolist()) == [2023, 2024]
    rows = {int(r.season): r for _, r in out.iterrows()}
    # 2024: v9c 50, v8 60 -> delta -10, winner v8
    assert rows[2024]["delta"] == pytest.approx(-10.0)
    assert rows[2024]["winner"] == "v8"
    # 2023: v9c 50, v8 30 -> delta +20, winner v9c
    assert rows[2023]["delta"] == pytest.approx(20.0)
    assert rows[2023]["winner"] == "v9c"
```

- [ ] **Step 2: Run the test, verify it fails**

```bash
cd "C:/Users/alden/MarchMadness/.claude/worktrees/feat-v9c-clean-rerun"
python -m pytest tests/test_v9c_per_season_breakdown.py -v 2>&1 | tail -10
```

Expected: FAIL with `ModuleNotFoundError: No module named 'src.v9c_per_season_breakdown'`.

### Task 1.2: Implement the script

**Files:** Create `src/v9c_per_season_breakdown.py`.

- [ ] **Step 1: Write the script**

```python
"""Per-season bracket-points breakdown for v9-C clean re-run.

The v9-C 15-cell sweep driver (sweep_v9_weights.py) emits 22-season
totals only. This script reads the v9-C winning cell's per-cell
pairwise CSV plus the clean v8 pairwise CSV, scores each season
individually via score_pairwise_path, and writes a per-season
comparison CSV. Used in the recovery-step-5 v9-C clean-rerun
findings note to show v9-C's W/L spread (matches PR 9's "6W-3L-13T"
profile reporting).

Inputs (CLI args, all required):
  --v9c-pairwise   Path to v9-C winning cell's pairwise CSV
                   (e.g. output/v9c_sweep/pairwise_v9_WU1.25_WM0.csv).
  --v8-pairwise    Path to clean v8 pairwise CSV
                   (e.g. output/pairwise_v8.csv).
  --output         Output CSV path
                   (e.g. output/v9c_clean_per_season.csv).

Output schema:
  season, v8_pts, v9c_pts, delta, winner
where delta = v9c_pts - v8_pts and winner in {'v8', 'v9c', 'tie'}
with tie when abs(delta) < 0.5.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.score_chalk_brackets import score_pairwise_path

TIE_THRESHOLD = 0.5


def _winner(delta: float) -> str:
    if abs(delta) < TIE_THRESHOLD:
        return "tie"
    return "v9c" if delta > 0 else "v8"


def build_breakdown(v9c_pairwise: str, v8_pairwise: str) -> pd.DataFrame:
    """Score each pairwise CSV per-season, return a comparison DataFrame."""
    v9c = score_pairwise_path(v9c_pairwise)["per_season_pts"]
    v8 = score_pairwise_path(v8_pairwise)["per_season_pts"]
    seasons = sorted(set(v9c) | set(v8))
    rows = []
    for s in seasons:
        v8_pts = float(v8.get(s, 0.0))
        v9c_pts = float(v9c.get(s, 0.0))
        delta = v9c_pts - v8_pts
        rows.append({
            "season": int(s),
            "v8_pts": v8_pts,
            "v9c_pts": v9c_pts,
            "delta": delta,
            "winner": _winner(delta),
        })
    return pd.DataFrame(rows, columns=["season", "v8_pts", "v9c_pts",
                                       "delta", "winner"])


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--v9c-pairwise", required=True)
    parser.add_argument("--v8-pairwise", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    df = build_breakdown(args.v9c_pairwise, args.v8_pairwise)
    df.to_csv(args.output, index=False)
    print(f"Wrote {len(df)} seasons to {args.output}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the test, verify it passes**

```bash
cd "C:/Users/alden/MarchMadness/.claude/worktrees/feat-v9c-clean-rerun"
python -m pytest tests/test_v9c_per_season_breakdown.py -v 2>&1 | tail -10
```

Expected: 1 passed.

- [ ] **Step 3: Run the full test suite to verify no regressions**

```bash
python -m pytest -q 2>&1 | tail -10
```

Expected: existing 137+ tests pass, plus 1 new = 138+ passed, 0 failed.

- [ ] **Step 4: Commit script + test**

```bash
git add src/v9c_per_season_breakdown.py tests/test_v9c_per_season_breakdown.py
git commit -m "$(cat <<'EOF'
feat(v9c-clean-rerun): per-season bracket-points breakdown script

Reads v9-C winning cell's per-cell pairwise CSV plus clean v8
pairwise CSV, scores each season via score_pairwise_path, emits
per-season comparison (season, v8_pts, v9c_pts, delta, winner).
Fills the gap that the existing 15-cell sweep driver only emits
22-season totals; the findings note needs the per-season W/L
spread to characterize the result's durability.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Phase 2: Re-run v8 LOSO baseline on clean v4

### Task 2.1: Snapshot current (leaky-baseline) `pairwise_v8.csv` locally

**Files:** `output/pairwise_v8_pre_clean_rerun.csv` (local-only, NOT committed)

Snapshot before overwriting so the findings note can quote both numbers, and so we have an audit trail if the clean run produces a surprise.

- [ ] **Step 1: Verify pre-rerun pairwise_v8.csv exists and snapshot it**

```bash
cd "C:/Users/alden/MarchMadness/.claude/worktrees/feat-v9c-clean-rerun"
wc -l output/pairwise_v8.csv  # expect ~48,466 lines
cp output/pairwise_v8.csv output/pairwise_v8_pre_clean_rerun.csv
ls -la output/pairwise_v8_pre_clean_rerun.csv
```

Expected: snapshot file exists with same byte count as `pairwise_v8.csv`.

### Task 2.2: Run `train_stage2.py` end-to-end on clean v4

**Files:** Overwrite `output/pairwise_v8.csv` (tracked artifact).

`train_stage2.py:main()` does the full double-LOSO build via `build_v8_pairwise()`. Reads `output/pairwise_v4.csv`, writes `output/pairwise_v8.csv`. Expected runtime ~2-3 minutes.

- [ ] **Step 1: Confirm input pairwise_v4.csv is the clean version**

```bash
cd "C:/Users/alden/MarchMadness/.claude/worktrees/feat-v9c-clean-rerun"
wc -l output/pairwise_v4.csv
git log -1 --oneline output/pairwise_v4.csv
```

Expected: 48,466 lines (header + 48,465 data); last commit on this file should be `f204dbf feat(v4-clean-loso-regen): regen pairwise_v4 under clean pipeline` (PR 21).

- [ ] **Step 2: Run train_stage2.py**

```bash
python -u src/train_stage2.py 2>&1 | tee output/v8_clean_rerun.log
```

Expected: completes in 2-3 min; final stdout shows "Saved output/pairwise_v8.csv" or equivalent. If the script fails with a missing-data error, the Phase 0 junction is missing — re-do Phase 0 Task 0.1.

- [ ] **Step 3: Verify the new pairwise_v8.csv is shaped right**

```bash
wc -l output/pairwise_v8.csv  # expect ~48,466
diff <(head -1 output/pairwise_v8.csv) <(head -1 output/pairwise_v8_pre_clean_rerun.csv)
```

Expected: same line count as before; same header (schema unchanged).

- [ ] **Step 4: Sanity-diff vs leaky baseline**

```bash
python -c "
import pandas as pd
old = pd.read_csv('output/pairwise_v8_pre_clean_rerun.csv')
new = pd.read_csv('output/pairwise_v8.csv')
print(f'old rows: {len(old):,}; new rows: {len(new):,}')
m = old.merge(new, on=['season','team_a','team_b'], suffixes=('_old','_new'))
import numpy as np
diff = (m['p_a_wins_new'] - m['p_a_wins_old']).abs()
print(f'mean abs delta: {diff.mean():.4f}; max: {diff.max():.4f}; '
      f'pairs with delta > 0.05: {int((diff > 0.05).sum())}')
"
```

Expected: non-trivial deltas (mean abs delta likely 0.02-0.10, given v4 LL shifted +0.122). If mean abs delta < 0.001, the clean run produced ~identical v8 -- that would mean the v8 trainer is insensitive to v4 input, which is unexpected; pause and inspect.

- [ ] **Step 5: Commit clean pairwise_v8.csv (force-add per existing pattern)**

```bash
git add output/pairwise_v8.csv
git commit -m "$(cat <<'EOF'
data(v9c-clean-rerun): regen pairwise_v8.csv under clean v4 baseline

train_stage2.py double-LOSO over 22 seasons against the clean
pairwise_v4.csv (PR 21). Replaces the leaky-baseline v8 OOF.
Stage-2 trainer code unchanged (PR 6/8 defaults; not tuned).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Phase 3: Re-run v9-C 15-cell sweep on clean v4

### Task 3.1: Run the v9-C sweep with auto v8 comparison

**Files:** Overwrite `output/v9c_sweep/`, `output/v9c_sweep_results.csv` (gitignored). Driver auto-loads `output/pairwise_v8.csv` (clean, from Phase 2) for the comparison and anchor sanity gate.

- [ ] **Step 1: Run the sweep**

```bash
cd "C:/Users/alden/MarchMadness/.claude/worktrees/feat-v9c-clean-rerun"
V9_FEATURE_SET=v9c python -u src/sweep_v9_weights.py 2>&1 | tee output/v9c_clean_sweep_run.log
```

Expected runtime: 5-7 minutes (15 cells x 22 LOSO seasons). Final stdout shows:
- A 15-row results table sorted by `total_brkt_pts` descending.
- `v8 baseline:    NNNN.N pts` (clean v8's 22-season total).
- `anchor (1, 0): NNNN.N pts (delta +/- N.NN)`.
- Either `Anchor cell reproduces v8 within 5 pts -- sweep is valid.` OR a `WARNING: anchor cell does not reproduce v8 within 5 pts ...`.
- A `WINNER:` or `NO WINNER:` line for the best cell.

- [ ] **Step 2: Inspect the anchor sanity gate result**

```bash
grep -E "^(WARNING|Anchor|v8 baseline|anchor|best cell|WINNER|NO WINNER):" output/v9c_clean_sweep_run.log
```

If `WARNING: anchor cell does not reproduce v8 within 5 pts` appears: HALT. Inspect per-game LL/Acc in the log (the trainer should match v8's clean numbers to ~3 decimals; if not, debug `train_upset_model.py` parameterization before trusting any cell). Document the WARNING in the findings note's anchor-sanity section but do NOT proceed to Phase 5's revert/stay decision until the trainer is sane.

If anchor is fine: continue.

- [ ] **Step 3: Verify the sweep results CSV is shaped right**

```bash
wc -l output/v9c_sweep_results.csv  # expect 16 (header + 15 cells)
head -1 output/v9c_sweep_results.csv
```

Expected: 16 lines. Header should include `w_upset, w_miss, total_brkt_pts, ll_loso_weighted_mean, acc_loso_weighted_mean, pairwise_csv` (the existing schema; column order may vary).

- [ ] **Step 4: Identify the winning cell**

```bash
python -c "
import pandas as pd
df = pd.read_csv('output/v9c_sweep_results.csv')
df = df.sort_values('total_brkt_pts', ascending=False).reset_index(drop=True)
print(df.to_string(index=False))
print()
print(f'Best cell: W_UPSET={df.iloc[0].w_upset}, W_MISS={df.iloc[0].w_miss}, total={df.iloc[0].total_brkt_pts:.1f}')
print(f'pairwise_csv: {df.iloc[0].pairwise_csv}')
"
```

Expected: prints the full sorted table and the winning cell's `pairwise_csv` path. Record `WINNER_W_UPSET` and `WINNER_W_MISS` for Phase 4.

---

## Phase 4: Per-season W/L breakdown for the winning cell

### Task 4.1: Run the breakdown script on winner + clean v8

**Files:** Create `output/v9c_clean_per_season.csv` (local-only, NOT committed).

- [ ] **Step 1: Run the breakdown script**

Substitute the winner's pairwise CSV path from Task 3.1 Step 4. The path follows the template `output/v9c_sweep/pairwise_v9_WU{u}_WM{m}.csv`; for example, if PR 9's winner (W_U=1.25, W_M=0.0) repeats, the path is `output/v9c_sweep/pairwise_v9_WU1.25_WM0.csv`.

```bash
cd "C:/Users/alden/MarchMadness/.claude/worktrees/feat-v9c-clean-rerun"
python -u src/v9c_per_season_breakdown.py \
  --v9c-pairwise "output/v9c_sweep/pairwise_v9_WU<U>_WM<M>.csv" \
  --v8-pairwise output/pairwise_v8.csv \
  --output output/v9c_clean_per_season.csv 2>&1 | tee -a output/v9c_clean_sweep_run.log
```

Replace `<U>` and `<M>` with the winner's weights from Task 3.1.

Expected: prints 22 rows, one per season; `season, v8_pts, v9c_pts, delta, winner` columns. Output CSV has 22 data rows + 1 header.

- [ ] **Step 2: Compute the W/L tally**

```bash
python -c "
import pandas as pd
df = pd.read_csv('output/v9c_clean_per_season.csv')
counts = df['winner'].value_counts().to_dict()
total_delta = df['delta'].sum()
print(f'W/L: {counts}')
print(f'aggregate delta (v9c - v8): {total_delta:+.2f} brkt pts')
"
```

Expected: prints win/loss/tie counts that sum to 22, plus the aggregate delta (which should match the sweep's reported best-cell delta).

---

## Phase 5: Apply decision matrix

The aggregate delta from Phase 4 Step 2 is the production-decision input.

### Task 5.1: Branch on the decision

- [ ] **Step 1: Check the delta and the anchor gate**

Read `output/v9c_clean_sweep_run.log` and confirm:
- Anchor cell `abs(delta) <= 5` pts (sanity gate passed).
- Winning cell's `delta_vs_v8` value (= aggregate delta from Phase 4).

Apply the matrix:
- `delta > 0`: skip Task 5.2; go to Task 5.3 (stay branch).
- `delta <= 0`: skip Task 5.3; go to Task 5.2 (revert branch).
- Anchor `abs(delta) > 5`: HALT. Findings note documents the anchor failure and recommends debugging `train_upset_model.py` before any production action; revert/stay is deferred until trainer is verified sane.

### Task 5.2: REVERT branch (only if `delta <= 0`)

**Files:** Overwrite `output/pairwise_probs.json` (tracked artifact) via `predict_2026_stage2.py`.

- [ ] **Step 1: Snapshot current (v9-C) pairwise_probs.json**

```bash
cp output/pairwise_probs.json output/pairwise_probs_pre_revert.json
```

Local-only; not committed. Audit trail.

- [ ] **Step 2: Re-run predict_2026_stage2.py to restore v8-corrected output**

```bash
python -u src/predict_2026_stage2.py 2>&1 | tee output/predict_2026_stage2_revert_run.log
```

Expected: writes `output/pairwise_probs_v8_2026.json` (versioned snapshot, gitignored) and overwrites `output/pairwise_probs.json` (tracked).

- [ ] **Step 3: Verify the revert touched both files**

```bash
diff output/pairwise_probs.json output/pairwise_probs_v8_2026.json && echo "files match"
diff output/pairwise_probs.json output/pairwise_probs_pre_revert.json && echo "ERROR: revert was no-op"
```

Expected: first diff prints "files match" (canonical was overwritten with v8 contents); second diff prints non-empty output (the revert actually changed the canonical artifact). If second diff prints "ERROR: revert was no-op", the v9-C and v8 outputs were identical — investigate, do not commit.

- [ ] **Step 4: Commit the revert**

```bash
git add output/pairwise_probs.json
git commit -m "$(cat <<'EOF'
data(v9c-clean-rerun): revert pairwise_probs.json to v8-corrected output

v9-C lost to v8 on the clean v4 baseline (best-cell delta <= 0).
Restoring v8 stage-2 corrector as the production model. v9-C's
+43 vs v8 swap-in evidence (PR 9) was driven by the Vegas leak
in v4's stage-1 features (PR 19, PR 21, PR 22).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

(After this step, skip Task 5.3 and proceed to Phase 6.)

### Task 5.3: STAY branch (only if `delta > 0`)

**Files:** Overwrite `output/pairwise_v9.csv` (tracked artifact) with the winning cell's pairwise CSV.

- [ ] **Step 1: Copy the winning cell's pairwise CSV over the canonical artifact**

Substitute the winner's path from Task 3.1 Step 4.

```bash
cp "output/v9c_sweep/pairwise_v9_WU<U>_WM<M>.csv" output/pairwise_v9.csv
wc -l output/pairwise_v9.csv  # expect ~48,466
```

- [ ] **Step 2: Verify pairwise_probs.json does NOT need re-running**

```bash
ls -la output/pairwise_probs.json output/pairwise_probs_v9c_2026.json
```

`pairwise_probs.json` should already be the v9-C-corrected one from PR 10's swap (timestamp matches `pairwise_probs_v9c_2026.json`). If it doesn't, re-run `python src/predict_2026_v9c.py` separately.

If the winning cell's weights differ from PR 10's `(W_U=1.25, W_M=0.0)`, then `pairwise_probs.json` reflects the old weights and should be re-applied. In that case run:

```bash
python -u src/predict_2026_v9c.py 2>&1 | tee output/predict_2026_v9c_stay_run.log
```

Note: `predict_2026_v9c.py` hardcodes `PROD_W_UPSET=1.25, PROD_W_MISS=0.0` — if the new winner is a different cell, also edit those constants in `src/predict_2026_v9c.py` before running. Commit the constant change in this step's commit.

- [ ] **Step 3: Commit the stay**

```bash
git add output/pairwise_v9.csv
# Conditionally include if predict_2026_v9c.py was edited and re-run:
# git add src/predict_2026_v9c.py output/pairwise_probs.json
git commit -m "$(cat <<'EOF'
data(v9c-clean-rerun): regen pairwise_v9.csv under clean v4 baseline

v9-C beats v8 on the clean v4 baseline (best-cell delta > 0).
v9-C stays in production. Winning cell preserved as the canonical
pairwise_v9.csv artifact.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Phase 6: Findings note + TODO.md update

### Task 6.1: Write the findings note

**Files:** Create `docs/notes/2026-05-04-v9c-clean-rerun.md`.

The note has 9 sections per the spec. Use the data captured in Phases 2–5:
- v8 baseline number from `output/v9c_clean_sweep_run.log` (line `v8 baseline: NNNN.N pts`).
- 15-cell sweep table from `output/v9c_sweep_results.csv`.
- Per-season breakdown from `output/v9c_clean_per_season.csv`.
- Anchor sanity from log (`anchor (1, 0): ... delta ...`).
- Pre/post v8 delta from Task 2.2 Step 4.
- Production state change from Phase 5 (revert vs stay).

- [ ] **Step 1: Write the note**

Template (substitute concrete numbers from the run; keep section headers verbatim):

```markdown
# v9-C Clean Re-run -- Findings

**Date:** 2026-05-04
**Branch:** feat/v9c-clean-rerun
**Verdict:** **<STAY|REVERT>.** Best v9-C cell delta vs clean v8: <+/-N.NN brkt pts>.
**Spec:** `docs/superpowers/specs/2026-05-04-v9c-clean-rerun-design.md`
**Plan:** `docs/superpowers/plans/2026-05-04-v9c-clean-rerun.md`
**Recovery context:** TODO.md "CONTAMINATION DISCOVERED 2026-05-04" -> step 5, item 1.

## TL;DR

<One paragraph: clean v8 baseline = NNNN.N pts (was NNNN.N leaky); v9-C
best cell at (W_U=X.XX, W_M=Y.YY) = NNNN.N pts; delta vs clean v8 =
+/-N.NN; W/L over 22 seasons = AW-BL-CT; production action = stay/revert>.

## Methods

- Input: `output/pairwise_v4.csv` (clean, PR 21; 48,465 rows).
- v8 baseline: `python src/train_stage2.py` -> `output/pairwise_v8.csv`.
- v9-C sweep: `V9_FEATURE_SET=v9c python src/sweep_v9_weights.py` -> 15
  cells in `output/v9c_sweep/`, results in `output/v9c_sweep_results.csv`.
- Per-season breakdown: `src/v9c_per_season_breakdown.py`.
- Hyperparameter confound: v9-C/v8 trainers reuse PR 6/8/9 untuned XGB
  defaults (no leak-baseline confound). `pairwise_v4.csv` carries PR 21's
  tuned-XGB-on-leaky-baseline confound (documented effect <0.02 LL).

## Clean v8 baseline

| metric | leaky (PR 9) | clean (this PR) | delta |
|---|---|---|---|
| 22-season total brkt pts | 2670 | NNNN | +/-NN |

<discussion of how v8's score shifted under clean v4>

## 15-cell v9-C sweep

| W_UPSET | W_MISS | total_brkt_pts | delta vs v8 | LL | Acc |
|---|---|---|---|---|---|
| ... | ... | ... | ... | ... | ... |

(Sorted by total_brkt_pts descending. PR 9 winning cell highlighted.)

## Winning cell per-season W/L

| season | v8 pts | v9c pts | delta | winner |
|---|---|---|---|---|
| ... | ... | ... | ... | ... |

W/L over 22 seasons: AW-BL-CT. <interpretation: durable / fragile?>

## Anchor sanity check

(W_UPSET=1.0, W_MISS=0.0): NNNN.N pts vs clean v8 NNNN.N pts; delta
+/-N.NN brkt pts. <Within 5-pt gate / WARNING fired.>

## Discussion

<Connect to PR 22's clean upset-detection numbers (clean v4: 15.3% vs
Vegas 17.5%). If v9-C lost: v9-C was correcting noise rather than
signal -- v8's content-blind stage-2 outperforms upset-aware stage-2
when v4's upset signal is below random. If v9-C won: surprising; the
upset signal v4 still has is enough for v9-C's correction to add
value despite the headline "v4 has no upset edge" finding.>

## Production state change

<Stay: v9-C remains in production at cell (X, Y). pairwise_v9.csv
updated. predict_2026_v9c.py constants <unchanged|updated to (X, Y)>.>

OR

<Revert: v9-C reverted to v8. pairwise_probs.json restored via
predict_2026_stage2.py. predict_2026_v9c.py retained for audit; can
be removed in a separate cleanup PR if v9-C is fully retired.>

## TODO.md update

Step 5 item 1 marked done. **Marginal-rejections list expanded** to
cover candidates whose original rejection deltas were within the
+0.122 LL leak noise floor and weren't named in the original
recovery roadmap:

- Plain BT standalone (PR 12): standalone LL 0.565 vs leaky v4
  0.437 = -0.128 weaker (gate failed). Vs clean v4 0.5588 =
  ~tied. LL-blend gate likely flips PASS.
- Feature-view ensemble PEER_A/B (PR 14): PEER_A LL 0.5720 vs
  leaky v4 0.4345 = +0.1375 (5.5x clause-1 tolerance). Vs clean
  v4 0.5588 = +0.013 (within tolerance). Clause 1 likely flips
  PASS.
- HBT (PR 16): all 7 sigma cells failed gate clauses 2/3; HBT
  LL 0.619-0.757. Vs clean v4 0.5588: gap shrinks but HBT still
  weaker.
- Colley (PR 15): clause-2 delta +0.0053 LL on subset.
- Massey-decay hl=14d (PR 15): clause-2 delta +0.0057 LL on
  subset.

## Follow-ups (priority order, cheapest first)

1. Plain BT standalone re-eval (~30 min compute).
2. PEER_A/B feature-view ensemble re-eval (~20 min).
3. Colley + Massey-decay hl=14d re-eval (~30 min combined).
4. HBT re-eval (~5 min).
5. BT-as-feature for v9-C re-eval (named in original roadmap).
6. v9 weight-sweep family re-eval -- partly subsumed by this PR's
   v9-C 15-cell sweep on clean v4 (winning cell here is the
   v9-C answer; v9-B-grid-on-clean-v4 is a separate cheap PR).
7. 538 audit follow-up (parked on `feat/v4-gap-audit-fte`).
```

- [ ] **Step 2: Verify the note has all 9 sections**

```bash
grep -nE "^## " docs/notes/2026-05-04-v9c-clean-rerun.md
```

Expected: 9 H2 headers (TL;DR, Methods, Clean v8 baseline, 15-cell v9-C sweep, Winning cell per-season W/L, Anchor sanity check, Discussion, Production state change, TODO.md update, Follow-ups). If 10 sections appear because TL;DR is split, that's fine.

### Task 6.2: Update TODO.md

**Files:** Modify `TODO.md` (top-level, tracked).

- [ ] **Step 1: Edit step 5 item 1 to "DONE"**

Find the line in `TODO.md` matching:
```
5. **Re-run the swap-decided / swap-candidate evaluations against
   the clean baseline.** **Now the immediate next PR.** Priority order:
   - **v9-C production swap** (currently deployed -- top priority).
```

Replace the bullet `- **v9-C production swap** (currently deployed -- top priority).` with (substitute the actual verdict + delta):

```
- **[DONE -- PR <pending>]** v9-C production swap re-eval. Best cell
  (W_U=X.XX, W_M=Y.YY) at NNNN.N brkt pts vs clean v8 NNNN.N
  (delta +/-N.NN). <Verdict: stays in production / reverted to v8>.
  W/L over 22 seasons: AW-BL-CT. Findings:
  `docs/notes/2026-05-04-v9c-clean-rerun.md`.
```

- [ ] **Step 2: Add the expanded marginal-rejections list to step 5**

Find the existing marginal-rejections paragraph in step 5:

```
- The "marginal" rejections in `Tried and rejected` whose deltas
  were within ~0.05 LL or ~30 brkt pts of v4 (BT-as-feature at
  -0.0015 LL; v9 weight-sweep family at +18 to +20 pts).
```

Replace with:

```
- The "marginal" rejections in `Tried and rejected` whose deltas
  were within the +0.122 LL leak noise floor of v4. Two named in
  the original roadmap (BT-as-feature at -0.0015 LL; v9 weight-
  sweep family at +18 to +20 pts). **Five more added by the v9-C
  re-eval (recovery step 5 item 1) findings:**
    - Plain BT standalone (PR 12): standalone LL 0.565 = ~tied
      with clean v4 0.5588; LL-blend gate likely flips PASS.
    - Feature-view ensemble PEER_A/B (PR 14): PEER_A delta vs
      v4 was +0.1375 vs leaky; +0.013 vs clean (within
      5x clause-1 tolerance); clause 1 likely flips PASS.
    - HBT (PR 16): standalone LL 0.619-0.757; gap to clean v4
      shrinks but HBT still weaker.
    - Colley (PR 15): clause-2 delta +0.0053 LL.
    - Massey-decay hl=14d (PR 15): clause-2 delta +0.0057 LL.
```

- [ ] **Step 3: Verify the edit didn't break the markdown structure**

```bash
grep -nE "^## |^### |^[0-9]+\." TODO.md | head -30
```

Expected: existing headers and numbered items preserved; the v9-C re-eval status line appears as a child of step 5; the expanded marginal-rejections paragraph is in place.

- [ ] **Step 4: Commit findings + TODO update**

```bash
git add docs/notes/2026-05-04-v9c-clean-rerun.md TODO.md
git commit -m "$(cat <<'EOF'
docs(v9c-clean-rerun): findings + TODO update -- recovery step 5 item 1

<Stay|Revert>. Best v9-C cell at (W_U=X.XX, W_M=Y.YY) at NNNN.N
brkt pts vs clean v8 NNNN.N (delta +/-N.NN). W/L AW-BL-CT.

TODO.md updates:
- Mark step 5 item 1 done with the verdict.
- Expand marginal-rejections list with 5 candidates not named in the
  original recovery roadmap (plain BT, PEER_A/B, HBT, Colley,
  Massey-decay) -- their original gate failures fall within the
  +0.122 LL leak noise floor.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Phase 7: Final verification

### Task 7.1: Confirm full test suite passes

- [ ] **Step 1: Run pytest**

```bash
cd "C:/Users/alden/MarchMadness/.claude/worktrees/feat-v9c-clean-rerun"
python -m pytest -q 2>&1 | tail -10
```

Expected: 138+ passed (existing + 1 new), 0 failed.

### Task 7.2: Verify branch state

- [ ] **Step 1: Show all commits on the branch**

```bash
git log --oneline main..HEAD
```

Expected (revert branch):
```
<hash> docs(v9c-clean-rerun): findings + TODO update -- recovery step 5 item 1
<hash> data(v9c-clean-rerun): revert pairwise_probs.json to v8-corrected output
<hash> data(v9c-clean-rerun): regen pairwise_v8.csv under clean v4 baseline
<hash> feat(v9c-clean-rerun): per-season bracket-points breakdown script
76e850d spec(v9c-clean-rerun): re-eval v9-C production swap on clean v4 baseline
```

OR (stay branch):
```
<hash> docs(v9c-clean-rerun): findings + TODO update -- recovery step 5 item 1
<hash> data(v9c-clean-rerun): regen pairwise_v9.csv under clean v4 baseline
<hash> data(v9c-clean-rerun): regen pairwise_v8.csv under clean v4 baseline
<hash> feat(v9c-clean-rerun): per-season bracket-points breakdown script
76e850d spec(v9c-clean-rerun): re-eval v9-C production swap on clean v4 baseline
```

- [ ] **Step 2: Verify clean working tree**

```bash
git status --short
```

Expected: empty (no uncommitted changes; all artifacts either committed or local-only).

- [ ] **Step 3: Verify untracked output files are local-only-as-intended**

```bash
ls output/ | grep -E "^(pairwise_v8_pre_clean_rerun|v9c_clean_per_season|v9c_clean_sweep_run|v8_clean_rerun|pairwise_probs_pre_revert|predict_2026_stage2_revert_run)" 2>&1
```

Expected: lists local-only audit-trail files. None of these are committed (they're under `output/` which is gitignored except for the named tracked files).

### Task 7.3: PR readiness

- [ ] **Step 1: Push branch and open PR**

```bash
git push -u origin feat/v9c-clean-rerun
```

Then via `gh pr create` with a title like `feat(v9c-clean-rerun): <stay|revert> verdict on clean baseline (recovery step 5)` and a body summarizing the verdict, the W/L spread, and the expanded marginal-rejections list.

---

## Summary checklist (completion order)

- [ ] Phase 0: Data junction created.
- [ ] Phase 1: `v9c_per_season_breakdown.py` + test, committed.
- [ ] Phase 2: Clean `pairwise_v8.csv` regenerated, committed.
- [ ] Phase 3: 15-cell v9-C sweep run; anchor gate inspected.
- [ ] Phase 4: Per-season W/L breakdown produced.
- [ ] Phase 5: Decision matrix applied; revert OR stay branch executed and committed.
- [ ] Phase 6: Findings note + TODO.md update committed.
- [ ] Phase 7: Tests green; branch ready for PR.
