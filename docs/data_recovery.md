# Data Directory Recovery Runbook

**Last updated:** 2026-05-04 (second occurrence; first was 2026-05-02).

The `data/` directory tree on this repo has been silently wiped twice now
during git-worktree cleanup operations on Windows. This document is the
operational runbook so it can be diagnosed and recovered fast next time.

## Background: what gets wiped, and why

The repo uses Windows directory junctions (created with `mklink /J`) inside
git worktrees to share `data/` (or its subdirs) across worktrees without
copying multi-GB files. A junction `<worktree>\data` points at the main
repo's `<repo>\data\`. The OS does not strongly distinguish "delete the
junction" from "recursively delete what the junction points at" -- some
delete tools follow the junction and wipe the target.

Two known wipe vectors:

1. **PowerShell `(Get-Item $junction).Delete()`** -- documented in
   `memory/feedback_windows_junction_delete.md`. Despite .NET docs claiming
   `DirectoryInfo.Delete()` on a reparse point removes only the reparse
   point, on Windows 11 + PowerShell 5.1 it has wiped the target's
   contents. Cause of the **2026-05-02** wipe.
2. **Non-git filesystem cleanup of an orphan worktree dir.** When a
   worktree has been removed from git's perspective (`git worktree list`
   no longer shows it) but its on-disk directory remains -- e.g., the
   user manually `git worktree remove`d the wrong path, or a script
   force-deleted the dir with PowerShell or Explorer's "Delete" -- the
   dangling junction inside the orphan dir still points at main repo's
   `data/`, and any recursive delete of the orphan dir wipes the target.
   Suspected cause of the **2026-05-04** wipe (multiple empty worktree
   dirs and the data subdirs all share mtime 19:54).

What's actually deleted: every file under `data/raw/kaggle/`,
`data/raw/march-machine-learning-2026/`, `data/raw/vegas_lines/`, and
`data/cache/`. Tracked files at `data/bracket/2026.pdf`,
`data/raw/bracket_2026.csv`, `data/team_name_overrides.csv`, and
`data/training_data.tar.gz` are also wiped if junction was at `data/`
level (May 2 incident); they survive if junctions were at deeper paths
(May 4 incident left these in place).

## Detection

`git status` flags the wipe IF tracked files were deleted (May 2 case).
For the May 4 case (subdir-only wipe), git status is clean and the wipe
is invisible until a Python script blows up:

```text
FileNotFoundError: data/raw/march-machine-learning-2026/MNCAATourneySeeds.csv
```

**Quick health check.** Run this any time before an experiment, especially
right after worktree cleanup operations:

```bash
ls "C:/Users/alden/MarchMadness/data/raw/march-machine-learning-2026/" | wc -l   # expect ~28
ls "C:/Users/alden/MarchMadness/data/raw/kaggle/"                       | wc -l   # expect ~30
ls "C:/Users/alden/MarchMadness/data/raw/vegas_lines/"                  | wc -l   # expect 23 (ncaabb03..ncaabb25)
ls "C:/Users/alden/MarchMadness/data/training_data.tar.gz"                       # expect ~42 MB
```

If any of these come back empty / missing, run the recovery procedure.

## Inventory: what each dir contains and how to restore it

| Path | Contents | Restore source |
|---|---|---|
| `data/raw/march-machine-learning-2026/` | Kaggle's annual NCAA tourney bundle (MNCAATourney{Seeds,Slots,CompactResults,DetailedResults}.csv, MMasseyOrdinals.csv, etc.) | `data/training_data.tar.gz` (tracked) |
| `data/raw/kaggle/` | Supplemental Kaggle CSVs (538 Ratings, AP Poll, Barttorvik {Home,Away,Neutral}, Heat Check, KenPom, EvanMiya, Coach/Conference/Resume metadata, etc.) | `data/training_data.tar.gz` (tracked) |
| `data/raw/vegas_lines/` | The Prediction Tracker historical betting-line CSVs (`ncaabbYY.csv` for fall year YY, mapped by `_extract_season()` in `src/enhanced_model_v3.py:181-190`; YY=03..25 covers Kaggle seasons 2004..2026) | `data/training_data.tar.gz` (tracked) |
| `data/raw/bracket_2026.csv` | Per-year bracket structure CSV | git checkout (tracked file) |
| `data/team_name_overrides.csv` | Manual team-name → TeamID overrides (loaded by `src/ingest/team_mapping.py`) | git checkout (tracked file) |
| `data/training_data.tar.gz` | The recovery bundle itself, ~42 MB | git checkout (tracked file) |
| `data/bracket/` | Bracket PDFs, including `data/bracket/2026.pdf` | git checkout (tracked) |
| `data/cache/` | Reproducible derived artifacts (Massey, Colley parquet caches, etc.) | Regenerated automatically the next time the producer runs (`src/features/*.py`'s cache loaders) |

## Recovery procedure

### Step 1: Confirm the scope of the wipe

```bash
ls -la "C:/Users/alden/MarchMadness/data/raw/"
ls "C:/Users/alden/MarchMadness/data/raw/march-machine-learning-2026/" 2>&1 | head -3
ls "C:/Users/alden/MarchMadness/data/raw/kaggle/"                       2>&1 | head -3
ls "C:/Users/alden/MarchMadness/data/raw/vegas_lines/"                  2>&1 | head -3
ls "C:/Users/alden/MarchMadness/data/training_data.tar.gz"
git -C "C:/Users/alden/MarchMadness" status --short
```

Note which subdirs are empty vs. which still have content. If
`data/training_data.tar.gz` is also missing, skip ahead to **Step 2a**.

### Step 2: Restore tracked files (if any tracked files were lost)

If `git status` shows deleted tracked files (e.g.,
`data/training_data.tar.gz`, `data/team_name_overrides.csv`,
`data/raw/bracket_2026.csv`, `data/bracket/2026.pdf`):

```bash
git -C "C:/Users/alden/MarchMadness" checkout HEAD -- data/
```

This restores all tracked files under `data/`. Verify
`data/training_data.tar.gz` is back and ~42 MB before continuing.

### Step 2a: If `training_data.tar.gz` is also gone AND `git checkout` cannot recover it

This means the working tree's tracked file was force-deleted somewhere
upstream. Try restoring from git's object database:

```bash
git -C "C:/Users/alden/MarchMadness" show HEAD:data/training_data.tar.gz > "C:/Users/alden/MarchMadness/data/training_data.tar.gz"
ls -la "C:/Users/alden/MarchMadness/data/training_data.tar.gz"
```

If even that fails, the repo's git objects are damaged. Restore from the
GitHub remote:

```bash
git -C "C:/Users/alden/MarchMadness" fetch origin main
git -C "C:/Users/alden/MarchMadness" checkout origin/main -- data/training_data.tar.gz
```

### Step 3: Extract `training_data.tar.gz` to restore the gitignored Kaggle subdirs

The tarball contains three top-level dirs: `kaggle/`,
`march-machine-learning-2026/`, and `vegas_lines/`. Extract into
`data/raw/`:

```bash
tar -xzf "/c/Users/alden/MarchMadness/data/training_data.tar.gz" \
    -C "/c/Users/alden/MarchMadness/data/raw/"
```

Notes:
- Use the MINGW `/c/...` path for `tar`. Bare `C:` paths confuse tar
  (it treats `C` as a remote host).
- `data/raw/` already exists and may contain empty subdirs from the
  wipe; tar populates the dirs and does not collide.
- `data/cache/` is **not** in the tarball. It regenerates automatically
  on next pipeline run (each Massey/Colley cache loader checks
  freshness).

### Step 4: Verify the restore

```bash
ls "C:/Users/alden/MarchMadness/data/raw/march-machine-learning-2026/" | wc -l   # expect ~28
ls "C:/Users/alden/MarchMadness/data/raw/kaggle/"                       | wc -l   # expect ~30
ls "C:/Users/alden/MarchMadness/data/raw/vegas_lines/"                  | wc -l   # expect 23
```

If any count is materially off, re-run Step 3 (the tarball is
deterministic — same extract twice produces the same tree).

### Step 5: Re-create any worktree data junctions you need

For each active worktree where you need data access:

```bash
cmd //c "mklink /J <worktree>\\data\\raw\\march-machine-learning-2026 C:\\Users\\alden\\MarchMadness\\data\\raw\\march-machine-learning-2026"
cmd //c "mklink /J <worktree>\\data\\raw\\kaggle                       C:\\Users\\alden\\MarchMadness\\data\\raw\\kaggle"
cmd //c "mklink /J <worktree>\\data\\raw\\vegas_lines                  C:\\Users\\alden\\MarchMadness\\data\\raw\\vegas_lines"
```

Junction at the **subdir level** (not at `<wt>\data`) is preferred --
it limits the blast radius of a future cleanup-induced wipe to a single
subdir, and it does not conflict with tracked files at higher levels
(e.g. `data/team_name_overrides.csv`).

### Step 6: Verify scripts can find the data

Pick the cheapest sanity-check script that touches the data dirs:

```bash
cd "C:/Users/alden/MarchMadness"
python -c "
import pandas as pd
from pathlib import Path
DATA = Path('data/raw/march-machine-learning-2026')
seeds = pd.read_csv(DATA / 'MNCAATourneySeeds.csv')
print(f'seeds: {len(seeds)} rows, seasons {seeds.Season.min()}-{seeds.Season.max()}')
"
```

Expect: `seeds: 7000+ rows, seasons 1985-2026` (or current Kaggle range).

## Prevention

Three rules for any worktree cleanup on this repo:

1. **Never use PowerShell `(Get-Item).Delete()` on a path that is or
   contains a Windows directory junction.** Even if .NET docs claim it
   is safe, on this OS + PowerShell version it is not. (Memory note:
   `feedback_windows_junction_delete.md`.)
2. **Always use `git worktree remove <path>` to remove a worktree.**
   Git correctly handles the junction without following it. If `git
   worktree remove` fails with "permission denied" or "directory not
   empty", do NOT switch to a force-delete tool -- find what is
   holding the directory (most often a stale shell cwd inside the
   worktree) and fix that, then retry.
3. **Never recursively delete a worktree's on-disk directory by hand.**
   If you must (e.g., orphan dir whose git-side admin record is
   already gone), first verify the worktree contains no live
   junctions:

   ```bash
   # PowerShell:
   Get-ChildItem -Path "<worktree>" -Recurse -Force -Attributes ReparsePoint
   # bash via cmd:
   cmd //c "dir /AL /S <worktree>"
   ```

   If any junctions are listed, **remove each junction first** with
   `cmd //c rmdir <junction-path>` (which is the only known-safe Windows
   command for unlinking a junction without following it). Only then
   delete the now-junction-free worktree dir.

## Belt-and-braces: snapshot data state before risky operations

Before any worktree cleanup, run:

```bash
ls "C:/Users/alden/MarchMadness/data/raw/march-machine-learning-2026/" | wc -l > /tmp/data_pre.txt
ls "C:/Users/alden/MarchMadness/data/raw/kaggle/"                       | wc -l >> /tmp/data_pre.txt
ls "C:/Users/alden/MarchMadness/data/raw/vegas_lines/"                  | wc -l >> /tmp/data_pre.txt
cat /tmp/data_pre.txt
```

After cleanup, verify the same counts. A drop signals a wipe in progress
-- `git checkout HEAD -- data/` immediately, then run this runbook from
Step 1.

## Incident log

| Date       | Trigger | Recovery time | Notes |
|------------|---------|---------------|-------|
| 2026-05-02 | PowerShell `(Get-Item "$wt\data").Delete()` on a `<wt>\data` junction during feature-view-ensemble worktree cleanup | ~10 min | Tracked files surfaced via `git status`. Recovered via `git checkout HEAD -- data/` + tarball extract. |
| 2026-05-04 | Suspected non-git cleanup of orphan worktree dirs (`feat-v4-clean-loso-regen`, `feat+v4-gap-audit-vegas`) whose data junctions were still live; mtime alignment at 19:54 across dirs and data subdirs | ~5 min for raw data; ~3 hours for `pairwise_v4.csv` regen | Tracked files in `data/` survived (junctions were at `data/raw/<subdir>` level, not `data/`). Detected by Phase 0 of v9-C clean-rerun PR (subagent reported empty source dir during junction setup). **Compounding loss:** PR 21's clean `output/pairwise_v4.csv` was also wiped along with the data. That file was gitignored and lived only in the wiped worktree; PR 21's findings note (which IS tracked) reported the clean LL numbers, but the actual canonical artifact never made it to git. Discovered when Phase 2 of the v9-C re-run produced byte-identical leaky v8 (Phase 2 commit was reverted). Fix going forward: force-add `pairwise_v4.csv` and `pairwise_v8.csv` whenever they are regenerated.

## Canonical pairwise artifacts: tracked vs. should-be-tracked

`output/` is gitignored. Specific named files are force-added (`git add -f`)
as canonical artifacts that downstream consumers depend on. **Anything in
`output/` that is the result of a long compute (>5 min) and is consumed by
another script SHOULD be force-added** -- otherwise it lives only in the
working tree and dies with any wipe.

**Tracked today (force-added):**

| File | Producer | Consumer(s) |
|---|---|---|
| `output/pairwise_probs.json` | `predict_2026_v9c.py` (or `predict_2026_stage2.py` if v9-C reverted) | `postmortem_full.py`, `bracket_scorecard.py`, `alternate_bracket.py`, `iowa_impact.py`, `blend_sweep.py` |
| `output/pairwise_v9.csv` | `sweep_v9_weights.py` (winning cell) | downstream backtest scripts |
| `output/pairwise_bt.csv` | `train_bt_stage1.py` | v9-D BT-as-feature path |
| `output/pairwise_ensemble.csv` | `ensemble_stage1.py` | feature-view ensemble work |
| `output/pairwise_hbt_sigma_*.csv` (7 files) | `train_hbt_stage1.py` | HBT diagnostic |
| `output/pairwise_lr.csv` | `train_lr_stage1.py` | LR ensemble experiment |
| `output/pairwise_peer_a.csv`, `output/pairwise_peer_b.csv` | `train_peer_stage1.py` | feature-view ensemble |
| `output/pairwise_v4bt_w*.csv` (6 files) | `sweep_bt_bracket_points.py` | BT bracket-points sweep |
| `output/pairwise_v9c_*.csv` (8 files) | various v9-C variants | per-variant backtests |

**SHOULD be tracked but isn't (load-bearing gap that caused PR 21's loss
on 2026-05-04):**

| File | Producer | Consumer(s) | Compute cost |
|---|---|---|---|
| `output/pairwise_v4.csv` | `enhanced_model_v3.py` (under `MM_PAIRWISE_OUT`) | `train_stage2.py`, `train_upset_model.py`, `sweep_v9_weights.py`, `train_bt_stage1.py`, every diagnose_*.py that needs v4 LOSO | **~3 hours** |
| `output/pairwise_v8.csv` | `train_stage2.py` | `sweep_v9_weights.py` (v8 baseline gate), `predict_2026_stage2.py` | ~3 minutes |

When you regenerate either of these (e.g., as part of the recovery-step
work or any future v4 retraining), **immediately force-add the result
to git** with a `data(...)` commit so the next data wipe does not consume
it. Append-mode writers (`enhanced_model_v3.py:629` writes
`pairwise_v4.csv` with `mode="a"`) are particularly fragile: if the
existing file is leaky-baseline content, the regen appends clean rows
under it and `keep="last"`-dedup masks the leak from downstream readers
-- but if the file is wiped between regen and the next consumer run, the
recovery cost is the full 3-hour regen.

**Append-mode caveat for `pairwise_v4.csv`.** Before re-running the
regen, **delete any existing `output/pairwise_v4.csv` first** (it is
written in append mode, line 629 of `enhanced_model_v3.py`) -- otherwise
the file ends up with leaky rows from the prior content followed by
clean rows from the regen. Dedup-by-last hides the staleness but the
file size doubles on each rerun. Always start from an absent file:

```bash
rm -f output/pairwise_v4.csv
MM_PAIRWISE_OUT=output/pairwise_v4.csv MM_SKIP_DEFAULT_LOSO=1 \
MM_TUNED_PARAMS_V3='{"n_estimators": 424, "max_depth": 4, "learning_rate": 0.013940346079873234, "subsample": 0.8736932106048627, "colsample_bytree": 0.7760609974958406}' \
python -u src/enhanced_model_v3.py > output/regen_clean_log.txt 2>&1
```

After the run completes (~3 hours):

```bash
wc -l output/pairwise_v4.csv  # expect 48,466 (header + 48,465 single-orientation rows)
git add -f output/pairwise_v4.csv
git commit -m "data(<branch>): force-add canonical pairwise_v4.csv (clean baseline)"
```

(MM_TUNED_PARAMS_V3 reuses leaky-run hyperparameters per PR 21's
documented confound; expected effect <0.02 LL on the clean baseline.)

## Related files

- `memory/feedback_windows_junction_delete.md` -- the durable feedback
  memory entry that triggers this doc when relevant.
- `data/training_data.tar.gz` -- the recovery archive itself (tracked).
- `src/enhanced_model_v3.py:181-190` -- vegas_lines filename-to-season
  mapping (`ncaabbYY.csv` -> Kaggle Season YY+1).
- `src/enhanced_model_v3.py:606-630` -- `MM_PAIRWISE_OUT` writer
  (append mode); see "Append-mode caveat" above before re-running.
- `src/enhanced_model_v3.py:972-976` -- `MM_SKIP_DEFAULT_LOSO` env
  gate (skips Step 6's untuned default LOSO; halves regen runtime).
- `src/enhanced_model_v3.py:997-1006` -- `MM_TUNED_PARAMS_V3` env
  gate (reuses tuned hyperparameters as JSON; saves an Optuna pass).
- `src/ingest/` -- ingest scripts for live (non-recovery) data flows.
  `kaggle2026_loader.py`, `kaggle_loader.py`, `cbbd_loader.py`,
  `massey_loader.py`. None of these load `vegas_lines/` (it has no
  ingest script in the repo -- it is a one-time data drop bundled into
  `training_data.tar.gz`).
