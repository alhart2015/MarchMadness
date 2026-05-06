# Colley Full LOSO Backtest Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Re-wire `colley_rating` into v4's `compute_all_features`, regenerate `output/pairwise_v4_with_colley.csv` via clean LOSO, retrain v8 on top, score 22-season LL + bracket points, and apply the spec's Reject/Clear/Marginal ladder vs canonical clean baselines (LL 0.5588 / brkt 2069).

**Architecture:** Un-revert the 12-line removal at commit `3b4c374` so `compute_all_features` produces a 68-feature matrix. Use `MM_PAIRWISE_OUT` to write the regen output to a non-canonical path, preserving canonical `pairwise_v4.csv` as the no-Colley baseline. v8 stage-2 retrains via existing `train_stage2.py`. Bracket points come from existing `score_chalk_brackets.py` `score_pairwise_path`.

**Tech Stack:** Python 3.11, pandas, numpy, xgboost (via `enhanced_model_v3.py` LOSO trainer + `train_stage2.py` corrector), pytest. Worktree-based isolation; data subdir junctions for `march-machine-learning-2026/`, `kaggle/`, `vegas_lines/` already established at worktree setup.

**Branch:** `feat/colley-full-loso-backtest` (worktree at `.claude/worktrees/feat-colley-full-loso-backtest`)
**Spec:** `docs/superpowers/specs/2026-05-05-colley-full-loso-backtest-design.md`

---

### Task 1: Pre-flight + commit spec/plan

**Goal:** Confirm tests pass and the data subdirs are populated; commit spec + plan.

**Files:**
- Read-only: `tests/test_features/test_colley_matrix.py`,
  `data/raw/march-machine-learning-2026/`,
  `data/raw/kaggle/`, `data/raw/vegas_lines/`

- [x] **Step 1: Pre-flight (already done at worktree setup)**

```bash
ls "C:/Users/alden/MarchMadness/data/raw/march-machine-learning-2026/" | wc -l   # expect ~28+
ls "C:/Users/alden/MarchMadness/data/raw/kaggle/"                       | wc -l   # expect ~30+
ls "C:/Users/alden/MarchMadness/data/raw/vegas_lines/"                  | wc -l   # expect 23
python -m pytest tests/test_features/test_colley_matrix.py -q
```
Expected: counts at runbook minima (35 / 38 / 23 confirmed at setup); `6 passed` for tests.

- [ ] **Step 2: Commit spec + plan**

```bash
git add docs/superpowers/specs/2026-05-05-colley-full-loso-backtest-design.md docs/superpowers/plans/2026-05-05-colley-full-loso-backtest.md
git commit -m "$(cat <<'EOF'
plan(colley-full-loso-backtest): spec + plan -- if-pass branch from PR 27

PR 27's clean-baseline clause-2 result for Colley flipped to PASS
(-0.0100 LL on 3-season subset). This PR executes the original Colley
spec's full-LOSO-backtest if-pass branch: re-wire colley_rating into
compute_all_features, regen pairwise_v4_with_colley.csv via clean LOSO,
retrain v8 on top, score 22-season LL + bracket points against clean
baselines (LL 0.5588 / brkt 2069), apply Reject/Clear/Marginal ladder.

Spec details decision matrix and pre-registered predictions.
Plan details task-by-task execution; primary cost is ~3h pairwise regen.
EOF
)"
```
Expected: `[feat/colley-full-loso-backtest <hash>]`. 2 files changed.

---

### Task 2: Re-wire `colley_rating` into `compute_all_features`

**Goal:** Un-revert the 12-line removal at commit `3b4c374`. Inverse of `git show 3b4c374 -- src/enhanced_model.py`.

**Files:**
- Modify: `src/enhanced_model.py` (3 hunks; lines 198, 342, 381 in the post-revert state)

- [ ] **Step 1: Apply Hunk 1 (top of `compute_all_features`)**

Insert immediately after line 198 (`seasons = [s for s in seasons if s >= 2003]`):

```python

    # -- Colley-matrix ratings (cached) -----------------------------------
    from src.features.colley_matrix import load_colley_ratings
    colley_full = load_colley_ratings(reg)
```

Use the Edit tool with old_string anchored at the `# -- Build KenPom -> Kaggle ID mapping --------------------------------` comment that follows.

- [ ] **Step 2: Apply Hunk 2 (per-season Colley map)**

Insert immediately after line 342 (the seed_map population loop's closing line `seed_map[int(row["TeamID"])] = _parse_seed_number(row["Seed"])`), with one blank line above:

```python

        # -- 2j: Colley rating ---------------------------------------------
        season_colley_df = colley_full[colley_full["Season"] == season]
        colley_map = dict(zip(season_colley_df["TeamID"], season_colley_df["colley_rating"]))
```

- [ ] **Step 3: Apply Hunk 3 (per-team row assembly)**

Insert immediately after line 381 (`row_data.update(massey_features[tid])` block), before the `# Conference strength` comment:

```python

            # Colley rating
            if tid in colley_map:
                row_data["colley_rating"] = colley_map[tid]
```

- [ ] **Step 4: Verify the wire-in is correct**

```bash
git diff src/enhanced_model.py | grep -E "^\+" | grep -v "^+++" | wc -l
# expect 12 (8 code + 4 blank/comment matching the original commit)

git show 3b4c374 -- src/enhanced_model.py | grep -E "^-" | grep -v "^---" | sed 's/^-/+/' > /tmp/expected.diff
git diff src/enhanced_model.py | grep -E "^\+" | grep -v "^+++" > /tmp/actual.diff
diff /tmp/expected.diff /tmp/actual.diff
# expect: empty diff (the 12 added lines exactly match the 12 deleted lines from the revert)
```
Expected: empty `diff` output. If non-empty, fix the 3 hunks until they exactly invert `3b4c374`.

- [ ] **Step 5: Smoke-test that the import line resolves**

```bash
python -c "from src.features.colley_matrix import load_colley_ratings; print('ok')"
```
Expected: `ok`. (Already passes -- confirmed at worktree pre-flight.)

- [ ] **Step 6: Commit wire-in**

```bash
git add src/enhanced_model.py
git commit -m "$(cat <<'EOF'
feat(colley-full-loso-backtest): re-wire colley_rating into compute_all_features

Un-revert commit 3b4c374. 12 lines across 3 hunks restored:
- Top: load_colley_ratings(reg) cached call.
- Per-season block 2j: colley_map for the season.
- Per-team row assembly: row_data["colley_rating"] = colley_map[tid].

Inverse of 3b4c374 (which itself reverted b8cefe6). Triggered by PR 27's
clean-baseline clause-2 PASS (-0.0100 LL on 3-season subset). Full
22-season LOSO regen + bracket-points scoring follow in subsequent
tasks per docs/superpowers/plans/2026-05-05-colley-full-loso-backtest.md.
EOF
)"
```
Expected: `[feat/colley-full-loso-backtest <hash>]`. 1 file changed; +12 / -0.

---

### Task 3: Regen `pairwise_v4_with_colley.csv` via clean LOSO

**Goal:** Generate the 22-season LOSO pairwise predictions with `colley_rating` in feature_cols. ~3 hours compute.

**Files:**
- Reads: feature pipeline via `prepare_loso_inputs()`,
  `data/raw/march-machine-learning-2026/*.csv`,
  `data/raw/kaggle/*.csv`,
  `data/raw/vegas_lines/*.csv`
- Writes: `output/pairwise_v4_with_colley.csv` (NEW canonical artifact),
  `output/cv_per_season_v3.csv` (per-season LL/acc, overwritten),
  `data/cache/colley_ratings.parquet` (regenerated; gitignored)
- Side effects: cache rebuild (~4 min) on first call

- [ ] **Step 1: Pre-clean (append-mode caveat)**

```bash
rm -f output/pairwise_v4_with_colley.csv
ls output/pairwise_v4_with_colley.csv 2>&1 | head -1
```
Expected: `ls: cannot access ...`. The append-mode writer
(`enhanced_model_v3.py:606-630`) requires the target be absent, otherwise
PR 21's leaky rows would prepend to clean rows. Setting a fresh
`MM_PAIRWISE_OUT` path avoids the canonical `pairwise_v4.csv` anyway,
but defensive.

- [ ] **Step 2: Kick off the regen (background, ~3 hours)**

```bash
mkdir -p output
MM_PAIRWISE_OUT=output/pairwise_v4_with_colley.csv \
MM_SKIP_DEFAULT_LOSO=1 \
MM_TUNED_PARAMS_V3='{"n_estimators": 424, "max_depth": 4, "learning_rate": 0.013940346079873234, "subsample": 0.8736932106048627, "colsample_bytree": 0.7760609974958406}' \
nohup python -u src/enhanced_model_v3.py > output/regen_colley_log.txt 2>&1 &
echo "regen PID: $!"
```
Expected: process kicks off; PID printed. Use `tail -f
output/regen_colley_log.txt` to watch progress (LOSO season prints
every ~7 minutes).

(For executors using a synchronous Bash with `run_in_background=true`
or equivalent: same env vars, same command, monitor via the regen log.)

- [ ] **Step 3: Wait + monitor**

Poll the log periodically. Expected milestones:
- 0-4 min: cache build (`colley_ratings`, `massey_mov_ratings`, efficiency).
- 4-8 min: feature matrix construction.
- 8-180 min: 22 LOSO iterations, each ~7 min.
- ~180 min: final summary print.

If `FIT ERROR` lines appear, halt and investigate. (PR 21 noted a
post-write `NameError` in the v3 final summary -- already fixed; verify
this run completes without error.)

- [ ] **Step 4: Verify the output**

```bash
wc -l output/pairwise_v4_with_colley.csv
# expect 48,466 (header + 48,465 single-orientation rows; same shape as canonical)

head -1 output/pairwise_v4_with_colley.csv
# expect: season,team_a,team_b,p_a_wins (header; or whatever the canonical header is)

python -c "
import pandas as pd
df = pd.read_csv('output/pairwise_v4_with_colley.csv')
print(f'rows: {len(df):,}')
print(f'unique pairs: {df.groupby([\"season\", \"team_a\", \"team_b\"]).ngroups:,}')
print(f'seasons: {sorted(df[\"season\"].unique())}')
print(f'mean p_a_wins: {df[\"p_a_wins\"].mean():.4f}')
"
```
Expected: ~48,465 rows; 22 seasons spanning 2003-2025 (no 2020); mean
p_a_wins ~0.5 (any orientation symmetry artefact, same as canonical).

- [ ] **Step 5: Force-add canonical artifact**

```bash
git add -f output/pairwise_v4_with_colley.csv
git commit -m "$(cat <<'EOF'
data(colley-full-loso-backtest): force-add pairwise_v4_with_colley.csv (clean LOSO + colley)

NEW canonical artifact (force-added per docs/data_recovery.md policy).
22-season clean LOSO with colley_rating in feature_cols. Generated via
MM_PAIRWISE_OUT=output/pairwise_v4_with_colley.csv,
MM_SKIP_DEFAULT_LOSO=1, MM_TUNED_PARAMS_V3=<PR 21 params>.

Distinct from canonical output/pairwise_v4.csv (no colley); both kept
side by side until verdict is locked. If verdict = Clear in this PR,
the canonical cutover is a separate follow-up PR.
EOF
)"
```
Expected: 1 file changed; ~3-4 MB.

---

### Task 4: Compute LL + accuracy summary (with vs without Colley)

**Goal:** Produce a per-season + aggregate LL/acc table comparing
`pairwise_v4_with_colley.csv` against canonical `pairwise_v4.csv`.

**Files:**
- Reads: `output/pairwise_v4_with_colley.csv`,
  `output/pairwise_v4.csv` (canonical, no colley),
  `data/raw/march-machine-learning-2026/MNCAATourneyCompactResults.csv`
- Writes: temp data captured into the next task's summary JSON

- [ ] **Step 1: Score both pairwise CSVs against actuals**

```bash
python -c "
import pandas as pd
from pathlib import Path
import math

DATA = Path('data/raw/march-machine-learning-2026')
results = pd.read_csv(DATA / 'MNCAATourneyCompactResults.csv')
results = results[results['DayNum'] >= 134]   # tourney games only (134 = Round of 64)

def score(pairwise_path, label):
    df = pd.read_csv(pairwise_path).drop_duplicates(['season', 'team_a', 'team_b'], keep='last')
    pmap = {}
    for _, r in df.iterrows():
        a, b = int(r['team_a']), int(r['team_b'])
        pmap[(int(r['season']), min(a, b), max(a, b))] = float(r['p_a_wins']) if a < b else 1 - float(r['p_a_wins'])
    rows = []
    for season, gdf in results.groupby('Season'):
        ll_sum = 0.0
        n_correct = 0
        n = 0
        for _, g in gdf.iterrows():
            w, l = int(g['WTeamID']), int(g['LTeamID'])
            key = (int(season), min(w, l), max(w, l))
            if key not in pmap:
                continue
            p_lo_wins = pmap[key]
            p_w = p_lo_wins if w < l else 1 - p_lo_wins
            p_w = min(max(p_w, 1e-15), 1 - 1e-15)
            ll_sum += -math.log(p_w)
            n_correct += int(p_w > 0.5)
            n += 1
        if n:
            rows.append({'season': int(season), 'n': n, 'll': ll_sum / n, 'acc': n_correct / n})
    out = pd.DataFrame(rows)
    print(f'{label}: 22-season mean LL = {out[\"ll\"].mean():.4f}; mean acc = {out[\"acc\"].mean():.4f}')
    return out

with_col = score('output/pairwise_v4_with_colley.csv', 'with colley')
without_col = score('output/pairwise_v4.csv', 'without colley')

# Sanity: LOSO CV file should reproduce the without-colley numbers.
# Per PR 21 baseline: mean LL 0.5588, mean acc 70.66%.
assert abs(without_col['ll'].mean() - 0.5588) < 0.01, f'baseline drift: {without_col[\"ll\"].mean():.4f} vs 0.5588'

merged = with_col.merge(without_col, on='season', suffixes=('_with', '_without'))
merged['ll_delta'] = merged['ll_with'] - merged['ll_without']
merged['acc_delta'] = merged['acc_with'] - merged['acc_without']
print()
print(merged[['season', 'n', 'll_without', 'll_with', 'll_delta', 'acc_without', 'acc_with', 'acc_delta']].to_string(index=False))
print()
print(f'mean ll_delta: {merged[\"ll_delta\"].mean():+.4f}  (negative = colley helps)')
print(f'mean acc_delta: {merged[\"acc_delta\"].mean():+.4f}  (positive = colley helps)')
print(f'seasons where colley helps on LL:  {(merged[\"ll_delta\"] < 0).sum()} / {len(merged)}')
print(f'seasons where colley helps on acc: {(merged[\"acc_delta\"] > 0).sum()} / {len(merged)}')

merged.to_csv('output/_colley_per_season_ll_acc.csv', index=False)
" 2>&1 | tee output/_colley_ll_summary.txt
```
Expected: per-season table prints; `mean ll_delta` is the headline
clean-baseline LL delta (compare to spec's pre-registered prediction
in `[-0.005, -0.001]`); the canonical-baseline drift assertion passes.

If the assertion fires (canonical pairwise_v4.csv mean LL deviates from
PR 21's 0.5588 by >0.01), halt -- something is off with the canonical
file.

- [ ] **Step 2: Apply LL portion of the ladder**

Note (do not commit yet -- bracket-points half is in Task 6):

| LL_delta | Implication |
|---|---|
| `>= +0.001`              | Reject (regardless of brkt) |
| `<= -0.005`              | Clear (regardless of brkt; barring catastrophic brkt-only fail at `<= +10`) |
| in `(-0.005, +0.001)`    | Continue to bracket-points (decides Marginal vs Clear) |

Capture the LL-side decision marker for the findings doc.

---

### Task 5: Retrain v8 stage-2 against the new pairwise

**Goal:** Generate `output/pairwise_v8_with_colley.csv` -- v8 stage-2
trained on `pairwise_v4_with_colley.csv` instead of canonical
`pairwise_v4.csv`.

**Files:**
- Reads: `output/pairwise_v4_with_colley.csv`,
  `data/raw/march-machine-learning-2026/MNCAATourney*.csv`
- Writes: `output/pairwise_v8_with_colley.csv` (NEW canonical artifact)

`src/train_stage2.py` reads `output/pairwise_v4.csv` directly (hardcoded
path). To redirect it without modifying the script, swap the canonical
file via temporary backup-and-restore around the `train_stage2.py` run.
Confirmed by reading `src/train_stage2.py` lines 22-36.

- [ ] **Step 1: Backup canonical, swap, run, restore**

```bash
# Backup the canonical pairwise_v4.csv (committed, force-added file).
cp output/pairwise_v4.csv output/pairwise_v4.csv.canonical_bak

# Swap the colley regen into the canonical slot for the duration of the v8 run.
cp output/pairwise_v4_with_colley.csv output/pairwise_v4.csv

# Run v8 trainer.
python src/train_stage2.py 2>&1 | tee output/_v8_with_colley_log.txt

# Move the v8 output to its non-canonical name.
mv output/pairwise_v8.csv output/pairwise_v8_with_colley.csv

# Restore canonical pairwise_v4.csv.
mv output/pairwise_v4.csv.canonical_bak output/pairwise_v4.csv
```
Expected: ~3-5 minutes; `train_stage2.py` summary prints WT MEAN
stage-1 LL / WT MEAN stage-2 LL pair (PR 24 baseline: 0.558 / 0.552).

- [ ] **Step 2: Sanity-check the swap-and-restore**

```bash
md5sum output/pairwise_v4.csv
# expect: 795d8ddfcd7a0a09a50c3732825c6316 (canonical, unchanged)

ls -la output/pairwise_v8_with_colley.csv output/pairwise_v8.csv 2>&1
# expect: pairwise_v8_with_colley.csv exists (new); pairwise_v8.csv is the canonical (untouched)
md5sum output/pairwise_v8.csv
# expect: 102467bc485c20ffecc7e6644b46c85a (clean baseline, unchanged from PR 24)
```
Expected: canonical `pairwise_v4.csv` md5 unchanged; canonical
`pairwise_v8.csv` md5 unchanged (we restored it via the swap-back).
If either md5 differs, halt and investigate.

- [ ] **Step 3: Force-add new v8 artifact**

```bash
git add -f output/pairwise_v8_with_colley.csv
git commit -m "$(cat <<'EOF'
data(colley-full-loso-backtest): force-add pairwise_v8_with_colley.csv

NEW canonical artifact (force-added per docs/data_recovery.md policy).
v8 stage-2 trained against pairwise_v4_with_colley.csv. Generated via
swap-canonical-and-restore pattern on src/train_stage2.py (which reads
the canonical path directly).

Distinct from canonical output/pairwise_v8.csv (PR 24 clean v8 over
no-colley v4). Both kept side by side until verdict is locked.
EOF
)"
```
Expected: 1 file changed; ~3-4 MB.

---

### Task 6: Score 22-season bracket points + apply ladder

**Goal:** Compute clean v4_with_colley + v8_with_colley bracket points
across 22 seasons; compare to clean v8 baseline (2069 brkt pts per PR
24); apply Reject/Clear/Marginal ladder.

**Files:**
- Reads: `output/pairwise_v8_with_colley.csv`,
  `data/raw/march-machine-learning-2026/MNCAATourney*.csv`,
  `output/_colley_per_season_ll_acc.csv` (from Task 4)
- Writes: `output/colley_full_loso_summary.json` (NEW canonical artifact)

- [ ] **Step 1: Score per-season bracket points**

`src/score_chalk_brackets.py:score_pairwise_path(path)` (line 194)
takes a single path and returns
`{"total_pts": float, "per_season_pts": {season: pts}}` covering all
seasons present in the pairwise CSV. No need to slice per-season.

```bash
python -c "
import pandas as pd
import sys
sys.path.insert(0, '.')
from src.score_chalk_brackets import score_pairwise_path

with_col = score_pairwise_path('output/pairwise_v8_with_colley.csv')
without_col = score_pairwise_path('output/pairwise_v8.csv')

ws = with_col['per_season_pts']
ns = without_col['per_season_pts']

rows = []
for s in sorted(set(ws) | set(ns)):
    pw = float(ws.get(s, 0.0))
    pn = float(ns.get(s, 0.0))
    rows.append({'season': int(s), 'pts_without': pn, 'pts_with': pw, 'delta': pw - pn})
df_b = pd.DataFrame(rows)
print(df_b.to_string(index=False))
print()
total_with = float(with_col['total_pts'])
total_without = float(without_col['total_pts'])
print(f'22-season total brkt pts: with colley = {total_with}; without (canonical) = {total_without}')
print(f'brkt_delta = {total_with - total_without:+.1f}  (positive = colley helps)')
print(f'baseline check: canonical sum = {total_without} vs PR 24 = 2069 (expect identical or within 5 of bracket-walk noise)')
print(f'seasons where colley helps: {(df_b[\"delta\"] > 0).sum()} / {len(df_b)}')
print(f'seasons where colley hurts: {(df_b[\"delta\"] < 0).sum()} / {len(df_b)}')
print(f'ties: {(df_b[\"delta\"] == 0).sum()} / {len(df_b)}')
df_b.to_csv('output/_colley_per_season_brkt.csv', index=False)
" 2>&1 | tee output/_colley_brkt_summary.txt
```
Expected: per-season table; `total_without` agrees with PR 24's 2069
to within ~5 brkt pts (the scorer's bracket-walk has small
non-determinism on tied chalk slots, documented in PR 24's anchor
sanity check). If `total_without` differs from 2069 by >5, halt --
scoring methodology has drifted and `brkt_delta` is not comparable.

Note: `score_pairwise_path` returns floats (not ints). `brkt_delta` is
a float in this run; cast to `round(brkt_delta)` for the integer
threshold checks in the summary JSON if all underlying scores are
integer-valued (they typically are in this scorer per its 1/2/4/8/16/32
weighting).

- [ ] **Step 2: Combine LL + brkt into the summary JSON**

```bash
python -c "
import pandas as pd
import json

ll = pd.read_csv('output/_colley_per_season_ll_acc.csv')
br = pd.read_csv('output/_colley_per_season_brkt.csv')

merged = ll.merge(br, on='season', how='outer')

ll_delta = float(ll['ll_delta'].mean())
acc_delta = float(ll['acc_delta'].mean())
brkt_delta = float(br['delta'].sum())   # score_pairwise_path returns floats

# Apply ladder per spec, evaluated in order:
if ll_delta >= 0.001 or brkt_delta <= 10:
    verdict = 'Reject'
    reason = 'LL_delta >= +0.001 OR brkt_delta <= +10'
elif ll_delta <= -0.005 or brkt_delta >= 25:
    verdict = 'Clear'
    reason = 'LL_delta <= -0.005 OR brkt_delta >= +25'
else:
    verdict = 'Marginal'
    reason = 'LL_delta in (-0.005, +0.001) AND brkt_delta in (+10, +25)'

summary = {
    'aggregate': {
        'mean_ll_with_colley': float(ll['ll_with'].mean()),
        'mean_ll_without_colley': float(ll['ll_without'].mean()),
        'll_delta': ll_delta,
        'mean_acc_with_colley': float(ll['acc_with'].mean()),
        'mean_acc_without_colley': float(ll['acc_without'].mean()),
        'acc_delta': acc_delta,
        'brkt_total_with_colley': float(br['pts_with'].sum()),
        'brkt_total_without_colley': float(br['pts_without'].sum()),
        'brkt_delta': brkt_delta,
    },
    'baselines_pr_24_pr_21': {
        'mean_ll_clean_v4': 0.5588,
        'mean_acc_clean_v4': 0.7066,
        'brkt_clean_v8_22season': 2069,
    },
    'thresholds': {
        'reject_ll_delta_ge': 0.001,
        'reject_brkt_delta_le': 10,
        'clear_ll_delta_le': -0.005,
        'clear_brkt_delta_ge': 25,
    },
    'verdict': verdict,
    'verdict_reason': reason,
    'per_season': merged.to_dict(orient='records'),
}

with open('output/colley_full_loso_summary.json', 'w') as f:
    json.dump(summary, f, indent=2, default=str)

print(json.dumps({k: v for k, v in summary.items() if k != 'per_season'}, indent=2, default=str))
print()
print(f'VERDICT: {verdict}')
print(f'  ll_delta:   {ll_delta:+.4f}')
print(f'  brkt_delta: {brkt_delta:+.1f}')
" 2>&1 | tee output/_colley_verdict.txt
```
Expected: JSON written; verdict printed.

- [ ] **Step 3: Force-add summary JSON**

```bash
git add -f output/colley_full_loso_summary.json
git commit -m "$(cat <<'EOF'
data(colley-full-loso-backtest): force-add colley_full_loso_summary.json

NEW canonical artifact. Per-season + aggregate LL / acc / brkt-pts
comparison (with vs without colley_rating) plus Reject/Clear/Marginal
verdict per spec ladder. Force-added per docs/data_recovery.md
canonical-artifact policy.

Verdict: <<see verdict line in body>>.
EOF
)"
```
Expected: 1 file changed; ~30-50 KB.

---

### Task 7: Findings doc + TODO update

**Goal:** Document the verdict + numbers; update TODO.md per the spec's
decision-matrix rows.

**Files:**
- Create: `docs/notes/2026-05-05-colley-full-loso-backtest.md`
- Modify: `TODO.md` (recovery step 5 sub-priority list, line 183)

- [ ] **Step 1: Read the verdict + numbers**

```bash
cat output/_colley_verdict.txt
cat output/_colley_ll_summary.txt
cat output/_colley_brkt_summary.txt
```

- [ ] **Step 2: Write findings doc**

Create `docs/notes/2026-05-05-colley-full-loso-backtest.md`. Required
sections:

- **TL;DR**: one paragraph. Verdict + key numbers (LL_delta, brkt_delta)
  + threshold breach. One sentence on whether the spec's pre-registered
  prediction held.
- **Methods**: input pipeline (clean v4 LOSO with colley_rating wired
  in), regen command + flags, swap-and-restore pattern for v8 stage-2,
  bracket-points scoring approach. Note the recurring data wipe + tar
  extract at worktree setup (engineering follow-up at TODO "Test-suite
  hygiene").
- **Aggregate verdict table**: 4 rows (LL_delta, acc_delta, brkt_delta,
  Verdict).
- **Per-season LL + acc + brkt deltas**: 22-row table including 2003
  and 2025 inclusive (no 2020). Highlight any season where the sign
  of LL_delta and brkt_delta disagree.
- **Spec ladder application**: which clause fired (or did not fire) and
  why. If verdict is Marginal, name the closest threshold and the
  delta to it.
- **Comparison to PR 27 / generalized lesson**:
  - Did the 22-season LL_delta dilute the 3-season subset's -0.0100 by
    the predicted 3-5x factor?
  - Were the largest-magnitude per-season helps in the same seasons
    that helped on the subset (2019, 2022, 2024)?
  - Cross-reference Massey-decay-14d's diff outcome as the contrast
    that supports the "W/L-only structurally distinct from clean v4"
    hypothesis.
- **Verdict + recommendation**: per the spec's decision matrix.
- **Files of record**: list all new + modified files.

- [ ] **Step 3: Update TODO.md per the verdict**

Find the bullet starting `**Colley full LOSO backtest -- NOW THE
IMMEDIATE NEXT PR.**` (currently line 183) and replace per the spec's
decision-matrix branch:

- Verdict = Clear:
```markdown
   - **[DONE -- PR <pending>] Colley CLEARed the bar.** LL_delta=<x>
     LL on 22-season aggregate (vs +0.001 reject / -0.005 clear bars);
     brkt_delta=<y> brkt pts (vs +10 / +25 bars). Wire-in retained on
     this branch. Cutover follow-up: replace canonical
     `output/pairwise_v4.csv` with `pairwise_v4_with_colley.csv` via
     a separate PR (preserves PR 24's clean-baseline numbers as the
     "no colley" reference). Findings:
     `docs/notes/2026-05-05-colley-full-loso-backtest.md`.
```

- Verdict = Marginal:
```markdown
   - **[DONE -- PR <pending>] Colley MARGINAL on full LOSO.** LL_delta=<x>
     LL on 22-season aggregate (cleared +0.001 reject bar but did not
     hit -0.005 clear bar); brkt_delta=<y> brkt pts (in (+10, +25)).
     Wire-in retained on this branch as audit artifact; not promoted to
     v4-stack. Candidate follow-up: hyperparameter retune on the
     colley-augmented feature matrix (Optuna pass; ~30 min compute).
     Findings: `docs/notes/2026-05-05-colley-full-loso-backtest.md`.
```

- Verdict = Reject:
```markdown
   - **[DONE -- PR <pending>] Colley REJECTed on full LOSO.** LL_delta=<x>
     LL on 22-season aggregate (>= +0.001 reject bar) AND/OR brkt_delta=<y>
     brkt pts (<= +10 reject bar). Wire-in REVERTED on this branch.
     Closes Colley as v4-stack feature. Generalized lesson: the
     3-season clause-2 PASS in PR 27 over-represented Colley-helpful
     seasons; W/L-only opponent-adjusted strength does not survive the
     22-season test under clean v4. Findings:
     `docs/notes/2026-05-05-colley-full-loso-backtest.md`.
```

- [ ] **Step 4: Verify TODO edits compile**

```bash
sed -n '180,200p' TODO.md
```
Expected: the "Colley full LOSO backtest" bullet is replaced cleanly
with the verdict-specific text; surrounding bullets unchanged.

- [ ] **Step 5: If verdict = Reject, ALSO revert the wire-in**

ONLY in the Reject branch:

```bash
# Revert the wire-in commit (Task 2).
git revert <Task-2-commit-sha> --no-edit
```
Expected: 1 file changed; -12 lines. The revert commit lands on the
branch alongside the audit artifacts (regen pairwise + summary JSON
remain force-added as the "verdict evidence").

If verdict = Clear or Marginal, SKIP this step (wire-in stays).

- [ ] **Step 6: Cleanup scratch files**

```bash
rm -f output/_colley_per_season_ll_acc.csv \
       output/_colley_per_season_brkt.csv \
       output/_colley_ll_summary.txt \
       output/_colley_brkt_summary.txt \
       output/_colley_verdict.txt \
       output/_v8_with_colley_log.txt \
       output/regen_colley_log.txt
```

- [ ] **Step 7: Commit findings + TODO update**

```bash
git add docs/notes/2026-05-05-colley-full-loso-backtest.md TODO.md
git commit -m "$(cat <<'EOF'
docs(colley-full-loso-backtest): findings + TODO update -- Colley <<VERDICT>>

LL_delta=<<+/-x.xxxx>> LL on 22-season aggregate.
brkt_delta=<<+/-y>> brkt pts vs clean v8 baseline (2069).
Verdict: <<Clear / Marginal / Reject>> per spec decision matrix
(docs/superpowers/specs/2026-05-05-colley-full-loso-backtest-design.md).

<<one-sentence summary of what the verdict means for the v4-stack
roadmap and recovery step 5.>>

Findings: docs/notes/2026-05-05-colley-full-loso-backtest.md.
TODO step 5 sub-priority advanced.
EOF
)"
```
Expected: 2 files changed.

---

### Task 8: Push branch + open PR

**Goal:** Get the branch up on GitHub and open the PR.

- [ ] **Step 1: Push branch**

```bash
git push -u origin feat/colley-full-loso-backtest
```
Expected: `Branch 'feat/colley-full-loso-backtest' set up to track ...`.

- [ ] **Step 2: Open PR**

```bash
gh pr create --title "Colley full LOSO backtest on clean v4 -- <<VERDICT>>" --body "$(cat <<'EOF'
## Summary
- Executes the original Colley spec's if-pass branch, triggered by PR 27's clean-baseline clause-2 PASS (-0.0100 LL on 3-season subset).
- Re-wires `colley_rating` into `compute_all_features` (un-revert of commit 3b4c374); regenerates `pairwise_v4_with_colley.csv` via clean LOSO; retrains v8 stage-2 on top; scores 22-season LL + bracket points head-to-head against canonical clean baselines (LL 0.5588 / brkt 2069).
- Verdict: **<<Clear / Marginal / Reject>>** per the spec's Reject/Clear/Marginal ladder. LL_delta=<<x>> LL; brkt_delta=<<y>> brkt pts.

## Test plan
- [x] `python -m pytest tests/test_features/test_colley_matrix.py -q` pass (6/6)
- [x] wire-in diff exactly inverts commit 3b4c374 (12 lines across 3 hunks)
- [x] regen produced `pairwise_v4_with_colley.csv` of the expected shape (~48,465 rows)
- [x] v8 retrained against the new pairwise without disturbing canonical `pairwise_v4.csv` md5 (swap-and-restore pattern)
- [x] canonical `pairwise_v8.csv` baseline reproduced to within 5 brkt pts
- [x] verdict applies the spec ladder cleanly; findings note has zero `<<...>>` placeholders
- [x] TODO.md step 5 sub-priority list advanced per decision matrix
EOF
)"
```
Expected: PR URL printed.

---

## Self-review checklist (the implementer should run this at end of plan)

- [ ] All relevant tests pass: `python -m pytest tests/test_features/test_colley_matrix.py tests/test_features/test_massey_matrix.py -q`. Expect 15 passed.
- [ ] Wire-in diff exactly inverts commit 3b4c374 (no stray whitespace).
- [ ] Canonical `output/pairwise_v4.csv` md5 unchanged at end of run (`795d8ddfcd7a0a09a50c3732825c6316`).
- [ ] Canonical `output/pairwise_v8.csv` md5 unchanged at end of run (`102467bc485c20ffecc7e6644b46c85a`).
- [ ] `output/pairwise_v4_with_colley.csv` and `output/pairwise_v8_with_colley.csv` exist, force-added, distinct from canonical.
- [ ] `output/colley_full_loso_summary.json` is well-formed JSON with `verdict in {'Reject', 'Clear', 'Marginal'}`.
- [ ] Findings doc has zero `<<...>>` placeholders.
- [ ] TODO.md step 5 list reflects the actual verdict.
- [ ] No scratch files left under `output/_*.txt`, `output/_*.csv`, `output/regen_colley_log.txt`.
- [ ] Worktree on branch `feat/colley-full-loso-backtest`, branch pushed, PR open.
- [ ] Commit graph: plan commit (Task 1), wire-in commit (Task 2), data commit pairwise_v4 (Task 3), data commit pairwise_v8 (Task 5), data commit summary JSON (Task 6), [optional revert commit if Reject (Task 7)], docs commit (Task 7).
