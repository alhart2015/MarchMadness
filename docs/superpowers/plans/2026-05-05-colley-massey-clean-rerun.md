# Colley + Massey-Decay Clean Re-eval Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Re-run PR 15's clause-2 LL-headroom gate for the two remaining
named items on the recovery step 5 marginal-rejections list (Colley,
Massey-decay-14d) against the post-PR-19 clean v4 feature pipeline.
Apply a per-candidate verdict-dependent decision: any candidate PASS ->
promote to "next-immediate full LOSO backtest" sub-priority; both FAIL ->
close marginal-rejections list cleanly.

**Architecture:** Reuse the existing `clause2_decay_massey.py` runner
verbatim for Massey-decay-14d. Mirror it as a new `clause2_colley.py`
to avoid temporary wire-in of the reverted `compute_all_features` Colley
column. Both scripts emit JSON artifacts that are force-added.

**Tech Stack:** Python 3.11, pandas, numpy, xgboost (LOSO trainer),
pytest. Worktree-based isolation; data junctions for
`march-machine-learning-2026/`, `kaggle/`, `vegas_lines/` already
established at worktree setup.

**Branch:** `feat/colley-massey-clean-rerun` (worktree at `.claude/worktrees/feat-colley-massey-clean-rerun`)
**Spec:** `docs/superpowers/specs/2026-05-05-colley-massey-clean-rerun-design.md`

---

### Task 1: Run pre-flight tests

**Goal:** Confirm the colley + massey solver tests pass under the clean
worktree before regen. (Pre-flight already done at worktree setup;
re-run as a fast verification.)

**Files:**
- Read-only: `tests/test_features/test_colley_matrix.py`,
  `tests/test_features/test_massey_matrix.py`

- [x] **Step 1: Run tests**

```bash
python -m pytest tests/test_features/test_colley_matrix.py tests/test_features/test_massey_matrix.py -q
```
Expected: `15 passed` (6 colley + 9 massey). If any fail, halt.
Confirmed at worktree setup: 15/15 passed in 11.83s.

---

### Task 2: Write `src/clause2_colley.py`

**Goal:** New standalone clause-2 runner for Colley, mirroring
`src/clause2_decay_massey.py`. No wire-in to `compute_all_features`
required; the runner constructs the merged feature matrix locally.

**Files:**
- Create: `src/clause2_colley.py`
- Read-only: `src/clause2_decay_massey.py` (template),
  `src/features/colley_matrix.py` (provides `compute_colley_ratings`),
  `src/diagnose_colley.py` (provides `GATE_SUBSET_SEASONS`,
  `LL_HEADROOM_MAX`)

- [ ] **Step 1: Implement clause2_colley.py**

Structure to mirror `clause2_decay_massey.py`:
- `from src.diagnose_colley import GATE_SUBSET_SEASONS, LL_HEADROOM_MAX`
- `from src.features.colley_matrix import compute_colley_ratings`
- Standard `_ROOT` / sys.path bootstrap.
- `main()`:
  - `inputs = prepare_loso_inputs()`
  - Drop pre-existing `colley_rating` from fm + feature_cols if present
    (defensive; wire-in is reverted but safe).
  - `colley = compute_colley_ratings(regular)`; merge into fm on
    (TeamID, Season).
  - `cols_with = list(feature_cols_full) + ["colley_rating"]`
  - `cols_without = list(feature_cols_full)`
  - Two `leave_one_season_out_cv_weighted` runs with
    `allowed_holdouts=GATE_SUBSET_SEASONS`.
  - Compute per-season + mean deltas.
  - Write `output/diag_clause2_colley.json`.
  - Print summary table + `CLAUSE 2: PASS/FAIL`.
  - sys.exit(0) on PASS, 1 on FAIL.

- [ ] **Step 2: Confirm syntax + imports**

```bash
python -c "import ast; ast.parse(open('src/clause2_colley.py').read())"
python -c "from src.features.colley_matrix import compute_colley_ratings; print('ok')"
```
Expected: no error printed; `ok`.

- [ ] **Step 3: Commit spec + plan + clause2_colley.py**

```bash
git add docs/superpowers/specs/2026-05-05-colley-massey-clean-rerun-design.md docs/superpowers/plans/2026-05-05-colley-massey-clean-rerun.md src/clause2_colley.py

git commit -m "$(cat <<'EOF'
plan(colley-massey-clean-rerun): spec + plan + clause2_colley runner

spec defines per-candidate decision matrix (FAIL/FAIL most likely,
PASS/PASS most surprising). plan details task-by-task execution.
clause2_colley.py mirrors clause2_decay_massey.py to avoid temporary
wire-in of the reverted colley_rating column to compute_all_features.
EOF
)"
```
Expected: `[feat/colley-massey-clean-rerun <hash>]`. 3 files changed.

---

### Task 3: Run both clause-2 runners

**Goal:** Generate clean-baseline clause-2 numbers for both candidates.
First runner pays the ~4 min `prepare_loso_inputs()` cold-start (cache
rebuild); subsequent runners are ~30 sec each.

**Files:**
- Reads: feature pipeline via `prepare_loso_inputs()`,
  `data/raw/march-machine-learning-2026/*.csv`,
  `data/raw/kaggle/*.csv`,
  `data/raw/vegas_lines/*.csv`,
  `output/pairwise_v4.csv` (NOT used directly; this is a feature-addition
  gate, not an LL-blend gate)
- Writes: `output/clause2_decay_massey_hl14.json` (overwrite),
  `output/diag_clause2_colley.json` (new)
- Side effects: rebuilds `data/cache/colley_ratings.parquet`,
  `data/cache/massey_mov_ratings.parquet` (and any efficiency caches
  needed by `prepare_loso_inputs()`)

- [ ] **Step 1: Run Colley clause 2 (cold-start)**

```bash
python -u src/clause2_colley.py 2>&1 | tee output/_clause2_colley_log.txt
```
Expected: ~4-5 min total (~4 min cache rebuild + ~30 sec LOSO subset).
Log shows two LOSO blocks each with 3 per-season summary lines (2019,
2022, 2024); no `FIT ERROR` lines. Final summary prints `CLAUSE 2: PASS`
or `FAIL` and `output/diag_clause2_colley.json` written.

- [ ] **Step 2: Run Massey-decay-14d clause 2 (warm cache)**

```bash
python -u src/clause2_decay_massey.py 14 2>&1 | tee output/_clause2_massey_log.txt
```
Expected: ~30 sec (cache warm from Step 1). Log shows the same
structure; final summary prints `CLAUSE 2: PASS` or `FAIL` and
`output/clause2_decay_massey_hl14.json` written.

- [ ] **Step 3: Sanity gates on both JSONs**

```bash
python -c "
import json
c = json.load(open('output/diag_clause2_colley.json'))
m = json.load(open('output/clause2_decay_massey_hl14.json'))
assert c['subset_seasons'] == [2019, 2022, 2024], c['subset_seasons']
assert m['subset_seasons'] == [2019, 2022, 2024], m['subset_seasons']
print(f\"Colley:        without={c['mean_ll_without_colley']:.4f}  with={c['mean_ll_with_colley']:.4f}  delta={c['mean_ll_delta']:+.4f}  pass={c['pass']}\")
print(f\"Massey hl=14d: without={m['mean_ll_without_massey']:.4f}  with={m['mean_ll_with_massey']:.4f}  delta={m['mean_ll_delta']:+.4f}  pass={m['pass']}\")
print()
print('PR 15 leaky baseline for comparison:')
print('  Colley:        without=0.4388  with=0.4440  delta=+0.0053  pass=False')
print('  Massey hl=14d: without=0.4388  with=0.4445  delta=+0.0057  pass=False')
print()
without_diff = abs(c['mean_ll_without_colley'] - m['mean_ll_without_massey'])
print(f'sanity: Colley vs Massey ll_without agree to {without_diff:.6f} (expect ~1e-6 -- both compute v4-without on same fm shape)')
"
```
Expected: both `subset_seasons` exactly `[2019, 2022, 2024]`; both
`mean_ll_without_*` shifted from 0.4388; print captured for findings.
The two `mean_ll_without_*` values may differ by up to ~0.001 (XGB
tuned-params re-run is not byte-deterministic across processes despite
same data); report actual difference in findings.

- [ ] **Step 4: Capture verdict summary for Task 5**

```bash
python -c "
import json
c = json.load(open('output/diag_clause2_colley.json'))
m = json.load(open('output/clause2_decay_massey_hl14.json'))
def verdict(d):
    return 'PASS' if d['pass'] else 'FAIL'
print(f'COLLEY: {verdict(c)}  delta={c[\"mean_ll_delta\"]:+.4f}')
for r in c['per_season']:
    sign = '+' if r['ll_delta'] >= 0 else ''
    print(f'  {r[\"season\"]}: with={r[\"ll_with\"]:.4f} without={r[\"ll_without\"]:.4f} delta={sign}{r[\"ll_delta\"]:.4f}')
print()
print(f'MASSEY hl=14d: {verdict(m)}  delta={m[\"mean_ll_delta\"]:+.4f}')
for r in m['per_season']:
    sign = '+' if r['ll_delta'] >= 0 else ''
    print(f'  {r[\"season\"]}: with={r[\"ll_with\"]:.4f} without={r[\"ll_without\"]:.4f} delta={sign}{r[\"ll_delta\"]:.4f}')
" | tee output/_clause2_summary.txt
```
Expected: a multi-line summary saved to `output/_clause2_summary.txt`.
Read in Task 5 step 1.

---

### Task 4: Force-add diag JSONs

**Goal:** Persist both diagnostic JSONs as canonical artifacts. Per the
canonical-artifact policy in `docs/data_recovery.md`, both are
load-bearing audit artifacts under gitignored `output/` and must be
force-added so the next data wipe does not consume them.

**Files:**
- Force-add (always): `output/clause2_decay_massey_hl14.json`,
  `output/diag_clause2_colley.json`

- [ ] **Step 1: Confirm files exist and are non-empty**

```bash
ls -la output/clause2_decay_massey_hl14.json output/diag_clause2_colley.json
```
Expected: each 1-3 KB.

- [ ] **Step 2: Force-add + commit (2 files)**

```bash
git add -f output/clause2_decay_massey_hl14.json output/diag_clause2_colley.json

git commit -m "$(cat <<'EOF'
data(colley-massey-clean-rerun): force-add 2 diag JSONs under clean v4

output/diag_clause2_colley.json: NEW canonical artifact. Per-season
ll_with / ll_without / ll_delta on subset {2019, 2022, 2024} with
clean-trained v4 feature pipeline (post-PR-19 vegas leak fix).

output/clause2_decay_massey_hl14.json: regenerated by
clause2_decay_massey.py against the same clean pipeline. Replaces
PR 15's leaky-baseline numbers.

Both force-added per docs/data_recovery.md canonical-artifact policy
(load-bearing audit artifacts under gitignored output/).
EOF
)"
```
Expected: `[feat/colley-massey-clean-rerun <hash>] data(...): ...`.
2 files changed.

- [ ] **Step 3: Cleanup scratch logs**

```bash
rm -f output/_clause2_colley_log.txt output/_clause2_massey_log.txt
```
Note: keep `output/_clause2_summary.txt` until after Task 5; remove at
the very end.

---

### Task 5: Write findings doc + apply decision matrix to TODO.md

**Goal:** Document both verdicts + new numbers vs PR 15, advance
recovery step 5 marginal-rejections list per the spec's per-candidate
decision matrix.

**Files:**
- Create: `docs/notes/2026-05-05-colley-massey-clean-rerun.md`
- Modify: `TODO.md` (recovery section, step 5 sub-priority list)

- [ ] **Step 1: Read the verdict summary from Task 3**

```bash
cat output/_clause2_summary.txt
```
Note all values. All filled into the findings doc as actual numbers --
no placeholders. Delete `output/_clause2_summary.txt` at end of Task 5.

- [ ] **Step 2: Write the findings doc**

Create `docs/notes/2026-05-05-colley-massey-clean-rerun.md`. Mirror
PR 24/25/26 structure but adapted for two-candidate per-clause-2
re-eval. Required sections:

- **TL;DR**: one paragraph. Verdict for each candidate + key numbers
  (clean clause-2 delta, threshold, PASS/FAIL). One sentence on
  whether the redundancy story held across the baseline shift.
- **Methods**: input pipeline (post-PR-19 clean v4), runner commands,
  subset_seasons, threshold inherited from PR 15 (`+0.001`). Note the
  data wipe + recovery (extracted training_data.tar.gz, set up
  junctions; first runner rebuilt Massey/Colley/efficiency caches).
- **Per-candidate clause-2 tables**: one section each. For each
  candidate, a 4-row table (3 seasons + mean) with `ll_with`,
  `ll_without`, `ll_delta` for both PR 15 (leaky) and this PR (clean),
  side by side.
- **Comparison to PR 15 / generalized lesson**:
  - Did either delta shrink to PASS, or stay in the +0.005 ballpark?
  - Did the WITHOUT-baseline shift in the direction PR 21 predicted
    (clean v4 LL on this subset > leaky LL on this subset)?
  - Does the result confirm or refute the spec's "redundancy is
    structural, not threshold-tight" prediction?
  - Cross-reference the PR 24/25/26 residual-correlation jumps as the
    *opposite* shift (clean v4 makes models-as-peers errors more
    correlated; clean v4 makes features-as-features deltas roughly
    unchanged because they compare two same-model arms). The
    distinction is informative for any future feature-vs-model
    diagnostic decision.
- **Verdict + recommendation**: per candidate. Closing the marginal-
  rejections list note if both FAIL.
- **Files of record**: list all new/modified files.

- [ ] **Step 3: Update `TODO.md` per decision matrix**

Find the "step 5 sub-priorities" list under "5. Re-run the swap-decided
/ swap-candidate evaluations against the clean baseline." Two bullets
need replacement, plus a closing-note edit if both FAIL.

**Original Colley bullet (line 161):**
```markdown
       - Colley (PR 15): clause-2 delta +0.0053 LL.
```
**Original Massey bullet (line 162):**
```markdown
       - Massey-decay hl=14d (PR 15): clause-2 delta +0.0057 LL.
```

**Apply ONE of the four decision-matrix replacements per the spec's
combined-verdict cases.** All four use the same `[DONE -- PR <pending>]`
prefix and cite `docs/notes/2026-05-05-colley-massey-clean-rerun.md`.

If FAIL/FAIL (most likely): both bullets become single-line "[DONE]
gate FAILED on clean baseline (delta +X.XXXX); robust NO-GO across both
baselines" lines. Add a closing summary above the bullets:

```markdown
   - **All five named items closed across PR 24, 25, 26, and this PR.**
     Recovery step 5 marginal-rejections list is fully unwound.
```

If any PASS combination: per the spec's decision matrix table, add a
"NOW THE IMMEDIATE NEXT PR" promotion bullet with the full LOSO backtest
follow-up.

Also update the parent recovery-step-5 line if marginal-rejections list
closes -- find:

```markdown
   - The "marginal" rejections in `Tried and rejected` whose deltas
     were within the +0.122 LL leak noise floor of v4. Two named in
     the original roadmap (BT-as-feature at -0.0015 LL; v9 weight-
     sweep family at +18 to +20 pts). **Two still standing on the v9-C
     re-eval (recovery step 5 item 1) findings list (three closed
     across PR 24, PR 25, and this PR):**
```

Update the "(three closed across...)" parenthetical to reflect the new
state per the verdicts.

- [ ] **Step 4: Verify TODO edits compile**

```bash
sed -n '120,170p' TODO.md
```
Expected: the "step 5 sub-priorities" section reads cleanly with new
verdicts in place.

- [ ] **Step 5: Commit findings + TODO update**

```bash
rm output/_clause2_summary.txt
git add docs/notes/2026-05-05-colley-massey-clean-rerun.md TODO.md
git commit -m "$(cat <<'EOF'
docs(colley-massey-clean-rerun): findings + TODO update -- recovery step 5 marginals #4-5

Verdict Colley: <<PASS or FAIL>> (delta=<<+/-x.xxxx>>).
Verdict Massey-decay-14d: <<PASS or FAIL>> (delta=<<+/-x.xxxx>>).
<<one-sentence summary of what the combined verdict means for the
marginal-rejections list>>.

Findings: docs/notes/2026-05-05-colley-massey-clean-rerun.md.
Spec:     docs/superpowers/specs/2026-05-05-colley-massey-clean-rerun-design.md.
Plan:     docs/superpowers/plans/2026-05-05-colley-massey-clean-rerun.md.
TODO step 5 sub-priority list advanced per spec decision matrix.
EOF
)"
```
Expected: `[feat/colley-massey-clean-rerun <hash>]`. 2 files changed.

---

### Task 6: Push branch + open PR

**Goal:** Get the branch up on GitHub and open the PR.

- [ ] **Step 1: Push branch**

```bash
git push -u origin feat/colley-massey-clean-rerun
```
Expected: `Branch 'feat/colley-massey-clean-rerun' set up to track ...`.

- [ ] **Step 2: Open PR with summary + test plan**

```bash
gh pr create --title "Re-eval colley + massey-decay-14d on clean v4 baseline" --body "$(cat <<'EOF'
## Summary
- Closes the last two named items on the recovery step 5 marginal-rejections list (Colley, Massey-decay-14d) by re-running their clause-2 LL-headroom gate against the post-PR-19 clean v4 feature pipeline.
- Adds `src/clause2_colley.py`, a standalone clause-2 runner mirroring `src/clause2_decay_massey.py`, to avoid the temporary wire-in that `src/diagnose_colley.py` would otherwise require.
- Force-adds both `output/clause2_decay_massey_hl14.json` and `output/diag_clause2_colley.json` per the canonical-artifact policy.

## Test plan
- [x] `python -m pytest tests/test_features/test_colley_matrix.py tests/test_features/test_massey_matrix.py -q` pass (15/15)
- [x] both clause-2 runners ran cleanly under clean v4 stack
- [x] sanity gates: `subset_seasons == [2019, 2022, 2024]` for both; `mean_ll_without_*` materially shifted from PR 15's 0.4388
- [x] verdicts per spec decision matrix; findings note references actual numbers (no `<<...>>` placeholders)
- [x] TODO.md step 5 sub-priority list advanced per decision matrix
EOF
)"
```
Expected: PR URL printed. Capture and report.

---

## Self-review checklist (the implementer should run this at end of plan)

- [ ] All relevant tests pass: `python -m pytest tests/test_features/test_colley_matrix.py tests/test_features/test_massey_matrix.py -q`. Expect 15 passed.
- [ ] Three commits on the branch: spec+plan+code (Task 2), data (Task 4), docs (Task 5).
- [ ] `git log --oneline main..HEAD` shows the plan, data, and docs commits in order.
- [ ] `output/diag_clause2_colley.json` is well-formed JSON with `subset_seasons == [2019, 2022, 2024]` and a numeric `mean_ll_delta`.
- [ ] `output/clause2_decay_massey_hl14.json` regenerated (numbers differ from PR 15's leaky values).
- [ ] Findings doc has zero `<<...>>` placeholders.
- [ ] `TODO.md` step 5 list reflects the actual verdicts.
- [ ] No scratch files left under `output/_*.txt`.
- [ ] Worktree on branch `feat/colley-massey-clean-rerun`, branch pushed, PR open.
