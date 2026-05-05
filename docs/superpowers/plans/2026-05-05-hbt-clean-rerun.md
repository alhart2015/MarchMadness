# HBT Clean Re-eval Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Re-run PR 16's per-cell 3-clause LL-blend gate
(`src/diagnose_hbt_vs_v4.py`) over the same 7-cell sigma sweep
{0.05, 0.10, 0.20, 0.50, 1.00, 2.00, 5.00} against clean
`pairwise_v4.csv` (PR 23 force-add) and clean-trained HBT cells. Apply
a verdict-dependent decision: any cell PASS -> promote HBT v9-C
correction + bracket-points backtest to next sub-priority; all cells
FAIL -> close HBT from the marginal-rejections list.

**Architecture:** Rerun the existing `train_hbt_stage1.py` to
regenerate the 7 per-sigma CSVs under the post-PR-19 clean v4 feature
pipeline, then rerun the existing `diagnose_hbt_vs_v4.py`. No
source-code changes. All 7 HBT cells expected to differ byte-wise
from tracked PR 16 outputs because HBT priors include Vegas-affected
v4 features.

**Tech Stack:** Python 3.11, pandas, numpy, scipy (L-BFGS-B), pytest.
Worktree-based isolation; data junctions for `march-machine-learning-2026/`,
`kaggle/`, `vegas_lines/` already established at worktree setup.

**Branch:** `feat/hbt-clean-rerun` (worktree at `.claude/worktrees/feat-hbt-clean-rerun`)
**Spec:** `docs/superpowers/specs/2026-05-05-hbt-clean-rerun-design.md`

---

### Task 1: Capture pre-run md5s and run existing tests

**Goal:** Record the pre-run state of the 8 tracked HBT artifacts (for
findings-doc diff narrative) and verify HBT test suite is clean before
modifying anything.

**Files:**
- Read-only: `output/pairwise_hbt_sigma_*.csv` (7 files), `output/diag_hbt_sweep.json`
- Read-only: `tests/test_features/test_hierarchical_bt.py`,
  `tests/test_train_hbt_stage1.py`,
  `tests/test_diagnose_hbt_vs_v4.py`

- [ ] **Step 1: md5 the 7 tracked HBT CSVs + diag JSON**

```bash
md5sum output/pairwise_hbt_sigma_*.csv output/diag_hbt_sweep.json | tee output/_md5_hbt_pre.txt
```
Expected: 8 lines, each a 32-char hex hash. Note for findings doc.

- [ ] **Step 2: Run HBT-related tests**

```bash
python -m pytest tests/test_features/test_hierarchical_bt.py tests/test_train_hbt_stage1.py tests/test_diagnose_hbt_vs_v4.py -q
```
Expected: 20 passed (5 + 7 + 8). If any fail, halt and reconcile before
running the trainer -- a broken trainer or diagnostic invalidates the
re-eval.

---

### Task 2: Regen 7 HBT CSVs against clean v4 features

**Goal:** Run `train_hbt_stage1.py` against the post-PR-19 clean v4
feature pipeline. Each sigma cell unlinks its existing CSV (line 142-144)
and writes fresh; no append-mode caveat.

**Files:**
- Modify (overwrite): `output/pairwise_hbt_sigma_0.05.csv`,
  `output/pairwise_hbt_sigma_0.10.csv`,
  `output/pairwise_hbt_sigma_0.20.csv`,
  `output/pairwise_hbt_sigma_0.50.csv`,
  `output/pairwise_hbt_sigma_1.00.csv`,
  `output/pairwise_hbt_sigma_2.00.csv`,
  `output/pairwise_hbt_sigma_5.00.csv`
- Read-only: `src/train_hbt_stage1.py`, `src/features/hierarchical_bt.py`,
  `src/enhanced_model_v3.py` (via `prepare_loso_inputs()`)

- [ ] **Step 1: Run trainer (cold-cache cold-start)**

```bash
python -u src/train_hbt_stage1.py 2>&1 | tee output/_train_hbt_log.txt
```
Expected: ~4 min `prepare_loso_inputs()` cold-start (data/cache wiped
this morning) + ~4 min trainer = ~8 min total. 7 sigma blocks, each
with 22 per-season summary lines + final summary `wrote 48,465 pairs`.
No `FIT ERROR` lines.

- [ ] **Step 2: Sanity-check the 7 CSVs**

```bash
wc -l output/pairwise_hbt_sigma_*.csv
```
Expected: 48,466 lines each (header + 48,465 rows). All 7 files same
length.

- [ ] **Step 3: Capture post-run md5s**

```bash
md5sum output/pairwise_hbt_sigma_*.csv | tee output/_md5_hbt_post.txt
```
Expected: 7 hashes, ALL different from `_md5_hbt_pre.txt` (HBT priors
shifted because v4 features shifted under PR 19's Vegas filter). If
any cell's hash matches its pre-run hash, halt -- evidence the priors
are not seeing the clean v4 features.

---

### Task 3: Run clean-baseline diagnostic + sanity gates

**Goal:** Run the per-cell 3-clause gate with clean inputs and verify
sanity gates (matched-game count, v4 LL, HBT LL shift).

**Files:**
- Reads: `output/pairwise_v4.csv`, 7 x `output/pairwise_hbt_sigma_*.csv`,
  `data/raw/march-machine-learning-2026/MNCAATourneyCompactResults.csv`
- Writes: `output/diag_hbt_sweep.json` (overwrite)

- [ ] **Step 1: Confirm input files exist + are the expected shape**

```bash
ls -la output/pairwise_v4.csv output/pairwise_hbt_sigma_*.csv && wc -l output/pairwise_v4.csv
```
Expected: pairwise_v4.csv ~48,466 lines (PR 23 force-add); each HBT CSV
~48,466 lines.

- [ ] **Step 2: Run the diagnostic**

```bash
python src/diagnose_hbt_vs_v4.py 2>&1 | tee output/_diag_hbt_log.txt
```
Expected stdout: header, sweep table (7 rows), threshold line,
verdict line. Exit code 0 if any cell passes, 1 if all fail -- both
acceptable; we ACT on the verdict in Task 5.

- [ ] **Step 3: Sanity gate -- `n_games == 1449` for every cell**

```bash
python -c "
import json
d = json.load(open('output/diag_hbt_sweep.json'))
ns = sorted({c['n_games'] for c in d['cells']})
print('n_games seen:', ns)
assert ns == [1449], ns
"
```
Expected: `n_games seen: [1449]`. If different, halt -- pair coverage
drift between v4 and HBT, likely a join/dedup issue.

- [ ] **Step 4: Sanity gate -- `ll_v4` matches clean baseline (within 0.005 LL)**

```bash
python -c "
import json
d = json.load(open('output/diag_hbt_sweep.json'))
lls = sorted({round(c['ll_v4'], 4) for c in d['cells']})
print('ll_v4 seen:', lls)
assert all(abs(ll - 0.5588) < 0.005 for ll in lls), lls
"
```
Expected: `ll_v4 seen: [0.555-0.563]` (one value, since v4 is the
shared baseline). If outside, halt -- the diagnostic is not seeing the
clean v4 we think.

- [ ] **Step 5: Sanity gate -- HBT LLs shifted vs PR 16 (no cell byte-identical)**

```bash
python -c "
import json
d = json.load(open('output/diag_hbt_sweep.json'))
# PR 16 leaky baseline LLs (from docs/notes/2026-05-03-hierarchical-bt.md)
PRIOR = {0.05: 0.6194, 0.10: 0.6305, 0.20: 0.6220, 0.50: 0.6306,
         1.00: 0.6507, 2.00: 0.6880, 5.00: 0.7569}
for c in d['cells']:
    s = round(c['sigma'], 2)
    new_ll = round(c['ll_hbt'], 4)
    delta = new_ll - PRIOR[s]
    flag = 'SAME' if abs(delta) < 1e-4 else f'shift {delta:+.4f}'
    print(f'  sigma={s:.2f}  ll_hbt: {PRIOR[s]:.4f} -> {new_ll:.4f}  ({flag})')
    assert abs(delta) > 1e-4, f'sigma={s} byte-identical to PR 16; clean v4 not flowing through'
"
```
Expected: 7 lines each showing a non-zero shift. If any cell is
byte-identical to PR 16, halt -- the priors are not picking up clean
v4 features (PR 19 leak fix may not be flowing through).

- [ ] **Step 6: Capture verdict + headline numbers for Task 5**

```bash
python -c "
import json
d = json.load(open('output/diag_hbt_sweep.json'))
best = d.get('best_passing_cell')
print(f\"VERDICT: {'ANY-PASS (cell={:.2f})'.format(best['sigma']) if best else 'ALL-FAIL'}\")
print()
print(f\"  {'sigma':>6}  {'r':>7}  {'ll_hbt':>7}  {'w_opt':>6}  {'headroom':>9}  c1 c2 c3  verdict\")
for c in d['cells']:
    yn = lambda b: 'Y' if b else 'N'
    v = 'PASS' if c['passes_all'] else 'FAIL'
    print(f\"  {c['sigma']:>6.2f}  {c['r']:>+7.3f}  {c['ll_hbt']:>7.4f}  {c['w_opt']:>6.2f}  {c['headroom']:>+9.4f}  {yn(c['passes_r'])}  {yn(c['passes_w'])}  {yn(c['passes_headroom'])}  {v}\")
print()
print(f\"  v4 standalone LL:  {d['cells'][0]['ll_v4']:.4f}\")
print(f\"  thresholds: r < {d['thresholds']['r_max']}, w in [{d['thresholds']['w_low']}, {d['thresholds']['w_high']}], headroom > {d['thresholds']['headroom_min']}\")
" | tee output/_diag_hbt_summary.txt
```
Expected: a multi-line summary saved to `output/_diag_hbt_summary.txt`.
Read in Task 5 step 1.

---

### Task 4: Force-add diagnostic artifacts

**Goal:** Persist the 7 regenerated HBT CSVs + the diagnostic JSON.
Per the canonical-artifact policy in `docs/data_recovery.md`, all 8
files are tracked-via-force-add and must be re-added (overwriting
their PR 16 versions) so the next data wipe does not consume them.

**Files:**
- Force-add (always): 7 x `output/pairwise_hbt_sigma_*.csv`,
  `output/diag_hbt_sweep.json`

- [ ] **Step 1: Confirm files exist and are non-empty**

```bash
ls -la output/diag_hbt_sweep.json output/pairwise_hbt_sigma_*.csv
```
Expected: JSON ~5-10 KB; each pairwise CSV ~2-3 MB.

- [ ] **Step 2: Force-add + commit (8 files)**

```bash
git add -f output/diag_hbt_sweep.json output/pairwise_hbt_sigma_*.csv

git commit -m "$(cat <<'EOF'
data(hbt-clean-rerun): force-add 7 sigma cells + diag JSON under clean v4

output/pairwise_hbt_sigma_<S>.csv (7 files): regenerated by
train_hbt_stage1.py against the clean v4 feature pipeline (post-PR-19
vegas leak fix). HBT's per-team prior s_i ~ Normal(beta . v4_features,
sigma^2) uses the full 67-feature v4 matrix including the seven Vegas
features filtered by PR 19, so all 7 cells shift vs PR 16. Sigma sweep
unchanged: {0.05, 0.10, 0.20, 0.50, 1.00, 2.00, 5.00}.

output/diag_hbt_sweep.json: re-emitted under clean pairwise_v4.csv
(PR 23 baseline) + clean-trained HBT cells. Replaces PR 16's
leaky-baseline numbers.

Both force-added per docs/data_recovery.md canonical-artifact policy
(load-bearing audit artifacts under gitignored output/).
EOF
)"
```
Expected: `[feat/hbt-clean-rerun <hash>] data(hbt-clean-rerun): ...`.
8 files changed.

- [ ] **Step 3: Cleanup scratch files (keep summary until end of Task 5)**

```bash
rm -f output/_md5_hbt_pre.txt output/_md5_hbt_post.txt output/_train_hbt_log.txt output/_diag_hbt_log.txt
```
Note: keep `output/_diag_hbt_summary.txt` until after Task 5; remove
at the very end.

---

### Task 5: Write findings doc + apply decision matrix to TODO.md

**Goal:** Document the verdict + new numbers vs PR 16, and advance the
priority list per the spec's decision matrix.

**Files:**
- Create: `docs/notes/2026-05-05-hbt-clean-rerun.md`
- Modify: `TODO.md` (recovery section, "step 5 sub-priorities" list)

- [ ] **Step 1: Read the verdict summary from Task 3**

```bash
cat output/_diag_hbt_summary.txt
```
Note all values. All filled into the findings doc as actual numbers --
no placeholders. Delete `output/_diag_hbt_summary.txt` at end of Task 5.

- [ ] **Step 2: Write the findings doc**

Create `docs/notes/2026-05-05-hbt-clean-rerun.md`. Mirror PR 24/PR 25
structure (TL;DR, methods, gate result table, standalone metrics,
comparison-to-PR-16, discussion, verdict, recommendation, files-of-record,
follow-ups). Replace `<<...>>` placeholders with actual values.

Required sections:
- **TL;DR**: one paragraph. Verdict + key numbers (best-cell r, w_opt,
  headroom; spread of HBT LL across sigma). One sentence on what
  flipped vs PR 16 (residual-correlation jump pattern? all cells still
  FAIL clauses 2/3?).
- **Methods**: input file md5s (pre and post), trainer + diagnostic
  commands, n_games per cell, threshold values inherited from PR 16.
  Note the data wipe + recovery (extracted training_data.tar.gz, set
  up junctions, trainer rebuilt Massey/Colley/efficiency caches).
- **Per-cell sweep table**: 7 rows with sigma, r, ll_hbt, acc_hbt,
  w_opt, headroom, c1/c2/c3 PASS/FAIL flags, cell verdict.
- **Standalone metrics**: clean v4 LL + acc, HBT LL+acc per sigma cell
  (or summarized as a range).
- **Comparison to PR 16**: side-by-side table. Key columns: sigma,
  ll_hbt (leaky vs clean), r (leaky vs clean), w_opt (leaky vs clean),
  headroom (leaky vs clean), c1 PASS/FAIL (leaky vs clean), cell verdict.
- **Discussion**: 3-4 paragraphs. Touch points:
  - Did residual correlation track PR 24's BT pattern (r 0.577 -> 0.868)
    or PR 25's PEER_B pattern (rho 0.45 -> 0.726)? Compare quantitatively.
  - Did HBT standalone LL shift on the clean baseline? Direction +
    magnitude per sigma cell. Does the non-monotonic curve over sigma
    persist?
  - If any cell PASSED: which one, why, and what does the production
    follow-up look like (v9-C correction + bracket-points backtest).
  - If all cells FAILED: confirm the prior PR 16 verdict with strengthened
    framing. Note whether the framing-correction concerns from PR 16
    ("LL gate may be filtering on the wrong metric") are settled by
    PR 17's bracket-points re-test on plain BT (which agreed with the
    LL gate) -- closing HBT here is robust across both leaky and clean
    LL gates.
- **Verdict + recommendation**: branch on PASS / FAIL.

- [ ] **Step 3: Update `TODO.md` per decision matrix**

Find the "step 5 sub-priorities" list under "5. Re-run the swap-decided
/ swap-candidate evaluations against the clean baseline". Apply ONE of
the two patches.

**If GATE FAILED (all cells)**: Mark "HBT (PR 16) re-eval" sub-priority
done with FAIL verdict. Find the bullet currently reading:

```markdown
       - HBT (PR 16): standalone LL 0.619-0.757; gap to clean v4
         shrinks but HBT still weaker. **NEXT IMMEDIATE PR** (~5 min
         compute). Note: PR 24 + this PR both showed residual-
         correlation jumps to r ~0.7-0.87 on the clean baseline, which
         strongly predicts HBT's clause 2 also flips FAIL. HBT re-eval
         is mostly closing the marginal-rejections list cleanly rather
         than expecting a flip.
```

Replace with:

```markdown
       - **[DONE -- PR <pending>]** HBT (PR 16) re-eval. **GATE FAILED**
         (all 7 cells). Best cell sigma=<<x.xx>>: r=<<x.xxx>>,
         w_opt=<<x.xx>>, headroom=<<+/-x.xxxx>>. <<one-sentence on
         residual-correlation pattern: did clause 1 flip FAIL on most
         cells matching PR 24/25, or stay PASS but with c2/c3 still
         dominating>>. Robust NO-GO across both leaky and clean baselines
         closes HBT as a stage-1 ensemble peer. Findings:
         `docs/notes/2026-05-05-hbt-clean-rerun.md`.
```

**If GATE PASSED (any cell)**: Mark sub-priority done with PASS verdict;
promote HBT v9-C correction + bracket-points backtest. Replace bullet with:

```markdown
       - **[DONE -- PR <pending>]** HBT (PR 16) re-eval. **GATE PASSED**
         on cell sigma=<<x.xx>>: r=<<x.xxx>>, w_opt=<<x.xx>>,
         headroom=<<+x.xxxx>>. <<one-sentence on what flipped>>. Findings:
         `docs/notes/2026-05-05-hbt-clean-rerun.md`.
       - **HBT v9-C correction + bracket-points backtest -- NOW THE
         IMMEDIATE NEXT PR.** Per the original HBT spec's if-pass branch:
         apply the best-passing cell's HBT predictions to v9-C, score
         22-season bracket points head-to-head against the canonical
         v4 + v9-C baseline (2069 brkt pts on clean v8). ~3 hours compute.
```

- [ ] **Step 4: Verify TODO edits compile (no broken markdown)**

```bash
sed -n '125,160p' TODO.md
```
Expected: the "step 5 sub-priorities" section reads cleanly.

- [ ] **Step 5: Commit findings + TODO update**

```bash
rm output/_diag_hbt_summary.txt
git add docs/notes/2026-05-05-hbt-clean-rerun.md TODO.md docs/superpowers/specs/2026-05-05-hbt-clean-rerun-design.md docs/superpowers/plans/2026-05-05-hbt-clean-rerun.md
git commit -m "$(cat <<'EOF'
docs(hbt-clean-rerun): findings + TODO update -- recovery step 5 marginal #3

Verdict: <<PASS or FAIL>> (best-cell sigma=<<x.xx>>, r=<<x.xxx>>,
headroom=<<+/-x.xxxx>>). <<one-sentence summary of what flipped or
didn't and what comes next>>.

Findings: docs/notes/2026-05-05-hbt-clean-rerun.md.
Spec:     docs/superpowers/specs/2026-05-05-hbt-clean-rerun-design.md.
Plan:     docs/superpowers/plans/2026-05-05-hbt-clean-rerun.md.
TODO step 5 sub-priority list advanced per spec decision matrix.
EOF
)"
```
Expected: `[feat/hbt-clean-rerun <hash>]`. 4 files changed.

---

## Self-review checklist (the implementer should run this at end of plan)

- [ ] All relevant tests pass: `python -m pytest tests/test_features/test_hierarchical_bt.py tests/test_train_hbt_stage1.py tests/test_diagnose_hbt_vs_v4.py -q`. Expect 20 passed.
- [ ] Two commits on the branch: data (Task 4), docs (Task 5). Spec + plan committed at the start of execution.
- [ ] `git log --oneline main..HEAD` shows the spec, plan, data, and docs commits in order.
- [ ] `output/diag_hbt_sweep.json` shows clean-baseline numbers (`ll_v4 ~ 0.555-0.563`, all 7 `ll_hbt` shifted vs PR 16).
- [ ] Findings doc has zero `<<...>>` placeholders.
- [ ] `TODO.md` step 5 list reflects the actual verdict.
- [ ] No scratch files left under `output/_*.txt`.
- [ ] Worktree on branch `feat/hbt-clean-rerun`, ready to push + open PR.
