# Feature-View Ensemble PEER_A/B Clean Re-eval Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Re-run PR 14's 3-clause pre-sweep gate (`src/diagnose_feature_view_ensemble.py`) against clean `pairwise_v4.csv` (PR 23 force-add) and clean-trained PEER_A / PEER_B pairwise CSVs. Apply a verdict-dependent decision: PASS -> promote E1+E2 sweeps to next sub-priority; FAIL -> drop feature-view ensemble from the marginal-rejections list.

**Architecture:** Rerun the existing `train_peer_stage1.py` for both peers under the post-PR-19 clean v4 feature pipeline, then rerun the existing `diagnose_feature_view_ensemble.py`. No source-code changes. PEER_A is expected byte-identical to the tracked PR 14 output (no Vegas features in PEER_A's 40 columns); PEER_B is expected to shift (Vegas features present).

**Tech Stack:** Python 3.11, pandas, numpy, scikit-learn, xgboost (already in repo), pytest. Worktree-based isolation; data junctions for `march-machine-learning-2026/`, `kaggle/`, `vegas_lines/` already established at worktree setup.

**Branch:** `feat/feature-view-clean-rerun` (worktree at `.claude/worktrees/feat-feature-view-clean-rerun`)
**Spec:** `docs/superpowers/specs/2026-05-05-feature-view-clean-rerun-design.md`

---

### Task 1: PEER_A byte-equality reproducibility check

**Goal:** Confirm `output/pairwise_peer_a.csv` (tracked from PR 14) reproduces byte-for-byte from `train_peer_stage1.py --peer a` against the clean v4 feature pipeline. The "PEER_A is invariant across the leaky->clean transition" claim is load-bearing for this PR's framing.

**Files:**
- Read-only: `src/train_peer_stage1.py`, tracked `output/pairwise_peer_a.csv`
- Temp: `output/pairwise_peer_a_repro.csv` (deleted after compare on equality; kept for force-add on mismatch)

- [ ] **Step 1: Capture md5 of tracked file**

```bash
md5sum output/pairwise_peer_a.csv | tee output/_md5_peer_a_tracked.txt
```
Expected: a 32-char hex hash. Note for findings doc.

- [ ] **Step 2: Run PEER_A trainer to temp path**

```bash
python src/train_peer_stage1.py --peer a --output output/pairwise_peer_a_repro.csv 2>&1 | tail -10
```
Expected: per-season summary lines + final summary; total_pairs ~48,465. No tracebacks. First call rebuilds Massey/Colley/efficiency caches; expect ~10-15 min.

- [ ] **Step 3: Capture md5 of repro file**

```bash
md5sum output/pairwise_peer_a_repro.csv | tee output/_md5_peer_a_repro.txt
```
Expected: same 32-char hex hash as Step 1 (PEER_A's 40 features are non-Vegas; clean baseline shouldn't shift them).

- [ ] **Step 4: Compare byte-equality**

```bash
cmp output/pairwise_peer_a.csv output/pairwise_peer_a_repro.csv && echo "BYTE-EQUAL" || echo "MISMATCH"
```
Expected: `BYTE-EQUAL`.

- [ ] **Step 5: On byte-equal, delete temp; on mismatch, keep temp + record diff**

```bash
# Run only the matching branch:
# (a) byte-equal: delete temp
rm output/pairwise_peer_a_repro.csv && echo "TEMP-CLEAN"
# (b) mismatch: do not delete; capture sample diff and proceed with the
# repro file as the new ground truth, with the discrepancy documented in
# the findings doc.
# diff <(head -200 output/pairwise_peer_a.csv) <(head -200 output/pairwise_peer_a_repro.csv) | head -40
```
Expected (a) `TEMP-CLEAN`. Note: this task does NOT commit anything -- it's a procedural gate for the spec's framing.

---

### Task 2: PEER_B regen + replace tracked

**Goal:** Run `train_peer_stage1.py --peer b` against the clean v4 feature pipeline. Expected to differ from the tracked PR 14 output because PEER_B includes the seven Vegas features filtered by PR 19. Replace the tracked CSV with the new run output.

**Files:**
- Modify: `output/pairwise_peer_b.csv` (overwrite with new run output)
- Read-only: `src/train_peer_stage1.py`, tracked `output/pairwise_peer_b.csv` (md5 captured first)

- [ ] **Step 1: Capture md5 of tracked file**

```bash
md5sum output/pairwise_peer_b.csv | tee output/_md5_peer_b_tracked.txt
```
Expected: a 32-char hex hash. Note for findings doc (this is the "leaky" PEER_B md5).

- [ ] **Step 2: Run PEER_B trainer (overwriting tracked)**

```bash
python src/train_peer_stage1.py --peer b 2>&1 | tail -10
```
The trainer wipes any prior output at its default path (line 165 of `train_peer_stage1.py`) and writes fresh. Expected: per-season summary lines + final summary; total_pairs ~48,465. Massey/Colley/efficiency caches reused from Task 1, so this should be much faster than Task 1 (~3-5 min).

- [ ] **Step 3: Capture md5 of new file**

```bash
md5sum output/pairwise_peer_b.csv | tee output/_md5_peer_b_clean.txt
```
Expected: a 32-char hex hash differing from Step 1's. If equal, halt -- the leak fix did not propagate to PEER_B (which would mean PR 19's `filter_vegas_to_pre_tournament()` is not being called; would invalidate PR 21's clean baseline as well).

- [ ] **Step 4: Verify schema unchanged**

```bash
head -1 output/pairwise_peer_b.csv && wc -l output/pairwise_peer_b.csv
```
Expected: header `season,team_a,team_b,p_a_wins`; ~48,466 lines (header + 48,465 rows).

---

### Task 3: Run clean-baseline diagnostic + sanity gates

**Goal:** Run the 3-clause gate with clean inputs and verify all 4 sanity gates pass (matched-game count, v4 LL, PEER_A LL invariance, PEER_B coverage match).

**Files:**
- Reads: `output/pairwise_v4.csv`, `output/pairwise_peer_a.csv`, `output/pairwise_peer_b.csv`, `data/raw/march-machine-learning-2026/MNCAATourneyCompactResults.csv`
- Writes: `output/diag_feature_view_ensemble.json` (overwrite)

- [ ] **Step 1: Confirm input files exist + are the expected shape**

```bash
ls -la output/pairwise_v4.csv output/pairwise_peer_a.csv output/pairwise_peer_b.csv && wc -l output/pairwise_v4.csv output/pairwise_peer_a.csv output/pairwise_peer_b.csv
```
Expected: each ~48,466 lines. v4 from PR 23 force-add; peer_a unchanged (or replaced if Task 1 mismatched); peer_b replaced this PR.

- [ ] **Step 2: Run the diagnostic**

```bash
python src/diagnose_feature_view_ensemble.py 2>&1 | tee output/_diag_stdout.txt
```
Expected stdout: header, n_played_games line, per-game LL section, optimal weights section, residual r line, clause checks, VERDICT line. Exit code 0 if PASS, 1 if FAIL -- both are acceptable; we ACT on the verdict in Task 5.

- [ ] **Step 3: Sanity gate -- `n_played_games == 1449`**

```bash
python -c "import json; d = json.load(open('output/diag_feature_view_ensemble.json')); n = d['diagnostic']['n_played_games']; print('n', n); assert n == 1449, n"
```
Expected: `n 1449`. If different, halt -- pair coverage drift (likely a join/dedup issue).

- [ ] **Step 4: Sanity gate -- `ll_v4` matches clean baseline (within 0.005 LL)**

```bash
python -c "import json; d = json.load(open('output/diag_feature_view_ensemble.json')); ll = d['diagnostic']['ll_v4']; print(f'll_v4 = {ll:.4f}'); assert abs(ll - 0.5588) < 0.005, ll"
```
Expected: `ll_v4 ~ 0.555-0.563`. If outside, halt -- the diagnostic is not seeing the clean v4 we think.

- [ ] **Step 5: Sanity gate -- PEER_A LL invariant (within 0.001 LL of PR 14's 0.5720)**

```bash
python -c "import json; d = json.load(open('output/diag_feature_view_ensemble.json')); ll = d['diagnostic']['ll_peer_a']; print(f'll_peer_a = {ll:.4f}'); assert abs(ll - 0.5720) < 0.001, ll"
```
Expected: `ll_peer_a ~ 0.5715-0.5725`. Tighter tolerance because Task 1 should have proved byte-equality. If outside this band but Task 1 said byte-equal, the diagnostic's join/dedup behaviour drifted (unlikely but flag-worthy).

- [ ] **Step 6: Capture verdict + headline numbers for Task 5**

```bash
python -c "
import json
d = json.load(open('output/diag_feature_view_ensemble.json'))
g = d['gate']
diag = d['diagnostic']
print(f\"VERDICT: {'PASS' if g['pass'] else 'FAIL'}\")
print(f\"  reason: {g['reason']}\")
print(f\"  n_played_games: {diag['n_played_games']}\")
print(f\"  ll_v4:        {diag['ll_v4']:.4f}\")
print(f\"  ll_peer_a:    {diag['ll_peer_a']:.4f}  (delta_a {diag['clauses']['per_peer_ll_ceiling']['delta_a']:+.4f})\")
print(f\"  ll_peer_b:    {diag['ll_peer_b']:.4f}  (delta_b {diag['clauses']['per_peer_ll_ceiling']['delta_b']:+.4f})\")
print(f\"  ll_2blend:    {diag['ll_2blend_optimal']:.4f}\")
print(f\"  ll_3blend:    {diag['ll_3blend_optimal']:.4f}\")
print(f\"  w_2blend:     A={diag['w_2blend_optimal']:.3f}, B={1-diag['w_2blend_optimal']:.3f}\")
print(f\"  w_3blend:     v4={diag['w_3blend_optimal'][0]:.3f}, A={diag['w_3blend_optimal'][1]:.3f}, B={diag['w_3blend_optimal'][2]:.3f}\")
print(f\"  rho_residual: {diag['rho_residual']:+.3f}\")
print(f\"  headroom:     {diag['headroom_2blend_vs_v4']:+.4f}\")
print(f\"  clauses: pll={diag['clauses']['per_peer_ll_ceiling']['pass']}, rho={diag['clauses']['residual_correlation']['pass']}, headroom={diag['clauses']['blend_headroom']['pass']}\")
" | tee output/_diag_summary.txt
```
Expected: a multi-line summary saved to `output/_diag_summary.txt`. Read in Task 5 step 1.

---

### Task 4: Force-add diagnostic artifacts

**Goal:** Persist the new diagnostic JSON + the regenerated PEER_B CSV (and optionally PEER_A if Task 1 mismatched). Per the canonical-artifact policy in `docs/data_recovery.md`, artifacts under gitignored `output/` get force-added when load-bearing for a finding.

**Files:**
- Force-add (always): `output/diag_feature_view_ensemble.json`, `output/pairwise_peer_b.csv`
- Force-add (conditional, only on Task 1 mismatch): `output/pairwise_peer_a.csv`

- [ ] **Step 1: Confirm files exist and are non-empty**

```bash
ls -la output/diag_feature_view_ensemble.json output/pairwise_peer_b.csv
```
Expected: JSON ~1-2 KB; pairwise_peer_b.csv ~2-3 MB.

- [ ] **Step 2: Force-add + commit (PEER_B + diag JSON; PEER_A on mismatch only)**

```bash
git add -f output/diag_feature_view_ensemble.json output/pairwise_peer_b.csv
# If Task 1 mismatched, also include:
# git add -f output/pairwise_peer_a.csv

git commit -m "$(cat <<'EOF'
data(feature-view-clean-rerun): force-add diag JSON + clean PEER_B csv

output/diag_feature_view_ensemble.json: re-emitted under clean
pairwise_v4.csv (PR 23 baseline) + clean-trained PEER_B. Replaces
PR 14's leaky-baseline numbers.

output/pairwise_peer_b.csv: regenerated by train_peer_stage1.py --peer b
against the clean v4 feature pipeline (post-PR-19 vegas leak fix).
PEER_B's 27 features include 7 Vegas features that shifted under the
fix. Old md5 <<...>>; new md5 <<...>>.

PEER_A unchanged: its 40 features (efficiency, four factors, KenPom,
Massey, conf strength, season summary) contain no Vegas inputs and
reproduce byte-identical from the trainer under clean baseline. Tracked
output/pairwise_peer_a.csv from PR 14 stays.

Both force-added per docs/data_recovery.md canonical-artifact policy
(load-bearing audit artifacts under gitignored output/).
EOF
)"
```
Expected: `[feat/feature-view-clean-rerun <hash>] data(feature-view-clean-rerun): ...`. 2 (or 3 on mismatch) files changed.

- [ ] **Step 3: Cleanup scratch files**

```bash
rm -f output/_md5_peer_a_tracked.txt output/_md5_peer_a_repro.txt output/_md5_peer_b_tracked.txt output/_md5_peer_b_clean.txt output/_diag_stdout.txt
```
Note: keep `output/_diag_summary.txt` until after Task 5; remove at the very end.

---

### Task 5: Write findings doc + apply decision matrix to TODO.md

**Goal:** Document the verdict + new numbers vs PR 14, and advance the priority list per the spec's decision matrix.

**Files:**
- Create: `docs/notes/2026-05-05-feature-view-clean-rerun.md`
- Modify: `TODO.md` (recovery section, "step 5 sub-priorities" list)

- [ ] **Step 1: Read the verdict summary from Task 3**

```bash
cat output/_diag_summary.txt
```
Note all values. All filled into the findings doc as actual numbers -- no placeholders. Delete `output/_diag_summary.txt` at end of Task 5.

- [ ] **Step 2: Write the findings doc**

Create `docs/notes/2026-05-05-feature-view-clean-rerun.md`. Mirror PR 24's structure (TL;DR, methods, gate result table, standalone metrics, comparison-to-PR-14, discussion, verdict, recommendation, files-of-record, follow-ups). Replace `<<...>>` placeholders with actual values.

Required sections:
- **TL;DR**: one paragraph. Verdict + key numbers (peer LLs, residual r, headroom). One sentence on what flipped vs PR 14.
- **Methods**: input file md5s, diagnostic command, n_played_games, threshold values inherited from PR 14. Note the data wipe + recovery (extracted training_data.tar.gz, set up junctions).
- **Gate result table**: 3 clauses with values + PASS/FAIL.
- **Standalone metrics table**: clean v4 / PEER_A / PEER_B LL + accuracy.
- **3-blend side observation**: w_3blend, ll_3blend, what it suggests.
- **Comparison to PR 14**: side-by-side. Key columns: ll_v4, ll_peer_a, ll_peer_b, delta_a, delta_b, rho, optimal_w_2blend, headroom, gate verdict.
- **Discussion**: 3-4 paragraphs. Touch points:
  - Whether PEER_A LL was invariant as predicted (the load-bearing claim).
  - How PEER_B LL shifted (delta vs leaky 0.4566).
  - How rho changed -- compare to PR 24's r=0.868 finding for v4 vs BT residuals (different quantity but related: do feature-view peers also share clean-v4's "hard regular-season-information" failure mode?).
  - What the 3-blend optimum suggests about PEER_B as a residual feature on top of v4 (PR 14's side observation: w_3blend = (0.757, 0.0, 0.243); compare to clean number).
- **Verdict + recommendation**: branch on PASS / FAIL.

- [ ] **Step 3: Update `TODO.md` per decision matrix**

Find the "step 5 sub-priorities" list under "5. Re-run the swap-decided / swap-candidate evaluations against the clean baseline". Apply ONE of the two patches:

**If GATE PASSED:** Mark "Feature-view ensemble PEER_A/B" sub-priority done with verdict + numbers; promote "Feature-view ensemble E1+E2 sweeps" to next sub-priority position. Find the bullet currently reading:

```markdown
       - Feature-view ensemble PEER_A/B (PR 14): PEER_A delta vs v4
         was +0.1375 vs leaky; +0.013 vs clean (within 5x clause-1
         tolerance); clause 1 likely flips PASS. **NEXT IMMEDIATE PR.**
```

Replace with:

```markdown
       - **[DONE -- PR <pending>]** Feature-view ensemble PEER_A/B re-eval. **GATE PASSED**
         under clean baseline. delta_a=<<+x.xxxx>>, delta_b=<<+x.xxxx>>,
         rho=<<x.xx>>, w_2blend=(<<x.xx>>, <<x.xx>>), headroom=<<+x.xxxx>> LL.
         <<one-sentence "what flipped">>. Findings:
         `docs/notes/2026-05-05-feature-view-clean-rerun.md`.
       - **Feature-view ensemble E1+E2 sweeps -- NOW THE IMMEDIATE
         NEXT PR.** Per PR 14's gated sweep design: E1 = blend(peer_A, peer_B)
         no v4; E2 = blend(v4, peer_A, peer_B). ~hours compute.
```

**If GATE FAILED:** Mark "Feature-view ensemble PEER_A/B" sub-priority done with FAIL verdict; do not promote E1+E2; advance to next remaining marginal-rejection (HBT). Replace the bullet with:

```markdown
       - **[DONE -- PR <pending>]** Feature-view ensemble PEER_A/B re-eval. **GATE FAILED**
         under clean baseline. delta_a=<<+x.xxxx>>, delta_b=<<+x.xxxx>>,
         rho=<<x.xx>>, w_2blend=(<<x.xx>>, <<x.xx>>), headroom=<<+/-x.xxxx>> LL.
         <<which clauses fail and why>>. Robust NO-GO across both leaky
         and clean baselines closes feature-view ensemble at K=2 semantic
         split. Findings: `docs/notes/2026-05-05-feature-view-clean-rerun.md`.
```

- [ ] **Step 4: Verify TODO edits compile (no broken markdown)**

```bash
head -150 TODO.md | tail -80
```
Expected: the "step 5 sub-priorities" section reads cleanly.

- [ ] **Step 5: Commit findings + TODO update**

```bash
rm output/_diag_summary.txt
git add docs/notes/2026-05-05-feature-view-clean-rerun.md TODO.md docs/superpowers/plans/2026-05-05-feature-view-clean-rerun.md
git commit -m "$(cat <<'EOF'
docs(feature-view-clean-rerun): findings + TODO update -- recovery step 5 marginal #2

Verdict: <<PASS or FAIL>> (delta_a=<<+x.xxxx>>, rho=<<x.xx>>, headroom=<<+/-x.xxxx>>).
<<one-sentence summary of what flipped or didn't and what comes next>>.

Findings: docs/notes/2026-05-05-feature-view-clean-rerun.md.
Plan: docs/superpowers/plans/2026-05-05-feature-view-clean-rerun.md.
TODO step 5 sub-priority list advanced per spec decision matrix.
EOF
)"
```
Expected: `[feat/feature-view-clean-rerun <hash>]`. 3 files changed.

---

## Self-review checklist (the implementer should run this at end of plan)

- [ ] All relevant tests pass: `python -m pytest tests/test_diagnose_feature_view_ensemble.py tests/test_train_peer_stage1.py tests/test_feature_views.py tests/test_ensemble_stage1.py -q`. Expect 23+ passed.
- [ ] Three commits on the branch: spec (committed at the start of plan execution), data (Task 4), docs (Task 5). PEER_A force-add is a fourth commit only on mismatch.
- [ ] `git log --oneline main..HEAD` shows the spec, plan, data, and docs commits in order.
- [ ] `output/diag_feature_view_ensemble.json` shows clean-baseline numbers (`ll_v4 ~ 0.555-0.563`, `ll_peer_a ~ 0.5720`).
- [ ] Findings doc has zero `<<...>>` placeholders.
- [ ] `TODO.md` step 5 list reflects the actual verdict.
- [ ] No scratch files left under `output/_*.txt`.
- [ ] Worktree on branch `feat/feature-view-clean-rerun`, ready to push + open PR.
