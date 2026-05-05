# Plain BT Standalone Clean Re-eval Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Re-run PR 12's 3-clause LL-blend gate (`src/diagnose_bt_vs_v4.py`) with the clean `pairwise_v4.csv` from main (PR 23's force-add) and apply a verdict-dependent decision: PASS → promote plain BT bracket-points re-test (PR 17 redo) to next sub-priority; FAIL → drop plain BT from the marginal-rejections list. Add a `--curve-out` flag to expose the full LL(w) curve so the findings doc can characterize shape, not just optimum.

**Architecture:** Single-script change (`src/diagnose_bt_vs_v4.py`) gated by a 1-test TDD cycle, then a one-shot diagnostic run against existing tracked CSVs (`output/pairwise_bt.csv` from PR 12, `output/pairwise_v4.csv` from PR 23). All compute is small (~10 sec total). No production-side changes. Reproducibility check (BT byte-equality) before drawing conclusions.

**Tech Stack:** Python 3.11, pandas, numpy, scikit-learn (already in repo), pytest. Worktree-based isolation; data junction at `data/raw/march-machine-learning-2026/`.

**Branch:** `feat/plain-bt-clean-rerun` (worktree at `.claude/worktrees/feat-plain-bt-clean-rerun`)
**Spec:** `docs/superpowers/specs/2026-05-05-plain-bt-clean-rerun-design.md`

---

### Task 1: BT byte-equality reproducibility check

**Goal:** Confirm `output/pairwise_bt.csv` (tracked from PR 12) reproduces byte-for-byte from `train_bt_stage1.py` against current data + library state. If not, halt and investigate before drawing conclusions.

**Files:**
- Read-only: `src/train_bt_stage1.py`, tracked `output/pairwise_bt.csv`
- Temp: `output/pairwise_bt_repro.csv` (deleted after compare)

- [ ] **Step 1: Confirm temp path doesn't exist (append-mode safety)**

```bash
ls output/pairwise_bt_repro.csv 2>&1 | grep -q "No such" && echo "OK: no preexisting temp" || (echo "ERROR: temp exists, delete first"; rm output/pairwise_bt_repro.csv)
```
Expected: `OK: no preexisting temp` (or auto-cleanup then OK).

- [ ] **Step 2: Capture md5 of tracked file**

```bash
md5sum output/pairwise_bt.csv
```
Expected: a 32-char hex hash + `output/pairwise_bt.csv`. Note the hex hash for the findings doc.

- [ ] **Step 3: Run BT trainer to temp path (~8 sec)**

```bash
python src/train_bt_stage1.py --out output/pairwise_bt_repro.csv 2>&1 | tail -5
```
Expected: per-season summary lines + final summary. No tracebacks.

- [ ] **Step 4: Capture md5 of repro file**

```bash
md5sum output/pairwise_bt_repro.csv
```
Expected: same 32-char hex hash as Step 2 (hashes should match).

- [ ] **Step 5: Verify byte-equality (halt on mismatch)**

```bash
cmp output/pairwise_bt.csv output/pairwise_bt_repro.csv && echo "BYTE-EQUAL: proceed" || echo "MISMATCH: halt and investigate"
```
Expected: `BYTE-EQUAL: proceed`. If `MISMATCH`, stop the plan, surface the diff to the user, do NOT continue to Task 2 — the assumption "BT is unchanged across leaky→clean transition" is load-bearing for the spec's conclusions.

- [ ] **Step 6: Cleanup temp file**

```bash
rm output/pairwise_bt_repro.csv && echo "TEMP-CLEAN"
```
Expected: `TEMP-CLEAN`. No file in `output/` named `pairwise_bt_repro.csv`. Note: this task does NOT commit anything — it's a procedural gate.

---

### Task 2: Add `--curve-out` flag (TDD)

**Goal:** Expose the full LL(w) blend curve as a 2-column CSV for findings-doc plotting. JSON shape unchanged for backward compat.

**Files:**
- Modify: `src/diagnose_bt_vs_v4.py`
- Modify: `tests/test_diagnose_bt_vs_v4.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_diagnose_bt_vs_v4.py`:

```python
def test_curve_csv_written_with_101_rows_matching_optimum(tmp_path):
    """`--curve-out` writes a 2-column CSV: 101 rows of (w, ll_blend),
    monotonic w from 0.00 to 1.00, value at the optimum row equals
    optimal_ll from the JSON to 1e-6."""
    import json
    import pandas as pd
    from src.diagnose_bt_vs_v4 import main

    pw_a = tmp_path / "v4.csv"
    pw_b = tmp_path / "bt.csv"
    _write_pairwise(pw_a, [
        (2003, 1100 + i, 1200 + i, 0.85) for i in range(10)
    ])
    _write_pairwise(pw_b, [
        (2003, 1100 + i, 1200 + i, 0.55) for i in range(10)
    ])
    results_csv = tmp_path / "results.csv"
    pd.DataFrame([
        {"Season": 2003, "WTeamID": 1100 + i, "LTeamID": 1200 + i, "DayNum": 136}
        for i in range(10)
    ]).to_csv(results_csv, index=False)

    out_json = tmp_path / "diag.json"
    out_curve = tmp_path / "curve.csv"
    # main() reads results from DATA constant by default; we monkey-
    # patch by setting cwd OR (simpler) calling compute_diagnostic + the
    # new helper directly. Use main() with a custom results path via
    # passing --pairwise-v4 / --pairwise-bt that compute_diagnostic loads,
    # and rely on results_df in the inner function path for unit testing.
    # Pattern matches existing tests' use of compute_diagnostic.
    from src.diagnose_bt_vs_v4 import compute_diagnostic, _write_curve
    diag = compute_diagnostic(str(pw_a), str(pw_b), results_df=pd.read_csv(results_csv))
    _write_curve(str(out_curve), diag["ll_at_w"])

    df = pd.read_csv(out_curve)
    assert list(df.columns) == ["w", "ll_blend"]
    assert len(df) == 101
    assert df["w"].iloc[0] == pytest.approx(0.0)
    assert df["w"].iloc[-1] == pytest.approx(1.0)
    # Row at the optimum w (rounded to nearest 0.01) matches optimal_ll.
    opt_row_idx = round(diag["optimal_w"] * 100)
    assert df["ll_blend"].iloc[opt_row_idx] == pytest.approx(diag["optimal_ll"], abs=1e-6)
```

- [ ] **Step 2: Run the new test, verify failure**

```bash
python -m pytest tests/test_diagnose_bt_vs_v4.py::test_curve_csv_written_with_101_rows_matching_optimum -v 2>&1 | tail -10
```
Expected: FAIL with `ImportError: cannot import name '_write_curve'` (the helper doesn't exist yet).

- [ ] **Step 3: Implement `_write_curve` helper in `src/diagnose_bt_vs_v4.py`**

Insert immediately after the existing imports block, before the constants (around line 23, before `DATA = Path(...)`):

```python
def _write_curve(path: str, ll_at_w: list) -> None:
    """Write the LL(w) blend curve to a 2-column CSV.

    Format: header `w,ll_blend`; 101 data rows for w in [0.00, 1.00]
    step 0.01. Both columns formatted to 6 decimals.
    """
    import numpy as np
    ws = np.linspace(0.0, 1.0, 101)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write("w,ll_blend\n")
        for w, ll in zip(ws, ll_at_w):
            f.write(f"{w:.6f},{ll:.6f}\n")
```

- [ ] **Step 4: Add `--curve-out` arg + call site to `main()`**

In the `main()` function, after `parser.add_argument("--out-json", ...)` (currently line 187), add:

```python
    parser.add_argument("--curve-out", default="output/diag_bt_vs_v4_curve.csv",
                        help="Where to write the LL(w) blend curve (CSV)")
```

Then, in `main()` after `compute_diagnostic(...)` returns and before the JSON write (currently around line 194), add:

```python
    _write_curve(args.curve_out, diag["ll_at_w"])
    print(f"  saved {args.curve_out}")
```

- [ ] **Step 5: Run the new test, verify pass**

```bash
python -m pytest tests/test_diagnose_bt_vs_v4.py::test_curve_csv_written_with_101_rows_matching_optimum -v 2>&1 | tail -10
```
Expected: PASS.

- [ ] **Step 6: Run full diagnostic + trainer test suite, verify no regressions**

```bash
python -m pytest tests/test_diagnose_bt_vs_v4.py tests/test_train_bt_stage1.py -q 2>&1 | tail -10
```
Expected: 8 passed (previously 7; +1 new).

- [ ] **Step 7: Commit code change**

```bash
git add src/diagnose_bt_vs_v4.py tests/test_diagnose_bt_vs_v4.py
git commit -m "$(cat <<'EOF'
feat(plain-bt-clean-rerun): expose full LL(w) curve via --curve-out flag

src/diagnose_bt_vs_v4.py already computes the 101-cell LL(w) curve
internally (line ~98) but drops it from the saved JSON to keep the
file slim. Add --curve-out flag (default output/diag_bt_vs_v4_curve.csv)
that always writes the curve as a 2-column CSV (w, ll_blend). JSON
shape unchanged. Adds one unit test asserting 101 rows + value at
optimum row matches optimal_ll to 6 decimals.

Recovery step 5 marginal #1 prep for the actual re-eval against clean
pairwise_v4.csv.
EOF
)"
```
Expected: `[feat/plain-bt-clean-rerun <hash>] feat(plain-bt-clean-rerun): expose full LL(w) curve via --curve-out flag`. 2 files changed.

---

### Task 3: Run clean-baseline diagnostic + sanity gates

**Goal:** Run the diagnostic with the clean `pairwise_v4.csv` from main and verify all 3 sanity gates pass. Capture the gate verdict for use in Task 5.

**Files:**
- Reads: `output/pairwise_v4.csv`, `output/pairwise_bt.csv`, `data/raw/march-machine-learning-2026/MNCAATourneyCompactResults.csv`
- Writes: `output/diag_bt_vs_v4.json` (overwrite), `output/diag_bt_vs_v4_curve.csv` (new file)

- [ ] **Step 1: Confirm input files exist + are the expected versions**

```bash
ls -la output/pairwise_v4.csv output/pairwise_bt.csv && wc -l output/pairwise_v4.csv output/pairwise_bt.csv
```
Expected: `pairwise_v4.csv` ~48,466 lines (header + 48,465 rows, per `data_recovery.md`), `pairwise_bt.csv` ~48,466 lines.

- [ ] **Step 2: Run the diagnostic**

```bash
python src/diagnose_bt_vs_v4.py --pairwise-v4 output/pairwise_v4.csv --pairwise-bt output/pairwise_bt.csv 2>&1 | tee output/_diag_stdout.txt
```
Expected stdout: standalone metrics, residual r, disagreement breakdown, optimal-weight section, VERDICT line. Two `saved` lines at end (curve + JSON). Process exit code 0 if PASS, 1 if FAIL — both are acceptable (we ACT on the verdict in Task 5, we don't fail the task on FAIL).

- [ ] **Step 3: Sanity gate — `n_games == 1449`**

```bash
python -c "import json; d = json.load(open('output/diag_bt_vs_v4.json')); print('n_games', d['diagnostic']['n_games']); assert d['diagnostic']['n_games'] == 1449"
```
Expected: `n_games 1449`. If different, halt — the matched-game set has drifted from PR 12; reconcile before proceeding (likely a pair-coverage issue in `pairwise_v4.csv`).

- [ ] **Step 4: Sanity gate — `ll_v4` matches clean baseline (within 0.005 LL)**

```bash
python -c "import json; d = json.load(open('output/diag_bt_vs_v4.json')); ll = d['diagnostic']['ll_v4']; print(f'll_v4 = {ll:.4f}'); assert abs(ll - 0.5588) < 0.005, f'll_v4 {ll:.4f} too far from clean baseline 0.5588'"
```
Expected: `ll_v4 ≈ 0.555-0.563`. If outside that band, halt — the diagnostic is not seeing the clean v4 we think it is (could be the wrong CSV, a stale dedup, or a join bug).

- [ ] **Step 5: Sanity gate — curve CSV shape**

```bash
wc -l output/diag_bt_vs_v4_curve.csv && head -3 output/diag_bt_vs_v4_curve.csv && tail -1 output/diag_bt_vs_v4_curve.csv
```
Expected: 102 lines (header + 101 data); header `w,ll_blend`; first data row `0.000000,...`; last row `1.000000,...`.

- [ ] **Step 6: Capture gate verdict + headline numbers for Task 5**

```bash
python -c "
import json
d = json.load(open('output/diag_bt_vs_v4.json'))
g = d['gate']
diag = d['diagnostic']
print(f\"VERDICT: {'PASS' if g['pass'] else 'FAIL'}\")
print(f\"  reason: {g['reason']}\")
print(f\"  r_residual: {diag['r_residual']:.4f}\")
print(f\"  optimal_w:  {diag['optimal_w']:.2f}\")
print(f\"  headroom:   {diag['headroom']:+.4f}\")
print(f\"  ll_v4:      {diag['ll_v4']:.4f}\")
print(f\"  ll_bt:      {diag['ll_bt']:.4f}\")
print(f\"  acc_v4:     {diag['acc_v4']:.4f}\")
print(f\"  acc_bt:     {diag['acc_bt']:.4f}\")
" | tee output/_diag_summary.txt
```
Expected: a 9-line summary saved to `output/_diag_summary.txt` (gitignored under `output/`; the underscore prefix marks it as scratch and it does not get force-added). Read it in Task 5 step 1.

---

### Task 4: Force-add diagnostic artifacts

**Goal:** Persist the JSON + curve CSV to git per the canonical-artifact policy. These files are gitignored under `output/` but get force-added when load-bearing for a finding.

**Files:**
- Force-add: `output/diag_bt_vs_v4.json`, `output/diag_bt_vs_v4_curve.csv`

- [ ] **Step 1: Confirm both files exist and are non-empty**

```bash
ls -la output/diag_bt_vs_v4.json output/diag_bt_vs_v4_curve.csv
```
Expected: JSON ~600 bytes; curve CSV ~3-4 KB.

- [ ] **Step 2: Force-add + commit**

```bash
git add -f output/diag_bt_vs_v4.json output/diag_bt_vs_v4_curve.csv
git commit -m "$(cat <<'EOF'
data(plain-bt-clean-rerun): force-add diag JSON + curve CSV under clean v4

output/diag_bt_vs_v4.json: re-emitted under clean pairwise_v4.csv
(PR 23 baseline). Replaces PR 12's leaky-baseline numbers.

output/diag_bt_vs_v4_curve.csv: new artifact; 101-row LL(w) blend curve
emitted by --curve-out flag from the previous commit. Used by the
findings note to characterize blend shape across the full w range.

Both force-added per docs/data_recovery.md canonical-artifact policy
(load-bearing audit artifacts under gitignored output/).
EOF
)"
```
Expected: `[feat/plain-bt-clean-rerun <hash>] data(plain-bt-clean-rerun): force-add diag JSON + curve CSV under clean v4`. 2 files changed.

---

### Task 5: Write findings doc + apply decision matrix to TODO.md

**Goal:** Document the verdict + new numbers vs PR 12, and advance the priority list per the spec's decision matrix. The findings doc structure is fixed; the TODO.md update branches on PASS vs FAIL.

**Files:**
- Create: `docs/notes/2026-05-05-plain-bt-clean-rerun.md`
- Modify: `TODO.md` (recovery section, "step 5 sub-priorities" list)

- [ ] **Step 1: Read the verdict summary from Task 3**

```bash
cat output/_diag_summary.txt
```
Note the values (`r_residual`, `optimal_w`, `headroom`, `ll_v4`, `ll_bt`, `acc_v4`, `acc_bt`, verdict). All filled into the findings doc as actual numbers — no placeholders. After Task 5's commit, delete the scratch file: `rm output/_diag_stdout.txt output/_diag_summary.txt`.

- [ ] **Step 2: Write the findings doc**

Create `docs/notes/2026-05-05-plain-bt-clean-rerun.md` with the structure below. Replace `<<...>>` placeholders with actual values from `/tmp/diag_summary.txt`. The verdict line uses the actual PASS/FAIL result. The "Recommendation" section branches on the verdict.

```markdown
# Plain BT Standalone Re-eval (Clean Baseline) -- Findings

**Date:** 2026-05-05
**Branch:** feat/plain-bt-clean-rerun
**Verdict:** **<<PASS or FAIL>>.** Gate clauses: r=<<r>>, optimal_w=<<w>>, headroom=<<+/-x.xxxx>>.
**Spec:** `docs/superpowers/specs/2026-05-05-plain-bt-clean-rerun-design.md`
**Plan:** `docs/superpowers/plans/2026-05-05-plain-bt-clean-rerun.md`
**Recovery context:** TODO.md "CONTAMINATION DISCOVERED 2026-05-04" -> step 5,
sub-priority "Plain BT standalone re-eval" (named highest signal/noise of the
marginal-rejections list in `docs/notes/2026-05-04-v9c-clean-rerun.md` § Follow-ups).

## TL;DR

Re-running PR 12's 3-clause LL-blend gate against the clean
`pairwise_v4.csv` (PR 23 force-added) <<flips/keeps>> the verdict.
Standalone LL: clean v4 <<x.xxxx>>, BT <<x.xxxx>> (delta <<+/-x.xxxx>>).
Optimal blend `w_v4=<<x.xx>>` <<inside/outside>> the gate band [0.30, 0.85];
headroom <<+/-x.xxxx>> LL <<above/below>> the 0.005 threshold; residual
correlation r=<<x.xxx>> <<below/above>> 0.60. <<One-sentence "what flipped
vs PR 12 and why">>.

## Methods

- Inputs (read-only):
  - `output/pairwise_v4.csv` (clean baseline, force-added in PR 23,
    md5 `<<...>>`).
  - `output/pairwise_bt.csv` (PR 12 force-add, byte-equal to a fresh
    `train_bt_stage1.py` rerun this PR -- md5 `<<tracked-md5>>`).
- Diagnostic: `python src/diagnose_bt_vs_v4.py --pairwise-v4 output/pairwise_v4.csv
  --pairwise-bt output/pairwise_bt.csv`. Same gate thresholds as PR 12
  (`GATE_R_MAX=0.60`, `GATE_W_LOW=0.30`, `GATE_W_HIGH=0.85`,
  `GATE_HEADROOM_MIN=0.005`).
- Procedure-side change this PR: added `--curve-out` flag (default
  `output/diag_bt_vs_v4_curve.csv`) so the full LL(w) curve is persisted
  alongside the slim JSON. Curve has 101 cells (`w` in [0.00, 1.00]
  step 0.01).
- Matched-game count: `n_games = 1449` (identical to PR 12).
  Reproducibility: BT csv byte-equal to tracked.

## Gate result

| measure                               | value     | clause              |
|---------------------------------------|-----------|---------------------|
| Pearson r(residual_v4, residual_bt)   | **<<x.xxx>>** | **<<PASS or FAIL>>** (< 0.60)     |
| optimal blend weight w_v4 (cheating)  | **<<x.xx>>**  | **<<PASS or FAIL>>** ([0.30, 0.85]) |
| headroom = LL_v4 - LL_optimal         | **<<+/-x.xxxx>>** | **<<PASS or FAIL>>** (> 0.005)    |
| **gate verdict**                      | -         | **<<PASS or FAIL>>**          |

## Standalone metrics (1449 played 2003-2025 tournament games)

| metric                 | clean v4 | BT     |
|------------------------|----------|--------|
| weighted-mean log loss | <<x.xxxx>> | <<x.xxxx>> |
| weighted-mean accuracy | <<x.xxx>>  | <<x.xxx>>  |

## Disagreement breakdown

| outcome                  | count | %      |
|--------------------------|-------|--------|
| both correct             | <<n>> | <<x.x>>%  |
| v4 only correct          | <<n>> | <<x.x>>%  |
| BT only correct          | <<n>> | <<x.x>>%  |
| both wrong               | <<n>> | <<x.x>>%  |
| total disagreements      | <<n>> | <<x.x>>%  |

When v4 and BT disagree on the predicted winner, BT is right
<<n>>/(<<n>>+<<n>>) = <<x.x>>% of the time (vs PR 12's 27.9%).

## Selected w values from `diag_bt_vs_v4_curve.csv`

| w    | ll_blend  |
|------|-----------|
| 0.00 | <<x.xxxx>> (= ll_bt) |
| 0.25 | <<x.xxxx>> |
| 0.50 | <<x.xxxx>> |
| <<optimal_w>> | <<x.xxxx>> (= optimal_ll) |
| 0.75 | <<x.xxxx>> |
| 1.00 | <<x.xxxx>> (= ll_v4) |

Shape: <<one sentence on shape, e.g. "monotone decreasing in w" or
"shallow U with optimum near w=0.65" or "essentially flat across [0.4, 0.9]">>.

## Comparison to PR 12 (leaky baseline)

| measure                              | PR 12 (leaky) | this PR (clean) |
|--------------------------------------|---------------|-----------------|
| ll_v4                                | 0.4369        | <<x.xxxx>>      |
| ll_bt                                | 0.5650        | <<x.xxxx>>      |
| delta                                | -0.1281 (BT weaker) | <<+/-x.xxxx>> |
| residual r                           | 0.577         | <<x.xxx>>       |
| optimal_w                            | 0.98          | <<x.xx>>        |
| headroom                             | +0.0000       | <<+/-x.xxxx>>   |
| disagreement rate                    | 24.0%         | <<x.x>>%        |
| BT-when-disagree correct             | 27.9%         | <<x.x>>%        |
| gate verdict                         | FAIL          | <<PASS or FAIL>>      |

## Discussion

<<2-4 sentences interpreting the result. Touch points:
- Whether the leak removal was sufficient to flip the gate, or only the
  strength-gap clauses, or neither.
- What this implies about BT-class peers (does the gate confirm BT is
  worth a bracket-points re-test, or close it for good?)
- How residual correlation r changed (we expect it to stay near 0.58
  since BT structure is unchanged; large drift would suggest a join /
  matched-set issue we should investigate).
- Whether disagreement rate moved -- if it dropped, v4 has gotten weaker
  in v4-correct cases; if it rose, BT and clean-v4 agree more often
  than BT and leaky-v4 did.>>

## Verdict + recommendation

<<branch on verdict>>

### If GATE PASSED:

Plain BT is a viable LL-blend partner with clean v4. The next experiment
in this thread is the bracket-points re-test (PR 17 redo): does an
LL-blend with `w_v4 in [<<x.xx>>, 0.95]` translate to bracket-points
gain over `v4-alone + v8 stage-2`? PR 17 found NO-GO on the leaky
baseline (every non-anchor cell lost; best non-anchor `w_v4=0.90` at
-29 pts), but PR 17 was on a baseline that has since shifted -601 brkt
pts (per PR 23). The re-test is ~3 hr compute (22-season v9-C backtest
× 6 sweep cells). **Promote to next sub-priority** (TODO.md update below).

### If GATE FAILED:

Plain BT does NOT clear the LL-blend gate even on the clean baseline.
The robust NO-GO across both baselines closes BT as a stage-1 ensemble
peer. Bracket-points re-test (PR 17 redo) skipped -- LL-gate failure
is sufficient to drop BT from the marginal-rejections list. Next
sub-priority becomes "Feature-view ensemble PEER_A/B re-eval" (~20 min
compute), which has weaker-but-similar standalone strength
characteristics and is the next most likely to flip from the clean
baseline shift.

## TODO.md update (this PR commits the update)

<<see TODO.md diff -- one of two paths depending on verdict, per spec
decision matrix>>

## Files of record

- `src/diagnose_bt_vs_v4.py` (modified: added `--curve-out` flag +
  `_write_curve` helper, ~15 lines)
- `tests/test_diagnose_bt_vs_v4.py` (modified: added 1 test for curve
  CSV, ~25 lines)
- `output/diag_bt_vs_v4.json` (overwritten with clean numbers; force-added)
- `output/diag_bt_vs_v4_curve.csv` (new tracked artifact, 101 rows)
- `docs/superpowers/specs/2026-05-05-plain-bt-clean-rerun-design.md`
- `docs/superpowers/plans/2026-05-05-plain-bt-clean-rerun.md`
```

- [ ] **Step 3: Update `TODO.md` per decision matrix**

Open `TODO.md` and find the "step 5 sub-priorities" list (around lines 82-127, under "5. Re-run the swap-decided / swap-candidate evaluations against the clean baseline"). Apply ONE of the two patches below depending on Task 3's verdict:

**If GATE PASSED:** Mark "Plain BT standalone re-eval" sub-priority done with verdict + numbers; promote "Plain BT bracket-points re-test (PR 17 finding)" to "Now the immediate next PR" position with explicit ~3 hr compute warning. Find the bullet currently reading `- **Plain BT bracket-points** (PR 17 finding) -- still pending.` and replace with:

```markdown
   - **[DONE -- PR <pending>]** Plain BT standalone re-eval. **GATE PASSED**
     under clean baseline. r=<<r>>, optimal_w=<<w>>, headroom=<<+x.xxxx>> LL.
     Standalone LL: clean v4 <<x.xxxx>>, BT <<x.xxxx>> (delta <<+/-x.xxxx>>).
     PR 12's two failing clauses (degenerate w_v4=0.98 and headroom=+0.0000)
     both flipped now that BT and v4 are within ~<<x.xxx>> LL of each other
     standalone. Findings: `docs/notes/2026-05-05-plain-bt-clean-rerun.md`.
   - **Plain BT bracket-points re-test (PR 17 redo) -- NOW THE IMMEDIATE
     NEXT PR.** ~3 hr compute (22-season v9-C backtest × 6 sweep cells).
     PR 17 found NO-GO on the leaky baseline but the baseline shifted
     -601 brkt pts in PR 23 -- the re-test is load-bearing for whether
     plain BT actually contributes to the production metric.
```

**If GATE FAILED:** Mark "Plain BT standalone re-eval" sub-priority done with FAIL verdict; remove "Plain BT bracket-points" entirely from the marginal-rejections list. Find the bullet currently reading `- **Plain BT bracket-points** (PR 17 finding) -- still pending.` and replace with:

```markdown
   - **[DONE -- PR <pending>]** Plain BT standalone re-eval. **GATE FAILED**
     under clean baseline. r=<<r>>, optimal_w=<<w>>, headroom=<<+/-x.xxxx>>
     LL. <<which clauses fail>>. Robust NO-GO across both leaky and
     clean baselines closes plain BT as a stage-1 ensemble peer.
     Plain BT bracket-points re-test (PR 17 redo) skipped -- LL-gate
     failure is sufficient.
     Findings: `docs/notes/2026-05-05-plain-bt-clean-rerun.md`.
   - **Plain BT bracket-points** (PR 17 finding) -- DROPPED. LL-gate
     failure under clean baseline closes plain BT.
```

Also, in the "Five more added by the v9-C re-eval" subsection (around lines 105-115), update the line currently reading `       - Plain BT standalone (PR 12): standalone LL 0.565 = ~tied with clean v4 0.5588; LL-blend gate likely flips PASS.` to record the actual verdict:

**PASS path:**
```markdown
       - Plain BT standalone (PR 12): standalone LL 0.565 = ~tied with
         clean v4 0.5588; LL-blend gate flipped PASS this PR (see above).
```

**FAIL path:**
```markdown
       - Plain BT standalone (PR 12): standalone LL 0.565 = ~tied with
         clean v4 0.5588 BUT LL-blend gate stayed FAIL this PR -- the
         strength-gap collapse was insufficient (see above).
```

- [ ] **Step 4: Verify `TODO.md` edits compile (no broken markdown)**

```bash
head -130 TODO.md | tail -55
```
Expected: the "step 5 sub-priorities" section reads cleanly with the new "[DONE -- PR <pending>]" line and the next-priority advancement.

- [ ] **Step 5: Commit findings + TODO update**

```bash
git add docs/notes/2026-05-05-plain-bt-clean-rerun.md TODO.md docs/superpowers/plans/2026-05-05-plain-bt-clean-rerun.md
git commit -m "$(cat <<'EOF'
docs(plain-bt-clean-rerun): findings + TODO update -- recovery step 5 marginal #1

Verdict: <<PASS or FAIL>> (r=<<r>>, optimal_w=<<w>>, headroom=<<+/-x.xxxx>>).
<<one-sentence summary of what flipped or didn't and what comes next>>.

Findings: docs/notes/2026-05-05-plain-bt-clean-rerun.md.
Plan: docs/superpowers/plans/2026-05-05-plain-bt-clean-rerun.md.
TODO step 5 sub-priority list advanced per spec decision matrix.
EOF
)"
```
Expected: `[feat/plain-bt-clean-rerun <hash>]`. 3 files changed.

---

## Self-review checklist (the implementer should run this at end of plan)

- [ ] All tests pass: `python -m pytest tests/test_diagnose_bt_vs_v4.py tests/test_train_bt_stage1.py -q` returns 8 passed.
- [ ] Three commits on the branch: spec (already on branch from brainstorm), feat (Task 2), data (Task 4), docs (Task 5).
- [ ] `git log --oneline main..HEAD` shows: spec, feat, data, docs (4 commits).
- [ ] `output/diag_bt_vs_v4.json` shows clean-baseline numbers (`ll_v4 ≈ 0.555-0.563`).
- [ ] `output/diag_bt_vs_v4_curve.csv` exists with 102 lines.
- [ ] Findings doc has zero `<<...>>` placeholders (all replaced with actual values).
- [ ] `TODO.md` step 5 list reflects the actual verdict.
- [ ] Worktree is on branch `feat/plain-bt-clean-rerun`, ready to push + open PR.
