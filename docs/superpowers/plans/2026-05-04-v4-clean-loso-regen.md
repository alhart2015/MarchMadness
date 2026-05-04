# v4 clean LOSO regeneration -- Plan

**Date:** 2026-05-04
**Branch:** feat/v4-clean-loso-regen
**Spec:** `docs/superpowers/specs/2026-05-04-v4-clean-loso-regen-design.md`

## Approach

This is a regen + record PR, not a code-change PR. The leak fix
ships in main via PR 19; the Massey/KenPom audit ships in main via
PR 20. All we do here is run `enhanced_model_v3.py` end-to-end,
diff per-season metrics against the snapshotted leaky baseline,
and write up the shift.

Phase 1 captures the new baseline. Phase 2 documents and ships.
Each phase is small, atomic, and has explicit completion criteria.

## Phase 1: Capture clean LOSO numbers

### Step 1.1: Confirm pre-regen state

Already done at branch setup time:
- `output/cv_per_season_v3_leaky_snapshot.csv` exists in this
  worktree (copied from main repo's working tree before any regen).
- `output/cv_per_season_v3.csv` is absent in this worktree
  (will be created by the script).
- `output/pairwise_v4.csv` is absent in this worktree
  (will be created by the script via `MM_PAIRWISE_OUT`).

Verify:

```
ls output/cv_per_season_v3_leaky_snapshot.csv  # must exist
ls output/cv_per_season_v3.csv 2>&1 | head -1  # "No such file"
ls output/pairwise_v4.csv 2>&1 | head -1       # "No such file"
```

### Step 1.2: Add MM_SKIP_DEFAULT_LOSO gate to enhanced_model_v3.py

Wrap Step 6's call to `leave_one_season_out_cv_weighted` in an
`if not os.environ.get("MM_SKIP_DEFAULT_LOSO"):` guard. Keep
`fm_filled` construction unconditional (Step 8 needs it). Print a
"skipping" message when the env var is set so the log is
unambiguous about which path ran.

Step 6's pairwise rows are dedup'd away (`keep="last"`) by every
downstream consumer; its CV log loss / accuracy values are
console-printed only and not persisted. The skip is safe.

### Step 1.3: Run the clean LOSO end-to-end

```
MM_PAIRWISE_OUT=output/pairwise_v4.csv \
MM_SKIP_DEFAULT_LOSO=1 \
MM_TUNED_PARAMS_V3='{"n_estimators": 424, "max_depth": 4, \
"learning_rate": 0.013940346079873234, \
"subsample": 0.8736932106048627, \
"colsample_bytree": 0.7760609974958406}' \
python -u src/enhanced_model_v3.py > output/regen_clean_log.txt 2>&1
```

`-u` forces unbuffered stdout so the log file shows progress in
real time (full-buffering on file redirect can hide hours of
progress otherwise). Tuned params come from the leaky run's
`output/v4_tuned_params.json` -- documented confound; see spec.

Expected runtime: ~3 hours (Step 8 LOSO over 22 seasons; Steps
1-5 data load + feature matrix; bracket sim).

### Step 1.3: Verify outputs are the right shape

```
wc -l output/pairwise_v4.csv      # expect 48466 (header + 48465 data)
wc -l output/cv_per_season_v3.csv # expect 23 (header + 22 seasons)
```

If pairwise count diverges materially from 48,465, the LOSO loop
either skipped or doubled a season -- investigate before
proceeding.

### Completion criteria for Phase 1

- Pipeline exits 0.
- Both CSVs exist with the expected row counts.
- `output/regen_clean_log.txt` is captured.

## Phase 2: Diff, document, commit

### Step 2.1: Compute the leak shift

Inline Python, not a new module (one-shot diff, no reuse value):

```python
import pandas as pd

leaky = pd.read_csv("output/cv_per_season_v3_leaky_snapshot.csv")
clean = pd.read_csv("output/cv_per_season_v3.csv")

merged = leaky.merge(
    clean, on="season", suffixes=("_leaky", "_clean"), how="outer", indicator=True
)
assert (merged["_merge"] == "both").all(), "season mismatch leaky vs clean"
merged = merged.drop(columns="_merge")

merged["delta_ll"]  = merged["log_loss_clean"] - merged["log_loss_leaky"]
merged["delta_acc"] = merged["accuracy_clean"] - merged["accuracy_leaky"]

cols = ["season", "log_loss_leaky", "log_loss_clean", "delta_ll",
        "accuracy_leaky", "accuracy_clean", "delta_acc", "n_games_clean"]
print(merged[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

print()
print(f"22-season mean LL  leaky: {merged['log_loss_leaky'].mean():.4f}")
print(f"22-season mean LL  clean: {merged['log_loss_clean'].mean():.4f}")
print(f"22-season mean LL  delta: {merged['delta_ll'].mean():+.4f}")
print(f"22-season mean acc leaky: {merged['accuracy_leaky'].mean():.4f}")
print(f"22-season mean acc clean: {merged['accuracy_clean'].mean():.4f}")
print(f"22-season mean acc delta: {merged['delta_acc'].mean():+.4f}")
```

Save the printed output (not the script -- this is one-shot
diagnostic, no module). The numbers go into the findings note.

### Step 2.2: Write the findings note

`docs/notes/2026-05-04-v4-clean-loso-regen.md` follows the shape
in the spec's "Comparison output" section:

1. Aggregate shift (mean LL/acc leaky vs clean, delta).
2. Per-season table (markdown).
3. Largest shifts (top 3 |delta_LL|) with a one-line gut check
   on whether the direction matches the leak hypothesis.
4. Anchor verdict (pass-as-expected / pass-and-flag /
   surprising-pass per spec criteria).
5. Downstream impact list (recovery step 5 candidates).

Keep it under ~80 lines. Match the prose style of
`docs/notes/2026-05-04-massey-kenpom-leak-audit.md` and
`docs/notes/2026-05-04-bt-bracket-points.md`.

### Step 2.3: Update TODO.md

Edits:

- Move recovery step 3 ("Regenerate `output/pairwise_v4.csv` via
  clean LOSO") out of the active recovery list and into the
  "Done" section with the actual numbers.
- Annotate the recovery-roadmap header's leaky-baseline citations
  (mean LL=0.4369, per-season acc ~80.4%) with "(pre-fix)" so a
  future reader understands which numbers are now historical.
- Update step 4's wording: instead of "Re-run audit against the
  regenerated `pairwise_v4.csv`", note that the regen has shipped
  and step 4 is the immediate next PR.

### Step 2.4: Verification checkpoint (CLAUDE.md "FORCED VERIFICATION")

```
pytest -v 2>&1 | tail -20
```

All tests must pass. Specifically expect green from
`tests/test_vegas_leak_filter.py` and `tests/test_kp_leak_guard.py`
(no code change here; those tests just need to keep passing).

State the test outcome and the clean-LOSO mean LL/acc numbers in
the final commit message and PR body.

### Step 2.5: Commit and PR

Single commit on the feature branch:

- Files changed:
  - `docs/superpowers/specs/2026-05-04-v4-clean-loso-regen-design.md` (new)
  - `docs/superpowers/plans/2026-05-04-v4-clean-loso-regen.md` (new)
  - `docs/notes/2026-05-04-v4-clean-loso-regen.md` (new)
  - `TODO.md` (edit)
- Commit message:
  ```
  feat(v4-clean-loso-regen): regen pairwise_v4 + cv_per_season under
  clean Vegas pipeline; document leak shift

  Recovery step 3 of 5. PR 19 closed the Vegas-feature leak; PR 20
  cleared Massey + KenPom of the same class. This PR runs the
  pipeline end-to-end and records the LOSO shift.

  22-season mean LL  leaky -> clean: <X.XXXX> -> <X.XXXX> (<+X.XXXX>)
  22-season mean acc leaky -> clean: <XX.X%>  -> <XX.X%>  (<+X.X%>)

  Verdict: <pass-as-expected | pass-and-flag | surprising-pass>.
  ```
- Push, open PR via gh.

### Completion criteria for Phase 2

- Findings note exists with all 5 sections filled.
- TODO.md edits applied; recovery step 3 in Done.
- `pytest -v` passes; output captured in PR description.
- PR opened and URL printed back.

## Risks and mitigations

- **Risk:** The script writes more files than the spec lists
  (e.g. `bracket_2026_real_*.csv`, `bracket.html`,
  `pairwise_probs.json`). These are gitignored, but local state
  changes. **Mitigation:** None needed; main repo's copies of
  those files are unaffected because each worktree has its own
  working tree.
- **Risk:** Optuna study cache mismatch between main and worktree
  could trigger a fresh tuning run, wasting compute and producing
  slightly different hyperparameters than the leaky baseline. The
  leaky baseline's hyperparameters are themselves a confound -- if
  the clean run uses the same cached study, we measure pure leak
  impact; if it re-tunes, we measure leak + tuning-noise.
  **Mitigation:** Note in findings if the log shows fresh tuning;
  if so, the verdict still stands directionally but the absolute
  delta has a small confound. The recovery roadmap accepts this.
- **Risk:** The clean LL is essentially identical to leaky
  ("surprising-pass"). **Mitigation:** Per spec, this is a flag
  and we investigate before merging.
