# v9-C Production Swap (Stage-2 Corrector Path) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `src/predict_2026_v9c.py` (mirror of `src/predict_2026_stage2.py` for v9-C) so analysis scripts pick up v9-C-corrected 2026 predictions via the canonical `output/pairwise_probs.json`.

**Architecture:** Single new entry-point script that trains v9-C on all 22 LOSO seasons (W_UPSET=1.25, W_MISS=0.0, feature_set="v9c"), applies it to v4's 2026 raw predictions with apply-time round resolution via `build_pair_round_lookup(2026, slots, seeds)`, writes a versioned snapshot, and overwrites the canonical `output/pairwise_probs.json`. Live bracket pipeline (`generate_bracket_real.py`) is untouched -- it's pure-v4-MC today and stays that way.

**Tech Stack:** Python 3, pandas, numpy, xgboost, pytest. Reuses existing `src/train_upset_model.py` ingredients; no new dependencies.

**Spec:** `docs/superpowers/specs/2026-05-01-v9c-production-swap-design.md`

**Branch:** feat/v9c-production-swap (already created; spec committed at 713b5a0)

---

## File Structure

| File | Change | Responsibility |
|---|---|---|
| `src/predict_2026_v9c.py` | Create | Train v9-C on all LOSO data, apply to 2026 v4 JSON, write canonical + versioned outputs. |
| `tests/test_predict_2026_v9c.py` | Create | One smoke test on tiny synthetic 2026 fixture. |
| `output/pairwise_probs.json` | Modify (already tracked) | Overwritten by the new script with v9-C-corrected 2026 predictions. |
| `output/pairwise_probs_v9c_2026.json` | Create (run output, local-only) | Versioned v9-C 2026 snapshot; not committed (mirroring v8 convention). |
| `output/predict_2026_v9c_run.log` | Create (run output, local-only) | Driver log; not committed. |
| `TODO.md` | Modify | Move "Production swap to v9-C" from active queue #1 to Done. |

---

## Task 1: Smoke test (TDD)

**Files:**
- Create: `tests/test_predict_2026_v9c.py`

- [ ] **Step 1: Write the smoke test**

Create `tests/test_predict_2026_v9c.py` with the following content:

```python
"""Smoke test for src/predict_2026_v9c.py.

Builds tiny synthetic Kaggle-shaped inputs for one historical season
and 2026, monkeypatches input file paths, then runs main() and
asserts the output JSON files exist and have plausible content.
"""
import json
import sys
from pathlib import Path

import pandas as pd
import pytest


def _write_synthetic_inputs(tmp_path: Path) -> dict:
    """Create the input files predict_2026_v9c.main() reads.

    One historical season (2024) with two played games gives the
    trainer a per-game training row to fit on. 2026 has 4 seeded
    teams forming the same bracket-walk shape.
    """
    data_dir = tmp_path / "data" / "raw" / "march-machine-learning-2026"
    data_dir.mkdir(parents=True)
    out_dir = tmp_path / "output"
    out_dir.mkdir()

    # Pairwise v4: one historical season + 2026 (the 2026 rows are
    # not actually consumed by main() -- v4 2026 lives in JSON --
    # but the trainer needs LOSO data, which load_per_game_data_with_upset
    # reads from this CSV.)
    pd.DataFrame({
        "season":  [2024, 2024, 2024],
        "team_a":  [1, 1, 2],
        "team_b":  [2, 3, 3],
        "p_a_wins": [0.7, 0.6, 0.55],
    }).to_csv(out_dir / "pairwise_v4.csv", index=False)

    # Seeds: 2024 + 2026, four teams each.
    pd.DataFrame({
        "Season": [2024, 2024, 2024, 2024, 2026, 2026, 2026, 2026],
        "Seed":   ["W01", "W08", "W09", "W16",
                   "W01", "W08", "W09", "W16"],
        "TeamID": [1, 2, 3, 4, 1, 2, 3, 4],
    }).to_csv(data_dir / "MNCAATourneySeeds.csv", index=False)

    # Slots: same bracket shape for both years (R1 + R2).
    pd.DataFrame({
        "Season": [2024]*3 + [2026]*3,
        "Slot":   ["R1W1", "R1W8", "R2W1",  "R1W1", "R1W8", "R2W1"],
        "StrongSeed": ["W01", "W08", "R1W1", "W01", "W08", "R1W1"],
        "WeakSeed":   ["W16", "W09", "R1W8", "W16", "W09", "R1W8"],
    }).to_csv(data_dir / "MNCAATourneySlots.csv", index=False)

    # Compact results: two played games in 2024 so trainer has rows.
    pd.DataFrame({
        "Season": [2024, 2024],
        "DayNum": [136, 138],
        "WTeamID": [1, 1],
        "WScore": [70, 75],
        "LTeamID": [2, 3],
        "LScore": [60, 65],
    }).to_csv(data_dir / "MNCAATourneyCompactResults.csv", index=False)

    # v4 2026 raw predictions: 6 pair-pairs across the 4 seeded teams.
    v4_2026 = {
        "1_2": 0.62,
        "1_3": 0.58,
        "1_4": 0.78,
        "2_3": 0.51,
        "2_4": 0.66,
        "3_4": 0.60,
    }
    with open(out_dir / "pairwise_probs_v4.json", "w") as f:
        json.dump(v4_2026, f)

    return {
        "data_dir": data_dir,
        "out_dir": out_dir,
        "pairwise_v4_csv": str(out_dir / "pairwise_v4.csv"),
        "results_csv": str(data_dir / "MNCAATourneyCompactResults.csv"),
        "seeds_csv": str(data_dir / "MNCAATourneySeeds.csv"),
        "slots_csv": str(data_dir / "MNCAATourneySlots.csv"),
        "v4_json": str(out_dir / "pairwise_probs_v4.json"),
        "canonical_json": str(out_dir / "pairwise_probs.json"),
        "v9c_versioned_json": str(out_dir / "pairwise_probs_v9c_2026.json"),
    }


def test_predict_2026_v9c_smoke(tmp_path, monkeypatch):
    """End-to-end smoke: synthetic inputs -> main() -> two JSON outputs
    written with the expected schema and value range.
    """
    paths = _write_synthetic_inputs(tmp_path)

    # Run main() with cwd = tmp_path so the relative paths in the
    # script resolve against our synthetic files.
    monkeypatch.chdir(tmp_path)

    # The script imports DATA from train_upset_model, which is set at
    # import time to "data/raw/march-machine-learning-2026". Patch it
    # to point at our synthetic data dir before main() runs.
    import src.train_upset_model as tum
    monkeypatch.setattr(tum, "DATA", paths["data_dir"])
    import src.predict_2026_v9c as p2026
    monkeypatch.setattr(p2026, "DATA", paths["data_dir"])

    p2026.main()

    # Both output files exist.
    assert Path(paths["canonical_json"]).exists()
    assert Path(paths["v9c_versioned_json"]).exists()

    # Versioned and canonical have the same content (script writes the
    # same dict twice).
    canon = json.loads(Path(paths["canonical_json"]).read_text())
    versioned = json.loads(Path(paths["v9c_versioned_json"]).read_text())
    assert canon == versioned

    # All 6 input pair-pair keys are present in the output.
    expected_keys = {"1_2", "1_3", "1_4", "2_3", "2_4", "3_4"}
    assert set(canon.keys()) == expected_keys

    # All probabilities are in (0, 1) (not 0 or 1 exactly; xgboost
    # almost never produces hard 0/1 with regularization).
    for k, p in canon.items():
        assert 0.0 < p < 1.0, f"pair {k} probability out of range: {p}"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_predict_2026_v9c.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'src.predict_2026_v9c'` (the import at line `import src.predict_2026_v9c as p2026` fails because the script doesn't exist yet).

- [ ] **Step 3: Commit the failing test**

```bash
git add tests/test_predict_2026_v9c.py
git commit -m "test(v9c): smoke test for predict_2026_v9c

Synthetic 2024 + 2026 fixture; asserts main() writes both canonical
and versioned JSON with all input pair keys and probabilities in
(0, 1). Failing as expected (script doesn't exist yet)."
```

---

## Task 2: Implement `src/predict_2026_v9c.py`

**Files:**
- Create: `src/predict_2026_v9c.py`

- [ ] **Step 1: Write the script**

Create `src/predict_2026_v9c.py` with the following content:

```python
"""Apply v9-C stage-2 corrector to v4's 2026 pairwise predictions.

Trains v9-C on ALL 22 LOSO seasons of v4 out-of-fold data with the
PR 9 winning weights (W_UPSET=1.25, W_MISS=0.0, feature_set='v9c'),
then applies it to v4's 2026 raw-pair JSON predictions. Writes
output/pairwise_probs_v9c_2026.json (versioned snapshot) and
overwrites output/pairwise_probs.json (canonical for analysis
scripts).

Production-swap path established in PR 9. See:
- docs/notes/2026-05-01-v9c-feature-stripped.md (LOSO findings)
- docs/superpowers/specs/2026-05-01-v9c-production-swap-design.md
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.train_upset_model import (
    DATA,
    build_pair_round_lookup,
    compute_sample_weights,
    fit_upset_model,
    load_per_game_data_with_upset,
    parse_seed,
    upset_features,
)

# Production weights from PR 9 winning cell.
PROD_W_UPSET = 1.25
PROD_W_MISS = 0.0
PROD_FEATURE_SET = "v9c"


def main():
    print("Loading v9-C training data (all 22 LOSO seasons)...")
    per_game = load_per_game_data_with_upset(
        "output/pairwise_v4.csv",
        str(DATA / "MNCAATourneyCompactResults.csv"),
        str(DATA / "MNCAATourneySeeds.csv"),
    )
    print(f"  {len(per_game):,} rows ({per_game.season.nunique()} seasons)")

    print(f"Training v9-C (W_UPSET={PROD_W_UPSET}, W_MISS={PROD_W_MISS}, "
          f"feature_set={PROD_FEATURE_SET}) on all seasons...")
    X = upset_features(per_game, feature_set=PROD_FEATURE_SET)
    y = per_game["label"].values
    w = compute_sample_weights(per_game, w_upset=PROD_W_UPSET,
                               w_miss=PROD_W_MISS)
    model = fit_upset_model(X, y, w)

    print("Loading v4 2026 pairwise predictions...")
    with open("output/pairwise_probs_v4.json") as f:
        v4_probs = json.load(f)
    print(f"  {len(v4_probs):,} pair-pairs")

    print("Loading 2026 seeds + slots...")
    seeds_df = pd.read_csv(DATA / "MNCAATourneySeeds.csv")
    seeds_df["seed_int"] = seeds_df["Seed"].apply(parse_seed)
    seeds_2026 = {int(r.TeamID): r.seed_int for _, r in seeds_df.iterrows()
                  if r.Season == 2026 and r.seed_int is not None}
    slots_df = pd.read_csv(DATA / "MNCAATourneySlots.csv")
    pair_round_2026 = build_pair_round_lookup(2026, slots_df, seeds_df)
    print(f"  {len(seeds_2026)} seeds; "
          f"{len(pair_round_2026)} pair-round entries")

    print("Applying v9-C to each 2026 pair...")
    adjusted = {}
    skipped = 0
    feat_rows = []
    keys_with_seeds = []
    for key, p_stage1 in v4_probs.items():
        a_str, b_str = key.split("_")
        a, b = int(a_str), int(b_str)
        seed_a = seeds_2026.get(a)
        seed_b = seeds_2026.get(b)
        if seed_a is None or seed_b is None:
            adjusted[key] = float(p_stage1)  # passthrough
            skipped += 1
            continue
        a_canon, b_canon = (a, b) if a < b else (b, a)
        rnd = float(pair_round_2026.get((a_canon, b_canon), 0))
        feat_rows.append({
            "p_stage1": float(p_stage1),
            "seed_a": float(seed_a),
            "seed_b": float(seed_b),
            "abs_seed_diff": float(abs(seed_a - seed_b)),
            "round": rnd,
        })
        keys_with_seeds.append(key)

    if feat_rows:
        X_apply = upset_features(pd.DataFrame(feat_rows),
                                 feature_set=PROD_FEATURE_SET)
        p_v9c = model.predict_proba(X_apply)[:, 1]
        for key, p_new in zip(keys_with_seeds, p_v9c):
            adjusted[key] = round(float(p_new), 4)

    print(f"  Adjusted: {len(keys_with_seeds):,}; "
          f"passthrough (no seeds): {skipped}")

    out_path = "output/pairwise_probs_v9c_2026.json"
    with open(out_path, "w") as f:
        json.dump(adjusted, f)
    print(f"Saved: {out_path}")

    # Overwrite the canonical pairwise_probs.json so analysis scripts
    # pick up v9-C corrections.
    with open("output/pairwise_probs.json", "w") as f:
        json.dump(adjusted, f)
    print("Overwrote: output/pairwise_probs.json")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the smoke test to verify it passes**

Run: `pytest tests/test_predict_2026_v9c.py -v`

Expected: PASS. The smoke test should complete in a few seconds (training xgboost on 4 rows is near-instant).

If it fails on `numpy` import being unused: remove `import numpy as np`. The script uses pandas only at the import-line level; numpy is implicitly used through xgboost / sklearn but not directly. (This is a small style fix; either keep or drop the import.)

If it fails on `monkeypatch.setattr(p2026, "DATA", paths["data_dir"])` because `predict_2026_v9c` re-imports `DATA` at module load: this is the standard from-import binding and should work since the test patches both the source (`tum.DATA`) and the consumer (`p2026.DATA`) -- if it doesn't, fall back to patching only `tum.DATA` and remove the second monkeypatch line.

- [ ] **Step 3: Commit the implementation**

```bash
git add src/predict_2026_v9c.py
git commit -m "feat(v9c): predict_2026_v9c.py -- production stage-2 corrector

Mirrors predict_2026_stage2.py for v9-C: trains on all 22 LOSO
seasons with W_UPSET=1.25, W_MISS=0.0, feature_set='v9c'; applies
to v4's 2026 raw JSON predictions with apply-time round resolution
via build_pair_round_lookup(2026, ...); writes versioned snapshot
and overwrites the canonical output/pairwise_probs.json.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: Run the production swap

**Files:**
- Modify (overwrite): `output/pairwise_probs.json` (already tracked in main as a canonical artifact)
- Create (local-only): `output/pairwise_probs_v9c_2026.json`
- Create (local-only): `output/predict_2026_v9c_run.log`

- [ ] **Step 1: Verify input prerequisites are present**

Run:
```bash
ls -la output/pairwise_v4.csv output/pairwise_probs_v4.json data/raw/march-machine-learning-2026/MNCAATourneySlots.csv data/raw/march-machine-learning-2026/MNCAATourneySeeds.csv data/raw/march-machine-learning-2026/MNCAATourneyCompactResults.csv
```

Expected: all 5 files exist with non-zero size. If any are missing, halt and investigate -- the script will crash at the load step.

- [ ] **Step 2: Snapshot the current canonical file before overwrite**

Run: `cp output/pairwise_probs.json output/pairwise_probs_pre_v9c_swap.json`

This gives a local-only audit trail in case we need to compare before-and-after spot checks.

- [ ] **Step 3: Run the production swap**

Run: `python src/predict_2026_v9c.py 2>&1 | tee output/predict_2026_v9c_run.log`

Expected runtime: <30 seconds (training xgboost on ~3000 LOSO rows + applying to ~2000 2026 pairs).

Expected log content:
- "Loading v9-C training data" line + row count
- "Training v9-C (W_UPSET=1.25, W_MISS=0.0, feature_set=v9c)" line
- "Loading v4 2026 pairwise predictions" line + pair count
- "Loading 2026 seeds + slots" line + seed and pair-round counts
- "Applying v9-C to each 2026 pair" line + adjusted/passthrough counts
- "Saved: output/pairwise_probs_v9c_2026.json"
- "Overwrote: output/pairwise_probs.json"

- [ ] **Step 4: Verify output files exist and have expected schema**

Run:
```bash
python -c "
import json
canon = json.load(open('output/pairwise_probs.json'))
versioned = json.load(open('output/pairwise_probs_v9c_2026.json'))
v4 = json.load(open('output/pairwise_probs_v4.json'))
print(f'canon keys: {len(canon)}; versioned keys: {len(versioned)}; v4 keys: {len(v4)}')
print(f'canon == versioned (same content): {canon == versioned}')
print(f'canon keys match v4 keys: {set(canon) == set(v4)}')
sample = list(canon.items())[:3]
print('sample (canon):', sample)
"
```

Expected:
- `canon keys` and `versioned keys` both equal `v4 keys` count (~2016).
- `canon == versioned` is `True`.
- `canon keys match v4 keys` is `True`.
- Sample probabilities are floats in (0, 1).

- [ ] **Step 5: Spot-check vs v8 (the model is genuinely different)**

Run:
```bash
python -c "
import json
v8 = json.load(open('output/pairwise_probs_v8_2026.json'))
v9c = json.load(open('output/pairwise_probs_v9c_2026.json'))
common = set(v8) & set(v9c)
diffs = [(k, v9c[k] - v8[k]) for k in common]
big_diffs = [d for d in diffs if abs(d[1]) > 0.01]
huge_diffs = [d for d in diffs if abs(d[1]) > 0.5]
print(f'common keys: {len(common)}')
print(f'diffs > 0.01: {len(big_diffs)} (success criterion: >= 50)')
print(f'diffs > 0.5:  {len(huge_diffs)} (success criterion: 0)')
print('top 5 absolute diffs:')
for k, d in sorted(diffs, key=lambda x: abs(x[1]), reverse=True)[:5]:
    print(f'  {k}: v8={v8[k]:.4f} v9c={v9c[k]:.4f} delta={d:+.4f}')
"
```

Expected per the spec's success criteria:
- `diffs > 0.01` count is at least 50 (model is genuinely different from v8).
- `diffs > 0.5` count is exactly 0 (no catastrophic flips).

If either criterion fails, **halt and investigate** -- do not commit. A `diffs > 0.5` count > 0 likely indicates a feature-column mismatch or a slots-lookup bug. A `diffs > 0.01` count < 50 likely indicates the script silently fell through to passthrough on most pairs.

- [ ] **Step 6: Commit the canonical artifact change**

```bash
git add output/pairwise_probs.json
git commit -m "feat(v9c): regenerate output/pairwise_probs.json with v9-C corrections

Ran src/predict_2026_v9c.py to overwrite the canonical analysis
artifact with v9-C stage-2 predictions for 2026. Versioned snapshot
at output/pairwise_probs_v9c_2026.json (local-only, not tracked).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

`output/pairwise_probs_v9c_2026.json`, `output/predict_2026_v9c_run.log`, and `output/pairwise_probs_pre_v9c_swap.json` stay local-only -- no git add for them (they're under the gitignored `output/` directory).

---

## Task 4: Full pytest gate

**Files:**
- None modified. Verification step.

- [ ] **Step 1: Run the full test suite**

Run: `pytest -v 2>&1 | tail -30`

Expected: All 138 tests PASS (137 from PR 9 + 1 new smoke test).

- [ ] **Step 2: If any failure, halt and debug**

Do not proceed to Task 5 until pytest is green. The most likely failure is the new smoke test if Task 2's `monkeypatch.setattr(p2026, "DATA", ...)` line doesn't take (see Task 2 Step 2 fallback note).

---

## Task 5: Update TODO.md

**Files:**
- Modify: `TODO.md`

- [ ] **Step 1a: Remove the v9-C swap item from active queue and renumber**

Use the Edit tool with this exact `old_string` -> `new_string` (replace_all=false). This single edit removes the now-completed swap item AND renumbers the remaining queue items 2/3/4 to 1/2/3 in one pass:

```
old_string:
1. **Production swap to v9-C.** Findings clear the spec's swap-in
   bar (+43 vs v8, F4/E8 lens distinctly better). Separate follow-up
   commit on `feat/v9-b-followup`: flip defaults to v9-C (W_UPSET=1.25,
   W_MISS=0.0, 5-feature set), regenerate `output/pairwise_v9c.csv`,
   point bracket pipeline at it, regenerate the 2026 chalk bracket and
   spot-check the picks that flip. Findings:
   `docs/notes/2026-05-01-v9c-feature-stripped.md`.
2. **Ensemble of model classes.** XGBoost + logistic regression +
   small neural net averaged (or stacked). The TODO already had this
   under Tier C. The hypothesis: different model classes capture
   partially-uncorrelated error patterns. Risk: if all three see the
   same features and reach the same ~80% R64 / ~50-60% deep-round
   ceiling, the errors are highly correlated and ensembling won't help
   much. Position #2 after the v9-C swap lands.
3. **External rankings (538, KenPom-public, BPI as features).** Note:
   we already have BPI, Sagarin, KenPom (POM), Bart Torvik (TRK), RPI
   via Massey ordinals (config.yaml lines 30-36). Truly external would
   be 538's tournament forecast or Vegas prop-bet predictions, which
   need data sourcing outside the Kaggle archive.
4. **Roster-level returning-experience.** Player-level data is not in
   the Kaggle Mania archive; would need an external roster CSV per
   season. Different signal from coach experience.

new_string:
1. **Ensemble of model classes.** XGBoost + logistic regression +
   small neural net averaged (or stacked). The TODO already had this
   under Tier C. The hypothesis: different model classes capture
   partially-uncorrelated error patterns. Risk: if all three see the
   same features and reach the same ~80% R64 / ~50-60% deep-round
   ceiling, the errors are highly correlated and ensembling won't help
   much. Promoted to position #1 after the v9-C swap landed.
2. **External rankings (538, KenPom-public, BPI as features).** Note:
   we already have BPI, Sagarin, KenPom (POM), Bart Torvik (TRK), RPI
   via Massey ordinals (config.yaml lines 30-36). Truly external would
   be 538's tournament forecast or Vegas prop-bet predictions, which
   need data sourcing outside the Kaggle archive.
3. **Roster-level returning-experience.** Player-level data is not in
   the Kaggle Mania archive; would need an external roster CSV per
   season. Different signal from coach experience.
```

- [ ] **Step 1b: Append the new Done entry**

Use the Edit tool with this exact `old_string` -> `new_string` (replace_all=false). The anchor is the trailing text of the most recent Done entry (v9-C feature-stripped) followed by the two blank lines and the "Architecture Rethink" header. Insert the new entry between the trailing text and the blank lines:

```
old_string:
  Findings: docs/notes/2026-05-01-v9c-feature-stripped.md.



## Architecture Rethink (Tier C)

new_string:
  Findings: docs/notes/2026-05-01-v9c-feature-stripped.md.
- **v9-C production swap (2026-05-01).** Added
  `src/predict_2026_v9c.py` (mirror of `src/predict_2026_stage2.py`
  for v9-C). Trains on all 22 LOSO seasons with W_UPSET=1.25,
  W_MISS=0.0, feature_set='v9c'; applies to v4's 2026 JSON via
  apply-time round lookup; writes versioned snapshot
  `output/pairwise_probs_v9c_2026.json` and overwrites the
  canonical `output/pairwise_probs.json` (the file analysis scripts
  consume). Live bracket pipeline (`generate_bracket_real.py`)
  unchanged -- it's pure-v4-MC today and v8 was never wired in
  there either; live-bracket stage-2 integration is a separate
  follow-up. Spec:
  `docs/superpowers/specs/2026-05-01-v9c-production-swap-design.md`.



## Architecture Rethink (Tier C)
```

The two blank lines between the last Done entry and the Architecture Rethink header are preserved as-is.

- [ ] **Step 2: Verify ASCII compliance**

Run: `python -c "open('TODO.md', encoding='utf-8').read().encode('ascii')" && echo "ASCII OK"`

Expected: prints `ASCII OK`.

- [ ] **Step 3: Commit**

```bash
git add TODO.md
git commit -m "docs: TODO update -- v9-C production swap landed

Moves 'Production swap to v9-C' from active queue #1 to Done.
Ensemble of model classes promotes back to #1.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 6: Open PR

**Files:**
- None modified.

- [ ] **Step 1: Push the branch and open a PR**

```bash
git push -u origin feat/v9c-production-swap
gh pr create --title "v9-C production swap (stage-2 corrector path)" --body "$(cat <<'EOF'
## Summary
- Adds `src/predict_2026_v9c.py` -- mirror of `src/predict_2026_stage2.py` for v9-C
- Trains v9-C on all 22 LOSO seasons (W_UPSET=1.25, W_MISS=0.0, feature_set='v9c'), applies to v4's 2026 raw JSON predictions, writes a versioned snapshot, and overwrites the canonical `output/pairwise_probs.json` so analysis scripts (`postmortem_full.py`, `bracket_scorecard.py`, `alternate_bracket.py`, `iowa_impact.py`, `blend_sweep.py`) pick up v9-C corrections
- Live bracket pipeline (`generate_bracket_real.py`) intentionally unchanged -- it's pure-v4-MC today and v8 was never wired in there either

## Test plan
- [ ] `pytest tests/test_predict_2026_v9c.py -v` passes (smoke test on synthetic 2024 + 2026 fixture)
- [ ] `pytest -v` full suite passes (138 tests)
- [ ] `python src/predict_2026_v9c.py` runs end-to-end without error
- [ ] Output JSON has the same key set as `output/pairwise_probs_v4.json`
- [ ] >= 50 pairs differ from v8 by > 0.01 (model is genuinely different)
- [ ] 0 pairs flip by > 0.5 (no catastrophic miscalibration)

## Out of scope
- Live bracket integration of stage-2 (separate design -- bracket HTML is pure-v4-MC today and stays that way)
- Modifying `predict_2026_stage2.py` (v8 path stays available for comparison)
- Changing trainer defaults in `train_upset_model.py` (LOSO trainer's defaults preserve PR 8/9 reproducibility)

Spec: docs/superpowers/specs/2026-05-01-v9c-production-swap-design.md
Plan: docs/superpowers/plans/2026-05-01-v9c-production-swap.md
PR 9 (LOSO findings): https://github.com/alhart2015/MarchMadness/pull/9

Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

Return the PR URL when done.
