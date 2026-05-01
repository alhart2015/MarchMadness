# v9-C Production Swap (Stage-2 Corrector Path) -- Design

**Date:** 2026-05-01
**Branch:** feat/v9c-production-swap
**Predecessors:**
- v9-C spec: `docs/superpowers/specs/2026-05-01-v9c-feature-stripped-design.md`
- v9-C plan: `docs/superpowers/plans/2026-05-01-v9c-feature-stripped.md`
- v9-C findings: `docs/notes/2026-05-01-v9c-feature-stripped.md`
- v8 production reference: `src/predict_2026_stage2.py`

## Motivation

PR 9 established v9-C as a clear winner over v8 in the LOSO backtest
(+43 brkt pts, F4/E8 lens distinctly better) and recommended swapping
v9-C into production. This is the swap.

The relevant production path is the **stage-2 corrector for 2026**:
`src/predict_2026_stage2.py` trains v8 stage-2 on all 22 LOSO seasons
of v4 OOF predictions, applies it to v4's 2026 raw-pair JSON
predictions, and overwrites `output/pairwise_probs.json` so analysis
scripts (`postmortem_full.py`, `bracket_scorecard.py`,
`alternate_bracket.py`, `iowa_impact.py`, `blend_sweep.py`) consume
v8-corrected probabilities.

The live bracket pipeline (`src/generate_bracket_real.py`) does NOT
use stage-2 today and is out of scope here -- bracket HTML stays
pure-v4 + Monte Carlo, same as it currently is. Wiring stage-2 into
the live bracket is a separate design (see follow-ups).

## Scope

**In scope.**

- New file `src/predict_2026_v9c.py`. Mirrors
  `src/predict_2026_stage2.py` in shape: trains the stage-2 corrector
  on all 22 LOSO seasons, applies it to v4's 2026 raw predictions,
  writes a versioned snapshot, then overwrites the canonical
  `output/pairwise_probs.json`.
- The new script trains v9-C (5 features, W_UPSET=1.25, W_MISS=0.0)
  using the existing `train_upset_model.py` ingredients
  (`load_per_game_data_with_upset`, `upset_features`,
  `compute_sample_weights`, `fit_upset_model`).
- Apply-time round resolution via `build_pair_round_lookup(2026,
  slots, seeds)` -- same helper PR 8 added; verified 2026 has 67
  slot rows + 68 seeded teams in the Kaggle data.
- Versioned output: `output/pairwise_probs_v9c_2026.json` (parallel
  to `predict_2026_stage2.py`'s `output/pairwise_probs_v8_2026.json`).
- Canonical overwrite: `output/pairwise_probs.json` -- the file
  consumed by analysis scripts.
- One smoke test ensuring the new script can be imported and its
  apply function builds a 5-column matrix on a tiny synthetic input.
- Commit the regenerated `output/pairwise_probs.json` (already
  tracked in main as a canonical artifact -- this is a modification,
  not a force-add). The versioned snapshot
  `output/pairwise_probs_v9c_2026.json` stays local-only, mirroring
  the existing convention (v8's `output/pairwise_probs_v8_2026.json`
  is not tracked either).

**Out of scope.**

- Wiring stage-2 into the live bracket pipeline
  (`generate_bracket_real.py`). Today's bracket HTML is pure-v4 + MC
  and v8 was never wired in either; making the live bracket reflect
  v9-C is a separate behavior change worth its own design.
- Modifying `predict_2026_stage2.py`. v8 path stays available for
  comparison/audit. If we want to retire v8 later that's a separate
  cleanup.
- Changing trainer defaults in `src/train_upset_model.py`. Defaults
  are calibrated for the LOSO backtest (W_UPSET=3.0, feature_set
  "v9b") and changing them would silently break the reproducibility
  of PR 8 / PR 9 numbers if anyone re-runs the trainer with no args.
  The new production script sets v9-C weights and feature set
  explicitly at call time, mirroring how `predict_2026_stage2.py`
  doesn't depend on v8's `W_UPSET`/`W_MISS` constants.
- Regenerating bracket-pipeline outputs (`output/bracket.html`,
  `output/bracket_data.json`, `output/bracket_compact.json`,
  `output/bracket_2026.csv`). These are pure-v4-MC and are not
  affected by the stage-2 swap.
- Regenerating analysis script outputs (`output/postmortem*` etc.).
  Those scripts read `output/pairwise_probs.json` at call time and
  pick up the new v9-C corrections automatically the next time the
  user runs them.

## Approach

### New file: `src/predict_2026_v9c.py`

Structure mirrors `src/predict_2026_stage2.py` (88 LOC) closely so
the diff between them is the v9-C-specific bits only:

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
PROD_W_MISS  = 0.0
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
    print(f"  {len(seeds_2026)} seeds; {len(pair_round_2026)} pair-round entries")

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

Key differences vs `predict_2026_stage2.py`:

- Imports from `src.train_upset_model` instead of `src.train_stage2`.
- Production weight constants (`PROD_W_UPSET`, `PROD_W_MISS`,
  `PROD_FEATURE_SET`) at module top so they're explicit and grep-able.
- Loads `MNCAATourneySlots.csv` and builds `pair_round_2026` lookup
  -- v8 doesn't need round; v9-C does.
- Per-pair feature row includes `round` (from the lookup) instead of
  the v8 4-feature shape.
- Apply step calls `upset_features(df, feature_set='v9c')` to assemble
  the 5-column matrix in the same column order the v9-C model was
  trained on.
- Out-path filenames change `v8` -> `v9c`.

### Tests

Add `tests/test_predict_2026_v9c.py` with one test:

- **`test_predict_2026_v9c_smoke`** -- monkeypatch the file paths to
  point at tiny synthetic CSVs (1 season, 4 teams, 6 pair-pairs in
  v4 JSON; 2026 entries in seeds + slots). Run `main()`. Assert the
  canonical `output/pairwise_probs.json` and the versioned
  `output/pairwise_probs_v9c_2026.json` are written, contain the
  expected pair keys, and have probability values in (0, 1).

The existing test convention in this repo is to write integration
tests against tiny synthetic Kaggle-shaped inputs (see
`tests/test_sweep_v9_weights.py:_write_minimal_inputs`). Reuse that
shape -- inline a similar fixture in the new test file.

`predict_2026_stage2.py` itself has no tests, so this is a slight
positive deviation from the v8 baseline. Worth doing because the
v9-C apply path has more moving parts (slots lookup, 5-column matrix)
and a smoke test catches the most common breakage (file missing,
schema mismatch, slots not resolved for 2026).

### Run

After the script is in and tested:

```bash
python src/predict_2026_v9c.py 2>&1 | tee output/predict_2026_v9c_run.log
```

Verify:
- `output/pairwise_probs_v9c_2026.json` written.
- `output/pairwise_probs.json` overwritten and now matches the v9c
  file byte-for-byte (or differs only in JSON key order).
- Spot-check a few high-stakes pairs (e.g., 1-vs-16, 5-vs-12) to
  confirm probabilities are reasonable (not all 0, not all 1, not
  systematically inverted).

### Commit canonical artifact

`output/` is gitignored, but `output/pairwise_probs.json` is already
tracked in main (force-added previously as a canonical artifact).
Modifying it in this branch is a normal `git add`, not a force-add.
The versioned snapshot stays local-only:

```bash
git add src/predict_2026_v9c.py tests/test_predict_2026_v9c.py
git add output/pairwise_probs.json   # already tracked
git commit -m "feat(v9c): production stage-2 swap -- predict_2026_v9c.py"
```

`output/pairwise_probs_v9c_2026.json` and
`output/predict_2026_v9c_run.log` are local-only; not committed.

## Success criteria

- `pytest -v tests/test_predict_2026_v9c.py` passes.
- `pytest -v` full suite stays green (existing 137 tests + 1 new).
- `python src/predict_2026_v9c.py` runs end-to-end without error.
- `output/pairwise_probs_v9c_2026.json` contains the expected number
  of keys (matches `output/pairwise_probs_v4.json`'s key count).
- `output/pairwise_probs.json` after the run has the same content as
  `output/pairwise_probs_v9c_2026.json` (Python's `json.dump`
  preserves dict insertion order, and the script writes the same
  `adjusted` dict twice, so the two files should be byte-identical
  in practice).
- Pair probabilities differ from `output/pairwise_probs_v8_2026.json`
  in at least 50 of the ~2000 pairs by > 0.01 (the model is
  genuinely different from v8, not a mislabeled re-run of v8). No
  pair flips by more than 0.5 (the model is not catastrophically
  miscalibrated -- v9-C's anchor reproduced v8 exactly in the LOSO
  backtest, so apply-time predictions should be in the same general
  range).
- `predict_2026_stage2.py` is unchanged.
- `train_upset_model.py` is unchanged.

## Risks and mitigations

- **2026 round lookup misses some pairs.** The bracket-walk helper
  resolves pair-meeting rounds for seeded teams. Pairs of teams that
  never meet in the bracket structure get round=0 by fallback. Same
  behavior as the v9-C LOSO sweep, where the per-cell pairwise CSVs
  fell back to 0 for the small subset of unresolvable pairs and the
  numbers still landed at +43 vs v8. Mitigation: the test smoke-checks
  that the round column is non-zero for at least one pair on a 2026
  fixture.
- **2026 v4 predictions stale.** `output/pairwise_probs_v4.json` was
  last written Apr 28 (per `ls -la`). If the v4 model has been
  re-tuned since, the v9-C application is layered on stale stage-1.
  Mitigation: this is the same staleness condition v8's swap
  inherited; if v4 is rerun, both v8 and v9-C scripts should be
  rerun. Documented but not addressed in this work.
- **Force-add of `output/pairwise_probs.json` could surprise.**
  Existing canonical-artifact pattern force-adds specific named
  files (`output/pairwise_v9.csv` was force-added in PR 8). Mitigation:
  use `git add -f` only for the two named files, never for the
  whole `output/` directory.
- **Smoke test exercises real model fit.** XGBoost on a tiny
  synthetic dataset can be flaky; the test asserts shape + value
  range, not specific predictions. Same approach as
  `tests/test_sweep_v9_weights.py`.

## Follow-ups (not in this spec)

- **Live bracket integration of stage-2.** A separate design
  question: "should `generate_bracket_real.py` apply the v9-C
  corrector before Monte Carlo?" Current behavior is pure-v4-MC.
  Wiring stage-2 in would change the live bracket picks (and any
  EV strategy outputs). Worth its own brainstorm + measurement
  (does the bracket get notably different picks? do the F4/E8
  improvements transfer to the live bracket?).
- **Retire `predict_2026_stage2.py`.** Once v9-C is the canonical
  production path and v8 is no longer being audited against, the
  v8 script can be deleted. Deferred until users explicitly stop
  consulting v8.
- **Update LOSO trainer defaults.** Same argument as above: only
  worth doing if the LOSO trainer's `main()` is genuinely the v9-C
  production entry point. Today it's a backtest tool; defaults
  should preserve the canonical exploration values.
