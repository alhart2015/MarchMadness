# v4 clean LOSO regeneration -- Design

**Date:** 2026-05-04
**Branch:** feat/v4-clean-loso-regen
**Status:** spec
**Recovery roadmap step:** 3 of 5 (TODO.md "CONTAMINATION DISCOVERED 2026-05-04")

## Problem

PRs 19 and 20 closed two leakage suspects:

- PR 19 (`feat/v4-vegas-leak-fix`, merged 2026-05-04): drops
  tournament games from the per-team Vegas-feature aggregates by
  filtering `daynum >= 134` before `compute_vegas_features` and
  `_build_vegas_team_records_with_dates`.
- PR 20 (`feat/audit-massey-kenpom-leak`, merged 2026-05-04): audited
  Massey + KenPom data sources; no leak found, defensive guard
  `kp_leak_guard` added in `src/enhanced_model.py` and
  `src/kaggle_submission.py`.

The current `output/pairwise_v4.csv` and `output/cv_per_season_v3.csv`
on disk were generated under the leaky pipeline (mean per-season
LL=0.4369, acc=79.7% over 22 LOSO seasons). Every downstream
consumer that reads `pairwise_v4.csv` (audit framework, ensemble
diagnostics, v9-C stage-2 corrector, BT bracket-points sweep, etc.)
is anchored on the leaky values. We need a clean baseline before
re-running any of those.

## Scope

**In scope (this PR):**

1. Run `enhanced_model_v3.py` end-to-end with the fixed feature
   pipeline (post-PR-19) to regenerate:
   - `output/pairwise_v4.csv` (per-(season, team_a, team_b) LOSO
     test-fold predictions; emitted via `MM_PAIRWISE_OUT` env var)
   - `output/cv_per_season_v3.csv` (per-season LL / Brier / acc /
     AUC / n_games over the 22 LOSO test seasons)
2. Compare per-season LL + acc to the pre-fix snapshot
   (`output/cv_per_season_v3_leaky_snapshot.csv`, captured before
   regen, gitignored).
3. Write findings note
   `docs/notes/2026-05-04-v4-clean-loso-regen.md` with the per-
   season delta table, 22-season aggregate shift, and a verdict on
   whether the shift matches the spec's anchor expectations
   (LL 0.45-0.47, acc 73-77%).
4. Update TODO.md: move recovery step 3 to "Done" with the actual
   numbers; the leaky-baseline `cv_per_season_v3.csv` numbers cited
   in the recovery roadmap stop being load-bearing.

**Out of scope (separate PRs, tracked in TODO.md):**

- Step 4: re-run `src/audit_v4_gap_vegas.py` against the clean
  pairwise CSV and update / retract the "no weak spots" verdict.
- Step 5: re-run swap-decided / swap-candidate evaluations (v9-C
  production swap, v8 vs v9-C, plain BT bracket points,
  marginal rejections within ~0.05 LL of v4).

**One small in-scope code change:** add an `MM_SKIP_DEFAULT_LOSO`
env-var gate to `src/enhanced_model_v3.py` that skips Step 6
(default-params LOSO CV). Step 6's pairwise rows are dedup'd away
by every downstream consumer of `pairwise_v4.csv` (which all use
`drop_duplicates(..., keep="last")`); its only output is a console
log of "default params would have scored ..." which is useful in
the original training-run context but not in a regen. With the
gate, a clean regen runs ~half as long. Combined with reusing
the leaky run's tuned hyperparameters via the existing
`MM_TUNED_PARAMS_V3` env var, the regen drops from ~5-6 hours to
~3 hours.

**Reuse-leaky-tuned-params confound:** the leaky-run hyperparameters
(`n_estimators=424, max_depth=4, lr=0.0139, subsample=0.874,
colsample=0.776`, Optuna seed=42) were chosen on the leaky
training distribution. Re-tuning on the clean distribution might
pick slightly different values. We accept this confound: this PR
measures "leak removed, same architecture and hyperparameters" --
the spec's pre-registered shift direction (LL up, acc down) does
not depend on hyperparameter reoptimization, and a separate retune
PR can be done if the clean numbers warrant it.

## Anchors and pre-registered expectations

PR 19's spec (`2026-05-04-v4-vegas-leak-fix-design.md`) pre-
registered the directional shift after the fix lands and a clean
regen runs:

| metric                       | leaky (pre-fix)   | expected (clean)            |
|------------------------------|-------------------|-----------------------------|
| v4 LOSO log loss (22-season) | 0.4369            | higher; perhaps 0.45-0.47   |
| v4 LOSO accuracy (per-season)| ~79.7% mean       | lower; perhaps 73-77%       |

Direction of shift is anchored by the leak's own structure: the
Vegas aggregates inflated tournament-success-correlated features
for teams that won tournament games and deflated them for teams
that lost early. Removing the leak removes that signal from the
test row, so test-time LL goes up (worse) and accuracy goes down.

**Verdict criteria for this PR:**

- **Pass-as-expected**: clean mean LL is higher than 0.4369 and
  within roughly [0.43, 0.50] (i.e., the leak was real but not
  catastrophic; v4 is still a reasonable model).
- **Pass-and-flag**: clean LL > 0.50 (i.e., v4 was much weaker than
  reported; major retractions required for downstream verdicts).
- **Surprising-pass**: clean LL <= 0.4369 (i.e., the filtering had
  ~zero effect on aggregate metrics). Would prompt investigation
  -- the leak was measurable in raw feature values (UConn 2024
  vegas_avg_margin shifted +1.98), so a no-op aggregate result
  would imply the model never used the leaky channels in any way
  that affected its calibration on average. Possible but unlikely.

The 22-season mean is the headline; per-season variance is
expected (the leak signal is correlated with tournament success,
so seasons with surprising tournaments shift more than chalk
seasons).

## Procedure

```
# In the worktree:
rm -f output/pairwise_v4.csv  # MM_PAIRWISE_OUT appends; start clean.
MM_PAIRWISE_OUT=output/pairwise_v4.csv \
MM_SKIP_DEFAULT_LOSO=1 \
MM_TUNED_PARAMS_V3="$(cat output/v4_tuned_params.json | tr -d '\n')" \
python -u src/enhanced_model_v3.py > output/regen_clean_log.txt 2>&1
```

`output/cv_per_season_v3.csv` is overwritten by the script
(`enhanced_model_v3.py:1019`). The leaky-baseline snapshot at
`output/cv_per_season_v3_leaky_snapshot.csv` (already captured
pre-regen in this branch) is the comparison reference.

`enhanced_model_v3.py`'s end-to-end run also writes
`output/bracket_2026_real.csv`, `output/pairwise_probs.json`,
`output/bracket_data.json`, `output/bracket.html`, etc. -- full
2026 bracket pipeline. Those side-outputs are *also* now produced
under the clean pipeline; v9-C re-application against the clean
`output/pairwise_probs.json` is recovery step 5 and out of scope
here.

## Comparison output (findings note shape)

`docs/notes/2026-05-04-v4-clean-loso-regen.md` will contain:

1. **Aggregate shift** -- 22-season mean LL / acc, leaky vs clean,
   delta.
2. **Per-season table** -- season, LL_leaky, LL_clean, delta_LL,
   acc_leaky, acc_clean, delta_acc, n.
3. **Largest shifts** -- top 3 seasons by |delta_LL| with a sentence
   about whether the direction matches the leak's hypothesized
   structure (champ/runner-up seasons should worsen more under the
   fix; first-round-upset seasons should improve or stay flat).
4. **Anchor verdict** -- pass-as-expected / pass-and-flag /
   surprising-pass per the criteria above.
5. **Downstream impact list** -- which findings notes / verdicts
   need re-evaluation in step 5.

## Success criteria

1. **Pipeline runs to completion** end-to-end without error;
   `output/pairwise_v4.csv` exists with 48,465 rows (the canonical
   pair count under the skip-gate -- one tuned-params pass; without
   the gate the file would have ~96,930 rows and downstream
   consumers would dedup with `keep="last"` to the same 48,465) and
   `output/cv_per_season_v3.csv` has 22 rows.
2. **`pytest -v`** passes (existing tests; no new tests added).
   Specifically `tests/test_vegas_leak_filter.py` (PR 19's
   integration tests) and `tests/test_kp_leak_guard.py` (PR 20's
   guard tests) are already in main and must stay green.
3. **Findings note written** with anchor verdict explicitly
   stated.
4. **TODO.md updated**: recovery step 3 moved to Done with the
   final numbers; leaky-baseline references in the recovery
   roadmap header retired or annotated as "pre-fix".

## What this does NOT establish

- v4's actual position vs Vegas / 538 / etc. -- the audit framework
  needs to be re-run against the clean CSV (step 4). The clean LL
  number does NOT by itself answer "does v4 still beat Vegas?" --
  Vegas's own LL comparison reference (~0.5447 in the PR 18 audit)
  is unchanged; the comparison is the next PR's job.
- Whether v9-C still adds points over v8 on the clean baseline.
  That's step 5.
- A new ground truth for any of the rejected experiments
  (HBT, plain BT, BT-as-feature, feature-view-ensemble,
  Massey-matrix, Colley). Those rejections are documented in
  TODO.md "Tried and rejected" with notes on which need re-eval
  vs which don't.

## Files of record

```
docs/superpowers/specs/2026-05-04-v4-clean-loso-regen-design.md  -- this
docs/superpowers/plans/2026-05-04-v4-clean-loso-regen.md         -- plan
docs/notes/2026-05-04-v4-clean-loso-regen.md                     -- findings
TODO.md                                                          -- recovery step 3 -> Done

src/enhanced_model_v3.py                          -- adds MM_SKIP_DEFAULT_LOSO env-var gate; fixes orphaned-locals NameError in final summary

output/cv_per_season_v3.csv                       -- regenerated (gitignored)
output/cv_per_season_v3_leaky_snapshot.csv        -- pre-regen reference (gitignored)
output/pairwise_v4.csv                            -- regenerated (gitignored)
output/regen_clean_log.txt                        -- pipeline log (gitignored)
```
