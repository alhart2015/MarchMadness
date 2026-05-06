# Colley Full LOSO Backtest -- Design

**Date:** 2026-05-05
**Branch:** feat/colley-full-loso-backtest
**Recovery context:** TODO.md "CONTAMINATION DISCOVERED 2026-05-04" -> step 5 sub-priority "Colley full LOSO backtest -- NOW THE IMMEDIATE NEXT PR"
**Predecessors:**
- Original Colley spec (clauses 1+2): `docs/superpowers/specs/2026-05-03-colley-matrix-feature-design.md`
- Original Colley findings (clause 2 FAIL on leaky): `docs/notes/2026-05-03-colley.md`
- Wire-in revert: commit `3b4c374`
- Clean clause-2 PASS (this is the trigger): `docs/notes/2026-05-05-colley-massey-clean-rerun.md` (PR 27)
- Clean v4 baseline: `docs/notes/2026-05-04-v4-clean-loso-regen.md` (PR 21, mean LL 0.5588, mean acc 70.7%)
- v9-C revert (production stage-2 is now v8): `docs/notes/2026-05-04-v9c-clean-rerun.md` (PR 24, clean v8 = 2069 brkt pts)
- Procedure runbook: `docs/data_recovery.md`

## Motivation

PR 27's clean-baseline clause-2 result for Colley flipped from leaky FAIL
(+0.0053 LL on 3-season subset) to clean PASS (-0.0100 LL). All three
subset seasons help under clean v4. Per the original Colley spec's
if-pass branch, clause-2 PASS triggers the full 22-season LOSO + bracket-
points backtest. This PR executes that branch.

The structural argument is unchanged: Colley produces an
opponent-adjusted W/L rating (no margin information). The PR 27 finding
is that v4's clean 67-feature stack does NOT extract opponent-adjusted
W/L-only signal as redundantly as it extracted opponent-adjusted *margin*
signal (where Massey-decay-14d still FAILed clause 2 at +0.0018 LL).
This PR tests whether that clean-baseline marginal information survives
the 22-season test set and converts to bracket points.

## Scope

**In scope.**

- Re-wire `colley_rating` into `compute_all_features` (un-revert commit
  `3b4c374`'s 12-line removal).
- Regenerate `output/pairwise_v4_with_colley.csv` via clean LOSO
  (`enhanced_model_v3.py` with `MM_SKIP_DEFAULT_LOSO=1` and
  `MM_TUNED_PARAMS_V3=<PR 21 tuned params>`). ~3 hours compute.
- Compute 22-season LL + accuracy from the regen pairwise; compare
  against canonical clean v4 baseline (LL 0.5588, acc 70.7%).
- Retrain v8 stage-2 against the new pairwise -> generate
  `output/pairwise_v8_with_colley.csv`. ~3-5 minutes.
- Score 22-season bracket points for the Colley-augmented stack:
  v4_with_colley + v8_with_colley vs canonical clean v8 baseline (2069
  brkt pts per PR 24).
- Apply the spec's Reject/Clear/Marginal ladder (see Decision Matrix
  below) and update TODO.md accordingly.
- Findings doc + force-add of canonical artifacts.

**Out of scope.**

- v9-C re-eval. v9-C was reverted to v8 in PR 24 (every cell lost on
  clean baseline; -140 brkt pts at best). v8 is production. No need
  to re-sweep v9-C against the Colley-augmented stage-1 -- if v9-C
  could not extract upset signal from the cleaner stage-1 (PR 24's
  finding), it will not extract upset signal from a slightly-stronger
  stage-1 either. If v8_with_colley clears the gate by a comfortable
  margin (>>+25 brkt pts), v9-C re-eval becomes a follow-up; otherwise
  closed.
- Production live-bracket wiring. `output/pairwise_probs.json` regen
  from 2026 stage-1 is a separate follow-up that already exists in
  TODO.md (recovery step 5: "regenerate v4's 2026 stage-1
  predictions").
- Bayesian-prior tuning, time-decay weighting, Colley-Massey blends.
  Out of scope per the original Colley spec's Out-of-scope list.
- Hyperparameter retuning. Reuse PR 21's tuned XGB parameters via
  `MM_TUNED_PARAMS_V3` (documented confound; expected effect <0.02 LL).
  If verdict is Marginal, hyperparameter retuning is a candidate
  follow-up.

## Decision Matrix

Per the original Colley spec's full-LOSO-backtest section (lines
207-216), evaluated in order against the canonical clean v8 baseline:

| Order | Verdict | Trigger | Action |
|---|---|---|---|
| 1 | **Reject** | `LL_delta >= +0.001` LL OR `brkt_delta <= +10` brkt pts | Revert wire-in. Close Colley as feature. Update TODO. |
| 2 | **Clear** | `LL_delta <= -0.005` LL OR `brkt_delta >= +25` brkt pts | Keep wire-in. Promote Colley to v4-stack. Plan v4.1 release follow-up. Update TODO. |
| 3 | **Marginal** | otherwise | Document; do not ship. Hyperparameter retune is a candidate follow-up. Update TODO. |

`LL_delta` = clean v4-with-colley mean LOSO LL minus clean v4 baseline
(0.5588). Negative is better (lower LL).
`brkt_delta` = clean v4-with-colley + v8-with-colley 22-season bracket
points minus clean v8 baseline (2069). Positive is better (more pts).

## Baselines (from prior PRs, do not re-derive)

- **Clean v4 LOSO LL (mean of 22 seasons):** 0.5588 (PR 21,
  `docs/notes/2026-05-04-v4-clean-loso-regen.md`).
- **Clean v4 LOSO accuracy:** 70.66% (PR 21, same).
- **Clean v8 22-season bracket pts:** 2069 (PR 24,
  `docs/notes/2026-05-04-v9c-clean-rerun.md`).
- **Per-season LL/acc table:** PR 21 findings doc, table at lines 27-52.

## Wire-in: exact diff

The revert at commit `3b4c374` removed 12 lines across 3 hunks of
`src/enhanced_model.py`. To wire back in, restore each hunk verbatim:

**Hunk 1 (top of `compute_all_features`, after seasons filter):**

```python
# Insert after line 198 ("seasons = [s for s in seasons if s >= 2003]"):

    # -- Colley-matrix ratings (cached) -----------------------------------
    from src.features.colley_matrix import load_colley_ratings
    colley_full = load_colley_ratings(reg)
```

**Hunk 2 (per-season block, between seed features and assembly loop):**

```python
# Insert after line 342 ("seed_map[int(row['TeamID'])] = _parse_seed_number(row['Seed'])"):

        # -- 2j: Colley rating ---------------------------------------------
        season_colley_df = colley_full[colley_full["Season"] == season]
        colley_map = dict(zip(season_colley_df["TeamID"], season_colley_df["colley_rating"]))
```

**Hunk 3 (per-team row assembly, between Massey and conf_strength):**

```python
# Insert after line 381 ("row_data.update(massey_features[tid])"):

            # Colley rating
            if tid in colley_map:
                row_data["colley_rating"] = colley_map[tid]
```

The line numbers above reflect the current `src/enhanced_model.py`
state (post-revert). Confirmed via `git show 3b4c374 -- src/enhanced_model.py`
(diff inverted).

## Output artifacts

**New (force-added per `docs/data_recovery.md` canonical-artifact policy):**

- `output/pairwise_v4_with_colley.csv` -- 22-season LOSO pairwise
  predictions with `colley_rating` in feature_cols. Distinct from
  canonical `output/pairwise_v4.csv` (no colley); both kept side by
  side until verdict is locked, at which point either:
  - Verdict = Clear -> `pairwise_v4_with_colley.csv` becomes the new
    canonical `pairwise_v4.csv` (in a separate follow-up PR; this PR
    leaves both files and waits for explicit cutover).
  - Verdict = Reject/Marginal -> `pairwise_v4_with_colley.csv` is
    retained as audit artifact; canonical stays as-is.
- `output/pairwise_v8_with_colley.csv` -- v8 stage-2 retrained on
  `pairwise_v4_with_colley.csv`.
- `output/colley_full_loso_summary.json` -- per-season LL / acc / brkt
  pts comparison (with vs without Colley) plus aggregate verdict.
- `docs/notes/2026-05-05-colley-full-loso-backtest.md` -- findings.

**New (NOT committed):**

- `data/cache/colley_ratings.parquet` (regenerated by load_colley_ratings;
  gitignored per existing pattern).

**Modified:**

- `src/enhanced_model.py` -- 12-line wire-in (un-revert).
- `TODO.md` -- recovery step 5 sub-priority list updated per verdict.

## Risks

1. **Verdict flips between LL and bracket points (e.g. LL passes but
   brkt fails by single-season noise).** Per spec, ladder is OR-of-fail
   for Reject and OR-of-pass for Clear, evaluated in order, so a single
   strong signal can land Reject or Clear cleanly. The Marginal bucket
   is the catch-all for "both metrics weakly positive but neither
   crosses the bar."
2. **3-hour regen wasted.** Mitigated by re-using PR 21's tuned XGB
   params (no Optuna pass) and `MM_SKIP_DEFAULT_LOSO=1` (skips Step 6
   default-params LOSO -- its rows are dedup'd away). Append-mode
   caveat: `rm -f output/pairwise_v4_with_colley.csv` before run.
3. **Cache invalidation.** `data/cache/colley_ratings.parquet` is
   producer-versioned. PR 27 already populated it under the same
   producer version (v1); the new worktree's cache is fresh and will
   rebuild on first call -- no stale-cache risk.
4. **Append-mode collision with canonical pairwise_v4.csv.** The
   `MM_PAIRWISE_OUT` env var directs the writer to a specific path.
   Setting `MM_PAIRWISE_OUT=output/pairwise_v4_with_colley.csv`
   isolates this regen from the canonical file. Verified by reading
   `src/enhanced_model_v3.py:606-630`.
5. **Reproducibility.** XGB tuned-params re-runs are not byte-identical
   across processes (PR 27 noted ~1e-6 LL noise floor, with one notable
   exception of byte-identical agreement). The 22-season LL deltas of
   interest are >> this noise floor, so the verdict is robust.
6. **Pre-existing v3 NameError post-write.** Per PR 21 findings, fixed
   in `enhanced_model_v3.py` final summary. Verify still fixed in this
   worktree before regen.

## Pre-registered prediction

Pre-registered predictions are documented for falsification clarity:

- **LL_delta:** Most likely in `[-0.005, -0.001]` -- i.e. clears the
  +0.001 reject bar and lands in Marginal or Clear. The clause-2
  delta on 3 holdout seasons was -0.0100 LL; on 22 seasons the
  delta typically dilutes by ~3-5x because most seasons see less
  movement than the worst seasons (PR 21's per-season LL shifts
  ranged +0.025 to +0.190; the leak-driven delta concentrated in
  high-leak seasons). Subset {2019, 2022, 2024} averaged a per-season
  shift of +0.150 LL on the leak fix vs the 22-season mean of +0.122
  LL -- so 22-season Colley delta could be ~0.6-0.7x of the subset's
  -0.0100, i.e. -0.006 to -0.007 LL on the aggregate.
  - If true, lands in Marginal (`-0.005 < delta < -0.001`) or barely
    Clear (`delta <= -0.005`). Realistic prior is ~50/50 between
    Marginal and Clear.
- **brkt_delta:** Most likely in `[-15, +30]` -- bracket points are
  noisier than LL because chalk-flip thresholds amplify small
  probability shifts. The +25 Clear threshold is plausible but not
  the central tendency; +0 to +20 Marginal is the most likely
  bucket.
- **Combined verdict:** Marginal (~50%), Clear (~30%), Reject (~20%).
  The 20% Reject probability accounts for the case where Colley's
  3-season subset PASS does not generalize to the broader 22-season
  test (or where bracket-points noise lands the brkt delta at <= +10
  even with positive LL).

If LL_delta lands at +0 to -0.001 LL (between PR 27's clause-2 -0.0100
and the +0.001 threshold), the most likely explanation is that
PR 27's 3-season subset over-represented Colley-helpful seasons. 2019
and 2022 had the largest individual flips (-0.024 and -0.021 swings on
the leak fix); 2003 (which improved -0.007 LL on the leak fix, the
only season that did) is the canary for "Colley adds noise where v4
was already calibrated."

## TODO.md update plan

The "Colley full LOSO backtest -- NOW THE IMMEDIATE NEXT PR" line
(currently line 183 of TODO.md) becomes one of:

- Verdict = Clear: replace with `[DONE -- PR <pending>] Colley CLEARs
  the bar (LL_delta=<x>, brkt_delta=<y>). Promote to v4-stack;
  cutover follow-up = "Replace canonical pairwise_v4.csv with
  pairwise_v4_with_colley.csv via separate PR." Findings:
  docs/notes/2026-05-05-colley-full-loso-backtest.md.`
- Verdict = Marginal: replace with `[DONE -- PR <pending>] Colley
  MARGINAL on full LOSO (LL_delta=<x>, brkt_delta=<y>). Wire-in
  retained on this branch as audit artifact; not promoted. Candidate
  follow-up: hyperparameter retune. Findings: <findings doc>.`
- Verdict = Reject: replace with `[DONE -- PR <pending>] Colley
  REJECTed on full LOSO (LL_delta=<x>, brkt_delta=<y>). Wire-in
  reverted; closes Colley as v4-stack feature. Findings: <findings
  doc>.`

In all three cases, the parent recovery step 5 marginal-rejections
list closing summary stays accurate (4 closed, 1 advanced via this PR).

## Files of record

Created on this branch:
- `docs/superpowers/specs/2026-05-05-colley-full-loso-backtest-design.md` (this)
- `docs/superpowers/plans/2026-05-05-colley-full-loso-backtest.md`
- `output/pairwise_v4_with_colley.csv` (force-added)
- `output/pairwise_v8_with_colley.csv` (force-added)
- `output/colley_full_loso_summary.json` (force-added)
- `docs/notes/2026-05-05-colley-full-loso-backtest.md`

Modified:
- `src/enhanced_model.py` -- 12-line wire-in
- `TODO.md` -- recovery step 5 update

Untouched:
- `src/features/colley_matrix.py` (the solver)
- `src/diagnose_colley.py` (cheap-gate runner; not used in this PR)
- Canonical `output/pairwise_v4.csv`, `output/pairwise_v8.csv` (kept
  intact; cutover is a separate follow-up)
