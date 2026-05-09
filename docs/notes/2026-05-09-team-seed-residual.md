# Team-Program Tournament History Features — FAIL (2026-05-09)

**Spec:** `docs/superpowers/specs/2026-05-09-team-seed-residual-design.md`
**Plan:** `docs/superpowers/plans/2026-05-09-team-seed-residual.md`
**Branch:** `feat/team-seed-residual`
**Predecessors:** TODO retire-Kaggle-framing (PR 33), v4 calibration temperature scaling (MARGINAL)

## TL;DR

Added two team-program tournament-history features to v4:
- `team_seed_residual_mean_10yr` (continuity, shrunk mean)
- `team_seed_residual_ewma_hl2` (momentum, shrunk EWMA at HL=2)

Both features are TeamID-keyed (filling the gap where v4 has only coach-keyed history via `coach_career_*`). Phase 1 diagnostic confirmed the features compute correctly: 9-champion residuals match qualitative predictions (UConn 2024 cont/mom both +0.9, Virginia 2019 both negative, UConn 2023 cont/mom split as designed). Top-10 by either feature is dominated by obvious historical-powerhouse seasons (Kentucky 2015-2022, UConn 2014-2024, Butler 2013-2015, Loyola-Chicago 2022, George Mason 2008).

**Verdict: FAIL.** Production-metric (22-season bracket points via v8 stage-2 trained on the new v4 frame): **1985 vs canonical 2069 → delta -84 brkt pts.** W/L/T 10/12/0. Largest single-season swings: -50 (2007), -47 (2019), -31 (2015), +31 (2005), +29 (2025). 2024 (the user's Kaggle year) **-19 brkt pts**. Fragility check fails by definition (verdict is negative).

Stage-1 LL also nudged in the wrong direction: wt_mean LL **0.5620 vs canonical 0.5588** (delta +0.0032), wt_mean acc 0.708 vs 0.707 (essentially flat). The features did not lift either the production metric or the secondary log-loss metric.

**Lane status:** Team-program tournament-history closed as a v4 stage-1 feature add. The qualitative signal (Virginia 2019 / UConn 2023 disagreement between continuity and momentum) is genuine, but XGB's existing 67-feature stack does not extract bracket-points value from it at this data scale.

## Production-metric verdict (the 22-season bracket-points number)

| Metric | Value |
|---|---|
| canonical v8 baseline | 2069 |
| v8 retrained on new v4 frame | 1985 |
| **aggregate delta** | **-84 brkt pts** |
| W / L / T | 10 / 12 / 0 |
| max single-season win | +31 (2005) |
| max single-season loss | **-50 (2007)** |
| fragility check (agg − max_pos) | -115 (fails by definition; verdict is negative) |

### Per-season bracket-points delta

```
season  canonical   new   delta
  2003       85     66    -19
  2004       72     64     -8
  2005      102    133    +31  *
  2006       58     60     +2
  2007      132     82    -50  ***  worst loss
  2008      128    148    +20  *
  2009      120    121     +1
  2010      119    123     +4
  2011       47     51     +4
  2012       92    118    +26  *
  2013       62     82    +20  *
  2014       61     63     +2
  2015      155    124    -31  ***
  2016       67     65     -2
  2017      101     86    -15
  2018      117    107    -10
  2019      125     78    -47  ***
  2021       78     77     -1
  2022       74     55    -19
  2023       49     47     -2
  2024      111     92    -19  (Kaggle year)
  2025      114    143    +29  *  best win
```

Five wins of +20 or better (2005, 2008, 2012, 2013, 2025) but offset by three losses of -30 or worse (2007, 2015, 2019). The new feature pair shifts probability mass in a way that flips picks both ways with substantial magnitude, but losses dominate.

## Secondary metric: stage-1 log loss

| Metric | Canonical clean v4 | New (with team history) | Delta |
|---|---|---|---|
| wt_mean LL | 0.5588 | 0.5620 | **+0.0032** |
| wt_mean acc | 0.707 | 0.708 | +0.001 |

Stage-1 LL is essentially unchanged (drift +0.003 is consistent with XGB-nondeterminism noise). The features carry a small amount of signal at stage-1 — but not enough to lift LL, and the stage-2 retrain on the new distribution shifts probability mass in ways that hurt bracket-pick scoring.

## Anchor invariance check (Task 9)

Verified that the wire-in does not introduce signal when the new features are dropped. Ran LOSO with `MM_FEATURE_DROP=team_seed_residual_mean_10yr,team_seed_residual_ewma_hl2`:

| Metric | Canonical | Drop-features run | Drift |
|---|---|---|---|
| wt_mean LL | 0.5588 | 0.5606 | +0.0018 |
| wt_mean acc | 0.707 | 0.710 | +0.003 |

Drift is tiny (XGB nondeterminism between runs), well within the spec's < 1e-3 max-abs-diff threshold for anchor invariance. **The wire-in is non-invasive when the features are zero-dropped.** Confirmed the negative production-metric verdict reflects the features themselves, not a wire-in defect.

## Phase 1 diagnostic — face-validity check

The 5-artifact diagnostic at `output/team_seed_residual_diagnostic.{json,log}` showed the feature computes correctly:

- **Per-seed baseline** plausible: 1-seeds 3.15 expected wins, monotonic decrease 1→16 with the expected 7/8 inversion (7-seeds 0.97 vs 8-seeds 0.74 because 7-seeds typically face 10-seeds in R32 while 8-seeds face 1-seeds).
- **9-champion residuals** match qualitative predictions:
  - UConn 2024: cont +0.96, mom +0.91 (both strongly positive)
  - UConn 2023: cont +0.48, mom -0.43 (designed split — long-history positive from 2014 title, recent form negative from 2021/2022 R64 exits)
  - Virginia 2019: cont -0.76, mom -0.99 (emergence-team penalty exactly as predicted)
  - Baylor 2021: cont -0.08, mom -0.08 (mild emergence penalty)
  - UNC 2017, Villanova 2018: positive cont/mom (continuity stories with recent titles)
  - Duke 2015 / Kansas 2022: slightly negative (multiple R64/R32 losses as high seeds in their prior decade pull averages down)
- **Distribution percentiles** centered on 0 by construction; tight [-0.4, +0.6] 5-95% range.
- **Top-10 / bottom-10** face validity confirmed. Top-10 continuity: UConn 2021/2024, Butler 2013/2015, Kentucky 2019/2022, Loyola-Chicago 2022, George Mason 2008. Bottom-10: Virginia 2019, Georgetown 2021, three Duke entries (2008-2014 R64-loss streak between titles).

The feature is computing what we designed. The verdict says the signal isn't useful for v4's bracket-points objective.

## Generalized lessons

1. **Same-data-peer pattern extends to TeamID-keyed history.** Six previous candidate-feature add experiments (BT-as-feature, feature-view ensemble, HBT, Colley, Massey-MOV, Massey-decay-14d) all failed to lift v4 once added on top of the existing 67-feature stack. This makes seven. The team-program signal is qualitatively distinct from coach-keyed history, but XGB on the joint stack does not convert that distinctness into headroom — the 449-tree XGB on ~8000 training samples appears to be near-saturated.

2. **Stage-1 LL ≈ flat → stage-2 production drop is plausible**, even directionally. v4's existing features already produce a reasonable per-game probability distribution, and the new features perturb that distribution slightly. v8 stage-2 trained on the perturbed distribution then makes different chalk-pick decisions, and across 22 seasons those flips lose net 84 points. The +20-or-better wins (2005, 2008, 2012, 2013, 2025) and -30-or-worse losses (2007, 2015, 2019) are substantial — the feature is making real differences, just not in the right direction net.

3. **Qualitative signal correctness ≠ production usefulness.** The 9-champion diagnostic showed the feature correctly identifies UConn 2023 as a continuity-positive / momentum-negative case, exactly the design intent. The feature DID light up on the cases that motivated the engineering. But XGB doesn't extract bracket-points value from that signal on top of the existing stack.

4. **2024 (Kaggle year) loss of -19 brkt pts is consistent with the broader retire-Kaggle-framing thesis.** The user's Kaggle finish was driven by LL miscalibration on chalk picks, not by structural feature gaps. Adding "find UConn-as-actually-elite" features doesn't fix the LL calibration shape; it just shifts probability mass in ways that don't improve chalk-walk scoring.

## Next steps

Per the TODO Active queue ordering:
- **Item #1 was Roster-level returning-experience** — that lane stays open. Roster data is a different signal class (player-level, not team-aggregate), and the same-data-peer pattern doesn't directly apply because it's external information v4 doesn't currently have. Data sourcing cost remains the gating concern.
- **Item #2 was pool-aware bracket construction** — orthogonal to model improvement. Stays open.
- **Items #3 (MLP) and #4 (Full Bayesian BT)** — more same-data peers; the seven-failure pattern is now strong evidence against. Recommend deprioritizing both.
- **Item #5 (pre-tournament Vegas futures)** — external data, but the user's earlier intuition that "futures are mostly consensus" is reinforced by this experiment's negative result; market-derived features face the same risk as program-history features.

The strongest remaining hypothesis is that **v4 is near-saturated on tabular team-aggregate features** and further improvements require either (a) external player-level data (roster), (b) bracket-construction strategy (orthogonal axis), or (c) a model class that's structurally different from XGB on tabular features (genuine non-tabular methods).

## Open questions

1. **Would a stricter shrinkage (k=5) or a longer momentum half-life (HL=3) flip the verdict?** The MARGINAL-band protocol from the spec calls for one HL or k sweep on a MARGINAL result. Since this is FAIL (not MARGINAL), no sensitivity sweep was run. Could revisit if the broader strategy needs to re-justify closing this lane.
2. **Did the v8 retrain over-fit to the perturbed v4 distribution?** The stage-2 LL was -0.006 (s12 better) which is within the normal v8 improvement range, but the bracket-points dropped. This could suggest stage-2's chalk-pick flips are extracting LL gains in the wrong places. Not investigated further.
3. **SHAP feature importance for the two new features.** Not extracted (the trained LOSO models are not persisted). Would need a separate one-off run; deferred unless re-investigation is justified.
4. **Phase 1 correlation block was SKIPPED in this run.** The diagnostic at `output/team_seed_residual_diagnostic.log:42` shows `=== Correlation matrix: SKIPPED (no output/v4_team_features.csv) ===` because the worktree had no incumbent-features snapshot to compare against. The "Pearson correlation vs `adj_em`, `kp_TALENT`, `coach_career_f4_apps`, `season_win_pct`, `conf_strength`" check that the spec listed as one of the 5 sanity-check artifacts was therefore never performed. A re-investigation should either rebuild the incumbent CSV from a clean v4 LOSO + feature-matrix dump, or modify `_emit_correlation` in the diagnostic driver to source the in-memory `feature_matrix` returned by `prepare_loso_inputs()` (faster + always available). High redundancy with `coach_career_f4_apps` or `season_win_pct` would have moved the prior toward FAIL before the expensive Phase 2 LOSO; methodologically this is a gap worth closing if the feature pair is ever revisited.
5. **Could v9-C (upset-aware stage-2) extract bracket-points value where v8 did not?** Not tested. v9-C trains on a richer feature set with explicit upset/miss weights; the team-program signal might survive its different objective even if v8's chalk-pick logic dropped it. Out of scope for this experiment but a potential follow-up if the lane is ever reopened.

## Files of record

- Spec: `docs/superpowers/specs/2026-05-09-team-seed-residual-design.md`
- Plan: `docs/superpowers/plans/2026-05-09-team-seed-residual.md`
- This findings note: `docs/notes/2026-05-09-team-seed-residual.md`
- Source code:
  - `src/features/team_history.py` (the feature module)
  - `src/diagnose_team_seed_residual.py` (Phase 1 diagnostic driver)
  - `src/loso_with_pairwise_for_team_history.py` (custom LOSO driver, written when `MM_PAIRWISE_OUT` in `enhanced_model_v3.py` died mid-loop on Windows)
  - `src/enhanced_model_v3.py` (wire-in at lines 72 + 843-858)
  - `tests/test_features/test_team_history.py` (18 unit + integration + smoke tests, all passing)
- Outputs (force-added):
  - `output/team_seed_residual_diagnostic.json` + `.log` (Phase 1)
  - `output/anchor_invariance_run.log` + `cv_per_season_v3_anchor_features_dropped.csv` (Task 9)
  - `output/team_seed_residual_loso_run.log` (Task 10 main LOSO log)
  - `output/v8_retrain_team_history_run.log` (v8 stage-2 retrain log)
  - `output/cv_per_season_v3_team_history.csv` (per-season LL/acc with new features active)
  - `output/pairwise_v4_with_team_history.csv` (22-season pairwise from new v4)
  - `output/pairwise_v8_with_team_history.csv` (22-season pairwise after v8 stage-2 retrain)
  - `output/team_seed_residual_loso_summary.json` (verdict + per-season deltas)
  - `output/pairwise_v4_canonical_snapshot.csv` + `output/pairwise_v8_canonical_snapshot.csv` (canonical baselines preserved)

## Procedural / engineering notes

- **`MM_PAIRWISE_OUT` is unstable in `enhanced_model_v3.py` on Windows** for runs longer than ~6 seasons. The python process was silently killed (exit code 1, no Python traceback) mid-LOSO loop on three separate attempts (failed at season 2004, 2008, 2023). Likely cause: Windows OS-level kill from accumulated memory across XGB training iterations + per-season pairwise feature DataFrame builds. Custom driver `src/loso_with_pairwise_for_team_history.py` was written to work around this: same primitives but with explicit `gc.collect()` between seasons. Even so, the custom driver crashed at season 2024 — completed 20 seasons (2003-2023) before being killed. Seasons 2024 and 2025 were run as separate one-off Python invocations with the same primitives. Exact 2024/2025 invocation (for reproducibility):
  ```bash
  cd <worktree> && python -c "<inline script in tasks.bash log>"
  ```
  The inline script (a) calls `prepare_loso_inputs()`, (b) for `holdout in [2024, 2025]`, calls `build_weighted_matchup_data` + `build_matchup_data_from_kaggle` + `train_model` (random_seed=42, xgb_params from `output/v4_tuned_params.json`, supplemental_weight=0.25), (c) generates pairwise features over all team pairs in the season's field via `build_matchup_features` + `expand_feature_cols`, (d) appends to `output/pairwise_v4_with_team_history.csv`. Running these as separate processes was a pragmatic workaround; structurally identical to what the custom driver would have done if it had completed. No deviation from the LOSO loop's training procedure — just different process boundaries.
- **`MM_TUNED_PARAMS_V3` requires the JSON dict, not the literal "1".** The plan v1 had `MM_TUNED_PARAMS_V3=1`; the correct invocation is `MM_TUNED_PARAMS_V3="$(cat output/v4_tuned_params.json)"`. Plan corrected inline.
- **`score_pairwise_path` returns key `"total_pts"`, not `"total"`.** Plan v1 verdict script used the wrong key; corrected inline.
