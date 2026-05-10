# GNN Stage-1 Peer Phase 2 LOSO -- FAIL (2026-05-10)

**Spec:** `docs/superpowers/specs/2026-05-09-non-tabular-model-class-scoping-design.md`
**Plan:** `docs/superpowers/plans/2026-05-10-gnn-phase2-loso-after-phase1-retraction.md`
**Branch:** `feat/non-tabular-model-class-scoping`
**Predecessors:** GNN Phase 1 (RS-prediction, 2026-05-09) -- RETRACTED (Massey-baseline leak via `ranking_day=133` + RS-prediction is a structurally biased proxy for tournament prediction). Phase 2 supersedes that result.

## TL;DR

Phase 2 trains a single cross-season GNN on 21 tournament seasons (LOSO over 22 seasons, 2003-2025 ex-2020) and evaluates the held-out season's bracket games. Two encoder variants were tested: a default GraphSAGE encoder (the structurally-faithful Phase 2 baseline) and an edge-attr-aware GINE encoder consuming normalized `[score_diff, site_indicator, days_rest, days_from_start]` (the plan's MARGINAL row's single authorized structural variant). The LL-blend gate against v4 was evaluated on the 22-season pooled pairwise (1449 unique tournament games x 2 orientations = 2898 evaluations).

**Verdict: FAIL.** The SAGE encoder produced the closer-of-the-two result and missed the gate by 0.0011 LL: standalone GNN LL 0.6060 vs v4 LL 0.5579; r_residual 0.5495 (PASS, < 0.60 -- signals are complementary); optimal blend weight w* = 0.80 (PASS, in [0.40, 0.85] -- non-degenerate); **headroom +0.0039 LL** (FAIL, threshold +0.0050). The edge-attr-aware GINE variant was worse: standalone LL 0.6293, r_residual 0.5180 (PASS), w* = 0.95 (FAIL, degenerate), headroom +0.0003 (FAIL, 10x short). Per the plan, only one structural variant was authorized before the final verdict; neither rescued the gate.

This is the eighth same-data-equivalent FAIL in a row (BT-as-feature, feature-view ensemble, HBT, Colley, Massey-MOV, Massey-decay-14d, team-seed-residual, and now GNN-Phase-2). The signal is **regime-dependent rather than uniformly weak**: 9-10 seasons where the GNN adds real per-season blend headroom (best: 2022 +0.0687, 2017 +0.0498, 2003 +0.0482, 2011 +0.0455) versus 8 seasons where the GNN is strictly worse and the optimal blend collapses to all-v4 with zero headroom (2005, 2006, 2007, 2008, 2010, 2012, 2019, 2021). The pooled +0.0039 LL is the net of these, not a flat low signal.

**Lane status:** GNN closed as a v4 LL-blend peer. v8 stage-2 retrain was not run (gate failed, so production-metric scoring would not be meaningful). Phase 1's verdict is RETRACTED in the same commit that introduces this note: the Phase 1 Massey baseline used `ranking_day=133` (Selection Sunday), which means the comparison rankings already incorporated the held-out RS test games -- the Phase 1 GNN was disadvantaged against a leaked baseline on a structurally biased task (RS-prediction is not the same target as tournament prediction).

## Production-metric verdict (LL-blend gate)

The plan defines the production gate at the LL-blend level rather than v8 bracket-points, because the Phase 2 protocol is "promote to v8 retrain only if the LL-blend gate clears +0.005 with non-degenerate w* and r_residual < 0.60." If the LL gate fails the v8 retrain is skipped.

**Primary run -- SAGE encoder:**

| Metric | Value | Gate | Result |
|---|---|---|---|
| v4 standalone LL (pooled 2898 eval) | 0.5579 | -- | -- |
| GNN standalone LL | 0.6060 | -- | (worse than v4) |
| v4 standalone acc | 0.702 | -- | -- |
| GNN standalone acc | 0.668 | -- | -- |
| r_residual (v4 vs GNN logit residual corr) | 0.5495 | < 0.60 | PASS |
| optimal blend weight w* | 0.80 | in [0.40, 0.85] | PASS |
| LL-blend headroom (LL_v4 - LL_blend) | +0.0039 | >= +0.0050 | **FAIL** |

**Secondary run -- edge-attr-aware GINE encoder:**

| Metric | Value | Gate | Result |
|---|---|---|---|
| GNN (edge-attr) standalone LL | 0.6293 | -- | (worse than SAGE) |
| GNN (edge-attr) standalone acc | 0.625 | -- | -- |
| r_residual | 0.5180 | < 0.60 | PASS |
| optimal blend weight w* | 0.95 | in [0.40, 0.85] | **FAIL** |
| LL-blend headroom | +0.0003 | >= +0.0050 | **FAIL** |

Both variants fail the +0.005 LL headroom threshold. SAGE fails only clause 3 (headroom) by 0.0011 LL; the edge-attr variant fails both clause 2 (degenerate w*) and clause 3 (10x short on headroom). v8 retrain not run.

## Per-season detail (SAGE encoder)

Per-season blend numbers from `output/cv_per_season_gnn_phase2_blend.csv`. `n` is per-season tournament games (single-orientation); LLs are pooled across both orientations. `w*` is the per-season optimal blend weight on v4 (so w*=1.00 means the per-season optimum is all-v4, i.e. GNN is strictly worse).

```
season  n  ll_v4   ll_gnn  r_res    w*   opt_ll  headroom
2003   64  0.5776  0.5507  0.396  0.38  0.5293  +0.0482
2004   64  0.5429  0.5658  0.482  0.63  0.5298  +0.0131
2005   64  0.5207  0.6378  0.569  1.00  0.5207  +0.0000
2006   64  0.5775  0.6436  0.701  1.00  0.5775  +0.0000
2007   64  0.4808  0.6325  0.626  1.00  0.4808  +0.0000
2008   64  0.4787  0.6751  0.596  1.00  0.4787  +0.0000
2009   64  0.5090  0.5251  0.508  0.60  0.4969  +0.0121
2010   64  0.5564  0.6805  0.540  1.00  0.5564  +0.0000
2011   67  0.6681  0.6305  0.595  0.30  0.6226  +0.0455
2012   67  0.5368  0.6388  0.604  1.00  0.5368  +0.0000
2013   67  0.6142  0.6067  0.667  0.46  0.5901  +0.0241
2014   67  0.5777  0.5623  0.663  0.39  0.5548  +0.0229
2015   67  0.4781  0.5202  0.643  0.89  0.4775  +0.0006
2016   67  0.5829  0.6264  0.604  0.81  0.5800  +0.0029
2017   67  0.5467  0.5030  0.581  0.25  0.4968  +0.0498
2018   67  0.5827  0.6377  0.648  0.85  0.5799  +0.0028
2019   67  0.5088  0.5977  0.567  0.99  0.5088  +0.0000
2021   66  0.5822  0.6748  0.458  0.99  0.5822  +0.0000
2022   67  0.6431  0.5810  0.563  0.23  0.5744  +0.0687
2023   67  0.6241  0.6667  0.593  0.77  0.6179  +0.0062
2024   67  0.6068  0.6346  0.628  0.70  0.5988  +0.0079
2025   67  0.4687  0.5365  0.417  0.81  0.4639  +0.0048
```

**Aggregate (across 1449 unique tournament games x 2 orientations = 2898 evaluations):**
- v4 standalone: LL 0.5579, acc 0.702
- GNN (SAGE) standalone: LL 0.6060, acc 0.668
- r_residual = 0.5495, optimal_w = 0.80, headroom = +0.0039

**Edge-attr GINE encoder aggregate:**
- GNN (edge-attr) standalone: LL 0.6293, acc 0.625
- r_residual = 0.5180, optimal_w = 0.95, headroom = +0.0003
- Per-season detail in `output/cv_per_season_gnn_phase2_edge_attr_blend.csv`

### Per-season highlights

**Best SAGE per-season headrooms (GNN adds genuine complementary signal):**
- 2022: +0.0687 (GNN beats v4 strongly; w* = 0.23 means the per-season optimum uses 77% GNN)
- 2017: +0.0498 (GNN beats v4 outright; w* = 0.25)
- 2003: +0.0482 (w* = 0.38)
- 2011: +0.0455 (w* = 0.30)

**Worst SAGE per-season headrooms (GNN strictly worse; blend collapses to v4):**
- 8 seasons with w* = 1.00 and zero headroom: 2005, 2006, 2007, 2008, 2010, 2012, 2019, 2021

The GNN signal is regime-dependent: 9-10 seasons of genuine value vs 8 seasons of strict harm. The aggregate +0.0039 is the net, not a uniformly weak signal.

### Disagreement breakdown (SAGE)

When v4 and GNN disagreed on the picked side (391/1449 games = 27% of games):
- v4 was right: 219 (56%)
- GNN was right: 172 (44%)

Consistent with v4's lower standalone LL: where they disagree, v4 wins the slim majority.

## What this means

Per the spec's sequel-ordering matrix, the GNN candidate is now closed against the v4 LL-blend target. Re-rank:

- **Candidate 4 (self-supervised team embeddings via RS margin prediction)** stays promoted as TODO Active queue item #3. Its saturation-break theory (latent style/matchup specificity learned from RS box-score margins) operates at the team-pair level rather than via multi-hop graph message-passing. The GNN's regime-dependent value pattern (real signal in some seasons, strict harm in others) is suggestive that a structurally different representation might still cleanly clear the gate even though graph-topology message-passing did not.
- **Candidate 3 (box-score four-factor distributional model)** stays alive -- different signal class (variance distribution across games, not season aggregates or graph topology).
- **Candidate 2 (sequence model on per-team trajectory)** stays deprioritized -- shares the aggregated-feature risk profile that the GNN result reinforces. KenPom's recent-form scalars and v4's tempo features likely absorb most of the team-trajectory signal.

The Phase 2 result hardens the saturation-on-tabular-features hypothesis. Eight same-data-equivalent FAILs (seven leak-safe priors + GNN Phase 2) is strong evidence that v4 on its current 67-feature stack at this data scale (~8000 training samples) is at or near a saturation ceiling that further same-data peers do not break.

## Methodological notes

1. **Phase 1 retraction.** The Phase 1 finding (RS-prediction sanity check, GNN vs scalar Massey baseline, FAIL by 0.10 LL) is RETRACTED. Two reasons: (a) `evaluate_massey_baseline` used `ranking_day=133` (Selection Sunday), so the Massey ratings the GNN was compared against already incorporated the held-out RS test games -- a data leak in the baseline's favor; (b) RS-prediction is structurally biased as a proxy for tournament prediction, because tournament games are neutral-court, single-elimination, and field-restricted to selection-eligible teams, none of which the per-team RS feature distribution faithfully represents. Phase 1 now carries a retraction header pointing here.

2. **Cross-season parameter sharing.** One GNN was trained on all 21 train-seasons' tournament outcomes per LOSO holdout; encoder and decoder weights are shared globally across seasons. The global team_index spans all seasons, so `nn.Embedding(num_teams, hidden_dim)` has one row per TeamID across the full 2003-2025 range. This is **structurally different from Phase 1's per-season independent training**: Phase 2 has ~2,800 training pairs per holdout vs Phase 1's ~250 per season.

3. **Cross-season embedding-sharing caveat.** Team identity persists across seasons via the global embedding (UConn 2003 and UConn 2024 share an embedding row). Per-season GraphSAGE message-passing on each season's RS graph contextualizes the row to that season's neighborhood, but the base team identity vector is shared across years. This is by design (cross-season transfer) but limits the model's ability to represent same-program-across-years variation (different coaching staffs, roster turnover, era-specific style changes). Worth noting for future variants -- e.g., a (team, era) compound embedding or an explicit season-conditioning input could relax this.

4. **Edge-attr structural variant tested.** Per the plan's MARGINAL row, the user authorized one structural change (not hyperparameter fishing) before declaring the final verdict. Tested: a GINE encoder consuming normalized edge_attr `[score_diff, site_indicator, days_rest, days_from_start]`. Result: standalone LL got worse (0.6293 vs SAGE 0.6060), and the gate failed harder (clauses 2 + 3 vs SAGE's clause 3 only). Edge attributes did not rescue the verdict.

5. **No hyperparameter sweep.** The plan's MARGINAL row also offered `hidden_dim in {32, 128}` and `lr in {5e-4}` variants, but the user explicitly excluded those as post-hoc fishing. Discipline preserved. The verdict stands on the structurally-motivated runs only (SAGE default + one authorized structural variant).

6. **Determinism.** `seed=42` set for torch / numpy / python at the start of every LOSO holdout. Documented in `output/gnn_phase2_loso_summary.json` and the edge-attr summary.

7. **Phase 2's eighth-failure pattern.** The seven prior failures (BT-feature, feature-view ensemble, HBT, Colley, Massey-MOV, Massey-decay-14d, team-seed-residual) were leak-safe and remain valid as same-data-peer null results. Phase 2 makes it eight in a row. Per the spec's sequel-ordering matrix, Candidate 4 (self-supervised team embeddings via RS margin prediction) stays promoted as TODO Active queue item #3.

8. **Wall-clock.** SAGE sweep: 9.6 min wall-clock, 2.9 min total train time. Edge-attr sweep: 21.7 min wall-clock, 11.3 min total train time. CPU only, no GPU. The wall-clock cost is dominated by graph build + per-holdout encoder forward passes, not by training epochs.

## Open questions

1. **Would Candidate 4 (self-supervised team embeddings on RS margin prediction) clear the gate where the GNN did not?** The GNN's regime-dependent value pattern (9-10 winning seasons vs 8 losing seasons) is genuine signal -- it just doesn't aggregate to +0.005 net. A representation learned from a denser objective (per-game margin prediction across all RS games, ~5,000 pairs per season vs ~70 tournament games) might be lower-variance and clear the gate. Worth a Phase 1 sanity check before LOSO commitment.

2. **Per-season heterogeneity: structural feature or noise?** The 8 zero-headroom seasons (2005-2010 cluster of 5, plus 2012, 2019, 2021) raise a question: is there an era-specific structural reason the GNN strictly underperforms there, or is it sampling variance over 64-67 games per season? The 2005-2010 clustering is suggestive (era of changing 1-bid-conference parity, fewer transfers), but not investigated.

3. **Would a (team, era) compound embedding lift the SAGE result?** The cross-season embedding-sharing caveat (point 3 above) is a real architectural choice; relaxing it (e.g., one embedding row per (TeamID, era) bucket where era boundaries align with major NCAA structural shifts) would test whether same-program-across-years variation is part of the residual that's preventing the +0.005 net headroom. Not tested -- would count as a third structural variant beyond the MARGINAL-row authorization.

4. **v9-C (upset-aware stage-2) instead of v8?** The gate is defined against v8's training distribution. v9-C trains on a richer feature set with explicit upset/miss weights; its objective rewards exactly the kind of regime-dependent pick-flipping the GNN provides on its winning seasons (2017 / 2022 are both notable upset-heavy years where the GNN beats v4 outright). Out of scope here, but a plausible follow-up if the lane is reopened.

5. **No per-season fragility check.** The gate metric is pooled, not per-season-max-loss-bounded. Even if the +0.005 threshold were cleared, a -47-style worst-season loss (cf team-seed-residual 2019) would still need to be checked before v8 retrain. Moot here since the gate fails, but worth noting for Candidate 4's eventual Phase 2 design.

## Files of record

- This findings note: `docs/notes/2026-05-10-gnn-phase2-loso.md`
- Phase 1 findings (now retracted): `docs/notes/2026-05-09-gnn-phase1.md`
- Plan: `docs/superpowers/plans/2026-05-10-gnn-phase2-loso-after-phase1-retraction.md`
- Spec: `docs/superpowers/specs/2026-05-09-non-tabular-model-class-scoping-design.md`
- Code: `src/gnn_stage1_peer/loso.py`, `src/gnn_stage1_peer/model.py` (SAGE + EdgeAttrAwareEncoder), `src/run_gnn_phase2.py`, `src/diagnose_gnn_vs_v4.py`
- SAGE outputs (force-added):
  - `output/pairwise_gnn_phase2.csv`
  - `output/gnn_phase2_loso_summary.json`
  - `output/gnn_phase2_loso_per_holdout.json`
  - `output/gnn_phase2_loso_run.log`
  - `output/diag_gnn_vs_v4.json`
  - `output/diag_gnn_vs_v4_curve.csv`
  - `output/cv_per_season_gnn_phase2_blend.csv`
- Edge-attr outputs (force-added):
  - `output/pairwise_gnn_phase2_edge_attr.csv`
  - `output/gnn_phase2_loso_summary_edge_attr.json`
  - `output/gnn_phase2_loso_per_holdout_edge_attr.json`
  - `output/gnn_phase2_loso_run_edge_attr.log`
  - `output/diag_gnn_vs_v4_edge_attr.json`
  - `output/diag_gnn_vs_v4_edge_attr_curve.csv`
  - `output/cv_per_season_gnn_phase2_edge_attr_blend.csv`
