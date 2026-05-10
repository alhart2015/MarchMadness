# Non-Tabular Stage-1 Peer Model Class -- Scoping Design

**Date:** 2026-05-09
**Branch:** `feat/non-tabular-model-class-scoping`
**Type:** Scoping spec (no implementation; deliverable is a written
candidate landscape + go/no-go recommendation on allocating experiment
effort to a structurally-different model class)
**Predecessors:**
- Team-program tournament-history features -- FAIL (PR 34, 2026-05-09):
  seventh same-data-peer null result; elevated the saturation
  hypothesis to dominant.
  `docs/notes/2026-05-09-team-seed-residual.md`.
- TODO retire-Kaggle-framing (PR 33): production objective is the
  22-season bracket-points backtest (clean baseline 2069), with
  log-loss as a secondary signal.
- 538 + Vegas audits (clean baseline): localized 1 + 6 weak spots
  respectively; no upset-detection edge over Vegas.

## TL;DR

Seven same-data-peer feature additions to v4 have failed on bracket
points (BT-as-feature, feature-view, HBT, Colley, Massey-MOV,
Massey-decay-14d, team-seed-residual). The saturation hypothesis --
v4's 449-tree XGB on the 67-feature team-aggregate stack is near-
saturated -- is now the dominant explanation. Three escape hatches
have been identified:

1. Roster-level returning-experience (genuinely new data) -- TODO
   Active queue #1
2. Pool-aware bracket construction (orthogonal decision layer) --
   TODO Active queue #2
3. Structurally different model class (this spec)

This scoping spec applies a strict inclusion criterion to lane #3:
**candidates must consume the underlying raw data in a way XGB on
the 67-feature stack does not see.** Same-feature candidates (MLP,
Bayesian BT) remain excluded -- they're already deprioritized in
TODO #3 and #4 for the same-data-peer reason.

Four candidates are documented. Lead candidate is a Graph Neural
Network on the regular-season game graph. Phase 1 sanity check
(regular-season prediction vs scalar Massey baseline) gates the
expensive Phase 2 LOSO experiment. **Total budget if all phases
run: ~2.5 weeks. Total budget if killed at Phase 1: ~3 days.**
The other three candidates are documented as deferred sequels and
are re-ranked based on the GNN's failure mode.

## Motivation

The seven-failure pattern is summarized in the team-seed-residual
findings (`docs/notes/2026-05-09-team-seed-residual.md`):

> XGB's existing 67-feature stack does not extract bracket-points
> value from [the team-program signal] at this data scale.
> [...] The strongest remaining hypothesis is that v4 is near-
> saturated on tabular team-aggregate features and further
> improvements require either (a) external player-level data
> (roster), (b) bracket-construction strategy (orthogonal axis),
> or (c) a model class that's structurally different from XGB on
> tabular features (genuine non-tabular methods).

Lane (c) was theoretically open but practically not on the active
queue -- TODO #3 (MLP) and #4 (Bayesian BT) are scoped model-class
candidates but both consume the same 67 features that XGB does, so
they collapse back to the same-data-peer risk profile. This spec
exists to either promote (c) to a real lane by identifying a
candidate with a different risk profile, or formally close it.

## Hypothesis under test

> A model class that consumes the underlying raw basketball data in
> a structurally different way -- e.g., as a graph, a sequence, a
> distributional rep, or a learned latent embedding -- can extract
> signal that XGB on the 67-feature flat stack misses, sufficient
> to clear the BT-class LL-blend gate as a stage-1 peer to v4 AND
> the +25 bracket-points gate on a v8 retrain.

If the lead candidate (GNN) clears both gates: lane #3 promotes to
a real saturation-break path; sequels #2-4 enter the queue as
follow-ups. If the lead candidate fails: re-rank or close the lane
based on the failure mode.

## Inclusion criterion: "different inputs"

A candidate qualifies if it consumes underlying data in a way XGB
on the 67-feature stack does not. Three example signal classes:

- **Graph topology** of the regular-season game graph (multi-hop
  schedule structure with edge-level margin/site/temporal modulation)
- **Per-team game-by-game sequence** (temporal trajectory across
  the season, including injury recovery and conference-tournament form)
- **Fine-grained game logs** (variance distribution of the four
  factors across games, not just season means)

Excluded:

- **MLP on the 67-feature stack** -- same inputs, different
  functional form. Already deprioritized in TODO #3.
- **Bayesian BT in any reparameterization** -- same inputs, latent
  reformulation. Already deprioritized in TODO #4.
- **Embeddings into XGB** -- any candidate that re-flattens the
  underlying signal as embeddings consumed by XGB collapses back
  to the seven-failure-pattern risk profile.

Rationale: the seven-failure pattern is now strong evidence that
*same inputs, different model* doesn't work at v4's data scale.
The whole theoretical justification for treating "model class" as
a saturation-break lane is that a structurally different inductive
bias accesses information XGB can't see -- but only if it actually
consumes data XGB doesn't.

## Role in the v4/v8 stack: stage-1 peer

Each candidate produces per-game `p(team A beats team B)` for
tournament games in each LOSO season. Output is a 22-season
pairwise frame in the same shape as `pairwise_v4.csv`. Evaluated
via the BT-class LL-blend gate against v4 -- the same gate that
BT, HBT, feature-view, Colley, Massey-MOV/decay went through.

If LL-blend gate clears: ensemble with v4 -> retrain v8 stage-2
on blended frame -> check 22-season bracket-points delta vs
canonical 2069.

This role is chosen over alternatives because:

- **Apples-to-apples with prior model-class experiments.** Enables
  same-pattern comparison; the LL-blend gate's clauses are
  empirically calibrated against six prior runs.
- **Lower bar than stage-1 standalone replacement.** A candidate
  doesn't need to BEAT v4 on standalone LL -- just disagree
  usefully with v4. If a candidate is strong enough to be a
  standalone replacement, it'll show up as a strong stage-1 peer
  first (the converse is not true).
- **Cleaner saturation-break test than embeddings-into-XGB.**
  Embeddings re-flatten the new signal into XGB's view, putting
  the candidate back at same-data-peer risk.

## Candidate landscape

Four candidates are documented. Each lists its saturation-break
theory (what it sees that XGB doesn't), input format, output
format, prototype cost, and dominant collapse risk.

### Candidate 1 (lead): Graph Neural Network on regular-season game graph

- **Input:** One graph per season. Nodes = ~350 D-I teams. Edges
  = each regular-season game played, with edge features (margin,
  site, days-rest, days-from-season-end). Built only from games
  before tournament start. Cross-season edges from prior seasons'
  tournaments are leak-safe and recommended as additional graph
  input (~1,400 historical-tournament edges per season's graph;
  flagged as A/B-testable in Phase 1).
- **Saturation-break theory:** Multi-hop schedule topology with
  edge-level margin/site/temporal modulation. Massey and KenPom
  encode strength-of-schedule as a *per-team scalar*; a GNN
  encodes structural context per team across multi-hop graph
  structure. Edge attributes (site, rest, temporal) are
  information scalar Massey doesn't see -- rules out the trivial
  "GNN re-derives Massey" outcome.
- **Output:** Two-team readout from the trained graph + matchup
  MLP -> p(A beats B in tournament).
- **Tooling:** PyTorch + PyTorch Geometric (or DGL). CPU-tractable
  at this data scale.
- **Cost:** Phase 1 ~3 days. Phase 2 ~1.5 weeks. v8 retrain ~2 days.
  **Total ~2.5 weeks** if all phases run.
- **Dominant collapse risk:** Massey/KenPom already extract the
  bulk of the relational signal. Mitigation: edge attributes that
  scalar Massey doesn't capture; Phase 1 gate ensures the GNN
  beats Massey on RS prediction before committing to LOSO.

### Candidate 2: Sequence model on per-team game-by-game trajectory

- **Input:** Per team, a chronological sequence of (opponent_id,
  opp_strength, margin, site, days-rest, win/loss) tuples through
  the regular season. Encoded with an LSTM or small transformer.
- **Saturation-break theory:** Temporal trajectory -- *how* a
  team trended into the tournament. v4 has season-aggregate
  features and EWMA-style coach features but no per-team
  game-by-game sequence representation. "Peaking vs declining,"
  injury recovery, late-conference-tournament form are hidden in
  season averages.
- **Output:** Two team encodings -> matchup scorer -> p(A beats B).
- **Tooling:** PyTorch sequence models. CPU-tractable.
- **Cost:** Phase 1 ~3 days. Phase 2 ~1.5 weeks.
- **Dominant collapse risk:** Highest of the four. KenPom recent-
  form / Massey late-season weighting partially encode this.
  Mitigation: explicitly separate "season-end form" from "season-
  mean strength" and check whether the sequence rep adds beyond
  scalar form indices.

### Candidate 3: Box-score four-factor distributional model

- **Input:** Per game, four-factor box score (eFG%, TO%, OREB%,
  FT/FGA) plus pace, for each team. Aggregated to per-team
  *distributional* representations -- means, variances, skews, and
  game-context conditional means (vs top-50, vs road, etc.) over
  the season.
- **Saturation-break theory:** v4 carries season-mean four-factor
  stats from KenPom but not the across-game *variance distribution*.
  A team that wins by relying on 3-point variance has different
  tournament-survival odds than a team that wins through consistent
  rebounding + eFG%, even at identical adjusted efficiency. This is
  real signal Sagarin-style aggregates discard.
- **Output:** Per-team distributional rep -> matchup interaction
  scorer -> p(A beats B).
- **Tooling:** Box-score data wrangling (Kaggle `Events*.csv` or
  KenPom game-detail) is the long pole. Modeling itself is small
  (custom feature pipeline + LR or small NN).
- **Cost:** Phase 1 ~5 days (data wrangling dominates). Phase 2 ~1 week.
- **Dominant collapse risk:** Box-score data quality pre-2003 is
  uneven; variance signal might be thin in practice. Mitigation:
  subset to seasons with full box-score coverage in Phase 1;
  expand if signal exists.

### Candidate 4: Self-supervised team embeddings via regular-season margin prediction

- **Input:** All regular-season game outcomes (team_A, team_B,
  margin, site). Pretrain team embeddings + optional style vectors
  end-to-end to predict signed margin.
- **Saturation-break theory:** Latent style/matchup specificity.
  v4 features are *team-only* -- XGB sees A's stats and B's stats
  separately and learns generic interactions. A learned style
  vector per team can encode "team X's offense interacts
  specifically with team Y's defense" -- interactions invisible
  when teams are treated as independent feature rows.
- **Output:** Dot-product (or learned scorer) of A and B
  embeddings + game features -> p(A beats B).
- **Tooling:** PyTorch matrix-factorization-style. CPU-tractable.
- **Cost:** Phase 1 ~3 days. Phase 2 ~1 week.
- **Dominant collapse risk:** Could collapse to global strength
  (re-derive Massey). Mitigation: hold a strength scalar fixed and
  learn a residual style component, or use high-rank embeddings +
  regularization.

## LOSO experiment architecture (applies to all candidates)

For each LOSO season S in {2003-2025} \ {2020}:

- **Training labels (supervised signal):** Tournament-game
  outcomes from all 21 seasons except S. Optional supplemental:
  late-season RS games and conference-tournament games at lower
  weight (mirrors v4's `supplemental_weight=0.25`).
- **Training graphs / sequences / distributional reps (model
  input):** Every season's RS-derived data, including S's. RS
  data is consumed at every training step and at inference -- it
  is input, not labels.
- **Test (held out):** S's tournament-game outcomes. Predictions
  use S's RS-only data as input. Cross-season tournament games
  from seasons < S are leak-safe and recommended as additional
  input edges in S's graph (~1,400 historical-tournament edges).

This mirrors v4 LOSO exactly: v4 trains on tournament games from
21 seasons, with features computed from each season's RS-derived
inputs (Massey, KenPom, etc.). The new candidates consume the
RS-derived inputs differently -- as a graph (Candidate 1),
sequence (Candidate 2), distributional rep (Candidate 3), or
learned embedding (Candidate 4) -- but the LOSO scope is identical.

**Two leak-safety asymmetries to be explicit about:**

1. **Within-season:** S's RS data is leak-safe input even when
   predicting S's tournament games (RS games end before Selection
   Sunday).
2. **Cross-season:** Tournament games in seasons < S are leak-safe
   and CAN be added as edges/inputs for S's graph (Candidate 1) or
   as additional supervision (all candidates).

## Phase 1 -- pre-LOSO sanity check

Before the expensive Phase 2 LOSO, validate that the candidate has
signal at the regular-season game level.

**GNN (lead) sanity check:**

- Train on RS games before March 1 of season S.
- Predict held-out late-season games (March 1 -> Selection Sunday).
- Baseline: scalar Massey rating differential.
- **Gate:** GNN must beat scalar Massey LL by >= 0.005 averaged
  over 5+ test seasons (e.g., 2018-2025 ex-2020).
- **If fails:** kill before Phase 2 -- Massey has already
  extracted the graph signal.

Adapted gates for the other candidates:

- **Sequence model (Candidate 2):** beat KenPom recent-form scalar
  on RS late-season prediction by >= 0.005 LL.
- **Box-score model (Candidate 3):** beat KenPom adjusted four-
  factor on RS prediction by >= 0.005 LL.
- **Self-supervised embeddings (Candidate 4):** beat Massey
  RS-prediction LL by >= 0.005, OR show reconstruction quality
  on a held-out RS subset that's meaningfully above the
  strength-only baseline.

Phase 1 also serves as a **wall-clock timing check** for Phase 2
cost estimation. If a single season's Phase 1 training exceeds
~30 minutes on CPU, re-evaluate Phase 2 scope (push to expanded
out-of-scope or seek GPU access).

## Phase 2 -- LOSO experiment

If Phase 1 passes:

- For each of 22 LOSO seasons (2003-2025 ex-2020), train the
  candidate on the LOSO scope defined above.
- Predict every tournament game in season S.
- Output: 22-season pairwise frame in the same shape as
  `pairwise_v4.csv`.

## Gate criteria (BT-class LL-blend, applied to candidate pairwise vs v4 pairwise)

**Three clauses (all must pass):**

1. **Disagreement correlation r >= 0.60.** When candidate and v4
   disagree on the picked side, the candidate must be right >= 45%
   of the time. r < 0.60 means the candidate is a same-data peer
   in disguise: where it disagrees with v4, it's not adding
   information.
2. **Blend weight w_v4 non-degenerate.** The LL-optimal blend
   weight on v4 must be in [0.40, 0.85]. w_v4 >= 0.95 means the
   candidate is being weighted to ~zero (no signal); w_v4 <= 0.30
   means the candidate is dominating, which would suggest the
   candidate alone should be tested as a stage-1 replacement (out
   of scope here).
3. **Blend headroom >= +0.005 LL** over v4 standalone. The blended
   LL on the 22-season tournament games must beat v4-standalone
   LL by at least 0.005 (the BT-class headroom threshold).

If all three pass: retrain v8 stage-2 on the v4+candidate-blended
frame (same procedure as the team-history experiment). Check
22-season bracket-points delta vs canonical 2069.

**Bracket-points gate:** delta >= +25 pts over 22 LOSO seasons
(PR 17's pass bar for stage-1 peer additions on the production
metric).

## Kill criteria (any one closes the lane on the candidate)

1. **Phase 1 fail:** Candidate doesn't beat its scalar baseline
   by >= 0.005 LL on RS prediction. Indicates the structural
   signal is already extracted by existing scalar features.
2. **Phase 2 clause 1 fail:** Disagreement correlation r < 0.60.
   Candidate is a same-data peer in disguise.
3. **Phase 2 clause 2 fail (high side only):** Optimal blend
   weight w_v4 >= 0.95. Candidate is being weighted to ~zero;
   carries no usable signal in the blend. (The opposite case --
   w_v4 <= 0.30, candidate dominating -- is NOT a kill: it
   indicates the candidate alone should be re-scoped as a
   stage-1 standalone replacement, which is out of scope for
   this spec but a clean re-direct rather than a closure.)
4. **Phase 2 clause 3 fail:** Blend headroom < +0.005 LL.
   Candidate doesn't add information at the tournament-prediction
   level.
5. **Bracket-points re-test fail:** delta < +25 pts on 22 LOSO
   seasons. Candidate has LL signal but doesn't translate to
   chalk-bracket production metric.

## Cost estimate

| Phase | GNN (lead) | Other candidates |
|---|---|---|
| Phase 1 (RS sanity check) | ~3 person-days | ~3-5 days each (Box-score: 5 days due to data wrangling) |
| Phase 2 (22-season LOSO) | ~1.5 weeks | ~1-1.5 weeks each |
| v8 retrain (if gate clears) | ~2 days | ~2 days each |
| **Total if all phases run** | **~2.5 weeks** | ~2-3 weeks each |
| **Total if killed at Phase 1** | ~3 days | ~3-5 days each |

## Sequel ordering

After the GNN result lands, re-rank the remaining candidates based
on what failed:

- **GNN passes both gates:** add as stage-1 peer; #2-4 deferred
  unless multi-peer ensemble is desired (separate complexity-vs-
  marginal-gain decision).
- **GNN fails Phase 1 (Massey absorbs structural signal):** rank
  up #4 (self-supervised embeddings -- similar saturation-break
  theory at team level), keep #3 (box-score -- distinct signal
  class), deprioritize #2 (sequence -- same "already aggregated"
  risk extends to recent-form scalars).
- **GNN passes Phase 1 but fails Phase 2 LL-blend gate:** rank up
  #2 (sequence -- different signal type, may transfer better to
  tournament distribution), keep #3, deprioritize #4 (similar risk
  profile if global-strength collapse on RS doesn't carry
  tournament gain).
- **GNN passes LL-blend but fails bracket-points re-test:** treat
  as an eighth same-data-peer-in-spirit failure. Don't run #2-4
  unless one has a meaningfully different bracket-points theory
  beyond stage-1 LL-blend signal.

## Out of scope

- **Same-feature candidates** (MLP, Bayesian BT) -- already
  deprioritized in TODO Active queue items #3 and #4.
- **GPU-scale compute** (transformer pretraining over multi-season
  game-detail corpus). Reconsider if Phase 1 timing on a small
  candidate suggests the lane is bottlenecked on compute capacity
  rather than signal availability.
- **Stage-2 replacement candidates** (model that consumes raw data
  and produces bracket picks directly). Would require full v8
  redesign; out of scope for stage-1 peer scoping.
- **Champion-only or round-conditional prediction.** No per-game
  p(win) -> can't be evaluated via LL-blend gate.
- **External data candidates** (roster-level, Vegas futures).
  Tracked separately in TODO Active queue #1 and #5.

## Procedural requirements

These reflect lessons from prior LOSO experiments. They are
**hard requirements**, not suggestions, for any candidate that
proceeds to Phase 2.

### Persist all output data via git

Force-add every output that needs to survive beyond the branch's
working life. **Do not rely on `.gitignore`-permitted artifacts
surviving cleanup or worktree teardown.**

Concrete must-force-add list, mirroring what the team-seed-
residual experiment did (PR 34):

- Phase 1 diagnostic JSON + log
- Phase 2 LOSO run log
- Per-season LL/acc CSV (`cv_per_season_*.csv`)
- 22-season pairwise frame (the candidate's output)
- v8 retrain log + retrained pairwise frame (if Phase 2 gate clears)
- Verdict summary JSON + .txt
- Anchor invariance log + drop-features pairwise frame

The plan document (the eventual
`docs/superpowers/plans/2026-05-09-non-tabular-model-class-scoping.md`)
must include explicit `git add -f <path>` steps in each of these
phases. The data wipe of 2026-05-04 lost PR 21's clean
`pairwise_v4.csv` because it was gitignored and lived only in a
wiped worktree; that lesson applies forward.

### Branch workflow

Work happens on a regular branch in the main repo, not in
`.claude/worktrees/`. Create the branch with `git checkout -b
feat/<name>` from the main repo path; do not use `git worktree add`
unless explicit parallelism is required and the user confirms.

### `MM_PAIRWISE_OUT` instability on Windows

Per the team-seed-residual findings, `MM_PAIRWISE_OUT` in
`enhanced_model_v3.py` proved unstable on Windows for runs longer
than ~6-20 seasons (silent OS kill mid-LOSO loop). The custom
driver pattern from `src/loso_with_pairwise_for_team_history.py`
(explicit `gc.collect()` between seasons) is reusable and
recommended as the starting point for any candidate's Phase 2
LOSO loop. Engineering follow-up to refactor v3 itself remains
queued in TODO; until then, candidate-specific drivers are the
practical path.

## Anchor invariance check

Before claiming a Phase 2 result, run a drop-candidate variant of
the LOSO loop with the candidate's pairwise output replaced by
v4's. Verify the resulting LL is within < 1e-3 max-abs-diff of
canonical v4. This confirms the wire-in is non-invasive and that
any LL drift in the with-candidate run reflects the candidate's
signal, not a wire-in defect. (Same pattern as the team-history
experiment's Task 9.)

## Open questions / methodological notes

1. **"Meaningfully beat scalar Massey" Phase 1 threshold of
   +0.005 LL** is calibrated to the BT-class LL-blend headroom
   gate. If RS-prediction base rate is unstable across test
   seasons, this may need to be expressed as a per-season win rate
   ("beat in 4 of 5 test seasons") instead of a pooled LL delta.
2. **Edge attribute richness for the GNN.** Recommend including
   site, rest, days-from-season-end, and home/away-streak features
   per edge. Massey doesn't see these; rules out the trivial "GNN
   re-derives Massey" outcome and provides headroom for the GNN to
   demonstrate distinct signal.
3. **Per-LOSO-season GNN training wall-clock on CPU** is the
   biggest unknown. Phase 1 doubles as a timing check; if a single
   season exceeds ~30 minutes on CPU, escalate scope (GPU access
   or simpler architecture).
4. **Cross-season tournament edges in S's graph.** Recommend
   including (~1,400 historical-tournament edges per season's
   graph); flag as A/B-testable in Phase 1 (compare Phase 1 GNN
   with vs without prior-tournament edges).
5. **Multi-peer ensemble after Phase 2 success.** If the GNN
   passes both gates, an open question is whether to also evaluate
   #2-4 as additional peers in a multi-peer ensemble, or to stop
   at one peer per saturation-break theory class. Defer until
   GNN's result is known; the answer depends on whether the GNN's
   signal is "everything in #2-4 plus more" or "different from
   #2-4 in ways that an ensemble could combine."
6. **What if GNN's success undermines the saturation hypothesis?**
   A clean GNN pass (Phase 1 + Phase 2 LL-blend + bracket-points)
   would be evidence that the saturation hypothesis was incomplete
   -- specifically, that what looked like "tabular saturation" was
   really "the wrong inductive bias for the data." Would prompt
   a TODO-Active-queue rewrite to elevate model-class lanes more
   broadly. No need to pre-decide that rewrite; document the
   contingency.

## Files of record

- This scoping spec: `docs/superpowers/specs/2026-05-09-non-tabular-model-class-scoping-design.md`
- (After implementation plan is written) Implementation plan:
  `docs/superpowers/plans/2026-05-09-non-tabular-model-class-scoping.md`
- Predecessor findings: `docs/notes/2026-05-09-team-seed-residual.md`
- Strategic context: `TODO.md` Active queue (item #3 to be replaced
  by this scoping's recommendation upon completion)
