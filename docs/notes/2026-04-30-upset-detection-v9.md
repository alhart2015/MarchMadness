# Upset-Detection Sub-Model (v9) -- Findings

## Summary

v9 (4-feature, upset-weighted) and v9-B (7-feature fallback) both LOST against
v8 by ~1100 bracket points over 22 LOSO seasons; keep v8 in production and
abandon the upset-detection direction as currently architected.

## Numbers

| Model | 22-season TOTAL bracket pts | 22-season MEAN per season | Weighted-mean LOSO log loss |
|-------|-----------------------------|---------------------------|-----------------------------|
| v4    | 2661                        | 121.0                     | 0.437                       |
| v8    | 2670                        | 121.4                     | 0.432                       |
| v9-A  | 1552 (-1118 vs v8)          | 70.5                      | 0.690 (+0.258 vs v8)        |
| v9-B  | 1588 (-1082 vs v8)          | 72.2                      | 0.699 (+0.267 vs v8)        |

## Per-round chalk-bracket accuracy

| Round | v8     | v9-A   | v9-B   |
|-------|--------|--------|--------|
| R64   | 82.8%  | 60.6%  | 60.9%  |
| R32   | 77.5%  | 41.9%  | 39.9%  |
| S16   | 65.1%  | 38.9%  | 37.7%  |
| E8    | 55.2%  | 32.2%  | 35.6%  |
| F4    | 60.5%  | 25.6%  | 25.6%  |
| Champ | 42.9%  | 23.8%  | 28.6%  |

## Verdict

LOSE on both variants. Keep v8 in production. Abandon the
upset-detection direction as currently architected. Detailed reasoning
follows.

## Diagnostic: why v9 collapsed

### 1. Weighting magnitude is the root cause

With W_UPSET=3.0 and W_MISS=4.0, an upset row where v4 was confidently wrong
gets weight ~15 vs ~1 for a typical non-upset row. With ~27% of training rows
flagged as upsets (392 of ~1450 games), the model is pulled hard toward
predicting upsets even where they are unwarranted. R64 chalk accuracy collapses
from 82.8% (v8) to ~60% (v9). The model "sees" upsets that are not there.

The sanity check is the cleanest proof: running v9 with W_UPSET=1.0, W_MISS=0.0
(uniform weights) produces LL=0.432 and Acc=80.6%, matching v8 exactly. The v9
trainer is architecturally correct. The catastrophic regression is purely due to
weighting magnitude, not a code bug.

### 2. Feature extension did not help

v9-B added 3 features (round, v4 confidence |p_v4 - 0.5|, is_higher_seed)
hoping the model could learn when not to over-predict upsets. It scored 36
bracket pts higher than v9-A (1588 vs 1552), but still -1082 vs v8. Both
feature sets were dominated by the weighting bias; richer features could not
rescue a mis-calibrated loss function.

### 3. Train/apply asymmetry on round in v9-B

pairwise_v4.csv has no DayNum, so the apply path always uses round=0 for every
pair while training rows have round in {1..6}. Any tree splits learned on round
will not fire at inference time. This is not the cause of the failure -- v9-B
was going to lose regardless because it shared v9-A's weighting bias -- but it
would need to be fixed before any future round-aware experiment. Fixing it
requires resolving each (team_a, team_b) pair to its bracket-slot round during
application, which was not done in this work.

## What this means for next steps

Per the spec's success-criteria table, both variants of v9 LOST by far
more than the +/-3 noise band (-1118 and -1082 pts respectively).
Decision per spec:

- **Keep v8 in production.** No code-path change in src/train_stage2.py;
  no change to the bracket pipeline.
- **Abandon the upset-detection direction in TODO.md** as currently
  architected (binary "did A win?" target with upset-weighted loss).
  Promote item #2 ("Ensemble of model classes") to active queue
  position #1.
- **Open question for any future revisit:** could a milder upset
  weighting (W_UPSET in {1.5, 2.0}, W_MISS in {0, 1}) sit alongside
  v8 productively? The current data argues for no: log loss
  monotonically degrades from W=1 (matches v8) to W=3 (catastrophe),
  suggesting the global optimum on this objective is at W=1 (i.e.,
  v8). But the bracket-points objective is not the same as log loss --
  a future attempt could sweep at low weights and target bracket pts
  directly. Documented but not promoted.

## Known limitations

- **v9-B's round feature has a train/apply asymmetry.** pairwise_v4.csv
  carries no DayNum, so the apply path uses round=0 for every pair
  while training rows have round in 1..6. Any tree splits the trainer
  learned on round will not fire at inference time. Fixing this would
  require resolving each (team_a, team_b) pair to its bracket-slot
  round during application -- not done in this work because the
  weighting-magnitude failure dominates either way.

## Files / commits

- Spec: docs/superpowers/specs/2026-04-30-upset-detection-design.md
- Plan: docs/superpowers/plans/2026-04-30-upset-detection.md
- Code: src/train_upset_model.py (16 unit tests passing)
- Outputs: output/pairwise_v9.csv, output/v9_eval.csv
- This writeup: docs/notes/2026-04-30-upset-detection-v9.md
