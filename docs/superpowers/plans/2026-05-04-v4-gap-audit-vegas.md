# v4 Gap Audit vs Vegas -- Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a per-bucket diagnostic of where v4 specifically underperforms Vegas closing-line implied probabilities on tournament games. Single audit note + JSON + 3 PNGs.

**Architecture:** One driver script `src/audit_v4_gap_vegas.py`. Reuses `load_vegas_lines` + `_build_vegas_name_to_kaggle_map` + `_resolve_vegas_name` from `src/enhanced_model_v2.py` and `score_chalk_brackets`-style joins on tournament outcomes.

**Tech Stack:** pandas, numpy, scipy.stats (norm.cdf for spread-to-prob), matplotlib (PNGs), pytest.

**Spec:** `docs/superpowers/specs/2026-05-04-v4-gap-audit-vegas-design.md`

---

## File Structure

**Created (committed):**

- `src/audit_v4_gap_vegas.py` -- driver script (~350 LOC).
  - Public: `run_audit(v4_csv, out_dir, out_json) -> dict`
  - Private: `_spread_to_prob(spread, sigma=11.0) -> float`
  - Private: `_build_vegas_lookup(vegas_df, teams, spellings) -> dict[(season, day, ta, tb), float]`
  - Private: `_join_v4_vegas_outcomes(...)` -> per-game DataFrame
  - Private: `_compute_bucket_metrics(df, bucket_col) -> dict`
  - Private: `_calibration_table(p_pred, y_actual, n_bins=10) -> list[dict]`
  - Private: `_ece(cal_table, n_total) -> float`
  - Private: `_save_calibration_plot`, `_save_per_bucket_bar_plot`
- `tests/test_audit_v4_gap_vegas.py` -- 6+ unit tests (~150 LOC).
- Spec + plan (this file).

**Generated (committed via `git add -f`):**

- `output/v4_gap_audit_vegas.json` -- per-bucket metrics.
- `output/v4_gap_calibration_overall.png`
- `output/v4_gap_calibration_by_round.png`
- `output/v4_gap_per_bucket_ll_delta.png`
- `output/v4_gap_audit_log.txt`
- `docs/notes/2026-05-04-v4-gap-audit-vegas.md` (findings).

**Modified:**

- `TODO.md` -- mark item #1 done; promote 538 audit to #1 explicitly.

---

## Phase 1: Vegas join + per-bucket metrics

Single phase. Audit is small enough that breaking it further would be theatre.

### Task 1: Implement `src/audit_v4_gap_vegas.py`

**Files:**
- Create: `src/audit_v4_gap_vegas.py`
- Test: `tests/test_audit_v4_gap_vegas.py`

- [ ] **Step 1: Write failing unit tests for the math + bucketing**

```python
"""Unit tests for src/audit_v4_gap_vegas.py."""
import math

import numpy as np
import pandas as pd
import pytest

from src.audit_v4_gap_vegas import (
    SIGMA,
    _calibration_table,
    _compute_bucket_metrics,
    _ece,
    _seed_diff_bucket,
    _spread_to_prob,
    _v4_confidence_quintile,
)


def test_spread_to_prob_anchors():
    """SIGMA=11. spread=0 -> 0.5; spread=11 -> N(0,1).cdf(1) ~= 0.8413;
    spread=-5.5 -> 1 - N(0,1).cdf(0.5) = 0.3085."""
    from scipy.stats import norm

    assert abs(_spread_to_prob(0.0) - 0.5) < 1e-9
    assert abs(_spread_to_prob(11.0) - norm.cdf(1.0)) < 1e-9
    assert abs(_spread_to_prob(-5.5) - norm.cdf(-0.5)) < 1e-9


def test_seed_diff_bucket_boundaries():
    """0-2, 3-5, 6-9, 10-15."""
    assert _seed_diff_bucket(0) == "0-2"
    assert _seed_diff_bucket(2) == "0-2"
    assert _seed_diff_bucket(3) == "3-5"
    assert _seed_diff_bucket(5) == "3-5"
    assert _seed_diff_bucket(6) == "6-9"
    assert _seed_diff_bucket(9) == "6-9"
    assert _seed_diff_bucket(10) == "10-15"
    assert _seed_diff_bucket(15) == "10-15"


def test_v4_confidence_quintile_boundaries():
    """Quintiles by predicted prob for the favored side."""
    assert _v4_confidence_quintile(0.55) == "0.50-0.60"
    assert _v4_confidence_quintile(0.60) == "0.50-0.60"
    assert _v4_confidence_quintile(0.61) == "0.60-0.70"
    assert _v4_confidence_quintile(0.95) == "0.90-1.00"
    # Mirror probabilities below 0.5 (the "favored side" interpretation):
    assert _v4_confidence_quintile(0.40) == "0.50-0.60"
    assert _v4_confidence_quintile(0.05) == "0.90-1.00"


def test_calibration_table_perfect():
    """Empirical rate equals predicted bin midpoint for a perfectly
    calibrated synthetic dataset."""
    rng = np.random.default_rng(0)
    n = 5000
    p = rng.uniform(0.5, 1.0, size=n)
    y = (rng.random(n) < p).astype(int)
    table = _calibration_table(p, y, n_bins=10)
    # ECE should be small.
    ece = _ece(table)
    assert ece < 0.03


def test_calibration_table_overconfident():
    """A model that says 0.9 but only wins 0.7 of the time should have
    high ECE."""
    n = 1000
    p = np.full(n, 0.9)
    y = (np.arange(n) < 700).astype(int)  # 700/1000 = 0.7 win rate
    table = _calibration_table(p, y, n_bins=10)
    ece = _ece(table)
    assert ece > 0.15


def test_compute_bucket_metrics_aggregates_correctly():
    """Three games in one bucket; compute LL + accuracy by hand and
    verify."""
    df = pd.DataFrame([
        {"bucket": "R64", "p_v4": 0.8, "p_vegas": 0.7, "winner_is_a": 1},
        {"bucket": "R64", "p_v4": 0.6, "p_vegas": 0.55, "winner_is_a": 1},
        {"bucket": "R64", "p_v4": 0.4, "p_vegas": 0.5, "winner_is_a": 0},
    ])
    by_bucket = _compute_bucket_metrics(df, "bucket")
    cell = by_bucket["R64"]
    assert cell["n_games"] == 3
    # ll_v4 = -mean(log(0.8), log(0.6), log(0.6))
    expected_ll = -np.mean([np.log(0.8), np.log(0.6), np.log(0.6)])
    assert abs(cell["ll_v4"] - expected_ll) < 1e-9
    # acc_v4: 0.8>0.5 chalk hit; 0.6>0.5 hit; 0.4<0.5 hit (since winner_is_a=0)
    assert cell["acc_v4"] == pytest.approx(1.0)
```

Run: expect ImportError.

- [ ] **Step 2: Implement the module**

`src/audit_v4_gap_vegas.py` outline:

```python
"""Audit v4's tournament-game predictions against Vegas closing-line
implied probabilities, broken down by round, higher-vs-lower-seed
status, v4-confidence quintile, and seed-difference magnitude.

Spec: docs/superpowers/specs/2026-05-04-v4-gap-audit-vegas-design.md
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.enhanced_model_v2 import (
    load_vegas_lines,
    _build_vegas_name_to_kaggle_map,
    _resolve_vegas_name,
)

SIGMA = 11.0
DATA = Path("data/raw/march-machine-learning-2026")
DEFAULT_OUT_JSON = "output/v4_gap_audit_vegas.json"

# Tournament round inference from DayNum (Kaggle convention):
# 134-135: First Four, 136-137: R64, 138-139: R32, 143-144: S16,
# 145-146: E8, 152: F4, 154: Champ
ROUND_BY_DAYNUM = {
    134: "FF", 135: "FF",
    136: "R64", 137: "R64",
    138: "R32", 139: "R32",
    143: "S16", 144: "S16",
    145: "E8",  146: "E8",
    152: "F4",  153: "F4",
    154: "Champ",
}

CONFIDENCE_BIN_EDGES = [0.50, 0.60, 0.70, 0.80, 0.90, 1.00]
CONFIDENCE_BIN_LABELS = ["0.50-0.60", "0.60-0.70", "0.70-0.80",
                          "0.80-0.90", "0.90-1.00"]


def _spread_to_prob(spread, sigma=SIGMA):
    return float(norm.cdf(float(spread) / sigma))


def _v4_confidence_quintile(p_a):
    """Map predicted prob (for either side) to confidence quintile of
    the favored side (>= 0.5)."""
    p_fav = max(p_a, 1.0 - p_a)
    for lo, hi, label in zip(CONFIDENCE_BIN_EDGES[:-1],
                              CONFIDENCE_BIN_EDGES[1:],
                              CONFIDENCE_BIN_LABELS):
        if lo <= p_fav <= hi:
            return label
    return CONFIDENCE_BIN_LABELS[-1]


def _seed_diff_bucket(d):
    if d <= 2:
        return "0-2"
    if d <= 5:
        return "3-5"
    if d <= 9:
        return "6-9"
    return "10-15"


def _round_from_daynum(daynum):
    return ROUND_BY_DAYNUM.get(int(daynum), "OTHER")


def _calibration_table(p_pred, y_actual, n_bins=10):
    """Build per-bin (predicted-mid, empirical-rate, n) table over [0,1]."""
    edges = np.linspace(0, 1, n_bins + 1)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (p_pred >= lo) & (p_pred < hi if hi < 1.0 else p_pred <= hi)
        n = int(mask.sum())
        if n == 0:
            empirical = None
        else:
            empirical = float(y_actual[mask].mean())
        out.append({
            "bin": [float(lo), float(hi)],
            "mid": float((lo + hi) / 2),
            "n": n,
            "empirical": empirical,
        })
    return out


def _ece(cal_table):
    """Expected calibration error: weighted-by-n absolute gap from
    diagonal."""
    n_total = sum(b["n"] for b in cal_table)
    if n_total == 0:
        return float("nan")
    s = 0.0
    for b in cal_table:
        if b["empirical"] is None:
            continue
        s += (b["n"] / n_total) * abs(b["mid"] - b["empirical"])
    return float(s)


def _compute_bucket_metrics(df, bucket_col):
    """For each unique value of df[bucket_col], compute LL + accuracy +
    calibration."""
    out = {}
    for value, sub in df.groupby(bucket_col):
        n = len(sub)
        if n == 0:
            continue
        # winner_is_a is 1 if team_a (the lower TeamID) won, else 0.
        # p_v4 and p_vegas are P(team_a wins). The actual outcome is winner_is_a.
        eps = 1e-15
        p_v4_w = np.where(sub["winner_is_a"] == 1, sub["p_v4"], 1 - sub["p_v4"])
        p_ve_w = np.where(sub["winner_is_a"] == 1, sub["p_vegas"], 1 - sub["p_vegas"])
        ll_v4 = float(-np.mean(np.log(np.clip(p_v4_w, eps, 1 - eps))))
        ll_ve = float(-np.mean(np.log(np.clip(p_ve_w, eps, 1 - eps))))
        acc_v4 = float(((sub["p_v4"] >= 0.5).astype(int) == sub["winner_is_a"]).mean())
        acc_ve = float(((sub["p_vegas"] >= 0.5).astype(int) == sub["winner_is_a"]).mean())
        cal_v4 = _calibration_table(sub["p_v4"].to_numpy(),
                                     sub["winner_is_a"].to_numpy())
        cal_ve = _calibration_table(sub["p_vegas"].to_numpy(),
                                     sub["winner_is_a"].to_numpy())
        out[str(value)] = {
            "n_games": int(n),
            "ll_v4": ll_v4,
            "ll_vegas": ll_ve,
            "ll_delta": ll_v4 - ll_ve,
            "acc_v4": acc_v4,
            "acc_vegas": acc_ve,
            "ece_v4": _ece(cal_v4),
            "ece_vegas": _ece(cal_ve),
            "mean_p_v4_minus_vegas": float((sub["p_v4"] - sub["p_vegas"]).mean()),
            "calibration_v4": cal_v4,
            "calibration_vegas": cal_ve,
        }
    return out
```

Plus the join logic:

```python
def _vegas_to_seasonday(vegas_df, day_zero_by_season):
    """Add (season, daynum) columns to vegas_df via the date string and
    each season's DayZero."""
    # For each row, parse date "MM/DD/YYYY" -> datetime, then find which
    # season (by season-end-year), and compute DayNum from DayZero.
    ...

def _build_per_game_audit_df(
    v4_csv, vegas_df, name_resolution, results_df, seeds_df,
    day_zero_by_season,
) -> pd.DataFrame:
    """Per-game DataFrame keyed by (season, daynum, team_a, team_b) with
    columns p_v4, p_vegas, winner_is_a, round, higher_seed, seed_a,
    seed_b."""
    ...

def run_audit(v4_csv, out_dir, out_json) -> dict:
    ...
```

For the join:
1. Load `MNCAATourneyCompactResults` (per-game truth).
2. Load `MNCAATourneySeeds` (seeds per (Season, TeamID)).
3. Load `MSeasons.csv` for `DayZero`.
4. Load Vegas lines via `load_vegas_lines()`.
5. For each Vegas row: parse date -> compute (season, daynum). Resolve home_id, road_id via existing fuzzy matcher.
6. Build a lookup `(season, daynum, min(home_id, road_id), max(home_id, road_id)) -> p_a_wins_vegas`. If `home_id < road_id`, p_a_wins_vegas = `_spread_to_prob(line)`; else `1 - _spread_to_prob(line)`.
7. For each tournament-game row: find the Vegas lookup with matching (season, daynum, team_pair). Allow daynum +/- 1 slack to absorb timezone / scheduling drift.
8. For each pairing: also look up p_v4 from the v4 pairwise CSV (where team_a < team_b).
9. Compute `winner_is_a = 1 if WTeamID == team_a else 0`.
10. Bucket each row.

- [ ] **Step 3: Run unit tests**

`pytest -v tests/test_audit_v4_gap_vegas.py` -- expect PASS.

- [ ] **Step 4: Run the audit on real data**

```bash
python src/audit_v4_gap_vegas.py \
    --v4 output/pairwise_v4.csv \
    --out-dir output/ \
    --out-json output/v4_gap_audit_vegas.json \
    2>&1 | tee output/v4_gap_audit_log.txt
```

Estimated wall time: ~30-60s (Vegas CSV loading + name resolution is the slow part).

- [ ] **Step 5: Verify anchors**

From the JSON's `overall` block:

- `ll_v4` ~= 0.4369 (matches plain-BT diagnostic finding) -- if not, halt and debug join/LL.
- `acc_vegas` in [0.70, 0.72].
- `join_coverage.n_both / n_tournament_games >= 0.60`. If not, halt.

If anchors pass, the per-bucket breakdown is trustworthy.

- [ ] **Step 6: Inspect findings + write the note**

Read the JSON's `weak_spots` array (top buckets sorted by `ll_delta` with `n >= 50`). Write `docs/notes/2026-05-04-v4-gap-audit-vegas.md`:

```markdown
# v4 Gap Audit vs Vegas -- Findings

**Date:** 2026-05-04
**Branch:** feat/v4-gap-audit-vegas
**Verdict:** [N weak spots identified | v4 is at Vegas-tier overall, gap is elsewhere]
**Spec:** ...

## TL;DR
[1-paragraph summary of where v4 specifically loses + the immediate
 followup audit (538) that's queued.]

## Setup recap
[Inputs, sample sizes, anchor results, join coverage, total wall time.]

## Headline numbers
| metric | v4 | Vegas |
|--------|----|----|
| log loss | 0.4369 | NN.NN |
| accuracy | 0.805 | NN.NN |
| ECE | NN.NN | NN.NN |

## Calibration overall
[Embed output/v4_gap_calibration_overall.png + brief commentary]

## Per-bucket weak spots
[Top 3-5 bucket signatures with ll_delta >= 0.02 and n >= 50,
 each with a concrete interpretation + candidate engineering target.]

## Calibration by round
[Embed output/v4_gap_calibration_by_round.png + commentary.]

## What this implies for the queue
[Move 538 audit to active queue #1 explicitly. List 1-2 concrete
 engineering experiments suggested by the weak-spot signatures.]

## Files of record
```

- [ ] **Step 7: Update `TODO.md`**

- Move active queue item #1 (v4 gap audit) to "Done" (with the verdict + key numbers).
- Promote item #2 (external rankings / 538) to active queue #1, *or* if 538 audit is genuinely the immediate followup (not engineering against the weak spots), make 538 audit the new #1 and the engineering followups a new #2-5 cluster.

- [ ] **Step 8: Force-add output artifacts + ASCII verify + commit**

```bash
git add -f output/v4_gap_audit_vegas.json output/v4_gap_calibration_*.png \
            output/v4_gap_per_bucket_ll_delta.png output/v4_gap_audit_log.txt
git add docs/notes/2026-05-04-v4-gap-audit-vegas.md TODO.md \
        src/audit_v4_gap_vegas.py tests/test_audit_v4_gap_vegas.py \
        docs/superpowers/specs/2026-05-04-v4-gap-audit-vegas-design.md \
        docs/superpowers/plans/2026-05-04-v4-gap-audit-vegas.md
```

- [ ] **Step 9: Run pytest + ASCII verification**

```bash
pytest -v tests/test_audit_v4_gap_vegas.py
pytest -v   # full suite
for f in [the new .md files] [the new .py files]; do
    python -c "open('$f').read().encode('ascii')" || echo "$f FAIL"
done
```

**Phase 1 exit criterion:** audit run with anchors passing, weak-spot list written, TODO updated, all tests green, ASCII clean.

---

## Risks (carried from spec)

1. **Vegas team-name resolution gaps in older seasons.** Reuse existing fuzzy matcher; report per-season coverage. Findings note's `join_coverage` makes any hole visible.
2. **Date-to-Season alignment errors.** Sanity-check via `MSeasons.csv` `DayZero`. Spot-check a few known games.
3. **Spread-to-prob convention (SIGMA=11).** Existing-codebase choice; sensitivity sweep is a follow-up.
4. **Bucket multiplicity / cherry-picking.** Findings note must report all cells in JSON; only top-3-5 highlighted as weak spots, with `n >= 50` and `ll_delta >= 0.02` thresholds. No p-hacking.

## Out-of-scope follow-ups (from spec)

- 538 tournament-forecast audit (next branch, sourcing-first).
- Sensitivity sweep over SIGMA.
- Per-team Vegas-vs-v4 outliers.
- Acting on weak-spot findings (each is its own engineering experiment).
