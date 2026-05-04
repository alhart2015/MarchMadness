# Massey + KenPom/Barttorvik Leak Audit -- Findings

**Date:** 2026-05-04
**Branch:** feat/audit-massey-kenpom-leak
**Verdict:** **NO LEAK in either source today.** Massey is clean by file
construction. KenPom Barttorvik mixes pre-tournament rating columns with one
post-tournament label (`ROUND`), but the v3 feature pipeline uses an explicit
17-column allowlist that does not include `ROUND`. Defensive guard added so a
future change to the allowlist cannot silently regress this property.
**Stage 3 of the recovery plan (clean-LOSO regen of `pairwise_v4.csv`) is
unblocked.**
**Context:** Stage 2 of the 5-PR recovery sequence following the Vegas leak
discovery in PR 19. TODO.md frames this as a cheap (~30 min) read-only audit
that only opens a fix PR if a leak is found.

## TL;DR

- **`MMasseyOrdinals.csv`:** max `RankingDayNum` per `Season` is 133
  (Selection Sunday) for every season except 2020 (128, COVID). No rows
  at `RankingDayNum >= 134` (NCAA tournament). The "latest day per season"
  filter in `src/ingest/kaggle_loader.py:42` and `src/enhanced_model.py:88`
  selects pre-tournament rankings by file convention. **Clean.**
- **`KenPom Barttorvik.csv`:** 68 rows per `YEAR` (NCAA field size). The file
  mixes data classes:
  - **Pre-tournament:** `GAMES`, `W`, `L`, all rating/efficiency columns
    (`KADJ EM`, `BADJ EM`, `BARTHAG`, four-factors, etc.). Verified by
    spot-check: 2018 Virginia W=31, L=2 matches their pre-NCAA record;
    2019 Virginia (champion) W=29, L=3 also pre-NCAA; 2024 UConn (champion)
    W=31, L=3 ditto.
  - **Post-tournament:** `ROUND` (1=champion, 2=runner-up, ..., 64=R64 loss,
    68=First Four loss). For 2026 (in-progress), all `ROUND=0`. Textbook
    label leak if used as a feature.
- **Three feature-builder paths use a `kp_cols`-style allowlist; all three
  exclude `ROUND`. No leak today.**
  1. **`compute_all_features` (`src/enhanced_model.py:312`)** -- this is
     what the v3 LOSO training run actually calls
     (`enhanced_model_v3.py:706` imports it from v1). Hardcoded inline
     allowlist of 17 columns (`KADJ EM`, `KADJ O`, `KADJ D`, `BARTHAG`,
     `TALENT`, `EXP`, `ELITE SOS`, `WAB`, `EFG%`, `EFG%D`, `TOV%`,
     `TOV%D`, `OREB%`, `DREB%`, `FTR`, `FTRD`, `K TEMPO`). No `kp_cols`
     kwarg, so a future leak requires editing the source list directly.
     Comment added pointing to this audit note.
  2. **`build_all_team_features` (`src/kaggle_submission.py:312`)** --
     used for Kaggle submission feature assembly. Same 17-column
     allowlist as default. Exposes `kp_cols=None` as a kwarg, so a
     caller could pass a leaky list. Runtime guard added (raises
     `ValueError` if any column in `kp_cols` is in
     `_KP_POST_TOURNAMENT_COLUMNS = {"ROUND"}`) plus 4 unit tests.
  3. **`_KENPOM_FEATURES` (`src/features/feature_matrix_v2.py:42`)** --
     used by the live bracket-generation pipeline
     (`generate_bracket_real.py`, `run_pipeline.py`). 26-column module-
     level constant. Excludes `ROUND` today; **left untouched in this
     PR** because the surrounding file has pre-existing non-ASCII
     box-drawing characters (`─`) and the live-bracket leak class is
     limited (2026 KenPom rows have `ROUND=0` for an in-progress
     season). Recommended followup: ASCII-clean the file in a separate
     PR and add the same defensive comment.
- **`_get_top_n_team_ids` (`src/enhanced_model_v3.py:487`)** filters by
  `KADJ EM RANK`, which is derived from `KADJ EM` -- a pre-tournament
  rating. Not a leak.

## What was checked

### 1. Massey raw data

```
m = pd.read_csv("data/raw/march-machine-learning-2026/MMasseyOrdinals.csv")
m.groupby("Season")["RankingDayNum"].max().describe()
# count    24
# mean    132.79
# min     128       (2020 -- COVID, tournament cancelled)
# max     133       (Selection Sunday)
# Seasons with max >= 134: <none>
```

The Kaggle Mania CSV ships pre-tournament. The `latest_day` filter
(applied identically in two places) is therefore correct.

### 2. KenPom Barttorvik raw data

103 columns, 68 rows per `YEAR` (the NCAA field). Suspicious columns:
`SEED`, `ROUND`, `GAMES`, `W`, `L`, `WAB`. Probed by team-season:

| YEAR | TEAM        | GAMES | W  | L  | SEED | ROUND | KADJ EM |
|------|-------------|-------|----|----|------|-------|---------|
| 2024 | Connecticut | 34    | 31 | 3  | 1    | 1     | 32.213  |
| 2024 | Purdue      | 33    | 29 | 4  | 1    | 2     | 29.118  |
| 2018 | Virginia    | 33    | 31 | 2  | 1    | 64    | 32.153  |
| 2019 | Virginia    | 32    | 29 | 3  | 1    | 1     | 35.655  |
| 2026 | Alabama     | 32    | 23 | 9  | 4    | 0     | 25.720  |

W/L for 2018 Virginia is 31-2, matching their pre-NCAA record (R64 loss
to UMBC made them 31-3). W/L for 2024 UConn is 31-3 -- their pre-NCAA
record (six NCAA wins made them 37-3). `GAMES`, `W`, `L`, and the
ratings derived from them are pre-tournament.

`ROUND` is the post-NCAA elimination round. Distribution per year:
`{1, 2, 4, 8, 16, 32, 64, 68}` for completed seasons; `{0}` for 2026.

`SEED` is set on Selection Sunday (DayNum 132-133) and is pre-tournament
by definition.

### 3. Where these columns are used

```
$ grep -rn '"ROUND"' src/
src/analyze_defense_conference.py:35:  df = df[df["ROUND"].isin(ROUND_TO_WINS)].copy()
src/analyze_defense_conference.py:36:  df["rounds_won"] = df["ROUND"].map(ROUND_TO_WINS)
```

Only an analysis script uses `ROUND`. **No feature pipeline references
it.** The two feature-building call sites (`enhanced_model.py:315` and
`kaggle_submission.py:324`) both define an explicit `kp_cols` allowlist
that excludes `ROUND` (and `GAMES`, `W`, `L`, `SEED`).

## What this means for the recovery plan

- **Stage 2 closes with no fix PR needed** for the leak class itself.
- **Stage 3 (regenerate `output/pairwise_v4.csv` via clean LOSO)** is
  unblocked. The PR 19 Vegas filter is the only behavior change since
  the contaminated training run; Massey + KenPom flow into v4 unchanged.
- **The "what's NOT contaminated" list in TODO.md stands.** Within-
  season redundancy diagnostics (Massey-vs-adj_em correlation 0.957,
  Colley-vs-massey_composite 0.948, etc.) are not affected -- those
  inputs were already pre-tournament.

## Defensive guard added

Even though the audit found no leak, the proximity is uncomfortable: the
KenPom Barttorvik file ships a post-tournament label one column away from
the rating columns we feed into the model. A future contributor adding,
say, `WIN%` to `kp_cols` would not introduce a leak (it's pre-tournament),
but adding `ROUND` would silently break LOSO. To make the convention
machine-checkable rather than tribal:

```python
# src/kaggle_submission.py: build_all_team_features
_KP_POST_TOURNAMENT_COLUMNS = frozenset({"ROUND"})

def build_all_team_features(...):
    if kp_cols is None:
        kp_cols = [...]
    leaked = set(kp_cols) & _KP_POST_TOURNAMENT_COLUMNS
    if leaked:
        raise ValueError(
            f"kp_cols includes post-tournament leak columns: {sorted(leaked)}. "
            f"These are populated AFTER NCAA games and must not be used as "
            f"features in LOSO CV. See "
            f"docs/notes/2026-05-04-massey-kenpom-leak-audit.md."
        )
```

A unit test asserts the guard fires on `kp_cols=["ROUND"]` and is silent
on the default allowlist.

## Open audit followups (not in this PR)

- **`538 Ratings.csv`, `AP Poll Data.csv`, `Coach Results.csv`,
  `Conference Results.csv` and other files in `data/raw/kaggle/`** were
  not audited here. Active queue item #1 (538 v4 gap audit) will touch
  the 538 file directly; if 538 ratings get added to the feature matrix
  later, that audit needs a similar leak check before wire-in.
- **`data/raw/kaggle/Resumes.csv`** (referenced in
  `feature_matrix_v2.py:_RESUME_FEATURES`) is not currently fed into v3.
  If it gets wired in later, audit `NET_RPI`, `ELO`, `WAB_RANK`, `Q1_W`,
  `Q2_W`, `Q3_Q4_L` for pre-tournament status (selection-committee
  resume metrics are pre-tournament by definition, but the file should
  still be spot-checked).
