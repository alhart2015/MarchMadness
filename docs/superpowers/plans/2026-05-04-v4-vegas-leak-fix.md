# v4 Vegas-feature leakage fix -- Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop v4's per-team-per-season Vegas aggregate features
(`vegas_avg_*`, `vegas_ats_pct`, `vegas_power_rating`,
`vegas_consistency`, `vegas_late_spread_delta`) from including the
holdout season's NCAA tournament games at LOSO test time.

**Architecture:** New helper `filter_vegas_to_pre_tournament(vegas_df,
seasons_csv_path)` that drops rows with `daynum >= 134`. Wire it into
`enhanced_model_v3.py` between `load_vegas_lines()` and
`compute_vegas_features()` / `_build_vegas_team_records_with_dates()`.
Keep the full `vegas_df` available to `_build_r64_lines()` (which
intentionally consumes tournament-game lines for R64 line blending).

**Tech Stack:** Python 3.11, pandas, pytest. ASCII only (Windows cp1252).

**Spec:** `docs/superpowers/specs/2026-05-04-v4-vegas-leak-fix-design.md`

---

## File Structure

| File | Role |
|------|------|
| `src/enhanced_model_v3.py` (modify, ~30 LOC added) | Define `filter_vegas_to_pre_tournament`; replace `vegas_df` with the filtered version at the three feature-computation call sites |
| `tests/test_vegas_leak_filter.py` (new) | Unit + smoke tests for the filter |
| `TODO.md` (modify) | Record contamination + recovery roadmap |

`load_vegas_lines()`, `compute_vegas_features()`, and
`_build_vegas_team_records_with_dates()` all keep their current
signatures.

---

## Task 1: Failing unit test for `filter_vegas_to_pre_tournament`

**Files:**
- Create: `tests/test_vegas_leak_filter.py`

- [ ] **Step 1: Write the failing tests**

```python
"""Unit + smoke tests for filter_vegas_to_pre_tournament.

Spec: docs/superpowers/specs/2026-05-04-v4-vegas-leak-fix-design.md
"""
from pathlib import Path
import pandas as pd
import pytest

from src.enhanced_model_v3 import filter_vegas_to_pre_tournament


def _make_seasons_csv(tmp_path: Path) -> Path:
    """Minimal MSeasons.csv with DayZero=11/01/2024 for season 2025."""
    p = tmp_path / "MSeasons.csv"
    pd.DataFrame({
        "Season": [2024, 2025],
        "DayZero": ["10/30/2023", "11/01/2024"],
        "RegionW": ["W", "W"], "RegionX": ["X", "X"],
        "RegionY": ["Y", "Y"], "RegionZ": ["Z", "Z"],
    }).to_csv(p, index=False)
    return p


def test_drops_tournament_rows(tmp_path):
    """Rows with daynum >= 134 (NCAA tournament First Four onward) are dropped."""
    seasons_csv = _make_seasons_csv(tmp_path)
    # DayZero for season 2025 is 11/01/2024.
    # 2024-11-01 = day 0, 2025-03-15 = day 134 (First Four), 2025-04-07 = day 157.
    df = pd.DataFrame({
        "season": [2025, 2025, 2025, 2025],
        "date": ["11/15/2024", "03/05/2025", "03/14/2025", "04/07/2025"],
        "home": ["A", "B", "C", "D"],
        "road": ["E", "F", "G", "H"],
        "line": [3.0, -2.0, 5.0, 1.0],
        "hscore": [70, 80, 65, 75],
        "rscore": [60, 70, 75, 70],
        "neutral": [0, 0, 1, 1],
    })
    out = filter_vegas_to_pre_tournament(df, seasons_csv_path=seasons_csv)
    # daynums: 14, 124, 133, 157. Drops the row with daynum=157 (April 7).
    # 03/14/2025 = day 133 = stays. 03/15/2025 would be day 134 = drop.
    assert len(out) == 3
    assert "04/07/2025" not in out["date"].tolist()
    # Schema preserved.
    assert list(out.columns) == list(df.columns)


def test_drops_first_four_day_134(tmp_path):
    """Boundary: daynum == 134 is the First Four day; must be dropped."""
    seasons_csv = _make_seasons_csv(tmp_path)
    # Season 2025 DayZero=11/01/2024. day 134 = 03/15/2025.
    df = pd.DataFrame({
        "season": [2025, 2025],
        "date": ["03/14/2025", "03/15/2025"],  # daynum 133, 134
        "home": ["A", "B"], "road": ["C", "D"],
        "line": [1.0, 2.0],
        "hscore": [70, 70], "rscore": [60, 60], "neutral": [1, 1],
    })
    out = filter_vegas_to_pre_tournament(df, seasons_csv_path=seasons_csv)
    assert len(out) == 1
    assert out["date"].iloc[0] == "03/14/2025"


def test_empty_input(tmp_path):
    """Empty input -> empty output, schema preserved."""
    seasons_csv = _make_seasons_csv(tmp_path)
    df = pd.DataFrame(columns=["season", "date", "home", "road", "line",
                                 "hscore", "rscore", "neutral"])
    out = filter_vegas_to_pre_tournament(df, seasons_csv_path=seasons_csv)
    assert len(out) == 0
    assert list(out.columns) == list(df.columns)


def test_unknown_season_kept_with_warning(tmp_path, capsys):
    """A row whose season has no DayZero entry is kept with a warning."""
    seasons_csv = _make_seasons_csv(tmp_path)
    df = pd.DataFrame({
        "season": [9999],
        "date": ["03/15/2025"],
        "home": ["A"], "road": ["B"], "line": [1.0],
        "hscore": [70], "rscore": [60], "neutral": [1],
    })
    out = filter_vegas_to_pre_tournament(df, seasons_csv_path=seasons_csv)
    assert len(out) == 1
    captured = capsys.readouterr()
    assert "9999" in captured.out or "9999" in captured.err


def test_unparseable_date_kept_with_warning(tmp_path, capsys):
    """A row with an unparseable date is kept (defensive, do not silently drop)."""
    seasons_csv = _make_seasons_csv(tmp_path)
    df = pd.DataFrame({
        "season": [2025],
        "date": ["not-a-date"],
        "home": ["A"], "road": ["B"], "line": [1.0],
        "hscore": [70], "rscore": [60], "neutral": [0],
    })
    out = filter_vegas_to_pre_tournament(df, seasons_csv_path=seasons_csv)
    assert len(out) == 1
    captured = capsys.readouterr()
    assert "unparseable" in captured.out.lower() or "unparseable" in captured.err.lower()


def test_smoke_real_data_2024_uconn():
    """Integration: with real Vegas data, 2024 UConn (TeamID 1163) has
    vegas_avg_margin near +16.16 (regular-season-only) after filter,
    not +18.13 (full season)."""
    from src.enhanced_model_v3 import load_vegas_lines, compute_vegas_features
    DATA = Path("data/raw/march-machine-learning-2026")
    teams = pd.read_csv(DATA / "MTeams.csv")
    spellings = pd.read_csv(DATA / "MTeamSpellings.csv", encoding="latin-1")
    vegas_df = load_vegas_lines()
    vegas_df_filtered = filter_vegas_to_pre_tournament(vegas_df)
    feats, _ = compute_vegas_features(vegas_df_filtered, teams, spellings)
    row = feats[(feats["TeamID"] == 1163) & (feats["Season"] == 2024)]
    assert len(row) == 1, "expected exactly one (UConn 2024) row"
    margin = float(row["vegas_avg_margin"].iloc[0])
    # Regular-season-only is ~+16.16. Filtered should be within 0.10 of that.
    assert 16.0 < margin < 16.30, f"expected ~16.16 reg-only, got {margin:.2f}"
```

- [ ] **Step 2: Run tests, verify they fail**

```bash
python -m pytest tests/test_vegas_leak_filter.py -v
```

Expected: All tests FAIL with `ImportError: cannot import name
'filter_vegas_to_pre_tournament' from 'src.enhanced_model_v3'`.

- [ ] **Step 3: Commit the failing tests**

```bash
git add tests/test_vegas_leak_filter.py
git commit -m "test(v4-vegas-leak-fix): add failing tests for filter_vegas_to_pre_tournament"
```

---

## Task 2: Implement `filter_vegas_to_pre_tournament`

**Files:**
- Modify: `src/enhanced_model_v3.py` (add helper function, place near
  the other Vegas helpers around line 192)

- [ ] **Step 1: Add the helper function**

Insert immediately AFTER `_vegas_file_to_season` (line ~191) and
BEFORE `def load_vegas_lines()` (line ~193):

```python
def filter_vegas_to_pre_tournament(
    vegas_df: pd.DataFrame,
    seasons_csv_path: Path | None = None,
) -> pd.DataFrame:
    """Drop rows whose daynum (date - DayZero[season]) is >= 134.

    134 is the First Four day in Kaggle's DayNum convention. Anything
    from the First Four onward is NCAA tournament and must NOT feed
    into v4's per-team-per-season Vegas aggregates -- otherwise it
    leaks tournament outcomes into LOSO test features.

    Returns a copy with the same schema as `vegas_df`. Rows whose
    season is missing from MSeasons.csv or whose date is unparseable
    are KEPT (defensive: a data hiccup must not silently delete
    legitimate regular-season rows). Both cases emit a warning.

    Spec: docs/superpowers/specs/2026-05-04-v4-vegas-leak-fix-design.md
    """
    if seasons_csv_path is None:
        seasons_csv_path = MANIA_DIR / "MSeasons.csv"

    if vegas_df.empty:
        return vegas_df.copy()

    seasons = pd.read_csv(seasons_csv_path)
    day_zero = {}
    for _, r in seasons.iterrows():
        try:
            day_zero[int(r["Season"])] = datetime.strptime(
                str(r["DayZero"]).strip(), "%m/%d/%Y"
            )
        except (ValueError, TypeError):
            continue

    out_mask = []
    n_unknown_season = 0
    n_unparseable_date = 0
    for season, date_str in zip(vegas_df["season"], vegas_df["date"]):
        dz = day_zero.get(int(season))
        if dz is None:
            n_unknown_season += 1
            out_mask.append(True)  # keep
            continue
        try:
            d = datetime.strptime(str(date_str).strip(), "%m/%d/%Y")
        except (ValueError, TypeError):
            n_unparseable_date += 1
            out_mask.append(True)  # keep (defensive)
            continue
        daynum = (d - dz).days
        out_mask.append(daynum < 134)

    if n_unknown_season:
        unknown_seasons = sorted({int(s) for s in vegas_df["season"]
                                   if int(s) not in day_zero})
        print(f"  warning: {n_unknown_season} Vegas rows have unknown "
              f"DayZero (seasons {unknown_seasons}); keeping them")
    if n_unparseable_date:
        print(f"  warning: {n_unparseable_date} Vegas rows have "
              f"unparseable dates; keeping them")

    out = vegas_df.loc[pd.Series(out_mask, index=vegas_df.index)].copy()
    return out
```

Imports needed at the top of the file (verify they're already there):
- `from datetime import datetime` -- check around line ~20
- `from pathlib import Path` -- check around line ~20

If `datetime` is not yet imported, add `from datetime import datetime`
near the other top-level imports.

- [ ] **Step 2: Run unit tests, verify they pass**

```bash
python -m pytest tests/test_vegas_leak_filter.py -v
```

Expected: All 6 tests PASS.

- [ ] **Step 3: Commit**

```bash
git add src/enhanced_model_v3.py
git commit -m "feat(v4-vegas-leak-fix): add filter_vegas_to_pre_tournament helper"
```

---

## Task 3: Wire the filter into the v3 LOSO + production paths

**Files:**
- Modify: `src/enhanced_model_v3.py` (3 call sites)

- [ ] **Step 1: Filter inside `prepare_loso_inputs()`**

Locate the block around line 648-682. Currently:

```python
vegas_df = load_vegas_lines()
print(f"  Loaded {len(vegas_df):,} Vegas line records across {vegas_df['season'].nunique()} seasons")

vegas_features, name_resolution = compute_vegas_features(
    vegas_df, data["teams"], data["spellings"]
)
...
vegas_team_records = _build_vegas_team_records_with_dates(vegas_df, name_resolution)
```

Change to:

```python
vegas_df = load_vegas_lines()
print(f"  Loaded {len(vegas_df):,} Vegas line records across {vegas_df['season'].nunique()} seasons")

# Filter NCAA tournament games out before computing per-team-per-season
# aggregates. Otherwise season S tournament outcomes leak into season S
# feature rows at LOSO test time. Keep `vegas_df` (full) for the R64 line-
# blending consumer downstream, which intentionally needs tournament lines.
vegas_df_pre_tourney = filter_vegas_to_pre_tournament(vegas_df)
print(f"  Filtered to {len(vegas_df_pre_tourney):,} pre-tournament rows "
      f"({len(vegas_df) - len(vegas_df_pre_tourney):,} tournament rows dropped)")

vegas_features, name_resolution = compute_vegas_features(
    vegas_df_pre_tourney, data["teams"], data["spellings"]
)
...
vegas_team_records = _build_vegas_team_records_with_dates(vegas_df_pre_tourney, name_resolution)
```

The `inputs` dict at line 849 still carries the FULL `vegas_df` (so
`_build_r64_lines()` downstream gets it).

- [ ] **Step 2: Filter inside the women's pipeline**

Locate line 1302-1303:

```python
vegas_df = load_vegas_lines()
vegas_features, _ = compute_vegas_features(vegas_df, m_teams, m_spellings)
```

Change to:

```python
vegas_df = load_vegas_lines()
vegas_df_pre_tourney = filter_vegas_to_pre_tournament(vegas_df)
vegas_features, _ = compute_vegas_features(vegas_df_pre_tourney, m_teams, m_spellings)
```

- [ ] **Step 3: Run smoke + unit tests**

```bash
python -m pytest tests/test_vegas_leak_filter.py -v
```

Expected: All 6 tests still PASS (the smoke test now exercises the
real wiring).

- [ ] **Step 4: Commit**

```bash
git add src/enhanced_model_v3.py
git commit -m "feat(v4-vegas-leak-fix): wire filter into LOSO + women pipeline call sites"
```

---

## Task 4: Quantitative leak check (manual verification)

**Files:**
- Run a one-shot script (no file changes).

- [ ] **Step 1: Reproduce the pre/post leak measurement**

```bash
python << 'EOF'
import sys
sys.path.insert(0, '.')
import pandas as pd
from src.enhanced_model_v3 import (
    load_vegas_lines, compute_vegas_features, filter_vegas_to_pre_tournament,
)
from pathlib import Path

DATA = Path('data/raw/march-machine-learning-2026')
teams = pd.read_csv(DATA / 'MTeams.csv')
spellings = pd.read_csv(DATA / 'MTeamSpellings.csv', encoding='latin-1')
vegas = load_vegas_lines()
vegas_clean = filter_vegas_to_pre_tournament(vegas)
print(f'Full: {len(vegas):,}  Filtered: {len(vegas_clean):,}  Dropped: {len(vegas)-len(vegas_clean):,}')

f_full, _ = compute_vegas_features(vegas, teams, spellings)
f_clean, _ = compute_vegas_features(vegas_clean, teams, spellings)

for season, tid, label in [
    (2024, 1163, '2024 UConn (champs)'),
    (2024, 1345, '2024 Purdue (runner-up)'),
    (2023, 1163, '2023 UConn (champs)'),
    (2018, 1438, '2018 Virginia (lost R64)'),
]:
    a = f_full[(f_full.TeamID==tid) & (f_full.Season==season)]
    b = f_clean[(f_clean.TeamID==tid) & (f_clean.Season==season)]
    if len(a)==0 or len(b)==0:
        print(f'{label}: missing'); continue
    am = float(a['vegas_avg_margin'].iloc[0])
    bm = float(b['vegas_avg_margin'].iloc[0])
    print(f'{label}: full margin={am:+.2f}  clean margin={bm:+.2f}  diff={am-bm:+.2f}')
EOF
```

Expected output (numbers should match the spec's quantified leak
table within ~0.05):

- 2024 UConn: full=+18.13, clean ~+16.16 (delta ~+1.98)
- 2024 Purdue: full=+13.46, clean ~+11.48 (delta ~+1.98)
- 2023 UConn: full=+14.42, clean ~+13.42 (delta ~+1.00)
- 2018 Virginia: full=+11.87, clean ~+12.70 (delta ~-0.83)

If clean values do not match the regular-season-only numbers within
0.10 tolerance, stop -- the filter has a bug. Investigate.

---

## Task 5: Update TODO.md with contamination + recovery roadmap

**Files:**
- Modify: `TODO.md`

- [ ] **Step 1: Add a new top-level section after the front matter**

Insert immediately after the `# Future Work` heading and before the
`## Tried and rejected` section a new section titled
`## CONTAMINATION DISCOVERED 2026-05-04 (active recovery)`. Body:

```markdown
## CONTAMINATION DISCOVERED 2026-05-04 (active recovery)

**TL;DR.** v4's Vegas-derived per-team-per-season features
(`vegas_avg_*`, `vegas_ats_pct`, `vegas_power_rating`,
`vegas_consistency`, `vegas_late_spread_delta`) were computed over
the full Vegas dataset INCLUDING NCAA tournament games. In LOSO CV,
this leaks the holdout season's tournament outcomes into the test
feature row for season S. v4's reported LOSO accuracy of ~80.4% per-
season and the PR 18 finding "v4 beats Vegas everywhere" cannot be
trusted at face value. Falsified by the user's actual Kaggle finish
of 2159 / 3462 -- a model that genuinely beats Vegas in every
bucket does not finish in the bottom half of a real prediction
contest. Discovery thread: 2026-05-04 chat investigation following
the PR 18 merge. Quantified leak: 2024 UConn vegas_avg_margin
+1.98 above regular-season-only; 2024 Purdue +1.98; 2018 Virginia
-0.83. Leak correlates with tournament success.

### Recovery plan (5 PRs, in order)

1. **Filter the leak.** PR `feat/v4-vegas-leak-fix`: add
   `filter_vegas_to_pre_tournament()` and wire it before
   `compute_vegas_features` and `_build_vegas_team_records_with_dates`.
   No regen, no eval changes. Spec:
   `docs/superpowers/specs/2026-05-04-v4-vegas-leak-fix-design.md`.

2. **Audit Massey + KenPom inputs for the same class of leak.**
   `data["massey"]` and the KenPom snapshots are loaded at
   `load_all_data()`. If either is end-of-season-INCLUDING-tournament
   ranking, same fix pattern applies. Cheap (~30 min) read-only audit;
   only opens a fix PR if a leak is found.

3. **Regenerate `output/pairwise_v4.csv` via clean LOSO.** Run
   `enhanced_model_v3.py` end-to-end with the fixed feature pipeline.
   Capture per-season LL + accuracy. Compare to current numbers in
   `output/cv_per_season_v3.csv`. Document the shift.

4. **Re-run the v4-vs-Vegas audit.** `python src/audit_v4_gap_vegas.py`
   against the regenerated `pairwise_v4.csv`. Update findings note
   `docs/notes/2026-05-04-v4-gap-audit-vegas.md` with the corrected
   numbers and retract the "no weak spots" verdict if appropriate.
   The 538 audit (currently active queue #1) stays queued.

5. **Re-run the swap-decided / swap-candidate evaluations against
   the clean baseline.** Priority order:
   - **v9-C production swap** (currently deployed -- top priority).
   - **v8 vs v9-C** bracket-points head-to-head.
   - **Plain BT bracket-points** (PR 17 finding).
   - The "marginal" rejections in `Tried and rejected` whose deltas
     were within ~0.05 LL or ~30 brkt pts of v4 (BT-as-feature at
     -0.0015 LL; v9 weight-sweep family at +18 to +20 pts).
   Big-magnitude rejections (-93 quality wins, -105 LR ensemble,
   +0.0057 Massey-decay clause-2 fail, etc.) do not need re-eval --
   a baseline shift of 0.02-0.05 LL won't flip them.

### What's NOT contaminated

- Diagnostics computed within a season across teams (e.g. Massey-
  vs-adj_em correlation = 0.957) -- the leak shifts both sides
  similarly within a season; redundancy verdicts stand.
- Anchor-equality checks (e.g. "weights (1.0, 0.0) reproduces
  pairwise_v4 byte-equal") -- these test plumbing, not signal.
- Selection of non-v4 models against absolute thresholds (e.g.
  Plain BT standalone LL=0.565 vs v4's 0.437 -- the gap survives
  any plausible shift in v4).
- The PR 18 audit's *framework* (per-bucket LL/acc/ECE), only the
  numerical verdict.
```

- [ ] **Step 2: Commit**

```bash
git add TODO.md
git commit -m "docs(v4-vegas-leak-fix): record contamination + 5-PR recovery roadmap in TODO.md"
```

---

## Task 6: Final pytest sweep

- [ ] **Step 1: Run the full test suite**

```bash
python -m pytest -v 2>&1 | tail -30
```

Expected: all tests pass. Special attention to:
- `tests/test_vegas_leak_filter.py` (new) -- 6 tests pass.
- `tests/test_audit_v4_gap_vegas.py` (existing) -- 10 tests pass
  unchanged (uses `load_vegas_lines()` directly, not affected by the
  filter wiring).
- `tests/test_features/`, `tests/test_ingest/`,
  `tests/test_integration.py` -- per CLAUDE.md mandatory subset for
  ingest/feature-touching changes.

If anything fails, stop and investigate.

- [ ] **Step 2: Push branch and open PR**

```bash
git push -u origin feat/v4-vegas-leak-fix
gh pr create --title "feat(v4-vegas-leak-fix): drop NCAA tournament games from Vegas feature aggregates" --body "$(cat <<'EOF'
## Summary

- Adds `filter_vegas_to_pre_tournament()` (drops rows with daynum >= 134).
- Wires it in before `compute_vegas_features` and `_build_vegas_team_records_with_dates` in both the men's LOSO + production path and the women's pipeline. `load_vegas_lines()` and `_build_r64_lines()` are unchanged -- the audit and R64 line-blending consumers still see the full data.
- Records the contamination discovery + 5-PR recovery roadmap in `TODO.md`.

## Why

PR 18's audit reported v4 beats Vegas everywhere (delta -0.114 LL, +10.3pp accuracy). Investigation showed `compute_vegas_features` aggregates over ALL Vegas-line games per (TeamID, Season) -- including NCAA tournament games. In LOSO CV, this leaks the holdout season's tournament outcomes into season-S feature rows at test time. Direct measurement: 2024 UConn vegas_avg_margin is +1.98 above regular-season-only; the leak correlates with tournament success. Falsified by the user's actual Kaggle finish of 2159 / 3462. This PR fixes the feature pipeline; the regen of `pairwise_v4.csv` and re-audit are subsequent PRs in the recovery sequence (see TODO.md).

## Test plan

- [x] 6 unit + smoke tests in `tests/test_vegas_leak_filter.py` green
- [x] `tests/test_audit_v4_gap_vegas.py` (10 tests) unchanged-and-green
- [x] `pytest -v tests/test_features tests/test_ingest tests/test_integration.py` green
- [x] Quantitative leak check: 2024 UConn / Purdue / 2023 UConn / 2018 Virginia all show clean vegas_avg_margin matching regular-season-only within 0.05

## Out of scope (next PRs in the recovery sequence)

- Massey + KenPom leak audit
- Regenerate `output/pairwise_v4.csv`
- Re-run `audit_v4_gap_vegas.py`
- Re-run swap-candidate evaluations (v9-C production, v8 vs v9-C, plain BT bracket points)

Spec: `docs/superpowers/specs/2026-05-04-v4-vegas-leak-fix-design.md`
Plan: `docs/superpowers/plans/2026-05-04-v4-vegas-leak-fix.md`
EOF
)"
```

---

## Self-review checklist

- [x] Spec coverage: filter helper (T2), wiring (T3), unit tests (T1),
  smoke verification (T4), TODO.md recovery (T5), full pytest (T6).
  Massey/KenPom audit + regen explicitly out of scope (next PRs).
- [x] No placeholders / TBDs / "fill in" / stub-only steps.
- [x] Type / signature consistency: `filter_vegas_to_pre_tournament(
  vegas_df: pd.DataFrame, seasons_csv_path: Path | None = None) ->
  pd.DataFrame` is the same shape in T1 (test imports), T2
  (definition), T3 (callers).
- [x] ASCII-only -- spot checked for em dashes, smart quotes, arrows.
- [x] No leakage between tasks: T1 alone fails, T2 makes T1 pass, T3
  makes the smoke test in T1 pass with real data, T4 is a manual
  one-shot, T5 is docs-only, T6 is the verification gate.
