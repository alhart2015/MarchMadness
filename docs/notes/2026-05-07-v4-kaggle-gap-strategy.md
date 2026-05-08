# v4 Kaggle Gap -- Strategy & Open Questions

**Date:** 2026-05-07
**Context:** Strategy note synthesizing the audits (Vegas, 538), the
per-season variance check, and the Kaggle finish (2159 / 3462) into a
coherent picture of v4's gaps. Captures conclusions reached in a
2026-05-07 conversation that were at risk of being glossed over.
**Status:** Reference document. Future sessions should read this before
starting any "new feature for v4" work.

---

## The motivating fact

**Kaggle 2026 March Madness: 2159 / 3462.** Bottom 38th percentile.
2/3 of competitors using the same base data beat v4. This is the
empirical thing we are trying to fix.

It is NOT a "v4 is broken" finding -- v4's clean LOSO log loss (0.5588)
is competitive against multiple public benchmarks, and on the
22-season aggregate it does well. It IS a "v4 has structural ceilings"
finding -- something about the model class, the feature stack, or the
external information available to other competitors but not to us
explains the gap.

## What the audits have established (and ruled out)

Four audits + one diagnostic have been run against clean v4. The
collective finding is that **v4 is competitive on log loss but has
real ceilings on bracket-points scoring, and the ceilings are partly
information-related and partly model-class-related**.

### Ceilings we've measured

1. **No upset-detection edge.** v4 catches **15.3%** of tournament
   upsets; Vegas catches **17.5%**; 538 catches **16.9%**. Bracket
   scoring rewards correctly-predicted upsets disproportionately, so
   this is the largest single explainer of "competitive on LL,
   mid-pack on bracket points." Source: `docs/notes/2026-05-04-v4-gap-audit-vegas.md`,
   `docs/notes/2026-05-04-v4-gap-audit-fte.md`.

2. **Stage-2 marginal value is tiny.** v9-C (the production stage-2)
   adds **+43 bracket points over 22 seasons** vs v8 = **~+2 pts/year**.
   The v4 stage-1 carries the load; we have no way to lift it on its
   own data. Source: `docs/notes/2026-05-01-v9c-feature-stripped.md`.

3. **Same-data peers all fail.** Four model classes tested as
   blend partners against clean v4: plain BT, hierarchical BT,
   feature-view ensemble, Colley-as-feature. **All four failed
   clause 2 of the LL-blend gate** (residual correlation > 0.6
   against v4). The pattern is so consistent across model classes
   that we now treat "same-data peer" as a closed lane. Source:
   `docs/notes/2026-05-04-v9c-clean-rerun.md` and the four
   per-experiment notes (PRs 24-27).

4. **Calibration shape varies year to year.** Per-season variance
   check (PR 30): ECE has **~25% CV across 21 seasons** (mean 0.134,
   std 0.034). 2011 is a 2.2-sigma outlier on `ll_v4`; 2015 and 2023
   are 2-sigma ECE outliers. From the inside, you can't predict which
   year v4 will be well-calibrated -- so even with perfect 22-season
   aggregate calibration, single-tournament Kaggle scoring is exposed
   to per-season variance with no in-distribution hedge. Source:
   `docs/notes/2026-05-07-v4-per-season-variance.md`.

5. **v4 hedges chalk.** 538 audit (n=298 chalk-favored matchups
   where the chalk pick won): v4 LL **0.322** vs 538 LL **0.247**,
   delta **+0.0754**. Both models pick the same favorite (mean
   prob delta +0.001), but 538 commits more probability mass to that
   favorite. Bracket scoring rewards confidence on calls you got
   right; v4 leaves chalk-pick probability on the floor. Source:
   `docs/notes/2026-05-04-v4-gap-audit-fte.md`.

### Things we've ruled out

- **Single-bad-season hypothesis.** 2024 (the year of the user's
  Kaggle finish) was unremarkable in v4's per-season frame: ll_v4
  = 0.591 (only 0.034 above the 21-season mean), not flagged on any
  variance metric, accuracy 71.4% (above mean). The Kaggle finish
  is NOT a 2024 fluke. Per-season variance check, PR 30.
- **Single weak bucket hypothesis.** Vegas surfaced 6 weak spots
  (E8, upsets, S16, mid-seed-gap, 0.80-0.90 confidence band, large-
  seed-gap). 538 surfaced 1 weak spot (chalk picks). The two
  benchmarks find *different* weak spots, implying calibration shape
  is the bottleneck rather than any single bucket. Engineering
  against any one bucket is therefore under-motivated.
- **Single missing model class.** As noted above, four same-data
  peers tried, four failed the same residual-correlation clause.

## Vegas data: what we have, what we can't have

This was the source of confusion in the 2026-05-07 strategy
discussion. The TL;DR is:

| Signal | Available at submission time? | Coverage | In v4 today? |
|--------|---|---|---|
| Regular-season per-game closing lines | yes (after season) | 22 seasons, 1000s of games | YES (post-leak-fix) |
| Tournament R64 per-game closing lines | **YES, leak-free** (set before tipoff, public) | 22 seasons * 32 = ~700 games | **NO** -- not currently a feature |
| Tournament R32+ per-game closing lines | **NO** -- matchups don't exist | -- | **NO, and impossible by construction** |
| Pre-tournament championship futures (per-team) | **YES** (posted before tournament) | 64 teams * 22 seasons = ~1400 entries | **NO** -- needs sourcing |
| Pre-tournament Final-Four reach futures | YES | same | NO |

**The 2026-05-04 leak fix removed `vegas_avg_*`, `vegas_ats_pct`,
`vegas_power_rating`, `vegas_consistency`, `vegas_late_spread_delta`
from v4's feature stack** because those season-aggregate features
were computed over the *full* Vegas dataset including tournament
games -- which leaked tournament outcomes into the per-team-per-season
feature row in LOSO CV. The fix correctly excluded tournament games
from those aggregates. But the fix did NOT add back any of the
*leak-free* tournament-time Vegas signals listed above. **That's
the data gap.**

### Why R32+ closing lines are impossible

In a real Kaggle submission, you submit pairwise probabilities for
all 2016 possible R64+ team-pair matchups, but you submit at the
*start* of the tournament -- before any R64 games have been played.
The R32 game between the winner of (X1 vs X16) and the winner of
(X8 vs X9) doesn't exist as a posted line until both R64 games have
finished. By construction, no sportsbook posts an R32 line for an
unresolved matchup. Same for S16, E8, F4, Champ.

**This means R64 closing lines are the only per-game leak-free Vegas
signal we can use for a tournament-time prediction.** They cover 32
of the 63 scored bracket games (~51% of the bracket-points denominator).

### Why pre-tournament futures could cover the rest

Sportsbooks post:
- **Championship odds** for each of 64 teams (P(team i wins title))
- **Reach-Final-Four odds** for each team
- **Reach-Elite-Eight odds** for each team
- **Win-region odds** (regional champion)

These are *aggregated* probabilities over all paths a team could
take through the bracket. They're posted before R64 tips off and
they're per-team rather than per-pair. From championship odds you
can analytically solve for an implied team-strength rating (under
a tree of pairwise BT matchups consistent with the bracket
structure). That implied rating is usable as a feature for ALL
2016 pairwise predictions -- not just R64.

**Catch:** historical futures data is harder to source than per-game
lines. Per-game lines are in the existing `data/raw/vegas/ncaabb*.csv`
archive (the `load_vegas_lines()` data); futures historical archives
require Vegas Insider, Action Network, or sportsbook API access.

## Data flaws vs structural flaws -- the honest answer

The user's question on 2026-05-07: *are the people who beat me using
better models, or better data?*

**Both, but data is the stronger hypothesis** for these reasons:

1. **Same-data peers all fail.** 4 model classes can't beat clean v4
   on its own 67-feature representation. Whatever signal the leaderboard
   leaders have, it's almost certainly not a clever same-data
   reformulation -- if it were, our peers would have surfaced some
   of it.
2. **Vegas does beat v4 head-to-head on the aggregate.** LL +0.0148,
   accuracy -0.7pp on 22 seasons / 1326 games. Vegas is *external
   information* (closing lines encode market consensus that v4 has
   no access to in its current feature stack). The size of the gap
   gives us a concrete upper bound on what "Vegas-as-feature" could
   buy.
3. **R64-specific gap is +0.012 LL.** From the Vegas audit's
   `by_round`: R64 `ll_v4`=0.5164, `ll_vegas`=0.5045 on 648 games.
   That delta translates into bracket points through v9-C, but the
   exact translation is unknown until tested.
4. **Median Kaggle competitor isn't running a quant fund.** 2/3 of
   3462 entries beat v4. That's *the median*. The median competitor
   plausibly has KenPom (in v4 already), maybe Massey (in v4 already),
   AND probably **at least Vegas closing lines for the R64 games**
   if they're using any external public data at all. Per-game
   closing lines are the cheapest, most-public external signal.

**That said, the structural flaws are also real.** Even with Vegas
closing lines wired in, v4's no-upset-detection ceiling (#1 above)
and low-stage-2-margin (#2) cap the headroom. A perfect-calibration
v4 + Vegas-line composite still won't pick the 11-seed F4 run.

The right framing: **the data gap is fixable; the structural
ceiling is harder**. Closing the data gap will likely move v4 from
2159/3462 toward the median (call it 1500-1700/3462). Closing the
structural ceiling -- if it's closeable at all on public data --
likely requires either (a) an upset-specific external signal we
haven't named yet (roster-level returning experience, injury data,
NIL movement), or (b) a model-class change beyond same-data peers
(genuine non-tabular methods, Bayesian posterior over team-pair
matchups instead of point estimates, etc.).

## Strategic ordering

Per the post-PR-30 TODO:

1. **External Data #1:** Vegas as a feature for v4. Per the
   discussion, this splits into:
   - **#1a (cheap, this PR's scope): R64 closing-line blend at
     apply time.** Hard-override or learned blend of v4 with the R64
     closing line for the 32 R64 pairs in each tournament. 22 seasons
     of training/eval data already in repo (`load_vegas_lines()`).
     ~50-200 LOC. Cheap-falsification gate: 22-season bracket points
     vs canonical v4 + v9-C (2069 baseline).
   - **#1b (more work, deferred): pre-tournament futures-derived
     team-strength as a v4 feature.** Needs futures data sourcing.
     Useful for ALL pair predictions including R32+. Run only if #1a
     wins materially -- the same-data-peer pattern is a real risk
     even for a futures-derived feature.

2. **Calibration-shape engineering** (TODO active queue #2). Backup
   if Vegas-as-feature falsifies. Temperature scaling / isotonic
   regression on a tournament-only validation set, gated by 22-season
   bracket points (not just LL). Addresses the "v4 hedges chalk"
   finding directly.

3. **MLP, full Bayesian BT, roster-level data, etc.** (TODO #3+).
   Large engineering investments for hypothesized small gains.
   Re-prioritize if both #1 and #2 fail to move the needle.

## Open questions / honest unknowns

1. **What does the median Kaggle competitor actually use?** This is
   a guess based on what's most plausible (Vegas R64 lines + KenPom
   + simple stage-2). We don't have data on the actual feature stacks
   of the 1300 entries that beat v4. A different and possibly cheaper
   hypothesis: most of the gap is just *noise* in single-tournament
   bracket scoring, and v4 is genuinely close to median in expectation
   but happened to draw the wrong year. The variance check's ECE-CV
   finding (#4 above) is consistent with this. **If true, the
   2159/3462 finish is ~1.0 sigma below v4's expected finish, not a
   structural deficit.** No way to disambiguate without multiple
   years of Kaggle finishes; one data point is one data point.

2. **Does R64-LL improvement transfer to R64-bracket-points?** v9-C
   was trained on v4's R64 distribution. An R64 override changes the
   stage-1 distribution that stage-2 sees, possibly degrading stage-2
   in subtle ways. The PR 17 BT-bracket-points re-test
   (`docs/notes/2026-05-04-bt-bracket-points.md`) found that LL
   improvements on stage-1 didn't translate cleanly to bracket points
   when v9-C wasn't retrained on the new stage-1 distribution. This
   is the largest single risk on the #1a experiment.

3. **Is there a leak-free way to use R64 lines that doesn't flow
   through the per-team season aggregate features that bit us before?**
   Yes -- the R64 closing line is a per-game signal, attached to
   exactly the 32 R64 games in each tournament's pairwise frame. It
   never enters team-aggregate or season-aggregate features. So the
   PR 19 leak-fix is unaffected.

4. **Does using Vegas R64 lines reintroduce a SIGMA-conversion
   artifact?** The Vegas audit used SIGMA=11 for spread-to-prob
   conversion and noted it "may be too peaky for tournament games."
   If we use R64 lines as a feature/blend, we either (a) commit to
   a SIGMA value, or (b) feed the spread directly as a feature and
   let the blend learn the SIGMA. (b) is cleaner.

5. **Can we use 538 forecasts as a feature instead?** Not really --
   538's archive only covers 2016-2023 (7 seasons), and 2014/2015
   predate the API while 2024/2025 weren't archived. The 7-season
   ceiling is structural unless an alternate source is found. Vegas
   has 22 seasons. For the same engineering cost, Vegas gives 3x
   the training signal.

## Files of record (cumulative for the 2026-05-04..2026-05-07 work)

- `docs/notes/2026-05-04-v4-clean-loso-regen.md` -- clean baseline
  measurement (PR 21)
- `docs/notes/2026-05-04-v4-gap-audit-vegas.md` -- 6 weak spots vs
  Vegas (PR 22 / clean re-run)
- `docs/notes/2026-05-04-v4-gap-audit-fte.md` -- 1 weak spot vs 538
  (PR 29)
- `docs/notes/2026-05-07-v4-per-season-variance.md` -- variance MIXED
  (PR 30)
- `docs/notes/2026-05-04-v9c-clean-rerun.md` -- production stage-2
  reverted to v8 under clean baseline
- `docs/notes/2026-05-05-{plain-bt,feature-view,hbt,colley-massey,
  colley-full-loso}-clean-rerun.md` -- four same-data peer
  experiments closed
- This note: `docs/notes/2026-05-07-v4-kaggle-gap-strategy.md`

## Bottom line

v4 has both data flaws and structural ceilings. The data flaws are
fixable with existing-in-repo Vegas closing-line data; the structural
ceilings are harder but not the immediate target. The next experiment
is the R64 closing-line blend (this PR's scope) -- explicit-spec
proof or falsification of the data hypothesis. Spec:
`docs/superpowers/specs/2026-05-07-v4-r64-line-blend-design.md`.
