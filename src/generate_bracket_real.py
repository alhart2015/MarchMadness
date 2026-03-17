"""Generate the 2026 March Madness bracket predictions from the ACTUAL bracket PDF.

Pipeline
--------
1. Load data and train the prediction model (skips Optuna, uses default params).
2. Load the actual 2026 bracket from data/raw/bracket_2026.csv.
   First Four slots are pre-resolved using KADJ EM (higher rating advances):
     South  16-seed : Lehigh       (KADJ EM -10.37 > Prairie View A&M -10.69)
     Midwest 16-seed: UMBC         (KADJ EM  -1.67 > Howard           -3.19)
     West   11-seed : North Carolina St. (KADJ EM 19.60 > Texas       19.03)
     Midwest 11-seed: SMU          (KADJ EM  18.09 > Miami OH          8.26)
3. Pre-compute all pairwise win probabilities (64x63 pairs) in a single batch.
4. Run 10,000-iteration Monte Carlo simulation using the lookup table.
5. Print advancement probabilities, chalk + EV bracket picks, and export CSV.

Usage
-----
    python src/generate_bracket_real.py
or
    python -m src.generate_bracket_real
"""

import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

# ── Logging / path setup ──────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

_HERE = Path(__file__).resolve().parent   # src/
_ROOT = _HERE.parent                       # project root
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

KAGGLE_DIR   = _ROOT / "data" / "raw" / "kaggle"
BRACKET_CSV  = _ROOT / "data" / "raw" / "bracket_2026.csv"
OUTPUT_DIR   = _ROOT / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Region / bracket constants ────────────────────────────────────────────────
REGIONS = ["East", "Midwest", "South", "West"]   # sorted alphabetically for simulation

# Standard NCAA first-round seed pairings within a region
FIRST_ROUND_PAIRS = [(1, 16), (8, 9), (5, 12), (4, 13), (6, 11), (3, 14), (7, 10), (2, 15)]

ROUND_NAMES = {1: "R64", 2: "R32", 3: "S16", 4: "E8", 5: "F4", 6: "Champ"}


# ── Bracket loading ───────────────────────────────────────────────────────────

def load_actual_bracket(bracket_csv: Path) -> pd.DataFrame:
    """Load the actual 2026 bracket from CSV.

    Returns DataFrame with columns: Region, Seed (int), TeamID (int), TeamName.
    """
    df = pd.read_csv(bracket_csv)
    required = {"Region", "Seed", "TeamID", "TeamName"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Bracket CSV missing columns: {missing}")
    df["Seed"] = df["Seed"].astype(int)
    df["TeamID"] = df["TeamID"].astype(int)
    assert len(df) == 64, f"Expected 64 teams in bracket CSV, got {len(df)}"
    return df.reset_index(drop=True)


# ── Pairwise probability lookup ───────────────────────────────────────────────

def precompute_win_probs(
    bracket: pd.DataFrame,
    feature_matrix: pd.DataFrame,
    feature_cols: list,
    model,
) -> Dict[tuple, float]:
    """Pre-compute P(team_a beats team_b) for all pairs in the bracket in one batch.

    Returns dict mapping (team_a_id, team_b_id) -> probability.
    """
    team_ids = bracket["TeamID"].tolist()
    n = len(team_ids)

    # Build feature lookup: team_id -> feature array
    feat_lookup: Dict[int, np.ndarray] = {}
    for tid in team_ids:
        row = feature_matrix[feature_matrix["TeamID"] == tid]
        if row.empty:
            raise ValueError(f"TeamID {tid} not found in feature matrix")
        feat_lookup[tid] = row.iloc[0][feature_cols].values.astype(float)

    # Build all ordered pairs (a, b) where a != b
    pairs = [(team_ids[i], team_ids[j]) for i in range(n) for j in range(n) if i != j]

    # Build batch feature matrix: each row is feat(a) - feat(b)
    rows = [feat_lookup[a] - feat_lookup[b] for a, b in pairs]
    X_batch = pd.DataFrame(rows, columns=feature_cols)
    X_batch = X_batch.fillna(0.0)

    # Single batch prediction
    proba_batch = model.predict_proba(X_batch)[:, 1]  # P(class=1) = P(a wins)

    win_prob: Dict[tuple, float] = {}
    for (a, b), p in zip(pairs, proba_batch):
        win_prob[(a, b)] = float(p)

    return win_prob


# ── Fast Monte Carlo simulation ───────────────────────────────────────────────

def simulate_tournament_fast(
    bracket: pd.DataFrame,
    win_prob: Dict[tuple, float],
    n_simulations: int = 10_000,
    random_seed: int = 42,
) -> dict:
    """Fast Monte Carlo using pre-computed win probability lookup.

    Returns dict with advancement_counts, champions, n_simulations.
    """
    rng = np.random.default_rng(random_seed)
    regions = sorted(bracket["Region"].unique())

    # Build region -> {seed: team_id}
    region_teams: Dict[str, Dict[int, int]] = {}
    for region in regions:
        rdf = bracket[bracket["Region"] == region]
        region_teams[region] = dict(zip(rdf["Seed"].astype(int), rdf["TeamID"].astype(int)))

    advancement: Dict[int, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
    champions: Dict[int, int] = defaultdict(int)

    def play_game(a: int, b: int) -> int:
        p = win_prob.get((a, b), 0.5)
        return a if rng.random() < p else b

    def simulate_region(seed_to_tid: Dict[int, int]) -> int:
        """Simulate 4 rounds within a region, return regional champion."""
        # Round 1
        round_winners = []
        for s_a, s_b in FIRST_ROUND_PAIRS:
            a = seed_to_tid[s_a]
            b = seed_to_tid[s_b]
            w = play_game(a, b)
            round_winners.append(w)
            advancement[w][1] += 1

        # Rounds 2-4
        current = round_winners
        for rnd in range(2, 5):
            next_round = []
            for i in range(0, len(current), 2):
                w = play_game(current[i], current[i + 1])
                next_round.append(w)
                advancement[w][rnd] += 1
            current = next_round

        return current[0]

    for sim in range(n_simulations):
        regional_champs = []
        for region in regions:
            champ = simulate_region(region_teams[region])
            regional_champs.append(champ)
            advancement[champ][5] += 1

        # Final Four: regions play in fixed bracket pairs
        # Standard: East vs West (or South) and Midwest vs the other
        # Actual NCAA 2026: East vs West in one semifinal, South vs Midwest in the other
        # regions list sorted = [East, Midwest, South, West]
        # idx:                    0      1        2      3
        # East(0) vs West(3), Midwest(1) vs South(2)
        semi1 = play_game(regional_champs[0], regional_champs[3])  # East vs West
        semi2 = play_game(regional_champs[1], regional_champs[2])  # Midwest vs South
        advancement[semi1][5] += 0   # already counted from regional win
        advancement[semi2][5] += 0
        champion = play_game(semi1, semi2)
        advancement[champion][6] += 1
        champions[champion] += 1

        if (sim + 1) % 2000 == 0:
            print(f"    ...{sim + 1:,} / {n_simulations:,} simulations done")

    return {
        "advancement_counts": dict(advancement),
        "champions": dict(champions),
        "n_simulations": n_simulations,
    }


def get_advancement_probabilities(
    advancement_counts: dict,
    n_simulations: int,
) -> dict:
    return {
        team_id: {r: count / n_simulations for r, count in rounds.items()}
        for team_id, rounds in advancement_counts.items()
    }


# ── Bracket display helpers ───────────────────────────────────────────────────

def print_bracket_picks(
    picks: dict,
    bracket: pd.DataFrame,
    strategy_name: str,
) -> None:
    """Print a human-readable bracket for a given strategy."""
    id_to_name   = dict(zip(bracket["TeamID"], bracket["TeamName"]))
    id_to_seed   = dict(zip(bracket["TeamID"], bracket["Seed"]))
    id_to_region = dict(zip(bracket["TeamID"], bracket["Region"]))

    round_labels = {
        1: "Round of 64   (R64)",
        2: "Round of 32   (R32)",
        3: "Sweet 16      (S16)",
        4: "Elite Eight   (E8 )",
        5: "Final Four    (F4 )",
        6: "CHAMPION",
    }

    print(f"\n{'=' * 70}")
    print(f"  BRACKET: {strategy_name}")
    print(f"{'=' * 70}")

    for rnd in range(1, 7):
        teams = picks.get(rnd, [])
        print(f"\n  {round_labels[rnd]}")
        print(f"  {'-' * 52}")
        for tid in teams:
            name   = id_to_name.get(tid, str(tid))
            seed   = id_to_seed.get(tid, "?")
            region = id_to_region.get(tid, "?")
            print(f"    ({seed:>2}) {name:<30}  [{region}]")
    print()


def print_advancement_table(
    advancement_probs: dict,
    bracket: pd.DataFrame,
    top_n: int = 30,
) -> None:
    id_to_name   = dict(zip(bracket["TeamID"], bracket["TeamName"]))
    id_to_seed   = dict(zip(bracket["TeamID"], bracket["Seed"]))
    id_to_region = dict(zip(bracket["TeamID"], bracket["Region"]))

    sorted_teams = sorted(
        advancement_probs.items(),
        key=lambda x: x[1].get(6, 0),
        reverse=True,
    )

    header = (
        f"{'#':>3}  {'Seed':>4}  {'Team':<26}  {'Region':<10}  "
        f"{'R64':>6}  {'R32':>6}  {'S16':>6}  {'E8':>6}  {'F4':>6}  {'Champ':>7}"
    )
    print(header)
    print("-" * len(header))

    for rank, (tid, probs) in enumerate(sorted_teams[:top_n], start=1):
        name   = id_to_name.get(tid, str(tid))
        seed   = id_to_seed.get(tid, "?")
        region = id_to_region.get(tid, "?")
        print(
            f"{rank:>3}  {seed:>4}  {name:<26}  {region:<10}  "
            f"{probs.get(1,0):>6.1%}  {probs.get(2,0):>6.1%}  "
            f"{probs.get(3,0):>6.1%}  {probs.get(4,0):>6.1%}  "
            f"{probs.get(5,0):>6.1%}  {probs.get(6,0):>7.2%}"
        )


def print_champion_probs(
    advancement_probs: dict,
    bracket: pd.DataFrame,
    top_n: int = 15,
) -> None:
    id_to_name = dict(zip(bracket["TeamID"], bracket["TeamName"]))
    id_to_seed = dict(zip(bracket["TeamID"], bracket["Seed"]))

    sorted_teams = sorted(
        advancement_probs.items(),
        key=lambda x: x[1].get(6, 0),
        reverse=True,
    )

    print(f"\n{'Rank':>4}  {'Seed':>4}  {'Team':<28}  {'Champ%':>8}  {'F4%':>7}  {'E8%':>7}")
    print("-" * 65)
    for rank, (tid, probs) in enumerate(sorted_teams[:top_n], start=1):
        name = id_to_name.get(tid, str(tid))
        seed = id_to_seed.get(tid, "?")
        print(
            f"{rank:>4}  {seed:>4}  {name:<28}  "
            f"{probs.get(6,0):>8.2%}  {probs.get(5,0):>7.2%}  {probs.get(4,0):>7.2%}"
        )


def print_full_bracket_by_region(bracket: pd.DataFrame) -> None:
    """Print the loaded bracket grouped by region for verification."""
    print()
    for region in ["East", "West", "South", "Midwest"]:
        rdf = bracket[bracket["Region"] == region].sort_values("Seed")
        print(f"  {region.upper()} REGION")
        print(f"  {'-' * 50}")
        for _, row in rdf.iterrows():
            print(f"    Seed {int(row['Seed']):>2}  {row['TeamName']:<28}  (ID={int(row['TeamID'])})")
        print()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:

    # ──────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("STEP 1 — Loading raw data")
    print("=" * 70)

    from src.ingest.kaggle2026_loader import load_kaggle2026_data
    data = load_kaggle2026_data(str(KAGGLE_DIR))

    kenpom      = data["kenpom"]
    matchups    = data["matchups"]
    resumes     = data["resumes"]
    ratings_538 = data["ratings_538"]

    print(f"  KenPom rows  : {len(kenpom):,}  (seasons {kenpom['YEAR'].min()}-{kenpom['YEAR'].max()})")
    print(f"  Matchup rows : {len(matchups):,}")
    print(f"  Resume rows  : {len(resumes):,}")
    print(f"  538 rows     : {len(ratings_538):,}")

    # ──────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("STEP 2 — Building tournament results")
    print("=" * 70)

    from src.ingest.build_tournament_results import build_tournament_results
    tourney_results = build_tournament_results(matchups)

    print(f"  Games parsed : {len(tourney_results):,}")
    print(f"  Seasons      : {sorted(tourney_results['Season'].unique())}")

    # ──────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("STEP 3 — Building feature matrix")
    print("=" * 70)

    from src.features.feature_matrix_v2 import build_feature_matrix_v2, get_feature_cols
    feature_matrix = build_feature_matrix_v2(kenpom, resumes, ratings_538)

    feature_cols = get_feature_cols(feature_matrix)
    print(f"  Feature matrix : {len(feature_matrix):,} rows x {len(feature_matrix.columns)} cols")
    print(f"  Feature columns: {len(feature_cols)}")
    print(f"  Seasons        : {int(feature_matrix['Season'].min())}-{int(feature_matrix['Season'].max())}")

    # ──────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("STEP 4 — Building matchup training data")
    print("=" * 70)

    from src.models.matchup import build_matchup_data

    training_seasons = sorted(tourney_results["Season"].unique())
    tourney_fm = feature_matrix[feature_matrix["Season"].isin(training_seasons)]

    # Drop columns with >20% NaN in training matchups
    nan_thresh = 0.20
    tmp_X, _ = build_matchup_data(tourney_fm, tourney_results, feature_cols)
    if not tmp_X.empty:
        null_fracs = tmp_X.isna().mean()
        drop_cols = null_fracs[null_fracs > nan_thresh].index.tolist()
        if drop_cols:
            print(f"  Dropping {len(drop_cols)} high-NaN cols: {drop_cols}")
            feature_cols = [c for c in feature_cols if c not in drop_cols]

    X_all, y_all = build_matchup_data(tourney_fm, tourney_results, feature_cols)
    X_all = X_all.fillna(X_all.median())

    print(f"  Training samples : {len(X_all):,}  (win rate {y_all.mean():.3f})")
    print(f"  Features used    : {len(feature_cols)}")

    # ──────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("STEP 5 — Training model (default XGBoost params, no Optuna)")
    print("=" * 70)

    from src.models.train import train_model
    model = train_model(X_all, y_all, random_seed=42)
    print("  Model trained successfully.")

    # ──────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("STEP 6 — Loading ACTUAL 2026 bracket from CSV")
    print("=" * 70)

    bracket = load_actual_bracket(BRACKET_CSV)

    print(f"  Bracket loaded : {len(bracket)} teams across {bracket['Region'].nunique()} regions")
    print(f"  Source         : {BRACKET_CSV}")
    print()
    print("  First Four resolutions (higher KADJ EM advances):")
    print("    South  16-seed: Lehigh       (KADJ EM -10.37 vs Prairie View A&M -10.69)")
    print("    Midwest 16-seed: UMBC        (KADJ EM  -1.67 vs Howard           -3.19)")
    print("    West   11-seed: NC State     (KADJ EM  19.60 vs Texas            19.03)")
    print("    Midwest 11-seed: SMU         (KADJ EM  18.09 vs Miami OH          8.26)")

    print_full_bracket_by_region(bracket)

    # ──────────────────────────────────────────────────────────────────────────
    print("=" * 70)
    print("STEP 7 — Preparing feature matrix for simulation")
    print("=" * 70)

    fm_2026 = feature_matrix[feature_matrix["Season"] == 2026].copy()
    sim_team_ids = set(bracket["TeamID"].tolist())
    fm_2026_tourney = fm_2026[fm_2026["TeamID"].isin(sim_team_ids)].copy()

    print(f"  2026 teams in feature matrix : {len(fm_2026_tourney)}")

    # Fill NaN with per-column median from training data
    for col in feature_cols:
        if col in fm_2026_tourney.columns:
            col_median = tourney_fm[col].median() if col in tourney_fm.columns else 0.0
            fm_2026_tourney[col] = fm_2026_tourney[col].fillna(col_median)

    missing_ids = sim_team_ids - set(fm_2026_tourney["TeamID"].tolist())
    if missing_ids:
        id_to_name = dict(zip(bracket["TeamID"], bracket["TeamName"]))
        print(f"  WARNING: {len(missing_ids)} teams missing from feature matrix: "
              f"{[id_to_name.get(i, str(i)) for i in missing_ids]}")
    else:
        print("  All 64 bracket teams found in feature matrix.")

    # ──────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("STEP 8 — Pre-computing pairwise win probabilities")
    print("=" * 70)

    print(f"  Computing {len(bracket)} x {len(bracket)-1} = {len(bracket)*(len(bracket)-1):,} pair probabilities...")
    win_prob = precompute_win_probs(bracket, fm_2026_tourney, feature_cols, model)
    print(f"  Done. Lookup table has {len(win_prob):,} entries.")

    # ──────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("STEP 9 — Monte Carlo simulation (10,000 iterations)")
    print("=" * 70)

    print("  Running simulation...")
    sim_results = simulate_tournament_fast(
        bracket=bracket,
        win_prob=win_prob,
        n_simulations=10_000,
        random_seed=42,
    )
    print("  Simulation complete.")

    advancement_probs = get_advancement_probabilities(
        sim_results["advancement_counts"],
        sim_results["n_simulations"],
    )

    # ──────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("RESULTS — Championship Probabilities (Top 15)")
    print("=" * 70)

    print_champion_probs(advancement_probs, bracket, top_n=15)

    # ──────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("RESULTS — Advancement Probabilities (Top 30)")
    print("=" * 70)

    print_advancement_table(advancement_probs, bracket, top_n=30)

    # ──────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("RESULTS — Bracket Picks")
    print("=" * 70)

    from src.bracket.strategies import chalk_bracket, expected_value_bracket

    chalk_picks = chalk_bracket(bracket, advancement_probs)
    ev_picks    = expected_value_bracket(bracket, advancement_probs)

    print_bracket_picks(chalk_picks, bracket, "CHALK  (highest probability each round)")
    print_bracket_picks(ev_picks,    bracket, "EXPECTED VALUE  (maximize points per slot)")

    # ──────────────────────────────────────────────────────────────────────────
    print("=" * 70)
    print("RESULTS — Champion summary")
    print("=" * 70)

    id_to_name = dict(zip(bracket["TeamID"], bracket["TeamName"]))
    id_to_seed = dict(zip(bracket["TeamID"], bracket["Seed"]))

    chalk_champ = chalk_picks[6][0]
    ev_champ    = ev_picks[6][0]
    top_mc      = sorted(advancement_probs.items(), key=lambda x: x[1].get(6, 0), reverse=True)[0]

    print(f"\n  Chalk champion      : ({id_to_seed[chalk_champ]:>2}) {id_to_name[chalk_champ]:<28}"
          f"  {advancement_probs[chalk_champ].get(6,0):.2%} championship prob")
    print(f"  EV champion         : ({id_to_seed[ev_champ]:>2}) {id_to_name[ev_champ]:<28}"
          f"  {advancement_probs[ev_champ].get(6,0):.2%} championship prob")
    print(f"  MC most likely champ: ({id_to_seed[top_mc[0]]:>2}) {id_to_name[top_mc[0]]:<28}"
          f"  {top_mc[1].get(6,0):.2%} championship prob")

    # ──────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("STEP 10 — Exporting to CSV")
    print("=" * 70)

    from src.bracket.output import export_bracket_csv
    csv_path = str(OUTPUT_DIR / "bracket_2026_real.csv")
    export_bracket_csv(advancement_probs, bracket, csv_path)
    print(f"  Saved: {csv_path}")

    bracket_csv_path = str(OUTPUT_DIR / "bracket_2026_real_structure.csv")
    bracket.to_csv(bracket_csv_path, index=False)
    print(f"  Saved: {bracket_csv_path}")

    print("\n" + "=" * 70)
    print("  Done!  2026 March Madness (REAL bracket) predictions complete.")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
