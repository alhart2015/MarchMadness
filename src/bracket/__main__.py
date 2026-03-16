"""CLI entry point: python -m src.bracket"""

import logging
from pathlib import Path

import joblib
import pandas as pd

from src.config import load_config
from src.models.train import predict_matchup
from src.bracket.simulator import load_bracket, simulate_tournament, get_advancement_probabilities
from src.bracket.strategies import chalk_bracket, expected_value_bracket
from src.bracket.output import format_advancement_table, export_bracket_csv

logging.basicConfig(level=logging.INFO, format="%(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    config = load_config()
    processed_dir = Path(config["data"]["processed_dir"])
    predict_season = config["seasons"]["predict_season"]

    # Load model
    import glob
    model_files = sorted(glob.glob("models/xgb_*.pkl"))
    if not model_files:
        raise FileNotFoundError("No trained model found in models/. Run 'python -m src.models' first.")
    model_path = model_files[-1]
    meta_path = model_path.replace(".pkl", "_meta.json")

    import json
    with open(meta_path) as f:
        meta = json.load(f)
    feature_cols = meta["feature_cols"]

    model = joblib.load(model_path)
    logger.info("Loaded model: %s (%d features)", model_path, len(feature_cols))

    # Load data
    feature_matrix = pd.read_parquet(processed_dir / "feature_matrix.parquet")
    current_features = feature_matrix[feature_matrix["Season"] == predict_season]
    teams = pd.read_parquet(processed_dir / "teams.parquet")

    # Load bracket
    bracket_path = f"data/raw/bracket_{predict_season}.csv"
    bracket = load_bracket(bracket_path)

    # Validate all bracket teams have features
    missing = set(bracket["TeamID"]) - set(current_features["TeamID"])
    if missing:
        raise ValueError(f"Bracket teams missing from feature matrix: {missing}")

    # Define predict function for simulator
    def predict_fn(a_features: dict, b_features: dict) -> float:
        diff = {col: a_features[col] - b_features[col] for col in feature_cols}
        X = pd.DataFrame([diff])
        return predict_matchup(model, X)

    # Simulate
    logger.info("Running %d simulations", config["model"]["n_simulations"])
    sim_results = simulate_tournament(
        bracket=bracket,
        feature_matrix=current_features,
        predict_fn=predict_fn,
        feature_cols=feature_cols,
        n_simulations=config["model"]["n_simulations"],
        random_seed=config["model"]["random_seed"],
    )

    # Advancement probabilities
    probs = get_advancement_probabilities(sim_results["advancement_counts"], sim_results["n_simulations"])

    # Output
    print("\n" + "=" * 80)
    print("ADVANCEMENT PROBABILITIES")
    print("=" * 80)
    print(format_advancement_table(probs, teams))

    # Champion probabilities
    champ_probs = {tid: count / sim_results["n_simulations"] for tid, count in sim_results["champions"].items()}
    team_map = dict(zip(teams["TeamID"], teams["TeamName"]))
    print("\n" + "=" * 80)
    print("CHAMPION PROBABILITIES")
    print("=" * 80)
    for tid, prob in sorted(champ_probs.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {team_map.get(tid, str(tid)):25s} {prob:.1%}")

    # Strategies
    scoring = config["bracket"]["scoring"]
    for strategy_name in config["bracket"]["strategies"]:
        if strategy_name == "chalk":
            picks = chalk_bracket(bracket, probs)
        elif strategy_name == "expected_value":
            picks = expected_value_bracket(bracket, probs, scoring=scoring)
        else:
            logger.warning("Unknown strategy: %s", strategy_name)
            continue

        print(f"\n{'=' * 80}")
        print(f"BRACKET: {strategy_name.upper()}")
        print(f"{'=' * 80}")
        champion = picks[6][0] if picks.get(6) else None
        if champion:
            print(f"  Champion: {team_map.get(champion, str(champion))}")
        final_four = picks.get(5, [])[:4]
        print(f"  Final Four: {[team_map.get(t, str(t)) for t in final_four]}")

    # Export
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    export_bracket_csv(probs, teams, str(output_dir / f"bracket_{predict_season}.csv"))
    logger.info("Bracket exported to output/bracket_%d.csv", predict_season)


if __name__ == "__main__":
    main()
