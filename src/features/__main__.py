"""CLI entry point: python -m src.features"""

import logging
from pathlib import Path

import pandas as pd

from src.config import load_config
from src.features.efficiency import compute_adjusted_efficiency
from src.features.four_factors import compute_four_factors
from src.features.feature_matrix import build_feature_matrix

logging.basicConfig(level=logging.INFO, format="%(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    config = load_config()
    processed_dir = Path(config["data"]["processed_dir"])

    # Load ingested data
    results = pd.read_parquet(processed_dir / "regular_season.parquet")
    seeds = pd.read_parquet(processed_dir / "seeds.parquet")
    massey = pd.read_parquet(processed_dir / "massey.parquet")
    team_conferences = pd.read_parquet(processed_dir / "team_conferences.parquet")

    start = config["seasons"]["train_start"]
    end = config["seasons"]["predict_season"]
    eff_cfg = config["efficiency"]
    massey_systems = config["massey"]["systems"]

    all_features = []
    for season in range(start, end + 1):
        season_results = results[results["Season"] == season]
        if len(season_results) == 0:
            logger.warning("No data for season %d, skipping", season)
            continue

        logger.info("Computing features for season %d", season)

        efficiency = compute_adjusted_efficiency(
            season_results, season=season,
            iterations=eff_cfg["iterations"],
            hca=eff_cfg["home_court_advantage"],
            half_life_days=eff_cfg["recency_half_life_days"],
            ridge_alpha=eff_cfg["ridge_alpha"],
        )

        ff = compute_four_factors(season_results, season=season)

        season_massey = massey[massey["Season"] == season] if "Season" in massey.columns else massey

        matrix = build_feature_matrix(
            efficiency=efficiency,
            four_factors=ff,
            seeds=seeds,
            massey=season_massey,
            results=results,
            season=season,
            massey_systems=massey_systems,
            team_conferences=team_conferences,
        )
        matrix["Season"] = season
        all_features.append(matrix)

    combined = pd.concat(all_features, ignore_index=True)
    output_path = processed_dir / "feature_matrix.parquet"
    combined.to_parquet(output_path, index=False)
    logger.info("Feature matrix saved: %s (%d rows, %d seasons)", output_path, len(combined), combined["Season"].nunique())


if __name__ == "__main__":
    main()
