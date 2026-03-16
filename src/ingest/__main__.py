"""CLI entry point: python -m src.ingest"""

import logging
from pathlib import Path

from src.config import load_config
from src.ingest.kaggle_loader import load_kaggle_data
from src.ingest.cbbd_loader import load_cbbd_data
from src.ingest.massey_loader import load_massey_composite
from src.ingest.team_mapping import build_team_mapping
from src.ingest.validation import validate_ingested_data

logging.basicConfig(level=logging.INFO, format="%(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    config = load_config()
    kaggle_dir = config["data"]["kaggle_dir"]
    processed_dir = Path(config["data"]["processed_dir"])
    cache_dir = config["data"]["cache_dir"]
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Source 1: Kaggle (required)
    logger.info("Loading Kaggle data from %s", kaggle_dir)
    data = load_kaggle_data(kaggle_dir)

    logger.info("Validating ingested data")
    validate_ingested_data(data)

    # Source 2: cbbd API (optional)
    predict_season = config["seasons"]["predict_season"]
    cbbd_data = load_cbbd_data(season=predict_season, cache_dir=cache_dir)
    if cbbd_data is not None:
        data["cbbd"] = cbbd_data
        # Build team mapping for cbbd names -> Kaggle IDs
        cbbd_mapping = build_team_mapping(
            kaggle_teams=data["teams"],
            external_names=cbbd_data["team"].tolist(),
            overrides_path=config["data"]["team_overrides"],
            auto_threshold=config["matching"]["auto_accept_threshold"],
            review_threshold=config["matching"]["review_threshold"],
        )
        cbbd_data["TeamID"] = cbbd_data["team"].map(cbbd_mapping)
        data["cbbd"] = cbbd_data

    # Source 3: Massey composite (optional)
    massey_composite = load_massey_composite(cache_dir=cache_dir)
    if massey_composite is not None:
        data["massey_composite"] = massey_composite

    # Save processed data as parquet
    for name, df in data.items():
        output_path = processed_dir / f"{name}.parquet"
        df.to_parquet(output_path, index=False)
        logger.info("Saved %s -> %s (%d rows)", name, output_path, len(df))

    logger.info("Ingestion complete")


if __name__ == "__main__":
    main()
