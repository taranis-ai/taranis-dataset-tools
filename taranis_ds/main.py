"""
main.py

Run the pipeline
"""

from pathlib import Path

import pandas as pd

from taranis_ds.config import Config
from taranis_ds.log import get_logger
from taranis_ds.misc import check_config, save_df_to_table
from taranis_ds.persist import get_db_connection


logger = get_logger(__name__)


def save_to_db():
    if not check_config("PROCESSED_DATASET_PATH", str):
        return

    if not Path(Config.PROCESSED_DATASET_PATH).exists():
        logger.error("%s does not exist", Config.PROCESSED_DATASET_PATH)
        return

    if not Config.PROCESSED_DATASET_PATH.endswith(".json"):
        logger.error("%s must be in .json format", Config.PROCESSED_DATASET_PATH)
        return

    try:
        df = pd.read_json(Config.PROCESSED_DATASET_PATH)
    except ValueError as e:
        logger.error("Could not load %s. Error: %s", Config.PROCESSED_DATASET_PATH, e)
        return

    connection = get_db_connection(Config.DB_PATH, "results")
    save_df_to_table(df, connection)


def run():
    pass


if __name__ == "__main__":
    run()
