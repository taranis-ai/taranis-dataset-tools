"""
main.py

Run the pipeline
"""

from pathlib import Path

import pandas as pd

import taranis_ds
from taranis_ds.config import VALID_TASKS, Config
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
        # get correct subset of columns if exists
        df = df[["id", "news_item_id", "title", "content", "tokens", "language"]]
    except ValueError as e:
        logger.error("Could not load %s. Error: %s", Config.PROCESSED_DATASET_PATH, e)
        return

    connection = get_db_connection(Config.DB_PATH, "results")
    save_df_to_table(df, connection)


def run():
    for task in VALID_TASKS:
        if task in Config.TASKS:
            module = getattr(taranis_ds, task)
            module.run()


if __name__ == "__main__":
    run()
