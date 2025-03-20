"""
preprocess.py

Run pre-processing steps on raw data exported from Taranis-AI
Save processed results to an SQLite DB
"""

import sqlite3

import pandas as pd
from transformers import AutoTokenizer

from taranis_ds.config import Config
from taranis_ds.log import get_logger
from taranis_ds.misc import check_config, detect_language
from taranis_ds.persist import check_table_exists, get_db_connection


logger = get_logger(__name__)


def df_iterator(df: pd.DataFrame, bs: int):
    # yield batches of size bs from df

    for i in range(0, len(df), bs):
        yield df.iloc[i : i + bs]


def get_tokens(df: pd.DataFrame, tokenizer_name: str) -> list:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    token_lens = []

    text_iter = df_iterator(df, 32)

    for batch in text_iter:
        texts = batch["content"].to_list()
        tokens = tokenizer(texts)["input_ids"]
        token_lens.extend(list(map(len, tokens)))
    return token_lens


def preprocess_taranis_dataset(ds_path: str, tokenizer_name: str, max_tokens: int | None = None) -> pd.DataFrame:
    df = pd.read_json(ds_path)

    df = df.explode("news_items", ignore_index=True)

    # create columns for content, title & news_item_id from the news_item
    df["content"] = df["news_items"].apply(lambda item: item["content"])
    df["title"] = df["news_items"].apply(lambda item: item["title"])
    df["news_item_id"] = df["news_items"].apply(lambda item: item["id"])

    # remove NoneType, empty and duplicated News items
    df = df[~df["news_items"].apply(lambda item: item["content"] is None or item["content"] == "")]
    df = df[~df["content"].duplicated()]

    df["tokens"] = get_tokens(df, tokenizer_name)
    df["language"] = df["content"].apply(detect_language)
    df = df[df["language"] != "err"]

    if max_tokens is not None:
        df = df[df["tokens"] <= max_tokens]

    return df[["id", "news_item_id", "title", "content", "tokens", "language"]]


def save_df_to_table(df: pd.DataFrame, table_name: str, connection: sqlite3.Connection) -> int:
    existing_df = pd.read_sql(f"SELECT * FROM {table_name}", connection, coerce_float=False)
    new_df = df[~df["id"].isin(existing_df["id"])]  # get only rows that are not already in db
    if new_df.empty:
        logger.info("No new entries to save in database")
        return 0

    rows = new_df.to_sql(table_name, connection, if_exists="append", index=False)
    return rows


def run():
    for conf_name, conf_type in [("TARANIS_DATASET_PATH", str), ("PREPROCESS_TOKENIZER", str), ("CYBERSEC_CLASS_ENDPOINT", str)]:
        if not check_config(conf_name, conf_type):
            logger.error("Skipping preprocess step")
            return

    if not check_config("PREPROCESS_MAX_TOKENS", int, required=False):
        logger.info("Config PREPROCESS_MAX_TOKENS was not set. Imposing no limit on maximum news item length")

    connection = get_db_connection(Config.DB_PATH, init=True)
    df = preprocess_taranis_dataset(Config.TARANIS_DATASET_PATH, Config.PREPROCESS_TOKENIZER, Config.PREPROCESS_MAX_TOKENS)
    logger.info("Saving preprocessed data to %s", Config.DB_PATH)

    if check_table_exists(connection, Config.TABLE_NAME):
        logger.info("Table %s already exists, update it with new entries", Config.TABLE_NAME)
        written_rows = save_df_to_table(df, Config.TABLE_NAME, connection)
        logger.info("%s rows written to %s", written_rows, Config.TABLE_NAME)
    else:
        logger.info("Creating new table %s", Config.TABLE_NAME)
        df.to_sql(Config.TABLE_NAME, connection, index=False)
    connection.close()


if __name__ == "__main__":
    run()
