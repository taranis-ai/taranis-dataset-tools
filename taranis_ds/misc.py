"""
misc.py

Shared functionality across multiple modules
e.g. Taranis Dataset loading
"""

import sqlite3

import pandas as pd
from iso639 import Lang
from iso639.exceptions import InvalidLanguageValue
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException

from taranis_ds.config import Config
from taranis_ds.log import get_logger


logger = get_logger(__name__)


def save_df_to_table(df: pd.DataFrame, connection: sqlite3.Connection) -> int:
    existing_df = pd.read_sql("SELECT * FROM results", connection, coerce_float=False)
    new_df = df[~df["id"].isin(existing_df["id"])]  # get only rows that are not already in db
    if new_df.empty:
        logger.info("No new entries to save in database")
        return 0

    return new_df.to_sql("results", connection, if_exists="append", index=False) or 0


def check_config(name: str, conf_type: type, required: bool = True):
    if not getattr(Config, name, None):
        if required:
            logger.error("Config %s was not set", name)
        return False
    if not isinstance(getattr(Config, name), conf_type):
        if required:
            logger.error("Config %s is not of type", conf_type)
        return False
    return True


def convert_language(lang_code: str) -> str:
    try:
        language = Lang(lang_code).name.lower()
    except InvalidLanguageValue:
        return "english"

    return language


def detect_language(text: str) -> str:
    try:
        return detect(text)
    except LangDetectException:
        return "err"
