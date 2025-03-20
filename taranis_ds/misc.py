"""
misc.py

Shared functionality across multiple modules
e.g. Taranis Dataset loading
"""

from pathlib import Path

import pandas as pd
from config import Config
from iso639 import Lang
from iso639.exceptions import InvalidLanguageValue
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException

from taranis_ds.log import get_logger


logger = get_logger(__name__)


def check_config(name: str, conf_type: type, required: bool = True):
    if name not in Config:
        if required:
            logger.error("Config %s was not set", name)
        return False
    if not isinstance(getattr(Config, name), conf_type):
        if required:
            logger.error("Config %s is not of type", conf_type)
        return False


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


def load_taranis_ds(dataset_path: str) -> pd.DataFrame:
    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        print(f"The dataset file at path {str(dataset_path)} does not exist")
        return pd.DataFrame()

    if dataset_path.suffix == ".json":
        return pd.read_json(dataset_path)
    elif dataset_path.suffix == ".pkl":
        return pd.read_pickle(dataset_path)
    else:
        print(f"Wrong file type {dataset_path.suffix}. Can only load .json or .pkl files.")
        return pd.DataFrame()
