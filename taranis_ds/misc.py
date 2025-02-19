"""
misc.py

Shared functionality across multiple modules
e.g. Taranis Dataset loading
"""

import pandas as pd
from pathlib import Path
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException


def detect_lang(text: str) -> str:
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
