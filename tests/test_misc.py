import logging
import pandas as pd
from taranis_ds.config import Config
from taranis_ds.misc import save_df_to_table, check_config, convert_language

logger = logging.getLogger(__name__)


def test_save_df_to_table(test_db):
    df = pd.DataFrame([{"id": 1, "col1": "Some text", "col2": 666},
                       {"id": 2, "col1": "Yet another text", "col2": 90},
                       {"id": 3, "col1": "three", "col2": 22},
                       ])
    assert save_df_to_table(df, test_db) == 3
    assert save_df_to_table(df, test_db) == 0

def test_check_config():
    assert check_config("PREPROCESS_MAX_TOKENS", int)
    assert check_config("DB_PATH", str)
    assert not check_config("DB_PATH", int)
    assert not check_config("UNKNOWN_CONFIG", dict)


def test_convert_language():
    assert convert_language("fr") == "french"
    assert convert_language("de") == "german"
    assert convert_language("en") == "english"
    assert convert_language("ru") == "russian"
