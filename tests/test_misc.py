import sqlite3
import logging

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from taranis_ds.misc import save_df_to_table

logger = logging.getLogger(__name__)


def test_save_df_to_table(test_db):
    df = pd.DataFrame([{"id": 1, "col1": "Some text", "col2": 666},
                       {"id": 2, "col1": "Yet another text", "col2": 90},
                       {"id": 3, "col1": "three", "col2": 22},
                       ])
    assert save_df_to_table(df, test_db) == 3
    assert save_df_to_table(df, test_db) == 0