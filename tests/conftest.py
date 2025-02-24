import os
import pytest
import json
import pandas as pd
from dotenv import load_dotenv
import sqlite3

test_dir = os.path.dirname(os.path.abspath(__file__))
env_file = os.path.join(test_dir, ".env.test")
current_path = os.getcwd()

load_dotenv(dotenv_path=env_file, override=True)

@pytest.fixture(scope="session")
def taranis_dataset_path():
    yield os.path.abspath(os.path.join(test_dir, f"{os.getenv('TARANIS_DATASET_PATH')}"))

@pytest.fixture(scope="session")
def results_db_path():
    yield os.path.abspath(os.path.join(test_dir, f"{os.getenv('DB_PATH')}"))

@pytest.fixture(scope="session")
def test_db_path():
    yield os.path.abspath(os.path.join(test_dir, "test.db"))

@pytest.fixture(scope="session")
def tokenizer():
    yield os.getenv("PREPROCESS_TOKENIZER")

@pytest.fixture(scope="session")
def taranis_dataset_json(taranis_dataset_path):
    yield json.loads(taranis_dataset_path)

@pytest.fixture(scope="session")
def taranis_dataset_df():
    yield pd.read_json(taranis_dataset_path)

@pytest.fixture(scope="session")
def test_db(test_db_path):
    conn = sqlite3.Connection(test_db_path)
    conn.execute("CREATE TABLE test(id INTEGER PRIMARY KEY, col1 TEXT, col2 INTEGER)")
    yield conn

    os.remove(test_db_path)

@pytest.fixture(scope="session")
def results_db(results_db_path):
    conn = sqlite3.Connection(results_db_path)
    conn.execute("CREATE TABLE results(id TEXT PRIMARY KEY, news_item_id TEXT, title TEXT, content TEXT, tokens INTEGER, language TEXT)")
    yield conn

    os.remove(results_db_path)
