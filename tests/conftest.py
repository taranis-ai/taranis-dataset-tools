import os
import pytest
import json
import pandas as pd
from dotenv import load_dotenv


test_dir = os.path.dirname(os.path.abspath(__file__))
env_file = os.path.join(test_dir, ".env.test")
current_path = os.getcwd()

load_dotenv(dotenv_path=env_file, override=True)

@pytest.fixture(scope="session")
def taranis_dataset_path():
    yield os.path.abspath(os.path.join(test_dir, f"{os.getenv('TARANIS_DATASET_PATH')}"))

@pytest.fixture(scope="session")
def tokenizer():
    yield os.getenv("PREPROCESS_TOKENIZER")

@pytest.fixture(scope="session")
def taranis_dataset_json(taranis_dataset_path):
    yield json.loads(taranis_dataset_path)

@pytest.fixture(scope="session")
def taranis_dataset_df():
    yield pd.read_json(taranis_dataset_path)
