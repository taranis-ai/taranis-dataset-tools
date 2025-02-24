import os
import pytest
import json
import pandas as pd
from dotenv import load_dotenv


base_dir = os.path.dirname(os.path.abspath(__file__))
env_file = os.path.join(base_dir, ".env.test")
current_path = os.getcwd()

load_dotenv(dotenv_path=env_file, override=True)

@pytest.fixture(scope="session")
def taranis_dataset_json():
    yield json.loads("assets/taranis_story_export_feb_2025_tiny.json")

@pytest.fixture(scope="session")
def taranis_dataset_df():
    yield pd.read_json("assets/taranis_story_export_feb_2025_tiny.json")
