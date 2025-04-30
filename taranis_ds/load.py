"""
load.py

Load all news items from a Taranis AI instance
"""

import json
from json import JSONDecodeError
from pathlib import Path
from typing import Any

import requests

from taranis_ds.config import Config
from taranis_ds.log import get_logger
from taranis_ds.misc import check_config


logger = get_logger(__name__)


def fetch_taranis_stories(taranis_url: str, auth_endpoint: str, export_endpoint: str, username: str, password: str) -> list[dict[str, Any]]:
    try:
        auth_response = requests.post(
            f"{taranis_url}{auth_endpoint}",
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            json={"username": username, "password": password},
            timeout=10,
        )
        auth_response.raise_for_status()
    except requests.HTTPError as e:
        logger.error(f"Could not authenticate to {taranis_url}{auth_endpoint}: {e}")
        return []

    try:
        auth_token = auth_response.json().get("access_token")
        if not auth_token:
            logger.error("No access_token found in auth response")
            return []
    except JSONDecodeError:
        logger.error("Failed to parse auth response JSON")
        return []

    try:
        export_response = requests.get(
            f"{taranis_url}{export_endpoint}", headers={"Accept": "application/json", "Authorization": f"Bearer {auth_token}"}, timeout=10
        )
        export_response.raise_for_status()
    except requests.HTTPError as e:
        logger.error(f"Failed to export stories from {taranis_url}{export_endpoint}: {e}")
        return []

    try:
        return export_response.json()
    except JSONDecodeError:
        logger.error("Could not parse exported stories JSON")
        return []


def run():
    logger.info("Running load step")
    for conf_name, conf_type in [
        ("TARANIS_DATASET_PATH", str),
        ("TARANIS_INSTANCE_URL", str),
        ("TARANIS_AUTH_ENDPOINT", str),
        ("TARANIS_EXPORT_ENDPOINT", str),
        ("TARANIS_ADMIN_USERNAME", str),
        ("TARANIS_ADMIN_PASSWORD", str),
    ]:
        if not check_config(conf_name, conf_type):
            logger.error("Skipping load step")
            return

    if Path(Config.TARANIS_DATASET_PATH).exists():
        logger.error("%s does already exist! Will not overwrite!", Config.TARANIS_DATASET_PATH)
        return

    if not Config.TARANIS_DATASET_PATH.endswith(".json"):
        logger.error("%s must be in .json format", Config.TARANIS_DATASET_PATH)
        return

    logger.info("Fetching stories from %s", Config.TARANIS_INSTANCE_URL)
    stories = fetch_taranis_stories(
        Config.TARANIS_INSTANCE_URL,
        Config.TARANIS_AUTH_ENDPOINT,
        Config.TARANIS_EXPORT_ENDPOINT,
        Config.TARANIS_ADMIN_USERNAME,
        Config.TARANIS_ADMIN_PASSWORD,
    )

    logger.info("Saving stories to %s", Config.TARANIS_DATASET_PATH)
    with open(Config.TARANIS_DATASET_PATH, "w") as f:
        json.dump(stories, f)


if __name__ == "__main__":
    run()
