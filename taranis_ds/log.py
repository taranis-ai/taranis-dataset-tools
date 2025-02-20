"""
log.py

Setup logger
"""

import logging
from config import Config


def get_logger(name: str):
    logger = logging.getLogger(name)
    if Config.DEBUG:
        log_level = logging.DEBUG
    else:
        log_level = logging.INFO

    logger.setLevel(log_level)

    if not logger.handlers:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)

        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s", "%Y-%m-%d %H:%M:%S")
        console_handler.setFormatter(formatter)

        logger.addHandler(console_handler)

    return logger
