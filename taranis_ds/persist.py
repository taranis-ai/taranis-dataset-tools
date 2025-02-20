"""
persist.py

Deals with saving results from the pipeline to disk and loading them upon continuation
"""

import sqlite3
from pathlib import Path
from typing import List, Tuple
from log import get_logger

logger = get_logger(__name__)


def check_table_exists(connection: sqlite3.Connection, table_name: str) -> bool:
    tables = connection.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}'").fetchall()
    return tables != []


def check_column_exists(connection: sqlite3.Connection, table_name: str, column_name: str) -> bool:
    table_info = connection.execute(f"PRAGMA table_info({table_name})").fetchall()
    columns = [entry[1] for entry in table_info]
    return column_name in columns


def init_db(db_path: str, table_name: str):
    connection = sqlite3.Connection(db_path)
    if check_table_exists(connection, "table_name"):
        logger.info("Table %s already exists", table_name)
        connection.close()

    connection.execute(
        f"CREATE TABLE {table_name}(id TEXT PRIMARY KEY, news_item_id TEXT, title TEXT, content TEXT, tokens INTEGER, language TEXT)"
    )
    connection.close()


def get_db_connection(db_path: str) -> sqlite3.Connection:
    if not Path(db_path).exists():
        raise RuntimeError(f"Database at {db_path} does not exist. You need to create it first")
    connection = sqlite3.Connection(db_path)
    return connection


def insert_column(connection: sqlite3.Connection, table_name: str, column_name: str, column_type: str):
    if check_table_exists(connection, table_name):
        logger.info("Table %s already exists.", table_name)
        return

    try:
        query = f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type}"
        logger.debug("Running SQL query: %s", query)
        connection.execute(query)
    except sqlite3.OperationalError as e:
        raise RuntimeError(f"Cannot add column {column_name} to table {table_name}. Error: {e}")


def update_row(connection: sqlite3.Connection, table_name: str, row_id: str, columns: List[str], values: List[str | int]):
    update_statements = []
    for col, val in zip(columns, values):
        if isinstance(val, int):
            update_statements.append(f"{col} = {val}")
        elif isinstance(val, str):
            update_statements.append(f"{col} = '{val}'")
    update_stmt = (", ").join(update_statements)

    with connection:
        try:
            query = f"UPDATE {table_name} SET {update_stmt} WHERE id = '{row_id}'"
            logger.debug("Running SQL query: %s", query)
            result = connection.execute(query)
        except sqlite3.OperationalError as e:
            raise RuntimeError(f"Failed to update row with id {row_id}. Error: {e}")

        if result.rowcount != 1:
            raise RuntimeError(f"Could not update row with id {row_id} for unknown reason")


def run_query(connection: sqlite3.Connection, query: str) -> List[Tuple]:
    try:
        logger.debug("Running SQL query: %s", query)
        result = connection.execute(query).fetchall()
    except sqlite3.OperationalError as e:
        raise RuntimeError(f"Failed to execute query {query}. Error: {e}")
    return result
