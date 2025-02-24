import os
import pytest
from taranis_ds import persist
import sqlite3

def test_check_table_exists(test_db):
    assert persist.check_table_exists(test_db, "test")
    assert not persist.check_table_exists(test_db, "other_table")

def test_check_column_exists(test_db):
    assert persist.check_column_exists(test_db, "test", "id")
    assert not persist.check_column_exists(test_db, "test", "other_column")

def test_init_db(test_db_path):
    persist.init_db(test_db_path, "test")
    assert os.path.exists(test_db_path)

    connection = sqlite3.Connection(test_db_path)
    assert persist.check_table_exists(connection, "test")
    os.remove(test_db_path)

def test_get_db_connection():
    with pytest.raises(RuntimeError) as exception:
        conn = persist.get_db_connection("unknown_path.db")
    assert str(exception.value) == f"Database at unknown_path.db does not exist. You need to create it first"


def test_insert_column(test_db):
    persist.insert_column(test_db, "test", "new_column", "TEXT")
    assert persist.check_column_exists(test_db, "test", "new_column")

    with pytest.raises(RuntimeError) as exception:
        persist.insert_column(test_db, "unknown_table", "new_column", "TEXT")
    assert str(exception.value) == f"Table unknown_table does not exist."


def test_update_row(test_db):
    with test_db:
        test_db.execute("INSERT INTO test (id, col1, col2) VALUES (1, 'test', 55)")

    persist.update_row(test_db, "test", "1", ["col1", "col2"], ["new_val", 90])
    result = test_db.execute("SELECT * FROM test").fetchall()
    assert result == [(1, "new_val", 90)]

    with pytest.raises(RuntimeError) as exception:
        persist.update_row(test_db, "test", "2", ["col1", "col2"], ["new_val", 90])
    assert str(exception.value) == "Could not update row with id 2."


def test_run_query(test_db):
    with test_db:
        test_db.execute("INSERT INTO test (id, col1, col2) VALUES (1, 'test', 55)")
    result = persist.run_query(test_db, "SELECT * FROM test")
    assert isinstance(result, list)
    assert result[0] == (1, "test", 55)