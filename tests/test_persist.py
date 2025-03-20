import os
import pytest
from taranis_ds import persist
import sqlite3

def test_check_table_exists(test_db):
    assert persist.check_table_exists(test_db, "results")
    assert not persist.check_table_exists(test_db, "other_table")

def test_check_column_exists(test_db):
    assert persist.check_column_exists(test_db, "results", "id")
    assert not persist.check_column_exists(test_db, "results", "other_column")

def test_init_db(test_db_path):
    persist.init_db(test_db_path, "results")
    assert os.path.exists(test_db_path)

    connection = sqlite3.Connection(test_db_path)
    assert persist.check_table_exists(connection, "results")
    os.remove(test_db_path)

def test_insert_column(test_db):
    persist.insert_column(test_db, "results", "new_column", "TEXT")
    assert persist.check_column_exists(test_db, "results", "new_column")

    with pytest.raises(RuntimeError) as exception:
        persist.insert_column(test_db, "unknown_table", "new_column", "TEXT")
    assert str(exception.value) == "Table unknown_table does not exist."


def test_update_row(test_db):
    with test_db:
        test_db.execute("INSERT INTO results (id, col1, col2) VALUES (1, 'test', 55)")

    persist.update_row(test_db, "results", "1", ["col1", "col2"], ["new_val", 90])
    result = test_db.execute("SELECT * FROM results").fetchall()
    assert result == [(1, "new_val", 90)]

    with pytest.raises(RuntimeError) as exception:
        persist.update_row(test_db, "results", "2", ["col1", "col2"], ["new_val", 90])
    assert str(exception.value) == "Could not update row with id 2."


def test_run_query(test_db):
    with test_db:
        test_db.execute("INSERT INTO results (id, col1, col2) VALUES (1, 'test', 55)")
    result = persist.run_query(test_db, "SELECT * FROM results")
    assert isinstance(result, list)
    assert result[0] == (1, "test", 55)