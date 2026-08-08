# SPDX-FileCopyrightText: 2018-2025 Stichting DuckDB Foundation
# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: MIT AND Apache-2.0
#
# Modified by Vane contributors.

import gc
import tempfile
from pathlib import Path

import pytest

import vane

pyarrow = pytest.importorskip("pyarrow")
pytest.importorskip("pyarrow.parquet")


class TestArrowUnregister:
    def test_arrow_unregister1(self, duckdb_cursor):
        parquet_filename = str(Path(__file__).parent / "data" / "userdata1.parquet")
        arrow_table_obj = pyarrow.parquet.read_table(parquet_filename)
        connection = vane.connect(":memory:")
        connection.register("arrow_table", arrow_table_obj)

        connection.execute("SELECT * FROM arrow_table;").to_arrow_table()
        connection.unregister("arrow_table")
        with pytest.raises(vane.CatalogException, match="Table with name arrow_table does not exist"):
            connection.execute("SELECT * FROM arrow_table;").to_arrow_table()
        with pytest.raises(vane.CatalogException, match="View with name arrow_table does not exist"):
            connection.execute("DROP VIEW arrow_table;")
        connection.execute("DROP VIEW IF EXISTS arrow_table;")

    def test_arrow_unregister2(self, duckdb_cursor):
        with tempfile.NamedTemporaryFile() as tmp:
            db = tmp.name

        connection = vane.connect(db)
        parquet_filename = str(Path(__file__).parent / "data" / "userdata1.parquet")
        arrow_table_obj = pyarrow.parquet.read_table(parquet_filename)
        connection.register("arrow_table", arrow_table_obj)
        connection.unregister("arrow_table")  # Attempting to unregister.
        connection.close()
        # Reconnecting while Arrow Table still in mem.
        connection = vane.connect(db)
        assert len(connection.execute("PRAGMA show_tables;").fetchall()) == 0
        with pytest.raises(vane.CatalogException, match="Table with name arrow_table does not exist"):
            connection.execute("SELECT * FROM arrow_table;").to_arrow_table()
        connection.close()
        del arrow_table_obj
        gc.collect()
        # Reconnecting after Arrow Table is freed.
        connection = vane.connect(db)
        assert len(connection.execute("PRAGMA show_tables;").fetchall()) == 0
        with pytest.raises(vane.CatalogException, match="Table with name arrow_table does not exist"):
            connection.execute("SELECT * FROM arrow_table;").to_arrow_table()
        connection.close()
