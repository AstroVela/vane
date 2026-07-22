# SPDX-FileCopyrightText: 2018-2025 Stichting DuckDB Foundation
# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: MIT AND Apache-2.0
#
# Modified by Vane contributors.

import gc
import tempfile

import pandas as pd
import pytest

import vane


class TestPandasUnregister:
    def test_pandas_unregister1(self, duckdb_cursor):
        df = pd.DataFrame([[1, 2, 3], [4, 5, 6]])
        connection = vane.connect(":memory:")
        connection.register("dataframe", df)

        df2 = connection.execute("SELECT * FROM dataframe;").fetchdf()  # noqa: F841
        connection.unregister("dataframe")
        with pytest.raises(vane.CatalogException, match="Table with name dataframe does not exist"):
            connection.execute("SELECT * FROM dataframe;").fetchdf()
        with pytest.raises(vane.CatalogException, match="View with name dataframe does not exist"):
            connection.execute("DROP VIEW dataframe;")
        connection.execute("DROP VIEW IF EXISTS dataframe;")

    def test_pandas_unregister2(self, duckdb_cursor):
        with tempfile.NamedTemporaryFile() as tmp:
            db = tmp.name

        connection = vane.connect(db)
        df = pd.DataFrame([[1, 2, 3], [4, 5, 6]])

        connection.register("dataframe", df)
        connection.unregister("dataframe")  # Attempting to unregister.
        connection.close()

        # Reconnecting while DataFrame still in mem.
        connection = vane.connect(db)
        assert len(connection.execute("PRAGMA show_tables;").fetchall()) == 0

        with pytest.raises(vane.CatalogException, match="Table with name dataframe does not exist"):
            connection.execute("SELECT * FROM dataframe;").fetchdf()

        connection.close()

        del df
        gc.collect()

        # Reconnecting after DataFrame freed.
        connection = vane.connect(db)
        assert len(connection.execute("PRAGMA show_tables;").fetchall()) == 0
        with pytest.raises(vane.CatalogException, match="Table with name dataframe does not exist"):
            connection.execute("SELECT * FROM dataframe;").fetchdf()
        connection.close()
