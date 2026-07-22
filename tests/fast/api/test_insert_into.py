# SPDX-FileCopyrightText: 2018-2025 Stichting DuckDB Foundation
# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: MIT AND Apache-2.0
#
# Modified by Vane contributors.

import pytest
from pandas import DataFrame

import vane


class TestInsertInto:
    def test_insert_into_schema(self, duckdb_cursor):
        # open connection
        con = vane.connect()
        con.execute("CREATE SCHEMA s")
        con.execute("CREATE TABLE s.t (id INTEGER PRIMARY KEY)")

        # make relation
        df = DataFrame([1], columns=["id"])
        rel = con.from_df(df)

        rel.insert_into("s.t")

        assert con.execute("select * from s.t").fetchall() == [(1,)]

        # This should fail since this will go to default schema
        with pytest.raises(vane.CatalogException):
            rel.insert_into("t")

        # If we add t in the default schema it should work.
        con.execute("CREATE TABLE t (id INTEGER PRIMARY KEY)")
        rel.insert_into("t")
        assert con.execute("select * from t").fetchall() == [(1,)]
