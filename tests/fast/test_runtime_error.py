# SPDX-FileCopyrightText: 2018-2025 Stichting DuckDB Foundation
# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: MIT AND Apache-2.0
#
# Modified by Vane contributors.

import pandas as pd
import pytest

import vane


def closed():
    return pytest.raises(vane.ConnectionException, match="Connection already closed")


def no_result_set():
    return pytest.raises(vane.InvalidInputException, match="No open result set")


class TestRuntimeError:
    def test_fetch_error(self):
        con = vane.connect()
        con.execute("create table tbl as select 'hello' i")
        with pytest.raises(vane.ConversionException):
            con.execute("select i::int from tbl").fetchall()

    def test_df_error(self):
        con = vane.connect()
        con.execute("create table tbl as select 'hello' i")
        with pytest.raises(vane.ConversionException):
            con.execute("select i::int from tbl").df()

    def test_arrow_error(self):
        pytest.importorskip("pyarrow")

        con = vane.connect()
        con.execute("create table tbl as select 'hello' i")
        with pytest.raises(vane.ConversionException):
            con.execute("select i::int from tbl").to_arrow_table()

    def test_register_error(self):
        con = vane.connect()
        py_obj = "this is a string"
        with pytest.raises(vane.InvalidInputException, match='Python Object "this is a string" of type "str"'):
            con.register(py_obj, "v")

    def test_arrow_fetch_table_error(self):
        pytest.importorskip("pyarrow")

        con = vane.connect()
        arrow_object = con.execute("select 1").to_arrow_table()
        arrow_relation = con.from_arrow(arrow_object)
        res = arrow_relation.execute()
        res.close()
        with pytest.raises(vane.InvalidInputException, match="There is no query result"):
            res.to_arrow_table()

    def test_arrow_record_batch_reader_error(self):
        pytest.importorskip("pyarrow")

        con = vane.connect()
        arrow_object = con.execute("select 1").to_arrow_table()
        arrow_relation = con.from_arrow(arrow_object)
        res = arrow_relation.execute()
        res.close()
        with pytest.raises(vane.ProgrammingError, match="There is no query result"):
            res.to_arrow_reader(1)

    def test_relation_cache_fetchall(self):
        conn = vane.connect()
        df_in = pd.DataFrame(
            {
                "numbers": [1, 2, 3, 4, 5],
            }
        )
        conn.execute("create view x as select * from df_in")
        rel = conn.query("select * from x")
        del df_in
        with pytest.raises(vane.ProgrammingError, match="Table with name df_in does not exist"):
            # Even when we preserve ExternalDependency objects correctly, this is not supported
            # Relations only save dependencies for their immediate TableRefs,
            # so the dependency of 'x' on 'df_in' is not registered in 'rel'
            rel.fetchall()

    def test_relation_cache_execute(self):
        conn = vane.connect()
        df_in = pd.DataFrame(
            {
                "numbers": [1, 2, 3, 4, 5],
            }
        )
        conn.execute("create view x as select * from df_in")
        rel = conn.query("select * from x")
        del df_in
        with pytest.raises(vane.ProgrammingError, match="Table with name df_in does not exist"):
            rel.execute()

    def test_relation_query_error(self):
        conn = vane.connect()
        df_in = pd.DataFrame(
            {
                "numbers": [1, 2, 3, 4, 5],
            }
        )
        conn.execute("create view x as select * from df_in")
        rel = conn.query("select * from x")
        del df_in
        with pytest.raises(vane.CatalogException, match="Table with name df_in does not exist"):
            rel.query("bla", "select * from bla")

    def test_conn_broken_statement_error(self):
        conn = vane.connect()
        df_in = pd.DataFrame(
            {
                "numbers": [1, 2, 3, 4, 5],
            }
        )
        conn.execute("create view x as select * from df_in")
        del df_in
        with pytest.raises(vane.CatalogException, match="Table with name df_in does not exist"):
            conn.execute("select 1; select * from x; select 3;")

    def test_conn_prepared_statement_error(self):
        conn = vane.connect()
        conn.execute("create table integers (a integer, b integer)")
        with pytest.raises(
            vane.InvalidInputException,
            match="Values were not provided for the following prepared statement parameters: 2",
        ):
            conn.execute("select * from integers where a =? and b=?", [1])

    def test_closed_conn_exceptions(self):
        conn = vane.connect()
        conn.close()
        df_in = pd.DataFrame(
            {
                "numbers": [1, 2, 3, 4, 5],
            }
        )

        with closed():
            conn.register("bla", df_in)

        with closed():
            conn.from_query("select 1")

        with closed():
            conn.table("bla")

        with closed():
            conn.table("bla")

        with closed():
            conn.view("bla")

        with closed():
            conn.values("bla")

        with closed():
            conn.table_function("bla")

        with closed():
            conn.from_df(df_in)

        with closed():
            conn.from_csv_auto("bla")

        with closed():
            conn.from_parquet("bla")

        with closed():
            conn.from_arrow("bla")

    def test_missing_result_from_conn_exceptions(self):
        conn = vane.connect()

        with no_result_set():
            conn.fetchone()

        with no_result_set():
            conn.fetchall()

        with no_result_set():
            conn.fetchnumpy()

        with no_result_set():
            conn.fetchdf()

        with no_result_set():
            conn.fetch_df_chunk()

        with no_result_set():
            conn.to_arrow_table()

        with no_result_set():
            conn.to_arrow_reader()
