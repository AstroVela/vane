# SPDX-FileCopyrightText: 2018-2025 Stichting DuckDB Foundation
# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: MIT AND Apache-2.0
#
# Modified by Vane contributors.

# cursor description
import tempfile

import pytest

import vane


def check_exception(f):
    had_exception = False
    try:
        f()
    except BaseException:
        had_exception = True
    assert had_exception


class TestConnectionClose:
    def test_connection_close(self, duckdb_cursor):
        with tempfile.NamedTemporaryFile() as tmp:
            db = tmp.name
        con = vane.connect(db)
        cursor = con.cursor()
        cursor.execute("create table a (i integer)")
        cursor.execute("insert into a values (42)")
        con.close()
        check_exception(lambda: cursor.execute("select * from a"))

    def test_open_and_exit(self):
        with pytest.raises(TypeError), vane.connect():
            # This exception does not get swallowed by DuckDBPyConnection's __exit__
            raise TypeError()

    def test_reopen_connection(self, duckdb_cursor):
        with tempfile.NamedTemporaryFile() as tmp:
            db = tmp.name
        con = vane.connect(db)
        cursor = con.cursor()
        cursor.execute("create table a (i integer)")
        cursor.execute("insert into a values (42)")
        con.close()
        con = vane.connect(db)
        cursor = con.cursor()
        results = cursor.execute("select * from a").fetchall()
        assert results == [(42,)]

    def test_get_closed_default_conn(self, duckdb_cursor):
        con = vane.connect()
        vane.set_default_connection(con)
        vane.close()

        # 'vane.close()' closes this connection, because we explicitly set it as the default
        with pytest.raises(vane.ConnectionException, match="Connection Error: Connection already closed"):
            con.sql("select 42").fetchall()

        default_con = vane.default_connection()
        default_con.sql("select 42").fetchall()
        default_con.close()

        # This does not error because the closed connection is silently replaced
        vane.sql("select 42").fetchall()

        # Show that the 'default_con' is still closed
        with pytest.raises(vane.ConnectionException, match="Connection Error: Connection already closed"):
            default_con.sql("select 42").fetchall()

        vane.close()

        # This also does not error because we silently receive a new connection
        con2 = vane.connect(":default:")
        con2.sql("select 42").fetchall()
