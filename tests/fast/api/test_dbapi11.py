# SPDX-FileCopyrightText: 2018-2025 Stichting DuckDB Foundation
# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: MIT AND Apache-2.0
#
# Modified by Vane contributors.

# cursor description

import tempfile

import vane


def check_exception(f):
    had_exception = False
    try:
        f()
    except Exception:
        had_exception = True
    assert had_exception


class TestReadOnly:
    def test_readonly(self, duckdb_cursor):
        with tempfile.NamedTemporaryFile() as tmp:
            db = tmp.name

        # this is forbidden
        check_exception(lambda: vane.connect(":memory:", True))

        con_rw = vane.connect(db, False)
        con_rw.cursor().execute("create table a (i integer)")
        con_rw.cursor().execute("insert into a values (42)")
        con_rw.close()

        con_ro = vane.connect(db, True)
        con_ro.cursor().execute("select * from a").fetchall()
        check_exception(lambda: con_ro.execute("delete from a"))
        con_ro.close()

        con_rw = vane.connect(db, False)
        con_rw.cursor().execute("drop table a")
        con_rw.close()
