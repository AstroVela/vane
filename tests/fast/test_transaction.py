# SPDX-FileCopyrightText: 2018-2025 Stichting DuckDB Foundation
# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: MIT AND Apache-2.0
#
# Modified by Vane contributors.

import vane


class TestConnectionTransaction:
    def test_transaction(self, duckdb_cursor):
        con = vane.connect()
        con.execute("create table t (i integer)")
        con.execute("insert into t values (1)")

        con.begin()
        con.execute("insert into t values (1)")
        assert con.execute("select count (*) from t").fetchone()[0] == 2
        con.rollback()
        assert con.execute("select count (*) from t").fetchone()[0] == 1
        con.begin()
        con.execute("insert into t values (1)")
        assert con.execute("select count (*) from t").fetchone()[0] == 2
        con.commit()
        assert con.execute("select count (*) from t").fetchone()[0] == 2
