# SPDX-FileCopyrightText: 2018-2025 Stichting DuckDB Foundation
# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: MIT AND Apache-2.0
#
# Modified by Vane contributors.

import vane


class TestContextManager:
    def test_context_manager(self, duckdb_cursor):
        with vane.connect(database=":memory:", read_only=False) as con:
            assert con.execute("select 1").fetchall() == [(1,)]
