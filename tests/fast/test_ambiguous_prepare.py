# SPDX-FileCopyrightText: 2018-2025 Stichting DuckDB Foundation
# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: MIT AND Apache-2.0
#
# Modified by Vane contributors.

import vane


class TestAmbiguousPrepare:
    def test_bool(self, duckdb_cursor):
        conn = vane.connect()
        res = conn.execute("select ?, ?, ?", (True, 42, [1, 2, 3])).fetchall()
        assert res[0][0]
        assert res[0][1] == 42
        assert res[0][2] == [1, 2, 3]
