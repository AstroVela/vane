# SPDX-FileCopyrightText: 2018-2025 Stichting DuckDB Foundation
# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: MIT AND Apache-2.0
#
# Modified by Vane contributors.

import pytest

import vane


class TestUDFTransactionality:
    @pytest.mark.xfail(reason="fetchone() does not realize the stream result was closed before completion")
    def test_type_coverage(self, duckdb_cursor):
        rel = duckdb_cursor.sql("select * from range(4096)")
        res = rel.fetchone()
        assert res == (0,)

        def my_func(x: str) -> int:
            return int(x)

        duckdb_cursor.create_function("test", my_func)

        with pytest.raises(vane.InvalidInputException, match="result closed"):
            res = rel.fetchone()
