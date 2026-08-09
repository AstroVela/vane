# SPDX-FileCopyrightText: 2018-2025 Stichting DuckDB Foundation
# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: MIT AND Apache-2.0
#
# Modified by Vane contributors.

from pathlib import Path

import pytest

import vane

script_path = Path(__file__).parent


class TestTableFunction:
    def test_table_function(self, duckdb_cursor):
        path = str(script_path / ".." / "data/integers.csv")
        rel = duckdb_cursor.table_function("read_csv", [path])
        res = rel.fetchall()
        assert res == [(1, 10, 0), (2, 50, 30)]

        # Provide only a string as argument, should error, needs a list
        with pytest.raises(vane.InvalidInputException, match=r"'params' has to be a list of parameters"):
            rel = duckdb_cursor.table_function("read_csv", path)
