# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

import duckdb


def test_explode_fixed_array(duckdb_cursor):
    relation = duckdb_cursor.sql(
        """
        SELECT *
        FROM (
            VALUES
                (10, [20, 21]::INTEGER[2], 30),
                (11, NULL::INTEGER[2], 31),
                (12, [22, NULL]::INTEGER[2], 32)
        ) data(x, a, y)
        """
    )

    exploded = relation.explode("a")

    assert exploded.columns == ["x", "a", "y"]
    assert exploded.types == [duckdb.sqltypes.INTEGER] * 3
    assert exploded.fetchall() == [
        (10, 20, 30),
        (10, 21, 30),
        (12, 22, 32),
        (12, None, 32),
    ]
    assert duckdb_cursor.sql("SELECT 1").fetchall() == [(1,)]
