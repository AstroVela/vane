# SPDX-FileCopyrightText: 2018-2025 Stichting DuckDB Foundation
# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: MIT AND Apache-2.0
#
# Modified by Vane contributors.

import pytest

_ = pytest.importorskip("vane.experimental.spark")


from spark_namespace.sql.types import Row


class TestSparkReadParquet:
    def test_read_parquet(self, duckdb_cursor, spark, tmp_path):
        file_path = tmp_path / "basic.parquet"
        file_path = file_path.as_posix()
        duckdb_cursor.execute(
            f"COPY (select 42 a, true b, 'this is a long string' c) to '{file_path}' (FORMAT PARQUET)"
        )
        df = spark.read.parquet(file_path)
        res = df.collect()
        assert res == [Row(a=42, b=True, c="this is a long string")]
