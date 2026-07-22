# SPDX-FileCopyrightText: 2018-2025 Stichting DuckDB Foundation
# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: MIT AND Apache-2.0
#
# Modified by Vane contributors.

import pytest

_ = pytest.importorskip("vane.experimental.spark")
from spark_namespace.sql.functions import col, concat_ws
from spark_namespace.sql.types import Row


class TestReplaceEmpty:
    def test_replace_empty(self, spark):
        data = [
            ("firstRowFirstColumn", "firstRowSecondColumn"),
            ("2ndRowFirstColumn", "2ndRowSecondColumn"),
        ]
        df = spark.createDataFrame(data, ["firstColumn", "secondColumn"])
        df = df.withColumn("concatted", concat_ws(" ", col("firstColumn"), col("secondColumn")))
        res = df.select("concatted").collect()
        assert res == [
            Row(concatted="firstRowFirstColumn firstRowSecondColumn"),
            Row(concatted="2ndRowFirstColumn 2ndRowSecondColumn"),
        ]
