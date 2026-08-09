# SPDX-FileCopyrightText: 2018-2025 Stichting DuckDB Foundation
# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: MIT AND Apache-2.0
#
# Modified by Vane contributors.

import pytest

_ = pytest.importorskip("vane.experimental.spark")

from spark_namespace.sql import functions as F


class TestSparkFunctionsBase64:
    def test_base64(self, spark):
        data = [
            ("quack",),
        ]
        res = (
            spark.createDataFrame(data, ["firstColumn"])
            .withColumn("encoded_value", F.base64(F.col("firstColumn")))
            .select("encoded_value")
            .collect()
        )
        assert res[0].encoded_value == "cXVhY2s="

    def test_base64ColString(self, spark):
        data = [
            ("quack",),
        ]
        res = (
            spark.createDataFrame(data, ["firstColumn"])
            .withColumn("encoded_value", F.base64("firstColumn"))
            .select("encoded_value")
            .collect()
        )
        assert res[0].encoded_value == "cXVhY2s="

    def test_unbase64(self, spark):
        data = [
            ("cXVhY2s=",),
        ]
        res = (
            spark.createDataFrame(data, ["firstColumn"])
            .withColumn("decoded_value", F.unbase64(F.col("firstColumn")))
            .select("decoded_value")
            .collect()
        )
        assert res[0].decoded_value == b"quack"
