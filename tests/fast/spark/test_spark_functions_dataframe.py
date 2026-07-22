# SPDX-FileCopyrightText: 2018-2025 Stichting DuckDB Foundation
# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: MIT AND Apache-2.0
#
# Modified by Vane contributors.

import pytest

_ = pytest.importorskip("vane.experimental.spark")
from spark_namespace.sql import functions as F


class TestSparkFunctionsArray:
    def test_broadcast(self, spark):
        data = [
            ([1, 2, 2], 2),
            ([2, 4, 5], 3),
        ]

        df = spark.createDataFrame(data, ["firstColumn", "secondColumn"])
        df_broadcast = F.broadcast(df)

        assert df.collect() == df_broadcast.collect()
