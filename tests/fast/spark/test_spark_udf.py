# SPDX-FileCopyrightText: 2018-2025 Stichting DuckDB Foundation
# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: MIT AND Apache-2.0
#
# Modified by Vane contributors.

import pytest

_ = pytest.importorskip("vane.experimental.spark")


class TestSparkUDF:
    def test_udf_register(self, spark):
        def to_upper_fn(s: str) -> str:
            return s.upper()

        with pytest.raises(NotImplementedError, match="UDF registration is disabled"):
            spark.udf.register("to_upper_fn", to_upper_fn)
