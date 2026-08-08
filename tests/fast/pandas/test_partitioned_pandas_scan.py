# SPDX-FileCopyrightText: 2018-2025 Stichting DuckDB Foundation
# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: MIT AND Apache-2.0
#
# Modified by Vane contributors.

import numpy
import pandas as pd

import vane


class TestPartitionedPandasScan:
    def test_parallel_pandas(self, duckdb_cursor):
        con = vane.connect()
        df = pd.DataFrame({"i": numpy.arange(10000000)})

        con.register("df", df)

        seq_results = con.execute("SELECT SUM(i) FROM df").fetchall()

        con.execute("PRAGMA threads=4")
        parallel_results = con.execute("SELECT SUM(i) FROM df").fetchall()

        assert seq_results[0][0] == 49999995000000
        assert parallel_results[0][0] == 49999995000000
