# SPDX-FileCopyrightText: 2018-2025 Stichting DuckDB Foundation
# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: MIT AND Apache-2.0
#
# Modified by Vane contributors.

import pandas as pd

import vane


class TestPandasAcceptFloat16:
    def test_pandas_accept_float16(self, duckdb_cursor):
        df = pd.DataFrame({"col": [1, 2, 3]})
        df16 = df.astype({"col": "float16"})  # noqa: F841
        con = vane.connect()
        con.execute("CREATE TABLE tbl AS SELECT * FROM df16")
        con.execute("select * from tbl")
        df_result = con.fetchdf()
        df32 = df.astype({"col": "float32"})
        assert (df32["col"] == df_result["col"]).all()
