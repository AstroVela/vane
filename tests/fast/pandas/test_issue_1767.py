#!/usr/bin/env python
# SPDX-FileCopyrightText: 2018-2025 Stichting DuckDB Foundation
# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: MIT AND Apache-2.0
#
# Modified by Vane contributors.


import pandas as pd
import pytest

import vane


# Join from pandas not matching identical strings #1767
@pytest.mark.usefixtures("ray_query")
class TestIssue1767:
    @pytest.mark.xfail(
        strict=True,
        raises=RuntimeError,
        reason="Ray FULL JOIN of Pandas sources finishes a scan queue without an explicit split batch",
    )
    def test_unicode_join_pandas(self, duckdb_cursor):
        A = pd.DataFrame({"key": ["a", "п"]})
        B = pd.DataFrame({"key": ["a", "п"]})
        con = vane.connect(":memory:")
        arrow = con.register("A", A).register("B", B)
        q = arrow.query("""SELECT key FROM "A" FULL JOIN "B" USING ("key") ORDER BY key""")
        result = q.df()

        d = {"key": ["a", "п"]}
        df = pd.DataFrame(data=d)
        pd.testing.assert_frame_equal(result, df, check_dtype=False)
