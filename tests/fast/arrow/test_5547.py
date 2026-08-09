# SPDX-FileCopyrightText: 2018-2025 Stichting DuckDB Foundation
# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: MIT AND Apache-2.0
#
# Modified by Vane contributors.

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

import vane

pa = pytest.importorskip("pyarrow")


def test_5547():
    num_rows = 2**17 + 1

    tbl = pa.Table.from_pandas(
        pd.DataFrame.from_records(
            [
                {
                    "id": i,
                    "nested": {
                        "a": i,
                    },
                }
                for i in range(num_rows)
            ]
        )
    )

    con = vane.connect()
    expected = tbl.to_pandas()
    result = con.execute(
        """
		SELECT * FROM tbl
    """
    ).df()

    assert_frame_equal(expected, result)

    con.close()
