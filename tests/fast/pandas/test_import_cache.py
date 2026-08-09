# SPDX-FileCopyrightText: 2018-2025 Stichting DuckDB Foundation
# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: MIT AND Apache-2.0
#
# Modified by Vane contributors.

import importlib.util

import pandas as pd
import pytest

import vane


@pytest.mark.parametrize(
    "string_dtype",
    [
        "python",
        pytest.param(
            "pyarrow", marks=pytest.mark.skipif(not importlib.util.find_spec("pyarrow"), reason="pyarrow not installed")
        ),
    ],
)
def test_import_cache_explicit_dtype(string_dtype):
    df = pd.DataFrame(  # noqa: F841
        {
            "id": [1, 2, 3],
            "value": pd.Series(["123.123", pd.NaT, pd.NA], dtype=pd.StringDtype(storage=string_dtype)),
        }
    )
    con = vane.connect()
    result_df = con.query("select id, value from df").df()

    assert pd.isna(result_df["value"][1])
    assert pd.isna(result_df["value"][2])


def test_import_cache_implicit_dtype():
    df = pd.DataFrame({"id": [1, 2, 3], "value": pd.Series(["123.123", pd.NaT, pd.NA])})  # noqa: F841
    con = vane.connect()
    result_df = con.query("select id, value from df").df()

    assert pd.isna(result_df["value"][1])
    assert pd.isna(result_df["value"][2])
