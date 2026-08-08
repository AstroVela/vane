# SPDX-FileCopyrightText: 2018-2025 Stichting DuckDB Foundation
# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: MIT AND Apache-2.0
#
# Modified by Vane contributors.

import pytest

import vane
from vane.sqltypes import VARCHAR

pa = pytest.importorskip("pyarrow")


class TestArrowBufferSize:
    def test_arrow_buffer_size(self):
        con = vane.connect()

        # All small string
        res = con.query("select 'bla'").to_arrow_table()
        assert res[0][0].type == pa.string()
        res = con.query("select 'bla'").to_arrow_reader()
        assert res.schema[0].type == pa.string()

        # All Large String
        con.execute("SET arrow_large_buffer_size=True")
        res = con.query("select 'bla'").to_arrow_table()
        assert res[0][0].type == pa.large_string()
        res = con.query("select 'bla'").to_arrow_reader()
        assert res.schema[0].type == pa.large_string()

        # All small string again
        con.execute("SET arrow_large_buffer_size=False")
        res = con.query("select 'bla'").to_arrow_table()
        assert res[0][0].type == pa.string()
        res = con.query("select 'bla'").to_arrow_reader()
        assert res.schema[0].type == pa.string()

    def test_arrow_buffer_size_udf(self):
        def just_return(x):
            return x

        con = vane.connect()
        con.create_function("just_return", just_return, [VARCHAR], VARCHAR, type="arrow")

        res = con.query("select just_return('bla')").to_arrow_table()

        assert res[0][0].type == pa.string()

        # All Large String
        con.execute("SET arrow_large_buffer_size=True")

        res = con.query("select just_return('bla')").to_arrow_table()
        assert res[0][0].type == pa.large_string()
