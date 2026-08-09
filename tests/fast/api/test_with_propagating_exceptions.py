# SPDX-FileCopyrightText: 2018-2025 Stichting DuckDB Foundation
# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: MIT AND Apache-2.0
#
# Modified by Vane contributors.

import pytest

import vane


class TestWithPropagatingExceptions:
    def test_with(self):
        # Should propagate exception raised in the 'with vane.connect() ..'
        with pytest.raises(vane.ParserException, match=r"syntax error at or near *"), vane.connect() as con:
            con.execute("invalid")

        # Does not raise an exception
        with vane.connect() as con:
            con.execute("select 1")
