# SPDX-FileCopyrightText: 2018-2025 Stichting DuckDB Foundation
# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: MIT AND Apache-2.0
#
# Modified by Vane contributors.

import datetime

import vane


class TestDateTimeDate:
    def test_date_infinity(self):
        con = vane.connect()
        # Positive infinity
        con.execute("SELECT 'infinity'::DATE")
        result = con.fetchall()
        # datetime.date.max
        assert result == [(datetime.date(9999, 12, 31),)]

        con.execute("SELECT '-infinity'::DATE")
        result = con.fetchall()
        # datetime.date.min
        assert result == [(datetime.date(1, 1, 1),)]

    def test_date_infinity_roundtrip(self):
        con = vane.connect()

        # positive infinity
        con.execute("select $1, $1 = 'infinity'::DATE", [datetime.date.max])
        res = con.fetchall()
        assert res == [(datetime.date.max, False)]

        # negative infinity
        con.execute("select $1, $1 = '-infinity'::DATE", [datetime.date.min])
        res = con.fetchall()
        assert res == [(datetime.date.min, False)]
