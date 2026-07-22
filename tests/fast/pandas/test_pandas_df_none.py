# SPDX-FileCopyrightText: 2018-2025 Stichting DuckDB Foundation
# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: MIT AND Apache-2.0
#
# Modified by Vane contributors.

import vane


class TestPandasDFNone:
    # This used to decrease the ref count of None
    def test_none_deref(self):
        con = vane.connect()
        df = con.sql("select NULL::VARCHAR as a from range(1000000)").df()  # noqa: F841
