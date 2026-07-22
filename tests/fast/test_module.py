# SPDX-FileCopyrightText: 2018-2025 Stichting DuckDB Foundation
# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: MIT AND Apache-2.0
#
# Modified by Vane contributors.

import vane


class TestModule:
    def test_paramstyle(self):
        assert vane.paramstyle == "qmark"

    def test_threadsafety(self):
        assert vane.threadsafety == 1

    def test_apilevel(self):
        assert vane.apilevel == "2.0"
