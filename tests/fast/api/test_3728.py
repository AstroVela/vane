# SPDX-FileCopyrightText: 2018-2025 Stichting DuckDB Foundation
# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: MIT AND Apache-2.0
#
# Modified by Vane contributors.

import vane


class Test3728:
    def test_3728_describe_enum(self, duckdb_cursor):
        # Create an in-memory database, but the problem is also present in file-backed DBs
        cursor = vane.connect(":memory:")

        # Create an arbitrary enum type
        cursor.execute("CREATE TYPE mood AS ENUM ('sad', 'ok', 'happy');")

        # Create a table where one or more columns are enum typed
        cursor.execute("CREATE TABLE person (name text, current_mood mood);")

        # This fails with "RuntimeError: Not implemented Error: unsupported type: mood"
        assert cursor.table("person").execute().description == [
            ("name", "VARCHAR", None, None, None, None, None),
            ("current_mood", "ENUM('sad', 'ok', 'happy')", None, None, None, None, None),
        ]
