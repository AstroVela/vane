# SPDX-FileCopyrightText: 2018-2025 Stichting DuckDB Foundation
# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: MIT AND Apache-2.0
#
# Modified by Vane contributors.

import sys

import vane


def test_duckdb_api():
    res = vane.execute("SELECT name, value FROM duckdb_settings() WHERE name == 'duckdb_api'")
    formatted_python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    assert res.fetchall() == [("duckdb_api", f"python/{formatted_python_version}")]
