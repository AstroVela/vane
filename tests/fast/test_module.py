# SPDX-FileCopyrightText: 2018-2025 Stichting DuckDB Foundation
# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: MIT AND Apache-2.0
#
# Modified by Vane contributors.

import pytest

import vane


class TestModule:
    def test_paramstyle(self):
        assert vane.paramstyle == "qmark"

    def test_threadsafety(self):
        assert vane.threadsafety == 1

    def test_apilevel(self):
        assert vane.apilevel == "2.0"

    @pytest.mark.parametrize("connection_argument", ["omitted", "none", "explicit"])
    def test_module_wrappers_route_connection(self, connection_argument):
        previous_default = vane.default_connection()
        default = vane.connect()
        explicit = vane.connect()
        try:
            default.execute("CREATE TABLE module_values AS SELECT 11 AS value")
            explicit.execute("CREATE TABLE module_values AS SELECT 22 AS value")
            default.execute("CREATE TYPE module_kind AS ENUM ('default')")
            explicit.execute("CREATE TYPE module_kind AS ENUM ('explicit')")
            vane.set_default_connection(default)

            kwargs = {}
            if connection_argument == "none":
                kwargs["connection"] = None
            elif connection_argument == "explicit":
                kwargs["connection"] = explicit
            expected_value = 22 if connection_argument == "explicit" else 11
            expected_kind = "explicit" if connection_argument == "explicit" else "default"

            vane.execute("SELECT value FROM module_values", **kwargs)
            assert vane.fetchone(**kwargs) == (expected_value,)
            assert vane.table("module_values", **kwargs).fetchall() == [(expected_value,)]
            assert str(vane.sqltype("module_kind", **kwargs)) == f"ENUM('{expected_kind}')"
        finally:
            vane.set_default_connection(previous_default)
            explicit.close()
            default.close()

    def test_module_arrow_overloads(self):
        import pyarrow as pa

        from vane import _native

        with vane.connect() as connection:
            source = pa.table({"value": [11, 22]})
            relation = _native.arrow(source, connection=connection)
            assert relation.fetchall() == [(11,), (22,)]

            connection.execute("SELECT 33 AS value UNION ALL SELECT 44 AS value ORDER BY value")
            reader = _native.arrow(1, connection=connection)
            assert isinstance(reader, pa.RecordBatchReader)
            assert reader.read_all().to_pydict() == {"value": [33, 44]}
