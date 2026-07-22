# SPDX-FileCopyrightText: 2018-2025 Stichting DuckDB Foundation
# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: MIT AND Apache-2.0
#
# Modified by Vane contributors.

import numpy as np
import pytest

import vane


class TestVariantFetchall:
    """Tests for fetchall/fetchone with VARIANT columns (should all pass)."""

    def test_integer(self):
        result = vane.sql("SELECT 42::VARIANT AS v").fetchone()
        assert result[0] == 42

    def test_string(self):
        result = vane.sql("SELECT 'hello'::VARIANT AS v").fetchone()
        assert result[0] == "hello"

    def test_boolean(self):
        result = vane.sql("SELECT true::VARIANT AS v").fetchone()
        assert result[0] is True

    def test_double(self):
        result = vane.sql("SELECT 3.14::DOUBLE::VARIANT AS v").fetchone()
        assert abs(result[0] - 3.14) < 1e-10

    def test_null(self):
        result = vane.sql("SELECT NULL::VARIANT AS v").fetchone()
        assert result[0] is None

    def test_list(self):
        result = vane.sql("SELECT [1, 2, 3]::VARIANT AS v").fetchone()
        assert result[0] == [1, 2, 3]

    def test_struct(self):
        result = vane.sql("SELECT {'a': 1, 'b': 2}::VARIANT AS v").fetchone()
        assert result[0] == {"a": 1, "b": 2}

    def test_nested_struct(self):
        result = vane.sql("SELECT {'x': {'y': 42}}::VARIANT AS v").fetchone()
        assert result[0] == {"x": {"y": 42}}

    def test_map(self):
        result = vane.sql("SELECT MAP {'key1': 'val1', 'key2': 'val2'}::VARIANT AS v").fetchone()
        val = result[0]
        # VARIANT converts maps to a list of key/value structs
        assert val == [{"key": "key1", "value": "val1"}, {"key": "key2", "value": "val2"}]

    def test_multiple_rows_mixed_types(self):
        result = vane.sql("""
            SELECT * FROM (
                VALUES (42::VARIANT), ('hello'::VARIANT), (true::VARIANT), ([1,2]::VARIANT)
            ) AS t(v)
        """).fetchall()
        assert result[0][0] == 42
        assert result[1][0] == "hello"
        assert result[2][0] is True
        assert result[3][0] == [1, 2]

    def test_variant_from_table(self):
        con = vane.connect()
        con.execute("CREATE TABLE t (v VARIANT)")
        con.execute("INSERT INTO t VALUES (42::VARIANT), ('hello'::VARIANT)")
        result = con.execute("SELECT * FROM t").fetchall()
        assert result[0][0] == 42
        assert result[1][0] == "hello"

    def test_variant_as_map_key(self):
        """The original repro that motivated VARIANT support."""
        result = vane.sql("""
            SELECT MAP {42::VARIANT: 'answer'} AS m
        """).fetchone()
        # MAP with VARIANT keys is returned as a struct with key/value arrays
        assert result[0] == {"key": [42], "value": ["answer"]}


class TestVariantFetchNumpy:
    """Tests for fetchnumpy with VARIANT columns."""

    def test_single_row(self):
        result = vane.sql("SELECT 42::VARIANT AS v").fetchnumpy()
        assert result["v"][0] == 42

    def test_multiple_rows(self):
        """Exercises chunk_offset > 0 — this was broken by Bug A/B."""
        result = vane.sql("""
            SELECT * FROM (
                VALUES (1::VARIANT), (2::VARIANT), (3::VARIANT)
            ) AS t(v)
        """).fetchnumpy()
        values = list(result["v"])
        assert values == [1, 2, 3]

    def test_null_handling(self):
        result = vane.sql("""
            SELECT * FROM (
                VALUES (42::VARIANT), (NULL::VARIANT), (99::VARIANT)
            ) AS t(v)
        """).fetchnumpy()
        arr = result["v"]
        assert arr[0] == 42
        assert arr[1] is np.ma.masked or arr[1] is None
        assert arr[2] == 99

    def test_mixed_types(self):
        result = vane.sql("""
            SELECT * FROM (
                VALUES (42::VARIANT), ('hello'::VARIANT), (true::VARIANT)
            ) AS t(v)
        """).fetchnumpy()
        values = list(result["v"])
        assert values[0] == 42
        assert values[1] == "hello"
        assert values[2] is True


class TestVariantFetchDF:
    """Tests for Pandas df() with VARIANT columns (goes through numpy)."""

    def test_basic(self):
        df = vane.sql("SELECT 42::VARIANT AS v").df()
        assert df["v"].iloc[0] == 42

    def test_multiple_types(self):
        df = vane.sql("""
            SELECT * FROM (
                VALUES (42::VARIANT), ('hello'::VARIANT), (true::VARIANT)
            ) AS t(v)
        """).df()
        assert df["v"].iloc[0] == 42
        assert df["v"].iloc[1] == "hello"
        assert df["v"].iloc[2] is True

    def test_null_handling(self):
        df = vane.sql("""
            SELECT * FROM (
                VALUES (42::VARIANT), (NULL::VARIANT), (99::VARIANT)
            ) AS t(v)
        """).df()
        assert df["v"].iloc[0] == 42
        assert df["v"].iloc[2] == 99


class TestVariantArrow:
    """Tests for Arrow/Polars — blocked on DuckDB core Arrow support."""

    @pytest.mark.xfail(strict=True, reason="Arrow export for VARIANT not yet supported in DuckDB core")
    def test_to_arrow_table(self):
        vane.sql("SELECT 42::VARIANT AS v").arrow()

    @pytest.mark.xfail(strict=True, reason="Arrow export for VARIANT not yet supported in DuckDB core")
    def test_fetch_arrow_reader(self):
        vane.sql("SELECT 42::VARIANT AS v").fetch_arrow_reader()

    @pytest.mark.xfail(strict=True, reason="Polars uses Arrow, which doesn't support VARIANT yet")
    def test_polars(self):
        vane.sql("SELECT 42::VARIANT AS v").pl()


class TestVariantIngestion:
    """Tests for Python → DuckDB VARIANT ingestion."""

    def test_insert_with_params(self):
        con = vane.connect()
        con.execute("CREATE TABLE t (v VARIANT)")
        con.execute("INSERT INTO t VALUES ($1::VARIANT)", [42])
        result = con.execute("SELECT * FROM t").fetchone()
        assert result[0] == 42


class TestVariantType:
    """Tests for VARIANT in the type system."""

    def test_type_from_string(self):
        t = vane.type("VARIANT")
        assert t.id == "variant"

    def test_variant_constant(self):
        from vane.sqltypes import VARIANT

        assert VARIANT is not None
        assert VARIANT.id == "variant"

    def test_children_raises(self):
        t = vane.type("VARIANT")
        with pytest.raises(vane.InvalidInputException, match="not nested"):
            _ = t.children

    def test_sqltypes_variant(self):
        from vane.sqltypes import VARIANT

        assert VARIANT.id == "variant"


class TestVariantPySpark:
    """Tests for PySpark VARIANT type mapping."""

    def test_variant_converts_to_variant_type(self):
        from vane.experimental.spark.sql.type_utils import convert_type
        from vane.experimental.spark.sql.types import VariantType

        t = vane.type("VARIANT")
        spark_type = convert_type(t)
        assert isinstance(spark_type, VariantType)
