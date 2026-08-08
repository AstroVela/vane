# SPDX-FileCopyrightText: 2018-2025 Stichting DuckDB Foundation
# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: MIT AND Apache-2.0
#
# Modified by Vane contributors.

from pathlib import Path

import pytest

import vane
from vane import FunctionExpression
from vane.sqltypes import BIGINT


@pytest.mark.parametrize("fields", [[None], {"a": None}])
def test_struct_type_rejects_null_type_holder(fields):
    with pytest.raises(vane.InvalidInputException, match="object has to be a list of DuckDBPyType"):
        vane.struct_type(fields)


def test_execute_rejects_null_statement_holder(duckdb_cursor):
    with pytest.raises(
        vane.InvalidInputException,
        match="Please provide either a DuckDBPyStatement or a string representing the query",
    ):
        duckdb_cursor.execute(None)


def test_value_rejects_null_type_holder(duckdb_cursor):
    value = vane.Value(1, None)
    with pytest.raises(
        vane.InvalidInputException,
        match="The 'type' of a Value should be of type DuckDBPyType",
    ):
        duckdb_cursor.execute("select ?", [value])


@pytest.mark.parametrize("dtype", [[None], {"category_id": None}])
def test_read_csv_rejects_null_type_holder(duckdb_cursor, dtype):
    csv_path = Path(__file__).parent / "data" / "category.csv"
    with pytest.raises(vane.InvalidInputException, match="can not be converted to a DuckDB Type"):
        duckdb_cursor.read_csv(csv_path, dtype=dtype)


def test_none_parameter_annotation_uses_explicit_udf_type(duckdb_cursor):
    def add_one(value: None):
        return value + 1

    duckdb_cursor.create_function("issue_159_add_one", add_one, [BIGINT], BIGINT)
    assert duckdb_cursor.sql("select issue_159_add_one(41)").fetchone() == (42,)


def test_none_still_implicitly_converts_to_sql_null_expression(duckdb_cursor):
    expression = FunctionExpression("greatest", None, 42)
    assert duckdb_cursor.sql("select 1").select(expression).fetchone() == (42,)
